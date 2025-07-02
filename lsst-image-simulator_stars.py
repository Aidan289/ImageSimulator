import sys
import os
import logging
import galsim
import copy
import numpy as np
from datetime import datetime
from astropy.table import Table, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
# from scipy.integrate import simps
from scipy.integrate import simpson

# Define some global constants
c_light = 2.998e8
h_planck = 6.626e-34

# From L. Y. Aaron Yung: < LINK TO PAPER >
cosmo = LambdaCDM(H0=67.8, Om0=0.308, Ode0=0.692)


def mag2uJy(mag):
    """ Helper function to convert AB magnitude to uJy flux """

    return 10 ** (-1 * (mag - 23.9) / 2.5)


def uJy2mag(uJy):
    """Helper function to convert uJy flux to AB magnitude"""

    return -2.5 * np.log10(uJy) + 23.9


def uJy2galflux(uJy, wavelengths, response):
    """ Helper function to convert flux from uJy to photon/cm^2/s """

    wavelengths = (wavelengths * u.um).to(u.m)
    # The galaxy flux is throughput * luminous density in photon/cm^2/s/Hz * nu_del integrated over nu (frequency)
    flux = simpson(y=response * (uJy * u.uJy).to(u.photon / u.cm ** 2 / u.s / u.Hz, equivalencies=
    u.spectral_density(wavelengths)).value * c_light / (wavelengths ** 2), x=wavelengths)

    return flux


def line(x, *p):
    """ Helper function for line-fitting """

    m, b = p
    return m * x + b


def effwave(wavelengths, response):
    '''Function to compute effective wavelength from the bandpass file.
        Assumes AB magnitude standard.
        Formula from <ADD LINK> Koornneef et al. 1986'''

    numer = simpson(y=wavelengths * wavelengths * response, x=wavelengths)
    denom = simpson(y=wavelengths * response, x=wavelengths)

    eff_wav = numer / denom

    return eff_wav

def get_datetime():
    now=datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def main(filter, user_t_exp, pixel_scale, dust=True, noise=True, catalogue_path=''):
    """ This script uses tabulated data from Aaron Yung's SAM catalogue to generate a simulated LSST image """

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("LSST 2")

    logger.info('Starting COSMOS LSST Image Script')

    if dust == True: print('Using dust magnitudes')
    else: print('Using intrinsic (dust-less) magnitudes')

    seed = 356789

    #  LSST specifications
    D = 6.423  # aperture in m
    g = 1.  # gain e/ADU

    # Set FOV
    fov_x = 0.44  # degrees
    fov_y = 0.56  # degrees
    # Set exposure time based on user input

    # Select filter for simulated image
    bandpass_file = telescope_config_path+"filter_lsst_" + filter + ".dat"
    bandpass_data = np.loadtxt(bandpass_file)

    # Specify which psf fits file to use for each filter selection
    # (Note: uv_split_bb and u_split_bb PSF fits files are not yet available, so we are using uv and u
    # PSFs respectively)
    psf_map = {'uv': 'uv', 'uv_split_bb': 'uv', 'u_split_bb': 'u', 'u': 'u', 'g': 'g'}
    psf_band = psf_map[filter]

    #  Read in characteristics of the bandpass
    eff_wav = effwave(bandpass_data[:, 0], bandpass_data[:, 1]) * 1000  # 1000 factor is to convert from um to nm

    logger.info('Considering the ' + filter + ' filter with effective wavelength ' + str(round(eff_wav, 0)) + ' nm')

    # Create PSF
    lsst_psf = galsim.Convolve([galsim.Gaussian(fwhm=0.8), galsim.Gaussian(fwhm=0.3)]) # same as ImSim
    
    directory = catalogue_path

    nfolder = 0

    # Define approx. center of lightcone field
    print("Creating full image canvas.")

    # Read a relatively large file
    data = Table.read('roman_hls_galaxia.dat', format='ascii', comment='#')

    RA_list = np.array(data['RA(2000)'])
    DEC_list = np.array(data['DEC(2000)'])

    # Shift RAs to center around 180
    # If >180 degrees, subtract 360
    #for i in range(0, len(RA_list)):
    #    if RA_list[i] >= 180:
    #        RA_list[i] -= 180  # -360+180=-180
    #    else:
    #        RA_list[i] += 180

    #data.add_column(col=RA_list, index=2, name='RA2')
    #print(data)

    # Use mean RA, DEC to define approximate field center
    cen_ra = np.mean(RA_list)
    cen_dec = np.mean(DEC_list)
    print(cen_ra, cen_dec)

    cen_coord = galsim.CelestialCoord(cen_ra * galsim.degrees, cen_dec * galsim.degrees)

    #  Create field image based on CASTOR specifications
    image_size_x = fov_x * 3600 / pixel_scale
    image_size_y = fov_y * 3600 / pixel_scale
    image = {}
    for t_exp in user_t_exp:
        image[t_exp] = galsim.Image(image_size_x, image_size_y)

        #  Define the WCS
        affine_wcs = galsim.PixelScale(pixel_scale).affine().shiftOrigin(image[t_exp].center)  # used to be withOrigin
        wcs = galsim.TanWCS(affine_wcs, world_origin=cen_coord)
        image[t_exp].wcs = wcs

    # Create mask for LSST's FOV. Overshoot as seems to miss the edges
    lbound = cen_ra - fov_x #/ 2
    rbound = cen_ra + fov_x #/ 2
    ubound = cen_dec + fov_y #/ 2
    dbound = cen_dec - fov_y #/ 2
    print(lbound,rbound,ubound,dbound)

    
    mask = np.logical_and(np.logical_and(RA_list >= lbound, RA_list <= rbound),
                          np.logical_and(DEC_list >= dbound, DEC_list <= ubound))

    #  Ignore galaxies outside of our simulated region
    data = data[mask]
    print(data)

    #  Get coordinates of selected galaxies
    RAs_to_draw = RA_list[mask]
    print(RAs_to_draw)
    DECs_to_draw = DEC_list[mask]
    tot = len(RAs_to_draw)

    count = 0
    FFTerror = []
    MemError = []

    #  Cycle through the list of stars
    for i in range(0, tot):

        if i % 1000 == 0:
            logger.info('Stamping object ' + str(i) + ' of ' + str(tot) + '...')

        # Assume disk/bulge mag ratio is the same in intrinsic and dust conditions.
        # intrinsic mags
        if data['CASTOR_' + filter][i] != '*':
            star_mag = np.float(data['LSST_' + filter][i])

        # convert to uJy flux
        star_uJyflux = mag2uJy(star_mag)



        gsp = galsim.GSParams(minimum_fft_size=16, maximum_fft_size=16384)

        
        #  Put coordinates into Galsim format
        coords = SkyCoord(ra=RAs_to_draw, dec=DECs_to_draw, unit=u.degree)
        coord_ra_list = coords.ra.hour
        coord_dec_list = coords.dec.degree
        coord = galsim.CelestialCoord(coord_ra_list[i] * galsim.hours, coord_dec_list[i] * galsim.degrees)        

        #  Convert coordinates to positions on image
        image_pos = wcs.toImage(coord)
        local_wcs = wcs.local(image_pos)
        ix = int(image_pos.x)
        iy = int(image_pos.y)


        ##Exposure time dependent properties grouped together for looping:
        for t_exp in user_t_exp:  # exposure time in seconds
            # convert flux to ADU
            star_flux = uJy2galflux(star_uJyflux, bandpass_data[:, 0], bandpass_data[:, 1]) \
                        * g * t_exp * np.pi * (D * 100. / 2) ** 2
            # print("disk flux:", disk_flux)


            #disk = galsim.Sersic(n=1, half_light_radius=data['r_disk'][i], flux=disk_flux)
            star=galsim.DeltaFunction(flux=star_flux)
            
            
            #  Convolve with PSF
            final = galsim.Convolve([star, lsst_psf])


            #  Create postage stamp image of galaxy
            try:
                stamp = final.drawImage(wcs=local_wcs, method='fft', nx=int(20 / pixel_scale),
                                        ny=int(20 / pixel_scale))
            except galsim.errors.GalSimFFTSizeError:
                print('FFTSizeError, skipping object')
                # print('Skipped object properties:\n', data[i])
                FFTerror.append(i)
                data['Error?'][i] = 1
                count += 1
                continue
            except ValueError:
                print('Value Error, skipping object')
                count += 1
                continue
            except MemoryError:
                print('Memory Error, skipping object')
                # print('Skipped object properties:\n', data[i])
                MemError.append(i)
                data['Error?'][i] = 1
                count += 1
                continue
            stamp.setCenter(ix, iy)

            #  Verify galaxy within CASTOR FOV
            bounds = stamp.bounds & image[t_exp].bounds
            if not bounds.isDefined():
                #print("Stamp:", stamp.bounds)
                #print("Image:", image[t_exp].bounds)
                #print('Galaxy off image')
                continue
            elif bounds.isDefined():
                #  Save postage stamp of galaxy to the field image
                image[t_exp][bounds] += stamp[bounds]

    dt=get_datetime()#include datetime of creation in filename to make sure files arent overwritten
    
    logger.info('Stamps complete, ' + str(count) + ' errors')
    np.savetxt(output_messages_path+dt+'_LSST_RomanStarCat_2019PSFs_' + filter + '_' + str(t_exp) + 's_pixscale'+str(pixel_scale).replace('.','p') +'FFTerrors.txt', FFTerror)
    np.savetxt(output_messages_path+dt+'_LSST_RomanStarCat_2019PSFs_' + filter + '_' + str(t_exp) + 's_pixscale'+str(pixel_scale).replace('.','p') + 'MemErrors.txt', MemError)
    
    for t_exp in user_t_exp:
        #  Add noise to field image and save
        file_name1 = os.path.join(output_images_path+dt+'_LSST_RomanStarCat_2019PSFs_' + filter + '_' + str(t_exp) +'s_pixscale'+str(pixel_scale).replace('.','p') + '.fits')
        if noise == True:
            # Calculate sky background
            # convert to ADU/pixel
            #sky_brightness = 22.26 # mag/arcsec^2
            #sky_flux = uJy2galflux(mag2uJy(sky_brightness),bandpass_data[:, 0], bandpass_data[:, 1]) # photons/s/cm^2/arcsec^2
            sky_level = 0#sky_flux * pixel_scale**2 * g * t_exp * np.pi * (D * 100. / 2) ** 2
        
            logger.info('Applying sky background of ' + str(sky_level) + ' counts')

            rng = galsim.BaseDeviate(t_exp)  # a random seed, different for each exposure
            ccd_noise = galsim.CCDNoise(rng, sky_level=sky_level, gain=g,
                                        read_noise=0)#9.)  # includes read noise, sky noise and shot noise, sky_level in ADU/pixel
            image[t_exp].addNoise(ccd_noise)
            print("Added CCD noise (including read noise, sky noise and shot noise)")
            #dark_current = 0.02 * t_exp  # dark current = 0.02 e/pixel/s
            #dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(rng, dark_current))  # includes dark current
            #image[t_exp].addNoise(dark_noise)
            #print("Added dark current noise")

        # Save image to FITS file
        image[t_exp].write(file_name1)
    return


if __name__=='__main__':
    
    ''' User Configuration Section '''
    # Choose image generation parameters
    filters = ["g"]  # available: u, uv, g, uv_split_bb, u_split_bb
    exp_times = [39000,3000,18000,2000]  # enter any exposure time in seconds (s)
    pixel_scale = 0.2  # arcsec/pixel
    dust = True  # specify if you want to use intrinsic or dust (observed) magnitudes
    noise = True
    #path to Aaron's catalogue
    catalogue_path = ''#/arc/projects/CASTOR/Simulations/wide.0/'
    output_messages_path = 'OutputMessages/'
    output_images_path = 'OutputImages/'
    telescope_config_path = '/arc/projects/CASTOR/MockPhotometry/LSST_config/'
    
    print('filters',filters,'pxscale',pixel_scale,'dust',dust,\
          'noise',noise,'exptimes',exp_times)
    
    # Run program for all specified filters and image types
    for filt in filters:
        print(filt)
        #for exp_time in exp_times:
        main(filt, exp_times, pixel_scale, dust=dust, noise=noise, catalogue_path=catalogue_path)

