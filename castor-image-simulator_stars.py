import sys
import os
import logging
import galsim
import copy
import numpy as np
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
    u.spectral_density(wavelengths)).value * c_light / (wavelengths ** 2), x=wavelengths, even="avg")

    return flux


def line(x, *p):
    """ Helper function for line-fitting """

    m, b = p
    return m * x + b


def effwave(wavelengths, response):
    '''Function to compute effective wavelength from the bandpass file.
        Assumes AB magnitude standard.
        Formula from <ADD LINK> Koornneef et al. 1986'''

    numer = simpson(y=wavelengths * wavelengths * response, x=wavelengths, even="avg")
    denom = simpson(y=wavelengths * response, x=wavelengths, even="avg")

    eff_wav = numer / denom

    return eff_wav


def skynoise(band, pixel_scale):
    """ Helper function to calculate sky noise per pixel (photons/s/cm^2/pixel) """

    #  Load filter and calculcate effective wavelength
    file_name = telescope_config_path+"etc_passband_castor." + band
    filter = np.loadtxt(file_name)
    eff_lam = effwave(filter[:, 0], filter[:, 1]) * 1000  # 1000 factor is to convert from um to nm

    #  Convert wavelengths to angstroms for consistency with the noise spectrum
    filter[:, 0] = (filter[:, 0] * u.um).to(u.angstrom)

    #  Load noise spectrum
    noise_spectrum = np.loadtxt(telescope_config_path+'hst_background_continuum.txt', skiprows=5)

    #  Load airglow spikes
    airglow_spikes = np.loadtxt(telescope_config_path+'hst_background_airglow.txt', skiprows=5, usecols=[0, 4])

    #  Select the background information from these files
    background = noise_spectrum[:, 3]
    background_airglow = airglow_spikes[:, 1]

    #  Create functions for the filter response and the sky background
    filt_response = interp1d(filter[:, 0], filter[:, 1])
    sky_spectrum = interp1d(noise_spectrum[:, 0], background)

    #  Integrate the product of these two function
    # Converts from erg/s/cm^2/arcsec^2 to photons/s/cm^2/arcsec^2
    wavs = np.linspace(min(filter[:, 0]) + 1, max(filter[:, 0]) - 1, 1000)  # in Angstroms (1e-10 m)
    combined_func = [filt_response(wav) * sky_spectrum(wav) * 1e-7 / (h_planck * c_light / (wav * 1e-10)) for wav in
                     wavs]
    # 6.626e-34 * 2.998e8 / (wav * 1E-10) is the energy of a photon of (Angstrom) wavelength wav
    flux_phot = simpson(combined_func, wavs)

    #  Add the airglow spikes
    for i in range(0, len(airglow_spikes[:, 0])):
        J_per_phot = h_planck * c_light / (airglow_spikes[i, 0] * 1e-10)  # energy of the airglow wavelength photon

        flux_phot += (filt_response(airglow_spikes[i, 0]) * background_airglow[i] * 1e-7 / J_per_phot)
        # ^ Also converts from erg/s/cm^2/arcsec^2 to photons/s/cm^2/arcsec^2
        #       - 1e-7  converts from erg to J
        #       - / J_per_phot converts from J to photons

    #  Calculate area of pixel in arcsec^2
    pixel_area = pixel_scale ** 2

    #  Calculate and return background in photons/s/cm^2/pixel
    return pixel_area * flux_phot


def main(filter, user_t_exp, pixel_scale, dust=True, noise=True, catalogue_path='', center=None, fov_name=-99):
    """ This script uses tabulated data from Aaron Yung's SAM catalogue to generate a simulated CASTOR image """

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("Castor 2")

    logger.info('Starting COSMOS Castor Image Script')

    if dust == True: print('Using dust magnitudes')
    else: print('Using intrinsic (dust-less) magnitudes')

    seed = 356789

    #  CASTOR specifications
    D = 1.  # aperture in m
    g = 1.  # gain e/ADU

    # Set FOV
    fov_x = 0.44  # degrees
    fov_y = 0.56  # degrees
    # Set exposure time based on user input

    # Select filter for simulated image
    bandpass_file = telescope_config_path+"etc_passband_castor." + filter
    bandpass_data = np.loadtxt(bandpass_file)

    # Specify which psf fits file to use for each filter selection
    # (Note: uv_split_bb and u_split_bb PSF fits files are not yet available, so we are using uv and u
    # PSFs respectively)
    psf_map = {'uv': 'NUV', 'uv_split_bb': 'NUV', 'u_split_bb': 'U', 'u': 'U', 'g': 'G'}
    psf_band = psf_map[filter]

    #  Read in characteristics of the bandpass
    eff_wav = effwave(bandpass_data[:, 0], bandpass_data[:, 1]) * 1000  # 1000 factor is to convert from um to nm

    logger.info('Considering the ' + filter + ' filter with effective wavelength ' + str(round(eff_wav, 0)) + ' nm')

    # Fetch filter-appropriate PSF
    psf_file = telescope_config_path+ psf_band+'samples_median_withJitter_X00-000d_Y00-000d_S0-001mm_resampledx3.fits'
    #New PSFs from Honeywell (Aug 2023), taking median of the 10 realizations and adding 
    #0.023'' spacecraft jitter by convolving with a Gaussian 
    
    #psf_file = telescope_config_path+psf_band + "_jitter_blk.fits" ##old PSFs
    
    psf_pixel_scale = 0.03 #new PSFs are 10x undersampled, then x3 upsampled #0.1
    castor_psf = galsim.InterpolatedImage(psf_file, scale=psf_pixel_scale, flux=1.)

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

    if center is None:
        # Use mean RA, DEC to define approximate field center
        cen_ra = np.mean(RA_list)
        cen_dec = np.mean(DEC_list)
        print(cen_ra, cen_dec)
        cen_coord = galsim.CelestialCoord(cen_ra * galsim.degrees, cen_dec * galsim.degrees)
        
    else:
        cen_ra = center[0]
        cen_dec = center[1]
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

    # Create mask for CASTOR's FOV. Overshoot as seems to miss the edges
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
            star_mag = np.float(data['CASTOR_' + filter][i])

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
            final = galsim.Convolve([star, castor_psf])


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


    logger.info('Stamps complete, ' + str(count) + ' errors')
    np.savetxt(output_messages_path+'CASTOR_RomanStarCat_2023PSFs_'+'_FOV{}_'.format(fov_name) + filter + '_' + str(t_exp) + 's_pixscale'+str(pixel_scale).replace('.','p') +'FFTerrors.txt', FFTerror)
    np.savetxt(output_messages_path+'CASTOR_RomanStarCat_2023PSFs_'+'_FOV{}_'.format(fov_name) + filter + '_' + str(t_exp) + 's_pixscale'+str(pixel_scale).replace('.','p') + 'MemErrors.txt', MemError)
    
    for t_exp in user_t_exp:
        #  Add noise to field image and save
        file_name1 = os.path.join(output_images_path+'CASTOR_RomanStarCat_2023PSFs_'+'_FOV{}_'.format(fov_name) + filter + '_' + str(t_exp) +'s_pixscale'+str(pixel_scale).replace('.','p') + '.fits')
        if noise == True:
            # Calculate sky background
            #bkg = skynoise(filter, pixel_scale)
            sky_level = 0#bkg * g * t_exp * np.pi * (D * 100. / 2) ** 2
            logger.info('Applying sky background of ' + str(sky_level) + ' counts')

            rng = galsim.BaseDeviate(t_exp)  # a random seed, different for each exposure
            ccd_noise = galsim.CCDNoise(rng, sky_level=sky_level, gain=g,
                                        read_noise=0)#2)  # includes read noise, sky noise and shot noise
            image[t_exp].addNoise(ccd_noise)
            print("Added CCD noise (including read noise, sky noise and shot noise)")
            #dark_current = 0.01 * t_exp  # dark current = 0.01 e/pixel/s
            #dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(rng, dark_current))  # includes dark current
            #image[t_exp].addNoise(dark_noise)
            #print("Added dark current noise")

        # Save image to FITS file
        image[t_exp].write(file_name1)
    return


if __name__=='__main__':
    
    ''' User Configuration Section '''
    # Choose image generation parameters
    filters = ["g"]#,"u" "uv", "g", "uv_split_bb", "u_split_bb"]  # available: u, uv, g, uv_split_bb, u_split_bb
    exp_times = [2000,36000,360000]#  # enter any exposure time in seconds (s)
    pixel_scale = 0.1  # arcsec/pixel
    dust = True  # specify if you want to use intrinsic or dust (observed) magnitudes
    noise = True
    #path to Aaron's catalogue
    catalogue_path = ''
    output_messages_path = 'OutputMessages/'
    output_images_path = 'OutputImages/'
    telescope_config_path = '/arc/projects/CASTOR/MockPhotometry/CASTOR_config/'
    
    print('filters',filters,'pxscale',pixel_scale,'dust',dust,\
          'noise',noise,'exptimes',exp_times)
    
    
    
    RA_centers = [29.79, 30.00, 30.21, 29.79, 30.00, 30.21]
    DEC_centers = [-40.21,-40.21,-40.21, -39.77,-39.77,-39.77]
    for fov_id in [1,2,3,4,5]:
        print('')
        print('================ FOV {} ================'.format(fov_id))
        print('')
        center=[RA_centers[fov_id],DEC_centers[fov_id]]

        print('filters',filters,'pxscale',pixel_scale,'dust',dust,\
              'noise',noise,'exptimes',exp_times,'FOV',fov_id)

        # Run program for all specified filters and image types
        for filt in filters:
            print(filt)
            #for exp_time in exp_times:
            main(filt, exp_times, pixel_scale, dust=dust, noise=noise, catalogue_path=catalogue_path,center=center,fov_name=fov_id)