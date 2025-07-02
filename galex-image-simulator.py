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

def main(filter, user_t_exp, pixel_scale, dust=True, noise=True, catalogue_path='',center=None,fov_name=-99):
    """ This script uses tabulated data from Aaron Yung's SAM catalogue to generate a simulated GALEX image """

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("GALEX 2")

    logger.info('Starting COSMOS GALEX Image Script')

    if dust == True: print('Using dust magnitudes')
    else: print('Using intrinsic (dust-less) magnitudes')

    seed = 356789

    #  GALEX specifications
    D = 0.5  # aperture in m
    g = 1.  # gain e/ADU
    
    if filter=="NUV":
        TEL_PSF_FWHM = 6.0 #arcsec
        TEL_dark_current = 0 #e/pixel/s 
        TEL_sky_level = 26.5# mag/arcsec^2   #20000 #counts/s
    else:
        TEL_PSF_FWHM = 4.5 #(instrument-limited values)
        TEL_dark_current = 0 #e/pixel/s 
        TEL_sky_level = 27.5# mag/arcsec^2   #20000 #counts/s
    TEL_read_noise=0 #e/pixel 
    

    # Set FOV
    fov_x = 0.44  # degrees
    fov_y = 0.56  # degrees
    # Set exposure time based on user input

    # Select filter for simulated image
    bandpass_file = telescope_config_path+"filter_galex_" + filter + ".dat"
    bandpass_data = np.loadtxt(bandpass_file)


    #  Read in characteristics of the bandpass
    eff_wav = effwave(bandpass_data[:, 0], bandpass_data[:, 1]) * 1000  # 1000 factor is to convert from um to nm

    logger.info('Considering the ' + filter + ' filter with effective wavelength ' + str(round(eff_wav, 0)) + ' nm')

    # Create PSF
    GALEX_psf = galsim.Gaussian(fwhm=TEL_PSF_FWHM) 
    
    directory = catalogue_path

    nfolder = 0

    # Define approx. center of lightcone field
    print("Creating full image canvas.")

    # Read a relatively large file
    header=['halo_id_nbody', 'gal_id', 'gal_type', 'z_nopec', 'redshift', 'ra', 'dec', 'm_vir', 'V_vir', 'r_vir', 'c_NFW', 'spin', 'mstar_diffuse', 'm_hot_halo', 'Z_hot_halo', 'v_disk', 'r_disk', 'sigma_bulge', 'rbulge', 'mhalo', 'mstar', 'mcold', 'mbulge', 'mbh', 'maccdot', 'maccdot_radio', 'Zstar', 'Zcold', 'mstardot', 'sfr_ave', 'meanage', 'tmerge', 'tmajmerge', 'cosi', 'tauV0', 'maccdot_BH', 'sfr10myr', 'mstarold', 'ageold', 'zstarold', 'UV1500_rest', 'UV1500_rest_bulge', 'UV1500_rest_dust', 'UV2300_rest', 'UV2300_rest_bulge', 'UV2300_rest_dust', 'UV2800_rest', 'UV2800_rest_bulge', 'UV2800_rest_dust', 'U_rest', 'U_rest_bulge', 'U_rest_dust', 'B_rest', 'B_rest_bulge', 'B_rest_dust', 'V_rest', 'V_rest_bulge', 'V_rest_dust', 'R_rest', 'R_rest_bulge', 'R_rest_dust', 'I_rest', 'I_rest_bulge', 'I_rest_dust', 'J_rest', 'J_rest_bulge', 'J_rest_dust', 'H_rest', 'H_rest_bulge', 'H_rest_dust', 'K_rest', 'K_rest_bulge', 'K_rest_dust', 'galex_FUV', 'galex_FUV_bulge', 'galex_FUV_dust', 'galex_NUV', 'galex_NUV_bulge', 'galex_NUV_dust', 'sdss_u', 'sdss_u_bulge', 'sdss_u_dust', 'sdss_g', 'sdss_g_bulge', 'sdss_g_dust', 'sdss_r', 'sdss_r_bulge', 'sdss_r_dust', 'sdss_i', 'sdss_i_bulge', 'sdss_i_dust', 'sdss_z', 'sdss_z_bulge', 'sdss_z_dust', 'acsf435w', 'acsf435w_bulge', 'acsf435w_dust', 'acsf606w', 'acsf606w_bulge', 'acsf606w_dust', 'acsf775w', 'acsf775w_bulge', 'acsf775w_dust', 'acsf814w', 'acsf814w_bulge', 'acsf814w_dust', 'acsf850lp', 'acsf850lp_bulge', 'acsf850lp_dust', 'wfc3f275w', 'wfc3f275w_bulge', 'wfc3f275w_dust', 'wfc3f336w', 'wfc3f336w_bulge', 'wfc3f336w_dust', 'wfc3f105w', 'wfc3f105w_bulge', 'wfc3f105w_dust', 'wfc3f125w', 'wfc3f125w_bulge', 'wfc3f125w_dust', 'wfc3f160w', 'wfc3f160w_bulge', 'wfc3f160w_dust', 'ctio_U', 'ctio_U_bulge', 'ctio_U_dust', 'CFHTLS_u', 'CFHTLS_u_bulge', 'CFHTLS_u_dust', 'musyc_u38', 'musyc_u38_bulge', 'musyc_u38_dust', 'UKIRT_J', 'UKIRT_J_bulge', 'UKIRT_J_dust', 'UKIRT_H', 'UKIRT_H_bulge', 'UKIRT_H_dust', 'UKIRT_K', 'UKIRT_K_bulge', 'UKIRT_K_dust', 'irac_ch1', 'irac_ch1_bulge', 'irac_ch1_dust', 'irac_ch2', 'irac_ch2_bulge', 'irac_ch2_dust', 'NIRCam_F070W', 'NIRCam_F070W_bulge', 'NIRCam_F070W_dust', 'NIRCam_F090W', 'NIRCam_F090W_bulge', 'NIRCam_F090W_dust', 'NIRCam_F115W', 'NIRCam_F115W_bulge', 'NIRCam_F115W_dust', 'NIRCam_F150W', 'NIRCam_F150W_bulge', 'NIRCam_F150W_dust', 'NIRCam_F200W', 'NIRCam_F200W_bulge', 'NIRCam_F200W_dust', 'NIRCam_F277W', 'NIRCam_F277W_bulge', 'NIRCam_F277W_dust', 'NIRCam_F356W', 'NIRCam_F356W_bulge', 'NIRCam_F356W_dust', 'NIRCam_F444W', 'NIRCam_F444W_bulge', 'NIRCam_F444W_dust', 'NIRCam_F140M', 'NIRCam_F140M_bulge', 'NIRCam_F140M_dust', 'NIRCam_F162M', 'NIRCam_F162M_bulge', 'NIRCam_F162M_dust', 'NIRCam_F182M', 'NIRCam_F182M_bulge', 'NIRCam_F182M_dust', 'NIRCam_F210M', 'NIRCam_F210M_bulge', 'NIRCam_F210M_dust', 'NIRCam_F250M', 'NIRCam_F250M_bulge', 'NIRCam_F250M_dust', 'NIRCam_F335M', 'NIRCam_F335M_bulge', 'NIRCam_F335M_dust', 'NIRCam_F360M', 'NIRCam_F360M_bulge', 'NIRCam_F360M_dust', 'NIRCam_F410M', 'NIRCam_F410M_bulge', 'NIRCam_F410M_dust', 'NIRCam_F430M', 'NIRCam_F430M_bulge', 'NIRCam_F430M_dust', 'NIRCam_F460M', 'NIRCam_F460M_bulge', 'NIRCam_F460M_dust', 'NIRCam_F480M', 'NIRCam_F480M_bulge', 'NIRCam_F480M_dust', 'Euclid_VIS', 'Euclid_VIS_bulge', 'Euclid_VIS_dust', 'Euclid_Y', 'Euclid_Y_bulge', 'Euclid_Y_dust', 'Euclid_J', 'Euclid_J_bulge', 'Euclid_J_dust', 'Euclid_H', 'Euclid_H_bulge', 'Euclid_H_dust', 'Roman_F062', 'Roman_F062_bulge', 'Roman_F062_dust', 'Roman_F087', 'Roman_F087_bulge', 'Roman_F087_dust', 'Roman_F106', 'Roman_F106_bulge', 'Roman_F106_dust', 'Roman_F129', 'Roman_F129_bulge', 'Roman_F129_dust', 'Roman_F146', 'Roman_F146_bulge', 'Roman_F146_dust', 'Roman_F158', 'Roman_F158_bulge', 'Roman_F158_dust', 'Roman_F184', 'Roman_F184_bulge', 'Roman_F184_dust', 'Roman_F213', 'Roman_F213_bulge', 'Roman_F213_dust', 'LSST_u', 'LSST_u_bulge', 'LSST_u_dust', 'LSST_g', 'LSST_g_bulge', 'LSST_g_dust', 'LSST_r', 'LSST_r_bulge', 'LSST_r_dust', 'LSST_i', 'LSST_i_bulge', 'LSST_i_dust', 'LSST_z', 'LSST_z_bulge', 'LSST_z_dust', 'LSST_y', 'LSST_y_bulge', 'LSST_y_dust', 'DECam_u', 'DECam_u_bulge', 'DECam_u_dust', 'DECam_g', 'DECam_g_bulge', 'DECam_g_dust', 'DECam_r', 'DECam_r_bulge', 'DECam_r_dust', 'DECam_i', 'DECam_i_bulge', 'DECam_i_dust', 'DECam_z', 'DECam_z_bulge', 'DECam_z_dust', 'DECam_Y', 'DECam_Y_bulge', 'DECam_Y_dust', 'NEWFIRM_K_atm', 'NEWFIRM_K_atm_bulge', 'NEWFIRM_K_atm_dust', 'VISTA_z', 'VISTA_z_bulge', 'VISTA_z_dust', 'VISTA_Y', 'VISTA_Y_bulge', 'VISTA_Y_dust', 'VISTA_J', 'VISTA_J_bulge', 'VISTA_J_dust', 'VISTA_H', 'VISTA_H_bulge', 'VISTA_H_dust', 'VISTA_Ks', 'VISTA_Ks_bulge', 'VISTA_Ks_dust', 'HSC_g', 'HSC_g_bulge', 'HSC_g_dust', 'HSC_r', 'HSC_r_bulge', 'HSC_r_dust', 'HSC_i', 'HSC_i_bulge', 'HSC_i_dust', 'HSC_z', 'HSC_z_bulge', 'HSC_z_dust', 'HSC_Y', 'HSC_Y_bulge', 'HSC_Y_dust', 'CFHT_J', 'CFHT_J_bulge', 'CFHT_J_dust', 'CFHT_Ks', 'CFHT_Ks_bulge', 'CFHT_Ks_dust', 'CASTOR.g', 'CASTOR.g_bulge', 'CASTOR.g_dust', 'CASTOR.u_split_bb', 'CASTOR.u_split_bb_bulge', 'CASTOR.u_split_bb_dust', 'CASTOR.u', 'CASTOR.u_bulge', 'CASTOR.u_dust', 'CASTOR.uv_split_bb', 'CASTOR.uv_split_bb_bulge', 'CASTOR.uv_split_bb_dust', 'CASTOR.uv', 'CASTOR.uv_bulge', 'CASTOR.uv_dust']
    
    
    relevant_columns=['ra', 'dec', 'galex_FUV', 'galex_FUV_bulge', 'galex_FUV_dust', 'galex_NUV', 'galex_NUV_bulge', 'galex_NUV_dust', 'r_disk', 'rbulge',
                                     'cosi', 'redshift']
    
    #data = Table.read(directory+'part_z0.70_z0.80/lightcone.dat', format='ascii', comment='#', names=header,
                     # include_names=relevant_columns,guess=False,data_start=0)
    data = Table.read(directory+'part_z3.75_z4.00/lightcone_nc.dat', format='ascii', comment='#', names=header,
                      #'part_z0.70_z0.80/lightcone_nc.dat', format='ascii', comment='#', names=header,
                      include_names=relevant_columns,guess=False,data_start=0)
    RA_list = data['ra']
    DEC_list = data['dec']

    # Shift RAs to center around 180
    # If >180 degrees, subtract 360
    for i in range(0, len(RA_list)):
        if RA_list[i] >= 180:
            RA_list[i] -= 180  # -360+180=-180
        else:
            RA_list[i] += 180

    data.add_column(col=RA_list, index=2, name='RA2')
    print(data)

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

    #  Create field image based on GALEX specifications
    image_size_x = fov_x * 3600 // pixel_scale
    image_size_y = fov_y * 3600 // pixel_scale
    print(image_size_x,image_size_y)
    
    image = {}
    for t_exp in user_t_exp:
        image[t_exp] = galsim.Image(image_size_x, image_size_y)

        #  Define the WCS
        affine_wcs = galsim.PixelScale(pixel_scale).affine().shiftOrigin(image[t_exp].center)  # used to be withOrigin
        wcs = galsim.TanWCS(affine_wcs, world_origin=cen_coord)
        image[t_exp].wcs = wcs

    # Create mask for GALEX's FOV
    lbound = cen_ra - fov_x / 2
    rbound = cen_ra + fov_x / 2
    ubound = cen_dec + fov_y / 2
    dbound = cen_dec - fov_y / 2

    print(lbound,rbound,ubound,dbound)
    # Loop through the redshift bin folders
    for folder in os.listdir(directory):
        # print folder name
        print('__________'+str(folder)+'__________')

        # Read in the table that the SAM data is saved in
        try:
            data = Table.read(directory + folder + '/lightcone_nc.dat', format='ascii', comment='#', names=header,
                      include_names=relevant_columns,guess=False,data_start=0)
        except:
            data1 = Table.read(directory + folder + '/lightcone_1.dat', format='ascii', comment='#', names=header,
                      include_names=relevant_columns,guess=False,data_start=0)
            data2 = Table.read(directory + folder + '/lightcone_2.dat', format='ascii', comment='#', names=header,
                      include_names=relevant_columns,guess=False,data_start=0)
            data = vstack([data1, data2], join_type='exact')

        RA_list = data['ra']
        #print(RA_list)
        DEC_list = data['dec']

        # Shift RAs
        # If >180 degrees, subtract 360
        for i in range(0, len(RA_list)):
            if RA_list[i] >= 180:
                RA_list[i] -= 180  # -360+180=-180
            else:
                RA_list[i] += 180

        data.add_column(col=RA_list, index=2, name='RA2')

        mask = np.logical_and(np.logical_and(RA_list >= lbound, RA_list <= rbound),
                              np.logical_and(DEC_list >= dbound, DEC_list <= ubound))

        #  Ignore galaxies outside of our simulated region
        data = data[mask]
        print(data)

        #  Get coordinates of selected galaxies
        RAs_to_draw = data['RA2']   # used to be RAs_to_draw = data['ra']
        print(RAs_to_draw)
        DECs_to_draw = data['dec']
        tot = len(RAs_to_draw)
            
        #  Put coordinates into Galsim format
        coords = SkyCoord(ra=RAs_to_draw, dec=DECs_to_draw, unit=u.degree)
        coord_ra_list = coords.ra.hour
        coord_dec_list = coords.dec.degree
        
        count = 0
        FFTerror = []
        MemError = []

        data.add_column(col=np.zeros(len(data['ra'])), name='Error?')

        # Assume disk/bulge mag ratio is the same in intrinsic and dust conditions.
        # intrinsic mags
        disk_mag = np.array(data['galex_' + filter])
        bulge_mag = np.array(data['galex_' + filter + '_bulge'])
        # dust mag (disk+bulge combined)
        dust_mag = np.array(data['galex_' + filter + '_dust'])

        # convert to uJy flux
        disk_uJyflux = mag2uJy(disk_mag)
        bulge_uJyflux = mag2uJy(bulge_mag)
        dust_uJyflux = mag2uJy(dust_mag)

        # get the individual dust mags
        bulge_dust_uJyflux = dust_uJyflux * bulge_uJyflux / (disk_uJyflux + bulge_uJyflux)
        disk_dust_uJyflux = dust_uJyflux * disk_uJyflux / (disk_uJyflux + bulge_uJyflux)

        # select dust or intrinsic mags based on user input
        if dust == True:
            use_uJy_disk = disk_dust_uJyflux  # dust mags
            use_uJy_bulge = bulge_dust_uJyflux
        else:
            use_uJy_disk = disk_uJyflux  # intrinsic mags, no dust
            use_uJy_bulge = bulge_uJyflux


        z = np.array(data['redshift'])
        arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z) * u.kpc / u.arcsec  # arcsec/kpc at redshift z
        r_disk = np.array(data['r_disk']) * arcsec_per_kpc
        r_bulge = np.array(data['rbulge']) * arcsec_per_kpc
        cosi = np.array(data['cosi'])
            
            
        #  Cycle through the list of galaxies
        for i in range(0, tot):

            if i % 1000 == 0:
                logger.info('Stamping object ' + str(i) + ' of ' + str(tot) + '...')

            
            gsp = galsim.GSParams(minimum_fft_size=16, maximum_fft_size=16384)
            
            
            # randomly generate beta between 0 and 2pi
            obj_rng = galsim.UniformDeviate(seed + i)
            Beta = galsim.Angle(obj_rng() * 360.0, unit=galsim.degrees)
 
            coord = galsim.CelestialCoord(coord_ra_list[i] * galsim.hours, coord_dec_list[i] * galsim.degrees)        
            
            #  Convert coordinates to positions on image
            image_pos = wcs.toImage(coord)
            local_wcs = wcs.local(image_pos)
            ix = int(image_pos.x)
            iy = int(image_pos.y)
            
            
            ##Exposure time dependent properties grouped together for looping:
            for t_exp in user_t_exp:  # exposure time in seconds
                # convert flux to ADU
                disk_flux = uJy2galflux(use_uJy_disk[i], bandpass_data[:, 0], bandpass_data[:, 1]) \
                            * g * t_exp * np.pi * (D * 100. / 2) ** 2
                # print("disk flux:", disk_flux)

                bulge_flux = uJy2galflux(use_uJy_bulge[i], bandpass_data[:, 0], bandpass_data[:, 1]) \
                             * g * t_exp * np.pi * (D * 100. / 2) ** 2
                # print("bulge flux:", bulge_flux)

                disk = galsim.Sersic(n=1, half_light_radius=r_disk[i], flux=disk_flux)

                if r_bulge[i] != 0.0:
                    bulge = galsim.Sersic(n=4, half_light_radius=r_bulge[i], flux=bulge_flux,
                                          trunc=10 * r_bulge[i],
                                          gsparams=gsp)  # assuming these are scale radii
                    gal = disk + bulge
                else:
                    gal = disk
                    # print('no bulge', data['rbulge'][i])


                gal = gal.shear(q=cosi[i], beta=Beta)

                #  Convolve with PSF
                final = galsim.Convolve([gal, GALEX_psf])


                #  Create postage stamp image of galaxy
                try:
                    if int(20 * r_disk[i] / pixel_scale)<10:
                        wdth=10
                    else:
                        wdth=int(20 * r_disk[i] / pixel_scale)
                    stamp = final.drawImage(wcs=local_wcs, method='fft', nx=wdth,
                                            ny=wdth)
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
                    print("Stamp:", stamp.bounds)
                    print("Image:", image[t_exp].bounds)
                    print('Galaxy off image')
                    continue
                elif bounds.isDefined():
                    #  Save postage stamp of galaxy to the field image
                    image[t_exp][bounds] += stamp[bounds]

        #if nfolder == 0: break  # used for testing, if only want to use first few redshift bins
        nfolder += 1

    dt=get_datetime()#include datetime of creation in filename to make sure files arent overwritten
    
    logger.info('Stamps complete, ' + str(count) + ' errors')
    np.savetxt(output_messages_path + dt + '_GALEX_AaronsCat_wide{}_'.format(simulation_num) +'_FOV{}_'.format(fov_name) + filter + '_' + str(t_exp) + 's_pixscale'+str(pixel_scale).replace('.','p') +'FFTerrors.txt', FFTerror)
    np.savetxt(output_messages_path + dt + '_GALEX_AaronsCat_wide{}_'.format(simulation_num) +'_FOV{}_'.format(fov_name) + filter + '_' + str(t_exp) + 's_pixscale'+str(pixel_scale).replace('.','p') + 'MemErrors.txt', MemError)
    
    for t_exp in user_t_exp:
        #  Add noise to field image and save
        file_name1 = os.path.join(output_images_path + dt + '_GALEX_AaronsCat_wide{}_'.format(simulation_num)  +'_FOV{}_'.format(fov_name) + filter + '_' + str(t_exp) +'s_pixscale'+str(pixel_scale).replace('.','p') + '.fits')
        if noise == True:
            # Calculate sky background
            # convert to ADU/pixel
            sky_brightness = TEL_sky_level # mag/arcsec^2
            sky_flux = uJy2galflux(mag2uJy(sky_brightness),bandpass_data[:, 0], bandpass_data[:, 1]) # photons/s/cm^2/arcsec^2
            sky_level = sky_flux * pixel_scale**2 * g * t_exp * np.pi * (D * 100. / 2) ** 2
            
            #sky_level=TEL_sky_level * t_exp #counts
            
            logger.info('Applying sky background of ' + str(sky_level) + ' counts')

            rng = galsim.BaseDeviate(t_exp)  # a random seed, different for each exposure
            ccd_noise = galsim.CCDNoise(rng, sky_level=sky_level, gain=g,
                                        read_noise=TEL_read_noise)  # includes read noise, sky noise and shot noise, sky_level in ADU/pixel
            image[t_exp].addNoise(ccd_noise)
            print("Added CCD noise (including read noise, sky noise and shot noise)")
            if TEL_dark_current>0:
                dark_current = TEL_dark_current * t_exp  # dark current = 0.02 e/pixel/s
                dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(rng, dark_current))  # includes dark current
                image[t_exp].addNoise(dark_noise)
                print("Added dark current noise, "+str(dark_noise))

        # Save image to FITS file
        image[t_exp].write(file_name1)
    return


if __name__=='__main__':
    
    ''' User Configuration Section '''
    # Choose image generation parameters
    filters = ["NUV","FUV"]  # available: NUV, FUV
    exp_times = [100,1500,30000,200000]  # enter any exposure time in seconds (s) ##UPDATE
    pixel_scale = 1.5  # arcsec/pixel
    dust = True  # specify if you want to use intrinsic or dust (observed) magnitudes
    noise = True
    #path to Aaron's catalogue
    simulation_num = 4
    catalogue_path = '/arc/projects/CASTOR/Simulations/wide.{}/'.format(simulation_num)
    output_messages_path = 'OutputMessages/'
    output_images_path = 'OutputImages/'
    telescope_config_path = '/arc/projects/CASTOR/MockPhotometry/GALEX_config/'
    
    print('filters',filters,'pxscale',pixel_scale,'dust',dust,\
          'noise',noise,'exptimes',exp_times)
    
    RA_centers = [179.52, 179.98, 180.44, 179.52, 179.98, 180.44]
    DEC_centers = [-0.335,-0.335,-0.335, 0.335, 0.335, 0.335]
    for fov_id in [0,4,5]: #number in 0-5, chooses which FOV from the 6 that will fit in the full catalogue area
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