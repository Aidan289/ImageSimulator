from astropy.io import fits

"""
exp_times = [39000,3000,18000,2000]

for exp_time in exp_times:
    galaxy_file='OutputImages/LSST_AaronsCat_2023PSFs_g_{}s_pixscale0p2.fits'.format(exp_time)
    star_file='OutputImages/LSST_RomanStarCat_2023PSFs_g_{}s_pixscale0p2.fits'.format(exp_time)
    out_file='OutputImages/LSST_RomanStarCatANDAaronsCat_2023PSFs_g_{}s_pixscale0p2.fits'.format(exp_time)

    with fits.open(galaxy_file) as galaxy_fits:
        with fits.open(star_file) as star_fits:
            combined_data=galaxy_fits[0].data+star_fits[0].data
            galaxy_fits[0].data=combined_data
            galaxy_fits.writeto(out_file,overwrite=True)
"""           
            
            
exp_times = [2000,36000,360000]
pxscale='1'

for fov in [1,2,3,4,5]:
    for exp_time in exp_times:
        galaxy_file='OutputImages/CASTOR_AaronsCat_wide4_2023PSFs__FOV{}_g_{}s_pixscale0p{}.fits'.format(fov,exp_time,pxscale)
        star_file='OutputImages/CASTOR_RomanStarCat_2023PSFs__FOV{}_g_{}s_pixscale0p{}.fits'.format(fov,exp_time,pxscale)
        out_file='OutputImages/CASTOR_RomanStarCatANDAaronsCat_wide4_2023PSFs__FOV{}_g_{}s_pixscale0p{}.fits'.format(fov,exp_time,pxscale)

        with fits.open(galaxy_file) as galaxy_fits:
            with fits.open(star_file) as star_fits:
                combined_data=galaxy_fits[0].data+star_fits[0].data
                star_fits[0].data=combined_data
                star_fits.writeto(out_file,overwrite=True)