
import argparse
import astropy.io.fits as fits
from astropy.io import ascii
#from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import numpy as np
#import numpy.fft as fft
import matplotlib.pyplot as plt
import os
from subprocess import call
from scipy import ndimage
from scipy import stats
import time
import importlib
# these are important to speed up the FFTs
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1.)

#from photutils import CircularAperture
#from photutils import make_source_mask
from photutils import Background2D, SigmaClip, MedianBackground

# for PSF fitting - see https://lmfit.github.io/lmfit-py/index.html
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit

from sip_to_pv import *

import resource
from skimage import restoration
import inpaint

# some global parameter settings
#telescope = 'kmtnet'
#telescope = 'meerlicht'
#telescope = 'omegacam'
telescope = 'kmtnet'

#KMTNet/OmegaWHITE
#if telescope=='kmtnet' or telescope=='omegacam' or telescope=='p48':
subimage_size = 1024     # size of subimages
subimage_border = 28     # border around subimage to avoid edge effects
#elif telescope=='meerlicht':
#MeerLICHT:
#subimage_size = 960      # size of subimages
#subimage_border = 32     # border around subimage to avoid edge effects
#trimmed MeerLICHT images:
#subimage_size = 950      # size of subimages
#subimage_border = 37     # border around subimage to avoid edge effects

# background estimation: these are the optional methods to estimate the
# backbround and its standard deviation (STD):
# (1) clipped median and STD of each object-masked subimage (simplest)
# (2) background and STD/RMS map determined by SExtractor (fastest)
# (3) improved background and STD map using masking of all sources (recommended)
# (4) similar to 3 but using photutils' Background2D (very very slow!)
bkg_method = 2           # background method to use
bkg_nsigma = 3           # data outside mean +- nsigma * stddev are
                         # clipped; used in methods 1, 3 and 4
bkg_boxsize = 128        # size of region used to determine
                         # background in methods 2, 3 and 4
bkg_filtersize = 5       # size of filter used for smoothing the above
                         # regions for method 2, 3 and 4

# ZOGY parameters
fratio_local = True      # determine fratio (Fn/Fr) from subimage (T) or full frame (F)
dxdy_local = False       # determine dx and dy from subimage (T) or full frame (F)
transient_nsigma = 5     # required significance in Scorr for transient detection

# optional fake stars
nfakestars = 1           # number of fake stars to be added to each subimage
                         # if 1: star will be at the center, if > 1: randomly distributed
fakestar_s2n = 50        # required signal-to-noise ratio of the fake stars    

# switch on/off different functions
dosex = False            # do extra SExtractor run (already done inside Astrometry.net)
dosex_psffit = False     # do extra SExtractor run with PSF fitting

# header keywords from which certain values are taken; these should be
# present in the header, but the names can be changed here
key_gain = 'GAIN'
key_ron = 'RDNOISE'
key_satlevel = 'SATURATE'
key_ra = 'RA'
key_dec = 'DEC'
key_pixscale = 'PIXSCALE'
key_exptime = 'EXPTIME'
key_seeing = 'SEEING'    # does not need to be present - is estimated
                         # using parameters below
#PTF:
#if telescope=='p48':
#key_ron = 'READNOI'
#key_satlevel = 'SATURVAL'
#key_ra = 'OBJRA'
#key_dec = 'OBJDEC'

# for seeing estimate
fwhm_imafrac = 0.25      # fraction of image area that will be used
                         # for initial seeing estimate
fwhm_detect_thresh = 10. # detection threshold for fwhm SExtractor run
fwhm_class_sort = False  # sort objects according to CLASS_STAR (T)
                         # or by FLUX_AUTO (F)
fwhm_frac = 0.25         # fraction of objects, sorted in brightness
                         # or class_star, used for fwhm estimate

# PSF parameters
use_single_psf = False   # use the same central PSF for all subimages
psf_clean_factor = 0     # pixels with values below (PSF peak * this
                         # factor) are set to zero; if this parameter
                         # is zero, no cleaning is done
psf_radius = 5           # PSF radius in units of FWHM used to build the PSF
                         # this determines the PSF_SIZE in psfex.config
                         # and size of the VIGNET in sex.params
psf_sampling = 2.0       # sampling factor used in PSFex - if zero, it
                         # is automatically determined for the new and
                         # ref image (~FWHM/4.5); if non-zero, it is
                         # fixed to the same sampling for both images

# Astrometry.net's tweak order
astronet_tweak_order = 3

# path and names of configuration files
cfg_dir = 'Config/'
sex_cfg = cfg_dir+'sex.config'     # SExtractor configuration file
sex_cfg_psffit = cfg_dir+'sex_psffit.config' # same for PSF-fitting version
sex_par = cfg_dir+'sex.params'     # SExtractor output parameters definition file
sex_par_psffit = cfg_dir+'sex_psffit.params' # same for PSF-fitting version
psfex_cfg = cfg_dir+'psfex.config' # PSFex configuration file
swarp_cfg = cfg_dir+'swarp.config' # SWarp configuration file

apphot_radii = [0.67, 1, 1.5, 2, 3, 5, 7, 10] # list of radii in units
                                              # of FWHM used for
                                              # aperture photometry in
                                              # SExtractor general

redo = True              # execute functions even if output file exist
verbose = True           # print out extra info
timing = True            # (wall-)time the different functions
display = False           # show intermediate fits images
make_plots = True        # make diagnostic plots and save them as pdf
show_plots = False        # show diagnostic plots


################################################################################

def optimal_subtraction(new_fits, ref_fits, ref_fits_remap=None, sub=None,
                        telescope=None, log=None, subpipe=False):
    
    """Function that accepts a new and a reference fits image, finds their
    WCS solution using Astrometry.net, runs SExtractor (inside
    Astrometry.net), PSFex to extract the PSF from the images, and
    performs Barak's optimal subtraction to produce the subtracted
    image (D), the significance image (S), and the corrected
    significance image (Scorr - see Zackay, Ofek & Gal-Yam 2016, ApJ,
    830, 27).

    Requirements:
    - Astrometry.net (in particular "solve-field" and index files)
    - SExtractor
    - SWarp
    - PSFex
    - ds9
    - sip_to_pv module from David Shupe: 
      https://github.com/stargaser/sip_tpv/blob/master/sip_to_pv.py
    - pyfftw to speed up the many FFTs performed
    - the other modules imported at the top
 
    Written by Paul Vreeswijk (pmvreeswijk@gmail.com) with vital input
    from Barak Zackay and Eran Ofek. Adapted by Kerry Paterson for
    integration into pipeline for MeerLICHT (ptrker004@myuct.ac.za).

    """

    start_time1 = os.times()

    if subpipe and telescope is not None:

        # In the case of a subpipe run (and telescope is defined), all
        # the global parameters below are taken from the subpipe
        # settings file (Constants) for a particular telescope rather
        # than their definitions at the top of this file. For the
        # parameter descriptions, see above.

        Constants = importlib.import_module('Utils.Constants_'+telescope)
        
        # make these global parameters
        global subimage_size, subimage_border, bkg_method, bkg_nsigma, bkg_boxsize, bkg_filtersize, fratio_local, dxdy_local, transient_nsigma, nfakestars, fakestar_s2n, dosex, dosex_psffit, pixelscale, fwhm_imafrac, fwhm_detect_thresh, fwhm_class_sort, fwhm_frac, use_single_psf, psf_clean_factor, psf_radius, psf_sampling, cfg_dir, sex_cfg, sex_cfg_psffit, sex_par, sex_par_psffit, psfex_cfg, swarp_cfg, apphot_radii, redo, verbose, timing, display, make_plots, show_plots

        subimage_size = Constants.Imager.subimage_size
        subimage_border = Constants.Imager.subimage_border

        bkg_method = Constants.bkg_method
        bkg_nsigma = Constants.bkg_nsigma
        bkg_boxsize = Constants.bkg_boxsize
        bkg_filtersize = Constants.bkg_filtersize

        fratio_local  = Constants.fratio_local
        dxdy_local = Constants.dxdy_local
        transient_nsigma = Constants.transient_nsigma

        nfakestars = Constants.nfakestars
        fakestar_s2n = Constants.fakestar_s2n

        dosex = Constants.dosex
        dosex_psffit = Constants.dosex_psffit

        # pixelscale - this parameter is not used anywhere below
        pixelscale = Constants.Imager.pixel_scale

        fwhm_imafrac = Constants.fwhm_imafrac
        fwhm_detect_thresh = Constants.fwhm_detect_thresh
        fwhm_class_sort = Constants.fwhm_class_sort
        fwhm_frac = Constants.fwhm_frac
        
        use_single_psf = Constants.use_single_psf
        psf_clean_factor = Constants.psf_clean_factor        
        psf_radius = Constants.psf_radius
        psf_sampling = Constants.psf_sampling

        cfg_dir = Constants.cfg_dir
        sex_cfg = Constants.sex_cfg
        sex_cfg_psffit = Constants.sex_cfg_psffit
        sex_par = Constants.sex_par
        sex_par_psffit = Constants.sex_par_psffit
        psfex_cfg = Constants.psfex_cfg
        swarp_cfg = Constants.swarp_cfg

        apphot_radii = Constants.apphot_radii

        redo = Constants.redo
        verbose = Constants.verbose
        timing = Constants.timing
        display = Constants.display
        make_plots = Constants.make_plots
        show_plots = Constants.show_plots

    # global parameters
    global base_new, base_ref
    global fwhm_new, fwhm_ref
        
    # define the base names of input fits files, base_new and
    # base_ref, as global so they can be used in any function in this
    # module
    if subpipe:
        # in case of subpipe, input images will have been WCS transformed
        # already and have '_wcs.fits' in the name
        base_new = new_fits.split('_wcs.fits')[0]
        base_ref = ref_fits.split('_wcs.fits')[0]
    else:
        base_new = new_fits.split('.fits')[0]
        base_ref = ref_fits.split('.fits')[0]
        
    # read in header of new_fits
    t = time.time()
    with fits.open(new_fits) as hdulist:
        header_new = hdulist[0].header
    keywords = ['NAXIS2', 'NAXIS1', key_gain, key_ron, key_satlevel,
                key_ra, key_dec, key_pixscale]
    ysize_new, xsize_new, gain_new, readnoise_new, satlevel_new, ra_new, dec_new, pixscale_new = read_header(header_new, keywords)
    if verbose:
        print keywords
        print read_header(header_new, keywords)

    # read in header of ref_fits
    with fits.open(ref_fits) as hdulist:
        header_ref = hdulist[0].header
    ysize_ref, xsize_ref, gain_ref, readnoise_ref, satlevel_ref, ra_ref, dec_ref, pixscale_ref = read_header(header_ref, keywords)
    if verbose:
        print keywords
        print read_header(header_ref, keywords)

    if not subpipe:    

        # in case of a subpipe run, the tasks below: running
        # Astrometry.net and remapping the ref image to the new image,
        # have already been done so this block can be skipped.
        
        # run SExtractor for seeing estimate of new_fits:
        sexcat_new = base_new+'.sexcat'
        fwhm_new, fwhm_std_new = run_sextractor(base_new+'.fits', sexcat_new, sex_cfg,
                                                sex_par, pixscale_new, fraction=fwhm_imafrac)
        print 'fwhm_new, fwhm_std_new', fwhm_new, fwhm_std_new
        print 'fwhm from header', header_new['SEEING']
        
        # write seeing (in arcseconds) to header
        #seeing_new = fwhm_new * pixscale_new
        #seeing_new_str = str('{:.2f}'.format(seeing_new))
        #header_new[key_seeing] = (seeing_new_str, '[arcsec] seeing estimated from central '+str(fwhm_imafrac))

        
        # TEST: add additional electrons to chunk of image to see the
        # reason for the shift between SExtractor background and those of
        # methods 3 and 4
        with fits.open(base_new+'.fits') as hdulist:
            header_test = hdulist[0].header
            data_test = hdulist[0].data
        data_test[:,0:512] += 50.
        fits.writeto(base_new+'.fits', data_test, header_test, clobber=True)
        
        
        # determine WCS solution of new_fits
        new_fits_wcs = base_new+'_wcs.fits'
        if not os.path.isfile(new_fits_wcs) or redo:
            result = run_wcs(base_new+'.fits', new_fits_wcs, ra_new, dec_new,
                             gain_new, readnoise_new, fwhm_new, pixscale_new, 'new')

        # run SExtractor for seeing estimate of ref_fits:
        sexcat_ref = base_ref+'.sexcat'
        fwhm_ref, fwhm_std_ref = run_sextractor(base_ref+'.fits', sexcat_ref, sex_cfg,
                                                sex_par, pixscale_ref, fraction=fwhm_imafrac)
        print 'fwhm_ref, fwhm_std_ref', fwhm_ref, fwhm_std_ref
        print 'fwhm from header', header_ref['SEEING']
        
        # write seeing (in arcseconds) to header
        #seeing_ref = fwhm_ref * pixscale_ref
        #seeing_ref_str = str('{:.2f}'.format(seeing_ref))
        #header_ref[key_seeing] = (seeing_ref_str, '[arcsec] seeing estimated from central '+str(fwhm_imafrac))

        # determine WCS solution of ref_fits
        ref_fits_wcs = base_ref+'_wcs.fits'
        if not os.path.isfile(ref_fits_wcs) or redo:
            result = run_wcs(base_ref+'.fits', ref_fits_wcs, ra_ref, dec_ref,
                             gain_ref, readnoise_ref, fwhm_ref, pixscale_ref, 'ref')


        # remap ref to new
        ref_fits_remap = base_ref+'_wcs_remap.fits'
        if not os.path.isfile(ref_fits_remap) or redo:
            result = run_remap(base_new+'_wcs.fits', base_ref+'_wcs.fits', ref_fits_remap,
                               [ysize_new, xsize_new], gain=gain_new, config=swarp_cfg)


    if subpipe:
        fwhm_new = header_new['SEEING']
        fwhm_ref = header_ref['SEEING']
            
    # initialize full output images
    data_D_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
    data_S_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
    data_Scorr_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
    data_Fpsf_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
    data_Fpsferr_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
    if nfakestars>0:
        data_new_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_ref_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        
    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = \
        centers_cutouts(subimage_size, ysize_new, xsize_new)

    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border
    nsubs = centers.shape[0]
    if verbose:
        print 'nsubs', nsubs
        for i in range(nsubs):
            print 'i', i
            print 'cuts_ima[i]', cuts_ima[i]
            print 'cuts_ima_fft[i]', cuts_ima_fft[i]
            print 'cuts_fft[i]', cuts_fft[i]
            
    # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with new,
    # ref, psf and background images

    data_new, psf_new, psf_orig_new, data_new_bkg, data_new_bkg_std = \
        prep_optimal_subtraction(base_new+'_wcs.fits', nsubs, 'new', fwhm_new)
    data_ref, psf_ref, psf_orig_ref, data_ref_bkg, data_ref_bkg_std = \
        prep_optimal_subtraction(base_ref+'_wcs.fits', nsubs, 'ref', fwhm_ref,
                                 remap=ref_fits_remap)


    # get x, y and fratios from matching PSFex stars across entire frame
    x_fratio, y_fratio, fratio, dra, ddec = get_fratio_radec(base_new+'_wcs.psfexcat',
                                                             base_ref+'_wcs.psfexcat',
                                                             base_new+'_wcs.sexcat',
                                                             base_ref+'_wcs.sexcat')
    
    dx = dra / pixscale_new
    dy = ddec / pixscale_new 

    # fratio is in counts, convert to electrons, in case gains of new
    # and ref images are not identical
    fratio *= gain_new / gain_ref
    
    dr = np.sqrt(dx**2 + dy**2)
    if verbose: print 'standard deviation dr over the full frame:', np.std(dr) 
    dr_full = np.sqrt(np.median(dr)**2 + np.std(dr)**2)
    dx_full = np.sqrt(np.median(dx)**2 + np.std(dx)**2)
    dy_full = np.sqrt(np.median(dy)**2 + np.std(dy)**2)
    #dr_full = np.std(dr)
    #dx_full = np.std(dx)
    #dy_full = np.sdata_new, psf_new, psf_orig_new, data_new_bkg, data_new_bkg_stdtd(dy)
    if verbose:
        print 'np.median(dr), np.std(dr)', np.median(dr), np.std(dr)
        print 'np.median(dx), np.std(dx)', np.median(dx), np.std(dx)
        print 'np.median(dy), np.std(dy)', np.median(dy), np.std(dy)
        print 'dr_full, dx_full, dy_full', dr_full, dx_full, dy_full
    
    #fratio_median, fratio_std = np.median(fratio), np.std(fratio)
    fratio_mean_full, fratio_std_full, fratio_median_full = clipped_stats(fratio, nsigma=2)
    if verbose:
        print 'fratio_mean_full, fratio_std_full, fratio_median_full', \
            fratio_mean_full, fratio_std_full, fratio_median_full
    
    if make_plots:
        # plot y vs x
        plt.axis((0,xsize_new,0,ysize_new))
        plt.plot(x_fratio, y_fratio, 'go') 
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dxdy.pdf')
        if show_plots: plt.show()
        plt.close()

        # plot dy vs dx
        plt.axis((-1,1,-1,1))
        plt.plot(dx, dy, 'go') 
        plt.xlabel('dx (pixels)')
        plt.ylabel('dy (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dxdy.pdf')
        if show_plots: plt.show()
        plt.close()
        
        # plot dr vs x_fratio
        plt.axis((0,xsize_new,0,1))
        plt.plot(x_fratio, dr, 'go')
        plt.xlabel('x (pixels)')
        plt.ylabel('dr (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('drx.pdf')
        if show_plots: plt.show()
        plt.close()

        # plot dr vs y_fratio
        plt.axis((0,ysize_new,0,1))
        plt.plot(y_fratio, dr, 'go')
        plt.xlabel('y (pixels)')
        plt.ylabel('dr (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dry.pdf')
        if show_plots: plt.show()
        plt.close()

        # plot dr as function of distance from the image center
        xcenter = xsize_new/2
        ycenter = ysize_new/2
        dist = np.sqrt((x_fratio-xcenter)**2 + (y_fratio-ycenter)**2)
        plt.axis((0,np.amax(dist),0,1))
        plt.plot(dist, dr, 'go')
        plt.xlabel('distance from image center (pixels)')
        plt.ylabel('dr (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('drdist.pdf')
        if show_plots: plt.show()
        plt.close()
                
        # plot dx vs x_fratio
        plt.axis((0,xsize_new,-1,1))
        plt.plot(x_fratio, dx, 'go')
        plt.xlabel('x (pixels)')
        plt.ylabel('dx (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dxx.pdf')
        if show_plots: plt.show()
        plt.close()

        # plot dy vs y_fratio
        plt.axis((0,ysize_new,-1,1))
        plt.plot(y_fratio, dy, 'go')
        plt.xlabel('y (pixels)')
        plt.ylabel('dy (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dyy.pdf')
        if show_plots: plt.show()
        plt.close()

    # initialize fakestar flux arrays if fake star(s) are being added
    # - this is to make a comparison plot of the input and output flux
    if nfakestars>0:
        fakestar_flux_input = np.ndarray(nsubs)
        fakestar_flux_output = np.ndarray(nsubs)
        fakestar_fluxerr_output = np.ndarray(nsubs)        
        fakestar_s2n_output = np.ndarray(nsubs)
        
    start_time2 = os.times()
            
    print '\nexecuting run_ZOGY on subimages ...'

    for nsub in range(nsubs):

        if timing: tloop = time.time()
        
        if verbose:
            print '\nNsub:', nsub+1
            print '----------'
            
        # refer to background and STD subimage with a shorter
        # parameter name
        bkg_new = data_new_bkg[nsub]
        bkg_ref = data_ref_bkg[nsub]
        std_new = data_new_bkg_std[nsub]
        std_ref = data_ref_bkg_std[nsub]

        # good place to make the corresponding variance images
        # N.B.: these are single images (i.e. not a cube) the size of
        # a subimage, so does not need the [nsub] index
        var_new = data_new[nsub] + readnoise_new**2 
        var_ref = data_ref[nsub] + readnoise_ref**2
        # alternative:
        #var_new = data_new[nsub] - bkg_new + std_new**2
        #var_ref = data_ref[nsub] - bkg_ref + std_ref**2
        
        if nfakestars>0:
            # add fake star(s) to new image
            if nfakestars==1:
                # place it at the center of the new subimage
                xpos = xsize_fft/2
                ypos = ysize_fft/2
                psf_hsize = psf_size_new/2
                index_temp = [slice(ypos-psf_hsize, ypos+psf_hsize+1),
                              slice(xpos-psf_hsize, xpos+psf_hsize+1)]
                # Use function [flux_optimal_s2n] to estimate flux needed
                # for star with S/N of [fakestar_s2n].
                fakestar_flux, fakestar_data = flux_optimal_s2n (psf_orig_new[nsub],
                                                                 data_new[nsub][index_temp],
                                                                 bkg_new[index_temp], readnoise_new,
                                                                 fakestar_s2n, fwhm=fwhm_new)
                # multiply psf_orig_new to contain fakestar_flux
                psf_fakestar = psf_orig_new[nsub] * fakestar_flux
                # add fake star to new image
                data_new[nsub][index_temp] += psf_fakestar
                # and variance image
                var_new[index_temp] += psf_fakestar
                
                if verbose:
                    print 'fakestar_flux: {} e-'.format(fakestar_flux)
                    flux, fluxerr, mask = flux_optimal(psf_orig_new[nsub], psf_orig_new[nsub],
                                                       fakestar_data, bkg_new[index_temp],
                                                       std_new[index_temp], readnoise_new)
                    print 'recovered flux, fluxerr, S/N', flux, fluxerr, flux/fluxerr
                
                    # check S/N with Eq. 51 from Zackay & Ofek 2017, ApJ, 836, 187
                    print 'S/N check', get_s2n_ZO(psf_orig_new[nsub], fakestar_data,
                                                  bkg_new[index_temp], readnoise_new)

                    # check S/N with Eqs. from Naylor (1998)
                    flux, fluxerr = get_optflux_Naylor(psf_orig_new[nsub], fakestar_data,
                                                       bkg_new[index_temp],
                                                       fakestar_data+readnoise_new**2)
                    print 'Naylor recovered flux, fluxerr, S/N', flux, fluxerr, flux/fluxerr
            

            else:
                # place stars in random positions across the subimage,
                # keeping subimage_border + psf_size_new/2 pixels off
                # each edge
                edge = subimage_border + psf_size_new/2 + 1
                xpos_rand = np.random.rand(nfakestars)*(xsize_fft-2*edge) + edge
                ypos_rand = np.random.rand(nfakestars)*(ysize_fft-2*edge) + edge
                for nstar in range(nfakestars):
                    xpos = np.int(xpos_rand[nstar])
                    ypos = np.int(ypos_rand[nstar])
                    psf_hsize = psf_size_new/2
                    index_temp = [slice(ypos-psf_hsize, ypos+psf_hsize+1),
                                  slice(xpos-psf_hsize, xpos+psf_hsize+1)]
                    fakestar_flux, fakestar_data = flux_optimal_s2n (psf_orig_new[nsub],
                                                                     data_new[nsub][index_temp],
                                                                     bkg_new[index_temp], readnoise_new,
                                                                     fakestar_s2n, fwhm=fwhm_new)
                    psf_fakestar = psf_orig_new[nsub] * fakestar_flux
                    data_new[nsub][index_temp] += psf_fakestar                
                    var_new[index_temp] += psf_fakestar
                
            # for plot of input vs. output flux; in case nfakestars >
            # 1, only the flux from the last one is recorded
            fakestar_flux_input[nsub] = fakestar_flux
                    
        # subtract the background
        data_new[nsub] -= bkg_new
        data_ref[nsub] -= bkg_ref

        # replace saturated pixel values with zero
        #data_new[nsub][data_new[nsub] > 0.95*satlevel_new] = 0.
        #data_ref[nsub][data_ref[nsub] > 0.95*satlevel_ref] = 0.

        # start with full-frame values
        fratio_mean, fratio_std, fratio_median = fratio_mean_full, fratio_std_full, fratio_median_full
        if fratio_local:
            # get median fratio from PSFex stars across subimage
            subcut = cuts_ima[nsub]
            # convert x,y_fratio pixel coordinates to indices
            y_fratio_index = (y_fratio-0.5).astype(int)
            x_fratio_index = (x_fratio-0.5).astype(int)
            # mask of entries in fratio arrays that lie within current subimage
            mask_sub_fratio = ((y_fratio_index >= subcut[0]) & (y_fratio_index < subcut[1]) & 
                               (x_fratio_index >= subcut[2]) & (x_fratio_index < subcut[3]))
            # require at least 10 values
            if np.sum(mask_sub_fratio) >= 10:
                # determine local fratios
                fratio_mean, fratio_std, fratio_median = clipped_stats(fratio[mask_sub_fratio], nsigma=2)
                if verbose:
                    print 'sub image fratios:', fratio[mask_sub_fratio]
                    
        # adopt full-frame values, also if local fratio_median is more
        # than 2 sigma (full frame) away from the full-frame value
        if not fratio_local or (np.abs(fratio_median-fratio_median_full)/fratio_std_full > 2.):
            fratio_mean, fratio_std, fratio_median = fratio_mean_full, fratio_std_full, fratio_median_full

        if verbose:
            print 'np.abs(fratio_median-fratio_median_full)/fratio_std_full', np.abs(fratio_median-fratio_median_full)/fratio_std_full
            print 'adopted fratio_mean, fratio_std, fratio_median', fratio_mean, fratio_std, fratio_median            
            
        # and the same for dx and dy
        if dxdy_local and any(mask_sub_fratio):
            dx_sub = np.sqrt(np.median(dx[mask_sub_fratio])**2 + np.std(dx[mask_sub_fratio])**2)
            dy_sub = np.sqrt(np.median(dy[mask_sub_fratio])**2 + np.std(dy[mask_sub_fratio])**2)
            if dx_sub > 2.*dx_full or not np.isfinite(dx_sub):
                dx_sub = dx_full
            if dy_sub > 2.*dy_full or not np.isfinite(dy_sub):
                dy_sub = dy_full
        else:
            dx_sub = dx_full
            dy_sub = dy_full

        # option 1: set f_ref to unity
        #f_ref = 1.
        #f_new = f_ref * fratio_median
        # option 2: set f_new to unity
        f_new = 1.
        f_ref = f_new / fratio_median
        if verbose:
            print 'f_new, f_ref', f_new, f_ref
            print 'dx_sub, dy_sub', dx_sub, dy_sub

        # test: put sharp source in new
        do_test = False
        if do_test:
            data_ref[nsub][:] = 0.
            data_new[nsub][:] = 0.
            data_new[nsub][xpos-1, ypos-1] = 1.
            data_new[nsub][xpos-1, ypos] = 1.
            data_new[nsub][xpos-1, ypos+1] = 1.
            data_new[nsub][xpos, ypos-1] = 1.
            data_new[nsub][xpos, ypos] = 3.
            data_new[nsub][xpos, ypos+1] = 1.
            data_new[nsub][xpos+1, ypos-1] = 1.
            data_new[nsub][xpos+1, ypos] = 1.
            data_new[nsub][xpos+1, ypos+1] = 1.
        
        # call Barak's function
        data_D, data_S, data_Scorr, data_Fpsf, data_Fpsferr = run_ZOGY(data_ref[nsub], data_new[nsub], 
                                                                       psf_ref[nsub], psf_new[nsub], 
                                                                       np.median(std_ref),
                                                                       np.median(std_new), 
                                                                       f_ref, f_new,
                                                                       var_ref, var_new,
                                                                       dx_sub, dy_sub)

        # check that robust std of Scorr is around unity
        if verbose:
            mean_Scorr, std_Scorr, median_Scorr = clipped_stats(data_Scorr, clip_zeros=False)
            print 'mean_Scorr, median_Scorr, std_Scorr', mean_Scorr, median_Scorr, std_Scorr
            mean_S, std_S, median_S = clipped_stats(data_S, clip_zeros=False)
            print 'mean_S, median_S, std_S', mean_S, median_S, std_S
            
        # if fake star(s) was (were) added to the subimages, compare
        # the input flux (the same for the entire subimage) with the
        # PSF flux determined by run_ZOGY. If multiple stars were
        # added, then this comparison is done for the last of them.
        if nfakestars>0:
            fakestar_flux_output[nsub] = data_Fpsf[xpos, ypos]
            fakestar_fluxerr_output[nsub] = data_Fpsferr[xpos, ypos]
            # and S/N from Scorr
            fakestar_s2n_output[nsub] = data_Scorr[xpos, ypos]
            
        # put sub images without the borders into output frames
        subcut = cuts_ima[nsub]
        index_subcut = [slice(subcut[0],subcut[1]), slice(subcut[2],subcut[3])]
        x1, y1 = subimage_border, subimage_border
        x2, y2 = x1+subimage_size, y1+subimage_size
        index_extract = [slice(y1,y2), slice(x1,x2)]

        data_D_full[index_subcut] = data_D[index_extract] / gain_new
        data_S_full[index_subcut] = data_S[index_extract]
        data_Scorr_full[index_subcut] = data_Scorr[index_extract]
        data_Fpsf_full[index_subcut] = data_Fpsf[index_extract]
        data_Fpsferr_full[index_subcut] = data_Fpsferr[index_extract]
        if nfakestars>0:
            data_new_full[index_subcut] = (data_new[nsub][index_extract] +
                                           bkg_new[index_extract]) / gain_new
            data_ref_full[index_subcut] = (data_ref[nsub][index_extract] +
                                           bkg_ref[index_extract]) / gain_ref
        

        if display and (nsub==0 or nsub == nsubs/2 or nsub==nsubs-1):

            # just for displaying purpose:
            fits.writeto('D.fits', data_D.astype(np.float32), clobber=True)
            fits.writeto('S.fits', data_S.astype(np.float32), clobber=True)
            fits.writeto('Scorr.fits', data_Scorr.astype(np.float32), clobber=True)
            fits.writeto('Scorr_abs.fits', np.abs(data_Scorr).astype(np.float32), clobber=True)
            #fits.writeto('Scorr_1sigma.fits', data_Scorr_1sigma, clobber=True)
        
            # write new and ref subimages to fits
            subname = '_sub'+str(nsub)
            newname = base_new+'_wcs'+subname+'.fits'
            #fits.writeto(newname, ((data_new[nsub]+bkg_new)/gain_new).astype(np.float32), clobber=True)
            fits.writeto(newname, data_new[nsub].astype(np.float32), clobber=True)
            refname = base_ref+'_wcs'+subname+'.fits'
            #fits.writeto(refname, ((data_ref[nsub]+bkg_ref)/gain_ref).astype(np.float32), clobber=True)
            fits.writeto(refname, data_ref[nsub].astype(np.float32), clobber=True)
            # variance images
            fits.writeto('Vnew.fits', var_new.astype(np.float32), clobber=True)
            fits.writeto('Vref.fits', var_ref.astype(np.float32), clobber=True)
            # background images
            fits.writeto('bkg_new.fits', bkg_new.astype(np.float32), clobber=True)
            fits.writeto('bkg_ref.fits', bkg_ref.astype(np.float32), clobber=True)
            
            
            # and display
            cmd = ['ds9','-zscale',newname,refname,'D.fits','S.fits','Scorr.fits']
            cmd = ['ds9','-zscale',newname,refname,'D.fits','S.fits','Scorr.fits',
                   'Vnew.fits', 'Vref.fits', 'bkg_new.fits', 'bkg_ref.fits',
                   'VSn.fits', 'VSr.fits', 'VSn_ast.fits', 'VSr_ast.fits',
                   'Sn.fits', 'Sr.fits', 'kn.fits', 'kr.fits', 'Pn_hat.fits', 'Pr_hat.fits',
                   'psf_ima_config_new_sub.fits', 'psf_ima_config_ref_sub.fits',
                   'psf_ima_resized_norm_new_sub.fits', 'psf_ima_resized_norm_ref_sub.fits', 
                   'psf_ima_center_new_sub.fits', 'psf_ima_center_ref_sub.fits', 
                   'psf_ima_shift_new_sub.fits', 'psf_ima_shift_ref_sub.fits']

            result = call(cmd)

        if timing: print 'wall-time spent in nsub loop', time.time()-tloop

    # find transient sources in Scorr
    #Scorr_peaks = ndimage.filters.maximum_filter(data_Scorr_full)
    #transient_nsigma = 5     # required significance in Scorr for transient detection
    #Scorr_peaks = Scorr_peaks[Scorr_peaks > transient_nsigma]
    #Scorr_peaks_mask = (data_Scorr_full == Scorr_peaks)
    # alternavitvely, use SExtractor:
    #fits.writeto('Scorr.fits', data_Scorr_full, clobber=True)
    #sex_trans_cfg = cfg_dir+'sex_trans.config'     # SExtractor configuration file
    #result = run_sextractor('Scorr.fits', 'trans.cat', sex_trans_cfg, sex_par, pixscale_new,
    #                        fwhm=fwhm_new)
    
    end_time = os.times()
    dt_usr  = end_time[2] - start_time2[2]
    dt_sys  = end_time[3] - start_time2[3]
    dt_wall = end_time[4] - start_time2[4]
    print
    print "Elapsed user time in {0}:  {1:.3f} sec".format("optsub", dt_usr)
    print "Elapsed CPU time in {0}:  {1:.3f} sec".format("optsub", dt_sys)
    print "Elapsed wall time in {0}:  {1:.3f} sec".format("optsub", dt_wall)
        
    dt_usr  = end_time[2] - start_time1[2]
    dt_sys  = end_time[3] - start_time1[3]
    dt_wall = end_time[4] - start_time1[4]
    print
    print "Elapsed user time in {0}:  {1:.3f} sec".format("total", dt_usr)
    print "Elapsed CPU time in {0}:  {1:.3f} sec".format("total", dt_sys)
    print "Elapsed wall time in {0}:  {1:.3f} sec".format("total", dt_wall)

    # write full new, ref, D and S images to fits
    if nfakestars>0:
        fits.writeto('new.fits', data_new_full, header_new, clobber=True)
        fits.writeto('ref.fits', data_ref_full, header_ref, clobber=True)
    if not subpipe:
        fits.writeto('D.fits', data_D_full, clobber=True)
        fits.writeto('S.fits', data_S_full, clobber=True)
        fits.writeto('Scorr.fits', data_Scorr_full, clobber=True)
        fits.writeto('Scorr_abs.fits', np.abs(data_Scorr_full), clobber=True)
        fits.writeto('Fpsf.fits', data_Fpsf_full, clobber=True)
        fits.writeto('Fpsferr.fits', data_Fpsferr_full, clobber=True)
    if subpipe:
        fits.writeto('D.fits', data_D_full, clobber=True)
        fits.writeto('S.fits', data_S_full, clobber=True)
        fits.writeto('Scorr.fits', data_Scorr_full, clobber=True)
        header_new.add_comment('Propagated header from new image to sub image.')
        fits.writeto(sub, np.abs(data_Scorr_full), header_new, clobber=True)
        fits.writeto('Fpsf.fits', data_Fpsf_full, clobber=True)
        fits.writeto('Fpsferr.fits', data_Fpsferr_full, clobber=True)
                
    # make comparison plot of flux input and output
    if make_plots and nfakestars>0:

        x = np.arange(nsubs)+1
        y = fakestar_flux_input
        plt.plot(x, y, 'ko')
        plt.xlabel('subimage number')
        plt.ylabel('true flux (e-)')
        plt.title('fake stars true input flux')
        plt.savefig('fakestar_flux_input.pdf')
        if show_plots: plt.show()
        plt.close()

        #plt.axis((0,nsubs,0,2))
        x = np.arange(nsubs)+1
        y = (fakestar_flux_input - fakestar_flux_output) / fakestar_flux_input
        yerr = fakestar_fluxerr_output / fakestar_flux_input
        plt.errorbar(x, y, yerr=yerr, fmt='ko')
        plt.xlabel('subimage number')
        plt.ylabel('(true flux - ZOGY flux) / true flux')
        plt.title('fake stars true input flux vs. ZOGY Fpsf output flux')
        plt.savefig('fakestar_flux_input_vs_ZOGYoutput.pdf')
        if show_plots: plt.show()
        plt.close()

        # same for S/N as determined by Scorr
        y = fakestar_s2n_output
        plt.plot(x, y, 'ko')
        plt.xlabel('subimage number')
        plt.ylabel('S/N from Scorr')
        plt.title('signal-to-noise ratio from Scorr')
        plt.savefig('fakestar_S2N_ZOGYoutput.pdf')
        if show_plots: plt.show()
        plt.close()
        
    # and display
    if nfakestars>0:
        cmd = ['ds9','-zscale','new.fits','ref.fits','D.fits','S.fits','Scorr.fits',
               'Fpsf.fits', 'Fpsferr.fits']
    else:
        cmd = ['ds9','-zscale',new_fits,ref_fits_remap,'D.fits','S.fits','Scorr.fits',
               'Fpsf.fits', 'Fpsferr.fits']
    result = call(cmd)


################################################################################

def get_optflux_xycoords (psfex_bintable, D, S, S_std, RON, xcoords, ycoords,
                          dx2, dy2, dxy, satlevel=60000,
                          psf_oddsized=True, psffit=False):
    
    """Function that returns the optimal flux and its error (using the
       function [flux_optimal] of a source at pixel positions
       [xcoords], [ycoords] given the inputs: .psf file produced by
       PSFex [psfex_bintable], data [D], sky [S], sky standard
       deviation [S_std] and read noise [RON]. [D], [S] and [RON] are
       assumed to be in electrons.
    
       [D] is a 2D array meant to be the full image. [S] can be a 2D
       array with the same shape as [D] or a scalar. [xcoords] and
       [ycoords] are arrays, and the output flux and its error will be
       arrays as well.
    
       The function will also return D(_replaced) where any saturated
       values in the PSF footprint of the [xcoords],[ycoords]
       coordinates that are being processed is replaced by the
       expected flux according to the PSF.

    """
        
    print '\nexecuting get_optflux_xycoords ...'
    t = time.time()

    # make sure x and y have same length
    if np.isscalar(xcoords) or np.isscalar(ycoords):
        print 'Error: xcoords and ycoords should be arrays'
    else:
        assert len(xcoords) == len(ycoords)
        
    # initialize output arrays
    ncoords = len(xcoords)
    flux_opt = np.zeros(ncoords)
    fluxerr_opt = np.zeros(ncoords)
    if psffit:
        flux_psf = np.zeros(ncoords)
        fluxerr_psf = np.zeros(ncoords)
        x_psf = np.zeros(ncoords)
        y_psf = np.zeros(ncoords)
        chi2_psf = np.zeros(ncoords)
        
    D_replaced = np.copy(D)
    
    # get dimensions of D
    ysize, xsize = np.shape(D)

    # get PSF images at x- and y-coordinates using function
    # [get_psf_xycoords]
    Pcube_noshift, Pcube_shift, xshift_array, yshift_array =\
        get_psf_xycoords (psfex_bintable, xcoords, ycoords, psf_oddsized=psf_oddsized)    

    # get psf_size from Pcube
    psf_size = np.shape(Pcube_noshift)[1]
    psf_hsize = psf_size/2
    
    # loop coordinates
    for i in range(ncoords):

        # extract data around position to use
        # indices of pixel in which [x],[y] is located
        # in case of odd-sized psf:
        if psf_oddsized:
            xpos = int(xcoords[i]-0.5)
            ypos = int(ycoords[i]-0.5)
        else:
            # in case of even-sized psf:
            xpos = int(xcoords[i])
            ypos = int(ycoords[i])

        # check if position is within image
        if ypos<0 or ypos>=ysize or xpos<0 or xpos>=xsize:
            #print 'Position x,y='+str(xpos)+','+str(ypos)+' outside image - skipping'
            continue
            
        # if PSF footprint is partially off the image, just go ahead
        # with the pixels on the image
        y1 = max(0, ypos-psf_hsize)
        x1 = max(0, xpos-psf_hsize)
        if psf_oddsized:
            y2 = min(ysize, ypos+psf_hsize+1)
            x2 = min(xsize, xpos+psf_hsize+1)
            # make sure axis sizes are odd
            if (x2-x1) % 2 == 0:
                if x1==0:
                    x2 -= 1
                else:
                    x1 += 1
            if (y2-y1) % 2 == 0:
                if y1==0:
                    y2 -= 1
                else:
                    y1 += 1
        else:
            y2 = min(ysize, ypos+psf_hsize)
            x2 = min(xsize, xpos+psf_hsize)
            # make sure axis sizes are even
            if (x2-x1) % 2 != 0:
                if x1==0:
                    x2 -= 1
                else:
                    x1 += 1
            if (y2-y1) % 2 != 0:
                if y1==0:
                    y2 -= 1
                else:
                    y1 += 1
        index = [slice(y1,y2),slice(x1,x2)]

        # extract subsection from D, S, and S_std
        D_sub = np.copy(D[index])
        if not np.isscalar(S):
            S_sub = S[index]
            S_std_sub = S_std[index]
        else:
            S_sub = S
            S_std_sub = S_std

        y1_P = y1 - (ypos - psf_hsize)
        x1_P = x1 - (xpos - psf_hsize)
        y2_P = y2 - (ypos - psf_hsize)
        x2_P = x2 - (xpos - psf_hsize)
        index_P = [slice(y1_P,y2_P),slice(x1_P,x2_P)]
        
        P_sub_shift = Pcube_shift[i][index_P]
        P_sub_noshift = Pcube_noshift[i][index_P]
        
        # could provide mask to flux_optimal, so that saturated pixels
        # can already be flagged, and flux_optimal could return a mask
        # indicating the pixels that were rejected

        # create mask of saturated pixels
        mask_sat = (D_sub >= satlevel)
        # add adjacent pixels to these
        mask_sat = ndimage.binary_dilation(mask_sat, structure=np.ones((3,3)).astype('bool'))
        # and its inverse
        mask_nonsat = ~mask_sat
        

        # call flux_optimal
        #print 'xcoords[i], ycoords[i]', xcoords[i], ycoords[i]
        #print 'dx2[i], dy2[i]', dx2[i], dy2[i]
        flux_opt[i], fluxerr_opt[i], mask_opt = flux_optimal (P_sub_shift, P_sub_noshift, D_sub,
                                                              S_sub, S_std_sub, RON, mask_use=mask_nonsat,
                                                              dx2=dx2[i], dy2=dy2[i], dxy=dxy[i])
        
        # if psffit=True, perform PSF fitting
        if psffit:
            flux_psf[i], fluxerr_psf[i], x_psf[i], y_psf[i], chi2_psf[i] =\
                flux_psffit (P_sub_noshift, D_sub, S_sub, RON, flux_opt[i],
                             xshift_array[i], yshift_array[i], mask_use=mask_nonsat)
            
        #print 'i, flux_opt[i], fluxerr_opt[i], flux_psf[i], fluxerr_psf[i]',\
        #    i, flux_opt[i], fluxerr_opt[i], flux_psf[i], fluxerr_psf[i]

        
        # if any pixels close to the center of the object are
        # saturated, replace them
        mask_inner = (P_sub_shift >= 0.25*np.amax(P_sub_shift))
        if np.any(mask_sat[mask_inner]):

            # replace saturated values
            D_replaced[index][mask_sat] = P_sub_shift[mask_sat] * flux_opt[i] + S_sub[mask_sat]

            # and put through [flux_optimal] once more without a
            # saturated pixel mask
            #flux_opt[i], fluxerr_opt[i], mask_opt = flux_optimal (P_sub_shift, P_sub_noshift, D_sub,
            #                                                      S_sub, S_std_sub, RON,
            #                                                      dx2=dx2[i], dy2=dy2[i], dxy=dxy[i])
            #D_replaced[index][mask_opt] = P_sub_shift[mask_opt] * flux_opt[i] + S_sub[mask_opt]

            if False and display:
                result = ds9_arrays(D_sub=D_sub, mask_nonsat=mask_nonsat.astype(int), 
                                    maskopt=mask_opt.astype(int),
                                    S_sub=S_sub, P_sub=P_sub_shift,
                                    D_replaced=D_replaced[index])
                
    if False and display:
        ds9_arrays(D=D, D_replaced=D_replaced)
    
    if psffit and make_plots:
        # compare xshift/yshift_array with psf xy shifts
        dx = xshift_array - x_psf
        dy = yshift_array - y_psf
        print 'np.median(dx), np.median(dy)', np.median(dx), np.median(dy)
        limits = (-2, 2, -2, 2)
        plt.axis(limits)
        plt.plot(dx, dy, 'ko')
        plt.xlabel('dx')
        plt.ylabel('dy')
        plt.show()
        plt.close()

    if timing: print 'wall-time spent in get_optflux_xycoords', time.time()-t

    if psffit:
        return flux_opt, fluxerr_opt, D_replaced, flux_psf, fluxerr_psf
    else:
        return flux_opt, fluxerr_opt, D_replaced
            

################################################################################

def flux_psffit(P, D, S, RON, flux_opt, xshift, yshift, mask_use=None):

    # if S is a scalar, expand it to 2D array
    if np.isscalar(S):
        S = np.ndarray(P.shape).fill(S)
        
    # replace negative values in D with the sky
    D[D<0] = S[D<0]

    # define objective function: returns the array to be minimized
    def fcn2min(params, P, D, S, RON, mask_use):

        xshift = params['xshift'].value
        yshift = params['yshift'].value
        flux_psf = params['flux_psf'].value

        # produce model by shifting and scaling psf
        #model = flux_psf * ndimage.shift(P, (yshift, xshift))
        # using Eran's shift function 
        model = flux_psf * image_shift_fft(P, xshift, yshift)
        
        err = np.sqrt(RON**2 + D)
        
        # residual
        resid = (D - S - model) / err
        
        #print 'xshift, yshift, flux_psf, chi2', xshift, yshift, flux_psf, np.sum(resid)
        #ds9_arrays(P=P, D=D, S=S, DminS=D-S, model=model, err=err, resid=resid)
        
        # return flattened array
        if mask_use is not None:
            return resid[mask_use].flatten()
        else:
            return resid.flatten()
        
    # create a set of Parameters
    params = Parameters()
    params.add('xshift', value=xshift, min=-2, max=2, vary=True)
    params.add('yshift', value=yshift, min=-2, max=2, vary=True)
    params.add('flux_psf', value=flux_opt)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(P, D, S, RON, mask_use))
    result = minner.minimize()

    chi2 = np.sum(fcn2min(result.params, P, D, S, RON, mask_use)**2)
    chi2_red = chi2 / D[mask_use].size

    return result.params['flux_psf'].value, result.params['flux_psf'].stderr,\
        result.params['xshift'].value, result.params['yshift'].value, chi2_red
    

################################################################################

def get_optflux (P, D, S, V):

    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], data [D], sky [S] and variance [V].  All are
    assumed to be in electrons rather than counts. These can be 1- or
    2-dimensional arrays with the same shape, while the sky can also
    be a scalar. See Horne 1986, PASP, 98, 609 and Naylor 1998, MNRAS,
    296, 339.

    """

    # and optimal flux and its error
    denominator = np.sum(P**2/V)
    optflux = np.sum((P*(D-S)/V)) / denominator
    optfluxerr = 1./np.sqrt(denominator)
    return optflux, optfluxerr


################################################################################

def get_optflux_Eran (P, P_noshift, D, S, V):

    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], the PSF shifted by the fractional pixel shift
    [P_shift], data [D], sky [S] and variance [V].  All are assumed to
    be in electrons rather than counts. These can be 1- or
    2-dimensional arrays with the same shape, while the sky can also
    be a scalar. See Eqs. 36 and 37 of Zackay & Ofek 2017, ApJ, 836,
    187.

    """

    # and optimal flux and its error
    denominator = np.sum((P_noshift*P)/V)
    optflux = np.sum((P_noshift*(D-S)/V)) / denominator
    optfluxerr = 1./np.sqrt(denominator)
    return optflux, optfluxerr

################################################################################

def get_optflux_Naylor (P, D, S, V):

    """Function that calculates signal-to-noise ratio using Eqs. 8, 10 and
    11 from Naylor 1998. All inputs are assumed to be in electrons
    rather than counts. These can be 1- or 2-dimensional lists, while
    the sky is also allowed to be a scalar.

    """

    # weights
    denominator = np.sum(P**2/V)
    W = (P/V) / denominator
    # and optimal flux and its error
    optflux = np.sum(W*(D-S))
    optfluxerr = np.sqrt(np.sum(W**2*V))
    return optflux, optfluxerr

################################################################################

def get_s2n_ZO (P, D, S, RON):

    """Function that calculates signal-to-noise ratio using Eq. 51 from
    Zackay & Ofek 2017, ApJ, 836, 187.  All inputs are assumed to be
    in electrons rather than counts. These can be 1- or 2-dimensional
    lists, while the sky is also allowed to be a scalar. 

    """

    T0 = np.sum(D-S)
    V = D+RON**2
    #s2n = np.sqrt(np.sum( (D-S)**2 / V ))
    s2n = np.sqrt(np.sum( (T0*P)**2 / V ))

    return s2n

################################################################################

def flux_optimal (P, P_noshift, D, S, S_std, RON, nsigma_inner=10,
                  nsigma_outer=5, max_iters=10, epsilon=1e-3, mask_use=None,
                  add_V_ast=False, dx2=0, dy2=0, dxy=0):
    
    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], data [D], sky [S], sky standard deviation [S_std]
    and read-out noise [RON].  This makes use of function
    [get_optflux] or [get_optflux_Eran]."""

    # if S is a scalar, expand it to 2D array
    if np.isscalar(S):
        S = np.ndarray(P.shape).fill(S)

    if add_V_ast:
        # calculate astrometric variance
        dDdy = D - np.roll(D,1,axis=0)
        dDdx = D - np.roll(D,1,axis=1)
        dDdxy = D - np.roll(D,1,axis=(0,1))
        V_ast = np.abs(dx2) * dDdx**2 + np.abs(dy2) * dDdy**2 + np.abs(dxy) * dDdxy**2
        
    # if input mask [mask_use] was not provided, create
    # it with same shape as D with all elements set to True.
    if mask_use is None: 
        mask_use = np.ones(D.shape, dtype=bool)
    # do not use any negative pixel values in D
    mask_use = ((mask_use) & (D>=0))
    
    # [mask_inner] - the central pixels within about 2xFWHM of the
    # object center (where P values are higher than 0.25 of the
    # central P value); [mask_outer] is the region outside that
    mask_inner = (P >= 0.25*np.amax(P))
    mask_outer = ~mask_inner
    mask_temp = np.ones(D.shape, dtype=bool)
    
    # loop
    flux_opt_old = float('inf')
    for i in range(max_iters):

        if i==0:
            # initial variance estimate (see Eq. 12 from Horne 1986)
            V = RON**2 + D
        else:
            # improved variance (see Eq. 13 from Horne 1986)
            V = RON**2 + S + flux_opt * P

        if add_V_ast:
            V += V_ast
                        
        # optimal flux
        #flux_opt, fluxerr_opt = get_optflux_Eran(P[mask], P_noshift[mask], D[mask], S[mask], V[mask])
        flux_opt, fluxerr_opt = get_optflux(P[mask_use], D[mask_use], S[mask_use], V[mask_use])
                    
        #print 'i, flux_opt, fluxerr_opt', i, flux_opt, fluxerr_opt, abs(flux_opt_old-flux_opt)/flux_opt

        if abs(flux_opt_old-flux_opt)/abs(flux_opt) < epsilon:
            break
        flux_opt_old = flux_opt

        # reject any discrepant values; use [nsigma_inner] as the
        # rejection criterium for the inner region defined by
        # [mask_inner]; outside of that use [nsigma_outer]
        sigma2 = (D - flux_opt * P - S)**2 / V

        mask_temp[mask_inner] = (sigma2[mask_inner] > nsigma_inner**2)
        mask_temp[mask_outer] = (sigma2[mask_outer] > nsigma_outer**2)
        mask_use[mask_temp] = False

    if False:
        print 'no. of rejected pixels', np.sum(mask_use==False)
        print 'np.amax((D - flux_opt * P - S)**2 / V)', np.amax(sigma2)
        
        if not add_V_ast:
            V_ast = np.zeros(D.shape)
            
        result = ds9_arrays(data=D, psf=P, sky=S, variance=V, fluxoptPsky = flux_opt*P+S,
                            data_min_fluxoptP_min_sky=(D - flux_opt * P - S),
                            data_min_fluxoptP_min_sky_squared_div_variance=sigma2,
                            mask_use=mask_use.astype(int), V_ast=V_ast)

    return flux_opt, fluxerr_opt, mask_use
    

################################################################################

def flux_optimal_s2n (P, D, S, RON, s2n, fwhm=5., max_iters=10, epsilon=1e-6):
    
    """Similar to function [flux_optimal] above, but this function returns
    the total flux sum(D-S) required for the point source to have a
    particular signal-to-noise ratio [s2n]. This function is used to
    estimate the flux of the fake stars that are being added to the
    image with a required S/N [fakestar_s2n].

    """

    # replace negative values in D with the sky
    if np.isscalar(S):
        D[D<0] = S
    else:
        D[D<0] = S[D<0]

    # keep a copy of original image
    D_orig = np.copy(D)
        
    for i in range(max_iters):
        if i==0:
            # initial estimate of variance (scalar)
            #V = RON**2 + D_orig
            V = RON**2 + S
            # and flux (see Eq. 13 of Naylor 1998)
            flux_opt = s2n * fwhm * np.sqrt(np.median(V)) / np.sqrt(2*np.log(2)/np.pi)
        else:
            # estimate new flux based on fluxerr_opt of previous iteration
            flux_opt = s2n * fluxerr_opt 
            # improved estimate of variance (2D list)
            V = RON**2 + S + flux_opt * P

        # new estimate of D
        #D = D_orig + flux_opt * P
        D = S + flux_opt * P

        # get optimal flux
        #flux_opt, fluxerr_opt = get_optflux(P, D, D_orig, V)
        flux_opt, fluxerr_opt = get_optflux(P, D, S, V)

        # break out of loop if S/N sufficiently close
        if abs(flux_opt/fluxerr_opt - s2n) / s2n < epsilon:
            break
        
    return flux_opt, D
    

################################################################################

def clipped_stats(array, nsigma=3, max_iters=10, epsilon=1e-6, clip_upper10=False,
                  clip_zeros=True, get_median=True, get_mode=False, mode_binsize=0.1,
                  verbose=False, show_hist=False):
    
    # remove zeros
    if clip_zeros:
        array = array[array.nonzero()]
        
    if clip_upper10:
        index_upper = int(0.9*array.size+0.5)
        array = np.sort(array.flatten())[:index_upper]

    mean_old = float('inf')
    for i in range(max_iters):
        mean = array.mean()
        std = array.std()
        if abs(mean_old-mean)/mean < epsilon:
            break
        mean_old = mean
        index = ((array>(mean-nsigma*std)) & (array<(mean+nsigma*std)))
        array = np.copy(array[index])
        
    # add median
    if get_median:
        median = np.median(array)
        if abs(median-mean)/mean>0.1:
            print 'Warning: mean and median in clipped_stats differ by more than 10%'
            print '   mean: ', mean
            print ' median: ', median
            
    # and mode
    if get_mode:
        bins = np.arange(np.int(np.amin(array)), np.int(np.amax(array)), 0.5)
        hist, bin_edges = np.histogram(array, bins)
        index = np.argmax(hist)
        mode = (bins[index]+bins[index+1])/2.
        if abs(mode-mean)/mean>0.1:
            print 'Warning: mean and mode in clipped_stats differ by more than 10%'

    if show_hist:
        bins = np.arange(np.int(np.amin(array)), np.int(np.amax(array)), 0.5)
        hist, bin_edges = np.histogram(array, bins)
        plt.hist(array, bins, color='green')
        x1,x2,y1,y2 = plt.axis()
        plt.plot([mean, mean], [y2,y1], color='red')
        plt.plot([mean+std, mean+std], [y2,y1], color='red', linestyle='--')
        plt.plot([mean-std, mean-std], [y2,y1], color='red', linestyle='--')
        if get_median:
            plt.plot([median, median], [y2,y1], color='magenta')
        if get_mode:
            plt.plot([mode, mode], [y2,y1], color='blue')
        if show_plots: plt.show()
        plt.close()
            
    if get_mode:
        if get_median:
            return mean, std, median, mode
        else:
            return mean, std, mode
    else:
        if get_median:
            return mean, std, median
        else:
            return mean, std
        
        
################################################################################

def read_header(header, keywords):

    values = []
    for i in range(len(keywords)):
        if keywords[i] in header:
            if verbose:
                print 'keywords[i], header[keywords[i]]', keywords[i], header[keywords[i]]
            values.append(header[keywords[i]])
        else:
            print 'Error: keyword', keywords[i], 'not present in header - change keyword name or add manually'
            raise SystemExit
    return values

################################################################################
    
def prep_optimal_subtraction(input_fits, nsubs, imtype, fwhm, remap=None):
    
    print '\nexecuting prep_optimal_subtraction ...'
    t = time.time()
    
    # read in input_fits
    with fits.open(input_fits) as hdulist:
        header_wcs = hdulist[0].header
        data_wcs = hdulist[0].data
    # if remapped image is provided, read that also
    if remap is not None:
        with fits.open(remap) as hdulist:
            header_remap = hdulist[0].header
            data_remap = hdulist[0].data

    # read original mask image (N.B.: this is different from the
    # objmask below which is a mask of the objects detected by
    # SExtractor)
    mask_fits = input_fits.replace('_wcs.fits', '_mask.fits')
    with fits.open(mask_fits) as hdulist:
        data_mask = hdulist[0].data
            
    # replace NANs with zero, and +-infinity with large +-numbers
    # data = np.nan_to_num(data)
    # get gain, readnoise and pixscale from header_wcs
    gain = header_wcs[key_gain]
    readnoise = header_wcs[key_ron]
    pixscale = header_wcs[key_pixscale]
    satlevel = header_wcs[key_satlevel]
    ysize, xsize = np.shape(data_wcs)
    # convert counts to electrons
    data_wcs *= gain
    if remap is not None:
        data_remap *= gain

    # ------------------------------
    # construction of background map
    # ------------------------------
        
    # fits filenames for background and std/RMS maps and object mask
    # as produced by SExtractor (i.e. in the case of the ref image
    # before remapping)
    if imtype=='new':
        base = base_new
    else:
        base = base_ref

    # in case of the reference image for subpipe, these parameters
    # should point to the images that have already been created at the
    # reference building stage - still to be implemented
    bkg_fits = base+'_bkg.fits'
    bkg_std_fits = base+'_bkg_std.fits'
    objmask_fits = base+'_objmask.fits'
    
    # read in SExtractor's object mask to use in background
    # estimation for methods 1,3 and 4.
    with fits.open(objmask_fits) as hdulist:
        data_objmask = hdulist[0].data
    
    # read in SExtractor's background and RMS/std maps which have
    # already been produced.    
    if bkg_method==2:
        with fits.open(bkg_fits) as hdulist:
            data_bkg = hdulist[0].data * gain
        with fits.open(bkg_std_fits) as hdulist:
            data_bkg_std = hdulist[0].data * gain

    # construct background image using [get_back]; in the case of
    # the reference image these data need to refer to the image
    # before remapping
    if bkg_method==3:
        data_bkg, data_bkg_std = get_back(data_wcs, data_objmask)

    # similar as above, but now photutils' Background2D is used
    # inside [get_back]
    if bkg_method==4:
        data_bkg, data_bkg_std = get_back(data_wcs, data_objmask,
                                          use_photutils=True)

    if imtype=='ref':
        # in case of the reference image, the background maps
        # produced above need to be projected to the coordinate
        # frame of the new or remapped reference image. At the
        # moment this is done with swarp, but this is a very slow
        # solution - try to improve.

        # update headers of the background and std/RMS fits image
        # with that of the original wcs-corrected reference image
        # for all background methods except 1
        if bkg_method!=1:
            fits.writeto(bkg_fits, (data_bkg/gain).astype(np.float32),
                         header=header_wcs, clobber=True)
            fits.writeto(bkg_std_fits, (data_bkg_std/gain).astype(np.float32),
                         header=header_wcs, clobber=True)
            # project ref image background maps to new image
            bkg_fits_remap = base_ref+'_bkg_remap.fits'
            if not os.path.isfile(bkg_fits_remap) or redo:
                result = run_remap(base_new+'_wcs.fits', bkg_fits, bkg_fits_remap,
                                   [ysize, xsize], gain=gain, config=swarp_cfg,
                                   resampling_type='NEAREST')
            bkg_std_fits_remap = base_ref+'_bkg_std_remap.fits'
            if not os.path.isfile(bkg_std_fits_remap) or redo:
                result = run_remap(base_new+'_wcs.fits', bkg_std_fits, bkg_std_fits_remap,
                                   [ysize, xsize], gain=gain, config=swarp_cfg,
                                   resampling_type='NEAREST')
            # and read back into array, replacing the previous arrays
            with fits.open(bkg_fits_remap) as hdulist:
                data_bkg = hdulist[0].data * gain
            with fits.open(bkg_std_fits_remap) as hdulist:
                data_bkg_std = hdulist[0].data * gain
        # only for method 1 the objmask needs to be projected
        else:
            fits.writeto(objmask_fits, data_objmask.astype(np.float32),
                         header=header_wcs, clobber=True)
            objmask_fits_remap = base_ref+'_objmask_remap.fits'
            if not os.path.isfile(objmask_fits_remap) or redo:
                result = run_remap(base_new+'_wcs.fits', objmask_fits, objmask_fits_remap,
                                   [ysize, xsize], gain=gain, config=swarp_cfg,
                                   resampling_type='NEAREST')
            with fits.open(objmask_fits_remap) as hdulist:
                data_objmask = hdulist[0].data

        # in any case, the ref mask image needs to be remapped
        # write the mask with the updated header to a temporary file
        fits.writeto('mask_ref_temp.fits', data_mask.astype(np.int),
                     header=header_wcs, clobber=True)
        # project ref image background maps to new image
        mask_fits_remap = base_ref+'_mask_remap.fits'
        if not os.path.isfile(mask_fits_remap) or redo:
            result = run_remap(base_new+'_wcs.fits', 'mask_ref_temp.fits', mask_fits_remap,
                               [ysize, xsize], gain=gain, config=swarp_cfg,
                               resampling_type='NEAREST')
        # read this remapped mask back into data_mask    
        with fits.open(mask_fits_remap) as hdulist:
            # remapping turns mask values into floats
            data_mask = (hdulist[0].data+0.5).astype(int)
            
    # let data refer to data_remap in case of ref image
    # and the original wcs-corrected image otherwise
    if remap is not None:
        data = data_remap
    else:
        data = data_wcs

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size, ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border

    # If [bkg_method]==1 (median) then make it here. For this method
    # in case of a reference image, the background and the STD/RMS
    # maps are determined from the remapped image directly. For the
    # other methods, they are determined from the original ref image
    # and subsequently mapped to the new image.
    if bkg_method==1:
        # initialize arrays
        data_bkg = np.zeros(data.shape)
        data_bkg_std = np.zeros(data.shape)
        # prepare mask_use based on data_objmask image built by
        # SExtractor, and remapped to the new image if needed
        mask_reject = ((data_objmask==0) | (data<=0))
        mask_use = ~mask_reject
        if verbose:
            print 'np.sum(mask_reject)', np.sum(mask_reject)

        for nsub in range(nsubs):
            subcut = cuts_ima[nsub]
            index_subcut = [slice(subcut[0],subcut[1]), slice(subcut[2],subcut[3])]
            # now determine background for method 1, where clipped median
            # of each subimage is used
            mask_use_sub = mask_use[index_subcut]
            mean, std, median = clipped_stats(data[index_subcut], nsigma=bkg_nsigma)
            if verbose:
                print '        nsub+1, mean, std, median', nsub+1, mean, std, median
            mean, std, median = clipped_stats(data[index_subcut][mask_use_sub], nsigma=bkg_nsigma)
            if verbose:
                print 'masked: nsub+1, mean, std, median', nsub+1, mean, std, median
            data_bkg[index_subcut] = median
            data_bkg_std[index_subcut] = std

    # In case of new image and background method other than 2, or
    # reference and background methods 3 and 4, write the background
    # and its RMS/STD arrays to fits. The units in these images are
    # ADU.
    # For subpipe this needs to be implemented such that the original
    # reference image background maps are not overwritten.
    if (imtype=='new' and bkg_method!=2) or (imtype=='ref' and bkg_method>2):
        bkg_fits = input_fits.replace('_wcs.fits', '_bkg.fits')
        fits.writeto(bkg_fits, (data_bkg/gain).astype(np.float32), clobber=True)
        bkg_std_fits = input_fits.replace('_wcs.fits', '_bkg_std.fits')
        fits.writeto(bkg_std_fits, (data_bkg_std/gain).astype(np.float32), clobber=True)
            
            
    # fix pixels using [fixpix] function which requires data (remapped
    # image in case, mask and background. Note
    # that in case background method 1 was selected, the background
    # used here is that determined by SExtractor
    data = fixpix (data, data_mask, data_bkg, satlevel=satlevel*gain)

    
    # determine psf of input image with get_psf function - needs to be
    # done before optimal fluxes are determined as the latter needs
    # the .psf output file from PSFex
    psf, psf_orig = get_psf(input_fits, header_wcs, nsubs, imtype, fwhm, pixscale)

    
    # -------------------------------
    # determination of optimal fluxes
    # -------------------------------

    # Get estimate of optimal flux for all sources in the new
    # image. For the reference image this should already have been
    # done when it was prepared.

    # For the reference image the [data] is read from the remapped
    # image, while the coordinates are from the original image, so to
    # make it work below temporarily, transform the coordinates
    # from the original reference image to the remapped image.

    if timing: t1 = time.time()
    print 'deriving optimal fluxes ...'
    
    # first read SExtractor fits table
    sexcat = input_fits.replace('.fits', '.sexcat')
    with fits.open(sexcat) as hdulist:
        data_sex = hdulist[2].data
    # read in positions and their errors
    xwin = data_sex['XWIN_IMAGE']
    ywin = data_sex['YWIN_IMAGE']    
    # skip coordinates outside the image
    #mask_use = ((xwin>0) & (xwin<(xsize+0.5)) & (ywin>0) & (ywin<(ysize+0.5)))
    xwin = data_sex['XWIN_IMAGE']#[mask_use]
    ywin = data_sex['YWIN_IMAGE']#[mask_use]
    errx2win = data_sex['ERRX2WIN_IMAGE']#[mask_use]
    erry2win = data_sex['ERRY2WIN_IMAGE']#[mask_use]
    errxywin = data_sex['ERRXYWIN_IMAGE']#[mask_use]
        
    if imtype == 'ref':
        # first infer ra, dec corresponding to x, y pixel positions in
        # the original ref image, using the .wcs file from
        # Astrometry.net
        wcs = WCS(base_ref+'.wcs')
        ra_temp, dec_temp = wcs.all_pix2world(xwin, ywin, 1)
        # then convert ra, dec back to x, y in the coordinate
        # frame of the new or remapped reference image
        wcs = WCS(base_new+'.wcs')
        xwin, ywin = wcs.all_world2pix(ra_temp, dec_temp, 1,
                                       tolerance=1e-3, adaptive=True,
                                       quiet=True)
        
    psfex_bintable = input_fits.replace('.fits', '.psf')

    fitpsf = False
    if fitpsf:
        flux_opt, fluxerr_opt, data_replaced, flux_psf, fluxerr_psf =\
            get_optflux_xycoords (psfex_bintable, data, data_bkg, data_bkg_std, readnoise,
                                  xwin, ywin, errx2win, erry2win, errxywin,
                                  satlevel=satlevel*gain, psffit=fitpsf)
    else:
        flux_opt, fluxerr_opt, data_replaced =\
            get_optflux_xycoords (psfex_bintable, data, data_bkg, data_bkg_std, readnoise,
                                  xwin, ywin, errx2win, erry2win, errxywin,
                                  satlevel=satlevel*gain)
        

    # uncomment this line to use image with saturated stars replaced
    # with psf estimate
    data = data_replaced

    # flux_opt is in e-, while flux_auto and flux_psf from
    # SExtractor catalog are in counts
    flux_opt /= gain
    fluxerr_opt /= gain
    if fitpsf:
        flux_psf /= gain
        fluxerr_psf /= gain
        
    # merge these two columns with sextractor catalog
    cols = [] 
    cols.append(fits.Column(name='FLUX_OPT', format='D', array=flux_opt))
    cols.append(fits.Column(name='FLUXERR_OPT', format='D', array=fluxerr_opt))
    if fitpsf:
        cols.append(fits.Column(name='FLUX_PSF', format='D', array=flux_psf))
        cols.append(fits.Column(name='FLUXERR_PSF', format='D', array=fluxerr_psf))
    orig_cols = data_sex.columns
    new_cols = fits.ColDefs(cols)
    hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)
    newcat = input_fits.replace('.fits', '.sexcat_fluxopt')
    hdu.writeto(newcat, clobber=True)
    # This fits table which includes the optimal fluxes still needs to
    # be converted to LDAC format although this is not really needed
    # anymore since PSFex is already finished. At this stage could
    # also remove the VIGNET entries in the SExtractor catalog which
    # take up a lot of space, but only for the NEW image.

    
    if timing: print 'wall-time spent deriving optimal fluxes', time.time()-t1

    # split full image into subimages to be used in run_ZOGY - this
    # needs to be done after determination of optimal fluxes as
    # otherwise the potential replacement of the saturated pixels will
    # not be taken into account

    # determine cutouts (already done above just before bkg_method 1
    # background is made)
    #centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size, ysize, xsize)
    #ysize_fft = subimage_size + 2*subimage_border
    #xsize_fft = subimage_size + 2*subimage_border

    fftdata = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    fftdata_bkg = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    fftdata_bkg_std = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    for nsub in range(nsubs):
        fftcut = cuts_fft[nsub]
        index_fft = [slice(fftcut[0],fftcut[1]), slice(fftcut[2],fftcut[3])]
        subcutfft = cuts_ima_fft[nsub]
        index_fftdata = [slice(subcutfft[0],subcutfft[1]), slice(subcutfft[2],subcutfft[3])]                       
        fftdata[nsub][index_fft] = data[index_fftdata]
        fftdata_bkg[nsub][index_fft] = data_bkg[index_fftdata]
        fftdata_bkg_std[nsub][index_fft] = data_bkg_std[index_fftdata]


    if make_plots:
        # compare flux_opt with flux_auto
        index = ((data_sex['FLUX_AUTO']>0) & (data_sex['FLAGS']==0))
        class_star = data_sex['CLASS_STAR'][index]
        flux_auto = data_sex['FLUX_AUTO'][index]
        fluxerr_auto = data_sex['FLUXERR_AUTO'][index]
        s2n_auto = flux_auto / fluxerr_auto
        flux_opt = flux_opt[index]
        fluxerr_opt = fluxerr_opt[index]
        if fitpsf:
            flux_mypsf = flux_psf[index]
            fluxerr_mypsf = fluxerr_psf[index]
        
        flux_diff = (flux_opt - flux_auto) / flux_auto
        fluxerr_diff = fluxerr_opt / flux_auto
        limits = (1,2*np.amax(s2n_auto),-0.2,0.2)
        plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                      xlabel='S/N (AUTO)', ylabel='(FLUX_OPT - FLUX_AUTO) / FLUX_AUTO', 
                      filename='fluxopt_vs_fluxauto_'+imtype+'.pdf',
                      title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

        if fitpsf:
            # compare flux_mypsf with flux_auto
            flux_diff = (flux_mypsf - flux_auto) / flux_auto
            fluxerr_diff = fluxerr_mypsf / flux_auto
            plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_MYPSF - FLUX_AUTO) / FLUX_AUTO', 
                          filename='fluxmypsf_vs_fluxauto_'+imtype+'.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
        
            # compare flux_opt with flux_mypsf
            flux_diff = (flux_opt - flux_mypsf) / flux_mypsf
            fluxerr_diff = fluxerr_opt / flux_mypsf
            plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_OPT - FLUX_MYPSF) / FLUX_MYPSF', 
                          filename='fluxopt_vs_fluxmypsf_'+imtype+'.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

        # compare flux_opt with flux_aper 2xFWHM
        for i in range(0,8):
            aper_str = str(apphot_radii[i])

            flux_aper = data_sex['FLUX_APER'][index,i]
            fluxerr_aper = data_sex['FLUXERR_APER'][index,i]
            flux_diff = (flux_opt - flux_aper) / flux_aper
            fluxerr_diff = fluxerr_opt / flux_aper
            plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_OPT - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                          filename='fluxopt_vs_fluxaper'+aper_str+'xFWHM_'+imtype+'.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            flux_diff = (flux_auto - flux_aper) / flux_aper
            fluxerr_diff = fluxerr_auto / flux_aper
            plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_AUTO - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                          filename='fluxauto_vs_fluxaper'+aper_str+'xFWHM_'+imtype+'.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            if fitpsf:
                flux_diff = (flux_mypsf - flux_aper) / flux_aper
                fluxerr_diff = fluxerr_mypsf / flux_aper
                plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                              xlabel='S/N (AUTO)', ylabel='(FLUX_MYPSF - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                              filename='fluxmypsf_vs_fluxaper'+aper_str+'xFWHM_'+imtype+'.pdf',
                              title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
            

        # compare with flux_psf if psffit catalog available
        if os.path.isfile(sexcat+'_psffit'):
            # read SExtractor psffit fits table
            with fits.open(sexcat+'_psffit') as hdulist:
                data_sex = hdulist[2].data
                
            flux_sexpsf = data_sex['FLUX_PSF'][index]
            fluxerr_sexpsf = data_sex['FLUXERR_PSF'][index]
            s2n_sexpsf = data_sex['FLUX_PSF'][index] / data_sex['FLUXERR_PSF'][index]
            
            flux_diff = (flux_sexpsf - flux_opt) / flux_opt
            fluxerr_diff = fluxerr_sexpsf / flux_opt
            plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_SEXPSF - FLUX_OPT) / FLUX_OPT', 
                          filename='fluxsexpsf_vs_fluxopt_'+imtype+'.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            if fitpsf:
                # and compare 'my' psf with SExtractor psf
                flux_diff = (flux_sexpsf - flux_mypsf) / flux_mypsf
                fluxerr_diff = fluxerr_sexpsf / flux_mypsf
                plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                              xlabel='S/N (AUTO)', ylabel='(FLUX_SEXPSF - FLUX_MYPSF) / FLUX_MYPSF', 
                              filename='fluxsexpsf_vs_fluxmypsf_'+imtype+'.pdf',
                              title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
            
            # and compare auto with psf
            flux_diff = (flux_sexpsf - flux_auto) / flux_auto
            fluxerr_diff = fluxerr_sexpsf / flux_auto
            plot_scatter (s2n_auto, flux_diff, fluxerr_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_SEXPSF - FLUX_AUTO) / FLUX_AUTO', 
                          filename='fluxsexpsf_vs_fluxauto_'+imtype+'.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

        
    if timing: print 'wall-time spent in prep_optimal_subtraction', time.time()-t

    return fftdata, psf, psf_orig, fftdata_bkg, fftdata_bkg_std
    

################################################################################

def fixpix (data, mask_in, bkg, satlevel=60000.):

    if timing: t = time.time()
    print '\nexecuting fixpix ...'

    # Replace pixels that correspond to zeros in data with the
    # background; this will ensure that the borders on the sides of
    # the entire image and the parts of the image where the new and
    # ref do not overlap can be handled by run_ZOGY.
    data[data==0] = bkg[data==0]

    # now try to clean the image from artificially sharp features such
    # as saturated and pixels as defined in data_mask - the FFTs in
    # [run_ZOGY] produce large-scale features surrounding these sharp
    # features in the subtracted image. Currently the (KMTNet) masks
    # do not seem to completely define all the bad pixels/columns
    # correctly - temporarily remake a mask here from negative or
    # saturated pixels
    mask = np.zeros(data.shape)
    mask[data < 0] = 1 
    mask[data >= satlevel] = 4

    # try just replacing the bad pixels with the background
    data_fixed = np.copy(data)
    mask_replace = ((mask==1))
    data_fixed[mask_replace] = bkg[mask_replace]

    # using restoration.inpaint_biharmonic
    # replace nonzero pixels with 1
    #mask[mask != 0] = 1
    # data values need to be between -1 and 1
    #norm = np.amax(np.abs(data))
    #data_fixed = restoration.inpaint_biharmonic(data/norm, mask, multichannel=False)
    #data_fixed *= norm

    # using inpaint.py from https://github.com/Technariumas/Inpainting
    #array : 2d np.ndarray - an array containing NaN elements that have to be replaced
    #max_iter : int - the number of iterations
    #kernel_size : int - the size of the kernel, default is 1
    #method : str - the method used to replace invalid values. Valid options are
    # `localmean`, 'idw'.
    # replace bad/saturated pixels with nans
    #data[mask != 0] = np.nan
    #data_fixed = inpaint.replace_nans(data, max_iter=5, kernel_radius=1,
    #                                  kernel_sigma=2, method='localmean')

    if timing: print 'wall-time spent in fixpix', time.time() - t

    #ds9_arrays(data=data, data_fixed=data_fixed)

    return data_fixed

        
################################################################################

def get_back (data, data_objmask, use_photutils=False, clip=True):
    
    """Function that returns the background of the image [data].  If
    use_photutils is True then apply the photutils' Background2D,
    while otherwise a clipped median is determined for each subimage
    which is masked using SExtractor's '-OBJECTS' image provided in
    [data_objmask]. The subimages (with size: [bkg_boxsize]) are then
    median filtered and resized to the size of the input image."""

    if timing: t = time.time()
    print '\nexecuting get_back ...'

    # masking using photutils
    #mask sources mask = make_source_mask(data, snr=2, npixels=5, dilate_size=11)

    # the above photutils masking process takes way too long for our
    # needs; use the SExtractor '-OBJECTS' image instead, which is a
    # (SExtractor) background-subtracted image with all pixels where
    # objects were detected set to zero (-OBJECTS)

    # mask all pixels with zeros in [data_objmask] or that have
    # non-positive pixel values in [data]
    mask_reject = ((data_objmask==0) | (data<=0))

    if use_photutils:
        t1 = time.time()
        # use the photutils Background2D function
        sigma_clip = SigmaClip(sigma=bkg_nsigma, iters=10)
        bkg_estimator = MedianBackground()
        # if bkg_boxsize does not fit integer times into the x- or
        # y-dimension of the shape, Background2D below fails if
        # edge_method='pad', which is the recommended method.  Use
        # edge_method='crop' instead.
        bkg = Background2D(data, bkg_boxsize, filter_size=bkg_filtersize,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                           mask=mask_reject, edge_method='crop')
        background, background_std = bkg.background, bkg.background_rms
        
    else:

        # mask to use (opposite of mask_zero)
        mask_use = ~mask_reject
        
        # determine clipped median and RMS/std in data with objects
        # masked
        if clip:
            # get clipped_stats mean, std and median 
            mean_full, std_full, median_full = clipped_stats(data[mask_use])
        else:
            median_full = np.median(data[mask_use])
            std_full = np.std(data[mask_use])
        if verbose:
            print 'Background median and std/RMS in object-masked image', median_full, std_full

        # loop through subimages the size of bkg_boxsize, and
        # determine median from the masked data
        ysize, xsize = data.shape
        centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(bkg_boxsize,
                                                                           ysize, xsize)        

        # loop subimages
        if ysize % bkg_boxsize != 0 or xsize % bkg_boxsize !=0:
            print 'Warning: [bkg_boxsize] does not fit integer times in image'
            print '         remaining pixels will be edge-padded'
        nysubs = ysize / bkg_boxsize
        nxsubs = xsize / bkg_boxsize
        # prepare output median and std output arrays
        mesh_median = np.ndarray((nysubs, nxsubs))
        mesh_std = np.ndarray((nysubs, nxsubs))
        nsub = -1
        mask_minsize = 0.5*bkg_boxsize**2
        for i in range(nxsubs):
            for j in range(nysubs):
                nsub += 1
                subcut = cuts_ima[nsub]
                data_sub = data[subcut[0]:subcut[1], subcut[2]:subcut[3]]
                mask_sub = mask_use[subcut[0]:subcut[1], subcut[2]:subcut[3]]
                if np.sum(mask_sub) > mask_minsize:
                    if clip:
                        # get clipped_stats mean, std and median 
                        mean, std, median = clipped_stats(data_sub[mask_sub], clip_upper10=True)
                    else:
                        median = np.median(data_sub[mask_sub])
                        std = np.std(data_sub[mask_sub])
                else:
                    # if less than half of the elements of mask_sub
                    # are True, use values from entire masked image
                    median, std = median_full, std_full
                    if verbose:
                        print 'Warning: using median and std of entire masked image for this background patch'
                        print '  nsub', nsub
                        print '  subcut', subcut
                        print '  np.sum(mask_sub) / bkg_boxsize**2', np.float(np.sum(mask_sub)) / bkg_boxsize**2
                        
                # fill median and std arrays
                mesh_median[j,i] = median
                mesh_std[j,i] = std

        # median filter the meshes with filter of size [bkg_filtersize]
        shape_filter = (bkg_filtersize, bkg_filtersize)
        mesh_median_filt = ndimage.filters.median_filter(mesh_median, shape_filter)
        mesh_std_filt = ndimage.filters.median_filter(mesh_std, shape_filter)

        # resize low-resolution meshes
        background = ndimage.zoom(mesh_median_filt, bkg_boxsize)
        background_std = ndimage.zoom(mesh_std_filt, bkg_boxsize)

        #ds9_arrays(data_objmask=data_objmask, mesh_median=mesh_median, mesh_median_filt=mesh_median_filt,
        #           background=background, background_std=background_std)
        
        # if shape of the background is not equal to input [data]
        # then pad the background images
        if data.shape != background.shape:
            t1 = time.time()
            ypad = ysize - background.shape[0]
            xpad = xsize - background.shape[1]
            background = np.pad(background, ((0,ypad),(0,xpad)), 'edge')
            background_std = np.pad(background_std, ((0,ypad),(0,xpad)), 'edge')                   
            print 'time to pad', time.time()-t1
            #ds9_arrays(data=data, data_objmask=data_objmask,
            #           background=background, background_std=background_std)
            #np.pad seems quite slow; alternative:
            #centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(bkg_boxsize,
            #                                                                   ysize, xsize,
            #                                                                   get_remainder=True)
            # these now include the remaining patches
                        
            
    if timing: print 'wall-time spent in get_back', time.time() - t

    return background, background_std
    

################################################################################

def plot_scatter (x, y, yerr, limits, corder, cmap='rainbow_r', symbol='o',
                  xlabel='', ylabel='', legendlabel='', title='', filename='',
                  simple=False):

    plt.axis(limits)
    #xplt.errorbar(x, y, yerr=yerr, linestyle="None", color='k')
    plt.scatter(x, y, c=corder, cmap=cmap, alpha=0.75, label=legendlabel)
    plt.xscale('log')
    if legendlabel!='':
        plt.legend(numpoints=1, fontsize='medium')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename != '':
        plt.savefig(filename)
    if show_plots: plt.show()
    plt.close()

################################################################################

def get_psf(image, ima_header, nsubs, imtype, fwhm, pixscale):

    """Function that takes in [image] and determines the actual Point
    Spread Function as a function of position from the full frame, and
    returns a cube containing the psf for each subimage in the full
    frame.

    """

    global psf_size_new
    
    if timing: t = time.time()
    print '\nexecuting get_psf ...'

    # determine image size from header
    xsize, ysize = ima_header['NAXIS1'], ima_header['NAXIS2']
    
    # run sextractor on image; this step is no longer needed as it is
    # done inside Astrometry.net, producing the same catalog as an
    # independent SExtractor run would.
    sexcat = image.replace('.fits', '.sexcat')
    if (not os.path.isfile(sexcat) or redo) and dosex:
        result = run_sextractor(image, sexcat+'_alt', sex_cfg, sex_par, pixscale, fwhm=fwhm)
        
    # run psfex on SExtractor output catalog
    psfexcat = image.replace('.fits', '.psfexcat')
    #if not os.path.isfile(psfexcat) or redo:
    print 'sexcat', sexcat
    print 'psfexcat', psfexcat
    result = run_psfex(sexcat, psfex_cfg, psfexcat, fwhm, imtype)

    # again run SExtractor, but now using output PSF from PSFex, so
    # that PSF-fitting can be performed for all objects. The output
    # columns defined in [sex_par_psffit] include several new columns
    # related to the PSF fitting.
    if (not os.path.isfile(sexcat+'_psffit') or redo) and dosex_psffit:
        result = run_sextractor(image, sexcat+'_psffit', sex_cfg_psffit,
                                sex_par_psffit, pixscale, fitpsf=True, fwhm=fwhm)
        
    # read in PSF output binary table from psfex
    psfex_bintable = image.replace('.fits', '.psf')
    with fits.open(psfex_bintable) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data[0][0][:]

    # read in some header keyword values
    polzero1 = header['POLZERO1']
    polzero2 = header['POLZERO2']
    polscal1 = header['POLSCAL1']
    polscal2 = header['POLSCAL2']
    poldeg = header['POLDEG1']
    psf_fwhm = header['PSF_FWHM']
    psf_samp = header['PSF_SAMP']
    # [psf_size_config] is the size of the PSF as defined in the PSFex
    # configuration file ([PSF_SIZE] parameter), which is the same as
    # the size of the [data] array
    psf_size_config = header['PSFAXIS1']
    if verbose:
        print 'polzero1                   ', polzero1
        print 'polscal1                   ', polscal1
        print 'polzero2                   ', polzero2
        print 'polscal2                   ', polscal2
        print 'order polynomial:          ', poldeg
        print 'PSF FWHM:                  ', psf_fwhm
        print 'PSF sampling size (pixels):', psf_samp
        print 'PSF size defined in config:', psf_size_config
        
    # call centers_cutouts to determine centers
    # and cutout regions of the full image
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size, ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border

    if imtype == 'ref':

        # in case of the ref image, the PSF was determined from the
        # original image, while it will be applied to the remapped ref
        # image. So the centers of the cutouts in the remapped ref
        # image need to be mapped back to those in the original
        # reference image to get the PSF from the proper
        # coordinates. Easiest to do this using astropy.wcs, which
        # would also take care of any potential rotation and scaling.

        # first infer ra, dec corresponding to x, y pixel positions
        # (centers[:,1] and centers[:,0], respectively, using the
        # [new].wcs file from Astrometry.net
        wcs = WCS(base_new+'.wcs')
        ra_temp, dec_temp = wcs.all_pix2world(centers[:,1], centers[:,0], 1)
        # then convert ra, dec back to x, y in the original ref image
        wcs = WCS(base_ref+'.wcs')
        centers[:,1], centers[:,0] = wcs.all_world2pix(ra_temp, dec_temp, 1)
        
    # initialize output PSF array

    # [psf_size] is the PSF size in image pixels,
    # i.e. [psf_size_config] multiplied by the PSF sampling (roughly
    # 4-5 pixels per FWHM) which is set by the [psf_sampling] parameter.
    # If set to zero, it is automatically determined by PSFex.
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    # if this is even, make it odd
    if psf_size % 2 == 0:
        psf_size += 1
    if verbose:
        print 'fwhm                      :', fwhm
        print 'final image PSF size      :', psf_size
    # now change psf_samp slightly:
    psf_samp_update = float(psf_size) / float(psf_size_config)
    if imtype == 'new': psf_size_new = psf_size
    # [psf_ima] is the corresponding cube of PSF subimages
    psf_ima = np.zeros((nsubs,psf_size,psf_size), dtype='float32')
    # [psf_ima_center] is [psf_ima] broadcast into images of xsize_fft
    # x ysize_fft
    psf_ima_center = np.zeros((nsubs,ysize_fft,xsize_fft), dtype='float32')
    # [psf_ima_shift] is [psf_ima_center] shifted - this is
    # the input PSF image needed in the zogy function
    psf_ima_shift = np.zeros((nsubs,ysize_fft,xsize_fft), dtype='float32')
    
    # loop through nsubs and construct psf at the center of each
    # subimage, using the output from PSFex that was run on the full
    # image
    for nsub in range(nsubs):
        
        x = (centers[nsub,1] - polzero1) / polscal1
        y = (centers[nsub,0] - polzero2) / polscal2

        if nsubs==1 or use_single_psf:
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + \
                          data[3] * y + data[4] * x * y + data[5] * y**2
            elif poldeg==3:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 + \
                          data[4] * y + data[5] * x * y + data[6] * x**2 * y + \
                          data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3

        # resample PSF image at image pixel scale
        psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update)
        # clean and normalize PSF
        psf_ima_resized_norm = clean_norm_psf(psf_ima_resized, psf_clean_factor)

        psf_ima[nsub] = psf_ima_resized_norm
        if verbose and nsub==0:
            print 'psf_samp, psf_samp_update', psf_samp, psf_samp_update
            print 'np.shape(psf_ima_config)', np.shape(psf_ima_config)
            print 'np.shape(psf_ima)', np.shape(psf_ima)
            print 'np.shape(psf_ima_resized)', np.shape(psf_ima_resized)
            print 'psf_size ', psf_size
            
        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            print 'WARNING: image not even in one or both dimensions!'
            
        xcenter_fft, ycenter_fft = xsize_fft/2, ysize_fft/2
        if verbose and (nsub==0 or nsub==nsubs-1):
            print 'xcenter_fft, ycenter_fft ', xcenter_fft, ycenter_fft

        psf_hsize = psf_size/2
        index = [slice(ycenter_fft-psf_hsize, ycenter_fft+psf_hsize+1), 
                 slice(xcenter_fft-psf_hsize, xcenter_fft+psf_hsize+1)]
        psf_ima_center[nsub][index] = psf_ima_resized_norm

        # perform fft shift
        psf_ima_shift[nsub] = fft.fftshift(psf_ima_center[nsub])

        if display:
            fits.writeto('psf_ima_config_'+imtype+'_sub.fits', psf_ima_config, clobber=True)
            fits.writeto('psf_ima_resized_norm_'+imtype+'_sub.fits',
                         psf_ima_resized_norm.astype(np.float32), clobber=True)
            fits.writeto('psf_ima_center_'+imtype+'_sub.fits',
                         psf_ima_center[nsub].astype(np.float32), clobber=True)            
            fits.writeto('psf_ima_shift_'+imtype+'_sub.fits',
                         psf_ima_shift[nsub].astype(np.float32), clobber=True)            

    #ds9_arrays(psf_ima_config=psf_ima_config, psf_ima_resized_norm=psf_ima_resized_norm)
            
    if timing: print 'wall-time spent in get_psf', time.time() - t

    return psf_ima_shift, psf_ima

################################################################################

def get_psf_xycoords(psfex_bintable, xcoords, ycoords, psf_oddsized=True, order=3):

    """Function that takes in .psf file produced by PSFex and returns a
    cube containing the original PSF and the shifted PSF at the
    coordinate arrays [x], [y]

    """

    if timing: t = time.time()
    print '\nexecuting get_psf_xycoords ...'

    # number of coordinates
    ncoords = len(xcoords)

    # read in PSF output binary table from psfex
    with fits.open(psfex_bintable) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data[0][0][:]

    # read in some header keyword values
    polzero1 = header['POLZERO1']
    polzero2 = header['POLZERO2']
    polscal1 = header['POLSCAL1']
    polscal2 = header['POLSCAL2']
    poldeg = header['POLDEG1']
    psf_fwhm = header['PSF_FWHM']
    psf_samp = header['PSF_SAMP']
    # [psf_size_config] is the size of the PSF grid as defined in the
    # PSFex configuration file ([PSF_SIZE] parameter)
    psf_size_config = header['PSFAXIS1']
    if verbose:
        print 'polzero1                   ', polzero1
        print 'polscal1                   ', polscal1
        print 'polzero2                   ', polzero2
        print 'polscal2                   ', polscal2
        print 'order polynomial:          ', poldeg
        print 'PSF FWHM:                  ', psf_fwhm
        print 'PSF sampling size (pixels):', psf_samp
        print 'PSF size defined in config:', psf_size_config
        
    # initialize output PSF array

    # [psf_size] is the PSF size in image pixels,
    # i.e. [psf_size_config] multiplied by the PSF sampling (roughly
    # 4-5 pixels per FWHM) which is set by the [psf_sampling] parameter.
    # If set to zero, it is automatically determined by PSFex.
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    # depending on [psf_oddsized], make the psf size odd or even
    if psf_oddsized:
        if psf_size % 2 == 0:
            psf_size += 1
    else:
        if psf_size % 2 != 0:
            psf_size += 1
    # now change psf_samp slightly:
    psf_samp_update = float(psf_size) / float(psf_size_config)

    # [psf_ima] is the corresponding cube of PSF subimages
    psf_cube_shift = np.ndarray((ncoords,psf_size,psf_size), dtype='float32')
    psf_cube_noshift = np.ndarray((ncoords,psf_size,psf_size), dtype='float32')
    xshift_array = np.zeros(ncoords)
    yshift_array = np.zeros(ncoords)
    
    # loop through coordinates and construct psf
    for i in range(ncoords):

        x = (int(xcoords[i]) - polzero1) / polscal1
        y = (int(ycoords[i]) - polzero2) / polscal2
        
        if ncoords==1 or use_single_psf:
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + \
                                 data[3] * y + data[4] * x * y + data[5] * y**2
            elif poldeg==3:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 + \
                                 data[4] * y + data[5] * x * y + data[6] * x**2 * y + \
                                 data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3

        # shift to the subpixel center of the object (object at
        # fractional pixel position 0.5,0.5 doesn't need the PSF to
        # shift as the PSF image is constructed to be even
        if psf_oddsized:
            xshift = xcoords[i]-np.round(xcoords[i])
            yshift = ycoords[i]-np.round(ycoords[i])
        else:
            xshift = (xcoords[i]-int(xcoords[i])-0.5)
            yshift = (ycoords[i]-int(ycoords[i])-0.5)
        xshift_array[i] = xshift
        yshift_array[i] = yshift
                    
        # if [psf_samp_update] is lower than unity, then perform this
        # shift before the PSF image is re-sampled to the image
        # pixels, as the original PSF will have higher resolution in
        # that case
        if psf_samp_update < 1:
            # multiply with PSF sampling to get shift in units of image
            # pixels
            xshift *= psf_samp_update
            yshift *= psf_samp_update
            # shift PSF
            psf_ima_shift = ndimage.shift(psf_ima_config, (yshift, xshift), order=order)
            # using Eran's function:
            #psf_ima_shift = image_shift_fft(psf_ima_config, xshift, yshift)
            # resample PSF image at image pixel scale
            psf_ima_shift_resized = ndimage.zoom(psf_ima_shift, psf_samp_update, order=order)
            # also resample non-shifted PSF image at image pixel scale
            psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update, order=order)
        else:
            # resample PSF image at image pixel scale
            psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update, order=order)
            # shift PSF
            psf_ima_shift_resized = ndimage.shift(psf_ima_resized, (yshift, xshift), order=order)
            # using Eran's function:
            #psf_ima_shift_resized = image_shift_fft(psf_ima_resized, xshift, yshift)

        # clean and normalize PSF
        psf_cube_shift[i] = clean_norm_psf(psf_ima_shift_resized, psf_clean_factor)

        # also return normalized PSF without any shift
        psf_cube_noshift[i] = clean_norm_psf(psf_ima_resized, psf_clean_factor)
        
    if timing: print 'wall-time spent in get_psf_xycoords', time.time() - t

    return psf_cube_noshift, psf_cube_shift, xshift_array, yshift_array

################################################################################

def get_fratio_radec(psfcat_new, psfcat_ref, sexcat_new, sexcat_ref, use_flux_opt=True):

    """Function that takes in output catalogs of stars used in the PSFex
    runs on the new and the ref image, and returns the arrays with
    pixel coordinates (!) x, y (in the new frame) and fratios for the
    matching stars. In addition, it provides the difference in stars'
    RAs and DECs in arcseconds between the two catalogs.

    """
    
    t = time.time()
    print '\nexecuting get_fratio_radec ...'
    
    def readcat (psfcat):
        table = ascii.read(psfcat, format='sextractor')
        number = table['SOURCE_NUMBER']
        x = table['X_IMAGE']
        y = table['Y_IMAGE']
        norm = table['NORM_PSF']
        return number, x, y, norm
        
    # read psfcat_new
    number_new, x_new, y_new, norm_new = readcat(psfcat_new)
    # read psfcat_ref
    number_ref, x_ref, y_ref, norm_ref = readcat(psfcat_ref)

    def xy2radec (number, sexcat):
        # read SExtractor fits table
        with fits.open(sexcat) as hdulist:
            data = hdulist[2].data
            ra_sex = data['ALPHAWIN_J2000']
            dec_sex = data['DELTAWIN_J2000']
        # loop numbers and record in ra, dec
        ra = []
        dec = []
        for n in number:
            ra.append(ra_sex[n-1])
            dec.append(dec_sex[n-1])      
        return np.array(ra), np.array(dec)
    
    # get ra, dec corresponding to x, y
    ra_new, dec_new = xy2radec(number_new, sexcat_new)
    ra_ref, dec_ref = xy2radec(number_ref, sexcat_ref)

    # now find matching entries
    x_new_match = []
    y_new_match = []
    dra_match = []
    ddec_match = []
    fratio = []
    nmatch = 0
    for i_new in range(len(x_new)):
        # calculate distance to ref objects
        dra = 3600.*(ra_new[i_new]-ra_ref)*np.cos(dec_new[i_new]*np.pi/180.)
        ddec = 3600.*(dec_new[i_new]-dec_ref)
        dist = np.sqrt(dra**2 + ddec**2)
        # minimum distance and its index
        dist_min, i_ref = np.amin(dist), np.argmin(dist)
        if dist_min < 3.:
            nmatch += 1
            x_new_match.append(x_new[i_new])
            y_new_match.append(y_new[i_new])
            dra_match.append(dra[i_ref])
            ddec_match.append(ddec[i_ref])
            # append ratio of normalized counts to fratios
            fratio.append(norm_new[i_new] / norm_ref[i_ref])
                        
    if verbose:
        print 'fraction of PSF stars that match', float(nmatch)/len(x_new)
            
    if timing: print 'wall-time spent in get_fratio_radec', time.time()-t

    return np.array(x_new_match), np.array(y_new_match), np.array(fratio), \
        np.array(dra_match), np.array(ddec_match)

################################################################################

def centers_cutouts(subsize, ysize, xsize, get_remainder=False):
    
    """Function that determines the input image indices (!) of the centers
    (list of nsubs x 2 elements) and cut-out regions (list of nsubs x
    4 elements) of image with the size xsize x ysize. Subsize is the
    fixed size of the subimages, e.g. 512 or 1024. The routine will
    fit as many of these in the full frames, and for the moment it
    will ignore any remaining pixels outside."""
    
    nxsubs = xsize / subsize
    nysubs = ysize / subsize
    if get_remainder:
        if xsize % subsize != 0:
            nxsubs += 1
            remainder_x = True
        else:
            remainder_x = False
        if ysize % subsize != 0:
            nysubs += 1
            remainder_y = True
        else:
            remainder_y = False

    nsubs = nxsubs * nysubs
    print 'nxsubs, nysubs, nsubs', nxsubs, nysubs, nsubs

    centers = np.ndarray((nsubs, 2), dtype=int)
    cuts_ima = np.ndarray((nsubs, 4), dtype=int)
    cuts_ima_fft = np.ndarray((nsubs, 4), dtype=int)
    cuts_fft = np.ndarray((nsubs, 4), dtype=int)
    sizes = np.ndarray((nsubs, 2), dtype=int)

    ysize_fft = subsize + 2*subimage_border
    xsize_fft = subsize + 2*subimage_border
        
    nsub = -1
    for i in range(nxsubs): 
        nx = subsize
        if get_remainder and i == nxsubs-1 and remainder_x:
            nx = xsize % subsize
        for j in range(nysubs):
            ny = subsize
            if get_remainder and j == nysubs-1 and remainder_y:
                ny = ysize % subsize
            x = i*subsize + nx/2
            y = j*subsize + ny/2
            nsub += 1
            centers[nsub] = [y, x]
            cuts_ima[nsub] = [y-ny/2, y+ny/2, x-nx/2, x+nx/2]
            y1 = np.amax([0,y-ny/2-subimage_border])
            x1 = np.amax([0,x-nx/2-subimage_border])
            y2 = np.amin([ysize,y+ny/2+subimage_border])
            x2 = np.amin([xsize,x+nx/2+subimage_border])
            cuts_ima_fft[nsub] = [y1,y2,x1,x2]
            cuts_fft[nsub] = [y1-(y-ny/2-subimage_border),ysize_fft-(y+ny/2+subimage_border-y2),
                              x1-(x-nx/2-subimage_border),xsize_fft-(x+nx/2+subimage_border-x2)]
            sizes[nsub] = [ny, nx]
            
    return centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes

################################################################################

def show_image(image):

    im = plt.imshow(np.real(image), origin='lower', cmap='gist_heat',
                    interpolation='nearest')
    plt.show(im)

################################################################################

def ds9_arrays(**kwargs):

    cmd = ['ds9', '-zscale', '-zoom', '4', '-cmap', 'heat']
    for name, array in kwargs.items():
        # write array to fits
        fitsfile = 'ds9_'+name+'.fits'
        fits.writeto(fitsfile, np.array(array).astype(np.float32), clobber=True)            
        # append to command
        cmd.append(fitsfile)

    #print 'cmd', cmd
    result = call(cmd)
    
################################################################################

def run_wcs(image_in, image_out, ra, dec, gain, readnoise, fwhm, pixscale, imtype):

    if timing: t = time.time()
    print '\nexecuting run_wcs ...'
    
    scale_low = 0.99 * pixscale
    scale_high = 1.01 * pixscale
    
    # round fwhm to 2 decimals
    fwhm = float('{:.2f}'.format(fwhm))
    # determine seeing
    seeing = fwhm * pixscale
    
    sexcat = image_out.replace('.fits','.sexcat')

    # in case of new image and [psf_sampling] is set to zero, scale
    # the size of the VIGNET output in the output catalog with
    # 2*[psf_radius]*[fwhm]
    if imtype=='new' and psf_sampling == 0.:
        # replace VIGNET size in SExtractor parameter file based on [psf_radius]
        size_vignet = np.int(np.ceil(2.*psf_radius*fwhm))
        # make sure it's odd (not sure if this is important; suggested in
        # PSFex manual)
        if size_vignet % 2 == 0: size_vignet += 1
        size_vignet_str = str((size_vignet, size_vignet))
        sex_par_temp = sex_par+'_temp'
        with open(sex_par, 'rt') as file_in:
            with open(sex_par_temp, 'wt') as file_out:
                for line in file_in:
                    file_out.write(line.replace('VIGNET(99,99)', 'VIGNET'+size_vignet_str))
        if verbose:
            print 'VIGNET size:', size_vignet_str
    # for the reference image, or if [psf_sampling] is non-zero, the
    # default VIGNET size as defined in the SExtractor config file is
    # used, at the moment this is (99,99)
    else:
        sex_par_temp = sex_par
            
    #scampcat = image_in.replace('.fits','.scamp')
    cmd = ['solve-field', '--no-plots', '--no-fits2fits',
           '--sextractor-config', sex_cfg,
           '--x-column', 'XWIN_IMAGE', '--y-column', 'YWIN_IMAGE',
           '--sort-column', 'FLUX_AUTO',
           '--no-remove-lines',
           '--keep-xylist', sexcat,
           # ignore existing WCS headers in FITS input images
           #'--no-verify', 
           #'--code-tolerance', str(0.01), 
           #'--quad-size-min', str(0.1),
           # for KMTNet images restrict the max quad size:
           '--quad-size-max', str(0.1),
           # number of field objects to look at:
           #'--depth', str(10),
           #'--scamp', scampcat,
           image_in,
           '--tweak-order', str(astronet_tweak_order), '--scale-low', str(scale_low),
           '--scale-high', str(scale_high), '--scale-units', 'app',
           '--ra', str(ra), '--dec', str(dec), '--radius', str(2.),
           '--new-fits', image_out, '--overwrite']

    # prepare aperture radii string 
    apphot_diams = np.array(apphot_radii) * 2 * fwhm
    apphot_diams_str = ",".join(apphot_diams.astype(str))
    if verbose:
        print 'aperture diameters used for PHOT_APERTURES', apphot_diams_str
    
    cmd_sex = 'sex -SEEING_FWHM '+str(seeing)+' -PARAMETERS_NAME '+sex_par_temp\
              +' -PHOT_APERTURES '+apphot_diams_str+' -BACK_SIZE '+str(bkg_boxsize)\
              +' -BACK_FILTERSIZE '+str(bkg_filtersize)
    
    # add commands to produce BACKGROUND, BACKGROUND_RMS and
    # background-subtracted image with all pixels where objects were
    # detected set to zero (-OBJECTS). These are used to build an
    # improved background map. 
    bkg = image_in.replace('.fits','_bkg.fits')
    bkg_std = image_in.replace('.fits','_bkg_std.fits')
    objmask = image_in.replace('.fits','_objmask.fits')
    cmd_sex += ' -CHECKIMAGE_TYPE BACKGROUND,BACKGROUND_RMS,-OBJECTS -CHECKIMAGE_NAME '\
               +bkg+','+bkg_std+','+objmask
        
    cmd += ['--sextractor-path', cmd_sex]
    if verbose:
        print 'Astrometry.net command:', cmd
    
    result = call(cmd)

    if timing: t2 = time.time()

    # this is the file containing just the WCS solution from Astrometry.net
    wcsfile = image_in.replace('.fits', '.wcs')

    use_wcs_xy2rd = False
    if use_wcs_xy2rd:
        # run Astrometry.net's wcs-xy2rd on the unix command line to
        # convert XWIN_IMAGE and YWIN_IMAGE to RA and DEC (saved in a
        # two-column fits table [radecfile]) from the [sexcat] and
        # .wcs output files created by Astrometry.net
        radecfile = image_in.replace('.fits', '.radec')
        cmd = ['wcs-xy2rd', '-w', wcsfile, '-i', sexcat, '-o', radecfile,
               '-X', 'XWIN_IMAGE', '-Y', 'YWIN_IMAGE']
        result = call(cmd)
        # read file with new ra and dec
        with fits.open(radecfile) as hdulist:
            data_newradec = hdulist[1].data
        newra = data_newradec['RA']
        newdec = data_newradec['DEC']

    # convert SIP header keywords from Astrometry.net to PV keywords
    # that swarp, scamp (and sextractor) understand using this module
    # from David Shupe:
    sip_to_pv(image_out, image_out, tpv_format=False)

    # read data from SExtractor catalog produced in Astrometry.net
    with fits.open(sexcat) as hdulist:
        data_sexcat = hdulist[1].data

    if not use_wcs_xy2rd:
        # instead of wcs-xy2rd, use astropy.WCS to find RA, DEC
        # corresponding to XWIN_IMAGE, YWIN_IMAGE, based on WCS info
        # saved by Astrometry.net in .wcs file (wcsfile). The 3rd
        # parameter to wcs.all_pix2world indicates the pixel
        # coordinate of the frame origin. This avoids having to save
        # the new RAs and DECs to file and read them back into python
        # arrays. Although it gives a command line warning, it
        # provides the same RA and DEC as wcs-xy2rd and also as
        # SExtractor run independently on the WCS-ed image (i.e.  the
        # image_out in this function). The warning is the mismatch
        # between NAXES in the .wcs image (0) and that expected
        # by the routine (2).      
        wcs = WCS(wcsfile)
        newra, newdec = wcs.all_pix2world(data_sexcat['XWIN_IMAGE'],
                                          data_sexcat['YWIN_IMAGE'],
                                          1)

    # read header of WCS image produced by Astrometry.net to be put in
    # data part of the LDAC_IMHEAD extension of the LDAC fits table
    # below
    with fits.open(image_out) as hdulist:
        header_wcsimage = hdulist[0].header

    # add header of .axy extension as the SExtractor keywords are there,
    # although PSFex only seems to use 2 of them: SEXGAIN and SEXBKDEV.
    # Astrometry.net does not provide these values (zeros), so their
    # values need to be set.
    axycat = image_in.replace('.fits','.axy')
    with fits.open(axycat) as hdulist:
        header_axycat = hdulist[0].header
    header_axycat['FITSFILE'] = image_out
    header_axycat['SEXGAIN'] = gain
    # estimate background r.m.s. (needed by PSFex) from BACKGROUND column in sexcat
    header_axycat['SEXBKDEV'] = np.sqrt(np.median(data_sexcat['BACKGROUND'])
                                        * gain + readnoise) / gain
    print 'background r.m.s. estimate:', np.sqrt(np.median(data_sexcat['BACKGROUND'])
                                                 * gain + readnoise)/gain
        
    # replace old ra and dec with new ones
    data_sexcat['ALPHAWIN_J2000'] = newra
    data_sexcat['DELTAWIN_J2000'] = newdec

    # convert FITS table to LDAC format needed by PSFex
    result = fits2ldac(header_wcsimage+header_axycat,
                       data_sexcat, sexcat, doSort=True)
    
    if timing:
        print 'extra time for creating LDAC fits table', time.time()-t2
        print 'wall-time spent in run_wcs', time.time()-t

################################################################################

def fits2ldac (header4ext2, data4ext3, fits_ldac_out, doSort=True):

    """This function converts the binary FITS table from Astrometry.net to
    a binary FITS_LDAC table that can be read by PSFex. [header4ext2]
    is what will be recorded as a single long string in the data part
    of the 2nd extension of the output table [fits_ldac_out], and
    [data4ext3] is the data part of an HDU that will define both the
    header and data parts of extension 3 of [fits_ldac_out].

    """

    # convert header to single (very) long string
    ext2_str = header4ext2.tostring(endcard=False, padding=False)

    # if the following line is not added, the very end of the data
    # part of extension 2 is written to a fits table such that PSFex
    # runs into a segmentation fault when attempting to read it (took
    # me ages to find out!).
    ext2_str += 'END                                                                          END'

    # read into string array
    ext2_data = np.array([ext2_str])

    # determine format string for header of extention 2
    formatstr = str(len(ext2_str))+'A'
    # create table 1
    col1 = fits.Column(name='Field Header Card', array=ext2_data, format=formatstr)
    cols = fits.ColDefs([col1])
    ext2 = fits.BinTableHDU.from_columns(cols)
    # make sure these keywords are in the header
    ext2.header['EXTNAME'] = 'LDAC_IMHEAD'
    ext2.header['TDIM1'] = '(80, {0})'.format(len(ext2_str)/80)

    # simply create extension 3 from [data4ext3]
    ext3 = fits.BinTableHDU(data4ext3)
    # extname needs to be as follows
    ext3.header['EXTNAME'] = 'LDAC_OBJECTS'

    # sort output table by number column if needed
    if doSort:
        index_sort = np.argsort(ext3.data['NUMBER'])
        ext3.data = ext3.data[index_sort]
    
    # create primary HDU
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    prihdu.header['EXPTIME'] = header4ext2['EXPTIME']
    prihdu.header['FILTNAME'] = header4ext2['FILTNAME']
    prihdu.header['SEEING'] = header4ext2['SEEING']
    prihdu.header['BKGSIG'] = header4ext2['SEXBKDEV']

    
    # write hdulist to output LDAC fits table
    hdulist = fits.HDUList([prihdu, ext2, ext3])
    hdulist.writeto(fits_ldac_out, clobber=True)
    hdulist.close()
    
################################################################################
    
def run_remap(image_new, image_ref, image_out, image_out_size,
              gain, config=swarp_cfg, resampling_type='LANCZOS3',
              projection_err=0.001):
        
    """Function that remaps [image_ref] onto the coordinate grid of
       [image_new] and saves the resulting image in [image_out] with
       size [image_size].
    """

    if timing: t = time.time()
    print '\nexecuting run_remap ...'

    # read headers
    t = time.time()
    with fits.open(image_new) as hdulist:
        header_new = hdulist[0].header
    with fits.open(image_ref) as hdulist:
        header_ref = hdulist[0].header
        
    # create .head file with header info from [image_new]
    header_out = header_new[:]
    # copy some keywords from header_ref
    #for key in [key_exptime, key_satlevel, key_gain, key_ron, key_seeing]:
    for key in [key_exptime, key_satlevel, key_gain, key_ron]:
        header_out[key] = header_ref[key]
    # delete some others
    for key in ['WCSAXES', 'NAXIS1', 'NAXIS2']:
        del header_out[key]
    # write to .head file
    with open(image_out.replace('.fits','.head'),'w') as newrefhdr:
        for card in header_out.cards:
            newrefhdr.write(str(card)+'\n')

    size_str = str(image_out_size[1]) + ',' + str(image_out_size[0]) 
    cmd = ['swarp', image_ref, '-c', config, '-IMAGEOUT_NAME', image_out, 
           '-IMAGE_SIZE', size_str, '-GAIN_DEFAULT', str(gain),
           '-RESAMPLING_TYPE', resampling_type,
           '-PROJECTION_ERR', str(projection_err)]
    result = call(cmd)
    
    if timing: print 'wall-time spent in run_remap', time.time()-t

################################################################################

def run_sextractor(image, cat_out, file_config, file_params, pixscale,
                   fitpsf=False, fraction=1.0, fwhm=5.0):

    """Function that runs SExtractor on [image], and saves the output
       catalog in [outcat], using the configuration file [file_config]
       and the parameters defining the output recorded in the
       catalogue [file_params]. If [fitpsf] is set to True,
       SExtractor will perform PSF fitting photometry using the PSF
       built by PSFex. If [fraction] is less than the default 1.0,
       SExtractor will be run on a fraction [fraction] of the area of
       the full image.

    """

    if timing: t = time.time()
    print '\nexecuting run_sextractor ...'

    # if fraction less than one, run SExtractor on specified fraction of
    # the image
    if fraction < 1.:

        # read in input image and header
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            data = hdulist[0].data
        # get input image size from header
        ysize, xsize = read_header(header, ['NAXIS2', 'NAXIS1'])
        
        # determine cutout from [fraction]
        center_x = np.int(xsize/2+0.5)
        center_y = np.int(ysize/2+0.5)
        halfsize_x = np.int((xsize * np.sqrt(fraction))/2.+0.5)
        halfsize_y = np.int((ysize * np.sqrt(fraction))/2.+0.5)
        data_fraction = data[center_y-halfsize_y:center_y+halfsize_y,
                             center_x-halfsize_x:center_x+halfsize_x]

        # write small image to fits
        image_fraction = image.replace('.fits','_fraction.fits')
        fits.writeto(image_fraction, data_fraction.astype(np.float32), header, clobber=True)

        # make image point to image_fraction
        image = image_fraction
        cat_out = cat_out+'_fraction'


    # the input fwhm determines the SEEING_FWHM (important for
    # star/galaxy separation) and the radii of the apertures used for
    # aperture photometry. If fwhm is not provided as input, it will
    # assume fwhm=5.0 pixels.
    fwhm = float('{:.2f}'.format(fwhm))
    # determine seeing
    seeing = fwhm * pixscale
    # prepare aperture diameter string to provide to SExtractor 
    apphot_diams = np.array(apphot_radii) * 2 * fwhm
    apphot_diams_str = ",".join(apphot_diams.astype(str))
    
    # run sextractor from the unix command line
    cmd = ['sex', image, '-c', file_config, '-CATALOG_NAME', cat_out, 
           '-PARAMETERS_NAME', file_params, '-PIXEL_SCALE', str(pixscale),
           '-SEEING_FWHM', str(seeing),'-PHOT_APERTURES',apphot_diams_str,
           '-BACK_SIZE', str(bkg_boxsize), '-BACK_FILTERSIZE', str(bkg_filtersize)]

    # in case of fraction being less than 1: only care about higher S/N detections
    if fraction < 1.: cmd += ['-DETECT_THRESH', str(fwhm_detect_thresh)]
    
    # provide PSF file from PSFex
    if fitpsf: cmd += ['-PSF_NAME', image.replace('.fits', '.psf')]

    # run command
    result = call(cmd)

    # get estimate of seeing from output catalog
    fwhm, fwhm_std = get_fwhm(cat_out, fwhm_frac, class_Sort=fwhm_class_sort)

    if timing: print 'wall-time spent in run_sextractor', time.time()-t
    return fwhm, fwhm_std


################################################################################

def get_fwhm (cat_ldac, fraction, class_Sort = False, get_elongation=False):

    """Function that accepts a FITS_LDAC table produced by SExtractor and
    returns the FWHM and its standard deviation in pixels.  The
    columns that need to be present in the fits table are 'FLAGS',
    'FLUX_AUTO' and 'CLASS_STAR'. By default, the function takes the
    brightest [fraction] of objects, and determines the median FWHM
    from them using sigma clipping. If [class_Sort] is True, it
    instead takes the fraction of objects with the highest CLASS_STAR
    value, which ideally is 1 for stars and 0 for galaxies. However,
    if the SEEING_FWHM that was used for the SExtractor run was off
    from the real value, the CLASS_STAR is not reliable.

    """
 
    if timing: t = time.time()
    print '\nexecuting get_fwhm ...'

    with fits.open(cat_ldac) as hdulist:
        data = hdulist[2].data

    # these arrays correspond to objecst with flag==0 and flux_auto>0.
    index = (data['FLAGS']==0) & (data['FLUX_AUTO']>0.)
    fwhm = data['FWHM_IMAGE'][index]
    class_star = data['CLASS_STAR'][index]
    flux_auto = data['FLUX_AUTO'][index]
    mag_auto = -2.5*np.log10(flux_auto)
    if get_elongation:
        elongation = data['ELONGATION'][index]
    
    if class_Sort:
        # sort by CLASS_STAR
        index_sort = np.argsort(class_star)
    else:
        # sort by FLUX_AUTO
        index_sort = np.argsort(flux_auto)

    # select fraction of targets
    index_select = np.arange(-np.int(len(index_sort)*fraction+0.5),-1)
    fwhm_select = fwhm[index_sort][index_select] 
    if get_elongation:
        elongation_select = elongation[index_sort][index_select] 
            
    # print warning if few stars are selected
    if len(fwhm_select) < 10:
        print 'WARNING: fewer than 10 objects are selected for FWHM determination'
    
    # determine mean, median and standard deviation through sigma clipping
    fwhm_mean, fwhm_std, fwhm_median = clipped_stats(fwhm_select)
    if verbose:
        print 'catalog', cat_ldac
        print 'fwhm_mean, fwhm_median, fwhm_std', fwhm_mean, fwhm_median, fwhm_std
    if get_elongation:
        # determine mean, median and standard deviation through sigma clipping
        elongation_mean, elongation_std, elongation_median = clipped_stats(elongation_select)
        if verbose:
            print 'elongation_mean, elongation_median, elongation_std',\
                elongation_mean, elongation_median, elongation_std
            
        
    if make_plots:

        # best parameter to plot vs. FWHM is MAG_AUTO
        mag_auto_select = mag_auto[index_sort][index_select]

        # to get initial values before discarding flagged objects
        index = (data['FLUX_AUTO']>0.)
        fwhm = data['FWHM_IMAGE'][index]
        flux_auto = data['FLUX_AUTO'][index]
        mag_auto = -2.5*np.log10(flux_auto)

        plt.plot(fwhm, mag_auto, 'bo')
        x1,x2,y1,y2 = plt.axis()
        plt.plot(fwhm_select, mag_auto_select, 'go')
        plt.plot([fwhm_median, fwhm_median], [y2,y1], color='red')
        fwhm_line = fwhm_median-fwhm_std
        plt.plot([fwhm_line, fwhm_line], [y2,y1], 'r--')
        fwhm_line = fwhm_median+fwhm_std
        plt.plot([fwhm_line, fwhm_line], [y2,y1], 'r--')
        plt.axis((0,20,y2,y1))
        plt.xlabel('FWHM (pixels)')
        plt.ylabel('MAG_AUTO')
        plt.savefig('fwhm.pdf')
        if show_plots: plt.show()
        plt.close()

        if get_elongation:
            elongation = data['ELONGATION'][index]

            plt.plot(elongation, mag_auto, 'bo')
            x1,x2,y1,y2 = plt.axis()
            plt.plot(elongation_select, mag_auto_select, 'go')
            plt.plot([elongation_median, elongation_median], [y2,y1], color='red')
            elongation_line = elongation_median-elongation_std
            plt.plot([elongation_line, elongation_line], [y2,y1], 'r--')
            elongation_line = elongation_median+elongation_std
            plt.plot([elongation_line, elongation_line], [y2,y1], 'r--')
            plt.axis((0,20,y2,y1))
            plt.xlabel('ELONGATION (A/B)')
            plt.ylabel('MAG_AUTO')
            plt.savefig('elongation.pdf')
            if show_plots: plt.show()
            plt.close()
            
    if timing: print 'wall-time spent in get_fwhm', time.time()-t

    if get_elongation:
        return fwhm_median, fwhm_std, elongation_median, elongation_std
    else:
        return fwhm_median, fwhm_std
    

################################################################################

def run_psfex(cat_in, file_config, cat_out, fwhm, imtype):
    
    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""

    if timing: t = time.time()

    if psf_sampling == 0:
        # determine [psf_size_config] based on [psf_radius], which is
        # 2 * [psf_radius] * FWHM / sampling factor. The sampling
        # factor used in PSFex below is now forced such that FWHM /
        # sampling factor =4.5, so:
        psf_sampling_use = fwhm / 4.5
        size = psf_radius*2*4.5
        # in case of reference image, scale this size by the ratio of
        # FWHMs so that the final psf stamp of the reference image is
        # as large as the psf stamp of the new image
        if imtype=='ref':
            size *= fwhm_new/fwhm_ref
        # convert to integer
        size = np.int(size+0.5)
        # make sure it's odd
        if size % 2 == 0: size += 1
        psf_size_config = str(size)+','+str(size)
    else:
        # use [psf_sampling] in PSFex
        psf_sampling_use = psf_sampling
        # use some reasonable default size
        psf_size_config = '45,45'

    if verbose:
        print 'psf_sampling_use', psf_sampling_use
        print 'psf_size_config', psf_size_config
        
    # get FWHM and ELONGATION to limit the PSFex configuration
    # parameters SAMPLE_FWHMRANGE and SAMPLE_MAXELLIP
    #fwhm, fwhm_std, elongation, elongation_std = get_fwhm(cat_in, 0.05, class_Sort=False,
    #                                                      get_elongation=True)
    #print 'fwhm, fwhm_std, elongation, elongation_std', fwhm, fwhm_std, elongation, elongation_std
    #sample_fwhmrange = str(fwhm-fwhm_std)+','+str(fwhm+fwhm_std)
    #print 'sample_fwhmrange', sample_fwhmrange
    #maxellip = (elongation+3.*elongation_std-1)/(elongation+3.*elongation_std+1)
    #maxellip_str= str(np.amin([maxellip, 1.]))
    #print 'maxellip_str', maxellip_str
    
    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config, '-PSF_SAMPLING', str(psf_sampling_use)]
    #       '-SAMPLE_FWHMRANGE', sample_fwhmrange,
    #       '-SAMPLE_MAXELLIP', maxellip_str]
    result = call(cmd)    

    if timing: print 'wall-time spent in run_psfex', time.time()-t

################################################################################

def clean_norm_psf(psf_array, clean_factor):

    # psf_array is assumed to be square
    ysize, xsize = psf_array.shape
    assert xsize == ysize
    
    # set values in the corners of the PSF image to zero
    hsize = xsize/2
    # even
    if xsize % 2 == 0:
        x = np.arange(-hsize, hsize)
    # odd
    else:
        x = np.arange(-hsize, hsize+1)
    xx, yy = np.meshgrid(x, x, sparse=True)
    psf_array[(xx**2+yy**2)>hsize**2] = 0

    # set any negative values to zero
    #psf_array[psf_array<0] = 0
    
    if clean_factor != 0:
        mask_clean = (psf_array < (np.amax(psf_array) * clean_factor))
        psf_array[mask_clean] = 0.

    # normalize
    psf_array_norm = psf_array / np.sum(psf_array)

    return psf_array_norm
    
################################################################################

def run_ZOGY(R,N,Pr,Pn,sr,sn,fr,fn,Vr,Vn,dx,dy):

# edited Barak's original code to include variances sigma_n**2 and
# sigma_r**2 (see Eq. 9, here sn and sr) and Fn and Fr which are
# assumed to be unity in Barak's code.
    
    if timing: t = time.time()

    R_hat = fft.fft2(R)
    N_hat = fft.fft2(N)
    Pn_hat = fft.fft2(Pn)
    #if psf_clean_factor!=0:
        # clean Pn_hat
        #Pn_hat = clean_psf(Pn_hat, psf_clean_factor)
    Pn_hat2_abs = np.abs(Pn_hat**2)

    Pr_hat = fft.fft2(Pr)
    #if psf_clean_factor!=0:
        # clean Pr_hat
        #Pr_hat = clean_psf(Pr_hat, psf_clean_factor)
    Pr_hat2_abs = np.abs(Pr_hat**2)

    sn2 = sn**2
    sr2 = sr**2
    #beta = fn/fr
    #beta2 = beta**2
    fn2 = fn**2
    fr2 = fr**2
    fD = fr*fn / np.sqrt(sn2*fr2+sr2*fn2)
    
    denominator = sn2*fr2*Pr_hat2_abs + sr2*fn2*Pn_hat2_abs
    if np.any(denominator==0):
        print 'Warning: denominator contains zero(s)'
        
    #denominator_beta = sn2*Pr_hat2_abs + beta2*sr2*Pn_hat2_abs

    D_hat = (fr*Pr_hat*N_hat - fn*Pn_hat*R_hat) / np.sqrt(denominator)
    # alternatively using beta:
    #D_hat = (Pr_hat*N_hat - beta*Pn_hat*R_hat) / np.sqrt(denominator_beta)

    D = np.real(fft.ifft2(D_hat)) / fD
    
    P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)
    #alternatively using beta:
    #P_D_hat = np.sqrt(sn2+beta2*sr2)*(Pr_hat*Pn_hat) / np.sqrt(denominator_beta)

    #P_D = np.real(fft.ifft2(P_D_hat))
    #print 'np.sum(P_D)', np.sum(P_D)
    
    S_hat = fD*D_hat*np.conj(P_D_hat)
    S = np.real(fft.ifft2(S_hat))

    # alternative way to calculate S
    #S_hat = (fn*fr2*Pr_hat2_abs*np.conj(Pn_hat)*N_hat -
    #         fr*fn2*Pn_hat2_abs*np.conj(Pr_hat)*R_hat) / denominator
    #S = np.real(fft.ifft2(S_hat))

    # PMV 2017/01/18: added following part based on Eqs. 25-31
    # from Barak's paper
    kr_hat = fr*fn2*np.conj(Pr_hat)*Pn_hat2_abs / denominator
    kr = np.real(fft.ifft2(kr_hat))
    kr2 = kr**2
    kr2_hat = fft.fft2(kr2)

    kn_hat = fn*fr2*np.conj(Pn_hat)*Pr_hat2_abs / denominator
    kn = np.real(fft.ifft2(kn_hat))
    kn2 = kn**2
    kn2_hat = fft.fft2(kn2)

    # checks
    #print 'sum(Pn)', np.sum(Pn)
    #print 'sum(Pr)', np.sum(Pr)
    #print 'sum(kn)', np.sum(kn)
    #print 'sum(kr)', np.sum(kr)
    #print 'fD', fD
    #print 'fD squared', fD**2
    
    Vr_hat = fft.fft2(Vr)
    Vn_hat = fft.fft2(Vn)

    VSr = np.real(fft.ifft2(Vr_hat*kr2_hat))
    VSn = np.real(fft.ifft2(Vn_hat*kn2_hat))

    dx2 = dx**2
    dy2 = dy**2
    # and calculate astrometric variance
    Sn = np.real(fft.ifft2(kn_hat*N_hat))
    dSndy = Sn - np.roll(Sn,1,axis=0)
    dSndx = Sn - np.roll(Sn,1,axis=1)
    VSn_ast = dx2 * dSndx**2 + dy2 * dSndy**2
    
    Sr = np.real(fft.ifft2(kr_hat*R_hat))
    dSrdy = Sr - np.roll(Sr,1,axis=0)
    dSrdx = Sr - np.roll(Sr,1,axis=1)
    VSr_ast = dx2 * dSrdx**2 + dy2 * dSrdy**2

    if verbose:
        print 'fD', fD
        #print 'kr_hat is finite?', np.all(np.isfinite(kr_hat))
        #print 'kn_hat is finite?', np.all(np.isfinite(kn_hat))
        #print 'dSrdx is finite?', np.all(np.isfinite(dSrdx))
        #print 'dSrdy is finite?', np.all(np.isfinite(dSrdy))
        #print 'dSndy is finite?', np.all(np.isfinite(dSndy))
        #print 'dSndx is finite?', np.all(np.isfinite(dSndx))
        #print 'VSr_ast is finite?', np.all(np.isfinite(VSr_ast))
        #print 'VSn_ast is finite?', np.all(np.isfinite(VSn_ast))
        #print 'dx is finite?', np.isfinite(dx)
        #print 'dy is finite?', np.isfinite(dy)
    
    if display:
        fits.writeto('Pn_hat.fits', np.real(Pn_hat).astype(np.float32), clobber=True)
        fits.writeto('Pr_hat.fits', np.real(Pr_hat).astype(np.float32), clobber=True)
        fits.writeto('kr.fits', np.real(kr).astype(np.float32), clobber=True)
        fits.writeto('kn.fits', np.real(kn).astype(np.float32), clobber=True)
        fits.writeto('Sr.fits', Sr.astype(np.float32), clobber=True)
        fits.writeto('Sn.fits', Sn.astype(np.float32), clobber=True)
        fits.writeto('VSr.fits', VSr.astype(np.float32), clobber=True)
        fits.writeto('VSn.fits', VSn.astype(np.float32), clobber=True)
        fits.writeto('VSr_ast.fits', VSr_ast.astype(np.float32), clobber=True)
        fits.writeto('VSn_ast.fits', VSn_ast.astype(np.float32), clobber=True)

    # and finally S_corr
    V_S = VSr + VSn
    V_ast = VSr_ast + VSn_ast
    V = V_S + V_ast
    #S_corr = S / np.sqrt(V)
    # make sure there's no division by zero
    S_corr = np.copy(S)
    S_corr[V>0] /= np.sqrt(V[V>0])

    # PMV 2017/03/05: added following PSF photometry part based on
    # Eqs. 41-43 from Barak's paper
    F_S =  np.sum((fn2*Pn_hat2_abs*fr2*Pr_hat2_abs) / denominator)
    # divide by the number of pixels in the images (related to do
    # the normalization of the ffts performed)
    F_S /= R.size
    if verbose:
        print 'F_S', F_S
    # an alternative (slower) way to calculate the same F_S:
    #F_S_array = fft.ifft2((fn2*Pn_hat2_abs*fr2*Pr_hat2_abs) / denominator)
    #F_S = F_S_array[0,0]

    alpha = S / F_S
    alpha_std = np.zeros(alpha.shape)
    alpha_std[V_S>=0] = np.sqrt(V_S[V_S>=0]) / F_S

    if timing:
        print 'wall-time spent in optimal subtraction', time.time()-t
        #print 'peak memory used in run_ZOGY in GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9
    
    return D, S, S_corr, alpha, alpha_std

################################################################################

def optimal_binary_image_subtraction(R,N,Pr,Pn,sr,sn):

# original code from Barak (this assumes fr and fn are unity, and it
# does not calculate the variance images needed for Scorr):

    R_hat = fft.fft2(R)
    N_hat = fft.fft2(N)
    Pn_hat = fft.fft2(Pn)
    Pr_hat = fft.fft2(Pr)
    G_hat = (Pr_hat*N_hat - Pn_hat*R_hat) / np.sqrt((sr**2*abs(Pn_hat**2) + sn**2*abs(Pr_hat**2)))
    P_G_hat = (Pr_hat*Pn_hat) / np.sqrt((sr**2*abs(Pn_hat**2) + sn**2*abs(Pr_hat**2)))
    S_hat = G_hat*conj(P_G_hat)
    #S_hat = (conj(Pn_hat)*np.abs(Pr_hat)**2*N_hat - conj(Pr_hat)*np.abs(Pn_hat)**2*R_hat) / (sr**2*abs(Pn_hat**2) + sn**2*abs(Pr_hat**2))
    S = fft.ifft2(S_hat)
    G = fft.ifft2(G_hat)
    P_G = real(fft.ifft2(P_G_hat))
    return S/std(S[15::30,15::30]), G/std(G[15::30,15::30]), P_G / sum(P_G)

################################################################################

def image_shift_fft(Image, DX, DY):
    
    phase = 2

    NY, NX = np.shape(Image)

    Nr = fft.ifftshift(np.arange(-np.floor(NY/2),np.ceil(NY/2)))
    Nc = fft.ifftshift(np.arange(-np.floor(NX/2),np.ceil(NX/2)))
    Nc, Nr = np.meshgrid(Nc,Nr);
    
    # Fourier Transform shift theorem
    image_fft2 = fft.fft2(Image) * np.exp(-1.j*2.*np.pi*(Nr*(DY/NY)+Nc*(DX/NX)))
    image_shifted = fft.ifft2(image_fft2) * np.exp(-1.j*phase)

    return np.abs(image_shifted)
    

# Original MATLAB function provided by Eran:
# 
# function [ShiftedImage,NY,NX,Nr,Nc]=image_shift_fft(Image,DX,DY,NY,NX,Nr,Nc)
# % Shift Image using the sub pixel Fourier shift theorem (sinc interp.)
# % Package: AstroIm
# % Description: Shift an image using the FFT shift thorem. This works well
# %              when the image does not contain sharp artifacts.
# %              Sharp artifacts will produce ringing.
# %              Note that the shift is defined on the content of the image,
# %              rather than the image boundries - e.g., the stars will be
# %              shifted in the requested direction.
# % Input  : - An image (2D matrix).
# %          - X shift to apply to input image.
# %          - Y shift to apply to input image.
# %          - NY (supply for faster performences). See output.
# %          - NX (supply for faster performences). See output.
# %          - Nr (supply for faster performences). See output.
# %          - Nc (supply for faster performences). See output.
# % Output : - Shifted image with the same size as the input image.
# %          - NY
# %          - NX
# %          - Nr
# %          - Nc
# % See also: AstroIm.imagefft_shift_fft.m, SIM/image_shift_fft.m,
# %           SIM/imagefft_shift_fft.m
# % License: GNU general public license version 3
# % Tested : Matlab R2015b
# %     By : Eran O. Ofek                    May 2016
# %    URL : http://weizmann.ac.il/home/eofek/matlab/
# % Example: ShiftedImage=AstroIm.image_shift_fft(Image,1.22,-3.1);
# % Reliable: 2
# %--------------------------------------------------------------------------
# 
# Phase = 2;
# 
# Nim = 1;
# if (nargin<4),
#     % NY, NX, Nr, Nc are not provided by user
#     [NY,NX,Nim] = size(Image);
#     Nr = ifftshift((-fix(NY/2):ceil(NY/2)-1));
#     Nc = ifftshift((-fix(NX/2):ceil(NX/2)-1));
#     [Nc,Nr] = meshgrid(Nc,Nr);
# end
# 
# % Fourier Transform shift theorem
# if (Nim==1),
#     ShiftedImage = ifft2(fft2(Image).*exp(-1i.*2.*pi.*(DY.*Nr./NY + DX.*Nc./NX))).*exp(-1i.*Phase)# ;
# else
#     % Image is cube
#     % not operational
#     %ShiftedImage = ifft2(fft2(Image).*exp(-1i.*2.*pi.*(  bsxfun(@times,DY,shiftdim(Nr,-1))./NY + # bsxfun(@times,DX,shiftdim(Nc,-1))./NX))).*exp(-1i.*Phase);
# end
# ShiftedImage = abs(ShiftedImage);

################################################################################

def main():
    """Wrapper allowing optimal_subtraction to be run from the command line"""
    
    parser = argparse.ArgumentParser(description='Run optimal_subtraction on images')
    parser.add_argument('new_fits', help='filename of new image')
    parser.add_argument('ref_fits', help='filename of ref image')
    parser.add_argument('--ref_fits_remap', default=None, help='remapped ref image')
    parser.add_argument('--sub', default=None, help='sub image')
    parser.add_argument('--telescope', default=None, help='telescope')
    parser.add_argument('--log', default=None, help='help')
    parser.add_argument('--subpipe', default=False, help='subpipe')
    args = parser.parse_args()
    optimal_subtraction(args.new_fits, args.ref_fits, args.ref_fits_remap, args.sub, args.telescope, args.log, args.subpipe)
        
if __name__ == "__main__":
    main()
