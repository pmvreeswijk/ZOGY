
import argparse
import astropy.io.fits as fits
from astropy.io import ascii
#from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import numpy as np
#import numpy.fft as fft
import matplotlib.pyplot as plt
import os
import subprocess
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
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground

# for PSF fitting - see https://lmfit.github.io/lmfit-py/index.html
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit

# see https://github.com/stargaser/sip_tpv (version June 2017):
# download from GitHub and "python install setup.py --user" for local
# install or "sudo python install setup.py" for system install
from sip_tpv import sip_to_pv

import resource
from skimage import restoration
#import inpaint
import logging
import sys

#from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.dummy import Lock

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from numpy.lib.recfunctions import append_fields, drop_fields, rename_fields


################################################################################

def optimal_subtraction(new_fits=None, ref_fits=None, log=None):
    
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
      https://github.com/stargaser/sip_tpv
    - pyfftw to speed up the many FFTs performed
    - the other modules imported at the top
 
    Written by Paul Vreeswijk (pmvreeswijk@gmail.com) with vital input
    from Barak Zackay and Eran Ofek. Adapted by Kerry Paterson for
    integration into pipeline for MeerLICHT (ptrker004@myuct.ac.za).

    """

    start_time1 = os.times()
    
    # initialise log
    if log is None:
        log = logging.getLogger() #create logger
        log.setLevel(logging.INFO) #set level of logger
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s") #set format of logger
        logging.Formatter.converter = time.gmtime #convert time in logger to UCT
        if new_fits is not None:
            filehandler = logging.FileHandler(new_fits.replace('.fits','.log'), 'w+') #create log file
        elif ref_fits is not None:
            filehandler = logging.FileHandler(ref_fits.replace('.fits','.log'), 'w+') #create log file
        else:
            filehandler = logging.FileHandler('log', 'w+') #create log file
        filehandler.setFormatter(formatter) #add format to log file
        log.addHandler(filehandler) #link log file to logger
        if C.verbose:
            streamhandler = logging.StreamHandler() #create print to screen logging
            streamhandler.setFormatter(formatter) #add format to screen logging
            log.addHandler(streamhandler) #link logger to screen logging
    
    # define booleans [new] and [ref] indicating
    # that the corresponding image is provided and exists
    def set_bool (image_fits):
        ima_bool = False
        if image_fits is not None:
            if os.path.isfile(image_fits):
                ima_bool = True
            else:
                log.info('Error: {} does not exist'.format(image_fits))
        return ima_bool
    new = set_bool (new_fits)
    ref = set_bool (ref_fits)
    if not new and not ref:
        log.info('Error: no valid input image(s) provided')
        raise SystemExit

    # global parameters
    if new:
        global base_new, fwhm_new
        # define the base names of input fits files as global so they
        # can be used in any function in this module
        base_new = new_fits.split('.fits')[0]
    if ref:
        global base_ref, fwhm_ref
        # define the base names of input fits files as global so they
        # can be used in any function in this module
        base_ref = ref_fits.split('.fits')[0]
    # if either one of [base_new] or [base_ref] is not defined, set it
    # to the value of their counterpart as they're used below, a.o.
    # in function [get_psf]
    if not new: base_new = base_ref
    if not ref: base_ref = base_new
    if new and ref: base_newref = base_new+'_'+base_ref
    
    keywords = ['NAXIS2', 'NAXIS1', C.key_gain, C.key_ron, C.key_satlevel,
                C.key_ra, C.key_dec, C.key_pixscale]
    if new:
        # read in header of new_fits
        t = time.time()
        with fits.open(new_fits) as hdulist:
            header_new = hdulist[0].header
        ysize_new, xsize_new, gain_new, readnoise_new, satlevel_new, ra_new, dec_new, pixscale_new = read_header(header_new, keywords, log)

    if ref:
        # read in header of ref_fits
        with fits.open(ref_fits) as hdulist:
            header_ref = hdulist[0].header
        ysize_ref, xsize_ref, gain_ref, readnoise_ref, satlevel_ref, ra_ref, dec_ref, pixscale_ref = read_header(header_ref, keywords, log)


    # function to run SExtractor on fraction of the image, applied
    # below to new and/or ref image
    def sex_fraction (base, sexcat, pixscale, imtype, header, log):
        fwhm, fwhm_std = run_sextractor(base+'.fits', sexcat, C.sex_cfg, C.sex_par,
                                        pixscale, log, fit_psf=False, return_fwhm=True,
                                        fraction=C.fwhm_imafrac, fwhm=5.0, save_bkg=False,
                                        update_vignet=False)
        log.info('fwhm_{}: {:.3f} +- {:.3f}'.format(imtype, fwhm, fwhm_std))
        # if SEEING keyword exists, report its value in the log
        if 'SEEING' in header:
            log.info('fwhm from header: ' + str(header['SEEING']))

        return fwhm, fwhm_std
            
    # if [new_fits] is not defined, [fwhm_new]=None ensures that code
    # does not crash in function [update_vignet_size] which uses both
    # [fwhm_new] and [fwhm_max]
    fwhm_new = None
    if new:
        # run SExtractor for seeing estimate of new_fits and ref_fits;
        # both new and ref need to have their fwhm determined before
        # continuing, as both [fwhm_new] and [fwhm_ref] are required
        # to determine the VIGNET size set in the full SExtractor run
        sexcat_new = base_new+'.sexcat_ldac'
        fwhm_new, fwhm_std_new = sex_fraction(base_new, sexcat_new, pixscale_new, 'new',
                                              header_new, log)

    fwhm_ref = None
    if ref:
        # do the same for the reference image
        sexcat_ref = base_ref+'.sexcat_ldac'
        fwhm_ref, fwhm_std_ref = sex_fraction(base_ref, sexcat_ref, pixscale_ref, 'ref',
                                              header_ref, log)

    # function to run SExtractor on full image, followed by Astrometry.net
    # to find the WCS solution, applied below to new and/or ref image
    def sex_wcs (base, sexcat, sex_params, pixscale, fwhm, update_vignet,
                 fits_mask, ra, dec, xsize, ysize, log):

        # run SExtractor on full image
        if not os.path.isfile(sexcat) or C.redo:
            result = run_sextractor(base+'.fits', sexcat, C.sex_cfg, sex_params,
                                    pixscale, log, fit_psf=False, return_fwhm=False,
                                    fraction=1.0, fwhm=fwhm, save_bkg=True,
                                    update_vignet=update_vignet, mask=fits_mask)
            # copy the LDAC binary fits table output from SExtractor (with
            # '_ldac' in the name) to a normal binary fits table;
            # Astrometry.net needs the latter, but PSFEx needs the former,
            # so keep both
            ldac2fits (sexcat, sexcat.replace('_ldac',''), log)

        # determine WCS solution of new_fits
        fits_wcs = base+'_wcs.fits'
        if not os.path.isfile(fits_wcs) or C.redo:
            if not C.skip_wcs:
                result = run_wcs(base+'.fits', fits_wcs, ra, dec, pixscale, xsize, ysize, log)
            else:
                # just copy original image to _wcs.fits image
                cmd = ['cp', base+'.fits', fits_wcs]
                result = subprocess.call(cmd)               
                # same for the SExtractor catalog
                cmd = ['cp', base+'.sexcat', base+'_wcs.sexcat']
                result = subprocess.call(cmd)                

        # read the header including WCS info
        with fits.open(fits_wcs) as hdulist:
            header_wcs = hdulist[0].header
        # if .wcs header file does not exist (e.g. if
        # [C.skip_wcs]==True), then create it here as it is used in
        # various places
        wcsfile = base+'.wcs'
        if not os.path.isfile(wcsfile):
            hdu = fits.PrimaryHDU(header=header_wcs)
            hdu.writeto(wcsfile)

        return header_wcs

    if new:
        # now run SExtractor on the full image with the above seeing
        # estimate, saving the background images
        # if mask is present, provide it to SExtractor as the flag image
        new_fits_mask = base_new+'_mask.fits'
        if not os.path.isfile(new_fits_mask): new_fits_mask = None
        header_new_wcs = sex_wcs(base_new, sexcat_new, C.sex_par, pixscale_new, fwhm_new, True,
                                 new_fits_mask, ra_new, dec_new, xsize_new, ysize_new, log)

    if ref:
        # same steps for the reference image
        ref_fits_mask = base_ref+'_mask.fits'
        if not os.path.isfile(ref_fits_mask): ref_fits_mask = None   
        header_ref_wcs = sex_wcs(base_ref, sexcat_ref, C.sex_par_ref, pixscale_ref, fwhm_ref, False,
                                 ref_fits_mask, ra_ref, dec_ref, xsize_ref, ysize_ref, log)
        # N.B.: two differences with new image: SExtractor parameter
        # file (new: C.sex_ar, ref: C.sex_par_ref) and update_vignet
        # boolean (new: True, ref: False). For the ref image, this
        # will lead to the VIGNET size to be as defined in the
        # parameter file [C.sex_par_ref], which by default is set to
        # the large value: (99,99). Instead of scaling it to the FWHM
        # and [C.psf_radius]. This is to be able to compare different
        # new images with different FWHMs to the same reference image
        # without needing to run SExtractor and possibly also PSFEx
        # again for the reference image.

            
    # initialise [ref_fits_remap] here to None; this is used below in
    # call to [prep_image_subtraction] if either [new_fits] or
    # [ref_fits] are not defined
    ref_fits_remap = None        
    if new and ref:
        # remap ref to new
        ref_fits_remap = base_ref+'_wcs_remap.fits'
        if not os.path.isfile(ref_fits_remap) or C.redo:
            # if reference image is poorly sampled, could use bilinear
            # interpolation for the remapping using SWarp - this
            # removes artefacts around bright stars (see Fig.6 in the
            # SWarp User's Guide). However, despite these artefacts,
            # the Scorr image still appears to be better with LANCZOS3
            # than when BILINEAR is used.
            resampling_type='LANCZOS3' 
            # if fwhm_ref <= 2: resampling_type='BILINEAR'
            result = run_remap(base_new+'_wcs.fits', base_ref+'_wcs.fits', ref_fits_remap,
                               [ysize_new, xsize_new], gain=gain_new, log=log, config=C.swarp_cfg,
                               resampling_type=resampling_type, resample='Y')

        # initialize full output images
        data_D_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_S_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_Scorr_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_Fpsf_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_Fpsferr_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_new_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_ref_full = np.ndarray((ysize_new, xsize_new), dtype='float32')


    # determine cutouts
    if new:
        xsize = xsize_new
        ysize = ysize_new
    else:
        xsize = xsize_ref
        ysize = ysize_ref
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = \
        centers_cutouts(C.subimage_size, ysize, xsize, log)
    nxsubs = xsize/C.subimage_size
    nysubs = ysize/C.subimage_size
    
    ysize_fft = C.subimage_size + 2*C.subimage_border
    xsize_fft = C.subimage_size + 2*C.subimage_border
    nsubs = centers.shape[0]
    if C.verbose:
        log.info('nsubs ' + str(nsubs))
        #for i in range(nsubs):
        #    log.info('i ' + str(i))
        #    log.info('cuts_ima[i] ' + str(cuts_ima[i]))
        #    log.info('cuts_ima_fft[i] ' + str(cuts_ima_fft[i]))
        #    log.info('cuts_fft[i] ' + str(cuts_fft[i]))
            
    # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with new,
    # ref, psf and background images
    if new:
        data_new, psf_new, psf_orig_new, data_new_bkg, data_new_bkg_std = \
            prep_optimal_subtraction(base_new+'_wcs.fits', nsubs, 'new', fwhm_new, log,
                                     fits_mask=new_fits_mask)

    # same for [ref_fits]; if either [new_fits] was not defined,
    # [ref_fits_remap] will be None
    if ref_fits is not None:
        data_ref, psf_ref, psf_orig_ref, data_ref_bkg, data_ref_bkg_std = \
            prep_optimal_subtraction(base_ref+'_wcs.fits', nsubs, 'ref', fwhm_ref, log,
                                     fits_mask=ref_fits_mask, ref_fits_remap=ref_fits_remap)

    if C.verbose and new:
        log.info('data_new.dtype {}'.format(data_new.dtype))
        log.info('psf_new.dtype {}'.format(psf_new.dtype))
        log.info('data_new_bkg.dtype {}'.format(data_new_bkg.dtype))
        log.info('data_new_bkg_std.dtype {}'.format(data_new_bkg_std.dtype))
    
    if new and ref:
        # get x, y and fratios from matching PSFex stars across entire frame
        x_fratio, y_fratio, fratio, dra, ddec = get_fratio_radec(base_new+'.psfexcat',
                                                                 base_ref+'.psfexcat',
                                                                 base_new+'_wcs.sexcat',
                                                                 base_ref+'_wcs.sexcat',log)    
        # convert to pixels
        dx = dra / pixscale_new
        dy = ddec / pixscale_new 

        # fratio is in counts, convert to electrons, in case gains of new
        # and ref images are not identical
        fratio *= gain_new / gain_ref
    
        dr = np.sqrt(dx**2 + dy**2)
        if C.verbose: log.info('standard deviation dr over full frame [pixels]: ' +str(np.std(dr)) )
        dr_full = np.sqrt(np.median(dr)**2 + np.std(dr)**2)
        dx_full = np.sqrt(np.median(dx)**2 + np.std(dx)**2)
        dy_full = np.sqrt(np.median(dy)**2 + np.std(dy)**2)
        #dr_full = np.std(dr)
        #dx_full = np.std(dx)
        #dy_full = np.sdata_new, psf_new, psf_orig_new, data_new_bkg, data_new_bkg_stdtd(dy)
        if C.verbose:
            log.info('np.median(dr), np.std(dr) [pixels]: ' + str(np.median(dr)) + ', ' + str(np.std(dr)))
            log.info('np.median(dx), np.std(dx) [pixels]: ' + str(np.median(dx)) + ', ' + str(np.std(dx)))
            log.info('np.median(dy), np.std(dy) [pixels]: ' + str(np.median(dy)) + ', ' + str(np.std(dy)))
            log.info('dr_full, dx_full, dy_full [pixels]: ' + str(dr_full) + ', ' + str(dx_full) + ', ' + str(dy_full))
    
        #fratio_median, fratio_std = np.median(fratio), np.std(fratio)
        fratio_mean_full, fratio_std_full, fratio_median_full = clipped_stats(fratio, nsigma=2, log=log)
        if C.verbose:
            log.info('fratio_mean_full, fratio_std_full, fratio_median_full: ' + str(fratio_mean_full) + ', ' + str(fratio_std_full) + ', ' + str(fratio_median_full))
            
        if C.make_plots:
                
            def plot (x, y, limits, xlabel, ylabel, filename, annotate=True):
                plt.axis(limits)
                plt.plot(x, y, 'go', markersize=5, markeredgecolor='k')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
                if annotate:
                    plt.annotate('STD(dx): {:.3f} pixels'.format(np.std(dx)), xy=(0.02,0.95), xycoords='axes fraction')
                    plt.annotate('STD(dy): {:.3f} pixels'.format(np.std(dy)), xy=(0.02,0.90), xycoords='axes fraction')
                    plt.annotate('STD(dr): {:.3f} pixels'.format(np.std(dr)), xy=(0.02,0.85), xycoords='axes fraction')
                if filename != '': plt.savefig(filename)
                if C.show_plots: plt.show()
                plt.close()

            plot (x_fratio, y_fratio, (0,xsize_new,0,ysize_new), 'x (pixels)', 'y (pixels)',
                  base_newref+'_xy.pdf', annotate=False)
            plot (dx, dy, (-1,1,-1,1), 'dx (pixels)', 'dy (pixels)',
                  base_newref+'_dxdy.pdf')
            plot (x_fratio, dr, (0,xsize_new,0,1), 'x (pixels)', 'dr (pixels)',
                  base_newref+'_drx.pdf', annotate=False)
            plot (y_fratio, dr, (0,ysize_new,0,1), 'y (pixels)', 'dr (pixels)',
                  base_newref+'_dry.pdf', annotate=False)
            plot (x_fratio, dx, (0,xsize_new,-1,1), 'x (pixels)', 'dx (pixels)',
                  base_newref+'_dxx.pdf')
            plot (y_fratio, dy, (0,ysize_new,-1,1), 'y (pixels)', 'dy (pixels)',
                  base_newref+'_dyy.pdf')
            # plot dr as function of distance from the image center
            xcenter = xsize_new/2
            ycenter = ysize_new/2
            dist = np.sqrt((x_fratio-xcenter)**2 + (y_fratio-ycenter)**2)
            plot (dist, dr, (0,np.amax(dist),0,1), 'distance from image center (pixels)',
                  'dr (pixels)', base_newref+'_drdist.pdf')

        # initialize fakestar flux arrays if fake star(s) are being added
        # - this is to make a comparison plot of the input and output flux
        if C.nfakestars>0:
            fakestar_flux_input = np.ndarray(nsubs)
            fakestar_flux_output = np.ndarray(nsubs)
            fakestar_fluxerr_output = np.ndarray(nsubs)        
            fakestar_s2n_output = np.ndarray(nsubs)
        
        start_time2 = os.times()
            
        log.info('Executing run_ZOGY on subimages ...')

    def zogy_subloop (nsub):
    #for nsub in range(nsubs):

        if C.timing: tloop = time.time()
        
        if C.verbose:
            log.info('Nsub: ' +str(nsub+1))
            log.info('----------')
            
        # refer to background and STD subimage with a shorter
        # parameter name
        bkg_new = data_new_bkg[nsub]
        bkg_ref = data_ref_bkg[nsub]
        std_new = data_new_bkg_std[nsub]
        std_ref = data_ref_bkg_std[nsub]

        # replace infinite values and nans with the background
        mask_infnan = ~np.isfinite(data_new[nsub])
        data_new[nsub][mask_infnan] = bkg_new[mask_infnan]
        mask_infnan = ~np.isfinite(data_ref[nsub])
        data_ref[nsub][mask_infnan] = bkg_ref[mask_infnan]

        # pixels with zero values in ref need to be set to zero in new
        # as well, to avoid subtracting non-overlapping image part
        mask_refzero = (data_ref[nsub]==0.)
        data_new[nsub][mask_refzero] = 0.
        
        # replace values <= 0 with the background
        mask_nonpos = (data_new[nsub] <= 0.)
        data_new[nsub][mask_nonpos] = bkg_new[mask_nonpos]
        mask_nonpos = (data_ref[nsub] <= 0.)
        data_ref[nsub][mask_nonpos] = bkg_ref[mask_nonpos]

        # good place to make the corresponding variance images
        # N.B.: these are single images (i.e. not a cube) the size of
        # a subimage, so does not need the [nsub] index
        var_new = data_new[nsub] + readnoise_new**2 
        var_ref = data_ref[nsub] + readnoise_ref**2
        # alternative:
        #var_new = data_new[nsub] - bkg_new + std_new**2
        #var_ref = data_ref[nsub] - bkg_ref + std_ref**2
        
        if C.nfakestars>0:
            # add fake star(s) to new image
            if C.nfakestars==1:
                # place it at the center of the new subimage
                xpos = xsize_fft/2
                ypos = ysize_fft/2
                psf_hsize = psf_size_new/2
                index_temp = [slice(ypos-psf_hsize, ypos+psf_hsize+1),
                              slice(xpos-psf_hsize, xpos+psf_hsize+1)]
                # Use function [flux_optimal_s2n] to estimate flux needed
                # for star with S/N of [C.fakestar_s2n].
                fakestar_flux, fakestar_data = flux_optimal_s2n (psf_orig_new[nsub],
                                                                 data_new[nsub][index_temp],
                                                                 bkg_new[index_temp], readnoise_new,
                                                                 C.fakestar_s2n, fwhm=fwhm_new)
                # multiply psf_orig_new to contain fakestar_flux
                psf_fakestar = psf_orig_new[nsub] * fakestar_flux
                # add fake star to new image
                data_new[nsub][index_temp] += psf_fakestar
                # and variance image
                var_new[index_temp] += psf_fakestar
                
                if C.verbose:
                    log.info('fakestar_flux: {} e-'.format(fakestar_flux))
                    flux, fluxerr, mask = flux_optimal(psf_orig_new[nsub], psf_orig_new[nsub],
                                                       fakestar_data, bkg_new[index_temp],
                                                       std_new[index_temp], readnoise_new, log=log)
                    log.info('recovered flux, fluxerr, S/N: ' + str(flux) + ', ' + str(fluxerr) + ', ' + str(flux/fluxerr))
                
                    # check S/N with Eq. 51 from Zackay & Ofek 2017, ApJ, 836, 187

                    s2n = get_s2n_ZO(psf_orig_new[nsub], fakestar_data, bkg_new[index_temp], readnoise_new)
                    log.info('S/N check (Eq. 51 Zackay & Ofek 2017): ' + str(s2n))
                    
                    # check S/N with Eqs. from Naylor (1998)
                    flux, fluxerr = get_optflux_Naylor(psf_orig_new[nsub], fakestar_data,
                                                       bkg_new[index_temp],
                                                       fakestar_data+readnoise_new**2)
                    log.info('Naylor recovered flux, fluxerr, S/N: ' + str(flux) + ', ' + str(fluxerr) + ', ' + str(flux/fluxerr))
            

            else:
                # place stars in random positions across the subimage,
                # keeping C.subimage_border + psf_size_new/2 pixels off
                # each edge
                edge = C.subimage_border + psf_size_new/2 + 1
                xpos_rand = np.random.rand(C.nfakestars)*(xsize_fft-2*edge) + edge
                ypos_rand = np.random.rand(C.nfakestars)*(ysize_fft-2*edge) + edge
                for nstar in range(C.nfakestars):
                    xpos = np.int(xpos_rand[nstar])
                    ypos = np.int(ypos_rand[nstar])
                    psf_hsize = psf_size_new/2
                    index_temp = [slice(ypos-psf_hsize, ypos+psf_hsize+1),
                                  slice(xpos-psf_hsize, xpos+psf_hsize+1)]
                    fakestar_flux, fakestar_data = flux_optimal_s2n (psf_orig_new[nsub],
                                                                     data_new[nsub][index_temp],
                                                                     bkg_new[index_temp], readnoise_new,
                                                                     C.fakestar_s2n, fwhm=fwhm_new)
                    psf_fakestar = psf_orig_new[nsub] * fakestar_flux
                    data_new[nsub][index_temp] += psf_fakestar                
                    var_new[index_temp] += psf_fakestar
                
            # for plot of input vs. output flux; in case C.nfakestars >
            # 1, only the flux from the last one is recorded
            fakestar_flux_input[nsub] = fakestar_flux
                    
        # subtract the background
        data_new[nsub] -= bkg_new
        data_ref[nsub] -= bkg_ref

        # replace saturated pixel values with zero
        #data_new[nsub][data_new[nsub] > 0.95*satlevel_new] = 0.
        #data_ref[nsub][data_ref[nsub] > 0.95*satlevel_ref] = 0.

        # read subcut, which defines the pixel indices [y1 y2 x1 x2]
        # identifying the corners of the subimage in the entire
        # input/output image coordinate frame
        subcut = cuts_ima[nsub]
                
        # start with full-frame values
        fratio_mean, fratio_std, fratio_median = fratio_mean_full, fratio_std_full, fratio_median_full
        if C.fratio_local:
            # get median fratio from PSFex stars across subimage
            #subcut = cuts_ima[nsub]
            # convert x,y_fratio pixel coordinates to indices
            y_fratio_index = (y_fratio-0.5).astype(int)
            x_fratio_index = (x_fratio-0.5).astype(int)
            # mask of entries in fratio arrays that lie within current subimage
            mask_sub_fratio = ((y_fratio_index >= subcut[0]) & (y_fratio_index < subcut[1]) & 
                               (x_fratio_index >= subcut[2]) & (x_fratio_index < subcut[3]))
            # require at least 10 values
            if np.sum(mask_sub_fratio) >= 10:
                # determine local fratios
                fratio_mean, fratio_std, fratio_median = clipped_stats(fratio[mask_sub_fratio],
                                                                       nsigma=2, log=log)
                if C.verbose:
                    log.info('sub image fratios: ' + str(fratio[mask_sub_fratio]))
                    
        # adopt full-frame values, also if local fratio_median is more
        # than 2 sigma (full frame) away from the full-frame value
        if not C.fratio_local or (np.abs(fratio_median-fratio_median_full)/fratio_std_full > 2.):
            fratio_mean, fratio_std, fratio_median = fratio_mean_full, fratio_std_full, fratio_median_full

        if C.verbose:
            log.info('np.abs(fratio_median-fratio_median_full)/fratio_std_full: ' + str(np.abs(fratio_median-fratio_median_full)/fratio_std_full))
            log.info('adopted fratio_mean, fratio_std, fratio_median: ' + str(fratio_mean) + ', ' + str(fratio_std) + ', ' + str(fratio_median))
            
        # and the same for dx and dy
        if C.dxdy_local and any(mask_sub_fratio):
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
        if C.verbose:
            log.info('f_new, f_ref: ' + str(f_new) + ', ' + str(f_ref))
            log.info('dx_sub, dy_sub: ' + str(dx_sub) + ', ' + str(dy_sub))

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


        #if nthreads > 1: lock.acquire()    
            
        # call Barak's function
        data_D, data_S, data_Scorr, data_Fpsf, data_Fpsferr = run_ZOGY(data_ref[nsub], data_new[nsub], 
                                                                       psf_ref[nsub], psf_new[nsub], 
                                                                       np.median(std_ref),
                                                                       np.median(std_new), 
                                                                       f_ref, f_new,
                                                                       var_ref, var_new,
                                                                       dx_sub, dy_sub, log)
        #if nthreads > 1: lock.release()

        # check that robust std of Scorr is around unity
        if C.verbose:
            mean_Scorr, std_Scorr, median_Scorr = clipped_stats(data_Scorr, clip_zeros=False,
                                                                log=log)
            log.info('mean_Scorr, median_Scorr, std_Scorr: ' + str(mean_Scorr) + ', ' + str(median_Scorr) + ', ' + str(std_Scorr))
            mean_S, std_S, median_S = clipped_stats(data_S, clip_zeros=False, log=log)
            log.info('mean_S, median_S, std_S: ' + str(mean_S) + ', ' + str(median_S) + ', ' + str(std_S))
            
        # if fake star(s) was (were) added to the subimages, compare
        # the input flux (the same for the entire subimage) with the
        # PSF flux determined by run_ZOGY. If multiple stars were
        # added, then this comparison is done for the last of them.
        if C.nfakestars>0:
            fakestar_flux_output[nsub] = data_Fpsf[xpos, ypos]
            fakestar_fluxerr_output[nsub] = data_Fpsferr[xpos, ypos]
            # and S/N from Scorr
            fakestar_s2n_output[nsub] = data_Scorr[xpos, ypos]
            
        # put sub images without the borders into output frames
        #subcut = cuts_ima[nsub]
        index_subcut = [slice(subcut[0],subcut[1]), slice(subcut[2],subcut[3])]
        x1, y1 = C.subimage_border, C.subimage_border
        x2, y2 = x1+C.subimage_size, y1+C.subimage_size
        index_extract = [slice(y1,y2), slice(x1,x2)]

        data_D_full[index_subcut] = data_D[index_extract] #/ gain_new
        data_S_full[index_subcut] = data_S[index_extract]
        data_Scorr_full[index_subcut] = data_Scorr[index_extract]
        data_Fpsf_full[index_subcut] = data_Fpsf[index_extract]
        data_Fpsferr_full[index_subcut] = data_Fpsferr[index_extract]
        data_new_full[index_subcut] = (data_new[nsub][index_extract] +
                                       bkg_new[index_extract]) #/ gain_new
        data_ref_full[index_subcut] = (data_ref[nsub][index_extract] +
                                       bkg_ref[index_extract]) #/ gain_ref

        if C.display and (nsub==0 or nsub==nysubs-1 or nsub==nsubs/2 or
                        nsub==nsubs-nysubs or nsub==nsubs-1):

            # just for displaying purpose:
            fits.writeto('D.fits', data_D.astype(np.float32), overwrite=True)
            fits.writeto('S.fits', data_S.astype(np.float32), overwrite=True)
            fits.writeto('Scorr.fits', data_Scorr.astype(np.float32), overwrite=True)
            #fits.writeto('Scorr_1sigma.fits', data_Scorr_1sigma, overwrite=True)
        
            # write new and ref subimages to fits
            subname = '_sub'+str(nsub)
            newname = base_new+'_wcs'+subname+'.fits'
            #fits.writeto(newname, ((data_new[nsub]+bkg_new)/gain_new).astype(np.float32), overwrite=True)
            fits.writeto(newname, data_new[nsub].astype(np.float32), overwrite=True)
            refname = base_ref+'_wcs'+subname+'.fits'
            #fits.writeto(refname, ((data_ref[nsub]+bkg_ref)/gain_ref).astype(np.float32), overwrite=True)
            fits.writeto(refname, data_ref[nsub].astype(np.float32), overwrite=True)
            # variance images
            fits.writeto('Vnew.fits', var_new.astype(np.float32), overwrite=True)
            fits.writeto('Vref.fits', var_ref.astype(np.float32), overwrite=True)
            # background images
            fits.writeto('bkg_new.fits', bkg_new.astype(np.float32), overwrite=True)
            fits.writeto('bkg_ref.fits', bkg_ref.astype(np.float32), overwrite=True)
            fits.writeto('std_new.fits', std_new.astype(np.float32), overwrite=True)
            fits.writeto('std_ref.fits', std_ref.astype(np.float32), overwrite=True)

            # and display
            cmd = ['ds9','-zscale',newname,refname,'D.fits','S.fits','Scorr.fits']
            cmd += ['Vnew.fits', 'Vref.fits', 'bkg_new.fits', 'bkg_ref.fits',
                    'std_new.fits', 'std_ref.fits',
                    'VSn.fits', 'VSr.fits', 'VSn_ast.fits', 'VSr_ast.fits',
                    'Sn.fits', 'Sr.fits', 'kn.fits', 'kr.fits', 'Pn_hat.fits', 'Pr_hat.fits',
                    'psf_ima_config_new_sub'+str(nsub)+'.fits', 'psf_ima_config_ref_sub'+str(nsub)+'.fits',
                    'psf_ima_resized_norm_new_sub'+str(nsub)+'.fits', 'psf_ima_resized_norm_ref_sub'+str(nsub)+'.fits', 
                    'psf_ima_center_new_sub'+str(nsub)+'.fits', 'psf_ima_center_ref_sub'+str(nsub)+'.fits', 
                    'psf_ima_shift_new_sub'+str(nsub)+'.fits', 'psf_ima_shift_ref_sub'+str(nsub)+'.fits']

            result = subprocess.call(cmd)

        if C.timing:
            log.info('wall-time spent in nsub loop ' +str(time.time()-tloop))
            log.info('peak memory (in GB) used in nsub loop {}'.
                     format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    # call above function [zogy_subloop] with pool.map
    # only if both [new_fits] and [ref_fits] are defined
    if new and ref:
        pool = ThreadPool(1)
        lock = Lock()
        pool.map(zogy_subloop, range(nsubs))
        pool.close()
        pool.join()
        
        # compute statistics on full Scorr image and show histogram
        if C.verbose:
            mean_Scorr, std_Scorr, median_Scorr = clipped_stats (data_Scorr_full, clip_zeros=False,
                                                                 make_hist=C.make_plots,
                                                                 name_hist=base_newref+'_Scorr_hist.pdf',
                                                                 log=log)
            log.info('mean_Scorr, median_Scorr, std_Scorr: ' + str(mean_Scorr) + ', ' +
                     str(median_Scorr) + ', ' + str(std_Scorr))
    
        # write full images to fits
        header_new.add_comment('Propagated header from new image (including WCS solution)')
        fits.writeto(base_newref+'_D.fits', data_D_full, header_new_wcs, overwrite=True)
        fits.writeto(base_newref+'_Scorr.fits', data_Scorr_full, header_new_wcs, overwrite=True)
        fits.writeto(base_newref+'_Scorr_neg.fits', np.negative(data_Scorr_full), header_new_wcs, overwrite=True)
        fits.writeto(base_newref+'_Fpsf.fits', data_Fpsf_full, header_new_wcs, overwrite=True)
        fits.writeto(base_newref+'_Fpsferr.fits', data_Fpsferr_full, header_new_wcs, overwrite=True)

        if C.display:
            fits.writeto('new.fits', data_new_full, header_new_wcs, overwrite=True)
            fits.writeto('ref.fits', data_ref_full, header_ref_wcs, overwrite=True)
            fits.writeto(base_newref+'_S.fits', data_S_full, header_new_wcs, overwrite=True)

        fits_mask_comb = base_newref+'_mask.fits'
        if not os.path.isfile(fits_mask_comb):
            fits_mask_comb = None
        
        # find transients using function [find_transients]
        transcat_pos = base_newref+'.transcat_pos'
        result = get_trans (base_newref+'_Scorr.fits', transcat_pos, pixscale_new, 'green', log,
                            mask=fits_mask_comb)
        # add PSF fluxes and errors 
        result = get_trans_flux (transcat_pos, data_Fpsf_full, data_Fpsferr_full)

        transcat_neg = base_newref+'.transcat_neg'
        result = get_trans (base_newref+'_Scorr_neg.fits', transcat_neg, pixscale_new, 'pink', log,
                            mask=fits_mask_comb)
        result = get_trans_flux (transcat_neg, data_Fpsf_full, data_Fpsferr_full)

    if args.telescope=='meerlicht' or args.telescope=='blackgem':
        # using the function [format_cat], write the new, ref and
        # transient output catalogues with the desired format, where the
        # thumbnail images (new, ref, D and Scorr) around each transient
        # are added as array columns in the transient catalogue.
        # new catalogue
        if new:
            cat_new = base_new+'_wcs.sexcat_fluxopt'
            cat_new_out = base_new+'_wcs.cat'
            result = format_cat (cat_new, cat_new_out, log, cat_type='new',
                                 header_toadd=header_new)
        # ref catalogue
        if ref:
            cat_ref = base_ref+'_wcs.sexcat_fluxopt'
            cat_ref_out = base_ref+'_wcs.cat'
            result = format_cat (cat_ref, cat_ref_out, log, cat_type='ref',
                                 header_toadd=header_ref)
        # trans catalogue
        if new and ref:
            cat_trans = base_newref+'.transcat_pos'
            cat_trans_out = base_newref+'.cat_trans'
            thumbnail_data = [data_new_full, data_ref_full, data_D_full, data_Scorr_full]
            thumbnail_keys = ['THUMBNAIL_RED', 'THUMBNAIL_REF', 'THUMBNAIL_D', 'THUMBNAIL_SCORR']
            result = format_cat (cat_trans, cat_trans_out, log, cat_type='trans',
                                 thumbnail_data=thumbnail_data, thumbnail_keys=thumbnail_keys,
                                 thumbnail_size=32, header_toadd=header_new)

    end_time = os.times()
    if new and ref:
        dt_usr  = end_time[2] - start_time2[2]
        dt_sys  = end_time[3] - start_time2[3]
        dt_wall = end_time[4] - start_time2[4]

        log.info("Elapsed user time in {0}:  {1:.3f} sec".format("optsub", dt_usr))
        log.info("Elapsed CPU time in {0}:  {1:.3f} sec".format("optsub", dt_sys))
        log.info("Elapsed wall time in {0}:  {1:.3f} sec".format("optsub", dt_wall))
        
    dt_usr  = end_time[2] - start_time1[2]
    dt_sys  = end_time[3] - start_time1[3]
    dt_wall = end_time[4] - start_time1[4]

    log.info("Elapsed user time in {0}:  {1:.3f} sec".format("total", dt_usr))
    log.info("Elapsed CPU time in {0}:  {1:.3f} sec".format("total", dt_sys))
    log.info("Elapsed wall time in {0}:  {1:.3f} sec".format("total", dt_wall))

    if new and ref:
        # make comparison plot of flux input and output
        if C.make_plots and C.nfakestars>0:
        
            x = np.arange(nsubs)+1
            y = fakestar_flux_input
            plt.plot(x, y, 'ko')
            plt.xlabel('subimage number')
            plt.ylabel('true flux (e-)')
            plt.title('fake stars true input flux')
            plt.savefig(base_newref+'_fakestar_flux_input.pdf')
            if C.show_plots: plt.show()
            plt.close()

            #plt.axis((0,nsubs,0,2))
            x = np.arange(nsubs)+1
            y = (fakestar_flux_input - fakestar_flux_output) / fakestar_flux_input
            yerr = fakestar_fluxerr_output / fakestar_flux_input
            plt.errorbar(x, y, yerr=yerr, fmt='ko')
            plt.xlabel('subimage number')
            plt.ylabel('(true flux - ZOGY flux) / true flux')
            plt.title('fake stars true input flux vs. ZOGY Fpsf output flux')
            plt.savefig(base_newref+'_fakestar_flux_input_vs_ZOGYoutput.pdf')
            if C.show_plots: plt.show()
            plt.close()

            # same for S/N as determined by Scorr
            y = fakestar_s2n_output
            plt.plot(x, y, 'ko')
            plt.xlabel('subimage number')
            plt.ylabel('S/N from Scorr')
            plt.title('signal-to-noise ratio from Scorr')
            plt.savefig(base_newref+'_fakestar_S2N_ZOGYoutput.pdf')
            if C.show_plots: plt.show()
            plt.close()
        
        # and display
        if C.display:
            if C.nfakestars>0:
                cmd = ['ds9','-zscale','new.fits','ref.fits','D.fits','Scorr.fits']
            else:
                cmd = ['ds9','-zscale',new_fits,ref_fits_remap,'D.fits','Scorr.fits']
            if os.path.isfile(transcat_pos+'_ds9regions'):
                cmd += ['-regions', transcat_pos+'_ds9regions']
            if os.path.isfile(transcat_neg+'_ds9regions'):
                cmd += ['-regions', transcat_neg+'_ds9regions']
            result = subprocess.call(cmd)


################################################################################

def xy_index_ref (ysize, xsize, wcs_new, wcs_ref, log):

    """Given an image with shape [ysize, xsize] and WCS solutions in
    [wcs_new] and [wcs_ref], return the masks [mask_new, mask_ref]
    identifying the pixels in the ref image that correspond to the
    pixels in the new image, i.e. the new image pixels [mask_new] map
    onto the reference image pixels [mask_ref]."""

    if C.timing: t = time.time()

    # Sample xx and yy every [step] pixels in each axis to perform the
    # mapping on a coarse grid, which is interpolated and expanded
    # back to the input grid below. This is done to avoid running the
    # functions [wcs.all_pix2world] and [wcs.all_world2pix] on each
    # and every image pixel.
    nsteps = 1000
    # It is important to use the [np.linspace] function below to
    # ensure that the coarse grid includes the first and last pixel of
    # each axis; the "+1" converts the indices to pixel coordinates.
    yy, xx = np.meshgrid(np.linspace(0, ysize-1, nsteps)+1,
                         np.linspace(0, xsize-1, nsteps)+1,
                         indexing='ij')
    ysize_coarse, xsize_coarse = yy.shape
    
    # flatten xx and yy into x and y
    x = xx.flatten()
    y = yy.flatten()
    
    # use [wcs_ref] file to get RA, DEC of pixel coordinates x, y
    wcs = WCS(wcs_new)
    ra, dec = wcs.all_pix2world(x, y, 1)

    # use [wcs_new] file to get x_ref, y_ref pixel coordinates
    # corresponding to ra, dec
    wcs = WCS(wcs_ref)
    x_ref, y_ref = wcs.all_world2pix(ra, dec, 1)

    # reshape
    xx_ref_coarse = x_ref.reshape((ysize_coarse, xsize_coarse))
    yy_ref_coarse = y_ref.reshape((ysize_coarse, xsize_coarse))
    
    # resize coarse grid to full input grid
    xx_ref = ndimage.zoom(xx_ref_coarse, 1.*xsize/xsize_coarse, order=1)
    yy_ref = ndimage.zoom(yy_ref_coarse, 1.*ysize/ysize_coarse, order=1)

    # define mask defining pixels in reference frame
    mask_new = ((xx_ref>=0.5) & (yy_ref>=0.5) & (xx_ref<=xsize+0.5) & (yy_ref<=ysize+0.5))
    
    # flip to get mask_new of pixels in new frame
    mask_ref = np.flip(np.flip(mask_new,1),0)

    if C.timing:
        log.info('wall-time spent in xy_index_ref ' + str(time.time()-t))
        log.info('peak memory (in GB) used in xy_index_ref {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return mask_new, mask_ref
        
        
################################################################################

def format_cat (cat_in, cat_out, log, thumbnail_data=None, thumbnail_keys=None,
                thumbnail_size=32, cat_type=None, header_toadd=None):

    """Function that formats binary fits table [cat_in] according to
        MeerLICHT/BlackGEM specifications and saves the resulting
        binary fits table [cat_out].

    """
    
    if C.timing: t = time.time()

    with fits.open(cat_in) as hdulist:
        prihdu = hdulist[0]
        header = hdulist[1].header
        data = hdulist[1].data

    if header_toadd is not None:
        header += header_toadd

    thumbnail_size2 = str(thumbnail_size**2)
        
    # this [formats] dictionary lists the output format, the output
    # column unit, and the desired format
    formats = {
        'NUMBER':         ['J', ''     , 'uint16'],
        'XWIN_IMAGE':     ['E', 'pix'  , 'flt32' ],
        'YWIN_IMAGE':     ['E', 'pix'  , 'flt32' ],
        'ERRX2WIN_IMAGE': ['E', 'pix^2', 'flt16' ],
        'ERRY2WIN_IMAGE': ['E', 'pix^2', 'flt16' ],
        'ERRXYWIN_IMAGE': ['E', 'pix^2', 'flt16' ],
        'X2WIN_IMAGE':    ['E', 'pix^2', 'flt16' ],
        'Y2WIN_IMAGE':    ['E', 'pix^2', 'flt16' ],
        'XYWIN_IMAGE':    ['E', 'pix^2', 'flt16' ],
        'ELONGATION':     ['E', ''     , 'flt16' ],
        'ALPHAWIN_J2000': ['E', 'deg'  , 'flt32' ],
        'DELTAWIN_J2000': ['E', 'deg'  , 'flt32' ],
        'FLAGS':          ['I', ''     , 'uint8' ],
        'IMAFLAGS_ISO':   ['I', ''     , 'uint8' ],
        'FWHM_IMAGE':     ['E', 'pix'  , 'flt16' ],
        'CLASS_STAR':     ['E', ''     , 'flt16' ],
        'FLUX_APER':      ['E', 'e-/s' , 'flt32' ],
        'FLUXERR_APER':   ['E', 'e-/s' , 'flt16' ],
        'BACKGROUND':     ['E', 'e-/s' , 'flt16' ],
        'FLUX_MAX':       ['E', 'e-/s' , 'flt16' ],
        'FLUX_AUTO':      ['E', 'e-/s' , 'flt32' ],
        'FLUXERR_AUTO':   ['E', 'e-/s' , 'flt16' ],
        'KRON_RADIUS':    ['E', 'pix'  , 'flt16' ],
        'FLUX_ISO':       ['E', 'e-/s' , 'flt32' ],
        'FLUXERR_ISO':    ['E', 'e-/s' , 'flt16' ],
        'ISOAREA_IMAGE':  ['E', 'pix^2', 'flt16' ],
        'MU_MAX':         ['E', 'e-/s' , 'flt16' ],
        'FLUX_RADIUS':    ['E', 'pix'  , 'flt16' ],
        'FLUX_PETRO':     ['E', 'e-/s' , 'flt32' ],
        'FLUXERR_PETRO':  ['E', 'e-/s' , 'flt16' ],
        'PETRO_RADIUS':   ['E', 'pix'  , 'flt16' ],
        'FLUX_OPT':       ['E', 'e-/s' , 'flt32' ],
        'FLUXERR_OPT':    ['E', 'e-/s' , 'flt16' ],
        'MAG_OPT':        ['E', 'e-/s' , 'flt32' ],
        'MAGERR_OPT':     ['E', 'e-/s' , 'flt16' ],
        'FLUX_PSF':       ['E', 'e-/s' , 'flt32' ],
        'FLUXERR_PSF':    ['E', 'e-/s' , 'flt16' ],
        'THUMBNAIL_RED':  [thumbnail_size2+'E', 'e-/s' , 'flt16' ],
        'THUMBNAIL_REF':  [thumbnail_size2+'E', 'e-/s' , 'flt16' ],
        'THUMBNAIL_D':    [thumbnail_size2+'E', 'e-/s' , 'flt16' ],
        'THUMBNAIL_SCORR':[thumbnail_size2+'E', 'e-/s' , 'flt16' ]
    }

    if cat_type is None:
        keys_to_record = data.names
    elif cat_type == 'ref':
        keys_to_record = ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE',
                          'ERRX2WIN_IMAGE', 'ERRY2WIN_IMAGE', 'ERRXYWIN_IMAGE', 
                          'X2WIN_IMAGE', 'Y2WIN_IMAGE', 'XYWIN_IMAGE',   
                          'ELONGATION', 'ALPHAWIN_J2000', 'DELTAWIN_J2000',
                          'FLAGS', 'IMAFLAGS_ISO', 'FWHM_IMAGE', 'CLASS_STAR',    
                          'FLUX_APER', 'FLUXERR_APER',  'BACKGROUND', 'FLUX_MAX',      
                          'FLUX_AUTO', 'FLUXERR_AUTO', 'KRON_RADIUS',   
                          'FLUX_ISO', 'FLUXERR_ISO', 'ISOAREA_IMAGE', 'MU_MAX', 'FLUX_RADIUS',
                          'FLUX_PETRO', 'FLUXERR_PETRO', 'PETRO_RADIUS',
                          'FLUX_OPT', 'FLUXERR_OPT', 'MAG_OPT', 'MAGERR_OPT']  
    elif cat_type == 'new':
        keys_to_record = ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE',
                          'ERRX2WIN_IMAGE', 'ERRY2WIN_IMAGE', 'ERRXYWIN_IMAGE', 
                          'ELONGATION', 'ALPHAWIN_J2000', 'DELTAWIN_J2000',
                          'FLAGS', 'IMAFLAGS_ISO', 'FWHM_IMAGE', 'CLASS_STAR',    
                          'FLUX_APER', 'FLUXERR_APER',  'BACKGROUND', 'FLUX_MAX',      
                          'FLUX_OPT', 'FLUXERR_OPT', 'MAG_OPT', 'MAGERR_OPT']  
    elif cat_type == 'trans':
        keys_to_record = ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE',
                          'ERRX2WIN_IMAGE', 'ERRY2WIN_IMAGE', 'ERRXYWIN_IMAGE', 
                          'ELONGATION', 'ALPHAWIN_J2000', 'DELTAWIN_J2000',
                          'FLAGS', 'IMAFLAGS_ISO', 'FWHM_IMAGE', 'CLASS_STAR',    
                          'FLUX_MAX', 'FLUX_PSF', 'FLUXERR_PSF']
        
    columns = []
    for key in keys_to_record:
        if key=='FLUX_APER' or key=='FLUXERR_APER':
            # loop apertures
            for i_ap in range(len(C.apphot_radii)):
                name = key+'_R'+str(C.apphot_radii[i_ap])+'xFWHM'
                col = fits.Column(name=name, format=formats[key][0], unit=formats[key][1], 
                                  disp=formats[key][2], array=data[key][:,i_ap])
                columns.append(col)
        else:
            if key in data.names:
                col = fits.Column(name=key, format=formats[key][0], unit=formats[key][1], 
                                  disp=formats[key][2], array=data[key])
                columns.append(col)
        
    # add [thumbnails]
    if thumbnail_data is not None and thumbnail_keys is not None:
        
        # number of thumbnail images to add
        nthumbnails = len(thumbnail_keys)
        
        # coordinates to loop
        xcoords = data['XWIN_IMAGE']
        ycoords = data['YWIN_IMAGE']
        ncoords = len(xcoords)

        # loop thumbnails
        for i_tn in range(nthumbnails):

            ysize, xsize = thumbnail_data[i_tn].shape
            
            # initialise output column
            data_col = np.zeros((ncoords,thumbnail_size,thumbnail_size))
            
            # loop x,y coordinates
            for i_pos in range(ncoords):

                # get index around x,y position using function [get_index_around_xy]
                index = get_index_around_xy(ysize, xsize, ycoords[i_pos], xcoords[i_pos],
                                            thumbnail_size)
                # record in data_col
                data_col[i_pos] = thumbnail_data[i_tn][index]
                        
            # add column to table
            dim_str = '('+str(thumbnail_size)+','+str(thumbnail_size)+')'
            key = thumbnail_keys[i_tn]
            col = fits.Column(name=key, format=formats[key][0], unit=formats[key][1], 
                              disp=formats[key][2], array=data_col, dim=dim_str)
            columns.append(col)

            
    coldefs = fits.ColDefs(columns)
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.header += header
    hdu.writeto(cat_out, overwrite=True)
    
    if C.timing:
        log.info('wall-time spent in format_cat ' + str(time.time()-t))
        log.info('peak memory (in GB) used in format_cat {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

        

################################################################################

def get_index_around_xy(ysize, xsize, ycoord, xcoord, size):

    if size % 2 == 0:
        oddsized = False
        xpos = int(xcoord)
        ypos = int(ycoord)
    else:
        oddsized = True
        xpos = int(xcoord-0.5)
        ypos = int(ycoord-0.5)

    hsize = size/2

    # if footprint is partially off the image, just go ahead
    # with the pixels on the image
    y1 = max(0, ypos-hsize)
    x1 = max(0, xpos-hsize)
    if oddsized:
        y2 = min(ysize, ypos+hsize+1)
        x2 = min(xsize, xpos+hsize+1)
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
        y2 = min(ysize, ypos+hsize)
        x2 = min(xsize, xpos+hsize)
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

    return [slice(y1,y2),slice(x1,x2)]


################################################################################

def get_trans(Scorr, transcat, pixscale, color, log, mask=None):

    if not os.path.isfile(transcat) or C.redo:
        result = run_sextractor(Scorr, transcat, C.sex_cfg_trans, C.sex_par_trans,
                                pixscale, log, fit_psf=False, return_fwhm=False,
                                fraction=1.0, fwhm=fwhm_new, save_bkg=False,
                                update_vignet=False, mask=mask)

    # read catalog
    with fits.open(transcat) as hdulist:
        data = hdulist[2].data

        # select objects based on FLUX_MAX, FLAG, CLASS_STAR
        index = ((data['FLUX_MAX'] >= C.transient_nsigma) &
                 (data['FLAGS'] == 0))

        # add constraint on IMAFLAGS_ISO which should be present in
        # SExtractor output in case that a mask image was provided
        if 'IMAFLAGS_ISO' in data.names:
            index = ((index) & (data['IMAFLAGS_ISO'] == 0))

        data_trans = data[:][index]

        hdulist[2].data = data_trans
        hdulist_new = fits.HDUList([hdulist[0], hdulist[2]])
        #hdulist_new.writeto(transcat+'_selected', overwrite=True)
        hdulist_new.writeto(transcat, overwrite=True)
        hdulist_new.close()

    # prepare ds9 region file
    f = open(transcat+'_ds9regions', 'w')
    ntrans = np.shape(data_trans)[0]
    log.info('ntrans: {}'.format(ntrans))
    for i in range(ntrans):
        f.write('circle({},{},{}) # color={} width=2 text={{{}}} font="times 7"\n'.
                format(data_trans['XWIN_IMAGE'][i], data_trans['YWIN_IMAGE'][i],
                       2.*fwhm_new, color, i))
    f.close()

    # find transient sources in Scorr
    #Scorr_peaks = ndimage.filters.maximum_filter(data_Scorr_full)
    #Scorr_peaks = Scorr_peaks[Scorr_peaks > C.transient_nsigma]
    #Scorr_peaks_mask = (data_Scorr_full == Scorr_peaks)
    # alternavitvely, use SExtractor:
    #fits.writeto('Scorr.fits', data_Scorr_full, overwrite=True)


################################################################################

def get_trans_flux (transcat, data_Fpsf, data_Fpsferr):
    
    # read catalog
    with fits.open(transcat) as hdulist:
        data = hdulist[1].data

    # coordinates to loop
    xcoords = data['XWIN_IMAGE']
    ycoords = data['YWIN_IMAGE']
    ncoords = len(xcoords)
 
    # initialise output fluxes
    flux_psf = np.zeros(ncoords)
    fluxerr_psf = np.zeros(ncoords)
        
    # loop coordinates
    for i in range(ncoords):

        xpos = int(xcoords[i]-0.5)
        ypos = int(ycoords[i]-0.5)

        flux_psf[i] = data_Fpsf[ypos, xpos]
        fluxerr_psf[i] = data_Fpsferr[ypos, xpos]
    
    # add PSF fluxes to catalog
    data = append_fields(data, ['FLUX_PSF','FLUXERR_PSF'],
                         [flux_psf, fluxerr_psf], usemask=False, asrecarray=True)

    # write updated catalog to file
    fits.writeto(transcat, data, overwrite=True)
    
    
################################################################################

def get_psfoptflux_xycoords (psfex_bintable, D, S, S_std, RON, xcoords, ycoords,
                             dx2, dy2, dxy, satlevel=50000,
                             psf_oddsized=True, psffit=False, log=None):
    
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
        
    log.info('Executing get_psfoptflux_xycoords ...')
    t = time.time()

    # make sure x and y have same length
    if np.isscalar(xcoords) or np.isscalar(ycoords):
        log.error('Error: xcoords and ycoords should be arrays')
    else:
        assert len(xcoords) == len(ycoords)
        
    # initialize output arrays
    ncoords = len(xcoords)
    flux_opt = np.zeros(ncoords)
    fluxerr_opt = np.zeros(ncoords)
    if psffit:
        flux_psf = np.zeros(ncoords)
        fluxerr_psf = np.zeros(ncoords)
        xshift_psf = np.zeros(ncoords)
        yshift_psf = np.zeros(ncoords)
        chi2_psf = np.zeros(ncoords)
        
    D_replaced = np.copy(D)
    
    # get dimensions of D
    ysize, xsize = np.shape(D)

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
    if C.verbose:
        log.info('polzero1                   ' + str(polzero1))
        log.info('polscal1                   ' + str(polscal1))
        log.info('polzero2                   ' + str(polzero2))
        log.info('polscal2                   ' + str(polscal2))
        log.info('order polynomial:          ' + str(poldeg))
        log.info('PSFex FWHM:                ' + str(psf_fwhm))
        log.info('PSF sampling size (pixels):' + str(psf_samp))
        log.info('PSF size defined in config:' + str(psf_size_config))
        
    # initialize output PSF array

    # [psf_size] is the PSF size in image pixels,
    # i.e. [psf_size_config] multiplied by the PSF sampling (roughly
    # 4-5 pixels per FWHM) which is set by the [C.psf_sampling] parameter.
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

    # define psf_hsize
    psf_hsize = psf_size/2
    
    if C.timing: log.info('wall-time spent in get_psfoptflux_xycoords before the loop ' + str(time.time()-t))

    # previously this was a loop; now turned to a function to
    # try pool.map multithreading below
    # loop coordinates
    #for i in range(ncoords):
    def loop_psfoptflux_xycoords(i):
    
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
            #continue
            return
            
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
        D_sub = D[index]
        if not np.isscalar(S):
            S_sub = S[index]
            S_std_sub = S_std[index]
        else:
            S_sub = S
            S_std_sub = S_std


        # get P_shift and P_noshift
        x = (int(xcoords[i]) - polzero1) / polscal1
        y = (int(ycoords[i]) - polzero2) / polscal2
        
        if ncoords==1 or C.use_single_psf:
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
        # shift if the PSF image is constructed to be even)
        if psf_oddsized:
            xshift = xcoords[i]-np.round(xcoords[i])
            yshift = ycoords[i]-np.round(ycoords[i])
        else:
            xshift = (xcoords[i]-int(xcoords[i])-0.5)
            yshift = (ycoords[i]-int(ycoords[i])-0.5)
                    
        # if [psf_samp_update] is lower than unity, then perform this
        # shift before the PSF image is re-sampled to the image
        # pixels, as the original PSF will have higher resolution in
        # that case
        order = 3
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
            # only required if psf-fitting is performed
            if psffit:
                psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update, order=order)
        else:
            # resample PSF image at image pixel scale
            psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update, order=order)
            # shift PSF
            psf_ima_shift_resized = ndimage.shift(psf_ima_resized, (yshift, xshift), order=order)
            # using Eran's function:
            #psf_ima_shift_resized = image_shift_fft(psf_ima_resized, xshift, yshift)

        # clean and normalize PSF
        psf_shift = clean_norm_psf(psf_ima_shift_resized, C.psf_clean_factor)
        # also return normalized PSF without any shift
        # only required if psf-fitting is performed
        if psffit:
            psf_noshift = clean_norm_psf(psf_ima_resized, C.psf_clean_factor)

        # extract subsection from psf_shift and psf_noshift
        y1_P = y1 - (ypos - psf_hsize)
        x1_P = x1 - (xpos - psf_hsize)
        y2_P = y2 - (ypos - psf_hsize)
        x2_P = x2 - (xpos - psf_hsize)
        index_P = [slice(y1_P,y2_P),slice(x1_P,x2_P)]
        
        P_shift = psf_shift[index_P]
        # only required if psf-fitting is performed
        if psffit:
            P_noshift = psf_noshift[index_P]
        
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
        flux_opt[i], fluxerr_opt[i], mask_opt = flux_optimal (P_shift, D_sub, S_sub, S_std_sub, RON,
                                                              mask_use=mask_nonsat,
                                                              dx2=dx2[i], dy2=dy2[i], dxy=dxy[i], log=log)
        
        # if psffit=True, perform PSF fitting
        if psffit:
            flux_psf[i], fluxerr_psf[i], xshift_psf[i], yshift_psf[i], chi2_psf[i] =\
            flux_psffit (P_noshift, D_sub, S_sub, RON, flux_opt[i],
                         xshift, yshift, mask_use=mask_nonsat, log=log)

            # xshift_psf and yshift_psf are the shifts with respect to
            # the integer xpos and ypos positions defined at the top
            # of this loop over the sources. This is because the fit
            # is using the PSF image that is not shifted
            # (P_noshift) to the exact source position.  Redefine
            # them here with respect to the fractional coordinates
            # xcoords and ycoords
            xshift_psf[i] += xshift
            yshift_psf[i] += yshift
                        
        #print 'i, flux_opt[i], fluxerr_opt[i], flux_psf[i], fluxerr_psf[i]',\
        #    i, flux_opt[i], fluxerr_opt[i], flux_psf[i], fluxerr_psf[i]

        
        # if any pixels close to the center of the object are
        # saturated, replace them
        mask_inner = (P_shift >= 0.25*np.amax(P_shift))
        if np.any(mask_sat[mask_inner]):

            # replace saturated values
            D_replaced[index][mask_sat] = P_shift[mask_sat] * flux_opt[i] + S_sub[mask_sat]

            # and put through [flux_optimal] once more without a
            # saturated pixel mask
            #flux_opt[i], fluxerr_opt[i], mask_opt = flux_optimal (P_shift, D_sub,
            #                                                      S_sub, S_std_sub, RON,
            #                                                      dx2=dx2[i], dy2=dy2[i], dxy=dxy[i])
            #D_replaced[index][mask_opt] = P_shift[mask_opt] * flux_opt[i] + S_sub[mask_opt]

            if False and C.display:
                result = ds9_arrays(D_sub=D_sub, mask_nonsat=mask_nonsat.astype(int), 
                                    maskopt=mask_opt.astype(int),
                                    S_sub=S_sub, P=P_shift,
                                    D_replaced=D_replaced[index])

    if C.timing: t1 = time.time()
    pool = ThreadPool(1)
    pool.map(loop_psfoptflux_xycoords, range(ncoords))
    pool.close()
    pool.join()
    if C.timing: log.info('wall-time spent in loop_psfoptflux_xycoords pool ' + str(time.time() - t1))
    if C.verbose: log.info('ncoords: {}'.format(ncoords))
    
    if False and C.display:
        ds9_arrays(D=D, D_replaced=D_replaced)

    if C.timing:
        log.info('wall-time spent in get_psfoptflux_xycoords ' + str(time.time()-t))
        log.info('peak memory (in GB) used in get_psfoptflux_xycoords {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
    
    if psffit:
        x_psf = xcoords + xshift_psf
        y_psf = ycoords + yshift_psf
        return flux_opt, fluxerr_opt, D_replaced, flux_psf, fluxerr_psf, x_psf, y_psf
    else:
        return flux_opt, fluxerr_opt, D_replaced
            

################################################################################

def flux_psffit(P, D, S, RON, flux_opt, xshift, yshift, mask_use=None, log=None):

    # if S is a scalar, expand it to 2D array
    if np.isscalar(S):
        S = np.ndarray(P.shape, dtype='float32').fill(S)
        
    # replace negative values in D with the sky
    D[D<0] = S[D<0]

    # define objective function: returns the array to be minimized
    def fcn2min(params, P, D, S, RON, mask_use):

        xshift = params['xshift'].value
        yshift = params['yshift'].value
        flux_psf = params['flux_psf'].value

        # shift the PSF image to the exact pixel position
        P_shift = ndimage.shift(P, (yshift, xshift))
        # alternatively, use Eran's shift function 
        #P_shift = image_shift_fft(P, xshift, yshift)
                
        #print 'sum of P, P_shift:', np.sum(P), np.sum(P_shift)
    
        # make sure that P_shift is equal to 1
        #P_shift /= np.sum(P_shift)
        
        # scale the shifted PSF
        model = flux_psf * P_shift

        # error estimate from the data themselves:
        #err = np.sqrt(RON**2 + D)

        # error estimate from the PSF model (at the moment this breaks
        # down at some point with a bright star combined with negative
        # values in the outskirts of P, which leads to nan values
        # because some RON**2+model+S values are negative):

        var = RON**2 + model + S
        var[var<0] = RON**2 + D[var<0]
        err = np.sqrt(var)
                
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
    params.add('flux_psf', value=flux_opt, min=0., vary=True)

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

def flux_optimal (P, D, S, S_std, RON, nsigma_inner=10, P_noshift=None,
                  nsigma_outer=5, max_iters=10, epsilon=1e-3, mask_use=None,
                  add_V_ast=False, dx2=0, dy2=0, dxy=0, log=None):
    
    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], data [D], sky [S], sky standard deviation [S_std]
    and read-out noise [RON].  This makes use of function
    [get_optflux] or [get_optflux_Eran]."""

    # if S is a scalar, expand it to 2D array
    if np.isscalar(S):
        S = np.ndarray(P.shape, dtype='float32').fill(S)

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
                    
        #print 'i, flux_opt, fluxerr_opt', i, flux_opt, fluxerr_opt, abs(flux_opt_old-flux_opt)/flux_opt, abs(flux_opt_old-flux_opt)/fluxerr_opt

        # original stopping criterium
        #if abs(flux_opt_old-flux_opt)/abs(flux_opt) < epsilon:
        #    break
        # suggestion by Steven:
        if abs(flux_opt_old-flux_opt)/fluxerr_opt < 0.1:
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
        log.info('no. of rejected pixels: ' + str(np.sum(mask_use==False)))
        log.info('np.amax((D - flux_opt * P - S)**2 / V): ' + str(np.amax(sigma2)))
        
        if not add_V_ast:
            V_ast = np.zeros(D.shape, dtype='float32')
            
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
    image with a required S/N [C.fakestar_s2n].

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
                  verbose=False, make_hist=False, name_hist=None, log=None):

    
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
        #log.info('i: {:d}, mean: {:.3f}, std: {:.3f}'.format(i, mean, std))
        if abs(mean_old-mean)/abs(mean) < epsilon:
            break
        mean_old = mean
        index = ((array>(mean-nsigma*std)) & (array<(mean+nsigma*std)))
        array = np.copy(array[index])
        
    # add median
    if get_median:
        median = np.median(array)
        if abs(median-mean)/mean>0.1:
            log.info('Warning: mean and median in clipped_stats differ by more than 10%')
            log.info('mean: ' + str(mean))
            log.info('median: ' + str(median))
            
    # and mode
    if get_mode:
        bins = np.arange(mean-nsigma*std, mean+nsigma*std, mode_binsize)
        hist, bin_edges = np.histogram(array, bins)
        index = np.argmax(hist)
        mode = (bins[index]+bins[index+1])/2.
        if abs(mode-mean)/mean>0.1:
            log.info('Warning: mean and mode in clipped_stats differ by more than 10%')

    if make_hist:
        bins = np.linspace(mean-nsigma*std, mean+nsigma*std)
        plt.hist(np.ravel(array), bins, color='green')
        x1,x2,y1,y2 = plt.axis()
        plt.plot([mean, mean], [y2,y1], color='black')
        plt.plot([mean+std, mean+std], [y2,y1], color='black', linestyle='--')
        plt.plot([mean-std, mean-std], [y2,y1], color='black', linestyle='--')
        title = 'mean (black line): {:.3f}, std: {:.3f}'.format(mean, std)
        if get_median:
            plt.plot([median, median], [y2,y1], color='blue')
            title += ', median (blue line): {:.3f}'.format(median)
        if get_mode:
            plt.plot([mode, mode], [y2,y1], color='red')
            title += ', mode (red line): {:.3f}'.format(mode)
        plt.title(title)
        if C.make_plots:
            if name_hist is None: name_hist = 'clipped_stats_hist.pdf'
            plt.savefig(name_hist)
        if C.show_plots: plt.show()
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

def read_header(header, keywords, log):

    values = []
    for i in range(len(keywords)):
        if keywords[i] in header:
            if C.verbose:
                log.info('key, value: ' + keywords[i] +', '+ str(header[keywords[i]]))
            values.append(header[keywords[i]])
        else:
            log.error('Error: keyword ' + keywords[i] + ' not present in header - change keyword name or add manually')
            raise SystemExit
    return values


################################################################################
    
def prep_optimal_subtraction(input_fits, nsubs, imtype, fwhm, log,
                             fits_mask=None, ref_fits_remap=None):

    log.info('Executing prep_optimal_subtraction ...')
    t = time.time()
       
    if imtype=='new':
        base = base_new
    else:
        base = base_ref

    # read in input_fits header
    with fits.open(input_fits) as hdulist:
        header_wcs = hdulist[0].header
        data_wcs = hdulist[0].data.astype(np.float32)

    # get gain, readnoise, pixscale and saturation level from header_wcs
    keywords = [C.key_gain, C.key_ron, C.key_pixscale, C.key_satlevel]
    gain, readnoise, pixscale, satlevel = read_header(header_wcs, keywords, log)
    ysize, xsize = np.shape(data_wcs)
        
    # read in background image
    fits_bkg = base+'_bkg.fits'
    with fits.open(fits_bkg) as hdulist:
        data_bkg = hdulist[0].data.astype(np.float32)
    # read in background std image
    fits_bkg_std = base+'_bkg_std.fits'
    with fits.open(fits_bkg_std) as hdulist:
        data_bkg_std = hdulist[0].data.astype(np.float16)

    # and mask image if not None
    if fits_mask is not None:
        with fits.open(fits_mask) as hdulist:
            data_mask = (hdulist[0].data+0.5).astype(np.int16)
            
    # if remapped image is provided, read that also
    if ref_fits_remap is not None:
        with fits.open(ref_fits_remap) as hdulist:
            data_ref_remap = hdulist[0].data.astype(np.float32)

        # and prepare the remapped background, std and mask images
        # instead of using SWarp to remap the background, std and mask
        # images, use the function [xy_index_ref] to determine indices
        # of pixels in reference image that correspond to pixels in
        # new image: image_new[mask_new] correspond to
        # image_ref[mask_ref]. This mapping needs to be done only
        # once, and can be used for all three remappings.
        mask_new, mask_ref = xy_index_ref (ysize, xsize,
                                           base_new+'.wcs', base_ref+'.wcs', log)

        # this function applies this mapping to an input fits image,
        # and returns the remapped data
        def get_data_remap (image_fits, mask_new, mask_ref):
            with fits.open(image_fits) as hdulist:
                data = hdulist[0].data
            # initialise remapped image to zero
            data_remap = np.zeros(data.shape)
            data_remap[mask_new] += data[mask_ref]
            return data_remap
                
        ref_fits_bkg_std = base_ref+'_bkg_std.fits'
        ref_fits_bkg = base_ref+'_bkg.fits'
        # remap reference image background
        data_ref_bkg_remap = get_data_remap(ref_fits_bkg, mask_new, mask_ref).astype(np.float32)
        # remap reference image background std
        data_ref_bkg_std_remap = get_data_remap(ref_fits_bkg_std, mask_new, mask_ref).astype(np.float16)
        # remap mask image if it exists
        if fits_mask is not None:
            data_ref_mask_remap = (get_data_remap(fits_mask, mask_new, mask_ref)+0.5).astype(np.int16)
                
            # prepare combined mask
            # first read in mask of new image
            data_new_mask = None
            new_fits_mask = base_new+'_mask.fits'
            if os.path.isfile(new_fits_mask):
                with fits.open(new_fits_mask) as hdulist:
                    data_new_mask = (hdulist[0].data+0.5).astype(np.int16)
            else:
                log.info('Warning: {} does not exist, while {} does'.
                         format(new_fits_mask, ref_fits_mask))
            # not sure how to combine; just add for now
            if data_new_mask is not None:
                data_mask_comb = (data_new_mask + data_ref_mask_remap).astype(np.int16)
            else:
                data_mask_comb = data_ref_mask_remap.astype(np.int16)
            # write to fits
            fits_mask_comb = base_new+'_'+base_ref+'_mask.fits'
            fits.writeto(fits_mask_comb, data_mask_comb, overwrite=True)
            
    # convert counts to electrons
    satlevel *= gain
    data_wcs *= gain
    data_bkg *= gain
    data_bkg_std *= gain
    if ref_fits_remap is not None:
        data_ref_remap *= gain
        data_ref_bkg_remap *= gain
        data_ref_bkg_std_remap *= gain
        
        # print warning if any pixel value is not finite
    if np.any(~np.isfinite(data_wcs)):
        log.info('Warning: not all pixel values are finite')
        log.info('         replacing NANs with zeros and +-inf with large +-numbers')        
        # replace NANs with zero, and +-infinity with large +-numbers
        #data_wcs = np.nan_to_num(data_wcs)


    # determine psf of input image with get_psf function - needs to be
    # done before optimal fluxes are determined
    psf, psf_orig = get_psf(input_fits, header_wcs, nsubs, imtype, fwhm, pixscale, log)

    # -------------------------------
    # determination of optimal fluxes
    # -------------------------------

    # Get estimate of optimal flux for all sources in the new and ref
    # image if not already done so.

    # [mypsffit] determines if PSF-fitting part is also performed;
    # this is different from SExtractor PSF-fitting
    mypsffit = False
    
    newcat = input_fits.replace('.fits', '.sexcat_fluxopt')
    if not os.path.isfile(newcat) or C.redo:
    
        if C.timing: t1 = time.time()
        log.info('deriving optimal fluxes ...')
    
        # first read SExtractor fits table
        sexcat = input_fits.replace('.fits', '.sexcat')
        with fits.open(sexcat) as hdulist:
            data_sex = hdulist[1].data

        # remove entries close to the edge
        if 'IMAFLAGS_ISO' in data_sex.columns.names:
            mask_use = (data_sex['IMAFLAGS_ISO'] != 2)
            data_sex = data_sex[:][mask_use]
            
        # read in positions and their errors
        xwin = data_sex['XWIN_IMAGE']
        ywin = data_sex['YWIN_IMAGE']
        errx2win = data_sex['ERRX2WIN_IMAGE']
        erry2win = data_sex['ERRY2WIN_IMAGE']
        errxywin = data_sex['ERRXYWIN_IMAGE']

        psfex_bintable = input_fits.replace('_wcs.fits', '.psf')

        if mypsffit:
            flux_opt, fluxerr_opt, data_replaced, flux_psf, fluxerr_psf, x_psf, y_psf =\
                get_psfoptflux_xycoords (psfex_bintable, data_wcs, data_bkg, data_bkg_std,
                                         readnoise, xwin, ywin, errx2win, erry2win, errxywin,
                                         satlevel=satlevel, psffit=mypsffit, log=log)
        else:
            flux_opt, fluxerr_opt, data_replaced =\
                get_psfoptflux_xycoords (psfex_bintable, data_wcs, data_bkg, data_bkg_std,
                                         readnoise, xwin, ywin, errx2win, erry2win, errxywin,
                                         satlevel=satlevel, log=log)
                
        if C.timing:
            log.info('wall-time spent deriving optimal fluxes ' + str(time.time()-t1))
            log.info('peak memory (in GB) used deriving optimal fluxes {}'.
                     format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

        if C.timing: t2 = time.time()

        # determine image zeropoint if ML/BG calibration catalog exists
        if os.path.isfile(C.cal_cat):

            # read calibration catalog
            with fits.open(C.cal_cat) as hdulist:
                data_cal = hdulist[1].data

            # use .wcs file to get RA, DEC of central pixel
            if imtype=='new':
                wcs = WCS(base_new+'.wcs')
            else:
                wcs = WCS(base_ref+'.wcs')
            ra_center, dec_center = wcs.all_pix2world(xsize/2, ysize/2, 1)
            log.info('ra_center: {}, dec_center: {}'.format(ra_center, dec_center))
            
            # use function [find_stars] to select stars in calibration
            # catalog that are within the current field-of-view
            dist_deg = np.amax([xsize/2, ysize/2]) * pixscale / 3600.
            mask_field = find_stars(data_cal, ra_center, dec_center, dist_deg, log)
            data_cal = data_cal[:][mask_field]
                
            # read a few extra header keywords needed in [get_apply_zp]
            keywords = [C.key_exptime, 'FILTNAME', 'DATE-OBS']
            exptime, filt, obsdate = read_header(header_wcs, keywords, log)

            if C.verbose:
                log.info('exptime: {}, filter: {}, obsdate: {}'.format(exptime, filt, obsdate))
            
            ra_sex = data_sex['ALPHAWIN_J2000']
            dec_sex = data_sex['DELTAWIN_J2000']
            ra_cal = data_cal['RA']
            dec_cal = data_cal['DEC']
            mag_cal = data_cal[filt]
            magerr_cal = data_cal['err_'+str(filt)]

            # get airmasses
            airmass_sex = get_airmass(ra_sex, dec_sex, obsdate, log)
            log.info('median airmass: {}'.format(np.median(airmass_sex)))
            
            mag_opt, magerr_opt, zp, zp_std = \
                get_apply_zp(ra_sex, dec_sex, airmass_sex, flux_opt, fluxerr_opt,
                             ra_cal, dec_cal, mag_cal, magerr_cal, exptime, filt, imtype, log)
            

        data_sex = append_fields(data_sex, ['FLUX_OPT','FLUXERR_OPT'] ,
                                 [flux_opt, fluxerr_opt], usemask=False, asrecarray=True)
        data_sex = drop_fields(data_sex, 'VIGNET')

        if os.path.isfile(C.cal_cat):
            data_sex = append_fields(data_sex, ['MAG_OPT','MAGERR_OPT'] ,
                                     [mag_opt, magerr_opt], usemask=False, asrecarray=True)

        # write updated catalog to file
        fits.writeto(newcat, data_sex, overwrite=True)
                        
        if C.timing: log.info('wall-time spent creating binary fits table including fluxopt '
                            + str(time.time()-t2))
    

    # split full image into subimages to be used in run_ZOGY - this
    # needs to be done after determination of optimal fluxes as
    # otherwise the potential replacement of the saturated pixels will
    # not be taken into account

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(C.subimage_size,
                                                                       ysize, xsize, log)
    ysize_fft = C.subimage_size + 2*C.subimage_border
    xsize_fft = C.subimage_size + 2*C.subimage_border

    if ref_fits_remap is not None:
        data = data_ref_remap
        data_bkg = data_ref_bkg_remap
        data_bkg_std = data_ref_bkg_std_remap
        if fits_mask is not None:
            data_mask = data_ref_mask_remap
    else:
        data = data_wcs
    
    # fix pixels using [fixpix] function which requires data (remapped
    # image in case of reference image), mask and background
    if fits_mask is not None:
        data = fixpix (data, data_mask, data_bkg, log, satlevel=satlevel)
        
    if C.timing: t2 = time.time()

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

    if C.timing: log.info('wall-time spent filling fftdata cubes ' + str(time.time()-t2))

    if C.make_plots:

        # in case optimal flux block above was skipped, the SExtractor
        # catalogue with FLUX_OPT needs to be read in here to be able
        # to make the plots below
        try:
            data_sex['FLUX_AUTO'][0]
        except NameError:
            # read SExtractor fits table
            with fits.open(newcat) as hdulist:
                data_sex = hdulist[1].data
            # and define flux_opt and fluxerr_opt
            flux_opt = data_sex['FLUX_OPT']
            fluxerr_opt = data_sex['FLUXERR_OPT']
            if mypsffit:
                flux_mypsf = data_sex['FLUX_PSF']
                fluxerr_mypsf = data_sex['FLUXERR_PSF']
                x_psf = data_sex['X_PSF']
                y_psf = data_sex['Y_PSF']
        else:
            if C.verbose:
                log.info('data_sex array is already defined; no need to read it in')
                
        # compare flux_opt with flux_auto
        index = ((data_sex['FLUX_AUTO']>0) & (data_sex['FLAGS']==0))
        class_star = data_sex['CLASS_STAR'][index]
        flux_auto = data_sex['FLUX_AUTO'][index] * gain
        fluxerr_auto = data_sex['FLUXERR_AUTO'][index] * gain
        s2n_auto = flux_auto / fluxerr_auto
        flux_opt = flux_opt[index]
        fluxerr_opt = fluxerr_opt[index]
        x_win = data_sex['XWIN_IMAGE'][index]
        y_win = data_sex['YWIN_IMAGE'][index]
        fwhm_image = data_sex['FWHM_IMAGE'][index]
        if mypsffit:
            flux_mypsf = flux_psf[index]
            fluxerr_mypsf = fluxerr_psf[index]
            x_psf = x_psf[index]
            y_psf = y_psf[index]
            
        flux_diff = (flux_opt - flux_auto) / flux_auto
        limits = (1,2*np.amax(s2n_auto),-0.3,0.3)
        plot_scatter (s2n_auto, flux_diff, limits, class_star,
                      xlabel='S/N (AUTO)', ylabel='(FLUX_OPT - FLUX_AUTO) / FLUX_AUTO', 
                      filename=base+'_fluxopt_vs_fluxauto.pdf',
                      title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

        if mypsffit:
            # compare flux_mypsf with flux_auto
            flux_diff = (flux_mypsf - flux_auto) / flux_auto
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_MYPSF - FLUX_AUTO) / FLUX_AUTO', 
                          filename=base+'_fluxmypsf_vs_fluxauto.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
        
            # compare flux_opt with flux_mypsf
            flux_diff = (flux_opt - flux_mypsf) / flux_mypsf
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_OPT - FLUX_MYPSF) / FLUX_MYPSF', 
                          filename=base+'_fluxopt_vs_fluxmypsf.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            # compare x_win and y_win with x_psf and y_psf
            dist_win_psf = np.sqrt((x_win-x_psf)**2+(y_win-y_psf)**2)
            plot_scatter (s2n_auto, dist_win_psf, (1,2*np.amax(s2n_auto),-2.,2.), class_star, 
                          xlabel='S/N (AUTO)', ylabel='distance XY_WIN vs. XY_MYPSF', 
                          filename=base+'_xyposition_win_vs_mypsf_class.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
            plot_scatter (s2n_auto, dist_win_psf, (1,2*np.amax(s2n_auto),-2.,2.), fwhm_image, 
                          xlabel='S/N (AUTO)', ylabel='distance XY_WIN vs. XY_MYPSF', 
                          filename=base+'_xyposition_win_vs_mypsf_fwhm.pdf',
                          title='rainbow color coding follows FWHM_IMAGE')
            
        # compare flux_opt with flux_aper 2xFWHM
        for i in range(len(C.apphot_radii)):
            aper_str = str(C.apphot_radii[i])

            flux_aper = data_sex['FLUX_APER'][index,i] * gain
            fluxerr_aper = data_sex['FLUXERR_APER'][index,i] * gain
            flux_diff = (flux_opt - flux_aper) / flux_aper
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_OPT - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                          filename=base+'_fluxopt_vs_fluxaper'+aper_str+'xFWHM.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            flux_diff = (flux_auto - flux_aper) / flux_aper
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_AUTO - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                          filename=base+'_fluxauto_vs_fluxaper'+aper_str+'xFWHM.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            if mypsffit:
                flux_diff = (flux_mypsf - flux_aper) / flux_aper
                plot_scatter (s2n_auto, flux_diff, limits, class_star,
                              xlabel='S/N (AUTO)', ylabel='(FLUX_MYPSF - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                              filename=base+'_fluxmypsf_vs_fluxaper'+aper_str+'xFWHM.pdf',
                              title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
            

        # compare with flux_psf if psffit catalog available
        sexcat_ldac_psffit = input_fits.replace('.fits', '.sexcat_ldac_psffit')
        if os.path.isfile(sexcat_ldac_psffit):
            # read SExtractor psffit fits table
            with fits.open(sexcat_ldac_psffit) as hdulist:
                data_sex = hdulist[2].data
                
            flux_sexpsf = data_sex['FLUX_PSF'][index] * gain
            fluxerr_sexpsf = data_sex['FLUXERR_PSF'][index] * gain
            s2n_sexpsf = data_sex['FLUX_PSF'][index] / data_sex['FLUXERR_PSF'][index]
            
            flux_diff = (flux_sexpsf - flux_opt) / flux_opt
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_SEXPSF - FLUX_OPT) / FLUX_OPT', 
                          filename=base+'_fluxsexpsf_vs_fluxopt.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            if mypsffit:
                # and compare 'my' psf with SExtractor psf
                flux_diff = (flux_sexpsf - flux_mypsf) / flux_mypsf
                plot_scatter (s2n_auto, flux_diff, limits, class_star,
                              xlabel='S/N (AUTO)', ylabel='(FLUX_SEXPSF - FLUX_MYPSF) / FLUX_MYPSF', 
                              filename=base+'_fluxsexpsf_vs_fluxmypsf.pdf',
                              title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
            
            # and compare auto with SExtractor psf
            flux_diff = (flux_sexpsf - flux_auto) / flux_auto
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel='S/N (AUTO)', ylabel='(FLUX_SEXPSF - FLUX_AUTO) / FLUX_AUTO', 
                          filename=base+'_fluxsexpsf_vs_fluxauto.pdf',
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

        
    if C.timing:
        log.info('wall-time spent in prep_optimal_subtraction ' + str(time.time()-t))
        log.info('peak memory (in GB) used in prep_optimal_subtraction {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return fftdata, psf, psf_orig, fftdata_bkg, fftdata_bkg_std
    

################################################################################

def get_apply_zp (ra_sex, dec_sex, airmass_sex, flux_opt, fluxerr_opt,
                  ra_cal, dec_cal, mag_cal, magerr_cal, exptime, filt, imtype, log):

    if C.timing: t = time.time()
    log.info('Executing get_apply_zp ...')

    # maximum distance in degrees between sources to match
    dist_max = 1./3600

    # return zeropoints in array with same number of rows as ra_sex
    # allows to potentially build a zeropoint map as a function of x,y
    # coordinates
    nrows = np.shape(ra_sex)[0]
    zp_array = np.zeros(nrows)

    # instrumental magnitudes and errors
    mag_sex_inst = np.zeros(nrows)-1
    magerr_sex_inst = np.zeros(nrows)-1
    mask_pos = (flux_opt > 0.)
    mag_sex_inst[mask_pos] = -2.5*np.log10(flux_opt[mask_pos]/exptime)
    pogson = 2.5/np.log(10.)
    magerr_sex_inst[mask_pos] = pogson*fluxerr_opt[mask_pos]/flux_opt[mask_pos]

    ncal = np.shape(ra_cal)[0]
    # loop calibration stars and find a match in SExtractor sources
    for i in range(ncal):
        
        # make a big cut in declination in the SExtractor arrays to
        # speed up distance calculation below
        mask_cut = (np.abs(dec_sex-dec_cal[i])<=dist_max)
        ra_sex_temp = ra_sex[mask_cut]
        dec_sex_temp = dec_sex[mask_cut]
        
        # calculate distances using function [haversine]
        dist = haversine(ra_sex_temp, dec_sex_temp, ra_cal[i], dec_cal[i])

        # prepare match mask the size of SExtractor catalog
        mask_match = np.zeros(nrows).astype('bool')
        mask_match[mask_cut] = (dist <= dist_max)

        if np.sum(mask_match)==1:
            # there's one match, calculate its zeropoint
            # need to calculate airmass for each star, as around A=2,
            # difference in airmass across the FOV is 0.1, i.e. a 5% change
            zp_array[mask_match] = mag_cal[i] - mag_sex_inst[mask_match] + \
                                   airmass_sex[mask_match] * C.ext_coeff[filt]
            if C.verbose: 
                log.info('ra_cal: {}, dec_cal: {}, mag_cal: {}'.
                         format(ra_cal[i], dec_cal[i], mag_cal[i], zp_array[mask_match]))

    if imtype=='new':
        base = base_new
    else:
        base = base_ref
                
    # determine median zeropoint
    # use values with nonzero values and where flux_opt>0
    mask_nonzero = ((zp_array>0) & (mask_pos))
    zp_mean, zp_std, zp_median = clipped_stats(zp_array[mask_nonzero], make_hist=C.make_plots,
                                               name_hist=base+'_zp_hist.pdf', log=log)
    if C.verbose:
        log.info('number of useful calibration stars in the field: {}'.
                 format(np.sum(mask_nonzero)))
        log.info('zp_mean: {:.3f}, zp_median: {:.3f}, zp_std: {:.3f}'.
                 format(zp_mean, zp_median, zp_std))

    # now apply this zeropoint to SExtractor sources
    mag_sex = zp_median + mag_sex_inst - airmass_sex*C.ext_coeff[filt]
    # set magnitudes of sources with non-positive optimal fluxes to -1
    mag_sex[~mask_pos] = -1
    magerr_sex = magerr_sex_inst
    magerr_sex[~mask_pos] = -1
    # could add error in zeropoint determination
        
    if C.timing:
        log.info('wall-time spent in get_apply_zp ' + str(time.time() - t))
        log.info('peak memory (in GB) used in get_apply_zp {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return mag_sex, magerr_sex, zp_median, zp_std


################################################################################

def find_stars (data, ra, dec, radec_range, log):

    if C.timing: t = time.time()
    log.info('Executing find_stars ...')

    # find entries in [data] within [radec_range] of [ra] and [dec]
    index_data = np.zeros(data.shape[0]).astype('bool')
    # make a big cut in the data array to speed up calculations
    index_calc = (np.abs(data['DEC']-dec)<=radec_range)
    ra_cat = data['RA'][index_calc]
    dec_cat = data['DEC'][index_calc]

    # find within circle:
    #dsigma = haversine(ra_cat, dec_cat, ra, dec)
    #index_data[index_calc] = (dsigma<=radec_range)

    # find within box:
    dsigma_ra = haversine(ra_cat, dec_cat, ra, dec_cat)
    dsigma_dec = haversine(ra_cat, dec_cat, ra_cat, dec)
    index_data[index_calc] = ((dsigma_ra<=radec_range) & (dsigma_dec<=radec_range))

    if C.timing:
        log.info('wall-time spent in find_stars ' + str(time.time() - t))
        log.info('peak memory (in GB) used in find_stars {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return index_data

################################################################################
        
def haversine (ra1, dec1, ra2, dec2):

    """Function that calculates angle in degrees between RA, DEC
    coordinates ra1, dec1 and ra2, dec2. Input coordinates can be
    scalars or arrays. """
    
    # convert to radians
    ra1, ra2, dec1, dec2 = map(np.radians, [ra1, ra2, dec1, dec2])
    
    d_ra = np.abs(ra1-ra2)
    d_dec = np.abs(dec1-dec2)
    
    a = np.sin(d_dec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return map(np.degrees, [c])[0]


################################################################################

def get_airmass (ra, dec, obsdate, log):

    if C.timing: t = time.time()
    log.info('Executing get_airmass ...')

    location = EarthLocation(lat=C.obs_lat, lon=C.obs_long, height=C.obs_height)
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    coords_altaz = coords.transform_to(AltAz(obstime=Time(obsdate), location=location))

    if C.timing:
        log.info('wall-time spent in get_airmass ' + str(time.time() - t))
        log.info('peak memory (in GB) used in get_airmass {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return coords_altaz.secz

        
################################################################################

def fixpix (data, data_mask, data_bkg, log, satlevel=60000.):

    if C.timing: t = time.time()
    log.info('Executing fixpix ...')

    # now try to clean the image from artificially sharp features such
    # as saturated and pixels as defined in data_mask - the FFTs in
    # [run_ZOGY] produce large-scale features surrounding these sharp
    # features in the subtracted image. Currently the (KMTNet) masks
    # do not seem to completely define all the bad pixels/columns
    # correctly - temporarily remake a mask here from negative or
    # saturated pixels
    #mask = np.zeros(data.shape, dtype='int8')
    #mask[data < 0] = 1 
    #mask[data >= satlevel] = 4

    # try just replacing the edge pixels with the background
    data_fixed = np.copy(data)
    mask_replace = (data_mask==2)
    data_fixed[mask_replace] = data_bkg[mask_replace]

    # Replace pixels that correspond to zeros in data with the
    # background; this will ensure that the borders on the sides of
    # the entire image and the parts of the image where the new and
    # ref do not overlap can be handled by run_ZOGY.
    data_fixed[data_fixed==0] = data_bkg[data_fixed==0]

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

    if C.timing:
        log.info('wall-time spent in fixpix ' + str(time.time() - t))
        log.info('peak memory (in GB) used in fixpix {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    #ds9_arrays(data=data, data_fixed=data_fixed)

    return data_fixed

        
################################################################################

def get_back (data, data_objmask, log, use_photutils=False, clip=True):
    
    """Function that returns the background of the image [data].  If
    use_photutils is True then apply the photutils' Background2D,
    while otherwise a clipped median is determined for each subimage
    which is masked using SExtractor's '-OBJECTS' image provided in
    [data_objmask]. The subimages (with size: [C.bkg_boxsize]) are then
    median filtered and resized to the size of the input image."""

    if C.timing: t = time.time()
    log.info('Executing get_back ...')

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
        sigma_clip = SigmaClip(sigma=C.bkg_nsigma, iters=10)
        bkg_estimator = MedianBackground()
        # if C.bkg_boxsize does not fit integer times into the x- or
        # y-dimension of the shape, Background2D below fails if
        # edge_method='pad', which is the recommended method.  Use
        # edge_method='crop' instead.
        bkg = Background2D(data, C.bkg_boxsize, filter_size=C.bkg_filtersize,
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
            mean_full, std_full, median_full = clipped_stats(data[mask_use],
                                                             nsigma=C.bkg_nsigma, log=log)
        else:
            median_full = np.median(data[mask_use])
            std_full = np.std(data[mask_use])
        if C.verbose:
            log.info('Background median and std/RMS in object-masked image: ' + str(median_full) + ', ' + str(std_full))

        # loop through subimages the size of C.bkg_boxsize, and
        # determine median from the masked data
        ysize, xsize = data.shape
        centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(C.bkg_boxsize,
                                                                           ysize, xsize, log)        
        nsubs = centers.shape[0]

        # loop subimages
        if ysize % C.bkg_boxsize != 0 or xsize % C.bkg_boxsize !=0:
            log.info('Warning: [C.bkg_boxsize] does not fit integer times in image')
            log.info('         remaining pixels will be edge-padded')
        nysubs = ysize / C.bkg_boxsize
        nxsubs = xsize / C.bkg_boxsize
        # prepare output median and std output arrays
        mesh_median = np.ndarray(nsubs)
        mesh_std = np.ndarray(nsubs)

        mask_minsize = 0.5*C.bkg_boxsize**2

        # previously this was a loop; now turned to a function to
        # try pool.map multithreading below
        #def get_median_std (nsub):
        for nsub in range(nsubs):
            subcut = cuts_ima[nsub]
            data_sub = data[subcut[0]:subcut[1], subcut[2]:subcut[3]]
            mask_sub = mask_use[subcut[0]:subcut[1], subcut[2]:subcut[3]]
            if np.sum(mask_sub) > mask_minsize:
                if clip:
                    # get clipped_stats mean, std and median 
                    mean, std, median = clipped_stats(data_sub[mask_sub], clip_upper10=True,
                                                      nsigma=C.bkg_nsigma, log=log)
                else:
                    median = np.median(data_sub[mask_sub])
                    std = np.std(data_sub[mask_sub])
            else:
                # if less than half of the elements of mask_sub
                # are True, use values from entire masked image
                median, std = median_full, std_full
                #if C.verbose:
                #    log.info('Warning: using median and std of entire masked image for this background patch')
                #    log.info('nsub' + str(nsub))
                #    log.info('subcut' + str(subcut))
                #    log.info('np.sum(mask_sub) / C.bkg_boxsize**2: ' + str(np.float(np.sum(mask_sub)) / C.bkg_boxsize**2))

            # fill median and std arrays
            mesh_median[nsub] = median
            mesh_std[nsub] = std

        #if C.timing: t1 = time.time()
        #pool = ThreadPool(1)
        #pool.map(get_median_std, range(nsubs))
        #pool.close()
        #pool.join()
        #if C.timing: log.info('wall-time spent in get_back pool ' + str(time.time() - t1))

        # reshape and transpose
        mesh_median = mesh_median.reshape((nxsubs, nysubs)).transpose()
        mesh_std = mesh_std.reshape((nxsubs, nysubs)).transpose()

        # median filter the meshes with filter of size [C.bkg_filtersize]
        shape_filter = (C.bkg_filtersize, C.bkg_filtersize)
        mesh_median_filt = ndimage.filters.median_filter(mesh_median, shape_filter)
        mesh_std_filt = ndimage.filters.median_filter(mesh_std, shape_filter)

        # resize low-resolution meshes
        background = ndimage.zoom(mesh_median_filt, C.bkg_boxsize)
        background_std = ndimage.zoom(mesh_std_filt, C.bkg_boxsize)

        #ds9_arrays(data_objmask=data_objmask, mesh_median=mesh_median,
        #           mesh_median_filt=mesh_median_filt,
        #           background=background, background_std=background_std)
        
        # if shape of the background is not equal to input [data]
        # then pad the background images
        if data.shape != background.shape:
            t1 = time.time()
            ypad = ysize - background.shape[0]
            xpad = xsize - background.shape[1]
            background = np.pad(background, ((0,ypad),(0,xpad)), 'edge')
            background_std = np.pad(background_std, ((0,ypad),(0,xpad)), 'edge')                   
            log.info('time to pad ' + str(time.time()-t1))
            #ds9_arrays(data=data, data_objmask=data_objmask,
            #           background=background, background_std=background_std)
            #np.pad seems quite slow; alternative:
            #centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(C.bkg_boxsize,
            #                                                                   ysize, xsize, log,
            #                                                                   get_remainder=True)
            # these now include the remaining patches
                        
            
    if C.timing:
        log.info('wall-time spent in get_back ' + str(time.time() - t))
        log.info('peak memory (in GB) used in get_back {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return background.astype(np.float32), background_std.astype(np.float16)

################################################################################

def plot_scatter (x, y, limits, corder, cmap='rainbow_r', symbol='o',
                  xlabel='', ylabel='', legendlabel='', title='', filename='',
                  simple=False, xscale='log', yscale='linear'):

    plt.axis(limits)
    plt.scatter(x, y, c=corder, cmap=cmap, alpha=1, label=legendlabel, edgecolors='black')
    plt.xscale(xscale)
    plt.yscale(yscale)
    if legendlabel!='':
        plt.legend(numpoints=1, fontsize='medium')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename != '':
        plt.savefig(filename)
    if C.show_plots: plt.show()
    plt.close()

################################################################################

def get_psf(image, ima_header, nsubs, imtype, fwhm, pixscale, log):

    """Function that takes in [image] and determines the actual Point
    Spread Function as a function of position from the full frame, and
    returns a cube containing the psf for each subimage in the full
    frame.

    """

    global psf_size_new
    
    if C.timing: t = time.time()
    log.info('Executing get_psf ...')

    # determine image size from header
    xsize, ysize = ima_header['NAXIS1'], ima_header['NAXIS2']
    
    # run psfex on SExtractor output catalog
    #
    # If the PSFEx output file is already present with the same
    # [psf_size_config] as currently required, then skip [run_psfex].
    skip_psfex = False
    psfex_bintable = image.replace('_wcs.fits', '.psf')    
    if os.path.isfile(psfex_bintable) and not C.redo:
        with fits.open(psfex_bintable) as hdulist:
            header = hdulist[1].header
            data = hdulist[1].data[0][0][:]
        # use function [get_samp_PSF_config_size] to determine [psf_samp]
        # and [psf_size_config]
        psf_samp, psf_size_config = get_samp_PSF_config_size()
        log.info('psf_samp, psf_size_config: '+str(psf_samp)+' '+str(psf_size_config))
        # check that the above [psf_size_config] is the same as
        # the size of the images in the data array, or equivalently,
        # the value of the header parameters 'PSFAXIS1' or 'PSFAXIS2'
        if psf_size_config == header['PSFAXIS1']:
            skip_psfex = True
            if C.verbose:
                log.info('Skipping run_psfex for image: '+image)
                        
    if not skip_psfex:
        psfexcat = image.replace('_wcs.fits', '.psfexcat')
        sexcat_ldac = image.replace('_wcs.fits', '.sexcat_ldac')
        log.info('sexcat_ldac ' + sexcat_ldac)
        log.info('psfexcat ' + psfexcat)
    
        # feed PSFEx only with zero-flag sources
        sexcat_ldac_zeroflags = sexcat_ldac+'_zeroflags'
        with fits.open(sexcat_ldac) as hdulist:
            data = hdulist[2].data
            index = ((data['FLAGS']==0) | (data['FLAGS']==1))
            data['NUMBER'][index] = np.arange(np.sum(index))+1
            hdulist[2].data = data[:][index]
            #& (data['FLUX_AUTO']>0.) & (data['FLUXERR_AUTO']>0.) & (data['FLUX_AUTO']/data['FLUXERR_AUTO']>20.)
            hdulist_new = fits.HDUList(hdulist)
            hdulist_new.writeto(sexcat_ldac_zeroflags, overwrite=True)
            hdulist_new.close()
            
        #result = run_psfex(sexcat_ldac_zeroflags, C.psfex_cfg, psfexcat, log, fwhm, imtype)
        # something goes wrong when feeding the above zeroflags catalog to PSFEx
        # not sure what - too restrictive? For now, just supply the full ldac catalog.
        result = run_psfex(sexcat_ldac, C.psfex_cfg, psfexcat, imtype, log)


    # If [C.dosex_psffit] parameter is set, then again run SExtractor,
    # but now using output PSF from PSFEx, so that PSF-fitting can be
    # performed for all objects. The output columns defined in
    # [C.sex_par_psffit] include several new columns related to the PSF
    # fitting.
    sexcat_ldac_psffit = image.replace('.fits', '.sexcat_ldac_psffit')
    if (not os.path.isfile(sexcat_ldac_psffit) or C.redo) and C.dosex_psffit:
        sexcat_ldac_psffit = image.replace('.fits', '.sexcat_ldac_psffit')
        result = run_sextractor(image, sexcat_ldac_psffit, C.sex_cfg_psffit,
                                C.sex_par_psffit, pixscale, log, fit_psf=True, fwhm=fwhm)
        
    # If not already done so above, read in PSF output binary table
    # from psfex, containing the polynomial coefficient images
    if not ('header' in dir()):
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
    if C.verbose:
        log.info('polzero1                   ' + str(polzero1))
        log.info('polscal1                   ' + str(polscal1))
        log.info('polzero2                   ' + str(polzero2))
        log.info('polscal2                   ' + str(polscal2))
        log.info('order polynomial:          ' + str(poldeg))
        log.info('PSFex FWHM:                ' + str(psf_fwhm))
        log.info('PSF sampling size (pixels):' + str(psf_samp))
        log.info('PSF size defined in config:' + str(psf_size_config))
        
    # call centers_cutouts to determine centers
    # and cutout regions of the full image
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(C.subimage_size, ysize, xsize, log)
    ysize_fft = C.subimage_size + 2*C.subimage_border
    xsize_fft = C.subimage_size + 2*C.subimage_border
    nxsubs = xsize/C.subimage_size
    nysubs = ysize/C.subimage_size

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

    # [psf_size] is the PSF size in image pixels:
    #   [psf_size] = [psf_size_config] * [psf_samp]
    # where [psf_size_config] is the size of the square
    # image on which PSFEx constructs the PSF.
    # If global parameter [C.psf_sampling] is set, then
    #   [psf_samp] = [psf_samling]
    # where [psf_samp(ling)] is the PSF sampling step in image pixels.
    # If [C.psf_sampling] is set to zero, [psf_samp] is determined as follows:
    #   [psf_samp] = [C.psf_samp_fwhmfrac] * FWHM in pixels
    # where [C.psf_samp_fwhmfrac] is a global parameter which should be set
    # to about 0.25 so for an oversampled image with FWHM~8: [psf_samp]~2,
    # while an undersampled image with FWHM~2: [psf_samp]~1/4
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    # if this is even, make it odd
    if psf_size % 2 == 0:
        psf_size += 1
    if C.verbose:
        log.info('FWHM                      : ' + str(fwhm))
        log.info('final image PSF size      : ' + str(psf_size))
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
    #for nsub in range(nsubs):
    #
    # previously this was a loop; now turned to a function to
    # try pool.map multithreading below
    def loop_psf_sub(nsub):
    
        x = (centers[nsub,1] - polzero1) / polscal1
        y = (centers[nsub,0] - polzero2) / polscal2

        if nsubs==1 or C.use_single_psf:
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
        psf_ima_resized_norm = clean_norm_psf(psf_ima_resized, C.psf_clean_factor)

        psf_ima[nsub] = psf_ima_resized_norm
        if C.verbose and nsub==0:
            log.info('psf_samp, psf_samp_update: ' + str(psf_samp) + ', ' + str(psf_samp_update))
            log.info('np.shape(psf_ima_config): ' + str(np.shape(psf_ima_config)))
            log.info('np.shape(psf_ima): ' + str(np.shape(psf_ima)))
            log.info('np.shape(psf_ima_resized): ' + str(np.shape(psf_ima_resized)))
            log.info('psf_size: ' + str(psf_size))
            
        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            log.info('WARNING: image not even in one or both dimensions!')
            
        xcenter_fft, ycenter_fft = xsize_fft/2, ysize_fft/2
        if C.verbose and nsub==0:
            log.info('xcenter_fft, ycenter_fft: ' + str(xcenter_fft) + ', ' + str(ycenter_fft))

        psf_hsize = psf_size/2
        index = [slice(ycenter_fft-psf_hsize, ycenter_fft+psf_hsize+1), 
                 slice(xcenter_fft-psf_hsize, xcenter_fft+psf_hsize+1)]
        psf_ima_center[nsub][index] = psf_ima_resized_norm

        # perform fft shift
        psf_ima_shift[nsub] = fft.fftshift(psf_ima_center[nsub])


        if C.display and (nsub==0 or nsub==nysubs-1 or nsub==nsubs/2 or
                        nsub==nsubs-nysubs or nsub==nsubs-1):
            if imtype=='new':
                base = base_new
            else:
                base = base_ref
            fits.writeto(base+'_psf_ima_config_sub'+str(nsub)+'.fits', psf_ima_config, overwrite=True)
            fits.writeto(base+'_psf_ima_resized_norm_sub'+str(nsub)+'.fits',
                         psf_ima_resized_norm.astype(np.float32), overwrite=True)
            fits.writeto(base+'_psf_ima_center_sub'+str(nsub)+'.fits',
                         psf_ima_center[nsub].astype(np.float32), overwrite=True)            
            fits.writeto(base+'_psf_ima_shift_sub'+str(nsub)+'.fits',
                         psf_ima_shift[nsub].astype(np.float32), overwrite=True)            


    # call above function [get_psf_sub] with pool.map
    if C.timing: t1 = time.time()
    pool = ThreadPool(1)
    pool.map(loop_psf_sub, range(nsubs))
    pool.close()
    pool.join()
    if C.timing:
        log.info('wall-time spent in loop_psf_sub pool ' + str(time.time() - t1))
        log.info('wall-time spent in get_psf ' + str(time.time() - t))
        log.info('peak memory (in GB) used in get_psf {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return psf_ima_shift, psf_ima


################################################################################

def get_fratio_radec(psfcat_new, psfcat_ref, sexcat_new, sexcat_ref, log):

    """Function that takes in output catalogs of stars used in the PSFex
    runs on the new and the ref image, and returns the arrays with
    pixel coordinates (!) x, y (in the new frame) and fratios for the
    matching stars. In addition, it provides the difference in stars'
    RAs and DECs in arcseconds between the two catalogs.

    """
    
    t = time.time()
    log.info('Executing get_fratio_radec ...')
    
    def readcat (psfcat):
        table = ascii.read(psfcat, format='sextractor')
        # mask of entries with FLAGS_PSF=0
        mask_zero = (table['FLAGS_PSF']==0)
        number = table['SOURCE_NUMBER'][mask_zero]
        x = table['X_IMAGE'][mask_zero]
        y = table['Y_IMAGE'][mask_zero]
        norm = table['NORM_PSF'][mask_zero]
        return number, x, y, norm
        
    # read psfcat_new
    number_new, x_new, y_new, norm_new = readcat(psfcat_new)
    # read psfcat_ref
    number_ref, x_ref, y_ref, norm_ref = readcat(psfcat_ref)

    if C.verbose:
        log.info('new: number of PSF stars with zero FLAGS: '+str(len(x_new)))
        log.info('ref: number of PSF stars with zero FLAGS: '+str(len(x_ref)))
    
    def xy2radec (number, sexcat):
        # read SExtractor fits table
        with fits.open(sexcat) as hdulist:
            data = hdulist[1].data
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
                        
    if C.verbose:
        log.info('fraction of PSF stars that match: ' + str(float(nmatch)/len(x_new)))
            
    if C.timing:
        log.info('wall-time spent in get_fratio_radec ' + str(time.time()-t))
        log.info('peak memory (in GB) used in get_fratio_radec {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return np.array(x_new_match), np.array(y_new_match), np.array(fratio), \
        np.array(dra_match), np.array(ddec_match)

################################################################################

def centers_cutouts(subsize, ysize, xsize, log, get_remainder=False):
    
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
    log.info('nxsubs, nysubs, nsubs: ' + str(nxsubs) + ', ' + str(nysubs) + ', ' + str(nsubs))

    centers = np.ndarray((nsubs, 2), dtype=int)
    cuts_ima = np.ndarray((nsubs, 4), dtype=int)
    cuts_ima_fft = np.ndarray((nsubs, 4), dtype=int)
    cuts_fft = np.ndarray((nsubs, 4), dtype=int)
    sizes = np.ndarray((nsubs, 2), dtype=int)

    ysize_fft = subsize + 2*C.subimage_border
    xsize_fft = subsize + 2*C.subimage_border
        
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
            y1 = np.amax([0,y-ny/2-C.subimage_border])
            x1 = np.amax([0,x-nx/2-C.subimage_border])
            y2 = np.amin([ysize,y+ny/2+C.subimage_border])
            x2 = np.amin([xsize,x+nx/2+C.subimage_border])
            cuts_ima_fft[nsub] = [y1,y2,x1,x2]
            cuts_fft[nsub] = [y1-(y-ny/2-C.subimage_border),ysize_fft-(y+ny/2+C.subimage_border-y2),
                              x1-(x-nx/2-C.subimage_border),xsize_fft-(x+nx/2+C.subimage_border-x2)]
            sizes[nsub] = [ny, nx]
            
    return centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes

################################################################################

def show_image(image):

    im = plt.imshow(np.real(image), origin='lower', cmap='gist_heat',
                    interpolation='nearest')
    plt.show(im)

################################################################################

def ds9_arrays(regions=None, **kwargs):

    cmd = ['ds9', '-zscale', '-zoom', '4', '-cmap', 'heat']
    if regions is not None:
        cmd += ['-regions', regions]
    for name, array in kwargs.items():
        # write array to fits
        fitsfile = 'ds9_'+name+'.fits'
        fits.writeto(fitsfile, np.array(array).astype(np.float32), overwrite=True)            
        # append to command
        cmd.append(fitsfile)

    #print 'cmd', cmd
    result = subprocess.call(cmd)

    
################################################################################

def run_wcs(image_in, image_out, ra, dec, pixscale, width, height, log):

    if C.timing: t = time.time()
    log.info('Executing run_wcs ...')
    
    scale_low = 0.99 * pixscale
    scale_high = 1.01 * pixscale

    basename = image_in.replace('.fits','')
    sexcat = image_in.replace('.fits','.sexcat')

    # feed Astrometry.net only with zero-flag sources
    with fits.open(sexcat) as hdulist:
        data = hdulist[1].data
    index = ((data['FLAGS']==0))
    #& (data['FLUX_AUTO']>0.) & (data['FLUXERR_AUTO']>0.) & (data['FLUX_AUTO']/data['FLUXERR_AUTO']>20.)
    sexcat_zeroflags = sexcat+'_zeroflags'
    fits.writeto(sexcat_zeroflags, data[:][index], overwrite=True)
    
    #scampcat = image_in.replace('.fits','.scamp')
    cmd = ['solve-field', '--no-plots', #'--no-fits2fits', cloud version of astrometry does not have this arg
           '--x-column', 'XWIN_IMAGE', '--y-column', 'YWIN_IMAGE',
           '--sort-column', 'FLUX_AUTO',
           '--no-remove-lines', '--uniformize', '0',
           # only work on 1000 brightest sources
           '--objs', '1000',
           '--width', str(width), '--height', str(height),           
           #'--keep-xylist', sexcat,
           # ignore existing WCS headers in FITS input images
           #'--no-verify', 
           #'--code-tolerance', str(0.01), 
           #'--quad-size-min', str(0.1),
           # for KMTNet images restrict the max quad size:
           '--quad-size-max', str(0.1),
           # number of field objects to look at:
           #'--depth', str(10),
           #'--scamp', scampcat,
           sexcat_zeroflags,
           '--tweak-order', str(C.astronet_tweak_order), '--scale-low', str(scale_low),
           '--scale-high', str(scale_high), '--scale-units', 'app',
           '--ra', str(ra), '--dec', str(dec), '--radius', str(2.),
           '--new-fits', image_out, '--overwrite',
           '--out', basename
    ]

    # if C.verbose:
    #     log.info('Astrometry.net command: '+ cmd)
    
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)

    if os.path.exists("%s.solved"%basename) and status==0:
        os.remove("%s.solved"%basename)
        os.remove("%s.match"%basename)
        os.remove("%s.rdls"%basename)
        os.remove("%s.corr"%basename)
        os.remove("%s-indx.xyls"%basename)
    else:
        log.error("Solving WCS failed.")
        return 'error'

    if C.timing: t2 = time.time()

    # read image_in
    with fits.open(image_in) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data

    # read header saved in .wcs 
    wcsfile = basename+'.wcs'
    with fits.open(wcsfile) as hdulist:
        header_wcs = hdulist[0].header

    # convert SIP header keywords from Astrometry.net to PV keywords
    # that swarp, scamp (and sextractor) understand using this module
    # from David Shupe: sip_to_pv

    # new version (June 2017) of sip_to_pv works on image header
    # rather than header+image (see below); the header is modified in
    # place; compared to the old version this saves an image write
    result = sip_to_pv(header_wcs, tpv_format=True, preserve=False)
        
    # add original and wcs headers and write image_out
    fits.writeto(image_out, data, header+header_wcs, overwrite=True)

    # using the old version of sip_to_pv (before June 2017):
    #status = sip_to_pv(image_out, image_out, log, tpv_format=True)
    #if status == False:
    #    log.error('sip_to_pv failed.')
    #    return 'error'

             
    # read data from SExtractor catalog
    with fits.open(sexcat) as hdulist:
        prihdu = hdulist[0]
        data_sexcat = hdulist[1].data

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

    # replace old ra and dec with new ones
    data_sexcat['ALPHAWIN_J2000'] = newra
    data_sexcat['DELTAWIN_J2000'] = newdec

    # write output catalog
    # contains RA and DEC, so putting _wcs in the name
    fits.writeto(basename+'_wcs.sexcat', data_sexcat, overwrite=True)
        
    if C.timing:
        #log.info('extra time for creating LDAC fits table ' + str(time.time()-t2))
        log.info('wall-time spent in run_wcs ' + str(time.time()-t))
        log.info('peak memory (in GB) used in run_wcs {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

        
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
    # prihdu.header['SEEING'] = header4ext2['SEEING'] #need to calculte and add
    prihdu.header['BKGSIG'] = header4ext2['SEXBKDEV']

    
    # write hdulist to output LDAC fits table
    hdulist = fits.HDUList([prihdu, ext2, ext3])
    hdulist.writeto(fits_ldac_out, overwrite=True)
    hdulist.close()

    
################################################################################

def ldac2fits (cat_ldac, cat_fits, log):

    """This function converts the LDAC binary FITS table from SExtractor
    to a common binary FITS table (that can be read by Astrometry.net) """

    if C.timing: t = time.time()
    log.info('Executing ldac2fits ...')

    # read input table and write out primary header and 2nd extension
    with fits.open(cat_ldac) as hdulist:
        hdulist_new = fits.HDUList([hdulist[0], hdulist[2]])
        hdulist_new.writeto(cat_fits, overwrite=True)
        hdulist_new.close()
        
    if C.timing:
        log.info('wall-time spent in ldac2fits ' + str(time.time()-t))
        log.info('peak memory (in GB) used in ldac2fits {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    
################################################################################
    
def run_remap(image_new, image_ref, image_out, image_out_size,
              gain, log, config=None, resample='Y', resampling_type='LANCZOS3',
              projection_err=0.001, mask=None, header_only='N'):
        
    """Function that remaps [image_ref] onto the coordinate grid of
       [image_new] and saves the resulting image in [image_out] with
       size [image_size].
    """

    if C.timing: t = time.time()
    log.info('Executing run_remap ...')

    with fits.open(image_new) as hdulist:
        header_new = hdulist[0].header
    with fits.open(image_ref) as hdulist:
        header_ref = hdulist[0].header
        
    # create .head file with header info from [image_new]
    header_out = header_new[:]
    # copy some keywords from header_ref
    #for key in [C.key_exptime, C.key_satlevel, C.key_gain, C.key_ron, C.key_seeing]:
    for key in [C.key_exptime, C.key_satlevel, C.key_gain, C.key_ron]:
        header_out[key] = header_ref[key]
    # delete some others
    for key in ['WCSAXES', 'NAXIS1', 'NAXIS2']:
        if key in header_out: del header_out[key]
    # write to .head file
    with open(image_out.replace('.fits','.head'),'w') as newrefhdr:
        for card in header_out.cards:
            newrefhdr.write(str(card)+'\n')

    size_str = str(image_out_size[1]) + ',' + str(image_out_size[0]) 
    cmd = ['swarp', image_ref, '-c', config, '-IMAGEOUT_NAME', image_out, 
           '-IMAGE_SIZE', size_str, '-GAIN_DEFAULT', str(gain),
           '-RESAMPLE', resample,
           '-RESAMPLING_TYPE', resampling_type,
           '-PROJECTION_ERR', str(projection_err)]
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)
    if status != 0:
        log.error('Swarp failed with exit code '+str(status)+'.')
        return 'error'
    
    if C.timing:
        log.info('wall-time spent in run_remap ' + str(time.time()-t))
        log.info('peak memory (in GB) used in run_remap {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    
################################################################################

def get_fwhm (cat_ldac, fraction, log, class_sort=False, get_elongation=False):

    """Function that accepts a FITS_LDAC table produced by SExtractor and
    returns the FWHM and its standard deviation in pixels.  The
    columns that need to be present in the fits table are 'FLAGS',
    'FLUX_AUTO' and 'CLASS_STAR'. By default, the function takes the
    brightest [fraction] of objects, and determines the median FWHM
    from them using sigma clipping. If [class_sort] is True, it
    instead takes the fraction of objects with the highest CLASS_STAR
    value, which ideally is 1 for stars and 0 for galaxies. However,
    if the SEEING_FWHM that was used for the SExtractor run was off
    from the real value, the CLASS_STAR is not reliable.

    """
 
    if C.timing: t = time.time()
    log.info('Executing get_fwhm ...')

    with fits.open(cat_ldac) as hdulist:
        data = hdulist[2].data

    # these arrays correspond to objecst with flag==0 and flux_auto>0.
    # add a S/N requirement
    index = (data['FLAGS']==0) & (data['FLUX_AUTO']>0.) & \
            (data['FLUXERR_AUTO']>0.) & (data['FLUX_AUTO']/data['FLUXERR_AUTO']>20.)
    fwhm = data['FWHM_IMAGE'][index]
    class_star = data['CLASS_STAR'][index]
    flux_auto = data['FLUX_AUTO'][index]
    mag_auto = -2.5*np.log10(flux_auto)
    if get_elongation:
        elongation = data['ELONGATION'][index]
    
    if class_sort:
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
        log.info('WARNING: fewer than 10 objects are selected for FWHM determination')
    
    # determine mean, median and standard deviation through sigma clipping
    fwhm_mean, fwhm_std, fwhm_median = clipped_stats(fwhm_select, log=log)
    if C.verbose:
        log.info('catalog: ' + cat_ldac)
        log.info('fwhm_mean: {:.3f}, fwhm_median: {:.3f}, fwhm_std: {:.3f}'.
                 format(fwhm_mean, fwhm_median, fwhm_std))
    if get_elongation:
        # determine mean, median and standard deviation through sigma clipping
        elongation_mean, elongation_std, elongation_median = clipped_stats(elongation_select,
                                                                           log=log)
        if C.verbose:
            log.info('elongation_mean: {:.3f}, elongation_median: {:.3f}, elongation_std: {:.3f}'.
                     format(elongation_mean, elongation_median, elongation_std))
            
        
    if C.make_plots:

        # best parameter to plot vs. FWHM is MAG_AUTO
        mag_auto_select = mag_auto[index_sort][index_select]

        # to get initial values before discarding flagged objects
        index = (data['FLUX_AUTO']>0.)
        fwhm = data['FWHM_IMAGE'][index]
        flux_auto = data['FLUX_AUTO'][index]
        mag_auto = -2.5*np.log10(flux_auto)

        plt.plot(fwhm, mag_auto, 'bo', markersize=5, markeredgecolor='k')
        x1,x2,y1,y2 = plt.axis()
        plt.plot(fwhm_select, mag_auto_select, 'go', markersize=5, markeredgecolor='k')
        plt.plot([fwhm_median, fwhm_median], [y2,y1], color='red')
        fwhm_line = fwhm_median-fwhm_std
        plt.plot([fwhm_line, fwhm_line], [y2,y1], 'r--')
        fwhm_line = fwhm_median+fwhm_std
        plt.plot([fwhm_line, fwhm_line], [y2,y1], 'r--')
        plt.axis((0,20,y2,y1))
        plt.xlabel('FWHM (pixels)')
        plt.ylabel('MAG_AUTO')
        plt.title('median FWHM: {:.2f} $\pm$ {:.2f} pixels'.format(fwhm_median, fwhm_std))
        plt.savefig(cat_ldac+'_fwhm.pdf')
        plt.title(cat_ldac)
        if C.show_plots: plt.show()
        plt.close()

        if get_elongation:
            elongation = data['ELONGATION'][index]

            plt.plot(elongation, mag_auto, 'bo', markersize=5, markeredgecolor='k')
            x1,x2,y1,y2 = plt.axis()
            plt.plot(elongation_select, mag_auto_select, 'go', markersize=5, markeredgecolor='k')
            plt.plot([elongation_median, elongation_median], [y2,y1], color='red')
            elongation_line = elongation_median-elongation_std
            plt.plot([elongation_line, elongation_line], [y2,y1], 'r--')
            elongation_line = elongation_median+elongation_std
            plt.plot([elongation_line, elongation_line], [y2,y1], 'r--')
            plt.axis((0,20,y2,y1))
            plt.xlabel('ELONGATION (A/B)')
            plt.ylabel('MAG_AUTO')
            plt.savefig(cat_ldac+'_elongation.pdf')
            if C.show_plots: plt.show()
            plt.close()
            
    if C.timing:
        log.info('wall-time spent in get_fwhm {:.3f}'.format(time.time()-t))
        log.info('peak memory (in GB) used in get_fwhm {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    if get_elongation:
        return fwhm_median, fwhm_std, elongation_median, elongation_std
    else:
        return fwhm_median, fwhm_std
    

################################################################################

def update_vignet_size (sex_par_in, sex_par_out, log):

    # in case [C.psf_sampling] is set to zero, scale the size of the
    # VIGNET output in the output catalog with 2*[C.psf_radius]*[fwhm]
    # where fwhm is taken to be the largest of global parameters
    # [fwhm_new] and [fwhm_ref]
    if C.psf_sampling == 0.:
        fwhm_vignet = np.amax([fwhm_new, fwhm_ref])
        size_vignet = np.int(np.ceil(2.*C.psf_radius*fwhm_vignet))
        # make sure it's odd
        if size_vignet % 2 == 0: size_vignet += 1
        # provide a warning if it's very large
        if size_vignet > 99:
            log.info('Warning: VIGNET size is larger than 99 pixels: ' + str(size_vignet))
    else:
        # otherwise set it to a reasonably large value
        size_vignet = 99

    # append the VIGNET size to the SExtractor parameter file
    # [sex_par_in] and write it to a temporary file [sex_par_out] to
    # be used later by SExtractor
    size_vignet_str = str((size_vignet, size_vignet))
    with open(sex_par_in, 'rt') as file_in:
        with open(sex_par_out, 'wt') as file_out:
            for line in file_in:
                file_out.write(line)
            file_out.write('VIGNET'+size_vignet_str+'\n')
        if C.verbose:
            log.info('VIGNET size: ' + str(size_vignet_str))

################################################################################

def run_sextractor(image, cat_out, file_config, file_params, pixscale, log,
                   fit_psf=False, return_fwhm=True, fraction=1.0, fwhm=5.0, save_bkg=True,
                   update_vignet=True, mask=None):

    """Function that runs SExtractor on [image], and saves the output
       catalog in [outcat], using the configuration file [file_config]
       and the parameters defining the output recorded in the
       catalogue [file_params]. If [fit_psf] is True, SExtractor will
       perform PSF fitting photometry using the PSF built by PSFex. If
       [return_fwhm] is True, as estimate of the image FWHM and its
       standard deviation is returned using SExtractor's seeing
       estimate of the detected sources; if False, it will return
       zeros. If [fraction] is less than the default 1.0, SExtractor
       will be run on a fraction [fraction] of the area of the full
       image. Sextractor will use the input value [fwhm], which is
       important for the star-galaxy classification. If [save-bkg] is
       True, the background image, its standard deviation and the
       -OBJECTS image (background-subtracted image with all objects
       masked with zero values), all produced by SExtractor, will be
       saved. If [C.bkg_method] is not set to 1 (use SExtractor's
       background), then improve the estimates of the background and
       its standard deviation."""

    if C.timing: t = time.time()
    log.info('Executing run_sextractor ...')

    # if fraction less than one, run SExtractor on specified fraction of
    # the image
    if fraction < 1.:

        # read in input image and header
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            data = hdulist[0].data
        # get input image size from header
        ysize, xsize = read_header(header, ['NAXIS2', 'NAXIS1'], log)
        
        # determine cutout from [fraction]
        center_x = np.int(xsize/2+0.5)
        center_y = np.int(ysize/2+0.5)
        halfsize_x = np.int((xsize * np.sqrt(fraction))/2.+0.5)
        halfsize_y = np.int((ysize * np.sqrt(fraction))/2.+0.5)
        data_fraction = data[center_y-halfsize_y:center_y+halfsize_y,
                             center_x-halfsize_x:center_x+halfsize_x]

        # write small image to fits
        image_fraction = image.replace('.fits','_fraction.fits')
        fits.writeto(image_fraction, data_fraction.astype(np.float32), header, overwrite=True)

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
    apphot_diams = np.array(C.apphot_radii) * 2 * fwhm
    apphot_diams_str = ",".join(apphot_diams.astype(str))

    # update size of VIGNET
    if update_vignet:
        update_vignet_size (file_params, file_params+'_temp', log)
        file_params = file_params+'_temp'

    if mask is not None:
        # and add line in parameter file to include IMAFLAG_ISO
        if 'temp' in file_params:
            with open(file_params, 'a') as myfile:
                myfile.write('IMAFLAGS_ISO\n')
        else:
            with open(file_params, 'rt') as file_in:
                with open(file_params+'_temp', 'wt') as file_out:
                    for line in file_in:
                        file_out.write(line)
                    file_out.write('IMAFLAGS_ISO\n')
            file_params = file_params+'_temp'

                    
    # run sextractor from the unix command line
    cmd = ['sex', image, '-c', file_config, '-CATALOG_NAME', cat_out, 
           '-PARAMETERS_NAME', file_params, '-PIXEL_SCALE', str(pixscale),
           '-SEEING_FWHM', str(seeing),'-PHOT_APERTURES',apphot_diams_str,
           '-BACK_SIZE', str(C.bkg_boxsize), '-BACK_FILTERSIZE', str(C.bkg_filtersize)]

    # add commands to produce BACKGROUND, BACKGROUND_RMS and
    # background-subtracted image with all pixels where objects were
    # detected set to zero (-OBJECTS). These are used to build an
    # improved background map. 
    if save_bkg:
        bkg = image.replace('.fits','_bkg.fits')
        bkg_std = image.replace('.fits','_bkg_std.fits')
        objmask = image.replace('.fits','_objmask.fits')
        cmd += ['-CHECKIMAGE_TYPE', 'BACKGROUND,BACKGROUND_RMS,-OBJECTS',
                '-CHECKIMAGE_NAME', bkg+','+bkg_std+','+objmask]
    
    # in case of fraction being less than 1: only care about higher S/N detections
    if fraction < 1.: cmd += ['-DETECT_THRESH', str(C.fwhm_detect_thresh)]
    
    # provide PSF file from PSFex
    if fit_psf: cmd += ['-PSF_NAME', image.replace('_wcs.fits', '.psf')]

    # provide mask image if not None
    if mask is not None:
        log.info('mask: {}'.format(mask))
        cmd += ['-FLAG_IMAGE', mask, '-FLAG_TYPE', 'OR']
        
        
    # run command
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)

    log.info('peak memory (in GB) used in run_sextractor before get_back {}'.
             format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    # improve background estimate if [C.bkg_method] not set to 1 (= use
    # background determined by SExtractor)
    if save_bkg and C.bkg_method != 1:

        # read in SExtractor's object mask created above
        with fits.open(objmask) as hdulist:
            data_objmask = hdulist[0].data.astype(np.float16)

        # read in input image
        with fits.open(image) as hdulist:
            data = hdulist[0].data.astype(np.float32)

        # construct background image using [get_back]; in the case of
        # the reference image these data need to refer to the image
        # before remapping
        if C.bkg_method==2:
            data_bkg, data_bkg_std = get_back(data, data_objmask, log)

        # similar as above, but now photutils' Background2D is used
        # inside [get_back]
        elif C.bkg_method==3:
            data_bkg, data_bkg_std = get_back(data, data_objmask, log,
                                              use_photutils=True)

        # write the improved background and standard deviation to fits
        # overwriting the fits images produced by SExtractor
        fits.writeto(bkg, data_bkg.astype(np.float32), overwrite=True)
        fits.writeto(bkg_std, data_bkg_std.astype(np.float32), overwrite=True)
    
    if return_fwhm:
        # get estimate of seeing from output catalog
        fwhm, fwhm_std = get_fwhm(cat_out, C.fwhm_frac, log, class_sort=C.fwhm_class_sort)
    else:
        fwhm = 0.
        fwhm_std = 0.
                
    if C.timing:
        log.info('wall-time spent in run_sextractor ' + str(time.time()-t))
        log.info('peak memory (in GB) used in run_sextractor {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

    return fwhm, fwhm_std

################################################################################

def run_psfex(cat_in, file_config, cat_out, imtype, log):
    
    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""

    if C.timing: t = time.time()
        
    # use function [get_samp_PSF_config_size] to determine [psf_samp]
    # and [psf_size_config] required to run PSFEx
    psf_samp, psf_size_config = get_samp_PSF_config_size()
    psf_size_config_str = str(psf_size_config)+','+str(psf_size_config)

    if C.verbose:
        log.info('psf_size_config: ' + str(psf_size_config))

    # get FWHM and ELONGATION to limit the PSFex configuration
    # parameters SAMPLE_FWHMRANGE and SAMPLE_MAXELLIP
    #fwhm, fwhm_std, elongation, elongation_std = get_fwhm(cat_in, 0.05, class_sort=False,
    #                                                      get_elongation=True)
    #print 'fwhm, fwhm_std, elongation, elongation_std', fwhm, fwhm_std, elongation, elongation_std
    #sample_fwhmrange = str(fwhm-fwhm_std)+','+str(fwhm+fwhm_std)
    #print 'sample_fwhmrange', sample_fwhmrange
    #maxellip = (elongation+3.*elongation_std-1)/(elongation+3.*elongation_std+1)
    #maxellip_str= str(np.amin([maxellip, 1.]))
    #print 'maxellip_str', maxellip_str

    # Need to check whether the VIGNET size from the SExtractor run is
    # sufficient large compared to [psf_samp] and [psf_size_config].
    
    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config_str, '-PSF_SAMPLING', str(psf_samp)]
    #       '-SAMPLE_FWHMRANGE', sample_fwhmrange,
    #       '-SAMPLE_MAXELLIP', maxellip_str]

    if C.make_plots:

        if imtype=='new':
            base = base_new
        else:
            base = base_ref
            
        cmd += ['-CHECKPLOT_DEV', 'PDF',
                '-CHECKPLOT_TYPE', 'FWHM, ELLIPTICITY, COUNTS, COUNT_FRACTION, CHI2, RESIDUALS',
                '-CHECKPLOT_NAME', base+'_psfex_fwhm,'+base+'_psfex_ellip,'+base+'_psfex_counts,'+
                base+'_psfex_countfrac,'+base+'_psfex_chi2,'+base+'_psfex_resi']
        cmd += ['-CHECKIMAGE_TYPE', 'CHI,PROTOTYPES,SAMPLES,RESIDUALS,SNAPSHOTS,BASIS',
                '-CHECKIMAGE_NAME', base+'_chi,'+base+'_psfex_proto,'+base+'_psfex_samp,'+
                base+'_psfex_resi,'+base+'_psfex_snap,'+base+'_psfex_basis']

    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)

    if C.timing:
        log.info('wall-time spent in run_psfex ' + str(time.time()-t))
        log.info('peak memory (in GB) used in run_psfex {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

        
################################################################################

def get_samp_PSF_config_size():

    # [psf_size] is the PSF size in image pixels:
    #   [psf_size] = [psf_size_config] * [psf_samp]
    # where [psf_size_config] is the size of the square
    # image on which PSFEx constructs the PSF.
    # If global parameter [C.psf_sampling] is set, then
    #   [psf_samp] = [psf_samling]
    # where [psf_samp(ling)] is the PSF sampling step in image pixels.
    # If [C.psf_sampling] is set to zero, [psf_samp] is determined as follows:
    #   [psf_samp] = [C.psf_samp_fwhmfrac] * FWHM in pixels
    # where [C.psf_samp_fwhmfrac] is a global parameter which should be set
    # to about 0.25 so for an oversampled image with FWHM~8: [psf_samp]~2,
    # while an undersampled image with FWHM~2: [psf_samp]~1/4
    fwhm_samp = np.amax([fwhm_new, fwhm_ref])
    if C.psf_sampling == 0:
        psf_samp = C.psf_samp_fwhmfrac * fwhm_samp
    else:
        psf_samp = C.psf_sampling

    # throughout this function, the maximum of [fwhm_new] and
    # [fwhm_ref] is used for the FWHM, so that for both the new and
    # ref image the [psf_size_config] is the same, which results in a
    # better subtraction; see also the function [update_vignet_size]
    # where this is also done
    
    # determine [psf_size_config] based on [C.psf_radius], which is
    #   [psf_size_config] = [psf_size] / [psf_samp]
    #   [psf_size_config] = 2 * [C.psf_radius] * FWHM / [psf_samp]
    # and since:
    #   [psf_samp] = [C.psf_samp_fwhmfrac] * FWHM in pixels
    # this can be written:
    #   [psf_size_config] = 2 * [C.psf_radius] / [C.psf_samp_fwhmfrac]
    # this is independent of the image FWHM since the FWHM is sampled
    # by a fixed number of steps defined by [C.psf_samp_fwhmfrac]
    psf_size_config = 2. * C.psf_radius * fwhm_samp / psf_samp

    # convert to integer
    psf_size_config = np.int(psf_size_config+0.5)
    # make sure it's odd
    if psf_size_config % 2 == 0: psf_size_config += 1

    return psf_samp, psf_size_config


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

    # CHECK! set any negative values to zero
    #psf_array[psf_array<0] = 0
    
    if clean_factor != 0:
        mask_clean = (psf_array < (np.amax(psf_array) * clean_factor))
        psf_array[mask_clean] = 0.

    # normalize
    psf_array_norm = psf_array / np.sum(psf_array)

    return psf_array_norm
    
################################################################################

def run_ZOGY(R,N,Pr,Pn,sr,sn,fr,fn,Vr,Vn,dx,dy,log):

# edited Barak's original code to include variances sigma_n**2 and
# sigma_r**2 (see Eq. 9, here sn and sr) and Fn and Fr which are
# assumed to be unity in Barak's code.
    
    if C.timing: t = time.time()

    R_hat = fft.fft2(R)
    N_hat = fft.fft2(N)
    Pn_hat = fft.fft2(Pn)
    #if C.psf_clean_factor!=0:
    #clean Pn_hat
    #Pn_hat = clean_psf(Pn_hat, C.psf_clean_factor)
    Pn_hat2_abs = np.abs(Pn_hat**2)

    Pr_hat = fft.fft2(Pr)
    #if C.psf_clean_factor!=0:
    # clean Pr_hat
    #Pr_hat = clean_psf(Pr_hat, C.psf_clean_factor)
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
        log.info('Warning: denominator contains zero(s)')
        
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

    if C.verbose:
        log.info('fD: ' +str(fD))
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
    
    if C.display:
        fits.writeto('Pn_hat.fits', np.real(Pn_hat).astype(np.float32), overwrite=True)
        fits.writeto('Pr_hat.fits', np.real(Pr_hat).astype(np.float32), overwrite=True)
        fits.writeto('kr.fits', np.real(kr).astype(np.float32), overwrite=True)
        fits.writeto('kn.fits', np.real(kn).astype(np.float32), overwrite=True)
        fits.writeto('Sr.fits', Sr.astype(np.float32), overwrite=True)
        fits.writeto('Sn.fits', Sn.astype(np.float32), overwrite=True)
        fits.writeto('VSr.fits', VSr.astype(np.float32), overwrite=True)
        fits.writeto('VSn.fits', VSn.astype(np.float32), overwrite=True)
        fits.writeto('VSr_ast.fits', VSr_ast.astype(np.float32), overwrite=True)
        fits.writeto('VSn_ast.fits', VSn_ast.astype(np.float32), overwrite=True)

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
    if C.verbose:
        log.info('F_S; ' + str(F_S))
    # an alternative (slower) way to calculate the same F_S:
    #F_S_array = fft.ifft2((fn2*Pn_hat2_abs*fr2*Pr_hat2_abs) / denominator)
    #F_S = F_S_array[0,0]

    alpha = S / F_S
    alpha_std = np.zeros(alpha.shape)
    alpha_std[V_S>=0] = np.sqrt(V_S[V_S>=0]) / F_S

    if C.timing:
        log.info('wall-time spent in run_ZOGY ' + str(time.time()-t))
        log.info('peak memory (in GB) used in run_ZOGY {}'.
                 format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
    
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
    parser.add_argument('--new_fits', default=None, help='filename of new image')
    parser.add_argument('--ref_fits', default=None, help='filename of ref image')
    parser.add_argument('--telescope', default=None, help='telescope')
    parser.add_argument('--log', default=None, help='help')
    parser.add_argument('--verbose', default=None, help='verbose')
    
    #global_pars(args.telescope)
    # replaced [global_pars] function with importing
    # Utils/Constants_[telescope} file as C; all former global
    # parameters are now referred to as C.[parameter name]
    global C, args
    args = parser.parse_args()

    if args.telescope is not None:
        C = importlib.import_module('Utils.Constants_'+args.telescope)
    else:
        C = importlib.import_module('Utils.Constants')

    # if verbosity is provided through args.verbose, it will overwrite
    # the corresponding setting in Constants (C.verbose)
    if args.verbose is not None:
        C.verbose = args.verbose

    optimal_subtraction(args.new_fits, args.ref_fits, args.log)

if __name__ == "__main__":
    main()

