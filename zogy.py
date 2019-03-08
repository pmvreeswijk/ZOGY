
import argparse
import astropy.io.fits as fits
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np
#import numpy.fft as fft
import matplotlib
#matplotlib.use('PDF')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import subprocess
from scipy import ndimage
import time
import importlib
# these are important to speed up the FFTs
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
#pyfftw.interfaces.cache.set_keepalive_time(1.)

# for PSF fitting - see https://lmfit.github.io/lmfit-py/index.html
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit

# see https://github.com/stargaser/sip_tpv (version June 2017):
# download from GitHub and "python setup.py install --user" for local
# install or "sudo python setup.py install" for system install
from sip_tpv import sip_to_pv

import resource
from skimage import restoration, measure
#import inpaint

import logging
import traceback

from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.convolution import convolve, convolve_fft

from numpy.lib.recfunctions import append_fields, drop_fields, rename_fields, stack_arrays

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

#from memory_profiler import profile
#import objgraph

__version__ = '0.8.1'

################################################################################

#@profile
def optimal_subtraction(new_fits=None,      ref_fits=None,
                        new_fits_mask=None, ref_fits_mask=None,
                        set_file='Settings.set_zogy', log=None,
                        verbose=None, nthread=1, telescope=None):
    
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
    
    # make nthreads a global parameter instead of passing it on
    # through different functions, as it is used in many places
    global nthreads
    nthreads = nthread
    # this needs to be done here to allow [optimal_subtraction]
    # to be run not from the command line

    global tel
    tel = telescope
    
    # import settings file as C such that all parameters defined there
    # can be referred to as C.[parameter]
    global C
    C = importlib.import_module(set_file)

    # if verbosity is provided through input parameter [verbose], it
    # will overwrite the corresponding setting in Constants
    # (C.verbose)
    if verbose is not None:
        C.verbose = verbose
    
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
        if get_par(C.verbose,tel):
            streamhandler = logging.StreamHandler() #create print to screen logging
            streamhandler.setFormatter(formatter) #add format to screen logging
            log.addHandler(streamhandler) #link logger to screen logging
    

    log.info('tel {}'.format(tel))

    # define booleans [new] and [ref] indicating
    # that the corresponding image is provided and exists
    def set_bool (image_fits):
        ima_bool = False
        if image_fits is not None:
            if os.path.isfile(image_fits):
                ima_bool = True
            else:
                log.info('file {} does not exist'.format(image_fits))
        return ima_bool
    new = set_bool (new_fits)
    ref = set_bool (ref_fits)
    if not new and not ref:
        log.critical('no valid input image(s) provided')
        if log:
            return 'critical', 'No valid input image - please check filenames/paths.'
        else:
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
    if new and ref:
        global base_newref
        base_newref = base_new

    # check if configuration files exist; if not exit
    def check_files (filelist, log):
        for filename in filelist:
            if not os.path.isfile(filename):
                log.critical('{} does not exist'.format(filename))
                if log:
                    return 'critical', filename+' does not exist'
                else:
                    raise SystemExit
            else:
                # write date modified to log
                with open(filename) as f:
                    lines = f.readlines()
                for line in lines:
                    if '#' in line and 'date' in line.lower():
                        log.info('date stamp of {}: {}'
                                 .format(filename, line.strip().split()[-1]))
                        break

    check_files([get_par(C.sex_cfg,tel), get_par(C.psfex_cfg,tel),
                 get_par(C.swarp_cfg,tel)], log)
    if new:
        check_files([get_par(C.sex_par,tel)], log)
    if ref:
        check_files([get_par(C.sex_par_ref,tel)], log)
    if get_par(C.dosex_psffit,tel):
        check_files([get_par(C.sex_cfg_psffit,tel),
                     get_par(C.sex_par_psffit,tel)], log)

    # the elements in [keywords] should be defined as strings, but do
    # not refer to the actual keyword names; the latter are
    # constructed inside the [read_header] function
    keywords = ['naxis2', 'naxis1', 'gain', 'ron', 'satlevel', 'ra', 'dec', 'pixscale']

    if new:
        # read in header of new_fits
        t = time.time()
        header_new = read_hdulist (new_fits, ext_header=0)
        ysize_new, xsize_new, gain_new, readnoise_new, satlevel_new, ra_new, dec_new, pixscale_new = (
            read_header(header_new, keywords, log))

    if ref:
        # read in header of ref_fits
        header_ref = read_hdulist (ref_fits, ext_header=0)
        ysize_ref, xsize_ref, gain_ref, readnoise_ref, satlevel_ref, ra_ref, dec_ref, pixscale_ref = (
            read_header(header_ref, keywords, log))


    # function to run SExtractor on fraction of the image, applied
    # below to new and/or ref image
    def sex_fraction (base, sexcat, pixscale, imtype, header, log):
        fwhm, fwhm_std, elong, elong_std = run_sextractor(base+'.fits', sexcat, get_par(C.sex_cfg,tel),
                                                          get_par(C.sex_par,tel), pixscale, log, header,
                                                          fit_psf=False, return_fwhm_elong=True,
                                                          fraction=get_par(C.fwhm_imafrac,tel),
                                                          fwhm=5.0, save_bkg=False, update_vignet=False)
        log.info('fwhm_{}: {:.3f} +- {:.3f}'.format(imtype, fwhm, fwhm_std))
        # if SEEING keyword exists, report its value in the log
        if 'SEEING' in header:
            log.info('fwhm from header: ' + str(header['SEEING']))

        # add header keyword(s):
        header['S-FWHM'] = (fwhm, '[pix] Sextractor FWHM estimate')
        header['S-FWSTD'] = (fwhm_std, '[pix] sigma (STD) FWHM estimate')
        header['S-SEEING'] = (fwhm*pixscale, '[arcsec] SExtractor seeing estimate')
        header['S-SEESTD'] = (fwhm_std*pixscale, '[arcsec] sigma (STD) SExtractor seeing')
        header['S-ELONG'] = (elong, 'SExtractor ELONGATION (A/B) estimate')
        header['S-ELOSTD'] = (elong_std, 'sigma (STD) SExtractor ELONGATION (A/B)')
        
        return fwhm, fwhm_std, elong, elong_std
            
    # if [new_fits] is not defined, [fwhm_new]=None ensures that code
    # does not crash in function [update_vignet_size] which uses both
    # [fwhm_new] and [fwhm_max]
    fwhm_new = 0
    if new:
        # run SExtractor for seeing estimate of new_fits and ref_fits;
        # both new and ref need to have their fwhm determined before
        # continuing, as both [fwhm_new] and [fwhm_ref] are required
        # to determine the VIGNET size set in the full SExtractor run
        sexcat_new = base_new+'_ldac.fits'
        fwhm_new, fwhm_std_new, elong_new, elong_std_new = sex_fraction(
            base_new, sexcat_new, pixscale_new, 'new', header_new, log)

    fwhm_ref = 0
    if ref:
        # do the same for the reference image
        sexcat_ref = base_ref+'_ldac.fits'
        fwhm_ref, fwhm_std_ref, elong_ref, elong_std_ref = sex_fraction(
            base_ref, sexcat_ref, pixscale_ref, 'ref', header_ref, log)

    # function to run SExtractor on full image, followed by Astrometry.net
    # to find the WCS solution, applied below to new and/or ref image
    def sex_wcs (base, sexcat, sex_params, pixscale, fwhm, update_vignet, imtype,
                 fits_mask, ra, dec, xsize, ysize, header, log):

        # run SExtractor on full image
        if not os.path.isfile(sexcat) or get_par(C.redo,tel):
            try:
                result = run_sextractor(base+'.fits', sexcat, get_par(C.sex_cfg,tel),
                                        sex_params, pixscale, log, header, fit_psf=False,
                                        return_fwhm_elong=False, fraction=1.0, fwhm=fwhm,
                                        save_bkg=True, update_vignet=update_vignet,
                                        imtype=imtype, mask=fits_mask)
            except Exception as e:
                SE_processed = False
                log.info(traceback.format_exc())
                log.error('exception was raised during [run_sextractor]: {}'.format(e))  
            else:
                SE_processed = True
                
            # copy the LDAC binary fits table output from SExtractor (with
            # '_ldac' in the name) to a normal binary fits table;
            # Astrometry.net needs the latter, but PSFEx needs the former,
            # so keep both
            ldac2fits (sexcat, base+'_cat.fits', log)

            # add header keyword(s):
            header['S-P'] = (SE_processed, 'successfully processed by SExtractor?')
            # SExtractor version
            cmd = ['sex', '-v']
            result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            version = result.stdout.read().split()[2].decode('UTF-8')
            header['S-VERS'] = (version, 'SExtractor version used')

        # determine WCS solution
        data_cal = None
        if ('CTYPE1' not in header.keys() and 'CTYPE2' not in header.keys()) or get_par(C.redo,tel):
            try:
                if not get_par(C.skip_wcs,tel):
                    data_cal = run_wcs(base+'.fits', base+'.fits', ra, dec, pixscale, xsize, ysize,
                                       header, log)
            except Exception as e:
                WCS_processed = False
                log.info(traceback.format_exc())
                log.error('exception was raised during [run_wcs]: {}'.format(e))  
            else:
                WCS_processed = True

                # add header keyword(s):
                header['A-P'] = (WCS_processed, 'successfully processed by Astrometry.net?')

        return data_cal

    if new:
        # now run above function [sex_wcs] on new image
        data_cal_new = sex_wcs(base_new, sexcat_new, get_par(C.sex_par,tel), pixscale_new,
                               fwhm_new, True, 'new', new_fits_mask, ra_new, dec_new, xsize_new,
                               ysize_new, header_new, log)

    if ref:
        # and reference image
        data_cal_ref = sex_wcs(base_ref, sexcat_ref, get_par(C.sex_par_ref,tel), pixscale_ref,
                               fwhm_ref, True, 'ref', ref_fits_mask, ra_ref, dec_ref, xsize_ref,
                               ysize_ref, header_ref, log)
        # N.B.: two differences with new image: SExtractor parameter
        # file (new: C.sex_par, ref: C.sex_par_ref) and update_vignet
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
        # initialize header to be recorded for keywords related to the
        # comparison of new and ref
        header_zogy = fits.Header()

        # remap ref to new
        ref_fits_remap = base_ref+'_remap.fits'
        ref_fits_remap = get_remap_name(base_new, base_ref, ref_fits_remap)
            
        if not os.path.isfile(ref_fits_remap) or get_par(C.redo,tel):
            # if reference image is poorly sampled, could use bilinear
            # interpolation for the remapping using SWarp - this
            # removes artefacts around bright stars (see Fig.6 in the
            # SWarp User's Guide). However, despite these artefacts,
            # the Scorr image still appears to be better with LANCZOS3
            # than when BILINEAR is used.
            resampling_type='LANCZOS3'
            # if fwhm_ref <= 2: resampling_type='BILINEAR'
            try:
                result = run_remap(base_new+'.fits', base_ref+'.fits', ref_fits_remap,
                                   [ysize_new, xsize_new], gain=gain_new, log=log,
                                   config=get_par(C.swarp_cfg,tel),
                                   resampling_type=resampling_type, resample='Y')
            except Exception as e:
                remap_processed = False
                log.info(traceback.format_exc())
                log.error('exception was raised during [run_remap]: {}'.format(e))  
            else:
                remap_processed = True
            header_zogy['SWARP-P'] = (remap_processed, 'reference image successfully SWarped?')

            
    # determine cutouts
    if new:
        xsize = xsize_new
        ysize = ysize_new
    else:
        xsize = xsize_ref
        ysize = ysize_ref
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(get_par(C.subimage_size,tel),
                                                                       ysize, xsize, log)
    nxsubs = xsize/get_par(C.subimage_size,tel)
    nysubs = ysize/get_par(C.subimage_size,tel)
    
    ysize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    xsize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    nsubs = centers.shape[0]
    if get_par(C.verbose,tel):
        log.info('nsubs ' + str(nsubs))
        #for i in range(nsubs):
        #    log.info('i ' + str(i))
        #    log.info('cuts_ima[i] ' + str(cuts_ima[i]))
        #    log.info('cuts_ima_fft[i] ' + str(cuts_ima_fft[i]))
        #    log.info('cuts_fft[i] ' + str(cuts_fft[i]))
            
    # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with new,
    # ref, psf and background images
    if new:
        data_new, psf_new, psf_orig_new, data_new_bkg, data_new_bkg_std, data_new_mask = (
            prep_optimal_subtraction(base_new+'.fits', nsubs, 'new', fwhm_new, header_new,
                                     log, fits_mask=new_fits_mask, data_cal=data_cal_new)
        )
            
    # same for [ref_fits]; if either [new_fits] was not defined,
    # [ref_fits_remap] will be None
    if ref_fits is not None:
        data_ref, psf_ref, psf_orig_ref, data_ref_bkg, data_ref_bkg_std, data_ref_mask = (
            prep_optimal_subtraction(base_ref+'.fits', nsubs, 'ref', fwhm_ref, header_ref,
                                     log, fits_mask=ref_fits_mask, ref_fits_remap=ref_fits_remap,
                                     data_cal=data_cal_ref)
        )
            
    if get_par(C.verbose,tel) and new:
        log.info('data_new.dtype {}'.format(data_new.dtype))
        log.info('psf_new.dtype {}'.format(psf_new.dtype))
        log.info('data_new_bkg.dtype {}'.format(data_new_bkg.dtype))
        log.info('data_new_bkg_std.dtype {}'.format(data_new_bkg_std.dtype))
        log.info('data_new_mask.dtype {}'.format(data_new_mask.dtype))
    
    if new and ref:
        # get x, y and fratios from matching PSFex stars across entire
        # frame the "_sub" output arrays are the values to be used for
        # the subimages below in the function [zogy_subloop]
        x_fratio, y_fratio, fratio, dx, dy, fratio_sub, dx_sub, dy_sub = (
            get_fratio_dxdy(base_new+'_psfex.cat', base_ref+'_psfex.cat',
                            base_new+'_cat.fits', base_ref+'_cat.fits',
                            header_new, header_ref, 
                            nsubs, cuts_ima, log, header_zogy))
        
        # fratio is in counts, convert to electrons, in case gains of new
        # and ref images are not identical
        fratio *= gain_new / gain_ref
        fratio_sub *= gain_new / gain_ref
                
        if get_par(C.make_plots,tel):

            dx_std = np.std(dx)
            dy_std = np.std(dy)
                        
            def plot (x, y, limits, xlabel, ylabel, filename, annotate=True):
                plt.axis(limits)
                plt.plot(x, y, 'go', color='tab:blue', markersize=5, markeredgecolor='k')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
                if annotate:
                    plt.annotate('STD(dx): {:.3f} pixels'.format(dx_std), xy=(0.02,0.95), xycoords='axes fraction')
                    plt.annotate('STD(dy): {:.3f} pixels'.format(dy_std), xy=(0.02,0.90), xycoords='axes fraction')
                if filename != '':
                    plt.savefig(filename)
                if get_par(C.show_plots,tel):
                    plt.show()
                plt.close()

            plot (x_fratio, y_fratio, (0,xsize_new,0,ysize_new), 'x (pixels)', 'y (pixels)',
                  base_newref+'_xy.pdf', annotate=False)
            plot (dx, dy, (-1,1,-1,1), 'dx (pixels)', 'dy (pixels)',
                  base_newref+'_dxdy.pdf')
            #plot (x_fratio, dr, (0,xsize_new,0,1), 'x (pixels)', 'dr (pixels)',
            #      base_newref+'_drx.pdf', annotate=False)
            #plot (y_fratio, dr, (0,ysize_new,0,1), 'y (pixels)', 'dr (pixels)',
            #      base_newref+'_dry.pdf', annotate=False)
            plot (x_fratio, dx, (0,xsize_new,-1,1), 'x (pixels)', 'dx (pixels)',
                  base_newref+'_dxx.pdf')
            plot (y_fratio, dy, (0,ysize_new,-1,1), 'y (pixels)', 'dy (pixels)',
                  base_newref+'_dyy.pdf')
            # plot dx and dy as function of distance from the image center
            xcenter = xsize_new/2
            ycenter = ysize_new/2
            dist = np.sqrt((x_fratio-xcenter)**2 + (y_fratio-ycenter)**2)
            plot (dist, dx, (0,np.amax(dist),0,1), 'distance from image center (pixels)',
                  'dx (pixels)', base_newref+'_dxdist.pdf')
            plot (dist, dy, (0,np.amax(dist),0,1), 'distance from image center (pixels)',
                  'dy (pixels)', base_newref+'_dydist.pdf')

        # initialize fakestar flux arrays if fake star(s) are being added
        # - this is to make a comparison plot of the input and output flux
        if get_par(C.nfakestars,tel)>0:
            nfake = nsubs * get_par(C.nfakestars,tel)
            fakestar_xcoord = np.ndarray(nfake, dtype=int)
            fakestar_ycoord = np.ndarray(nfake, dtype=int)
            fakestar_flux_input = np.ndarray(nfake)
            fakestar_flux_output = np.ndarray(nfake)
            fakestar_fluxerr_output = np.ndarray(nfake)        
            fakestar_s2n_output = np.ndarray(nfake)

            # add the fake stars to the new subimages
            for nsub in range(nsubs):
                index_fake = tuple([slice(nsub*get_par(C.nfakestars,tel),
                                          (nsub+1)*get_par(C.nfakestars,tel))])
                fakestar_xcoord[index_fake], fakestar_ycoord[index_fake], \
                    fakestar_flux_input[index_fake] = add_fakestars (psf=psf_orig_new[nsub],
                                                                     data=data_new[nsub],
                                                                     bkg=data_new_bkg[nsub],
                                                                     readnoise=readnoise_new,
                                                                     fwhm=fwhm_new, log=log)


        start_time2 = os.times()

        if get_par(C.timing,tel):
            t_zogypool = time.time()

        log.info('Executing run_ZOGY on subimages ...')

        # tried to use multiprocessing using Pool, but ran into error:
        # 'IOError: bad message length', which is probably due to
        # arrays passed on to function being too large
        zogy_subloop_partial = partial(zogy_subloop,
                                       data_ref=data_ref,
                                       data_new=data_new,
                                       psf_ref=psf_ref,
                                       psf_new=psf_new,
                                       data_ref_bkg=data_ref_bkg,
                                       data_new_bkg=data_new_bkg,
                                       data_ref_bkg_std=data_ref_bkg_std,
                                       data_new_bkg_std=data_new_bkg_std,
                                       readnoise_ref=readnoise_ref,
                                       readnoise_new=readnoise_new,
                                       fratio_sub=fratio_sub,
                                       dx_sub=dx_sub, dy_sub=dy_sub,
                                       log=log)
        pool = ThreadPool(1)
        try:
            results_pool_zogy = pool.map(zogy_subloop_partial, range(nsubs))
            pool.close()
            pool.join()
        except Exception as e:
            zogy_processed = False
            log.info(traceback.format_exc())
            log.error('exception was raised during [zogy_subloop]: {}'.format(e))  
        else:
            zogy_processed = True
            
        if get_par(C.timing,tel):
            log_timing_memory (t0=t_zogypool, label='ZOGY pool', log=log)
        
        # loop over results from pool and paste subimages
        # into output images

        # first initialize full output images
        data_D_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        #data_S_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_Scorr_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_Fpsf_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_Fpsferr_full = np.ndarray((ysize_new, xsize_new), dtype='float32')

        data_new_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_ref_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
        data_new_mask_full = np.ndarray((ysize_new, xsize_new), dtype='uint8')
        data_ref_mask_full = np.ndarray((ysize_new, xsize_new), dtype='uint8')

        #objgraph.show_most_common_types()

        for nsub in range(nsubs):

            # using results from pool:
            data_D, data_S, data_Scorr, data_Fpsf, data_Fpsferr = results_pool_zogy[nsub]
            
            # if one or more fake stars were added to the subimages,
            # compare the input flux with the PSF flux determined by
            # run_ZOGY.
            if get_par(C.nfakestars,tel)>0:
                index_fake = tuple([slice(nsub*get_par(C.nfakestars,tel),
                                          (nsub+1)*get_par(C.nfakestars,tel))])
                x_fake = fakestar_xcoord[index_fake]-1
                y_fake = fakestar_ycoord[index_fake]-1
                fakestar_flux_output[index_fake] = data_Fpsf[y_fake, x_fake]
                fakestar_fluxerr_output[index_fake] = data_Fpsferr[y_fake, x_fake]
                # and S/N from Scorr
                fakestar_s2n_output[index_fake] = data_Scorr[y_fake, x_fake]
                # x,y coords are in the fft frame; infer the x,y pixel
                # coordinates in the full image by using [subcutfft],
                # which defines the pixel indices [y1 y2 x1 x2]
                # identifying the corners of the fft subimage in the
                # entire input/output image coordinate frame; used various
                # times below
                subcutfft = cuts_ima_fft[nsub]
                fakestar_xcoord[index_fake] += subcutfft[2]
                fakestar_ycoord[index_fake] += subcutfft[0]

            # put sub images without the borders into output frames
            subcut = cuts_ima[nsub]
            index_subcut = tuple([slice(subcut[0],subcut[1]), slice(subcut[2],subcut[3])])
            x1, y1 = get_par(C.subimage_border,tel), get_par(C.subimage_border,tel)
            x2, y2 = x1+get_par(C.subimage_size,tel), y1+get_par(C.subimage_size,tel)
            index_extract = tuple([slice(y1,y2), slice(x1,x2)])

            data_D_full[index_subcut] = data_D[index_extract] #/ gain_new
            #data_S_full[index_subcut] = data_S[index_extract]
            data_Scorr_full[index_subcut] = data_Scorr[index_extract]
            data_Fpsf_full[index_subcut] = data_Fpsf[index_extract]
            data_Fpsferr_full[index_subcut] = data_Fpsferr[index_extract]

            data_new_full[index_subcut] = data_new[nsub][index_extract]
            data_ref_full[index_subcut] = data_ref[nsub][index_extract]
            data_new_mask_full[index_subcut] = data_new_mask[nsub][index_extract]
            data_ref_mask_full[index_subcut] = data_ref_mask[nsub][index_extract]
        
            if get_par(C.display,tel) and (nsub==0 or nsub==nysubs-1 or nsub==nsubs/2 or
                              nsub==nsubs-nysubs or nsub==nsubs-1):

                subend = '_sub'+str(nsub)+'.fits'

                # just for displaying purpose:
                fits.writeto(base_newref+'_D'+subend, data_D.astype('float32'), overwrite=True)
                fits.writeto(base_newref+'_S'+subend, data_S.astype('float32'), overwrite=True)
                fits.writeto(base_newref+'_Scorr'+subend, data_Scorr.astype('float32'), overwrite=True)
        
                # write new and ref subimages to fits
                newname = base_new+subend
                refname = base_ref+subend
                fits.writeto(newname, data_new[nsub].astype('float32'), overwrite=True)
                fits.writeto(refname, data_ref[nsub].astype('float32'), overwrite=True)

                # background images
                fits.writeto(base_new+'_bkg'+subend, data_new_bkg[nsub].astype('float32'), overwrite=True)
                fits.writeto(base_ref+'_bkg'+subend, data_ref_bkg[nsub].astype('float32'), overwrite=True)
                fits.writeto(base_new+'_std'+subend, data_new_bkg_std[nsub].astype('float32'), overwrite=True)
                fits.writeto(base_ref+'_std'+subend, data_ref_bkg_std[nsub].astype('float32'), overwrite=True)
                # masks
                fits.writeto(base_new+'_mask'+subend, data_new_mask[nsub].astype('uint8'), overwrite=True)
                fits.writeto(base_ref+'_mask'+subend, data_ref_mask[nsub].astype('uint8'), overwrite=True)
            
                # and display
                cmd = ['ds9','-zscale',newname,refname, base_newref+'_D'+subend,
                       base_newref+'_S'+subend, base_newref+'_Scorr'+subend, 
                       base_new+'_bkg'+subend, base_ref+'_bkg'+subend,
                       base_new+'_std'+subend, base_ref+'_std'+subend,
                       base_new+'_mask'+subend, base_ref+'_mask'+subend,
                       base_newref+'_VSn.fits', base_newref+'_VSr.fits',
                       base_newref+'_VSn_ast.fits', base_newref+'_VSr_ast.fits',
                       base_newref+'_Sn.fits', base_newref+'_Sr.fits',
                       base_newref+'_kn.fits', base_newref+'_kr.fits',
                       base_newref+'_Pn_hat.fits', base_newref+'_Pr_hat.fits',
                       base_new+'_psf_ima_config'+subend,
                       base_ref+'_psf_ima_config'+subend,
                       base_new+'_psf_ima_resized_norm'+subend,
                       base_ref+'_psf_ima_resized_norm'+subend, 
                       base_new+'_psf_ima_center'+subend,
                       base_ref+'_psf_ima_center'+subend, 
                       base_new+'_psf_ima_shift'+subend,
                       base_ref+'_psf_ima_shift'+subend]
            
                result = subprocess.call(cmd)


        # compute statistics on Scorr image and show histogram
        # discarding the edge, and using a fraction of the total image
        edge = 100
        nstat = int(0.05 * (xsize_new*ysize_new))
        x_stat = (np.random.rand(nstat)*(xsize_new-2*edge)).astype(int) + edge
        y_stat = (np.random.rand(nstat)*(ysize_new-2*edge)).astype(int) + edge
        mean_Scorr, std_Scorr, median_Scorr = (
            clipped_stats (data_Scorr_full[y_stat,x_stat], clip_zeros=True,
                           make_hist=get_par(C.make_plots,tel), name_hist=base_newref+'_Scorr_hist.pdf',
                           hist_xlabel='value in Scorr image', log=log))
        if get_par(C.verbose,tel):
            log.info('Scorr mean: {:.3f} , median: {:.3f}, std: {:.3f}'
                     .format(mean_Scorr, median_Scorr, std_Scorr))

        # compute statistics on Fpsferr image
        mean_Fpsferr, std_Fpsferr, median_Fpsferr = (
            clipped_stats (data_Fpsferr_full[y_stat,x_stat], make_hist=get_par(C.make_plots,tel),
                           name_hist=base_newref+'_Fpsferr_hist.pdf',
                           hist_xlabel='value in Fpsferr image', log=log))
        if get_par(C.verbose,tel):
            log.info('Fpsferr mean: {:.3f} , median: {:.3f}, std: {:.3f}'
                     .format(mean_Fpsferr, median_Fpsferr, std_Fpsferr))

        # add header keyword(s):
        header_zogy['Z-P'] = (zogy_processed, 'successfully processed by ZOGY?')
        header_zogy['Z-V'] = (__version__, 'ZOGY version used')
        header_zogy['Z-REF'] = (base_ref.split('/')[-1]+'.fits', 'name reference image')
        header_zogy['Z-SIZE'] = (get_par(C.subimage_size,tel), '[pix] size of (square) ZOGY subimages')
        header_zogy['Z-BSIZE'] = (get_par(C.subimage_border,tel), '[pix] size of ZOGY subimage borders')
        header_zogy['Z-SCMED'] = (median_Scorr, 'median Scorr full image')
        header_zogy['Z-SCSTD'] = (std_Scorr, 'sigma (STD) Scorr full image')
        header_zogy['Z-FPEMED'] = (median_Fpsferr, '[e-] median Fpsferr full image')
        header_zogy['Z-FPESTD'] = (std_Fpsferr, '[e-] sigma (STD) Fpsferr full image')

        # find transients using function [get_trans_alt], which
        # applies threshold cuts directly on Scorr for the transient
        # detection, rather than running SExtractor (see below)
        ntrans = get_trans (data_new_full, data_ref_full, data_Scorr_full,
                            data_Fpsf_full, data_Fpsferr_full,
                            data_new_mask_full, data_ref_mask_full, header_new, log)

        # add header keyword(s):
        header_zogy['T-NSIGMA'] = (get_par(C.transient_nsigma,tel),
                                   '[sigma] input transient detection threshold')
        lflux3 = 3.*median_Fpsferr
        lflux5 = 5.*median_Fpsferr
        lflux = float(get_par(C.transient_nsigma,tel)) * median_Fpsferr
        header_zogy['T-LFLUX3'] = (lflux3, '[e-] full-frame transient 3-sigma limiting flux')
        header_zogy['T-LFLUX5'] = (lflux5, '[e-] full-frame transient 5-sigma limiting flux')
        header_zogy['T-LFLUX'] = (lflux, '[e-] full-frame transient {}-sigma limiting flux'
                                  .format(get_par(C.transient_nsigma,tel)))
        header_zogy['T-NTRANS'] = (ntrans, 'number of >= {}-sigma transients (pre-vetting)'
                                   .format(get_par(C.transient_nsigma,tel)))

        # infer limiting magnitudes from corresponding limiting
        # fluxes using zeropoint and median airmass
        if 'PC-ZP' in header_new and 'PC-AIRM' in header_new:
            keywords = ['exptime', 'filter']
            exptime, filt = read_header(header_new, keywords, log)
            if tel=='LCO': filt=filt[0]
            zeropoint = header_new['PC-ZP']
            airmass = header_new['PC-AIRM']
            [lmag3, lmag5, lmag] = apply_zp([lflux3, lflux5, lflux],
                                            zeropoint, airmass, exptime, filt, log)

            header_zogy['T-LMAG3'] = (lmag3, '[mag] full-frame transient 3-sigma limiting mag')
            header_zogy['T-LMAG5'] = (lmag5, '[mag] full-frame transient 5-sigma limiting mag')
            header_zogy['T-LMAG'] = (lmag, '[mag] full-frame transient {}-sigma limiting mag' 
                                     .format(get_par(C.transient_nsigma,tel)))


        if get_par(C.nfakestars,tel)==0:
            
            # still write these header keywords
            header_zogy['T-NFAKE'] = (get_par(C.nfakestars,tel), 'number of fake stars added to full frame')
            header_zogy['T-FAKESN'] = (get_par(C.fakestar_s2n,tel), 'fake stars input S/N')
            
        else:

            # compare input and output flux
            fluxdiff = (fakestar_flux_input - fakestar_flux_output) / fakestar_flux_input
            fluxdiff_err = fakestar_fluxerr_output / fakestar_flux_input
            fd_mean, fd_std, fd_median = (clipped_stats(fluxdiff, clip_zeros=False, log=log))
            fderr_mean, fderr_std, fderr_median = (clipped_stats(fluxdiff_err, clip_zeros=False, log=log))

            # add header keyword(s):
            nfake = len(fakestar_flux_input)
            header_zogy['T-NFAKE'] = (nfake, 'number of fake stars added to full frame')
            header_zogy['T-FAKESN'] = (get_par(C.fakestar_s2n,tel), 'fake stars input S/N')

            # write to ascii file
            filename = base_new+'_fakestars.dat'
            f = open(filename, 'w')
            f.write('{:1} {:11} {:11} {:12} {:12} {:16} {:11} {:11}\n'
                    .format('#', 'xcoord[pix]', 'ycoord[pix]', 'flux_in[e-]', 'flux_out[e-]',
                            'fluxerr_out[e-]', 'S/N_input', 'S/N_output'))
            for i in range(nfake):
                f.write('{:11.2f} {:11.2f} {:12.2e} {:12.2e} {:16.2e} {:11.2f} {:11.2f}\n'
                        .format(fakestar_xcoord[i], fakestar_ycoord[i],
                                fakestar_flux_input[i], fakestar_flux_output[i], fakestar_fluxerr_output[i],
                                get_par(C.fakestar_s2n,tel), fakestar_s2n_output[i]))
            f.close()

            # make comparison plot of flux input and output
            if get_par(C.make_plots,tel):
            
                x = np.arange(nsubs*get_par(C.nfakestars,tel))+1
                y = fakestar_flux_input
                plt.plot(x, y, 'o', color='tab:blue', markersize=7, markeredgecolor='k')
                plt.xlabel('fakestar number (total: nsubs x C.nfakestars)')
                plt.ylabel('true flux (e-)')
                plt.title('fake stars true input flux')
                plt.savefig(base_newref+'_fakestar_flux_input.pdf')
                if get_par(C.show_plots,tel): plt.show()
                plt.close()

                #plt.axis((0,nsubs,0,2))
                plt.errorbar(x, fluxdiff, yerr=fluxdiff_err, linestyle='None', ecolor='k', capsize=2)
                plt.plot(x, fluxdiff, 'o', color='tab:blue', markersize=7, markeredgecolor='k')
                plt.xlabel('fakestar number (total: nsubs x C.nfakestars)')
                plt.ylabel('(true flux - ZOGY flux) / true flux')
                plt.title('true flux vs. ZOGY Fpsf; mean:{:.3f}, std:{:.3f}, data err:{:.3f}'
                          .format(fd_mean, fd_std, fderr_mean))
                plt.savefig(base_newref+'_fakestar_flux_input_vs_ZOGYoutput.pdf')
                if get_par(C.show_plots,tel): plt.show()
                plt.close()

                # same for S/N as determined by Scorr
                y = fakestar_s2n_output
                plt.plot(x, y, 'o', color='tab:blue', markersize=7, markeredgecolor='k')
                plt.xlabel('fakestar number (total: nsubs x C.nfakestars)')
                plt.ylabel('S/N from Scorr')
                plt.title('fakestars signal-to-noise ratio from Scorr')
                plt.savefig(base_newref+'_fakestar_S2N_ZOGYoutput.pdf')
                if get_par(C.show_plots,tel): plt.show()
                plt.close()


        # write full ZOGY output images to fits
        if get_par(C.timing,tel):
            t_fits = time.time() 

        header_newzogy = header_new + header_zogy
        #header_newzogy.add_comment('many keywords, incl. WCS solution, are from corresponding image')
        fits.writeto(base_newref+'_D.fits', data_D_full, header_newzogy, overwrite=True)
        #fits.writeto(base_newref+'_S.fits', data_S_full, header_newzogy, overwrite=True)
        fits.writeto(base_newref+'_Scorr.fits', data_Scorr_full, header_newzogy, overwrite=True)
        fits.writeto(base_newref+'_Fpsf.fits', data_Fpsf_full, header_newzogy, overwrite=True)
        fits.writeto(base_newref+'_Fpsferr.fits', data_Fpsferr_full, header_newzogy, overwrite=True)
        
        if get_par(C.timing,tel):
            log_timing_memory (t0=t_fits, label='writing D, Scorr, Fpsf and Fpsferr fits images', log=log)
        
        if get_par(C.display,tel):
            fits.writeto('new.fits', data_new_full, header_new, overwrite=True)
            fits.writeto('ref.fits', data_ref_full, header_ref, overwrite=True)
            fits.writeto('new_mask.fits', data_new_mask_full, header_new, overwrite=True)
            fits.writeto('ref_mask.fits', data_ref_mask_full, header_ref, overwrite=True)

                                
    # using the function [format_cat], write the new, ref and
    # transient output catalogues with the desired format, where the
    # thumbnail images (new, ref, D and Scorr) around each transient
    # are added as array columns in the transient catalogue.

    # new catalogue
    if new:
        exptime_new = read_header(header_new, ['exptime'], log)
        cat_new = base_new+'_cat.fits'
        cat_new_out = base_new+'_cat.fits'
        header_cat = read_hdulist(cat_new, ext_header=1)
        if 'FORMAT-P' not in header_cat.keys():
            result = format_cat (cat_new, cat_new_out, log, cat_type='new',
                                 header_toadd=header_new, exptime=exptime_new,
                                 apphot_radii=get_par(C.apphot_radii,tel))
    # ref catalogue
    if ref:
        exptime_ref = read_header(header_ref, ['exptime'], log)
        cat_ref = base_ref+'_cat.fits'
        cat_ref_out = base_ref+'_cat.fits'
        header_cat = read_hdulist(cat_ref, ext_header=1)
        if 'FORMAT-P' not in header_cat.keys():
            result = format_cat (cat_ref, cat_ref_out, log, cat_type='ref',
                                 header_toadd=header_ref, exptime=exptime_ref,
                                 apphot_radii=get_par(C.apphot_radii,tel))
    # trans catalogue
    if new and ref:
        cat_trans = base_newref+'.transcat'
        cat_trans_out = base_newref+'_trans.fits'
        thumbnail_data = [data_new_full, data_ref_full, data_D_full, data_Scorr_full]
        thumbnail_keys = ['THUMBNAIL_RED', 'THUMBNAIL_REF', 'THUMBNAIL_D', 'THUMBNAIL_SCORR']
        # need to take care of objects closer than 32/2 pixels to
        # the full image edge in creation of thumbnails - results
        # in an error if transients are close to the edge
        result = format_cat (cat_trans, cat_trans_out, log, cat_type='trans',
                             thumbnail_data=thumbnail_data, thumbnail_keys=thumbnail_keys,
                             thumbnail_size=100, header_toadd=header_newzogy,
                             exptime=exptime_new, apphot_radii=get_par(C.apphot_radii,tel))

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
        # and display
        if get_par(C.display,tel):
        #if True:
            cmd = ['ds9', '-zscale', 'new.fits', 'new_mask.fits',
                   'ref.fits', 'ref_mask.fits', 
                   base_newref+'_D.fits', base_newref+'_Scorr.fits']
            # add ds9 regions
            if get_par(C.make_plots,tel):
                cmd += ['-regions', base_newref+'_ds9regions.txt']
            result = subprocess.call(cmd)
    # shutdown internal log before exiting
    if log is None:
        logging.shutdown()
    return 'info', 'Successfully ran ZOGY on image.'


################################################################################

def get_par (par, tel):

    """Function to check if [par] is a dictionary with one of the keys
       being [tel] or the alphabetic part of [tel] (e.g. 'BG'), and if
       so, return the corresponding value. Otherwise just return the
       parameter value."""

    par_val = par
    if type(par) is dict:
        if tel in par.keys():
            par_val = par[tel]
        else:
            # cut off digits from [tel]
            tel_base = ''.join([char for char in tel if char.isalpha()])
            if tel_base in par.keys():
                par_val = par[tel_base]
        
    return par_val


################################################################################

def add_fakestars (psf, data, bkg, readnoise, fwhm, log):

    """Function to add fakestars to the image as defined in [data] (this
    array is updated in place) with the PSF image as defined in
    [psf]. The number of stars added is set by [C.nfakestars]; the
    first star is added at the center of the (sub)image [data], while
    any additional stars are randomly distributed across the
    image. The S/N is determined by [C.fakestar_s2n] and [readnoise];
    the inputs [bkg] and [fwhm] are only used to double-check the S/N
    calculation performed by function [flux_optimal_s2n] with two
    different functions [get_s2n_ZO] and [get_optflux_Naylor]. The
    size of the image regions that are updated is half that of the
    global parameter [psf_size_new].

    The function returns lists that contain: 1) the x pixel
    coordinates, 2) the y pixel coordinates and 3) the fluxes of the
    fake stars that were added.

    """

    ysize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    xsize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    psf_hsize = psf_size_new/2
    
    # place stars in random positions across the image, keeping
    # C.subimage_border + psf_size_new/2 pixels off each edge
    edge = get_par(C.subimage_border,tel) + psf_size_new/2 + 1
    xpos = (np.random.rand(get_par(C.nfakestars,tel))*(xsize_fft-2*edge) + edge).astype(int)
    ypos = (np.random.rand(get_par(C.nfakestars,tel))*(ysize_fft-2*edge) + edge).astype(int)
    # place first star at the center of the image
    xpos[0] = xsize_fft/2
    ypos[0] = ysize_fft/2
    flux_fakestar = np.zeros(get_par(C.nfakestars,tel))
    
    for nstar in range(get_par(C.nfakestars,tel)):
            
        index_temp = tuple([slice(ypos[nstar]-psf_hsize, ypos[nstar]+psf_hsize+1),
                            slice(xpos[nstar]-psf_hsize, xpos[nstar]+psf_hsize+1)])

        # Use function [flux_optimal_s2n] to estimate flux needed for
        # star with S/N of [C.fakestar_s2n].  This S/N estimate
        # includes the Poisson noise from any object that happens to
        # be present in the image at the fakestar position.  If this
        # should be just the background instead, replace data with bkg.
        flux_fakestar[nstar] = flux_optimal_s2n (psf, data[index_temp], readnoise,
                                                 get_par(C.fakestar_s2n,tel), fwhm=fwhm)
        # multiply psf_orig_new to contain fakestar_flux
        psf_fakestar = psf * flux_fakestar[nstar]
        # add fake star to new image
        data[index_temp] += psf_fakestar

        if get_par(C.verbose,tel):
            data_fakestar = psf_fakestar + bkg[index_temp]
            log.info('fakestar flux: {} e-'.format(flux_fakestar[nstar]))
            flux, fluxerr = flux_optimal(psf, data_fakestar, bkg[index_temp],
                                         readnoise, log=log)
            log.info('recovered flux: {}, fluxerr: {}, S/N: {}'.format(flux, fluxerr, flux/fluxerr))
            
            # check S/N with Eq. 51 from Zackay & Ofek 2017, ApJ, 836, 187
            s2n = get_s2n_ZO(psf, data_fakestar, bkg[index_temp],
                             data_fakestar+readnoise**2)
            #log.info('S/N check (Eq. 51 Zackay & Ofek 2017): {}'.format(s2n))
            
            # check S/N with Eqs. from Naylor (1998)
            flux, fluxerr = get_optflux_Naylor(psf, data_fakestar, bkg[index_temp],
                                               data_fakestar+readnoise**2)
            #log.info('Naylor recovered flux: {}, fluxerr: {}, S/N: {}'.format(flux, fluxerr, flux/fluxerr))

    return xpos+1, ypos+1, flux_fakestar
            
            
################################################################################

def read_hdulist (fits_file, ext_data=None, ext_header=None, dtype=None,
                  columns=None):


    # read data if [ext_data] is defined
    if ext_data is not None:

        if type(ext_data)==int:
            # if single extension is provided, read data into fitsrec array
            with fits.open(fits_file) as hdulist:
                data = hdulist[ext_data].data

            # convert to [dtype] if it is defined
            if dtype is not None:
                data = data.astype(dtype, copy=False)

        else:
            # if multiple extensions are provided, read data into Table array
            for n_ext, ext in enumerate(ext_data):
                with fits.open(fits_file) as hdulist:
                    data_temp = hdulist[ext].data
                # convert to table, as otherwise concatenation of
                # extensions below using [stack_arrays] is slow
                data_temp = Table(data_temp)
                # could also read fits extension into Table directly,
                # but this is about twice as slow as the 2 steps above
                #data_temp = Table.read(fits_file, hdu=ext)
                if n_ext==0:
                    data = data_temp
                else:
                    #data = stack_arrays((data, data_temp), asrecarray=True, usemask=False)
                    # following does not work if data is a fitsrec array and the
                    # array contains boolean fields, as these are incorrectly converted 
                    data = np.concatenate([data, data_temp])


    # read header if [ext_header] is defined
    if ext_header is not None:
        with fits.open(fits_file) as hdulist:
            header = hdulist[ext_header].header
            
    if columns is not None:
        # only return defined columns
        # no check is done whether they exist or not
        return [data[col] for col in columns]
    else:
        # return data and header depending on whether [ext_data]
        # and [ext_header] are defined or not
        if ext_data is not None:
            if ext_header is not None:
                return data, header
            else:
                return data
        else:
            if ext_header is not None:
                return header
            else:
                return 
            

################################################################################

def format_cat (cat_in, cat_out, log=None, thumbnail_data=None, thumbnail_keys=None,
                thumbnail_size=64, cat_type=None, header_toadd=None, exptime=0.,
                apphot_radii=None):

    """Function that formats binary fits table [cat_in] according to
        MeerLICHT/BlackGEM specifications and saves the resulting
        binary fits table [cat_out].

    """

    if cat_in is not None:

        if get_par(C.timing,tel): t = time.time()

        with fits.open(cat_in) as hdulist:
            prihdu = hdulist[0]
            header = hdulist[1].header
            data = hdulist[1].data

        if header_toadd is not None:
            header += header_toadd

    else:
        header = header_toadd
            
    thumbnail_size2 = str(thumbnail_size**2)
        
    # this [formats] dictionary lists the output format, the output
    # column unit, and the desired format
    formats = {
        'NUMBER':         ['J', ''     ], #, 'uint16'],
        'XWIN_IMAGE':     ['E', 'pix'  ], #, 'flt32' ],
        'YWIN_IMAGE':     ['E', 'pix'  ], #, 'flt32' ],
        'ERRX2WIN_IMAGE': ['E', 'pix^2'], #, 'flt16' ],
        'ERRY2WIN_IMAGE': ['E', 'pix^2'], #, 'flt16' ],
        'ERRXYWIN_IMAGE': ['E', 'pix^2'], #, 'flt16' ],
        'X2WIN_IMAGE':    ['E', 'pix^2'], #, 'flt16' ],
        'Y2WIN_IMAGE':    ['E', 'pix^2'], #, 'flt16' ],
        'XYWIN_IMAGE':    ['E', 'pix^2'], #, 'flt16' ],
        'ELONGATION':     ['E', ''     ], #, 'flt16' ],
        'ALPHAWIN_J2000': ['D', 'deg'  ], #, 'flt64' ],
        'DELTAWIN_J2000': ['D', 'deg'  ], #, 'flt64' ],
        'FLAGS':          ['I', ''     ], #, 'uint8' ],
        'IMAFLAGS_ISO':   ['I', ''     ], #, 'uint8' ],
        'FWHM_IMAGE':     ['E', 'pix'  ], #, 'flt16' ],
        'CLASS_STAR':     ['E', ''     ], #, 'flt16' ],
        'FLUX_APER':      ['E', 'e-/s' ], #, 'flt32' ],
        'FLUXERR_APER':   ['E', 'e-/s' ], #, 'flt16' ],
        'BACKGROUND':     ['E', 'e-/s' ], #, 'flt16' ],
        'FLUX_MAX':       ['E', 'e-/s' ], #, 'flt16' ],
        'FLUX_AUTO':      ['E', 'e-/s' ], #, 'flt32' ],
        'FLUXERR_AUTO':   ['E', 'e-/s' ], #, 'flt16' ],
        'KRON_RADIUS':    ['E', 'pix'  ], #, 'flt16' ],
        'FLUX_ISO':       ['E', 'e-/s' ], #, 'flt32' ],
        'FLUXERR_ISO':    ['E', 'e-/s' ], #, 'flt16' ],
        'ISOAREA_IMAGE':  ['E', 'pix^2'], #, 'flt16' ],
        'MU_MAX':         ['E', 'mag'  ], #, 'flt16' ],
        'FLUX_RADIUS':    ['E', 'pix'  ], #, 'flt16' ],
        'FLUX_PETRO':     ['E', 'e-/s' ], #, 'flt32' ],
        'FLUXERR_PETRO':  ['E', 'e-/s' ], #, 'flt16' ],
        'PETRO_RADIUS':   ['E', 'pix'  ], #, 'flt16' ],
        'FLUX_OPT':       ['E', 'e-/s' ], #, 'flt32' ],
        'FLUXERR_OPT':    ['E', 'e-/s' ], #, 'flt16' ],
        'MAG_OPT':        ['E', 'mag'  ], #, 'flt32' ],
        'MAGERR_OPT':     ['E', 'mag'  ], #, 'flt16' ],
        'FLUX_PSF':       ['E', 'e-/s' ], #, 'flt32' ],
        'FLUXERR_PSF':    ['E', 'e-/s' ], #, 'flt16' ],
        'MAG_PSF':        ['E', 'mag'  ], #, 'flt32' ],
        'MAGERR_PSF':     ['E', 'mag'  ], #, 'flt16' ],
        'S2N':            ['E', ''     ], #, 'flt16' ],
        'THUMBNAIL_RED':  [thumbnail_size2+'E', 'e-' ], #, 'flt16' ],
        'THUMBNAIL_REF':  [thumbnail_size2+'E', 'e-' ], #, 'flt16' ],
        'THUMBNAIL_D':    [thumbnail_size2+'E', 'e-' ], #, 'flt16' ],
        'THUMBNAIL_SCORR':[thumbnail_size2+'E', 'e-' ], #, 'flt16' ]
    }

    if cat_type is None:
        if cat_in is not None:
            keys_to_record = data.dtype.names
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
                          'S2N', 'FLUX_PSF', 'FLUXERR_PSF', 'MAG_PSF', 'MAGERR_PSF']


    def get_col (key, key_new, data_key):

        # function that returns column definition based on input
        # [key], [key_new], and [data_key]; for most fields [key] and
        # [key_new] are the same, except for 'FLUX_APER' and
        # 'FLUXERR_APER', which are split into the separate apertures,
        # and the aperture sizes enter in the new key name as well.
        
        # if exposure time is non-zero, modify all 'e-/s' columns accordingly
        if exptime != 0:
            if formats[key][1]=='e-/s':
                data_key /= exptime 
        else:
            if log is not None:
                log.warn('input [exptime] in function [format_cat] is zero')

        if cat_in is not None:
            col = fits.Column(name=key_new, format=formats[key][0],
                              unit=formats[key][1], #disp=formats[key][2],
                              array=data_key)
        # if [cat_in] is None, define the column but without the data;
        # this is used for making a table with the same field
        # definitions but without any entries
        else:
            col = fits.Column(name=key_new, format=formats[key][0],
                              unit=formats[key][1]) #, disp=formats[key][2])

        return col   

    
    columns = []
    for key in keys_to_record:

        if key=='FLUX_APER' or key=='FLUXERR_APER':
            # update column names of aperture fluxes to include radii
            # loop apertures
            for i_ap in range(len(apphot_radii)):
                key_new = '{}_R{}xFWHM'.format(key, apphot_radii[i_ap])
                columns.append(get_col (key, key_new, data[key][:,i_ap]))
        else:
            if key in data.dtype.names:
                columns.append(get_col (key, key, data[key]))

                
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
                index_full, index_small = get_index_around_xy(ysize, xsize, ycoords[i_pos],
                                                              xcoords[i_pos], thumbnail_size)

                # record in data_col
                try:
                    data_col[i_pos][index_small] = thumbnail_data[i_tn][index_full]                        
                except ValueError as ve:
                    if log is not None:
                        log.info('skipping object at x,y: {:.0f},{:.0f} due to ValueError: {}'.
                                 format(xcoords[i_pos], ycoords[i_pos], ve))
                    
            # add column to table
            dim_str = '('+str(thumbnail_size)+','+str(thumbnail_size)+')'
            key = thumbnail_keys[i_tn]
            col = fits.Column(name=key, format=formats[key][0], unit=formats[key][1], 
                              #disp=formats[key][2],
                              array=data_col, dim=dim_str)
            columns.append(col)

            
    if cat_in is not None:
        # add header keyword indicating catalog was successfully formatted
        header['FORMAT-P'] = (True, 'successfully formatted catalog')

        if get_par(C.timing,tel) and log is not None:
            log_timing_memory (t0=t, label='format_cat', log=log)

    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header += header
    hdu.writeto(cat_out, overwrite=True)
            
    return


################################################################################

def get_index_around_xy(ysize, xsize, ycoord, xcoord, size):

    # size is assumed to be even!
    xpos = int(xcoord)
    ypos = int(ycoord)
    hsize = int(size/2)

    # if footprint is partially off the image, just go ahead
    # with the pixels on the image
    y1 = max(0, ypos-hsize)
    x1 = max(0, xpos-hsize)
    y2 = min(ysize, ypos+hsize)
    x2 = min(xsize, xpos+hsize)
    index = tuple([slice(y1,y2),slice(x1,x2)])
    
    # also determine indices of small image of sizexsize
    y1_small = max(0, hsize-ypos)
    x1_small = max(0, hsize-xpos)
    y2_small = min(size, size-(ypos+hsize-ysize))
    x2_small = min(size, size-(xpos+hsize-xsize))
    index_small = tuple([slice(y1_small,y2_small),slice(x1_small,x2_small)])
    
    return index, index_small


################################################################################

def get_trans (data_new, data_ref, data_Scorr, data_Fpsf, data_Fpsferr,
               data_new_mask, data_ref_mask, header_new, log):

    """Function that selects transient candidates from the significance
    array (data_Scorr), and estimates a number of SExtractor-like
    quantities (N.B.: the normal versions, not the windowed ones!),
    such as position and shape parameters, and also returns
    the peak significance, the PSF flux and error from the ZOGY images
    Scorr, Fpsf and Fpsferr.

    Possible extension in the future: fit the PSF to data_Scorr,
    data_Fpsf, data_Fpsferr to infer the exact position, peak
    significance, peak PSF flux and its error, and an alternative
    measurement of the PSF-fit flux of the candidate transient.  These
    quantities and the chi-square of the fits could be used to assess
    the reality of the transient, and weed out some additional bogus
    transients. The PSF to be fit to these data arrays is a
    combination of the PSFs of the new and ref image, i.e. P_D in
    ZOGY-speak.

    """

    if get_par(C.timing,tel): t = time.time()

    # mask of pixels with absolute values >= C.transient_sigma
    mask_significant_init = (np.abs(data_Scorr) >= get_par(C.transient_nsigma,tel)).astype('uint8')
    
    # mask of pixels beyond neighbour_nsigma
    neighbour_nsigma = 2.
    mask_neighbours = (np.abs(data_Scorr) >= neighbour_nsigma).astype('uint8')
    
    # let significant mask grow until all neighbours are included
    mask_significant = ndimage.morphology.binary_propagation (mask_significant_init,
                                                              mask=mask_neighbours).astype('uint8')

    # label the connecting pixels
    data_Scorr_regions, nregions = ndimage.label(mask_significant, structure=None)

    # to also include pixels connecting only diagonally, use this structure:
    #struct = np.ones((3,3), dtype=bool)
    #data_Scorr_regions, nregions = ndimage.label(data_Scorr, structure=struct)

    # skimage.measure.label is similar to ndimage.label, except that
    # it requires neighbours to have the same pixel value; see
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    # also: it is about twice as slow as ndimage.label
    #data_Scorr_regions, nregions = measure.label(mask_significant, return_num=True, connectivity=1)
    # use [connectivity=1] to not include pixels that are only diagonally connected

    #ds9_arrays(mask_significant_init=mask_significant_init, mask_neighbours=mask_neighbours,
    #           mask_significant=mask_significant, data_Scorr=data_Scorr,
    #           data_Scorr_regions=data_Scorr_regions)

    if get_par(C.verbose,tel):
        log.info('nregions: {}'.format(nregions))
        
    # using skimage.measure.regionprops; see
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    # for list of attributes of [props]
    #region = measure.regionprops(data_Scorr_regions, intensity_image=data_Scorr, cache=True)
    region = measure.regionprops(data_Scorr_regions, cache=True)

    # initialize arrays
    mask_keep = np.zeros(nregions, dtype=bool)
    npixels_array = np.zeros(nregions)
    x_array = np.zeros(nregions)
    y_array = np.zeros(nregions)
    errx2_array = np.zeros(nregions)
    erry2_array = np.zeros(nregions)
    errxy_array = np.zeros(nregions)
    elongation_array = np.zeros(nregions)
    Scorr_array = np.zeros(nregions)
    flux_array = np.zeros(nregions)
    fluxerr_array = np.zeros(nregions)
    
    t1 = time.time()
    # loop over the regions:
    for i in range(nregions):
        
        # determine the indices of the input data arrays corresponding
        # to the current region with the significant pixels
        region_temp = region[i]
        coords = region_temp.coords
        y_index = coords[:,0]
        x_index = coords[:,1]
        index_region = tuple([y_index, x_index])

        # rectangular bounding box of the current region; N.B.: this
        # includes pixels that are not significant and is mostly for
        # displaying purpose
        bbox = region_temp.bbox
        # xmin, ymin, xmax, ymax are indices, not pixel coordinates;
        # they can be directly used to slice an array, i.e. xmax and
        # ymax already have 1 added
        xmin, ymin, xmax, ymax = bbox
        index_bbox = tuple([slice(xmin,xmax),slice(ymin,ymax)])
         
        # check if region is affected by one or more flagged pixels in
        # the input new and ref mask arrays; for the moment, discard
        # the region if sum of flags in either new or ref mask is
        # nonzero
        if (np.sum(data_new_mask[index_region]) > 0 or
            np.sum(data_ref_mask[index_region])):
            continue

        # discard if region area is too small or too big
        npixels = len(coords)
        if npixels < 3 or npixels > 1000:
            continue

        # discard if region contains both positively significant as
        # negatively significant values
        if (np.amax(data_Scorr[index_region]) >= get_par(C.transient_nsigma,tel) and
            np.amin(data_Scorr[index_region]) <= -get_par(C.transient_nsigma,tel)):
            continue

        # x and y indices of peak significance
        index_peak = np.abs(data_Scorr[index_region]).argmax()
        XPEAK = x_index[index_peak]
        YPEAK = y_index[index_peak]
        Scorr_peak = data_Scorr[index_region][index_peak]
        if False:
            log.info('XPEAK: {}, YPEAK: {}, Scorr_peak: {}'.format(XPEAK, YPEAK, Scorr_peak))

        # flux and fluxerr
        flux_peak = data_Fpsf[index_region][index_peak]
        fluxerr_peak = data_Fpsferr[index_region][index_peak]
        
        color_ds9 = 'green'
        data_Scorr_region = np.copy(data_Scorr[index_region])
        if Scorr_peak < 0.:
            color_ds9 = 'pink'
            data_Scorr_region = -data_Scorr_region
        
        # using function [trans_measure], obtain a number of
        # quantities, such as central pixel coordinates and
        # elongation, weighted with image [intensity]
        X, Y, X2, Y2, XY, ERRX2, ERRY2, ERRXY, A, B, THETA, ERRA, ERRB, ERRTHETA = (
            trans_measure(data_Scorr_region, x_index+1, y_index+1,
                          var_bkg=data_Fpsferr[index_region]))
        if B!=0:
            ELONGATION = A/B
        else:
            ELONGATION = 100.
        ELLIPTICITY = 1-B/A

        if False:
            log.info('X: {:.2f}, Y: {:.2f}, X2: {:2f}, Y2: {:2f}, XY: {:.2f}, ERRX2: {:.2f}, ERRY2: {:.2f}, ERRXY: {:.2f}, A: {:.2f}, B: {:.2f}, THETA: {:.2f}, ERRA: {:.2f}, ERRB: {:.2f}, ERRTHETA: {:.2f}, ELONGATION: {:.2f}, ELLIPTICITY: {:.2f}'.format(X, Y, X2, Y2, XY, ERRX2, ERRY2, ERRXY, A, B, THETA, ERRA, ERRB, ERRTHETA, ELONGATION, ELLIPTICITY))

        # try fitting P_D (combination of PSFs of new and ref images)
        # to D, Scorr, Fpsf and Fpsferr images in order to:
        #
        # (1) use chi2 of PSF fit to D to discard fake transients
        #
        # (2) improve the estimate of the peak value in Scorr, Fpsf
        #     and Fpsferr, which should be possible as the PSF is
        #     better sampled than the image pixels
        
        
        # fill output lists
        mask_keep[i] = True
        npixels_array[i] = npixels
        x_array[i] = X
        y_array[i] = Y
        errx2_array[i] = ERRX2
        erry2_array[i] = ERRY2
        errxy_array[i] = ERRXY
        elongation_array[i] = ELONGATION
        Scorr_array[i] = Scorr_peak
        flux_array[i] = flux_peak
        fluxerr_array[i] = fluxerr_peak
        
        if False:
            ds9_arrays(new=data_new[index_bbox],
                       ref=data_ref[index_bbox],
                       #D=data_D[index_bbox],
                       Scorr=data_Scorr[index_bbox],
                       Fpsf=data_Fpsf[index_bbox],
                       Fpsferr=data_Fpsferr[index_bbox],
                       mask_new=data_new_mask[index_bbox],
                       mask_ref=data_ref_mask[index_bbox])
            
    # loop arrays and discard entries within some number of
    # pixels (3xFWHM?) of each other 
    #for i in range(ntrans):

    ntrans = np.sum(mask_keep)
    log.info('ntrans: {}'.format(ntrans))
    
    # need to convert psf fluxes to magnitudes by applying the zeropoint
    keywords = ['exptime', 'filter']
    exptime, filt = read_header(header_new, keywords, log)
    if tel=='LCO': filt=filt[0]
    # get zeropoint from [header_new]
    if 'PC-ZP' in header_new.keys():
        zp = header_new['PC-ZP']
    else:
        zp = get_par(C.zp_default,tel)[filt]
    # get airmass from [header_new]
    if 'PC-AIRM' in header_new.keys():
        airmass = header_new['PC-AIRM']
    elif 'AIRMASSC' in header_new.keys():
        airmass = header_new['PC-AIRM']
    # get magnitudes corresponding to fluxes
    mag_array, magerr_array = apply_zp(flux_array, zp, airmass, exptime, filt, log,
                                       fluxerr=fluxerr_array, zp_std=None)
    
    # create output table:
    table_all = Table([x_array, y_array, errx2_array, erry2_array, errxy_array,
                       elongation_array, Scorr_array, flux_array, fluxerr_array,
                       mag_array, magerr_array],
                      names=('XWIN_IMAGE', 'YWIN_IMAGE', 'ERRX2WIN_IMAGE', 'ERRY2WIN_IMAGE',
                             'ERRXYWIN_IMAGE', 'ELONGATION', 'S2N', 'FLUX_PSF', 'FLUXERR_PSF',
                             'MAG_PSF', 'MAGERR_PSF'))

    # keep relevant transients
    table = table_all[mask_keep]
    # add number
    table['NUMBER'] = np.arange(ntrans)+1
    # add RA and DEC
    wcs = WCS(header_new)
    ra, dec = wcs.all_pix2world(table['XWIN_IMAGE'], table['YWIN_IMAGE'], 1)
    table['ALPHAWIN_J2000'] = ra
    table['DELTAWIN_J2000'] = dec
    
    # create output fits catalog
    table.write(base_newref+'.transcat', format='fits', overwrite=True)

    # determine output transient catalogue array, containing
    # columns similar to these:
    #keys_to_record = ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE',
    #                  'ERRX2WIN_IMAGE', 'ERRY2WIN_IMAGE', 'ERRXYWIN_IMAGE', 
    #                  'ELONGATION', 'ALPHAWIN_J2000', 'DELTAWIN_J2000',
    #                  'FLAGS', 'IMAFLAGS_ISO', 'FWHM_IMAGE', 'CLASS_STAR',    
    #                  'FLUX_MAX', 'FLUX_PSF', 'FLUXERR_PSF']
    # e.g.: x, y, errx, erry, alphawin_j2000, deltawin_j2000, flags,
    #       peak significance and error, peak Fpsf and error, peak Fpsferr and error,
    #       PSF flux and error from D, chi2 (should be similar for all fits; if not,
    #       record separate chi2 for each fit).

    # prepare ds9 region file using function [prep_ds9regions]
    if get_par(C.make_plots,tel):
        result = prep_ds9regions(base_newref+'_ds9regions.txt',
                                 x_array[mask_keep], y_array[mask_keep], 
                                 radius=5., width=2, color=color_ds9)

    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_trans', log=log)
        
    return ntrans


################################################################################

def prep_ds9regions(filename, x, y, radius=5, width=2, color='green'):

    """Function that creates a text file with the name [filename] that
    can be used to mark objects at an array of pixel positions (x,y)
    with a circle when displaying an image with ds9."""

    # prepare ds9 region file
    f = open(filename, 'w')
    ncoords = len(x)
    for i in range(ncoords):
        f.write('circle({},{},{}) # color={} width={} text={{{}}} font="times 7"\n'.
                format(x[i], y[i], radius, color, width, i))
    f.close()
    return
    

################################################################################

def trans_measure(I, x_index, y_index, var_bkg=0.):

    """Function that determines a number of positional quantities
    (adopting the definitions as described in the SExtractor manual)
    based on the input intensity array [I] and the corresponding
    indices of the x,y coordinates [x_index] and [y_index]. These
    three input arrays should have the same shape. [var_bkg]
    determines the local background noise, which is added to [I] to
    estimate the flux uncertainty used in the position uncertainty
    calculations.

    """
    
    assert I.shape == x_index.shape
    assert x_index.shape == y_index.shape

    # determine position (in pixel coordinates) weighted by I
    I_sum = np.sum(I)
    X = np.sum(I * x_index) / I_sum
    Y = np.sum(I * y_index) / I_sum

    # 2nd order moments: X2, Y2, XY
    X2 = np.sum(I * x_index**2) / I_sum - X**2
    Y2 = np.sum(I * y_index**2) / I_sum - Y**2
    XY = np.sum(I * x_index * y_index) / I_sum - X*Y
    
    # Position errors: ERRX2, ERRY2, ERRXY
    I_var = I + var_bkg
    I_sum2 = I_sum**2
    ERRX2 = np.sum(I_var * (x_index - X)**2) / I_sum2
    ERRY2 = np.sum(I_var * (y_index - Y)**2) / I_sum2
    ERRXY = np.sum(I_var * (x_index - X)*(y_index - Y)) / I_sum2

    # use function [get_shape_parameters] to get A, B, THETA and their
    # errors
    A, B, THETA, ERRA, ERRB, ERRTHETA = get_shape_parameters (X2, Y2, XY, ERRX2, ERRY2, ERRXY)

    return X, Y, X2, Y2, XY, ERRX2, ERRY2, ERRXY, A, B, THETA, ERRA, ERRB, ERRTHETA


################################################################################

def get_shape_parameters (x2, y2, xy, errx2, erry2, errxy):

    """Function to calculate a, b, theta, erra, errb, errtheta from x2,
    y2, xy, errx2, erry2, errxy (see SExtractor manual). """

    # Eq. 24 from SExtractor manual:
    term1 = (x2 + y2) / 2
    term2 = np.sqrt(0.25*(x2-y2)**2 + xy**2)
    a = np.sqrt(term1 + term2)
    # Eq. 25 from SExtractor manual:
    if term1 >= term2:
        b = np.sqrt(term1 - term2)
    else:
        b = 0.
    # Eq. 21 from SExtractor manual:
    theta = 0.5*np.arctan2( 2*xy, x2-y2 )
    theta *= 180 / np.pi

    #elongation = a / b
    #ellipticity = 1 - (b / a)

    # Eq. 36 from SExtractor manual:
    term1 = (errx2 + erry2) / 2
    term2 = np.sqrt(0.25*(errx2-erry2)**2 + errxy**2)
    erra = np.sqrt(term1 + term2)
    # Eq. 37 from SExtractor manual:
    if term1 >= term2:
        errb = np.sqrt(term1 - term2)
    else:
        errb = 0.
    # Eq. 38 from SExtractor manual:
    errtheta = 0.5*np.arctan2( 2*errxy, errx2-erry2 )
    errtheta *= 180 / np.pi

    # don't need the calculations below for now
    if False:
        # Ellipse parameters: CXX, CYY, CXY
        denominator = np.sqrt(0.25*(x2-y2)**2 + xy**2)
        # Eq. 27 from SExtractor manual:
        cxx = y2 / denominator
        # Eq. 28 from SExtractor manual:
        cyy = x2 / denominator
        # Eq. 29 from SExtractor manual:
        cxy = -2*xy / denominator
        
        # and their errors
        denominator = np.sqrt(0.25*(errx2-erry2)**2 + errxy**2)
        # Eq. 39 from SExtractor manual:
        cxx = erry2 / denominator
        # Eq. 40 from SExtractor manual:
        cyy = errx2 / denominator
        # Eq. 41 from SExtractor manual:
        cxy = -2*errxy / denominator

    return a, b, theta, erra, errb, errtheta


################################################################################

def get_psfoptflux_xycoords (psfex_bintable, D, S, D_mask, RON, xcoords, ycoords,
                             dx2=0, dy2=0, dxy=0, satlevel=50000, replace_satdata=False,
                             psf_oddsized=True, psffit=False, get_limflux=False,
                             limflux_nsigma=5., log=None):

    """Function that returns the optimal flux and its error (using the
       function [flux_optimal] of a source at pixel positions
       [xcoords], [ycoords] given the inputs: .psf file produced by
       PSFex [psfex_bintable], data [D], sky [S] and read-out noise
       [RON]. [D], [S] and [RON] are assumed to be in electrons.
    
       [D] is a 2D array meant to be the full image. [S] can be a 2D
       array with the same shape as [D] or a scalar. [xcoords] and
       [ycoords] are arrays, and the output flux and its error will be
       arrays as well.
    
       If [replace_satdata]=True, the function will update [D] where
       any saturated values in the PSF footprint of the
       [xcoords],[ycoords] coordinates that are being processed is
       replaced by the expected flux according to the PSF.

       If [get_limflux]=True, the function will return the limiting
       flux (at significance [limflux_nsigma]) at the input
       coordinates using the function [flux_optimal_s2n] in
       [flux_opt]; [fluxerr_opt] will contain zeros in that case, and
       PSF fitting is not performed even if [psffit]=True.

    """
        
    log.info('Executing get_psfoptflux_xycoords ...')
    if get_par(C.timing,tel): t = time.time()

    # make sure x and y have same length
    if np.isscalar(xcoords) or np.isscalar(ycoords):
        log.error('xcoords and ycoords should be arrays')
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
        
    # get dimensions of D
    ysize, xsize = np.shape(D)

    # read in PSF output binary table from psfex
    data, header = read_hdulist(psfex_bintable, ext_data=1, ext_header=1)
    data = data[0][0][:]
    #with fits.open(psfex_bintable) as hdulist:
    #    header = hdulist[1].header
    #    data = hdulist[1].data[0][0][:]

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
    if get_par(C.verbose,tel):
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
    psf_hsize = int(psf_size/2)

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
            #print ('Position x,y='+str(xpos)+','+str(ypos)+' outside image - skipping')
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
        index = tuple([slice(y1,y2),slice(x1,x2)])

        # extract subsection from D, S and D_mask
        D_sub = D[index]
        D_mask_sub = D_mask[index]
        if np.isscalar(S):
            S_sub = S
        else:
            S_sub = S[index]

        # get P_shift and P_noshift
        x = (int(xcoords[i]) - polzero1) / polscal1
        y = (int(ycoords[i]) - polzero2) / polscal2

        if ncoords==1 or get_par(C.use_single_psf,tel):
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = (data[0] + data[1] * x + data[2] * x**2 + 
                                  data[3] * y + data[4] * x * y + data[5] * y**2)
            elif poldeg==3:
                psf_ima_config = (data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 +
                                  data[4] * y + data[5] * x * y + data[6] * x**2 * y +
                                  data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3)

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
        order = 2
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
        psf_shift = clean_norm_psf(psf_ima_shift_resized, get_par(C.psf_clean_factor,tel))
        # also return normalized PSF without any shift
        # only required if psf-fitting is performed
        if psffit:
            psf_noshift = clean_norm_psf(psf_ima_resized, get_par(C.psf_clean_factor,tel))

        # extract subsection from psf_shift and psf_noshift
        y1_P = y1 - (ypos - psf_hsize)
        x1_P = x1 - (xpos - psf_hsize)
        y2_P = y2 - (ypos - psf_hsize)
        x2_P = x2 - (xpos - psf_hsize)
        index_P = tuple([slice(y1_P,y2_P),slice(x1_P,x2_P)])
        
        P_shift = psf_shift[index_P]
        # only required if psf-fitting is performed
        if psffit:
            P_noshift = psf_noshift[index_P]
        
        if get_limflux:
            # determine limiting flux at this position using flux_optimal_s2n
            flux_opt[i] = flux_optimal_s2n (P_shift, S_sub, RON, limflux_nsigma, fwhm=psf_fwhm)

        else:

            # use only those pixels that are not affected
            # by any bad pixels, cosmic rays, saturation, etc.
            mask_use = (D_mask_sub==0)
            
            # call flux_optimal
            flux_opt[i], fluxerr_opt[i] = flux_optimal (P_shift, D_sub, S_sub, RON,
                                                        mask_use=mask_use,
                                                        dx2=dx2[i], dy2=dy2[i], dxy=dxy[i], log=log)
        
            # if psffit=True, perform PSF fitting
            if psffit:
                flux_psf[i], fluxerr_psf[i], xshift_psf[i], yshift_psf[i], chi2_psf[i] = (
                    flux_psffit (P_noshift, D_sub, S_sub, RON, flux_opt[i], xshift, yshift,
                                 mask_use=mask_use, log=log)
                )

                # xshift_psf and yshift_psf are the shifts with respect to
                # the integer xpos and ypos positions defined at the top
                # of this loop over the sources. This is because the fit
                # is using the PSF image that is not shifted
                # (P_noshift) to the exact source position.  Redefine
                # them here with respect to the fractional coordinates
                # xcoords and ycoords
                xshift_psf[i] += xshift
                yshift_psf[i] += yshift
                        
                #print ('i, flux_opt[i], fluxerr_opt[i], flux_psf[i], fluxerr_psf[i]',
                #       i, flux_opt[i], fluxerr_opt[i], flux_psf[i], fluxerr_psf[i])

            if replace_satdata:        
                # first determine mask of saturated pixels:
                mask_sat = ((D_mask_sub==get_par(C.mask_value['saturated'],tel)) |
                            (D_mask_sub==get_par(C.mask_value['saturated-connected'],tel)))
                # if any pixels close to the center of the object are
                # saturated, replace them
                mask_inner = (P_shift >= 0.25*np.amax(P_shift))

                if np.any(mask_sat[mask_inner]):

                    # replace all saturated values
                    D[index][mask_sat] = P_shift[mask_sat] * flux_opt[i] + S_sub[mask_sat]

                    # and put through [flux_optimal] once more without a
                    # saturated pixel mask
                    #flux_opt[i], fluxerr_opt[i] = flux_optimal (P_shift, D_sub,
                    #                                                      S_sub, RON,
                    #                                                      dx2=dx2[i], dy2=dy2[i], dxy=dxy[i])
                    #D[index][mask_use] = P_shift[mask_use] * flux_opt[i] + S_sub[mask_use]

                    if get_par(C.display,tel):
                        result = ds9_arrays(D_sub=D_sub, mask_nonsat=~mask_sat.astype(int), 
                                            maskopt=mask_use.astype(int), S_sub=S_sub, P=P_shift,
                                            D_replaced=D[index])

    if get_par(C.timing,tel): t1 = time.time()
    pool = ThreadPool(nthreads)
    pool.map(loop_psfoptflux_xycoords, range(ncoords), chunksize=1000)
    pool.close()
    pool.join()
    if get_par(C.verbose,tel): log.info('ncoords: {}'.format(ncoords))

    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_psfoptflux_xycoords', log=log)

    if psffit:
        x_psf = xcoords + xshift_psf
        y_psf = ycoords + yshift_psf
        return flux_opt, fluxerr_opt, flux_psf, fluxerr_psf, x_psf, y_psf
    else:
        return flux_opt, fluxerr_opt
            

################################################################################

def flux_psffit(P, D, S, RON, flux_opt, xshift, yshift, mask_use=None, log=None):

    # if S is a scalar, expand it to 2D array
    if np.isscalar(S):
        S = S * np.ones(P.shape, dtype='float32')
        
    # replace negative values in D with the sky
    #D[D<0] = S[D<0]

    # define objective function: returns the array to be minimized
    def fcn2min(params, P, D, S, RON, mask_use):

        xshift = params['xshift'].value
        yshift = params['yshift'].value
        flux_psf = params['flux_psf'].value

        # shift the PSF image to the exact pixel position
        P_shift = ndimage.shift(P, (yshift, xshift))
        # alternatively, use Eran's shift function 
        #P_shift = image_shift_fft(P, xshift, yshift)
                
        #print ('sum of P, P_shift:', np.sum(P), np.sum(P_shift))
    
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
        
        #print ('xshift, yshift, flux_psf, chi2', xshift, yshift, flux_psf, np.sum(resid))
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
    if P.ndim!=1:
        P = P.ravel()
        D = D.ravel()
        V = V.ravel()
        if not np.isscalar(S):
            S = S.ravel()

    # and optimal flux and its error
    P_over_V = P/V
    denominator = np.dot(P, P_over_V)
    if denominator!=0:
        optflux = np.dot(P_over_V, (D-S)) / denominator
        optfluxerr = 1./np.sqrt(denominator)
    else:
        optflux = 0.
        optfluxerr = 0.
        
    # previously the optimal flux and error were calculated as follows
    # (a bit slower than above):
    #denominator = np.sum(P**2/V)
    #optflux = np.sum((P*(D-S)/V)) / denominator
    #optfluxerr = 1./np.sqrt(denominator)
    
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

def get_s2n_ZO (P, D, S, V):

    """Function that calculates signal-to-noise ratio using Eq. 51 from
    Zackay & Ofek 2017, ApJ, 836, 187.  All inputs are assumed to be
    in electrons rather than counts. These can be 1- or 2-dimensional
    lists, while the sky is also allowed to be a scalar. 

    """

    T0 = np.sum(D-S)
    #s2n = np.sqrt(np.sum( (D-S)**2 / V ))
    s2n = np.sqrt(np.sum( (T0*P)**2 / V ))

    return s2n

################################################################################

def flux_optimal (P, D, S, RON, nsigma_inner=10, P_noshift=None,
                  nsigma_outer=5, max_iters=10, epsilon=1e-3, mask_use=None,
                  add_V_ast=False, dx2=0, dy2=0, dxy=0, log=None):
    
    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], data [D], sky [S] and read-out noise [RON].  This
    makes use of function [get_optflux] or [get_optflux_Eran]. """

    # if S is a scalar, expand it to 2D array
    if np.isscalar(S):
        S = S * np.ones(P.shape, dtype='float32')

    if add_V_ast:
        # calculate astrometric variance
        dDdy = D - np.roll(D,1,axis=0)
        dDdx = D - np.roll(D,1,axis=1)
        dDdxy = D - np.roll(D,1,axis=(0,1))
        V_ast = np.abs(dx2) * dDdx**2 + np.abs(dy2) * dDdy**2 + np.abs(dxy) * dDdxy**2
        
    # if input mask [mask_use] was not provided, create it with same
    # shape as D with all elements set to True.
    if mask_use is None: 
        mask_use = np.ones(D.shape, dtype=bool)
    # do not use any negative pixel values in D
    mask_use[D<0] = False
    
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
                    
        #print ('i, flux_opt, fluxerr_opt', i, flux_opt, fluxerr_opt,
        #       abs(flux_opt_old-flux_opt)/flux_opt, abs(flux_opt_old-flux_opt)/fluxerr_opt)
        if fluxerr_opt==0.:
            break

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

    return flux_opt, fluxerr_opt
    

################################################################################

def flux_optimal_s2n (P, S, RON, s2n, fwhm=5., max_iters=10, epsilon=1e-6):
    
    """Similar to function [flux_optimal] above, but this function returns
    the total flux required for the point source to have a particular
    signal-to-noise ratio [s2n], given the PSF image [P], the sky
    background [S] (can be image or scalar), and the read-out noise
    [RON]. This function is used to estimate the flux of the fake
    stars that are being added to the image with a required S/N
    [C.fakestar_s2n].

    Note that the image itself can be provided as the sky background
    to calculate the flux required to reach the required S/N with
    respect to the image.

    """

    for i in range(max_iters):
        if i==0:
            # initial estimate of variance (scalar)
            V = RON**2 + S
            # and flux (see Eq. 13 of Naylor 1998)
            flux_opt = s2n * fwhm * np.sqrt(np.median(V)) / np.sqrt(2*np.log(2)/np.pi)
        else:
            # estimate new flux based on fluxerr_opt of previous iteration
            flux_opt = s2n * fluxerr_opt 
            # improved estimate of variance (2D list)
            V = RON**2 + S + flux_opt * P

        # new estimate of D
        D = S + flux_opt * P

        # get optimal flux
        flux_opt, fluxerr_opt = get_optflux(P, D, S, V)

        # break out of loop if S/N sufficiently close
        if abs(flux_opt/fluxerr_opt - s2n) / s2n < epsilon:
            break
        
    return flux_opt
    

################################################################################

def clipped_stats(array, nsigma=3, max_iters=10, epsilon=1e-6, clip_upper_frac=0,
                  clip_zeros=True, get_median=True, get_mode=False, mode_binsize=0.1,
                  verbose=False, make_hist=False, name_hist=None, hist_xlabel=None,
                  log=None):

    if verbose and get_par(C.timing,tel) and log is not None:
        log.info('Executing clipped_stats ...')
        t = time.time()

    # remove zeros
    if clip_zeros:
        array = array[array.nonzero()]
        
    if clip_upper_frac != 0:
        index_upper = int((1.-clip_upper_frac)*array.size+0.5)
        array = np.sort(array.flatten(), kind='quicksort')[:index_upper]

    mean_old = float('inf')
    for i in range(max_iters):
        mean = array.mean()
        std = array.std()
        if abs(mean_old-mean)/abs(mean) < epsilon:
            break
        mean_old = mean
        index = ((array>(mean-nsigma*std)) & (array<(mean+nsigma*std)))
        array = array[index]

    # add median
    if get_median:
        median = np.median(array)
        if abs(median-mean)/mean>0.1 and log is not None:
            log.info('Warning: mean and median in clipped_stats differ by more than 10%')
            log.info('mean: {:.3f}, median: {:.3f}'.format(mean, median))
            
    # and mode
    if get_mode:
        bins = np.arange(mean-nsigma*std, mean+nsigma*std, mode_binsize)
        hist, bin_edges = np.histogram(array, bins)
        index = np.argmax(hist)
        mode = (bins[index]+bins[index+1])/2.
        if abs(mode-mean)/mean>0.1 and log is not None:
            log.info('Warning: mean and mode in clipped_stats differ by more than 10%')

    if make_hist:
        bins = np.linspace(mean-nsigma*std, mean+nsigma*std)
        plt.hist(np.ravel(array), bins, color='tab:blue')
        x1,x2,y1,y2 = plt.axis()
        plt.plot([mean, mean], [y2,y1], color='black')
        plt.plot([mean+std, mean+std], [y2,y1], color='black', linestyle='--')
        plt.plot([mean-std, mean-std], [y2,y1], color='black', linestyle='--')
        title = 'mean (black line): {:.3f}, std: {:.3f}'.format(mean, std)
        if get_median:
            plt.plot([median, median], [y2,y1], color='tab:orange')
            title += ', median (orange line): {:.3f}'.format(median)
        if get_mode:
            plt.plot([mode, mode], [y2,y1], color='tab:red')
            title += ', mode (red line): {:.3f}'.format(mode)
        plt.title(title)
        if hist_xlabel is not None:
            plt.xlabel(hist_xlabel)
        plt.ylabel('number') 
        if get_par(C.make_plots,tel):
            if name_hist is None: name_hist = 'clipped_stats_hist.pdf'
            plt.savefig(name_hist)
        if get_par(C.show_plots,tel): plt.show()
        plt.close()

    if verbose and get_par(C.timing,tel) and log is not None:
        log_timing_memory (t0=t, label='clipped_stats', log=log)
        
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

    # list with values to return
    values = []
    # loop keywords
    for key in keywords:
        # use function [get_keyvalue] (see below) to return the value
        # from either the variable defined in Constants module, or
        # from the fits header using the keyword name defined in the
        # Constants module
        values.append(get_keyvalue(key, header, log))

    if len(values)==1:
        return values[0]
    else:
        return values


################################################################################

def get_keyvalue (key, header, log):
    
    # check if [key] is defined in Constants module
    var = 'C.'+key
    try:
        value = eval(var)
    except:
        # if it does not work, try using the value of the keyword name
        # (defined in Constants module) from the fits header instead
        try:
            key_name = eval('C.key_'+key)
        except:
            log.critical('either [{}] or [{}] needs to be defined in [settings_file]'.
                      format(key, 'key_'+key))
            if log:
                return 'critical', key+' not defined in setting file.'
            else:
                raise SystemExit
        else:
            if key_name in header:
                value = header[key_name]
            else:
                log.critical('keyword {} not present in header'.format(key_name))
                if log:
                    return 'critical', key_name+' not in header.'
                else:
                    raise SystemExit

    if get_par(C.verbose,tel):
        log.info('keyword: {}, adopted value: {}'.format(key, value))
            
    return value


################################################################################

def get_remap_name(new_name, ref_name, remap_name): 

    # in case full paths are provided for the input images and the new
    # and ref directories are different, the remapped reference images
    # should end up in the new directory

    def get_dir_name (name):
        dir_name = '.'
        if '/' in name:
            dir_name = '/'.join(name.split('/')[:-1])
        return dir_name

    new_dir = get_dir_name(new_name)
    ref_dir = get_dir_name(ref_name)
    if new_dir != ref_dir:
        remap_name = '{}/{}'.format(new_dir, remap_name.split('/')[-1])

    return remap_name
        
            
################################################################################

def prep_optimal_subtraction(input_fits, nsubs, imtype, fwhm, header, log,
                             fits_mask=None, ref_fits_remap=None, data_cal=None):

    log.info('Executing prep_optimal_subtraction ...')
    t = time.time()
       
    if imtype=='new':
        base = base_new
    else:
        base = base_ref

    # read in input_fits header
    data_wcs, header_wcs = read_hdulist (input_fits, ext_data=0, ext_header=0,
                                         dtype='float32')

    # get gain, readnoise, pixscale and saturation level from header
    keywords = ['gain', 'ron', 'pixscale', 'satlevel']
    gain, readnoise, pixscale, satlevel = read_header(header, keywords, log)
    ysize, xsize = np.shape(data_wcs)

    # read in background image
    fits_bkg = base+'_bkg.fits'
    if os.path.exists(fits_bkg):
        data_bkg = read_hdulist (fits_bkg, ext_data=0, dtype='float32')
    else:
        # if it does not exist, create it from the background mesh
        fits_bkg_mini = base+'_bkg_mini.fits'
        data_bkg_mini = read_hdulist (fits_bkg_mini, ext_data=0, dtype='float32')        
        data_bkg = mini2back (data_bkg_mini, data_wcs.shape, log,
                              order_interp=2, bkg_boxsize=get_par(C.bkg_boxsize,tel))

    # same for background std image
    fits_bkg_std = base+'_bkg_std.fits'
    if os.path.exists(fits_bkg_std):
        data_bkg_std = read_hdulist (fits_bkg_std, ext_data=0, dtype='float32')
    else:
        # if it does not exist, create it from the background mesh
        fits_bkg_std_mini = base+'_bkg_std_mini.fits'
        data_bkg_std_mini = read_hdulist (fits_bkg_std_mini, ext_data=0, dtype='float32')        
        data_bkg_std = mini2back (data_bkg_std_mini, data_wcs.shape, log,
                                  order_interp=1, bkg_boxsize=get_par(C.bkg_boxsize,tel))

    # function to create a minimal mask of saturated pixels and the
    # adjacent pixels from input data, in case mask image is not
    # provided
    def create_mask (data, satlevel):
        # saturated pixels
        data_mask = np.zeros(data.shape, dtype='uint8')
        mask_sat = (data >= 0.8*satlevel)
        data_mask[mask_sat] += get_par(C.mask_value['saturated'],tel)
        # pixels connected to saturated pixels
        mask_sat_adj = ndimage.binary_dilation(mask_sat, structure=np.ones((3,3)).astype('bool'))
        mask_sat_adj[mask_sat] = False
        data_mask[mask_sat_adj] += get_par(C.mask_value['saturated-connected'],tel)
        return data_mask
    
    # and read in mask image
    if fits_mask is not None:
        data_mask = read_hdulist (fits_mask, ext_data=0, dtype='uint8')
    else:
        # if mask image is not provided, use function [create_mask] to
        # create a minimal mask
        data_mask = create_mask(data_wcs, satlevel)

    # if remapped image is provided, read that also
    if ref_fits_remap is not None:
        data_ref_remap = read_hdulist (ref_fits_remap, ext_data=0, dtype='float32')

        # the reference background maps and mask need to be projected
        # to the coordinate frame of the new or remapped reference
        # image. At the moment this is done with swarp, but this is a
        # very slow solution. Tried to improve this using functions
        # [xy_index_ref] and [get_data_remap] (see older zogy
        # versions), but these fail when there is rotation between the
        # images, resulting in rotated masks.

        use_swarp = True
        if use_swarp:

            # local function to help with remapping of the background
            # maps and mask
            def run_swarp (fits2remap, data2remap, header2remap=header_wcs,
                           fits2remap2=base_new+'.fits'):

                # update headers of fits image with that of the
                # original wcs-corrected reference image
                fits.writeto(fits2remap, data2remap, header2remap, overwrite=True)
                # project fits image to new image
                fits_out = fits2remap.replace('.fits', '_remap.fits')
                fits_out = get_remap_name(fits2remap2, fits2remap, fits_out)
                if not os.path.isfile(fits_out) or get_par(C.redo,tel):
                    result = run_remap(fits2remap2, fits2remap, fits_out,
                                       [ysize, xsize], gain=gain, log=log,
                                       config=get_par(C.swarp_cfg,tel),
                                       resampling_type='NEAREST',
                                       dtype=data2remap.dtype.name)
                data_remapped = read_hdulist (fits_out, ext_data=0)
                return data_remapped
                
            # remap reference image background
            data_ref_bkg_remap = run_swarp(fits_bkg, data_bkg)
            # remap reference image background std
            data_ref_bkg_std_remap = run_swarp(fits_bkg_std, data_bkg_std)
            # remap reference mask image if it exists
            if fits_mask is not None:
                # SWarp turns integer mask into float during processing,
                # so need to add 0.5 and convert to integer again
                data_ref_remap_mask = (run_swarp(fits_mask, data_mask)+0.5).astype('uint8')

        # if mask image is not provided, use above function [create_mask] to
        # create a minimal mask from the remapped reference data
        if fits_mask is None:
            data_ref_remap_mask = create_mask(data_ref_remap, satlevel)
        
    # convert counts to electrons
    satlevel *= gain
    data_wcs *= gain
    data_bkg *= gain
    data_bkg_std *= gain
    # fix pixels using function [fixpix]
    if imtype=='new':
        data_wcs = fixpix (data_wcs, data_bkg, log, satlevel=satlevel, data_mask=data_mask,
                           fwhm=fwhm, base=base)

    if ref_fits_remap is not None:
        data_ref_remap *= gain
        data_ref_bkg_remap *= gain
        data_ref_bkg_std_remap *= gain
        # fix pixels using function [fixpix] also in remapped reference image
        data_ref_remap = fixpix (data_ref_remap, data_ref_bkg_remap, log, satlevel=satlevel,
                                 data_mask=data_ref_remap_mask, fwhm=fwhm, base=base)

    # print warning if any pixel value is not finite
    # if np.any(~np.isfinite(data_wcs)):
    #    log.info('Warning: not all pixel values are finite')
    #    log.info('         replacing NANs with zeros and +-inf with large +-numbers')        
    #    # replace NANs with zero, and +-infinity with large +-numbers
    #    data_wcs = np.nan_to_num(data_wcs)

    # add header keyword(s) regarding background
    # pre-fixed with S as background is produced in SExtractor module
    bkg_mean, bkg_std, bkg_median = clipped_stats(data_bkg, nsigma=10., log=log)
    header['S-BKG'] = (bkg_median, '[e-] median background full image')
    header['S-BKGSTD'] = (bkg_std, '[e-] sigma (STD) background full image')

    # determine psf of input image with get_psf function - needs to be
    # done before optimal fluxes are determined
    psf, psf_orig = get_psf(input_fits, header, nsubs, imtype, fwhm, pixscale, log)

    # -------------------------------
    # determination of optimal fluxes
    # -------------------------------

    # Get estimate of optimal flux for all sources in the new and ref
    # image if not already done so.

    # [mypsffit] determines if PSF-fitting part is also performed;
    # this is different from SExtractor PSF-fitting
    mypsffit = False

    # first read SExtractor fits table
    sexcat = base+'_cat.fits'
    data_sex = read_hdulist (sexcat, ext_data=1)
    
    if 'FLUX_OPT' not in data_sex.dtype.names or get_par(C.redo,tel):
        
        if get_par(C.timing,tel): t1 = time.time()
        log.info('deriving optimal fluxes ...')
    
        # read in positions and their errors
        xwin = data_sex['XWIN_IMAGE']
        ywin = data_sex['YWIN_IMAGE']
        errx2win = data_sex['ERRX2WIN_IMAGE']
        erry2win = data_sex['ERRY2WIN_IMAGE']
        errxywin = data_sex['ERRXYWIN_IMAGE']

        psfex_bintable = base+'_psf.fits'

        if mypsffit:
            flux_opt, fluxerr_opt, flux_psf, fluxerr_psf, x_psf, y_psf = (
                get_psfoptflux_xycoords (psfex_bintable, data_wcs, data_bkg, data_mask,
                                         readnoise, xwin, ywin, errx2win, erry2win, errxywin,
                                         satlevel=satlevel, replace_satdata=False, psffit=mypsffit, log=log)
            )
        else:
            flux_opt, fluxerr_opt = (
                get_psfoptflux_xycoords (psfex_bintable, data_wcs, data_bkg, data_mask,
                                         readnoise, xwin, ywin, errx2win, erry2win, errxywin,
                                         satlevel=satlevel, replace_satdata=False, log=log)
            )
            
        # determine 5-sigma limiting flux using
        # [get_psfoptflux_xycoords] with [get_limflux]=True for random
        # coordinates across the field
        nlimits = 101
        edge = 100
        xlim = np.random.rand(nlimits)*(xsize-2*edge) + edge
        ylim = np.random.rand(nlimits)*(ysize-2*edge) + edge
        def calc_limflux (nsigma):
            # '__' is to disregard the 2nd output array from [get_psfoptflux_xycoords]
            limflux_array, __ = get_psfoptflux_xycoords (psfex_bintable, data_wcs, data_bkg, data_mask,
                                                         readnoise, xlim, ylim, satlevel=satlevel,
                                                         get_limflux=True, limflux_nsigma=nsigma, log=log)
            limflux_mean, limflux_std, limflux_median = clipped_stats(limflux_array, log=log)
            if get_par(C.verbose,tel):
                log.info('{}-sigma limiting flux; mean: {}, std: {}, median: {}'
                         .format(nsigma, limflux_mean, limflux_std, limflux_median))
            return limflux_median

        limflux_3sigma = calc_limflux (3.)
        limflux_5sigma = calc_limflux (5.)

        # add header keyword(s):
        mask_neg = (flux_opt < 0.)
        header['NOBJ-NEG'] = (np.sum(mask_neg), 'number of objects with negative optimal flux')
        header['LIMFLUX3'] = (limflux_3sigma, '[e-] full-frame 3-sigma limiting flux')
        header['LIMFLUX5'] = (limflux_5sigma, '[e-] full-frame 5-sigma limiting flux')
        
        if get_par(C.timing,tel):
            log_timing_memory (t0=t1, label='deriving optimal fluxes', log=log)

        if get_par(C.timing,tel): t2 = time.time()

        # read a few extra header keywords needed in [get_zp] and [apply_zp]
        keywords = ['exptime', 'filter', 'obsdate']
        exptime, filt, obsdate = read_header(header, keywords, log)
        if tel=='LCO': filt=filt[0]
        if get_par(C.verbose,tel):
            log.info('exptime: {}, filter: {}, obsdate: {}'.format(exptime, filt, obsdate))

        # get airmasses for SExtractor catalog sources
        ra_sex = data_sex['ALPHAWIN_J2000']
        dec_sex = data_sex['DELTAWIN_J2000']
        flags_sex = data_sex['FLAGS']

        lat = get_par(C.obs_lat,tel)
        lon = get_par(C.obs_long,tel)
        height = get_par(C.obs_height,tel)
        airmass_sex = get_airmass(ra_sex, dec_sex, obsdate, 
                                  lat, lon, height, log=log)
        airmass_sex_median = float(np.median(airmass_sex))
        log.info('median airmass: {}'.format(airmass_sex_median))
        
        # use WCS solution in input [header] to get RA, DEC of central pixel
        wcs = WCS(header)
        ra_center, dec_center = wcs.all_pix2world(xsize/2, ysize/2, 1)
        log.info('ra_center: {}, dec_center: {}'.format(ra_center, dec_center))

        # determine airmass at image center
        airmass_center = get_airmass(ra_center, dec_center, obsdate,
                                     lat, lon, height, log=log)
        header['AIRMASSC'] = (float(airmass_center), 'airmass at image center')

        # determine image zeropoint if ML/BG calibration catalog exists
        ncalstars=0
        if os.path.isfile(get_par(C.cal_cat,tel)):
            
            # add header keyword(s):
            cal_name = get_par(C.cal_cat,tel).split('/')[-1]
            header['PC-CAT-F'] = (cal_name, 'photometric catalog')
            #caldate = time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(C.cal_cat)))

            # Only execute the following block if input [data_cal] is
            # not defined; if [C.cal_cat] exists, [data_cal] should
            # have been already produced by [run_wcs] so that it can
            # be re-used here.
            if data_cal is None:
                # determine cal_cat fits extensions to read using
                # [get_ext_list] (each 1 degree strip in declination is
                # recorded in its own extension in the calibration catalog)
                fov_half_deg = np.amax([xsize, ysize]) * pixscale / 3600. / 2
                ext_list = get_ext_list (dec_center, fov_half_deg, zone_size=60.)
                #print ('dec_center: {}, ext_list: {}'.format(dec_center, ext_list))

                # read calibration catalog
                data_cal = read_hdulist (get_par(C.cal_cat,tel), ext_data=ext_list)

                # use function [find_stars] to select stars in calibration
                # catalog that are within the current field-of-view
                mask_field = find_stars(data_cal['ra'], data_cal['dec'], ra_center, dec_center,
                                        fov_half_deg, log)
                index_field = np.where(mask_field)[0]
                data_cal = data_cal[index_field]

            ncalstars = np.shape(data_cal)[0]
            log.info('number of potential photometric calibration stars in FOV: {}'.format(ncalstars))
            header['PC-TNCAL'] = (ncalstars, 'total number of photcal stars in FOV')

            # test: limit calibration catalog entries
            if 'chi2' in data_cal.dtype.names:
                mask_cal = (data_cal['chi2'] <= 10.)
                #data_cal = data_cal[:][mask_cal]
                data_cal = data_cal[mask_cal]

            ncalstars = np.shape(data_cal)[0]
            log.info('number of phot.cal. stars in FOV after chi2 cut: {}'.format(ncalstars))

            # requirements on presence of survey input filters for the
            # calibration of a ML/BG filter:
            #u: Gaia and (GALEX NUV or SM u or SM v or SDSS u)
            #g: Gaia and (GALEX NUV or SM u or SM v or SDSS u or SDSS g)
            #q: Gaia and (GALEX NUV or SM u or SM v or SDSS u or SDSS g or PS1 g or SM g)
            #r: Gaia: 0
            #i: Gaia and (2MASS J or PS1 z or PS1 y or SDSS z or SM z): 0
            #z: Gaia and (2MASS J or PS1 y): 0

            # prepare [filt_req] dictionary
            filt_req = {}
            # these filters are required for all filters, boolean AND
            filt_req['all'] = ['Gaia2r_G', 'Gaia2r_Gbp', 'Gaia2r_Grp', '2MASS_J']
            # these filters are required for specific filters, boolean OR
            filt_req['u'] = ['GALEX_NUV', 'SM_u', 'SM_v', 'SDSS_u']
            filt_req['g'] = ['GALEX_NUV', 'SM_u', 'SM_v', 'SDSS_u', 'SDSS_g']
            filt_req['q'] = ['GALEX_NUV', 'SM_u', 'SM_v', 'SDSS_u', 'SDSS_g', 'PS1_g', 'SM_g']
            filt_req['i'] = ['PS1_z', 'PS1_y', 'SDSS_z', 'SM_z']
            filt_req['z'] = ['PS1_y']

            # [mask_cal] is mask for entries in [data_cal] where all
            # filters listed in [filt_req['all']] are present (True)
            mask_cal = np.all([data_cal[col] for col in filt_req['all']], axis=0)
            data_cal = data_cal[mask_cal]
            
            # loop the the filter keys of [filt_req]
            if filt in filt_req.keys():
                # [mask_cal_filt] is mask for for entries in the
                # updated [data_cal] for which all filters in
                # [filt_req[current filter]] are present
                mask_cal_filt = np.any([data_cal[col] for col in filt_req[filt]], axis=0)
                # if less than [C.phot_ncal_min] stars left, drop filter requirements and hope for the best!
                if np.sum(mask_cal_filt) >= get_par(C.phot_ncal_min,tel):
                    data_cal = data_cal[mask_cal_filt]
                else:
                    log.info('Warning: less than {} calibration stars with default filter requirements'.
                             format(get_par(C.phot_ncal_min,tel)))
                    log.info('filter: {}, requirements (any one of these): {}'.format(filt, filt_req[filt]))
                    log.info('dropping this specific requirement and hoping for the best')
                    
            # This is the number of photometric calibration stars
            # after the chi2 and filter requirements cut.
            ncalstars = np.shape(data_cal)[0]
            log.info('number of photometric stars in FOV after filter cut: {}'.format(ncalstars))
            header['PC-FNCAL'] = (ncalstars, 'number of photcal stars after filter cut')
            
            # pick only main sequence stars
            if False:
                if 'spectype' in data_cal.dtype.names:
                    mask_cal = ['V' in data_cal['spectype'][i] and
                                'IV' not in data_cal['spectype'][i] and
                                'M' not in data_cal['spectype'][i]
                                for i in range(np.shape(data_cal)[0])]
                    #data_cal = data_cal[:][mask_cal]
                    data_cal = data_cal[mask_cal]
                

            # only continue if calibration stars are present in the FOV
            # and filter is present in calibration catalog
            if ncalstars>0:
                ra_cal = data_cal['ra']
                dec_cal = data_cal['dec']
                mag_cal = data_cal[filt]
                magerr_cal = data_cal['err_'+str(filt)]
                # infer the zeropoint
                mask_zp = ((flux_opt>0.) & (flags_sex<=1))
                zp, zp_std, ncal_used = get_zp(ra_sex[mask_zp], dec_sex[mask_zp],
                                               airmass_sex[mask_zp], flux_opt[mask_zp],
                                               fluxerr_opt[mask_zp], ra_cal, dec_cal,
                                               mag_cal, magerr_cal, exptime, filt,
                                               imtype, log, data_cal)
                header['PC-NCAL'] = (ncal_used, 'number of brightest photcal stars used')

            header['PC-NCMAX'] = (get_par(C.phot_ncal_max,tel),
                                  'input max. number of photcal stars to use')
            header['PC-NCMIN'] = (get_par(C.phot_ncal_min,tel),
                                  'input min. number of stars to apply filter cut')

            if get_par(C.timing,tel):
                log_timing_memory (t0=t2, label='determining photometric calibration', log=log)

        else:
            log.info('Warning: photometric calibration catalog {} not found!'.format(get_par(C.cal_cat,tel)))

        # if there are no photometric calibration stars (either
        # because no photometric calibration catalog was provided, or
        # no calibration stars could be found in this particular
        # field), use the default zeropoints defined in the Settings
        # module
        if ncalstars==0:
            header['PC-P'] = (False, 'successfully processed by photometric calibration?')
            zp = get_par(C.zp_default,tel)[filt]
            zp_std = 0.

        # apply the zeropoint
        mag_opt, magerr_opt = apply_zp(flux_opt, zp, airmass_sex, exptime, filt, log,
                                       fluxerr=fluxerr_opt, zp_std=None)

        # infer limiting magnitudes from corresponding limiting
        # fluxes using zeropoint and median airmass
        [limmag_3sigma] = apply_zp([limflux_3sigma], zp, airmass_sex_median, exptime, filt, log)
        log.info('3-sigma limiting magnitude: {}'.format(limmag_3sigma))
        [limmag_5sigma] = apply_zp([limflux_5sigma], zp, airmass_sex_median, exptime, filt, log)
        log.info('5-sigma limiting magnitude: {}'.format(limmag_5sigma))
        
        # add header keyword(s):
        header['PC-P'] = (True, 'successfully processed by phot. calibration?')
        header['PC-ZPDEF'] = (get_par(C.zp_default,tel)[filt], '[mag] default filter zeropoint in settings file')
        header['PC-ZP'] = (zp, '[mag] zeropoint=m_AB+2.5*log10(flux[e-/s])+A*k')
        header['PC-ZPSTD'] = (zp_std, '[mag] sigma (STD) zeropoint sigma')
        header['PC-EXTCO'] = (get_par(C.ext_coeff,tel)[filt], '[mag] filter extinction coefficient (k)')
        header['PC-AIRM'] = (airmass_sex_median, 'median airmass of calibration stars')
        header['LIMMAG3'] = (limmag_3sigma, '[mag] full-frame 3-sigma limiting magnitude')
        header['LIMMAG5'] = (limmag_5sigma, '[mag] full-frame 5-sigma limiting magnitude')

        # if these optimal fluxes and magnitudes already present in catalog,
        # delete them; this could happen in case [C.redo] is set to True
        if 'FLUX_OPT' in data_sex.dtype.names:
            data_sex = drop_fields(data_sex, ['FLUX_OPT','FLUXERR_OPT','MAG_OPT','MAGERR_OPT'])

        data_sex = append_fields(data_sex, ['FLUX_OPT','FLUXERR_OPT'] ,
                                 [flux_opt, fluxerr_opt], usemask=False, asrecarray=True)
        
        data_sex = append_fields(data_sex, ['MAG_OPT','MAGERR_OPT'] ,
                                 [mag_opt, magerr_opt], usemask=False, asrecarray=True)

        # write updated catalog to file
        fits.writeto(sexcat, data_sex, overwrite=True)
                        
        if get_par(C.timing,tel):
            log_timing_memory (t0=t2, label='creating binary fits table including fluxopt', log=log)

    # split full image into subimages to be used in run_ZOGY - this
    # needs to be done after determination of optimal fluxes as
    # otherwise the potential replacement of the saturated pixels will
    # not be taken into account

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(get_par(C.subimage_size,tel),
                                                                       ysize, xsize, log)
    ysize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    xsize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)

    if ref_fits_remap is not None:
        data = data_ref_remap
        data_bkg = data_ref_bkg_remap
        data_bkg_std = data_ref_bkg_std_remap
        data_mask = data_ref_remap_mask
    else:
        data = data_wcs
    
    if get_par(C.timing,tel): t2 = time.time()

    fftdata = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    fftdata_bkg = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    fftdata_bkg_std = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    fftdata_mask = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='uint8')
    for nsub in range(nsubs):
        fftcut = cuts_fft[nsub]
        index_fft = tuple([slice(fftcut[0],fftcut[1]), slice(fftcut[2],fftcut[3])])
        subcutfft = cuts_ima_fft[nsub]
        index_fftdata = tuple([slice(subcutfft[0],subcutfft[1]), slice(subcutfft[2],subcutfft[3])])
        fftdata[nsub][index_fft] = data[index_fftdata]
        fftdata_bkg[nsub][index_fft] = data_bkg[index_fftdata]
        fftdata_bkg_std[nsub][index_fft] = data_bkg_std[index_fftdata]
        fftdata_mask[nsub][index_fft] = data_mask[index_fftdata]

    if get_par(C.timing,tel):
        log_timing_memory (t0=t2, label='filling fftdata cubes', log=log)

    if get_par(C.make_plots,tel):

        # in case optimal flux block above was skipped, the SExtractor
        # catalogue with FLUX_OPT needs to be read in here to be able
        # to make the plots below
        try:
            data_sex['FLUX_OPT'][0]
        except NameError:
            # read SExtractor fits table
            data_sex = read_hdulist (sexcat, ext_data=1)
            # and define flux_opt and fluxerr_opt
            flux_opt = data_sex['FLUX_OPT']
            fluxerr_opt = data_sex['FLUXERR_OPT']
            # and corresponding calibrated magnitudes
            if os.path.isfile(get_par(C.cal_cat,tel)) and 'mag_opt' in locals():
                mag_opt = data_sex['MAG_OPT']
                magerr_opt = data_sex['MAGERR_OPT']
            if mypsffit:
                flux_mypsf = data_sex['FLUX_PSF']
                fluxerr_mypsf = data_sex['FLUXERR_PSF']
                x_psf = data_sex['X_PSF']
                y_psf = data_sex['Y_PSF']
            # read a few extra header keywords needed below
            keywords = ['exptime', 'filter', 'obsdate']
            exptime, filt, obsdate = read_header(header, keywords, log)
        else:
            if get_par(C.verbose,tel):
                log.info('data_sex array is already defined; no need to read it in')

                
        # filter arrays by FLAG
        index = ((data_sex['FLUX_AUTO']>0) & (data_sex['FLAGS']==0))
        class_star = data_sex['CLASS_STAR'][index]
        flux_auto = data_sex['FLUX_AUTO'][index] * gain
        fluxerr_auto = data_sex['FLUXERR_AUTO'][index] * gain
        s2n_auto = flux_auto / fluxerr_auto
        flux_opt = data_sex['FLUX_OPT'][index]
        fluxerr_opt = data_sex['FLUXERR_OPT'][index]
        if os.path.isfile(get_par(C.cal_cat,tel)) and 'mag_opt' in locals():
            mag_opt = data_sex['MAG_OPT'][index]
            magerr_opt = data_sex['MAGERR_OPT'][index]
        x_win = data_sex['XWIN_IMAGE'][index]
        y_win = data_sex['YWIN_IMAGE'][index]
        fwhm_image = data_sex['FWHM_IMAGE'][index]
        if mypsffit:
            flux_mypsf = flux_psf[index]
            fluxerr_mypsf = fluxerr_psf[index]
            x_psf = x_psf[index]
            y_psf = y_psf[index]
            
        if os.path.isfile(get_par(C.cal_cat,tel)) and 'mag_opt' in locals():
            # histogram of all 'good' objects as a function of magnitude
            bins = np.arange(12, 22, 0.2)
            plt.hist(np.ravel(mag_opt), bins, color='tab:blue')
            x1,x2,y1,y2 = plt.axis()
            title = 'filter: {}, exptime: {:.0f}s'.format(filt, exptime)
            if 'limmag_5sigma' in locals():
                limmag = np.float(limmag_5sigma)
                plt.plot([limmag, limmag], [y1,y2], color='black', linestyle='--')
                title += ', lim. mag (5$\sigma$; dashed line): {:.2f}'.format(limmag)
            plt.title(title)
            plt.xlabel(filt+' magnitude')
            plt.ylabel('number')
            plt.savefig(base+'_magopt.pdf')
            if get_par(C.show_plots,tel): plt.show()
            plt.close()

        # compare flux_opt with flux_auto
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
            
        # compare flux_opt with flux_aper
        for i in range(len(get_par(C.apphot_radii,tel))):

            aper = get_par(C.apphot_radii[i],tel)
            field = 'FLUX_APER'
            field_err = 'FLUXERR_APER'
            field_format = 'FLUX_APER_R{}xFWHM'.format(aper)
            field_format_err = 'FLUXERR_APER_R{}xFWHM'.format(aper)

            if field in data_sex.dtype.names:
                flux_aper = data_sex[field][index,i] * gain
                fluxerr_aper = data_sex[field_err][index,i] * gain
            elif field_format in data_sex.dtype.names:
                flux_aper = data_sex[field_format][index] * gain
                fluxerr_aper = data_sex[field_format_err][index] * gain
                
            flux_diff = (flux_opt - flux_aper) / flux_aper
            xlabel = 'S/N (AUTO)'
            ylabel = '(FLUX_OPT - {}) / {}'.format(field_format, field_format)

            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel=xlabel, ylabel=ylabel,
                          filename='{}_fluxopt_vs_fluxaper_{}xFWHM.pdf'.format(base, aper),
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')

            flux_diff = (flux_auto - flux_aper) / flux_aper
            ylabel = '(FLUX_AUTO - {}) / {}'.format(field_format, field_format)
            plot_scatter (s2n_auto, flux_diff, limits, class_star,
                          xlabel=xlabel, ylabel=ylabel,
                          filename='{}_fluxauto_vs_fluxaper_{}xFWHM.pdf'.format(base, aper),
                          title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')


            if mypsffit:
                flux_diff = (flux_mypsf - flux_aper) / flux_aper
                plot_scatter (s2n_auto, flux_diff, limits, class_star,
                              xlabel='S/N (AUTO)', ylabel='(FLUX_MYPSF - FLUX_APER ('+aper_str+'xFWHM)) / FLUX_APER ('+aper_str+'xFWHM)', 
                              filename=base+'_fluxmypsf_vs_fluxaper'+aper_str+'xFWHM.pdf',
                              title='rainbow color coding follows CLASS_STAR: from purple (star) to red (galaxy)')
            

        # compare with flux_psf if psffit catalog available
        sexcat_ldac_psffit = base+'_cat_ldac_psffit.fits'
        if os.path.isfile(sexcat_ldac_psffit):
            # read SExtractor psffit fits table
            data_sex = read_hdulist (sexcat_ldac_psffit, ext_data=2)
                
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

        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='prep_optimal_subtraction', log=log)

    #if get_par(C.verbose,tel):
    #    log.info('fftdata.dtype {}'.format(fftdata.dtype))
    #    log.info('psf.dtype {}'.format(psf.dtype))
    #    log.info('psf_orig.dtype {}'.format(psf_orig.dtype))
    #    log.info('fftdata_bkg.dtype {}'.format(fftdata_bkg.dtype))
    #    log.info('fftdata_bkg_std.dtype {}'.format(fftdata_bkg_std.dtype))
    
    return fftdata, psf, psf_orig, fftdata_bkg, fftdata_bkg_std, fftdata_mask
    

################################################################################

def get_zp (ra_sex, dec_sex, airmass_sex, flux_opt, fluxerr_opt,
            ra_cal, dec_cal, mag_cal, magerr_cal, exptime, filt, imtype, log, data_cal):

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing get_zp ...')

    if imtype=='new':
        base = base_new
    else:
        base = base_ref
                    
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
    mag_sex_inst = -2.5*np.log10(flux_opt/exptime)
    pogson = 2.5/np.log(10.)
    magerr_sex_inst = pogson*fluxerr_opt/flux_opt

    # sort calibration catalog arrays in brightness
    index_sort = np.argsort(mag_cal)
    ra_cal = ra_cal[index_sort]
    dec_cal = dec_cal[index_sort]
    mag_cal = mag_cal[index_sort]
    magerr_cal = magerr_cal[index_sort]

    ncal = np.shape(ra_cal)[0]
    nmatch = 0
    # loop calibration stars and find a match in SExtractor sources
    for i in range(ncal):

        mask_match = find_stars(ra_sex, dec_sex, ra_cal[i], dec_cal[i],
                                dist_max, log, search='circle')

        if np.sum(mask_match)==1:
            # there's one match, calculate its zeropoint
            # need to calculate airmass for each star, as around A=2,
            # difference in airmass across the FOV is 0.1, i.e. a 5% change
            zp_array[mask_match] = (mag_cal[i] - mag_sex_inst[mask_match] +
                                    airmass_sex[mask_match]*get_par(C.ext_coeff,tel)[filt])

            #if get_par(get_par(C.verbose,tel):
            #    if 'spectype' in data_cal.dtype.names:
            #        log.info('ra_cal: {}, dec_cal: {}, mag_cal: {}, magerr_sex: {}, zp_array: {}, spectype: {}, chi2: {}, absdev: {}'.
            #                 format(ra_cal[i], dec_cal[i], mag_cal[i],
            #                        magerr_sex_inst[mask_match], zp_array[mask_match],
            #                        data_cal['spectype'][i], data_cal['chi2'][i], data_cal['absdev'][i]))
            #    else:
            #        log.info('ra_cal: {}, dec_cal: {}, mag_cal: {}, magerr_sex: {}, zp_array: {}'.
            #                 format(ra_cal[i], dec_cal[i], mag_cal[i],
            #                        magerr_sex_inst[mask_match], zp_array[mask_match]))

                    
            # done when number of matches equals [C.phot_ncal_max]
            nmatch += 1
            if nmatch == get_par(C.phot_ncal_max,tel):
                break
                
                
    # determine median zeropoint
    zp_mean, zp_std, zp_median = clipped_stats(zp_array, clip_zeros=True,
                                               make_hist=get_par(C.make_plots,tel),
                                               name_hist=base+'_zp_hist.pdf',
                                               hist_xlabel=filt+' zeropoint (mag)',
                                               log=log)
    if get_par(C.verbose,tel):
        log.info('zp_mean: {:.3f}, zp_median: {:.3f}, zp_std: {:.3f}'.
                 format(zp_mean, zp_median, zp_std))

    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_zp', log=log)

    return zp_median, zp_std, nmatch


################################################################################

def apply_zp (flux, zp, airmass, exptime, filt, log,
              fluxerr=None, zp_std=None):

    """Function that converts the array [flux] into calibrated magnitudes
    using [zp] (a scalar), [airmass] (scalar or array with the same
    size as [flux]), exptime (scalar) and [filt]. If [fluxerr] is
    provided, the function will also return the magnitude errors. If
    [zp_std] is provided, it is summed quadratically to the magnitude
    errors. The output will be numpy arrays with the same number of
    elements as the input flux."""
    
    if get_par(C.timing,tel): t = time.time()
    #log.info('Executing apply_zp ...')

    # make sure input fluxes are numpy arrays
    flux = np.asarray(flux)
    if fluxerr is not None:
        fluxerr = np.asarray(fluxerr)
    
    # instrumental magnitudes 
    nrows = len(flux)
    mag_inst = np.zeros(nrows)
    mask_pos = (flux > 0.)
    mag_inst[mask_pos] = -2.5*np.log10(flux[mask_pos]/exptime)
    # now convert the instrumental mags
    # N.B.: airmass correction is relative to airmass at which
    # atmospheric extinction was already included in the calibration
    # catalog
    mag = zp + mag_inst - airmass * get_par(C.ext_coeff,tel)[filt]
    # set magnitudes of sources with non-positive fluxes to -1
    mag[~mask_pos] = -1
    
    # and determine errors if [fluxerr] is provided
    if fluxerr is not None:
        pogson = 2.5/np.log(10.)
        magerr = np.zeros(nrows)
        magerr[mask_pos] = pogson*fluxerr[mask_pos]/flux[mask_pos]
        if zp_std is not None:
            # add zp_std to output magnitude error
            magerr = np.sqrt(magerr**2 + zp_std**2)
            # provide warning if zp_std is large
            if zp_std>0.1:
                log.info('Warning: zp_std is larger than 0.1 mag: {}'.
                         format(zp_std))
        # set errors of sources with non-positive fluxes to -1
        magerr[~mask_pos] = -1
        
    #if get_par(C.timing,tel):
    #    log_timing_memory (t0=t, label='apply_zp', log=log)

    if fluxerr is not None:
        return mag, magerr
    else:
        return mag


################################################################################

def field_stars (ra_cat, dec_cat, ra, dec, dist, log, search='box'):

    # find entries in [ra_cat] and [dec_cat] within [dist] of
    # [ra] and [dec]
    mask_data = np.zeros(len(ra_cat), dtype='bool')
    # make a big cut in arrays ra_cat and dec_cat to speed up
    mask_cut = (np.abs(dec_cat-dec)<=dist)
    ra_cat_cut = ra_cat[mask_cut]
    dec_cat_cut = dec_cat[mask_cut]
    
    center = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
    targets = SkyCoord(ra=ra_cat_cut*u.deg, dec=dec_cat_cut*u.deg, frame="icrs")
    separation = center.separation(targets).deg

    if search=='circle':
        # find within circle:
        mask_data[mask_cut] = (separation<=dist)
    else:
        posangle = center.position_angle(targets).to(u.deg).rad
        mask_data[mask_cut] = ((abs(separation*np.sin(posangle)) <= dist) & 
                               (abs(separation*np.cos(posangle)) <= dist))

    return mask_data


################################################################################

def find_stars (ra_cat, dec_cat, ra, dec, dist, log, search='box'):

    if get_par(C.timing,tel): t = time.time()
    #log.info('Executing find_stars ...')

    # find entries in [ra_cat] and [dec_cat] within [dist] of
    # [ra] and [dec]
    mask_data = np.zeros(len(ra_cat), dtype='bool')
    # make a big cut in arrays ra_cat and dec_cat to speed up
    mask_cut = (np.abs(dec_cat-dec)<=dist)
    ra_cat_cut = ra_cat[mask_cut]
    dec_cat_cut = dec_cat[mask_cut]

    if search=='circle':
        # find within circle:
        dsigma = haversine(ra_cat_cut, dec_cat_cut, ra, dec)
        mask_data[mask_cut] = (dsigma<=dist)
    else:
        # find within box:
        dsigma_ra = haversine(ra_cat_cut, dec_cat_cut, ra, dec_cat_cut)
        dsigma_dec = np.abs(dec_cat_cut-dec)
        mask_data[mask_cut] = ((dsigma_ra<=dist) & (dsigma_dec<=dist))

    #if get_par(C.timing,tel):
    #    log_timing_memory (t0=t, label='find_stars', log=log)

    return mask_data


################################################################################
        
def haversine (ra1, dec1, ra2, dec2):

    """Function that calculates (absolute) angle in degrees between RA,
    DEC coordinates ra1, dec1 and ra2, dec2. Input coordinates can be
    scalars or arrays; all are assumed to be in decimal degrees."""
    
    # convert to radians
    ra1, ra2, dec1, dec2 = map(np.radians, [ra1, ra2, dec1, dec2])
    
    d_ra = np.abs(ra1-ra2)
    d_dec = np.abs(dec1-dec2)
    
    a = np.sin(d_dec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return list(map(np.degrees, [c]))[0]


################################################################################

def zone (dec, zone_size=0.5):

    # zones of [zone_size] arcminute in size:
    #return np.floor((90 + dec)/0.0083333).astype(int)
    return np.floor((90 + dec)*(60./zone_size)).astype(int)
    # see http://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_external_catalogues/ssec_dm_panstarrs1_original_valid.html


################################################################################
    
def get_ext_list (dec_center, fov_half_deg, zone_size=60):
    
    ext_min = 1+zone(max(-89.999,dec_center-fov_half_deg), zone_size=zone_size)
    ext_max = 1+zone(min(+89.999,dec_center+fov_half_deg), zone_size=zone_size)
    # ext_max+1 ensures that ext_max is included
    return range(ext_min, ext_max+1)

    
################################################################################

def get_airmass (ra, dec, obsdate, lat, lon, height, log=None):

    if log is not None:
        log.info('Executing get_airmass ...')

    location = EarthLocation(lat=lat, lon=lon, height=height)
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    coords_altaz = coords.transform_to(AltAz(obstime=Time(obsdate), location=location))

    return coords_altaz.secz

        
################################################################################

def fixpix (data, data_bkg, log, satlevel=60000., data_mask=None, fwhm=3,
            base=None):

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing fixpix ...')

    data_fixed = np.copy(data)
    
    # replace infinite values and nans with the background
    #mask_infnan = ~np.isfinite(data)
    #data[mask_infnan] = data_bkg[mask_infnan]
    #n_infnan = np.sum(mask_infnan)
    #if n_infnan>0:
    #    log.info('Warning: number of infinite/nan numbers in image: {}'.
    #             format(n_infnan))

    # replace edge pixels with the background
    mask_edge = (data_mask==get_par(C.mask_value['edge'],tel))
    #data[mask_edge] = 0.
    # or with the background
    data_fixed[mask_edge] = data_bkg[mask_edge]

    # now try to clean the image from artificially sharp features such
    # as saturated and pixels as defined in data_mask - the FFTs in
    # [run_ZOGY] produce large-scale features surrounding these sharp
    # features in the subtracted image.

    # first replace bad/saturated pixels with nans
    mask_inpaint = ((data_mask == C.mask_value['bad']) |
                    (data_mask == C.mask_value['saturated']) |
                    (data_mask == C.mask_value['saturated-connected']))
        
    print ('np.sum(mask_inpaint): {}'.format(np.sum(mask_inpaint)))        
    data_fixed[mask_inpaint] = np.nan
    
    # using inpaint.py from https://github.com/Technariumas/Inpainting
    #array : 2d np.ndarray - an array containing NaN elements that have to be replaced
    #max_iter : int - the number of iterations
    #kernel_size : int - the size of the kernel, default is 1
    #method : str - the method used to replace invalid values. Valid options are
    # `localmean`, 'idw'.
    #data_fixed = inpaint.replace_nans(data, max_iter=5, kernel_radius=5,
    #                                  kernel_sigma=5, method='idw')

    # using astropy convolution: first defined a Gaussian kernel
    # scaled to the FWHM
    std_odd = int(fwhm/2.35)
    # or keep it small
    std_odd = 1
    if not std_odd % 2:
        std_odd += 1
    # prepare convolution kernel
    kernel = Gaussian2DKernel(std_odd, x_size=3, y_size=3)
        
    # astropy's convolution replaces the NaN pixels with a kernel-weighted
    # interpolation from their neighbors
    it=0
    while np.isnan(data_fixed).any():
        it+=1
        log.info ('iteration: {}'.format(it))
        #data_fixed = convolve(data_fixed, kernel)
        data_fixed = interpolate_replace_nans(data_fixed, kernel)
        log.info ('np.sum(np.isnan(data_fixed)): {}'.format(np.sum(np.isnan(data_fixed))))


    fits.writeto(base+'_fixed.fits', data_fixed, overwrite=True)
    
    # using restoration.inpaint_biharmonic
    # replace nonzero pixels with 1
    #mask[mask != 0] = 1
    # data values need to be between -1 and 1
    #norm = np.amax(np.abs(data))
    #data_fixed = restoration.inpaint_biharmonic(data/norm, mask, multichannel=False)
    #data_fixed *= norm

    
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='fix_pix', log=log)

    return data_fixed

        
################################################################################

def get_back (data, objmask, log, clip=True):
    
    """Function that returns the background of the image [data]. A clipped
    median is determined for each subimage which is masked using the
    object mask (created from SExtractor's '-OBJECTS' image, where
    objects have zero values). The subimages (with size:
    [C.bkg_boxsize]) are then median filtered and resized to the size
    of the input image.

    """

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing get_back ...')
    

    # use the SExtractor '-OBJECTS' image, which is a (SExtractor)
    # background-subtracted image with all pixels where objects were
    # detected set to zero (-OBJECTS), as the object mask

    # mask all pixels with zeros in [data_objmask] or that have
    # non-positive pixel values in [data]
    mask_reject = (objmask | (data<=0))

    # mask to use (opposite of mask_reject)
    mask_use = ~mask_reject
        
    #if get_par(C.timing,tel):
    #    log_timing_memory (t0=t, label='get_back -2', log=log)

    # determine background for fraction of the full masked image
    ysize, xsize = data.shape
    index_stat = get_index_stat (xsize*ysize, np.sum(mask_use))
        
    # determine clipped median and RMS/std in data with objects
    # masked
    if clip:
        # get clipped_stats mean, std and median 
        mean_full, std_full, median_full = clipped_stats(data[mask_use][index_stat],
                                                             nsigma=get_par(C.bkg_nsigma,tel),
                                                             log=None)
        #if get_par(C.timing,tel):
        #    log_timing_memory (t0=t, label='get_back -1', log=log)
    else:
        median_full = np.median(data[mask_use][index_stat])
        std_full = np.std(data[mask_use][index_stat])
        if get_par(C.verbose,tel):
            log.info('Background median in object-masked image: {:.3f} +- {:.3f}'
                     .format(median_full, std_full))


    # loop through subimages the size of C.bkg_boxsize, and
    # determine median from the masked data
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(get_par(C.bkg_boxsize,tel),
                                                                       ysize, xsize, log)        
    nsubs = centers.shape[0]

    # loop subimages
    if ysize % get_par(C.bkg_boxsize,tel) != 0 or xsize % get_par(C.bkg_boxsize,tel) !=0:
        log.info('Warning: [C.bkg_boxsize] does not fit integer times in image')
        log.info('         remaining pixels will be edge-padded')
    nysubs = int(ysize / get_par(C.bkg_boxsize,tel))
    nxsubs = int(xsize / get_par(C.bkg_boxsize,tel))
    # prepare output median and std output arrays
    mini_median = np.ndarray(nsubs, dtype='float32')
    mini_std = np.ndarray(nsubs, dtype='float32')

    # minimum fraction of background subimage pixels not to be
    # affected by the object mask
    mask_minsize = 0.5*get_par(C.bkg_boxsize,tel)**2
    
    # loop over background subimages and determine
    # their median and standard deviation
    t1=time.time()
    for nsub in range(nsubs):
        mini_median[nsub], mini_std[nsub] = get_median_std(
            nsub, cuts_ima, data, mask_use, mask_minsize, clip,
            median_full, std_full, log=None)
        
    # reshape and transpose
    mini_median = mini_median.reshape((nxsubs, nysubs)).transpose()
    mini_std = mini_std.reshape((nxsubs, nysubs)).transpose()

    #if get_par(C.timing,tel):
    #    log_timing_memory (t0=t, label='get_back after reshaping and transposing', log=log)

    # median filter the meshes with filter of size [C.bkg_filtersize]
    shape_filter = (get_par(C.bkg_filtersize,tel), get_par(C.bkg_filtersize,tel))
    mini_median_filt = ndimage.filters.median_filter(mini_median, shape_filter)
    mini_std_filt = ndimage.filters.median_filter(mini_std, shape_filter)

    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_back', log=log)

    return mini_median_filt, mini_std_filt


################################################################################

def mini2back (mini_filt, shape_data, log, order_interp=3, bkg_boxsize=None):
    
    if get_par(C.timing,tel): t = time.time()
    log.info('Executing mini2back ...')

    # resize low-resolution meshes, with order [order_interp], where
    # order=0: nearest
    # order=1: bilinear spline interpolation
    # order=2: quadratic spline interpolation
    # order=3: cubic spline interpolation
    background = ndimage.zoom(mini_filt, get_par(C.bkg_boxsize,tel), order=order_interp)

    # if shape of the background is not equal to input [data]
    # then pad the background images
    if shape_data != background.shape:
        t1 = time.time()
        ysize, xsize = shape_data
        ypad = ysize - background.shape[0]
        xpad = xsize - background.shape[1]
        background = np.pad(background, ((0,ypad),(0,xpad)), 'edge')
        log.info('time to pad ' + str(time.time()-t1))

        #np.pad seems quite slow; alternative:
        #centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(get_par(C.bkg_boxsize,tel),
        #                                                                   ysize, xsize, log,
        #                                                                   get_remainder=True)
        # these now include the remaining patches
                        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='mini2back', log=log)
        
    return background


################################################################################

def get_index_stat (npix, nmask, frac_nmask_max=0.5, frac_npix_stat=0.1):
    
    ratio = 1.*nmask/npix
    if ratio >= frac_nmask_max:
        nstat = int(frac_npix_stat * npix / ratio)
        index_stat = (np.random.rand(nstat)*nmask).astype(int)
    else:
        index_stat = np.arange(nmask)
        
    return index_stat


################################################################################

def get_median_std (nsub, cuts_ima, data, mask_use, mask_minsize, clip,
                    median_full, std_full, log=None):
    
    subcut = cuts_ima[nsub]
    data_sub = data[subcut[0]:subcut[1], subcut[2]:subcut[3]]
    mask_sub = mask_use[subcut[0]:subcut[1], subcut[2]:subcut[3]]

    # determine background for fraction of the masked image
    #ysize, xsize = data_sub.shape
    #index_stat = get_index_stat(xsize*ysize, np.sum(mask_sub), frac_npix_stat=0.5)

    if np.sum(mask_sub) > mask_minsize:
        if clip:
            # get clipped_stats mean, std and median 
            mean, std, median = clipped_stats(
                data_sub[mask_sub], clip_upper_frac=0,
                nsigma=get_par(C.bkg_nsigma,tel), log=log)
        else:
            median = np.median(data_sub[mask_sub])
            std = np.std(data_sub[mask_sub])
    else:
        # if less than half of the elements of mask_sub
        # are True, use values from entire masked image
        median, std = median_full, std_full
            
    # fill median and std arrays
    #mini_median[nsub] = median
    #mini_std[nsub] = std

    return median, std
        
            
################################################################################
            
def plot_scatter (x, y, limits, corder, cmap='rainbow_r', marker='o',
                  xlabel=None, ylabel=None, legendlabel=None, title=None, filename=None,
                  simple=False, xscale='log', yscale='linear'):

    plt.axis(limits)
    plt.scatter(x, y, c=corder, cmap=cmap, alpha=1, label=legendlabel, edgecolors='black')
    plt.xscale(xscale)
    plt.yscale(yscale)
    if legendlabel is not None:
        plt.legend(numpoints=1, fontsize='medium')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    if get_par(C.show_plots,tel): plt.show()
    plt.close()


################################################################################

def plot_scatter_hist (x, y, limits, color='tab:blue', marker='o', xlabel=None,
                       ylabel=None, title=None, label=None, labelpos=None,
                       filename=None):

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # the scatter plot:
    axScatter.scatter(x, y, color=color, marker=marker, s=20,
                      edgecolors='black')

    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)

    # limits scatter plot
    axScatter.set_xlim((limits[0], limits[1]))
    axScatter.set_ylim((limits[2], limits[3]))

    if xlabel is not None:
        axScatter.set_xlabel(xlabel, fontsize=11)
    if ylabel is not None:
        axScatter.set_ylabel(ylabel, fontsize=11)

    binwidth = 0.01
    xbins = np.arange(limits[0], limits[1] + binwidth, binwidth)
    axHistx.hist(x, bins=xbins, color=color, edgecolor='black')
    ybins = np.arange(limits[2], limits[3] + binwidth, binwidth)
    axHisty.hist(y, bins=ybins, orientation='horizontal',
                 color=color, edgecolor='black')

    yticks = axHistx.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    xticks = axHisty.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    
    # limits histograms
    axHistx.set_xlim((limits[0], limits[1]))
    axHisty.set_ylim((limits[2], limits[3]))
    
    if label is not None:
        for i in range(len(label)):
            plt.annotate(label[i], xy=(0,0), xytext=labelpos[i],
                         textcoords='figure fraction', fontsize=11)

    if title is not None:
        axHistx.set_title(title)
    if filename is not None:
        plt.savefig(filename)
    if filename is not None:
        plt.savefig(filename)
    if get_par(C.show_plots,tel):
        plt.show()
    plt.close()


################################################################################

def get_psf(image, header, nsubs, imtype, fwhm, pixscale, log):

    """Function that takes in [image] and determines the actual Point
    Spread Function as a function of position from the full frame, and
    returns a cube containing the psf for each subimage in the full
    frame.

    """

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing get_psf ...')

    global psf_size_new

    if imtype=='new':
        base = base_new
    else:
        base = base_ref
    
    # determine image size from header
    xsize, ysize = header['NAXIS1'], header['NAXIS2']
    
    # run psfex on SExtractor output catalog
    #
    # If the PSFEx output file is already present with the same
    # [psf_size_config] as currently required, then skip [run_psfex].
    skip_psfex = False
    psfex_bintable = base+'_psf.fits'
    if os.path.isfile(psfex_bintable) and not get_par(C.redo,tel):
        data, header_psf = read_hdulist (psfex_bintable, ext_data=1, ext_header=1)
        data = data[0][0][:]
        # use function [get_samp_PSF_config_size] to determine [psf_samp]
        # and [psf_size_config]
        psf_samp, psf_size_config = get_samp_PSF_config_size()
        log.info('psf_samp, psf_size_config: '+str(psf_samp)+' '+str(psf_size_config))
        # check that the above [psf_size_config] is the same as
        # the size of the images in the data array, or equivalently,
        # the value of the header parameters 'PSFAXIS1' or 'PSFAXIS2'
        if psf_size_config == header_psf['PSFAXIS1']:
            skip_psfex = True
            if get_par(C.verbose,tel):
                log.info('Skipping run_psfex for image: '+image)
                        
    if not skip_psfex:
        psfexcat = base+'_psfex.cat'
        sexcat_ldac = base+'_ldac.fits'
        log.info('sexcat_ldac: {}'.format(sexcat_ldac))
        log.info('psfexcat: {}'.format(psfexcat))

        if True:
            t_temp = time.time()
            # feed PSFEx only with selected sources
            sexcat_ldac_selected = base+'_ldac_4psfex.fits'
            with fits.open(sexcat_ldac) as hdulist:
                data_ldac = hdulist[2].data
                mask_ok = ((data_ldac['FLAGS']<=1) & (data_ldac['SNR_WIN']>=get_par(C.psf_stars_s2n_min,tel)))
                # sort by FLUX_AUTO
                #index_sort = np.argsort(data_ldac['FLUX_AUTO'][mask_ok])
                # select the faintest 20,000 above the s2n cut-off
                #data_ldac = data_ldac[:][mask_ok] #[index_sort][0:20000]
                data_ldac = data_ldac[mask_ok] #[index_sort][0:20000]
                hdulist[2].data = data_ldac
                hdulist_new = fits.HDUList(hdulist)
                hdulist_new.writeto(sexcat_ldac_selected, overwrite=True)
                hdulist_new.close()

                if get_par(C.make_plots,tel):
                    result = prep_ds9regions(base+'_ds9regions_psfstars.txt',
                                             data_ldac['XWIN_IMAGE'],
                                             data_ldac['YWIN_IMAGE'],
                                             radius=5., width=2, color='red')
                            
            log.info('time to create selection of LDAC catalog for PSFEx: {}'
                     .format(time.time()-t_temp))
            
        try:
            # selected catalog:
            result = run_psfex(sexcat_ldac_selected, get_par(C.psfex_cfg,tel), psfexcat, imtype, log)
            # full catalog:
            #result = run_psfex(sexcat_ldac, get_par(C.psfex_cfg,tel), psfexcat, imtype, log)
        except Exception as e:
            PSFEx_processed = False
            log.info(traceback.format_exc())
            log.error('exception was raised during [run_psfex]: {}'.format(e))  
        else:
            PSFEx_processed = True

            
    # If [C.dosex_psffit] parameter is set, then again run SExtractor,
    # but now using output PSF from PSFEx, so that PSF-fitting can be
    # performed for all objects. The output columns defined in
    # [C.sex_par_psffit] include several new columns related to the PSF
    # fitting.
    sexcat_ldac_psffit = base+'_ldac_psffit.fits'
    if (not os.path.isfile(sexcat_ldac_psffit) or get_par(C.redo,tel)) and get_par(C.dosex_psffit,tel):
        result = run_sextractor(image, sexcat_ldac_psffit, get_par(C.sex_cfg_psffit,tel),
                                get_par(C.sex_par_psffit,tel), pixscale, log, header,
                                fit_psf=True, update_vignet=False, fwhm=fwhm)
        
    # If not already done so above, read in PSF output binary table
    # from psfex, containing the polynomial coefficient images
    if not ('header_psf' in dir()):
        data, header_psf = read_hdulist (psfex_bintable, ext_data=1, ext_header=1)
        data = data[0][0][:]

    # read in some header keyword values
    polzero1 = header_psf['POLZERO1']
    polzero2 = header_psf['POLZERO2']
    polscal1 = header_psf['POLSCAL1']
    polscal2 = header_psf['POLSCAL2']
    poldeg = header_psf['POLDEG1']
    psf_fwhm = header_psf['PSF_FWHM']
    psf_samp = header_psf['PSF_SAMP']
    psf_chi2 = header_psf['CHI2']
    psf_nstars = header_psf['ACCEPTED']
    # [psf_size_config] is the size of the PSF as defined in the PSFex
    # configuration file ([PSF_SIZE] parameter), which is the same as
    # the size of the [data] array
    psf_size_config = header_psf['PSFAXIS1']
    if get_par(C.verbose,tel):
        log.info('polzero1                     ' + str(polzero1))
        log.info('polscal1                     ' + str(polscal1))
        log.info('polzero2                     ' + str(polzero2))
        log.info('polscal2                     ' + str(polscal2))
        log.info('order polynomial:            ' + str(poldeg))
        log.info('PSFex FWHM:                  ' + str(psf_fwhm))
        log.info('PSF sampling size (pixels):  ' + str(psf_samp))
        log.info('PSF size defined in config:  ' + str(psf_size_config))
        log.info('number of accepted PSF stars:' + str(psf_nstars))
        log.info('final reduced chi2 PSFEx fit:' + str(psf_chi2))
        
    # call centers_cutouts to determine centers
    # and cutout regions of the full image
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
        get_par(C.subimage_size,tel), ysize, xsize, log)
    ysize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    xsize_fft = get_par(C.subimage_size,tel) + 2*get_par(C.subimage_border,tel)
    nxsubs = xsize/get_par(C.subimage_size,tel)
    nysubs = ysize/get_par(C.subimage_size,tel)

    if imtype == 'ref':
        # in case of the ref image, the PSF was determined from the
        # original image, while it will be applied to the remapped ref
        # image. So the centers of the cutouts in the remapped ref
        # image need to be mapped back to those in the original
        # reference image to get the PSF from the proper
        # coordinates. Easiest to do this using astropy.wcs, which
        # would also take care of any potential rotation and scaling.

        # first infer ra, dec corresponding to x, y pixel positions
        # (centers[:,1] and centers[:,0], respectively, using the WCS
        # solution in [new].wcs file from Astrometry.net
        header_new_temp = read_hdulist (base_new+'.fits', ext_header=0)
        wcs = WCS(header_new_temp)
        ra_temp, dec_temp = wcs.all_pix2world(centers[:,1], centers[:,0], 1)
        # then convert ra, dec back to x, y in the original ref image;
        # since this block concerns the reference image, the input [header]
        # corresponds to the reference header
        wcs = WCS(header)
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
    if get_par(C.verbose,tel):
        log.info('FWHM                      : ' + str(fwhm))
        log.info('final image PSF size      : ' + str(psf_size))
    # now change psf_samp slightly:
    psf_samp_update = float(psf_size) / float(psf_size_config)
    if imtype == 'new': psf_size_new = psf_size
    # [psf_ima] is the corresponding cube of PSF subimages
    psf_ima = np.zeros((nsubs,psf_size,psf_size))
    # [psf_ima_center] is [psf_ima] broadcast into images of xsize_fft
    # x ysize_fft
    psf_ima_center = np.zeros((nsubs,ysize_fft,xsize_fft))
    # [psf_ima_shift] is [psf_ima_center] shifted - this is
    # the input PSF image needed in the [run_ZOGY] function
    psf_ima_shift = np.zeros((nsubs,ysize_fft,xsize_fft))

    # if [run_psfex] was executed successfully (see above), then add a
    # number of header keywords
    if not skip_psfex and PSFEx_processed:
        header['PSF-P'] = (PSFEx_processed, 'successfully processed by PSFEx?')   
        header['PSF-RAD'] = (get_par(C.psf_radius,tel), '[FWHM] radius in units of FWHM to build PSF')
        header['PSF-SIZE'] = (psf_size, '[pix] size PSF image')
        header['PSF-FRAC'] = (get_par(C.psf_samp_fwhmfrac,tel), '[FWHM] PSF sampling step in units of FWHM')
        header['PSF-SAMP'] = (psf_samp_update, '[pix] PSF sampling step (~ PSF-FRAC x FWHM)')
        header['PSF-CFGS'] = (psf_size_config, 'size PSF config. image (= PSF-SIZE / PSF-SAMP)')
        header['PSF-NOBJ'] = (psf_nstars, 'number of accepted PSF stars')
        header['PSF-FIX'] = (get_par(C.use_single_psf,tel), 'single fixed PSF used for entire image?')
        header['PSF-PLDG'] = (poldeg, 'degree polynomial used in PSFEx')
        header['PSF-CHI2'] = (psf_chi2, 'final reduced chi-squared PSFEx fit')
        header['PSF-FWHM'] = (psf_fwhm, '[pix] image FWHM inferred by PSFEx')
        #header['PSF-ELON'] = (psf_elon, 'median elongation of PSF stars')
        #header['PSF-ESTD'] = (psf_elon_std, 'elongation sigma (STD) of PSF stars')
        
        
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

        if nsubs==1 or get_par(C.use_single_psf,tel):
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = (data[0] + data[1] * x + data[2] * x**2 +
                                  data[3] * y + data[4] * x * y + data[5] * y**2)
            elif poldeg==3:
                psf_ima_config = (data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 +
                                  data[4] * y + data[5] * x * y + data[6] * x**2 * y +
                                  data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3)

        # resample PSF image at image pixel scale

        # PMV 2018/11/22: N.B.!: runtime warning (not in log file,
        # only to STDOUT) related to zoom below:
        # /usr/lib/python2.7/dist-packages/scipy/ndimage/interpolation.py:600:
        # UserWarning: From scipy 0.13.0, the output shape of zoom()
        # is calculated with round() instead of int() - for these
        # inputs the size of the returned array has changed.",
        # UserWarning)
        psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update)
        # clean and normalize PSF
        psf_ima_resized_norm = clean_norm_psf(psf_ima_resized, get_par(C.psf_clean_factor,tel))

        psf_ima[nsub] = psf_ima_resized_norm
        if get_par(C.verbose,tel) and nsub==0:
            log.info('psf_samp, psf_samp_update: ' + str(psf_samp) + ', ' + str(psf_samp_update))
            log.info('np.shape(psf_ima_config): ' + str(np.shape(psf_ima_config)))
            log.info('np.shape(psf_ima): ' + str(np.shape(psf_ima)))
            log.info('np.shape(psf_ima_resized): ' + str(np.shape(psf_ima_resized)))
            log.info('psf_size: ' + str(psf_size))
            
        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            log.info('Warning: image not even in one or both dimensions!')
            
        xcenter_fft, ycenter_fft = int(xsize_fft/2), int(ysize_fft/2)
        if get_par(C.verbose,tel) and nsub==0:
            log.info('xcenter_fft, ycenter_fft: ' + str(xcenter_fft) + ', ' + str(ycenter_fft))

        psf_hsize = int(psf_size/2)
        index = tuple([slice(ycenter_fft-psf_hsize, ycenter_fft+psf_hsize+1), 
                       slice(xcenter_fft-psf_hsize, xcenter_fft+psf_hsize+1)])
        psf_ima_center[nsub][index] = psf_ima_resized_norm

        # perform fft shift
        psf_ima_shift[nsub] = fft.fftshift(psf_ima_center[nsub])
        
        if get_par(C.display,tel) and (nsub==0 or nsub==nysubs-1 or nsub==nsubs/2 or
                        nsub==nsubs-nysubs or nsub==nsubs-1):
            if imtype=='new':
                base = base_new
            else:
                base = base_ref
            fits.writeto(base+'_psf_ima_config_sub'+str(nsub)+'.fits', psf_ima_config, overwrite=True)
            fits.writeto(base+'_psf_ima_resized_norm_sub'+str(nsub)+'.fits',
                         psf_ima_resized_norm.astype('float32'), overwrite=True)
            fits.writeto(base+'_psf_ima_center_sub'+str(nsub)+'.fits',
                         psf_ima_center[nsub].astype('float32'), overwrite=True)            
            fits.writeto(base+'_psf_ima_shift_sub'+str(nsub)+'.fits',
                         psf_ima_shift[nsub].astype('float32'), overwrite=True)            


    # call above function [get_psf_sub] with pool.map
    if get_par(C.timing,tel): t1 = time.time()
    pool = ThreadPool(nthreads)
    pool.map(loop_psf_sub, range(nsubs))
    pool.close()
    pool.join()
    if get_par(C.timing,tel):
        log_timing_memory (t0=t1, label='loop_psf_sub pool', log=log)
        log_timing_memory (t0=t, label='get_psf', log=log)

    return psf_ima_shift.astype('float32'), psf_ima.astype('float32')


################################################################################

def get_fratio_dxdy(psfcat_new, psfcat_ref, sexcat_new, sexcat_ref,
                    header_new, header_ref, nsubs, cuts_ima, log, header):
    
    """Function that takes in output catalogs of stars used in the PSFex
    runs on the new and the ref image, and returns the arrays with
    pixel coordinates (!) x, y (in the new frame) and fratios for the
    matching stars. In addition, it provides the difference in x- and
    y-coordinates between the catalogs after converting the reference
    image pixels to pixels in the new image."""
    
    t = time.time()
    log.info('Executing get_fratio_dxdy ...')
    
    def readcat (psfcat):
        table = ascii.read(psfcat, format='sextractor')
        # In PSFEx version 3.17.1 (last stable version), only stars
        # with zero flags are recorded in the output catalog. However,
        # in PSFEx version 3.18.2 all objects from the SExtractor
        # catalog are recorded, and in that case the entries with
        # FLAGS_PSF=0 need to be selected to speed up this function
        # significantly in case SExtractor detects many sources.
        if 'FLAGS_PSF' in table.colnames:
            mask_zero = (table['FLAGS_PSF']==0)
        else:
            mask_zero = np.ones(len(table), dtype=bool)
        number = table['SOURCE_NUMBER'][mask_zero]
        x = table['X_IMAGE'][mask_zero]
        y = table['Y_IMAGE'][mask_zero]
        norm = table['NORM_PSF'][mask_zero]
        return number, x, y, norm
        
    # read psfcat_new
    number_new, x_new, y_new, norm_new = readcat(psfcat_new)
    # read psfcat_ref
    number_ref, x_ref, y_ref, norm_ref = readcat(psfcat_ref)

    if get_par(C.verbose,tel):
        log.info('new: number of PSF stars with zero FLAGS: {}'.format(len(x_new)))
        log.info('ref: number of PSF stars with zero FLAGS: {}'.format(len(x_ref)))
    
    def xy2radec (number, sexcat):

        '''Function to return the RA and DEC from the binary fits SExtractor
        catalog [sexcat] using the columns ALPHAWIN_J2000 and
        DELTAWIN_J2000.  [number] is an array of integers, indicating
        the source number in the SExtractor catalog [sexcat].'''

        # read SExtractor fits table
        data = read_hdulist (sexcat, ext_data=1)
        ra_sex = data['ALPHAWIN_J2000']
        dec_sex = data['DELTAWIN_J2000']
        # loop numbers and record in ra, dec
        ra = []
        dec = []
        for n in number:
            ra.append(ra_sex[n-1])
            dec.append(dec_sex[n-1])
        return np.array(ra), np.array(dec)


    # get reference ra, dec corresponding to x, y
    #ra_new, dec_new = xy2radec(number_new, sexcat_new)
    #ra_ref, dec_ref = xy2radec(number_ref, sexcat_ref)
    # instead use wcs.all_pix2world
    wcs = WCS(header_ref)
    ra_ref, dec_ref = wcs.all_pix2world(x_ref, y_ref, 1)

    # convert the reference RA and DEC to pixels in the new frame
    wcs = WCS(header_new)
    x_ref2new, y_ref2new = wcs.all_world2pix(ra_ref, dec_ref, 1)

    # these can be compared to x_new and y_new
    # to find matching entries
    x_new_match = []
    y_new_match = []
    dx_match = []
    dy_match = []
    fratio_match = []
    nmatch = 0
    dist_max = 5. #pixels
    for i_new in range(len(x_new)):
        # calculate distance to ref objects
        dx_temp = x_new[i_new] - x_ref2new
        dy_temp = y_new[i_new] - y_ref2new
        dist = np.sqrt(dx_temp**2 + dy_temp**2)
        # minimum distance and its index
        dist_min, i_ref = np.amin(dist), np.argmin(dist)
        if dist_min <= dist_max:
            nmatch += 1
            x_new_match.append(x_new[i_new])
            y_new_match.append(y_new[i_new])
            dx_match.append(dx_temp[i_ref])
            dy_match.append(dy_temp[i_ref])
            # append ratio of normalized counts to fratios
            fratio_match.append(norm_new[i_new] / norm_ref[i_ref])
                        
    if get_par(C.verbose,tel):
        log.info('fraction of PSF stars that match: ' + str(float(nmatch)/len(x_new)))

    x_new_match = np.asarray(x_new_match)
    y_new_match = np.asarray(y_new_match)
    dx_match = np.asarray(dx_match)
    dy_match = np.asarray(dy_match)
    fratio_match = np.asarray(fratio_match)
        
    # now also determine arrays for fratio, dx and dy to be used in
    # function [zogy_subloop]:
    fratio_sub = np.zeros(nsubs)
    dx_sub = np.zeros(nsubs)
    dy_sub = np.zeros(nsubs)

    # calculate full-frame average standard deviation and median
    fratio_mean_full, fratio_std_full, fratio_median_full = clipped_stats(fratio_match, log=log)
    dx_mean, dx_std, dx_median = clipped_stats(dx_match, log=log)
    dy_mean, dy_std, dy_median = clipped_stats(dy_match, log=log)
    dx_full = np.sqrt(dx_mean**2 + dx_std**2)
    dy_full = np.sqrt(dy_mean**2 + dy_std**2)
    if get_par(C.verbose,tel):
        log.info('median dx: {:.3f} +- {:.3f} pixels'.format(dx_median, dx_std))
        log.info('median dy: {:.3f} +- {:.3f} pixels'.format(dy_median, dy_std))
        log.info('full-frame dx: {:.3f}, dy: {:.3f}'.format(dx_full, dy_full))

    # add header keyword(s):
    header['Z-DXYLOC'] = (get_par(C.dxdy_local,tel), 'star position offsets determined per subimage?')
    header['Z-DX'] = (dx_median, '[pix] dx median offset full image')
    header['Z-DXSTD'] = (dx_std, '[pix] dx sigma (STD) offset full image')
    header['Z-DY'] = (dy_median, '[pix] dy median offset full image')
    header['Z-DYSTD'] = (dy_std, '[pix] dy sigma (STD) offset full image')
    header['Z-FNRLOC'] = (get_par(C.fratio_local,tel), 'flux ratios (Fnew/Fref) determined per subimage?')
    header['Z-FNR'] = (fratio_median_full, 'median flux ratio (Fnew/Fref) full image')
    header['Z-FNRSTD'] = (fratio_std_full, 'sigma (STD) flux ratio (Fnew/Fref) full image')

    
    def local_or_full(value_local, value_full, std_full, log, nsigma=3):
        # function to return full-frame value if local value is more
        # than [nsigma] (full frame) away from the full-frame value
        if np.abs(value_local-value_full)/std_full > nsigma or not np.isfinite(value_local):
            if get_par(C.verbose,tel):
                log.info('np.abs(value_local-value_full)/std_full: {}'
                         .format(np.abs(value_local-value_full)/std_full))
                log.info('adopted value: {}'.format(value_full))
            return value_full
        else:
            return value_local

        
    # loop subimages
    for nsub in range(nsubs):
        
        # [subcut] defines the pixel indices [y1 y2 x1 x2] identifying
        # the corners of the subimage in the entire input/output image
        # coordinate frame; used various times below
        subcut = cuts_ima[nsub]

        # start with full-frame values
        fratio_mean, fratio_std, fratio_median = fratio_mean_full, fratio_std_full, fratio_median_full
        dx = dx_full
        dy = dy_full
        
        # determine mask of full-frame matched values belonging to
        # this subimage
        y_index = (y_new_match-0.5).astype('uint16')
        x_index = (x_new_match-0.5).astype('uint16')
        mask_sub = ((y_index >= subcut[0]) & (y_index < subcut[1]) & 
                    (x_index >= subcut[2]) & (x_index < subcut[3]))

        # require a minimum number of values before adopting local
        # values for fratio, dx and dy
        if np.sum(mask_sub) >= 15:
            
            if get_par(C.fratio_local,tel):
                # determine local fratios
                fratio_mean, fratio_std, fratio_median = clipped_stats(fratio_match[mask_sub],
                                                                       log=log)
                fratio_mean = local_or_full (fratio_mean, fratio_mean_full, fratio_std_full, log)
                    
            # and the same for dx and dy
            if get_par(C.dxdy_local,tel):
                # determine local values
                dx_mean, dx_std, dx_median = clipped_stats(dx_match[mask_sub], log=log)
                dy_mean, dy_std, dy_median = clipped_stats(dy_match[mask_sub], log=log)
                dx = np.sqrt(dx_mean**2 + dx_std**2)
                dy = np.sqrt(dy_mean**2 + dy_std**2)

                # adopt full-frame values if local values are more
                # than nsigma the full-frame values
                dx = local_or_full (dx, 0., dx_full, log)
                dy = local_or_full (dy, 0., dy_full, log)
                
        fratio_sub[nsub] = fratio_mean
        dx_sub[nsub] = dx
        dy_sub[nsub] = dy

        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_fratio_dxdy', log=log)

    return x_new_match, y_new_match, fratio_match, dx_match, dy_match, \
        fratio_sub, dx_sub, dy_sub


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
        # In PSFEx version 3.17.1 (last stable version), only stars
        # with zero flags are recorded in the output catalog. However,
        # in PSFEx version 3.18.2 all objects from the SExtractor
        # catalog are recorded, and in that case the entries with
        # FLAGS_PSF=0 need to be selected to speed up this function
        # significantly in case SExtractor detects many sources.
        if 'FLAGS_PSF' in table.colnames:
            mask_zero = (table['FLAGS_PSF']==0)
        else:
            mask_zero = np.ones(len(table), dtype=bool)
        number = table['SOURCE_NUMBER'][mask_zero]
        x = table['X_IMAGE'][mask_zero]
        y = table['Y_IMAGE'][mask_zero]
        norm = table['NORM_PSF'][mask_zero]
        return number, x, y, norm
        
    # read psfcat_new
    number_new, x_new, y_new, norm_new = readcat(psfcat_new)
    # read psfcat_ref
    number_ref, x_ref, y_ref, norm_ref = readcat(psfcat_ref)

    if get_par(C.verbose,tel):
        log.info('new: number of PSF stars with zero FLAGS: {}'.format(len(x_new)))
        log.info('ref: number of PSF stars with zero FLAGS: {}'.format(len(x_ref)))
    
    def xy2radec (number, sexcat):

        '''Function to return the RA and DEC from the binary fits SExtractor
        catalog [sexcat] using the columns ALPHAWIN_J2000 and
        DELTAWIN_J2000.  [number] is an array of integers, indicating
        the source number in the SExtractor catalog [sexcat].'''

        # read SExtractor fits table
        data = read_hdulist (sexcat, ext_data=1)
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
                        
    if get_par(C.verbose,tel):
        log.info('fraction of PSF stars that match: ' + str(float(nmatch)/len(x_new)))
            
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_fratio_radec', log=log)

    return (np.array(x_new_match), np.array(y_new_match), np.array(fratio),
            np.array(dra_match), np.array(ddec_match))


################################################################################

def centers_cutouts(subsize, ysize, xsize, log, get_remainder=False):
    
    """Function that determines the input image indices (!) of the centers
    (list of nsubs x 2 elements) and cut-out regions (list of nsubs x
    4 elements) of image with the size xsize x ysize. Subsize is the
    fixed size of the subimages, e.g. 512 or 1024. The routine will
    fit as many of these in the full frames, and for the moment it
    will ignore any remaining pixels outside."""
    
    nxsubs =int(xsize / subsize)
    nysubs = int(ysize / subsize)
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

    border = get_par(C.subimage_border,tel)
    ysize_fft = subsize + 2*border
    xsize_fft = subsize + 2*border

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
            y1 = np.amax([0,y-ny/2-border])
            x1 = np.amax([0,x-nx/2-border])
            y2 = np.amin([ysize,y+ny/2+border])
            x2 = np.amin([xsize,x+nx/2+border])
            cuts_ima_fft[nsub] = [y1,y2,x1,x2]
            cuts_fft[nsub] = [y1-(y-ny/2-border),ysize_fft-(y+ny/2+border-y2),
                              x1-(x-nx/2-border),xsize_fft-(x+nx/2+border-x2)]
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
        fits.writeto(fitsfile, np.array(array).astype('float32'), overwrite=True)            
        # append to command
        cmd.append(fitsfile)

    result = subprocess.call(cmd)

    
################################################################################

def run_wcs(image_in, image_out, ra, dec, pixscale, width, height, header, log):

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing run_wcs ...')
    
    scale_low = (1.-get_par(C.pixscale_varyfrac,tel)) * pixscale
    scale_high = (1.+get_par(C.pixscale_varyfrac,tel)) * pixscale

    base = image_in.replace('.fits','')
    sexcat = base+'_cat.fits'

    # feed Astrometry.net only with brightest sources; N.B.:
    # keeping only objects with zero FLAGS does not work well in crowded fields
    # read SExtractor catalogue (this is also used further down below in this function)
    data_sexcat = read_hdulist (sexcat, ext_data=1)
    nobjects = data_sexcat.shape[0]
    header['S-NOBJ'] = (nobjects, 'number of objects detected by SExtractor')
    # select stars for finding WCS solution
    mask_use = (data_sexcat['FLAGS']<=1)
    # sort in brightness (FLUX_AUTO)
    index_sort = np.argsort(data_sexcat['FLUX_AUTO'][mask_use])
    # select the brightest objects
    nbright = get_par(C.ast_nbright,tel)
    sexcat_bright = base+'_cat_bright.fits'
    #fits.writeto(sexcat_bright, data_sexcat[:][mask_use][index_sort][-nbright:], overwrite=True)
    fits.writeto(sexcat_bright, data_sexcat[mask_use][index_sort][-nbright:], overwrite=True)

    # create ds9 regions text file to show the brightest stars
    if get_par(C.make_plots,tel):
        result = prep_ds9regions(base+'_cat_bright_ds9regions.txt',
                                 data_sexcat['XWIN_IMAGE'][mask_use][index_sort][-nbright:],
                                 data_sexcat['YWIN_IMAGE'][mask_use][index_sort][-nbright:],
                                 radius=5., width=2, color='green')
        
    #scampcat = image_in.replace('.fits','.scamp')

    dir_out = '.'
    if '/' in base:
        dir_out = '/'.join(base.split('/')[:-1])

    cmd = ['solve-field', '--no-plots', #'--no-fits2fits', cloud version of astrometry does not have this arg
           '--x-column', 'XWIN_IMAGE', '--y-column', 'YWIN_IMAGE',
           '--sort-column', 'FLUX_AUTO',
           '--no-remove-lines', '--uniformize', '0',
           # only work on brightest sources
           #'--objs', '1000',
           '--width', str(width), '--height', str(height),           
           #'--keep-xylist', sexcat,
           # ignore existing WCS headers in FITS input images
           #'--no-verify', 
           #'--verbose',
           #'--verbose',
           #'--parity', 'neg',
           #'--code-tolerance', str(0.01), 
           #'--quad-size-min', str(0.1),
           # for KMTNet images restrict the max quad size:
           #'--quad-size-max', str(0.1),
           # number of field objects to look at:
           '--depth', '50,150,200,250,300,350,400,450,500',
           #'--scamp', scampcat,
           sexcat_bright,
           '--tweak-order', str(get_par(C.astronet_tweak_order,tel)), '--scale-low', str(scale_low),
           '--scale-high', str(scale_high), '--scale-units', 'app',
           '--ra', str(ra), '--dec', str(dec), '--radius', str(get_par(C.astronet_radius,tel)),
           '--new-fits', 'none', '--overwrite',
           '--out', base.split('/')[-1],
           '--dir', dir_out
    ]

    # log cmd executed
    cmd_str = ' '.join(cmd)
    log.info('Astrometry.net command executed:\n{}'.format(cmd_str))
    
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)

    # read header of .match file, which describes the quad match that
    # solved the image
    data_match = read_hdulist (base+'.match', ext_data=1)
        
    if os.path.exists("%s.solved"%base) and status==0:
        os.remove("%s.solved"%base)
        #os.remove("%s.match"%base)
        #os.remove("%s.rdls"%base)
        #os.remove("%s.corr"%base)
        #os.remove("%s-indx.xyls"%base)
    else:
        log.error("Solving WCS failed.")
        return 'error'

    if get_par(C.timing,tel): t2 = time.time()

    # read image_in
    data = read_hdulist (image_in, ext_data=0)

    # read header saved in .wcs 
    wcsfile = base+'.wcs'
    header_wcs = read_hdulist (wcsfile, ext_header=0)

    # remove HISTORY, COMMENT and DATE fields from Astrometry.net header
    # they are still present in the base+'.wcs' file
    header_wcs.pop('HISTORY', None)
    header_wcs.pop('COMMENT', None)
    header_wcs.pop('DATE', None)
    
    # add specific keyword indicating index file of match
    if data_match['HEALPIX'][0]!=-1:
        anet_index = 'index-{}-{:02d}.fits'.format(data_match['INDEXID'][0], data_match['HEALPIX'][0])
    else:
        anet_index = 'index-{}.fits'.format(data_match['INDEXID'][0])
    header_wcs['A-INDEX'] = (anet_index, 'name of index file WCS solution')

    # and pixelscale
    cd1_1 = header_wcs['CD1_1']
    cd1_2 = header_wcs['CD1_2']
    cd2_1 = header_wcs['CD2_1']
    cd2_2 = header_wcs['CD2_2']

    anet_pixscale_x = np.sqrt(cd1_1**2 + cd1_2**2) * 3600.
    anet_pixscale_y = np.sqrt(cd2_2**2 + cd2_1**2) * 3600.
    anet_pixscale = np.average([anet_pixscale_x, anet_pixscale_y])

    # and rotation
    anet_rot_x = np.arctan2( cd1_2, cd1_1) * (180./np.pi)
    anet_rot_y = np.arctan2(-cd2_1, cd2_2) * (180./np.pi)
    anet_rot = np.average([anet_rot_x, anet_rot_y])

    # add header keywords
    header_wcs['A-PSCALE'] = (anet_pixscale, '[arcsec/pix] pixel scale WCS solution')
    header_wcs['A-PSCALX'] = (anet_pixscale_x, '[arcsec/pix] X-axis pixel scale WCS solution')
    header_wcs['A-PSCALY'] = (anet_pixscale_y, '[arcsec/pix] Y-axis pixel scale WCS solution')

    header_wcs['A-ROT'] = (anet_rot, '[deg] rotation WCS solution (E of N for "up")')
    header_wcs['A-ROTX'] = (anet_rot_x, '[deg] X-axis rotation WCS (E of N for "up")')
    header_wcs['A-ROTY'] = (anet_rot_y, '[deg] Y-axis rotation WCS (E of N for "up")')

    # convert SIP header keywords from Astrometry.net to PV keywords
    # that swarp, scamp (and sextractor) understand using this module
    # from David Shupe: sip_to_pv

    # using the old version of sip_to_pv (before June 2017):
    #status = sip_to_pv(image_out, image_out, log, tpv_format=True)
    #if status == False:
    #    log.error('sip_to_pv failed.')
    #    return 'error'

    # new version (June 2017) of sip_to_pv works on image header
    # rather than header+image (see below); the header is modified in
    # place; compared to the old version this saves an image write
    result = sip_to_pv(header_wcs, tpv_format=True, preserve=False)

    # update input header with [header_wcs]
    header += header_wcs

    # use astropy.WCS to find RA, DEC corresponding to XWIN_IMAGE,
    # YWIN_IMAGE, based on WCS info saved by Astrometry.net in .wcs
    # file (wcsfile). The 3rd parameter to wcs.all_pix2world indicates
    # the pixel coordinate of the frame origin. Using astropy.WCS
    # avoids having to save the new RAs and DECs to file and read them
    # back into python arrays. It provides the same RA and DEC as
    # wcs-xy2rd and also as SExtractor run independently on the WCS-ed
    # image (i.e.  the image_out in this function).
    # N.B.: WCS accepts header objects - gets rid of old warning about
    # axis mismatch, as .wcs files have NAXIS=0, while proper image
    # header files have NAXIS=2
    wcs = WCS(header)
    newra, newdec = wcs.all_pix2world(data_sexcat['XWIN_IMAGE'],
                                      data_sexcat['YWIN_IMAGE'],
                                      1)

    # replace old ra and dec with new ones
    data_sexcat['ALPHAWIN_J2000'] = newra
    data_sexcat['DELTAWIN_J2000'] = newdec
    # replace catalog with RA and DEC columns
    fits.writeto(sexcat, data_sexcat, overwrite=True)
    
    if get_par(C.timing,tel):
        t3 = time.time()

    # check how well the WCS solution just found, compares with an
    # external catalog defined in Constants module
    if os.path.isfile(get_par(C.cal_cat,tel)):

        # use .wcs file to get RA, DEC of central pixel
        xsize = width
        ysize = height
        ra_center, dec_center = wcs.all_pix2world(xsize/2, ysize/2, 1)
        log.info('ra_center: {}, dec_center: {}'.format(ra_center, dec_center))

        # determine cal_cat min and max declination zone of field
        # to determine fits extensions (=zone+1) to read
        fov_half_deg = np.amax([xsize, ysize]) * pixscale / 3600. / 2
        ext_list = get_ext_list (dec_center, fov_half_deg, zone_size=60.)
        log.info('declination zone indices read from calibration catalog: {}'
                 .format(ext_list))

        t4 = time.time()
        # read calibration catalog
        data_cal = read_hdulist (get_par(C.cal_cat,tel), ext_data=ext_list)

        # use function [find_stars] to select stars in calibration
        # catalog that are within the current field-of-view
        mask_field = find_stars(data_cal['ra'], data_cal['dec'],
                                ra_center, dec_center, fov_half_deg, log)
        index_field = np.where(mask_field)[0]
        # N.B.: this [data_cal] array is returned by this function
        # [run_wcs] and also by [run_wcs] so that it can be re-used
        # for the the photometric calibration in [prep_optimal_subtraction]
        data_cal = data_cal[index_field]
        ra_ast = data_cal['ra']
        dec_ast = data_cal['dec']
        mag_ast = data_cal[get_par(C.ast_filter,tel)]

        n_aststars = np.shape(index_field)[0]
        log.info('number of potential astrometric stars in FOV: {}'.format(n_aststars))
        
        # add header keyword(s):
        cal_name = get_par(C.cal_cat,tel).split('/')[-1]
        header['A-CAT-F'] = (cal_name, 'astrometric catalog') 
        header['A-TNAST'] = (n_aststars, 'total number of astrometric stars in FOV')

        # Limit to brightest stars ([nbright] is defined above) in the field
        index_sort_ast = np.argsort(mag_ast)
        ra_ast_bright = ra_ast[index_sort_ast][0:nbright]
        dec_ast_bright = dec_ast[index_sort_ast][0:nbright]
        
        # calculate array of offsets between astrometry comparison
        # stars and any non-saturated SExtractor source; no match or
        # multiple matches will return a zero offset for that star.
        newra_bright = newra[mask_use][index_sort][-nbright:]
        newdec_bright = newdec[mask_use][index_sort][-nbright:]        
        dra_array, ddec_array = calc_offsets (newra_bright, newdec_bright,
                                              ra_ast_bright, dec_ast_bright, log=log)
        # convert to arcseconds
        dra_array *= 3600.
        ddec_array *= 3600.

        n_aststars_used = np.shape(dra_array)[0]
        log.info('number of astrometric stars used: {}'.format(n_aststars_used))
        header['A-NAST'] = (n_aststars_used, 'number of brightest stars used for WCS check')
        header['A-NAMAX'] = (get_par(C.ast_nbright,tel), 'input max. number of stars to use for WCS check')
        
        # calculate means, stds and medians
        dra_mean, dra_std, dra_median = clipped_stats(dra_array, nsigma=5, log=log)
        # ,make_hist=C.make_plots, name_hist=base+'_dra_hist.pdf', hist_xlabel='delta RA [arcsec]')
        ddec_mean, ddec_std, ddec_median = clipped_stats(ddec_array, nsigma=5, log=log)
        # ,make_hist=C.make_plots, name_hist=base+'_ddec_hist.pdf', hist_xlabel='delta DEC [arcsec]')
        
        log.info('dra_mean [arcsec]: {:.3f}, dra_std: {:.3f}, dra_median: {:.3f}'
                 .format(dra_mean, dra_std, dra_median))
        log.info('ddec_mean [arcsec]: {:.3f}, ddec_std: {:.3f}, ddec_median: {:.3f}'
                 .format(ddec_mean, ddec_std, ddec_median))
        
        # add header keyword(s):
        header['A-DRA'] = (dra_median, '[arcsec] dRA median offset to astrom. catalog')
        header['A-DRASTD'] = (dra_std, '[arcsec] dRA sigma (STD) offset')
        header['A-DDEC'] = (ddec_median, '[arcsec] dDEC median offset to astrom. catalog')
        header['A-DDESTD'] = (ddec_std, '[arcsec] dDEC sigma (STD) offset')

        if get_par(C.make_plots,tel):
            dr = np.sqrt(dra_std**2+ddec_std**2)
            limits = 5. * np.array([-dr,dr,-dr,dr])
            mask_nonzero = ((dra_array!=0.) & (ddec_array!=0.))
            label1 = 'dRA={:.3f}$\pm${:.3f}"'.format(dra_median, dra_std)
            label2 = 'dDEC={:.3f}$\pm${:.3f}"'.format(ddec_median, ddec_std)
            result = plot_scatter_hist(dra_array[mask_nonzero], ddec_array[mask_nonzero], limits,
                                       xlabel='delta RA [arcsec]', ylabel='delta DEC [arcsec]',
                                       label=[label1,label2], labelpos=[(0.77,0.9),(0.77,0.85)],
                                       filename=base+'_dRADEC.pdf', title=base)
            
    else:
        log.info('Warning: calibration catalog {} not found!'.format(get_par(C.cal_cat,tel)))
        data_cal = None

    # write image_out including header
    fits.writeto(image_out, data, header, overwrite=True)

    if get_par(C.timing,tel):
        log_timing_memory (t0=t3, label='calculate offset wrt external catalog', log=log)
        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='run_wcs', log=log)

    return data_cal
        

################################################################################

def calc_offsets (ra_sex, dec_sex, ra_ast, dec_ast, log):
    
    if get_par(C.timing,tel): t = time.time()
    log.info('Executing calc_offsets ...')

    # number of astrometry comparison sources
    n_ast = np.shape(ra_ast)[0]

    # prepare output arrays
    dra_array = np.zeros(n_ast)
    ddec_array = np.zeros(n_ast)
    
    # loop astrometry stars and find a match in SExtractor sources
    dist_max = 1./3600
    for i in range(n_ast):
        
        # make a big cut in declination in the SExtractor arrays to
        # speed up distance calculation below
        mask_cut = (np.abs(dec_sex-dec_ast[i])<=dist_max)
        ra_sex_temp = ra_sex[mask_cut]
        dec_sex_temp = dec_sex[mask_cut]
        
        # calculate distances using function [haversine]
        dist = haversine(ra_sex_temp, dec_sex_temp, ra_ast[i], dec_ast[i])
        
        # mask with match(es) between subset of ra_sex and dec_sex and
        # the source in the astrometry catalog
        mask_match = (dist <= dist_max)
        
        if np.sum(mask_match)==1:
            # if there's one match, calculate the RA, DEC offsets;
            # including the sign!
            ddec_array[i] = dec_sex_temp[mask_match]-dec_ast[i]
            # for RA, use the haversine function to get the right offset
            dra_array[i] = haversine(ra_sex_temp[mask_match], dec_sex_temp[mask_match],
                                     ra_ast[i], dec_sex_temp[mask_match])
            # add the sign (haversine provides absolute distances):
            if ra_sex_temp[mask_match] < ra_ast[i]:
                dra_array[i] *= -1
                        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='calc_offsets', log=log)

    return dra_array, ddec_array
    
    
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
    ext2 = fits.BinTableHDU.from_columns([col1])
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

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing ldac2fits ...')

    # read input table and write out primary header and 2nd extension
    columns = []
    with fits.open(cat_ldac) as hdulist:

        if False:
            # to make the fits tables more compact, convert the columns
            # with double precision datatypes ('D') to float32 ('E')
            # except for the RA and DEC columns
            data = hdulist[2].data
            cols = hdulist[2].columns

            for icol, key in enumerate(cols.names):
                format_new = cols.formats[icol]
                # shouldn't it be 'D' and 'E' instead of '1D' and '1E'
                # below?
                if '1D' in cols.formats[icol] and 'J2000' not in key:
                    format_new = '1E'
                    #data[key] = data[key].astype('float32')
                col = fits.Column(name=key, format=format_new, unit=cols.units[icol],
                                  array=data[key])
                columns.append(col)

            hdulist[2] = fits.BinTableHDU.from_columns(columns)

            # overwrite input ldac fits table with double formats
            # converted to float32
            #hdulist.writeto(cat_ldac, overwrite=True)

        # delete VIGNET column
        hdulist[2].data = drop_fields(hdulist[2].data, 'VIGNET')
        # and write regular fits file
        hdulist_new = fits.HDUList([hdulist[0], hdulist[2]])
        hdulist_new.writeto(cat_fits, overwrite=True)
        hdulist_new.close()

    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='ldac2fits', log=log)

    return

    
################################################################################
    
def run_remap(image_new, image_ref, image_out, image_out_size,
              gain, log, config=None, resample='Y', resampling_type='LANCZOS3',
              projection_err=0.001, mask=None, header_only='N',
              resample_suffix='_resamp.fits', resample_dir='.', dtype='float32'):
        
    """Function that remaps [image_ref] onto the coordinate grid of
       [image_new] and saves the resulting image in [image_out] with
       size [image_size].
    """
    
    if '/' in image_new:
        # set resample directory to that of the new image
        resample_dir = '/'.join(image_new.split('/')[:-1])
        
    # for testing of alternative way; for the moment switch on but
    # needs some further testing
    run_alt = True
    
    if get_par(C.timing,tel): t = time.time()
    log.info('Executing run_remap ...')

    header_new = read_hdulist (image_new, ext_header=0)
    header_ref = read_hdulist (image_ref, ext_header=0)
    
    # create .head file with header info from [image_new]
    header_out = header_new[:]
    # copy some keywords from header_ref
    for key in ['exptime', 'satlevel', 'gain', 'ron']:
        value = get_keyvalue(key, header_ref, log)
        try:
            key_name = eval('C.key_'+key)
        except:
            key_name = key.capitalize()
        header_out[key_name] = value

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
           '-PROJECTION_ERR', str(projection_err),
           '-NTHREADS', str(nthreads)]

    if run_alt:
        cmd += ['-COMBINE', 'N',
                '-RESAMPLE_DIR', resample_dir,
                '-RESAMPLE_SUFFIX', resample_suffix,
                '-DELETE_TMPFILES', 'N']

    # log cmd executed
    cmd_str = ' '.join(cmd)
    log.info('SWarp command executed:\n{}'.format(cmd_str))

    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)
    if status != 0:
        log.error('Swarp failed with exit code {}'.format(status))
        return 'error'

    if run_alt:
        image_resample = image_out.replace('_remap.fits', resample_suffix)
        data_resample, header_resample = read_hdulist(image_resample,
                                                      ext_data=0, ext_header=0)
        # SWarp turns integers (mask images) into floats, so making
        # sure that [data_resample] is in the correct format.  All the
        # inputs are fits image names, so have to include an
        # additional [dtype] input.
        if 'int' in dtype:
            data_resample = (data_resample+0.5).astype(dtype)

        # There should be just a shift between the resampled image and
        # the output image in case of COMBINE='Y', which is just
        # determined by which input pixels will end up in the output
        # image. Determine the "0,0" pixel in the output image that
        # corresponds to "0,0" in the input image:
        ra0, dec0 = WCS(header_resample).all_pix2world(0, 0, 0)
        x0, y0 = WCS(header_out).all_world2pix(ra0, dec0, 0)
        x0, y0 = int(x0+0.5), int(y0+0.5)

        # resampled image is a bit smaller than the original image
        # size
        ysize_resample, xsize_resample = np.shape(data_resample)
        # create zero output image with correct dtype
        data_remap = np.zeros(image_out_size, dtype=dtype)
        # and place resampled image in output image
        data_remap[y0:y0+ysize_resample,
                   x0:x0+xsize_resample] = data_resample

        # write to fits [image_out] with correct header
        if not os.path.isfile(image_out) or get_par(C.redo,tel):
            fits.writeto(image_out, data_remap, header_out, overwrite=True)
    
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='run_remap', log=log)

    return

    
################################################################################

def get_fwhm (cat_ldac, fraction, log, class_sort=False, get_elong=False):

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
 
    if get_par(C.timing,tel): t = time.time()
    log.info('Executing get_fwhm ...')

    data = read_hdulist (cat_ldac, ext_data=2)

    # these arrays correspond to objecst with flag==0 and flux_auto>0.
    # add a S/N requirement
    index = ((data['FLAGS']==0) & (data['FLUX_AUTO']>0.) &
             (data['FLUXERR_AUTO']>0.) & (data['FLUX_AUTO']/data['FLUXERR_AUTO']>20.))
    fwhm = data['FWHM_IMAGE'][index]
    class_star = data['CLASS_STAR'][index]
    flux_auto = data['FLUX_AUTO'][index]
    mag_auto = -2.5*np.log10(flux_auto)
    if get_elong:
        elong = data['ELONGATION'][index]
    
    if class_sort:
        # sort by CLASS_STAR
        index_sort = np.argsort(class_star)
    else:
        # sort by FLUX_AUTO
        index_sort = np.argsort(flux_auto)

    # select fraction of targets
    index_select = np.arange(-np.int(len(index_sort)*fraction+0.5),-1)
    fwhm_select = fwhm[index_sort][index_select] 
    if get_elong:
        elong_select = elong[index_sort][index_select] 
            
    # print warning if few stars are selected
    if len(fwhm_select) < 10:
        log.info('WARNING: fewer than 10 objects are selected for FWHM determination')
    
    # determine mean, median and standard deviation through sigma clipping
    fwhm_mean, fwhm_std, fwhm_median = clipped_stats(fwhm_select, log=log)
    if get_par(C.verbose,tel):
        log.info('catalog: ' + cat_ldac)
        log.info('fwhm_mean: {:.3f}, fwhm_median: {:.3f}, fwhm_std: {:.3f}'.
                 format(fwhm_mean, fwhm_median, fwhm_std))
    if get_elong:
        # determine mean, median and standard deviation through sigma clipping
        elong_mean, elong_std, elong_median = clipped_stats(elong_select, log=log)
        if get_par(C.verbose,tel):
            log.info('elong_mean: {:.3f}, elong_median: {:.3f}, elong_std: {:.3f}'.
                     format(elong_mean, elong_median, elong_std))
            
        
    if get_par(C.make_plots,tel):

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
        plt.axis((0,min(x2,15),y2,y1))
        plt.xlabel('FWHM (pixels)')
        plt.ylabel('MAG_AUTO')
        plt.title('median FWHM: {:.2f} $\pm$ {:.2f} pixels'.format(fwhm_median, fwhm_std))
        plt.savefig(cat_ldac.replace('.fits','')+'_fwhm.pdf')
        plt.title(cat_ldac)
        if get_par(C.show_plots,tel): plt.show()
        plt.close()

        if get_elong:

            elong = data['ELONGATION'][index]

            plt.plot(elong, mag_auto, 'bo', markersize=5, markeredgecolor='k')
            x1,x2,y1,y2 = plt.axis()
            plt.plot(elong_select, mag_auto_select, 'go', markersize=5, markeredgecolor='k')
            plt.plot([elong_median, elong_median], [y2,y1], color='red')
            elong_line = elong_median-elong_std
            plt.plot([elong_line, elong_line], [y2,y1], 'r--')
            elong_line = elong_median+elong_std
            plt.plot([elong_line, elong_line], [y2,y1], 'r--')
            plt.axis((0,min(x2,5),y2,y1))
            plt.xlabel('ELONGATION (A/B)')
            plt.ylabel('MAG_AUTO')
            plt.savefig(cat_ldac.replace('.fits','')+'_elongation.pdf')
            if get_par(C.show_plots,tel): plt.show()
            plt.close()
            
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='get_fwhm', log=log)

    if get_elong:
        return fwhm_median, fwhm_std, elong_median, elong_std
    else:
        return fwhm_median, fwhm_std
    

################################################################################

def update_vignet_size (sex_par_in, sex_par_out, imtype, log):

    if imtype=="ref":
        # set vignet size to the value defined in [C.size_vignet_ref]
        size_vignet = get_par(C.size_vignet_ref,tel)
    else:
        
        # in case [C.psf_sampling] is set to zero, scale the size of the
        # VIGNET output in the output catalog with 2*[C.psf_radius]*[fwhm]
        # where fwhm is taken to be the largest of global parameters
        # [fwhm_new] and [fwhm_ref]
        if get_par(C.psf_sampling,tel) == 0.:
            fwhm_vignet = np.amax([fwhm_new, fwhm_ref])
            size_vignet = np.int(np.ceil(2.*get_par(C.psf_radius,tel)*fwhm_vignet))
            # make sure it's odd
            if size_vignet % 2 == 0: size_vignet += 1
            # provide a warning if it's larger than the reference image
            # size
            if size_vignet > get_par(C.size_vignet_ref,tel):
                log.info('Warning: VIGNET size of {} is larger than ref image value of {}'
                         .format(size_vignet, get_par(C.size_vignet_ref,tel)))
        else:
            # otherwise set it to the value defined for the ref image
            size_vignet = get_par(C.size_vignet_ref,tel)

    # append the VIGNET size to the SExtractor parameter file
    # [sex_par_in] and write it to a temporary file [sex_par_out] to
    # be used by SExtractor
    size_vignet_str = str((size_vignet, size_vignet))
    with open(sex_par_in, 'rt') as file_in:
        with open(sex_par_out, 'wt') as file_out:
            for line in file_in:
                file_out.write(line)
            file_out.write('VIGNET'+size_vignet_str+'\n')
        if get_par(C.verbose,tel):
            log.info('VIGNET size: ' + str(size_vignet_str))

    return size_vignet


################################################################################

def run_sextractor(image, cat_out, file_config, file_params, pixscale, log, header,
                   fit_psf=False, return_fwhm_elong=True, fraction=1.0, fwhm=5.0, save_bkg=True,
                   update_vignet=True, imtype=None, mask=None):

    """Function that runs SExtractor on [image], and saves the output
       catalog in [outcat], using the configuration file [file_config]
       and the parameters defining the output recorded in the
       catalogue [file_params]. If [fit_psf] is True, SExtractor will
       perform PSF fitting photometry using the PSF built by PSFex. If
       [return_fwhm_elong] is True, an estimate of the image median
       FWHM and ELONGATION and their standard deviations are returned
       using SExtractor's seeing estimate of the detected sources; if
       False, it will return zeros. If [fraction] is less than the
       default 1.0, SExtractor will be run on a fraction [fraction] of
       the area of the full image. Sextractor will use the input value
       [fwhm], which is important for the star-galaxy
       classification. If [save-bkg] is True, the background image,
       its standard deviation and the -OBJECTS image
       (background-subtracted image with all objects masked with zero
       values), all produced by SExtractor, will be saved. If
       [C.bkg_method] is not set to 1 (use SExtractor's background),
       then improve the estimates of the background and its standard
       deviation.

    """

    if get_par(C.timing,tel): t = time.time()
    log.info('Executing run_sextractor ...')

    base = image.replace('.fits','')
    
    # if fraction less than one, run SExtractor on specified fraction of
    # the image
    if fraction < 1.:

        # read in input image and header
        data, header = read_hdulist (image, ext_data=0, ext_header=0)
        # get input image size from header
        ysize, xsize = read_header(header, ['naxis2', 'naxis1'], log)
        
        # determine cutout from [fraction]
        center_x = np.int(xsize/2+0.5)
        center_y = np.int(ysize/2+0.5)
        halfsize_x = np.int((xsize * np.sqrt(fraction))/2.+0.5)
        halfsize_y = np.int((ysize * np.sqrt(fraction))/2.+0.5)
        data_fraction = data[center_y-halfsize_y:center_y+halfsize_y,
                             center_x-halfsize_x:center_x+halfsize_x]

        # write small image to fits
        image_fraction = base+'_fraction.fits'
        fits.writeto(image_fraction, data_fraction.astype('float32'), header, overwrite=True)

        # make image point to image_fraction
        image = image_fraction
        cat_out = base+'_cat_fraction.fits'


    # the input fwhm determines the SEEING_FWHM (important for
    # star/galaxy separation) and the radii of the apertures used for
    # aperture photometry. If fwhm is not provided as input, it will
    # assume fwhm=5.0 pixels.
    fwhm = float('{:.2f}'.format(fwhm))
    # determine seeing
    seeing = fwhm * pixscale
    # prepare aperture diameter string to provide to SExtractor 
    apphot_diams = np.array(get_par(C.apphot_radii,tel)) * 2 * fwhm
    apphot_diams_str = ",".join(apphot_diams.astype(str))

    # update size of VIGNET
    if update_vignet:
        size_vignet = update_vignet_size (file_params, file_params+'_temp', imtype, log)
        file_params = file_params+'_temp'
        # write vignet_size to header
        header['S-VIGNET'] = (size_vignet, '[pix] size square VIGNET used in SExtractor')
        
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
           '-BACK_SIZE', str(get_par(C.bkg_boxsize,tel)), '-BACK_FILTERSIZE',
           str(get_par(C.bkg_filtersize,tel)),
           '-NTHREADS', str(nthreads), '-FILTER_NAME', get_par(C.sex_det_filt,tel),
           '-STARNNW_NAME', get_par(C.cfg_dir,tel)+'default.nnw']

    # add commands to produce BACKGROUND, BACKGROUND_RMS and
    # background-subtracted image with all pixels where objects were
    # detected set to zero (-OBJECTS). These are used to build an
    # improved background map. 
    if save_bkg:
        fits_bkg = base+'_bkg.fits'
        fits_bkg_std = base+'_bkg_std.fits'
        fits_objmask = base+'_objmask.fits'
        cmd += ['-CHECKIMAGE_TYPE', 'BACKGROUND,BACKGROUND_RMS,-OBJECTS',
                '-CHECKIMAGE_NAME', fits_bkg+','+fits_bkg_std+','+fits_objmask]
    
    # in case of fraction being less than 1: only care about higher S/N detections
    if fraction < 1.: cmd += ['-DETECT_THRESH', str(get_par(C.fwhm_detect_thresh,tel))]
    
    # provide PSF file from PSFex
    if fit_psf: cmd += ['-PSF_NAME', base+'_psf.fits']

    # provide mask image if not None
    if mask is not None:
        log.info('mask: {}'.format(mask))
        cmd += ['-FLAG_IMAGE', mask, '-FLAG_TYPE', 'OR']
        
    # log cmd executed
    cmd_str = ' '.join(cmd)
    log.info('SExtractor command executed:\n{}'.format(cmd_str))
        
    # run command
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)

    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='run_sextractor before get_back', log=log)

    # improve background estimate if [C.bkg_method] not set to 1 (= use
    # background determined by SExtractor)
    if save_bkg and get_par(C.bkg_method,tel) != 1:

        # read in SExtractor's object mask created above
        data_objmask = read_hdulist (fits_objmask, ext_data=0)
        objmask = (data_objmask==0)
        del data_objmask
       
        # read in input image
        data = read_hdulist (image, ext_data=0, dtype='float32')

        # construct background image using [get_back]; in the case of
        # the reference image these data need to refer to the image
        # before remapping
        if get_par(C.bkg_method,tel)==2:
            data_bkg_mini, data_bkg_std_mini = get_back(data, objmask, log)
           
            # write these filtered meshes to fits
            fits.writeto(base+'_bkg_mini.fits', data_bkg_mini, overwrite=True)
            fits.writeto(base+'_bkg_std_mini.fits', data_bkg_std_mini, overwrite=True)
            # and update headers with [C.bkg_boxsize]
            fits.setval(base+'_bkg_mini.fits', 'BKG_SIZE', value=get_par(C.bkg_boxsize,tel))
            fits.setval(base+'_bkg_std_mini.fits', 'BKG_SIZE', value=get_par(C.bkg_boxsize,tel))

            # now use function [mini2back] to turn filtered mesh of median
            # and std of backgroun regions into full background image and
            # its standard deviation
            data_bkg = mini2back (data_bkg_mini, data.shape, log,
                                  order_interp=2, bkg_boxsize=get_par(C.bkg_boxsize,tel))
            data_bkg_std = mini2back (data_bkg_std_mini, data.shape, log,
                                      order_interp=1, bkg_boxsize=get_par(C.bkg_boxsize,tel))


        # write the improved background and standard deviation to fits
        # overwriting the fits images produced by SExtractor
        fits.writeto(fits_bkg, data_bkg, overwrite=True)
        fits.writeto(fits_bkg_std, data_bkg_std, overwrite=True)

                
    if return_fwhm_elong:
        # get estimate of seeing and elongation from output catalog
        fwhm, fwhm_std, elong, elong_std = get_fwhm(
            cat_out, get_par(C.fwhm_frac,tel), log,
            class_sort=get_par(C.fwhm_class_sort,tel), get_elong=True)
        
    else:
        fwhm = 0.
        fwhm_std = 0.
        elong = 0.
        elong_std = 0.
        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='run_sextractor', log=log)

    return fwhm, fwhm_std, elong, elong_std


################################################################################

def run_psfex(cat_in, file_config, cat_out, imtype, log):
    
    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""

    if get_par(C.timing,tel): t = time.time()

    if imtype=='new':
        base = base_new
    else:
        base = base_ref

    # use function [get_samp_PSF_config_size] to determine [psf_samp]
    # and [psf_size_config] required to run PSFEx
    psf_samp, psf_size_config = get_samp_PSF_config_size()
    psf_size_config_str = str(psf_size_config)+','+str(psf_size_config)

    if get_par(C.verbose,tel):
        log.info('psf_size_config: ' + str(psf_size_config))

    # get FWHM and ELONGATION to limit the PSFex configuration
    # parameters SAMPLE_FWHMRANGE and SAMPLE_MAXELLIP
    #fwhm, fwhm_std, elong, elong_std = get_fwhm(cat_in, 0.05, class_sort=False,
    #                                                      get_elong=True)
    #print ('fwhm, fwhm_std, elong, elong_std', fwhm, fwhm_std, elong, elong_std)
    #sample_fwhmrange = str(fwhm-fwhm_std)+','+str(fwhm+fwhm_std)
    #print ('sample_fwhmrange', sample_fwhmrange)
    #maxellip = (elong+3.*elong_std-1)/(elong+3.*elong_std+1)
    #maxellip_str= str(np.amin([maxellip, 1.]))
    #print ('maxellip_str', maxellip_str)

    # Need to check whether the VIGNET size from the SExtractor run is
    # sufficient large compared to [psf_samp] and [psf_size_config].
    

    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config_str, '-PSF_SAMPLING', str(psf_samp),
           '-SAMPLE_MINSN', str(get_par(C.psf_stars_s2n_min,tel)),
           '-NTHREADS', str(nthreads)]
    #       '-SAMPLE_FWHMRANGE', sample_fwhmrange,
    #       '-SAMPLE_MAXELLIP', maxellip_str]

    if get_par(C.make_plots,tel):
        cmd += ['-CHECKPLOT_TYPE', 'FWHM, ELLIPTICITY, COUNTS, COUNT_FRACTION, CHI2, RESIDUALS',
                '-CHECKPLOT_DEV', 'PS',
                '-CHECKPLOT_ANTIALIAS', 'N',
                '-CHECKPLOT_NAME',
                'psfex_fwhm, psfex_ellip, psfex_counts, psfex_countfrac, psfex_chi2, psfex_resi']
        cmd += ['-CHECKIMAGE_TYPE', 'CHI,PROTOTYPES,SAMPLES,RESIDUALS,SNAPSHOTS,BASIS',
                '-CHECKIMAGE_NAME',
                'psfex_chi, psfex_proto, psfex_samp, psfex_resi, psfex_snap, psfex_basis']

    # log cmd executed
    cmd_str = ' '.join(cmd)
    log.info('PSFEx command executed:\n{}'.format(cmd_str))
        
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)

    # standard output of PSFEx is .psf; change this to _psf.fits
    psf_in = cat_in.replace('.fits', '.psf')
    psf_out = base+'_psf.fits'
    os.rename (psf_in, psf_out)

    cwd = os.getcwd()
    file_list = glob.glob('{}/psfex*'.format(cwd))
    for name in file_list:
        # DP: replaced tabs for this section with 8 spaces (gave
        # inconsistancy error in Tabs and spaces)
        short = name.split('/')[-1]
        prefix = '_'.join(short.split('_')[0:2])
        name_new = short.replace('ldac_4psfex', prefix)
        name_new = name_new.replace(prefix+'_','')
        name_new = '{}/{}'.format(cwd, name_new)
        log.info ('name: {}, name_new: {}'.format(name, name_new))
        os.rename (name, name_new)
        
        
    if get_par(C.timing,tel):
        log_timing_memory (t0=t, label='run_psfex', log=log)

    return

        
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
    if get_par(C.psf_sampling,tel) == 0:
        psf_samp = get_par(C.psf_samp_fwhmfrac,tel) * fwhm_samp
    else:
        psf_samp = get_par(C.psf_sampling,tel)

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
    psf_size_config = 2. * get_par(C.psf_radius,tel) * fwhm_samp / psf_samp

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
    hsize = int(xsize/2)
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

# function to run ZOGY on subimages
def zogy_subloop (nsub, data_ref, data_new,
                  psf_ref, psf_new,
                  data_ref_bkg, data_new_bkg,
                  data_ref_bkg_std, data_new_bkg_std,
                  readnoise_ref, readnoise_new,
                  fratio_sub, dx_sub, dy_sub, log=None):
    
    if get_par(C.timing,tel) and log is not None:
        t = time.time()
    
    if get_par(C.verbose,tel) and log is not None:
        log.info(' ')
        log.info('nsub: {}'.format(nsub+1))
        log.info('----------')

    # option 1: set f_ref to unity
    #f_ref = 1.
    #f_new = f_ref * np.mean(fratio_sub)
    # option 2: set f_new to unity
    fn = 1.
    fr = fn / fratio_sub[nsub]
    dx = dx_sub[nsub]
    dy = dy_sub[nsub]
        
    N = data_new[nsub]
    R = data_ref[nsub]
    Pn = psf_new[nsub]
    Pr = psf_ref[nsub]
            
    # before running zogy, pixels with zero values in ref need to
    # be set to zero in new as well, and vice versa, to avoid
    # subtracting non-overlapping image part
    mask_zero = ((R==0.) | (N==0.))
    N[mask_zero] = 0.
    R[mask_zero] = 0.
    
    # determine variance images before background is subtracted
    # N.B.: these are single images (i.e. not a cube) the size of
    # a subimage, so does not need the [nsub] index
    Vn = data_new[nsub] + readnoise_new**2
    Vr = data_ref[nsub] + readnoise_ref**2

    if np.sum(~mask_zero) != 0:
    
        # subtract the background where images are nonzero
        N[~mask_zero] -= data_new_bkg[nsub][~mask_zero]
        R[~mask_zero] -= data_ref_bkg[nsub][~mask_zero]
    
        # determine subimage s_new and s_ref from background RMS
        # images
        sn = np.median(data_new_bkg_std[nsub][~mask_zero])
        sr = np.median(data_ref_bkg_std[nsub][~mask_zero])

    else:

        log.warn('empty subimage; large shift between new and ref image?')
        
        sn = 1
        sr = 1
        
        
    if get_par(C.verbose,tel) and log is not None:
        log.info('fn: {}, fr: {}'.format(fn, fr))
        log.info('dx: {}, dy: {}'.format(dx, dy))
        log.info('sn: {}, sr: {}'.format(sn, sr))

    return run_ZOGY(R,N,Pr,Pn,sr,sn,fr,fn,Vr,Vn,dx,dy, log=log)


################################################################################
    
def run_ZOGY(R,N,Pr,Pn,sr,sn,fr,fn,Vr,Vn,dx,dy, log=None):

    if get_par(C.timing,tel) and log is not None:
        t = time.time()    

    # boolean [use_fftw] determines if initial forward fft2 is
    # initialized using pyfftw or not; for some reason this speeds up
    # all subsequent calls to convenience function [fft.fft2]
    # significantly, with a loop time of 0.2s instead of 0.3s.  If
    # nthreads>1 then this speed-up becomes less dramatic, e.g. with 4
    # threads, the loop time is 0.2s without [use_fftw] and 0.17s with
    # [use_fftw].
    use_fftw = True
    if use_fftw:
        R = R.astype('complex64')
        R_hat = np.zeros_like(R)
        fft_forward = pyfftw.FFTW(R, R_hat, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ),
                                  threads=nthreads, planning_timelimit=None)
        fft_forward()
    else:
        R_hat = fft.fft2(R, threads=nthreads)

    N_hat = fft.fft2(N, threads=nthreads)

    Pn_hat = fft.fft2(Pn, threads=nthreads)
    #if get_par(C.psf_clean_factor,tel)!=0:
    #clean Pn_hat
    #Pn_hat = clean_psf(Pn_hat, get_par(C.psf_clean_factor,tel))
    Pn_hat2_abs = np.abs(Pn_hat**2)
    
    Pr_hat = fft.fft2(Pr, threads=nthreads)
    #if get_par(C.psf_clean_factor,tel)!=0:
    # clean Pr_hat
    #Pr_hat = clean_psf(Pr_hat, get_par(C.psf_clean_factor,tel))
    Pr_hat2_abs = np.abs(Pr_hat**2)

    sn2 = sn**2
    sr2 = sr**2
    fn2 = fn**2
    fr2 = fr**2
    fD = (fr*fn) / np.sqrt(sn2*fr2+sr2*fn2)
    
    denominator = (sn2*fr2)*Pr_hat2_abs + (sr2*fn2)*Pn_hat2_abs
        
    D_hat = (fr*(Pr_hat*N_hat) - fn*(Pn_hat*R_hat)) / np.sqrt(denominator)

    if use_fftw:
        D = np.zeros_like(D_hat)
        fft_backward = pyfftw.FFTW(D_hat, D, axes=(0,1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ),
                                   threads=nthreads, planning_timelimit=None)
        fft_backward()
        D = np.real(D) / fD
    else:
        D = np.real(fft.ifft2(D_hat, threads=nthreads)) / fD
    
    P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)
    #P_D = np.real(fft.ifft2(P_D_hat, threads=nthreads))
    
    S_hat = fD*D_hat*np.conj(P_D_hat)
    S = np.real(fft.ifft2(S_hat, threads=nthreads))

    # alternative way to calculate S
    #S_hat = (fn*fr2*Pr_hat2_abs*np.conj(Pn_hat)*N_hat -
    #         fr*fn2*Pn_hat2_abs*np.conj(Pr_hat)*R_hat) / denominator
    #S = np.real(fft.ifft2(S_hat), threads=nthreads)

    # PMV 2017/01/18: added following part based on Eqs. 25-31
    # from Barak's paper
    kr_hat = (fr*fn2)*np.conj(Pr_hat)*Pn_hat2_abs / denominator
    kr = np.real(fft.ifft2(kr_hat, threads=nthreads))
    kr2 = kr**2
    kr2_hat = fft.fft2(kr2, threads=nthreads)

    kn_hat = (fn*fr2)*np.conj(Pn_hat)*Pr_hat2_abs / denominator
    kn = np.real(fft.ifft2(kn_hat, threads=nthreads))
    kn2 = kn**2
    kn2_hat = fft.fft2(kn2, threads=nthreads)

    Vr_hat = fft.fft2(Vr, threads=nthreads)
    Vn_hat = fft.fft2(Vn, threads=nthreads)

    VSr = np.real(fft.ifft2(Vr_hat*kr2_hat, threads=nthreads))
    VSn = np.real(fft.ifft2(Vn_hat*kn2_hat, threads=nthreads))

    dx2 = dx**2
    dy2 = dy**2
    # and calculate astrometric variance
    Sn = np.real(fft.ifft2(kn_hat*N_hat, threads=nthreads))
    dSndy = Sn - np.roll(Sn,1,axis=0)
    dSndx = Sn - np.roll(Sn,1,axis=1)
    VSn_ast = dx2 * dSndx**2 + dy2 * dSndy**2
    
    Sr = np.real(fft.ifft2(kr_hat*R_hat, threads=nthreads))
    dSrdy = Sr - np.roll(Sr,1,axis=0)
    dSrdx = Sr - np.roll(Sr,1,axis=1)
    VSr_ast = dx2 * dSrdx**2 + dy2 * dSrdy**2

    if get_par(C.display,tel):
        base = base_newref
        fits.writeto(base+'_Pn_hat.fits', np.real(Pn_hat).astype('float32'), overwrite=True)
        fits.writeto(base+'_Pr_hat.fits', np.real(Pr_hat).astype('float32'), overwrite=True)
        fits.writeto(base+'_kr.fits', np.real(kr).astype('float32'), overwrite=True)
        fits.writeto(base+'_kn.fits', np.real(kn).astype('float32'), overwrite=True)
        fits.writeto(base+'_Sr.fits', Sr.astype('float32'), overwrite=True)
        fits.writeto(base+'_Sn.fits', Sn.astype('float32'), overwrite=True)
        fits.writeto(base+'_VSr.fits', VSr.astype('float32'), overwrite=True)
        fits.writeto(base+'_VSn.fits', VSn.astype('float32'), overwrite=True)
        fits.writeto(base+'_VSr_ast.fits', VSr_ast.astype('float32'), overwrite=True)
        fits.writeto(base+'_VSn_ast.fits', VSn_ast.astype('float32'), overwrite=True)

    # and finally S_corr
    V_S = VSr + VSn
    V_ast = VSr_ast + VSn_ast
    V = V_S + V_ast
    #S_corr = S / np.sqrt(V)
    # make sure there's no division by zero
    S_corr = np.copy(S)
    #S_corr[V>0] /= np.sqrt(V[V>0])
    mask = (V>0)
    S_corr[mask] /= np.sqrt(V[mask])

    # PMV 2017/03/05: added following PSF photometry part based on
    # Eqs. 41-43 from Barak's paper
    F_S = fn2*fr2*np.sum((Pn_hat2_abs*Pr_hat2_abs) / denominator)
    # divide by the number of pixels in the images (related to do
    # the normalization of the ffts performed)
    F_S /= R.size
    # an alternative (slower) way to calculate the same F_S:
    #F_S_array = fft.ifft2((fn2*Pn_hat2_abs*fr2*Pr_hat2_abs) / denominator)
    #F_S = F_S_array[0,0]

    alpha = S / F_S
    alpha_std = np.zeros(alpha.shape)
    #alpha_std[V_S>=0] = np.sqrt(V_S[V_S>=0]) / F_S
    mask = (V_S>=0)
    alpha_std[mask] = np.sqrt(V_S[mask]) / F_S
    
    if get_par(C.timing,tel) and log is not None:
        log_timing_memory (t0=t, label='run_ZOGY', log=log)
    
    return D, S, S_corr, alpha, alpha_std


################################################################################
    
def run_ZOGY_backup(R,N,Pr,Pn,sr,sn,fr,fn,Vr,Vn,dx,dy, log=None):

    if get_par(C.timing,tel) and log is not None:
        t = time.time()    

    R_hat = fft.fft2(R)
    N_hat = fft.fft2(N)
    Pn_hat = fft.fft2(Pn)
    #if get_par(C.psf_clean_factor,tel)!=0:
    #clean Pn_hat
    #Pn_hat = clean_psf(Pn_hat, get_par(C.psf_clean_factor,tel))
    Pn_hat2_abs = np.abs(Pn_hat**2)

    Pr_hat = fft.fft2(Pr)
    #if get_par(C.psf_clean_factor,tel)!=0:
    # clean Pr_hat
    #Pr_hat = clean_psf(Pr_hat, get_par(C.psf_clean_factor,tel))
    Pr_hat2_abs = np.abs(Pr_hat**2)

    sn2 = sn**2
    sr2 = sr**2
    fn2 = fn**2
    fr2 = fr**2
    fD = fr*fn / np.sqrt(sn2*fr2+sr2*fn2)
    
    denominator = sn2*fr2*Pr_hat2_abs + sr2*fn2*Pn_hat2_abs
        
    D_hat = (fr*Pr_hat*N_hat - fn*Pn_hat*R_hat) / np.sqrt(denominator)
    D = np.real(fft.ifft2(D_hat)) / fD
    
    P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)
    #P_D = np.real(fft.ifft2(P_D_hat))
    
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

    #if get_par(C.display,tel):
    #    base = base_newref
    #    fits.writeto(base+'_Pn_hat.fits', np.real(Pn_hat).astype('float32'), overwrite=True)
    #    fits.writeto(base+'_Pr_hat.fits', np.real(Pr_hat).astype('float32'), overwrite=True)
    #    fits.writeto(base+'_kr.fits', np.real(kr).astype('float32'), overwrite=True)
    #    fits.writeto(base+'_kn.fits', np.real(kn).astype('float32'), overwrite=True)
    #    fits.writeto(base+'_Sr.fits', Sr.astype('float32'), overwrite=True)
    #    fits.writeto(base+'_Sn.fits', Sn.astype('float32'), overwrite=True)
    #    fits.writeto(base+'_VSr.fits', VSr.astype('float32'), overwrite=True)
    #    fits.writeto(base+'_VSn.fits', VSn.astype('float32'), overwrite=True)
    #    fits.writeto(base+'_VSr_ast.fits', VSr_ast.astype('float32'), overwrite=True)
    #    fits.writeto(base+'_VSn_ast.fits', VSn_ast.astype('float32'), overwrite=True)

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
    # an alternative (slower) way to calculate the same F_S:
    #F_S_array = fft.ifft2((fn2*Pn_hat2_abs*fr2*Pr_hat2_abs) / denominator)
    #F_S = F_S_array[0,0]

    alpha = S / F_S
    alpha_std = np.zeros(alpha.shape)
    alpha_std[V_S>=0] = np.sqrt(V_S[V_S>=0]) / F_S

    if get_par(C.timing,tel) and log is not None:
        log_timing_memory (t0=t, label='run_ZOGY', log=log)
    
    return D, S, S_corr, alpha, alpha_std


################################################################################

def log_timing_memory(t0, label, log=None):
  
    # ru_maxrss is in units of kilobytes on Linux; however, this seems
    # to be OS dependent as on mac os maverick it is in units of
    # bytes; see manpages of "getrusage"
    mem_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6

    if log is not None:
        log.info ('wall-time spent in {}: {:.4f} s'.format(label, time.time()-t0))
        log.info ('peak memory used in {}: {:.4f} GB'.format(label, mem_GB))
    else:
        print ('wall-time spent in {}: {:.4f} s'.format(label, time.time()-t0))
        print ('peak memory used in {}: {:.4f} GB'.format(label, mem_GB))
        

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
    parser.add_argument('--new_fits_mask', default=None, help='filename of new image mask')
    parser.add_argument('--ref_fits_mask', default=None, help='filename of ref image mask')
    parser.add_argument('--set_file', default='Settings.set_zogy', help='name of settings file')
    parser.add_argument('--log', default=None, help='help')
    parser.add_argument('--verbose', default=None, help='verbose')
    parser.add_argument('--nthreads', default=1, type=int, help='number of threads to use')
    parser.add_argument('--telescope', default=None, help='telescope')
    
    # replaced [global_pars] function with importing [set_file] as C;
    # all former global parameters are now referred to as C.[parameter
    # name]. This importing is done inside [optimal_subtraction] in
    # case it is not called from the command line.
    args = parser.parse_args()
    optimal_subtraction(args.new_fits, args.ref_fits, args.new_fits_mask, args.ref_fits_mask,
                        args.set_file, args.log, args.verbose, args.nthreads, args.telescope)

if __name__ == "__main__":
    main()
