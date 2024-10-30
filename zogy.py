import os
import sys
import subprocess
import glob
import tempfile
import shutil
import argparse
import importlib
import resource
import traceback
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
from functools import partial
import math
import collections
import itertools
import gc
import numbers
import psutil


import multiprocessing as mp
# define mp start method through context; this same method is used in
# blackbox, buildref and force_phot
mp_ctx = mp.get_context('spawn')


# set up log
import logging
import time
logfmt = ('%(asctime)s.%(msecs)03d [%(levelname)s, %(process)s] %(message)s '
          '[%(funcName)s, line %(lineno)d]')
datefmt = '%Y-%m-%dT%H:%M:%S'
logging.basicConfig(level='INFO', format=logfmt, datefmt=datefmt)
logFormatter = logging.Formatter(logfmt, datefmt)
logging.Formatter.converter = time.gmtime #convert time in logger to UTC
log = logging.getLogger()


# setting environment variable OMP_NUM_THREADS to number of threads,
# use value from environment variable SLURM_CPUS_PER_TASK if it is
# defined, otherwise set to 1 if OMP_NUM_THREADS was not defined
# already; needs to be done before numpy is imported
if os.environ.get('SLURM_CPUS_PER_TASK') is not None:
    os.environ['OMP_NUM_THREADS'] = os.environ.get('SLURM_CPUS_PER_TASK')
elif os.environ.get('OMP_NUM_THREADS') is None:
    os.environ['OMP_NUM_THREADS'] = '1'


import numpy as np
from numpy.polynomial.polynomial import polyvander2d, polygrid2d, polyval2d

from numpy.lib.recfunctions import append_fields, drop_fields
#from numpy.lib.recfunctions import rename_fields, stack_arrays

import astropy.io.fits as fits
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.table import Table, vstack, unique
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
# due to regular problems with downloading default IERS file (needed
# to compute UTC-UT1 corrections for e.g. sidereal time computation),
# Steven created a mirror of this file in a google storage bucket
#from astropy.utils import iers
#iers.conf.iers_auto_url = \
#    'https://storage.googleapis.com/blackbox-auxdata/timing/finals2000A.all'
#iers.conf.iers_auto_url_mirror = \
#    'http://maia.usno.navy.mil/ser7/finals2000A.all'

import fitsio

from scipy import ndimage
from scipy import interpolate
from scipy.spatial.distance import cdist

from skimage import restoration, measure
from skimage.util.shape import view_as_windows
#from skimage.util.shape import view_as_blocks


#import scipy.fft as fft
#import numpy.fft as fft
# these are important to speed up the FFTs
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()

# pyfftw.interfaces.cache.set_keepalive_time(1.)
# for PSF fitting - see https://lmfit.github.io/lmfit-py/index.html
from lmfit import minimize, Parameters, fit_report

# see https://github.com/stargaser/sip_tpv (version June 2017):
# download from GitHub and "python setup.py install --user" for local
# install or "sudo python setup.py install" for system install
from sip_tpv import sip_to_pv

import matplotlib
# matplotlib.use('PDF')
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# healpy
import healpy as hp

# in case google cloud is being used
from google.cloud import storage

# from memory_profiler import profile
# import objgraph

__version__ = '1.3.7'


################################################################################

# functions to let blackbox define multiprocessing instances of
# Lock(), Queue() and Pool() while using mp start method defined at
# top of this module

def get_mp_Lock():
    return mp_ctx.Lock()

def get_mp_Queue():
    return mp_ctx.Queue()

def get_mp_Pool(nproc=1):
    return mp_ctx.Pool(nproc)


################################################################################

#import linecache
#import tracemalloc

# use:
#tracemalloc.start()
#snapshot = tracemalloc.take_snapshot()
#display_top(snapshot)
#tracemalloc.stop()

# see https://docs.python.org/3/library/tracemalloc.html
def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s; file=%s; line=%s; size=%.3f MB"
              % (index, filename, frame.lineno, stat.size / 1024**2))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.3f MB" % (len(other), size / 1024**2))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.3f MB" % (total / 1024**2))


################################################################################

# @profile
def optimal_subtraction(new_fits=None,      ref_fits=None,
                        new_fits_mask=None, ref_fits_mask=None,
                        set_file='set_zogy', logfile=None,
                        redo_new=None, redo_ref=None,
                        verbose=None, nthreads=None, telescope='ML1',
                        keep_tmp=None):


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

    """

    global tel
    tel = telescope
    log.info ('telescope:       {}'.format(tel))


    # if input [nthreads] is not set, use the value of environment
    # variable 'OMP_NUM_THREADS' if that is present, otherwise set it
    # to 1
    if nthreads is None:
        nthreads = int(os.environ.get('OMP_NUM_THREADS', 1))

    log.info ('nthreads:        {}'.format(nthreads))


    # import settings file as set_zogy such that all parameters defined there
    # can be referred to as set_zogy.[parameter]
    global set_zogy
    set_zogy = importlib.import_module(set_file)

    # if verbosity is provided through input parameter [verbose], it
    # will overwrite the corresponding setting in the settings file
    # (set_zogy.verbose)
    if verbose is not None:
        set_zogy.verbose = str2bool(verbose)

    # same for redo_new and redo_ref
    if redo_new is not None:
        set_zogy.redo_new = str2bool(redo_new)
    if redo_ref is not None:
        set_zogy.redo_ref = str2bool(redo_ref)

    # same for keep_tmp
    if keep_tmp is not None:
        set_zogy.keep_tmp = str2bool(keep_tmp)


    start_time1 = os.times()

    # name of logfile
    if logfile is None:
        if new_fits is not None:
            logfile = new_fits.replace('.fits', '.log')
        elif ref_fits is not None:
            logfile = ref_fits.replace('.fits', '.log')


    # attach logfile to logging if not already attached
    handlers = log.handlers[:]
    for handler in handlers:
        if logfile in str(handler):
            break
    else:
        fileHandler = logging.FileHandler(logfile, 'a')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel('INFO')
        log.addHandler(fileHandler)
        log.info ('logfile created: {}'.format(logfile))


    # define booleans [new] and [ref] indicating
    # that the corresponding image is provided and exists
    def set_bool (image_fits):
        ima_bool = False
        if image_fits is not None:
            if isfile(image_fits):
                ima_bool = True
                # check that image is not in compressed format
                header_temp = read_hdulist (image_fits, get_data=False,
                                            get_header=True)
                #if ('TTYPE1' in header_temp and
                #    'COMPRESSED' in header_temp['TTYPE1']):
                if header_temp['NAXIS'] != 2:
                    msg = 'input images need to be uncompressed'
                    log.critical(msg)
                    raise RuntimeError(msg)
            else:
                log.info('file {} does not exist'.format(image_fits))
        return ima_bool


    new = set_bool (new_fits)
    ref = set_bool (ref_fits)
    if not new and not ref:
        msg = 'no valid input image(s) provided'
        log.critical(msg)
        raise RuntimeError(msg)

    # global parameters
    if new:
        global base_new, fwhm_new, pixscale_new #, ysize_new, xsize_new
        # define the base names of input fits files as global so they
        # can be used in any function in this module
        base_new = new_fits.split('.fits')[0]

    if ref:
        global base_ref, fwhm_ref, pixscale_ref
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
    def check_files (filelist):
        for filename in filelist:
            if not isfile(filename):
                msg = '{} does not exist'.format(filename)
                log.critical(msg)
                raise RuntimeError(msg)


    check_files([get_par(set_zogy.sex_cfg,tel), get_par(set_zogy.psfex_cfg,tel),
                 get_par(set_zogy.swarp_cfg,tel)])
    if new:
        check_files([get_par(set_zogy.sex_par,tel)])
    if ref:
        check_files([get_par(set_zogy.sex_par_ref,tel)])
    if get_par(set_zogy.psffit_sex,tel):
        check_files([get_par(set_zogy.sex_cfg_psffit,tel),
                     get_par(set_zogy.sex_par_psffit,tel)])

    # the elements in [keywords] should be defined as strings, but do
    # not refer to the actual keyword names; the latter are
    # constructed inside the [read_header] function
    keywords = ['naxis2', 'naxis1', 'gain', 'satlevel', 'ra', 'dec', 'pixscale']

    if new:
        # read in header of new_fits
        t = time.time()
        header_new = read_hdulist (new_fits, get_data=False, get_header=True)
        ysize_new, xsize_new, gain_new, satlevel_new, ra_new, dec_new, \
            pixscale_new = (read_header(header_new, keywords))

    if ref:
        # read in header of ref_fits
        header_ref = read_hdulist (ref_fits, get_data=False, get_header=True)
        ysize_ref, xsize_ref, gain_ref, satlevel_ref, ra_ref, dec_ref, \
            pixscale_ref = (read_header(header_ref, keywords))


    # function to run SExtractor on fraction of the image, applied
    # below to new and/or ref image
    def sex_fraction (base, sexcat, pixscale, imtype, header):

        fwhm, fwhm_std, elong, elong_std = run_sextractor(
            '{}.fits'.format(base), sexcat, get_par(set_zogy.sex_cfg,tel),
            get_par(set_zogy.sex_par,tel), pixscale, header,
            return_fwhm_elong=True, fraction=get_par(set_zogy.fwhm_imafrac,tel),
            fwhm=5.0, tel=tel, set_zogy=set_zogy)

        log.info('fwhm_{}: {:.3f} +- {:.3f}'.format(imtype, fwhm, fwhm_std))

        # add header keyword(s):
        header['Z-V'] = (__version__, 'ZOGY version used')
        seeing = fwhm * pixscale
        seeing_std = fwhm_std * pixscale

        header['S-FWHM'] = (fwhm, '[pix] Sextractor FWHM estimate')
        header['S-FWSTD'] = (fwhm_std, '[pix] sigma (STD) FWHM estimate')
        header['S-SEEING'] = (seeing, '[arcsec] SExtractor seeing estimate')
        header['S-SEESTD'] = (seeing_std, '[arcsec] sigma (STD) SExtractor seeing')

        header['S-ELONG'] = (elong, 'SExtractor ELONGATION (A/B) estimate')
        header['S-ELOSTD'] = (elong_std, 'sigma (STD) SExtractor ELONGATION (A/B)')

        return fwhm, fwhm_std, elong, elong_std


    # run SExtractor for seeing estimate of new_fits
    if new:
        sexcat_new = '{}_cat.fits'.format(base_new)
        keys_temp = ['S-FWHM', 'S-FWSTD', 'S-ELONG', 'S-ELOSTD']
        # read values from header if available and nonzero
        if np.all([k in header_new and header_new[k] > 0 for k in keys_temp]):
            fwhm_new, fwhm_std_new, elong_new, elong_std_new = [
                header_new[k] for k in keys_temp]
        else:
            fwhm_new, fwhm_std_new, elong_new, elong_std_new = sex_fraction(
                base_new, sexcat_new, pixscale_new, 'new', header_new)



    # same for the reference image
    if ref:
        sexcat_ref = '{}_cat.fits'.format(base_ref)
        keys_temp = ['S-FWHM', 'S-FWSTD', 'S-ELONG', 'S-ELOSTD']
        # read values from header if available and nonzero
        if np.all([k in header_ref and header_ref[k] > 0 for k in keys_temp]):
            fwhm_ref, fwhm_std_ref, elong_ref, elong_std_ref = [
                header_ref[k] for k in keys_temp]
        else:
            fwhm_ref, fwhm_std_ref, elong_ref, elong_std_ref = sex_fraction(
                base_ref, sexcat_ref, pixscale_ref, 'ref', header_ref)



    # function to run SExtractor on full image, followed by Astrometry.net
    # to find the WCS solution, applied below to new and/or ref image
    def sex_wcs (base, sexcat, sex_params, pixscale, fwhm, imtype, fits_mask,
                 ra, dec, xsize, ysize, header):

        # redo boolean to determine whether to rerun some parts
        # (source extractor, astrometry.net, psfex) even if those were
        # executed done
        redo = ((get_par(set_zogy.redo_new,tel) and imtype=='new') or
                (get_par(set_zogy.redo_ref,tel) and imtype=='ref'))

        log.info ('redo switch in [sex_wcs]: {}'.format(redo))


        if isfile(sexcat):
            header_cat = read_hdulist (sexcat, get_data=False, get_header=True)
            size_cat = header_cat['NAXIS2']

        # run SExtractor on full image
        if (not (isfile(sexcat) and size_cat > 0) or redo):

            try:
                SE_processed = False

                # run source-extractor with normal fits table as
                # output and without the VIGNET output column; the
                # LDAC version required by PSFEx is made just before
                # PSFEx is run
                result = run_sextractor(
                    '{}.fits'.format(base), sexcat, get_par(set_zogy.sex_cfg,tel),
                    sex_params, pixscale, header, fwhm=fwhm, imtype=imtype,
                    fits_mask=fits_mask, npasses=2, tel=tel, set_zogy=set_zogy,
                    nthreads=nthreads)


            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [run_sextractor]: {}'
                              .format(e))
            else:
                SE_processed = True
            finally:
                # add header keyword(s):
                header['S-P'] = (SE_processed, 'successfully processed by '
                                 'SExtractor?')
                # SExtractor version
                cmd = ['source-extractor', '-v']
                result = subprocess.run(cmd, capture_output=True)
                version = str(result.stdout).split()[-2]
                header['S-V'] = (version.strip(), 'SExtractor version used')
                if not SE_processed:
                    return False

            # copy the LDAC binary fits table output from SExtractor (with
            # '_ldac' in the name) to a normal binary fits table;
            # Astrometry.net needs the latter, but PSFEx needs the former,
            # so keep both

            # the LDAC catalog of higher S/N sources is now made just
            # before PSFEx is run, so this conversion can be skipped
            # ldac2fits (sexcat, '{}_cat.fits'.format(base))

            # now that normal fits catalog without thumbnails is
            # available, update its BACKGROUND column with improved
            # background estimate - this was previously done inside
            # [run_sextractor] but with large LDAC files this becomes
            # very RAM-expensive - switching off so that background
            # column lists the background of the reduced image,
            # because the BACKPHOTO_TYPE is LOCAL
            if False:
                if get_par(set_zogy.bkg_method,tel)==2:
                    update_bkgcol (base, header, imtype)


        else:
            log.info('output catalog {} or {} already present and redo flag '
                     'is set to False; skipping source-extractor run on {}'
                     .format(sexcat, '{}_cat.fits'.format(base),
                             '{}.fits'.format(base)))


        # determine WCS solution
        if ('CTYPE1' not in header and 'CTYPE2' not in header) or redo:
            try:
                if not get_par(set_zogy.skip_wcs,tel):
                    # delete some keywords that astrometry.net does
                    # not appear to overwrite
                    for key in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                                'CUNIT1', 'CUNIT2',
                                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                                'PROJP1', 'PROJP3',
                                'PV1_1', 'PV1_2', 'PV2_1', 'PV2_2']:
                        if key in header:
                            del header[key]
                    WCS_processed = False
                    fits_base = '{}.fits'.format(base)
                    run_wcs(fits_base, ra, dec, pixscale, xsize, ysize, header,
                            imtype)

            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [run_wcs]: {}'
                              .format(e))
            else:
                WCS_processed = True
                # update pixscale_new or pixscale_ref
                if 'A-PSCALE' in header:
                    if imtype=='new':
                        pixscale_new = header['A-PSCALE']
                    if imtype=='ref':
                        pixscale_ref = header['A-PSCALE']

            finally:
                # add header keyword(s):
                header['A-P'] = (WCS_processed, 'successfully processed by '
                                 'Astrometry.net?')
                # Astrometry.net version
                cmd = ['solve-field', '-h']
                result = subprocess.run(cmd, capture_output=True)
                version = str(result.stdout).split('Revision')[1].split(',')[0]
                header['A-V'] = (version.strip(), 'Astrometry.net version used')
                if not WCS_processed:
                    return False

        else:
            log.info('WCS solution (CTYPE1 and CTYPE2 keywords) present in '
                     'header of {}; skipping astrometric calibration'
                     .format('{}.fits'.format(base)))

        return True



    # initialize header_trans
    if new and ref:
        remap = True
        # initialize header to be recorded for keywords related to the
        # comparison of new and ref
        header_trans = fits.Header()
    else:
        remap = False


    if new:
        # now run above function [sex_wcs] on new image
        if fwhm_new > 0:
            success = sex_wcs(
                base_new, sexcat_new, get_par(set_zogy.sex_par,tel), pixscale_new,
                fwhm_new, 'new', new_fits_mask, ra_new, dec_new, xsize_new,
                ysize_new, header_new)
        else:
            success = False

        if not success:
            # leave because either FWHM estimate from initial
            # SExtractor is not positive, or an exception occurred
            # during [run_sextractor] or [run_wcs] inside [sex_wcs]
            if not ref:
                return header_new
            else:
                return header_new, header_trans


    if ref:
        # and reference image
        if fwhm_ref > 0:
            success = sex_wcs(
                base_ref, sexcat_ref, get_par(set_zogy.sex_par_ref,tel),
                pixscale_ref, fwhm_ref, 'ref', ref_fits_mask, ra_ref,
                dec_ref, xsize_ref, ysize_ref, header_ref)
        else:
            success = False

        if not success:
            # leave because either FWHM estimate from initial
            # SExtractor is not positive, or an exception occurred
            # during [run_sextractor] or [run_wcs] inside [sex_wcs]
            if not new:
                return header_ref
            else:
                return header_new, header_trans


    # check that new and ref image have at least some overlap; if not,
    # then leave
    if new and ref:

        # pixel positions of the 4 corners - with a margin of [dpix]
        # pixels - in the new image
        dpix = int((xsize_ref + ysize_ref) / 2. / 4.)
        xy_new = [(dpix,dpix), (xsize_new-dpix,dpix),
                  (dpix,ysize_new-dpix), (xsize_new-dpix, ysize_new-dpix)]
        x_new, y_new = zip(*xy_new)
        # convert these to ra, dec
        wcs_new = WCS(header_new)
        ra_new, dec_new = wcs_new.all_pix2world(x_new, y_new, 1)

        # convert these to pixel positions in the ref image
        wcs_ref = WCS(header_ref)
        x_ref, y_ref = wcs_ref.all_world2pix(ra_new, dec_new, 1)

        # check whether any of the new image "corners" are on the ref
        # image
        mask_on = np.zeros(4, dtype=bool)
        for i in range(len(x_ref)):
            if (x_ref[i] >= 0 and x_ref[i] <= xsize_ref and
                y_ref[i] >= 0 and y_ref[i] <= ysize_ref):
                mask_on[i] = True
                log.info ('corner {} on ref image: {}'.format(i, mask_on[i]))

        # if none of the corners are within the ref image, then there
        # is no or too little overlap
        if not np.any(mask_on):

            log.error ('overlap between new image {} and ref image {} is '
                       'less than {} pixels in the reference frame; leaving'
                       .format(new_fits, ref_fits, dpix))

            return header_new, header_trans



    # determine cutouts
    if new:
        xsize = xsize_new
        ysize = ysize_new
    else:
        xsize = xsize_ref
        ysize = ysize_ref


    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
        get_par(set_zogy.subimage_size,tel), ysize, xsize)
    nsubs = centers.shape[0]
    if get_par(set_zogy.verbose,tel):
        log.info('nsubs: {}'.format(nsubs))
        #for i in range(nsubs):
        #    log.info('i ' + str(i))
        #    log.info('cuts_ima[i] ' + str(cuts_ima[i]))
        #    log.info('cuts_ima_fft[i] ' + str(cuts_ima_fft[i]))
        #    log.info('cuts_fft[i] ' + str(cuts_fft[i]))


    # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with new,
    # psf and background images
    if new:

        try:
            # data_new, psf_new and data_new_bkg_std (same goes for
            # corresponding ref arrays below) are actually strings
            # pointing to the name of the (last) subimage file(s) in
            # which the subimage data arrays were saved;
            # [load_npy_fits] with input parameter nsub will read the
            # separate files properly
            dict_data_new, dict_psf_new, psf_orig_new, dict_data_new_bkg_std = (
                prep_optimal_subtraction('{}.fits'.format(base_new), nsubs,
                                         'new', fwhm_new, header_new,
                                         fits_mask=new_fits_mask,
                                         nthreads=nthreads))
        except Exception as e:
            log.exception('exception was raised during [prep_optimal_extraction]'
                          ' of new image {}: {}'.format(base_new, e))
            if not ref:
                return header_new
            else:
                return header_new, header_trans


    # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with ref,
    # psf and background images; if new and ref are provided, then
    # these images will be remapped to the new frame
    if ref:

        try:
            dict_data_ref, dict_psf_ref, psf_orig_ref, dict_data_ref_bkg_std = (
                prep_optimal_subtraction('{}.fits'.format(base_ref), nsubs,
                                         'ref', fwhm_ref, header_ref,
                                         fits_mask=ref_fits_mask, remap=remap,
                                         nthreads=nthreads))

        except Exception as e:
            log.exception('exception was raised during [prep_optimal_extraction]'
                          ' of reference image {}: {}'.format(base_ref, e))
            if not new:
                return header_ref
            else:
                return header_new, header_trans


        if remap:
            # for now set the value header below to True by hand;
            # should be checked properly, e.g. by try-except the
            # prep_optimal_subtraction function above, or by
            # try-except the swarp part inside it and let it pass
            # along the remap_processed variable to here
            header_trans['SWARP-P'] = (True, 'reference image successfully SWarped?')
            # SWarp version
            cmd = ['swarp', '-v']
            result = subprocess.run(cmd, capture_output=True)
            version = str(result.stdout).split()[-2]
            header_trans['SWARP-V'] = (version.strip(), 'SWarp version used')



    if new and ref:
        # get x, y and fratios from matching PSFex stars across entire
        # frame the "_subs" output arrays are the values to be used for
        # the subimages below in the function [run_ZOGY]
        use_optflux = get_par(set_zogy.fratio_optflux,tel)
        ok, x_fratio, y_fratio, dx, dy, fratio_subs, fratio_err_subs, dx_subs, \
            dy_subs = (get_fratio_dxdy ('{}_cat.fits'.format(base_new),
                                        '{}_cat.fits'.format(base_ref),
                                        '{}_psfex.cat'.format(base_new),
                                        '{}_psfex.cat'.format(base_ref),
                                        header_new, header_ref, nsubs, cuts_ima,
                                        header_trans, use_optflux=use_optflux))

        if not ok:
            # leave because of too few matching stars in new and ref
            # to determine fratio, dx and dy
            log.error ('too few matching stars in new and ref to determine '
                       'fratio, dx and dy')
            return header_new, header_trans



        if True:
            # compare this with the flux ratio inferred from the zeropoint
            # difference between new and ref:
            keys_temp = ['PC-ZP', 'AIRMASSC', 'PC-EXTCO']
            if (np.all([k in header_new for k in keys_temp]) and
                np.all([k in header_ref for k in keys_temp])):
                zp_new, airmass_new, extco_new = [header_new[k] for k in keys_temp]
                zp_ref, airmass_ref, extco_ref = [header_ref[k] for k in keys_temp]
                # mag_AB = ZP + mag_i - airmass * extco
                # mag_AB_ref = mag_AB_new leads to: dmag_i = mag_i_new - mag_i_ref
                dmag = ((zp_ref - airmass_ref*extco_ref) -
                        (zp_new - airmass_new*extco_new))
                fratio_zps = 10**(-0.4*dmag)  # fnew/fref
                log.info('fratio from delta ZPs: {}'.format(fratio_zps))

                # adopt this value for the moment
                #fratio = fratio_zps
                #fratio_subs[:] = fratio_zps



        # fratio is in counts if input images were in counts; convert
        # to electrons
        fratio_subs *= gain_new / gain_ref
        fratio_err_subs *= gain_new / gain_ref


        if get_par(set_zogy.make_plots,tel):

            dx_med = np.median(dx)
            dx_std = np.std(dx)

            dy_med = np.median(dy)
            dy_std = np.std(dy)


            def plot (x, y, limits, xlabel, ylabel, filename, annotate=True):
                plt.axis(limits)
                #plt.plot(x, y, 'go', color='tab:blue', markersize=5,
                #         markeredgecolor='k')
                plt.plot(x, y, 'ko', markersize=2)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title('{}\n vs {}'.format(new_fits, ref_fits), fontsize=12)
                if annotate:
                    plt.annotate('dx={:.3f} +/- {:.3f} pixels'
                                 .format(dx_med, dx_std),
                                 xy=(0.02,0.95), xycoords='axes fraction')
                    plt.annotate('dy={:.3f} +/- {:.3f} pixels'
                                 .format(dy_med, dy_std),
                                 xy=(0.02,0.90), xycoords='axes fraction')
                if filename != '':
                    plt.savefig(filename)
                if get_par(set_zogy.show_plots,tel):
                    plt.show()
                plt.close()


            plot (x_fratio, y_fratio, (0,xsize_new,0,ysize_new), 'x (pixels)',
                  'y (pixels)', '{}_xy.pdf'.format(base_newref), annotate=False)
            plot (dx, dy, (-1,1,-1,1), 'dx (pixels)', 'dy (pixels)',
                  '{}_dxdy.pdf'.format(base_newref))
            #plot (x_fratio, dr, (0,xsize_new,0,1), 'x (pixels)', 'dr (pixels)',
            #      base_newref+'_drx.pdf', annotate=False)
            #plot (y_fratio, dr, (0,ysize_new,0,1), 'y (pixels)', 'dr (pixels)',
            #      base_newref+'_dry.pdf', annotate=False)
            plot (x_fratio, dx, (0,xsize_new,-1,1), 'x (pixels)', 'dx (pixels)',
                  '{}_dxx.pdf'.format(base_newref))
            plot (y_fratio, dy, (0,ysize_new,-1,1), 'y (pixels)', 'dy (pixels)',
                  '{}_dyy.pdf'.format(base_newref))
            # plot dx and dy as function of distance from the image center
            xcenter = int(xsize_new/2)
            ycenter = int(ysize_new/2)
            dist = np.sqrt((x_fratio-xcenter)**2 + (y_fratio-ycenter)**2)
            plot (dist, dx, (0,np.amax(dist),0,1), 'distance from image center '
                  '(pixels)', 'dx (pixels)', '{}_dxdist.pdf'.format(base_newref))
            plot (dist, dy, (0,np.amax(dist),0,1), 'distance from image center '
                  '(pixels)', 'dy (pixels)', '{}_dydist.pdf'.format(base_newref))




        start_time2 = os.times()
        if get_par(set_zogy.timing,tel):
            t_zogy = time.time()

        log.info('executing run_ZOGY on subimages ...')
        mem_use (label='just before run_ZOGY')

        try:

            zogy_processed = False
            results_zogy = []

            for nsub in range(nsubs):

                # run ZOGY

                # load subimages
                data_new_sub = load_npy_fits (dict_data_new[nsub])
                psf_new_sub = load_npy_fits (dict_psf_new[nsub])
                data_new_bkg_std_sub = load_npy_fits (dict_data_new_bkg_std[nsub])

                data_ref_sub = load_npy_fits (dict_data_ref[nsub])
                psf_ref_sub = load_npy_fits (dict_psf_ref[nsub])
                data_ref_bkg_std_sub = load_npy_fits (dict_data_ref_bkg_std[nsub])

                # if use_FFTW=True: proces the first subimage (could
                # be any subimage) twice to avoid it showing vertical
                # bands (e.g. see
                # ML1_20190610_185110_red_Scorr.fits.fz). Not clear
                # why this happens - something to do with the
                # optimization planning done by FFTW for the first
                # image to speed up subsequent FFT calculations.
                use_FFTW = True
                if use_FFTW and nsub==0:
                    niter = 2
                else:
                    niter = 1

                for i in range(niter):
                    result_sub = run_ZOGY (nsub, data_ref_sub, data_new_sub,
                                           psf_ref_sub, psf_new_sub,
                                           data_ref_bkg_std_sub,
                                           data_new_bkg_std_sub,
                                           fratio_subs[nsub], fratio_err_subs[nsub],
                                           dx_subs[nsub], dy_subs[nsub],
                                           use_FFTW=use_FFTW, nthreads=nthreads)

                results_zogy.append(result_sub)


                # delete if not keeping intermediate/temporary files
                if not get_par(set_zogy.keep_tmp,tel):
                    list2remove = [dict_psf_new[nsub],
                                   dict_psf_ref[nsub],
                                   dict_data_new_bkg_std[nsub],
                                   dict_data_ref_bkg_std[nsub],
                                   dict_data_ref[nsub]]
                    # keep dict_data_new - needed below - if fake
                    # stars were added
                    if get_par(set_zogy.nfakestars,tel)==0:
                        list2remove.append(dict_data_new[nsub])
                    # remove
                    remove_files (list2remove)



        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [run_ZOGY]: {}'.format(e))
        else:
            zogy_processed = True
        finally:
            # add header keyword(s):
            header_trans['Z-P'] = (zogy_processed, 'successfully processed by ZOGY?')
            header_trans['Z-REF'] = (base_ref.split('/')[-1], 'name reference image')
            header_trans['Z-SIZE'] = (get_par(set_zogy.subimage_size,tel),
                                      '[pix] size of (square) ZOGY subimages')
            header_trans['Z-BSIZE'] = (get_par(set_zogy.subimage_border,tel),
                                       '[pix] size of ZOGY subimage borders')
            # if exception occurred in [run_ZOGY], leave
            if not zogy_processed:
                return header_new, header_trans


        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t_zogy, label='after run_ZOGY finished')



        # loop over results from pool and paste subimages into output
        # images

        # this definition of [index_extract], the part from the fft
        # images (including a border) that needs to go into the final
        # images, can be outside of the nsub loop below
        x1, y1 = (get_par(set_zogy.subimage_border,tel),
                  get_par(set_zogy.subimage_border,tel))
        x2, y2 = (x1+get_par(set_zogy.subimage_size,tel),
                  y1+get_par(set_zogy.subimage_size,tel))
        index_extract = tuple([slice(y1,y2), slice(x1,x2)])


        log.info ('creating full D, Scorr, Fpsf and Fpsferr images from '
                  'subimages and writing them to disk')


        #header_tmp = header_new + header_trans

        # merge header_new and header_trans
        header_tmp = header_new.copy(strip=True)
        header_tmp.update(header_trans.copy(strip=True))


        # assemble and write full output images, one by one using
        # function [create_full]
        names_zogy = ['{}_D.fits'.format(base_newref),
                      #'{}_S.fits'.format(base_newref),
                      '{}_Scorr.fits'.format(base_newref),
                      '{}_Fpsf.fits'.format(base_newref),
                      '{}_Fpsferr.fits'.format(base_newref)]

        for i_name, name in enumerate(names_zogy):
            # zogy files can be a mix of fits and numpy files so
            # [create_full] needs list of all images instead of
            # using single name to be used with [load_npy_fits]
            sublist = [l[i_name] for l in results_zogy]
            create_full (name, header_tmp, (ysize_new, xsize_new),
                         'float32', nsubs, cuts_ima, index_extract,
                         sublist=sublist)

            # delete if not keeping intermediate/temporary files
            if not get_par(set_zogy.keep_tmp,tel):
                remove_files (sublist)


        # if fake stars were added to the new image, also create
        # full image from data_new subimage files
        if get_par(set_zogy.nfakestars,tel)>0:
            log.info ('creating new image including fake stars '
                      'from subimages and writing it to disk')
            # subimage filenames are recorded in dictionary
            # [dict_data_new]
            sublist = [dict_data_new[i] for i in range(nsubs)]
            create_full ('{}_new.fits'.format(base_newref),
                         header_tmp, (ysize_new, xsize_new),
                         'float32', nsubs, cuts_ima, index_extract,
                         sublist=sublist)

            # delete if not keeping intermediate/temporary files
            if not get_par(set_zogy.keep_tmp,tel):
                sublist = [dict_data_new[i] for i in range(nsubs)]
                remove_files (sublist)


        mem_use (label='after creating full images')


        if get_par(set_zogy.display,tel):

            log.info ('displaying numerous subimages')

            path, name = os.path.split(base_newref)
            if len(path)>0:
                dir_newref = path
            else:
                dir_newref = '.'

            base_remap = '{}/{}'.format(dir_newref, base_ref.split('/')[-1])
            names_disp_list = ['{}.fits'.format(base_newref),
                               '{}_remap.fits'.format(base_remap),
                               '{}_bkg_std.fits'.format(base_newref),
                               '{}_bkg_std_remap.fits'.format(base_remap)]
            dtype_list = ['float32'] * 4

            if get_par(set_zogy.nfakestars,tel)>0:
                names_disp_list[0] = '{}_new.fits'.format(base_newref)

            if new_fits_mask is not None:
                names_disp_list.append(new_fits_mask)
                dtype_list.append('uint8')

            if ref_fits_mask is not None:
                fits_tmp = ref_fits_mask.replace('.fits', '_remap.fits')
                names_disp_list.append('{}/{}'.format(dir_newref,
                                                      fits_tmp.split('/')[-1]))
                dtype_list.append('uint8')

            for i_name, name in enumerate(names_disp_list):
                create_subs (name, dir_newref, nsubs, cuts_ima,
                             dtype_list[i_name], index_extract)

            # display
            names_full = [names_disp_list[0:2], names_zogy, names_disp_list[2:]]
            names_full = list(itertools.chain.from_iterable(names_full))
            display_subs(base_new, base_ref, nsubs, names_full)




        mem_use (label='just after run_ZOGY')

        if 'data_Scorr_full' not in locals():
            data_Scorr_full = read_hdulist('{}_Scorr.fits'.format(base_newref))

        if 'data_Fpsferr_full' not in locals():
            data_Fpsferr_full = read_hdulist('{}_Fpsferr.fits'.format(base_newref))

        # compute statistics on Scorr image and show histogram
        # discarding the edge, and using a fraction of the total image
        edge = 100
        nstat = int(0.1 * (xsize_new*ysize_new))
        x_stat = (np.random.rand(nstat)*(xsize_new-2*edge)).astype(int) + edge
        y_stat = (np.random.rand(nstat)*(ysize_new-2*edge)).astype(int) + edge
        mean_Scorr, median_Scorr, std_Scorr = sigma_clipped_stats (
            data_Scorr_full[y_stat,x_stat].astype(float))

        if get_par(set_zogy.verbose,tel):
            log.info('Scorr mean: {:.3f} , median: {:.3f}, std: {:.3f}'
                     .format(mean_Scorr, median_Scorr, std_Scorr))

        # make histrogram plot if needed
        if get_par(set_zogy.make_plots,tel):
            clipped_stats (data_Scorr_full[y_stat,x_stat], clip_zeros=True,
                           make_hist=get_par(set_zogy.make_plots,tel),
                           name_hist='{}_Scorr_hist.pdf'.format(base_newref),
                           hist_xlabel='value in Scorr image')


        # compute statistics on Fpsferr image
        mean_Fpsferr, median_Fpsferr, std_Fpsferr = sigma_clipped_stats (
            data_Fpsferr_full[y_stat,x_stat].astype(float))

        if get_par(set_zogy.verbose,tel):
            log.info('Fpsferr mean: {:.3f} , median: {:.3f}, std: {:.3f}'
                     .format(mean_Fpsferr, median_Fpsferr, std_Fpsferr))


        # convert Fpsferr image to limiting magnitude image
        mask_zero = (data_Fpsferr_full==0)
        data_Fpsferr_full[mask_zero] = median_Fpsferr
        exptime, filt = read_header(header_new, ['exptime', 'filter'])
        if 'PC-ZP' in header_new and 'AIRMASSC' in header_new:
            zp = header_new['PC-ZP']
            airm = header_new['AIRMASSC']
            ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]
            data_tlimmag = apply_zp((get_par(set_zogy.transient_nsigma,tel) *
                                     data_Fpsferr_full), zp, airm, exptime,
                                    ext_coeff).astype('float32')

        # do not try to write scaled uint8 image (see block above);
        # keep precision here and let fpack do the compression to some
        # selected q-level
        fits_tlimmag = '{}_trans_limmag.fits'.format(base_newref)
        fits.writeto(fits_tlimmag, data_tlimmag, header_tmp, overwrite=True)
        del data_tlimmag



        # add header keyword(s):
        header_trans['Z-SCMED'] = (median_Scorr, 'median Scorr full image')
        header_trans['Z-SCSTD'] = (std_Scorr, 'sigma (STD) Scorr full image')
        header_trans['Z-FPEMED'] = (median_Fpsferr/exptime,
                                    '[e-/s] median Fpsferr full image')
        header_trans['Z-FPESTD'] = (std_Fpsferr/exptime,
                                    '[e-/s] sigma (STD) Fpsferr full image')


        # if [std_Scorr] is very large, do not bother to continue as
        # too many spurious transients are likely to be identified
        std_Scorr_limit = 2.5
        if std_Scorr > std_Scorr_limit:
            log.error ('Scorr image standard deviation is larger than {}; not '
                       'continuing with the transient extraction as too many '
                       'spurious transients are likely to be identified'
                       .format(std_Scorr_limit))
            return header_new, header_trans


        # define fits files names
        path, name = os.path.split(base_newref)
        if len(path)>0:
            dir_newref = path
        else:
            dir_newref = '.'

        base_remap = '{}/{}'.format(dir_newref, base_ref.split('/')[-1])

        if get_par(set_zogy.nfakestars,tel)>0:
            fits_new = '{}_new.fits'.format(base_newref)
        else:
            fits_new = '{}.fits'.format(base_newref)

        fits_ref = '{}_remap.fits'.format(base_remap)
        fits_new_bkg_std = '{}_bkg_std.fits'.format(base_newref)
        fits_ref_bkg_std = '{}_bkg_std_remap.fits'.format(base_remap)
        fits_Scorr = '{}_Scorr.fits'.format(base_newref)
        fits_D = '{}_D.fits'.format(base_newref)
        fits_Fpsf = '{}_Fpsf.fits'.format(base_newref)
        fits_Fpsferr = '{}_Fpsferr.fits'.format(base_newref)

        if new_fits_mask is not None:
            fits_new_mask = new_fits_mask
        else:
            fits_new_mask = new_fits.replace('.fits', '_mask.fits')

        if ref_fits_mask is not None:
            fits_tmp = ref_fits_mask.replace('.fits', '_remap.fits')
            fits_ref_mask = '{}/{}'.format(dir_newref, fits_tmp.split('/')[-1])
        else:
            fits_tmp = ref_fits.replace('.fits', '_mask.fits')
            fits_ref_mask = '{}/{}'.format(dir_newref, fits_tmp.split('/')[-1])

        fits_new_psf = '{}_psf.fits'.format(base_new)
        fits_ref_psf = '{}_psf.fits'.format(base_ref)
        fits_new_cat = '{}_cat.fits'.format(base_new)
        fits_ref_cat = '{}_cat.fits'.format(base_ref)


        mem_use (label='just before get_trans')

        table_trans, dict_thumbnails = get_trans (
            fits_new, fits_ref, fits_D, fits_Scorr, fits_Fpsf, fits_Fpsferr,
            fits_new_mask, fits_ref_mask, fits_new_bkg_std, fits_ref_bkg_std,
            header_new, header_ref, header_trans, fits_new_psf, fits_ref_psf,
            nthreads=nthreads)

        # number of transients in table to add to header below
        ntrans = len(table_trans)

        # if one or more fake stars were added to the new image, merge
        # the fake star table with the output transient catalog
        if get_par(set_zogy.nfakestars,tel)>0:

            log.info ('extracting fake stars from data')
            table_fake = Table.read('{}_fake.fits'.format(base_newref))
            table_merged = merge_fakestars (
                table_trans, table_fake, 'trans', header_new)

            # write to fits table
            cat_trans = '{}.transcat'.format(base_newref)
            table_merged.write(cat_trans, format='fits', overwrite=True)



        mem_use (label='just after get_trans')


        # for ML/BG delete unnecesary images
        if tel[0:2] in ['ML','BG'] and not get_par(set_zogy.keep_tmp,tel):

            fits_new_bkg = '{}_bkg.fits'.format(base_newref)
            # ref_fits_bkg_std refers to image that was not remapped;
            # fits_ref refer to remapped images
            ref_fits_bkg_std = '{}_bkg_std.fits'.format(base_remap)
            list2remove = [fits_new_bkg, fits_new_bkg_std,
                           fits_ref, fits_ref_mask, fits_ref_bkg_std]

            if not get_par(set_zogy.use_new_transcat,tel):
                # these images below are used in the crossmatching
                # further below
                list2remove += [ref_fits, ref_fits_mask, ref_fits_bkg_std]

            # remove
            remove_files (list2remove, verbose=True)



        # add header keyword(s):
        header_trans['T-NSIGMA'] = (get_par(set_zogy.transient_nsigma,tel),
                                   '[sigma] input transient detection threshold')
        lflux = float(get_par(set_zogy.transient_nsigma,tel)) * median_Fpsferr
        header_trans['T-LFLUX'] = (lflux/exptime, '[e-/s] full-frame transient '
                                  '{}-sigma limiting flux'.format(
                                      get_par(set_zogy.transient_nsigma,tel)))
        header_trans['T-NTRANS'] = (ntrans, 'number of >= {}-sigma transients '
                                   '(pre-vetting)'.format(
                                       get_par(set_zogy.transient_nsigma,tel)))

        # add ratio of ntrans over total number of significant objects detected
        if 'NOBJECTS' in header_new:
            nobjects = header_new['NOBJECTS']
            header_trans['T-FTRANS'] = (ntrans/nobjects, 'transient fraction: '
                                        'T-NTRANS / NOBJECTS')


        # infer limiting magnitudes from corresponding limiting
        # fluxes using zeropoint and median airmass
        if 'PC-ZP' in header_new and 'AIRMASSC' in header_new:
            keywords = ['exptime', 'filter']
            exptime, filt = read_header(header_new, keywords)
            zeropoint = header_new['PC-ZP']
            airmass = header_new['AIRMASSC']
            ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]
            [[lmag], [lfnu]] = apply_zp([lflux], zeropoint, airmass, exptime,
                                        ext_coeff, return_fnu=True)
            header_trans['T-LFNU'] = (lfnu, '[microJy] full-frame transient '
                                      '{}-sigma limiting Fnu'.format(
                                      get_par(set_zogy.transient_nsigma,tel)))
            header_trans['T-LMAG'] = (lmag, '[mag] full-frame transient {}-sigma '
                                      'limiting mag'.format(
                                      get_par(set_zogy.transient_nsigma,tel)))


        # add fakestar header keywords
        header_trans['T-NFAKE'] = (nsubs * get_par(set_zogy.nfakestars,tel),
                                   'number of fake stars added to full frame')
        header_trans['T-FAKESN'] = (get_par(set_zogy.fakestar_s2n,tel),
                                    'fake stars input S/N')


        # apply Zafiirah's MeerCRAB module to the thumbnails in
        # the transient catalog just created, using the function
        # get_ML_prob_real
        ML_prob_real = None
        if get_par(set_zogy.ML_calc_prob,tel) and tel[0:2] in ['ML','BG']:

            try:
                ML_processed = False
                # vetting training version to be used
                ML_version = get_par(set_zogy.ML_version,tel)
                # and the corresponding model
                ML_model = get_par(set_zogy.ML_models,tel)[ML_version]

                log.info ('applying machine-learning real/bogus version {} with '
                          'model {} for {}'
                          .format(ML_version, ML_model.split('/')[-1], base_new))




                # depending on version, execute a different function
                if ML_version == '1':
                    ML_prob_real = get_ML_prob_real_Zafiirah (dict_thumbnails,
                                                              ML_model)
                elif ML_version == '2':
                    ML_prob_real = get_ML_prob_real_Diederik (dict_thumbnails,
                                                              ML_model,
                                                              size_use=40)
                elif ML_version == '3':
                    # updated version of Diederik (August 2024)
                    ML_prob_real = get_ML_prob_real_Diederik (dict_thumbnails,
                                                              ML_model,
                                                              size_use=30)


            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [get_ML_prob_real]: '
                              '{}'.format(e))
            else:
                ML_processed = True

            finally:
                header_trans['MC-P'] = (ML_processed, 'successfully '
                                        'processed by MeerCRAB?')
                # set version by hand
                #ML_version = prediction_phase.__version__
                header_trans['MC-V'] = (ML_version, 'MeerCRAB version used')
                header_trans['MC-MODEL'] = (ML_model.split('/')[-1],
                                            'MeerCRAB training model used')


            # if exception occurred in [get_ML_prob_real], leave
            if not ML_processed:
                return header_new, header_trans



        # updating headers of ZOGY images
        if get_par(set_zogy.timing,tel):
            t_fits = time.time()


        #header_tmp = header_new + header_trans
        #header_tmp['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')

        # merge header_new and header_trans
        header_tmp = header_new.copy(strip=True)
        header_tmp.update(header_trans.copy(strip=True))


        # images were already created above, just update the headers here
        for ext_tmp in ['D', 'Scorr', 'Fpsf', 'Fpsferr', 'trans_limmag']:

            fn_tmp = '{}_{}.fits'.format(base_newref, ext_tmp)
            update_imcathead (fn_tmp, header_tmp)



        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t_fits, label='after updating headers of '
                               'D, Scorr, Fpsf and Fpsferr fits images')



    mem_use (label='just before formatting catalogs')

    # using the function [format_cat], write the new, ref and
    # transient output catalogues with the desired format, where the
    # thumbnail images (new, ref, D and Scorr) around each transient
    # are added as array columns in the transient catalogue.

    # new catalogue
    if new:
        exptime_new = read_header(header_new, ['exptime'])
        cat_new = '{}_cat.fits'.format(base_new)


        # convert fits table to astropy Table
        table_new = Table.read(cat_new)


        # CHECK!!! - this should not be needed anymore, but need to
        # double-checck this
        if False:

            # add the zeropoint error to the fnu errors
            add_zperr (table_new, header_new, imtype='new', tel=tel)

            # add magnitude columns??
            add_magcols (table_new, overwrite=False)



        # write updated table, overwriting the input fits table
        table_new.write (cat_new, format='fits', overwrite=True)



        cat_new_out = cat_new
        header_new_cat = read_hdulist(cat_new, get_data=False, get_header=True)
        if not ('FORMAT-P' in header_new_cat and header_new_cat['FORMAT-P']):

            header2add = header_new.copy(strip=True)

            format_cat (cat_new, cat_new_out, cat_type='new',
                        header2add=header2add, exptime=exptime_new,
                        apphot_radii=get_par(set_zogy.apphot_radii,tel),
                        nfakestars= get_par(set_zogy.nfakestars,tel),
                        tel=tel, set_zogy=set_zogy)

    # ref catalogue
    if ref:
        exptime_ref = read_header(header_ref, ['exptime'])
        cat_ref = '{}_cat.fits'.format(base_ref)
        cat_ref_out = cat_ref
        header_ref_cat = read_hdulist(cat_ref, get_data=False, get_header=True)
        if not ('FORMAT-P' in header_ref_cat and header_ref_cat['FORMAT-P']):

            header2add = header_ref.copy(strip=True)

            format_cat (cat_ref, cat_ref_out, cat_type='ref',
                        header2add=header2add, exptime=exptime_ref,
                        apphot_radii=get_par(set_zogy.apphot_radii,tel),
                        tel=tel, set_zogy=set_zogy)

    # trans catalogue
    if new and ref:
        cat_trans = '{}.transcat'.format(base_newref)
        cat_trans_out = '{}_trans.fits'.format(base_newref)
        sexcat_new = '{}_sexcat.fits'.format(base_new)

        if get_par(set_zogy.use_new_transcat,tel):

            # add some additional columns to transient catalog, by crossmatching
            # with Gaia and reference catalogs, and performing forced photometry
            # at transients positions in new and ref image
            table_trans = trans_crossmatch_fphot (
                cat_trans,
                new_fits, new_fits_mask, sexcat_new,
                ref_fits, ref_fits_mask, cat_ref,
                nthreads=nthreads)


            # CHECK!!! - this should not be needed anymore, but need to
            # double-checck this
            if False:

                # add the zeropoint error to the fnu errors in transient table
                add_zperr (table_trans, header_new, header_ref=header_ref_cat,
                           imtype='trans', tel=tel)

                # add magnitudes corresponding to fnu and error columns in
                # transient catalog
                add_magcols (table_trans)



            # now that transient flux and flux in the reference image
            # at the transient position is known, remove transients
            # where most of their flux is explained by a negative flux
            # in the reference image
            fnu_zogy = table_trans['FNU_ZOGY'].value
            fnuerr_zogy = table_trans['FNUERR_ZOGY'].value
            fnu_opt_ref = table_trans['FNU_OPT_REF'].value
            fnuerr_opt_ref = table_trans['FNUERR_OPT_REF'].value
            fnu_tot = fnu_zogy + fnu_opt_ref
            fnuerr_tot = np.sqrt(fnuerr_zogy**2 + fnuerr_opt_ref**2)

            log.info ('fnu_tot.shape: {}'.format(fnu_tot.shape))
            log.info ('fnuerr_tot.shape: {}'.format(fnuerr_tot.shape))
            log.info ('fnu_opt_ref.shape: {}'.format(fnu_opt_ref.shape))
            log.info ('fnuerr_opt_ref.shape: {}'.format(fnuerr_opt_ref.shape))


            # CHECK!!! - skip for now
            if True:

                mask_discard = (((fnu_tot / fnuerr_tot) < 3) &
                                ((fnu_opt_ref / fnuerr_opt_ref) < -3))
                table_trans = table_trans[~mask_discard]
                log.info ('{} transients discarded whose flux can be explained '
                          'by a negative flux in reference image'
                          .format(np.sum(mask_discard)))

                # also discard transients with nonzero flag in reference image?
                mask_zeroflag_ref = (table_trans['FLAGS_MASK_REF'] == 0)
                table_trans = table_trans[mask_zeroflag_ref]

                log.info ('{} transients discarded with nonzero flags in '
                          'reference image'.format(np.sum(mask_zeroflag_ref)))



            # after these updates, write updated table, overwriting the input
            table_trans.write (cat_trans, format='fits', overwrite=True)



            # for ML/BG delete unnecesary images
            if tel[0:2] in ['ML','BG'] and not get_par(set_zogy.keep_tmp,tel):

                # remove
                list2remove = [ref_fits, ref_fits_mask, ref_fits_bkg_std]
                remove_files (list2remove, verbose=True)



        # need to take care of objects closer than 32/2 pixels to
        # the full image edge in creation of thumbnails - results
        # in an error if transients are close to the edge

        # update header_new with header_trans
        header2add = header_new.copy(strip=True)
        header2add.update(header_trans.copy(strip=True))

        # add various header keywords from the reference catalog to
        # the transient catalog
        add_refkeys(header2add, header_ref_cat)


        format_cat (cat_trans, cat_trans_out, cat_type='trans',
                    header2add=header2add,
                    exptime=exptime_new,
                    apphot_radii=get_par(set_zogy.apphot_radii,tel),
                    dict_thumbnails=dict_thumbnails,
                    save_thumbnails=get_par(set_zogy.save_thumbnails, tel),
                    size_thumbnails=get_par(set_zogy.size_thumbnails, tel),
                    ML_calc_prob=get_par(set_zogy.ML_calc_prob,tel),
                    ML_prob_real=ML_prob_real,
                    nfakestars= get_par(set_zogy.nfakestars,tel),
                    tel=tel, set_zogy=set_zogy)

        # test: calculate old ML_prob_real on the fits catalog
        #ML_model = get_par(set_zogy.ML_model,tel)
        #ML_prob_real_old = get_ML_prob_real_old (cat_trans_out, ML_model)


    mem_use (label='just after formatting catalogs')

    end_time = os.times()
    if new and ref:
        dt_usr  = end_time[2] - start_time2[2]
        dt_sys  = end_time[3] - start_time2[3]
        dt_wall = end_time[4] - start_time2[4]

        log.info('Elapsed user time in optsub: {:.2f}s'.format(dt_usr))
        log.info('Elapsed CPU time in optsub:  {:.2f}s'.format(dt_sys))
        log.info('Elapsed wall time in optsub: {:.2f}s'.format(dt_wall))

    dt_usr  = end_time[2] - start_time1[2]
    dt_sys  = end_time[3] - start_time1[3]
    dt_wall = end_time[4] - start_time1[4]

    log.info('Elapsed user time in zogy:   {:.2f}s'.format(dt_usr))
    log.info('Elapsed CPU time in zogy:    {:.2f}s'.format(dt_sys))
    log.info('Elapsed wall time in zogy:   {:.2f}s'.format(dt_wall))



    if new and ref:
        # return new and zogy header separately
        return header_new, header_trans
    elif new:
        return header_new
    elif ref:
        return header_ref




################################################################################

def add_refkeys (header_trans, header_ref):

    for key in ['S-SEEING', 'S-SEESTD', 'S-ELONG', 'S-ELOSTD',
                'PC-ZP', 'PC-ZPSTD', 'PC-ZPERR', 'LIMMAG', 'QC-FLAG']:

        if key in header_ref:

            # replace string before the dash (S-, PC-, etc.) with R-
            if key not in ['QC-FLAG']:
                key_ref = 'R-{}'.format(key.split('-')[-1])
            else:
                key_ref = 'R-QCFLAG'


            # add to transient header
            header_trans[key_ref] = (header_ref[key], header_ref.comments[key])

        else:

            log.warning ('keyword {} not present in reference header for {}'
                         .format(key, base_ref))


    return


################################################################################

def add_zperr (table_in, header, header_ref=None, imtype='trans', tel=None):

    """add photometric zeropoint error (taken from the header keyword
       PC-ZPERR or PC-ZPSTD) to all flux measurement error columns of
       [table_in] that start with FNUERR, and record the result in
       column FNUERRTOT with the same ending. This can be used for
       'new', 'ref' or 'trans' tables indicated with [imtype] - for
       transient catalogs both header and header_ref need to be
       defined. If imtype is new (ref), the header should refer to
       that of the new (ref) image or catalog. The input table
       [table_in] is updated in place.
    """


    if imtype=='trans' and header_ref is None:
        log.error ('parameter header_ref is not defined while imtype=\'trans\' '
                   'in zogy.add_zperr(); total errors related to _REF columns '
                   'in the transient catalog will not be computed correctly')


    # pogson constant
    pogson = 2.5/np.log(10)

    # loop columns
    for col in table_in.colnames:

        # initialize zperr to zero in case it fails to be defined
        # below
        zperr = 0

        # pick FNUERR columns
        if col.startswith('FNUERR_'):

            fnu = table_in[col.replace('FNUERR', 'FNU')].value
            fnuerr = table_in[col].value


            # this concerns columns in the input transient table
            # related to the reference image
            if (imtype == 'trans' and header_ref is not None
                and col.endswith('_REF')):

                if 'PC-ZPERR' in header_ref:
                    zperr = header_ref['PC-ZPERR']
                elif 'PC-ZPSTD' in header_ref:
                    zperr = header_ref['PC-ZPSTD']
                else:
                    log.error ('neither PC-ZPERR nor PC-ZPSTD found in ref '
                               'header in zogy.add_zperr(); adopting zperr=0')


            # for non-ML/BG images or (ML/BG) reference catalogs, no
            # need to take care of different read-out channels
            elif imtype=='ref' or tel not in ['ML1', 'BG2', 'BG3', 'BG4']:

                if 'PC-ZPERR' in header:
                    zperr = header['PC-ZPERR']
                elif 'PC-ZPSTD' in header:
                    zperr = header['PC-ZPSTD']
                else:
                    log.error ('neither PC-ZPERR nor PC-ZPSTD found in {}'
                               'header in zogy.add_zperr(); adopting zperr=0'
                               .format(imtype))

            else:

                # ML/BG: for new full-source and transient catalog,
                # need to convert pixel coordinates to channel number
                # as zeropoint is channel dependent
                if imtype == 'trans':

                    #xcoords = table_in['X_PSF_D']
                    #ycoords = table_in['Y_PSF_D']
                    xcoords = table_in['X_PEAK']
                    ycoords = table_in['Y_PEAK']

                elif imtype == 'new':

                    xcoords = table_in['X_POS']
                    ycoords = table_in['Y_POS']


                # N.B.: for coordinates off the image, channels will
                # be zero!!!
                channels = coords2chan (xcoords, ycoords)

                # add channels to table
                if 'CHANNEL' not in table_in.colnames:
                    table_in['CHANNEL'] = channels


                if 'PC-ZPERR' in header:
                    zperr = np.array([header['PC-ZPE{}'.format(chan)]
                                      for chan in channels])
                elif 'PC-ZPSTD' in header:
                    zperr = np.array([header['PC-ZPS{}'.format(chan)]
                                      for chan in channels])
                else:
                    log.error ('neither PC-ZPERR nor PC-ZPSTD found in {} '
                               'header in zogy.add_zperr(); adopting zperr=0'
                               .format(imtype))



            # add zeropoint error to measurement error
            fnuerr_zp = zperr * np.abs(fnu) / pogson
            fnuerrtot = np.sqrt(fnuerr**2 + fnuerr_zp**2).astype('float32')


            # for transient fluxes (_ZOGY and _PSF_D), additional
            # error due to comparison with reference image was
            # initially added here, but this is now (Oct 2024) done
            # inside [run_ZOGY] and when performing the PSF_D fit in
            # [get_trans] by using the error in the weighted mean of
            # fratio calculated in [get_fratio_dxdy] and recorded in
            # header keyword Z-FNRERR


            # record in new table column
            table_in[col.replace('FNUERR', 'FNUERRTOT')] = fnuerrtot



    return


################################################################################

def add_magcols (table_in, overwrite=True):

    # loop columns
    for col in table_in.colnames:

        # pick FNU columns
        if col.startswith('FNU_'):

            fnu = table_in[col]
            fnuerr = table_in[col.replace('FNU', 'FNUERR')]
            fnuerrtot = table_in[col.replace('FNU', 'FNUERRTOT')]


            # for transient flux, convert to absolute value so that
            # magnitude will not be set to 99
            if 'ZOGY' in col or 'PSF_D' in col:
                fnu = np.abs(fnu)


            # convert to mag using zogy.fnu2mag()
            mag, magerr, magerrtot = fnu2mag (fnu, fnuerr, fnuerrtot)


            # record in new table columns; if the mag/magerr/magerrtot
            # column already exists and overwrite is True, it will be
            # overwritten
            cols_dict = {'MAG': mag, 'MAGERR': magerr, 'MAGERRTOT': magerrtot}
            for key in cols_dict:
                col_mag = col.replace('FNU', key)
                if overwrite or col_mag not in table_in.colnames:
                    table_in[col_mag] = cols_dict[key]


    return


################################################################################

def strip_hdrkeys (header, keys2strip=None):

    # copy of input header
    header = header.copy()

    # strip image- and table-dependent keywords using astropy strip
    header.strip()

    # strip additional keywords if provided
    if keys2strip is not None:
        for key in keys2strip:
            if key in header:
                del header[key]


    return header


################################################################################

def trans_crossmatch_fphot (fits_transcat, fits_red, fits_red_mask, sexcat_new,
                            fits_ref, fits_ref_mask, fits_cat_ref, nthreads=1):

    """Cross match entries in transient catalog with the Gaia DR3
    catalog and the reference catalog to extract the nearest
    matches. Also perform forced photometry on the reference and
    reduced image at the transient position to extract the
    corresponding optimal magnitudes in those images. Altogether this
    will add the following columns to the input catalog:


    SOURCE_ID_NEAR_GAIA:  source ID of Gaia DR3 source nearest to the transient
    SEP_NEAR_GAIA:        [arcsec] separation to nearest Gaia DR3 source
    MAG_G_NEAR_GAIA       [mas] Gaia DR3 G-band magnitude (Vega) of nearest
                          source
    MAGERR_G_NEAR_GAIA    [mag] corresponding error
    SEP_BRIGHT_GAIA       [arcsec] separation to nearest bright (G<14) Gaia DR3
                          source
    MAG_G_BRIGHT_GAIA     [mag] Gaia DR3 G-band magnitude of nearest bright
                          (G<14) source


    NUMBER_NEAR_REF       number of reference catalogue source nearest to the
                          transient
    SEP_NEAR_REF:         [arcsec] separation to nearest source in ref catalogue
    FWHM_NEAR_REF         [pix] FWHM of nearest source in reference catalogue
    ELONG_NEAR_REF        elongation of nearest source in reference catalogue
    CLASS_STAR_NEAR_REF   SExtractor star/galaxy classification of nearest
                          ref-image source
    MAG_AUTO_NEAR_REF:    [mag] Kron/AUTO magnitude (AB) of nearest source in ref
                          catalog
    MAGERR_AUTO_NEAR_REF: [mag] corresponding error
    FNU_OPT_REF           [microJy] forced-photometry flux in reference image at
                          transient position
    FNUERR_OPT_REF        [microJy] corresponding error
    FLAGS_MASK_REF        OR comb. of bad pixels inner PSF profile in reference
                          image

    (BACKGROUND_REF:       [e-] estimated source background in global
                           background-subtracted ref image)


    SEP_NEAR_RED:         [arcsec] separation to nearest source in reduced image
    FWHM_NEAR_RED         [pix] FWHM of nearest source in reduced image
    ELONG_NEAR_RED        elongation of nearest source in reduced image
    CLASS_STAR_NEAR_RED   SExtractor star/galaxy classification of nearest
                          red-image source
    FNU_OPT_RED           [microJy] forced-photometry flux in reduced image at
                          transient position
    FNUERR_OPT_RED        [microJy] corresponding error
    FLAGS_MASK_RED        OR comb. of bad pixels inner PSF profile in reduced
                          image

    (BACKGROUND_RED:       [e-] estimated source background in global
                           background-subtracted red image)


    The above columns will be added to the input catalog, which will
    be overwritten.

    """

    # import forced photometry; done here and not at the top to try to
    # avoid cyclical imports as force_phot module contains import zogy
    import force_phot


    if get_par(set_zogy.timing,tel): t = time.time()
    log.info ('executing trans_crossmatch_fphot ...')


    # read selected columns from input transient catalog
    table_trans = Table.read(fits_transcat)

    #cols = ['RA_PEAK', 'DEC_PEAK', 'RA_PSF_D', 'DEC_PSF_D', 'CHI2_PSF_D']
    #table_trans = Table(fitsio.read(fits_transcat, ext=-1, columns=cols))

    # use RA/DEC_PSF_D
    ra_trans = table_trans['RA_PSF_D'].value
    dec_trans = table_trans['DEC_PSF_D'].value

    # could adopt RA/DEC peak in case PSF fit to D did not succeed
    if False:
        mask_peak = (table_trans['CHI2_PSF_D'].value == 0)
        ra_trans[mask_peak] = table_trans['RA_PEAK'][mask_peak].value
        dec_trans[mask_peak] = table_trans['DEC_PEAK'][mask_peak].value


    # crossmatch with Gaia DR3
    # ------------------------

    log.info ('cross-matching transients with Gaia DR3')

    # get some header info from reduced image
    header_new = read_hdulist(fits_red, get_data=False, get_header=True)
    keywords = ['naxis2', 'naxis1', 'pixscale', 'obsdate']
    ysize_new, xsize_new, pixscale_new, obsdate_new = read_header(header_new,
                                                                  keywords)
    if 'A-ROT' in header_new:
        rot_new = header_new['A-ROT']
    else:
        rot_new = 0


    # use functions [get_gaia_image_table] and - only if the
    # corresponding fits table does not exist yet -
    # [get_imagestars_hpindex] to extract the entries from the Gaia
    # input catalog that are relevant for this image, and apply a
    # proper motion correction
    fits_gaia = get_par(set_zogy.gaia_cat,tel)
    ra_center = header_new['RA-CNTR']
    dec_center = header_new['DEC-CNTR']
    # field-of-view
    fov_half_deg = np.amax([xsize_new, ysize_new]) * pixscale_new / 3600 / 2
    table_gaia = get_gaia_image_table (
        fits_red, fits_gaia, obsdate_new, ra_center, dec_center, fov_half_deg,
        rot_new)


    # find star in Gaia catalog nearest to transient coordinates
    ra_gaia = table_gaia['ra'].value
    dec_gaia = table_gaia['dec'].value
    __, index_gaia, sep_gaia, sep_ra, sep_dec = get_matches (
        ra_trans, dec_trans, ra_gaia, dec_gaia, return_offsets=True)


    # add relevant columns to table_trans
    table_trans['SOURCE_ID_NEAR_GAIA'] = table_gaia['source_id'][index_gaia]
    table_trans['SEP_NEAR_GAIA'] = sep_gaia.astype('float32')
    table_trans['MAG_G_NEAR_GAIA'] = table_gaia['mag_G'][index_gaia]
    table_trans['MAGERR_G_NEAR_GAIA'] = table_gaia['magerr_G'][index_gaia]


    # find nearest G<14 star in Gaia
    mask_bright = (table_gaia['mag_G'] < 14)
    table_gaia_bright = table_gaia[mask_bright]

    ra_gaia_bright = table_gaia_bright['ra'].value
    dec_gaia_bright = table_gaia_bright['dec'].value
    __, index_gaia_bright, sep_gaia_bright, __, __ = get_matches (
        ra_trans, dec_trans, ra_gaia_bright, dec_gaia_bright,
        return_offsets=True)


    # add relevant columns to table_trans
    table_trans['SEP_BRIGHT_GAIA'] = sep_gaia_bright.astype('float32')
    table_trans['MAG_G_BRIGHT_GAIA'] = \
        table_gaia_bright['mag_G'][index_gaia_bright]



    # crossmatch with reference catalog
    # ---------------------------------

    log.info ('cross-matching transients with reference catalog')

    # read ref catalog
    table_ref = Table.read(fits_cat_ref)

    # find star in ref catalog nearest to transient coordinates
    ra_ref = table_ref['RA'].value
    dec_ref = table_ref['DEC'].value
    __, index_ref, sep_ref, sep_ra, sep_dec = get_matches (
        ra_trans, dec_trans, ra_ref, dec_ref, return_offsets=True)


    # add relevant columns
    table_trans['NUMBER_NEAR_REF'] = table_ref['NUMBER'][index_ref]
    table_trans['SEP_NEAR_REF'] = sep_ref.astype('float32')
    table_trans['FWHM_NEAR_REF'] = table_ref['FWHM'][index_ref]
    table_trans['ELONG_NEAR_REF'] = table_ref['ELONGATION'][index_ref]
    table_trans['CLASS_STAR_NEAR_REF'] = table_ref['CLASS_STAR'][index_ref]
    table_trans['MAG_AUTO_NEAR_REF'] = table_ref['MAG_AUTO'][index_ref]
    table_trans['MAGERR_AUTO_NEAR_REF'] = table_ref['MAGERR_AUTO'][index_ref]


    # forced photometry in ref image
    # ------------------------------

    log.info ('performing forced photometry in reference image')

    # create input table to run forced photometry
    ntrans = len(ra_trans)
    index_in = np.arange(ntrans)
    table_fphot_ref = Table([index_in, ra_trans, dec_trans],
                            names=['INDEX_IN', 'RA_IN', 'DEC_IN'])

    # create dictionary: {image: [indices in table to consider]}
    image_indices_dict_ref = {fits_ref: np.arange(ntrans)}


    # set various parameters to be able to run forced photometry
    nsigma = get_par(set_zogy.source_nsigma,tel)
    apphot_radii = get_par(set_zogy.apphot_radii,tel)

    # use local or global background?
    bkg_global = (get_par(set_zogy.bkg_phototype,tel) == 'global')

    # background annulus radii, objmask and limiting fraction
    bkg_radii = get_par(set_zogy.bkg_radii,tel)
    bkg_objmask = get_par(set_zogy.bkg_objmask,tel)
    bkg_limfrac = get_par(set_zogy.bkg_limfrac,tel)


    # epoch to use for proper motion correction
    pm_epoch = get_par(set_zogy.cal_epoch,tel)


    # run forced photometry; N.B.: reference image is run as new image
    # (fullsource=True) to allow using the ref images that have been
    # copied to the tmp folder; otherwise the force_phot module will
    # look for the image in the reference folder/bucket
    table_fphot_ref = force_phot.force_phot (
        table_fphot_ref, image_indices_dict_ref, mask_list=[fits_ref_mask],
        trans=False, ref=False, fullsource=True, nsigma=nsigma,
        apphot_radii=apphot_radii, bkg_global=bkg_global, bkg_radii=bkg_radii,
        bkg_objmask=bkg_objmask, bkg_limfrac=bkg_limfrac, pm_epoch=pm_epoch,
        tel=tel, ncpus=nthreads)


    # add relevant columns; need to order by column 'INDEX_IN' and
    # take care that some input coordinates might be missing in output
    # table because the forced photometry failed (e.g. source just off
    # the image)
    index_out = table_fphot_ref['INDEX_IN'].value

    # initialise columns
    table_trans['FNU_OPT_REF'] = np.zeros(ntrans, dtype='float32')
    table_trans['FNUERR_OPT_REF'] = np.zeros(ntrans, dtype='float32')
    table_trans['FLAGS_MASK_REF'] = np.zeros(ntrans, dtype='uint8')

    # fill values at correct index
    table_trans['FNU_OPT_REF'][index_out] = table_fphot_ref['FNU_OPT']
    table_trans['FNUERR_OPT_REF'][index_out] = table_fphot_ref['FNUERR_OPT']
    table_trans['FLAGS_MASK_REF'][index_out] = table_fphot_ref['FLAGS_MASK']


    # crossmatch with new source-extractor catalog
    # --------------------------------------------

    log.info ('cross-matching transients with new source-extractor catalog')

    # read new source-extractor catalog
    table_new = Table.read(sexcat_new)

    # find star in new catalog nearest to transient coordinates
    ra_new = table_new['RA'].value
    dec_new = table_new['DEC'].value
    __, index_new, sep_new, sep_ra, sep_dec = get_matches (
        ra_trans, dec_trans, ra_new, dec_new, return_offsets=True)


    # add relevant columns
    table_trans['SEP_NEAR_RED'] = sep_new.astype('float32')
    table_trans['FWHM_NEAR_RED'] = table_new['FWHM'][index_new]
    table_trans['ELONG_NEAR_RED'] = table_new['ELONGATION'][index_new]
    table_trans['CLASS_STAR_NEAR_RED'] = table_new['CLASS_STAR'][index_new]



    # forced photometry in reduced image
    # ----------------------------------

    log.info ('performing forced photometry in reduced image')

    # create input table to run forced photometry
    ntrans = len(ra_trans)
    index_in = np.arange(ntrans)
    table_fphot_red = Table([index_in, ra_trans, dec_trans],
                            names=['INDEX_IN', 'RA_IN', 'DEC_IN'])

    # create dictionary: {image: [indices in table to consider]}
    image_indices_dict_red = {fits_red: np.arange(ntrans)}


    # run forced photometry
    table_fphot_red = force_phot.force_phot (
        table_fphot_red, image_indices_dict_red, mask_list=[fits_red_mask],
        trans=False, ref=False, fullsource=True, nsigma=nsigma,
        apphot_radii=apphot_radii, bkg_global=bkg_global, bkg_radii=bkg_radii,
        bkg_objmask=bkg_objmask, bkg_limfrac=bkg_limfrac, pm_epoch=pm_epoch,
        tel=tel, ncpus=nthreads)


    # add relevant columns; need to order by column 'INDEX_IN' and
    # take care that some input coordinates might be missing in output
    # table because the forced photometry failed (e.g. source just off
    # the image)
    index_out = table_fphot_red['INDEX_IN'].value

    # initialise columns
    table_trans['FNU_OPT_RED'] = np.zeros(ntrans, dtype='float32')
    table_trans['FNUERR_OPT_RED'] = np.zeros(ntrans, dtype='float32')
    table_trans['FLAGS_MASK_RED'] = np.zeros(ntrans, dtype='uint8')

    # fill values at correct index
    table_trans['FNU_OPT_RED'][index_out] = table_fphot_red['FNU_OPT']
    table_trans['FNUERR_OPT_RED'][index_out] = table_fphot_red['FNUERR_OPT']
    table_trans['FLAGS_MASK_RED'][index_out] = table_fphot_red['FLAGS_MASK']


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in trans_crossmatch_fphot')


    return table_trans


################################################################################

def display_subs (base_new, base_ref, nsubs, names):

    for nsub in range(nsubs):

        if show_sub(nsub):

            cmd = ['ds9','-zscale','-frame','lock','image']
            subend = 'sub{}.fits'.format(nsub)

            for name in names:
                base_name = name.replace('.fits','')
                cmd.append('{}_{}'.format(base_name, subend))

            cmd.append('{}_VSn_{}'.format(base_new, subend))
            cmd.append('{}_VSr_{}'.format(base_new, subend))
            cmd.append('{}_VSn_ast_{}'.format(base_new, subend))
            cmd.append('{}_VSr_ast_{}'.format(base_new, subend))
            cmd.append('{}_Sn_{}'.format(base_new, subend))
            cmd.append('{}_Sr_{}'.format(base_new, subend))
            cmd.append('{}_kn_{}'.format(base_new, subend))
            cmd.append('{}_kr_{}'.format(base_new, subend))
            cmd.append('{}_Pn_hat_{}'.format(base_new, subend))
            cmd.append('{}_Pr_hat_{}'.format(base_new, subend))

            cmd.append('{}_psf_ima_config_{}'.format(base_new, subend))
            cmd.append('{}_psf_ima_config_{}'.format(base_ref, subend))
            cmd.append('{}_psf_ima_{}'.format(base_new, subend))
            cmd.append('{}_psf_ima_{}'.format(base_ref, subend))
            cmd.append('{}_psf_ima_center_{}'.format(base_new, subend))
            cmd.append('{}_psf_ima_center_{}'.format(base_ref, subend))
            cmd.append('{}_psf_ima_shift_{}'.format(base_new, subend))
            cmd.append('{}_psf_ima_shift_{}'.format(base_ref, subend))
            cmd.append('{}_PD_{}'.format(base_new, subend))

            result = subprocess.run(cmd)


################################################################################

def create_subs (name_full, dir_sub, nsubs, cuts_ima, dtype_sub, index_extract,
                 header=None):

    # read full image (could be fits or numpy file)
    ext = name_full.split('.')[-1]
    if 'fits' in ext:
        data_full = read_hdulist(name_full)
    else:
        data_full = np.load(name_full)

    size_fft = (get_par(set_zogy.subimage_size,tel) +
                2*get_par(set_zogy.subimage_border,tel))
    shape_fft = (size_fft, size_fft)

    # loop subimages
    for nsub in range(nsubs):

        # save subimage only for particular subimages defined in
        # function [show_sub]
        if show_sub(nsub):

            # extract data_sub from data_full
            subcut = cuts_ima[nsub]
            index_subcut = tuple([slice(subcut[0],subcut[1]),
                                  slice(subcut[2],subcut[3])])
            data_sub = np.zeros(shape_fft, dtype=dtype_sub)
            data_sub[index_extract] = data_full[index_subcut]

            # save subimages in [base_sub] folder
            name_sub = '{}/{}'.format(dir_sub, name_full.split('/')[-1])
            # change extension
            name_sub = name_sub.replace('.fits', '_sub{}.fits'.format(nsub))
            fits.writeto(name_sub, data_sub, header, overwrite=True)


    return


################################################################################

def create_full (name_full, header_full, shape_full, dtype_full, nsubs, cuts_ima,
                 index_extract, data=None, sublist=None):

    # create full data array to fill
    data_full = np.zeros(shape_full, dtype=dtype_full)

    # loop subimages
    for nsub in range(nsubs):

        # put sub images without the borders (index_extract) into the
        # right subimage (index_subcut) in data_full
        subcut = cuts_ima[nsub]
        index_subcut = tuple([slice(subcut[0],subcut[1]),
                              slice(subcut[2],subcut[3])])

        if data is not None:
            data_sub = data[nsub]

        elif sublist is not None:
            # read subimage data from sublist element depending on
            # extension, use np.load or read_hdulist; this depends on
            # how files were saved at end of function [run_ZOGY] which
            # is determined by the parameter [set_zogy.display]
            ext = sublist[nsub].split('.')[-1]
            if 'fits' in ext:
                data_sub = read_hdulist(sublist[nsub])
            else:
                data_sub = np.load(sublist[nsub], mmap_mode='c')


        # paste subimage into data_full
        data_full[index_subcut] = data_sub[index_extract]


    # write full output image
    fits.writeto (name_full, data_full, header_full, overwrite=True)

    return name_full


################################################################################

def merge_fakestars (table, table_fake, cat_type, header):

    """function to merge [table_fake] with [table] of type [cat_type],
       such that all entries in the output table contain the columns
       from [table_fake], but only the fake stars have nonzero values
       for those columns.

    """

    if cat_type == 'trans':
        xcoords = table['X_POS_SCORR']
        ycoords = table['Y_POS_SCORR']

    else:
        xcoords = table['X_POS']
        ycoords = table['Y_POS']


    # matching radius in arcseconds
    dist_max = 2

    if True:

        # convert pixel coordinates to world
        wcs = WCS(header)
        ra, dec = wcs.all_pix2world(xcoords, ycoords, 1)
        ra_fake, dec_fake = wcs.all_pix2world(table_fake['X_FAKE'],
                                              table_fake['Y_FAKE'], 1)

        # find matches between table and table_fake
        index_fake, index = get_matches (ra_fake, dec_fake, ra, dec,
                                         dist_max=dist_max, return_offsets=False)

    else:

        # alternatively, find matches between table and table_fake
        # using pixel coordinates
        xcoords_fake = table_fake['X_FAKE']
        ycoords_fake = table_fake['Y_FAKE']
        if 'A-PSCALE' in header and header['A-PSCALE'] != 0:
            dist_max /= float(header['A-PSCALE'])
        else:
            dist_max = 5

        index_fake, index = get_matches_pix (xcoords_fake, ycoords_fake,
                                             xcoords, ycoords, dist_max=dist_max,
                                             return_offsets=False)


    # add zero-valued fake columns to table
    table.add_columns([0.0]*5, names=['X_FAKE', 'Y_FAKE',
                                     'SNR_FAKE_IN', 'E_FLUX_FAKE_IN',
                                     'MAG_FAKE_IN'])

    table['X_FAKE'][index] = table_fake['X_FAKE'][index_fake]
    table['Y_FAKE'][index] = table_fake['Y_FAKE'][index_fake]
    table['SNR_FAKE_IN'][index] = table_fake['SNR_FAKE_IN'][index_fake]
    table['E_FLUX_FAKE_IN'][index] = table_fake['E_FLUX_FAKE_IN'][index_fake]

    keywords = ['exptime', 'filter']
    exptime, filt = read_header(header, keywords)
    zeropoint = header['PC-ZP']
    airmass = header['AIRMASSC']

    flux_fake_in = table_fake['E_FLUX_FAKE_IN'][index_fake]
    ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]
    mag_fake_in = apply_zp (flux_fake_in, zeropoint, airmass, exptime, ext_coeff)
    table['MAG_FAKE_IN'][index] = mag_fake_in


    return table


################################################################################

def extract_fakestars_orig (table_fake, table_trans, data_Fpsf, data_Fpsferr,
                            data_Scorr, pixscale):

    # extract flux and flux error from Fpsf and Fpsferr; using
    # indices corresponding to table_fake x,y coordinates
    x_fake = (table_fake['X']+0.5).astype(int) - 1
    y_fake = (table_fake['Y']+0.5).astype(int) - 1
    table_fake['E_FLUX_OUT'] = data_Fpsf[y_fake, x_fake]
    table_fake['E_FLUXERR_OUT'] = data_Fpsferr[y_fake, x_fake]
    # and S/N from Scorr
    table_fake['SNR_OUT'] = data_Scorr[y_fake, x_fake]


    # add zero-valued fake columns to table_trans
    table_trans.add_columns([0.0]*4, names=['X_FAKE', 'Y_FAKE',
                                           'SNR_FAKE_IN', 'E_FLUX_FAKE_IN'])

    # match entries in table_trans with table_fake within some distance
    # and add table_fake column values to that specific entry
    dist2_max = (3./pixscale)**2 #pixels
    for i_fake in range(len(table_fake)):
        dist2 = ((table_trans['X_POS_SCORR'] - table_fake['X'][i_fake])**2 +
                 (table_trans['Y_POS_SCORR'] - table_fake['Y'][i_fake])**2)
        dist2_min, i_trans = np.amin(dist2), np.argmin(dist2)
        #log.info ('dist2_min: {}, i_trans: {}'.format(dist2_min, i_trans))
        if dist2_min <= dist2_max:
            table_trans['X_FAKE'][i_trans] = table_fake['X'][i_fake]
            table_trans['Y_FAKE'][i_trans] = table_fake['Y'][i_fake]
            table_trans['SNR_FAKE_IN'][i_trans] = table_fake['SNR_IN'][i_fake]
            table_trans['E_FLUX_FAKE_IN'][i_trans] = table_fake['E_FLUX_IN'][i_fake]
            # replace peak-pixel OUT values with actually extracted values
            table_fake['E_FLUX_OUT'][i_fake] = table_trans['E_FLUX_ZOGY'][i_trans]
            table_fake['E_FLUXERR_OUT'][i_fake] = table_trans['E_FLUXERR_ZOGY'][i_trans]
            table_fake['SNR_OUT'][i_fake] = table_trans['SNR_ZOGY'][i_trans]


    # overwrite output fits catalog created in [get_trans]
    table_trans.write('{}.transcat'.format(base_newref), format='fits',
                      overwrite=True)


    # create plot comparing input and extracted fluxes
    # compare input and output flux
    fluxdiff = ((table_fake['E_FLUX_IN'] - table_fake['E_FLUX_OUT']) /
                table_fake['E_FLUX_IN'])
    fluxdiff_err = table_fake['E_FLUXERR_OUT'] / table_fake['E_FLUX_IN']

    fd_mean, fd_median, fd_std = sigma_clipped_stats(fluxdiff.astype(float))
    fderr_mean, fderr_median, fderr_std = sigma_clipped_stats(fluxdiff_err
                                                              .astype(float))

    # write to ascii file
    filename = '{}_fakestars.dat'.format(base_new)
    f = open(filename, 'w')
    f.write('{:1} {:11} {:11} {:12} {:12} {:16} {:11} {:11}\n'
            .format('#', 'xcoord[pix]', 'ycoord[pix]', 'flux_in[e-]',
                    'flux_out[e-]', 'fluxerr_out[e-]', 'S/N_input',
                    'S/N_output'))

    for i in range(len(table_fake)):
        f.write('{:11.2f} {:11.2f} {:12.2e} {:12.2e} {:16.2e} {:11.2f} '
                '{:11.2f}\n'
                .format(table_fake['X'][i], table_fake['Y'][i],
                        table_fake['E_FLUX_IN'][i], table_fake['E_FLUX_OUT'][i],
                        table_fake['E_FLUXERR_OUT'][i],
                        table_fake['SNR_IN'][i],
                        table_fake['SNR_OUT'][i]))
    f.close()

    # make comparison plot of flux input and output
    if get_par(set_zogy.make_plots,tel):

        nfake = len(table_fake)
        x = np.arange(nfake)+1

        plt.errorbar(x, fluxdiff, yerr=fluxdiff_err, linestyle='None',
                     ecolor='k', capsize=2)
        plt.plot(x, fluxdiff, 'o', color='tab:blue', markersize=7,
                 markeredgecolor='k')
        plt.xlabel('fakestar number (total: {})'.format(nfake))
        plt.ylabel('(true flux - ZOGY flux) / true flux')
        plt.title('true flux vs. ZOGY Fpsf; mean:{:.3f}, std:{:.3f}, '
                  'data err:{:.3f}'.format(fd_mean, fd_std, fderr_mean))
        plt.savefig('{}_fakestar_flux_in_vs_ZOGYout.pdf'
                    .format(base_newref))
        if get_par(set_zogy.show_plots,tel): plt.show()
        plt.close()

        # same for S/N as determined by Scorr
        y = table_fake['SNR_OUT']
        plt.plot(x, y, 'o', color='tab:blue', markersize=7,
                 markeredgecolor='k')
        plt.xlabel('fakestar number (total: {})'.format(nfake))
        plt.ylabel('S/N from Scorr')
        plt.title('fakestars signal-to-noise ratio from Scorr')
        plt.savefig('{}_fakestar_SNR_ZOGYoutput.pdf'.format(base_newref))
        if get_par(set_zogy.show_plots,tel): plt.show()
        plt.close()


    return


################################################################################

def show_sub (nsub):

    # if nsub is corner or central subimage of subimage array,
    # return True, otherwise return False

    # create 2D integer array of subimages
    subsize = get_par(set_zogy.subimage_size,tel)
    ysize_new, xsize_new = get_par(set_zogy.shape_new,tel)
    nx = int(xsize_new / subsize)
    ny = int(ysize_new / subsize)
    arr = np.arange(nx*ny).reshape(ny,nx)
    subs2show = [arr[0,0], arr[0,-1], arr[ny//2,nx//2], arr[-1,0], arr[-1,-1]]

    if nsub in subs2show:
        return True
    else:
        return False


################################################################################

def robust_Z_norm (ims):
    """
    Applies robust Z norming to images

    Parameters
    ----------
    ims : numpy array with shape (batch, dim1, dim2, channels)
      Images to be normalized

    Returns
    ---------
    numpy array with shape (batch, dim1, dim2, channels)
      array of normalized images
    """


    org_shape = ims.shape # save original shape
    # Flatten and calculate median and median absolute deviation per image
    flat = ims.reshape((org_shape[0],org_shape[1]*org_shape[2],org_shape[3]))
    med = np.median(flat, axis = 1).reshape(org_shape[0],1,org_shape[3])
    mad = np.median(np.abs(flat-med), axis = 1).reshape(org_shape[0],1,
                                                        org_shape[3])

    # suggested addition by Fiore to avoid dividing by zero
    # when returning values
    epsilon = 1e-6
    mad = np.maximum(mad, epsilon)

    # Apply robust Z score formula according to:
    # "https://cloudxlab.com/assessment/displayslide/6286/robust-z-score-method"
    return ((0.6745*(flat-med))/ mad).reshape(org_shape)


################################################################################

def central_crop (ims, new_size):
    """
    Crops center of images

    Parameters
    ----------
    ims : numpy array with shape (batch, dim1, dim2, channels)
      Images to be normalised
    new_size : int
      new size of sides of images after cropping

    Returns
    ---------
    numpy array wih shape (batch, new_size, new_size, channels)
      cropped array
    """
    # Run through image axis
    for i in range(1,3):
        rel_side = ims.shape[i]
        if rel_side > new_size:
            border = (rel_side - new_size)//2
            # Slice along relevant axis
            ims = ims.take(indices = range(border,border+new_size), axis = i)

    return ims


################################################################################

def get_probability (events, model_file, normed_size = 40):
    """
    Predicts probability of event being real according to supplied model file

    Parameters
    ----------
    events : numpy array with shape (batch, dim1, dim2, channels)
      Events on which prediction should be applied. dim1 and dim2 must be at
      least larger than normed_size (default 40)
    model_file :  str
      location of tensorflow model file
    normed_size : int
      Size of images used in the normalization, defaults to 40. Can be same size
      as input of model

    Returns
    ---------
    numpy array with shape (batch)
      array of values between 0 and 1 giving probability of corresponding event
      in input array being real
  """

    # tensorflow
    import tensorflow as tf
    # to avoid tensorflow info and warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    # Import model and extract input shape
    model = tf.keras.models.load_model(model_file)
    input_shape = model.layers[0].input_shape[0]

    # Run checks on inputs
    if input_shape[1] != input_shape[2]:
        raise Exception("Model input does not meet expectation of square images")
    if input_shape[1] > normed_size:
        raise Exception("normed_size should not be smaller than size of expected "
                        "model input")
    if events.shape[3] != input_shape[3]:
        raise Exception(f"Incorrect number of image channels given. Expected "
                        "{input_shape[3]}")
    if events.shape[1] < normed_size or events.shape[2] <normed_size:
        raise Exception(f"Input images too small, must be at least be size "
                        "defined by normed_size ({normed_size}x{normed_size})")

    # Norm events after cropping center to relevant size
    normed_events = robust_Z_norm(central_crop(events, normed_size))

    # Run prediction and output array
    return model.predict(central_crop(normed_events, input_shape[1]),
                         verbose = False)[:,0]


################################################################################

def get_ML_prob_real_Diederik (dict_thumbnails, model, size_use=40):

    """function based on training by Diederik de Vries and Fiorenzo
    Stoppa of the 2nd vetting data set collected during 2022.  The
    functions above and used here, [robust_Z_norm], [central_crop] and
    [get_probability], were provided by Diederik. Similarly to
    [get_ML_prob_real_Zafiirah], it calculates the probability that a transient
    candidate is real, using the thumbnail files recorded in
    [dict_thumbnails] in combination with the trained model [model].

    In August 2024, Fiore forwarded the latest code from Diederik,
    which is saved here in the functions [get_probability_aug2024],
    [log_z_norm_super] and [central_crop_aug2024]; the latter function
    is only slightly different from the original central_crop (input
    parameter "new_size" is a shape with 4-element tuple, rather than
    an integer), but it is skipped anyway as the central cropping is
    done already inside this current function.

    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info ('executing get_ML_prob_real_Diederik ...')


    # size of thumbnails
    size_tn = get_par(set_zogy.size_thumbnails,tel)

    # thumbnail images are 100x100 pixels (default size_tn is 100),
    # need to extract the central 40x40 (30x30) pixels for Diederiks
    # (august 2024) model; this is also done in function
    # [central_crop] supplied by Diederik, but to save memory usage
    # crop the images as they are read into memory
    if size_tn > size_use:
        border = (size_tn - size_use)//2
        i1 = border
        i2 = border+size_use

        # first slice/dimension is nrows
        index = (slice(None,None), slice(i1,i2), slice(i1,i2))

    else:
        # otherwise, use full thumbnail imagse
        index = (slice(None,None), slice(None,None), slice(None,None))


    # initially list with data to be stacked after the loop
    list_2stack = []

    # loop [dict_thumbnails] and read corresponding numpy files
    for key in dict_thumbnails.keys():

        # data_thumbnail will have shape (nrows, size_use, size_use)
        data_thumbnail = np.load(dict_thumbnails[key], mmap_mode='c')[index]

        # add them to list to stack
        list_2stack.append(data_thumbnail)


    # stack data_thumbnail along last axis; shape expected is (nrows,
    # size, size, 4), where size is at least 40 for get_probability
    # and size equals 30 for get_probability_aug2024
    data_stack = np.stack(list_2stack, axis=-1)


    # obtain probabilities
    if 'supervised' in model:
        prob_real = get_probability_aug2024(data_stack, model)
    else:
        prob_real = get_probability(data_stack, model)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_ML_prob_real_Diederik')


    # return probabilities
    return prob_real


################################################################################

def log_z_norm_super(images, quantiles = [-57.209943275451664,
                                          -31.484833221435547,
                                          -182.8149905395508]):
    """
    Applies normalisation to images for use in supervised classification model

    Parameters
    ----------
    ims : numpy array with shape (batch, dim1, dim2, channels)
      Images to be normalized
    used_quantiles : list of floats
      values used for quantile cutoff of each channel. See thesis for details on
      normalisation function.

    Returns
    ---------
    numpy array with shape (batch, dim1, dim2, channels)
      array of normalized images
    """

    # according to Fiore, used_ims needs to be 4, i.e. number of
    # channels or thumbnail images
    used_ims = np.shape(images)[-1]

    images = np.reshape(images,(-1,900,used_ims)) # Flatten images

    for i in range(np.minimum(used_ims, 3)): # For loop runs faster than parralel
        # Remove lower quantile per image type and shifts
        quant = quantiles[i]
        images[...,i] = np.maximum(images[...,i],quant) - quant + 1

    for i in range(np.minimum(used_ims, 2)):
        # Take logarithm for New and Ref
        images[...,i] = np.log10(images[...,i])

    # Z norm the shifted images
    images = ((images - np.mean(images, axis = 1, keepdims=True)) /
              (np.std(images, axis = 1, keepdims =True)+1e-8))

    return images.reshape((-1,30,30,used_ims))


################################################################################

def central_crop_aug2024 (ims, new_size):

    """
    Crops center of images

    Parameters
    ----------
    ims : numpy array with shape (batch, dim1, dim2, channels)
      Images to be normalised
    new_size : tuple or list
      Shape of images after cropping

    Returns
    ---------
    numpy array wih shape (batch, new_size[1], new_size[2], channels) representing
      the cropped array
    """
    # Run through image axis
    for i in range(1,3):
        rel_side = ims.shape[i]
        if rel_side > new_size[i]:
            border = (rel_side - new_size[i])//2
            # Slice along relevant axis
            ims = ims.take(indices = range(border,border+new_size[i]), axis = i)
    return ims


################################################################################

def get_probability_aug2024 (events, model_file):

    """
    Predicts probability of event being real according to supplied model file

    Parameters
    ----------
    events : numpy array with shape (batch, dim1, dim2, channels)
      Events on which prediction should be applied. dim1 and dim2 must be at
      least larger than required image size for the model. Channels need to match
      channels expected by model.
    model_file :  str
      location of tensorflow model file

    Returns
    ---------
    numpy array with shape (batch)
      array of values between 0 and 1 giving probability of corresponding event
      in input array being real
    """

    # tensorflow
    import tensorflow as tf
    # to avoid tensorflow info and warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    # Import model and extract input shape
    model = tf.keras.models.load_model(model_file)
    input_shape = model.layers[0].input_shape[0]

    # Run checks on inputs
    if len(events.shape) != len(input_shape):
        raise Exception(f"Incorrect number of input dimensions found shape {events.shape} (number of input dimensions should be ({len(input_shape)}). For single events use batch of size 1")
    if events.shape[3] != input_shape[3]:
        raise Exception(f"Incorrect number of image channels given. Expected {input_shape[3]} but found {events.shape[3]}")
    if events.shape[1] < input_shape[1] or events.shape[2] < input_shape[2]:
        raise Exception(f"Event images smaller than model input. Model requires at least size ({input_shape[1]},{input_shape[2]}) but found event images of size ({events.shape[1]},{events.shape[2]})")

    # Norm events after cropping center to relevant size
    #normed_events = log_z_norm_super(central_crop_aug2024 (events, input_shape))
    # skip central_crop; images in events already contain the central 30x30 pixels
    normed_events = log_z_norm_super(events)

    # Run prediction and output array
    return model.predict(normed_events, verbose = False)[:,0]


################################################################################

def get_ML_prob_real_Zafiirah (dict_thumbnails, model, size_use=30,
                               factor_norm=255.):

    # needed for Zafiirah's machine learning package MeerCRAB
    from meerCRAB_code import prediction_phase


    """function based on Zafiirah's Jupyter notebook (see
    https://github.com/Zafiirah13/meercrab) which uses MeerCRAB's
    function [realbogus_prediction] to calculate the probability that
    a transient candidate is real, using the thumbnail files recorded
    in [dict_thumbnails] in combination with the trained model
    [model]. Most models require the central 30x30 pixels to be used
    rather than the full (100x100) thumbnails. The normalisation
    factor 255 is the one applied by Zafiirah to the thumbnails during
    the ML training.

    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info ('executing get_ML_prob_real_Zafiirah ...')

    # read fits table
    #table = Table.read(fits_table)

    # split set_zogy.ML_model parameter into model path and name
    model_path, model_name = os.path.split(model)
    # model_path input into [realbogus_prediction] requires trailing /
    model_path += '/'


    # thumbnail images are 100x100 pixels, need to extract the central
    # 30x30 pixels for most of Zafiirah's models

    # size of thumbnails
    size_tn = get_par(set_zogy.size_thumbnails,tel)

    if size_tn > size_use:
        border = (size_tn - size_use)//2
        i1 = border
        i2 = border+size_use

        # first slice/dimension is nrows
        index = (slice(None,None), slice(i1,i2), slice(i1,i2))

    else:
        # otherwise, use full thumbnail imagse
        index = (slice(None,None), slice(None,None), slice(None,None))


    # initially list with data to be stacked after the loop
    list_2stack = []


    # loop [dict_thumbnails] and read corresponding numpy files
    for key in dict_thumbnails.keys():

        # data_thumbnail will initially have shape (nrows, 100, 100),
        # potentially reduced to (nrows, 30, 30) using [index]
        data_thumbnail = np.load(dict_thumbnails[key], mmap_mode='c')[index]

        # append to list to be stacked, except for the 'SCORR'
        # thumbnail in case modelname does not contain 'NRDS'
        if not ('NRDS' not in model_name and '_SCORR' in key):
            list_2stack.append(data_thumbnail)


    # stack data_thumbnail along last axis; shape expected in
    # [realbogus_prediction] is (nrows, 30 or 100, 30 or 100, 3 or 4)
    # where last dimension depends on whether Scorr is included or not
    data_stack = np.stack(list_2stack, axis=-1)


    # normalise
    data_stack /= factor_norm


    # generate some transient ID (not important but required)
    #id_trans = np.arange(len(table))
    id_trans = np.arange(data_stack.shape[0])


    # threshold (not important but required)
    prob_thresh = 0.5
    ML_real_prob, __ = prediction_phase.realbogus_prediction(
        model_name, data_stack, id_trans, prob_thresh, model_path=model_path)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_ML_prob_real_Zafiirah')


    return ML_real_prob


################################################################################

def orient_data (data, header, header_out=None, MLBG_rot90_flip=False,
                 pixscale=0.564, tel=None):

    """Function to remap [data] from the CD matrix defined in [header] to
    the CD matrix taken from [header_out].  If the latter is not
    provided the output orientation will be North up, East left.

    If [MLBG_rot90_flip] is switched on and the data is from MeerLICHT or
    BlackGEM, the data will be oriented within a few degrees from
    North up, East left while preserving the pixel values in the new,
    *remapped* reference, D and Scorr images.

    """

    # rotation matrix:
    # R = [[dx * cos(theta),  dy * -sin(theta)],
    #      [dx * sin(theta),  dy * cos(theta)]]
    # with theta=0: North aligned with positive y-axis
    # and East with the positive x-axis (RA increases to the East)
    #
    # N.B.: np.dot(R, [[x], [y]]) = np.dot([x,y], R.T)
    #
    # matrices below are defined using the (WCS) header keywords
    # CD?_?:
    #
    # [ CD1_1  CD2_1 ]
    # [ CD1_2  CD2_2 ]
    #
    # orient [data] with its orientation defined in [header] to the
    # orientation defined in [header_out]. If the latter is not
    # provided, the output orientation will be North up, East left.

    # check if input data is square; if it is not, the transformation
    # will not be done properly.
    assert data.shape[0] == data.shape[1]

    # define data CD matrix, assumed to be in [header]
    CD_data = read_CD_matrix (header)

    # determine output CD matrix, either from [header_out] or North
    # up, East left
    if header_out is not None:
        CD_out = read_CD_matrix (header_out)
    else:
        # define the CD matrix with North up and East left, using the
        # input pixel scale
        cdelt = pixscale/3600
        CD_out = np.array([[-cdelt, 0], [0, cdelt]])


    # check if values of CD_data and CD_out are similar
    CD_close = [math.isclose(CD_data[i,j], CD_out[i,j], rel_tol=1e-3)
                for i in range(2) for j in range(2)]


    if np.all(CD_close):

        #log.info ('data CD matrix already similar to CD_out matrix; '
        #          'no need to remap data')

        # if CD matrix values are all very similar, do not bother to
        # do the remapping
        data2return = data

    elif MLBG_rot90_flip and tel in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']:

        #log.info ('for ML/BG: rotating data by exactly 90 degrees and for '
        #          'ML also flip left/right')

        # rotate data by exactly 90 degrees counterclockwise (when
        # viewing data with y-axis increasing to the top!) and for ML1
        # also flip in the East-West direction; for ML/BG this will
        # result in an image within a few degrees of the North up,
        # East left orientation while preserving the original pixel
        # values of the new, *remapped* reference, D and Scorr images.

        data2return = np.rot90(data, k=-1)
        if tel=='ML1':
            data2return = np.fliplr(data2return)

        # equivalent operation: data2return = np.flipud(np.rot90(data))

    else:

        #log.info ('remapping data from input CD matrix: {} to output CD '
        #          'matrix: {}'.format(CD_data, CD_out))

        # transformation matrix, which is the dot product of the
        # output CD matrix and the inverse of the data CD matrix
        CD_data_inv = np.linalg.inv(CD_data)
        CD_trans = np.dot(CD_out, CD_data_inv)

        # transpose and flip because [affine_transform] performs
        # np.dot(matrix, [[y],[x]]) rather than np.dot([x,y], matrix)
        matrix = np.flip(CD_trans.T)

        # offset, calculated from
        #
        # [xi - dxi, yo - dyo] = np.dot( [xo - dxo, yo - dyo], CD_trans )
        #
        # where xi, yi are the input coordinates corresponding to the
        # output coordinates xo, yo in data and dxi/o, dyi/o are the
        # corresponding offsets from the point of
        # rotation/transformation, resulting in
        #
        # [xi, yi] = np.dot( [xo, yo], CD_trans ) + offset
        # with
        # offset = -np.dot( [dxo, dyo], CD_trans ) + [dxi, dyi]
        # setting [dx0, dy0] and [dxi, dyi] to the center
        center = (np.array(data.shape)-1)/2
        offset = -np.dot(center, np.flip(CD_trans)) + center

        # infer transformed data
        data2return = ndimage.affine_transform(data, matrix, offset=offset,
                                               mode='nearest')


    return data2return


################################################################################

def read_CD_matrix (header):

    if ('CD1_1' in header and 'CD1_2' in header and
        'CD2_1' in header and 'CD2_2' in header):

        data2return = np.array([[header['CD1_1'], header['CD2_1']],
                                [header['CD1_2'], header['CD2_2']]])
    else:
        msg = 'one of CD?_? keywords not in header'
        log.critical(msg)
        raise KeyError(msg)
        data2return = None


    return data2return


################################################################################

def get_par (par, tel):

    """Function to check if [par] is a dictionary with one of the keys
       being [tel] or the alphabetic part of [tel] (e.g. 'BG'), and if
       so, return the corresponding value. Otherwise just return the
       parameter value."""

    par_val = par
    if type(par) is dict:
        if tel in par:
            par_val = par[tel]
        else:
            # cut off digits from [tel]
            tel_base = ''.join([char for char in tel if char.isalpha()])
            if tel_base in par:
                par_val = par[tel_base]

    return par_val


################################################################################

def get_bucket_name (path):

    """infer bucket- and filename from [path], which is expected
       to be gs://[bucket name]/some/path/file or [bucket
       name]/some/path/file; if [path] starts with a forward slash,
       empty strings will be returned"""

    bucket_name = path.split('gs://')[-1].split('/')[0]
    if len(bucket_name) > 0:
        # N.B.: returning filename without the starting '/'
        bucket_file = path.split(bucket_name)[-1][1:]
    else:
        bucket_file = ''

    return bucket_name, bucket_file


################################################################################

def isfile (filename):

    if filename[0:5] == 'gs://':

        storage_client = storage.Client()
        bucket_name, bucket_file = get_bucket_name (filename)
        # N.B.: bucket_file should not start with '/'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(bucket_file)
        return blob.exists()

    else:

        return os.path.isfile(filename)


################################################################################

def isdir (folder):

    if folder[0:5] == 'gs://':

        if folder[-1] != '/':
            folder = '{}/'.format(folder)

        storage_client = storage.Client()
        bucket_name, bucket_file = get_bucket_name (folder)
        blobs = storage_client.list_blobs(bucket_name, prefix=bucket_file,
                                          delimiter=None, max_results=1)
        nblobs = len(list(blobs))

        return nblobs > 0

    else:

        return os.path.isdir(folder)


################################################################################

def list_files (path, search_str='', end_str='', start_str=None,
                recursive=False):

    """function to return list of files starting with [path] (can be a
       folder or google cloud bucket name and path; this does not have
       to be a precise or complete folder/path, e.g. [path] can be
       some_path/some_file_basename), with possible [end_str] and
       containing [search_str] without any wildcards. If path is an
       exact folder, then [start_str] can be used as the beginning of
       the filename.

    """

    # is google cloud being used?
    google_cloud = (path[0:5] == 'gs://')


    # split input [path] into folder_bucket and prefix; if path is a
    # folder, need to add a slash at the end, otherwise the prefix
    # will be the name of the deepest folder
    if isdir(path) and path[-1] != '/':
        path = '{}/'.format(path)


    # if path is indeed a folder, then prefix will be an empty string,
    # which is fine below
    folder_bucket, prefix = os.path.split(path.split('gs://')[-1])


    # if path consists of just the bucket name including gs:// or just
    # a path without any slashes at all, the above will lead to an
    # empty folder_bucket and prefix=path; turn these around
    if len(folder_bucket)==0:
        folder_bucket = prefix
        prefix = ''


    # if prefix is empty and [start_str] is defined, use
    # that as the prefix
    if prefix=='' and start_str is not None:
        prefix = start_str


    # if not dealing with google cloud buckets, use glob
    if not google_cloud:

        #glob files
        if recursive:
            files = glob.glob('{}/**/{}*{}*{}'.format(folder_bucket, prefix,
                                                      search_str, end_str),
                              recursive=True)
            if path in files:
                files.remove(path)

        else:
            files = glob.glob('{}/{}*{}*{}'.format(folder_bucket, prefix,
                                                   search_str, end_str))

    else:

        # for buckets, use storage.Client().list_blobs; see
        # https://cloud.google.com/storage/docs/samples/storage-list-files-with-prefix#storage_list_files_with_prefix-python

        # setting delimiter to '/' restricts the results to only the
        # files in a given folder
        if recursive:
            delimiter = None
        else:
            delimiter = '/'


        # bucket name and prefix (e.g. gs://) to add to output files
        bucket_name, bucket_file = get_bucket_name(path)
        bucket_prefix = path.split(bucket_name)[0]


        if False:
            log.info ('folder_bucket: {}'.format(folder_bucket))
            log.info ('prefix: {}'.format(prefix))
            log.info ('path: {}'.format(path))
            log.info ('bucket_name: {}'.format(bucket_name))
            log.info ('bucket_file: {}'.format(bucket_file))


        # get the blobs
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=bucket_file,
                                          delimiter=delimiter)

        # collect blobs' names in list of files
        files = []
        for blob in blobs:

            filename = blob.name

            # check for search string; if search string is empty,
            # following if statement will be False
            if search_str not in filename:
                continue

            # check if filename ends with [end_str]
            len_ext = len(end_str)
            if len_ext > 0 and filename[-len_ext:] != end_str:
                # if not, continue with next blob
                continue

            # after surviving above checks, append filename including
            # the bucket prefix and name
            files.append('{}{}/{}'.format(bucket_prefix, bucket_name, filename))


    #log.info ('files returned by [list_files]: {}'.format(files))
    log.info ('number of files returned by [list_files]: {}'.format(len(files)))


    return files


################################################################################

def list_folders (path, search_str='', depth=None):

    """function to return list of existing folders starting with
       [path] (can be a folder or google cloud bucket name and path;
       this does not have to be a precise or complete folder/path,
       e.g. [path] can be some_path/some_file_basename), optionally
       containing [search_str] without any wildcards. [depth]
       determines how many folder depths are returned, starting from
       the root directory, where only the folders at that precise
       depth are returned and not their parent folders; for google
       cloud buckets the root folder (depth=1) is considered to start
       after the bucket name, i.e. gs://[bucket name]/[root
       folder]/etc. If [depth] is not defined, all existing folders
       are returned.

    """


    # use function [list_files] to list all files containing 'search_str'
    file_list = list_files (path, search_str=search_str, recursive=True)


    # running in google cloud?
    google_cloud = (path[0:5]=='gs://')


    # remove potential 'gs://' and bucket name
    if google_cloud:

        # infer bucket name
        bucket_name, __ = get_bucket_name(path)

        # chop off 'gs://[bucket_name]'
        file_list = [fn.split(bucket_name)[-1] for fn in file_list]


    if depth is None:
        # list all existing folders
        folder_list = ['/'.join(fn.split('/')[:-1]) for fn in file_list]
    else:
        # create folder list with depth [depth]
        folder_list = ['/'.join(fn.split('/')[:depth+1]) for fn in file_list
                       # line added by Danielle to make sure only
                       # folders with this precise depth are included
                       # also for the linux file system
                       if len(fn.rstrip('/').split('/')) > depth]


    # remove duplicate folders and sort
    folder_list = sorted(list(set(folder_list)))


    # append 'gs://[bucket_name]' if needed
    if google_cloud:
        folder_list = ['gs://{}{}'.format(bucket_name, fn) for fn in folder_list]


    # only folders, not files; added by Danielle as she noticed on the
    # normal (linux) file system also files were returned
    isfolder = [isdir(fn) for fn in folder_list]
    folder_list = np.array(folder_list)[isfolder].tolist()


    log.info ('folder_list: {}'.format(folder_list))


    return folder_list


################################################################################

def read_hdulist (fits_file, get_data=True, get_header=False,
                  ext_name_indices=None, dtype=None, memmap=True):

    """Function to read the data (if [get_data] is True) and/or header
    (if [get_header] is True) of the input [fits_file].  The fits file
    can be an image or binary table, and can be compressed (with the
    compressions that astropy.io can handle, such as .gz and .fz). If
    [ext_name_indices] is defined, which can be an integer, a string
    matching the extension's keyword EXTNAME or a list or numpy array
    of integers, those extensions are retrieved.

    """

    t0 = time.time()

    if isfile(fits_file):
        fits_file_read = fits_file

    else:
        # if fits_file does not exist, look for compressed versions or
        # files without the .fz or .gz extension
        if isfile('{}.fz'.format(fits_file)):
            fits_file_read = '{}.fz'.format(fits_file)
        elif isfile(fits_file.replace('.fz','')):
            fits_file_read = fits_file.replace('.fz','')
        elif isfile('{}.gz'.format(fits_file)):
            fits_file_read = '{}.gz'.format(fits_file)
        elif isfile(fits_file.replace('.gz','')):
            fits_file_read = fits_file.replace('.gz','')
        else:
            raise FileNotFoundError ('file not found: {}'.format(fits_file))



    with fits.open(fits_file_read, memmap=memmap) as hdulist:

        n_exts = len(hdulist)

        # if [ext_name_indices] is a range, or list or numpy ndarray
        # of integers, loop over these extensions and concatenate the
        # data into one astropy Table; it is assumed the extension
        # formats are identical to one another - this is used to read
        # specific extensions from e.g. the calibration catalog.
        if type(ext_name_indices) in [list, range, np.ndarray]:

            for i_ext, ext in enumerate(ext_name_indices):

                # get header from first extension as they should be
                # all identical, except for NAXIS2 (nrows)
                if get_header and i_ext==0:
                    header = hdulist[ext].header

                if get_data:
                    # read extension
                    data_temp = hdulist[ext].data
                    # convert to table, as otherwise concatenation of
                    # extensions below using [stack_arrays] is slow
                    data_temp = Table(data_temp)
                    # could also read fits extension into Table directly,
                    # but this is about twice as slow as the 2 steps above
                    #data_temp = Table.read(fits_file_read, hdu=ext)
                    if i_ext==0:
                        data = data_temp
                    else:
                        #data = stack_arrays((data, data_temp),asrecarray=True,
                        #                    usemask=False)
                        # following does not work if data is a fitsrec
                        # array and the array contains boolean fields, as
                        # these are incorrectly converted; therefore the
                        # conversion to a Table above
                        data = np.concatenate([data, data_temp])
                        # could also use the following instead, but
                        # since the above is working ...
                        #data = vstack([data, data_temp])


                    log.info ('added {} rows from extension {} of {}'
                              .format(len(data_temp), ext, fits_file))


        else:
            # otherwise read the extension defined by [ext_name_indices]
            # or simply the last extension
            if type(ext_name_indices) in [int, str]:
                ext = ext_name_indices
            else:
                ext = n_exts-1

            if get_data:
                data = hdulist[ext].data
                # convert to [dtype] if it is defined
                if dtype is not None:
                    data = data.astype(dtype, copy=False)

            if get_header:
                header = hdulist[ext].header



    msg_tmp = ''
    if get_header and not get_data:
        msg_tmp = 'header of '

    log.info ('wall-time spent in read_hdulist to read {}{}: {:.3f} s'
              .format(msg_tmp, fits_file_read, time.time()-t0))


    # return data and header depending on whether [get_data]
    # and [get_header] are defined or not
    if get_data:
        if get_header:
            return data, header
        else:
            return data
    else:
        if get_header:
            return header
        else:
            log.error ('parameters [get_data] and [get_header] are both False '
                       'in function [zogy.read_hdlist]; returning None'
                       )
            return None


################################################################################

def read_exts_fitsio (fits_file, ext_name_indices):

    # use fitsio.FITS to read extensions listed in [ext_name_indices]
    # of input [fits_file] and return a single data array
    for i_ext, ext in enumerate(ext_name_indices):

        # read extension
        data_temp = fitsio.FITS(fits_file)[ext][:]

        # convert to table, as otherwise concatenation of
        # extensions below using [stack_arrays] is slow
        data_temp = Table(data_temp)
        # could also read fits extension into Table directly,
        # but this is about twice as slow as the 2 steps above
        #data_temp = Table.read(fits_file_read, hdu=ext)
        if i_ext==0:
            data = data_temp
        else:
            #data = stack_arrays((data, data_temp),asrecarray=True,
            #                    usemask=False)
            # following does not work if data is a fitsrec
            # array and the array contains boolean fields, as
            # these are incorrectly converted; therefore the
            # conversion to a Table above
            data = np.concatenate([data, data_temp])
            # could also use the following instead, but
            # since the above is working ...
            #data = vstack([data, data_temp])


        log.info ('added {} rows from extension {} of {}'
                  .format(len(data_temp), ext, fits_file))


    return data




################################################################################

def update_imcathead (filename, header, create_hdrfile=False, use_fitsio=False):


    """Update the existing header of input fits file [filename] (which
    can be compressed) with [header]; if create_hdrfile is True, an
    additional fits file with just the header is also created, ending
    with "_hdr.fits". Alternatively, the input fits file can also be
    updated using fitsio.

    This function replaces obsolete update_cathead() and
    update_imhead() (and related copy_header() and process_keys()
    functions.

    """

    t = time.time()


    # add DATEFILE time stamp
    header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')


    if use_fitsio:

        # caveat: fitsio.write_keys results in slightly different
        # floating point values in the headers, e.g. 2016.0 is
        # converted to 2016., and other floats are rounded to a
        # precision a decimal or two less than the original header
        log.info ('updating header of {} with input header (using fitsio)'
                  .format(filename))
        header = header.copy(strip=True)
        with fitsio.FITS(filename, 'rw') as hdulist:
            for key in header:
                hdulist[-1].write_key(key, header[key], header.comments[key])


        if create_hdrfile:
            # read header in astropy format to make separate header
            # file further down below
            with fits.open(filename, memmap=True) as hdulist:
                header_updated = hdulist[-1].header


    else:

        # using astropy, open fits file in update mode
        with fits.open(filename, 'update', memmap=True,
                       output_verify='warn') as hdulist:


            # update header using header update method
            log.info ('updating header of {} with input header (using '
                      'astropy update)' .format(filename))
            hdulist[-1].header.update(header.copy(strip=True))


            if create_hdrfile:
                # save updated header to create header file below
                header_updated = hdulist[-1].header.copy()



    # create separate header file with updated header
    if create_hdrfile:
        hdulist = fits.HDUList(fits.PrimaryHDU(header=header_updated))
        hdulist.writeto('{}_hdr.fits'.format(filename.split('.fits')[0]),
                        overwrite=True, output_verify='warn')


    log_timing_memory (t0=t, label='in update_imcathead')


    return


################################################################################

def format_cat (cat_in, cat_out, cat_type=None, header2add=None,
                exptime=0, apphot_radii=None, dict_thumbnails=None,
                save_thumbnails=False, size_thumbnails=100, ML_calc_prob=False,
                ML_prob_real=None, nfakestars=0, tel=None, set_zogy=None):


    """Function that formats binary fits table [cat_in] according to
       MeerLICHT/BlackGEM specifications for [cat_type] 'new', 'ref'
       or 'trans', and saves the resulting binary fits table
       [cat_out]. If [cat_in] is None, the output fits table will
       contain the same column definitions but without any data
       entries.

    """

    if get_par(set_zogy.timing,tel):
        t = time.time()

    log.info('executing format_cat ...')
    mem_use (label='at start of format_cat')


    if cat_in is not None:

        # read data and header of [cat_in]
        with fits.open(cat_in, memmap=True) as hdulist:
            prihdu = hdulist[0]
            header = hdulist[1].header
            data = hdulist[1].data


        # strip header from image- or table-specific keywords
        header.strip()


        if header2add is not None:
            header.update(header2add.copy(strip=True))


        log.info ('format catalog: {}, type: {}: column names: {}'
                  .format(cat_in, cat_type, data.dtype.names))

    else:

        # if no [cat_in] is provided, just define the header using
        # [header2add]
        header = header2add.copy(strip=True)




    # this [formats] dictionary contains the output format, the output
    # column unit and the column description
    tn_fmt = '{}E'.format(size_thumbnails**2)
    formats = {
        'NUMBER':              ['J', ''         , 'Running object number'],
        'X_POS':               ['E', 'pix'      , 'X-pixel coordinate'],
        'Y_POS':               ['E', 'pix'      , 'Y-pixel coordinate'],
        'XVAR_POS':            ['E', 'pix^2'    , 'Variance of X-pixel coordinate'],
        'YVAR_POS':            ['E', 'pix^2'    , 'Variance of Y-pixel coordinate'],
        'XYCOV_POS':           ['E', 'pix^2'    , 'Co-variance of XY position'],
        'CXX':                 ['E', 'pix^(-2)' , 'CXX object ellipse parameter'],
        'CYY':                 ['E', 'pix^(-2)' , 'CYY object ellipse parameter'],
        'CXY':                 ['E', 'pix^(-2)' , 'CXY object ellipse parameter'],
        'A':                   ['E', 'pix'      , 'Profile RMS along major axis'],
        'B':                   ['E', 'pix'      , 'Profile RMS along minor axis'],
        'THETA':               ['E', 'deg'      , 'Position angle (CCW/x)'],
        'ELONGATION':          ['E', ''         , 'Elongation (=A/B)'],
        'RA':                  ['D', 'deg'      , 'Right ascension (ICRS)'],
        'DEC':                 ['D', 'deg'      , 'Declination (ICRS)'],
        'FLAGS':               ['I', ''         , 'SExtractor extraction flags'],
        'FLAGS_MASK':          ['I', ''         , 'OR-combined flags over ISO/inner profile in input mask'],
        'FWHM':                ['E', 'pix'      , 'FWHM assuming a Gaussian core'],
        'CLASS_STAR':          ['E', ''         , 'SExtractor star/galaxy classification'],
        'E_FLUX_APER':         ['E', 'e-/s'     , 'Electron flux within radius x FWHM'],
        'E_FLUXERR_APER':      ['E', 'e-/s'     , 'Electron flux error within radius x FWHM'],
        'MAG_APER':            ['E', 'mag'      , 'Aperture AB mag within radius x FWHM'],
        'MAGERR_APER':         ['E', 'mag'      , 'Aperture AB mag error within radius x FWHM'],
        'MAGERRTOT_APER':      ['E', 'mag'      , 'Aperture AB mag total error (incl. ZP error) within radius x FWHM'],
        'FNU_APER':            ['E', 'uJy'      , 'Aperture flux within radius x FWHM'],
        'FNUERR_APER':         ['E', 'uJy'      , 'Aperture flux error within radius x FWHM'],
        'FNUERRTOT_APER':      ['E', 'uJy'      , 'Aperture flux total error (incl. ZP error) within radius x FWHM'],
        'BACKGROUND':          ['E', 'e-'       , 'Sky background estimated from sky annulus'],
        #'E_FLUX_MAX':          ['E', 'e-/s'    ],
        'E_FLUX_AUTO':         ['E', 'e-/s'     , 'Electron flux within a Kron-like elliptical aperture'],
        'E_FLUXERR_AUTO':      ['E', 'e-/s'     , 'Electron flux error within a Kron-like elliptical aperture'],
        'MAG_AUTO':            ['E', 'mag'      , 'AB mag within a Kron-like elliptical aperture'],
        'MAGERR_AUTO':         ['E', 'mag'      , 'AB mag error within a Kron-like elliptical aperture'],
        'MAGERRTOT_AUTO':      ['E', 'mag'      , 'AB mag total error (incl. ZP error) within a Kron-like elliptical aperture'],
        'KRON_RADIUS':         ['E', 'pix'      , 'Kron aperture(s)'],
        'E_FLUX_ISO':          ['E', 'e-/s'     , 'Isophotal electron flux'],
        'E_FLUXERR_ISO':       ['E', 'e-/s'     , 'Isophotal electron flux error'],
        'MAG_ISO':             ['E', 'mag'      , 'Isophotal AB mag'],
        'MAGERR_ISO':          ['E', 'mag'      , 'Isophotal AB mag error'],
        'MAGERRTOT_ISO':       ['E', 'mag'      , 'Isophotal AB mag total error (incl. ZP error)'],
        'ISOAREA':             ['I', 'pix^2'    , 'Isophotal area above analysis threshold'],
        'MU_MAX':              ['E', 'mag/pix^2', 'Peak surface brightness above background'],
        'FLUX_RADIUS':         ['E', 'pix'      , '50% Fraction-of-light radius'],
        'E_FLUX_PETRO':        ['E', 'e-/s'     , 'Electron flux within a Petrosian-like elliptical aperture'],
        'E_FLUXERR_PETRO':     ['E', 'e-/s'     , 'Electron flux error within a Petrosian-like elliptical aperture'],
        'MAG_PETRO':           ['E', 'mag'      , 'AB mag within a Petrosian-like elliptical aperture'],
        'MAGERR_PETRO':        ['E', 'mag'      , 'AB mag error within a Petrosian-like elliptical aperture'],
        'MAGERRTOT_PETRO':     ['E', 'mag'      , 'AB mag total error (incl. ZP error) within a Petrosian-like elliptical aperture'],
        'PETRO_RADIUS':        ['E', 'pix'      , 'Petrosian aperture(s)'],
        'E_FLUX_OPT':          ['E', 'e-/s'     , 'Optimal electron flux'],
        'E_FLUXERR_OPT':       ['E', 'e-/s'     , 'Optimal electron flux error'],
        'MAG_OPT':             ['E', 'mag'      , 'Optimal AB mag'],
        'MAGERR_OPT':          ['E', 'mag'      , 'Optimal AB mag error'],
        'MAGERRTOT_OPT':       ['E', 'mag'      , 'Optimal AB mag total error (incl. ZP error)'],
        'FNU_OPT':             ['E', 'uJy'      , 'Optimal flux'],
        'FNUERR_OPT':          ['E', 'uJy'      , 'Optimal flux error'],
        'FNUERRTOT_OPT':       ['E', 'uJy'      , 'Optimal flux total error (incl. ZP error)'],
        # transient:
        # ==========
        'X_PEAK':              ['I', 'pix'      , 'Integer X-pixel coordinate of peak in significance/Scorr image'],
        'Y_PEAK':              ['I', 'pix'      , 'Integer Y-pixel coordinate of peak in significance/Scorr image'],
        'CHANNEL':             ['I', ''         , 'ML/BG read-out channel number (based on X_PEAK, Y_PEAK)'],
        'RA_PEAK':             ['D', 'deg'      , 'Right ascension (ICRS; based on X_PEAK, Y_PEAK)'],
        'DEC_PEAK':            ['D', 'deg'      , 'Declination (ICRS, based on X_PEAK, Y_PEAK)'],
        'SNR_ZOGY':            ['E', ''         , 'Transient signal-to-noise ratio (=peak value in significance/Scorr image'],
        'E_FLUX_ZOGY':         ['E', 'e-/s'     , 'Transient PSF electron flux'],
        'E_FLUXERR_ZOGY':      ['E', 'e-/s'     , 'Transient PSF electron flux error'],
        'MAG_ZOGY':            ['E', 'mag'      , 'Transient PSF AB mag'],
        'MAGERR_ZOGY':         ['E', 'mag'      , 'Transient PSF AB mag error'],
        'MAGERRTOT_ZOGY':      ['E', 'mag'      , 'Transient PSF AB mag total error (incl. ZP error)'],
        'FNU_ZOGY':            ['E', 'uJy'      , 'Transient PSF flux'],
        'FNUERR_ZOGY':         ['E', 'uJy'      , 'Transient PSF flux error'],
        'FNUERRTOT_ZOGY':      ['E', 'uJy'      , 'Transient PSF flux total error (incl. ZP error)'],
        #
        'X_POS_SCORR':         ['E', 'pix'      , 'X-pixel coordinate in significance/Scorr image'],
        'Y_POS_SCORR':         ['E', 'pix'      , 'Y-pixel coordinate in significance/Scorr image'],
        'RA_SCORR':            ['D', 'deg'      , 'Right ascension (ICRS; based on X_POS_SCORR, Y_POS_SCORR)'],
        'DEC_SCORR':           ['D', 'deg'      , 'Declination (ICRS; based on X_POS_SCORR, Y_POS_SCORR)'],
        'ELONG_SCORR':         ['E', ''         , 'Elongation in significance/Scorr image'],
        'FLAGS_SCORR':         ['I', ''         , 'SExtractor extraction flags in significance/Scorr image'],
        'FLAGS_MASK_SCORR':    ['I', ''         , 'OR-combined flags over inner profile in input new and reference masks'],
        #
        'X_PSF_D':             ['E', 'pix'      , 'X-pixel coordinate from PSF fit to difference image'],
        'XERR_PSF_D':          ['E', 'pix'      , 'X-pixel coordinate error from PSF fit to difference image'],
        'Y_PSF_D':             ['E', 'pix'      , 'Y-pixel coordinate from PSF fit to difference image'],
        'YERR_PSF_D':          ['E', 'pix'      , 'Y-pixel coordinate error from PSF fit to difference image'],
        'RA_PSF_D':            ['D', 'deg'      , 'Right ascension (ICRS; based on X_PSF_D, Y_PSF_D)'],
        'DEC_PSF_D':           ['D', 'deg'      , 'Declination (ICRS; based on X_PSF_D, Y_PSF_D)'],
        'E_FLUX_PSF_D':        ['E', 'e-/s'     , 'Transient electron flux from PSF fit to difference image'],
        'E_FLUXERR_PSF_D':     ['E', 'e-/s'     , 'Transient electron flux error from PSF fit to difference image'],
        'MAG_PSF_D':           ['E', 'mag'      , 'Transient AB mag from PSF fit to difference image'],
        'MAGERR_PSF_D':        ['E', 'mag'      , 'Transient AB mag error from PSF fit to difference image'],
        'MAGERRTOT_PSF_D':     ['E', 'mag'      , 'Transient AB mag total error (incl. ZP error) from PSF fit to difference image'],
        'FNU_PSF_D':           ['E', 'uJy'      , 'Transient flux from PSF fit to difference image'],
        'FNUERR_PSF_D':        ['E', 'uJy'      , 'Transient flux error from PSF fit to difference image'],
        'FNUERRTOT_PSF_D':     ['E', 'uJy'      , 'Transient flux total error (incl. ZP error) from PSF fit to difference image'],
        'CHI2_PSF_D':          ['E', ''         , 'Reduced chi-square of PSF fit to difference image'],
        #
        'X_MOFFAT_D':          ['E', 'pix'      , 'X-pixel coordinate from Moffat fit to difference image'],
        'XERR_MOFFAT_D':       ['E', 'pix'      , 'X-pixel coordinate error from Moffat fit to difference image'],
        'Y_MOFFAT_D':          ['E', 'pix'      , 'Y-pixel coordinate from Moffat fit to difference image'],
        'YERR_MOFFAT_D':       ['E', 'pix'      , 'Y-pixel coordinate error from Moffat fit to difference image'],
        'RA_MOFFAT_D':         ['D', 'deg'      , 'Right ascension (ICRS; based on X_MOFFAT_D, Y_MOFFAT_D)'],
        'DEC_MOFFAT_D':        ['D', 'deg'      , 'Declination (ICRS; based on X_MOFFAT_D, Y_MOFFAT_D)'],
        'FWHM_MOFFAT_D':       ['E', 'pix'      , 'Average FWHM from Moffat fit to difference image'],
        'ELONG_MOFFAT_D':      ['E', ''         , 'Elongation from Moffat fit to difference image'],
        'CHI2_MOFFAT_D':       ['E', ''         , 'Reduced chi-square of Moffat fit to difference image'],
        #
        'X_GAUSS_D':           ['E', 'pix'      , 'X-pixel coordinate from Gauss fit to difference image'],
        'XERR_GAUSS_D':        ['E', 'pix'      , 'X-pixel coordinate error from Gauss fit to difference image'],
        'Y_GAUSS_D':           ['E', 'pix'      , 'Y-pixel coordinate from Gauss fit to difference image'],
        'YERR_GAUSS_D':        ['E', 'pix'      , 'Y-pixel coordinate error from Gauss fit to difference image'],
        'RA_GAUSS_D':          ['D', 'deg'      , 'Right ascension (ICRS; based on X_GAUSS_D, Y_GAUSS_D)'],
        'DEC_GAUSS_D':         ['D', 'deg'      , 'Declination (ICRS; based on X_GAUSS_D, Y_GAUSS_D)'],
        'FWHM_GAUSS_D':        ['E', 'pix'      , 'Average FWHM from Gauss fit to difference image'],
        'ELONG_GAUSS_D':       ['E', ''         , 'Elongation from Gauss fit to difference image'],
        'CHI2_GAUSS_D':        ['E', ''         , 'Reduced chi-square of Gauss fit to difference image'],
        #
        'CLASS_REAL':          ['E', ''         , 'Machine-learning probability that transient is real'],
        'X_FAKE':              ['E', 'pix'      , 'X-pixel coordinate of inserted/fake transient'],
        'Y_FAKE':              ['E', 'pix'      , 'Y-pixel coordinate of inserted/fake transient'],
        'SNR_FAKE_IN':         ['E', ''         , 'Signal-to-noise ratio of inserted/fake transient'],
        'E_FLUX_FAKE_IN':      ['E', 'e-/s'     , 'Electron flux of inserted/fake transient'],
        'MAG_FAKE_IN':         ['E', 'mag'      , 'AB mag of inserted/fake transient'],
        'FNU_FAKE_IN':         ['E', 'uJy'      , 'Flux of inserted/fake transient'],
        'THUMBNAIL_RED':       [tn_fmt, 'e-'    , 'Reduced image thumbnail centered on transient'],
        'THUMBNAIL_REF':       [tn_fmt, 'e-'    , 'Reference image thumbnail centered on transient'],
        'THUMBNAIL_D':         [tn_fmt, 'e-'    , 'Difference image thumbnail centered on transient'],
        'THUMBNAIL_SCORR':     [tn_fmt, 'sigma' , 'Significance/Scorr image thumbnail centered on transient'],
        #
        'SOURCE_ID':           ['K', ''         , 'Gaia source ID'],
        'LIMMAG_OPT':          ['E', 'mag'      , 'Optimal limiting AB magnitude'],
        'SNR_OPT':             ['E', ''         , 'Optimal signal-to-noise ratio'],
        #
        'SOURCE_ID_NEAR_GAIA': ['K', ''         , 'Source ID of Gaia source nearest to the transient'],
        'SEP_NEAR_GAIA':       ['E', 'arcsec'   , 'Separation to nearest Gaia source'],
        'MAG_G_NEAR_GAIA':     ['E', 'mag'      , 'Gaia G-band magnitude (Vega) of nearest source'],
        'MAGERR_G_NEAR_GAIA':  ['E', 'mag'      , 'Gaia G-band magnitude error (Vega) of nearest source'],
        'SEP_BRIGHT_GAIA':     ['E', 'arcsec'   , 'Separation to nearest bright (G<14) Gaia source'],
        'MAG_G_BRIGHT_GAIA':   ['E', 'mag'      , 'Gaia DR3 G-band magnitude of nearest bright (G<14) source'],
        #
        'NUMBER_NEAR_REF':     ['J', ''         , 'Running number of source in reference image nearest to the transient'],
        'SEP_NEAR_REF':        ['E', 'arcsec'   , 'Separation to nearest source in reference image'],
        'FWHM_NEAR_REF':       ['E', 'pix'      , 'FWHM of nearest source in reference image'],
        'ELONG_NEAR_REF':      ['E', ''         , 'Elongation of nearest source in reference image'],
        'CLASS_STAR_NEAR_REF': ['E', ''         , 'SExtractor star/galaxy classification of nearest source in reference image'],
        'MAG_AUTO_NEAR_REF':   ['E', 'mag'      , 'Kron/AUTO AB mag of nearest source in reference image'],
        'MAGERR_AUTO_NEAR_REF':['E', 'mag'      , 'Kron/AUTO AB mag error of nearest source in reference image'],
        'MAGERRTOT_AUTO_NEAR_REF':['E', 'mag'   , 'Kron/AUTO AB mag total error (incl. ZP error) of nearest source in reference image'],
        'FNU_OPT_REF':         ['E', 'uJy'      , 'Forced-photometry optimal flux in reference image at transient position'],
        'FNUERR_OPT_REF':      ['E', 'uJy'      , 'Forced-photometry optimal flux error in reference image at transient position'],
        'FNUERRTOT_OPT_REF':   ['E', 'uJy'      , 'Forced-photometry optimal flux total error (incl. ZP error) in reference image at transient position'],
        'MAG_OPT_REF':         ['E', 'mag'      , 'Forced-photometry optimal AB mag in reference image at transient position'],
        'MAGERR_OPT_REF':      ['E', 'mag'      , 'Forced-photometry optimal AB mag error in reference image at transient position'],
        'MAGERRTOT_OPT_REF':   ['E', 'mag'      , 'Forced-photometry optimal AB mag total error (incl. ZP error) in reference image at transient position'],
        'FLAGS_MASK_REF':      ['I', ''         , 'OR-combined flags over inner profile in reference image mask'],
        #
        #'NUMBER_NEAR_RED':     ['J', ''         , 'Running number of full-source catalogue source nearest to the transient'],
        'SEP_NEAR_RED':        ['E', 'arcsec'   , 'Separation to nearest source in reduced image'],
        'FWHM_NEAR_RED':       ['E', 'pix'      , 'FWHM of nearest source in reduced image'],
        'ELONG_NEAR_RED':      ['E', ''         , 'Elongation of nearest source in reduced image'],
        'CLASS_STAR_NEAR_RED': ['E', ''         , 'SExtractor star/galaxy classification of nearest source in reduced image'],
        'FNU_OPT_RED':         ['E', 'uJy'      , 'Forced-photometry optimal flux in reduced image at transient position'],
        'FNUERR_OPT_RED':      ['E', 'uJy'      , 'Forced-photometry optimal flux error in reduced image at transient position'],
        'FNUERRTOT_OPT_RED':   ['E', 'uJy'      , 'Forced-photometry optimal flux total error (incl. ZP error) in reduced image at transient position'],
        'MAG_OPT_RED':         ['E', 'mag'      , 'Forced-photometry optimal AB mag in reduced image at transient position'],
        'MAGERR_OPT_RED':      ['E', 'mag'      , 'Forced-photometry optimal AB mag error in reduced image at transient position'],
        'MAGERRTOT_OPT_RED':   ['E', 'mag'      , 'Forced-photometry optimal AB mag total error (incl. ZP error) in reduced image at transient position'],
        'FLAGS_MASK_RED':      ['I', ''         , 'OR-combined flags over inner profile in reduced image mask'],
    }


    # columns to record for Gaia forced-photometry full-source catalog
    # in AB magnitudes
    if not get_par(set_zogy.record_fnu,tel):
        keys_to_record_gaia = ['SOURCE_ID', 'X_POS', 'Y_POS', 'FLAGS_MASK',
                               'BACKGROUND', 'MAG_APER', 'MAGERR_APER',
                               #'E_FLUX_OPT', 'E_FLUXERR_OPT',
                               'MAG_OPT', 'MAGERR_OPT', 'SNR_OPT', 'LIMMAG_OPT']
    else:
        # in Fnu
        keys_to_record_gaia = ['SOURCE_ID', 'X_POS', 'Y_POS', 'FLAGS_MASK',
                               'BACKGROUND', 'FNU_APER', 'FNUERR_APER',
                               'FNU_OPT', 'FNUERR_OPT',
                               # CHECK!!! - following columns
                               # are just for testing and should be
                               # removed when going to production
                               'FNUERRTOT_OPT',
                               'MAG_OPT', 'MAGERR_OPT', 'MAGERRTOT_OPT',
                               'SNR_OPT', 'LIMMAG_OPT']


    if cat_type is None:
        # if no [cat_type] is provided, define the keys to record from
        # the data columns
        if cat_in is not None:
            keys_to_record = data.dtype.names


    elif cat_type == 'ref':

        # for the reference catalog of telescopes other than ML/BG for
        # which Gaia forced photometry is performed
        if (get_par(set_zogy.force_phot_gaia,tel) and
            tel not in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']):

            # in case of force photometry on Gaia input catalog
            keys_to_record = keys_to_record_gaia

        else:
            # for the ML/BG reference catalog, stick to the source-extractor
            # detections
            keys_to_record = ['NUMBER', 'X_POS', 'Y_POS',
                              'XVAR_POS', 'YVAR_POS', 'XYCOV_POS',
                              'RA', 'DEC',
                              'CXX', 'CYY', 'CXY', 'A', 'B', 'THETA',
                              'ELONGATION', 'FWHM', 'CLASS_STAR',
                              'FLAGS', 'FLAGS_MASK',
                              'BACKGROUND',
                              'MAG_APER', 'MAGERR_APER', 'MAGERRTOT_APER',
                              'MAG_AUTO', 'MAGERR_AUTO', 'MAGERRTOT_AUTO',
                              'KRON_RADIUS',
                              'MAG_ISO', 'MAGERR_ISO', 'MAGERRTOT_ISO',
                              'ISOAREA', 'MU_MAX', 'FLUX_RADIUS',
                              'MAG_PETRO', 'MAGERR_PETRO', 'MAGERRTOT_PETRO',
                              'PETRO_RADIUS',
                              #'E_FLUX_OPT', 'E_FLUXERR_OPT',
                              'FNU_OPT', 'FNUERR_OPT', 'FNUERRTOT_OPT',
                              'MAG_OPT', 'MAGERR_OPT', 'MAGERRTOT_OPT']


    elif cat_type == 'new':

        if get_par(set_zogy.force_phot_gaia,tel):
            # in case of forced photometry on Gaia input catalog
            keys_to_record = keys_to_record_gaia
        else:
            keys_to_record = ['NUMBER', 'X_POS', 'Y_POS',
                              'XVAR_POS', 'YVAR_POS', 'XYCOV_POS',
                              'RA', 'DEC',
                              'ELONGATION', 'FWHM', 'CLASS_STAR',
                              'FLAGS', 'FLAGS_MASK', 'BACKGROUND',
                              'MAG_APER', 'MAGERR_APER',
                              'E_FLUX_OPT', 'E_FLUXERR_OPT',
                              'MAG_OPT', 'MAGERR_OPT']


    elif cat_type == 'trans':

        if not get_par(set_zogy.use_new_transcat,tel):

            # original set
            keys_to_record = ['NUMBER', 'X_PEAK', 'Y_PEAK',
                              'RA_PEAK', 'DEC_PEAK', 'SNR_ZOGY',
                              'E_FLUX_ZOGY', 'E_FLUXERR_ZOGY',
                              'MAG_ZOGY', 'MAGERR_ZOGY',
                              #
                              'X_POS_SCORR', 'Y_POS_SCORR',
                              #'XVAR_POS_SCORR', 'YVAR_POS_SCORR', 'XYCOV_POS_SCORR',
                              'RA_SCORR', 'DEC_SCORR', 'ELONG_SCORR',
                              'FLAGS_SCORR', 'FLAGS_MASK_SCORR',
                              #'MAG_OPT_D', 'MAGERR_OPT_D',
                              'X_PSF_D', 'XERR_PSF_D', 'Y_PSF_D', 'YERR_PSF_D',
                              'RA_PSF_D', 'DEC_PSF_D',
                              'MAG_PSF_D', 'MAGERR_PSF_D',
                              'CHI2_PSF_D',
                              #'X_MOFFAT_D', 'XERR_MOFFAT_D',
                              #'Y_MOFFAT_D', 'YERR_MOFFAT_D',
                              #'RA_MOFFAT_D', 'DEC_MOFFAT_D',
                              #'FWHM_MOFFAT_D', 'ELONG_MOFFAT_D', 'CHI2_MOFFAT_D',
                              'X_GAUSS_D', 'XERR_GAUSS_D',
                              'Y_GAUSS_D', 'YERR_GAUSS_D',
                              'RA_GAUSS_D', 'DEC_GAUSS_D',
                              'FWHM_GAUSS_D', 'ELONG_GAUSS_D', 'CHI2_GAUSS_D']

        else:

            # new set of transient columns (June 2024)
            keys_to_record = ['NUMBER', 'X_PEAK', 'Y_PEAK', 'CHANNEL',
                              'RA_PEAK', 'DEC_PEAK', 'SNR_ZOGY',
                              'FNU_ZOGY', 'FNUERR_ZOGY', 'FNUERRTOT_ZOGY',
                              'MAG_ZOGY', 'MAGERR_ZOGY', 'MAGERRTOT_ZOGY',
                              #
                              'FLAGS_SCORR', 'FLAGS_MASK_SCORR',
                              #
                              'X_PSF_D', 'XERR_PSF_D', 'Y_PSF_D', 'YERR_PSF_D',
                              'RA_PSF_D', 'DEC_PSF_D',
                              'FNU_PSF_D', 'FNUERR_PSF_D', 'FNUERRTOT_PSF_D',
                              'MAG_PSF_D', 'MAGERR_PSF_D', 'MAGERRTOT_PSF_D',
                              'CHI2_PSF_D',
                              #
                              'SOURCE_ID_NEAR_GAIA', 'SEP_NEAR_GAIA',
                              'MAG_G_NEAR_GAIA', 'MAGERR_G_NEAR_GAIA',
                              'SEP_BRIGHT_GAIA', 'MAG_G_BRIGHT_GAIA',
                              #
                              'NUMBER_NEAR_REF',
                              'SEP_NEAR_REF', 'FWHM_NEAR_REF', 'ELONG_NEAR_REF',
                              'CLASS_STAR_NEAR_REF',
                              'MAG_AUTO_NEAR_REF', 'MAGERR_AUTO_NEAR_REF',
                              'FNU_OPT_REF', 'FNUERR_OPT_REF', 'FNUERRTOT_OPT_REF',
                              'MAG_OPT_REF', 'MAGERR_OPT_REF', 'MAGERRTOT_OPT_REF',
                              'FLAGS_MASK_REF',
                              #
                              'SEP_NEAR_RED', 'FWHM_NEAR_RED', 'ELONG_NEAR_RED',
                              'CLASS_STAR_NEAR_RED',
                              'FNU_OPT_RED', 'FNUERR_OPT_RED', 'FNUERRTOT_OPT_RED',
                              'MAG_OPT_RED', 'MAGERR_OPT_RED', 'MAGERRTOT_OPT_RED',
                              'FLAGS_MASK_RED'
                              ]


        if ML_calc_prob and tel[0:2] in ['ML','BG']:

            keys_to_record.append('CLASS_REAL')

            # MeerCRAB probabilities ML_prob_real are now calculated
            # before format_cat
            if cat_in is not None and ML_prob_real is not None:
                # field CLASS_REAL is not yet included in data, so
                # append it using the probabilities initialised to -1;
                # the actual probabilities will be added after this
                # function is done when MeerCRAB is processed
                data = append_fields(data, 'CLASS_REAL', ML_prob_real,
                                     usemask=False, asrecarray=True)


    if nfakestars > 0 and (cat_type == 'new' or cat_type == 'trans'):

        # add fakestar columns
        keys_to_record.append('X_FAKE')
        keys_to_record.append('Y_FAKE')
        keys_to_record.append('SNR_FAKE_IN')
        keys_to_record.append('E_FLUX_FAKE_IN')
        keys_to_record.append('MAG_FAKE_IN')


    # rename any of the keys using this dictionary, such as the
    # pre-defined SExtractor column names
    #keys2rename = {'ALPHAWIN_J2000': 'RA_ICRS', 'DELTAWIN_J2000': 'DEC_ICRS'}
    keys2rename = {}


    def get_col (key, key_new, data_key=None):

        # function that returns column definition based on input
        # [key], [key_new], and [data_key]; for most fields [key] and
        # [key_new] are the same, except for 'E_FLUX_APER' and
        # 'E_FLUXERR_APER' or 'MAG_APER' and 'MAGERR_APER', which are
        # split into the separate apertures, and the aperture sizes
        # enter in the new key name as well.

        # if exposure time is non-zero, modify all 'e-/s' or
        # 'electron/s' columns accordingly
        if exptime != 0:
            if '/s' in formats[key][1]:
                data_key /= exptime
                #key_new = 'E-{}'.format(key_new)
                key_new = '{}'.format(key_new)
        else:
            # if [format_cat] is called from [qc], then dummy catalogs
            # are being made and no exptime is provided
            dumcat = np.any([header[k] for k in header if 'DUMCAT' in k])
            if not dumcat:
                log.warning('input [exptime] in function [format_cat] is zero')

        if data_key is not None:
            col = fits.Column(name=key_new, format=formats[key][0],
                              unit=formats[key][1], #disp=formats[key][2],
                              array=data_key)
        # if [data_key] is None, define the column but without the
        # data; this is used for making a table with the same field
        # definitions but without any entries
        else:
            col = fits.Column(name=key_new, format=formats[key][0],
                              unit=formats[key][1]) #, disp=formats[key][2])
        return col


    # using the above [get_col] function, loop through the keys to
    # record and define the list of columns
    columns = []
    for key in keys_to_record:

        if 'APER' in key:
            # update column names of aperture fluxes to include radii
            # loop apertures
            for i_ap in range(len(apphot_radii)):
                key_new = '{}_R{}xFWHM'.format(key, apphot_radii[i_ap])
                if cat_in is not None:
                    if key_new in data.dtype.names:
                        #columns.append(get_col (key, key_new, data[key][:,i_ap]))
                        columns.append(get_col (key, key_new, data[key_new]))
                else:
                    columns.append(get_col (key, key_new))

        else:

            # check if key needs to be renamed
            if key in keys2rename.keys():
                key_new = keys2rename[key]
            else:
                key_new = key

            if cat_in is not None:
                if key in data.dtype.names:
                    columns.append(get_col (key, key_new, data[key]))
            else:
                columns.append(get_col (key, key_new))



    # create hdu from columns
    hdu = fits.BinTableHDU.from_columns(columns, character_as_bytes=True)
    mem_use (label='after hdu creation in format_cat')


    # add column descriptions
    add_col_descriptions (hdu, formats)


    # save light version of transient catalog before heavy
    # thumbnails are added
    if cat_type == 'trans':

        header['FORMAT-P'] = (True, 'successfully formatted catalog')
        header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')

        #hdu.header += header
        hdu.header.update(header.copy(strip=True))
        hdu.writeto(cat_out.replace('.fits', '_light.fits'), overwrite=True)
        #del hdu



    # add [thumbnails] column definition and the corresponding data
    if save_thumbnails:


        # add column definitions
        dim_str = '({},{})'.format(size_thumbnails, size_thumbnails)
        for i_tn, key in enumerate(dict_thumbnails.keys()):

            # add column but without the data for the moment
            col = fits.Column(name=key, format=formats[key][0],
                              unit=formats[key][1], dim=dim_str)
            # append column
            columns.append(col)


        # recreate hdu with additional columns
        hdu = fits.BinTableHDU.from_columns(columns, character_as_bytes=True)
        mem_use (label='after hdu creation including thumbnails in format_cat')


        # add column descriptions
        add_col_descriptions (hdu, formats)


        # loop thumbnail data and add them to the hdu
        for i_tn, key in enumerate(dict_thumbnails.keys()):

            if dict_thumbnails[key] is not None:

                # read data_thumbnail from input [dict_thumbnails]
                hdu.data[key] = np.load(dict_thumbnails[key], mmap_mode='c')
                mem_use (label='after adding column {} to hdu.data in '
                         'format_cat'.format(key))

                # delete numpy file if not keeping
                # intermediate/temporary files
                if not get_par(set_zogy.keep_tmp,tel):
                    remove_files ([dict_thumbnails[key]])



    # add header keyword indicating catalog was successfully formatted
    header['FORMAT-P'] = (True, 'successfully formatted catalog')
    header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')


    # update hdu header
    hdu.header.update(header.copy(strip=True))

    # save hdu to fits
    hdu.writeto(cat_out, overwrite=True)

    # update header and create separate header fits file
    update_imcathead(cat_out, header, create_hdrfile=True)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in format_cat')


    return


################################################################################

def add_col_descriptions (hdu, formats):

    # add column descriptions to hdu header; column names are in the
    # TTYPE[i+1] keywords and the descriptions should be saved in
    # TCOMM[i+1]
    for ic, col0 in enumerate(hdu.data.dtype.names):
        ncol = ic + 1
        # edit aperture names
        if 'APER' in col0:
            if 'E_' in col0:
                col = '_'.join(col0.split('_')[:3])
            else:
                col = '_'.join(col0.split('_')[:2])
        else:
            col = col0

        # check if col is present in formats
        if col in formats:
            descr = formats[col][2]
        else:
            log.warning ('{} not in formats dictionary'.format(col))
            descr = ''

        # update relevant header keyword COMM
        hdu.header['TCOMM{}'.format(ncol)] = descr


################################################################################

def get_index_around_xy (ysize, xsize, ycoord, xcoord, size):

    """Function to retrieve indices around pixel coordinates [ycoord,
    xcoord] in original image of size (ysize, xsize) and a thumbnail
    image of size (size, size) onto which the original pixel values
    will be projected. Normally the shapes of the two returned indices
    will be (size, size), but not if the pixel coordinates are near
    the original image edge.

    The function also returns the pixel coordinates in the thumbnail
    image corresponding to ycoord, xcoord in the original image.

    N.B.: if (ycoord, xcoord) are pixel coordinates, then pixel
    coordinate (int(ycoord), int(xcoord)) will correspond to pixel
    coordinate (int(size/2), int(size/2)) in the thumbnail image. If
    instead they are pixel indices (i.e.  int(pixel coordinate) - 1),
    then index (ycoord, xcoord) will correspond to pixel index
    (int(size/2), int(size/2)) in the thumbnail index.

    """


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

    # also determine corresponding indices of thumbnail image, which
    # will not be [0:size, 0:size] if an object is near the image edge
    y1_tn = max(0, hsize-ypos)
    x1_tn = max(0, hsize-xpos)
    y2_tn = min(size, size-(ypos+hsize-ysize))
    x2_tn = min(size, size-(xpos+hsize-xsize))
    index_tn = tuple([slice(y1_tn,y2_tn),slice(x1_tn,x2_tn)])

    # pixel coordinates in the thumbnail image corresponding to
    # ycoord, xcoord in the original image
    ycoord_tn = ycoord - y1
    xcoord_tn = xcoord - x1

    return index, index_tn, ycoord_tn, xcoord_tn


################################################################################

def get_trans (fits_new, fits_ref, fits_D, fits_Scorr, fits_Fpsf, fits_Fpsferr,
               fits_new_mask, fits_ref_mask, fits_new_bkg_std, fits_ref_bkg_std,
               header_new, header_ref, header_trans, fits_new_psf, fits_ref_psf,
               nthreads=1, keep_all=False):

    """Function that selects transient candidates from the significance
    array (data_Scorr), and determines all regions with peak Scorr
    values above the set threshold, and down to where the region wings
    reaches the 2 sigma isochrone. Regions are discarded if they:

    - are too big or too small
    - contain both negatively and positively significant Scorr values
      (at 5 sigma level) within the same region
    - contain both negatively and positively significant Scorr values
      (at 5 sigma level) in this region and region at a pixel position
      that is due to the shift between the new and reference image
      (e.g. an artefact that is present in both the new and reference
      image will create transients with opposite significance at
      pixel positions equal to the shift between the images)
    - contain more flagged pixels than the maximum indicated in
      [set_zogy.transient_mask_max]

    Futhermore, a PSF fit to the D image is performed at the location
    of the filtered transients, using combination of the PSFs of the
    new and ref image, i.e. P_D in ZOGY-speak. This results in an
    independent estimate of the transient flux and error, as well
    as the pixel location and error. The resulting chi2 of the PSF
    fit is used to futher filter out transients.
    A moffat fit is also performed to the D image.
    """


    if get_par(set_zogy.timing,tel):
        t = time.time()

    # base name
    base = fits_Scorr.replace('_Scorr.fits','')


    # combine new and remapped ref mask
    data_new_mask = read_hdulist (fits_new_mask, dtype='uint8')
    data_ref_mask = read_hdulist (fits_ref_mask, dtype='uint8')
    data_newref_mask = (data_new_mask | data_ref_mask)
    fits_newref_mask = '{}_mask_newref.fits'.format(base)
    fits.writeto (fits_newref_mask, data_newref_mask, overwrite=True)
    del data_new_mask, data_ref_mask



    # read Scorr image and its header
    data_Scorr, header = read_hdulist (fits_Scorr, get_header=True,
                                       dtype='float32')


    # read a few header keywords
    pixscale = read_header(header, ['pixscale'])

    # for the FWHM, use the average of fwhm_new and fwhm_ref
    fwhm = (header_new['S-FWHM'] + header_ref['S-FWHM']) / 2


    # run Source Extractor on Scorr image and accompanying mask
    sexcat_pos = '{}_Scorr_cat_pos.fits'.format(base)
    result = run_sextractor(fits_Scorr, sexcat_pos,
                            get_par(set_zogy.sex_cfg,tel),
                            get_par(set_zogy.sex_par,tel), pixscale, header,
                            fwhm=fwhm, fits_mask=fits_newref_mask, npasses=1,
                            tel=tel, set_zogy=set_zogy, nthreads=nthreads,
                            Scorr_mode='init')


    # read background-subtracted output image (created by source
    # extractor) from 'initial' run above
    fits_Scorr_bkgsub = '{}_Scorr_bkgsub.fits'.format(base)
    data_Scorr_bkgsub = read_hdulist (fits_Scorr_bkgsub, dtype='float32')


    # keep a copy of the original Scorr image if keep_tmp is True
    if get_par(set_zogy.keep_tmp,tel):
        fits_Scorr_orig = '{}_Scorr_orig.fits'.format(base)
        shutil.copy2 (fits_Scorr, fits_Scorr_orig)

    # overwrite Scorr image with the background-subtracted Scorr
    # image
    os.rename (fits_Scorr_bkgsub, fits_Scorr)


    # determine statistics on background-subtracted Scorr image (done
    # already before, but possibly the background subtraction will
    # change the result slightly)
    edge = 100
    ysize, xsize = data_Scorr_bkgsub.shape
    nstat = int(0.1 * (xsize*ysize))
    x_stat = (np.random.rand(nstat)*(xsize-2*edge)).astype(int) + edge
    y_stat = (np.random.rand(nstat)*(ysize-2*edge)).astype(int) + edge
    log.info ('calculating Scorr statistics in [get_trans]')
    #mean, median, std = sigma_clipped_stats (data_Scorr)
    #print ('mean: {}, median: {}, std: {}'
    #       .format(mean, median, std))
    mean_Scorr, median_Scorr, std_Scorr = sigma_clipped_stats (
        data_Scorr_bkgsub[y_stat,x_stat].astype(float))
    log.info ('mean_Scorr: {:.3f}, median_Scorr: {:.3f}, std_Scorr: {:.3f}'
              .format(mean_Scorr, median_Scorr, std_Scorr))

    # update values in transient header
    header_trans['Z-SCMED'] = (median_Scorr, 'median Scorr full image')
    header_trans['Z-SCSTD'] = (std_Scorr, 'sigma (STD) Scorr full image')


    if False:
        # write normalized Scorr image to disk
        data_Scorr_bkgsub_norm = data_Scorr_bkgsub/std_Scorr
        fits_Scorr_bkgsub_norm = '{}_Scorr_bkgsub_norm.fits'.format(base)
        fits.writeto (fits_Scorr_bkgsub_norm, data_Scorr_bkgsub_norm, header,
                      overwrite=True)


    # run source extractor on the positive and negative
    # background-subtracted (and possibly normalized) Scorr images
    # with the above std_Scorr so that SExtractor uses the right
    # detection threshold
    result = run_sextractor(fits_Scorr, sexcat_pos,
                            get_par(set_zogy.sex_cfg,tel),
                            get_par(set_zogy.sex_par,tel), pixscale, header,
                            fwhm=fwhm, fits_mask=fits_newref_mask, npasses=1,
                            tel=tel, set_zogy=set_zogy, nthreads=nthreads,
                            Scorr_mode='pos', std_Scorr=std_Scorr)
                            #image_analysis=fits_D_neg, std_Scorr=std_Scorr)


    # prepare the negative Scorr and D images
    fits_Scorr_bkgsub_neg = '{}_Scorr_bkgsub_neg.fits'.format(base)
    fits.writeto (fits_Scorr_bkgsub_neg, -data_Scorr_bkgsub, header,
                  overwrite=True)


    data_D = read_hdulist(fits_D, dtype='float32')
    if False:
        # fits_D_neg is not used anymore
        fits_D_neg = '{}_D_neg.fits'.format(base)
        fits.writeto (fits_D_neg, -data_D, overwrite=True)


    sexcat_neg = '{}_Scorr_cat_neg.fits'.format(base)
    result = run_sextractor(fits_Scorr_bkgsub_neg, sexcat_neg,
                            get_par(set_zogy.sex_cfg,tel),
                            get_par(set_zogy.sex_par,tel), pixscale, header,
                            fwhm=fwhm, fits_mask=fits_newref_mask, npasses=1,
                            tel=tel, set_zogy=set_zogy, nthreads=nthreads,
                            Scorr_mode='neg', std_Scorr=std_Scorr)
                            #image_analysis=fits_D_neg, std_Scorr=std_Scorr)


    if not get_par(set_zogy.keep_tmp,tel):
        remove_files ([fits_Scorr_bkgsub_neg], verbose=True)


    # read positive table
    table_trans_pos = Table.read(sexcat_pos, memmap=True)
    # read off values at indices of X_PEAK and Y_PEAK
    index_x = table_trans_pos['X_PEAK'] - 1
    index_y = table_trans_pos['Y_PEAK'] - 1
    Scorr_peak_pos = data_Scorr_bkgsub[index_y, index_x]
    # add 'SNR_ZOGY' to table
    table_trans_pos.add_column (Scorr_peak_pos, name='SNR_ZOGY')

    # same for negative
    table_trans_neg = Table.read(sexcat_neg, memmap=True)
    index_x = table_trans_neg['X_PEAK'] - 1
    index_y = table_trans_neg['Y_PEAK'] - 1
    Scorr_peak_neg = data_Scorr_bkgsub[index_y, index_x]
    table_trans_neg.add_column (Scorr_peak_neg, name='SNR_ZOGY')

    # merge positive and negative catalogs
    table_trans = vstack([table_trans_pos, table_trans_neg])
    log.info ('total number of transients (pos+neg): {}'
              .format(len(table_trans)))


    # Filter on significance
    # ======================

    # filter by abs(SNR_ZOGY) >= 6; do not divide by [std_Scorr] -
    # the value in the background-subtracted Scorr image is assumed to
    # be the significance irrespective of the STD in the Scorr
    # image. This is corroborated by a test with fake stars of S/N=10
    # in an image compared with a deep reference image where the
    # resulting Scorr STD was 0.8: scaling the stars' significance
    # with 1/0.8 would boost their values beyond 10. In run_sextractor
    # the source-extractor threshold is adjusted by [std_Scorr],
    # because source extractor is using the actual STD in the image to
    # determine its detection threshold.
    nsigma_norm = get_par(set_zogy.transient_nsigma,tel) #/ std_Scorr
    mask_signif = np.abs(table_trans['SNR_ZOGY']) >= nsigma_norm
    table_trans = table_trans[mask_signif]
    log.info ('transient detection threshold: {}'.format(nsigma_norm))
    log.info ('ntrans after threshold cut: {}'.format(len(table_trans)))

    if get_par(set_zogy.make_plots,tel):
        ds9_rad = 2
        result = prep_ds9regions(
            '{}_ds9regions_trans_filt0_none.txt'.format(base),
            table_trans['X_POS'], table_trans['Y_POS'],
            radius=ds9_rad, width=2, color='cyan',
            value=table_trans['FLAGS_MASK'])


    # filter on FLAGS_MASK values (1 + 2 + 4 + 8 + 16 + 32 = 63 = 2**6 - 1)
    # ===========================

    mask_flags = np.zeros(len(table_trans), dtype=bool)
    masktype_discard = 63
    mask_value = get_par(set_zogy.mask_value,tel)
    # iterate over all mask values
    for val in mask_value.values():
        # check if this one is to be discarded
        if masktype_discard & val == val:
            mask_discard = (table_trans['FLAGS_MASK'] & val == val)
            mask_flags[mask_discard] = True
            log.info('discarding FLAGS_MASK value {}; no. of objects: {}'
                     .format(val, np.sum(mask_discard)))

    if not keep_all:
        table_trans = table_trans[~mask_flags]

    log.info ('ntrans after FLAGS_MASK cut: {}'.format(len(table_trans)))

    if get_par(set_zogy.make_plots,tel):
        ds9_rad += 2
        result = prep_ds9regions(
            '{}_ds9regions_trans_filt1_flagsmask.txt' .format(base),
            table_trans['X_POS'], table_trans['Y_POS'],
            radius=ds9_rad, width=2, color='blue',
            value=table_trans['FLAGS'])


    # filter on FLAGS values (all = 2**8 - 1 = 255)
    # =============================================

    mask_flags = np.zeros(len(table_trans), dtype=bool)
    masktype_discard = 255
    # iterate over all mask values (1 .. 128)
    for val in 2**np.arange(8):
        # check if this one is to be discarded
        if masktype_discard & val == val:
            mask_discard = (table_trans['FLAGS'] & val == val)
            mask_flags[mask_discard] = True
            log.info('discarding FLAGS value {}; no. of objects: {}'
                     .format(val, np.sum(mask_discard)))

    if not keep_all:
        table_trans = table_trans[~mask_flags]

    log.info ('ntrans after FLAGS cut: {}'.format(len(table_trans)))

    if get_par(set_zogy.make_plots,tel):
        ds9_rad += 2
        result = prep_ds9regions(
            '{}_ds9regions_trans_filt2_flags.txt'.format(base),
            table_trans['X_POS'], table_trans['Y_POS'],
            radius=ds9_rad, width=2, color='magenta',
            value=table_trans['ELONGATION'])


    # filter on ELONGATION
    # ====================

    mask_elong = (table_trans['ELONGATION'] <= 5)
    if not keep_all:
        table_trans = table_trans[mask_elong]

    log.info ('ntrans after ELONGATION cut: {}'.format(len(table_trans)))


    # PSF fit to D
    #=============

    # read fratio (Fn/Fr) from header_trans in order to scale the
    # reference image and its variance
    sn, sr, fratio, fratio_err = 1, 1, 1, 0
    if 'Z-FNR' in header_trans:
        fratio = header_trans['Z-FNR']

    if 'Z-FNRERR' in header_trans:
        fratio_err = header_trans['Z-FNRERR']

    log.info ('fratio from header_trans: {} +- {}'.format(fratio, fratio_err))


    # use sum of variances of new and ref images (where ref image
    # needs to be scaled with flux ratio fn/fr) as variance image to
    # use in psf fitting to D image (in get_psfoptflux)
    # initially used sn and sr as scalar estimate of the new and
    # ref image background standard deviation:
    #data_D_var = data_new + new_bkg_std**2 +
    #            (data_ref + ref_bkg_std**2) * fratio**2
    # using full background STD images
    data_new = read_hdulist(fits_new, dtype='float32')
    data_ref = read_hdulist(fits_ref, dtype='float32')
    data_new_bkg_std = read_hdulist(fits_new_bkg_std, dtype='float32')
    data_ref_bkg_std = read_hdulist(fits_ref_bkg_std, dtype='float32')
    data_D_var = ((np.maximum(data_new,0) + data_new_bkg_std**2) +
                  (np.maximum(data_ref,0) + data_ref_bkg_std**2) * fratio**2 +
                  (data_ref * fratio_err)**2)


    # try fitting P_D (combination of PSFs of new and ref images)
    # to D, Scorr, Fpsf and Fpsferr images in order to:
    #
    # (1) use chi2 of PSF fit to D to discard fake transients
    #
    # (2) improve the estimate of the peak value in Scorr, Fpsf
    #     and Fpsferr, which should be possible as the PSF is
    #     better sampled than the image pixels

    def help_psffit_D (psffit, moffat, gauss, get_flags_mask_central=False):

        # use [get_psfoptflux] to perform a PSF fit to D
        results = get_psfoptflux_mp (
            fits_new_psf, data_D, data_D_var, data_newref_mask,
            table_trans['X_PEAK'], table_trans['Y_PEAK'], psffit=psffit,
            moffat=moffat, gauss=gauss, psfex_bintable_ref=fits_ref_psf,
            data_new_bkg_std=data_new_bkg_std, data_ref_bkg_std=data_ref_bkg_std,
            header_new=header_new, header_ref=header_ref,
            Scorr_peak=table_trans['SNR_ZOGY'], set_zogy=set_zogy,
            get_flags_mask_central=get_flags_mask_central,
            nthreads=nthreads, tel=tel)

        return results



    # PSF fit to D, directly added as columns to table_trans
    colnames = ['X_PSF_D', 'XERR_PSF_D', 'Y_PSF_D', 'YERR_PSF_D',
                'E_FLUX_PSF_D', 'E_FLUXERR_PSF_D', 'CHI2_PSF_D', 'FLAGS_CENTRAL']
    table_trans.add_columns(
        help_psffit_D (True, False, False, get_flags_mask_central=True),
        names=colnames)

    log.info ('[get_trans] time after PSF fit to D: {}'.format(time.time()-t))


    # filter on CHI2_PSF_D
    # ====================

    if get_par(set_zogy.make_plots,tel):
        ds9_rad += 2
        result = prep_ds9regions(
            '{}_ds9regions_trans_filt3_elong.txt'.format(base),
            table_trans['X_POS'], table_trans['Y_POS'],
            radius=ds9_rad, width=2, color='blue',
            value=table_trans['CHI2_PSF_D'])


    # filter out transient candidates with high chi2 and non-finite values
    chi2_max = get_par(set_zogy.chi2_max,tel)
    chi2_snr_limit = get_par(set_zogy.chi2_snr_limit,tel)
    mask_keep = ((table_trans['CHI2_PSF_D'] <= chi2_max) |
                 (np.abs(table_trans['SNR_ZOGY']) >= chi2_snr_limit))



    row_numbers = np.arange(len(table_trans))+1
    # discard rows where fit values are infinite or NaN
    for col in colnames:
        mask_finite = np.isfinite(table_trans[col])
        nbad = np.sum(~mask_finite)
        if nbad > 0:
            mask_keep &= mask_finite
            log.warning ('column {} contains {} infinite or NaN value(s) for '
                         'image {}; discarding the corresponding row(s): {}'
                         .format(col, nbad, fits_new, row_numbers[~mask_finite]))
    # filter
    if not keep_all:
        table_trans = table_trans[mask_keep]

    log.info('ntrans after PSF_D fit chi2 filter: {}'.format(len(table_trans)))


    # filter on S/N_PSF_D
    # ===================

    # check S/N of E_FLUX_PSF_D
    s2n_psfD = np.zeros(len(table_trans))
    mask_nonzero = (table_trans['E_FLUXERR_PSF_D'] != 0)
    s2n_psfD[mask_nonzero] = (table_trans['E_FLUX_PSF_D'][mask_nonzero] /
                              table_trans['E_FLUXERR_PSF_D'][mask_nonzero])


    if get_par(set_zogy.make_plots,tel):
        ds9_rad += 2
        result = prep_ds9regions(
            '{}_ds9regions_trans_filt4_chi2_PSF_D.txt'.format(base),
            table_trans['X_POS'], table_trans['Y_POS'],
            radius=ds9_rad, width=2, color='yellow',
            value=s2n_psfD)


    # keep objects above sigma threshold, but make sure to also keep
    # objects for which E_FLUXERR_PSF_D=0 because of failing fit
    nsigma = get_par(set_zogy.transient_nsigma,tel)
    mask_keep = (np.abs(s2n_psfD) >= nsigma) | ~mask_nonzero
    if not keep_all:
        table_trans = table_trans[mask_keep]

    log.info('ntrans after PSF_D fit S/N filter: {}'.format(len(table_trans)))


    # filter on FLAGS_CENTRAL (determined while performing PSF fit to D above)
    # =======================

    # require central flags to be zero
    mask_keep = (table_trans['FLAGS_CENTRAL'] == 0)
    if not keep_all:
        table_trans = table_trans[mask_keep]

    log.info('ntrans after FLAGS_CENTRAL filter: {}'.format(len(table_trans)))


    # Gauss fit to D
    # ==============

    # add Gauss parameters in case of original transient catalog
    use_new_transcat = get_par(set_zogy.use_new_transcat,tel)
    if not use_new_transcat:

        # Gauss fit to D, directly added as columns to table_trans
        colnames = ['X_GAUSS_D', 'XERR_GAUSS_D', 'Y_GAUSS_D', 'YERR_GAUSS_D',
                    'FWHM_GAUSS_D', 'ELONG_GAUSS_D', 'CHI2_GAUSS_D']
        table_trans.add_columns(help_psffit_D (False, False, True),
                                names=colnames)

        log.info ('[get_trans] time after Gauss fit to D: {}'
                  .format(time.time()-t))


        if get_par(set_zogy.make_plots,tel):
            ds9_rad += 2
            result = prep_ds9regions(
                '{}_ds9regions_trans_filt5_s2n_PSF_D.txt'.format(base),
                table_trans['X_POS'], table_trans['Y_POS'],
                radius=ds9_rad, width=2, color='blue',
                value=table_trans['CHI2_GAUSS_D'])


        # filter out transient candidates with high chi2 values
        mask_keep = ((table_trans['CHI2_GAUSS_D'] <= chi2_max) |
                     (np.abs(table_trans['SNR_ZOGY']) >= chi2_snr_limit))

        # discard rows where fit values are infinite or NaN
        for col in colnames:
            mask_finite = np.isfinite(table_trans[col])
            nbad = np.sum(~mask_finite)
            if nbad > 0:
                mask_keep &= mask_finite
                log.warning ('column {} contains {} infinite or NaN values for '
                             'image {}; discarding the corresponding row(s)'
                             .format(col, nbad, fits_new))
        # filter
        if not keep_all:
            table_trans = table_trans[mask_keep]

        log.info('ntrans after Gauss fit chi2 filter: {}'
                 .format(len(table_trans)))


        if get_par(set_zogy.make_plots,tel):
            ds9_rad += 2
            result = prep_ds9regions(
                '{}_ds9regions_trans_filt6_chi2_GAUSS.txt'.format(base),
                table_trans['X_POS'], table_trans['Y_POS'],
                radius=ds9_rad, width=2, color='green')



    # determine RAs and DECs
    wcs_new = WCS(header_new)
    wcs_ref = WCS(header_ref)

    ra_peak, dec_peak = wcs_new.all_pix2world(table_trans['X_PEAK'],
                                              table_trans['Y_PEAK'], 1)

    # determine RA and DEC corresponding to x_psf_D and y_psf_D
    ra_psf_D, dec_psf_D = wcs_new.all_pix2world(table_trans['X_PSF_D'],
                                                table_trans['Y_PSF_D'], 1)

    if False:
        # determine RA and DEC corresponding to x_moffat and y_moffat
        ra_moffat, dec_moffat = wcs_new.all_pix2world(table_trans['X_MOFFAT_D'],
                                                      table_trans['Y_MOFFAT_D'],
                                                      1)


    # determine RA and DEC corresponding to x_pos and y_pos
    ra_D, dec_D = wcs_new.all_pix2world(table_trans['X_POS'],
                                        table_trans['Y_POS'], 1)

    # adding RAs and DECs to table
    table_trans.add_columns([ra_peak,   dec_peak,
                             ra_psf_D,  dec_psf_D,
                             ra_D,  dec_D],
                            names=['RA_PEAK',    'DEC_PEAK',
                                   'RA_PSF_D',   'DEC_PSF_D',
                                   'RA_SCORR',   'DEC_SCORR'])

    # add GAUSS_D columns for original catalog
    if not use_new_transcat:
        ra_gauss, dec_gauss = wcs_new.all_pix2world(table_trans['X_GAUSS_D'],
                                                    table_trans['Y_GAUSS_D'], 1)
        table_trans.add_columns([ra_gauss,  dec_gauss],
                                names=['RA_GAUSS_D', 'DEC_GAUSS_D'])



    # need to convert psf fluxes to magnitudes by applying the zeropoint
    keywords = ['exptime', 'filter', 'obsdate']
    exptime, filt, obsdate = read_header (header_new, keywords)


    # get zeropoint from [header_new]
    if 'PC-ZP' in header_new:
        zp = header_new['PC-ZP']
    else:
        zp = get_par(set_zogy.zp_default,tel)[filt]


    # corresponding error
    if 'PC-ZPERR' in header_new:
        zperr = header_new['PC-ZPERR']
    elif 'PC-ZPSTD' in header_new:
        zperr = header_new['PC-ZPSTD']
    else:
        log.warning ('keywords PC-ZPERR nor PC-ZPSTD found; adopting zperr=0')
        zperr = 0


    # for ML/BG: if set_zogy.MLBG_phot_apply_chanzp is True, replace
    # [zp] with array of zeropoints, one for each object; same for
    # [zp_err]
    if (tel[0:2] in ['ML','BG']
        and get_par(set_zogy.MLBG_phot_apply_chanzp,tel)):

        # even though MLBG_phot_apply_chanzp is True, channel
        # zeropoints may not be available and moreover they could be
        # zero; use image zp in those cases
        xcoords = table_trans['X_PEAK'].value
        ycoords = table_trans['Y_PEAK'].value

        zp_chan = [header_new['PC-ZP{}'.format(i+1)]
                   if 'PC-ZP{}'.format(i+1) in header_new
                   and header_new['PC-ZP{}'.format(i+1)] != 0
                   else zp for i in range(16)]
        zp = get_zp_coords (xcoords, ycoords, zp_chan, zp)


        # same for zperr
        zperr_chan = [header_new['PC-ZPE{}'.format(i+1)]
                      if 'PC-ZPE{}'.format(i+1) in header_new
                      and header_new['PC-ZPE{}'.format(i+1)] != 0
                      else zperr for i in range(16)]
        zperr = get_zp_coords (xcoords, ycoords, zperr_chan, zperr)



    # get airmass from [header_new]
    if 'AIRMASSC' in header_new:
        airmass = header_new['AIRMASSC']
    elif 'PC-AIRM' in header_new:
        airmass = header_new['PC-AIRM']


    # determine individual airmasses of transients to be able to
    # determine their magnitudes accurately also at high airmass
    lat = get_par(set_zogy.obs_lat,tel)
    lon = get_par(set_zogy.obs_lon,tel)
    height = get_par(set_zogy.obs_height,tel)
    airmass_trans = get_airmass (table_trans['RA_PEAK'].value,
                                 table_trans['DEC_PEAK'].value,
                                 obsdate, lat, lon, height)


    # get magnitudes corresponding to absolute fluxes; fluxes, which
    # can be negative for e.g. an object detected in the reference
    # image but not in the new image, are first converted to positive
    # fluxes. That it was a negative flux object is still clear from
    # the sign of Scorr_peak.
    data_Fpsf = read_hdulist (fits_Fpsf, dtype='float32')
    data_Fpsferr = read_hdulist (fits_Fpsferr, dtype='float32')
    # read off fluxes and errors at X_PEAK and Y_PEAK indices
    flux_peak = data_Fpsf[table_trans['Y_PEAK']-1, table_trans['X_PEAK']-1]
    fluxerr_peak = data_Fpsferr[table_trans['Y_PEAK']-1, table_trans['X_PEAK']-1]
    del data_Fpsf, data_Fpsferr


    ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]
    mag_peak, magerr_peak, magerrtot_peak, fnu_peak, fnuerr_peak, \
        fnuerrtot_peak = apply_zp (flux_peak, zp, airmass_trans, exptime,
                                   ext_coeff, fluxerr=fluxerr_peak,
                                   return_fnu=True, zperr=zperr)


    mag_psf_D, magerr_psf_D, magerrtot_psf_D, fnu_psf_D, fnuerr_psf_D, \
        fnuerrtot_psf_D = apply_zp (table_trans['E_FLUX_PSF_D'], zp,
                                    airmass_trans, exptime, ext_coeff,
                                    fluxerr=table_trans['E_FLUXERR_PSF_D'],
                                    return_fnu=True, zperr=zperr)


    log.info ('[get_trans] time after converting flux to mag: {}'
              .format(time.time()-t))

    # adding magnitudes and also flux_peak to table
    table_trans.add_columns([flux_peak, fluxerr_peak,
                             mag_peak,  magerr_peak, magerrtot_peak,
                             fnu_peak,  fnuerr_peak, fnuerrtot_peak,
                             mag_psf_D, magerr_psf_D, magerrtot_psf_D,
                             fnu_psf_D, fnuerr_psf_D, fnuerrtot_psf_D],
                            names=[
                                'E_FLUX_ZOGY', 'E_FLUXERR_ZOGY',
                                'MAG_ZOGY', 'MAGERR_ZOGY', 'MAGERRTOT_ZOGY',
                                'FNU_ZOGY', 'FNUERR_ZOGY', 'FNUERRTOT_ZOGY',
                                'MAG_PSF_D', 'MAGERR_PSF_D', 'MAGERRTOT_PSF_D',
                                'FNU_PSF_D', 'FNUERR_PSF_D', 'FNUERRTOT_PSF_D'])


    # change some of the column names
    colnames_new = {'X_POS':      'X_POS_SCORR',
                    'Y_POS':      'Y_POS_SCORR',
                    'XVAR_POS':   'XVAR_POS_SCORR',
                    'YVAR_POS':   'YVAR_POS_SCORR',
                    'XYCOV_POS':  'XYCOV_POS_SCORR',
                    'ELONGATION': 'ELONG_SCORR',
                    'FLAGS':      'FLAGS_SCORR',
                    'FLAGS_MASK': 'FLAGS_MASK_SCORR'}

    for key in colnames_new.keys():
        if key in table_trans.colnames:
            table_trans.rename_column (key, colnames_new[key])



    table_trans['NUMBER'] = np.arange(len(table_trans))+1
    ntrans = len(table_trans)

    # create output fits catalog; final table definition is determined
    # in [format_cat]
    table_trans.write('{}.transcat'.format(base_newref), format='fits',
                      overwrite=True)

    # define keys of thumbnails
    keys_thumbnails = ['THUMBNAIL_RED', 'THUMBNAIL_REF',
                       'THUMBNAIL_D', 'THUMBNAIL_SCORR']

    # extract the thumbnail images corresponding to the transients in
    # case either thumbnail data is being saved or MeerCRAB
    # probabilities need to be calculated for ML/BG
    if (get_par(set_zogy.save_thumbnails,tel) or
        (get_par(set_zogy.ML_calc_prob,tel) and tel[0:2] in ['ML','BG'])):

        n_thumbnails = len(keys_thumbnails)

        # coordinates to loop
        xcoords = table_trans['X_PEAK']
        ycoords = table_trans['Y_PEAK']
        ncoords = len(xcoords)

        # thumbnail size
        size_thumbnails = get_par(set_zogy.size_thumbnails,tel)

        # initialize dictionary with keys [keys_thumbnails] and values
        # the numpy filenames where data_thumbnail is saved
        dict_thumbnails = {}

        # list of full data images to use in loop
        data_full = [data_new, data_ref, data_D, data_Scorr_bkgsub]

        # size of full input images; assuming they have identical shapes
        ysize, xsize = data_new.shape

        # loop thumbnails
        for i_tn, key in enumerate(keys_thumbnails):

            # initialise output thumbnail column
            data_thumbnail = np.zeros((ncoords, size_thumbnails,
                                       size_thumbnails), dtype='float32')

            # loop x,y coordinates
            for i_pos in range(ncoords):

                # get index around x,y position using function
                # [get_index_around_xy]
                index_full, index_tn, __, __ = get_index_around_xy (
                    ysize, xsize, ycoords[i_pos], xcoords[i_pos],
                    size_thumbnails)


                try:

                    data_thumbnail[i_pos][index_tn] = data_full[i_tn][index_full]

                    # if [orient_thumbnails] is switched on,
                    # orient the thumbnails in North-up, East left
                    # orientation
                    if get_par(set_zogy.orient_thumbnails,tel):

                        # input reference data is the remapped
                        # reference image and its orientation is
                        # the same as that of the new, D and Scorr
                        # images, and so the same header
                        # (header2add=header_newzogy) should be
                        # used rather than the reference image
                        # header header_ref
                        #data_thumbnails[i_tn, i_pos] = orient_data (
                        #    data_thumbnails[i_tn, i_pos], header_new,
                        #    MLBG_rot90_flip=True, tel=tel)
                        data_thumbnail[i_pos] = orient_data (
                            data_thumbnail[i_pos], header_new,
                            MLBG_rot90_flip=True, pixscale=pixscale_new, tel=tel)


                except Exception as e:
                    log.exception('skipping remapping of {} at x,y: '
                                  '{:.0f},{:.0f} due to exception: {}'
                                  .format(key, xcoords[i_pos],
                                          ycoords[i_pos], e))

            # save thumbnail as numpy file and record key and name in
            # dictionary
            dict_thumbnails[key] = save_npy_fits (data_thumbnail, '{}_{}.npy'
                                                  .format(base, key))

    else:
        for key in keys_thumbnails:
            dict_thumbnails[key] = None


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_trans')

    return table_trans, dict_thumbnails


################################################################################

def get_edge_coords (data_mask):

    """function to extract x- and y-coordinates (not indices!) of the
    inner pixels of the edge pixels in the input boolean mask; returns
    two numpy arrays, one for x and one for y. If no edge pixels are
    present in the input mask, the arrays will be empty.

    """

    # identify edge pixels
    value_edge = get_par(set_zogy.mask_value['edge'],tel)
    mask_edge = (data_mask & value_edge == value_edge)

    # grow edge with one pixel and subtract the original edge mask
    mask_edge_inner = ndimage.binary_dilation(mask_edge).astype('bool')
    mask_edge_inner[mask_edge] = False

    # convert the resulting mask into x- and y-coordinates
    ysize, xsize = data_mask.shape
    x = np.arange(xsize)+1
    y = np.arange(ysize)+1
    xx, yy = np.meshgrid(x, y)

    return xx[mask_edge_inner], yy[mask_edge_inner]


################################################################################

def prep_ds9regions(filename, x, y, radius=5, width=2, color='green',
                    value=None):

    """Function that creates a text file with the name [filename] that
    can be used to mark objects at an array of pixel positions (x,y)
    with a circle when displaying an image with ds9."""

    # prepare ds9 region file
    f = open(filename, 'w')
    ncoords = len(x)
    for i in range(ncoords):
        if value is None:
            f.write('circle({},{},{}) # color={} width={}\n'
                    .format(x[i], y[i], radius, color, width))
        else:
            val_tmp = value[i]
            try:
                if 'float' in str(val_tmp.dtype):
                    val_tmp = '{:.2f}'.format(value[i])
            except:
                pass

            f.write('circle({},{},{}) # color={} width={} text={{{}}} '
                    'font="times 12"\n'
                    .format(x[i], y[i], radius, color, width, val_tmp))

    f.close()
    return


################################################################################

def trans_measure(I, x_index, y_index, var_bkg=0):

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
    A, B, THETA, ERRA, ERRB, ERRTHETA = get_shape_parameters (
        X2, Y2, XY, ERRX2, ERRY2, ERRXY)

    return X, Y, X2, Y2, XY, ERRX2, ERRY2, ERRXY, A, B, THETA, \
        ERRA, ERRB, ERRTHETA


################################################################################

def get_shape_parameters (x2, y2, xy, errx2, erry2, errxy):

    """Function to calculate a, b, theta, erra, errb, errtheta from x2,
    y2, xy, errx2, erry2, errxy (see SExtractor manual). """

    # Eq. 24 from SExtractor manual:
    term1 = (x2 + y2) / 2
    term2 = np.sqrt(0.25*(x2-y2)**2 + xy**2)
    if (term1 + term2) >= 0:
        a = np.sqrt(term1 + term2)
    else:
        a = 0
    # Eq. 25 from SExtractor manual:
    if term1 >= term2:
        b = np.sqrt(term1 - term2)
    else:
        b = 0
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
        errb = 0
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

def get_psfoptflux (psfex_bintable, D, bkg_var, D_mask, xcoords, ycoords,
                    psffit=False, moffat=False, gauss=False, get_limflux=False,
                    limflux_nsigma=5., psfex_bintable_ref=None,
                    data_new_bkg_std=None, data_ref_bkg_std=None,
                    header_new=None, header_ref=None, header_trans=None,
                    imtype=None, Scorr_peak=None, inject_fake=False,
                    nsigma_fake=10, remove_psf=False,
                    replace_sat_psf=False, replace_sat_nmax=100,
                    set_zogy=None, tel=None, fwhm=None, diff=True,
                    get_flags_mask_central=False, local_bkg=None,
                    nthreads=1):


    """Function that returns the optimal flux and its error (using the
       function [flux_optimal] of a source at pixel positions [xcoords],
       [ycoords] given the inputs: .psf file produced by PSFex
       [psfex_bintable], background-subtracted data [D] and background
       variance [bkg_var]. [D] and bkg_var are assumed to be in e-.

       [D] is a 2D array meant to be the full image; [bkg_var] can be a 2D
       array with the same shape as [D] or a scalar. [xcoords] and
       [ycoords] are arrays, and the output flux and its error will be
       arrays as well.

       If [replace_sat_psf]=True, the function will update [D] where
       any saturated values in the PSF footprint of the
       [xcoords],[ycoords] coordinates that are being processed is
       replaced by the expected flux according to the PSF. If the
       object contains more than [replace_sat_nmax] saturated pixels,
       the replacement is skipped for that object.

       If [get_limflux]=True, the function will return the limiting
       flux (at significance [limflux_nsigma]) at the input
       coordinates using the function [flux_optimal_s2n] in
       [flux_opt]; [fluxerr_opt] will contain zeros in that case, and
       PSF fitting is not performed even if [psffit]=True.

       If [inject_fake]=True, fake stars at the input [xcoords],
       [ycoords] will be added to the image.

    """

    log.info('executing get_psfoptflux ...')
    if get_par(set_zogy.timing,tel): t = time.time()

    # make sure x and y have same length
    if np.isscalar(xcoords) or np.isscalar(ycoords):
        log.error('xcoords and ycoords should be arrays')
    else:
        assert len(xcoords) == len(ycoords)

    # initialize output arrays
    ncoords = len(xcoords)
    flux_opt = np.zeros(ncoords, dtype='float32')
    fluxerr_opt = np.zeros_like(flux_opt)

    if get_flags_mask_central:
        flags_mask_central = np.zeros(ncoords, dtype='uint8')

    if psffit:
        flux_psf = np.zeros_like(flux_opt)
        fluxerr_psf = np.zeros_like(flux_opt)
        xshift_psf = np.zeros_like(flux_opt)
        yshift_psf = np.zeros_like(flux_opt)
        chi2_psf = np.zeros_like(flux_opt)
        xerr_psf = np.zeros_like(flux_opt)
        yerr_psf = np.zeros_like(flux_opt)

    if moffat or gauss:
        x_moffat = np.zeros(ncoords)
        xerr_moffat = np.zeros_like(flux_opt)
        y_moffat = np.zeros(ncoords)
        yerr_moffat = np.zeros_like(flux_opt)
        fwhm_moffat = np.zeros_like(flux_opt)
        elong_moffat = np.zeros_like(flux_opt)
        chi2_moffat = np.zeros_like(flux_opt)

    if inject_fake and np.isscalar(nsigma_fake):
        # convert to array
        nsigma_fake = (nsigma_fake * np.ones(ncoords)).astype('float32')


    # get dimensions of D
    ysize, xsize = np.shape(D)

    # read in PSF output binary table from psfex, containing the
    # polynomial coefficient images, and various PSF parameters using
    # the function [extract_psf_datapars]
    verbose = get_par(set_zogy.verbose,tel)
    results = extract_psf_datapars (psfex_bintable, verbose=verbose)
    (data_psf, header_psf, psf_fwhm, psf_samp, psf_size_config, psf_chi2,
     psf_nstars, polzero1, polscal1, polzero2, polscal2, poldeg) = results

    # same for reference image
    if psfex_bintable_ref is not None:

        results_ref = extract_psf_datapars (psfex_bintable_ref, verbose=verbose)
        (data_psf_ref, header_psf_ref, psf_fwhm_ref, psf_samp_ref,
         psf_size_config_ref, psf_chi2_ref, psf_nstars_ref, polzero1_ref,
         polscal1_ref, polzero2_ref, polscal2_ref, poldeg_ref) = results_ref


    # [psf_size] is the PSF size in image pixels, which determines the
    # size of [psf_ima] and [psf_ima_ref] below. For the photometry
    # measurements, [psf_size] should be defined by
    # 2*psf_rad_phot*fwhm_new or fwhm_ref; for the PSF fit to the
    # difference image, in which case the input imtype will be None,
    # the psf_size should be defined using the maximum of fwhm_new and
    # fwhm_ref.
    if imtype=='new':
        if fwhm is not None:
            fwhm_use = fwhm
        else:
            fwhm_use = fwhm_new

        # initial fwhm used for flux limits and Moffat/Gauss fits below
        fwhm_fit_init = fwhm_use

    elif imtype=='ref':
        if fwhm is not None:
            fwhm_use = fwhm
        else:
            fwhm_use = fwhm_ref

        fwhm_fit_init = fwhm_use

    elif imtype is None:
        fwhm_use = max(fwhm_new, fwhm_ref*(pixscale_ref/pixscale_new))
        fwhm_fit_init = (fwhm_new + fwhm_ref) / 2.


    psf_size = 2 * get_par(set_zogy.psf_rad_phot,tel) * fwhm_use
    # make sure [psf_size] is not larger than [psf_size_config] *
    # [psf_samp], which will happen if FWHM is very large
    psf_size = int(min(psf_size, psf_size_config * psf_samp))
    # it should also not be larger than the vignet size; for the PSF
    # fits to the D image, that would lead to an exception because the
    # psf_imas used to build the P_D below would not have the same
    # size
    size_vignet = get_par(set_zogy.size_vignet,tel)
    psf_size = int(min(psf_size, size_vignet))

    # force it to be odd
    if psf_size % 2 == 0:
        psf_size -= 1


    # bad pixel mask values used inside loop
    mask_value = get_par(set_zogy.mask_value,tel)
    value_sat = mask_value['saturated']
    value_satcon = mask_value['saturated-connected']


    log.info ('psf_size used in [get_psfoptflux]: {} pix for imtype: {}'
              .format(psf_size, imtype))


    # if reference image PSF is involved, infer ref image pixel
    # coordinates corresponding to the new image xcoords, ycoords
    if psfex_bintable_ref is not None:

        # convert xycoords to ra, dec in the new image
        wcs_new = WCS(header_new)
        ra_new, dec_new = wcs_new.all_pix2world(xcoords, ycoords, 1)

        # convert these to pixel positions in the ref image
        wcs_ref = WCS(header_ref)
        xcoords_ref, ycoords_ref = wcs_ref.all_world2pix(ra_new, dec_new, 1)



    # loop coordinates
    for i in range(ncoords):

        t_tmp = time.time()

        # using function [get_psf_ima], construct the PSF image with
        # shape (psf_size, psf_size) at xcoords[i], ycoords[i]; this
        # image is at the original pixel scale
        psf_clean_factor = get_par(set_zogy.psf_clean_factor,tel)
        psf_ima, __ = get_psf_ima (
            data_psf, xcoords[i], ycoords[i], psf_size,
            psf_samp, polzero1, polscal1, polzero2, polscal2, poldeg,
            imtype=imtype, psf_clean_factor=psf_clean_factor, tel=tel)


        if i % 10000 == 0:
            t_new = time.time()-t_tmp


        # determine indices of PSF square footprint (with size:
        # [psf_size]) in an image with shape (ysize, xsize) at pixel
        # coordinates xcoords[i], ycoords[i]; if the footprint is
        # partially off the image, index_P is needed to define the
        # subset of pixels in the PSF footprint that are on the full
        # image. After remapping of the ref image, these indices also
        # apply to the ref image.
        index, index_P = get_P_indices (
            xcoords[i], ycoords[i], xsize, ysize, psf_size)

        # if coordinates off the image, the function [get_P_indices]
        # returns None and the rest can be skipped; continue with next
        # source
        if index is None:
            log.warning ('index return from [get_P_indices] is None for object '
                         'at x,y: {:.2f},{:.2f}; returning zero flux and fluxerr'
                         .format(xcoords[i], ycoords[i]))
            continue

        # if [psfex_bintable_ref] is provided, the new image [psf_ima]
        # will be convolved with the ref image [psf_ima_ref], and
        # replaced with the resulting combined PSF image
        if psfex_bintable_ref is not None:

            # same for reference image; includes rotation to the
            # orientation of the new image
            psf_ima_ref, __ = get_psf_ima (
                data_psf_ref, xcoords_ref[i], ycoords_ref[i], psf_size,
                psf_samp_ref, polzero1_ref, polscal1_ref, polzero2_ref,
                polscal2_ref, poldeg_ref, imtype='ref',
                remap=True, header=header_ref, header_new=header_new,
                psf_clean_factor=psf_clean_factor, tel=tel)


            # [psf_ima_ref] needs to be rotated to the orientation of
            # the new image
            psf_ima_ref = orient_data (psf_ima_ref, header_ref,
                                       header_out=header_new, tel=tel)


            # setting image standard deviations and flux ratios to
            # unity; if values present in headers, use those instead
            sn, sr, fn, fr = 1, 1, 1, 1

            if data_new_bkg_std is not None and data_ref_bkg_std is not None:
                sn = np.median(data_new_bkg_std[index])
                sr = np.median(data_ref_bkg_std[index])
            elif header_new is not None:
                if 'S-BKGSTD' in header_new:
                    sn = header_new['S-BKGSTD']
                if 'S-BKGSTD' in header_ref:
                    sr = header_new['S-BKGSTD']

            if header_trans is not None:
                if 'Z-FNR' in header_trans:
                    fr = fn / header_new['Z-FNR']


            # could save some time in rest of the loop below by saving
            # P_D for each subimage when it is created in [run_zogy],
            # and reading them in this function; they would need to be
            # fftshifted back and extracted as the center as they are
            # defined for the entire subimage. Downside: single PSF is
            # used for the entire subimage - not so unreasonable.

            # now combine [psf_ima] and [psf_ima_ref]
            # into single psf image using FFT, first performing FFT shift
            Pn_fftshift = fft.fftshift(psf_ima)
            Pr_fftshift = fft.fftshift(psf_ima_ref)
            # fourier transforms
            Pn_hat = fft.fft2(Pn_fftshift)
            Pr_hat = fft.fft2(Pr_fftshift)
            Pn_hat2_abs = np.abs(Pn_hat**2)
            Pr_hat2_abs = np.abs(Pr_hat**2)

            # following definitions as in zogy core function
            sn2 = sn**2
            sr2 = sr**2
            fn2 = fn**2
            fr2 = fr**2
            fD = (fr*fn) / np.sqrt(sn2*fr2+sr2*fn2)
            denominator = (sn2*fr2)*Pr_hat2_abs + (sr2*fn2)*Pn_hat2_abs
            P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)

            # multiply, inverse FFT and shift
            P_D = fft.ifftshift(np.real(fft.ifft2(P_D_hat)))
            # needs shift of 1,1 pixel because above operations
            # are on a odd-sized image
            psf_ima = ndimage.shift(P_D, (1,1), order=2)
            # normalize
            psf_ima /= np.sum(psf_ima)

            # this provides the same result (apart from the sn, sr,
            # fn, fr ratios), but is ~10 times slower:
            #psf_ima_config = ndimage.convolve(psf_ima_config, psf_ima_config_ref)
            #psf_ima_config /= np.sum(psf_ima_config)



        # extract index_P
        P_shift = psf_ima[index_P]

        # extract subsection from D, D_mask
        D_sub = D[index]
        D_mask_sub = D_mask[index]


        # if bkg_var is a scalar, convert to an image
        if np.isscalar(bkg_var):
            bkg_var_sub = bkg_var * np.ones(D_sub.shape)

        # if bkg_var is a 2D array with the same shape as the input
        # data image (D), extract the subimage part
        elif np.shape(bkg_var) == np.shape(D):
            bkg_var_sub = bkg_var[index]

        # if input background variance is a string
        # 'calc_from_data', calculate it from the input subimage
        # (can be used when background standard deviation was not
        # calculated already, e.g. for the zogy difference (D)
        # image when transients are being fit using this function
        elif bkg_var == 'calc_from_data':
            D_sub_masked = np.ma.masked_array(D_sub, mask=D_mask_sub)
            D_sub_mean, D_sub_median, D_sub_std = sigma_clipped_stats (
                D_sub_masked.astype(float), mask_value=0)
            bkg_var_sub = D_sub_std**2

        else:
            # if none of the above, write error message to log
            log.error('input parameter {} with value {} in function {} '
                      'not understood'
                      .format('bkg_var', bkg_var, 'get_psfoptflux'))


        # define central mask of object using the PSF
        mask_central = (P_shift >= 0.01 * np.amax(P_shift))

        # if flags_mask_central is specified, calculate the combined
        # flags_mask of the central object area
        if get_flags_mask_central:
            flags_mask_central[i] = np.sum(np.unique(D_mask_sub[mask_central]))


        # determine optimal or psf or limiting flux
        if get_limflux:
            # determine limiting flux at this position using
            # flux_optimal_s2n; if Poisson noise of objects should be
            # taken into account, then add background-subtracted image
            # to the background variance: bkg_var_sub + D_sub
            flux_opt[i] = flux_optimal_s2n (P_shift, bkg_var_sub,
                                            limflux_nsigma, fwhm=fwhm_fit_init)

        elif inject_fake:

            # add star to input image with following flux; here the
            # Poisson noise of objects is added to the background
            # variance
            flux_opt[i] = flux_optimal_s2n (P_shift, bkg_var_sub + D_sub,
                                            nsigma_fake[i], fwhm=fwhm_fit_init)
            D_sub += flux_opt[i] * psf_ima


        else:

            # use only those pixels that are not affected by any bad
            # pixels, cosmic rays, saturation, edge pixels, etc.
            mask_use = (D_mask_sub==0)

            if mask_use.shape != mask_central.shape:
                log.warning ('mask_use.shape: {} not equal to '
                             'mask_central.shape: {} for object at x,y: '
                             '{:.2f},{:.2f}; returning zero flux and fluxerr'
                             .format(mask_use.shape, mask_central.shape,
                                     xcoords[i], ycoords[i]))
                continue

            else:
                if np.sum(mask_central) != 0:
                    frac_tmp = (np.sum(mask_use & mask_central) /
                                np.sum(mask_central))
                else:
                    frac_tmp = 0

                # if the fraction of good pixels around the source is less
                # than set_zogy.source_minpixfrac, then continue with next
                # source; optimal flux will be zero and source will not be
                # included in output catalog - may result in very
                # saturated stars not ending up in output catalog
                if frac_tmp < get_par(set_zogy.source_minpixfrac,tel):
                    # too many bad pixel objects to warn about
                    #log.warning ('fraction of useable pixels around source '
                    #             'at x,y: {:.0f},{:.0f}: {} is less than limit '
                    #             'set by set_zogy.source_minpixfrac: {}; '
                    #             'returning zero flux and flux error'
                    #             .format(xcoords[i], ycoords[i], frac_tmp,
                    #                     get_par(set_zogy.source_minpixfrac,tel)))
                    continue


            # perform optimal photometry measurements
            try:

                show=False

                if False:

                    # show optimal fit of particular object(s)
                    dist_tmp = np.sqrt((xcoords[i]-2500)**2 +
                                       (ycoords[i]-9450)**2)
                    if dist_tmp < 50:
                        show=True
                        log.info ('xcoords[i]: {}, ycoords[i]: {}'
                                  .format(xcoords[i], ycoords[i]))


                # pass input parameter [local_bkg] for this object to
                # [flux_optimal] if it is defined
                if local_bkg is not None:
                    sky_bkg = local_bkg[i]
                else:
                    sky_bkg = 0


                flux_opt[i], fluxerr_opt[i] = flux_optimal (
                    P_shift, D_sub, bkg_var_sub, mask_use=mask_use,
                    fwhm=fwhm_use, sky_bkg=sky_bkg, show=show, tel=tel)


            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('problem running [flux_optimal] on object at '
                              'pixel coordinates: x={}, y={}; returning zero '
                              'flux and fluxerr'.format(xcoords[i], ycoords[i]))

                continue


            # infer error image used in psffit and gauss/moffat fits below
            D_sub_err = np.sqrt(np.abs(bkg_var_sub))


            # if psffit=True, perform PSF fitting
            if psffit:

                if i==0:
                    show=True
                else:
                    show=False

                try:
                    xshift_psf[i], xerr_psf[i], yshift_psf[i], yerr_psf[i], \
                        flux_psf[i], fluxerr_psf[i], chi2_psf[i] = (
                            flux_psffit (P_shift, D_sub, D_sub_err, flux_opt[i],
                                         mask_use=mask_use, fwhm=fwhm_fit_init,
                                         show=show, max_nfev=200, diff=diff))

                except Exception as e:
                    #log.exception(traceback.format_exc())
                    log.exception('problem running [flux_psffit] on object at '
                                  'pixel coordinates: {}, {}; returning zero '
                                  'flux, fluxerr, (x,y) shifts and chi2'
                                  .format(xcoords[i], ycoords[i]))
                    continue


            # if moffat=True, perform Moffat fit
            if moffat or gauss:

                if i==0:
                    show=True
                else:
                    show=False

                try:

                    # fit 2D gauss
                    x_moffat[i], xerr_moffat[i], y_moffat[i], yerr_moffat[i], \
                        fwhm_moffat[i], elong_moffat[i], chi2_moffat[i] = \
                            fit_moffat_single (
                                D_sub, D_sub_err, mask_use=mask_use,
                                fit_gauss=gauss, fwhm=fwhm_fit_init,
                                P_shift=P_shift, show=show, max_nfev=200)

                    x_moffat[i] += index[1].start
                    y_moffat[i] += index[0].start

                except Exception as e:
                    #log.exception(traceback.format_exc())
                    log.exception('problem running [fit_moffat_single] on object '
                                  'at pixel coordinates: {}, {}; returning zeros'
                                  .format(xcoords[i], ycoords[i]))
                    continue


            # replace saturated+connected pixels with PSF profile
            if replace_sat_psf:

                if False:
                    if np.sqrt((xcoords[i]-2500)**2 +
                               (ycoords[i]-9450)**2) < 100:
                        log.info('xcoords[i]: {}, ycoords[i]: {}'
                                 .format(xcoords[i], ycoords[i]))

                # determine pixels to be replaced; avoid including
                # saturated pixels from other objects within the
                # footprint

                # mask of saturated+connected pixels in subimage:
                mask_sat_sub = (D_mask_sub & value_sat == value_sat)
                mask_satcon_sub = (mask_sat_sub |
                                   (D_mask_sub & value_satcon == value_satcon))

                # label saturated+connected pixels; also
                # diagonally-connected pixels are included in a region
                struct = np.ones((3,3), dtype=bool)
                regions, nregions = ndimage.label(mask_satcon_sub,
                                                  structure=struct)


                # loop regions
                for ir in range(nregions):

                    mask_region = (regions==ir+1)
                    nsat = np.sum(mask_sat_sub & mask_region)

                    if np.sqrt((xcoords[i]-2500)**2 +
                               (ycoords[i]-9450)**2) < 100:
                        log.info('xcoords[i]: {}, ycoords[i]: {}'
                                 .format(xcoords[i], ycoords[i]))
                        log.info('nsat: {}, sum(mask_central & mask_region): {}'
                                 .format(nsat, np.sum(mask_central &
                                                      mask_region)))

                    # check if there is overlap with central PSF region
                    if (np.sum(mask_central & mask_region) > 0 and
                        nsat > 0 and nsat <= replace_sat_nmax):

                        mask_satcon_use = (mask_satcon_sub & mask_region)


                        # replace these by the estimate of flux_opt * PSF + 2d
                        # fit to the sky background, determined in function
                        # [flux_optimal]
                        D_sub[mask_satcon_use] = (
                            P_shift[mask_satcon_use] * flux_opt[i] + sky_bkg)


                        if False:
                            D_sub_orig = np.copy(D_sub)
                            if np.sqrt((xcoords[i]-2500)**2 +
                                       (ycoords[i]-9450)**2) < 100:
                                D_sub_psf = P_shift * flux_opt[i] + sky_bkg
                                ds9_arrays (D_sub_orig=D_sub_orig, D_sub=D_sub,
                                            D_mask_sub=D_mask_sub.astype(int))


                    else:
                        log.warning ('saturated+connected pixels not replaced '
                                     'for object at x,y={},{}; '
                                     'np.sum(mask_central & mask_region): {}, '
                                     'nsat: {}'
                                     .format(int(xcoords[i]), int(ycoords[i]),
                                             np.sum(mask_central & mask_region),
                                             nsat))



        if remove_psf:
            if psffit:
                D_sub -= P_shift * flux_psf[i]
            else:
                D_sub -= P_shift * flux_opt[i]


        if False:
            if i % 10000 == 0:
                t_loop = time.time()-t_tmp
                log.info ('t_loop = {:.3f}s'.format(t_loop))
                log.info ('t_new = {:.3f}s; fraction: {:.3f}'
                          .format(t_new, t_new/t_loop))
                if psfex_bintable_ref is not None:
                    log.info ('t_ref = {:.3f}s; fraction: {:.3f}'
                              .format(t_ref, t_ref/t_loop))





    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_psfoptflux')


    # return optimal flux, which can refer to a limiting flux
    # or flux of an artificial/fake object that was added
    list2return = [flux_opt, fluxerr_opt]


    # PSF fit arrays to return
    if psffit:
        x_psf = xcoords + xshift_psf
        y_psf = ycoords + yshift_psf
        list2return = [x_psf, xerr_psf, y_psf, yerr_psf,
                       flux_psf, fluxerr_psf, chi2_psf]

    # Moffat/Gauss arrays to return
    if moffat or gauss:
        list2return = [x_moffat, xerr_moffat, y_moffat, yerr_moffat,
                       fwhm_moffat, elong_moffat, chi2_moffat]


    # if specified, add combined flags_mask of central PSF area
    if get_flags_mask_central:
        list2return += [flags_mask_central]


    #list2return = [elem for sublist in list2return for elem in sublist]
    #list2return = list(itertools.chain.from_iterable(list2return))

    return list2return


################################################################################

def pool_func (func, itemlist, *args, nproc=1):

    try:
        # list to record the results
        results = []
        # create pool of workers
        pool = mp_ctx.Pool(nproc)
        # loop items in itemlist
        for item in itemlist:
            args_tmp = [item]
            # add any additional arguments
            for arg in args:
                args_tmp.append(arg)

            # issue function and arguments to the worker pool
            results.append(pool.apply_async(func, args_tmp))

        # close the pool
        pool.close()
        # wait for the worker processes to exit
        pool.join()
        # get the results
        results = [r.get() for r in results]

    except Exception as e:
        log.exception ('exception was raised during [pool.apply_async({})]: '
                       '{}'.format(func, e))

    return results


################################################################################

def pool_func_lists (func, list_of_imagelists, *args, nproc=1):

    try:
        results = []
        pool = mp_ctx.Pool(nproc)
        for nlist, filelist in enumerate(list_of_imagelists):
            args_temp = [filelist]
            for arg in args:
                args_temp.append(arg[nlist])

            results.append(pool.apply_async(func, args_temp))

        pool.close()
        pool.join()
        results = [r.get() for r in results]
        #    log.info ('result from pool.apply_async: {}'.format(results))
        return results

    except Exception as e:
        #log.exception (traceback.format_exc())
        log.exception ('exception was raised during [pool.apply_async({})]: '
                       '{}'.format(func, e))

        raise RuntimeError


################################################################################

def get_apflux (xcoords, ycoords, data, data_mask, fwhm, objmask=None,
                apphot_radii=None, bkg_radii=None, bkg_limfrac=None,
                bkg_std_coords=None, set_zogy=None, tel=None, nthreads=1):


    """function to infer total local-background subtracted aperture
    fluxes (i.e. not per second) in image data [data] given a list of
    pixel coordinates and the apertures definition list or tuple
    [apphot_radii] and background annulus inner and outer radii
    [bkg_radii] - a 2-element list or tuple. If [bkg_radii] is None,
    the fluxes returned are without any local background
    subtraction. It is also possible to just determine the background
    value in the annulus without any fluxes by leaving [apphot_radii]
    undefined.

    The function returns a list of three float arrays:
    (1) the background-subtracted fluxes with shape (ncoords, nradii)
    (2) the corresponding flux error with shape (ncoords, nradii)
    (3) estimate of the local background with shape (ncoords)

    If [apphot_radii] or [bkg_radii] is None, the corresponding arrays
    will contain zeros.

    """


    log.info('executing get_apflux with {} thread(s) ...'.format(nthreads))
    if get_par(set_zogy.timing,tel): t = time.time()


    if apphot_radii is None and bkg_radii is None:
        log.error ('input [apphot_radii] and [bkg_radii] in '
                   'function zogy.get_apflux are both None; returning None')
        return None


    # split input coordinates into nthreads roughly-equal lists
    ncoords = len(xcoords)
    indices = list(np.linspace(0, ncoords, num=nthreads+1).astype(int))
    # indices_start_stop is a list of lists with the start and stop
    # indices for each of the threads to be run
    indices_start_stop = [indices[i:i+2] for i in range(len(indices)-1)]
    log.info ('get_apflux indices_start_stop: {}'.format(indices_start_stop))


    # read zoom factor from settings file
    fzoom = get_par(set_zogy.apphot_fzoom,tel)


    # if input objmask is not defined, replace it with null array with
    # same shape as input data
    if objmask is None:
        objmask = np.zeros_like(data, dtype=bool)


    # scale radii with fwhm
    if apphot_radii is not None:
        apphot_radii = np.array(apphot_radii) * fwhm


    if bkg_radii is not None:
        # bkg_radii could be list of strings, so convert to numpy
        # float array
        bkg_radii = np.array(bkg_radii) * fwhm


    # list of parareters to provide to [get_apflux_loop]
    pars = [xcoords, ycoords, data, data_mask,
            objmask, apphot_radii, bkg_radii, bkg_limfrac, fzoom,
            bkg_std_coords]


    # feed these lists to pool_func
    if nthreads == 1:
        # do not bother to multiprocess
        list2return = get_apflux_loop (indices_start_stop[0], *pars)
    else:
        # multiprocess
        results = pool_func (get_apflux_loop, indices_start_stop, *pars,
                             nproc=nthreads)
        # results is a list of output arrays; need to concatenate
        # these into single arrays, treating flux_ap, fluxerr_ap and
        # flux_bkg separately because of their different shapes
        flux_ap = np.concatenate(list(zip(*results))[0], axis=1)
        fluxerr_ap = np.concatenate(list(zip(*results))[1], axis=1)
        flux_bkg = np.concatenate(list(zip(*results))[2], axis=0)
        list2return = [flux_ap, fluxerr_ap, flux_bkg]


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_apflux')


    return list2return


################################################################################

def get_apflux_loop (index_start_stop, xcoords, ycoords, data, data_mask,
                     objmask, apphot_radii, bkg_radii, bkg_limfrac, fzoom,
                     bkg_std_coords, verbose=False):


    # limit coordinates to index_start_stop range
    i1, i2 = index_start_stop
    xcoords = xcoords[i1:i2]
    ycoords = ycoords[i1:i2]
    ncoords = len(xcoords)


    # initialize output arrays
    if apphot_radii is not None:
        nrad = len(apphot_radii)
    else:
        nrad = 1

    flux_ap = np.zeros((nrad, ncoords)).astype('float32')
    fluxerr_ap = np.zeros((nrad, ncoords)).astype('float32')
    flux_bkg = np.zeros(ncoords).astype('float32')


    # number of pixels in apertures and background annulus in ideal
    # case (object not near the center and without any masked pixels)
    if apphot_radii is not None:
        npix_ap_max = np.pi * apphot_radii**2
        if verbose:
            log.info ('npix_ap_max: {}'.format(npix_ap_max))

    if bkg_radii is not None:
        npix_bkg_max = np.pi * (bkg_radii[1]**2 - bkg_radii[0]**2)
        if verbose:
            log.info ('npix_bkg_max: {}'.format(npix_bkg_max))


    # loop over coordinates
    for i in range(ncoords):

        # initialize parameters used in flux error estimate below;
        # these will in principle be updated in the background annulus
        # determination, but not if [bkg_radii] are not provided or if
        # too many background pixels are masked
        k_stat = 0
        if bkg_std_coords is not None:
            bkg_std = bkg_std_coords[i]
        else:
            bkg_std = 0


        # first determine background flux
        if bkg_radii is not None:

            # size of region to consider around each object; needs to be even
            # for use in [get_index_around_xy]
            size = int(2 * np.ceil(bkg_radii[1] + 1))
            if verbose:
                log.info ('bkg size: {}'.format(size))


            # use function [get_sect_dist] to extract region around
            # coordinates and also infer the 2D distance array to
            # the coordinates
            data_sect, data_mask_sect, objmask_sect, dist, \
                yc_sect, xc_sect = get_sect_dist (data, data_mask, objmask,
                                                  xcoords[i], ycoords[i],
                                                  size, fzoom=1)


            # select valid background pixels, so including pixels in
            # [mask_bkg] that are not masked by bad pixels or objects
            # (which correspond to 1 or True in objmask)
            # N.B.: the background pixels used are the full pixels, i.e.
            # there is not subsampling done
            mask_bkg = ((dist >= bkg_radii[0]) &
                        (dist <= bkg_radii[1]) &
                        (data_mask_sect == 0) &
                        ~objmask_sect)
            npix_bkg_valid = np.sum(mask_bkg)
            if verbose:
                log.info ('npix_bkg_valid: {}'.format(npix_bkg_valid))


            # determine background if at least [bkg_limfrac] of the
            # pixels are valid; otherwise leave it at zero
            if (npix_bkg_valid / npix_bkg_max) >= bkg_limfrac:
                # determine the clipped median (or mean?)
                bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(
                    data_sect[mask_bkg])
                flux_bkg[i] = bkg_median
                # see fluxerr discussion below on k_stat
                k_stat = np.pi/2
                if verbose:
                    log.info ('mean: {}, median: {}, std: {}'
                              .format(mean, median, std))
                    log.info ('flux_bkg: {:.3f}, bkg_std: {:.3f}'
                              .format(flux_bkg[i], bkg_std))



        # do something similar for the apertures
        if apphot_radii is not None:

            # size of aperture region to consider around each object;
            # needs to be even for use in [get_index_around_xy]
            size = int(2 * np.ceil(apphot_radii[-1] + 1))
            if verbose:
                log.info ('ap size: {}'.format(size))


            # use function [get_sect_dist] to extract region around
            # coordinates and also infer the 2D distance array to
            # the coordinates
            data_sect, data_mask_sect, objmask_sect, dist, \
                yc_sect, xc_sect = get_sect_dist (data, data_mask, objmask,
                                                  xcoords[i], ycoords[i],
                                                  size, fzoom=fzoom)


            # increase the resolution of the image section so
            # that each pixel is split into fzoom x fzoom pixels
            data_sect_fine = blow_up (data_sect, fzoom)
            data_mask_sect_fine = blow_up (data_mask_sect, fzoom)


            # size of subpixels, i.e. size ratio between subsampled
            # and original pixels
            pixelsize = 1/fzoom**2


            # loop apphot_radii
            for i_rad, radius in enumerate(apphot_radii):


                # identify unmasked (sub)pixels within aperture radius
                mask_ap = ((dist <= radius) & (data_mask_sect_fine == 0))
                # number of valid (full) pixels
                npix_ap_valid = np.sum(mask_ap) * pixelsize
                if verbose:
                    log.info ('npix_ap_valid: {}'.format(npix_ap_valid))


                # sum of flux of unmasked pixels within radius,
                # accounting for the fine grid
                flux_ap[i_rad,i] = np.sum(data_sect_fine[mask_ap]) * pixelsize


                # subtract background if available
                if flux_bkg[i] != 0 and npix_bkg_valid != 0:
                    # only for the valid aperture pixels!
                    flux_ap[i_rad,i] -= flux_bkg[i] * npix_ap_valid
                    # pixel ratio aperture and background for fluxerr
                    npix_ratio_apbkg = npix_ap_valid / npix_bkg_valid
                else:
                    npix_ratio_apbkg = 0


                # calculate the flux error from the data variance
                # array, accounting for the error in background flux
                # determination; see also Masci
                # see https://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
                # or /Users/pmv/BlackGEM/PipeLine/Documents/ApPhotUncert.pdf
                # (see https://web.ipac.caltech.edu/staff/fmasci/home/mystats/ApPhotUncert_corr.pdf
                # or /Users/pmv/BlackGEM/PipeLine/Documents/ApPhotUncert_corr.pdf
                # for an elaborate estimate that includes pixel-to-pixel
                # correlations)
                # error = sqrt (flux_ap + area_ap*sky + area_ap**2 * var_bkg/area_bkg)
                #       = sqrt (flux_ap + area_ap*sky * (1 + area_ap/area_bkg))
                # since var_bkg ~ sky; this was used also in IRAF's apphot
                #       = sqrt (sum_ap (var_i) + area_ap**2 * var_bkg/area_bkg)
                # where var_i indicates pixel in variance image which
                # includes read-out, background and Poisson noise
                #fluxerr_ap[i_rad,i] = np.sqrt(
                #    np.sum(data_var_sect_fine[mask_ap]) * pixelsize +
                #    std_bkg[i]**2 * npix_ap_valid * npix_ratio_apbkg)

                # can also avoid using data_var and include "k" factor from Masci:
                # var = flux_ap + (Nap + k*Nap**2/Nbkg) * var_bkg
                fact_tmp = npix_ap_valid * (1 + k_stat * npix_ratio_apbkg)
                fluxerr_ap[i_rad,i] = np.sqrt(np.maximum(flux_ap[i_rad,i],0) +
                                              fact_tmp * bkg_std**2)

                if verbose:
                    log.info ('i: {}, i_rad: {}, flux_ap: {}, fluxerr_ap: {}'
                              .format(i, i_rad, flux_ap[i_rad,i],
                                      fluxerr_ap[i_rad,i]))


                if False:

                    fig,ax = plt.subplots(1)
                    ax.imshow(data_sect, origin='lower')
                    circ = Circle((xc_sect-1,yc_sect-1), radius, fill=False)
                    ax.add_patch(circ)
                    plt.show()

                    data_plt = np.copy(data_sect_fine)
                    data_plt[~mask_ap] = 0
                    plt.imshow(data_plt, origin='lower')
                    plt.show()



    return [flux_ap, fluxerr_ap, flux_bkg]


################################################################################

def blow_up (data_sect, fzoom):

    ysize_sect, xsize_sect = data_sect.shape
    return np.repeat(data_sect,fzoom**2).reshape(-1,xsize_sect,fzoom,fzoom) \
                                        .swapaxes(1,2).reshape(ysize_sect*fzoom,
                                                               xsize_sect*fzoom)


################################################################################

def get_sect_dist (data, data_mask, objmask, xcoord, ycoord, size, fzoom=1):

    # get image size from data
    ysize, xsize = data.shape


    # define the relevant indices to extract from data and other
    # arrays in [array_list], based on size
    index_sect, __, yc_sect, xc_sect = get_index_around_xy (
        ysize, xsize, ycoord, xcoord, size)


    # extract relevant section
    data_sect = data[index_sect]
    data_mask_sect = data_mask[index_sect]
    objmask_sect = objmask[index_sect]
    #if objmask is not None:
    #    objmask_sect = objmask[index_sect]
    #else:
    #    # create null array
    #    objmask_sect = np.zeros_like(data_sect, dtype=bool)


    # from the extracted section, create an array indicating
    # the distance to ycoord, xcoord
    ysize_sect, xsize_sect = data_sect.shape
    x = np.arange(0.5+0.5/fzoom, xsize_sect+0.5, 1/fzoom)
    y = np.arange(0.5+0.5/fzoom, ysize_sect+0.5, 1/fzoom)
    # 'xy' indexing, so 1st dimension of xx, yy and dist
    # indicate the y coordinate
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx-xc_sect)**2+(yy-yc_sect)**2)


    return data_sect, data_mask_sect, objmask_sect, dist, yc_sect, xc_sect


################################################################################

def get_psfoptflux_mp (psfex_bintable, D, bkg_var, D_mask, xcoords, ycoords,
                       psffit=False, moffat=False, gauss=False, get_limflux=False,
                       limflux_nsigma=5., psfex_bintable_ref=None,
                       data_new_bkg_std=None, data_ref_bkg_std=None,
                       header_new=None, header_ref=None, header_trans=None,
                       imtype=None, Scorr_peak=None, inject_fake=False,
                       nsigma_fake=10, remove_psf=False,
                       replace_sat_psf=False, replace_sat_nmax=100,
                       set_zogy=None, tel=None, fwhm=None, diff=True,
                       get_flags_mask_central=False, local_bkg=None,
                       get_psf_footprint=False, nthreads=1):



    # when creating mask of objects' psf central footprints, force
    # nthreads=1
    if get_psf_footprint:
        nthreads = 1


    log.info('executing get_psfoptflux_mp with {} thread(s) ...'
             .format(nthreads))
    if get_par(set_zogy.timing,tel): t = time.time()


    # get dimensions of D
    ysize, xsize = np.shape(D)

    # read in PSF output binary table from psfex, containing the
    # polynomial coefficient images, and various PSF parameters using
    # the function [extract_psf_datapars]
    verbose = get_par(set_zogy.verbose,tel)
    results = extract_psf_datapars (psfex_bintable, verbose=verbose)
    (data_psf, header_psf, psf_fwhm, psf_samp, psf_size_config, psf_chi2,
     psf_nstars, polzero1, polscal1, polzero2, polscal2, poldeg) = results

    # same for reference image
    if psfex_bintable_ref is not None:

        results_ref = extract_psf_datapars (psfex_bintable_ref, verbose=verbose)
        (data_psf_ref, header_psf_ref, psf_fwhm_ref, psf_samp_ref,
         psf_size_config_ref, psf_chi2_ref, psf_nstars_ref, polzero1_ref,
         polscal1_ref, polzero2_ref, polscal2_ref, poldeg_ref) = results_ref

    else:

        (data_psf_ref, header_psf_ref, psf_fwhm_ref, psf_samp_ref,
         psf_size_config_ref, psf_chi2_ref, psf_nstars_ref, polzero1_ref,
         polscal1_ref, polzero2_ref, polscal2_ref, poldeg_ref) = [None] * 12


    # [psf_size] is the PSF size in image pixels, which determines the
    # size of [psf_ima] and [psf_ima_ref] below. For the photometry
    # measurements, [psf_size] should be defined by
    # 2*psf_rad_phot*fwhm_new or fwhm_ref; for the PSF fit to the
    # difference image, in which case the input imtype will be None,
    # the psf_size should be defined using the maximum of fwhm_new and
    # fwhm_ref.
    if imtype=='new':
        if fwhm is not None:
            fwhm_use = fwhm
        else:
            fwhm_use = fwhm_new

        # initial fwhm used for flux limits and Moffat/Gauss fits below
        fwhm_fit_init = fwhm_use

    elif imtype=='ref':
        if fwhm is not None:
            fwhm_use = fwhm
        else:
            fwhm_use = fwhm_ref

        fwhm_fit_init = fwhm_use

    elif imtype is None:
        fwhm_use = max(fwhm_new, fwhm_ref*(pixscale_ref/pixscale_new))
        fwhm_fit_init = (fwhm_new + fwhm_ref) / 2.


    psf_size = 2 * get_par(set_zogy.psf_rad_phot,tel) * fwhm_use
    # make sure [psf_size] is not larger than [psf_size_config] *
    # [psf_samp], which will happen if FWHM is very large
    psf_size = int(min(psf_size, psf_size_config * psf_samp))
    # it should also not be larger than the vignet size; for the PSF
    # fits to the D image, that would lead to an exception because the
    # psf_imas used to build the P_D below would not have the same
    # size
    size_vignet = get_par(set_zogy.size_vignet,tel)
    psf_size = int(min(psf_size, size_vignet))

    # force it to be odd
    if psf_size % 2 == 0:
        psf_size -= 1


    log.info ('psf_size used in [get_psfoptflux_mp]: {} pix for imtype: {}'
              .format(psf_size, imtype))


    # split input coordinates into nthreads roughly-equal lists
    ncoords = len(xcoords)
    indices = list(np.linspace(0, ncoords, num=nthreads+1).astype(int))
    # indices_start_stop is a list of lists with the start and stop
    # indices for each of the threads to be run
    indices_start_stop = [indices[i:i+2] for i in range(len(indices)-1)]
    log.info ('get_psfoptflux_mp indices_start_stop: {}'
              .format(indices_start_stop))


    # set_zogy parameters that were previously inferred in
    # [get_psfoptflux_loop], but which makes the function unpickable
    # because of the reference to the set_zogy module, leading to the
    # multiprocessing breaking down
    psf_clean_factor = get_par(set_zogy.psf_clean_factor,tel)
    source_minpixfrac = get_par(set_zogy.source_minpixfrac,tel)
    mask_value = get_par(set_zogy.mask_value,tel)


    # list of parareters to provide to [get_psfoptflux_loop]
    pars = [xcoords, ycoords, data_psf, psf_size, psf_samp,
            polzero1, polscal1, polzero2, polscal2, poldeg, imtype, xsize, ysize,
            D, D_mask, bkg_var, data_psf_ref, psf_samp_ref,
            polzero1_ref, polscal1_ref,
            polzero2_ref, polscal2_ref, poldeg_ref,
            data_new_bkg_std, data_ref_bkg_std,
            header_new, header_ref, header_trans,
            psffit, moffat, gauss,
            get_limflux, limflux_nsigma, fwhm_fit_init,
            inject_fake, nsigma_fake,
            replace_sat_psf, replace_sat_nmax, remove_psf,
            fwhm_use, diff, get_flags_mask_central,
            psf_clean_factor, source_minpixfrac, mask_value,
            local_bkg, get_psf_footprint, tel]


    # feed these lists to pool_func
    if nthreads == 1:
        # do not bother to multiprocess
        list2return = get_psfoptflux_loop (indices_start_stop[0], *pars)

    else:
        # multiprocess
        results = pool_func (get_psfoptflux_loop, indices_start_stop, *pars,
                             nproc=nthreads)
        # results is a list of output arrays; need to concatenate these
        # into single arrays
        list2return = np.concatenate(results, axis=1)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_psfoptflux_mp')


    return list2return


################################################################################

def get_psfoptflux_loop (
        index_start_stop, xcoords, ycoords, data_psf, psf_size, psf_samp,
        polzero1, polscal1, polzero2, polscal2, poldeg, imtype, xsize, ysize,
        D, D_mask, bkg_var, data_psf_ref=None,  psf_samp_ref=None,
        polzero1_ref=None, polscal1_ref=None,
        polzero2_ref=None, polscal2_ref=None, poldeg_ref=None,
        data_new_bkg_std=None, data_ref_bkg_std=None,
        header_new=None, header_ref=None, header_trans=None,
        psffit=False, moffat=False, gauss=False,
        get_limflux=False, limflux_nsigma=5, fwhm_fit_init=None,
        inject_fake=False, nsigma_fake=10,
        replace_sat_psf=False, replace_sat_nmax=100, remove_psf=False,
        fwhm_use=None, diff=True, get_flags_mask_central=False,
        psf_clean_factor=None, source_minpixfrac=None, mask_value=None,
        local_bkg=None, get_psf_footprint=False, tel=None):


    #if remove_psf:
    #    fits_tmp = 'test_image.fits'
    #    fits.writeto (fits_tmp, D, overwrite=True)


    # loop indices
    i1, i2 = index_start_stop

    # limit coordinates to i1:i2 range
    xcoords = xcoords[i1:i2]
    ycoords = ycoords[i1:i2]

    # initialize output arrays
    ncoords = len(xcoords)
    flux_opt = np.zeros(ncoords, dtype='float32')
    fluxerr_opt = np.zeros_like(flux_opt)

    if get_flags_mask_central:
        flags_mask_central = np.zeros(ncoords, dtype='uint8')

    if psffit:
        flux_psf = np.zeros_like(flux_opt)
        fluxerr_psf = np.zeros_like(flux_opt)
        xshift_psf = np.zeros_like(flux_opt)
        yshift_psf = np.zeros_like(flux_opt)
        chi2_psf = np.zeros_like(flux_opt)
        xerr_psf = np.zeros_like(flux_opt)
        yerr_psf = np.zeros_like(flux_opt)

    if moffat or gauss:
        x_moffat = np.zeros(ncoords)
        xerr_moffat = np.zeros_like(flux_opt)
        y_moffat = np.zeros(ncoords)
        yerr_moffat = np.zeros_like(flux_opt)
        fwhm_moffat = np.zeros_like(flux_opt)
        elong_moffat = np.zeros_like(flux_opt)
        chi2_moffat = np.zeros_like(flux_opt)

    if inject_fake and np.isscalar(nsigma_fake):
        # convert to array
        nsigma_fake = (nsigma_fake * np.ones(ncoords)).astype('float32')


    if get_psf_footprint:
        mask_psf_footprint = np.zeros_like(D, dtype=bool)


    # bad pixel mask values for inside loop
    value_sat = mask_value['saturated']
    value_satcon = mask_value['saturated-connected']


    # if reference image PSF is involved, infer ref image pixel
    # coordinates corresponding to the new image xcoords, ycoords
    if data_psf_ref is not None:

        # convert xycoords to ra, dec in the new image
        wcs_new = WCS(header_new)
        ra_new, dec_new = wcs_new.all_pix2world(xcoords, ycoords, 1)

        # convert these to pixel positions in the ref image
        wcs_ref = WCS(header_ref)
        xcoords_ref, ycoords_ref = wcs_ref.all_world2pix(ra_new, dec_new, 1)



    # start loop over index_list; this is practically the same loop as
    # in the original [get_psfoptflux], except that parameter
    # psfex_bintable_ref is replaced by data_psf_ref
    for i in range(ncoords):

        t_tmp = time.time()

        # using function [get_psf_ima], construct the PSF image with
        # shape (psf_size, psf_size) at xcoords[i], ycoords[i]; this
        # image is at the original pixel scale
        #psf_clean_factor = get_par(set_zogy.psf_clean_factor,tel)
        psf_ima, __ = get_psf_ima (
            data_psf, xcoords[i], ycoords[i], psf_size,
            psf_samp, polzero1, polscal1, polzero2, polscal2, poldeg,
            imtype=imtype, psf_clean_factor=psf_clean_factor, tel=tel)


        if i % 1000 == 0:
            t_new = time.time()-t_tmp


        # determine indices of PSF square footprint (with size:
        # [psf_size]) in an image with shape (ysize, xsize) at pixel
        # coordinates xcoords[i], ycoords[i]; if the footprint is
        # partially off the image, index_P is needed to define the
        # subset of pixels in the PSF footprint that are on the full
        # image. After remapping of the ref image, these indices also
        # apply to the ref image.
        index, index_P = get_P_indices (
            xcoords[i], ycoords[i], xsize, ysize, psf_size)


        # if coordinates off the image, the function [get_P_indices]
        # returns None and the rest can be skipped; continue with next
        # source
        if index is None:
            log.warning ('index return from [get_P_indices] is None for object '
                         'at x,y: {:.2f},{:.2f}; returning zero flux and fluxerr'
                         .format(xcoords[i], ycoords[i]))
            continue


        # if [data_psf_ref] is provided, the new image [psf_ima]
        # will be convolved with the ref image [psf_ima_ref], and
        # replaced with the resulting combined PSF image
        if data_psf_ref is not None:

            # same for reference image; includes rotation to the
            # orientation of the new image
            psf_ima_ref, __ = get_psf_ima (
                data_psf_ref, xcoords_ref[i], ycoords_ref[i], psf_size,
                psf_samp_ref, polzero1_ref, polscal1_ref, polzero2_ref,
                polscal2_ref, poldeg_ref, imtype='ref',
                psf_clean_factor=psf_clean_factor,
                remap=True, header=header_ref, header_new=header_new, tel=tel)


            # setting image standard deviations and flux ratios to
            # unity; if values present in headers, use those instead
            sn, sr, fn, fr = 1, 1, 1, 1

            if data_new_bkg_std is not None and data_ref_bkg_std is not None:
                sn = np.median(data_new_bkg_std[index])
                sr = np.median(data_ref_bkg_std[index])
            elif header_new is not None:
                if 'S-BKGSTD' in header_new:
                    sn = header_new['S-BKGSTD']
                if 'S-BKGSTD' in header_ref:
                    sr = header_new['S-BKGSTD']

            if header_trans is not None:
                if 'Z-FNR' in header_trans:
                    fr = fn / header_new['Z-FNR']


            # could save some time in rest of the loop below by saving
            # P_D for each subimage when it is created in [run_zogy],
            # and reading them in this function; they would need to be
            # fftshifted back and extracted at the center as they are
            # defined for the entire subimage. Downside: single PSF is
            # used for the entire subimage - not so unreasonable.

            # now combine [psf_ima] and [psf_ima_ref]
            # into single psf image using FFT, first performing FFT shift
            Pn_fftshift = fft.fftshift(psf_ima)
            Pr_fftshift = fft.fftshift(psf_ima_ref)


            #if False:
            #    Pn = Pn_fftshift.astype('complex64')
            #    Pn_hat = np.zeros_like(Pn)
            #    fft_forward = pyfftw.FFTW(Pn, Pn_hat, axes=(0,1),
            #                              direction='FFTW_FORWARD',
            #                              flags=('FFTW_ESTIMATE', ),
            #                              threads=1, planning_timelimit=None)
            #    fft_forward()


            Pn_hat = fft.fft2(Pn_fftshift, threads=1)
            Pr_hat = fft.fft2(Pr_fftshift, threads=1)
            Pn_hat2_abs = np.abs(Pn_hat**2)
            Pr_hat2_abs = np.abs(Pr_hat**2)

            # following definitions as in zogy core function
            sn2 = sn**2
            sr2 = sr**2
            fn2 = fn**2
            fr2 = fr**2
            fD = (fr*fn) / np.sqrt(sn2*fr2+sr2*fn2)
            denominator = (sn2*fr2)*Pr_hat2_abs + (sr2*fn2)*Pn_hat2_abs
            P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)

            # multiply, inverse FFT and shift
            P_D = fft.ifftshift(np.real(fft.ifft2(P_D_hat, threads=1)))
            # needs shift of 1,1 pixel because above operations
            # are on a odd-sized image
            psf_ima = ndimage.shift(P_D, (1,1), order=2)
            # normalize
            psf_ima /= np.sum(psf_ima)

            # this provides the same result (apart from the sn, sr,
            # fn, fr ratios), but is ~10 times slower:
            #psf_ima_config = ndimage.convolve(psf_ima_config, psf_ima_config_ref)
            #psf_ima_config /= np.sum(psf_ima_config)



        # extract index_P
        P_shift = psf_ima[index_P]

        # extract subsection from D, D_mask
        D_sub = D[index]
        D_mask_sub = D_mask[index]


        # if bkg_var is a scalar, convert to an image
        if np.isscalar(bkg_var):
            bkg_var_sub = bkg_var * np.ones(D_sub.shape)

        # if bkg_var is a 2D array with the same shape as the input
        # data image (D), extract the subimage part
        elif np.shape(bkg_var) == np.shape(D):
            bkg_var_sub = bkg_var[index]

        # if input background variance is a string
        # 'calc_from_data', calculate it from the input subimage
        # (can be used when background standard deviation was not
        # calculated already, e.g. for the zogy difference (D)
        # image when transients are being fit using this function
        elif bkg_var == 'calc_from_data':
            D_sub_masked = np.ma.masked_array(D_sub, mask=D_mask_sub)
            D_sub_mean, D_sub_median, D_sub_std = sigma_clipped_stats (
                D_sub_masked.astype(float), mask_value=0)
            bkg_var_sub = D_sub_std**2

        else:
            # if none of the above, write error message to log
            log.error('input parameter {} with value {} in function {} '
                      'not understood'
                      .format('bkg_var', bkg_var, 'get_psfoptflux'))



        # define central mask of object using the PSF
        mask_central = (P_shift >= 0.01 * np.amax(P_shift))

        # if flags_mask_central is specified, calculate the combined
        # flags_mask of the central object area
        if get_flags_mask_central:
            flags_mask_central[i] = np.sum(np.unique(D_mask_sub[mask_central]))



        # for saturated stars, increase mask_central; on second
        # thought, don't bother with this, because PSF fits to the
        # saturated stars appear pretty poorly, with very negative
        # values
        #if flag_mask_central & value_sat == value_sat:
        #    mask_central = (P_shift != 0)


        if get_psf_footprint:
            mask_footprint = (P_shift >= 0.001 * np.amax(P_shift))
            mask_psf_footprint[index] += mask_footprint


        # determine optimal or psf or limiting flux
        elif get_limflux:
            # determine limiting flux at this position using
            # flux_optimal_s2n; if Poisson noise of objects should be
            # taken into account, then add background-subtracted image
            # to the background variance: bkg_var_sub + D_sub
            flux_opt[i] = flux_optimal_s2n (P_shift, bkg_var_sub,
                                            limflux_nsigma, fwhm=fwhm_fit_init)

        elif inject_fake:

            # add star to input image with following flux; here the
            # Poisson noise of objects is added to the background
            # variance
            flux_opt[i] = flux_optimal_s2n (P_shift, bkg_var_sub + D_sub,
                                            nsigma_fake[i], fwhm=fwhm_fit_init)
            D_sub += flux_opt[i] * psf_ima


        else:

            # use only those pixels that are not affected by any bad
            # pixels, cosmic rays, saturation, edge pixels, etc.
            mask_use = (D_mask_sub==0)

            if mask_use.shape != mask_central.shape:
                log.warning ('mask_use.shape: {} not equal to '
                             'mask_central.shape: {} for object at x,y: '
                             '{:.2f},{:.2f}; returning zero flux and fluxerr'
                             .format(mask_use.shape, mask_central.shape,
                                     xcoords[i], ycoords[i]))
                continue

            else:
                if np.sum(mask_central) != 0:
                    frac_tmp = (np.sum(mask_use & mask_central) /
                                np.sum(mask_central))
                else:
                    frac_tmp = 0

                # if the fraction of good pixels around the source is less
                # than set_zogy.source_minpixfrac, then continue with next
                # source; optimal flux will be zero and source will not be
                # included in output catalog - may result in very
                # saturated stars not ending up in output catalog
                #if frac_tmp < get_par(set_zogy.source_minpixfrac,tel):
                if frac_tmp < source_minpixfrac:
                    # do not warn about objects near the CCD edge
                    dpix_edge = 5
                    if (dpix_edge < xcoords[i] < xsize-dpix_edge and
                        dpix_edge < ycoords[i] < ysize-dpix_edge):
                        log.warning ('fraction of useable pixels around source '
                                     'at x,y: {:.0f},{:.0f}: {} is less than limit '
                                     'set by set_zogy.source_minpixfrac: {}; '
                                     'returning zero flux and flux error'
                                     .format(xcoords[i], ycoords[i], frac_tmp,
                                             source_minpixfrac))
                    continue


            # perform optimal photometry measurements
            try:

                show=False

                if False:

                    # show optimal fit of particular object(s)
                    dist_tmp = np.sqrt((xcoords[i]-2500)**2 +
                                       (ycoords[i]-9450)**2)
                    if dist_tmp < 50:
                        show=True
                        log.info ('xcoords[i]: {}, ycoords[i]: {}'
                                  .format(xcoords[i], ycoords[i]))

                # pass input parameter [local_bkg] for this object to
                # [flux_optimal] if it is defined
                if local_bkg is not None:
                    sky_bkg = local_bkg[i]
                else:
                    sky_bkg = 0


                flux_opt[i], fluxerr_opt[i] = flux_optimal (
                    P_shift, D_sub, bkg_var_sub, mask_use=mask_use,
                    fwhm=fwhm_use, sky_bkg=sky_bkg, show=show, tel=tel)


            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('problem running [flux_optimal] on object at pixel '
                              'coordinates: x={}, y={}; returning zero flux '
                              'and fluxerr'.format(xcoords[i], ycoords[i]))

                continue


            # infer error image used in psffit and gauss/moffat fits below
            D_sub_err = np.sqrt(np.abs(bkg_var_sub))


            # if psffit=True, perform PSF fitting
            if psffit:

                if i==0:
                    show=True
                else:
                    show=False

                try:
                    xshift_psf[i], xerr_psf[i], yshift_psf[i], yerr_psf[i], \
                        flux_psf[i], fluxerr_psf[i], chi2_psf[i] = (
                            flux_psffit (P_shift, D_sub, D_sub_err, flux_opt[i],
                                         mask_use=mask_use, fwhm=fwhm_fit_init,
                                         show=show, max_nfev=200, diff=diff))

                except Exception as e:
                    #log.exception(traceback.format_exc())
                    log.exception('problem running [flux_psffit] on object at '
                                  'pixel coordinates: {}, {}; returning zero '
                                  'flux, fluxerr, (x,y) shifts and chi2'
                                  .format(xcoords[i], ycoords[i]))
                    continue


            # if moffat=True, perform Moffat fit
            if moffat or gauss:

                if i==0:
                    show=True
                else:
                    show=False

                try:

                    # fit 2D gauss
                    x_moffat[i], xerr_moffat[i], y_moffat[i], yerr_moffat[i], \
                        fwhm_moffat[i], elong_moffat[i], chi2_moffat[i] = \
                            fit_moffat_single (
                                D_sub, D_sub_err, mask_use=mask_use,
                                fit_gauss=gauss, fwhm=fwhm_fit_init,
                                P_shift=P_shift, show=show, max_nfev=200)

                    x_moffat[i] += index[1].start
                    y_moffat[i] += index[0].start

                except Exception as e:
                    #log.exception(traceback.format_exc())
                    log.exception('problem running [fit_moffat_single] on object '
                                  'at pixel coordinates: {}, {}; returning zeros'
                                  .format(xcoords[i], ycoords[i]))
                    continue


            # replace saturated+connected pixels with PSF profile
            if replace_sat_psf:

                if False:
                    if np.sqrt((xcoords[i]-2500)**2 +
                               (ycoords[i]-9450)**2) < 100:
                        log.info('xcoords[i]: {}, ycoords[i]: {}'
                                 .format(xcoords[i], ycoords[i]))

                # determine pixels to be replaced; avoid including
                # saturated pixels from other objects within the
                # footprint

                # mask of saturated+connected pixels in subimage:
                mask_sat_sub = (D_mask_sub & value_sat == value_sat)
                mask_satcon_sub = (mask_sat_sub |
                                   (D_mask_sub & value_satcon == value_satcon))

                # label saturated+connected pixels; also
                # diagonally-connected pixels are included in a region
                struct = np.ones((3,3), dtype=bool)
                regions, nregions = ndimage.label(mask_satcon_sub,
                                                  structure=struct)


                # loop regions
                for ir in range(nregions):

                    mask_region = (regions==ir+1)
                    nsat = np.sum(mask_sat_sub & mask_region)

                    if np.sqrt((xcoords[i]-2500)**2 +
                               (ycoords[i]-9450)**2) < 100:
                        log.info('xcoords[i]: {}, ycoords[i]: {}'
                                 .format(xcoords[i], ycoords[i]))
                        log.info('nsat: {}, np.sum(mask_central & mask_region): {}'
                                 .format(nsat, np.sum(mask_central & mask_region)))

                    # check if there is overlap with central PSF region
                    if (np.sum(mask_central & mask_region) > 0 and
                        nsat > 0 and nsat <= replace_sat_nmax):

                        mask_satcon_use = (mask_satcon_sub & mask_region)

                        # replace these by the estimate of flux_opt *
                        # PSF + sky background
                        D_sub[mask_satcon_use] = (
                            P_shift[mask_satcon_use] * flux_opt[i] + sky_bkg)


                        if False:
                            D_sub_orig = np.copy(D_sub)
                            if np.sqrt((xcoords[i]-2500)**2 +
                                       (ycoords[i]-9450)**2) < 100:
                                D_sub_psf = P_shift * flux_opt[i] + sky_bkg
                                ds9_arrays (D_sub_orig=D_sub_orig, D_sub=D_sub,
                                            D_mask_sub=D_mask_sub.astype(int))


                    else:
                        log.warning ('saturated+connected pixels not replaced '
                                     'for object at x,y={},{}; '
                                     'np.sum(mask_central & mask_region): {}, '
                                     'nsat: {}'
                                     .format(int(xcoords[i]), int(ycoords[i]),
                                             np.sum(mask_central & mask_region),
                                             nsat))



        if remove_psf:
            if psffit:
                D_sub -= P_shift * flux_psf[i]
            else:
                D_sub -= P_shift * flux_opt[i]


        if False:
            if i % 1000 == 0:
                t_loop = time.time()-t_tmp
                log.info ('t_loop = {:.3f}s'.format(t_loop))
                log.info ('t_new = {:.3f}s; fraction: {:.3f}'
                          .format(t_new, t_new/t_loop))
                if data_psf_ref is not None:
                    log.info ('t_ref = {:.3f}s; fraction: {:.3f}'
                              .format(t_ref, t_ref/t_loop))



    # CHECK!!! - temporarily record D
    #if remove_psf:
    #    fits_tmp = 'test_image_psf_removed.fits'
    #    fits.writeto (fits_tmp, D, overwrite=True)


    # end of loop: for i in range(i1,i2):

    # return optimal flux, which can refer to a limiting flux
    # or flux of an artificial/fake object that was added
    list2return = [flux_opt, fluxerr_opt]


    # PSF fit arrays to return
    if psffit:
        x_psf = xcoords + xshift_psf
        y_psf = ycoords + yshift_psf
        list2return = [x_psf, xerr_psf, y_psf, yerr_psf,
                       flux_psf, fluxerr_psf, chi2_psf]

    # Moffat/Gauss arrays to return
    if moffat or gauss:
        list2return = [x_moffat, xerr_moffat, y_moffat, yerr_moffat,
                       fwhm_moffat, elong_moffat, chi2_moffat]


    # if specified, add combined flags_mask of central PSF area
    if get_flags_mask_central:
        list2return += [flags_mask_central]


    if get_psf_footprint:
        list2return = [mask_psf_footprint]


    return list2return


################################################################################

def get_psf_ima (data, xcoord, ycoord, psf_size, psf_samp, polzero1,
                 polscal1, polzero2, polscal2, poldeg, xshift=0, yshift=0,
                 imtype=None, remap=False, header=None, header_new=None,
                 psf_clean_factor=0, tel=None):

    """function to infer the PSF image with shape (psfsize, psfsize) at
    the original pixel scale at the input pixel coordinates (xcoord,
    ycoord). [data] is the output fits from PSFEx.

    """

    # infer the relatieve PSFEx coordinates x,y
    x = (int(xcoord) - polzero1) / polscal1
    y = (int(ycoord) - polzero2) / polscal2


    # obtain the PSF image at the PSFEx configuration pixel scale
    psf_ima_config = calc_psf_config (data, poldeg, x, y)


    # if remapping is done and the new and ref image have
    # different orientations, the PSF of the ref image needs
    # to be transformed to that of the new image
    if imtype=='ref' and remap:
        # remap [psf_ima_sub] from the ref frame to the new frame
        # using the function [orient_data]
        psf_ima_config = orient_data (psf_ima_config, header,
                                      header_out=header_new, tel=tel)


    # determine shift to the subpixel center of the object (object
    # at fractional pixel position 0.5,0.5 doesn't need the PSF to
    # shift if the PSF image is constructed to be even)
    # odd case:
    xshift = xcoord-np.round(xcoord)
    yshift = ycoord-np.round(ycoord)


    # shift the PSF image (if needed) and resize/resample at the
    # original pixel scale
    order = 2
    if (xshift==0 and yshift==0):

        # if no shift is required, simply resample [psf_ima_config] at
        # the original pixel scale using ndimage.zoom and
        # [psf_samp]
        psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp,
                                       order=order, mode='nearest')
    else:

        # otherwise, perform the shift on either [psf_ima_config] or
        # the resampled image, depending on the [psf_samp] being lower
        # or higher than 1, respectively

        if psf_samp < 1:

            # determine shift in config pixels
            xshift_config = xshift / psf_samp
            yshift_config = yshift / psf_samp
            # shift PSF
            psf_ima_shift = ndimage.shift(psf_ima_config,
                                          (yshift_config, xshift_config),
                                          order=order)
            # resample shifted PSF image at image pixel scale
            psf_ima_shift_resized = ndimage.zoom(psf_ima_shift, psf_samp,
                                                 order=order, mode='nearest')

        else:
            # resample PSF image at image pixel scale
            psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp,
                                           order=order, mode='nearest')
            # shift PSF
            psf_ima_shift_resized = ndimage.shift(psf_ima_resized,
                                                  (yshift, xshift), order=order)

        # let [psf_ima_resized] refer to [psf_ima_shift_resized]
        psf_ima_resized = psf_ima_shift_resized


    if False:
        psf_ima_resized_temp2 = np.copy(psf_ima_resized)

    # clean, cut and normalize PSF
    psf_ima_resized = clean_cut_norm_psf (psf_ima_resized, psf_clean_factor,
                                          cut_size=psf_size)


    if False:
        if xshift==0 and yshift==0:
            if (xcoord < 2640 and ycoord < 2640):
                ds9_arrays (psf_ima_config=psf_ima_config,
                            psf_ima_resized_temp=psf_ima_resized_temp,
                            psf_ima_resized_temp2=psf_ima_resized_temp2,
                            psf_ima_resized=psf_ima_resized)

        else:
            ds9_arrays (psf_ima_config=psf_ima_config,
                        psf_ima_resized_temp=psf_ima_resized_temp,
                        psf_ima_shift_resized=psf_ima_shift_resized,
                        psf_ima_resized_temp2=psf_ima_resized_temp2,
                        psf_ima_resized=psf_ima_resized)


    return psf_ima_resized, psf_ima_config


################################################################################

def get_P_indices (xcoord, ycoord, xsize, ysize, psf_size):

    # calculate pixel indices in image of size (ysize, xsize) that
    # correspond to the PSF image centered at (ycoord, xcoord), and
    # the corresponding pixel indices in the PSF image itself, which
    # is relevant if the (ycoord, xcoord) is less than psf_size/2 away
    # from the edge of the image.
    psf_hsize = int(psf_size/2)

    # extract data around position to use indices of pixel in which
    # [x],[y] is located in case of odd-sized psf:
    xpos = int(xcoord-0.5)
    ypos = int(ycoord-0.5)

    # check if position is within image
    if ypos<0 or ypos>=ysize or xpos<0 or xpos>=xsize:
        #print ('Position x,y='+str(xpos)+','+str(ypos)+' outside
        #image - skipping') continue
        return None, None

    # if PSF footprint is partially off the image, just go ahead
    # with the pixels on the image
    y1 = max(0, ypos-psf_hsize)
    x1 = max(0, xpos-psf_hsize)
    # assuming psf is oddsized
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

    index = tuple([slice(y1,y2),slice(x1,x2)])

    # extract subsection
    y1_P = y1 - (ypos - psf_hsize)
    x1_P = x1 - (xpos - psf_hsize)
    y2_P = y2 - (ypos - psf_hsize)
    x2_P = x2 - (xpos - psf_hsize)
    index_P = tuple([slice(y1_P,y2_P),slice(x1_P,x2_P)])

    return index, index_P


################################################################################

def get_psf_config (data, xcoord, ycoord, psf_oddsized, ysize, xsize,
                    psf_hsize, polzero1, polscal1, polzero2, polscal2, poldeg):

    # extract data around position to use indices of pixel in which
    # [x],[y] is located in case of odd-sized psf:
    if psf_oddsized:
        xpos = int(xcoord-0.5)
        ypos = int(ycoord-0.5)
    else:
        # in case of even-sized psf:
        xpos = int(xcoord)
        ypos = int(ycoord)

    # check if position is within image
    if ypos<0 or ypos>=ysize or xpos<0 or xpos>=xsize:
        #print ('Position x,y='+str(xpos)+','+str(ypos)+' outside image - skipping')
        #continue
        return None, None, None

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

    # construct PSF at x,y
    x = (int(xcoord) - polzero1) / polscal1
    y = (int(ycoord) - polzero2) / polscal2
    psf_ima_config = calc_psf_config (data, poldeg, x, y)

    # extract subsection
    y1_P = y1 - (ypos - psf_hsize)
    x1_P = x1 - (xpos - psf_hsize)
    y2_P = y2 - (ypos - psf_hsize)
    x2_P = x2 - (xpos - psf_hsize)
    index_P = tuple([slice(y1_P,y2_P),slice(x1_P,x2_P)])

    # index are the indices of the PSF footprint of the full image,
    # while index_P are the indices of the PSF image with size
    # psf_size x psf_size, which is needed in case the PSF footprint
    # is partially off the full image

    return psf_ima_config, index, index_P


################################################################################

def flux_psffit (P, D, D_err, flux_opt, mask_use=None, max_nfev=100, show=False,
                 fwhm=3, diff=True, fit_bkg=True, fit_inner=False):


    # define objective function: returns the array to be minimized
    def fcn2min(params, P, D, D_err, mask_use, diff):

        # convert params to dictionary of values
        par = params.valuesdict()

        # shift the PSF image
        P_shift = ndimage.shift(P, (par['yshift'], par['xshift']))

        # make sure that P_shift is equal to 1
        #P_shift /= np.sum(P_shift)

        # scale the shifted PSF
        model = par['flux_psf'] * P_shift

        # for the difference image (for which this function is used
        # exclusively at the moment), [D_err] already includes the
        # variance of both images, including the transient flux, so
        # adding the model flux would be too much
        var = D_err**2
        if not diff:
            var += np.abs(model) + np.max(par['sky'], 0)

        # sigma
        err = np.sqrt(var)

        # residuals
        mask_nonzero = (err != 0)
        resid = (D - par['sky'] - model)
        resid[mask_nonzero] /= err[mask_nonzero]
        resid[~mask_nonzero] = 0

        # return raveled (flattened) array
        if mask_use is not None:
            return resid[mask_use].ravel()
        else:
            return resid.ravel()



    # create mask with pixel values above 1 percent of the peak of the
    # PSF used to determine the chi-square of this region
    mask_inner = (P >= 0.01 * np.amax(P))


    # estimate background value and STD from pixels outside of inner
    # region
    mask_bkg = mask_use & ~mask_inner
    __, D_bkg, D_bkg_std = sigma_clipped_stats (D[mask_bkg].astype(float))


    # create a set of Parameters
    params = Parameters()
    hwhm = max(fwhm/2, 3)
    params.add('xshift', value=0, min=-hwhm, max=hwhm, vary=True)
    params.add('yshift', value=0, min=-hwhm, max=hwhm, vary=True)
    params.add('sky', value=D_bkg, min=D_bkg-D_bkg_std, max=D_bkg+D_bkg_std,
               vary=fit_bkg)

    # flux
    f0 = 3
    flux_min = -f0 * np.abs(flux_opt)
    flux_max =  f0 * np.abs(flux_opt)
    params.add('flux_psf', value=flux_opt, min=flux_min, max=flux_max, vary=True)


    if fit_inner:
        # only fit inner part of profile
        mask_use &= mask_inner


    # do leastsq model fit using minimize
    result = minimize(fcn2min, params, method='leastsq',
                      args=(P, D, D_err, mask_use, diff), max_nfev=max_nfev)


    xshift = result.params['xshift'].value
    yshift = result.params['yshift'].value
    xerr = result.params['xshift'].stderr
    yerr = result.params['yshift'].stderr
    flux_psf = result.params['flux_psf'].value
    fluxerr_psf = result.params['flux_psf'].stderr
    sky = result.params['sky'].value
    chi2 = result.chisqr
    chi2red = result.redchi
    if np.sum(mask_inner) != 0:
        chi2_inner = np.sum(fcn2min(result.params, P, D, D_err, mask_inner,
                                    diff)**2)
    else:
        chi2_inner = 0


    # in case fit did not succeed, stderrs will be None
    if xerr is None:
        xerr = 0
    if yerr is None:
        yerr = 0
    if fluxerr_psf is None:
        # also set flux and chi2 to zero as it will not be reliable
        fluxerr = 0
        fluxerr_psf = 0
        chi2_inner = 0


    # reduced chi2_inner
    denom = (np.sum(mask_inner) - result.nvarys)
    if denom != 0:
        chi2red_inner = chi2_inner / denom
    else:
        chi2red_inner = chi2_inner


    if show:

        log.info('----------------------------------------')
        log.info('PSF fit')
        log.info('----------------------------------------')
        log.info(fit_report(result))
        log.info('chi2red_inner:{:.3f}'.format(chi2red_inner))

        if False:

            P_shift = ndimage.shift(P, (yshift, xshift))
            model = flux_psf * P_shift
            resid = (D - sky - model) / D_err

            ds9_arrays (P_shift=P_shift, D=D, Dminsky=D-sky, model=model,
                        D_err=D_err, resid=resid, mask_use=mask_use.astype(int),
                        mask_inner=mask_inner.astype(int))


    return xshift, xerr, yshift, yerr, flux_psf, fluxerr_psf, chi2red_inner


################################################################################

def get_optflux (P, D, V):

    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], sky-subtracted data [D] and variance [V].  All are
    assumed to be in electrons rather than counts. These can be 1- or
    2-dimensional arrays with the same shape, while the sky can also
    be a scalar. See Horne 1986, PASP, 98, 609 and Naylor 1998, MNRAS,
    296, 339.

    """

    # avoid zeros in V
    mask_V0 = (V==0)

    optflux = 0
    optfluxerr = 0
    if np.sum(~mask_V0 != 0):
        denominator = np.sum(P[~mask_V0]**2/V[~mask_V0])
        if denominator > 0:
            optflux = np.sum((P[~mask_V0]*D[~mask_V0]/V[~mask_V0])) / denominator
            optfluxerr = 1./np.sqrt(denominator)


    return optflux, optfluxerr


    if False:

        # alternative calculation

        if P.ndim!=1:
            P = P.flatten()
            D = D.flatten()
            V = V.flatten()

        # and optimal flux and its error
        P_over_V = P/V
        denominator = np.dot(P, P_over_V)
        if denominator>0:
            optflux = np.dot(P_over_V, D) / denominator
            optfluxerr = 1./np.sqrt(denominator)
        else:
            optflux = 0
            optfluxerr = 0


################################################################################

def get_optflux_Naylor (P, D, V):

    """Function that calculates signal-to-noise ratio using Eqs. 8, 10 and
    11 from Naylor 1998. D(data) is assumed to have been sky
    subtracted. All inputs are assumed to be in electrons rather than
    counts. These can be 1- or 2-dimensional arrays.

    """

    # weights
    denominator = np.sum(P**2/V)
    W = (P/V) / denominator
    # and optimal flux and its error
    optflux = np.sum(W*D)
    optfluxerr = np.sqrt(np.sum(W**2*V))
    return optflux, optfluxerr


################################################################################

def get_s2n_ZO (P, D, V):

    """Function that calculates signal-to-noise ratio using Eq. 51 from
    Zackay & Ofek 2017, ApJ, 836, 187. D(data) is assumed to have been
    sky subtracted. All inputs are assumed to be in electrons rather
    than counts. These can be 1- or 2-dimensional arrays.

    """

    T0 = np.sum(D)
    #s2n = np.sqrt(np.sum( (D-S)**2 / V ))
    s2n = np.sqrt(np.sum( (T0*P)**2 / V ))

    return s2n


################################################################################

def flux_optimal (P, D, bkg_var, nsigma_inner=np.inf, nsigma_outer=5, max_iters=10,
                  epsilon=0.1, mask_use=None, add_V_ast=False, fwhm=None,
                  sky_bkg=None, dx2=0, dy2=0, dxy=0, show=False, tel=None):


    """Function that calculates optimal flux and corresponding error based
       on the PSF [P], sky-subtracted data [D] and background variance
       [bkg_var].  This makes use of function [get_optflux] or
       [get_optflux_Eran].

    """

    if add_V_ast:
        # calculate astrometric variance
        dDdy = D - np.roll(D,1,axis=0)
        dDdx = D - np.roll(D,1,axis=1)
        dDdxy = D - np.roll(D,1,axis=(0,1))
        V_ast = np.abs(dx2) * dDdx**2 + np.abs(dy2) * dDdy**2 + np.abs(dxy) * dDdxy**2


    if mask_use is None:
        # if input mask [mask_use] was not provided, create it with same
        # shape as D with all elements set to True.
        mask_use = np.ones(D.shape, dtype=bool)
    else:
        # make copy as otherwise [mask_use] is affected also outside
        # this function
        mask_use = np.copy(mask_use)


    # in case input [sky_bkg] is available
    if sky_bkg is not None and sky_bkg != 0:

        D = D - sky_bkg

        # add Poisson noise of sky background level
        if sky_bkg > 0:
            bkg_var = bkg_var + sky_bkg


    # [mask_inner] - the central pixels within about 1xFWHM
    # (assuming a Gaussian shape) of the object center (where P values
    # are higher than 0.01 times the central P value);
    # [mask_outer] is the region outside of that
    P_max = np.amax(P)
    mask_inner = (P >= 0.01*P_max)
    mask_outer = ~mask_inner
    # intialize mask array to keep track of rejected pixels in loop below
    mask_temp = np.ones(D.shape, dtype=bool)


    # loop
    flux_opt_old = float('inf')
    for i in range(max_iters):

        if i==0:
            # initial variance estimate (see Eq. 12 from Horne 1986)
            V = bkg_var + np.abs(D)

        else:
            # improved variance (see Eq. 13 from Horne 1986)
            V = bkg_var + np.abs(flux_opt) * P


        if add_V_ast:
            V += V_ast

        # optimal flux
        flux_opt, fluxerr_opt = get_optflux (P[mask_use],
                                             D[mask_use],
                                             V[mask_use])

        #print ('i, flux_opt, fluxerr_opt', i, flux_opt, fluxerr_opt,
        #       abs(flux_opt_old-flux_opt)/flux_opt,
        # abs(flux_opt_old-flux_opt)/fluxerr_opt)

        if fluxerr_opt==0:
            break

        # stopping criterium suggested by Steven:
        if abs(flux_opt_old-flux_opt)/np.abs(fluxerr_opt) < epsilon:
            break

        flux_opt_old = flux_opt

        # reject any discrepant values; use [nsigma_inner] as the
        # rejection criterium for the inner region defined by
        # [mask_inner]; outside of that use [nsigma_outer]
        sigma2 = (D - flux_opt * P)**2
        mask_pos = (V > 0)
        sigma2[mask_pos] /= V[mask_pos]

        mask_temp[mask_inner] = (sigma2[mask_inner] > nsigma_inner**2)
        mask_temp[mask_outer] = (sigma2[mask_outer] > nsigma_outer**2)
        mask_use[mask_temp] = False


    if show:

        log.info('no. of rejected pixels: {}'.format(np.sum(~mask_use)))
        log.info('np.amax((D - flux_opt * P)**2 / V): {}'.format(np.amax(sigma2)))
        log.info('flux_opt: {:.1f} +- {:.1f}'.format(flux_opt, fluxerr_opt))

        ds9_arrays(data=D, psf=P, bkg_var=bkg_var, variance=V,
                   fluxoptP = flux_opt*P, data_min_fluxoptP=(D - flux_opt * P),
                   data_min_fluxoptP_squared_div_variance=sigma2,
                   mask_use=mask_use.astype(int))


    return flux_opt, fluxerr_opt


################################################################################

def flux_optimal_s2n (P, bkg_var, s2n, fwhm=5., max_iters=15, epsilon=1e-7):

    """Similar to function [flux_optimal] above, but this function returns
    the total flux [in e-] required for the point source to have a
    particular signal-to-noise ratio [s2n], given the PSF image [P]
    and the background variance [bkg_var] (=sky background + RN**2).
    This function is used to estimate the limiting magnitude of an
    image at a set of coordinates (in get_psfoptflux), and
    also to estimate the flux of the fake stars that are being added
    to the image with a required S/N [set_zogy.fakestar_s2n].

    Note that the background-subtracted image itself should be added
    to [bkg_var] in order to calculate the flux required to reach the
    required S/N with respect to the image, i.e. taking into account
    the poisson noise of any object present in the image.

    """

    for i in range(max_iters):

        if i==0:

            # initial estimate of variance (scalar)
            V = bkg_var

            # and flux (see Eq. 13 of Naylor 1998)
            flux_opt = (s2n * fwhm * np.sqrt(np.median(V)) /
                        np.sqrt(2*np.log(2)/np.pi))

        else:

            # estimate new flux based on fluxerr_opt of previous iteration
            flux_opt = s2n * fluxerr_opt

            # improved estimate of variance (2D list)
            V = bkg_var + flux_opt * P


        # new estimate of D
        D = flux_opt * P

        # get optimal flux, avoiding zeros in V
        mask = (V!=0)
        if np.sum(mask) != 0:
            flux_opt, fluxerr_opt = get_optflux (P[mask], D[mask], V[mask])
        else:
            break

        # also break out of loop if S/N sufficiently close
        if fluxerr_opt != 0:
            if abs(flux_opt/fluxerr_opt - s2n) / s2n < epsilon:
                break


    return flux_opt


################################################################################

def clipped_stats(array, nsigma=3, max_iters=10, epsilon=1e-6, clip_upper_frac=0,
                  clip_zeros=True, get_median=True, get_mode=False, mode_binsize=0.1,
                  verbose=False, make_hist=False, name_hist=None, hist_xlabel=None,
                  use_median=False):

    if verbose and get_par(set_zogy.timing,tel):
        log.info('executing clipped_stats ...')
        t = time.time()

    # remove zeros
    if clip_zeros:
        array = array[array!=0]

    if clip_upper_frac != 0:
        index_upper = int((1.-clip_upper_frac)*array.size+0.5)
        array = np.sort(array.flatten(), kind='quicksort')[:index_upper]

    mean_old = float('inf')
    for i in range(max_iters):
        if array.size > 0:
            if not use_median:
                mean = array.mean()
            else:
                mean = np.median(array)
            std = array.std()
            if abs(mean_old-mean)/abs(mean) < epsilon:
                break
            mean_old = mean
            index = ((array>(mean-nsigma*std)) & (array<(mean+nsigma*std)))
            array = array[index]
        else:
            array = np.zeros(1)
            break


    # make sure to calculate mean if median was used in clipping
    if use_median:
        mean = array.mean()

    # add median
    if get_median:
        median = np.median(array)

    # and mode
    if get_mode:
        bins = np.arange(mean-nsigma*std, mean+nsigma*std, mode_binsize)
        hist, bin_edges = np.histogram(array, bins)
        index = np.argmax(hist)
        mode = (bins[index]+bins[index+1])/2.

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
            title = '{}, median (orange line): {:.3f}'.format(title, median)
        if get_mode:
            plt.plot([mode, mode], [y2,y1], color='tab:red')
            title = '{}, mode (red line): {:.3f}'.format(title, mode)
        plt.title(title)
        if hist_xlabel is not None:
            plt.xlabel(hist_xlabel)
        plt.ylabel('number')
        if get_par(set_zogy.make_plots,tel):
            if name_hist is None: name_hist = 'clipped_stats_hist.pdf'
            plt.savefig(name_hist)
        if get_par(set_zogy.show_plots,tel): plt.show()
        plt.close()

    if verbose and get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in clipped_stats')

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

    # list with values to return
    values = []
    # loop keywords
    for key in keywords:
        # use function [get_keyvalue] (see below) to return the value
        # from either the variable defined in settings file, or from
        # the fits header using the keyword name defined in the
        # settings file
        value = get_keyvalue(key, header)
        if key=='filter':
            value = str(value)
        values.append(value)

    if len(values)==1:
        return values[0]
    else:
        return values


################################################################################

def get_keyvalue (key, header):

    # check if [key] is defined in settings file
    var = 'set_zogy.{}'.format(key)
    try:
        value = eval(var)
    except:
        # if it does not work, try using the value of the keyword name
        # (defined in settings file) from the fits header instead
        try:
            key_name = eval('set_zogy.key_{}'.format(key))
        except:
            msg = ('either [{}] or [key_{}] needs to be defined in '
                   '[settings_file]'.format(key, key))
            log.critical(msg)
            raise RuntimeError(msg)
        else:
            if key_name in header:
                value = header[key_name]
            else:
                msg = 'keyword {} not present in header'.format(key_name)
                log.critical(msg)
                raise RuntimeError(msg)

    #log.info('keyword: {}, adopted value: {}'.format(key, value))

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

def get_Xchan_bool (tel, chancorr, imtype, std=False):

    # function to determine whether interpolation is allowed across
    # different channels in [mini2back]

    # for standard deviation
    if std:
        # for bkg_std, do not interpolate for ML/BG images, except for the
        # reference image
        if tel[0:2] in ['ML','BG'] and imtype!='ref':
            interp_Xchan = False
        else:
            interp_Xchan = True

    else:
        # for bkg, allow interpolation for non-ML/BG images or if
        # ML/BG channel correction factors were applied or if this
        # concerns the reference image
        if (tel not in ['ML1', 'BG2', 'BG3', 'BG4', 'BG'] or chancorr or
            imtype=='ref'):
            interp_Xchan = True
        else:
            interp_Xchan = False


    return interp_Xchan


################################################################################

def prep_optimal_subtraction(input_fits, nsubs, imtype, fwhm, header,
                             fits_mask=None, remap=False, nthreads=1):

    log.info('executing prep_optimal_subtraction ...')
    t = time.time()

    if imtype=='new':
        base = base_new
    else:
        base = base_ref

    # read input_fits
    data_wcs = read_hdulist (input_fits, dtype='float32')

    # get gain, pixscale and saturation level from header
    keywords = ['gain', 'pixscale', 'satlevel']
    gain, pixscale, satlevel = read_header(header, keywords)
    ysize, xsize = np.shape(data_wcs)


    # determine data_mask
    if fits_mask is not None:
        # read in mask image
        data_mask = read_hdulist (fits_mask, dtype='uint8')
        # create new mask or modify an existing one
        data_mask = create_modify_mask (data_wcs, satlevel, data_mask=data_mask)

    else:
        data_mask = None
        fits_mask = input_fits.replace('.fits', '_mask.fits')
        # create new mask or modify an existing one
        data_mask = create_modify_mask (data_wcs, satlevel, data_mask=data_mask)

        # as pointed out by PLirat through GitHub (see email from 23 Dec
        # 2021), if [fits_mask] is not provided, then an error occurs
        # below when remapping the reference image mask because the fits
        # file itself is required in [help_swarp] (and also in
        # [run_sextractor] if fake stars are being added), which was not
        # created before; added these lines as suggested by PLirat:
        header_mask = read_hdulist (input_fits, get_data=False, get_header=True)
        fits.writeto (fits_mask, data_mask, header_mask, overwrite=True)


    # check if background was already subtracted from input_fits
    if 'BKG-SUB' in header:
        bkg_sub = header['BKG-SUB']
    else:
        bkg_sub = False

    log.info('background already subtracted from {}?: {}'
             .format(input_fits, bkg_sub))


    # determine whether interpolation is allowed across different
    # channels in [mini2back] using function get_Xchan_bool
    chancorr = get_par(set_zogy.MLBG_chancorr,tel)
    interp_Xchan = get_Xchan_bool (tel, chancorr, imtype)
    interp_Xchan_std = get_Xchan_bool (tel, chancorr, imtype, std=True)


    # read in background std image in any case (i.e. also if
    # background was already subtracted); it is needed for the
    # variance image output
    fits_bkg_std = '{}_bkg_std.fits'.format(base)
    if isfile(fits_bkg_std):
        data_bkg_std = read_hdulist (fits_bkg_std, dtype='float32')
    else:
        # if it does not exist, create it from the background mesh
        fits_bkg_std_mini = '{}_bkg_std_mini.fits'.format(base)
        data_bkg_std_mini, header_mini = read_hdulist (
            fits_bkg_std_mini, get_header=True, dtype='float32')

        if 'BKG-SIZE' in header_mini:
            bkg_size = header_mini['BKG-SIZE']
        else:
            bkg_size = get_par(set_zogy.bkg_boxsize,tel)

        data_bkg_std = mini2back (data_bkg_std_mini, data_wcs.shape,
                                  order_interp=1, bkg_boxsize=bkg_size,
                                  interp_Xchan=interp_Xchan_std,
                                  timing=get_par(set_zogy.timing,tel))
        # and write to fits; needed below for SWarp
        fits.writeto (fits_bkg_std, data_bkg_std, overwrite=True)



    # in case the reference image needs to be remapped, SWarp the
    # reference image (already background subtracted), the background
    # STD and the mask to the frame of the new image
    if remap:

        # for now this is done with SWarp, but this is a rather slow
        # solution. Tried to improve this using functions
        # [xy_index_ref] and [get_data_remap] (see older zogy
        # versions), but these fail when there is rotation between the
        # images, resulting in rotated masks.


        # local function to help with remapping
        def help_swarp (fits2remap, data2remap, header2remap=header,
                        fits2remap2='{}.fits'.format(base_new), value_edge=0,
                        resampling_type='NEAREST', update_header=False):


            # update header of fits2remap image with that of the
            # original wcs-corrected reference image (needed if header
            # does not correspond to that of the reference image,
            # e.g. for the mask)
            if update_header:
                log.info('updating header of {}'.format(fits2remap))
                update_imcathead(fits2remap, header2remap)


            # project fits image to new image
            fits_out = fits2remap.replace('.fits', '_remap.fits')
            fits_out = get_remap_name(fits2remap2, fits2remap, fits_out)
            result = run_remap(fits2remap2, fits2remap, fits_out,
                               get_par(set_zogy.shape_new,tel), gain=gain,
                               config=get_par(set_zogy.swarp_cfg,tel),
                               resampling_type=resampling_type,
                               dtype=data2remap.dtype.name,
                               value_edge=value_edge,
                               timing=get_par(set_zogy.timing,tel),
                               nthreads=nthreads, tel=tel, set_zogy=set_zogy)
            data_remapped = read_hdulist (fits_out)


            return data_remapped


        # remap reference image; if it is poorly sampled, could use
        # bilinear interpolation for the remapping using SWarp - this
        # removes artefacts around bright stars (see Fig.6 in the
        # SWarp User's Guide). However, despite these artefacts, the
        # Scorr image still appears to be better with LANCZOS3 than
        # when BILINEAR is used.
        # N.B.: SWarp needs to run on the background-subtracted
        # reference image
        if not bkg_sub:
            fits_temp = fits_bkgsub
        else:
            fits_temp = input_fits

        data_ref_remap = help_swarp(fits_temp, data_wcs,
                                    resampling_type='LANCZOS3')

        # remap reference image background std; note that with
        # update_header set to True, the header is updated
        # with that of the wcs-corrected reference image
        data_ref_bkg_std_remap = help_swarp(fits_bkg_std, data_bkg_std,
                                            update_header=True)

        # remap reference mask image; SWarp turns integer
        # mask into float during processing, so need to add 0.5 and
        # convert to integer again
        data_ref_remap_mask = (help_swarp(
            fits_mask, data_mask, update_header=True,
            value_edge=get_par(set_zogy.mask_value['edge'],tel))
                               +0.5).astype('uint8')


    # convert counts to electrons
    satlevel *= gain
    data_wcs *= gain
    data_bkg_std *= gain


    # fix pixels using function [fixpix]
    mask_value = get_par(set_zogy.mask_value,tel)
    if imtype=='new':
        if get_par(set_zogy.interp_sat,tel):
            data_wcs = fixpix (data_wcs, satlevel=satlevel,
                               data_bkg_std=data_bkg_std, data_mask=data_mask,
                               imtype=imtype, header=header,
                               mask_value=mask_value,
                               timing=get_par(set_zogy.timing,tel),
                               base=base, keep_tmp=get_par(set_zogy.keep_tmp,tel),
                               along_row=get_par(set_zogy.interp_along_row,tel),
                               interp_func=get_par(set_zogy.interp_func,tel),
                               interp_dpix=get_par(set_zogy.interp_dpix,tel))

    if remap:

        # convert counts to electrons
        data_ref_remap *= gain
        #data_ref_bkg_remap *= gain
        data_ref_bkg_std_remap *= gain
        # fix pixels using function [fixpix] also in remapped
        # reference image; already done for ML/BG reference images
        if (get_par(set_zogy.interp_sat,tel) and
            tel not in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']):

            data_ref_remap = fixpix (
                data_ref_remap, satlevel=satlevel,
                data_bkg_std=data_ref_bkg_std_remap,
                data_mask=data_ref_remap_mask, imtype=imtype, header=header,
                mask_value=mask_value,
                timing=get_par(set_zogy.timing,tel), base=base,
                keep_tmp=get_par(set_zogy.keep_tmp,tel),
                along_row=get_par(set_zogy.interp_along_row,tel),
                interp_func=get_par(set_zogy.interp_func,tel),
                interp_dpix=get_par(set_zogy.interp_dpix,tel))


            if False:
                ds9_arrays (data_ref_remap=data_ref_remap)



    # determine psf of input image with get_psf function - needs to be
    # done before the optimal fluxes are determined, as it will run
    # PSFEx and create the _psf.fits file that is used in the optimal
    # flux determination. The fft-shifted subimage PSFs are saved in
    # numpy files, the filenames of which are recorded in the
    # dictionary [dict_psf] with the subimage integers as the keys.
    dict_psf, psf_orig = get_psf (input_fits, header, nsubs, imtype, fwhm,
                                  pixscale, remap, fits_mask, nthreads=nthreads)


    # -----------------------------------------------
    # injection of fake/artificial stars in new image
    # -----------------------------------------------

    # the fake stars injected will be added to the transient output
    # catalog so it can be checked whether they are detected and what
    # are their properties
    if imtype=='new' and get_par(set_zogy.nfakestars,tel)>0:

        # number of stars to add
        nfake = get_par(set_zogy.nfakestars,tel)
        radec = get_par(set_zogy.fakestar_radec,tel)

        # coordinates of stars to add; place them in random positions
        # across the image, keeping [edge] pixels off the image edges
        edge = 200
        rng = np.random.default_rng(seed=1)
        xcoords_fake = rng.uniform(low=edge,high=xsize-edge, size=nfake)
        ycoords_fake = rng.uniform(low=edge,high=ysize-edge, size=nfake)

        # replace first coordinates with [radec] if available
        if radec is not None and radec[0] is not None and radec[1] is not None:
            wcs_new = WCS(header)
            x_tmp, y_tmp = wcs_new.all_world2pix(radec[0], radec[1], 1)
            # make sure these coordinates are well within the image
            if (x_tmp > edge and x_tmp < xsize-edge and
                y_tmp > edge and y_tmp < ysize-edge):

                xcoords_fake[0] = x_tmp
                ycoords_fake[0] = y_tmp


        # signal-to-noise ratios of stars to add; can be a scalar,
        # resulting in the same value for all stars, or an array with
        # shape (nfake)
        nsigma_fake_in =  get_par(set_zogy.fakestar_s2n,tel) * np.ones(nfake)

        # use function [get_psfoptflux] to add the stars
        psfex_bintable = '{}_psf.fits'.format(base)
        fakestar_flux_in, __ = get_psfoptflux (
            psfex_bintable, data_wcs, data_bkg_std**2, data_mask, xcoords_fake,
            ycoords_fake, inject_fake=True, nsigma_fake=nsigma_fake_in,
            imtype=imtype, set_zogy=set_zogy, nthreads=1, tel=tel)


        # create table_fake to be used later to merge with full-source
        # or transient catalog; magnitude corresponding to
        # E_FLUX_FAKE_IN is added later on in [merge_fakestars]
        names_fake = ['X_FAKE', 'Y_FAKE', 'SNR_FAKE_IN', 'E_FLUX_FAKE_IN']
        table_fake = Table(
            [xcoords_fake, ycoords_fake, nsigma_fake_in, fakestar_flux_in],
            names=names_fake)

        # save
        fits_fake = '{}_fake.fits'.format(base)
        table_fake.write(fits_fake, overwrite=True)


        # for the inclusion of all fake stars measurements in the
        # catalog, need to re-run source-extractor
        fits_new = '{}_new.fits'.format(base)
        fits.writeto(fits_new, data_wcs, header, overwrite=True)
        sexcat = '{}_cat.fits'.format(base_new)

        result = run_sextractor(fits_new, sexcat, get_par(set_zogy.sex_cfg,tel),
                                get_par(set_zogy.sex_par,tel), pixscale_new,
                                header, fwhm=fwhm, imtype='new',
                                fits_mask=fits_mask, npasses=1, tel=tel,
                                set_zogy=set_zogy, nthreads=nthreads)

        # copy the LDAC binary fits table output from SExtractor (with
        # '_ldac' in the name) to a normal binary fits table;
        # Astrometry.net needs the latter, but PSFEx needs the former,
        # so keep both
        #sexcat = '{}_cat.fits'.format(base)
        #ldac2fits (sexcat_ldac, sexcat)



    # infer image rotation angle from header, used
    # to rotate the image properly to find objects on the
    # frame in [phot_calibrate] and [run_force_phot]
    if 'A-ROT' in header:
        rot = header['A-ROT']
    else:
        rot = 0


    # infer image central coordinates; this is used in forced
    # photometry and also further down below (for airmass at image
    # center and for the calibration catalog) RA-CNTR and DEC-CNTR
    # should have been added to header in run_wcs
    if 'RA-CNTR' in header and 'DEC-CNTR' in header:
        ra_center = header['RA-CNTR']
        dec_center = header['DEC-CNTR']
    else:
        # otherwise, use WCS solution in input [header] to get RA, DEC
        # of central pixel
        wcs = WCS(header)
        ra_center, dec_center = wcs.all_pix2world(xsize/2+0.5,
                                                  ysize/2+0.5, 1)


    # half the field-of-view (FoV) in degrees
    fov_half_deg = np.amax([xsize, ysize]) * pixscale / 3600 / 2


    # important header keywords needed in [phot_calibrate] and [apply_zp]
    keywords = ['exptime', 'filter', 'obsdate']
    exptime, filt, obsdate = read_header(header, keywords)

    # extinction coefficient
    ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]

    if get_par(set_zogy.verbose,tel):
        log.info('exptime: {}, filter: {}, obsdate: {}, ext_coeff: {}'
                 .format(exptime, filt, obsdate, ext_coeff))


    # -------------------
    # create Gaia objmask
    # -------------------

    # added option to create Gaia objmask using function
    # [create_objmask_gaia], but do not use
    if False and imtype=='new':

        fits_objmask_gaia = '{}_objmask_gaia.fits'.format(base)
        if not isfile (fits_objmask_gaia):

            log.info ('creating {}'.format(fits_objmask_gaia))
            fits_gaia = get_par(set_zogy.gaia_cat,tel)
            objmask_gaia = create_objmask_gaia (
                input_fits, header, fits_gaia, obsdate, ra_center, dec_center,
                fov_half_deg, rot, base, data_wcs, data_bkg_std, data_mask,
                imtype)

            # save to fits
            fits.writeto(fits_objmask_gaia, objmask_gaia.astype('uint8'),
                         overwrite=True)

        else:

            # if it exists, read it
            log.info ('reading {}'.format(fits_objmask_gaia))
            objmask_gaia = read_hdulist (fits_objmask_gaia, dtype=bool)



    # read object mask image (since Sep 2022 this is a uint8 image
    # with ones indicating the objects; previous to that it was a
    # float32 image with zeros indicating the objects)
    fits_objmask = '{}_objmask.fits'.format(base)
    if isfile (fits_objmask):
        objmask = read_hdulist (fits_objmask, dtype=bool)
    else:
        objmask = None


    # -----------------------
    # photometric calibration
    # -----------------------

    # redo boolean to determine whether to rerun some parts
    # (source extractor, astrometry.net, psfex) even if those were
    # executed done
    redo = ((get_par(set_zogy.redo_new,tel) and imtype=='new') or
            (get_par(set_zogy.redo_ref,tel) and imtype=='ref'))


    # only execute this phot_calibrate/limiting magnitude image block
    # if the photometric calibration was not already performed
    # successfully, or redo is switched on
    if ('PC-P' in header and header['PC-P'] == True and not redo
        and 'PC-ZP' in header and 'PC-ZPSTD' in header):

        # read zp and zp_std from header
        zp = header['PC-ZP']
        zp_std = header['PC-ZPSTD']
        if 'PC-ZPERR' in header:
            zp_err = header['PC-ZPERR']
        else:
            zp_err = 0

        log.warning ('zeropoint present in header; skipping photometric '
                     'calibration and creation of limiting magnitude image for '
                     '{}'.format(base))

    else:

        # add header keyword(s):
        fits_cal = get_par(set_zogy.cal_cat,tel)
        header['PC-CAT-F'] = (fits_cal.split('/')[-1], 'photom. cat')


        # run [phot_calibrate]
        if (isfile(fits_cal) and filt in get_par(set_zogy.zp_default,tel)):
            pc_ok = True
            zp, zp_std, zp_err, ncal_used, airmass_cal = phot_calibrate (
                fits_cal, header, exptime, filt, obsdate, base, ra_center,
                dec_center, fov_half_deg, data_wcs, data_bkg_std, data_mask,
                imtype, objmask, fwhm, nthreads, rot=rot)

            if zp is None or ncal_used==0:
                pc_ok = False

        else:
            pc_ok = False


        # if photometric calibration failed (either because no photometric
        # calibration catalog could be found, or no matching calibration
        # stars were present in this particular field), use the default
        # zeropoints defined in the settings
        if not pc_ok:
            zp = get_par(set_zogy.zp_default,tel)[filt]
            zp_std = 0
            zp_err = 0
            log.warning ('photometric calibration catalog {} not found '
                         'and/or filter {} not one of ugqriz; using the '
                         'default zeropoint: {} and zp_std=zp_err=0'
                         .format(get_par(set_zogy.cal_cat,tel), filt, zp))
        else:
            log.info ('inferred image zeropoint: {:.4f}, zp_std: {:.4f}, '
                      'zp_err: {:.4f} using {} calibration stars'
                      .format(zp, zp_std, zp_err, ncal_used))


        # add header keyword(s):
        header['PC-P'] = (pc_ok, 'successfully processed by phot. calibration?')
        header['PC-ZPDEF'] = (get_par(set_zogy.zp_default,tel)[filt],
                              '[mag] default filter zeropoint in settings file')
        header['PC-ZP'] = (zp, '[mag] zeropoint=m_AB+2.5*log10(flux[e-/s])+A*k')
        header['PC-ZPSTD'] = (zp_std, '[mag] weighted sigma (STD) zeropoint')
        header['PC-ZPERR'] = (zp_err, '[mag] weighted error zeropoint')
        header['PC-EXTCO'] = (ext_coeff, '[mag] filter extinction coefficient '
                              '(k)')
        if pc_ok:
            # airmass_cal is only present if photometric calibration
            # went fine
            header['PC-AIRM'] = (np.median(airmass_cal),
                                 'median airmass of calibration stars')


        # determine airmass at image center
        lat = get_par(set_zogy.obs_lat,tel)
        lon = get_par(set_zogy.obs_lon,tel)
        height = get_par(set_zogy.obs_height,tel)
        airmass_center, alt_center, az_center = get_airmass (
            ra_center, dec_center, obsdate, lat, lon, height, get_altaz=True)

        # in case of reference image and header airmass==1 (set to
        # unity in refbuild module used for ML/BG) then force
        # airmasses calculated above to be unity.  If the reference
        # image is a combination of multiple images, the airmass
        # calculation above will not be correct. It is assumed that
        # the fluxes in the combined reference image have been scaled
        # to an airmass of 1.
        if imtype=='ref' and 'AIRMASS' in header and header['AIRMASS']==1:
            airmass_center = 1

        log.info('airmass at image center: {:.3f}'.format(airmass_center))
        header['AIRMASSC'] = (airmass_center, 'airmass at image center')


        # also add altitude and azimuth corrsponding to the image center
        header['ALT-CNTR'] = (alt_center, '[deg] altitude at image center')
        header['AZ-CNTR'] = (az_center, '[deg] azimuth at image center')


        # using image center and location, calculate the barycentric
        # julian date corresponding to it; see:
        # https://docs.astropy.org/en/stable/time/index.html#barycentric-and-heliocentric-light-travel-time-corrections
        if 'MJD-OBS' in header:
            mjd_obs = header['MJD-OBS']
        elif 'DATE-OBS' in header:
            mjd_obs = Time(header['DATE-OBS'], format='isot').mjd
        else:
            log.error ('neither MJD-OBS nor DATE-OBS keyword found in header; '
                       'unable to determine Barycentric JD for {}'
                       .format(input_fits))

        if 'mjd_obs' in locals():
            location = EarthLocation(lat=lat, lon=lon, height=height)
            time_obs = Time(mjd_obs, format='mjd', scale='utc',
                            location=location)
            coords_center = SkyCoord(ra_center*u.deg, dec_center*u.deg)
            ltt_bary = time_obs.light_travel_time(coords_center)
            bjd_obs = (time_obs.tdb + ltt_bary).mjd
            header['BJD-OBS'] = (bjd_obs, '[d] Barycentric JD (using '
                                 'DATE-OBS, RA/DEC-CNTR)')



        # -------------------------------------------
        # determine image limiting flux and magnitude
        # -------------------------------------------

        # determine image limiting flux using [get_psfoptflux] with
        # [get_limflux]=True for random coordinates across the field
        nlimits = 501
        edge = 100
        xlim = np.random.rand(nlimits)*(xsize-2*edge) + edge
        ylim = np.random.rand(nlimits)*(ysize-2*edge) + edge


        # execute [get_psfoptflux]
        nsigma = get_par(set_zogy.source_nsigma,tel)
        psfex_bintable = '{}_psf.fits'.format(base)
        limflux_array, __ = get_psfoptflux (
            psfex_bintable, data_wcs, data_bkg_std**2, data_mask, xlim,
            ylim, get_limflux=True, limflux_nsigma=nsigma,
            imtype=imtype, set_zogy=set_zogy, nthreads=1, tel=tel)


        # clipped statistics
        limflux_mean, limflux_median, limflux_std = sigma_clipped_stats(
            limflux_array.astype(float), mask_value=0)
        if get_par(set_zogy.verbose,tel):
            log.info('{}-sigma limiting flux; mean: {:.2f}, std: {:.2f}, '
                     'median: {:.2f}'.format(nsigma, limflux_mean,
                                             limflux_std, limflux_median))


        # infer limiting magnitudes from limiting flux using zeropoint
        # and central airmass
        limflux = limflux_median
        [[limmag], [limfnu]] = apply_zp([limflux], zp, airmass_center, exptime,
                                        ext_coeff, return_fnu=True)
        log.info('{}-sigma limiting Fnu: {:.3f}'.format(nsigma, limfnu))
        log.info('{}-sigma limiting magnitude: {:.3f}'.format(nsigma, limmag))


        # add n-sigma limiting flux in e-/s, in microJy and limiting
        # magnitude to header
        header['LIMEFLUX'] = (limflux/exptime, '[e-/s] full-frame {}-sigma '
                              'limiting flux'.format(nsigma))
        header['LIMFNU'] = (limfnu, '[microJy] full-frame {}-sigma '
                            'limiting Fnu'.format(nsigma))
        header['LIMMAG'] = (limmag, '[mag] full-frame {}-sigma limiting mag'
                            .format(nsigma))



        # ------------------------------------
        # creation of limiting magnitude image
        # ------------------------------------

        # create limiting magnitude image; no need to do so for ML/BG
        # reference images except when zogy is run on reference image
        # only, i.e. remap=False
        if not (tel[0:2] in ['ML','BG'] and imtype=='ref' and remap):

            fits_limmag = '{}_limmag.fits'.format(base)
            create_limmag_image (fits_limmag, header, exptime, filt, data_wcs,
                                 data_bkg_std, data_mask, nsubs, psf_orig, remap,
                                 zp, airmass_center)



    # based on the settings parameter [force_phot_gaia], either
    # perform forced photometry on the coordinates from an input
    # (Gaia) catalog, or optimal photometry for all sources detected
    # already by source extractor


    # read catalog fits table
    fits_cat = '{}_cat.fits'.format(base)
    table_cat = Table.read (fits_cat, memmap=True)


    # do not bother to execute block below if MAG_OPT or FNU_OPT is
    # already present in [fits_cat] and redo is False
    psf_rad_phot = get_par(set_zogy.psf_rad_phot,tel)
    if (('MAG_OPT' in table_cat.colnames or 'FNU_OPT' in table_cat.colnames)
        and not redo
        # and PSF-RADP in header is equal to psf_rad_phot in the
        # settings file
        and ('PSF-RADP' in header and header['PSF-RADP'] == psf_rad_phot)):

        if get_par(set_zogy.force_phot_gaia,tel):
            label = 'forced photometry'
        else:
            label = 'optimal fluxes'

        log.warning ('skipping extraction of {} for {}'.format(label, base))


    else:

        if get_par(set_zogy.force_phot_gaia,tel) and imtype=='new':

            # ------------------------------------------
            # perform forced photometry on input catalog
            # ------------------------------------------

            # update header of input fits image with keywords added by
            # PSFEx and [phot_calibrate]; the function [run_force_phot]
            # requires the zeropoint to be present in the header
            update_imcathead (input_fits, header)


            # perform forced photometry on input_fits at relevant
            # positions in input catalog
            obsdate = read_header(header, ['obsdate'])
            table_cat = run_force_phot (
                input_fits, get_par(set_zogy.gaia_cat,tel), obsdate,
                ra_center, dec_center, fov_half_deg, fits_mask, nthreads,
                rot=rot)


            # make copy of source extractor catalog, before
            # overwriting it with the Gaia forced-photometry catalog
            sexcat = '{}_sexcat.fits'.format(base)
            if os.path.exists(fits_cat):
                shutil.copy2 (fits_cat, sexcat)


            # write fits table to file
            table_cat.write (fits_cat, format='fits', overwrite=True)


            # add number of Gaia objects on the image
            header['NGAIA'] = (len(table_cat), 'number of Gaia sources within '
                               'image footprint')


            # add n-sigma to header
            nsigma = get_par(set_zogy.source_nsigma,tel)
            header['NSIGMA'] = (nsigma, '[sigma] source detection threshold')


            # add the number of significant objects to header
            mask_nsigma = (table_cat['SNR_OPT'] > nsigma)
            header['NOBJECTS'] = (np.sum(mask_nsigma), 'number of >= {}-sigma '
                                  'Gaia objects'.format(nsigma))


        else:

            # -------------------------------
            # determination of optimal fluxes
            # -------------------------------

            # get estimate of optimal flux for all sources detected by
            # source extractor in the new or ref image

            # execute [infer_optimal_fluxmag]
            table_cat = infer_optimal_fluxmag (
                table_cat, header, exptime, filt, obsdate, base, data_wcs,
                data_bkg_std, data_mask, objmask, imtype, zp, fwhm, nthreads)


            # merge fake stars and catalog
            if imtype=='new' and get_par(set_zogy.nfakestars,tel)>0:
                table_merged = merge_fakestars (table_cat, table_fake, 'new',
                                                header)
                table_merged.write(fits_cat, format='fits', overwrite=True)

            else:
                # write updated fits table to file
                table_cat.write (fits_cat, format='fits', overwrite=True)


            # add number of Gaia objects on the image
            header['NGAIA'] = ('None', 'number of Gaia sources within image '
                               'footprint')



        # number of brightest channel stars used; this used to be only
        # relevant for Gaia forced photometry, but since Oct 2024 it
        # is also applied inside [infer_optimal_fluxmag]
        if tel[0:2] in ['ML','BG']:

            if imtype=='ref':
                # for the reference image, the image-average zeropoint
                # is used
                apply_chanzp = False
            else:
                apply_chanzp = get_par(set_zogy.MLBG_phot_apply_chanzp,tel)


            header['PC-ZPCHN'] = (apply_chanzp, 'applied channel ZPs instead of '
                                  'image-average?')

            ncal_min_chan = get_par(set_zogy.MLBG_phot_ncal_min_chan,tel)
            header['PC-NCMIN'] = (ncal_min_chan, 'required minimum number of '
                                  'chan. photcal stars')




    # add estimate of saturation magnitude to the header
    satmag = get_satmag (table_cat)
    header['MAG-SAT'] = (satmag, '[mag] magnitude brightest non-saturated '
                         'source')
    log.info('magnitude of brightest non-saturated source: {:.3f}'
             .format(satmag))


    # update header of input fits image with keywords added by
    # different functions
    update_imcathead ('{}.fits'.format(base), header)


    # ------------------------
    # preparation of subimages
    # ------------------------

    # split full image into subimages to be used in run_ZOGY - this
    # needs to be done after determination of optimal fluxes as
    # otherwise the potential replacement of the saturated pixels will
    # not be taken into account
    if remap:
        data = data_ref_remap
        data_bkg_std = data_ref_bkg_std_remap
        data_mask = data_ref_remap_mask
        del data_wcs
    else:
        data = data_wcs


    # ensure (once more) that edge pixels are set to zero
    value_edge = get_par(set_zogy.mask_value['edge'],tel)
    mask_edge = (data_mask & value_edge == value_edge)
    data[mask_edge] = 0


    # determine cutouts; in case ref image has different size than new
    # image, redefine [ysize] and [xsize] as up to this point they
    # refer to the shape of the original ref image, while shape of
    # remapped ref image should be used
    ysize, xsize = np.shape(data)
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
        get_par(set_zogy.subimage_size,tel), ysize, xsize)

    ysize_fft = (get_par(set_zogy.subimage_size,tel) +
                 2*get_par(set_zogy.subimage_border,tel))
    xsize_fft = (get_par(set_zogy.subimage_size,tel) +
                 2*get_par(set_zogy.subimage_border,tel))


    if get_par(set_zogy.timing,tel):
        t2 = time.time()


    # initialize subimage numpy fft data arrays
    fftdata_sub = np.zeros((ysize_fft, xsize_fft), dtype='float32')
    fftdata_bkg_std_sub = np.zeros((ysize_fft, xsize_fft),
                                   dtype='float32')

    # save fftdata and fftdata_bkg_std to numpy files, one for each
    # subimage, and let these dictionaries contain the filenames with
    # the subimage integer as the key
    dict_fftdata = {}
    dict_fftdata_bkg_std = {}

    for nsub in range(nsubs):

        # initialize to zero
        fftdata_sub[:] = 0
        fftdata_bkg_std_sub[:] = 0

        # fill the subimage fft data arrays
        fftcut = cuts_fft[nsub]
        index_fft = tuple([slice(fftcut[0],fftcut[1]),
                           slice(fftcut[2],fftcut[3])])
        subcutfft = cuts_ima_fft[nsub]
        index_fftdata = tuple([slice(subcutfft[0],subcutfft[1]),
                               slice(subcutfft[2],subcutfft[3])])
        fftdata_sub[index_fft] = data[index_fftdata]
        fftdata_bkg_std_sub[index_fft] = data_bkg_std[index_fftdata]

        # record the numpy filename in a dictionary
        dict_fftdata[nsub] = save_npy_fits(fftdata_sub, '{}_fftdata_sub{}.npy'
                                           .format(base, nsub))

        dict_fftdata_bkg_std[nsub] = save_npy_fits(fftdata_bkg_std_sub,
                                                   '{}_fftdata_bkg_std_sub{}.npy'
                                                   .format(base, nsub))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t2, label='in filling fftdata cubes')


    # call function [prep_plots] to make various plots
    if get_par(set_zogy.make_plots,tel) and 'E_FLUX_AUTO' in table_cat.colnames:
        # in case optimal flux block above was skipped, the SExtractor
        # catalogue with E_FLUX_OPT needs to be read in here to be able
        # to make the plots below
        if 'E_FLUX_OPT' not in table_cat.colnames:
            table_cat = Table.read(fits_cat, memmap=True)

        prep_plots (table_cat, header, base)


    # remove file(s) if not keeping intermediate/temporary files
    if not get_par(set_zogy.keep_tmp,tel):
        if 'fits_calcat_field' in locals():
            remove_files ([fits_calcat_field], verbose=True)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in prep_optimal_subtraction')


    return dict_fftdata, dict_psf, psf_orig, dict_fftdata_bkg_std


################################################################################

def get_satmag (table_cat):

    colnames = table_cat.colnames

    if 'FLAGS_MASK' in colnames:
        col_mask = 'FLAGS_MASK'
    elif 'MASK' in colnames:
        col_mask = 'MASK'
    else:
        log.warning ('FLAGS_MASK and MASK not in column names; not able to '
                     'estimate the flux of the brightest non-saturated source')
        return 99.0


    value_sat = get_par(set_zogy.mask_value,tel)['saturated']
    mask_sat = (table_cat[col_mask] & value_sat == value_sat)
    if 'MAG_OPT' not in table_cat.colnames:
        table_cat['MAG_OPT'] = fnu2mag(table_cat['FNU_OPT'].value)

    satmag = np.sort(table_cat[~mask_sat]['MAG_OPT'])[0]


    return satmag


################################################################################

def create_limmag_image (fits_limmag, header, exptime, filt, data_wcs,
                         data_bkg_std, data_mask, nsubs, psf_orig, remap, zp,
                         airmass_center):


    # create 1-sigma error image using:
    #
    #  noise = sqrt(flux_tot + area_ap * (rdnoise**2 + background))
    #        ~ sqrt(area_ap * (flux/pixel + (rdnoise**2 + background))
    #        = sqrt(area_ap * (data_wcs + data_bkg_std**2)
    #
    # where area_ap is the effective aperture area used, flux is
    # [data_wcs] and readnoise**2 + background is the same as
    # [data_bkg_std]**2.
    #
    # N.B.: this is only valid if the flux in a particular pixel
    # is the same as the average flux per pixel in the imaginary
    # aperture of size area_ap centered on that pixel, which is
    # fine for the background regions, but it will overestimate
    # the noise at e.g. the centers of stars, where it is wrongly
    # assumed the entire aperture contains the same flux per pixel
    # as the flux at the center. Would have to convolve the noise
    # image (data_wcs + data_bkg_std**2) with the PSF and
    # scale it to determine the proper limiting magnitude image

    area_ap = np.zeros(nsubs)
    for i in range(nsubs):
        # The effective area is inferred from the
        # PSF profile P (volume normalized to unity) as follows:
        #
        # var = 1 / sum(P**2/V)  - Eq. 9 from Horne (1986)
        #
        # assuming V is roughly equal for each pixel, then:
        #
        # 1 / sum(P**2/V) = V / sum(P**2) = area_ap * V
        # so: area_ap = 1 / sum(P**2)
        sum_P2 = np.sum(psf_orig[i]**2)
        if sum_P2 != 0:
            area_ap[i] = 1/sum_P2


    # set zero values to median
    mask_zero = (area_ap==0)
    if np.all(mask_zero):
        area_ap[:] = 1
    else:
        area_ap[mask_zero] = np.median(area_ap)


    log.info ('median effective aperture area for calculation of limiting '
              'magnitude image: {:.2f} pix, inferred from 1/sum(P**2)'
              .format(np.median(area_ap)))


    # reshape, interpolate and grow area_ap array (1D) to size of
    # data_wcs; the reference image may not have the same size as the
    # new image so that the [subimage_size] does not fit integer times
    # into the image - in that case use the new-image size and pad the
    # image to the actual size of the reference image
    ysize, xsize = data_wcs.shape
    ysize_new, xsize_new = get_par(set_zogy.shape_new,tel)
    subsize = get_par(set_zogy.subimage_size,tel)
    if remap:
        # remap is True, so the number of subimages is the same as
        # for the new image
        nx = int(xsize_new / subsize)
        ny = int(ysize_new / subsize)
    else:
        # need to take care that ref image may have a different
        # size than the new image
        nx = int(xsize / subsize)
        ny = int(ysize / subsize)



    # grow the image using ndimage.zoom
    data_area_ap = ndimage.zoom(area_ap.reshape((nx,ny)).transpose(),
                                subsize, order=2, mode='nearest')


    # if shape not that of input image [data_wcs], e.g. remap is False
    # and reference image has a different size than the new image,
    # need to pad it
    if data_area_ap.shape != data_wcs.shape:

        #log.info ('data_area_ap.shape: {}, data_wcs.shape: {}'
        #          .format(data_area_ap.shape, data_wcs.shape))

        if remap:

            # find x,y pixel position in [data_wcs] of origin
            # (x,y)=(1,1) of new image
            header_new = read_hdulist ('{}.fits'.format(base_new),
                                       get_data=False, get_header=True)
            wcs_new = WCS(header_new)
            x0 = y0 = np.array([1])
            ra0, dec0 = wcs_new.all_pix2world(x0, y0, 1)
            #log.info ('ra0: {}, dec0: {}'.format(ra0, dec0))
            wcs_ref = WCS(header)
            x0_ref, y0_ref = wcs_ref.all_world2pix(ra0, dec0, 1)
            #log.info ('x0_ref: {}, y0_ref: {}'.format(x0_ref, y0_ref))
            # determine padding on 4 sides
            nx_before = max(int(x0_ref[0])-1, 0)
            ny_before = max(int(y0_ref[0])-1, 0)
            nx_after = max(xsize - (int(x0_ref[0])+xsize_new-1), 0)
            ny_after = max(ysize - (int(y0_ref[0])+ysize_new-1), 0)

        else:

            # subimages were extracted starting from image index
            # (0,0), so pad only after up to the image size
            ny_before = nx_before = 0
            ny_after = ysize - ny*subsize
            nx_after = xsize - nx*subsize


        # pad
        pad_width = ((ny_before, ny_after), (nx_before, nx_after))
        log.info ('shape of [data_area_ap]: {} not equal to shape of '
                  '[data_wcs]: {}; padding it with pad_width: {}'
                  .format(data_area_ap.shape, data_wcs.shape, pad_width))
        data_area_ap = np.pad (data_area_ap, pad_width, mode='edge')


    # calculate error image
    data_err = np.sqrt(data_area_ap * (np.maximum(data_wcs,0) + data_bkg_std**2))

    # replace any zero values with a large value
    mask_zero = (data_err==0)
    data_err[mask_zero] = 100*np.median(data_bkg_std)


    # apply the zeropoint
    ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]
    data_limmag = apply_zp((get_par(set_zogy.source_nsigma,tel) * data_err),
                           zp, airmass_center, exptime, ext_coeff
                           ).astype('float32')


    # set limiting magnitudes at edges to zero; only consider 'real'
    # edge pixels for reference image
    value_edge = get_par(set_zogy.mask_value['edge'],tel)
    mask_edge = (data_mask == value_edge)
    data_limmag[mask_edge] = 0


    # write limiting magnitude image
    header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
    fits.writeto(fits_limmag, data_limmag, header, overwrite=True)


    return


################################################################################

def infer_optimal_fluxmag (table_cat, header, exptime, filt, obsdate, base,
                           data_wcs, data_bkg_std, data_mask, objmask, imtype,
                           zp, fwhm, nthreads):


    if get_par(set_zogy.timing,tel): t1 = time.time()
    log.info('deriving optimal fluxes ...')


    # read pixel positions
    xwin = table_cat['X_POS'].value
    ywin = table_cat['Y_POS'].value


    # use local or global background?
    bkg_global = (get_par(set_zogy.bkg_phototype,tel) == 'global')


    # determine local sky background to be supplied to be used in
    # [get_psfoptflux] as the background
    if bkg_global:

        local_bkg = None

    else:

        if True:

            # for consistency with the aperture flux measurements by
            # source extractor, adopt the background from the
            # BACKGROUND column in table_cat, which is the local
            # background determined by source extractor in a
            # rectangular aperture without any discarding of potential
            # objects in the sky region; that background was also used
            # by the aperture measurements
            local_bkg = table_cat['BACKGROUND']

        else:
            # alternatively, determine sky background using
            # [get_apflux]; in this case, should really also replace
            # the catalog aperture fluxes with those from function
            # get_apflux
            bkg_radii = get_par(set_zogy.bkg_radii,tel)
            bkg_limfrac = get_par(set_zogy.bkg_limfrac,tel)
            __, __, local_bkg = get_apflux (
                xwin, ywin, data_wcs, data_mask, fwhm, objmask=objmask,
                bkg_radii=bkg_radii, bkg_limfrac=bkg_limfrac, set_zogy=set_zogy,
                tel=tel, nthreads=nthreads)



    psfex_bintable = '{}_psf.fits'.format(base)
    flux_opt, fluxerr_opt = get_psfoptflux_mp (
        psfex_bintable, data_wcs, data_bkg_std**2, data_mask, xwin, ywin,
        psffit=False, imtype=imtype, local_bkg=local_bkg, set_zogy=set_zogy,
        nthreads=nthreads, tel=tel)

    # add optimal flux columns to table
    table_cat['E_FLUX_OPT'] = flux_opt
    table_cat['E_FLUXERR_OPT'] = fluxerr_opt


    # [mypsffit] determines if PSF-fitting part is also performed;
    # this is different from SExtractor PSF-fitting
    mypsffit = get_par(set_zogy.psffit,tel)
    if mypsffit:
        # add psffit quantities to table to be able to plot them later
        results = get_psfoptflux_mp (
            psfex_bintable, data_wcs, data_bkg_std**2, data_mask, xwin, ywin,
            psffit=True, imtype=imtype, local_bkg=local_bkg,
            set_zogy=set_zogy, nthreads=nthreads, tel=tel)

        colnames = ['X_PSF', 'XERR_PSF', 'Y_PSF', 'YERR_PSF',
                    'E_FLUX_PSF', 'E_FLUXERR_PSF', 'CHI2_PSF']
        table_cat.add_columns(results, names=colnames)


    # if required, replace saturated+connected pixels with PSF
    # profile for the saturated objects
    if get_par(set_zogy.replace_sat_psf,tel):

        # mask with objects identified as saturated
        mask_value = get_par(set_zogy.mask_value,tel)
        value_sat = mask_value['saturated']
        mask_sat = (table_cat['FLAGS_MASK'] & value_sat == value_sat)
        xwin_sat = table_cat['X_POS'][mask_sat]
        ywin_sat = table_cat['Y_POS'][mask_sat]
        sat_nmax = get_par(set_zogy.replace_sat_nmax,tel)
        get_psfoptflux_mp (psfex_bintable, data_wcs, data_bkg_std**2,
                           data_mask, xwin_sat, ywin_sat, imtype=imtype,
                           replace_sat_psf=True,
                           replace_sat_nmax=sat_nmax, set_zogy=set_zogy,
                           nthreads=1, tel=tel)


        # save data_wcs after modification in [get_psfoptflux] if
        # temporary files are kept
        if get_par(set_zogy.keep_tmp,tel):
            fits_updated = '{}_updated.fits'.format(base)
            fits.writeto(fits_updated, data_wcs, header, overwrite=True)


    if get_par(set_zogy.make_plots,tel):

        # make ds9 regions file with all detections
        result = prep_ds9regions(
            '{}_alldet_ds9regions.txt'.format(base), xwin, ywin,
            radius=2., width=1, color='blue',
            value=np.arange(1,len(xwin)+1))

        # and with flux_opt/fluxerr_opt < 5.
        index_nonzero = np.nonzero(fluxerr_opt)[0]
        mask_s2n_lt5 = np.zeros(flux_opt.shape, dtype='bool')
        mask_s2n_lt5[index_nonzero] = (
            np.abs(flux_opt[index_nonzero]/fluxerr_opt[index_nonzero])<5)
        mask_s2n_lt5 |= (flux_opt==0)
        result = prep_ds9regions(
            '{}_s2n_lt5_ds9regions.txt'.format(base), xwin[mask_s2n_lt5],
            ywin[mask_s2n_lt5], radius=2., width=1, color='red',
            value=np.arange(1,np.sum(mask_s2n_lt5)+1))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t1, label='in deriving optimal fluxes')


    if get_par(set_zogy.timing,tel): t2 = time.time()


    # add n-sigma to header; for new images this is nsigma
    nsigma = get_par(set_zogy.source_nsigma,tel)
    if imtype=='new':
        header['NSIGMA'] = (nsigma, '[sigma] source detection threshold')
    else:
        # for reference image, do not apply S/N >= nsigma cut
        # below to go as deep as possible and not discard any
        # faint galaxies
        header['NSIGMA'] = (0, '[sigma] source detection threshold')


    # get airmasses for SExtractor catalog sources
    ra_sex = table_cat['RA'].value
    dec_sex = table_cat['DEC'].value
    flags_sex = table_cat['FLAGS'].value
    xcoords_sex = table_cat['X_POS'].value
    ycoords_sex = table_cat['Y_POS'].value

    lat = get_par(set_zogy.obs_lat,tel)
    lon = get_par(set_zogy.obs_lon,tel)
    height = get_par(set_zogy.obs_height,tel)
    airmass_sex = get_airmass (ra_sex, dec_sex, obsdate, lat, lon, height)
    airmass_sex_median = np.median(airmass_sex)


    # in case of reference image and header airmass==1 (set to
    # unity in refbuild module used for ML/BG) then force
    # airmasses calculated above to be unity.  If the reference
    # image is a combination of multiple images, the airmass
    # calculation above will not be correct. It is assumed that
    # the fluxes in the combined reference image have been scaled
    # to an airmass of 1.
    if imtype=='ref' and 'AIRMASS' in header and header['AIRMASS']==1:
        airmass_sex[:] = 1
        airmass_sex_median = 1

    log.info('median airmass calibration stars: {:.3f}'
             .format(airmass_sex_median))



    # conversion of catalog fluxes to magnitudes
    # ==========================================

    # split aperture flux with shape (nrows, napps) into separate
    # columns
    apphot_radii = get_par(set_zogy.apphot_radii,tel)
    for col in table_cat.colnames:
        if col=='E_FLUX_APER' or col=='E_FLUXERR_APER':
            # update column names of aperture fluxes to include radii
            # loop apertures
            for i_ap in range(len(apphot_radii)):
                # rename column
                col_new = '{}_R{}xFWHM'.format(col, apphot_radii[i_ap])
                # append it to table_cat
                table_cat[col_new] = table_cat[col][:,i_ap]

            # delete original column
            del table_cat[col]



    # for ML/BG, if set_zogy.MLBG_phot_apply_chanzp is True, replace
    # [zp] with array of zeropoints, one for each object; same for
    # [zp_err]
    if (tel[0:2] in ['ML','BG'] and imptype=='new'
        and get_par(set_zogy.MLBG_phot_apply_chanzp,tel)):

        # even though MLBG_phot_apply_chanzp is True, channel
        # zeropoints may not be available and moreover they could be
        # zero; use image zp in those cases
        zp_chan = [header['PC-ZP{}'.format(i+1)]
                   if 'PC-ZP{}'.format(i+1) in header
                   and header['PC-ZP{}'.format(i+1)] != 0
                   else zp for i in range(16)]
        zp = get_zp_coords (xwin, ywin, zp_chan, zp)


        # same for zperr
        zp_err_chan = [header['PC-ZPE{}'.format(i+1)]
                       if 'PC-ZPE{}'.format(i+1) in header
                       and header['PC-ZP{}'.format(i+1)] != 0
                       else zp for i in range(16)]
        zp = get_zp_coords (xwin, ywin, zp_chan, zp)



    # convert fluxes to magnitudes and add to catalog
    colnames = table_cat.colnames
    ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]
    for col in colnames:

        if 'E_FLUX_' in col and 'RADIUS' not in col and 'FLUX_MAX' not in col:

            # define various column names
            col_flux = col
            col_fluxerr = col.replace('E_FLUX', 'E_FLUXERR')
            col_mag = col.replace('E_FLUX', 'MAG')
            col_magerr = col.replace('E_FLUX', 'MAGERR')
            col_magerrtot = col.replace('E_FLUX', 'MAGERRTOT')
            col_fnu = col.replace('E_FLUX', 'FNU')
            col_fnuerr = col.replace('E_FLUX', 'FNUERR')
            col_fnuerrtot = col.replace('E_FLUX', 'FNUERRTOT')


            # convert electron fluxes to mags and fnus
            mag_tmp, magerr_tmp, magerrtot_tmp, fnu_tmp, fnuerr_tmp, \
                fnuerrtot_tmp = apply_zp(table_cat[col_flux], zp, airmass_sex,
                                         exptime, ext_coeff,
                                         fluxerr=table_cat[col_fluxerr],
                                         return_fnu=True, zperr=zperr)


            # if present, delete original col_mag and col_fnu columns
            cols_magfnu = [col_mag, col_magerr, col_magerrtot,
                           col_fnu, col_fnuerr, col_fnuerrtot]

            for col_tmp in cols_magfnu:
                if col_tmp in colnames:
                    del table_cat[col_tmp]


            # add mag columns to table
            table_cat[col_mag] = mag_tmp.astype('float32')
            table_cat[col_magerr] = magerr_tmp.astype('float32')
            table_cat[col_magerrtot] = magerrtot_tmp.astype('float32')


            # add fnu columns to table
            table_cat[col_fnu] = fnu_tmp.astype('float32')
            table_cat[col_fnuerr] = fnuerr_tmp.astype('float32')
            table_cat[col_fnuerrtot] = fnuerrtot_tmp.astype('float32')



        # also convert MU_MAX in the ref catalog from e- to
        # magnitudes (per pix**2); SExtractor calculation is as
        # follows: -2.5 log10 (E_FLUX_MAX (in e-) / pixscale**2), so
        # easiest to use E_FLUX_MAX to convert to magnitude
        if 'MU_MAX' in col:
            if 'E_FLUX_MAX' in colnames:
                flux_max = table_cat['E_FLUX_MAX']
            else:
                flux_max = 10**(-0.4*table_cat['MU_MAX']) * pixscale**2

            mag_tmp = apply_zp(flux_max, zp, airmass_sex, exptime, ext_coeff)
            table_cat['MU_MAX'] = mag_tmp.astype('float32')



    # discard sources with flux S/N below get_par(set_zogy.source_nsigma,tel)
    # and with negative fluxes
    def get_mask (flux, fluxerr, nsigma):
        # mask with same size as flux array, initialized to False
        mask_all = np.zeros(flux.size, dtype=bool)
        # mask where fluxerr is not equal to zero
        mask_ok = (fluxerr!=0)
        # where fluxerr is not equal to zero, determine mask of
        # sources with positive fluxes and flux errors and sources
        # with significant S/N (flux/fluxerr > nsigma)
        mask_all[mask_ok] = ((flux[mask_ok]>0) & (fluxerr[mask_ok]>0) &
                             ((flux[mask_ok]/fluxerr[mask_ok]) >= nsigma))
        return mask_all


    # for reference image, filter out objects lower than S/N=nsigma
    # according to SExtractor AUTO fluxes and errors, if available
    if imtype=='ref' and ('E_FLUX_AUTO' in table_cat.colnames and
                          'E_FLUXERR_AUTO' in table_cat.colnames):
        log.info ('discarding sources with S/N_AUTO < {}'.format(nsigma))
        mask_nsigma = get_mask(table_cat['E_FLUX_AUTO'],
                               table_cat['E_FLUXERR_AUTO'], nsigma)
    else:
        # for new images or auto fluxes are not available, use the
        # optimal fluxes and errors
        log.info ('discarding sources with S/N_OPT < {}'.format(nsigma))
        mask_nsigma = get_mask(flux_opt, fluxerr_opt, nsigma)


    # update header
    header['NOBJECTS'] = (np.sum(mask_nsigma), 'number of >= {}-sigma '
                          'objects'.format(nsigma))


    return table_cat[mask_nsigma]


################################################################################

def limit_cal (table_cal, filt):

    # limit calibration catalog entries based on chi2 of stellar template fits
    if 'chi2' in table_cal.colnames:
        mask_cal = (np.isfinite(table_cal['chi2']) & (table_cal['chi2'] <= 10))
        #data_cal = data_cal[:][mask_cal]
        table_cal = table_cal[mask_cal]


    ncalstars = len(table_cal)
    log.info('number of phot.cal. stars in FOV after chi2 cut: {}'
             .format(ncalstars))


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
    filt_req['q'] = ['GALEX_NUV', 'SM_u', 'SM_v', 'SDSS_u', 'SDSS_g', 'PS1_g',
                     'SM_g']
    filt_req['i'] = ['PS1_z', 'PS1_y', 'SDSS_z', 'SM_z']
    filt_req['z'] = ['PS1_y']

    # [mask_cal] is mask for entries in [table_cal] where all
    # filters listed in [filt_req['all']] are present (True)
    mask_cal = np.all([table_cal[col] for col in filt_req['all']], axis=0)
    table_cal = table_cal[mask_cal]

    # loop the the filter keys of [filt_req]
    if filt in filt_req:
        # [mask_cal_filt] is mask for for entries in the
        # updated [table_cal] for which all filters in
        # [filt_req[current filter]] are present
        mask_cal_filt = np.any([table_cal[col] for col in filt_req[filt]],
                               axis=0)
        # if less than [set_zogy.phot_ncal_min] stars left,
        # drop filter requirements and hope for the best!
        phot_ncal_min = 10
        if np.sum(mask_cal_filt) >= phot_ncal_min:
            table_cal = table_cal[mask_cal_filt]
        else:
            log.warning('less than {} calibration stars with '
                        'default filter requirements'
                        .format(phot_ncal_min))
            log.info('filter: {}, requirements (any one of these): {}'
                     .format(filt, filt_req[filt]))
            log.info('dropping this specific requirement and hoping '
                     'for the best')


    return table_cal


################################################################################

def phot_calibrate (fits_cal, header, exptime, filt, obsdate, base, ra_center,
                    dec_center, fov_half_deg, data_wcs, data_bkg_std, data_mask,
                    imtype, objmask, fwhm, nthreads, rot=0):


    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing phot_calibrate ...')


    # read [fits_calcat_field] if it exists; it should have been
    # produced already by [run_wcs] when checking the astrometric
    # accuracy
    fits_calcat_field = '{}_calcat_field_{}.fits'.format(base, imtype)
    if isfile(fits_calcat_field):

        # read [fits_calcat_field]
        log.info ('reading {}'.format(fits_calcat_field))
        table_cal = Table.read(fits_calcat_field, memmap=True)

    else:

        # select calibration stars that are within the image
        if '20181115' in fits_cal:
            # old calibration catalog is indexed by 1-degree declination
            # zones, so need to use [get_imagestars_zones]
            table_cal = get_imagestars_zones (fits_cal, obsdate, ra_center,
                                              dec_center, fov_half_deg)

        else:
            # new calibration catalog is indexed by HEALPix indices
            table_cal = get_imagestars_hpindex (fits_cal, obsdate, ra_center,
                                                dec_center, fov_half_deg,
                                                rot=rot)

        log.info ('selected {} calibration stars within the image {}, with '
                  'central ra,dec: {:.4f},{:.4f}, a field-of-view of {:.3f} '
                  'degrees and DATE-OBS: {}'
                  .format(len(table_cal), base, ra_center, dec_center,
                          2*fov_half_deg, obsdate))



    ncalstars = len(table_cal)
    log.info('number of potential photometric calibration stars in FOV: {}'
             .format(ncalstars))
    header['PC-TNCAL'] = (ncalstars, 'total number of photcal stars in FOV')


    # for the old calibration catalog, limit [table_cal] with respect
    # to the catalog chi2 of the stellar template fits, and also the
    # filters
    if '20181115' in fits_cal:

        table_cal = limit_cal (table_cal, filt)

        # number of photometric calibration stars after the chi2 and
        # filter requirements cut.
        ncalstars = len(table_cal)
        log.info('number of photometric stars in FOV after filter cut: {}'
                 .format(ncalstars))
        header['PC-FNCAL'] = (ncalstars, 'number of photcal stars after '
                              'filter cut')


    # infer the pixel positions
    ra_cal = table_cal['ra'].value
    dec_cal = table_cal['dec'].value
    xpos, ypos = WCS(header).all_world2pix(ra_cal, dec_cal, 1)


    # use local or global background?
    bkg_global = (get_par(set_zogy.bkg_phototype,tel) == 'global')

    # determine local sky background to be supplied to be used in
    # [get_psfoptflux] as the background
    if bkg_global:

        local_bkg = None

    else:

        # determine sky background using [get_apflux]
        bkg_radii = get_par(set_zogy.bkg_radii,tel)
        bkg_limfrac = get_par(set_zogy.bkg_limfrac,tel)
        __, __, local_bkg = get_apflux (
            xpos, ypos, data_wcs, data_mask, fwhm, objmask=objmask,
            bkg_radii=bkg_radii, bkg_limfrac=bkg_limfrac, set_zogy=set_zogy,
            tel=tel, nthreads=1)



    # infer their optimal fluxes
    psfex_bintable = '{}_psf.fits'.format(base)
    flux_opt, fluxerr_opt, flags_mask_opt = get_psfoptflux_mp (
        psfex_bintable, data_wcs, data_bkg_std**2, data_mask, xpos, ypos,
        imtype=imtype, local_bkg=local_bkg, set_zogy=set_zogy,
        get_flags_mask_central=True, nthreads=1, tel=tel)


    # calculate the zeropoint (for ML/BG also calculate it per channel
    # and do the ZP surface fit), starting with the airmasses of the
    # calibration stars
    lat = get_par(set_zogy.obs_lat,tel)
    lon = get_par(set_zogy.obs_lon,tel)
    height = get_par(set_zogy.obs_height,tel)
    airmass_cal = get_airmass (ra_cal, dec_cal, obsdate, lat, lon, height)


    # only continue if calibration stars are present in the FOV
    zp = None
    zp_std = None
    zp_err = None
    ncal_used = 0
    if ncalstars>0:

        if tel in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']:
            if '20181115' in fits_cal:
                # old calibration catalog lacks _ML subscripts
                filt_subscript = ''
            else:
                # new catalog contains _ML and _BG subscripts
                filt_subscript = '_{}'.format(tel[0:2])
        else:
            # what to do for other telescopes?
            #filt_subscript = '_{}'.format(tel)
            filt_subscript = ''


        # extract calibration magnitudes
        mag_cal = table_cal['{}{}'.format(filt, filt_subscript)].value


        # pick calibration stars with good S/N and zero FLAGS_MASK
        snr_opt = np.zeros_like(flux_opt)
        mask_nonzero = (fluxerr_opt != 0)
        snr_opt[mask_nonzero] = (flux_opt[mask_nonzero] /
                                 fluxerr_opt[mask_nonzero])
        mask_zp = ((snr_opt >= 10) & (flags_mask_opt==0))
        log.info ('{} calibration catalog entries left after flux/mask filter '
                  .format(np.sum(mask_zp)))


        # collect individual zeropoints across entire image;
        # N.B. these output arrays are sorted in brightess according
        # to mag_cal
        x_array, y_array, zp_array, zperr_array = collect_zps (
            xpos[mask_zp], ypos[mask_zp], flux_opt[mask_zp],
            fluxerr_opt[mask_zp], mag_cal[mask_zp], airmass_cal[mask_zp],
            exptime, filt)


        # determine single zeropoint for entire image
        zp, zp_std, zp_err, ncal_used = calc_zp (
            x_array, y_array, zp_array, zperr_array, filt, imtype,
            data_wcs.shape, zp_type='single')


        header['PC-NCAL'] = (ncal_used,
                             'number of photcal stars used (S/N>=10,flags=0)')


        # after sorting in brightness and masking table_cal with
        # [mask_zp], add (x,y) positions, zeropoints and airmasses to
        # it, and overwrite [fits_calcat_field] that can be used to
        # inspect the zeropoints as a function of position and stellar
        # colour
        index_sort = np.argsort(mag_cal[mask_zp])
        table_cal = table_cal[mask_zp][index_sort]
        table_cal['x_pos'] = x_array.astype('float32')
        table_cal['y_pos'] = y_array.astype('float32')
        table_cal['zeropoint'] = zp_array.astype('float32')
        table_cal['airmass'] = airmass_cal[mask_zp][index_sort].astype('float32')
        table_cal.write (fits_calcat_field, format='fits', overwrite=True)


        # plot zeropoints vs. calibration star colour
        if get_par(set_zogy.make_plots,tel):

            log.info ('table_cal.colnames: {}'.format(table_cal.colnames))
            log.info ('filt_subscript: {}'.format(filt_subscript))

            col = ['g{}'.format(filt_subscript), 'r{}'.format(filt_subscript)]
            if (col[0] in table_cal.colnames and col[1] in table_cal.colnames):

                col_array = (table_cal[col[0]] - table_cal[col[1]])

                fig, ax = plt.subplots(1, 1)
                fig.subplots_adjust(hspace=0, wspace=0)
                fig.suptitle(base.split('/')[-1], fontsize=10)

                ax.plot(col_array, zp_array, 'o', color='tab:blue',
                        markersize=5, markeredgecolor='k')
                ax.set_ylabel('image zeropoints (mag)')
                ax.set_xlabel('{} - {} colour'.format(col[0], col[1]))

                plt.savefig('{}_zeropoint_vs_color_{}.pdf'.format(base, imtype))
                if get_par(set_zogy.show_plots,tel): plt.show()
                plt.close()



        # for MeerLICHT and BlackGEM only
        if tel[0:2] in ['ML','BG']:

            # calculate zeropoint for each channel - not
            # restricted anymore to brightest maximum number
            # defined by get_par(set_zogy.phot_ncal_use,tel)
            # which is used in the single-value ZP above
            zp_chan, zp_std_chan, zp_err_chan, ncal_chan = calc_zp (
                x_array, y_array, zp_array, zperr_array, filt, imtype,
                zp_type='channels')


            # replace channels points that have zero values (due to
            # them having less than ncal_use_chan stars available)
            # with the image zp; N.B.: the number of stars actually
            # used, denoted by ncal_chan, can be a bit less than the
            # number available due to sigma clipping in calc_zp
            mask_zero = (zp_chan == 0)
            zp_chan[mask_zero] = zp
            zp_std_chan[mask_zero] = zp_std
            zp_err_chan[mask_zero] = zp_err


            for i_chan in range(zp_chan.size):
                header['PC-ZP{}'.format(i_chan+1)] = (
                    zp_chan[i_chan], '[mag] chan {} weighted zeropoint'
                    .format(i_chan+1))
            for i_chan in range(zp_chan.size):
                header['PC-ZPS{}'.format(i_chan+1)] = (
                    zp_std_chan[i_chan], '[mag] chan {} sigma (STD) weighted '
                    'zeropoint'.format(i_chan+1))
            for i_chan in range(zp_chan.size):
                header['PC-ZPE{}'.format(i_chan+1)] = (
                    zp_err_chan[i_chan], '[mag] chan {} error weighted zeropoint'
                    .format(i_chan+1))
            for i_chan in range(zp_chan.size):
                header['PC-NCC{}'.format(i_chan+1)] = (
                    ncal_chan[i_chan], 'chan {} number of photcal stars used'
                    .format(i_chan+1))


            # fit low-order 2D polynomial to zeropoint array;
            # normalize coordinates by image size to make it
            # easier on the polynomial fit
            order = 2
            ysize, xsize = data_wcs.shape
            f_norm = max(ysize, xsize)
            # subtract 1 from coordinate arrays to convert fit
            # to pixel indices rather than pixel coordinates,
            # and then normalize to have coordinate values
            # around 0-1
            coeffs = polyfit2d ((x_array-1)/f_norm, (y_array-1)/f_norm,
                                zp_array, order=order)

            # evaluate at image grid in pixel indices
            x = np.arange(xsize)
            y = np.arange(ysize)
            zp_2dfit = polygrid2d(x/f_norm, y/f_norm, coeffs).T

            # N.B.: polynomials were fit with normalized x and
            # y indices (using [f_norm]), so coeffients should
            # be scaled accordingly if original image pixel
            # indices are used to infer the polynomial fit
            # (most natural)
            xy = np.arange(order+1)
            coeffs_power = np.sum(np.meshgrid(xy,xy), axis=0)
            coeffs_scaled = coeffs / (float(f_norm)**coeffs_power)

            # record fit coefficients in header
            header['PC-ZPFDG'] = (order,
                                  'zeropoint 2D polynomial fit degree')
            for nc, coeff in enumerate(coeffs_scaled.ravel()):
                if np.isfinite(coeff):
                    value = coeff
                else:
                    value = 'None'

                header['PC-ZPF{}'.format(nc)] = (
                    value, 'zeropoint 2D poly fit coefficient {}'.format(nc))


            if get_par(set_zogy.make_plots,tel):

                # save zeropoint arrays
                ascii.write([x_array, y_array, zp_array, zperr_array],
                            '{}_zp.dat'.format(base),
                            names=['x_array', 'y_array', 'zp_array',
                                   'zperr_array'], overwrite=True)

                # plot
                plt.imshow(zp_2dfit, origin='lower', aspect=1)
                plt.colorbar()
                plt.scatter(x_array, y_array, c=zp_array, marker='o',
                            edgecolor='k')
                plt.xlim(x.min(), x.max())
                plt.xlabel('X-axis')
                plt.ylim(y.min(), y.max())
                plt.ylabel('Y-axis')
                plt.title('polynomial fit (order={}) to image zeropoints'
                          .format(order))
                plt.savefig('{}_zp_2dfit.pdf'.format(base))
                plt.close()


            # determine median zeropoints on subimages to
            # check if they are constant across the image
            subsize = get_par(set_zogy.subimage_size,tel)
            zp_mini, zp_std_mini, zp_err_mini, ncal_mini = calc_zp (
                x_array, y_array, zp_array, zperr_array, filt, imtype,
                data_shape=data_wcs.shape, zp_type='background',
                boxsize=subsize)


            # if keeping intermediate/temporary files, save
            # these mini images
            if get_par(set_zogy.keep_tmp,tel):

                fits.writeto('{}_zp_subs.fits'.format(base),
                             zp_mini, overwrite=True)
                fits.writeto('{}_zp_std_subs.fits'.format(base),
                             zp_std_mini, overwrite=True)
                fits.writeto('{}_zp_err_subs.fits'.format(base),
                             zp_err_mini, overwrite=True)
                fits.writeto('{}_zp_ncal_subs.fits'.format(base),
                             ncal_mini.astype('int16'), overwrite=True)


            # add statistics of these arrays to header
            mask_use = ((zp_mini != 0) & (zp_std_mini != 0) & (zp_err_mini != 0))
            n_zps = np.sum(mask_use)
            if n_zps >= 4:
                # discard the highest and lowest value in stats
                arr_sort = np.sort(zp_mini[mask_use])
                max_diff = np.abs(arr_sort[-2] - arr_sort[1])
                arr_sort = np.sort(zp_std_mini[mask_use])
                max_std = arr_sort[-2]
                arr_sort = np.sort(zp_err_mini[mask_use])
                max_err = arr_sort[-2]
            else:
                log.warning ('too few subimages (<4) available to '
                             'determine maximum ZP difference, STD and ERR')
                max_diff = 'None'
                max_std = 'None'
                max_err = 'None'


            header['PC-TNSUB'] = (int(ysize/subsize)**2, 'total number of '
                                  'subimages available')
            header['PC-NSUB'] = (n_zps, 'number of subimages used for ZP '
                                 'statistics')
            header['PC-MZPD'] = (max_diff, '[mag] max. ZP difference between '
                                 'subimages')
            header['PC-MZPS'] = (max_std, '[mag] max. ZP sigma (STD) of '
                                 'subimages')
            header['PC-MZPE'] = (max_err, '[mag] max. ZP error of subimages')



    # end of block: if ncalstars > 0


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in phot_calibrate')


    return zp, zp_std, zp_err, ncal_used, airmass_cal


################################################################################

def remove_files (filelist, verbose=False):

    for f in filelist:
        if isfile(f):

            if f[0:5] == 'gs://':
                cmd = ['gsutil', '-q', 'rm', f]
                result = subprocess.run(cmd)
            else:
                os.remove(f)

            if verbose:
                log.info ('removing file {}'.format(f))

    return


################################################################################

def create_objmask_gaia (fits_in, header, fits_gaia, obsdate, ra_center,
                         dec_center, fov_half_deg, rot, base, data_wcs,
                         data_bkg_std, data_mask, imtype):


    # use functions [get_gaia_image_table] and - only if the
    # corresponding fits table does not exist yet -
    # [get_imagestars_hpindex] to extract the entries from the Gaia
    # input catalog that are relevant for this image, and apply a
    # proper motion correction
    table_gaia_image = get_gaia_image_table (
        fits_in, fits_gaia, obsdate, ra_center, dec_center, fov_half_deg, rot)


    # convert input RA/DEC from table to pixel coordinates
    ra = table_gaia_image['ra'].value
    dec = table_gaia_image['dec'].value
    xcoords, ycoords = WCS(header).all_world2pix(ra, dec, 1)


    # discard entries that are not finite
    mask_finite = (np.isfinite(xcoords) & np.isfinite(ycoords))

    # and on the image
    dpix_edge = 5
    ysize, xsize = data_wcs.shape
    mask_on = ((xcoords > dpix_edge) & (xcoords < xsize-dpix_edge) &
               (ycoords > dpix_edge) & (ycoords < ysize-dpix_edge))

    # combination of finite/on-image masks
    mask_ok = mask_finite & mask_on

    # update coordinates
    xcoords = xcoords[mask_ok]
    ycoords = ycoords[mask_ok]


    # create objmask, with True entries indicating the object
    # footprints
    psfex_bintable = '{}_psf.fits'.format(base)
    [objmask_gaia] = get_psfoptflux_mp (
        psfex_bintable, data_wcs, data_bkg_std**2, data_mask, xcoords, ycoords,
        imtype=imtype, set_zogy=set_zogy, nthreads=1, tel=tel,
        get_psf_footprint=True)


    return objmask_gaia


################################################################################

def run_force_phot (fits_in, fits_gaia, obsdate, ra_center, dec_center,
                    fov_half_deg, fits_mask, nthreads, rot=0):

    """function that performs forced photometry on [fits_in] at the
    RADECs listed in [fits_gaia] with assumed column/field names 'ra'
    and 'dec' for the coordinates and 'pmra' and 'pmdec' for the
    proper motion. If the proper motion columns are not present, no
    proper motion is applied.

    """

    # import forced photometry; done here and not at the top to try to
    # avoid cyclical imports as force_phot module contains import zogy
    import force_phot


    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing run_force_phot ...')


    # use functions [get_gaia_image_table] and - only if the
    # corresponding fits table does not exist yet -
    # [get_imagestars_hpindex] to extract the entries from the Gaia
    # input catalog that are relevant for this image, and apply a
    # proper motion correction
    table_gaia_image = get_gaia_image_table (
        fits_in, fits_gaia, obsdate, ra_center, dec_center, fov_half_deg, rot)


    # update name of ra/dec columns for use in force_phot
    table_gaia_image['ra'].name = 'RA_IN'
    table_gaia_image['dec'].name = 'DEC_IN'


    # sort by magnitude
    if 'mag_G' in table_gaia_image.colnames:
        log.info ('sorting table_gaia_image by column mag_G')
        table_gaia_image.sort('mag_G')


    log.info ('selected {} Gaia stars within the image {}\nwith central ra,dec: '
              '{:.4f},{:.4f}, field-of-view: {:.3f} degrees and DATE-OBS: {}'
              .format(len(table_gaia_image), fits_in, ra_center, dec_center,
                      2*fov_half_deg, obsdate))


    # create dictionary: {image: [indices in table to consider]}
    ncoords = len(table_gaia_image)
    image_indices_dict = {fits_in: np.arange(ncoords)}


    # set various parameters to be able to run forced photometry
    nsigma = get_par(set_zogy.source_nsigma,tel)
    apphot_radii = get_par(set_zogy.apphot_radii,tel)

    # use local or global background?
    bkg_global = (get_par(set_zogy.bkg_phototype,tel) == 'global')

    # background annulus radii, objmask and limiting fraction
    bkg_radii = get_par(set_zogy.bkg_radii,tel)
    bkg_objmask = get_par(set_zogy.bkg_objmask,tel)
    bkg_limfrac = get_par(set_zogy.bkg_limfrac,tel)


    # remove PSF after measuring?
    remove_psf = get_par(set_zogy.remove_psf,tel)
    if remove_psf:
        log.warning ('PSF of object is removed from image using flux '
                     'measurement')


    # epoch to use for proper motion correction
    pm_epoch = get_par(set_zogy.cal_epoch,tel)


    # run forced photometry
    table_force_phot = force_phot.force_phot (
        table_gaia_image, image_indices_dict, mask_list=[fits_mask], trans=False,
        ref=False, fullsource=True, nsigma=nsigma, apphot_radii=apphot_radii,
        bkg_global=bkg_global, bkg_radii=bkg_radii, bkg_objmask=bkg_objmask,
        bkg_limfrac=bkg_limfrac, pm_epoch=pm_epoch, remove_psf=remove_psf,
        tel=tel, ncpus=nthreads)


    # delete columns that are not needed
    colnames = table_force_phot.colnames
    # original Gaia ra and dec columns
    #cols2delete = ['ra', 'dec']
    cols2delete = ['RA_IN', 'DEC_IN', 'FILENAME']
    # flux, fluxerr or S/N related to aperture photometry
    cols2delete += [c for c in colnames
                    if ('FLUX' in c or 'SNR' in c) and 'APER' in c]
    for colname in colnames:
        if colname in cols2delete:
            # delete column
            del table_force_phot[colname]
        else:
            # convert remaining ones to upper case, except for the
            # aperture ones, which have a - confusing! - lower case x,
            # but keep like that because of consistency with original
            # catalogs
            if 'APER' not in colname:
                table_force_phot[colname].name = colname.upper()



    # sort by SOURCE_ID
    index_sort = np.argsort(table_force_phot['SOURCE_ID'])
    table_force_phot = table_force_phot[index_sort]


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in run_force_phot')


    return table_force_phot


################################################################################

def get_imagestars_zones (fits_cal, obsdate, ra_center, dec_center,
                          fov_half_deg):

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing get_imagestars_zones ...')


    # determine cal_cat fits extensions to read using
    # [get_zone_indices] (each 1 degree strip in declination is
    # recorded in its own extension in the calibration catalog)
    zone_indices = get_zone_indices (dec_center, fov_half_deg, zone_size=60)


    # read specific extensions (=zone_indices+1) of
    # calibration catalog
    t1 = time.time()
    data_cal = read_hdulist (get_par(set_zogy.cal_cat,tel),
                             ext_name_indices=zone_indices+1)
    log.info ('read extensions {} from {} in {:.3f}s'
              .format(zone_indices, fits_cal, time.time()-t1))


    # use function [find_stars] to select stars in calibration
    # catalog that are within the current field-of-view
    index_field = find_stars (data_cal['ra'], data_cal['dec'],
                              ra_center, dec_center, fov_half_deg, search='box')


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_imagestars_zones')


    # return relevant indices as an astropy Table
    return Table(data_cal[index_field])


################################################################################

def get_gaia_image_table (fits_in, fits_gaia, obsdate, ra_center, dec_center,
                          fov_half_deg, rot):

    # use function [get_imagestars_hpindex] to extract the entries
    # from the input catalog that are relevant for this image, and
    # apply a proper motion correction - if columns pmra and pmdec are
    # present in fits_gaia - from [obsdate] to the reference epoch
    # (2016 for Gaia DR3)
    image_gaia_cat = '{}_gaia_cat.fits'.format(fits_in.split('.fits')[0])
    if not isfile(image_gaia_cat):

        table_gaia_image = get_imagestars_hpindex (fits_gaia, obsdate,
                                                   ra_center, dec_center,
                                                   fov_half_deg, rot=rot)

        # save selected stars to fits catalog so it can be used later on
        table_gaia_image.write (image_gaia_cat, format='fits', overwrite=True)

    else:

        # if it already exists, just read the table
        log.info ('reading {}'.format(image_gaia_cat))
        table_gaia_image = Table.read(image_gaia_cat)


    return table_gaia_image


################################################################################

def get_imagestars_hpindex (fits_gaia, obsdate, ra_center, dec_center,
                            fov_half_deg, rot=0):

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing get_imagestars_hpindex ...')


    # infer healpixel NSIDE from keywords LEVEL or NSIDE in the 1st
    # extension header
    with fits.open(fits_gaia, memmap=True) as hdulist:
        header = hdulist[1].header

    if 'NSIDE' in header:
        nside = header['NSIDE']
    elif 'LEVEL' in header:
        level = header['LEVEL']
        nside = 2**level
    else:
        log.error ('keywords LEVEL nor NSIDE present in 1st extension header '
                   'of {}'.format(fits_gaia))


    # determine healpix indices within radius [radius_deg] (with a 10%
    # margin) from (ra_center, dec_center)
    radius_deg = 1.1 * fov_half_deg * np.sqrt(2)
    indices = healpix_nearby (ra_center, dec_center, radius_deg,
                              nside=nside, search='circle')


    # convert integer array of healpixel indices to list with fits
    # extensions
    ext_name_indices = list(indices+1)
    # add last extension with high proper motion stars
    ext_name_indices.append(-1)
    # read indices
    t1 = time.time()
    # using astropy:
    #data_gaia = read_hdulist (fits_gaia, ext_name_indices=ext_name_indices)
    # using fitsio (faster!):
    data_gaia = read_exts_fitsio (fits_gaia, ext_name_indices=ext_name_indices)


    log.info ('read extensions {} from {} in {:.3f}s'
              .format(ext_name_indices, fits_gaia, time.time()-t1))
    mem_use (label='after reading extensions from {}'.format(fits_gaia))


    # limit stars to within the APPROXIMATE image FOV (+5 arcmin)
    index_field = find_stars (data_gaia['ra'], data_gaia['dec'],
                              ra_center, dec_center,
                              fov_half_deg+5/60, search='box', rot=rot)
    data_gaia = data_gaia[index_field]


    # apply proper motion if available
    table_gaia = Table(data_gaia)
    colnames = table_gaia.colnames
    if 'pmra' in colnames and 'pmdec' in colnames:
        gaia_epoch = get_par(set_zogy.gaia_epoch,tel)
        table_gaia = apply_gaia_pm (table_gaia, obsdate, epoch=gaia_epoch,
                                    return_table=True)


    # now limit stars to within the PRECISE image FOV
    index_field = find_stars (table_gaia['ra'].value, table_gaia['dec'].value,
                              ra_center, dec_center, fov_half_deg, search='box',
                              rot=rot)
    table_gaia = table_gaia[index_field]


    # select the unique entries as the high proper motion stars can
    # appear twice
    table_gaia = unique(table_gaia, keys='source_id', keep='first')


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_imagestars_hpindex')


    # return relevant indices
    return table_gaia


################################################################################

def healpix_nearby (ra, dec, radius_deg, nside=4, search='circle', nest=True):

    """return indices (integer array) of healpixels for given [nside]
       that overlap with circle with radius [radius_deg] or a square
       image/box with halfsize of [radius_deg]/sqrt(2), centerered at
       [ra], [dec] (floats). The circle method is slightly faster; the
       box method can lead to fewer matching healpixels, because the
       box halfsize is smaller than the circle radius.
    """

    if search == 'circle':

        # convert ra, dec to healpy vec
        vec = hp.ang2vec(ra, dec, lonlat=True)

        # return indices of healpy pixels (nested or ring) which
        # overlap with circle centered at ra,dec and with radius
        # [radius_deg]
        indices = hp.query_disc(nside=nside, vec=vec,
                                radius=np.radians(radius_deg),
                                inclusive=True, nest=nest)

    elif search == 'box':

        # center coordinates
        center = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')

        # determine 4 corners of image or box with directional offset
        # method
        ras = []
        decs = []
        for posangle in np.arange(45,360,90):
            corner = center.directional_offset_by(posangle*u.deg,
                                                  radius_deg*u.deg)
            ras.append(corner.ra.deg)
            decs.append(corner.dec.deg)

        # convert corner ras, decs to healpy vec
        vecs = (hp.ang2vec(ras, decs, lonlat=True))

        # use query_polygon method to infer indices
        indices = hp.query_polygon(nside, vecs, inclusive=True, nest=nest)


    return indices


################################################################################

def apply_gaia_pm (table_gaia, obsdate, epoch=2016.0, return_table=False,
                   remove_pmcolumns=True, ra_col='ra', dec_col='dec',
                   pmra_col='pmra', pmdec_col='pmdec'):

    """function to correct the coordinates defined by [ra_col] and
    [dec_col] in the input table [table_gaia], for the corresponding
    proper motion defined in columns [pmra_col] and [pmdec_col],
    assumed to be in units of milli-arcseconds per year.  The input
    [obsdate] needs to be readable by astropy's time.Time, e.g. fits
    format yyyy-mm-ddThh:mm:ss.sss. That date is compared to the
    catalog [epoch] - 2016 for Gaia DR3 - to calculate the offset due
    to the proper motion."""


    log.info('executing apply_gaia_pm ...')


    # delta time in years between [obsdate] and epoch
    dtime_yr = Time(obsdate).decimalyear - epoch


    # convert table columns to numpy arrays
    coords_in = SkyCoord(table_gaia[ra_col].value * u.deg,
                         table_gaia[dec_col].value * u.deg, frame='icrs')


    # offset due to proper motion in arcseconds
    offset_ra = 1e-3 * table_gaia[pmra_col].value * dtime_yr
    offset_dec = 1e-3 * table_gaia[pmdec_col].value * dtime_yr


    # use astropy's SkyCoord.spherical_offsets_by method to calculate
    # coordinates corrected for offset due to proper motion
    coords_out = coords_in.spherical_offsets_by(offset_ra *u.arcsec,
                                                offset_dec*u.arcsec)


    # return astropy Table or just the ra and dec arrays
    if return_table:

        # replace ra and dec columns with corrected coordinates
        table_gaia[ra_col] = coords_out.ra
        table_gaia[dec_col] = coords_out.dec

        # remove proper motion columns if needed
        if remove_pmcolumns:
            del table_gaia[pmra_col]
            del table_gaia[pmdec_col]

        # return Astropy table
        return table_gaia

    else:

        # return the RA and DEC arrays
        return np.array(coords_out.ra), np.array(coords_out.dec)


################################################################################

def prep_plots (table, header, base):

    log.info ('preparing photometry plots ...')

    mypsffit = get_par(set_zogy.psffit,tel)
    if mypsffit:
        flux_mypsf = table['E_FLUX_PSF']
        fluxerr_mypsf = table['E_FLUXERR_PSF']
        x_psf = table['X_PSF']
        y_psf = table['Y_PSF']


    # read a few extra header keywords needed below
    keywords = ['gain', 'exptime', 'filter', 'obsdate']
    gain, exptime, filt, obsdate = read_header(header, keywords)


    # filter arrays by FLAG
    index = ((table['E_FLUX_AUTO']>0) & (table['FLAGS']==0))
    if 'FLAGS_MASK' in table.colnames:
        index &= (table['FLAGS_MASK']==0)

    class_star = table['CLASS_STAR'][index]
    flux_auto = table['E_FLUX_AUTO'][index] * gain
    fluxerr_auto = table['E_FLUXERR_AUTO'][index] * gain
    s2n_auto = flux_auto / fluxerr_auto
    flux_opt = table['E_FLUX_OPT'][index]
    fluxerr_opt = table['E_FLUXERR_OPT'][index]

    if 'MAG_OPT' in table.colnames:
        mag_opt = table['MAG_OPT'][index]
        magerr_opt = table['MAGERR_OPT'][index]

        x_array = np.asarray(mag_opt)
        xlabel = '{} magnitude (OPT)'.format(filt)
        limits = (np.amin(x_array)-0.1, np.amax(x_array)+0.1, -0.3, 0.3)

    else:
        x_array = s2n_auto
        xlabel = 'S/N (AUTO)'
        limits = (1, 2*np.amax(x_array), -0.3, 0.3)


    x_win = table['X_POS'][index]
    y_win = table['Y_POS'][index]
    fwhm_image = table['FWHM'][index]

    if mypsffit:
        flux_mypsf = flux_psf[index]
        fluxerr_mypsf = fluxerr_psf[index]
        x_psf = x_psf[index]
        y_psf = y_psf[index]
        # write catalog with psffit to output
        table.write(fits_cat.replace('.fits','_psffit.fits'), format='fits',
                    overwrite=True)

    if 'MAG_OPT' in table.colnames:
        # histogram of all 'good' objects as a function of magnitude
        bins = np.arange(12, 22, 0.2)
        plt.hist(np.ravel(mag_opt), bins, color='tab:blue')
        x1,x2,y1,y2 = plt.axis()
        title = 'filter: {}, exptime: {:.0f}s'.format(filt, exptime)
        if 'LIMMAG' in header:
            limmag = float(header['LIMMAG'])
            plt.plot([limmag, limmag], [y1,y2], color='black', linestyle='--')
            title = (r'{}, lim. mag (5$\sigma$; dashed line): {:.2f}'
                     .format(title, limmag))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('number')
        plt.savefig('{}_magopt.pdf'.format(base))
        if get_par(set_zogy.show_plots,tel): plt.show()
        plt.close()



    # compare flux_opt with flux_auto
    title = 'rainbow colors follow CLASS_STAR: from purple (star) to red (galaxy)'
    dmag = calc_mag (flux_opt, flux_auto)
    plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                  ylabel='delta magnitude (OPT - AUTO)',
                  filename='{}_opt_vs_auto.pdf'.format(base), title=title)

    if mypsffit:
        # compare flux_mypsf with flux_auto
        dmag = calc_mag (flux_mypsf, flux_auto)
        plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                      ylabel='delta magnitude (MYPSF - AUTO)',
                      filename='{}_mypsf_vs_auto.pdf'.format(base), title=title)

        # compare flux_opt with flux_mypsf
        dmag = calc_mag (flux_opt, flux_mypsf)
        plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                      ylabel='delta magnitude (OPT - MYPSF)',
                      filename='{}_opt_vs_mypsf.pdf'.format(base), title=title)

        # compare x_win and y_win with x_psf and y_psf
        dist_win_psf = np.sqrt((x_win-x_psf)**2+(y_win-y_psf)**2)
        plot_scatter (x_array, dist_win_psf, (limits[0],limits[1],-2.,2.),
                      class_star, xlabel=xlabel,
                      ylabel='distance XY_WIN vs. XY_MYPSF',
                      filename=('{}_xyposition_win_vs_mypsf_class.pdf'
                                .format(base)), title=title)

        plot_scatter (x_array, dist_win_psf, (limits[0],limits[1],-2.,2.),
                      fwhm_image, xlabel=xlabel,
                      ylabel='distance XY_WIN vs. XY_MYPSF',
                      filename=('{}_xyposition_win_vs_mypsf_fwhm.pdf'
                                .format(base)),
                      title='rainbow colors follow FWHM')


    # compare flux_opt with flux_aper
    for i in range(len(get_par(set_zogy.apphot_radii,tel))):

        aper = get_par(set_zogy.apphot_radii[i],tel)
        field = 'E_FLUX_APER'
        field_err = 'E_FLUXERR_APER'
        field_format = 'E_FLUX_APER_R{}xFWHM'.format(aper)
        field_format_err = 'E_FLUXERR_APER_R{}xFWHM'.format(aper)

        if field in table.colnames:
            flux_aper = table[field][index,i] * gain
            fluxerr_aper = table[field_err][index,i] * gain
        elif field_format in table.colnames:
            flux_aper = table[field_format][index] * gain
            fluxerr_aper = table[field_format_err][index] * gain


        dmag = calc_mag (flux_opt, flux_aper)
        ylabel='delta magnitude (OPT - APER_R{}xFWHM)'.format(aper)
        plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                      ylabel=ylabel,
                      filename=('{}_opt_vs_aper_{}xFWHM.pdf'.format(base,aper)),
                      title=title)

        dmag = calc_mag (flux_auto, flux_aper)
        ylabel='delta magnitude (AUTO - APER_R{}xFWHM)'.format(aper)
        plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                      ylabel=ylabel,
                      filename=('{}_auto_vs_aper_{}xFWHM.pdf'.format(base, aper)),
                      title=title)

        if mypsffit:
            dmag = calc_mag (flux_mypsf, flux_aper)
            ylabel='delta magnitude (MYPSF - APER_R{}xFWHM)'.format(aper)
            plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                          ylabel=ylabel,
                          filename=('{}_mypsf_vs_aper_{}xFWHM.pdf'
                                    .format(base, aper)), title=title)


    # compare with flux_psf if psffit catalog available
    sexcat_ldac_psffit = '{}_cat_ldac_psffit.fits'.format(base)
    if isfile(sexcat_ldac_psffit):
        # read SExtractor psffit fits table
        table = Table.read(sexcat_ldac_psffit, hdu=2, memmap=True)

        flux_sexpsf = table['E_FLUX_PSF'][index] * gain

        dmag = calc_mag (flux_sexpsf, flux_opt)
        plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                      ylabel='delta {}-band magnitude (SEXPSF - OPT)',
                      filename='{}_sexpsf_vs_opt.pdf'.format(base), title=title)

        if mypsffit:
            # and compare 'my' psf with SExtractor psf
            dmag = calc_mag (flux_sexpsf, flux_mypsf)
            plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                          ylabel='delta {}-band magnitude (SEXPSF - MYPSF)',
                          filename='{}_sexpsf_vs_mypsf.pdf'.format(base),
                          title=title)

        # and compare auto with SExtractor psf
        dmag = calc_mag (flux_sexpsf, flux_auto)
        plot_scatter (x_array, dmag, limits, class_star, xlabel=xlabel,
                      ylabel='delta {}-band magnitude (SEXPSF - AUTO)',
                      filename='{}_sexpsf_vs_auto.pdf'.format(base),
                      title=title)


    return


################################################################################

def calc_mag (flux1, flux2):

    mask_ok = ((flux2 != 0) & (flux1>0) & (flux2>0))
    dmag = np.zeros_like (flux1)
    dmag[mask_ok] = -2.5*np.log10(flux1[mask_ok]/flux2[mask_ok])

    return dmag


################################################################################

def load_npy_fits (filename):

    # determine file extension
    ext = filename.split('.')[-1]

    # load numpy or fits file
    if 'fits' in ext:
        data = read_hdulist(filename)
    else:
        data = np.load(filename, mmap_mode='c')


    return data


################################################################################

def save_npy_fits (data, filename, header=None):

    """function to save [data] to a file. Depending on the extension, it
    is saved as a numpy (.npy) or fits (.fits) file. If filename is
    not provided, a temporary numpy file is created. The filename used
    is returned.

    """

    if filename is None:
        # if filename is not provided, generate temporary file
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        filename = f.name

    ext = filename.split('.')[-1]
    if 'fits' in ext:
        # save [data] as fits file
        fits.writeto (filename, data, header, overwrite=True)
    else:
        # save [data] as numpy file in numpy folder
        dir_numpy = get_par(set_zogy.dir_numpy,tel)
        if dir_numpy is not None and len(dir_numpy) > 0:

            path, name = os.path.split(filename)
            if len(path)>0:
                dir_numpy = '{}/{}'.format(path, dir_numpy)

            # make folder if it does not exist yet
            if not isdir(dir_numpy):
                os.mkdir(dir_numpy)

            # update filename
            filename = '{}/{}'.format(dir_numpy, name)


        # if file already exists, remove it
        remove_files ([filename])

        # save numpy file
        np.save(filename, data)


    # return filename
    return filename


################################################################################

def create_modify_mask (data, satlevel, data_mask=None):

    """function to identify the saturated and adjacent pixels in input
       data and add these to an existing mask, or create a new mask if
       [data_mask] is not provided

    """

    if data_mask is None:
        data_mask = np.zeros(data.shape, dtype='uint8')

    value = get_par(set_zogy.mask_value['saturated'],tel)
    mask_sat_check = (data_mask & value == value)

    # if no saturated pixels already present, add them
    if np.sum(mask_sat_check) == 0:
        mask_sat = (data >= 0.8*satlevel)
        data_mask[mask_sat] += get_par(set_zogy.mask_value['saturated'],tel)
        # pixels connected to saturated pixels
        mask_sat_adj = ndimage.binary_dilation(mask_sat,structure=
                                               np.ones((3,3)).astype('bool'),
                                               iterations=1)
        mask_sat_adj[mask_sat] = False
        data_mask[mask_sat_adj] += get_par(set_zogy.mask_value
                                           ['saturated-connected'],tel)

    return data_mask


################################################################################

def collect_zps (x_array, y_array, flux_opt, fluxerr_opt, mag_cal, airmass_cal,
                 exptime, filt, magerr_cal=None):

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing collect_zps ...')

    # calculate zeropoints using individual airmasses, since at A=2
    # the difference in airmass across the FOV is 0.1, i.e. a 5%
    # change

    # instrumental magnitudes; N.B.: input flux_opt is guaranteed to
    # be positive
    mag_opt_inst = -2.5*np.log10(flux_opt/exptime)
    pogson = 2.5/np.log(10)
    # corresponding (symmetrical) error
    magerr_opt_inst = pogson * fluxerr_opt / flux_opt

    # zeropoints; N.B.: although an airmass of 1.2 was used to infer
    # the mag_cal, this same airmass was used for the (hypothetical)
    # standard star, so the zeropoints below need not be corrected for
    # this
    zp_array = (mag_cal - mag_opt_inst +
                airmass_cal * get_par(set_zogy.ext_coeff,tel)[filt])

    # determine error in individual zeropoints
    if magerr_cal is None:
        # if magerr_cal is not provided, then set zp error equal
        # to the measurment error
        zperr_array = magerr_opt_inst
    else:
        # in case magerr_cal is provided
        zperr_array = np.sqrt(magerr_opt_inst**2 + magerr_cal**2)


    # return x_array, y_array, zp_array and zperr_array in order of
    # brightness determined by [mag_cal]
    index_sort = np.argsort(mag_cal)

    return x_array[index_sort], y_array[index_sort], zp_array[index_sort], \
        zperr_array[index_sort]



################################################################################

def calc_zp (x_array, y_array, zp_array, zperr_array, filt, imtype,
             data_shape=None, zp_type='single', boxsize=None, nsigma=2.5):

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing calc_zp ...')


    if zp_type == 'single':

        # determine weighted mean zeropoint, requiring at least 5
        # non-zero values in zp_array
        if np.sum(zp_array != 0) >= 5:

            # clip outliers
            ma_clipped = sigma_clip(zp_array, sigma=nsigma)
            mask_use = ~ma_clipped.mask
            zp, zp_std, zp_err = weighted_mean (zp_array[mask_use],
                                                zperr_array[mask_use])
            nused = np.sum(mask_use)


            # make histrogram plot if needed
            if get_par(set_zogy.make_plots,tel):

                if imtype=='new':
                    base = base_new
                else:
                    base = base_ref

                clipped_stats (zp_array[mask_use], clip_zeros=True,
                               make_hist=get_par(set_zogy.make_plots,tel),
                               name_hist='{}_zp_hist.pdf'.format(base),
                               hist_xlabel='{} zeropoint (mag)'.format(filt))


        else:
            log.warning ('could not determine zeropoints weighted mean for lack '
                         'of calibration stars (<5); returning zeros')
            zp, zp_std, zp_err, nused = (0, 0, 0, 0)


    elif zp_type == 'channels':

        # determine zeropoints of the 16 channels of the
        # MeerLICHT/BlackGEM CCD

        ncal_min_chan = get_par(set_zogy.MLBG_phot_ncal_min_chan,tel)
        zp, zp_std, zp_err, __, nused = zps_weighted_mean (
            x_array, y_array, zp_array, zperr_array, 1320, 5280,
            (2,8), ncal_min=ncal_min_chan, nsigma=nsigma)

        # arrays need to be flattened to get 1D array with index = nchan-1
        zp = zp.ravel()
        zp_std = zp_std.ravel()
        zp_err = zp_err.ravel()
        nused = nused.ravel()



    elif zp_type == 'background':

        # determine zeropoints on the scale of the input boxsize
        ysize, xsize = data_shape
        if ysize % boxsize != 0 or xsize % boxsize !=0:
            log.warning ('input boxsize in function calc_zp does not fit '
                         'integer times in image')
        nysubs = int(ysize / boxsize)
        nxsubs = int(xsize / boxsize)
        zp, zp_std, zp_err, __, nused = zps_weighted_mean (
            x_array, y_array, zp_array, zperr_array, boxsize, boxsize,
            (nysubs, nxsubs), ncal_min=5, nsigma=nsigma)


        if False:
            # arrays need to be transposed to get the mini-images arrays
            zp = zp.T
            zp_std = zp_std.T
            zp_err = zp_err.T
            nused = nused.T


    if get_par(set_zogy.verbose,tel) and zp_type=='single':
        log.info('zp: {:.3f}; zp_std: {:.3f}; zp_err: {:.3f}; nused: {}'
                 .format(zp, zp_std, zp_err, nused))

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in calc_zp')


    return zp, zp_std, zp_err, nused


################################################################################

def zps_medarray (xcoords, ycoords, zps, dx, dy, array_shape, nval_use):

    """Function that returns three arrays, each with shape [array_shape],
    with the clipped median, standard deviation and number of
    zeropoint values [zps] over rectangles with size [dx] x [dy] and
    shape [array_shape]. This is used to calculate the median
    zeropoints over the MeerLICHT/BlackGEM channels, or to create a
    "mini" image with the median zeropoints over some box size,
    similar to the mini background and standard deviation images.

    """

    # initialize output arrays
    zps_median = np.zeros(array_shape).ravel().astype('float32')
    zps_std = np.zeros(array_shape).ravel().astype('float32')
    zps_nstars = np.zeros(array_shape).ravel().astype(int)


    # determine the indices of the origins of the subimages
    nsubs = zps_median.size
    ny, nx = array_shape
    index_x = np.tile(np.arange(nx) * dx, ny)
    index_y = np.repeat(np.arange(ny) * dy, nx)


    # indices corresponding to xcoords and ycoords
    x = (xcoords-0.5).astype(int)
    y = (ycoords-0.5).astype(int)


    # loop subimages and determine median zeropoint
    for nsub in range(nsubs):

        # determine to which subimage the coordinates belong
        mask_sub = ((x >= index_x[nsub]) &  (x < index_x[nsub]+dx) &
                    (y >= index_y[nsub]) &  (y < index_y[nsub]+dy))

        # record number of stars in subimage
        zps_nstars[nsub] = np.sum(mask_sub)

        # only determine median when sufficient number of values
        # [nval_use] are available; otherwise leave it at zero
        if np.sum(mask_sub) >= nval_use:

            # pick the brightest [nval_use] calibration stars
            mean, median, std = sigma_clipped_stats (zps[mask_sub][0:nval_use]
                                                     .astype(float))
            zps_median[nsub] = median
            zps_std[nsub] = std


    # return median values with shape [array_shape]
    return (zps_median.reshape(array_shape),
            zps_std.reshape(array_shape),
            zps_nstars.reshape(array_shape))


################################################################################

def zps_weighted_mean (xcoords, ycoords, zps, zpserr, dx, dy, array_shape,
                       ncal_min=15, nsigma=2.5):

    """Function that returns 5 arrays, each with shape [array_shape],
    with the clipped weighted mean zeropoint, the corresponding
    standard deviation, the error in the weighted mean, the number of
    calibration stars available and the number of stars that were used
    in the zeropoint determination, over rectangles with size [dx] x
    [dy] and shape [array_shape]. This is used to calculate the
    weighted zeropoints over the MeerLICHT/BlackGEM channels, or to
    create a "mini" image with the weighted zeropoints over some box
    size, similar to the mini background and standard deviation
    images.

    """

    # initialize output arrays
    zps_wmean = np.zeros(array_shape).ravel().astype('float32')
    zps_wstd = np.zeros(array_shape).ravel().astype('float32')
    zps_wmeanerr = np.zeros(array_shape).ravel().astype('float32')
    zps_nstars = np.zeros(array_shape).ravel().astype(int)
    zps_nused = np.zeros(array_shape).ravel().astype(int)


    # determine the indices of the origins of the subimages
    nsubs = zps_wmean.size
    ny, nx = array_shape
    index_x = np.tile(np.arange(nx) * dx, ny)
    index_y = np.repeat(np.arange(ny) * dy, nx)


    # indices corresponding to xcoords and ycoords
    x = (xcoords-0.5).astype(int)
    y = (ycoords-0.5).astype(int)


    # loop subimages and determine clipped weighted zeropoint
    for nsub in range(nsubs):

        # mask that identifies coordinates in current subimage
        mask_sub = ((x >= index_x[nsub]) &  (x < index_x[nsub]+dx) &
                    (y >= index_y[nsub]) &  (y < index_y[nsub]+dy))

        # number of stars available
        zps_nstars[nsub] = np.sum(mask_sub)


        # only determine zeropoint if sufficient number of values
        # [ncal_min] are available; otherwise leave it at zero
        if np.sum(mask_sub) >= ncal_min:

            # clip subimage outliers
            ma_clipped = sigma_clip(zps[mask_sub], sigma=nsigma)
            mask_use = ~ma_clipped.mask

            # weighted mean
            zps_use = zps[mask_sub][mask_use]
            zpserr_use = zpserr[mask_sub][mask_use]
            zps_wmean[nsub], zps_wstd[nsub], zps_wmeanerr[nsub] = \
                weighted_mean (zps_use, zpserr_use)

            # number of stars used in the zeropoint determination
            zps_nused[nsub] = len(zps_use)



    # return median values with shape [array_shape]
    return (zps_wmean.reshape(array_shape),
            zps_wstd.reshape(array_shape),
            zps_wmeanerr.reshape(array_shape),
            zps_nstars.reshape(array_shape),
            zps_nused.reshape(array_shape))


################################################################################

def weighted_mean (mag, magerr, magerr_wlim=0.001):

    """calculate weighted mean of sample provided by the arrays [mag]
       and [magerr], where the weighting is performed in flux
       units. The maximum weight is defined by [magerr_wlim], so that
       the brightest stars are not too dominant. The following three
       scalars are returned: weighted mean, the weighted standard
       deviation and the error on the weighted mean, in units of
       magnitudes.

       (June 2024): added method from dreamreduce
       (zeropoints.find_sigma) to correct for the fact that the flux
       errors are likely underestimated, by finding a single constant
       error to all magnitude errors such that the reduced chi2 of the
       distribution of values will become unity

    """


    # using [find_sigma], determine additional sigma or magnitude
    # error (due to measurement errors likely being an underestimate
    # of the true errors), which is found by adding a single constant
    # error to all magnitude errors such that the reduced chi2 of the
    # distribution of values will become unity
    try:
        res = mag - np.median(mag)
        sigma = find_sigma (res, magerr)
    except Exception as e:
        # in case of an exception in find_sigma, set sigma to zero
        log.warning ('following exception occurred in zogy.find_sigma, '
                     'setting sigma to zero: {}'.format(e))
        sigma = 0


    # total magnitude error
    magerrtot = np.sqrt(magerr**2 + sigma**2)


    # convert mag and magerrtot to flux
    pogson = 2.5/np.log(10)
    flux = 10**(-0.4*mag)
    fluxerrtot = flux * magerrtot / pogson


    # initialize weights to fluxerrtot corresponding to magerr_wlim
    # (which is in magnitude units)
    fluxerr_min = flux * magerr_wlim / pogson
    weights = 1/fluxerr_min**2
    # set usual weights for stars with flux errors above magerr_wlim
    # threshold
    mask = (pogson*fluxerrtot/flux > magerr_wlim)
    weights[mask] = 1/fluxerrtot[mask]**2


    # weighted mean flux and magnitude
    flux_wmean = np.average(flux, weights=weights)
    mag_wmean = -2.5*np.log10(flux_wmean)


    # weighted sample standard deviation
    flux_wstd = np.sqrt(np.sum(weights * (flux - flux_wmean)**2) /
                        np.sum(weights))
    mag_wstd = pogson * flux_wstd / flux_wmean


    # error in the weighted mean
    flux_wmeanerr = np.sqrt(1/np.sum(weights))
    mag_wmeanerr = pogson * flux_wmeanerr / flux_wmean


    return mag_wmean, mag_wstd, mag_wmeanerr


################################################################################

def find_sigma (res, err, axis=None, maxiter=15, epsilon=1e-2):


    # CHECK!!! - for debugging purposes
    #log.info ('res: {}'.format(res))
    #log.info ('np.shape(res): {}'.format(np.shape(res)))
    #log.info ('err: {}'.format(err))
    #log.info ('np.shape(err): {}'.format(np.shape(err)))


    # with too few valid values in case axis is None, return zero
    # immediately
    if axis is None and np.ma.count(res)<2:
        return 0


    # Search for a solution between 0 and 2.
    err1 = 0
    err2 = 2


    # flatten arrays if needed
    if axis is None:
        res = np.ravel(res)
        err = np.ravel(err)
    else:
        # err1 and 2 (and output err3) have one dimension less than
        # res and err
        err1 += np.zeros(np.sum(res, axis=axis, keepdims=True).shape)
        err2 += np.zeros_like(err1)



    # determined reduced chi-square with respect to the median at
    # beginning of interval
    chi2red = get_chi2red (res, err, err1, axis=axis)
    #log.info ('chi2red begin interval: {}'.format(chi2red))

    # if chi2red is still below 1, even when a zero sigma was added to
    # the individual errors, return err1 if chi2red is scalar or
    # create mask to set the corresponding values to zero, further
    # below
    if np.isscalar(chi2red):
        # boolean indicating that chi2red, err1, err2 and err3 are
        # scalars
        scalar = True
        if chi2red <= 1:
            return err1
    else:
        scalar = False
        mask1 = (chi2red <= 1)


    # determined reduced chi-square with respect to the median at end
    # of interval
    chi2red = get_chi2red (res, err, err2, axis=axis)
    #log.info ('chi2red end interval: {}'.format(chi2red))
    # if chi2red is still below 1, even when a zero sigma was added to
    # the individual errors, return err2 if chi2red is scalar or
    # create mask to set the corresponding values to zero, further
    # below
    if scalar:
        if chi2red > 1:
            return err2
    else:
        mask2 = (chi2red > 1)


    # Find the solution.
    for niter in range(maxiter):

        err3 = (err1 + err2)/2
        chi2red = get_chi2red (res, err, err3, axis=axis)

        if scalar:

            if chi2red > 1:
                # if chi2red > 1: move lower error up to err3 to decrease
                # chi2 in next iteration
                err1 = err3
            else:
                # if chi2red < 1: move upper error down to err3 to
                # increase chi2red in next iteration
                err2 = err3

        else:

            # use mask if not scalars
            mask = (chi2red > 1)
            err1[mask] = err3[mask]
            err2[~mask] = err3[~mask]



        # break if chi2red - 1 is sufficiently close to 1
        if np.max(np.abs(chi2red-1)) < epsilon:
            break



    err3 = (err2 + err1)/2

    if not scalar:
        err3[mask1] = 0
        err3[mask2] = 2


    if not np.isscalar(err3) and err3.size == 1:
        err3 = err3[0]


    return err3


################################################################################

def get_chi2red (res, err, sigma=0, axis=None):

    # flatten arrays if needed
    if axis is None:
        res = np.ravel(res)
        err = np.ravel(err)


    # median along axis
    med = np.ma.median(res, axis=axis, keepdims=True)


    # determine chi-square
    weights = 1/(err**2 + sigma**2)
    chi2 = np.sum(weights * (res-med)**2, axis=axis, keepdims=True)


    # nobs; np.ma.count also works for non-masked arrays
    nobs = np.ma.count(res, axis=axis, keepdims=True)


    # reduced chi2
    chi2red = chi2 / np.maximum(nobs-1, 1)


    # if chi2red is a 1-d array with a single item, convert it to a
    # scalar
    if axis is None and not np.isscalar(chi2red) and chi2red.size == 1:
        chi2red = chi2red[0]


    return chi2red


################################################################################

def apply_zp (flux, zp, airmass, exptime, ext_coeff, fluxerr=None,
              zperr=None, xcoords=None, ycoords=None, return_fnu=False):

    """Function that converts the array [flux in e-] into calibrated
    magnitudes using [zp in mag] (a scalar or array with same size as
    [flux]), [airmass] (scalar or array with the same size as [flux]),
    exptime (scalar) and filter-specific [ext_coeff].

    If [fluxerr] is provided, the function will also return the
    magnitude errors. If [zperr in mag] is provided, an additional
    total magnitude error is also returned, which is the quadratic
    average of fluxerr and zperr. The output will be numpy arrays with
    the same number of elements as the input flux.

    If [return_fnu] is True, then Fnu in microJy (and the
    corresponding Fnu and total error if relevant) are also returned
    alongside the magnitude (and its errors if relevant). If both
    fluxerr and zperr are not None, the arrays will be returned in the
    following order:

        mag, magerr, magerrtot, fnu, fnuerr, fnuerrtot

    """

    log.info('executing apply_zp ...')


    # Pogson constant
    pogson = 2.5/np.log(10)


    # make sure input fluxes are numpy arrays
    flux = np.asarray(flux)
    if fluxerr is not None:
        # and that fluxerr is not negative
        fluxerr = np.abs(fluxerr)


    # instrumental magnitudes
    mag_inst = np.zeros_like(flux)
    mask_pos = (flux > 0)
    mag_inst[mask_pos] = -2.5*np.log10(flux[mask_pos]/exptime)

    # convert the instrumental mags; N.B.: airmass correction is
    # relative to airmass at which atmospheric extinction was already
    # included in the calibration catalog
    mag = zp + mag_inst - airmass * ext_coeff
    # set magnitudes of sources with non-positive fluxes to 99
    mag[~mask_pos] = 99


    # start output list
    list2return = [mag]


    # determine magerr if [fluxerr] is provided
    if fluxerr is not None:

        magerr = np.zeros_like(flux)
        magerr[mask_pos] = pogson * fluxerr[mask_pos] / flux[mask_pos]
        list2return.append(magerr)


        # determine magerrtot if [zperr] is provided
        if zperr is not None:
            magerrtot = np.sqrt(magerr**2 + zperr**2)
            list2return.append(magerrtot)



    if return_fnu:

        # also infer Fnu in microJy
        fnu = (flux/exptime) * 10**(-0.4*(zp - airmass * ext_coeff - 23.9))
        list2return.append(fnu)


        # determine fnuerr if [fluxerr] is provided
        if fluxerr is not None:

            fnuerr = np.zeros_like(flux)
            nz = (flux != 0)
            fnuerr[nz] = np.abs(fnu[nz] * fluxerr[nz] / flux[nz])
            list2return.append(fnuerr)


            # determine fnuerrtot if [zperr] is provided
            if zperr is not None:
                # fnuerrtot should not be inferred from magerrtot, as
                # that is not defined for negative fluxes
                fnuerrtot = np.sqrt(fnuerr**2 + (zperr*fnu/pogson)**2)
                list2return.append(fnuerrtot)



    if len(list2return)==1:
        return list2return[0]
    else:
        return list2return



################################################################################

def apply_zp_mag2flux (mag, zp, airmass, exptime, ext_coeff, magerr=None):

    """Function that converts the calibrated magnitudes [mag] back to
    total flux in e- (not per second!) using [zp] (a scalar),
    [airmass] (scalar or array with the same size as [mag]), exptime
    (scalar) and filter-specific [ext_coeff]. If [magerr] is provided,
    the function will also return the flux errors.

    """

    log.info('executing apply_zp_mag2flux ...')

    # make sure input magnitudes are numpy arrays
    mag = np.asarray(mag)
    if magerr is not None:
        magerr = np.asarray(magerr)

    # instrumental magnitudes, assuming they are all positive
    mag_inst = mag - zp + airmass * ext_coeff

    # convert instrumental magnitudes to fluxes
    flux = exptime * 10**(-0.4*mag_inst)

    # and determine errors if [magerr] is provided
    if magerr is not None:
        pogson = 2.5/np.log(10)
        fluxerr = magerr * flux / pogson


    if magerr is not None:
        return flux, fluxerr
    else:
        return flux


################################################################################

def apply_zp_fnu2flux (fnu, zp, airmass, exptime, ext_coeff, fnuerr=None):

    """Function that converts the calibrated fluxes [fnu] back to
    total flux in e- (not per second!) using [zp] (a scalar),
    [airmass] (scalar or array with the same size as [fnu]), exptime
    (scalar) and filter-specific [ext_coeff]. If [fnuerr] is provided,
    the function will also return the flux errors.

    """

    log.info('executing apply_zp_fnu2flux ...')

    # convert total electron flux to Fnu
    flux = fnu * exptime / 10**(-0.4*(zp - airmass * ext_coeff - 23.9))

    if fnuerr is None:
        return flux
    else:
        fluxerr = flux * fnuerr / fnu
        return flux, fluxerr


################################################################################

def fnu2mag (fnu, fnuerr=None, fnuerrtot=None):

    # Pogson constant
    pogson = 2.5/np.log(10)


    # mask of positive fluxes
    mask_pos = (fnu > 0)
    mag = np.zeros_like(fnu)
    mag[mask_pos] = -2.5 * np.log10(fnu[mask_pos]) + 23.9
    # set mags of negative fluxes to +99
    mag[~mask_pos] = 99
    list2return = [mag]


    if fnuerr is not None:
        magerr = pogson * np.abs(fnuerr/fnu)
        # set magerr of negative fluxes to 99?
        #magerr[~mask_pos] = 99
        list2return.append(magerr)


    if fnuerrtot is not None:
        magerrtot = pogson * np.abs(fnuerrtot/fnu)
        # set magerrtot of negative fluxes to 99?
        #magerrtot[~mask_pos] = 99
        list2return.append(magerrtot)


    if len(list2return)==1:
        return list2return[0]
    else:
        return list2return


################################################################################

def get_zp_coords (xcoords, ycoords, zp_chan, zp):

    """function to return array of zeropoints for each object at array
    of pixel coordinates (xcoords, ycoords) based on the channel
    zeropoints [zp_chan] and to which channel the object belongs. For
    coordinates off the image, [zp] is returned.

    If all coordinates would be positive, this function would simplify
    to: zps = zp_chan[coords2chan(xcoords, ycoords) - 1]

    """

    # initialize output array
    zps = np.zeros_like(xcoords)

    # make sure zp_chan is a numpy array
    zp_chan = np.array(zp_chan)

    # use function coords2chan to infer channel number for input
    # coordinates
    chan_number = coords2chan(xcoords, ycoords)

    # assign channel zeropoint to coordinates on the image
    mask_on = (chan_number != 0)
    zps[mask_on] = zp_chan[chan_number[mask_on] - 1]

    # assign image zp to coordinates off the image
    zps[~mask_on] = zp


    return zps


################################################################################

def coords2chan (xcoords, ycoords):

    """Function to convert pixel coordinates (arrays) to channel
    number (not indices!) for MeerLICHT/BlackGEM images.

    The center of the bottom left pixel has xcoord,ycoord=1,1. The
    channel numbering is defined as follows when viewing an image
    (before any WCS rotation is applied):

       ---------------------------
       | 09 10 11 12 13 14 15 16 |
       | 09 10 11 12 13 14 15 16 |
       | 09 10 11 12 13 14 15 16 |
       | 09 10 11 12 13 14 15 16 |
       | 01 02 03 04 05 06 07 08 |
       | 01 02 03 04 05 06 07 08 |
       | 01 02 03 04 05 06 07 08 |
       | 01 02 03 04 05 06 07 08 |
       ---------------------------

    Should the coordinates be off the image, i.e. lower than 0.5 or
    greater than or equal to 10560.5, 0 is returned.

    """

    # set image size, number of channels and channel size for
    # MeerLICHT/BlackGEM CCD
    ysize, xsize = 10560, 10560
    nx, ny = 8, 2
    dx = xsize // nx
    dy = ysize // ny

    # get channel number using integer division
    chan_numbers = ((xcoords-0.5) // dx +
                    nx * ((ycoords-0.5) // dy) + 1).astype('uint8')

    # return zero for coordinates off the image
    mask_zero = ((xcoords < 0.5) | (xcoords >= xsize+0.5) |
                 (ycoords < 0.5) | (ycoords >= ysize+0.5))
    chan_numbers[mask_zero] = 0


    return chan_numbers


################################################################################

def find_stars (ra_cat, dec_cat, ra, dec, dist, rot=0, search='box', sort=False):

    """find entries in [ra_cat] and [dec_cat] within [dist] of [ra] and
       [dec]; all in degrees
    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing find_stars ...')

    # make a big cut in arrays ra_cat and dec_cat to speed up; use a
    # healthy margin in dec, as images near pole are curved in dec,
    # i.e. the image corners have different DECs than the pixels in
    # between the corners
    index_cut = np.nonzero(np.abs(dec_cat-dec)<=1.5*dist)[0]
    ra_cat_cut = ra_cat[index_cut]
    dec_cat_cut = dec_cat[index_cut]

    # separation in degrees; haversine is equivalent to astropy
    # separation method
    sep = haversine(ra_cat_cut, dec_cat_cut, ra, dec)

    if search=='circle':
        # find within circle:
        mask_dist = (sep<=dist)

    else:
        # field center
        center = SkyCoord(ra*u.deg, dec*u.deg)

        # targets
        targets = SkyCoord(ra=ra_cat_cut*u.deg, dec=dec_cat_cut*u.deg)

        # revert to old method for zero rotation
        if rot==0:
            dra, ddec = center.spherical_offsets_to(targets)
            mask_dist = ((abs(dra.deg) <= dist) & (abs(ddec.deg) <= dist))
        else:
            # create skyoffset frame centered at center with rotation, see
            # https://docs.astropy.org/en/stable/coordinates/matchsep.html#sky-offset-frames
            frame_rot = center.skyoffset_frame(rotation=Angle(rot*u.deg))

            # offsets between targets and center in rotated frame
            offsets = targets.transform_to(frame_rot)

            # determine which targets are on the rotated frame
            mask_dist = ((abs(offsets.lon.deg) <= dist) &
                         (abs(offsets.lat.deg) <= dist))


    # indices of sources in input catalog within circle or box
    index_dist = index_cut[mask_dist]

    # sort indices in distance if needed
    if sort:
        index_sort = np.argsort(sep[mask_dist])
        index_dist = index_dist[index_sort]


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in find_stars')


    return index_dist


################################################################################

def find_stars_orig (ra_cat, dec_cat, ra, dec, dist, search='box',
                     sort=False):

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing find_stars ...')

    # find entries in [ra_cat] and [dec_cat] within [dist] of
    # [ra] and [dec]
    # make a big cut in arrays ra_cat and dec_cat to speed up
    index_cut = np.nonzero(np.abs(dec_cat-dec)<=dist)[0]
    ra_cat_cut = ra_cat[index_cut]
    dec_cat_cut = dec_cat[index_cut]

    if search=='circle':
        # find within circle:
        dsigma = haversine(ra_cat_cut, dec_cat_cut, ra, dec)
        mask_dist = (dsigma<=dist)
    else:
        # find within box:
        dsigma_ra = haversine(ra_cat_cut, dec_cat_cut, ra, dec_cat_cut)
        dsigma_dec = np.abs(dec_cat_cut-dec)
        mask_dist = ((dsigma_ra<=dist) & (dsigma_dec<=dist))

        if sort:
            dsigma = np.sqrt(dsigma_ra**2 + dsigma_dec**2)


    # now select only entries within circle or box
    index_dist = index_cut[0][mask_dist]

    # sort indices in distance if needed
    if sort:
        index_sort = np.argsort(dsigma[mask_dist])
        index_dist = index_dist[index_sort]

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in find_stars')

    return index_dist


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

    """Returns index of declination zone for input [dec], where zones
    start at declination -90 and are [zone_size] arcminutes in size.

    """

    #return np.floor((90 + dec)/0.0083333).astype(int)
    return np.floor((90 + dec)*(60./zone_size)).astype(int)
    # see http://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_external_catalogues/ssec_dm_panstarrs1_original_valid.html


################################################################################

def get_zone_indices (dec_center, fov_half_deg, zone_size=60):

    """Returns numpy integer array of declination zone indices (of size
    [zone_size] arcminutes) that are covered by a telescope pointing
    at declination [dec_center] with a FOV of 2 x [fov_half_deg].

    """
    zone_min = zone(max(-89.9999,dec_center-fov_half_deg), zone_size=zone_size)
    zone_max = zone(min(+89.9999,dec_center+fov_half_deg), zone_size=zone_size)
    # zone_max+1 ensures that zone_max is included
    return np.arange(zone_min, zone_max+1)


################################################################################

def get_airmass (ra, dec, obsdate, lat, lon, height, get_altaz=False):

    log.info('executing get_airmass ...')

    location = EarthLocation(lat=lat, lon=lon, height=height)
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    coords_altaz = coords.transform_to(AltAz(obstime=Time(obsdate),
                                             location=location))

    # take care to convert Quantity coords_altaz.secz to numpy float
    # or array
    airmass = coords_altaz.secz.value

    if get_altaz:
        return airmass, coords_altaz.alt.deg, coords_altaz.az.deg
    else:
        return airmass


################################################################################

def fixpix (data, satlevel=60000, data_bkg_std=None, data_mask=None,
            imtype=None, header=None, mask_value=None, timing=True, base=None,
            keep_tmp=False, along_row=True, interp_func='spline',
            interp_dpix=5):

    """input data is assumed to be background subtracted"""


    if timing: t = time.time()
    log.info('executing fixpix ...')

    #data_fixed = np.copy(data)
    # no need for a copy
    data_fixed = data

    # replace edge pixels with zero
    value_edge = mask_value['edge']
    mask_edge = (data_mask & value_edge == value_edge)
    data_fixed[mask_edge] = 0
    # or with the background
    #data_fixed[mask_edge] = data_bkg[mask_edge]

    # now try to clean the image from artificially sharp features such
    # as saturated pixels as defined in data_mask - the FFTs in
    # [run_ZOGY] produce large-scale features surrounding these sharp
    # features in the subtracted image.

    # determine pixels to be replaced, starting with saturated ones
    value_sat = mask_value['saturated']
    #mask_2replace = (data_mask & value_sat == value_sat)
    value_satcon = mask_value['saturated-connected']
    mask_2replace = ((data_mask & value_sat == value_sat) |
                     (data_mask & value_satcon == value_satcon))

    if False:

        # consider pixels with values higher than 2/3 of the saturation
        # level to be saturated-connected pixels
        mask_connected = (data >= 0.67*satlevel)
        # grow mask_2replace to include these
        mask_2replace = ndimage.morphology.binary_propagation (
            mask_2replace, mask=mask_connected)

        # add one more line of pixels
        struct = np.ones((3,3), dtype=bool)
        mask_2replace = ndimage.binary_dilation(mask_2replace, structure=struct)


    # do not consider saturated(-connected) pixels on the edge
    mask_2replace &= ~mask_edge

    # fix saturated and saturated-connected pixels with function
    # [inter_pix] that interpolates these pixels with a spline or
    # gauss fit using [dpix] on either side
    if data_bkg_std is not None:
        data_err = np.sqrt(np.maximum(data_fixed,0) + data_bkg_std**2)
    else:
        if header is not None:
            # get readnoise from header
            #rdnoise = header['RDNOISE']
            rdnoise = read_header(header, ['rdnoise'])
        else:
            rdnoise = 10

        data_err = np.sqrt(np.maximum(data_fixed,0)+rdnoise**2)


    data_fixed = inter_pix (data_fixed, data_err, mask_2replace, k=3,
                            dpix=interp_dpix, along_row=along_row,
                            interp_func=interp_func)


    if False:

        # add a couple more pixels to look for negative values
        struct = np.ones((5,5), dtype=bool)
        mask_2replace = ndimage.binary_dilation(mask_2replace, structure=struct)

        # set values in [data_fixed[mask_2replace]] that turn out to be
        # lower than zero (e.g. due to spline fit starting on a wing of a
        # star) equal to the median of the surrounding pixels
        mask_neg = (mask_2replace & (data_fixed < 0))
        size_filter = 5
        data_fixed = fill_mask(data_fixed, size_filter, mask_2fill=mask_neg,
                               mask_valid=~mask_edge, use_median=True, smooth=False)


    if False:
        # try using astropy convolution to replace bad pixels using a
        # Gaussian2D kernel
        kernel = Gaussian2DKernel(1, x_size=3, y_size=3)

        # replace bad pixels with nans
        value_bad = mask_value['bad']
        mask_bad = (data_mask & value_bad == value_bad)
        data_fixed[mask_bad] = np.nan

        # astropy's convolution replaces the NaN pixels with a kernel-weighted
        # interpolation from their neighbors
        it=0
        while np.isnan(data_fixed).any():
            it+=1
            log.info ('iteration: {}'.format(it))
            #data_fixed = convolve(data_fixed, kernel)
            data_fixed = interpolate_replace_nans(data_fixed, kernel)
            log.info ('np.sum(np.isnan(data_fixed)): {}'
                      .format(np.sum(np.isnan(data_fixed))))


    if base is not None and keep_tmp:
        fits.writeto('{}_pixfixed.fits'.format(base), data_fixed, overwrite=True)


    if timing:
        log_timing_memory (t0=t, label='in fixpix')

    return data_fixed


################################################################################

def inter_pix (data, data_std, mask_2replace, dpix=7, k=3, along_row=True,
               interp_func='spline', order=2, fit_neg_values=False):

    """Function to replace mask pixels with spline fit along row or column"""

    #data_replaced = np.copy(data)
    data_replaced = np.copy(data)

    # if [data_std] is a scalar, convert it to an array with the same
    # shape as data
    if np.isscalar(data_std):
        data_std *= np.ones(np.shape(data))

    # label consecutive pixels in x as distinct region
    if along_row:
        struct = [[0,0,0],[1,1,1],[0,0,0]]
    else:
        struct = [[0,1,0],[0,1,0],[0,1,0]]

    regions, nregions = ndimage.label(mask_2replace, structure=struct)
    # determine slices corresponding to these regions
    obj_slices = ndimage.find_objects(regions)

    # iterate over regions to replace
    ymax, xmax = np.shape(data)

    for i in range(nregions):

        y_slice = obj_slices[i][0]
        x_slice = obj_slices[i][1]

        # _absc and _ordi refer to abscissa and ordinate of the spline
        # fit to be performed; depending on the input [along_row] the
        # abscissa and/or ordinate is along the x- and/or y-axis of
        # [data]
        if along_row:
            absc_slice = x_slice
            absc_max = xmax
            ordi_slice = y_slice
        else:
            absc_slice = y_slice
            absc_max = ymax
            ordi_slice = x_slice

        i_start = max(absc_slice.start-dpix, 0)
        i_stop = min(absc_slice.stop+dpix, absc_max)
        indx_absc = np.arange(i_start, i_stop)
        indx_ordi = ordi_slice.start

        if along_row:
            indx_x = indx_absc
            indx_y = indx_ordi
        else:
            indx_x = indx_ordi
            indx_y = indx_absc


        mask_absc = mask_2replace[indx_y, indx_x]
        data_absc = data_replaced[indx_y, indx_x]
        data_std_absc = data_std[indx_y, indx_x]

        # data to fit
        mask_fit = ~mask_absc
        # also avoid negative values, depending on the input
        # [fit_neg_values]
        if not fit_neg_values:
            mask_fit &= (data_absc>=0)

        if np.sum(mask_fit) < 2:
            continue

        x_fit = indx_absc[mask_fit]
        y_fit = data_absc[mask_fit]
        w_fit = 1./data_std_absc[mask_fit]

        try:

            if interp_func == 'spline':
                # spline fit
                fit = interpolate.UnivariateSpline(x_fit, y_fit, w=w_fit, k=k,
                                                   check_finite=True)
            elif interp_func == 'gauss':

                # alternatively, fit a 1d gauss
                params = Parameters()
                params.add('amp', value=np.amax(y_fit), min=0, #max=1.5e5,
                           vary=True)
                hdpix = int(dpix/2)
                params.add('mean', value=np.mean(indx_absc[mask_absc]),
                           min=indx_absc[hdpix], max=indx_absc[-hdpix],
                           vary=True)
                params.add('sigma', value=3., min=2, max=20, vary=True)
                params.add('a0', value=np.amin(y_fit), min=0, vary=True)
                #params.add('a1', value=0, vary=True)

                # do leastsq polynomial fit including z_err
                yerr_fit = data_std_absc[mask_fit]
                result = minimize (gauss2min, params, method='least_squares',
                                   args=(x_fit, y_fit, yerr_fit), max_nfev=100)

                p = list(result.params.valuesdict().values())
                fit = lambda x: gauss1d(p, x)

                #log.info (fit_report(result))

            elif interp_func == 'poly':

                p = np.polyfit(x_fit, y_fit, order)
                fit = lambda x: np.polyval(p, x)

            else:
                log.error ('interpolating function [interp_func] in function '
                           '[inter_pix] needs to be either \'spline\', \'gauss\''
                           'or \'poly\'')


        except Exception as e:
            log.exception(traceback.format_exc())
            log.exception('spline fit in [inter_pix] to region {} with '
                          'slice {} failed; pixel values not updated'
                          .format(i, obj_slices[i]))



        else:
            # replace masked entries with interpolated values
            absc_fill = indx_absc[mask_absc]
            fit_fill = np.maximum(fit(absc_fill), 0)
            if along_row:
                data_replaced[indx_ordi, absc_fill] = fit_fill
            else:
                data_replaced[absc_fill, indx_ordi] = fit_fill


        finally:

            if False:

                if along_row:
                    ycoord = indx_ordi + 1
                    xcoord = np.median(indx_absc)+1
                else:
                    xcoord = indx_ordi + 1
                    ycoord = np.median(indx_absc)+1

                log.info ('xcoord: {}, ycoord: {}'.format(xcoord, ycoord))

                #if True or np.sqrt((xcoord-10387)**2 + (ycoord-10212)**2) < 100:
                if True:

                    plt.errorbar (indx_absc+1, data_absc, data_std_absc,
                                  linestyle='None', ecolor='k', capsize=2)
                    plt.plot(indx_absc+1, data_absc, 'o', color='tab:blue',
                             markersize=5, markeredgecolor='k')
                    plt.plot(indx_absc[mask_absc]+1, data_absc[mask_absc], 'ro',
                             markersize=5, markeredgecolor='k')
                    indx_plot = np.linspace(indx_absc[0], indx_absc[-1], num=100)
                    plt.plot(indx_plot+1, fit(indx_plot), 'g-')
                    plt.xlabel('pixel position')
                    plt.ylabel('pixel value')
                    plt.title('xcoord, ycoord: {}, {}'.format(xcoord, ycoord))
                    plt.show()
                    plt.close()


    return data_replaced


################################################################################

def get_back_orig (data, header, objmask, imtype, clip=True, fits_mask=None):

    """Function that returns the background of the image [data]. A clipped
    median is determined for each subimage which is masked using the
    object mask (created from SExtractor's '-OBJECTS' image, where
    objects have zero values). The subimages (with size:
    [set_zogy.bkg_boxsize]) are then median filtered and resized to
    the size of the input image.

    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing get_back ...')

    # use the SExtractor '-OBJECTS' image, which is a (SExtractor)
    # background-subtracted image with all pixels where objects were
    # detected set to zero (-OBJECTS), as the object mask

    # mask all pixels with zeros in [data_objmask], which is objmask
    # (i.e. a True in objmask corresponds to 0 in data_objmask)
    mask_reject = np.copy(objmask)

    # also reject any masked pixel from input mask
    if fits_mask is not None:
        data_mask = read_hdulist (fits_mask, dtype='uint8')
        mask_reject |= (data_mask > 0)


    # reshape the input image in a 3D array of shape (nysubs, nxsubs,
    # number of pixels in background box), such that when taking
    # median along last axis, the mini image with the medians of the
    # subimages are immediately obtained; this needs to be done in a
    # masked array to be able to discard pixels in the object mask
    # (note that masked arrays consider mask values of True to be
    # invalid, i.e. are not used)
    data_masked = np.ma.masked_array(data, mask=mask_reject)

    # image size and number of background boxes along x and y
    ysize, xsize = data.shape
    boxsize = get_par(set_zogy.bkg_boxsize,tel)
    if ysize % boxsize != 0 or xsize % boxsize !=0:
        log.info('Warning: [set_zogy.bkg_boxsize] does not fit integer times in image')
        log.info('         remaining pixels will be edge-padded')
    nysubs = int(ysize / boxsize)
    nxsubs = int(xsize / boxsize)

    # reshape
    data_masked_reshaped = data_masked.reshape(
        nysubs,boxsize,-1,boxsize).swapaxes(1,2).reshape(nysubs,nxsubs,-1)
    # could also use skimage.util.shape.view_as_blocks:
    #block_shape = (boxsize, boxsize)
    #data_masked_reshaped = view_as_blocks(data_masked, block_shape).reshape(
    #    nysubs, nxsubs, -1)


    # get clipped statistics; if background box is big enough, do the
    # statistics on a random subset of pixels
    if boxsize > 300:
        index_stat = get_rand_indices((data_masked_reshaped.shape[2],))
        __, mini_median, mini_std = sigma_clipped_stats (
            data_masked_reshaped[:,:,index_stat].astype(float),
            sigma=get_par(set_zogy.bkg_nsigma,tel), axis=2, mask_value=0)
    else:
        __, mini_median, mini_std = sigma_clipped_stats (
            data_masked_reshaped.astype(float),
            sigma=get_par(set_zogy.bkg_nsigma,tel), axis=2, mask_value=0)


    # minimum fraction of background subimage pixels not to be
    # affected by the OR combination of object mask and image mask
    mask_minsize = 0.5*get_par(set_zogy.bkg_boxsize,tel)**2
    # if number of valid pixels along axis 2 is less than [mask_minsize]
    # then set that background box median to zero
    sum_bad_pixels = np.sum(data_masked_reshaped.mask, axis=2)
    mask_zero = (sum_bad_pixels >= mask_minsize)
    mini_median[mask_zero] = 0
    mini_std[mask_zero] = 0
    # set any possible nans to zero
    #mini_median[~np.isfinite(mini_median)] = 0
    #mini_std[~np.isfinite(mini_std)] = 0


    # for ML/BG images, determine relative correction factors between
    # different channels, possibly due to non-linearity, from the mini
    # background images
    if False and tel[0:2] in ['ML','BG']:

        # channel correction factors determined from low order 2D
        # polynomial fit and minimizing the residuals
        bkg_corr = bkg_corr_MLBG (mini_median, mini_std, data, header,
                                  correct_data=True, order=2)

        # save data to fits image if background correction was applied
        if bkg_corr:
            if imtype=='new':
                base = base_new
            else:
                base = base_ref

            fits.writeto ('{}.fits'.format(base), data, header, overwrite=True)


    # fill zeros and smooth mini images with median or mean filter
    # using function [fill_mask]
    size_filter = get_par(set_zogy.bkg_filtersize,tel)

    if tel not in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']:

        mini_median_filt = fill_zeros_filter (mini_median, size_filter,
                                              use_median=False)
        mini_std_filt = fill_zeros_filter (mini_std, size_filter,
                                           use_median=False)

    else:
        # for ML/BG this needs to be done per channel separately, to
        # avoid channels with different average count levels affecting
        # each other
        mini_shape = mini_median.shape
        mini_sec = get_section_MLBG (mini_shape)
        nchans = np.shape(mini_sec)[0]

        # prepare output arrays and loop channels
        mini_median_filt = np.zeros_like (mini_median)
        mini_std_filt = np.zeros_like (mini_std)
        for i_chan in range(nchans):
            sec_temp = mini_sec[i_chan]
            mini_median_filt[sec_temp] = fill_zeros_filter (
                mini_median[sec_temp], size_filter, use_median=False)
            mini_std_filt[sec_temp] = fill_zeros_filter (
                mini_std[sec_temp], size_filter, use_median=False)


    # estimate median and std of entire image from the values of the subimages
    bkg_median = np.median(mini_median_filt)
    bkg_std = np.median(mini_std_filt)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_back')


    if False:
        ds9_arrays(data=data, mask_reject=mask_reject, mini_median=mini_median,
                   mini_median_filt=mini_median_filt)

    # mini_median_filt and mini_std_filt are float64 (which is what
    # sigma_clipped_stats returns); return them as float32 below to
    # avoid the data type of the full background and its std image
    # being float64 as well
    return (mini_median_filt.astype('float32'), mini_std_filt.astype('float32'),
            bkg_median, bkg_std)


################################################################################

def get_back (data, header, fits_objmask, fits_mask=None,
              tel=None, set_zogy=None, imtype=None):


    """Function that returns the background of the image [data]. A clipped
    median and standard deviation is determined for each subimage
    which is masked using the object mask (created from SExtractor's
    '-OBJECTS' image, where objects have zero values). The subimages
    (with size: [set_zogy.bkg_boxsize] are then median filtered.

    For ML/BG images and if set_zogy.use_2Dfit is True, a low order
    polynomial 2D fit is performed to the filtered mini background and
    if the fit value is lower than the filtered image for a particular
    subimage, that value is used.

    For MeerLICHT/BlackGEM images, the 2D polynomial fit will be used
    to estimate the channel correction factors, probably due to
    non-linearity of the different channels at low count levels.  For
    this estimate the function [bkg_corr_MLBG] is used. If
    set_zogy.MLBG_chancorr is True and the factors are lower than
    set_zogy.MLBG_chancorr_limdev for all channels, the factors are
    applied to the input data.

    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing get_back ...')
    mem_use (label='at start of get_back')

    # use the SExtractor '-OBJECTS' image, which is a (SExtractor)
    # background-subtracted image with all pixels where objects were
    # detected set to zero (-OBJECTS), as the object mask

    # [objmask] is the mask indicating the zeros in the '-OBJECTS'
    # image (with True)
    data_objmask = read_hdulist(fits_objmask, dtype='float32')
    mask_reject = (np.abs(data_objmask)==0)
    del data_objmask


    # also reject any masked pixel from input mask
    if fits_mask is not None:
        data_mask = read_hdulist (fits_mask, dtype='uint8')
        mask_tmp = (data_mask > 0)
        mask_reject |= mask_tmp
        nmask = np.sum(mask_tmp)
    else:
        nmask = 0

    # let this mask grow a few pixels if masked fraction is not
    # already above limfrac_reject of all pixels; avoid counting the
    # masked pixels from the bad pixel mask - nmask - as there can be
    # many "edge" pixels in the ML/BG reference image for the grid
    # centering option
    limfrac_reject_image = 0.5
    mask_reject_old = mask_reject
    struct = np.ones((3,3), dtype=bool)
    for i in range(3):
        imfrac_rej = (np.sum(mask_reject)-nmask)/(mask_reject.size-nmask)
        if imfrac_rej < limfrac_reject_image:
            mask_reject = ndimage.binary_dilation(mask_reject,
                                                  structure=struct,
                                                  iterations=1)

    log.info ('fraction of masked pixels in OR combination of '
              'source-extractor\'s object mask and the input mask: {:.3f}'
              .format(np.sum(mask_reject)/mask_reject.size))

    if (np.sum(~np.isfinite(data))) > 0:
        log.warning ('input data contains infinite or nan values')



    # image size and number of background boxes along x and y
    ysize, xsize = data.shape

    # reshape the input image in a 3D array of shape (nysubs, nxsubs,
    # number of pixels in background box), such that when taking
    # median along last axis, the mini image with the medians of the
    # subimages are immediately obtained; this needs to be done in a
    # masked array to be able to discard pixels in the object mask
    # (note that masked arrays consider mask values of True to be
    # invalid, i.e. are not used)
    bkg_boxsize = get_par(set_zogy.bkg_boxsize,tel)
    if ysize % bkg_boxsize != 0 or xsize % bkg_boxsize != 0:
        log.warning ('[bkg_boxsize] does not fit integer times in image; '
                     'best to pick a boxsize that does')

    # number of x,y subimages
    nysubs = int(ysize / bkg_boxsize)
    nxsubs = int(xsize / bkg_boxsize)


    t0 = time.time()

    # construct masked array and reshape it so that new shape
    # is: nboxes in y, nboxes in z, nvalues in box
    data_masked = (np.ma.masked_array(data, mask=mask_reject)
                   .reshape(nysubs,bkg_boxsize,-1,bkg_boxsize)
                   .swapaxes(1,2).reshape(nysubs,nxsubs,-1))

    if (np.sum(~np.isfinite(data_masked))) > 0:
        log.warning ('data_masked contains infinite or nan values')


    # suppressing expected "RuntimeWarning: Mean of empty slice"
    # warning for boxes where all values are masked
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # get clipped statistics
        nsigma = get_par(set_zogy.bkg_nsigma,tel)
        __, mini_bkg, mini_std = sigma_clipped_stats(
            data_masked, sigma=nsigma, axis=2, mask_value=0)


    # set any nan or infinite values to zero
    mask_mini_finite = (np.isfinite(mini_bkg) & np.isfinite(mini_std))
    mini_bkg[~mask_mini_finite] = 0
    mini_std[~mask_mini_finite] = 0

    # convert to 'float32'; don't! - this somehow leads to factors fit
    # in bkg_corr_MLBG being fixed to 1.0 or very close to 1.0
    #mini_bkg = mini_bkg.astype('float32')
    #mini_std = mini_std.astype('float32')


    # mask background subimage if more than some fraction of all
    # subimage pixels are masked (by OR combination of object mask and
    # input image mask); do this iteratively starting with a large
    # fraction and decreasing it to a low fraction, to prevent too
    # many subimages being masked - the limit is again set by
    # [limfrac_reject_image]; avoid counting the masked pixels from
    # the bad pixel mask - nmask - as there can be many "edge" pixels
    # in the ML/BG reference image for the grid centering option
    nbox_mask = int(nmask / bkg_boxsize**2)
    for fraction in [1, 0.67, 0.5]:
        npix_lim = fraction * bkg_boxsize**2
        # if number of invalid pixels along axis 2 is greater than
        # or equal to [npix_lim] then mask that background
        # subimage
        npix_bad = np.sum(data_masked.mask, axis=2)
        mask_tmp = (npix_bad >= npix_lim)
        imfrac_rej = (np.sum(mask_tmp)-nbox_mask)/(mask_tmp.size-nbox_mask)
        if imfrac_rej < limfrac_reject_image:
            mask_mini_avoid = mask_tmp
        else:
            break

    # add any negative values to mask_mini_avoid; include zeros to
    # also count the nan/infinite values set to zero above
    mask_mini_avoid |= (mini_bkg <= 0)

    # set masked values to zero
    mini_bkg[mask_mini_avoid] = 0
    mini_std[mask_mini_avoid] = 0

    # report masked fraction of image
    log.info ('fraction of subimages that were masked: {:.2f}'
              .format(np.sum(mask_mini_avoid)/mask_mini_avoid.size))
    log.info ('time to get statistics: {:.3f}s'.format(time.time()-t0))


    # fill zeros (if needed) and smooth mini images with
    # median or mean filter using function [fill_mask]
    size_filter = get_par(set_zogy.bkg_filtersize,tel)


    if tel not in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']:

        # not a ML/BG image
        mini_bkg_filt = fill_mask (mini_bkg, size_filter,
                                   use_median=True, smooth=True)
        #mini_bkg_filt = fill_zeros_filter (mini_bkg, size_filter)

        # get readnoise from header
        rdnoise = read_header(header, ['rdnoise'])

        # create mini image with readnoise
        mini_rdnoise = np.zeros_like (mini_bkg)
        mini_rdnoise[:] = rdnoise

        # mini_bkg_filt and mini_rdnoise are combined further below


    else:

        # for ML/BG this needs to be done per channel separately, to
        # avoid channels with different average count levels affecting
        # each other
        mini_shape = mini_bkg.shape
        mini_sec = get_section_MLBG (mini_shape)
        nchans = np.shape(mini_sec)[0]


        # create mini image with rdnoise
        mini_rdnoise = np.zeros_like (mini_bkg)
        # factor to potentially boost noise a bit
        f_rdnoise = 1.0
        # loop channels
        for i_chan in range(nchans):

            # channel section
            sec_tmp = mini_sec[i_chan]

            rdn_str = 'RDN{}'.format(i_chan+1)
            if rdn_str not in header:
                log.warning ('keyword {} expected but not present in header'
                             .format(rdn_str))
            else:
                mini_rdnoise[sec_tmp] = f_rdnoise * header[rdn_str]


        # replace mini_std with sqrt(bkg + readnoise**2)
        mini_std = np.sqrt(mini_bkg + mini_rdnoise**2)


        mem_use (label='just before bkg_corr_MLBG')


        # now try correcting the mini images for the differences in
        # the channels; if [set_zogy.MLBG_chancorr] is True, then
        # mini_bkg, mini_std and data are corrected with the channel
        # correction factors in place.
        order_max = 2
        bkg_corr, mini_bkg_2Dfit = bkg_corr_MLBG (
            mini_bkg, mini_std, data, header,
            correct_data=get_par(set_zogy.MLBG_chancorr,tel),
            order_max=order_max, mask_reject=mask_mini_avoid, tel=tel,
            set_zogy=set_zogy, limfrac_reject_image=limfrac_reject_image)


        # save data to fits image if background correction was applied
        if bkg_corr:
            fits.writeto (fits_objmask.replace('_objmask',''), data, header,
                          overwrite=True)



        if not get_par(set_zogy.MLBG_use2Dfit,tel):

            # if MLBG_use2Dfit was not used, use local background,
            # filtered per channel
            mini_bkg_filt = np.zeros_like(mini_bkg)
            for i_chan in range(nchans):
                # channel section
                sec_tmp = mini_sec[i_chan]
                # filter
                mini_bkg_filt[sec_tmp] = fill_mask (
                    mini_bkg[sec_tmp], size_filter, mask_2fill=mask_mini_avoid,
                    use_median=True, smooth=True)

                #mini_bkg_filt[sec_tmp] = fill_zeros_filter (
                #    mini_bkg[sec_tmp], size_filter, mask_2fill=mask_mini_avoid,
                #    fill_zeros=True)


        else:

            # now that images have been corrected, determine the local
            # and global backgrounds based on [mini_bkg] and
            # [mini_bkg_2Dfit], respectively; take minimum of mini_bkg
            # and 2Dfit
            mini_bkg_min = np.amin([mini_bkg, mini_bkg_2Dfit], axis=0)
            # boxes for which too few pixels were available for a
            # reliable bkg estimate, adopt the 2Dfit value
            mini_bkg_min[mask_mini_avoid] = (mini_bkg_2Dfit[mask_mini_avoid])


            # finally, apply a median filter per channel
            mini_bkg_filt = np.zeros_like(mini_bkg)
            for i_chan in range(nchans):
                # channel section
                sec_tmp = mini_sec[i_chan]
                mini_bkg_filt[sec_tmp] = median_filter (
                    mini_bkg_min[sec_tmp], size_filter)

                #mini_bkg_filt[sec_tmp] = fill_zeros_filter (
                #    mini_bkg_min[sec_tmp], size_filter, fill_zeros=False)



    # replace mini_std_filt with bkg value + readnoise**2
    mini_std_filt = np.sqrt(mini_bkg_filt + mini_rdnoise**2)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_back')


    # mini_bkg and mini_std are float64 (which is what
    # sigma_clipped_stats returns); return them as float32 below to
    # avoid the data type of the full background and its std image
    # being float64 as well
    return mini_bkg_filt.astype('float32'), mini_std_filt.astype('float32')


################################################################################

def get_median_std (nsub, cuts_ima, data, mask_use, mask_minsize=0.5, clip=True):

    subcut = cuts_ima[nsub]
    data_sub = data[subcut[0]:subcut[1], subcut[2]:subcut[3]]
    mask_sub = mask_use[subcut[0]:subcut[1], subcut[2]:subcut[3]]

    if np.sum(mask_sub) > mask_minsize:
        if clip:
            # get clipped_stats mean, std and median
            __, median, std = sigma_clipped_stats(data_sub[mask_sub]
                                                  .astype(float))
        else:
            median = np.median(data_sub[mask_sub])
            std = np.std(data_sub[mask_sub])
    else:
        # set to zero
        median, std = 0, 0

    return median, std


################################################################################

def fill_zeros_filter (data_mini, size_filter, use_median=True,
                       fill_zeros=True, mask_2fill=None, apply_filter=True):

    # fill zeros in data_mini with median of surrounding 3x3
    # pixels, until no more zeros left
    if fill_zeros:

        if mask_2fill is None:
            mask_2fill_copy = np.copy((data_mini==0))
        else:
            mask_2fill_copy = np.copy(mask_2fill)


        # avoid infinite loop
        if not np.all(mask_2fill_copy):

            # save original data_mini to compare to inside loop
            data_mini_orig = np.copy(data_mini)
            # continue as long as mask_2fill_copy is not changing any more
            while True:

                # filter data_mini
                data_mini_filt = median_filter_mask (data_mini, ~mask_2fill_copy,
                                                     size_filter=size_filter,
                                                     use_median=use_median)
                # fill data_mini with filtered values at mask_2fill_copy
                data_mini[mask_2fill_copy] = data_mini_filt[mask_2fill_copy]
                # copy of mask_2fill_copy to see if it's still changing
                mask_2fill_copy_old = np.copy(mask_2fill_copy)
                # update mask_2fill_copy
                mask_2fill_copy &= (data_mini == data_mini_orig)
                # if mask_2fill_copy has not changed, break
                if np.all(mask_2fill_copy==mask_2fill_copy_old):
                    break

                if False:
                    log.info ('np.sum(mask_2fill_copy): {}, '
                              'mask_2fill_copy.size: {}'
                              .format(np.sum(mask_2fill_copy),
                                      mask_2fill_copy.size))

                    ds9_arrays (mask_2fill_copy=mask_2fill_copy.astype(int),
                                data_mini_orig=data_mini_orig,
                                data_mini_filt=data_mini_filt,
                                data_mini=data_mini)


    if apply_filter:
        if use_median:
            # median filter
            data_mini = ndimage.filters.median_filter(data_mini, size_filter)
        else:
            # mean filter
            weights = np.full((size_filter,size_filter), 1./size_filter**2)
            data_mini = ndimage.filters.convolve(data_mini, weights)


    return data_mini


################################################################################

def get_section_MLBG (data_shape):

    """function to determine channel definitions for Meerlicht/BlackGEM
    image with shape [data_shape]

    """

    ysize, xsize = data_shape
    # define number of channels in x and y; hardcoded here because
    # from this zogy.py module in principle there is no access to
    # the blackbox settings file
    ny, nx = 2, 8
    # and size of data section in each channel
    ysize_chan, xsize_chan = ysize//ny, xsize//nx

    # channel reduced data section slices; shape=(16,2)
    return tuple([(slice(y,y+ysize_chan), slice(x,x+xsize_chan))
                  for y in range(0,ysize,ysize_chan)
                  for x in range(0,xsize,xsize_chan)])


################################################################################

def bkg_corr_MLBG (mini_median, mini_std, data, header, correct_data=True,
                   order_max=2, mask_reject=None, tel=None,
                   set_zogy=None, limfrac_reject_image=None):


    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing bkg_corr_MLBG ...')
    mem_use (label='at start of bkg_corr_MLBG')


    # determine channel pixels for full and mini images
    data_sec = get_section_MLBG (data.shape)
    mini_shape = mini_median.shape
    mini_sec = get_section_MLBG (mini_shape)
    nchans = np.shape(mini_sec)[0]
    npix_chan = mini_median.size // nchans

    # in case mask_reject is None
    if mask_reject is None:
        mask_reject = np.zeros(mini_median.shape, dtype=bool)

    # define x, y grid to fit
    ysize_mini, xsize_mini = mini_median.shape
    x = np.arange(xsize_mini)
    y = np.arange(ysize_mini)
    # normalization factor for polynomial fit, used also inside
    # function mini2min
    f_norm = max(ysize_mini, xsize_mini)
    xx, yy = np.meshgrid(x, y)
    xx = xx[~mask_reject]
    yy = yy[~mask_reject]
    zz = mini_median[~mask_reject]

    # loop different polynomial orders up to [order_max]:
    chi2red_old = np.inf
    limfrac_reject_chan = 0.8
    limdev = get_par(set_zogy.MLBG_chancorr_limdev,tel)
    for order in range(order_max+1):

        mem_use (label='at start of order loop in bkg_corr_MLBG')

        # create a set of parameters with factors
        params = Parameters()
        for i_chan in range(nchans):

            # fix fit factor to 1 if more than some fraction of the
            # channel pixels are masked through mask_reject; N.B.:
            # this does not include any additionally rejected pixels
            # due to large deviation from the fit below
            npix_masked = np.sum(mask_reject[mini_sec[i_chan]])
            if (npix_masked / npix_chan) > limfrac_reject_chan:
                vary = False
                log.warning ('ratio npix_masked / npix_chan = {} / {} is '
                             'larger than allowed ({}); fixing channel '
                             'correction factors to 1.0'
                             .format(npix_masked, npix_masked,
                                     limfrac_reject_chan))
            else:
                vary = True

            params.add('factor{}'.format(i_chan+1), value=1.0, min=0.5, max=1.5,
                       vary=vary)

        # add polynomial coefficients to fit, with decent starting estimates
        deg = [order, order]
        deg = np.asarray(deg)
        vander = polyvander2d(xx.ravel()/f_norm, yy.ravel()/f_norm, deg)
        z_temp = zz.reshape((vander.shape[0],))
        coeffs_init, chi2, rank, s = np.linalg.lstsq(vander, z_temp, rcond=None)
        # set higher-crossterm coefficients to zero
        xy = np.arange(order+1)
        mask_cfit = (np.sum(np.meshgrid(xy,xy), axis=0) <= order).ravel()
        coeffs_init[~mask_cfit] = 0
        for i_coeff, coeff_init in enumerate(coeffs_init):
            params.add('coeff{}'.format(i_coeff), value=coeff_init,
                       vary=mask_cfit[i_coeff])

        log.info ('start coefficients: {}'.format(coeffs_init))

        # do least_squares polynomial fit in several iterations to include
        # rejection of outliers in residual map
        mask_reject_temp = np.copy(mask_reject)
        npix_reject_old = np.sum(mask_reject_temp)
        for it in range(5):
            result = minimize (mini2min, params, method='least_squares',
                               args=(mini_median, mini_std, mini_sec, order,
                                     mask_reject_temp, f_norm,), max_nfev=100)
            params = result.params
            chi2red = result.redchi

            mini_median_corr_temp, mini_std_corr_temp, fit_temp, resid_temp = \
                mini2min(params, mini_median, mini_std, mini_sec, order,
                         mask_reject_temp, f_norm, return_resid=False)


            # scale residuals with the chi-square value as standard
            # deviations (mini_std) are overestimated (e.g. if mean
            # was used, the error in the means would be divided by
            # np.sqrt(npixels)
            resid_temp /= np.sqrt(chi2red)
            # reject pixels that are outliers in residual map
            mask_reject_temp |= (np.abs(resid_temp) >= 3)
            npix_reject = np.sum(mask_reject_temp)

            log.info ('iteration: {}, chi2red: {}, np.sum(mask_reject_temp): {}'
                      .format(it, chi2red, np.sum(mask_reject_temp)))

            if (npix_reject == npix_reject_old or
                npix_reject/mask_reject_temp.size > limfrac_reject_image):
                break
            else:
                npix_reject_old = npix_reject


        # define and normalize factor_temp
        factor_temp = np.zeros(nchans)
        for i_chan in range(nchans):
            factor_temp[i_chan] = params['factor{}'.format(i_chan+1)].value

        factor_temp_norm = factor_temp / np.mean(factor_temp)

        # define and normalize coeff_temp
        ncoeffs = (order+1)**2
        coeff_temp = np.zeros(ncoeffs)
        for i_coeff in range(ncoeffs):
            coeff_temp[i_coeff] = params['coeff{}'.format(i_coeff)].value

        coeff_temp_norm = coeff_temp / np.mean(factor_temp)

        # fit one more time with factors fixed at normalized best-fit
        # values, or 1 if any of them deviates more than limdev
        mask_deviant = (np.abs(factor_temp_norm - 1) > limdev)
        if np.any(mask_deviant):
            log.warning ('factor deviation for channel(s) {} larger than {}: '
                         '{}; setting all channel factors to 1'
                         .format(list(np.nonzero(mask_deviant)[0]+1), limdev,
                                 factor_temp_norm[mask_deviant]))
            factor_temp_norm[:] = 1

        # easier to create new set of fit parameters rather than adjusting them
        params = Parameters()
        for i_chan in range(nchans):
            params.add('factor{}'.format(i_chan+1), value=factor_temp_norm[i_chan],
                       vary=False)
        for i_coeff in range(ncoeffs):
            params.add('coeff{}'.format(i_coeff), value=coeff_temp_norm[i_coeff],
                       vary=mask_cfit[i_coeff])

        # fit
        result = minimize (mini2min, params, method='least_squares',
                           args=(mini_median, mini_std, mini_sec, order,
                                 mask_reject_temp, f_norm), max_nfev=100)
        params = result.params
        chi2red = result.redchi
        log.info ('order: {}, chi2red: {}'.format(order, chi2red))
        log.info ('normalized factors for this intermediate fit: {}'
                  .format(factor_temp_norm))

        # check formal criterium to use
        if (chi2red_old / chi2red) > 2:
            chi2red_old = chi2red
            order_bf = order
            chi2red_bf = chi2red
            params_bf = params
            mask_reject_bf = mask_reject_temp
            result_bf = result
            resid_bf = resid_temp

        mem_use (label='at end of order loop in bkg_corr_MLBG')


    #log.info (fit_report(result_bf))
    log.info ('order used: {}, reduced chi-square: {}'.format(order_bf,
                                                              chi2red_bf))
    p = params_bf

    # save fit factors and coefficients in numpy arrays; N.B.: error
    # estimate is unreliable because factor and coefficients fit
    # parameters are typically correlated, so not considering the
    # errors
    factor = np.zeros(nchans)
    for i_chan in range(nchans):
        factor[i_chan] = p['factor{}'.format(i_chan+1)].value

    ncoeffs = (order_bf+1)**2
    coeff = np.zeros(ncoeffs)
    for i_coeff in range(ncoeffs):
        coeff[i_coeff] = p['coeff{}'.format(i_coeff)].value


    # one more call to mini2min with final fit parameters
    mini_median_corr, mini_std_corr, mini_median_2Dfit, __ = \
        mini2min(p, mini_median, mini_std, mini_sec, order_bf,
                 mask_reject_bf, f_norm, return_resid=False)


    if False:
        ds9_arrays (mini_median=mini_median,
                    mini_std=mini_std,
                    mini_median_corr=mini_median_corr,
                    mini_std_corr=mini_std_corr,
                    mini_median_2Dfit=mini_median_2Dfit,
                    resid_bf=resid_bf,
                    mask_reject_bf=mask_reject_bf.astype(int),
                    bkg_mini=np.amin([mini_median_corr,mini_median_2Dfit], axis=0)
                    )


    mem_use (label='just before correcting data in bkg_corr_MLBG')

    if correct_data and not np.all(factor==1):

        bkg_corr = True

        # mini median and std
        mini_median[:] = mini_median_corr
        mini_std[:] = mini_std_corr

        # full image data
        for i_chan in range(nchans):
            data[data_sec[i_chan]] *= factor[i_chan]

        log.info ('image channels modified with correction factors: {}'
                  .format(factor))

    else:

        bkg_corr = False
        log.info ('image channels *NOT* modified with correction factors: {}'
                  .format(factor))



    # add boolean to header
    header['BKG-CORR'] = (bkg_corr, 'channels corrected for background ratios?')
    header['BKG-CHI2'] = (chi2red_bf, 'reduced chi2 of background factor/poly fit')

    # add correction factors to header
    for i_chan in range(nchans):

        header['BKG-CF{}'.format(i_chan+1)] = (
            factor[i_chan],
            'channel {} correction factor'.format(i_chan+1))


    # N.B.: polynomials were fit with normalized x and y pixel indices
    # (using [f_norm]) of the mini image, so coefficients should be
    # scaled accordingly if original image pixel indices are used to
    # infer the polynomial fit (most natural)
    bkg_boxsize = get_par(set_zogy.bkg_boxsize,tel)
    xy = np.arange(order_bf+1)
    coeff_power = np.sum(np.meshgrid(xy,xy), axis=0).ravel()
    coeff_scaled = coeff / (float(f_norm * bkg_boxsize)**coeff_power)

    header['BKG-FDEG'] = (order_bf, 'degree background 2D polynomial fit')

    # add coefficients to header
    for i_coeff in range((order_bf+1)**2):

        header['BKG-FC{}'.format(i_coeff)] = (
            coeff_scaled[i_coeff],
            'background 2D poly fit coefficient {}'.format(i_coeff))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in bkg_corr_MLBG')


    return bkg_corr, mini_median_2Dfit


################################################################################

def mini2min (params, data, data_std, data_sec, order, mask_reject,
              f_norm, return_resid=True):


    # fit parameters
    p = params

    # define x, y grid to fit
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])

    # make copy of input data
    data_corr = np.copy(data)
    data_std_corr = np.copy(data_std)

    # modify channel sections with fit parameters
    nchans = np.shape(data_sec)[0]
    for i_chan in range(nchans):
        factor_chan = p['factor{}'.format(i_chan+1)].value
        data_corr[data_sec[i_chan]] *= factor_chan
        data_std_corr[data_sec[i_chan]] *= factor_chan

    # exclude pixels where data_std_corr is zero
    mask_fit = (data_std_corr != 0)

    # add input mask_reject
    if mask_reject is not None:
        mask_fit &= ~mask_reject

    # create model 2D polynomial fit from coefficients
    coeff = [p['coeff{}'.format(i)].value for i in range((order+1)**2)]
    coeff = np.array(coeff).reshape((order+1, order+1))
    fit = polygrid2d(x/f_norm, y/f_norm, coeff).T

    # residuals
    resid = np.zeros(data_corr.shape)
    resid[mask_fit] = ((data_corr[mask_fit] - fit[mask_fit]) /
                       data_std_corr[mask_fit])

    # return flattened residuals
    if return_resid:
        return resid[mask_fit].ravel()
    else:
        return data_corr, data_std_corr, fit, resid



################################################################################

def polyfcn2d (params, x, y, z, z_err, deg):

    # extract coefficients from params
    p = params.valuesdict()
    coeffs = [p['c{}'.format(i)] for i in range((deg+1)**2)]
    coeffs = np.array(coeffs).reshape(deg+1,deg+1)

    # determine z_model at x,y with coefficients
    z_model = polyval2d(x, y, coeffs)

    # determine residuals
    resid = (z - z_model) / z_err

    # return residuals
    return resid.ravel()


################################################################################

def polyfit2d (x, y, z, z_err=None, order=2, fit_higher_Xterms=False,
               verbose=False):

    # exclude higher cross-terms if input parameter
    # [fit_higher_Xterms] is False
    if not fit_higher_Xterms:
        xy = np.arange(order+1)
        xx, yy = np.meshgrid(xy, xy)
        mask_fit = (xx+yy <= order).ravel()
    else:
        mask_fit = np.ones((order+1)**2, dtype=bool)

    # polynomial fit without considering z_err
    deg = [order, order]
    deg = np.asarray(deg)
    vander = polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    vander[:,~mask_fit] = 0
    z_temp = z.reshape((vander.shape[0],))
    coeffs, resid, rank, s = np.linalg.lstsq(vander, z_temp, rcond=None)

    if verbose:
        #log.info ('vander: {}'.format(vander))
        log.info ('resid : {}'.format(resid))
        log.info ('rank  : {}'.format(rank))
        log.info ('s     : {}'.format(s))
        log.info ('coeffs: {}'.format(coeffs))

    if z_err is not None:

        # put coeffients in parameters
        coeffs[~mask_fit] = 0
        params = Parameters()
        nc = coeffs.size
        for i in range(nc):
            params.add('c{}'.format(i), value=coeffs[i], vary=mask_fit[i])

        # do leastsq polynomial fit including z_err
        result = minimize (polyfcn2d, params, method='least_squares',
                           args=(x, y, z, z_err, order), max_nfev=100)

        p = result.params.valuesdict()
        coeffs = np.array([p['c{}'.format(i)] for i in range(nc)])
        chi2red = result.redchi

        if verbose:
            log.info (fit_report(result))

        return coeffs.reshape(deg+1), chi2red

    else:

        # return fit coefficients
        return coeffs.reshape(deg+1)


################################################################################

def polyfit2d_orig (x, y, f, deg):

    """see 2nd most popular answer at
https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent

    """

    deg = [deg, deg]
    deg = np.asarray(deg)
    vander = polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f, rcond=None)[0]
    return c.reshape(deg+1)


################################################################################

def gauss2min (params, x, y, yerr=None):

    p = list(params.valuesdict().values())
    resid = gauss1d (p, x) - y
    if yerr is not None:
        mask_nonzero = (yerr!=0)
        resid[mask_nonzero] /= yerr[mask_nonzero]

    return resid


################################################################################

def gauss1d (p, x):

    # 1D Gaussian function including background up to 2nd order polynomial

    z = (x-p[1])/p[2]
    gauss = p[0]*np.exp(-0.5*z**2)

    for i in range(3,6):
        if len(p) > i:
            gauss += p[i] * x**(i-3)

    return gauss


################################################################################

def fill_mask (data, size_filter, mask_2fill=None, mask_valid=None,
               use_median=True, smooth=False):

    """function to replace values in [data] indicated with boolean
    [mask_2fill] with the mean or median - determined through
    [use_median] - of the surrounding values. The size of the filter
    is determined by [size_filter]. Only valid pixels, defined with
    boolean mask [mask_valid]; if not defined, all pixels are
    considered to be valid. If smooth is True, the entire image is
    mean- or median-filtered after the [mask_2fill] values were
    replaced. The updated data is returned.

    """

    # copy of input data and data size
    data_copy = np.copy(data)
    ysize, xsize = data.shape


    # copy of mask_2fill
    if mask_2fill is None:
        # if not defined, replace all zero values of input data
        mask_2fill_copy = np.copy(data==0)
    else:
        mask_2fill_copy = np.copy(mask_2fill)


    # if mask_valid not provided, consider all pixels to be valid
    if mask_valid is None:
        mask_valid_copy = np.ones(data_copy.shape, dtype=bool)
    else:
        mask_valid_copy = np.copy(mask_valid)


    # set mask_2fill_copy indices as invalid; this will be updated in loop
    mask_valid_copy[mask_2fill_copy] = False


    # keep going until no values are changing
    dpix = int(size_filter/2)
    while True:

        # indices of (remaining) True indices
        index_y, index_x = np.nonzero(mask_2fill_copy)

        npix = np.sum(mask_2fill_copy)
        for n in range(npix):

            j0 = max(index_y[n]-dpix, 0)
            j1 = min(index_y[n]+dpix+1, ysize)
            i0 = max(index_x[n]-dpix, 0)
            i1 = min(index_x[n]+dpix+1, xsize)

            data_tmp = data_copy[j0:j1,i0:i1]
            mask_tmp = mask_valid_copy[j0:j1,i0:i1]

            if np.sum(mask_tmp) >= 2:

                if use_median:
                    data_copy[index_y[n], index_x[n]] = np.median(data_tmp[mask_tmp])
                else:
                    data_copy[index_y[n], index_x[n]] = np.mean(data_tmp[mask_tmp])

                mask_valid_copy[index_y[n], index_x[n]] = True
                mask_2fill_copy[index_y[n], index_x[n]] = False


        # if mask_2fill_copy not changing anymore, break
        if np.sum(mask_2fill_copy) == npix:
            break


    if smooth:
        data_copy = median_filter(data_copy, size_filter, use_median=use_median)


    return data_copy


################################################################################

def median_filter (data, size_filter, use_median=True):

    """function to median- or mean-filter the entire input array [data]
    with a filter size set by [size_filter]; all values inside this
    box are used, including the central value. The updated data array
    is returned.

    """

    data_copy = np.copy(data)

    if use_median:
        # median filter
        data_copy = ndimage.filters.median_filter(data_copy, size_filter)
    else:
        # mean filter
        weights = np.full((size_filter,size_filter), 1./size_filter**2)
        data_copy = ndimage.filters.convolve(data_copy, weights)


    return data_copy


################################################################################

# function to return median filter, with size [filter_size], of array
# with mask (True=valid pixels)
def median_filter_mask (array, mask, size_filter=3, use_median=True):

    # array shape
    ysize, xsize = array.shape

    # pad array with (size_filter-1)//2 pixels with value 0
    dpix = int(size_filter-1)//2
    array_pad = np.pad(array, (dpix,dpix), 'constant')
    # and mask with value False
    mask_pad = np.pad(mask, (dpix,dpix), 'constant')

    # using skimage.util.shape.view_as_windows to construct cubes
    window_shape = (size_filter, size_filter)
    array_cube = view_as_windows (array_pad, window_shape).reshape(
        array.shape[0], array.shape[1], -1)
    mask_cube = view_as_windows (mask_pad, window_shape).reshape(
        mask.shape[0], mask.shape[1], -1)

    # create masked array
    array_masked = np.ma.masked_array(array_cube, mask=~mask_cube)

    if use_median:
        # return median filtered array
        return np.ma.median(array_masked, axis=2)
    else:
        # return mean filtered array
        return np.ma.mean(array_masked, axis=2)


################################################################################

def get_rand_indices (shape, fraction=0.2):

    """Given an input shape, this function returns a tuple of random
    integer arrays (with the ranges determined by shape), one for each
    axis/dimension. The total number of indices returned is the total
    size defined by [shape] times [fraction].

    """

    # determine size
    size = np.empty(shape).size

    # number of dimensions
    ndim = len(shape)

    # create list of integer arrays
    # N.B.: size needs to be the same for each axis
    index = [np.random.randint(shape[i],size=int(size*fraction))
             for i in range(ndim)]

    if ndim==1:
        index = index[0]

    # return tuple
    return tuple(index)


################################################################################

def mini2back (data_mini, output_shape, order_interp=3, bkg_boxsize=None,
               interp_Xchan=True, timing=True):

    if timing: t = time.time()
    log.info('executing mini2back ...')


    def help_mini2back (data_mini, output_shape):

        # resize low-resolution meshes, with order [order_interp], where
        # order=0: nearest
        # order=1: bilinear spline interpolation
        # order=2: quadratic spline interpolation
        # order=3: cubic spline interpolation
        background = ndimage.zoom(data_mini, bkg_boxsize, order=order_interp,
                                  mode='nearest')

        # if shape of the background is not equal to input
        # [output_data], then pad the background image
        if output_shape != background.shape:
            t1 = time.time()
            ysize, xsize = output_shape
            ypad = ysize - background.shape[0]
            xpad = xsize - background.shape[1]
            background = np.pad(background, ((0,ypad),(0,xpad)), 'edge')

            log.info('time to pad: {:.4f}'.format(time.time()-t1))

            #np.pad seems quite slow; alternative:
            #centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
            #    get_par(set_zogy.bkg_boxsize,tel), ysize, xsize,
            #    get_remainder=True)
            # these now include the remaining patches

        return background


    # if interp_Xchan is True, then expand the mini image to the full
    # image allowing the interpolation to cross the different channels
    if interp_Xchan:

        data_full = help_mini2back (data_mini, output_shape)

    else:

        # for ML/BG, determine the full image for each channel
        # separately and insert them into the output image
        data_sec = get_section_MLBG (output_shape)
        mini_shape = data_mini.shape
        mini_sec = get_section_MLBG (mini_shape)
        nchans = np.shape(mini_sec)[0]

        # prepare output array and loop channels
        data_full = np.zeros(output_shape, dtype='float32')
        channel_shape = np.shape(data_full[data_sec[0]])

        for i_chan in range(nchans):

            data_full[data_sec[i_chan]] = help_mini2back (
                data_mini[mini_sec[i_chan]], channel_shape)


    if timing:
        log_timing_memory (t0=t, label='in mini2back')


    return data_full


################################################################################

def plot_scatter (x, y, limits, corder, cmap='rainbow_r', marker='o',
                  markersize=2, xlabel=None, ylabel=None, legendlabel=None,
                  title=None, filename=None, simple=False, xscale='linear',
                  yscale='linear', binsize=0.5):

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=corder, cmap=cmap, alpha=0.5, label=legendlabel,
               edgecolors='black')
    ax.axis(limits)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


    bins = np.arange(int(np.min(x)),np.ceil(np.max(x)),binsize)
    indices = np.digitize(x, bins, right=True)
    median = np.zeros(len(bins))
    std = np.zeros(len(bins))
    for i in indices:
        mask = ((indices-1) == i)
        if np.sum(mask)>0:
            median[i] = np.median(y[mask])
            std[i] = np.std(y[mask])

    ax.plot(bins+binsize/2, median, color='tab:red', linestyle='dashed')
    ax.errorbar(bins+binsize/2, median, yerr=std, linestyle='None',
                capsize=5, color='tab:red')


    if legendlabel is not None:
        ax.legend(numpoints=1, fontsize='medium')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, fontsize=10)

    if filename is not None:
        fig.savefig(filename)

    if get_par(set_zogy.show_plots,tel): plt.show()
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
    if get_par(set_zogy.show_plots,tel):
        plt.show()
    plt.close()


################################################################################

def get_psf (image, header, nsubs, imtype, fwhm, pixscale, remap, fits_mask,
             nthreads=1):

    """Function that takes in [image] and determines the actual Point
    Spread Function as a function of position from the full frame, and
    returns a cube containing the psf for each subimage in the full
    frame.

    PSFEx runs on a SExtractor catalog and not on an image, and
    therefore the input [image] parameter is only used to print out
    the name and if PSF fitting is done by SExtractor
    (i.e. [set_zogy.psffit_sex] is set to True).

    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing get_psf ...')

    if imtype=='new':
        base = base_new
    else:
        base = base_ref


    # determine image size from header
    if not remap:
        xsize, ysize = header['NAXIS1'], header['NAXIS2']
    else:
        # if remap is True, force size of input ref image,
        # to be the same as that of the new image
        ysize, xsize = get_par(set_zogy.shape_new,tel)

    # number of subimages in x and y
    subsize = get_par(set_zogy.subimage_size,tel)
    nx = int(xsize / subsize)
    ny = int(ysize / subsize)


    # redo boolean to determine whether to rerun some parts (source
    # extractor, astrometry.net, psfex) even if those were executed
    # done
    redo = ((get_par(set_zogy.redo_new,tel) and imtype=='new') or
            (get_par(set_zogy.redo_ref,tel) and imtype=='ref'))


    # run psfex on SExtractor output catalog
    #
    # If the PSFEx output file is already present with right
    # psf_size_config or larger, then skip run_psfex
    skip_psfex=False
    psfex_bintable = '{}_psf.fits'.format(base)
    if isfile(psfex_bintable) and not redo:
        header_psf = read_hdulist (psfex_bintable, get_data=False,
                                   get_header=True)
        psf_samp, psf_size_config = get_samp_PSF_config_size(imtype)
        if header_psf['PSFAXIS1'] >= psf_size_config:
            skip_psfex = True
            header['PSF-P'] = (True, 'successfully processed by PSFEx?')
            if get_par(set_zogy.verbose,tel):
                log.info ('skipping run_psfex for image {} as existing PSFEx '
                          'output file: {} was built with [psf_size_config]: '
                          '{} equal to or larger than [size_vignet]/[psf_samp]'
                          .format(image, psfex_bintable, psf_size_config))


    if not skip_psfex:
        psfexcat = '{}_psfex.cat'.format(base)
        log.info('psfexcat: {}'.format(psfexcat))
        sexcat_ldac = '{}_cat_ldac.fits'.format(base)
        log.info('sexcat_ldac: {}'.format(sexcat_ldac))
        try:

            # need to create LDAC version of SExtractor catalog, including
            # the thumbnails around the sources that will be used by
            # PSFEx, i.e. with SNR >~ 20
            result = run_sextractor(
                '{}.fits'.format(base), sexcat_ldac,
                get_par(set_zogy.sex_cfg,tel), get_par(set_zogy.sex_par,tel),
                pixscale, header, fwhm=fwhm, psfex_mode=True, imtype=imtype,
                fits_mask=fits_mask, npasses=1, tel=tel, set_zogy=set_zogy,
                nthreads=nthreads)


            # set polynomial degree to use in [run_psfex]
            poldeg = get_par(set_zogy.psf_poldeg,tel)
            # but if only 1 subimage, use constant PSF
            if nsubs==1:
                poldeg = 0
            # size of axes of PSF output snap image; does not need to
            # be same as number of subimage
            nsnap = min(nx, ny)
            # run PSFEx:
            result = run_psfex(sexcat_ldac, get_par(set_zogy.psfex_cfg,tel),
                               psfexcat, imtype, poldeg, nsnap=nsnap,
                               limit_ldac=True, nthreads=nthreads)

        except Exception as e:
            PSFEx_processed = False
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [run_psfex]: {}'.format(e))

        else:
            PSFEx_processed = True


        header['PSF-P'] = (PSFEx_processed, 'successfully processed by PSFEx?')

        # PSFex version
        cmd = ['psfex', '-v']
        result = subprocess.run(cmd, capture_output=True)
        version = str(result.stdout).split()[-2]
        header['PSF-V'] = (version.strip(), 'PSFEx version used')

        # now that header is updated, raise exception that will be caught
        # in try-except block around [prep_optimal_subtraction]
        if not PSFEx_processed:
            raise Exception ('exception was raised during [run_psfex]')



    # If [set_zogy.psffit_sex] parameter is set, then again run SExtractor,
    # but now using output PSF from PSFEx, so that PSF-fitting can be
    # performed for all objects. The output columns defined in
    # [set_zogy.sex_par_psffit] include several new columns related to the PSF
    # fitting.
    sexcat_ldac_psffit = '{}_ldac_psffit.fits'.format(base)
    if (get_par(set_zogy.psffit_sex,tel) and
        (not isfile(sexcat_ldac_psffit) or redo)):

        result = run_sextractor(image, sexcat_ldac_psffit,
                                get_par(set_zogy.sex_cfg_psffit,tel),
                                get_par(set_zogy.sex_par_psffit,tel), pixscale,
                                header, fit_psf=True, fwhm=fwhm, tel=tel,
                                set_zogy=set_zogy, nthreads=nthreads)


    # read in PSF output binary table from psfex, containing the
    # polynomial coefficient images, and various PSF parameters using
    # the function [extract_psf_datapars]
    verbose = get_par(set_zogy.verbose,tel)
    results = extract_psf_datapars (psfex_bintable, verbose=verbose)
    (data_psf, header_psf, psf_fwhm, psf_samp, psf_size_config, psf_chi2,
     psf_nstars, polzero1, polscal1, polzero2, polscal2, poldeg) = results


    # call centers_cutouts to determine centers
    # and cutout regions of the full image
    subimage_size = get_par(set_zogy.subimage_size,tel)
    subimage_border = get_par(set_zogy.subimage_border,tel)
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
        subimage_size, ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border

    if imtype=='ref':
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
        header_new = read_hdulist ('{}.fits'.format(base_new),
                                   get_data=False, get_header=True)
        wcs = WCS(header_new)
        ra_temp, dec_temp = wcs.all_pix2world(centers[:,1], centers[:,0], 1)
        # then convert ra, dec back to x, y in the original ref image;
        # since this block concerns the reference image, the input [header]
        # corresponds to the reference header
        wcs = WCS(header)
        centers[:,1], centers[:,0] = wcs.all_world2pix(ra_temp, dec_temp, 1)


    # [psf_size] is the PSF size in image pixels, which determines the
    # size of [psf_ima_sub] below. For both the new and ref image, it
    # should be defined as 2*psf_rad_zogy*fwhm, where fwhm is the
    # maximum of fwhm_new and fwhm_ref. At least, if these are both
    # available; it is possible that [get_psf] is run on the new or
    # ref image only - use the input [fwhm] in that case.
    if ('fwhm_new' in globals() and 'fwhm_ref' in globals() and
        'pixscale_new' in globals() and 'pixscale_ref' in globals()):
        fwhm_use = max(fwhm_new, fwhm_ref*(pixscale_ref/pixscale_new))
    else:
        fwhm_use = fwhm

    psf_size = 2 * get_par(set_zogy.psf_rad_zogy,tel) * fwhm_use
    # make sure [psf_size] is not larger than [psf_size_config] *
    # [psf_samp], which will happen if FWHM is very large
    psf_size = int(min(psf_size, psf_size_config * psf_samp))
    # force it to be odd
    if psf_size % 2 == 0:
        psf_size -= 1


    if 'fwhm_new' in globals():
        log.info ('fwhm_new [pix]: {:.2f}'.format(fwhm_new))

    if 'fwhm_ref' in globals():
        log.info ('fwhm_ref [pix]: {:.2f}'.format(fwhm_ref))

    log.info ('fwhm_use [pix]: {:.2f}'.format(fwhm_use))

    log.info ('psf_size used in [get_psf]: {} pix for imtype: {}'
              .format(psf_size, imtype))
    log.info ('psf_samp used in [get_psf]: {:.2f} pix for imtype: {}'
              .format(psf_samp, imtype))

    # [psf_ima] is the corresponding cube of PSF subimages
    psf_ima = np.zeros((nsubs,psf_size,psf_size)).astype('float32')

    # [psf_ima_center] is [psf_ima] or [psf_ima_remap] at the
    # center of images of xsize_fft x ysize_fft; the former cube
    # is replace by psf_ima_center_sub inside the loop below to
    # save memory
    # psf_ima_center = np.zeros((nsubs,ysize_fft,xsize_fft))


    # if [run_psfex] was executed successfully (see above), then add a
    # number of header keywords
    if not skip_psfex and PSFEx_processed:
        header['PSF-P'] = (PSFEx_processed, 'successfully processed by PSFEx?')
        header['PSF-RAD'] = (get_par(set_zogy.psf_rad_zogy,tel),
                             '[FWHM] PSF radius used for optimal subtraction')
        header['PSF-RADP'] = (get_par(set_zogy.psf_rad_phot,tel),
                              '[FWHM] PSF radius used for optimal photometry')
        header['PSF-SIZE'] = (psf_size, '[pix] size PSF image for optimal subtraction')
        header['PSF-FRAC'] = (get_par(set_zogy.psf_samp_fwhmfrac,tel),
                              '[FWHM] PSF sampling step in units of FWHM')
        header['PSF-SAMP'] = (psf_samp,
                              '[pix] PSF sampling step (~ PSF-FRAC * FWHM)')
        header['PSF-CFGS'] = (psf_size_config,
                              '[config. pix] size PSF configuration image')
        header['PSF-NOBJ'] = (psf_nstars, 'number of accepted PSF stars')
        header['PSF-FIX'] = (poldeg==0, 'single fixed PSF used for entire image?')
        header['PSF-PLDG'] = (poldeg, 'degree polynomial used in PSFEx')
        header['PSF-CHI2'] = (psf_chi2, 'final reduced chi-squared PSFEx fit')
        # add PSF-FWHM in arcseconds using initial pixel scale
        header['PSF-FWHM'] = (psf_fwhm,
                              '[pix] image FWHM inferred by PSFEx')
        header['PSF-SEE'] = (psf_fwhm*pixscale,
                             '[arcsec] image seeing inferred by PSFEx')
        #header['PSF-ELON'] = (psf_elon, 'median elongation of PSF stars')
        #header['PSF-ESTD'] = (psf_elon_std, 'elongation sigma (STD) of PSF stars')


    # loop through nsubs and construct psf at the center of each
    # subimage, using the output from PSFex that was run on the full
    # image
    dict_psf_ima_shift = {}
    for nsub in range(nsubs):

        # using function [get_psf_ima], construct the PSF image with
        # shape (psf_size, psf_size) at the central coordinates of the
        # subimage; this image is at the original pixel scale
        if 'header_new' not in locals():
            header_new = None

        psf_clean_factor = get_par(set_zogy.psf_clean_factor,tel)
        psf_ima_sub, psf_ima_config_sub = get_psf_ima (
            data_psf, centers[nsub,1], centers[nsub,0], psf_size,
            psf_samp, polzero1, polscal1, polzero2, polscal2, poldeg,
            imtype=imtype, remap=remap, header=header, header_new=header_new,
            psf_clean_factor=psf_clean_factor)

        # record psf image in [psf_ima] cube to be used for the Moffat
        # and Gaussian fits to the PSFs at the end of this function
        psf_ima[nsub] = psf_ima_sub

        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage:
        # psf_ima_center
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            log.info('Warning: image not even in one or both dimensions!')

        xcenter_fft = int(xsize_fft/2)
        ycenter_fft = int(ysize_fft/2)
        if get_par(set_zogy.verbose,tel) and nsub==0:
            log.info('xcenter_fft: {}, ycenter_fft: {}'
                     .format(xcenter_fft, ycenter_fft))

        psf_hsize = int(psf_size/2)
        # psf_size should be odd, so need to add 1:
        index = tuple([slice(ycenter_fft-psf_hsize, ycenter_fft+psf_hsize+1),
                       slice(xcenter_fft-psf_hsize, xcenter_fft+psf_hsize+1)])
        psf_ima_center = np.zeros((ysize_fft,xsize_fft))
        psf_ima_center[index] = psf_ima_sub

        # perform fft shift, i.e. psf_ima_center is split into 4
        # quadrants which are flipped so that PSF center ends up on
        # the 4 corners of psf_ima_shift[nsub]
        psf_ima_shift_sub = fft.fftshift(psf_ima_center)

        # save psf_ima_shift_sub to a .npy file using function
        # [save_npy_fits] and record the filenames in
        # [dict_psf_ima_shift] with the subimage integers as the keys
        dict_psf_ima_shift[nsub] = save_npy_fits (
            psf_ima_shift_sub.astype('float32'), '{}_psf_ima_shift_sub{}.npy'
            .format(base, nsub))


        if (get_par(set_zogy.display,tel) and show_sub(nsub) and
            'base_newref' in globals()):

            fits.writeto('{}_psf_ima_config_sub{}.fits'.format(base, nsub),
                         psf_ima_config_sub.astype('float32'), overwrite=True)
            fits.writeto('{}_psf_ima_sub{}.fits'.format(base, nsub),
                         psf_ima_sub.astype('float32'), overwrite=True)
            fits.writeto('{}_psf_ima_center_sub{}.fits'.format(base, nsub),
                         psf_ima_center.astype('float32'), overwrite=True)
            fits.writeto('{}_psf_ima_shift_sub{}.fits'.format(base, nsub),
                         psf_ima_shift_sub.astype('float32'), overwrite=True)


    # now that PSFEx is done, fit elliptical Moffat and Gauss
    # functions to centers of subimages across the frame using the
    # [fit_moffat] function. N.B.: in case the ref image has a
    # different orientation than the new image, the Moffat/Gauss fits
    # are done in the orientation of the new frame!
    try:
        base_psf = '{}_psf'.format(base)
        fit_moffat(psf_ima, nx, ny, header, pixscale, base_psf, fit_gauss=False)
        fit_moffat(psf_ima, nx, ny, header, pixscale, base_psf, fit_gauss=True)
    except Exception as e:
        #log.info(traceback.format_exc())
        log.info('exception was raised during [fit_moffat]: {}'.format(e))



    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_psf')

    return dict_psf_ima_shift, psf_ima


################################################################################

def extract_psf_datapars (psfex_bintable, verbose=True):

    # read in PSF output binary table from psfex
    data_psf, header_psf = read_hdulist(psfex_bintable, get_header=True)
    data_psf = data_psf[0][0][:]

    # determine [poldeg] from shape of psf data; if too few stars are
    # available, PSFEx can lower this polynomial degree with respect
    # to the initial value defined in [set_zogy.psf_poldeg]
    poldeg = np.cumsum(range(10)).tolist().index(data_psf.shape[0])-1

    # read keyword values from header; [psf_size_config] is the size of
    # the PSF grid as defined in the PSFex configuration file
    # ([PSF_SIZE] parameter)
    keys = ['PSF_FWHM', 'PSF_SAMP', 'PSFAXIS1', 'CHI2', 'ACCEPTED']
    psf_fwhm, psf_samp, psf_size_config, psf_chi2, psf_nstars = (
        [header_psf[key] for key in keys])

    # following keys are only present if poldeg nonzero
    if poldeg != 0:
        keys = ['POLZERO1', 'POLZERO2', 'POLSCAL1', 'POLSCAL2']
        polzero1, polzero2, polscal1, polscal2 = (
            [header_psf[key] for key in keys])
    else:
        polzero1, polzero2 = 0, 0
        polscal1, polscal2 = 1, 1

    if verbose:
        if poldeg != 0:
            log.info('polzero1:                     {}'.format(polzero1))
            log.info('polscal1:                     {}'.format(polscal1))
            log.info('polzero2:                     {}'.format(polzero2))
            log.info('polscal2:                     {}'.format(polscal2))

        log.info('order polynomial:             {}'.format(poldeg))
        log.info('PSFex FWHM (pix):             {}'.format(psf_fwhm))
        log.info('PSF sampling size (pix):      {}'.format(psf_samp))
        log.info('PSF config size (config. pix):{}'.format(psf_size_config))
        log.info('number of accepted PSF stars: {}'.format(psf_nstars))
        log.info('final reduced chi2 PSFEx fit: {}'.format(psf_chi2))


    return (data_psf, header_psf, psf_fwhm, psf_samp, psf_size_config, psf_chi2,
            psf_nstars, polzero1, polscal1, polzero2, polscal2, poldeg)


################################################################################

def switch (sigma1, sigma2, theta):

    # theta below is the angle (with respect to +y) of the axis
    # corresponding to sigma1 or alpha1, which is the position angle
    # of the minor axis instead of the major axis in the case that
    # sigma1 < sigma2 (or alpha1 < alpha2). This function switches
    # sigma1 and sigma2 (or alpha1 and alpha2) so that the position
    # angle of the major axis is determined.
    switched = False
    if sigma1 < sigma2:
        sigma1, sigma2 = sigma2, sigma1
        switched = True
        if theta < 0:
            theta += 90
        else:
            theta -= 90

    if theta < -90:
        theta %= 90
    elif theta > 90:
        theta %= -90

    return sigma1, sigma2, theta, switched


################################################################################

def fit_moffat (psf_ima, nx, ny, header, pixscale, base_output, fit_gauss=False):

    if get_par(set_zogy.timing,tel): t = time.time()

    # arrays to be saved to output images; these are 1D and
    # will be reshaped below
    nsubs = nx * ny
    psfpeak = np.zeros(nsubs)
    amplitude = np.zeros(nsubs)
    fwhm_min = np.zeros(nsubs)
    fwhm_max = np.zeros(nsubs)
    theta = np.zeros(nsubs)
    elongation = np.zeros(nsubs)
    ellipticity = np.zeros(nsubs)
    chi2red = np.zeros(nsubs)
    if not fit_gauss:
        beta = np.zeros(nsubs)

    # make 2D x,y grid of pixel coordinates corresponding to PSF data
    # array to use in fit
    psf_size_config = np.shape(psf_ima)[1]
    xy = range(1,psf_size_config+1)
    xx, yy = np.meshgrid(xy, xy, indexing='xy')

    # create a set of Parameters
    params = Parameters()
    center = (psf_size_config-1)/2+1
    params.add('x0', value=center, min=center-5, max=center+5, vary=True)
    params.add('y0', value=center, min=center-5, max=center+5, vary=True)
    params.add('theta', value=0, min=-90, max=90, vary=True)
    params.add('amplitude', value=np.amax(psf_ima[int(nsubs/2)]), min=0,
               vary=True)
    params.add('background', value=0, min=-1, max=1, vary=True)

    if fit_gauss:
        params.add('sigma1', value=1, min=0.01, max=psf_size_config/4, vary=True)
        params.add('sigma2', value=1, min=0.01, max=psf_size_config/4, vary=True)
    else:
        params.add('beta', value=1, min=0.01, vary=True)
        params.add('alpha1', value=1, min=0.01, max=psf_size_config/4, vary=True)
        params.add('alpha2', value=1, min=0.01, max=psf_size_config/4, vary=True)


    for i in range(nsubs):

        # record peak of PSF
        psfpeak[i] = np.amax(psf_ima[i])


        # do leastsq model fit
        result = minimize(moffat2min, params, method='least_squares',
                          args=(psf_ima[i], xx, yy, fit_gauss), max_nfev=100)
        p = result.params.valuesdict()
        chi2red[i] = result.redchi


        if fit_gauss:
            sigma1, sigma2, theta[i], __ = switch(p['sigma1'], p['sigma2'],
                                                  p['theta'])
            fwhm_max[i] = sigma2fwhm(sigma1)
            fwhm_min[i] = sigma2fwhm(sigma2)
        else:
            beta[i] = p['beta']
            alpha1, alpha2, theta[i], __ = switch(p['alpha1'], p['alpha2'],
                                                  p['theta'])
            fwhm_max[i] = alpha2fwhm(alpha1, beta[i])
            fwhm_min[i] = alpha2fwhm(alpha2, beta[i])


        B, A = fwhm_min[i], fwhm_max[i]
        elongation[i] = A/B
        ellipticity[i] = 1 - B/A
        amplitude[i] = p['amplitude']


    if fit_gauss:
        label = 'Gauss'
    else:
        label = 'Moffat'


    # update original image header with some keywords
    header['PSF-PMIN'] = (np.amin(psfpeak),
                          '[sum(P)=1] min. peak value subimage PSFs')
    header['PSF-PMAX'] = (np.amax(psfpeak),
                          '[sum(P)=1] max. peak value subimage PSFs')
    header['PSF-PMED'] = (np.median(psfpeak),
                          '[sum(P)=1] median peak value subimage PSFs')
    header['PSF-PSTD'] = (np.std(psfpeak),
                          '[sum(P)=1] sigma (STD) peak subimage PSFs')

    if not fit_gauss:
        header['PSF-BMIN'] = (np.amin(beta),
                              '[pix] min. beta {} fit subimage PSFs'
                              .format(label))
        header['PSF-BMAX'] = (np.amax(beta),
                              '[pix] max. beta {} fit subimage PSFs'
                              .format(label))
        header['PSF-BMED'] = (np.median(beta),
                              '[pix] median beta {} fit subimage PSFs'
                              .format(label))
        header['PSF-BSTD'] = (np.amax(beta),
                              '[pix] sigma (STD) beta {} fit subimage PSFs'
                              .format(label))


    header['PSF-EMN{}'.format(label[0])] = (np.amin(elongation),
                                            'min. elongation {} fit '
                                            'subimage PSFs'.format(label))
    header['PSF-EMX{}'.format(label[0])] = (np.amax(elongation),
                                            'max. elongation {} fit '
                                            'subimage PSFs'.format(label))
    header['PSF-EMD{}'.format(label[0])] = (np.median(elongation),
                                            'median elongation {} fit '
                                            'subimage PSFs'.format(label))
    header['PSF-EST{}'.format(label[0])] = (np.std(elongation),
                                            'sigma (STD) elongation {} fit '
                                            'subimage PSFs'.format(label))

    fwhm_ave = (fwhm_max + fwhm_min) / 2
    seeing_ave = pixscale * fwhm_ave
    header['PSF-FMN{}'.format(label[0])] = (np.amin(fwhm_ave),
                                            '[pix] min. mean FWHM {} fits'
                                            .format(label))
    header['PSF-FMX{}'.format(label[0])] = (np.amax(fwhm_ave),
                                            '[pix] max. mean FWHM {} fits'
                                            .format(label))
    header['PSF-FMD{}'.format(label[0])] = (np.median(fwhm_ave),
                                            '[pix] median mean FWHM {} fits'
                                            .format(label))
    header['PSF-FST{}'.format(label[0])] = (np.std(fwhm_ave),
                                            '[pix] sigma (STD) mean FWHM {} '
                                            'fits'.format(label))


    if get_par(set_zogy.make_plots,tel):

        # reshape and transpose data arrays
        psfpeak = psfpeak.reshape((nx, ny)).transpose()
        amplitude = amplitude.reshape((nx, ny)).transpose()
        fwhm_min = fwhm_min.reshape((nx, ny)).transpose()
        fwhm_max = fwhm_max.reshape((nx, ny)).transpose()
        elongation = elongation.reshape((nx, ny)).transpose()
        ellipticity = ellipticity.reshape((nx, ny)).transpose()
        theta = theta.reshape((nx, ny)).transpose()
        chi2red = chi2red.reshape((nx, ny)).transpose()
        if not fit_gauss:
            beta = beta.reshape((nx, ny)).transpose()

        # output dictionary
        output = {'psfpeak': psfpeak, 'amplitude': amplitude,
                  'fwhm_min': fwhm_min, 'fwhm_max': fwhm_max,
                  'elongation': elongation, 'ellipticity': ellipticity,
                  'theta': theta, 'chi2red': chi2red}
        if not fit_gauss:
            output['beta'] = beta

        # name output multiple-extension fits file
        if not fit_gauss:
            fits_output = '{}{}'.format(base_output, '_moffat_pars.fits')
        else:
            fits_output = '{}{}'.format(base_output, '_gauss_pars.fits')

        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU())
        for key in output.keys():
            hdulist.append(fits.ImageHDU(data=output[key], name=key))
        # write image
        hdulist.writeto(fits_output, overwrite=True)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in fit_moffat')


################################################################################

def fit_moffat_single (image, image_err, mask_use=None, fit_gauss=False,
                       P_shift=None, fwhm=6, max_nfev=100, show=False,
                       fit_bkg=True, fit_inner=False):


    # make 2D x,y grid of pixel coordinates corresponding to PSF data
    # array to use in fit
    imsize_y, imsize_x = np.shape(image)
    x = range(1, imsize_x+1)
    y = range(1, imsize_y+1)
    xx, yy = np.meshgrid(x, y, indexing='xy')


    # estimate x- and ycenter (pixel coordinates) in
    # D_sub from maximum in P_shift; for objects near
    # image edge, this will not be the central pixel
    if P_shift is not None:
        yc_indx, xc_indx = np.unravel_index(np.argmax(P_shift),
                                            P_shift.shape)
        ycenter = yc_indx + 1
        xcenter = xc_indx + 1
        # create mask with pixel values above 1 percent of the peak of
        # the PSF used to determine the chi-square of this region
        mask_inner = (P_shift >= 0.01 * np.amax(P_shift))
    else:
        ycenter = (imsize_y-1)/2 + 1
        xcenter = (imsize_x-1)/2 + 1
        mask_inner = (np.sqrt((xx-xcenter)**2+(yy-ycenter)**2) < 2*fwhm)


    # estimate background value and STD from pixels outside of inner
    # region
    mask_bkg = mask_use & ~mask_inner
    __, val_bkg, val_bkg_std = sigma_clipped_stats (image[mask_bkg]
                                                    .astype(float))

    # min and max amplitude
    val_min = np.amin(image[mask_inner])
    val_max = np.amax(image[mask_inner])


    # create a set of Parameters
    params = Parameters()
    hwhm = max(fwhm/2, 3)
    params.add('x0', value=xcenter, min=xcenter-hwhm, max=xcenter+hwhm,
               vary=True)
    params.add('y0', value=ycenter, min=ycenter-hwhm, max=ycenter+hwhm,
               vary=True)
    params.add('theta', value=0, min=-90, max=90, vary=True)

    f0 = 3
    params.add('amplitude', value=1, min=f0*val_min, max=f0*val_max, vary=True)
    params.add('background', value=val_bkg, min=val_bkg-val_bkg_std,
               max=val_bkg+val_bkg_std, vary=fit_bkg)


    if fit_inner:
        # only fit inner part of profile
        mask_use &= mask_inner


    if fit_gauss:
        sigma = fwhm / sigma2fwhm(1.)
        params.add('sigma1', value=sigma, min=0.3*sigma, max=2*sigma, vary=True)
        params.add('sigma2', value=sigma, min=0.3*sigma, max=2*sigma, vary=True)
    else:
        params.add('beta', value=3, min=0.01, vary=True)
        params.add('alpha1', value=3, min=0.01, vary=True)
        params.add('alpha2', value=3, min=0.01, vary=True)


    # do leastsq model fit
    result = minimize(moffat2min, params, method='least_squares',
                      args=(image, xx, yy, fit_gauss, image_err, mask_use),
                      max_nfev=max_nfev)
    p = result.params.valuesdict()
    chi2red = result.redchi

    # chi2 within mask_inner
    if np.sum(mask_inner) != 0:
        chi2_inner = np.sum(moffat2min(result.params, image, xx, yy, fit_gauss,
                                       image_err, mask_inner)**2)
    else:
        chi2_inner = 0


    if fit_gauss:
        sigma1, sigma2, theta, switched = switch(p['sigma1'], p['sigma2'],
                                                 p['theta'])
        fwhm_max = sigma2fwhm(sigma1)
        fwhm_min = sigma2fwhm(sigma2)
    else:
        beta = p['beta']
        alpha1, alpha2, theta, switched = switch(p['alpha1'], p['alpha2'],
                                                 p['theta'])
        fwhm_max = alpha2fwhm(alpha1, beta)
        fwhm_min = alpha2fwhm(alpha2, beta)


    B, A = fwhm_min, fwhm_max
    elongation = A/B
    ellipticity = 1 - B/A
    amplitude = p['amplitude']
    background = p['background']
    background_err = result.params['background'].stderr
    fwhm_ave = (fwhm_min+fwhm_max)/2

    x0 = p['x0']
    y0 = p['y0']
    x0err = result.params['x0'].stderr
    y0err = result.params['y0'].stderr

    # in case fit did not succeed, stderrs will be None
    if x0err is None:
        x0err = 0
    if y0err is None:
        y0err = 0

    # in that case, also set chi2_inner to zero
    if x0err is None or y0err is None:
        chi2_inner = 0


    # reduced chi2_inner
    denom = max(1, (np.sum(mask_inner) - result.nvarys))
    if denom != 0:
        chi2red_inner = chi2_inner / denom
    else:
        chi2red_inner = chi2_inner


    # this block is to display the images related to the best-fit model
    if show:

        log.info('----------------------------------------')

        if fit_gauss:
            log.info('Gauss fit')
        else:
            log.info('Moffat fit')

        log.info('----------------------------------------')

        log.info(fit_report(result))

        log.info('fwhm_min:       {:.3f}'.format(fwhm_min))
        log.info('fwhm_max:       {:.3f}'.format(fwhm_max))
        log.info('fwhm (average): {:.3f}'.format(fwhm_ave))
        log.info('elongation:     {:.3f}'.format(elongation))
        log.info('theta:          {:.3f}'.format(theta))
        log.info('chi2red_inner:  {:.3f}'.format(chi2red_inner))
        log.info('success:        {}'.format(result.success))
        log.info('switched:       {}'.format(switched))


        if False:

            # create model image
            if fit_gauss:
                model_ima = EllipticalGauss2D (xx, yy, x0=x0, y0=y0,
                                               sigma1=sigma1, sigma2=sigma2,
                                               theta=theta, amplitude=amplitude,
                                               background=p['background'])
            else:
                model_ima = EllipticalMoffat2D (xx, yy, x0=x0, y0=y0, beta=beta,
                                                alpha1=alpha1, alpha2=alpha2,
                                                theta=theta, amplitude=amplitude,
                                                background=p['background'])

            resid = (image - model_ima) / image_err

            ds9_arrays(image=image, model_ima=model_ima, image_err=image_err,
                       resid=resid, mask_use=mask_use.astype(int),
                       mask_inner=mask_inner.astype(int), P_shift=P_shift)


    return x0, x0err, y0, y0err, fwhm_ave, elongation, chi2red_inner


################################################################################

def Elliptical2D_abc (x0, y0, x, y, sigx, sigy, theta_rad):

    """this function can be used for both the Moffat and Gauss 2D
    functions

     these a,b,c definotions are consistent with info on Gaussian 2D
     in astropy:
     https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html

     with sin (2*theta) = 2 * sin(theta) * cos(theta), and switching
     around B and C, these are identical to the definitions at:
     http://www.aspylib.com/doc/aspylib_fitting.html

     except that sig1/sigx and sig2/sigy have been switched around in
     a,b,c below to let sigy correspond to the input sigma1 for
     compatibility with fitting functions [fit_moffat] and
     [fit_moffat_single] that assume sigma1 defines the major axis;
     theta is zero in the +y direction and increases in the
     counterclockwise direction

    """


    cost = np.cos(theta_rad)
    sint = np.sin(theta_rad)
    sin2t = np.sin(2 * theta_rad)

    a = (cost / sigy)**2 + (sint / sigx)**2
    b = sin2t / sigy**2 - sin2t / sigx**2
    c = (sint / sigy)**2 + (cost / sigx)**2

    xdiff = x - x0
    ydiff = y - y0

    return a * xdiff**2 + b * xdiff * ydiff + c * ydiff**2


################################################################################

def EllipticalMoffat2D (x, y, x0=0, y0=0, beta=1, alpha1=1, alpha2=1,
                        theta=0, amplitude=1, background=0):

    rr_gg = Elliptical2D_abc (x0, y0, x, y, alpha1, alpha2, theta*np.pi/180)
    return background + amplitude * (1 + rr_gg) ** (-beta)


################################################################################

# function to convert Moffat alpha and beta to FWHM
def alpha2fwhm (alpha, beta):
    if alpha is not None and beta is not None:
        return alpha * 2 * np.sqrt(2**(1/beta)-1)
    else:
        return None


################################################################################

# similar function for Gaussian 2D as for EllipticalMoffat function above
def EllipticalGauss2D (x, y, x0=0, y0=0, sigma1=1, sigma2=1,
                       theta=0, amplitude=1, background=0):

    # note: theta input is assumed to be in degrees
    abc = Elliptical2D_abc (x0, y0, x, y, sigma1, sigma2, theta*np.pi/180)
    return background + amplitude * np.exp(-0.5*abc)


################################################################################

# function to convert Gaussian sigma to FWHM
def sigma2fwhm (sigma):
    if sigma is not None:
        return sigma * 2 * np.sqrt(2*np.log(2))
    else:
        return None


################################################################################

# define objective function: returns the array to be minimized
def moffat2min (params, image, xx, yy, fit_gauss, image_err=None, mask_use=None):

    p = params.valuesdict()

    if fit_gauss:

        model = EllipticalGauss2D (xx, yy, x0=p['x0'], y0=p['y0'],
                                   sigma1=p['sigma1'], sigma2=p['sigma2'],
                                   theta=p['theta'], amplitude=p['amplitude'],
                                   background=p['background'])
    else:

        model = EllipticalMoffat2D (xx, yy, x0=p['x0'], y0=p['y0'],
                                    beta=p['beta'],
                                    alpha1=p['alpha1'], alpha2=p['alpha2'],
                                    theta=p['theta'], amplitude=p['amplitude'],
                                    background=p['background'])

    # residuals
    if image_err is not None:
        err = image_err
    else:
        err = 1


    #resid = np.abs(image - model) / err
    resid = (image - model) / err

    if mask_use is not None:
        resid = resid[mask_use]

    # return flattened array
    return resid.flatten()


################################################################################

def calc_psf_config (data, poldeg, x, y):

    """function to construct the PSF image at relative coordinates x,y
    from the output data of PSFEx; the output image is in units of
    configuration pixels, which differ from the original image pixel
    scale with a factor of psf_samp.

    """

    if poldeg==0:
        psf_ima_config = data[0]

    elif poldeg==1:
        psf_ima_config = (data[0] + data[1] * x + data[2] * y)

    elif poldeg==2:
        psf_ima_config = (data[0] + data[1] * x + data[2] * x**2 +
                          data[3] * y + data[4] * x * y + data[5] * y**2)

    elif poldeg==3:
        psf_ima_config = (data[0] + data[1] * x + data[2] * x**2  + data[3] * x**3 +
                          data[4] * y + data[5] * x * y + data[6] * x**2 * y +
                          data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3)

    # alternatively (but not faster and less readable):
    #ndata = data.shape[0]
    #xy = range(0,poldeg+1)
    #xx, yy = np.meshgrid(xy, xy, sparse=True)
    #mask_xy = xx+yy <= poldeg
    #x_arr = np.array(x**xx)
    #y_arr = np.array(y**yy).reshape(poldeg+1,1)
    #xy_arr = (x_arr * y_arr)[mask_xy].reshape(ndata,1,1)
    #psf_ima_config = np.sum(xy_arr * data, axis=0)

    return psf_ima_config


################################################################################

def get_fratio_dxdy (cat_new, cat_ref, psfcat_new, psfcat_ref, header_new,
                     header_ref, nsubs, cuts_ima, header, use_optflux=False):

    """Function that takes in output catalogs of stars from the PSFex
    runs on the new and the ref image, and returns arrays with x,y
    pixel coordinates (!) (in the new frame) and flux ratios for the
    matching stars. If [use_optflux] is False, the ratios are inferred
    from the normalisation fluxes as saved in the NORM_PSF column of
    the PSFEx output file (ending in .._psf.cat). If [use_optflux] is
    True, the fratios are determined using the optimal fluxes in the
    catalogs.

    In addition, this function provides the difference in x- and
    y-coordinates of the new and ref catalogs after converting the
    reference image pixels to pixels in the new image through the WCS
    solutions of both images.

    """

    t = time.time()
    log.info('executing get_fratio_dxdy ...')


    #---------------------------------------------------------------------------
    def readcat (psfcat):

        """helper function to read PSFEx output ASCII catalog and
           extract x, y and normalisation factor from the PSF stars
        """

        table = ascii.read(psfcat, format='sextractor')
        # In PSFEx version 3.18.2 and higher all objects from the
        # input SExtractor catalog are recorded, and in that case the
        # entries with FLAGS_PSF=0 need to be selected.
        if 'FLAGS_PSF' in table.colnames:
            mask_psfstars = (table['FLAGS_PSF']==0)
        # In PSFEx version 3.17.1, only stars with zero flags are
        # recorded in the output catalog, so use the entire table
        else:
            mask_psfstars = np.ones(len(table), dtype=bool)

        x = table['X_IMAGE'][mask_psfstars]
        y = table['Y_IMAGE'][mask_psfstars]
        norm = table['NORM_PSF'][mask_psfstars]
        return x, y, norm

    #---------------------------------------------------------------------------

    # read psfcat_new
    x_psf_new, y_psf_new, norm_psf_new = readcat(psfcat_new)
    # read psfcat_ref
    x_psf_ref, y_psf_ref, norm_psf_ref = readcat(psfcat_ref)

    log.info('number of PSF stars in new: {}'.format(len(x_psf_new)))
    log.info('number of PSF stars in ref: {}'.format(len(x_psf_ref)))


    # get reference ra, dec corresponding to x, y using
    # wcs.all_pix2world
    wcs = WCS(header_ref)
    ra_psf_ref, dec_psf_ref = wcs.all_pix2world(x_psf_ref, y_psf_ref, 1)

    # same for new ra, dec and also convert the reference RA and DEC
    # to pixels in the new frame
    wcs = WCS(header_new)
    ra_psf_new, dec_psf_new = wcs.all_pix2world(x_psf_new, y_psf_new, 1)
    x_psf_ref2new, y_psf_ref2new = wcs.all_world2pix(ra_psf_ref, dec_psf_ref, 1)


    # use [get_matches] to find matches between new and ref coordinates
    # of PSF stars
    dist_max = 2
    idx_psf_new, idx_psf_ref = get_matches (ra_psf_new, dec_psf_new,
                                            ra_psf_ref, dec_psf_ref,
                                            dist_max=dist_max,
                                            return_offsets=False)

    dx_match = x_psf_new[idx_psf_new] - x_psf_ref2new[idx_psf_ref]
    dy_match = y_psf_new[idx_psf_new] - y_psf_ref2new[idx_psf_ref]
    fratio_match = norm_psf_new[idx_psf_new] / norm_psf_ref[idx_psf_ref]
    nmatch = len(fratio_match)

    # fratio_err cannot be estimated from the PSFEx output files; set
    # it to zero
    fratio_err_match = np.zeros_like(fratio_match)


    # correct for the ratio in ref/new exposure times; this method
    # uses FLUX_AUTO (see PHOTFLUX_KEY parameter in psfex.config),
    # saved in NORM_PSF column of PSFEx output file (.._red_psfex.cat)
    # which has not been converted to e-/s
    # N.B.: actually, fratio should not be converted to /s unit!!!
    #fratio_match *= header_ref['EXPTIME'] / header_new['EXPTIME']

    log.info('fraction of PSF stars that match: {}'
             .format((len(idx_psf_new)+len(idx_psf_ref))/
                     (len(x_psf_new)+len(x_psf_ref))))
    log.info ('median(fratio_match) using FLUX_AUTO: {}'
              .format(np.median(fratio_match)))


    if use_optflux:

        # match these PSF stars' coordinates to the new and ref catalogs to
        # extract their optimal fluxes
        table_new = Table.read(cat_new, memmap=True)
        table_ref = Table.read(cat_ref, memmap=True)

        if ('RA' in table_new.colnames and 'DEC' in table_new.colnames and
            'RA' in table_ref.colnames and 'DEC' in table_ref.colnames):

            ra_cat_new = table_new['RA'].value
            dec_cat_new = table_new['DEC'].value
            ra_cat_ref = table_ref['RA'].value
            dec_cat_ref = table_ref['DEC'].value

        else:

            # if RA,DEC not in catalog (e.g. the full-source ML/BG catalogs):

            # get cat ra, dec corresponding to x,y using wcs.all_pix2world
            wcs = WCS(header_new)
            ra_cat_new, dec_cat_new = wcs.all_pix2world(table_new['X_POS'].value,
                                                        table_new['Y_POS'].value,
                                                        1)
            # same for reference image
            wcs = WCS(header_ref)
            ra_cat_ref, dec_cat_ref = wcs.all_pix2world(table_ref['X_POS'].value,
                                                        table_ref['Y_POS'].value,
                                                        1)


        # find matches with PSF stars' positions (those PSF stars that
        # match between new and ref)
        __, idx_cat_new = get_matches (ra_psf_new[idx_psf_new],
                                       dec_psf_new[idx_psf_new],
                                       ra_cat_new, dec_cat_new,
                                       return_offsets=False)
        # now find the corresponding stars in table_ref
        idx_cat_tmp, idx_cat_ref = get_matches (ra_cat_new[idx_cat_new],
                                                dec_cat_new[idx_cat_new],
                                                ra_cat_ref, dec_cat_ref,
                                                return_offsets=False)
        # need to narrow down idx_cat_new in case not all of them
        # matched
        idx_cat_new = idx_cat_new[idx_cat_tmp]


        # select relevant entries of tables
        table_new = table_new[idx_cat_new]
        table_ref = table_ref[idx_cat_ref]


        if len(table_new)==len(table_ref) and len(table_new)> 0:

            if ('E_FLUX_OPT' in table_new.colnames and
                'E_FLUX_OPT' in table_ref.colnames):

                # final catalog fluxes have been saved in e-/s, while
                # the corresponding header EXPTIME indicates the image
                # EXPTIME; the intermediate catalogs are still in e-;
                # solution: check the column unit, and if in e-/s,
                # convert it back to e- to determine fratio
                flux_new = table_new['E_FLUX_OPT'].value
                flux_ref = table_ref['E_FLUX_OPT'].value
                # errors
                fluxerr_new = table_new['E_FLUXERR_OPT'].value
                fluxerr_ref = table_ref['E_FLUXERR_OPT'].value


                if '/s' in str(table_new['E_FLUX_OPT'].unit):
                    flux_new *= header_new['EXPTIME']
                    fluxerr_new *= header_new['EXPTIME']


                if '/s' in str(table_ref['E_FLUX_OPT'].unit):
                    flux_ref *= header_ref['EXPTIME']
                    fluxerr_ref *= header_ref['EXPTIME']


                # calculate ratio
                fratio_match = np.ones_like(flux_new)
                # nonzero mask
                m_nz = ((flux_ref != 0) & (flux_new != 0))
                fratio_match[m_nz] = flux_new[m_nz] / flux_ref[m_nz]

                # corresponding error
                fratio_err_match = fratio_match[m_nz] * np.sqrt(
                    (fluxerr_new[m_nz] / flux_new[m_nz])**2 +
                    (fluxerr_ref[m_nz] / flux_ref[m_nz])**2)



            elif (('MAG_OPT' in table_new.colnames or
                   'FNU_OPT' in table_new.colnames) and
                  ('MAG_OPT' in table_ref.colnames or
                   'FNU_OPT' in table_ref.colnames) and
                  'PC-ZP' in header_new and 'PC-ZP' in header_ref):

                # previously fluxes in electrons (or ADU) were saved
                # in output catalogs; if they are not available, infer
                # the fluxes from the optimal magnitudes (fnu in
                # microJy) using function [apply_zp_mag2flux]
                # ([apply_zp_fnu2flux])
                keywords = ['exptime', 'filter', 'obsdate']
                exptime_new, filt_new, obsdate_new = read_header (header_new,
                                                                  keywords)

                lat = get_par(set_zogy.obs_lat,tel)
                lon = get_par(set_zogy.obs_lon,tel)
                height = get_par(set_zogy.obs_height,tel)
                airmass_new = get_airmass (ra_cat_new[idx_cat_new],
                                           dec_cat_new[idx_cat_new],
                                           obsdate_new, lat, lon, height)
                zp_new = header_new['PC-ZP']
                # use extinction coefficient from header, as new and ref
                # may be different telescopes
                ext_coeff_new = header_new['PC-EXTCO']

                # check if MAG_OPT is present in table
                if 'MAG_OPT' in table_new.colnames:

                    flux_new, fluxerr_new = apply_zp_mag2flux (
                        table_new['MAG_OPT'].value, zp_new, airmass_new,
                        exptime_new, ext_coeff_new,
                        magerr=table_new['MAGERR_OPT'].value)

                else:

                    # if not, use FNU_OPT with function apply_zp_fnu2flux
                    flux_new, fluxerr_new = apply_zp_fnu2flux (
                        table_new['FNU_OPT'].value, zp_new, airmass_new,
                        exptime_new, ext_coeff_new,
                        fnuerr=table_new['FNUERR_OPT'].value)


                # same for reference image
                exptime_ref, filt_ref, obsdate_ref = read_header (header_ref,
                                                                  keywords)

                if 'AIRMASS' in header_ref and header_ref['AIRMASS']==1:
                    airmass_ref = 1
                    log.info ('airmass_ref: {}'.format(airmass_ref))
                else:
                    airmass_ref = get_airmass (ra_cat_ref[idx_cat_ref],
                                               dec_cat_ref[idx_cat_ref],
                                               obsdate_ref, lat, lon, height)

                zp_ref = header_ref['PC-ZP']
                ext_coeff_ref = header_ref['PC-EXTCO']

                # check if MAG_OPT is present in table
                if 'MAG_OPT' in table_ref.colnames:

                    flux_ref, fluxerr_ref = apply_zp_mag2flux (
                        table_ref['MAG_OPT'].value, zp_ref, airmass_ref,
                        exptime_ref, ext_coeff_ref,
                        magerr=table_ref['MAGERR_OPT'].value)

                else:

                    # if not, use FNU_OPT with function apply_zp_fnu2flux
                    flux_ref, fluxerr_ref = apply_zp_fnu2flux (
                        table_ref['FNU_OPT'].value, zp_ref, airmass_ref,
                        exptime_ref, ext_coeff_ref,
                        fnuerr=table_ref['FNUERR_OPT'].value)


                # calculate ratio
                fratio_match = np.ones_like(flux_new)
                # nonzero mask
                m_nz = ((flux_ref != 0) & (flux_new != 0))
                fratio_match[m_nz] = flux_new[m_nz] / flux_ref[m_nz]

                # corresponding error
                fratio_err_match = fratio_match[m_nz] * np.sqrt(
                    (fluxerr_new[m_nz] / flux_new[m_nz])**2 +
                    (fluxerr_ref[m_nz] / flux_ref[m_nz])**2)


            else:
                log.warning ('E_FLUX_OPT and/or MAG_OPT not available in '
                             'catalogs to infer flux ratios; using E_FLUX_AUTO')

        else:
            log.error ('zero or not an equal number of matches between PSF '
                       'stars and sources in new and reference catalogs; using '
                       'E_FLUX_AUTO for the flux ratio')


        log.info('median(fratio_match) using E_FLUX_OPT: {}'
                 .format(np.median(fratio_match)))



    # now also determine fratio, dx and dy for each subimage which can
    # be used in function [run_ZOGY]:
    fratio_subs = np.ones(nsubs)
    fratio_err_subs = np.zeros(nsubs)
    dx_subs = np.zeros(nsubs)
    dy_subs = np.zeros(nsubs)


    #---------------------------------------------------------------------------
    def get_mean_fratio (fratio, fratio_err, weighted=True):

        """helper function to calculate sigma-clipped (weighted) mean,
           median and std of fratio values; this is used below both
           for the full-frame and - if relevant - for the subimages
           calculations; the boolean [weighted] determines whether the
           normal or weighted mean is returned
        """

        # calculate full-frame average, standard deviation and median
        fratio_mean, fratio_med, fratio_std = (
            sigma_clipped_stats(fratio, mask_value=0))


        if weighted:
            # replace fratio_mean with weighted mean, where fratio_err
            # is first corrected such that distribution of fratio and
            # fratio_err has a reduced chi-square value of unity
            # (using function from dreamreduce: zeropoints.find_sigma)
            res = fratio - fratio_med
            sigma = find_sigma (res, fratio_err)
            log.info ('sigma in get_mean_fratio: {:.3f}'.format(sigma))
            fratio_vartot = fratio_err**2 + sigma**2

            # corresponding weights
            m_nz = (fratio_vartot != 0)
            weights = 1/fratio_vartot[m_nz]

            # weighted mean and its error
            fratio_mean = np.average(fratio[m_nz], weights=weights)
            fratio_werr = np.sqrt(1/np.sum(weights))

        else:
            # fratio_werr has not been defined yet; simply scale the
            # fratio_std with the square root of the number of
            # measurements
            fratio_werr = fratio_std / np.sqrt(len(fratio))


        return fratio_mean, fratio_med, fratio_std, fratio_werr

    #---------------------------------------------------------------------------


    # calculations below require a bare minimum of matches, otherwise
    # sigma_clipped_stats will return NaNs, which will cause an
    # exception as header values cannot contain NaNs
    nmatch_min = 15
    if nmatch > nmatch_min:

        success = True

        # fratio stats using function get_mean_fratio
        fratio_mean_full, fratio_med_full, fratio_std_full, fratio_err_full = \
            get_mean_fratio (fratio_match, fratio_err_match, weighted=True)


        # dx and dy
        dx_mean, dx_med, dx_std = sigma_clipped_stats(dx_match, mask_value=0)
        dy_mean, dy_med, dy_std = sigma_clipped_stats(dy_match, mask_value=0)
        dx_full = np.sqrt(dx_mean**2 + dx_std**2)
        dy_full = np.sqrt(dy_mean**2 + dy_std**2)

    else:

        success = False

        # set the values arbitrarily high
        fratio_mean_full, fratio_med_full = 100, 100
        fratio_std_full, fratio_err_full = 100, 100
        dx_mean, dx_med, dx_std = 100, 100, 100
        dy_mean, dy_med, dy_std = 100, 100, 100
        dx_full, dy_full = 100, 100



    log.info('full-frame fratio mean: {:.3f} ({:.3f}), med: {:.3f}, std: {:.3f}'
             .format(fratio_mean_full, fratio_err_full, fratio_med_full,
                     fratio_std_full))
    log.info('median dx: {:.3f} +- {:.3f} pix'.format(dx_med, dx_std))
    log.info('median dy: {:.3f} +- {:.3f} pix'.format(dy_med, dy_std))
    log.info('full-frame dx: {:.3f}, dy: {:.3f}'.format(dx_full, dy_full))


    # add header keyword(s):
    header['Z-DXYLOC'] = (get_par(set_zogy.dxdy_local,tel),
                          'star position offsets determined per subimage?')
    header['Z-DX'] = (dx_med, '[pix] dx median offset full image')
    header['Z-DXSTD'] = (dx_std, '[pix] dx sigma (STD) offset full image')
    header['Z-DY'] = (dy_med, '[pix] dy median offset full image')
    header['Z-DYSTD'] = (dy_std, '[pix] dy sigma (STD) offset full image')
    header['Z-FNROPT'] = (get_par(set_zogy.fratio_optflux,tel),
                          'optimal (T) or AUTO (F) flux used for flux ratio')
    header['Z-FNRLOC'] = (get_par(set_zogy.fratio_local,tel),
                          'flux ratios (Fnew/Fref) determined per subimage?')
    header['Z-FNR'] = (fratio_med_full,
                       'median flux ratio (Fnew/Fref) full image')
    header['Z-FNRSTD'] = (fratio_std_full,
                          'sigma (STD) flux ratio (Fnew/Fref) full image')
    header['Z-FNRERR'] = (fratio_err_full,
                          'weighted error flux ratio (Fnew/Fref) full image')


    #---------------------------------------------------------------------------
    def local_or_full (value_local, value_full, std_full, nsigma=3):
        # function to return full-frame value if local value is more
        # than [nsigma] (full frame) away from the full-frame value
        if (np.abs(value_local-value_full)/std_full > nsigma or
            not np.isfinite(value_local)):

            if get_par(set_zogy.verbose,tel):
                log.info('np.abs(value_local-value_full)/std_full: {}'
                         .format(np.abs(value_local-value_full)/std_full))
                log.info('adopted value: {}'.format(value_full))
            return value_full

        else:

            return value_local

    #---------------------------------------------------------------------------


    # loop subimages
    x_psf_new_match = x_psf_new[idx_psf_new]
    y_psf_new_match = y_psf_new[idx_psf_new]
    for nsub in range(nsubs):

        # start with full-frame values
        fratio_mean, fratio_std, fratio_med, fratio_err = (
            fratio_mean_full, fratio_std_full, fratio_med_full, fratio_err_full)
        dx, dy = dx_full, dy_full


        # [subcut] defines the pixel indices [y1 y2 x1 x2] identifying
        # the corners of the subimage in the entire input/output image
        # coordinate frame; used various times below
        subcut = cuts_ima[nsub]


        # determine mask of full-frame matched values belonging to
        # this subimage
        x_idx = (x_psf_new_match-0.5).astype('uint16')
        y_idx = (y_psf_new_match-0.5).astype('uint16')
        mask_sub = ((y_idx >= subcut[0]) & (y_idx < subcut[1]) &
                    (x_idx >= subcut[2]) & (x_idx < subcut[3]))


        # require a minimum number of values before adopting local
        # values for fratio, dx and dy
        if np.sum(mask_sub) >= nmatch_min:

            if get_par(set_zogy.fratio_local,tel):

                # local fratio stats using function get_mean_fratio
                fratio_mean, fratio_med, fratio_std, fratio_err =\
                    get_mean_fratio (fratio_match[mask_sub],
                                     fratio_err_match, weighted=True)
                fratio_mean = local_or_full (fratio_mean, fratio_mean_full,
                                             fratio_std_full)


            # and the same for dx and dy
            if get_par(set_zogy.dxdy_local,tel):

                # determine local values
                dx_mean, dx_med, dx_std = sigma_clipped_stats(
                    dx_match[mask_sub], mask_value=0)
                dy_mean, dy_med, dy_std = sigma_clipped_stats(
                    dy_match[mask_sub], mask_value=0)
                dx = np.sqrt(dx_mean**2 + dx_std**2)
                dy = np.sqrt(dy_mean**2 + dy_std**2)


                # adopt full-frame values if local values are more
                # than nsigma away from the full-frame values
                dx = local_or_full (dx, dx_full, dx_std)
                dy = local_or_full (dy, dy_full, dy_std)



        fratio_subs[nsub] = fratio_mean
        fratio_err_subs[nsub] = fratio_err
        dx_subs[nsub] = dx
        dy_subs[nsub] = dy


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_fratio_dxdy')

    return success, x_psf_new_match, y_psf_new_match, dx_match, dy_match, \
        fratio_subs, fratio_err_subs, dx_subs, dy_subs


################################################################################

def centers_cutouts(subsize, ysize, xsize, get_remainder=False):

    """Function that determines the input image indices (!) of the centers
    (list of nsubs x 2 elements) and cut-out regions (list of nsubs x
    4 elements) of image with the size xsize x ysize. Subsize is the
    fixed size of the subimages. The routine will fit as many of these
    in the full frames, and for the moment it will ignore any
    remaining pixels outside.

    """

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
    log.info('nxsubs: {}, nysubs: {}, nsubs: {}'.format(nxsubs, nysubs, nsubs))

    centers = np.zeros((nsubs, 2), dtype=int)
    cuts_ima = np.zeros((nsubs, 4), dtype=int)
    cuts_ima_fft = np.zeros((nsubs, 4), dtype=int)
    cuts_fft = np.zeros((nsubs, 4), dtype=int)
    sizes = np.zeros((nsubs, 2), dtype=int)

    border = get_par(set_zogy.subimage_border,tel)
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
            x = i*subsize + int(nx/2)
            y = j*subsize + int(ny/2)
            nsub += 1
            centers[nsub] = [y, x]
            cuts_ima[nsub] = [y-int(ny/2), y+int(ny/2), x-int(nx/2), x+int(nx/2)]
            y1 = np.amax([0,y-int(ny/2)-border])
            x1 = np.amax([0,x-int(nx/2)-border])
            y2 = np.amin([ysize,y+int(ny/2)+border])
            x2 = np.amin([xsize,x+int(nx/2)+border])
            cuts_ima_fft[nsub] = [y1,y2,x1,x2]
            cuts_fft[nsub] = [y1-(y-int(ny/2)-border),ysize_fft-(y+int(ny/2)+border-y2),
                              x1-(x-int(nx/2)-border),xsize_fft-(x+int(nx/2)+border-x2)]
            sizes[nsub] = [ny, nx]

    return centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes


################################################################################

def show_image(image):

    im = plt.imshow(np.real(image), origin='lower', cmap='gist_heat',
                    interpolation='nearest')
    plt.show(im)


################################################################################

def ds9_arrays(regions=None, **kwargs):

    cmd = ['ds9', '-zscale', '-zoom', '4', '-cmap', 'heat',
           '-frame', 'lock', 'image']

    if regions is not None:
        cmd += ['-regions', regions]

    for name, array in kwargs.items():
        # write array to fits
        fitsfile = 'ds9_{}.fits'.format(name)
        fits.writeto(fitsfile, np.array(array), overwrite=True)
        # append to command
        cmd.append(fitsfile)

    result = subprocess.run(cmd)


################################################################################

def create_brightcat (table_sex, nbright, base, use_allstars=False):

    # feed Astrometry.net only with brightest sources; N.B.: keeping
    # only objects with zero FLAGS does not work well in crowded
    # fields select stars for finding WCS solution
    #mask_use = (data_sexcat['FLAGS']<=1)

    # the above selection is not working well for crowded images
    # either; Danielle found the following to work well for both
    # non-crowded as crowded images, i.e. use all objects that are not
    # masked according to the input mask (any pixel within the
    # isophotal area of an object):
    if 'FLAGS_MASK' in table_sex.colnames:
        mask_use = (table_sex['FLAGS_MASK']==0)
    else:
        mask_use = (table_sex['FLAGS']<=3)


    # Gavin suggested using saturated stars is not much of an issue:
    # if use_allstars is True, disregard the FLAGs
    if use_allstars:
        mask_use = np.ones(len(table_sex), dtype=bool)


    # sort in brightness (E_FLUX_AUTO)
    if 'E_FLUX_AUTO' in table_sex.colnames:
        column_sort = 'E_FLUX_AUTO'
    elif 'E_FLUX_OPT' in table_sex.colnames:
        column_sort = 'E_FLUX_OPT'
    else:
        column_sort = 'E_FLUX_APER_R5xFWHM'


    index_sort = np.argsort(table_sex[mask_use][column_sort])
    # sorting is done by flux, so reverse [index_sort] to have the
    # brightest objects first
    index_sort = index_sort[::-1]


    # subset of table_sex, sorted in brightness
    table_sex_use_sort = table_sex[mask_use][index_sort]


    # select the brightest objects
    index_bright = uniform_subset (table_sex_use_sort['X_POS'].value,
                                   table_sex_use_sort['Y_POS'].value, nbright)
    table_sex_bright = table_sex_use_sort[index_bright]

    # save to fits
    sexcat_bright = '{}_cat_bright.fits'.format(base)
    table_sex_bright.write(sexcat_bright, format='fits', overwrite=True)


    # create ds9 regions text file to show the brightest stars
    if get_par(set_zogy.make_plots,tel):
        prep_ds9regions('{}_cat_bright_ds9regions.txt'.format(base),
                        table_sex_bright['X_POS'].value,
                        table_sex_bright['Y_POS'].value,
                        radius=5., width=2, color='green',
                        value=np.arange(1,nbright+1))


    return mask_use, column_sort, index_sort, index_bright


################################################################################

def run_wcs (image_in, ra, dec, pixscale, width, height, header, imtype):

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing run_wcs ...')


    # pixel scale range
    varyfrac = get_par(set_zogy.pixscale_varyfrac,tel)
    scale_low = (1 - varyfrac) * pixscale
    scale_high = (1 + varyfrac) * pixscale


    # read SExtractor catalogue (this is also used further down below
    # in this function)
    base = image_in.replace('.fits','')
    sexcat = '{}_cat.fits'.format(base)
    table_sex = Table.read(sexcat, memmap=True)


    # output folder
    dir_out = '.'
    if '/' in base:
        dir_out, __ = os.path.split(base)
        #dir_out = '/'.join(base.split('/')[:-1])


    # use brightest nbright stars from catalog
    nbright = get_par(set_zogy.ast_nbright,tel)

    # name of bright star catalog
    sexcat_bright = '{}_cat_bright.fits'.format(base)


    # string with depths
    dstep = 50
    depth_str = (str(list(range(dstep,nbright//3+1,dstep)))
                 .replace('[','').replace(']','').replace(' ',''))


    # solve-field command
    cmd = ['solve-field', '--no-plots',
           '--config', get_par(set_zogy.astronet_config,tel),
           #'--no-fits2fits', cloud version of astrometry does not have this arg
           '--x-column', 'X_POS',
           '--y-column', 'Y_POS',
           #'--sort-column', column_sort,  # added further below
           '--no-remove-lines',
           '--uniformize', '0',
           # only work on brightest sources
           #'--objs', str(nbright),
           '--width', str(width),
           '--height', str(height),
           #'--keep-xylist', sexcat,
           # ignore existing WCS headers in fits input images
           #'--no-verify',
           #'--verbose',
           #'--verbose',
           '--crpix-center',
           #'--code-tolerance', str(0.01),
           #'--quad-size-min', str(0.1),
           # for KMTNet images restrict the max quad size:
           #'--quad-size-max', str(0.1),
           # number of field objects to look at:
           '--depth', depth_str,
           #'--scamp', scampcat,
           # give up solving after the specified number of seconds of CPU time
           '--cpulimit', '120',
           sexcat_bright,
           '--tweak-order', str(get_par(set_zogy.astronet_tweak_order,tel)),
           '--scale-low', str(scale_low),
           '--scale-high', str(scale_high),
           '--scale-units', 'app',
           '--ra', str(ra),
           '--dec', str(dec),
           '--radius', str(get_par(set_zogy.astronet_radius,tel)),
           '--new-fits', 'none',
           '--overwrite',
           '--out', base.split('/')[-1],
           '--dir', dir_out
    ]


    # parity: whether transformation in CD matrix involves a
    # reflection or just rotation; for ML this is "neg" and for BG
    # this is "pos"; include this for ML and BG here, which makes it a
    # bit easier to find the solution, but not really needed
    if tel[0:2] in ['ML','BG']:
        if tel=='ML1':
            parity_str = 'neg'
        else:
            parity_str = 'pos'

        cmd += ['--parity', parity_str]



    if get_par(set_zogy.timing,tel):
        t2 = time.time()



    # perform 2 iterations of solve-field: first one does not use
    # stars that are flagged because of e.g. saturation; this makes it
    # more difficult for Astrometry.net to find matches with the index
    # files, which have a fixed bright magnitude limit. If that fails,
    # another attempt is made using all stars irrespective of their
    # flags. This will lead to a larger error on the solution, up to a
    # factor of about 2 for fields with many saturated stars, but that
    # is still acceptable.
    for niter in range(2):

        if niter == 0:
            mask_use, column_sort, index_sort, index_bright = create_brightcat (
                table_sex, nbright, base)
        else:
            mask_use, column_sort, index_sort, index_bright = create_brightcat (
                table_sex, nbright, base, use_allstars=True)


        # add sorting column; even though catalog with bright entries
        # is already sorted in flux, with brightest objects at the
        # top, this is sorting column is important to include
        cmd_process = cmd + ['--sort-column', column_sort]

        # log cmd executed
        cmd_str = ' '.join(cmd_process)
        log.info('Astrometry.net command executed:\n{}'.format(cmd_str))

        result = subprocess.run(cmd_process, capture_output=True)
        status = result.returncode
        log.info('stdout: {}'.format(result.stdout.decode('UTF-8')))
        log.info('stderr: {}'.format(result.stderr.decode('UTF-8')))
        log.info('status: {}'.format(status))


        if isfile('{}.solved'.format(base)) and status==0:

            # in case of success after 2nd attempt: re-determine
            # mask_use and index_sort for non-flagged stars, so
            # saturated stars are not included in solution statistics
            # calculated below
            if niter == 1:
                mask_use, column_sort, index_sort, index_bright = create_brightcat (
                    table_sex, nbright, base)


            # break out of loop
            break


        else:

            # try again using all stars, i.e. included the saturated ones
            if niter == 0:
                log.info ('solve-field (Astrometry.net) first attempt failed '
                          'with exit code {}; trying again using all bright '
                          'stars, irrespective of their flags'.format(status))
            else:
                msg = ('solve-field (Astrometry.net) failed with exit code {} '
                       'on second attempt'.format(status))
                log.exception(msg)
                raise Exception(msg)



    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t2, label='to execute solve-field')



    # read .match file, which describes the quad match that solved the
    # image, before it is deleted
    table_match = Table.read('{}.match'.format(base), memmap=True)

    # calculate median x- and y-position of the calibration stars
    # recorded by Astrometry.net, to see if they are well spread
    # across the entire image
    filename_xyls = '{}-indx.xyls'.format(base)
    if isfile(filename_xyls):

        table_xyls = Table.read(filename_xyls, memmap=True)
        log.info ('median normalized x-coord. of Anet index stars in FOV: '
                  '{:.2f} +- {:.2f}'
                  .format(np.median(table_xyls['X'])/width,
                          np.std(table_xyls['X'])/width))
        log.info ('median normalized y-coord. of Anet index stars in FOV: '
                  '{:.2f} +- {:.2f}'
                  .format(np.median(table_xyls['Y'])/height,
                          np.std(table_xyls['Y'])/height))


    # read header saved in .wcs
    wcsfile = '{}.wcs'.format(base)
    header_wcs = read_hdulist (wcsfile, get_data=False, get_header=True)


    # remove HISTORY, COMMENT and DATE fields from Astrometry.net header
    # they are still present in the base+'.wcs' file
    header_wcs.pop('HISTORY', None)
    header_wcs.pop('COMMENT', None)
    header_wcs.pop('DATE', None)


    # add specific keyword indicating index file of match
    if table_match['HEALPIX'][0] > 0:
        anet_index = 'index-{}-{:02d}.fits'.format(table_match['INDEXID'][0],
                                                   table_match['HEALPIX'][0])
    else:
        anet_index = 'index-{}.fits'.format(table_match['INDEXID'][0])


    header_wcs['A-INDEX'] = (anet_index, 'name of A.net index file WCS solution')

    # add number of calibration stars in index file
    header_wcs['A-NINDEX'] = (table_match['NINDEX'][0], 'number of stars in '
                              'A.net index file in FOV')

    # add number of matching quads
    header_wcs['A-NQUADS'] = (table_match['QMATCHED'][0], 'number of '
                              'matching A.net quads used for WCS')


    # and pixelscale; see Eq. 189 of Calabretta and Greisen
    # (previously the signs in the sin of CD1_2 and CD2_1 below were
    # switched around)
    CD1_1 = header_wcs['CD1_1']  # CD1_1 = CDELT1 *  cos (CROTA2)
    CD1_2 = header_wcs['CD1_2']  # CD1_2 = CDELT2 * -sin (CROTA2)
    CD2_1 = header_wcs['CD2_1']  # CD2_1 = CDELT1 *  sin (CROTA2)
    CD2_2 = header_wcs['CD2_2']  # CD2_2 = CDELT2 *  cos (CROTA2)


    # this is different than astronet's report of the pixel scale,
    # which includes the polynomial distortions
    anet_pixscale_x = np.sqrt(CD1_1**2 + CD2_1**2) * 3600
    anet_pixscale_y = np.sqrt(CD1_2**2 + CD2_2**2) * 3600
    anet_pixscale = np.average([anet_pixscale_x, anet_pixscale_y])

    # rotation with the angle between the North and the second axis
    # (y-axis) of the image, counted positive to the East;

    # determinant of the CD matrix, which indicates whether the
    # transformation is proper or improper - in the improper case the
    # rotation involves a reflection
    det = CD1_1 * CD2_2 - CD1_2 * CD2_1
    if det > 0:
        parity = 1
    else:
        parity = -1

    # confusingly, in astrometry.net, parity is indicated as "pos" if
    # above parity=-1; e.g. for BlackGEM det<0, so parity=-1, but
    # indicated as parity "pos" in astrometry.net output; may have to
    # do with astronomical convention that East is 90 degrees
    # counterclockwise from North, whereas the rotation is positive in
    # clockwise direction.
    anet_rot_x = -np.degrees(np.arctan2(parity*CD2_1, parity*CD1_1))
    anet_rot_y = -np.degrees(np.arctan2(-CD1_2, CD2_2))
    anet_rot = np.average([anet_rot_x, anet_rot_y])

    # in Astrometry.net code, pixscale and rotation are determined as
    # follows, which are not quite the same but consistent with the
    # values above; adopt these Anet determinations
    anet_pixscale = 3600*np.sqrt(np.abs(det))
    #T = parity*CD1_1 + CD2_2
    #A = parity*CD2_1 - CD1_2
    anet_rot = -np.degrees(np.arctan2(parity*CD2_1 - CD1_2,
                                      parity*CD1_1 + CD2_2))

    # add header keywords
    header_wcs['A-PSCALE'] = (anet_pixscale,
                              '[arcsec/pix] pixel scale WCS solution')
    header_wcs['A-PSCALX'] = (anet_pixscale_x,
                              '[arcsec/pix] X-axis pixel scale WCS solution')
    header_wcs['A-PSCALY'] = (anet_pixscale_y,
                              '[arcsec/pix] Y-axis pixel scale WCS solution')

    header_wcs['A-ROT'] = (anet_rot,
                           '[deg] rotation WCS solution (from N to y-axis)')
    header_wcs['A-ROTX'] = (anet_rot_x,
                            '[deg] X-axis rotation WCS (from N to y-axis)')
    header_wcs['A-ROTY'] = (anet_rot_y,
                            '[deg] Y-axis rotation WCS (from N to y-axis)')

    # convert SIP header keywords from Astrometry.net to PV keywords
    # that swarp, scamp (and sextractor) understand using this module
    # from David Shupe: sip_to_pv

    # using the old version of sip_to_pv (before June 2017):
    #status = sip_to_pv(image_in, image_in, tpv_format=True)
    #if status == False:
    #    log.error('sip_to_pv failed.')
    #    return 'error'

    # new version (June 2017) of sip_to_pv works on image header
    # rather than header+image (see below); the header is modified in
    # place; compared to the old version this saves an image write
    result = sip_to_pv(header_wcs, tpv_format=True, preserve=False)

    # strip header_wcs from any image- or table-dependent keywords
    header_wcs.strip()

    # update header with wcs keywords
    header.update(header_wcs)


    # use astropy.WCS to find RA, DEC corresponding to X_POS,
    # Y_POS, based on WCS info saved by Astrometry.net in .wcs
    # file (wcsfile). The 3rd parameter to wcs.all_pix2world indicates
    # the pixel coordinate of the frame origin. Using astropy.WCS
    # avoids having to save the new RAs and DECs to file and read them
    # back into python arrays. It provides the same RA and DEC as
    # wcs-xy2rd and also as SExtractor run independently on the WCS-ed
    # image.
    # N.B.: WCS accepts header objects - gets rid of old warning about
    # axis mismatch, as .wcs files have NAXIS=0, while proper image
    # header files have NAXIS=2
    wcs = WCS(header)
    newra, newdec = wcs.all_pix2world(table_sex['X_POS'], table_sex['Y_POS'], 1)

    # update catalog with new RA and DEC columns
    table_sex['RA'] = newra
    table_sex['DEC'] = newdec
    table_sex.write(sexcat, format='fits', overwrite=True)


    # also infer RA, DEC of central pixel
    xsize = width
    ysize = height
    ra_center, dec_center = wcs.all_pix2world(xsize/2+0.5, ysize/2+0.5, 1)
    log.info('ra_center: {:.4f}, dec_center: {:.4f}'.format(ra_center,
                                                            dec_center))

    # add RA-CNTR and DEC-CNTR to header
    header['RA-CNTR'] = (float(ra_center),
                         '[deg] RA (ICRS) at image center (astrom.net)')
    header['DEC-CNTR'] = (float(dec_center),
                          '[deg] DEC (ICRS) at image center (astrom.net)')


    if get_par(set_zogy.timing,tel):
        t3 = time.time()


    # check how well the WCS solution just found, compares with the
    # coordinates in the calibration catalog
    fits_cal = get_par(set_zogy.cal_cat,tel)
    if isfile (fits_cal):

        # field-of-view
        fov_half_deg = np.amax([xsize, ysize]) * pixscale / 3600 / 2

        # date of observation for proper motion correction
        obsdate = read_header(header, ['obsdate'])

        # select calibration stars that are within the image
        if '20181115' in fits_cal:
            # old calibration catalog is indexed by 1-degree declination
            # zones, so need to use [get_imagestars_zones]
            table_cal = get_imagestars_zones (fits_cal, obsdate, ra_center,
                                              dec_center, fov_half_deg)

        else:
            # new calibration catalog is indexed by HEALPix indices
            table_cal = get_imagestars_hpindex (fits_cal, obsdate, ra_center,
                                                dec_center, fov_half_deg,
                                                rot=anet_rot)


        log.info ('selected {} calibration stars within the image {}, with '
                  'central ra,dec: {:.4f},{:.4f}, a field-of-view of {:.3f} '
                  'degrees and DATE-OBS: {}'
                  .format(len(table_cal), base, ra_center, dec_center,
                          2*fov_half_deg, obsdate))


        # write table_cal to fits table so it can be re-used in the
        # photometric calibration; one file for each imtype as the
        # shift may be considerable
        if table_cal is not None:
            fits_calcat_field = '{}_calcat_field_{}.fits'.format(base, imtype)
            table_cal.write (fits_calcat_field, format='fits', overwrite=True)


        n_ast = len(table_cal)
        log.info('number of potential astrometric stars in FOV: {}'
                 .format(n_ast))


        # add header keyword(s):
        cal_name = fits_cal.split('/')[-1]
        header['A-CAT-F'] = (cal_name, 'astrom. cat')
        header['A-TNAST'] = (n_ast, 'total number of astrometric stars in FOV')


        # calculate array of offsets between astrometry comparison
        # stars and any non-saturated SExtractor source
        newra_use_sort = newra[mask_use][index_sort]
        newdec_use_sort = newdec[mask_use][index_sort]
        __, index_ast, __, dra_array, ddec_array = get_matches (
            newra_use_sort[index_bright], newdec_use_sort[index_bright],
            table_cal['ra'].value, table_cal['dec'].value, dist_max=2)


        # this is the number of matches that Anet could find
        # between brightest stars detected and the number of stars
        # in the index file for this field
        n_ast_used = table_match['NMATCH'][0]

        log.info('number of astrometric stars used: {}'.format(n_ast_used))
        header['A-NAST'] = (n_ast_used,
                            'number of brightest matching stars used for WCS')
        header['A-NAMAX'] = (get_par(set_zogy.ast_nbright,tel),
                             'input max. no. of stars detected to use for WCS')


        # calculate means, stds and medians
        dra_mean, dra_median, dra_std = sigma_clipped_stats(
            dra_array, sigma=5, mask_value=0)
        ddec_mean, ddec_median, ddec_std = sigma_clipped_stats(
            ddec_array, sigma=5, mask_value=0)

        log.info('dra_mean [arcsec]: {:.3f}, dra_std: {:.3f}, dra_median: {:.3f}'
                 .format(dra_mean, dra_std, dra_median))
        log.info('ddec_mean [arcsec]: {:.3f}, ddec_std: {:.3f}, ddec_median: '
                 '{:.3f}'.format(ddec_mean, ddec_std, ddec_median))

        # add header keyword(s):
        header['A-DRA'] = (dra_median,
                           '[arcsec] dRA median offset to astrom. catalog')
        header['A-DRASTD'] = (dra_std, '[arcsec] dRA sigma (STD) offset')
        header['A-DDEC'] = (ddec_median,
                            '[arcsec] dDEC median offset to astrom. catalog')
        header['A-DDESTD'] = (ddec_std, '[arcsec] dDEC sigma (STD) offset')


        # this is the number of matches found above between
        # brightest stars detected and the total number of catalog
        # stars in this field
        n_ast_offset = np.shape(dra_array)[0]
        header['A-NA-OFF'] = (n_ast_offset, 'number of matching stars '
                              'used in offset calc.')


        # check if original ML/BG filename contains 'adc'
        is_adcfile = 'ORIGFILE' in header and 'adc' in header['ORIGFILE'].lower()

        if get_par(set_zogy.make_plots,tel) or is_adcfile:

            log.info ('creating dRA vs dDEC plot, including histograms')

            # plot of dra vs. ddec in arcseconds, including histograms
            dr = np.sqrt(dra_std**2+ddec_std**2)
            dr = max(dr, 0.05)
            limits = 5. * np.array([-dr,dr,-dr,dr])
            mask_nonzero = ((dra_array!=0) & (ddec_array!=0))
            label1 = r'dRA={:.3f}$\pm${:.3f}"'.format(dra_median, dra_std)
            label2 = r'dDEC={:.3f}$\pm${:.3f}"'.format(ddec_median, ddec_std)
            title = '{}\n{}'.format(base.split('/')[-1], header['ORIGFILE'])

            result = plot_scatter_hist(
                dra_array[mask_nonzero], ddec_array[mask_nonzero], limits,
                xlabel='delta RA [arcsec]', ylabel='delta DEC [arcsec]',
                label=[label1,label2], labelpos=[(0.77,0.9),(0.77,0.85)],
                filename='{}_dRADEC.pdf'.format(base), title=title)


            # plot of dra vs. color and ddec vs. color, to check for
            # offset dependence on stellar color
            col = ['g_{}'.format(tel[0:2]), 'r_{}'.format(tel[0:2])]
            if (col[0] in table_cal.colnames and
                col[1] in table_cal.colnames and np.sum(mask_nonzero) >= 10):

                log.info ('creating dRA and dDEC vs color plots')

                col_array = (table_cal[col[0]][index_ast] -
                             table_cal[col[1]][index_ast])

                fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, sharey=False)
                fig.subplots_adjust(hspace=0, wspace=0)
                fig.suptitle(title, fontsize=10)

                # dRA subplot
                ax0.plot(col_array, dra_array, 'o', color='tab:blue',
                         markersize=5, markeredgecolor='k')
                ax0.set_ylabel('delta RA [arcsec]')
                # 1st degree polynomial fit to determin slope
                p_ra, V_ra = np.polyfit(col_array, dra_array, 1, cov=True)
                label_ra = r'slope={:.3f}$\pm${:.3f}'.format(p_ra[0],
                                                             np.sqrt(V_ra[0,0]))
                ax0.annotate(label_ra, xy=(1,1), xycoords='axes fraction',
                             fontsize=10, ha='right', va='top')
                # plot fit
                xp = np.linspace(np.amin(col_array), np.amax(col_array))
                ax0.plot (xp, xp*p_ra[0]+p_ra[1], color='k')

                # dDEC subplot
                ax1.plot(col_array, ddec_array, 'o', color='tab:blue',
                         markersize=5, markeredgecolor='k')
                ax1.set_ylabel('delta DEC [arcsec]')
                ax1.set_xlabel('{} - {} colour'.format(col[0], col[1]))

                # 1st degree polynomial fit to determin slope
                p_dec, V_dec = np.polyfit(col_array, ddec_array, 1, cov=True)
                label_dec = r'slope={:.3f}$\pm${:.3f}'.format(p_dec[0],
                                                              np.sqrt(V_dec[0,0]))
                ax1.annotate(label_dec, xy=(1,1), xycoords='axes fraction',
                             fontsize=10, ha='right', va='top')
                # plot fit
                ax1.plot (xp, xp*p_dec[0]+p_dec[1], color='k')


                plt.savefig('{}_dRADEC_vs_color.pdf'.format(base))
                if get_par(set_zogy.show_plots,tel): plt.show()
                plt.close()

                # also save fits file with calibration info and
                # offsets for ADC analysis
                table_cal_offsets = Table(table_cal[index_ast])
                table_cal_offsets['DRA'] = dra_array
                table_cal_offsets['DDEC'] = ddec_array
                table_cal_offsets.write ('{}_dRADEC_calcat.fits'.format(base),
                                         format='fits', overwrite=True)


    else:
        log.error ('calibration catalog {} not found!'
                   .format(get_par(set_zogy.cal_cat,tel)))



    # update header of input image
    update_imcathead (image_in, header)


    # remove file(s) if not keeping intermediate/temporary files
    if not get_par(set_zogy.keep_tmp,tel):

        if 'sexcat_bright' in locals():
            remove_files ([sexcat_bright], verbose=True)

        # astrometry.net output files
        list2remove = ['{}{}'.format(base, ext) for ext in
                       ['.solved', '.match', '.rdls', '.corr', '-indx.xyls',
                        '.axy', '.wcs']]
        remove_files (list2remove, verbose=True)



    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t3, label='to calculate offset wrt external catalog')

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in run_wcs')

    return


################################################################################

def uniform_subset (xcoords, ycoords, nselect, nregions=128):

    """Subdivide input coordinates over [nregions] of equal surface
    and return the indices of the first ~[nselect]/[nregions] entries
    for each region. If the input coordinates are ordered, the same
    order is preserved for the output indices; e.g. if sorted by
    brightness, the brightest objects for each region are returned.

    """

    # minimum and maximum coordinates
    xmin = min(xcoords)
    xmax = max(xcoords)
    ymin = min(ycoords)
    ymax = max(ycoords)
    xsize = xmax - xmin
    ysize = ymax - ymin

    # total image size set by min and max coordinates divided into
    # nregions, resulting in approximate size of square region (of
    # which nregions fit into entire image)
    size_region = int(np.sqrt(xsize*ysize/nregions))

    # number of coordinates to be selected per pixel
    n_perpix = nselect / (xsize*ysize)

    # list to record indices of uniform set of coordinates
    indices_out = []

    # loop over regions in x
    for x in np.arange(xmin, xmax, size_region):

        # effective region xsize; number of regions that fit into
        # total xsize is not an integer, so while normally dx =
        # size_region, it will be smaller when x is nearing xmax
        dx = min(size_region, xmax-x)

        # loop over regions in y
        for y in np.arange(ymin, ymax, size_region):
            dy = min(size_region, ymax-y)

            # indices of all coordinates in this region
            indices_tmp = np.nonzero((xcoords>=x) & (xcoords<(x+size_region)) &
                                     (ycoords>=y) & (ycoords<(y+size_region)))[0]

            # record subset of coordinates in this region in indices_out
            n = max(int(dx*dy*n_perpix+0.5),1)
            indices_out += list(indices_tmp[0:n])


    # return indices_out, which might have slightly more entries than
    # nselect
    return indices_out[0:nselect]


################################################################################

def get_matches (ra1, dec1, ra2, dec2, dist_max=None, return_offsets=True):

    """Find closest match for sources at [ra1], [dec1] among sources at
    [ra2], [dec2]. If [dist_max] is provided (units: arcseconds), only
    the (indices of) sources with total offsets within that limit are
    returned. The indices of the matching sources, one for ra1/dec1
    and one for ra2/dec2, are returned. If [return_offsets] is True,
    the total offset and the RA and DEC offsets from ra1/dec1 to
    ra2/dec2 (i.e. ra1 or dec1 + offset = ra2 or dec2) are also
    returned in units of arcseconds. The input ras and decs are
    assumed to be in units of degrees.

    """

    coords1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    index1 = np.arange(len(ra1))

    coords2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    index2, dist, __ = coords1.match_to_catalog_sky(coords2)

    if dist_max is not None:
        mask_match = (dist.arcsec <= dist_max)
        index1 = index1[mask_match]
        index2 = index2[mask_match]
        dist = dist[mask_match]

    if return_offsets:
        dra, ddec = coords1[index1].spherical_offsets_to(coords2[index2])
        return index1, index2, dist.arcsec, dra.arcsec, ddec.arcsec
    else:
        return index1, index2


################################################################################

def get_matches_pix (x1, y1, x2, y2, dist_max=None, return_offsets=True):

    """Find closest match for sources at pixel coordinates [x1], [y1]
    among sources at [x2], [y2]. If [dist_max] is provided (units:
    pixels), only the (indices of) sources with total offsets within
    that limit are returned. The indices of the matching sources, one
    for x1,y1 and one for x2,y2, are returned. If [return_offsets] is
    True, the total offset and the x and y offsets from x1,y1 to x2,y2
    are also returned.

    """

    # convert input coordinates to (x,y) pairs
    coords1 = np.array(list(zip(x1, y1)))
    coords2 = np.array(list(zip(x2, y2)))

    # use [scipy.spatial.distance.cdist] to calculate 2D distance
    # matrix with shape (ncoords1, ncoords2)
    dist = cdist(coords1, coords2)

    # apply distance cut
    if dist_max is not None:
        mask_dist = (dist <= dist_max)
    else:
        # use all
        mask_dist = np.ones_like(dist, dtype=bool)

    # convert dist to masked array, where distances larger than
    # dist_max - if provided - are masked out
    dist_masked = np.ma.masked_array(dist, mask=~mask_dist)

    # infer indices of (x2,y2) that are closest to (x1,y1)
    idx2 = np.ma.argmin(dist_masked, axis=1)

    # mask of mask_dist along axis=0 for which there is at least one
    # entry along axis=1
    mask_valid = np.any(mask_dist, axis=1)

    # indices of coords1
    idx1 = np.arange(len(coords1))

    # keep valid indices
    idx1 = idx1[mask_valid]
    idx2 = idx2[mask_valid]


    if return_offsets:
        dx = x2[idx2] - x1[idx1]
        dy = y2[idx2] - y1[idx1]
        dr = np.sqrt(dx**2 + dy**2)
        return idx1, idx2, dr, dx, dy
    else:
        return idx1, idx2


################################################################################

def ldac2fits_alt (cat_ldac, cat_fits):

    """This function converts the LDAC binary fits table from SExtractor
    to a common binary fits table (that can be read by Astrometry.net).
    It is taking up more memory for a very large LDAC file:

      memory use [GB]: rss=1.607, maxrss=1.735, vms=12.605 in ldac2fits

    compared to the original ldac2fits function:

      memory use [GB]: rss=0.430, maxrss=1.604, vms=3.811 in ldac2fits

    so keep using the original one which uses the record array helper
    function 'drop_fields', but that appears to be a valid numpy
    utility (see: https://numpy.org/devdocs/user/basics.rec.html).

    """

    if get_par(set_zogy.timing,tel):
        t = time.time()

    log.info('executing ldac2fits_alt ...')

    # read 2nd extension of input LDAC catalog
    table = Table.read(cat_ldac, hdu=2, memmap=True)

    # delete VIGNET column
    #del table['VIGNET']
    table.remove_column('VIGNET')

    # write output fits table
    table.write (cat_fits, format='fits', overwrite=True)

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in ldac2fits_alt')

    return


################################################################################

def ldac2fits (cat_ldac, cat_fits):

    """This function converts the LDAC binary fits table from SExtractor
    to a common binary fits table (that can be read by Astrometry.net) """

    if get_par(set_zogy.timing,tel):
        t = time.time()

    log.info('executing ldac2fits ...')

    # read input table and write out primary header and 2nd extension
    columns = []
    with fits.open(cat_ldac, memmap=True) as hdulist:

        # delete VIGNET column
        hdulist[2].data = drop_fields(hdulist[2].data, 'VIGNET')

        # and write regular fits file
        hdulist_new = fits.HDUList([hdulist[0], hdulist[2]])
        hdulist_new.writeto(cat_fits, overwrite=True)
        hdulist_new.close()


    # now that normal fits catalog (not LDAC) has been created, rename
    # some of the columns using the function rename_catcols
    rename_catcols(cat_fits)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in ldac2fits')

    return


################################################################################

def run_remap(image_new, image_ref, image_out, image_out_shape, gain=1,
              config=None, resample='Y', resampling_type='LANCZOS3',
              projection_err=0.001, mask=None, header_only='N',
              resample_suffix='_resamp.fits', resample_dir='.', dtype='float32',
              value_edge=0, timing=True, nthreads=0, oversampling=0, tel=None,
              set_zogy=None):

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

    if timing: t = time.time()
    log.info('executing run_remap ...')

    header_new = read_hdulist (image_new, get_data=False, get_header=True)
    header_ref = read_hdulist (image_ref, get_data=False, get_header=True)

    # create .head file with header info from [image_new]
    header_out = header_new.copy()

    # not necessary to copy these reference image keywords to the
    # remapped image as its header is not used anyway
    if False:
        # copy some keywords from header_ref
        for key in ['exptime', 'satlevel', 'gain']:
            value = get_keyvalue(key, header_ref)
            try:
                key_name = eval('set_zogy.key_{}'.format(key))
            except:
                key_name = key.capitalize()
            header_out[key_name] = value


    # delete some others
    for key in ['WCSAXES', 'NAXIS1', 'NAXIS2']:
        if key in header_out:
            del header_out[key]
    # write to .head file
    with open(image_out.replace('.fits','.head'),'w') as newrefhdr:
        for card in header_out.cards:
            newrefhdr.write('{}\n'.format(card))

    size_str = '{},{}'.format(image_out_shape[1], image_out_shape[0])
    cmd = ['swarp', image_ref, '-c', config, '-IMAGEOUT_NAME', image_out,
           '-IMAGE_SIZE', size_str, '-GAIN_DEFAULT', str(gain),
           '-RESAMPLE', resample,
           '-RESAMPLING_TYPE', resampling_type,
           '-OVERSAMPLING', str(oversampling),
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

    result = subprocess.run(cmd, capture_output=True)
    status = result.returncode
    log.info('stdout: {}'.format(result.stdout.decode('UTF-8')))
    log.info('stderr: {}'.format(result.stderr.decode('UTF-8')))
    log.info('status: {}'.format(status))


    if status != 0:
        msg = 'SWarp failed with exit code {}'.format(status)
        log.exception(msg)
        raise Exception(msg)

    if run_alt:
        image_resample = image_out.replace('_remap.fits', resample_suffix)
        data_resample, header_resample = read_hdulist(image_resample,
                                                      get_header=True)
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
        data_remap = np.zeros(image_out_shape, dtype=dtype)
        data_remap += value_edge
        # and place resampled image in output image
        index_resample = tuple([slice(y0,y0+ysize_resample),
                                slice(x0,x0+xsize_resample)])

        data_remap[index_resample] = data_resample

        # set edge values
        if value_edge != 0:

            # value_edge is nonzero, so assuming that this involves a
            # mask image; that can have islands of zeros in the edge
            # region; detect these and set them to [value_edge]

            # set edge pixels to zero and others to one
            mask_edge = (data_remap & value_edge == value_edge)
            data_label = np.zeros_like(data_remap, dtype=int)
            data_label[~mask_edge] = 1
            # detect features among the non-zero pixels
            struct = np.ones((3,3), dtype=bool)
            n_feat = ndimage.label (data_label, structure=struct,
                                    output=data_label)
            if n_feat > 1:
                # if more than one feature, set the pixels of all
                # features but the largest one to [value_edge]
                values_feat = np.arange(n_feat)+1
                sizes_feat = [np.sum(data_label==val) for val in values_feat]
                i_max = np.argmax(sizes_feat)
                for i, val in enumerate(values_feat):
                    if i != i_max:
                        mask_feat = (data_label==val)
                        data_remap[mask_feat] = value_edge



        # write to fits [image_out] with correct header; N.B.:
        # remapped reference bkg/std/mask image will currently be the
        # same for different new images
        header_out['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(image_out, data_remap, header_out, overwrite=True)


        # for ML/BG: remove resampled image and its weight to save
        # some disk space
        if tel[0:2] in ['ML','BG'] and not get_par(set_zogy.keep_tmp,tel):

            image_resample_weight = image_resample.replace('.fits',
                                                           '.weight.fits')
            remove_files ([image_resample, image_resample_weight], verbose=True)



    if timing:
        log_timing_memory (t0=t, label='in run_remap')

    return


################################################################################

def get_fwhm (fits_cat, fraction, class_sort=False, get_elong=False, nmin=5):

    """Function that accepts a fits table produced by SExtractor and
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

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing get_fwhm ...')

    data = read_hdulist (fits_cat)


    # these arrays correspond to objecst with flag==0 and flux_auto>0
    # add a S/N requirement
    index = ((data['FLAGS']==0) & (data['FLUX_AUTO']>0) &
             (data['FLUXERR_AUTO']>0) &
             (data['FLUX_AUTO']/data['FLUXERR_AUTO']>20))
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
    index_select = np.arange(-int(len(index_sort)*fraction+0.5),-1)
    fwhm_select = fwhm[index_sort][index_select]
    if get_elong:
        elong_select = elong[index_sort][index_select]


    # determine mean, median and standard deviation through sigma clipping
    __, fwhm_median, fwhm_std = sigma_clipped_stats(
        fwhm_select.astype(float), mask_value=0)
    # set values to zero if too few stars are selected or values
    # just determined are non-finite
    if (len(fwhm_select) < nmin or
        not (np.isfinite(fwhm_median) and np.isfinite(fwhm_std))):
        fwhm_median, fwhm_std = [0] * 2

    if get_par(set_zogy.verbose,tel):
        log.info('catalog: {}'.format(fits_cat))
        log.info('fwhm_median: {:.2f} pix, fwhm_std: {:.2f} pix'
                 .format(fwhm_median, fwhm_std))


    if get_elong:
        # determine mean, median and standard deviation through sigma clipping
        __, elong_median, elong_std = sigma_clipped_stats(
            elong_select.astype(float), mask_value=0)
        # set values to zero if too few stars are selected or values
        # just determined are non-finite
        if (len(elong_select) < nmin or
            not (np.isfinite(elong_median) and np.isfinite(elong_std))):
            elong_median, elong_std = [0] * 2

        if get_par(set_zogy.verbose,tel):
            log.info('elong_median: {:.3f}, elong_std: {:.3f}'
                     .format(elong_median, elong_std))



    if get_par(set_zogy.make_plots,tel):

        # best parameter to plot vs. FWHM is MAG_AUTO
        mag_auto_select = mag_auto[index_sort][index_select]

        # to get initial values before discarding flagged objects
        index = (data['FLUX_AUTO']>0)
        fwhm = data['FWHM_IMAGE'][index]
        flux_auto = data['FLUX_AUTO'][index]
        mag_auto = -2.5*np.log10(flux_auto)

        plt.plot(fwhm, mag_auto, 'bo', markersize=1)
        x1,x2,y1,y2 = plt.axis()
        plt.plot(fwhm_select, mag_auto_select, 'go', markersize=1)
        plt.plot([fwhm_median, fwhm_median], [y2,y1], color='red')
        fwhm_line = fwhm_median-fwhm_std
        plt.plot([fwhm_line, fwhm_line], [y2,y1], 'r--')
        fwhm_line = fwhm_median+fwhm_std
        plt.plot([fwhm_line, fwhm_line], [y2,y1], 'r--')
        plt.axis((0,min(x2,15),y2,y1))
        plt.xlabel('FWHM (pixels)')
        plt.ylabel('MAG_AUTO')
        plt.title(r'median FWHM: {:.2f} $\pm$ {:.2f} pixels'
                  .format(fwhm_median, fwhm_std))
        plt.savefig('{}_fwhm.pdf'.format(fits_cat.replace('.fits','')))
        plt.title(fits_cat)
        if get_par(set_zogy.show_plots,tel): plt.show()
        plt.close()

        if get_elong:

            elong = data['ELONGATION'][index]

            plt.plot(elong, mag_auto, 'bo', markersize=1)
            x1,x2,y1,y2 = plt.axis()
            plt.plot(elong_select, mag_auto_select, 'go', markersize=1)
            plt.plot([elong_median, elong_median], [y2,y1], color='red')
            elong_line = elong_median-elong_std
            plt.plot([elong_line, elong_line], [y2,y1], 'r--')
            elong_line = elong_median+elong_std
            plt.plot([elong_line, elong_line], [y2,y1], 'r--')
            plt.axis((0,min(x2,5),y2,y1))
            plt.xlabel('ELONGATION (A/B)')
            plt.ylabel('MAG_AUTO')
            plt.title(r'median ELONGATION: {:.2f} $\pm$ {:.2f}'
                      .format(elong_median, elong_std))
            plt.savefig('{}_elongation.pdf'.format(fits_cat.replace('.fits','')))
            if get_par(set_zogy.show_plots,tel): plt.show()
            plt.close()


    # show catalog entries with very low FWHM
    if get_par(set_zogy.make_plots,tel):
        mask_lowfwhm = (data['FWHM_IMAGE'] < fwhm_median-3*fwhm_std)
        result = prep_ds9regions('{}_lowfwhm_ds9regions.txt'
                                 .format(fits_cat.replace('.fits','')),
                                 data['XWIN_IMAGE'][mask_lowfwhm],
                                 data['YWIN_IMAGE'][mask_lowfwhm],
                                 radius=5., width=2, color='purple',
                                 value=data['FWHM_IMAGE'][mask_lowfwhm])


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_fwhm')


    if get_elong:
        return fwhm_median, fwhm_std, elong_median, elong_std
    else:
        return fwhm_median, fwhm_std


################################################################################

def run_sextractor (image, cat_out, file_config, file_params, pixscale,
                    header, fit_psf=False, return_fwhm_elong=False, fraction=1.0,
                    fwhm=5.0, psfex_mode=False, imtype=None, fits_mask=None,
                    npasses=2, tel=None, set_zogy=None, nthreads=0,
                    Scorr_mode=None, image_analysis=None, std_Scorr=1):


    """Function that runs SExtractor on [image], and saves the output
       catalog in [cat_out], using the configuration file
       [file_config] and the parameters defining the output recorded
       in the catalogue [file_params]. If [fit_psf] is True,
       SExtractor will perform PSF fitting photometry using the PSF
       built by PSFex. If [return_fwhm_elong] is True, an estimate of
       the image median FWHM and ELONGATION and their standard
       deviations are returned using SExtractor's seeing estimate of
       the detected sources; if False, it will return zeros. If
       [fraction] is less than the default 1.0, SExtractor will be run
       on a fraction [fraction] of the area of the full
       image. Sextractor will use the input value [fwhm], which is
       important for the star-galaxy classification. If [save-bkg] is
       True, the background image, its standard deviation and the
       -OBJECTS image (background-subtracted image with all objects
       masked with zero values), all produced by SExtractor, will be
       saved. If [set_zogy.bkg_method] is not set to 1 (use
       SExtractor's background), then improve the estimates of the
       background and its standard deviation.

    """

    if get_par(set_zogy.timing,tel): t = time.time()
    log.info('executing run_sextractor ...')

    base = image.replace('.fits','')

    # make copy of input image, as input image will get background
    # subtracted
    if get_par(set_zogy.keep_tmp,tel):
        image_orig = '{}_orig.fits'.format(base)
        if not isfile(image_orig):
            shutil.copy2 (image, image_orig)


    # if fraction less than one, run SExtractor on specified fraction of
    # the image
    if fraction < 1:

        # read input image and header
        data, header = read_hdulist (image, get_header=True)
        # get input image size from header
        xsize, ysize = header['NAXIS1'], header['NAXIS2']

        log.info ('xsize: {}, ysize: {}'.format(xsize, ysize))

        # determine cutout from [fraction]
        center_x = int(xsize/2+0.5)
        center_y = int(ysize/2+0.5)
        halfsize_x = int((xsize * np.sqrt(fraction))/2.+0.5)
        halfsize_y = int((ysize * np.sqrt(fraction))/2.+0.5)
        data_fraction = data[center_y-halfsize_y:center_y+halfsize_y,
                             center_x-halfsize_x:center_x+halfsize_x]

        # write small image to fits
        image_fraction = '{}_fraction.fits'.format(base)
        header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(image_fraction, data_fraction.astype('float32'), header,
                     overwrite=True)

        # make image point to image_fraction
        image = image_fraction
        cat_out = '{}_cat_fraction.fits'.format(base)


    # the input fwhm determines the SEEING_FWHM (important for
    # star/galaxy separation) and the radii of the apertures used for
    # aperture photometry. If fwhm is not provided as input, it will
    # assume fwhm=5.0 pixels.
    fwhm = float('{:.2f}'.format(fwhm))
    # determine seeing
    seeing = fwhm * pixscale
    # prepare aperture diameter string to provide to SExtractor
    apphot_diams = np.array(get_par(set_zogy.apphot_radii,tel)) * 2 * fwhm
    apphot_diams_str = ','.join(apphot_diams.astype(str))

    if psfex_mode or fits_mask is not None:

        # create updated parameters file starting from original file
        file_params_edited = '{}_params.txt'.format(base)
        shutil.copy2(file_params, file_params_edited)

        # update size of VIGNET
        if psfex_mode:
            size_vignet = get_par(set_zogy.size_vignet,tel)
            # write vignet_size to header
            header['S-VIGNET'] = (size_vignet, '[pix] size square VIGNET used '
                                  'in SExtractor')
            # append the VIGNET size to the temporary SExtractor
            # parameter file created above
            size_vignet_str = str((size_vignet, size_vignet))
            with open(file_params_edited, 'a') as myfile:
                myfile.write('VIGNET{}\n'.format(size_vignet_str))

        if fits_mask is not None:
            # and add line in parameter file to include IMAFLAG_ISO
            with open(file_params_edited, 'a') as myfile:
                myfile.write('IMAFLAGS_ISO\n')

        # use edited version for parameter file
        file_params = file_params_edited


    # check if background was already subtracted from image
    if 'BKG-SUB' in header:
        bkg_sub = header['BKG-SUB']
    else:
        header['BKG-SUB'] = (False, 'sky background was subtracted?')
        bkg_sub = False


    # determine whether interpolation is allowed across different
    # channels in [mini2back] using function get_Xchan_bool
    chancorr = get_par(set_zogy.MLBG_chancorr,tel)
    interp_Xchan = get_Xchan_bool (tel, chancorr, imtype)
    interp_Xchan_std = get_Xchan_bool (tel, chancorr, imtype, std=True)


    # get gain from header
    gain = read_header(header, ['gain'])


    log.info('background already subtracted?: {}'.format(bkg_sub))

    # do not apply weighting
    apply_weight = False

    # initialize cmd_dict dictionary that is filled in below for the
    # different options, which will be converted to a cmd list to be
    # executed from the unix command line; make sure it is an
    # insertion-ordered dictionary (default for dictionaries starting
    # from python3.6)
    cmd_dict = collections.OrderedDict()
    cmd_dict['source-extractor'] = image
    if image_analysis is not None:
        cmd_dict['source-extractor'] = '{},{}'.format(image, image_analysis)
    cmd_dict['-c'] = file_config
    cmd_dict['-BACK_TYPE'] = 'AUTO'
    cmd_dict['-BACK_VALUE'] = '0.0'
    cmd_dict['-BACK_SIZE'] = str(get_par(set_zogy.bkg_boxsize,tel))
    cmd_dict['-BACK_FILTERSIZE'] = str(get_par(set_zogy.bkg_filtersize,tel))
    cmd_dict['-BACKPHOTO_TYPE'] = get_par(set_zogy.bkg_phototype,tel).upper()
    cmd_dict['-BACKPHOTO_THICK'] = str(get_par(set_zogy.bkg_photothick,tel))
    cmd_dict['-VERBOSE_TYPE'] = 'QUIET'
    cmd_dict['-CATALOG_NAME'] = cat_out
    cmd_dict['-PARAMETERS_NAME'] = file_params
    cmd_dict['-PIXEL_SCALE'] = str(pixscale)
    cmd_dict['-SEEING_FWHM'] = str(seeing)
    cmd_dict['-PHOT_APERTURES'] = apphot_diams_str
    cmd_dict['-NTHREADS'] = str(nthreads)
    cmd_dict['-FILTER_NAME'] = get_par(set_zogy.sex_det_filt,tel)
    starnnw_name = '{}default.nnw'.format(get_par(set_zogy.cfg_dir,tel))
    cmd_dict['-STARNNW_NAME'] = starnnw_name
    cmd_dict['-CATALOG_TYPE'] = 'FITS_1.0'


    for npass in range(npasses):

        if npass==0:

            if apply_weight:
                cmd_dict['-WEIGHT_TYPE'] = 'BACKGROUND'

        else:

            # for FWHM estimate, or if background method is set to 1
            # (SExtractor method) or background had already been
            # subtracted, or Scorr_mode is not None: no need to do
            # multiple passes
            if (return_fwhm_elong or
                get_par(set_zogy.bkg_method,tel)==1 or
                bkg_sub or
                Scorr_mode is not None or
                psfex_mode):
                break

            log.info ('running 2nd pass of SExtractor')

            # save catalog of 1st/initial run to different filename
            if get_par(set_zogy.keep_tmp,tel):
                os.rename (cat_out, cat_out.replace('.fits', '_init.fits'))


            if apply_weight:
                cmd_dict['-WEIGHT_TYPE'] = 'MAP_RMS'
                cmd_dict['-WEIGHT_IMAGE'] = fits_bkg_std


        # add commands to produce BACKGROUND, BACKGROUND_RMS and
        # background-subtracted image with all pixels where objects
        # were detected set to zero (-OBJECTS). These are used to
        # build an improved background map.
        if (not return_fwhm_elong and
            npass==0 and
            not bkg_sub and
            Scorr_mode is None and
            not psfex_mode):

            fits_bkg = '{}_bkg.fits'.format(base)
            fits_bkg_std = '{}_bkg_std.fits'.format(base)
            fits_objmask = '{}_objmask1.fits'.format(base)
            fits_objects = '{}_objects1.fits'.format(base)
            fits_segm = '{}_segm1.fits'.format(base)
            cmd_dict['-CHECKIMAGE_TYPE'] = ('BACKGROUND,BACKGROUND_RMS,OBJECTS,'
                                            '-OBJECTS,SEGMENTATION')
            image_names = '{},{},{},{},{}'.format(fits_bkg, fits_bkg_std,
                                                  fits_objects, fits_objmask,
                                                  fits_segm)
            cmd_dict['-CHECKIMAGE_NAME'] = image_names

        elif (bkg_sub or npass>0) and not return_fwhm_elong and not psfex_mode:

            # if background was already subtracted or this is the 2nd
            # pass, still let SExtractor produce the object mask image
            fits_objects = '{}_objects.fits'.format(base)
            fits_objmask = '{}_objmask.fits'.format(base)
            fits_segm = '{}_segm.fits'.format(base)
            cmd_dict['-CHECKIMAGE_TYPE'] = 'OBJECTS,-OBJECTS,SEGMENTATION'
            image_names = '{},{},{}'.format(fits_objects, fits_objmask,
                                            fits_segm)
            cmd_dict['-CHECKIMAGE_NAME'] = image_names


        # in case of fwhm/elongation estimate: only consider higher
        # S/N detections
        if return_fwhm_elong:
            cmd_dict['-DETECT_THRESH'] = str(get_par(
                set_zogy.fwhm_detect_thresh,tel))


        elif Scorr_mode is not None:
            # detection threshold for transient detection of at least
            # 3 pixels above 2/3 * set_zogy.transient_nsigma; scale
            # with std_Scorr as source extractor will use actual Scorr
            # STD
            det_th = 0.66 * get_par(set_zogy.transient_nsigma,tel) / std_Scorr
            ana_th = det_th
            # increase in 'init' mode - exact detection threshold is
            # only important for background-subtracted image
            if Scorr_mode=='init':
                det_th *= 5

            cmd_dict['-DETECT_THRESH'] = str(det_th)
            cmd_dict['-DETECT_MINAREA'] = '3'
            cmd_dict['-DETECT_MAXAREA'] = '2000'
            cmd_dict['-ANALYSIS_THRESH'] = str(ana_th)
            cmd_dict['-CATALOG_TYPE'] = 'FITS_1.0'
            cmd_dict['-FILTER'] = 'N'
            #cmd_dict['-FILTER_THRESH'] = '2.0'
            #cmd_dict['-FILTER_NAME'] = '/Users/pmv/ZOGY/Config/gauss_3.0_5x5.conv'
            cmd_dict['-DEBLEND_NTHRESH'] = '8'
            cmd_dict['-DEBLEND_MINCONT'] = '0.1'
            # save background-subtracted Scorr image only in 'init' mode
            if Scorr_mode=='init':
                fits_bkgsub = '{}_bkgsub.fits'.format(base)
                cmd_dict['-CHECKIMAGE_TYPE'] = '-BACKGROUND'
                cmd_dict['-CHECKIMAGE_NAME'] = fits_bkgsub


        elif psfex_mode:
            # in psfex mode need to record in FITS_LDAC format and
            # adjust the detection thresholds to total S/N >~ 20
            cmd_dict['-CATALOG_TYPE'] = 'FITS_LDAC'
            det_th = max(0.2 * get_par(set_zogy.psf_stars_s2n_min,tel), 1.5)
            cmd_dict['-DETECT_THRESH'] = str(det_th)
            cmd_dict['-DETECT_MINAREA'] = '3'
            cmd_dict['-ANALYSIS_THRESH'] = str(det_th)



        # provide PSF file from PSFex
        if fit_psf:
            cmd_dict['-PSF_NAME'] = '{}_psf.fits'.format(base)


        # provide mask image and type if not None
        if fits_mask is not None:
            log.info('mask: {}'.format(fits_mask))
            cmd_dict['-FLAG_IMAGE'] = fits_mask
            cmd_dict['-FLAG_TYPE'] = 'OR'


        # convert cmd_dict to list
        cmd_list = list(itertools.chain.from_iterable(list(cmd_dict.items())))

        # log cmd executed
        cmd_str = ' '.join(cmd_list)
        log.info('SExtractor command executed:\n{}'.format(cmd_str))

        # run command
        result = subprocess.run(cmd_list, capture_output=True)
        status = result.returncode
        log.info('stdout: {}'.format(result.stdout.decode('UTF-8')))
        log.info('stderr: {}'.format(result.stderr.decode('UTF-8')))
        log.info('status: {}'.format(status))

        if status != 0:
            msg = 'SExtractor failed with exit code {}'.format(status)
            log.exception(msg)
            raise Exception(msg)

        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t, label='in run_sextractor before get_back')


        # improve background and its standard deviation estimate if
        # [set_zogy.bkg_method] not set to 1 (= use background
        # determined by SExtractor) and background had not already
        # been subtracted
        if (not return_fwhm_elong and
            npass==0 and
            get_par(set_zogy.bkg_method,tel)==2 and
            not bkg_sub and
            Scorr_mode is None and
            not psfex_mode):

            # read in input image; do not read header as that would
            # overwrite the header updates already done in this
            # function
            data = read_hdulist (image, dtype='float32')

            if fits_mask is not None:
                data_mask = read_hdulist (fits_mask, dtype='uint8')

            # construct background image using [get_back]; for ML/BG:
            # if the global background was already subtracted,
            # [data_bkg_mini] will be zero
            data_bkg_mini, data_bkg_std_mini = get_back (
                data, header, fits_objmask, fits_mask=fits_mask,
                tel=tel, set_zogy=set_zogy, imtype=imtype)


            # write these filtered meshes to fits and update their
            # headers with [set_zogy.bkg_boxsize]
            fits_tmp = '{}_bkg_mini.fits'.format(base)
            fits.writeto(fits_tmp, data_bkg_mini, overwrite=True)
            bkg_size = get_par(set_zogy.bkg_boxsize,tel)
            txt_tmp = '[pix] background boxsize used to create this image'
            fits.setval(fits_tmp, 'BKG-SIZE', value=bkg_size, comment=txt_tmp)


            # bkg STD
            fits_tmp = '{}_bkg_std_mini.fits'.format(base)
            fits.writeto(fits_tmp, data_bkg_std_mini, overwrite=True)
            fits.setval(fits_tmp, 'BKG-SIZE', value=bkg_size, comment=txt_tmp)


            # now use function [mini2back] to turn filtered mesh of median
            # and std of background regions into full background image and
            # its standard deviation
            bkg_size = get_par(set_zogy.bkg_boxsize,tel)
            data_bkg = mini2back (data_bkg_mini, data.shape, order_interp=2,
                                  bkg_boxsize=bkg_size,
                                  interp_Xchan=interp_Xchan,
                                  timing=get_par(set_zogy.timing,tel))
            data_bkg_std = mini2back (data_bkg_std_mini, data.shape,
                                      order_interp=1, bkg_boxsize=bkg_size,
                                      interp_Xchan=interp_Xchan_std,
                                      timing=get_par(set_zogy.timing,tel))


            # subtract the global background
            data -= data_bkg
            header['BKG-SUB'] = (True, 'sky background was subtracted?')

            header['S-BKG'] = (gain * np.median(data_bkg_mini), '[e-] median '
                               'background full image')
            header['S-BKGSTD'] = (gain * np.median(data_bkg_std_mini), '[e-] '
                                  'sigma (STD) background full image')


            # best to ensure that edge pixels are set to zero
            if fits_mask is not None:
                value_edge = get_par(set_zogy.mask_value['edge'],tel)
                mask_edge = (data_mask & value_edge == value_edge)
                data[mask_edge] = 0

            # save to fits; this image will be used to feed to
            # SExtractor in the next pass
            fits.writeto(image, data, header, overwrite=True)

            # write the improved background and standard deviation to fits
            # overwriting the fits images produced by SExtractor
            fits.writeto(fits_bkg, data_bkg, header, overwrite=True)
            fits.writeto(fits_bkg_std, data_bkg_std, header, overwrite=True)



    if return_fwhm_elong:
        # get estimate of seeing and elongation from output catalog
        fwhm, fwhm_std, elong, elong_std = get_fwhm(
            cat_out, get_par(set_zogy.fwhm_frac,tel),
            class_sort=get_par(set_zogy.fwhm_class_sort,tel), get_elong=True)

    else:
        fwhm = 0
        fwhm_std = 0
        elong = 0
        elong_std = 0



    # CHECK!!! - keep tmp copy
    #if get_par(set_zogy.keep_tmp,tel):
    #    shutil.copy2 (cat_out, cat_out.replace('.fits', '_init2.fits'))


    # now that catalog has been created, rename some of the columns
    # using the function rename_catcols, unless this involves the LDAC
    # catalog needed for PSFEx
    if not psfex_mode:
        rename_catcols(cat_out)


    # add number of objects detected (=number of catalog rows) to header
    header_catout = read_hdulist (cat_out, get_data=False, get_header=True)
    nobjects = header_catout['NAXIS2']
    log.info('number of objects detected by SExtractor: {}'.format(nobjects))

    if not return_fwhm_elong and not psfex_mode:
        header['S-NOBJ'] = (nobjects, 'number of objects detected by SExtractor')


    # also add header keyword(s) regarding background in case
    # background was determined by source-extractor
    if (get_par(set_zogy.bkg_method,tel)==1 and not return_fwhm_elong and
        not psfex_mode):
        data_bkg = read_hdulist (fits_bkg)
        header['S-BKG'] = (gain * np.median(data_bkg), '[e-] median background '
                           'full image')
        data_bkg_std = read_hdulist (fits_bkg_std)
        header['S-BKGSTD'] = (gain * np.median(data_bkg_std), '[e-] sigma (STD) '
                              'background full image')


    # save object mask created by source extractor, which is a float32
    # image with zeros indicating the objects, as an uint8 image with
    # ones indicating the objects
    fits_objmask = '{}_objmask.fits'.format(base)
    if isfile(fits_objmask) and not return_fwhm_elong and not psfex_mode:
        data_objmask = read_hdulist(fits_objmask, dtype='float32')
        objmask = (np.abs(data_objmask)==0)
        fits.writeto(fits_objmask, objmask.astype('uint8'), overwrite=True)


    # remove file(s) if not keeping intermediate/temporary files
    if not get_par(set_zogy.keep_tmp,tel):

        if 'image_fraction' in locals():
            remove_files ([image_fraction], verbose=True)

        if fraction < 1:
            remove_files ([cat_out], verbose=True)



    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in run_sextractor')

    return fwhm, fwhm_std, elong, elong_std


################################################################################

def update_bkgcol (base, header, imtype):

    """function to replace BACKGROUND column in the fits catalog with
       values from the improved background, if available; this is
       needed because the 2nd SExtractor run has been performed on a
       background-subtracted image

    """

    # fits table to update
    fits_cat = '{}_cat.fits'.format(base)

    # read [data_bkg] from full or mini fits files
    fits_bkg = '{}_bkg.fits'.format(base)
    fits_bkg_mini = '{}_bkg_mini.fits'.format(base)

    if isfile(fits_bkg):
        # read it from the full background image
        data_bkg = read_hdulist (fits_bkg, dtype='float32')
        log.info ('reading full background image')


    # if full image backgrounds not available, create it from the
    # background mesh
    if 'data_bkg' not in locals():

        # determine whether interpolation is allowed across different
        # channels in [mini2back] using function get_Xchan_bool
        chancorr = get_par(set_zogy.MLBG_chancorr,tel)
        interp_Xchan = get_Xchan_bool (tel, chancorr, imtype)

        # determine full image size from header
        data_shape = (header['NAXIS2'], header['NAXIS1'])

        if isfile(fits_bkg_mini):

            data_bkg_mini, header_mini = read_hdulist (
                fits_bkg_mini, get_header=True, dtype='float32')

            log.info ('reading background mini image')

            if 'BKG-SIZE' in header_mini:
                bkg_size = header_mini['BKG-SIZE']
            else:
                bkg_size = get_par(set_zogy.bkg_boxsize,tel)

            data_bkg = mini2back (data_bkg_mini, data_shape,
                                  order_interp=2, bkg_boxsize=bkg_size,
                                  interp_Xchan=interp_Xchan,
                                  timing=get_par(set_zogy.timing,tel))


    # replace the background column in the output catalog in case
    # 'data_bkg' exists - could be absent, e.g. for the co-added
    # reference image where the background was already subtracted from
    # the individual frames and no background image or mini image is
    # available
    if 'data_bkg' in locals():
        with fits.open(fits_cat, mode='update', memmap=True) as hdulist:
            data = hdulist[-1].data
            if 'BACKGROUND' in data.dtype.names:
                x_indices = (data['X_POS']-0.5).astype(int)
                y_indices = (data['Y_POS']-0.5).astype(int)
                #background = data_bkg[y_indices, x_indices]
                #hdulist[-1].data['BACKGROUND'] = background
                data['BACKGROUND'] = data_bkg[y_indices, x_indices]

                log.info ('updating initial BACKGROUND determination in '
                          'source-extractor catalog {} with improved values'
                          .format(fits_cat))

                if not np.any(data_bkg):
                    log.warning ('all new BACKGROUND values are zero for '
                                 'catalog {}'.format(fits_cat))


    return


################################################################################

def rename_catcols (cat_in, hdu_ext=1):

    """Function to rename particular columns in the fits table created
    by SExtractor.

    """

    # dictionary col_old2new is used to rename a number of
    # columns in the output new/ref/trans catalogs; this is
    # done here when converting the LDAC to normal fits
    # catalog, so the LDAC catalog - used by PSFEx - still
    # contains the original SExtractor column names. This way
    # the $ZOGYHOME/Config SExtractor and PSFEx configuration
    # files can also still contain the original names, but the
    # drawback is that old and new column names are mixed in
    # this zogy.py module.
    col_old2new = {'ALPHAWIN_J2000': 'RA',
                   'DELTAWIN_J2000': 'DEC',
                   'XWIN_IMAGE':     'X_POS',
                   'YWIN_IMAGE':     'Y_POS',
                   'ERRX2WIN_IMAGE': 'XVAR_POS',
                   'ERRY2WIN_IMAGE': 'YVAR_POS',
                   'ERRXYWIN_IMAGE': 'XYCOV_POS',
                   'FWHM_IMAGE':     'FWHM',
                   'X2WIN_IMAGE':    'X2AVE_POS',
                   'Y2WIN_IMAGE':    'Y2AVE_POS',
                   'XYWIN_IMAGE':    'XYAVE_POS',
                   'ISOAREA_IMAGE':  'ISOAREA',
                   'IMAFLAGS_ISO':   'FLAGS_MASK',
                   'XPEAK_IMAGE':    'X_PEAK',
                   'YPEAK_IMAGE':    'Y_PEAK',
                   'CXXWIN_IMAGE':   'CXX',
                   'CYYWIN_IMAGE':   'CYY',
                   'CXYWIN_IMAGE':   'CXY',
                   'AWIN_IMAGE':     'A',
                   'BWIN_IMAGE':     'B',
                   'THETAWIN_IMAGE': 'THETA'}


    def update_colname (table, col, col_new, cat_in):

        # if new column name does not exist yet, rename the old column
        if col_new not in table.colnames:
            table[col].name = col_new
            log.info ('renamed column {} to {}'.format(col, col_new))
        else:
            log.warning ('column {} already present in catalog {}'
                         .format(col_new, cat_in))


    # read correct extension of input catalog
    table = Table.read(cat_in, hdu=hdu_ext, memmap=True)

    # loop through above dictionary keys
    for col in col_old2new.keys():

        # check if column is present in table
        if col in table.colnames:

            # update column name
            col_new = col_old2new[col]
            update_colname (table, col, col_new, cat_in)

            # for column 'IMAFLAGS_ISO' preserve a copy of the old
            # column to possibly use in PSFEx to filter sources
            if False and col == 'IMAFLAGS_ISO':

                # create column with the old name if it does not
                # already exist
                if col not in table.colnames:
                    table[col] = table[col_new]
                    log.info ('keeping a copy of column {} with its original '
                              'name for possible use in PSFEx'.format(col))
                else:
                    log.warning ('column {} already present in catalog {}'
                                 .format(col, cat_in))


    # loop through all table columns to update names of flux columns
    for col in table.colnames:

        # prefix all flux column names with E_ to indicate unit is
        # electrons
        if ('FLUX' in col and 'E_' not in col and
            'RADIUS' not in col and 'GROWTHSTEP' not in col):

            # update column name
            col_new = 'E_{}'.format(col)
            update_colname (table, col, col_new, cat_in)


    # insert table into correct extension of input catalog; the
    # character_as_bytes determines whether to return bytes for string
    # columns when accessed from the HDU; saves some memory and is not
    # relevant as none of the [cat_in] column datatypes are strings
    with fits.open(cat_in, mode='update', memmap=True) as hdulist:
        hdulist[hdu_ext] = fits.table_to_hdu(table, character_as_bytes=True)


    return


################################################################################

def run_psfex (cat_in, file_config, cat_out, imtype, poldeg, nsnap=8,
               limit_ldac=True, nthreads=0):

    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""


    if get_par(set_zogy.timing,tel):
        t = time.time()


    if imtype=='new':
        base = base_new
    else:
        base = base_ref


    # select a subset of entries in the input ldac catalog to speed up
    # psfex; need to use fitsio to keep RAM memory under control as
    # even with memmap=True, astropy.io.fits will access entire table
    # even if selection on single column is needed
    if limit_ldac:

        mem_use ('before limiting LDAC')

        # info from astropy site:
        # For FITS binary tables, the data is stored row by row, and
        # it is possible to read only a subset of rows, but reading a
        # full column loads the whole table data into memory.
        #
        # options:
        # (1) loop table with large step and keep relevant rows;
        #     --> this will still access the entire table
        # (2) read table, delete thumbnail columns, make selection
        #     of rows in light table, read full table again and
        #     select relevant rows
        #     --> not working after testing it
        # (3) fitsio - yes!

        # read all rows but only specific columns from 2nd extension
        # of input catalog, where output data is a numpy recarray
        data_ldac = fitsio.read(cat_in, ext=2,
                                columns=['FLAGS', 'SNR_WIN', 'IMAFLAGS_ISO',
                                         'XWIN_IMAGE', 'YWIN_IMAGE'])

        # SExtractor flags 0 or 1 and S/N larger than value
        # defined in settings file; note that this is the same as
        # the default rejection mask on SExtractor FLAGS defined
        # in the psfex.config file: SAMPLE_FLAGMASK = 0x00fe
        # (=254) or '0b11111110'
        s2n = get_par(set_zogy.psf_stars_s2n_min,tel)
        index_ok = np.nonzero((data_ldac['FLAGS']<=1) &
                              (data_ldac['SNR_WIN']>=s2n) &
                              (data_ldac['IMAFLAGS_ISO']==0))[0]
        log.info ('number of PSF stars available with FLAGS<=1, SNR>{} '
                  'and IMAFLAGS_ISO==0: {}'.format(s2n, len(index_ok)))


        psf_stars_nmax = get_par(set_zogy.psf_stars_nmax,tel)
        if len(index_ok) > psf_stars_nmax:

            # shuffling the indices in place, so coordinates
            # are picked randomly irrespective of input order
            rng = np.random.default_rng()
            rng.shuffle(index_ok)

            # select [psf_stars_nmax] random set of PSF stars,
            # uniformly spread across the image
            index_uniform = uniform_subset (data_ldac['XWIN_IMAGE'][index_ok],
                                            data_ldac['YWIN_IMAGE'][index_ok],
                                            psf_stars_nmax)

            log.info ('too many PSFEx input stars ({}) available; narrowing it '
                      'down by randomly selecting {} stars uniformly spread '
                      'across the image'
                      .format(len(index_ok), len(index_uniform)))

            # update index_ok
            index_ok = index_ok[index_uniform]



        # read input catalog again, now all columns but only specific rows
        data_ldac_ok = fitsio.read(cat_in, ext=2, rows=index_ok)


        # write data_ldac_ok to 2nd extension of input table, first
        # need to delete the 2nd extension of the input table as
        # otherwise the existing rows will not be deleted but
        # overwritten
        with fits.open(cat_in, mode='update', memmap=True) as hdulist:
            del hdulist[2]

        # write data_ldac_ok; 2nd extension is newly created
        with fitsio.FITS(cat_in, 'rw') as fits_tmp:
            fits_tmp.write(data_ldac_ok, extname='LDAC_OBJECTS')


        if get_par(set_zogy.make_plots,tel):
            result = prep_ds9regions('{}_psfstars_ds9regions.txt'.format(base),
                                     data_ldac_ok['XWIN_IMAGE'],
                                     data_ldac_ok['YWIN_IMAGE'],
                                     radius=5., width=2, color='red')


        if get_par(set_zogy.timing,tel):
            log_timing_memory (
                t0=t,label='in limiting entries in LDAC input catalog for PSFEx')




    # use function [get_samp_PSF_config_size] to determine [psf_samp]
    # and [psf_size_config] required to run PSFEx
    psf_samp, psf_size_config = get_samp_PSF_config_size(imtype)
    psf_size_config_str = '{},{}'.format(psf_size_config, psf_size_config)

    if get_par(set_zogy.verbose,tel):
        log.info('psf_size_config: {} in [run_psfex] for imtype {}'
                 .format(psf_size_config, imtype))

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

    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config_str, '-PSF_SAMPLING', str(psf_samp),
           '-SAMPLE_MINSN', str(get_par(set_zogy.psf_stars_s2n_min,tel)),
           '-NTHREADS', str(nthreads), '-PSFVAR_NSNAP', str(nsnap),
           '-PSFVAR_DEGREES', str(poldeg)]
    #       '-SAMPLE_FWHMRANGE', sample_fwhmrange,
    #       '-SAMPLE_MAXELLIP', maxellip_str]

    if get_par(set_zogy.make_plots,tel):
        cmd += ['-CHECKPLOT_TYPE', 'FWHM, ELLIPTICITY, COUNTS, COUNT_FRACTION, '
                'CHI2, RESIDUALS',
                '-CHECKPLOT_DEV', 'PS',
                '-CHECKPLOT_ANTIALIAS', 'N',
                '-CHECKPLOT_NAME',
                'psfex_fwhm, psfex_ellip, psfex_counts, psfex_countfrac, '
                'psfex_chi2, psfex_resi']
        cmd += ['-CHECKIMAGE_TYPE', 'CHI,PROTOTYPES,SAMPLES,RESIDUALS,SNAPSHOTS,'
                'BASIS,MOFFAT,-MOFFAT,-SYMMETRICAL',
                '-CHECKIMAGE_NAME',
                'psfex_chi, psfex_proto, psfex_samp, psfex_resi, psfex_snap, '
                'psfex_basis, psfex_moffat, psfex_minmoffat, psfex_minsymmetrical']

    # log cmd executed
    cmd_str = ' '.join(cmd)
    log.info('PSFEx command executed:\n{}'.format(cmd_str))

    # possibility to run PSFEx multiple times with increasing
    # timeouts, as different runs on the same input catalog sometimes
    # results in a long execution time
    timeouts = [None]
    for timeout in timeouts:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except TimeoutExpired:
            # need to clean up any files that were created before
            # starting next iteration?
            pass
        else:
            # no exception occurred, leave loop
            break


    status = result.returncode
    log.info('stdout: {}'.format(result.stdout.decode('UTF-8')))
    log.info('stderr: {}'.format(result.stderr.decode('UTF-8')))
    log.info('status: {}'.format(status))

    if status != 0:
        msg = 'PSFEx failed with exit code {}'.format(status)
        log.exception(msg)
        raise Exception(msg)


    # standard output of PSFEx is .psf; change this to _psf.fits
    psf_in = cat_in.replace('.fits', '.psf')
    psf_out = '{}_psf.fits'.format(base)
    os.rename (psf_in, psf_out)

    # if zero stars were found by PSFEx, raise exception
    header_psf = read_hdulist(psf_out, get_data=False, get_header=True)
    if header_psf['ACCEPTED']==0:
        msg = ('no appropriate source found by PSFEx in [run_psfex] for catalog '
               '{}'.format(cat_in))
        log.exception (msg)
        raise Exception (msg)


    # no need to limit disk space of LDAC catalog anymore, as it is no
    # longer saved
    if False:
        # for the reference image, limit the size of the ldac catalog fits
        # file to be saved with only those catalog entries used by PSFEx
        # --> why not for new image as well?
        if imtype=='ref' or imtype=='new':
            psfexcat = '{}_psfex.cat'.format(base)
            table = ascii.read(psfexcat, format='sextractor')
            # In PSFEx version 3.18.2 all objects from the input
            # SExtractor catalog are recorded, and in that case the
            # entries with FLAGS_PSF=0 need to be selected.
            if 'FLAGS_PSF' in table.colnames:
                mask_psfstars = (table['FLAGS_PSF']==0)
            # In PSFEx version 3.17.1 (last stable version), only stars
            # with zero flags are recorded in the output catalog, so need
            # to pick the source number of the input catalog minus 1.
            else:
                mask_psfstars = table['SOURCE_NUMBER']-1

            # overwrite the input catalog with these selected stars
            with fits.open(cat_in, mode='update') as hdulist:
                hdulist[2].data = hdulist[2].data[mask_psfstars]


    # record the PSFEx output check images defined above, into
    # extensions of a single fits file
    if get_par(set_zogy.make_plots,tel):

        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU())

        cwd = os.getcwd()
        file_list = glob.glob('{}/psfex*'.format(cwd))
        for name in file_list:

            short = name.split('/')[-1]
            prefix = '_'.join(short.split('_')[0:2])

            if 'fits' in name:
                # read image data and header
                data, header = read_hdulist (name, get_header=True)
                # add extension to hdulist with extname=prefix
                hdulist.append(fits.ImageHDU(data=data, header=header,
                                             name=prefix))
                # remove individual image
                os.remove(name)
                log.info ('added fits image {} to multi-extension fits file'
                          .format(name))

            else:
                # if not a fits file, rename it
                name_new = short.replace('ldac', prefix)
                name_new = name_new.replace('{}_'.format(prefix),'')
                name_new = '{}/{}'.format(cwd, name_new)
                log.info ('name: {}, name_new: {}'.format(name, name_new))
                os.rename (name, name_new)

        # write image
        fits_output = '{}_psf_checkimages.fits'.format(base)
        hdulist.writeto(fits_output, overwrite=True)


    # remove file(s) if not keeping intermediate/temporary files
    if not get_par(set_zogy.keep_tmp,tel):
        remove_files ([cat_in], verbose=True)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in run_psfex')

    return


################################################################################

def get_samp_PSF_config_size (imtype):

    """function to determine [psf_samp] and [psf_config_size] to be used
       in [get_psf] and [run_psfex]

    """

    if imtype=='new':
        fwhm = fwhm_new
    elif imtype=='ref':
        fwhm = fwhm_ref


    # If [set_zogy.psf_sampling] is set to nonzero, then:
    #   [psf_samp] = [psf_samling]
    # where [psf_samp(ling)] is the PSF sampling step in image
    # pixels. If [set_zogy.psf_sampling] is set to zero, then:
    #   [psf_samp] = [set_zogy.psf_samp_fwhmfrac] * FWHM in pixels
    # where [set_zogy.psf_samp_fwhmfrac] is a global parameter which
    # should be set to about 0.25, so for an oversampled image with
    # FWHM~8: [psf_samp]~2, while for an undersampled image with
    # FWHM~2: [psf_samp]~1/4
    if get_par(set_zogy.psf_sampling,tel) == 0:
        psf_samp = get_par(set_zogy.psf_samp_fwhmfrac,tel) * fwhm
    else:
        psf_samp = get_par(set_zogy.psf_sampling,tel)

    # [psf_size_config] is the size of the square image on which PSFEx
    # constructs the PSF. That image is resized to the original pixel
    # scale in function [get_psf_ima] and cut to the [psf_size] needed
    # for the optimal photometry measurements or the PSFs needed for
    # the optimal subtraction. Make [psf_size_config] as large as
    # allowed by [set_zogy.size_vignet_ref] and [psf_samp] determined
    # above
    psf_size_config = get_par(set_zogy.size_vignet,tel) / psf_samp
    # convert to integer
    psf_size_config = int(np.ceil(psf_size_config))
    # make sure it's odd
    if psf_size_config % 2 == 0:
        psf_size_config += 1


    # provide warning if actually required psf_size_config is larger
    # than allowed by size_vignet
    psf_radius = max(get_par(set_zogy.psf_rad_phot,tel),
                     get_par(set_zogy.psf_rad_zogy,tel))

    if (2 * fwhm * psf_radius) > (psf_samp * psf_size_config):
        log.warning ('psf_size_config of {} is limited by VIGNET size of {}: '
                     '2 * FWHM_{} * max(psf_rad_phot, psf_rad_zogy) / '
                     'psf_samp = 2 * {:.2f} * {} / {:.2f} = {:.2f}'
                     .format(psf_size_config,
                             get_par(set_zogy.size_vignet,tel),
                             imtype, fwhm, psf_radius, psf_samp,
                             (2 * fwhm * psf_radius)/psf_samp))


    # now slightly change [psf_samp] such that [psf_samp] *
    # [psf_size_config] (the size in original image pixels) is exactly
    # an odd number
    val_tmp = int(psf_samp * psf_size_config)
    if val_tmp % 2 == 0:
        val_tmp += 1
    psf_samp = float(val_tmp) / float(psf_size_config)


    return psf_samp, psf_size_config


################################################################################

def clean_cut_norm_psf (psf_array, clean_factor, cut=True, cut_size=None):

    # psf_array is assumed to be square
    ysize, xsize = psf_array.shape
    assert xsize == ysize


    # if clean_factor is nonzero, clean array from low values in
    # outskirts
    if clean_factor != 0:
        mask_clean = (psf_array < (np.amax(psf_array) * clean_factor))
        psf_array[mask_clean] = 0


    # decrease size of input array to [cut_size] if that is smaller
    # than the input size
    if cut and cut_size is not None and cut_size < ysize:

        # these are the x,y indices of the central pixel in
        # [psf_array]; this is independent of size being odd/even
        xpos_center = int(xsize/2)
        ypos_center = int(ysize/2)

        # index does depend on output cut_size being odd/even
        if cut_size % 2 == 0:
            odd = 0
        else:
            odd = 1

        cut_hsize = int(cut_size/2.)
        index = (slice(ypos_center-cut_hsize, ypos_center+cut_hsize+odd),
                 slice(xpos_center-cut_hsize, xpos_center+cut_hsize+odd))

        # extract relevant subsection
        psf_array = psf_array[index]

        # update size of psf_array
        ysize, xsize = psf_array.shape


    # set values in the corners of the remaining PSF image to zero;
    # the exact half of the array is used to compare to the distance
    # grid
    dist, __, __ = dist_from_peak (psf_array)
    psf_array[dist>(ysize/2.)] = 0


    # also set any negative value to zero; switch this off because
    # this appears to bias the optimal magnitudes and also the Scorr
    # image is worse
    #psf_array[psf_array < 0] = 0

    # normalize
    psf_array /= np.sum(psf_array)


    # return array
    return psf_array


################################################################################

def dist_from_center (data):

    """function to return a grid with the same shape as [data] indicating
       the distance in pixels from the exact center of the square 2D
       input array to the center of the individual pixels

    """

    ysize, xsize = data.shape


    def get_range (size):
        hsize = int(size/2)
        if size % 2 == 0:
            # even case
            xy = np.arange(-hsize+0.5, hsize+0.5)
        else:
            # odd case
            xy = np.arange(-hsize, hsize+1)

        return xy


    x = get_range(xsize)
    y = get_range(ysize)

    # define the grid
    xx, yy = np.meshgrid(x, y, sparse=False)

    # return the distance
    return np.sqrt(xx**2+yy**2), xx, yy


################################################################################

def dist_from_peak (data):

    """function to return a grid with the same shape as [data] indicating
       the distance in pixels from the peak pixel in data

    """

    ysize, xsize = data.shape
    ypeak, xpeak = np.unravel_index(np.argmax(data), data.shape)

    x = np.arange(xsize) - xpeak
    y = np.arange(ysize) - ypeak

    # define the grid
    xx, yy = np.meshgrid(x, y, sparse=False)

    # return the distance
    return np.sqrt(xx**2+yy**2), xx, yy


################################################################################

def run_ZOGY (nsub, data_ref, data_new, psf_ref, psf_new, data_ref_bkg_std,
              data_new_bkg_std, fratio, fratio_err, dx, dy, use_FFTW=True,
              nthreads=1):

    """function to run ZOGY on a subimage"""

    if get_par(set_zogy.timing,tel):
        t = time.time()

    # option 1: set f_ref to unity
    #f_ref = 1.
    #f_new = f_ref * np.mean(fratio_sub)
    # option 2: set f_new to unity
    fn = 1.
    fr = fn / fratio
    #dx = dx
    #dy = dy

    N = np.copy(data_new)
    R = np.copy(data_ref)
    Pn = psf_new
    Pr = psf_ref

    # before running zogy, pixels with zero values in ref need to
    # be set to zero in new as well, and vice versa, to avoid
    # subtracting non-overlapping image part
    mask_zero = ((R==0) | (N==0))
    N[mask_zero] = 0
    R[mask_zero] = 0

    # determine subimage s_new and s_ref from background RMS
    # images
    if np.sum(~mask_zero) != 0:

        sn = np.median(data_new_bkg_std[~mask_zero])
        sr = np.median(data_ref_bkg_std[~mask_zero])
        # try providing full subimages for sn and sr
        #sn = data_new_bkg_std
        #sr = data_ref_bkg_std

    else:

        log.warning('empty subimage; large shift between new and ref image?')
        sn = 1
        sr = 1


    # variance estimate: background-subtracted image +
    # measured background variance
    Vn = N + data_new_bkg_std**2
    Vr = R + data_ref_bkg_std**2

    if get_par(set_zogy.verbose,tel):
        log.info('--------------------')
        log.info('nsub: {}'.format(nsub+1))
        log.info('--------------------')
        log.info('fn: {:.3f}, fr: {:.3f}, fratio: {:.3f}'.format(fn, fr, fratio))
        log.info('dx: {:.3f}, dy: {:.3f}'.format(dx, dy))
        log.info('sn: {:.3f}, sr: {:.3f}'.format(sn, sr))


    # boolean [use_FFTW] determines if initial forward fft2 is
    # initialized using pyfftw.FFTW or not; due to planning involved
    # this speeds up all subsequent calls to convenience function
    # [fft.fft2] significantly, with a loop time of 0.2s instead of
    # 0.3s.  If nthreads>1 then this speed-up becomes less dramatic,
    # e.g. with 4 threads, the loop time is 0.2s without [use_FFTW]
    # and 0.17s with [use_FFTW]. Sometimes, this seems to change the
    # results for the 1st subimage, in the same way for planner flags
    # FFTW_ESTIMATE (timing: ~97s on macbook) and FFTW_ESTIMATE
    # (~90s), but much worse for FFTW_PATIENT (~360s). So if this is
    # switched on, need to do a pre-processing of the 1st subimage.
    if use_FFTW:
        Rc = R.astype('complex64')
        R_hat = np.zeros_like(Rc)
        fft_forward = pyfftw.FFTW(Rc, R_hat, axes=(0,1), direction='FFTW_FORWARD',
                                  flags=('FFTW_ESTIMATE', ),
                                  threads=nthreads, planning_timelimit=None)
        fft_forward()
    else:
        R_hat = fft.fft2(R, threads=nthreads)

    N_hat = fft.fft2(N, threads=nthreads)

    Pn_hat = fft.fft2(Pn, threads=nthreads)
    #if get_par(set_zogy.psf_clean_factor,tel)!=0:
    #Pn_hat[Pn_hat<0] = 1e-6
    Pn_hat2_abs = np.abs(Pn_hat**2)

    Pr_hat = fft.fft2(Pr, threads=nthreads)
    #if get_par(set_zogy.psf_clean_factor,tel)!=0:
    #Pr_hat[Pr_hat<0] = 1e-6
    Pr_hat2_abs = np.abs(Pr_hat**2)

    sn2 = sn**2
    sr2 = sr**2
    fn2 = fn**2
    fr2 = fr**2
    fD = (fr*fn) / np.sqrt(sn2*fr2+sr2*fn2)

    denominator = (sn2*fr2)*Pr_hat2_abs + (sr2*fn2)*Pn_hat2_abs

    D_hat = (fr*(Pr_hat*N_hat) - fn*(Pn_hat*R_hat)) / np.sqrt(denominator)

    if use_FFTW:
        D = np.zeros_like(D_hat)
        fft_backward = pyfftw.FFTW(D_hat, D, axes=(0,1), direction='FFTW_BACKWARD',
                                   flags=('FFTW_MEASURE', ),
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


    if get_par(set_zogy.display,tel) and show_sub(nsub):
        base = base_newref
        subend = 'sub{}.fits'.format(nsub)
        fits.writeto('{}_Pn_hat_{}'.format(base, subend),
                     np.real(Pn_hat).astype('float32'), overwrite=True)
        fits.writeto('{}_Pr_hat_{}'.format(base, subend),
                     np.real(Pr_hat).astype('float32'), overwrite=True)
        fits.writeto('{}_kr_{}'.format(base, subend),
                     np.real(kr).astype('float32'), overwrite=True)
        fits.writeto('{}_kn_{}'.format(base, subend),
                     np.real(kn).astype('float32'), overwrite=True)
        fits.writeto('{}_Sr_{}'.format(base, subend),
                     Sr.astype('float32'), overwrite=True)
        fits.writeto('{}_Sn_{}'.format(base, subend),
                     Sn.astype('float32'), overwrite=True)
        fits.writeto('{}_VSr_{}'.format(base, subend),
                     VSr.astype('float32'), overwrite=True)
        fits.writeto('{}_VSn_{}'.format(base, subend),
                     VSn.astype('float32'), overwrite=True)
        fits.writeto('{}_VSr_ast_{}'.format(base, subend),
                     VSr_ast.astype('float32'), overwrite=True)
        fits.writeto('{}_VSn_ast_{}'.format(base, subend),
                     VSn_ast.astype('float32'), overwrite=True)

        P_D = fft.ifftshift(np.real(fft.ifft2(P_D_hat, threads=nthreads)))
        fits.writeto('{}_PD_{}'.format(base, subend),
                     P_D.astype('float32'), overwrite=True)



    # !!!CHECK!!! - need to add Vfratio = (R * fratio _err)**2 ??
    # Vfratio is added below to V_S, so that it influences both Scorr
    # and alpha_std (= PSF flux error)
    V_fratio = (R * fratio_err)**2
    #V_fratio = 0

    # and finally Scorr
    V_S = VSr + VSn + V_fratio
    V_ast = VSr_ast + VSn_ast
    V = V_S + V_ast
    #Scorr = S / np.sqrt(V)
    # make sure there's no division by zero
    Scorr = np.copy(S)
    #Scorr[V>0] /= np.sqrt(V[V>0])
    mask = (V>0)
    Scorr[mask] /= np.sqrt(V[mask])

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


    # save numpy arrays to file; the array variables will point to the
    # filenames instead
    base = base_newref
    if get_par(set_zogy.display,tel) and show_sub(nsub):
        # save as fits so it can be used to display
        subend = 'sub{}.fits'.format(nsub)
    else:
        # save as numpy file
        subend = 'sub{}.npy'.format(nsub)

    D = save_npy_fits (D, '{}_D_{}'.format(base, subend))
    #S = save_npy_fits (S, '{}_S_{}'.format(base, subend))
    Scorr = save_npy_fits (Scorr, '{}_Scorr_{}'.format(base, subend))
    alpha = save_npy_fits (alpha, '{}_Fpsf_{}'.format(base, subend))
    alpha_std = save_npy_fits (alpha_std, '{}_Fpsferr_{}'
                               .format(base, subend))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in run_ZOGY')

    #return D, S, Scorr, alpha, alpha_std
    return D, Scorr, alpha, alpha_std


################################################################################

def log_timing_memory(t0, label=''):

    log.info ('wall-time spent {}: {:.3f} s'.format(label, time.time()-t0))
    mem_use (label=label)

    return


################################################################################

def disk_use (path=None, label=''):


    if path is None:
        # it is is defined, use global parameter base_new, which
        # refers to base_ref if only ref image is provided
        if 'base_new' in globals():
            path = os.path.dirname(base_new)
        else:
            # otherwise use current directory
            path = '.'


    # if path does not exist, or is a google cloud bucket, shutil
    # command will fail
    if path[0:5] != 'gs://' and os.path.exists(path):

        total, used, free = shutil.disk_usage(path)
        norm = 1024**3
        log.info ('disk use [GiB]: total={:.3f}, used={:.3f}, free={:.3f} (path: '
                  '{}) {}'.format(total/norm, used/norm, free/norm, path, label))

    else:

        log.warning ('invalid path {} in zogy.disk_use')


    return


################################################################################

def mem_use (label=''):

    # see https://gmpy.dev/blog/2016/real-process-memory-and-environ-in-python

    # ru_maxrss is in units of kilobytes on Linux; however, this seems
    # to be OS dependent as on mac os it is in units of bytes; see
    # manpages of "getrusage"
    if sys.platform=='darwin':
        norm = 1024**3
    else:
        norm = 1024**2

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/norm

    full_info = psutil.Process().memory_full_info()
    uss = full_info.uss / 1024**3
    rss = full_info.rss / 1024**3
    vms = full_info.vms / 1024**3
    # swap currently (sep 2024) not working on mac OS, so leave it out for now
    #swap = full_info.swap / 1024**3

    log.info ('memory use [GB]: uss={:.3f}, rss={:.3f}, maxrss={:.3f}, '
              'vms={:.3f} {}' #, swap={:.3f} {}'
              .format(uss, rss, maxrss, vms, label))

    # also report disk use
    disk_use (label=label)

    return


################################################################################

def mem_use_old (label=''):

    # ru_maxrss is in units of kilobytes on Linux; however, this seems
    # to be OS dependent as on mac os it is in units of bytes; see
    # manpages of "getrusage"
    if sys.platform=='darwin':
        norm = 1024**3
    else:
        norm = 1024**2

    mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/norm
    mem_now = psutil.Process().memory_info().rss / 1024**3
    mem_virt = psutil.Process().memory_info().vms / 1024**3

    log.info ('memory use [GB]: rss={:.3f}, maxrss={:.3f}, vms={:.3f} in {}'
              .format(mem_now, mem_max, mem_virt, label))

    return


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

# from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


################################################################################

def main():
    """Wrapper allowing optimal_subtraction to be run from the command line"""

    parser = argparse.ArgumentParser(description='Run optimal_subtraction on images')
    parser.add_argument('--new_fits', type=str, default=None,
                        help='filename of new image')
    parser.add_argument('--ref_fits', type=str, default=None,
                        help='filename of ref image')
    parser.add_argument('--new_fits_mask', type=str, default=None,
                        help='filename of new image mask')
    parser.add_argument('--ref_fits_mask', type=str, default=None,
                        help='filename of ref image mask')
    parser.add_argument('--set_file', type=str, default='set_zogy',
                        help='name of settings file')
    parser.add_argument('--logfile', type=str, default=None,
                        help='if name is provided, an output logfile is created')
    parser.add_argument('--redo_new', default=None,
                        help='force re-doing new-image source-extractor and '
                        'astrometry.net parts even when products already present')
    parser.add_argument('--redo_ref', default=None,
                        help='force re-doing ref-image source-extractor and '
                        'astrometry.net parts even when products already present')
    parser.add_argument('--verbose', default=None,
                        help='increase verbosity level')
    parser.add_argument('--nthreads', type=int, default=None,
                        help='number of threads (CPUs) to use (default: None)')
    parser.add_argument('--telescope', type=str, default='ML1', help='telescope')
    parser.add_argument('--keep_tmp', default=None,
                        help='keep intermediate/temporary files')


    # replaced [global_pars] function with importing [set_file];
    # all former global parameters are now referred to as set_zogy.[parameter
    # name]. This importing is done inside [optimal_subtraction] in
    # case it is not called from the command line.
    args = parser.parse_args()
    optimal_subtraction(args.new_fits, args.ref_fits, args.new_fits_mask,
                        args.ref_fits_mask, args.set_file, args.logfile,
                        args.redo_new, args.redo_ref,
                        args.verbose, args.nthreads, args.telescope,
                        args.keep_tmp)

if __name__ == "__main__":
    main()
