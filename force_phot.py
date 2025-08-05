import os
import argparse
import re
import traceback
import subprocess


#import multiprocessing as mp
#mp_ctx = mp.get_context('spawn')


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

import numpy as np

import astropy.io.fits as fits
from astropy.coordinates import Angle, SkyOffsetFrame, SkyCoord
from astropy.table import Table, hstack, vstack, unique
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.io import ascii

import zogy
import set_zogy
set_zogy.verbose=False

import fitsio

from google.cloud import storage

# since version 0.9.3 (Feb 2023) this module was moved over from
# BlackBOX to ZOGY to be able to perform forced photometry on an input
# (Gaia) catalog inside ZOGY
__version__ = '1.3.0'


################################################################################

def force_phot (table_in, image_indices_dict, mask_list=None, trans=True,
                ref=True, fullsource=False, nsigma=3, apphot_radii=None,
                bkg_global=True, bkg_radii=None, bkg_objmask=True,
                bkg_limfrac=None, pm_epoch=2016.0, keys2add=None,
                keys2add_dtypes=None, thumbnails=False, size_thumbnails=None,
                remove_psf=False, tel=None, ncpus=1):


    """Forced photometry on MeerLICHT/BlackGEM images at the input
       coordinates provided through the input Astropy Table [table_in]
       and a dictionary [image_indices_dict] defining {image1:
       indices1, image2: indices2, ..}. The results are returned in a
       single astropy Table.

       A list of image masks can be provided through [mask_list],
       which should have the same number and order of elements as the
       images defined in the keys of [image_indices_dict].

       Depending on whether [trans] and/or [ref] and/or [fullsource]
       is set to True, forced photometry is done for the transient
       and/or reference and/or full-source catalogs.

       In the transient case, the transient magnitude (MAG_ZOGY) and
       corresponding error (MAGERR_ZOGY), its signal-to-noise ratio
       (SNR_ZOGY) and the [nsigma]-sigma transient limiting magnitude
       (TRANS_LIMMAG) are determined at the input coordinates and
       added as columns to the output table.

       In the fullsource case, the optimal magnitude (MAG_OPT) and
       corresponding error (MAGERR_OPT), its signal-to-noise ratio
       (SNR_OPT) and the [nsigma]-sigma limiting magnitude
       (TRANS_LIMMAG) are determined at the input coordinates and
       added as columns to the output table.

       Any header keyword listed in [keys2add] with the corresponding
       data type listed in [keys2add_dtypes] will be added as an
       output column, where the values are searched for in the input
       image headers.

       To speed up the processing, [ncpus] can be increased to the
       number of CPUs available.


    Parameters:
    -----------

    table_in: Astropy Table (no default) that needs to include at
              least the columns RA_IN and DEC_IN with the coordinates
              in decimal degrees at which the photometry will be
              determined. Any additional column will also be recorded
              in the output table.

    image_indices_dict: dictionary (no default) with reduced image
                        filenames - including full path - as keys and
                        the indices of [table_in] for which the RA_IN,
                        DEC_IN coordinates will be used to perform the
                        forced photometry.

    mask_list: list of masks (default=None) corresponding to the input
               images; the order of this list needs to be the same as
               the keys in [image_indices_dict]

    trans: boolean (default=True), if True the transient magnitudes
           and limits will be extracted

    ref: boolean (default=True), if True the reference magnitudes and
         limits will be extracted

    fullsource: boolean (default=False), if True the full-source
                magnitudes and limits will be extracted

    nsigma: float (default=3), the significance level at which the
            limiting magnitudes will be determined; this values will
            be indicated in the relevant output table column names

    apphot_radii: list or numpy array of radii (default=None) to use
                  for aperture photometry

    bkg_global: boolean (default=True) determining whether to use the
                global or local background determination in the
                photometry

    bkg_radii: 2-element list or numpy array (default=None) indicating
               the inner and outer radius of the sky annulus used for
               the local background determination

    bkg_objmask: boolean (default=True) determining whether to discard
                 pixels affected by objects (stars or galaxies, as
                 detected by source extractor) in the background
                 annulus to determine the local background; if set to
                 False, all pixels that have not been masked as
                 bad/saturated/etc. in the input mask will be used for
                 the local background determination

    bkg_limfrac: if more than this fraction of background annulus
                 pixels is affected by objects and/or masked pixels,
                 the global background is adopted

    pm_epoch: float (default=2016.0), proper motion reference epoch;
              if [table_in] includes the columns PMRA_IN and PMDEC_IN
              (units: mas/yr), the proper motion is taken into account
              using this reference epoch

    keys2add: list of strings (default=None); header keywords that
              will be added as columns to the output table

    keys2add_dtypes: list of dtypes (default=None); corresponding
                     dtypes of the header keywords provided in
                     [keys2add]


    ncpus: int (default=1); number of processes/tasks to use

    thumbnails: boolean (default=None) determining whether to include
                the thumbnail images of size [size_thumbnails] to the
                output catalog. Which thumbnails are included depends
                on [trans], [ref] and [fullsource]

    tel: str (default=None) indicating telescope (e.g. ML1, BG2)

    size_thumbnails: int (default=100), size in pixels of thumbnails

    remove_psf: bool (default=False), option to remove inferred PSF
                from image before measuring next object; if True, this
                will cause single CPU to be used even if multiple are
                available

    """


    # no point continuing if input [trans] and [fullsource] are both
    # set to False
    if not trans and not fullsource:
        log.error ('input parameters [trans] and [fullsource] are both set to '
                   'False; no data will be extracted')
        return None


    # extracting the reference magnitudes is applicable only to
    # MeerLICHT/BlackGEM images because of the specific directory
    # structure, so switch [ref] off if telescope is not ML/BG
    if tel not in ['ML1', 'BG2', 'BG3', 'BG4', 'BG']:
        log.info ('setting input parameter ref to False as telescope is not '
                  'one of MeerLICHT or BlackGEM telescopes')
        ref = False


    # to add keys or not
    if keys2add is not None and keys2add_dtypes is not None:
        add_keys = True
    else:
        add_keys = False


    # initialise output table, starting from the input table
    names = table_in.colnames
    dtypes = []
    for colname in names:
        dtypes += [table_in[colname].dtype]


    # add the filename and pixel positions
    names += ['FILENAME', 'X_POS_RED', 'Y_POS_RED']
    dtypes += ['U30', float, float]


    # initialize keyword columns
    if add_keys:
        for ikey, key in enumerate(keys2add):
            names.append(key)
            dtype_str = keys2add_dtypes[ikey]
            if dtype_str in ['float', 'int', 'bool']:
                if '32' in dtype_str:
                    dtype = np.float32(dtype_str)
                else:
                    dtype = eval(dtype_str)
            else:
                dtype = dtype_str

            dtypes.append(dtype)

            # if QC-FLAG is a requested keyword, while TQC-FLAG is
            # not, then add the latter if trans is True
            if key == 'QC-FLAG' and 'TQC-FLAG' not in keys2add and trans:
                names += ['TQC-FLAG']
                dtypes += ['U6']


    # add FLAGS_MASK_RED, which is determined irrespective of a match with
    # a full-source catalog source
    if 'FLAGS_MASK_RED' not in names:
        names += ['FLAGS_MASK_RED']
        dtypes += ['int16']


    # initialize columns to be determined below
    if fullsource:

        # optimal photometry columns
        names_fullsource = ['FLAGS_OPT_RED', 'BACKGROUND_RED', 'MAG_OPT_RED',
                            'MAGERR_OPT_RED', 'MAGERRTOT_OPT_RED', 'SNR_OPT_RED',
                            'LIMMAG_OPT_RED']
        names += names_fullsource
        dtypes += ['int16']
        dtypes += ['float32'] * 6


        # add corresponding fluxes
        names += ['FNU_OPT_RED', 'FNUERR_OPT_RED', 'FNUERRTOT_OPT_RED']
        dtypes += ['float32'] * 3


        # add aperture photometry columns if needed
        if apphot_radii is not None:

            for radius in apphot_radii:
                names += ['MAG_APER_R{}xFWHM_RED'.format(radius),
                          'MAGERR_APER_R{}xFWHM_RED'.format(radius),
                          'MAGERRTOT_APER_R{}xFWHM_RED'.format(radius),
                          'SNR_APER_R{}xFWHM_RED'.format(radius)]
                dtypes += ['float32'] * 4


                # add corresponding fluxes
                names += ['FNU_APER_R{}xFWHM_RED'.format(radius),
                          'FNUERR_APER_R{}xFWHM_RED'.format(radius),
                          'FNUERRTOT_APER_R{}xFWHM_RED'.format(radius)]
                dtypes += ['float32'] * 3



        # add thumbnail if relevant
        if thumbnails:
            names += ['THUMBNAIL_RED']
            dtypes += ['float32']



    if trans:

        names_trans = ['MAG_ZOGY', 'MAGERR_ZOGY', 'MAGERRTOT_ZOGY',
                       'SNR_ZOGY', 'LIMMAG_ZOGY']
        names += names_trans
        dtypes += ['float32'] * 5


        # add corresponding fluxes
        names += ['FNU_ZOGY', 'FNUERR_ZOGY', 'FNUERRTOT_ZOGY']
        dtypes += ['float32'] * 3


        # add thumbnails if relevant
        if thumbnails:
            names += ['THUMBNAIL_D', 'THUMBNAIL_SCORR']
            dtypes += ['float32'] * 2



    if ref:

        # add pixelcoordinates corresponding to input RA/DEC to table
        names += ['X_POS_REF', 'Y_POS_REF']
        dtypes += [float, float]


        # add FLAGS_MASK for the reference image
        if 'FLAGS_MASK_REF' not in names:
            names += ['FLAGS_MASK_REF']
            dtypes += ['int16']


        # magnitude, snr and limiting magnitude columns
        names_ref = ['FLAGS_OPT_REF', 'BACKGROUND_REF', 'MAG_OPT_REF',
                     'MAGERR_OPT_REF', 'MAGERRTOT_OPT_REF', 'SNR_OPT_REF',
                     'LIMMAG_OPT_REF']
        names += names_ref
        dtypes += ['int16']
        dtypes += ['float32'] * 6


        # add corresponding fluxes
        names += ['FNU_OPT_REF', 'FNUERR_OPT_REF', 'FNUERRTOT_OPT_REF']
        dtypes += ['float32'] * 3


        # add aperture photometry columns if needed
        if apphot_radii is not None:

            for radius in apphot_radii:
                names += ['MAG_APER_R{}xFWHM_REF'.format(radius),
                          'MAGERR_APER_R{}xFWHM_REF'.format(radius),
                          'MAGERRTOT_APER_R{}xFWHM_REF'.format(radius),
                          'SNR_APER_R{}xFWHM_REF'.format(radius)]
                dtypes += ['float32'] * 4


                # add corresponding fluxes
                names += ['FNU_APER_R{}xFWHM_REF'.format(radius),
                          'FNUERR_APER_R{}xFWHM_REF'.format(radius),
                          'FNUERRTOT_APER_R{}xFWHM_REF'.format(radius)]
                dtypes += ['float32'] * 3



        # in case trans==True, add these ZOGY+REF columns
        if trans:
            names += ['MAG_ZOGY_PLUSREF',
                      'MAGERR_ZOGY_PLUSREF',
                      'MAGERRTOT_ZOGY_PLUSREF',
                      'SNR_ZOGY_PLUSREF']
            dtypes += ['float32'] * 4

            # add corresponding fluxes
            names += ['FNU_ZOGY_PLUSREF',
                      'FNUERR_ZOGY_PLUSREF',
                      'FNUERRTOT_ZOGY_PLUSREF']
            dtypes += ['float32'] * 3



        # add thumbnail if relevant
        if thumbnails:
            names += ['THUMBNAIL_REF']
            dtypes += ['float32']




    # convert [image_indices_dict] to a list of [image, indices]
    # lists, so it is more easily processed by pool_func
    image_indices_list = []
    for k, v in image_indices_dict.items():
        image_indices_list.append([k,v])


    # for testing, limit the number of images
    #ntest = 50
    #image_indices_list = image_indices_list[:ntest]


    # if mask_list is not None, add the masks in the above list as a
    # third item, which will be unpacked correctly in [get_rows]
    # there must be a better way of doing this
    if mask_list is not None:
        image_indices_list = [[l[0][0], l[0][1], l[1]]
                              for l in zip(image_indices_list, mask_list)]


    # check if there are any images to process at all
    nimages = len(image_indices_list)
    log.info ('effective number of images from which to extract magnitudes: {}'
              .format(nimages))
    if nimages == 0:
        log.critical ('no images could be found matching the input coordinates ')


    # input parameters to [get_rows]
    pars = [table_in, trans, ref, fullsource, nsigma, apphot_radii,
            bkg_global, bkg_radii, bkg_objmask, bkg_limfrac, pm_epoch,
            keys2add, add_keys, names, dtypes, thumbnails, size_thumbnails,
            remove_psf]


    if nimages == 1:
        # for single image, execute [get_rows] without pool_func, and
        # provide ncpus as input parameter, which will multiprocess
        # the function zogy.get_psfoptflux_mp such that the different cpus
        # process different lists of objects on the same image
        table = get_rows (image_indices_list[0], *pars, ncpus=ncpus)
        table_list = [table]

    else:

        if ncpus > 1:
            # for multiple images, execute [get_rows] with pool_func, such
            # that the different cpus process different images
            table_list = zogy.pool_func (get_rows, image_indices_list, *pars,
                                         nproc=ncpus)
        else:
            table_list = []
            for image_indices in image_indices_list:
                table_list.append(get_rows(image_indices, *pars))



    # remove None entries, e.g. due to coordinates off the field
    log.info ('removing None entries in [table_list]')
    table_list = [item for item in table_list if item is not None]


    # finished multi-processing filenames
    log.info ('stacking individual tables into output table')
    ntables = len(table_list)
    if ntables > 0:
        table = vstack(table_list)
    else:
        return None


    # sort in time
    log.info ('sorting by FILENAME')
    table.sort(['FILENAME'])


    zogy.mem_use ('at end of [force_phot]')


    # return table
    return table


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

def remove_empty (list_in):

    while True:
        try:
            i = list_in.index('')
            log.warning ('removing empty string from list')
            list_in.pop(i)
        except:
            break


################################################################################

def add_drop_fz (filename):

    if zogy.isfile(filename):
        return filename
    elif zogy.isfile('{}.fz'.format(filename)):
        return '{}.fz'.format(filename)
    elif zogy.isfile(filename.split('.fz')[0]):
        return filename.split('.fz')[0]
    else:
        #return None
        log.warning ('no (un)compressed version of {} found'.format(filename))
        return filename


################################################################################

def get_rows (image_indices, table_in, trans, ref, fullsource, nsigma,
              apphot_radii, bkg_global, bkg_radii, bkg_objmask, bkg_limfrac,
              pm_epoch, keys2add, add_keys, names, dtypes, thumbnails,
              size_thumbnails, remove_psf, ncpus=None):


    # extract filenames and table indices from input list
    # [image_indices] consisting of [filename, [index1, index2, ..]]
    # with possibly fits_mask as a 3rd element
    if len(image_indices)==3:
        filename, indices, fits_mask = image_indices
    else:
        filename, indices = image_indices
        fits_mask = None


    log.info ('processing {}'.format(filename))
    zogy.mem_use ('[force_phot.get_rows] at start')


    # infer telescope name from basename
    tel = filename.split('/')[-1][0:3]


    # create a zero-valued table with shape ncoords x number of column
    # names, with the names and dtypes set by the corresponding input
    # parameters
    ncoords = len(indices)
    table = Table(np.zeros((ncoords,len(names))), names=names, dtype=dtypes)
    colnames = table.colnames

    # loop columns
    for colname in colnames:
        # copy values from table_in if it exists
        if colname in table_in.colnames:
            table[colname] = table_in[colname][indices]

        # set magnitude columns to +99
        if 'MAG' in colname:
            table[colname] = 99


    # read header
    basename = filename.split('.fits')[0]
    fits_red = '{}.fits.fz'.format(basename)
    fits_cat = '{}_cat.fits'.format(basename)
    fits_trans = '{}_trans.fits'.format(basename)
    # try to read transient catalog header, as it is more complete
    # than the full-source catalog header
    if zogy.isfile(fits_trans):
        hdrfile2read = '{}_hdr.fits'.format(fits_trans.split('.fits')[0])
        if not zogy.isfile(hdrfile2read):
            hdrfile2read = fits_trans

    elif zogy.isfile(fits_cat):
        hdrfile2read = '{}_hdr.fits'.format(fits_cat.split('.fits')[0])
        if not zogy.isfile(hdrfile2read):
            hdrfile2read = fits_cat

    elif zogy.isfile(fits_red):
        hdrfile2read = '{}_hdr.fits'.format(fits_red.split('.fits')[0])
        if not zogy.isfile(hdrfile2read):
            hdrfile2read = fits_red

    else:
        log.warning ('reduced image, full-source and transient catalog all '
                     'do not exist for {}; skipping its extraction'
                     .format(basename))
        return None


    # read header
    try:
        log.info ('reading header of {}'.format(hdrfile2read))
        # fitsio
        #header = fitsio.FITS(hdrfile2read)[-1].read_header()
        # astropy
        #header = zogy.read_hdulist (hdrfile2read, get_data=False, get_header=True)
        t0 = time.time()
        with fits.open(hdrfile2read) as hdulist:
            header = hdulist[-1].header

        log.info ('read {} in {:.3f}s'.format(hdrfile2read, time.time()-t0))

    except:
        log.exception ('trouble reading header of {}; skipping its extraction'
                       .format(hdrfile2read))
        return None



    # need to define proper shapes for thumbnail columns; if
    # [thumbnails] is set, adopt the corresponding input
    # [size_thumbnails]; otherwise, adopt the size of the thumbnails
    # in the transient catalog
    if thumbnails:
        size_tn = size_thumbnails
    else:
        size_tn = zogy.get_par(set_zogy.size_thumbnails,tel)

    for col in colnames:
        if 'THUMBNAIL' in col:
            table[col] = np.zeros((len(table), size_tn, size_tn),
                                  dtype='float32')


    # start to fill in table
    table['FILENAME'] = basename.split('/')[-1]


    # add header keywords to output table
    if add_keys:
        for key in keys2add:

            try:
                table[key] = header[key]
            except:
                table[key] = None
                log.warning ('keyword {} not in header of {}'
                             .format(key, hdrfile2read))

            if key=='QC-FLAG' and 'TQC-FLAG' not in keys2add and trans:
                try:
                    table['TQC-FLAG'] = header['TQC-FLAG']
                except:
                    table['TQC-FLAG'] = None
                    log.warning ('keyword TQC-FLAG not in header of {}'
                                 .format(hdrfile2read))



    # full-source; determining optimal flux
    # -------------------------------------
    if fullsource:

        # infer full-source magnitudes and S/N
        table = infer_mags (table, basename, fits_mask, nsigma, apphot_radii,
                            bkg_global, bkg_radii, bkg_objmask, bkg_limfrac,
                            pm_epoch, keys2add, add_keys, thumbnails, size_tn,
                            imtype='new', remove_psf=remove_psf, tel=tel,
                            ncpus=ncpus)



    # transient; extracting ZOGY fluxes
    # ---------------------------------
    if trans:

        # infer transient magnitudes and S/N
        table = infer_mags (table, basename, fits_mask, nsigma, apphot_radii,
                            bkg_global, bkg_radii, bkg_objmask, bkg_limfrac,
                            pm_epoch, keys2add, add_keys, thumbnails, size_tn,
                            imtype='trans', tel=tel, ncpus=ncpus)


    # reference; determining optimal fluxes
    # -------------------------------------
    if ref:


        # infer reference image basename including full path
        basename_ref = get_basename_ref (basename, filename, tel, header)
        log.info ('basename_ref: {}'.format(basename_ref))


        # if name of reference image could not be inferred, provide
        # warning
        if basename_ref is None:

            log.warning ('skipping extraction of magnitudes for {}'
                         .format(basename))

        else:

            # reference mask image; input parameter fits_mask refers to
            # the reduced image mask
            fits_mask = '{}_mask.fits.fz'.format(basename_ref.split('_red')[0])


            # infer reference magnitudes and S/N
            table = infer_mags (table, basename_ref, fits_mask, nsigma,
                                apphot_radii, bkg_global, bkg_radii, bkg_objmask,
                                bkg_limfrac, pm_epoch, keys2add, add_keys,
                                thumbnails, size_tn, imtype='ref', tel=tel,
                                ncpus=ncpus)


            # for transients, add any potential source in the reference
            # image to the transient flux and save the result in the
            # column MAG_ZOGY_PLUSREF
            if trans:

                fnu_zogy = np.array(table['FNU_ZOGY'])
                fnu_ref = np.array(table['FNU_OPT_REF'])
                fnu_tot = fnu_zogy + fnu_ref
                table['FNU_ZOGY_PLUSREF'] = fnu_tot.astype('float32')

                mag_tot = np.zeros_like(fnu_tot, dtype='float32') + 99
                mask_pos = (fnu_tot > 0)
                mag_tot[mask_pos] = -2.5 * np.log10(fnu_tot[mask_pos]) + 23.9
                table['MAG_ZOGY_PLUSREF'] = mag_tot.astype('float32')


                # the corresponding error
                fnuerr_zogy = np.array(table['FNUERR_ZOGY'])
                fnuerr_ref = np.array(table['FNUERR_OPT_REF'])
                fnuerr_tot = np.sqrt(fnuerr_zogy**2 + fnuerr_ref**2)
                table['FNUERR_ZOGY_PLUSREF'] = fnuerr_tot.astype('float32')

                magerr_tot = np.zeros_like(fnu_tot, dtype='float32') + 99
                pogson = 2.5 / np.log(10)
                magerr_tot[mask_pos] = (pogson * fnuerr_tot[mask_pos]
                                        / fnu_tot[mask_pos])
                table['MAGERR_ZOGY_PLUSREF'] = magerr_tot.astype('float32')


                # also determine the total error, which includes the
                # zeropoint error in both the new and reference image
                fnuerrtot_zogy = np.array(table['FNUERRTOT_ZOGY'])
                fnuerrtot_ref = np.array(table['FNUERRTOT_OPT_REF'])
                fnuerrtot_tot = np.sqrt(fnuerrtot_zogy**2 + fnuerrtot_ref**2)
                table['FNUERRTOT_ZOGY_PLUSREF'] = fnuerrtot_tot.astype('float32')

                magerrtot_tot = np.zeros_like(fnu_tot, dtype='float32') + 99
                magerrtot_tot[mask_pos] = (pogson * fnuerrtot_tot[mask_pos]
                                           / fnu_tot[mask_pos])
                table['MAGERRTOT_ZOGY_PLUSREF'] = magerrtot_tot.astype('float32')


                # S/N
                mask_nonzero = (fnuerr_tot != 0)
                snr_tot = np.zeros_like(fnu_tot, dtype='float32')
                snr_tot[mask_nonzero] = (fnu_tot[mask_nonzero] /
                                         fnuerr_tot[mask_nonzero])
                table['SNR_ZOGY_PLUSREF'] = snr_tot



    zogy.mem_use ('[force_phot.get_rows] at end')

    return table


################################################################################

def get_basename_ref (basename, filename, tel, header):

    if tel == 'ML1':
        # infer path to ref folder from basename
        ref_dir = '{}/ref'.format(basename.split('/red/')[0])
    else:
        # for BlackGEM, refer to GCP bucket; refer to the right
        # one depending on the processing environment
        # (test/staging/production)
        bucket_env = filename.split('gs://')[-1].split('blackgem-red')[0]
        ref_dir = 'gs://{}blackgem-ref'.format(bucket_env)


    # read field ID and reference image name from header
    if 'Z-REF' not in header:
        log.warning ('header keyword Z-REF with reference image name not '
                     'found for {}; returning None for basename_ref'
                     .format(basename))
        basename_ref = None
    else:
        # reference image basename including full path
        obj, ref_name = header['OBJECT'], header['Z-REF']
        basename_ref = '{}/{}/{}'.format(ref_dir, obj, ref_name)


    return basename_ref


################################################################################

def infer_mags (table, basename, fits_mask, nsigma, apphot_radii, bkg_global,
                bkg_radii, bkg_objmask, bkg_limfrac, pm_epoch, keys2add,
                add_keys, thumbnails, size_tn, imtype='new', remove_psf=False,
                tel='ML1', ncpus=None):


    log.info ('executing infer_mags() ...')


    # is google cloud being used?
    google_cloud = (basename[0:5] == 'gs://')


    # label in logging corresponding to 'new', 'ref' and 'trans' imtypes
    label_dict = {'new': 'full-source', 'ref': 'reference', 'trans': 'transient'}
    label = label_dict[imtype]


    # similar dictionary for additional string to add to output table colnames
    s2add_dict = {'new': '_RED', 'ref': '_REF', 'trans': '_TRANS'}
    s2add = s2add_dict[imtype]


    # shorthand
    new = (imtype == 'new')
    trans = (imtype == 'trans')
    ref = (imtype == 'ref')


    # define fits_red in any case
    fits_red = '{}.fits.fz'.format(basename)
    if not google_cloud:
        fits_red = add_drop_fz (fits_red)


    if trans:

        # filenames relevant for magtypes 'trans'
        fits_Fpsf = '{}_Fpsf.fits.fz'.format(basename)
        fits_trans = '{}_trans.fits'.format(basename)
        fits_tlimmag = '{}_trans_limmag.fits.fz'.format(basename)
        fits_Scorr = '{}_Scorr.fits.fz'.format(basename)
        fits_D = '{}_D.fits.fz'.format(basename)

        if not google_cloud:
            fits_Fpsf = add_drop_fz (fits_Fpsf)
            fits_tlimmag = add_drop_fz (fits_tlimmag)
            fits_Scorr = add_drop_fz (fits_Scorr)
            fits_D = add_drop_fz (fits_D)


        list2check = [fits_Fpsf, fits_trans, fits_tlimmag, fits_Scorr,
                      fits_D]


    else:

        # filenames relevant for magtype 'full-source' and 'reference'
        fits_limmag = '{}_limmag.fits.fz'.format(basename)
        psfex_bintable = '{}_psf.fits'.format(basename)
        fits_objmask = '{}_objmask.fits.fz'.format(basename)
        fits_bkg_std_mini = '{}_bkg_std_mini.fits.fz'.format(basename)

        if not google_cloud:
            fits_limmag = add_drop_fz (fits_limmag)
            fits_objmask = add_drop_fz (fits_objmask)
            fits_bkg_std_mini = add_drop_fz (fits_bkg_std_mini)


        list2check = [fits_red, psfex_bintable, fits_bkg_std_mini]


        if fits_mask is not None:
            if not google_cloud:
                fits_mask = add_drop_fz (fits_mask)

            list2check += [fits_mask]



    # check if required images/catalogs are available
    if not google_cloud:

        for fn in list2check:
            fn_tmp = add_drop_fz (fn)
            if fn_tmp is None or not zogy.isfile (fn_tmp):
                log.warning ('{} not found; skipping extraction of {} magnitudes '
                             'for {}'.format(fn_tmp, label, basename))
                return table

    else:
        # above check on each file separately is very slow
        # in Google cloud; instead get list of all files with
        # same basename
        flist = zogy.list_files(basename.split('_red')[0])
        #log.info ('flist: {}'.format(flist))

        # check if required files are in this list
        for fn in list2check:
            if fn is None or fn not in flist:
                log.warning ('{} not found; skipping extraction of {} magnitudes '
                             'for {}'.format(fn, label, basename))
                return table



    # read header
    try:

        hdrfile2read = '{}_hdr.fits'.format(fits_red.split('.fits')[0])
        #header = zogy.read_hdulist (hdrfile2read, get_data=False, get_header=True)
        with fits.open(hdrfile2read) as hdulist:
            header = hdulist[-1].header


        if 'IMAGEW' in header and 'IMAGEH' in header:
            # infer data shape from header keywords
            data_shape = (header['IMAGEH'], header['IMAGEW'])

        else:
            # need to read fits_red instead of _hdr.fits file until
            # both reduced image and reference image contain IMAGEW
            # and IMAGEH keywords (added by Astrometry.net for reduced
            # image; in buildref.py for ref image since Dec 2024)

            hdrfile2read = fits_red
            #header = zogy.read_hdulist (hdrfile2read, get_data=False,
            #                            get_header=True)
            with fits.open(hdrfile2read) as hdulist:
                header = hdulist[-1].header

            # infer data_shape from header
            if 'ZNAXIS2' in header and 'ZNAXIS1' in header:
                data_shape = (header['ZNAXIS2'], header['ZNAXIS1'])
            elif 'NAXIS2' in header and 'NAXIS1' in header:
                data_shape = (header['NAXIS2'], header['NAXIS1'])
            else:
                log.error ('not able to infer data shape from header of {}'
                           .format(fits_red))



        ysize, xsize = data_shape
        log.info ('data shape inferred from header of {}: {}'
                  .format(hdrfile2read, data_shape))



        # also read transient header
        if trans:
            hdrfile2read = '{}_hdr.fits'.format(fits_trans.split('.fits')[0])
            #header_trans = zogy.read_hdulist (hdrfile2read, get_data=False,
            #                                  get_header=True)
            with fits.open(hdrfile2read) as hdulist:
                header_trans = hdulist[-1].header

    except:
        log.exception ('trouble reading header of {}; skipping extraction of {} '
                       'magnitudes for {}'
                       .format(hdrfile2read, label, basename))
        return table



    # read FWHM from the header
    if 'S-FWHM' in header:
        fwhm = header['S-FWHM']
    elif 'PSF-FWHM' in header:
        fwhm = header['PSF-FWHM']
    else:
        fwhm = 5
        log.warning ('keywords S-FWHM nor PSF-FWHM present in the header '
                     'for {}; assuming fwhm=5 pix'.format(basename))




    # if proper motion need to be corrected for (if [pm_epoch] is not
    # None and [table_in] columns PMRA_IN and PMDEC_IN exist), extract
    # the image date of observation and apply the proper motion
    # correction; N.B.: this needs to be done here so that it is
    # correctly done also for the reference image, which has a
    # different DATE-OBS
    colnames = table.colnames
    if (pm_epoch is not None and 'DATE-OBS' in header and
        'PMRA_IN' in colnames and 'PMDEC_IN' in colnames):

        obsdate = header['DATE-OBS']
        ra, dec = zogy.apply_gaia_pm (
            table, obsdate, epoch=pm_epoch, return_table=False, ra_col='RA_IN',
            dec_col='DEC_IN', pmra_col='PMRA_IN', pmdec_col='PMDEC_IN',
            remove_pmcolumns=False)

    else:
        ra = table['RA_IN'].value
        dec = table['DEC_IN'].value


    # convert input RA/DEC from table to pixel coordinates; needs to
    # be done from table as it may shrink in size between different
    # calls to [infer_mags]
    xcoords, ycoords = WCS(header).all_world2pix(ra, dec, 1)


    # discard entries that were not finite or off the image; NB: this
    # means that any source that is off the reduced image, but present
    # in the reference image will not appear in the reference part of
    # the output table. A way around this is to set both [fullsource]
    # and [trans] to False. Could try to keep all coordinates, but
    # then would have to juggle with masks below, prone to mistakes.

    # make sure xcoords and ycoords are finite
    mask_finite = (np.isfinite(xcoords) & np.isfinite(ycoords))

    # and on the image
    dpix_edge = 5
    mask_on = ((xcoords > dpix_edge) & (xcoords < xsize-dpix_edge) &
               (ycoords > dpix_edge) & (ycoords < ysize-dpix_edge))

    # combination of finite/on-image masks; return if no coordinates
    # left
    mask_ok = mask_finite & mask_on
    if np.sum(mask_ok)==0:
        log.warning ('all of the inferred pixel coordinates are infinite/nan '
                     'and/or off the image; skipping extraction of {} '
                     'magnitudes for {}'.format(label, basename))
        return table


    ncoords_ok = np.sum(mask_ok)
    if np.sum(~mask_ok) != 0:
        log.info ('{} off-image or non-finite coordinates for {} extraction of '
                  '{}'.format(np.sum(~mask_ok), label, basename))
        #for i_c in range(np.sum(~mask_ok)):
        #    log.info ('xcoords: {}, ycoords: {}'
        #              .format(xcoords[~mask_ok][i_c], ycoords[~mask_ok][i_c]))

        xcoords = xcoords[mask_ok]
        ycoords = ycoords[mask_ok]
        table = table[mask_ok]


    # update table with coordinates
    if ref:
        table['X_POS_REF'] = xcoords
        table['Y_POS_REF'] = ycoords
    else:
        table['X_POS_RED'] = xcoords
        table['Y_POS_RED'] = ycoords



    # indices of pixel coordinates; need to be defined after
    # discarding coordinates off the image
    x_indices = (xcoords-0.5).astype(int)
    y_indices = (ycoords-0.5).astype(int)


    # determine several other header keyword values; NB: use of
    # mask_ok, which narrows the table down to valid coordinates
    exptime, filt, zp, zp_std, zp_err, airmass, ext_coeff = get_keys (
        header, table['RA_IN'], table['DEC_IN'], tel)


    # infer zeropoint nsubimages shape for either ref or new/trans
    if ref:
        zp_nsubs_shape = zogy.get_par(set_zogy.zp_nsubs_shape_ref,tel)
    else:
        zp_nsubs_shape = zogy.get_par(set_zogy.zp_nsubs_shape_new,tel)


    # in case zp_nsubs_shape is different from (1,1), replace [zp]
    # with array of zeropoints, one for each object; same for [zp_std]
    # and [zp_err]
    if zp_nsubs_shape == (1,1):

        # zp_coords are used further below to apply the zeropoints, so
        # let them refer to the single zeropoints in case zeropoint
        # shape is (1,1)
        zp_coords = zp
        zp_std_coords = zp_std
        zp_err_coords = zp_err

    else:

        # name of zeropoints numpy file
        fn_zps = '{}_zps.npy'.format(basename)

        # check if numpy file with zeropoint arrays exists
        if not zogy.isfile(fn_zps):

            # if not, revert to the old channel zeropoints
            log.warning ('{} not found; using channel zeropoints instead'
                         .format(fn_zps))
            zp_chan, zp_std_chan, zp_err_chan = zogy.get_zp_header (
                header, set_zogy=set_zogy, channels=True)

            # use get_zp_coords() to convert channel zeropoints, zp_std
            # and zp_err to coordinate-specific zeropoints
            zp_coords, zp_std_coords, zp_err_coords = zogy.get_chan_zp_coords(
                xcoords, ycoords, zp_chan, zp, zp_std_chan, zp_std,
                zp_err_chan, zp_err)

        else:

            if google_cloud:

                # read numpy file from GCP
                bucket_name, bucket_file = zogy.get_bucket_name (fn_zps)
                bucket = storage.Client().bucket(bucket_name)
                blob = bucket.blob(bucket_file)
                with blob.open('rb') as f:
                    zp_subs, zp_std_subs, zp_err_subs, zp_ncal_subs = np.load(f)

            else:

                # read numpy file
                zp_subs, zp_std_subs, zp_err_subs, zp_ncal_subs = np.load(fn_zps)


            # use get_zp_coords() to convert subimage zeropoints, zp_std
            # and zp_err to coordinate-specific zeropoints, i.e. zp,
            # zp_std and zp_err will have same length as number of
            # coordinates
            zp_coords, zp_std_coords, zp_err_coords = zogy.get_zp_coords (
                xcoords, ycoords, zp_subs, zp, zp_std_subs, zp_std,
                zp_err_subs, zp_err, (ysize, xsize), zp_nsubs_shape)




    if False:
        log.info ('exptime:   {}'.format(exptime))
        log.info ('filt:      {}'.format(filt))
        log.info ('zp:        {}'.format(zp))
        log.info ('airmass:   {}'.format(airmass))
        log.info ('ext_coeff: {}'.format(ext_coeff))


    # split between new/ref and transient extraction
    if not trans:

        # determine background standard deviation using [get_bkg_std]
        data_bkg_std = get_bkg_std (fits_bkg_std_mini, xcoords, ycoords,
                                    data_shape, imtype, tel)

        # object mask may not be available, so first check if it
        # exists
        if (fits_objmask is not None and zogy.isfile(fits_objmask)
            and bkg_objmask):
            #objmask = zogy.read_hdulist (fits_objmask, dtype=bool)
            with fits.open(fits_objmask) as hdulist:
                objmask = hdulist[-1].data.astype(bool)
        else:
            # if it does not exist, or input parameter bkg_objmask is
            # False, create an all-False object mask
            objmask = np.zeros (data_shape, dtype=bool)


        # read reduced image; need to use astropy method, as otherwise
        # this will lead to an exception in [zogy.get_psfoptflux_mp] as
        # (probably) the shape attribute is not available when data is
        # read through fitsio.FITS
        #data = zogy.read_hdulist (fits_red, dtype='float32')
        with fits.open(fits_red) as hdulist:
            data = hdulist[-1].data.astype('float32')


        # corresponding mask may not be available, so first check if
        # it exists
        if fits_mask is not None and zogy.isfile(fits_mask):
            log.info ('fits_mask used: {}'.format(fits_mask))
            #data_mask = zogy.read_hdulist (fits_mask, dtype='int16')
            with fits.open(fits_mask) as hdulist:
                data_mask = hdulist[-1].data.astype('int16')

            # mask can be read using fitsio.FITS, but only little bit
            # faster than astropy.io.fits
            #data_mask = fitsio.FITS(fits_mask)[-1][:,:]
        else:
            log.warning ('fits_mask {} does not exist; assuming that none of '
                         'the pixels are flagged'.format(fits_mask))
            data_mask = np.zeros (data_shape, dtype='int16')


        # add combined FLAGS_MASK column to output table using
        # [get_flags_mask_comb]
        table['FLAGS_MASK{}'.format(s2add)] = (
            get_flags_mask_comb(data_mask, xcoords, ycoords, fwhm, xsize, ysize))



        # aperture magnitudes
        # -------------------

        zogy.mem_use ('[force_phot.infer_mags] before calling [zogy.get_apflux]')


        # if [bkg_global] is True, set [bkg_radii] to None so that
        # a zero background flux will be adopted in [get_apflux]
        if bkg_global:
            bkg_radii = None


        # infer bkg_std at xcoords,ycoords; needed as input to apflux
        # in case bkg_std cannot be determined from background annulus
        if len(y_indices)==1:
            bkg_std_coords = np.array([data_bkg_std])
        else:
            bkg_std_coords = data_bkg_std[y_indices, x_indices]

        #log.info ('xcoords[0:10]: {}'.format(xcoords[0:10]))
        #log.info ('ycoords[0:10]: {}'.format(ycoords[0:10]))
        #log.info ('bkg_std_coords[0:10]: {}'.format(bkg_std_coords[0:10]))


        try:

            # determine aperture fluxes at pixel coordinates
            if ncpus is None:
                # submit to [zogy.get_apflux] with single thread,
                # as the multiprocessing is done at the image
                # level, i.e. each cpu is processing a different
                # image. The [force_phot] function only provides
                # ncpus to [get_rows] and this [infer_mags] in
                # case of a single image.
                flux_aps, fluxerr_aps, local_bkg = zogy.get_apflux (
                    xcoords, ycoords, data, data_mask, fwhm, objmask=objmask,
                    apphot_radii=apphot_radii, bkg_radii=bkg_radii,
                    bkg_limfrac=bkg_limfrac, bkg_std_coords=bkg_std_coords,
                    set_zogy=set_zogy, tel=tel, nthreads=1)

            else:
                # submit to [zogy.get_apflux] with [ncpu] threads
                # as this concerns a single image and the
                # multiprocessing should be done at the object
                # level, i.e. each cpu processes different objects
                # on the same image. The [force_phot] function
                # only provides ncpus to [get_rows] and this
                # [infer_mags] in case of a single image
                flux_aps, fluxerr_aps, local_bkg = zogy.get_apflux (
                    xcoords, ycoords, data, data_mask, fwhm, objmask=objmask,
                    apphot_radii=apphot_radii, bkg_radii=bkg_radii,
                    bkg_limfrac=bkg_limfrac, bkg_std_coords=bkg_std_coords,
                    set_zogy=set_zogy, tel=tel, nthreads=ncpus)


        except Exception as e:
            log.error ('exception was raised while executing '
                       '[zogy.get_apflux]; skipping extraction of {} '
                       'magnitudes for {}: {}'.format(label, basename, e))
            log.error (traceback.format_exc())
            return table


        # record local background in table; N.B.: local_bkg
        # inferred in [get_apflux] is also adopted below in the
        # optimal magnitudes determination
        table['BACKGROUND{}'.format(s2add)] = local_bkg.astype('float32')


        # add various aperture columns to table
        for i_rad, radius in enumerate(apphot_radii):

            # get flux for specific aperture radius
            flux_ap = flux_aps[i_rad]
            fluxerr_ap = fluxerr_aps[i_rad]


            # signal-to-noise ratio
            mask_nonzero = (fluxerr_ap != 0)
            snr_ap = np.zeros_like(flux_ap, dtype='float32')
            snr_ap[mask_nonzero] = (flux_ap[mask_nonzero] /
                                    fluxerr_ap[mask_nonzero])
            col_tmp = 'SNR_APER_R{}xFWHM{}'.format(radius, s2add)
            table[col_tmp] = snr_ap


            # infer calibrated magnitudes using the zeropoint
            if zp_coords is not None:

                mag_ap, magerr_ap, magerrtot_ap, \
                    fnu_ap, fnuerr_ap, fnuerrtot_ap = zogy.apply_zp (
                        flux_ap, zp_coords, airmass, exptime, ext_coeff,
                        fluxerr=fluxerr_ap, return_fnu=True,
                        zp_err=zp_std_coords)


                mask_pos = (flux_ap > 0)
                mag_ap[~mask_pos] = 99
                #magerr_ap[~mask_pos] = 99
                #magerrtot_ap[~mask_pos] = 99


                col_tmp = 'MAG_APER_R{}xFWHM{}'.format(radius, s2add)
                table[col_tmp] = mag_ap.astype('float32')
                col_tmp = 'MAGERR_APER_R{}xFWHM{}'.format(radius, s2add)
                table[col_tmp] = magerr_ap.astype('float32')
                col_tmp = 'MAGERRTOT_APER_R{}xFWHM{}'.format(radius, s2add)
                table[col_tmp] = magerrtot_ap.astype('float32')


                # add fluxes to table
                col_tmp = 'FNU_APER_R{}xFWHM{}'.format(radius, s2add)
                table[col_tmp] = fnu_ap.astype('float32')
                col_tmp = 'FNUERR_APER_R{}xFWHM{}'.format(radius, s2add)
                table[col_tmp] = fnuerr_ap.astype('float32')
                col_tmp = 'FNUERRTOT_APER_R{}xFWHM{}'.format(radius, s2add)
                table[col_tmp] = fnuerrtot_ap.astype('float32')



            else:
                if i_rad==0:
                    log.warning ('keyword PC-ZP not in header; unable to '
                                 'infer {} magnitudes for {}'
                                 .format(label, basename))


        # optimal magnitudes
        # ------------------

        zogy.mem_use ('[force_phot.infer_mags] before calling '
                      '[zogy.get_psfoptflux_mp]')

        try:

            # fit local background in zogy.get_psfoptflux_mp() if sky
            # value was not successfully determined by
            # zogy.get_apflux(), i.e. it is zero
            mask_fit_local_bkg = (local_bkg==0)


            # determine optimal fluxes at pixel coordinates
            if ncpus is None:
                # submit to [get_psfoptflux_mp] with single thread, as
                # the multiprocessing is done at the image level,
                # i.e. each cpu is processing a different image
                flux_opt, fluxerr_opt, local_bkg_opt, flags_opt = \
                    zogy.get_psfoptflux_mp(
                        psfex_bintable, data, data_bkg_std**2, data_mask, xcoords,
                        ycoords, imtype=imtype, fwhm=fwhm, local_bkg=local_bkg,
                        mask_fit_local_bkg=mask_fit_local_bkg, remove_psf=remove_psf,
                        get_flags_opt=True, set_zogy=set_zogy, tel=tel, nthreads=1)

            else:
                # submit to [get_psfoptflux_mp] with [ncpu] threads as
                # this concerns a single image and the multiprocessing
                # should be done at the object level, i.e. each cpu
                # processes different objects on the same image. The
                # [force_phot] function only provides ncpus to
                # [get_rows] and [infer_mags] in case of a single
                # image.
                flux_opt, fluxerr_opt, local_bkg_opt, flags_opt = \
                    zogy.get_psfoptflux_mp(
                        psfex_bintable, data, data_bkg_std**2, data_mask, xcoords,
                        ycoords, imtype=imtype, fwhm=fwhm, local_bkg=local_bkg,
                        mask_fit_local_bkg=mask_fit_local_bkg, remove_psf=remove_psf,
                        get_flags_opt=True, set_zogy=set_zogy, tel=tel, nthreads=ncpus)


                if False:
                    # !!!CHECK!!! - try psffit instead of optimal flux
                    __, __, __, __, flux_opt, fluxerr_opt, __ =  (
                        zogy.get_psfoptflux_mp (
                            psfex_bintable, data, data_bkg_std**2, data_mask,
                            xcoords, ycoords, imtype=imtype, fwhm=fwhm,
                            local_bkg=local_bkg,
                            mask_fit_local_bkg=mask_fit_local_bkg,
                            remove_psf=remove_psf, psffit=True,
                            set_zogy=set_zogy, tel=tel, nthreads=ncpus))


        except Exception as e:
            log.error ('exception was raised while executing '
                       '[zogy.get_psfoptflux_mp]; skipping extraction of {} '
                       'magnitudes for {}: {}'.format(label, basename, e))
            log.error (traceback.format_exc())
            return table


        zogy.mem_use ('[force_phot.infer_mags] after calling '
                      '[zogy.get_psfoptflux_mp]')


        if zp_coords is not None:
            # infer calibrated magnitudes using the zeropoint
            mag_opt, magerr_opt, magerrtot_opt, fnu_opt, fnuerr_opt, \
                fnuerrtot_opt = zogy.apply_zp (
                    flux_opt, zp_coords, airmass, exptime, ext_coeff,
                    fluxerr=fluxerr_opt, return_fnu=True, zp_err=zp_std_coords)


            mask_pos = (flux_opt > 0)
            mag_opt[~mask_pos] = 99
            #magerr_opt[~mask_pos] = 99

        else:
            log.warning ('keyword PC-ZP not in header; unable to infer {} '
                         'magnitudes for {}'.format(label, basename))


        # infer limiting magnitudes
        nsigma_orig = 5
        limmags = get_limmags (fits_limmag, google_cloud, y_indices, x_indices,
                               header, nsigma, nsigma_orig, imtype, label=label)


        # calculate signal-to-noise ratio; applies to either the SNR
        # of the limit or the matched source in the catalog
        snr_opt = np.zeros(ncoords_ok)
        mask_nonzero = (fluxerr_opt != 0)
        snr_opt[mask_nonzero] = (flux_opt[mask_nonzero] /
                                 fluxerr_opt[mask_nonzero])


        # update table
        table['FLAGS_OPT{}'.format(s2add)] = flags_opt.astype('int16')
        table['MAG_OPT{}'.format(s2add)] = mag_opt.astype('float32')
        table['MAGERR_OPT{}'.format(s2add)] = magerr_opt.astype('float32')
        table['MAGERRTOT_OPT{}'.format(s2add)] = magerrtot_opt.astype('float32')
        table['SNR_OPT{}'.format(s2add)] = snr_opt.astype('float32')
        table['LIMMAG_OPT{}'.format(s2add)] = (limmags.astype('float32'))


        # sky background may have been updated when determining
        # the optimal flux
        table['BACKGROUND{}'.format(s2add)] = local_bkg_opt.astype('float32')


        # add fluxes
        table['FNU_OPT{}'.format(s2add)] = fnu_opt.astype('float32')
        table['FNUERR_OPT{}'.format(s2add)] = fnuerr_opt.astype('float32')
        table['FNUERRTOT_OPT{}'.format(s2add)] = fnuerrtot_opt.astype('float32')


        # add thumbnail image
        if thumbnails:

            # thumbnail to add depends on fullsource and ref
            if ref:
                key_tn = 'THUMBNAIL_REF'
            else:
                key_tn = 'THUMBNAIL_RED'

            # extract thumbnail data
            table[key_tn] = get_thumbnail (data, data_shape, xcoords, ycoords,
                                           size_tn, key_tn, header, tel)



    elif trans:

        # read flux values at xcoords, ycoords
        Fpsf = get_pixel_values (fits_Fpsf, google_cloud, y_indices, x_indices,
                                 dpix=1)

        # get transient limiting magnitude at xcoord, ycoord
        # and convert it back to Fpsferr

        # read limiting magnitude at pixel coordinates
        nsigma_trans_orig = 6
        tlimmags = get_limmags (fits_tlimmag, google_cloud, y_indices, x_indices,
                                header_trans, nsigma, nsigma_trans_orig, imtype,
                                label=label)


        # zp, object airmass, ext_coeff and exptime were determined
        # above; for conversion from transient limiting magnitude to
        # Fpsferr the airmass at image center was used; N.B. tlimmags
        # were already converted from nsigma_trans_orig to nsigma
        # requested
        airmassc = header['AIRMASSC']
        Fpsferr = (10**(-0.4*(tlimmags - zp_coords + airmassc * ext_coeff))
                   * exptime / nsigma)


        # read off transient S/N from Scorr image
        snr_zogy = get_pixel_values (fits_Scorr, google_cloud,
                                     y_indices, x_indices, dpix=1)


        if zp_coords is not None:

            # infer calibrated magnitudes using the zeropoint
            mag_zogy, magerr_zogy, magerrtot_zogy, fnu_zogy, fnuerr_zogy, \
                fnuerrtot_zogy = zogy.apply_zp (
                    Fpsf, zp_coords, airmass, exptime, ext_coeff,
                    fluxerr=Fpsferr, return_fnu=True, zp_err=zp_std_coords)

            mask_zero = (Fpsf==0)
            mag_zogy[mask_zero] = 99
            #magerr_zogy[mask_zero] = 99

        else:

            mag_zogy = np.zeros(ncoords_ok, dtype='float32') + 99
            magerr_zogy = np.zeros(ncoords_ok, dtype='float32') + 99
            log.warning ('keyword PC-ZP not in header; unable to infer {} '
                         'magnitudes for {}'.format(label, basename))



        # update table
        table['MAG_ZOGY'] = mag_zogy.astype('float32')
        table['MAGERR_ZOGY'] = magerr_zogy.astype('float32')
        table['MAGERRTOT_ZOGY'] = magerrtot_zogy.astype('float32')
        table['SNR_ZOGY'] = snr_zogy.astype('float32')
        table['LIMMAG_ZOGY'] = tlimmags.astype('float32')


        # add fluxes
        table['FNU_ZOGY'] = fnu_zogy.astype('float32')
        table['FNUERR_ZOGY'] = fnuerr_zogy.astype('float32')
        table['FNUERRTOT_ZOGY'] = fnuerrtot_zogy.astype('float32')



        # add transient thumbnail images
        if thumbnails:

            fits_dict = {'D': fits_D, 'SCORR': fits_Scorr}
            for key in ['D', 'SCORR']:

                # shorthand
                key_tn = 'THUMBNAIL_{}'.format(key)
                fn = fits_dict[key]

                # check if file exists
                if zogy.isfile(fn):

                    if google_cloud:
                        # read data using astropy
                        #data = zogy.read_hdulist(fn)
                        with fits.open(fn) as hdulist:
                            data = hdulist[-1].data

                    else:
                        # read data using fitsio.FITS
                        data = fitsio.FITS(fn)[-1]

                    table[key_tn] = get_thumbnail (
                        data, data_shape, xcoords, ycoords, size_tn, key_tn,
                        header, tel)
                else:
                    log.warning ('{} not found; skipping extraction of {} for {}'
                                 .format(fn, key_tn, basename))



    return table


################################################################################

def get_thumbnail (data, data_shape, xcoords, ycoords, size_tn, key_tn, header,
                   tel):

    # number of coordinates
    ncoords = len(xcoords)

    # size of full input image
    ysize, xsize = data_shape

    # initialise output thumbnail array
    data_tn = np.zeros((ncoords, size_tn, size_tn), dtype='float32')

    # loop x,y coordinates
    for i_pos in range(ncoords):

        # get index around x,y position using function
        # [zogy.get_index_around_xy]
        x = xcoords[i_pos]
        y = ycoords[i_pos]
        index_full, index_tn, __, __ = zogy.get_index_around_xy(
            ysize, xsize, y, x, size_tn)


        try:

            data_tn[i_pos][index_tn] = data[index_full]

            # orient the thumbnails in North-up, East left
            # orientation
            data_tn[i_pos] = zogy.orient_data (data_tn[i_pos], header,
                                               MLBG_rot90_flip=True, tel=tel)


        except Exception as e:
            log.exception('skipping remapping of {} at x,y: {:.0f},{:.0f} due '
                          'to exception: {}'.format(key_tn, x, y, e))


    return data_tn


################################################################################

def get_limmags (fits_limmag, google_cloud, y_indices, x_indices, header,
                 nsigma, nsigma_orig, imtype, label='full-source'):


    # read limiting magnitude at pixel coordinates
    if zogy.isfile(fits_limmag):

        # infer limiting magnitudes
        limmags = get_pixel_values (fits_limmag, google_cloud, y_indices,
                                    x_indices, dpix=0, get_max=False)

        # convert limmag from number of sigma listed
        # in the image header to input [nsigma]
        if imtype == 'trans':

            if ('T-NSIGMA' in header and
                isinstance(header['T-NSIGMA'], (float, int)) and
                header['T-NSIGMA'] != 0):
                nsigma_orig = header['T-NSIGMA']

        else:

            if ('NSIGMA' in header and
                isinstance(header['NSIGMA'], (float, int)) and
                header['NSIGMA'] != 0):
                nsigma_orig = header['NSIGMA']


        # if nsigma and nsigma_orig not the same, adjust the limmags
        if nsigma_orig != nsigma:
            limmags += -2.5*np.log10(nsigma/nsigma_orig)


    else:
        log.warning ('{} not found; no {} limiting magnitude(s) '
                     'available'.format(fits_limmag, label))
        ncoords = len(y_indices)
        limmags = np.zeros(ncoords, dtype='float32')


    return limmags


################################################################################

def get_pixel_values (filename, google_cloud, y_indices=None, x_indices=None,
                      dpix=0, get_max=True):

    """infer pixel values from [filename] at indices [y_indices,
       x_indices]; if dpix > 0, the maximum value over a square block
       of pixels with width 2*dpix+1 is returned
    """

    t1 = time.time()

    if google_cloud:

        # use astropy.io.fits for google_cloud bucket files
        if False:
            #data = zogy.read_hdulist(filename)
            with fits.open(filename) as hdulist:
                data = hdulist[-1].data

            if y_indices is None or x_indices is None:
                values = data
            else:
                values = data[y_indices, x_indices]

        else:
            with fits.open(filename) as hdulist:

                # section indices must be slices, so need to extract
                # indices one by one instead of data[y_indices, x_indices]
                nvalues = len(y_indices)
                # determine dtype of data
                vals_tmp = hdulist[-1].section[
                    y_indices[0]-dpix:y_indices[0]+1+dpix,
                    x_indices[0]-dpix:x_indices[0]+1+dpix]
                if get_max:
                    value0 = np.max(vals_tmp)
                else:
                    value0 = np.min(vals_tmp)

                values = np.zeros(nvalues, dtype=value0.dtype)
                values[0] = value0 #[0][0]

                log.info ('{} value(s) extracted from {}:'.format(nvalues,
                                                                  filename))
                if nvalues > 1:
                    for i in range(1,nvalues):
                        vals_tmp = hdulist[-1].section[
                            y_indices[i]-dpix:y_indices[i]+1+dpix,
                            x_indices[i]-dpix:x_indices[i]+1+dpix] #[0][0]
                        if get_max:
                            values[i] = np.max(vals_tmp)
                        else:
                            values[i] = np.min(vals_tmp)

    else:

        # use fitsio otherwise
        data = fitsio.FITS(filename)[-1]

        # infer data values at indices
        if y_indices is None or x_indices is None:
            values = data[:,:]
        else:
            # fitsio data indices must be slices, so need to extract
            # indices one by one instead of data[y_indices, x_indices]
            nvalues = len(y_indices)
            # determine dtype of data
            dtype = data[y_indices[0]:y_indices[0]+1,
                         x_indices[0]:x_indices[0]+1].dtype
            values = np.zeros(nvalues, dtype=dtype)
            for i in range(nvalues):
                vals_tmp = data[y_indices[i]-dpix:y_indices[i]+1+dpix,
                                x_indices[i]-dpix:x_indices[i]+1+dpix] #[0][0]
                if get_max:
                    values[i] = np.max(vals_tmp)
                else:
                    values[i] = np.min(vals_tmp)


    log.info ('wall-time spent in get_pixel_values: {:.3f}s'
              .format(time.time()-t1))


    return values


################################################################################

def get_keys (header, ra_in, dec_in, tel):

    # infer the image zeropoint
    keys = ['EXPTIME', 'FILTER', 'DATE-OBS']
    exptime, filt, obsdate = [header[key] for key in keys]

    # get zeropoint from [header]
    if 'PC-ZP' in header:
        zp = header['PC-ZP']
    else:
        zp = None


    # zp_err using function in zogy
    __, zp_std, zp_err = zogy.get_zp_header(header, set_zogy=set_zogy)


    # determine airmass for ra_in and dec_in; for older reference
    # images, airmass of the combined image was forced to be unity, so
    # in that case adopt airmass=1. Otherwise, infer airmass(es) using
    # coordinates and obsdate
    if 'AIRMASS' in header and header['AIRMASS']==1.0:
        airmass = 1.0
    else:
        lat = zogy.get_par(set_zogy.obs_lat,tel)
        lon = zogy.get_par(set_zogy.obs_lon,tel)
        height = zogy.get_par(set_zogy.obs_height,tel)
        airmass = zogy.get_airmass(ra_in, dec_in, obsdate, lat, lon, height)


    # extinction coefficient
    ext_coeff = zogy.get_par(set_zogy.ext_coeff,tel)[filt]


    return exptime, filt, zp, zp_std, zp_err, airmass, ext_coeff


################################################################################

def get_bkg_std (fits_bkg_std_mini, xcoords, ycoords, data_shape, imtype, tel):

    # create background STD from mini image
    with fits.open(fits_bkg_std_mini) as hdulist:
        data_bkg_std_mini = hdulist[-1].data.astype('float32')
        header_mini = hdulist[-1].header


    if 'BKG-SIZE' in header_mini:
        bkg_size = header_mini['BKG-SIZE']
    else:
        bkg_size = zogy.get_par(set_zogy.bkg_boxsize,tel)


    if len(xcoords) == 1:
        # determine scalar bkg_std value from mini image at
        # xcoord, ycoord
        x_indices_mini = ((xcoords-0.5).astype(int)/bkg_size).astype(int)
        y_indices_mini = ((ycoords-0.5).astype(int)/bkg_size).astype(int)
        [data_bkg_std] = data_bkg_std_mini[y_indices_mini, x_indices_mini]

    else:
        # determine full bkg_std image from mini image

        # determine whether interpolation is allowed across
        # different channels in [zogy.mini2back] using function
        # [zogy.get_Xchan_bool]
        chancorr = zogy.get_par(set_zogy.MLBG_chancorr,tel)
        interp_Xchan_std = zogy.get_Xchan_bool (tel, chancorr, imtype,
                                                std=True)
        # if shape different from single ML/BG image, also
        # allow to do cross-channel interpolation
        if data_shape != zogy.get_par(set_zogy.shape_new,tel):
            interp_Xchan_std = True

        data_bkg_std = zogy.mini2back (
            data_bkg_std_mini, data_shape, order_interp=1,
            bkg_boxsize=bkg_size, interp_Xchan=interp_Xchan_std,
            timing=zogy.get_par(set_zogy.timing,tel))


    return data_bkg_std


################################################################################

def get_flags_mask_comb (data_mask, xcoords, ycoords, fwhm, xsize, ysize):

    # identify mask pixels within 2xFWHM of the pixel coordinate

    # full size of window around coordinates, make sure it is even
    size_4fwhm = int(4*fwhm+0.5)
    if size_4fwhm % 2 != 0:
        size_4fwhm += 1

    hsize = int(size_4fwhm/2)

    # define meshgrid
    xy = range(1, size_4fwhm+1)
    xx, yy = np.meshgrid(xy, xy, indexing='ij')

    # initialize flags_mask_comb
    ncoords = len(xcoords)
    flags_mask_comb = np.zeros(ncoords, dtype='int16')

    # loop coordinates
    for m in range(ncoords):

        # get index around x,y position using function
        # [zogy.get_index_around_xy]
        index_full, index_tn, __, __ = zogy.get_index_around_xy(
            ysize, xsize, ycoords[m], xcoords[m], size_4fwhm)


        if np.sum(data_mask[index_full]) != 0:

            # create zero-valued temporary thumbnail data_mask
            data_mask_tn = np.zeros((size_4fwhm, size_4fwhm), dtype='int16')

            # fill temporary thumbnail with values from full data_mask
            data_mask_tn[index_tn] = data_mask[index_full]

            # define mask of pixels within 2*fwhm of the pixel coordinates
            mask_central = (np.sqrt(
                (xx - (xcoords[m] - int(xcoords[m]) + hsize))**2 +
                (yy - (ycoords[m] - int(ycoords[m]) + hsize))**2) < 2*fwhm)

            # previously: add sum of unique values in the thumbnail
            # data mask to the output array flags_mask_comb
            #flags_mask_comb[m] = np.sum(np.unique(data_mask_tn[mask_central]))
            # however, this is not correct, as pixel mask value for
            # the reference image can be combination of e.g. 4 and 8 =
            # 12, so need to perform a bitwise OR combination instead of SUM
            flags_mask_comb[m] = np.bitwise_or.reduce(data_mask_tn[mask_central])



    return flags_mask_comb


################################################################################

def index_images (indices, table_in, mjds_obs, dtime_max, basenames,
                  radecs_cntr, fov_deg, pm_epoch):

    """function to return a dictionary: {index1: imagelist1, index2:
    imagelist2, etc} with the index of [table_in] and the
    corresponding list of images from [basenames] for which the RA,DEC
    coordinates of [table_in[index]] (assumed to be in columns 'RA_IN'
    and 'DEC_IN') is within the image boundaries determined by
    [radecs_cntr] and [fov_deg]. If [table_in] contains 'MJD_IN', the
    images are also filtered on abs(MJD_IN - mjd_obs) <
    [dtime_max]. If columns PMRA_IN and PMDEC_IN are present in
    [table_in], then the proper motion with respect to [pm_epoch] is
    taken into account.

    """

    # dictionary to return
    index_images_dict = {}


    # half the fov
    hfov_deg = fov_deg / 2


    # extract ra and dec
    ras_in = table_in['RA_IN'].value
    decs_in = table_in['DEC_IN'].value


    # if 'MJD_IN' column is present in [table_in], also
    # extract mjds_in
    if 'MJD_IN' in table_in.colnames:
        mjds_in = table_in['MJD_IN'].value
        filter_mjd = True
    else:
        filter_mjd = False


    # in case proper motion is included, add a buffer zone around each
    # image, determined by the maximum proper motion in table_in
    # multiplied by 30 years, to make sure that all sources that could
    # end up within the image are taken into consideration
    if 'PMRA_IN' in table_in.colnames and 'PMDEC_IN' in table_in.colnames:
        pm_buffer = (30/3.6e6) * np.concatenate(
            [table_in['PMRA_IN'].value, table_in['PMDEC_IN'].value]).max()
        #log.info ('proper motion buffer: {:.1f} arcsec'.format(3600*pm_buffer))
        hfov_deg += pm_buffer


    # loop input indices, corresponding to the indices of sources in
    # [table_in]
    for i in indices:

        # mask indicating the files to be used
        mask_files = (np.abs(radecs_cntr[:,1] - decs_in[i]) <= hfov_deg)

        # now determine whether radec is actually on which of
        # images[mask_files]; if proper motion is included, include an
        # additional buffer around the images is used in [hfov_deg]
        target = SkyCoord(ra=ras_in[i], dec=decs_in[i], unit='deg')
        centers = SkyCoord(ra=radecs_cntr[:,0][mask_files],
                           dec=radecs_cntr[:,1][mask_files], unit='deg')
        centers.transform_to(SkyOffsetFrame(origin=centers))
        target_centers = target.transform_to(SkyOffsetFrame(origin=centers))
        xi, eta = target_centers.lon, target_centers.lat
        mask_on = ((np.abs(xi) < hfov_deg*u.deg) &
                   (np.abs(eta) < hfov_deg*u.deg))
        mask_files[mask_files] = mask_on


        if False:
            # this block below is not needed, as the proper motion
            # buffer has already been taken into account, and should a
            # source fall outside the actual image, it will be
            # neglected in [infer_mags]

            # if proper motion is included, need to loop over the list
            # of relevant images only and determine the corrected
            # ra,dec for the MJD_OBS of that particular image
            if apply_pm:

                # corresponding indices
                index_files = np.nonzero(mask_files)[0]
                log.info ('len(index_files): {}'.format(len(index_files)))

                for i_file in index_files:

                    # convert MJD_OBS to date of observations
                    obsdate = Time(mjds_obs[i_file], format='mjd').isot
                    # infer corrected RA and DEC; note that they should be
                    # 1-element arrays
                    ra_corr, dec_corr = zogy.apply_gaia_pm (
                        table_in[i], obsdate, epoch=pm_epoch, return_table=False,
                        ra_col='RA_IN', dec_col='DEC_IN',
                        pmra_col='PMRA_IN', pmdec_col='PMDEC_IN')

                    # now determine whether RA,DEC corrected for proper
                    # motion is actually on this particular image
                    target = SkyCoord(ra=ra_corr, dec=dec_corr, unit='deg')
                    center = SkyCoord(ra=radecs_cntr[i_file,0],
                                      dec=radecs_cntr[i_file,1], unit='deg')
                    center.transform_to(SkyOffsetFrame(origin=center))
                    target_center = target.transform_to(SkyOffsetFrame(origin=center))
                    xi, eta = target_center.lon, target_center.lat
                    # N.B.: use original image (half) size again, so
                    # without the proper motion buffer
                    mask_on = ((np.abs(xi) < (fov_deg/2)*u.deg) &
                               (np.abs(eta) < (fov_deg/2)*u.deg))
                    # mask_on is 1-element mask
                    mask_files[i_file] = mask_on


        # also mask files based on MJD_IN if relevant
        if filter_mjd:
            mask_files &= (np.abs(mjds_obs - mjds_in[i]) <= dtime_max/24)


        # if any file is selected, add it/them to output dictionary
        # [index_images_dict], with the individual indices as the keys
        # and the lists of corresponding images as the values
        if np.any(mask_files):
            basenames_radec = list(np.array(basenames)[mask_files].astype(str))
            index_images_dict[i] = basenames_radec


    return index_images_dict


################################################################################

def get_headkeys (filenames):

    # filenames could be catalogs or images, so split using
    # '_red', but add _red to the basename
    basenames = ['{}_red'.format(fn.split('_red')[0]) for fn in filenames]

    nfiles = len(basenames)
    objects = np.zeros(nfiles, dtype=int)
    mjds_obs = np.zeros(nfiles)
    filts = np.zeros(nfiles, dtype=str)
    ras_cntr = np.zeros(nfiles)
    decs_cntr = np.zeros(nfiles)

    for nfile, basename in enumerate(basenames):

        # read header
        #header = zogy.read_hdulist ('{}_hdr.fits'.format(basename),
        #                            get_data=False, get_header=True)
        with fits.open('{}_hdr.fits'.format(basename)) as hdulist:
            header = hdulist[-1].header


        objects[nfile] = int(header['OBJECT'])
        if 'MJD-OBS' in header:
            mjds_obs[nfile] = header['MJD-OBS']
        elif 'DATE-OBS' in header:
            mjds_obs[nfile] = Time(header['DATE-OBS']).mjd
        else:
            log.error ('MJD-OBS or DATE-OBS not in header of {}'
                       .format(basename))
            mjds_obs[nfile] = None


        filts[nfile] = header['FILTER'].strip()
        ras_cntr[nfile] = header['RA-CNTR']
        decs_cntr[nfile] = header['DEC-CNTR']


    return objects, mjds_obs, filts, ras_cntr, decs_cntr


################################################################################

def get_mjd_mask (mjds, date_start, date_end, date_format):

    mask_mjd = np.ones(mjds.size, dtype=bool)
    if date_start is not None:
        mjd_start = Time(date_start, format=date_format).mjd
        mask_mjd &= (mjds >= mjd_start)

    if date_end is not None:
        mjd_end = Time(date_end, format=date_format).mjd
        mask_mjd &= (mjds <= mjd_end)


    return mask_mjd


################################################################################

def verify_lengths(p1, p2):

    [p1_str] = [p for p in globals() if globals()[p] is p1]
    [p2_str] = [p for p in globals() if globals()[p] is p2]
    err_message = ('input parameters {} and {} need to have the same length'
                   .format(p1_str, p2_str))

    # check if either one is None while the other is not, and vice versa
    if ((p1 is None and p2 is not None) or
        (p2 is None and p1 is not None)):
        log.error (err_message)
        raise SystemExit

    elif p1 is not None and p2 is not None:
        # also check that they have the same length
        if len(p1) != len(p2):
            log.error (err_message)
            raise SystemExit

    return


################################################################################

def create_col_descr(keys2add, header, ra_col, dec_col):

    col_descr = {
        'NUMBER_IN':         'line number of coordinates in input list',
        ra_col:              'input source right ascension (RA)',
        dec_col:             'input source declination (DEC)',
        'FILENAME':          'base filename of matching image',
        'X_POS_RED':         '[pix] x pixel coordinate corresponding to input RA/DEC in red image',
        'Y_POS_RED':         '[pix] y pixel coordinate corresponding to input RA/DEC in red image',
        'FLAGS_MASK_RED':    'OR-combined flagged pixels within 2xFWHM of coords in red image',
        'FLAGS_OPT_RED':     'Optimal photometry flags in red image',
        'BACKGROUND_RED':    '[e-] sky background estimated from sky annulus in red image',
        'MAG_OPT_RED':       '[mag] optimal AB magnitude in red image',
        'MAGERR_OPT_RED':    '[mag] optimal AB magnitude error in red image',
        'MAGERRTOT_OPT_RED': '[mag] optimal AB magnitude total error (incl. ZP error) in red image',
        'SNR_OPT_RED':       'signal-to-noise ratio in red image',
        'LIMMAG_OPT_RED':    '[mag] limiting AB magnitude at nsigma significance in red image',
        'FNU_OPT_RED':       '[microJy] flux in red image (AB mag = -2.5 log10 fnu + 23.9)',
        'FNUERR_OPT_RED':    '[microJy] flux error in red image',
        'FNUERRTOT_OPT_RED': '[microJy] flux total error (incl. ZP error) in red image',
        'MAG_APER_RED':      '[mag] aperture AB mag within radius x FWHM in red image',
        'MAGERR_APER_RED':   '[mag] aperture AB mag error within radius x FWHM in red image',
        'MAGERRTOT_APER_RED':'[mag] aperture AB mag total error (incl. ZP error) within radius x FWHM in red image',
        'SNR_APER_RED':      'aperture signal-to-noise ratio within radius x FWHM in red image',
        'FNU_APER_RED':      '[microJy] aperture flux within radius x FWHM in red image',
        'FNUERR_APER_RED':   '[microJy] aperture flux error within radius x FWHM in red image',
        'FNUERRTOT_APER_RED':'[microJy] aperture flux total error (incl. ZP error) within radius x FWHM in red image',
        #
        'MAG_ZOGY':          '[mag] transient AB magnitude',
        'MAGERR_ZOGY':       '[mag] transient AB magnitude error',
        'MAGERRTOT_ZOGY':    '[mag] transient AB magnitude total error (incl. ZP error)',
        'SNR_ZOGY':          'transient signal-to-noise ratio',
        'LIMMAG_ZOGY':       '[mag] transient limiting AB magnitude at input nsigma significance',
        'FNU_ZOGY':          '[microJy] transient flux in red image (mag = -2.5 log10 fnu + 23.9)',
        'FNUERR_ZOGY':       '[microJy] transient flux error',
        'FNUERRTOT_ZOGY':    '[microJy] transient flux total error  (incl. ZP error)',
        #
        'X_POS_REF':         '[pix] x pixel coordinate corresponding to input RA/DEC in ref image',
        'Y_POS_REF':         '[pix] y pixel coordinate corresponding to input RA/DEC in ref image',
        'FLAGS_MASK_REF':    'OR-combined flagged pixels within 2xFWHM of coords in ref image',
        'FLAGS_OPT_REF':     'Optimal photometry flags in ref image',
        'BACKGROUND_REF':    '[e-] sky background estimated from sky annulus in ref image',
        'MAG_OPT_REF':       '[mag] optimal AB magnitude in ref image',
        'MAGERR_OPT_REF':    '[mag] optimal AB magnitude error in ref image',
        'MAGERRTOT_OPT_REF': '[mag] optimal AB magnitude total error (incl. ZP error) in ref image',
        'SNR_OPT_REF':       'signal-to-noise ratio in ref image',
        'LIMMAG_OPT_REF':    '[mag] limiting AB magnitude at nsigma significance in ref image',
        'FNU_OPT_REF':       '[microJy] flux in ref image (AB mag = -2.5 log10 fnu + 23.9)',
        'FNUERR_OPT_REF':    '[microJy] flux error in ref image',
        'FNUERRTOT_OPT_REF': '[microJy] flux total error (incl. ZP error) in ref image',
        'MAG_APER_REF':      '[mag] aperture AB mag within radius x FWHM in ref image',
        'MAGERR_APER_REF':   '[mag] aperture AB mag error within radius x FWHM in ref image',
        'MAGERRTOT_APER_REF':'[mag] aperture AB mag total error (incl. ZP error) within radius x FWHM in ref image',
        'SNR_APER_REF':      'aperture signal-to-noise ratio within radius x FWHM in ref image',
        'FNU_APER_REF':      '[microJy] aperture flux within radius x FWHM in ref image',
        'FNUERR_APER_REF':   '[microJy] aperture flux error within radius x FWHM in ref image',
        'FNUERRTOT_APER_REF':'[microJy] aperture flux total error (incl. ZP error) within radius x FWHM in ref image',
        #
        'THUMBNAIL_RED':     'square thumbnail of the red image centered at input coords',
        'THUMBNAIL_REF':     'square thumbnail of the ref image centered at input coords',
        'THUMBNAIL_D':       'square thumbnail of the difference image centered at input coords',
        'THUMBNAIL_SCORR':   'square thumbnail of the significance image centered at input coords',
        #
        'MAG_ZOGY_PLUSREF':       '[mag] sum of ZOGY and ref image magnitude',
        'MAGERR_ZOGY_PLUSREF':    '[mag] sum of ZOGY and ref image magnitude errors',
        'MAGERRTOT_ZOGY_PLUSREF': '[mag] sum of ZOGY and ref image magnitude total errors (incl. ZP errors)',
        'SNR_ZOGY_PLUSREF':       'sum of ZOGY and ref image signal-to-noise ratio',
        'FNU_ZOGY_PLUSREF':       '[microJy] sum of ZOGY and ref image fluxes',
        'FNUERR_ZOGY_PLUSREF':    '[microJy] sum of ZOGY and ref image flux errors',
        'FNUERRTOT_ZOGY_PLUSREF': '[microJy] sum of ZOGY and ref image flux total errors (incl. ZP errors)',
        #
        'TQC-FLAG':          'transient QC flag (green|yellow|orange|red)',
        #
        # include PC-ZPERR manually in case it is not in the header
        # (initial reduction did not include this keyword)
        'PC-ZPERR':          '[mag] weighted error zeropoint',
    }


    # loop keywords of input header, where the column names are in the
    # TTYPE[i+1] keywords and the descriptions are in TCOMM[i+1]
    for i in range(header['TFIELDS']):
        key = header['TTYPE{}'.format(i+1)]
        # check if key is in the input keys2add
        if key in keys2add:
            # check if relevant TCOMM is in header
            key_descr = 'TCOMM{}'.format(i+1)
            if key_descr in header:
                # if so, add its description to col_descr
                descr = header[key_descr]
                col_descr[key] = descr
            else:
                log.warning ('{} description of column/key {} not available'
                             .format(key_descr, key))


    return col_descr


################################################################################

# from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool (v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


################################################################################

def str2None (value):
    if value == 'None':
        return None
    else:
        return value


################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform (transient) forced '
                                     'photometry on MeerLICHT/BlackGEM data')

    parser.add_argument('radecs', type=str,
                        help='comma-separated list of RA,DEC coordinates '
                        '(ra1,dec1,ra2,dec2,...) or the name of a file with '
                        'format [radecs_file_format] containing the RA and DEC '
                        'coordinates with the column names [ra_col] and '
                        '[dec_col] and optionally also a date of observation '
                        'with the column name [date_col] and astropy Time format '
                        '[date_format] and also optional proper motion columns '
                        'with column names [pmra_col] and [pmdec_col] at epoch '
                        '[pm_epoch]; RA and DEC can either be both in decimal '
                        'degrees or in colon-separated sexagesimal hours and '
                        'degrees for RA and DEC, respectively')

    parser.add_argument('file_out', type=str,
                        help='output filename with resulting magnitudes, where '
                        'the file type is based on the extension; if not '
                        'recognized by astropy.Table, it is forced to be a fits '
                        'file')

    parser.add_argument('--radecs_file_format', type=str, default=None,
                        help='astropy file format of [radecs] in case it points '
                        'to a file; (e)csv and fits files, a.o., are recognized '
                        'through their extension by astropy.Table; default=None')

    parser.add_argument('--ra_col', type=str, default='RA',
                        help='name of input RA column in [radecs]; default=RA')

    parser.add_argument('--dec_col', type=str, default='DEC',
                        help='name of input DEC column in [radecs]; default=DEC')

    parser.add_argument('--date_col', type=str, default='DATE-OBS',
                        help='name of optional input date of observation in '
                        '[radecs]; only needed in case you want to limit the '
                        'images to be observed within a time window with size '
                        '[dtime_max] around this input date; default=DATE-OBS')

    parser.add_argument('--date_format', type=str, default='isot',
                        help='astropy.time.Time format of [date_col]; '
                        'default=isot')

    parser.add_argument('--pmra_col', type=str, default='PMRA',
                        help='name of input RA proper motion column (unit: '
                        'mas/yr) in [radecs]; default=PMRA')

    parser.add_argument('--pmdec_col', type=str, default='PMDEC',
                        help='name of input DEC proper motion column (unit: '
                        'mas/yr) in [radecs]; default=PMDEC')

    parser.add_argument('--pm_epoch', type=float, default=2016.0,
                        help='proper motion reference epoch; default=2016.0 '
                        '(Gaia DR3)')

    parser.add_argument('--input_cols2copy', type=str, default=None,
                        help='comma-separated list of input column names to '
                        'add to the output fits table (optional); the columns '
                        '[ra_col], [dec_col] and [date_col] - the latter only '
                        'if it is present in [radecs] - are copied by default; '
                        'default=None')

    parser.add_argument('--specific_files', type=str, default=None,
                        help='comma-separated list with the filenames of the '
                        'reduced images to be used or the name of an ASCII file '
                        'listing the filenames in the first column without any '
                        'header/column name info; if left at the default of '
                        'None, the routine will use the header tables '
                        'containing all available images for ML or BG, defined '
                        'by parameter [telescope], to select the relevant '
                        'images based on the input coordinates [radecs] and '
                        'other possible constraints such as [filters], '
                        '[date_start] and [date_end]; default=None')

    parser.add_argument('--telescope', type=str, default='BG',
                        choices=['ML1', 'BG2', 'BG3', 'BG4', 'BG'],
                        help='telescope name (ML1, BG2, BG3, BG4 or BG); if set '
                        'to BG, files from any BG present in the header tables '
                        'will be used; this parameter is only relevant if '
                        '[specific_files] is left at its default None; '
                        'default=\'BG\'')

    parser.add_argument('--filters', type=str, default='ugqriz',
                        help='consider images in these filters only; '
                        'default=ugqriz')

    parser.add_argument('--date_start', type=str, default=None,
                        help='starting UTC date of observation in format '
                        '[date_format] for measurements to consider; '
                        'default=None')

    parser.add_argument('--date_end', type=str, default=None,
                        help='ending UTC date of observation in format '
                        '[date_format] for measurements to consider; '
                        'default=None')

    parser.add_argument('--trans', type=str2bool, default=True,
                        help='extract transient magnitudes?; default=True')

    parser.add_argument('--ref', type=str2bool, default=True,
                        help='extract reference magnitudes? N.B.: only works '
                        'for MeerLICHT/BlackGEM images!; default=True')

    parser.add_argument('--fullsource', type=str2bool, default=False,
                        help='extract full-source magnitudes?; default=False')

    parser.add_argument('--nsigma', type=int, default=3,
                        help='significance threshold for a detection; default=3')

    parser.add_argument('--apphot_radii', type=str2None, default='0.66,1.5,5',
                        help='radii in units of the image FWHM at which to '
                        'extract aperture photometry; default=0.66,1.5,5, '
                        'if set to None, no aperture photometry is performed')

    parser.add_argument('--bkg_global', type=str2bool, default=False,
                        help='for full-source and ref cases: use global '
                        'background estimate (T) or estimate local background '
                        'from annulus around the coordinates (F); default=False')

    parser.add_argument('--bkg_local_radii', type=str2None, default='5,7',
                        help='inner and outer radii in units of the image FWHM '
                        'of the sky annulus used to determine the aperture '
                        'photometry local background (sigma-clipped median); '
                        'default=5,7; if set to None, no local background is '
                        'subtracted (this also depends on [bkg_global], i.e. if '
                        'these radii are defined but [bkg_global]=True, no '
                        'local background is subtracted either')

    parser.add_argument('--bkg_local_objmask', type=str2bool, default=True,
                        help='discard pixels in the local background annulus '
                        'affected by objects (stars or galaxies, as detected by '
                        'source extractor) to improve the local background '
                        'determination; if set to False, all pixels that have '
                        'not been identified as bad/saturated/etc. in the input '
                        'mask will be used; default=True')

    parser.add_argument('--bkg_local_limfrac', type=float, default=0.5,
                        help='if more than this fraction of the background '
                        'annulus is masked, by objects and/or pixels identified '
                        'as bad/saturated/etc. in the input mask, the global '
                        'background value is adopted instead of the local one; '
                        'default=0.5')

    parser.add_argument('--include_fluxes', type=str2bool, default=True,
                        help='besides the optimal/aperture magnitudes, also '
                        'include the corresponding fluxes and their errors, '
                        'in units of microJy, in the output table; default=True')

    parser.add_argument('--thumbnails', type=str2bool, default=False,
                        help='extract thumbnail images around input coordinates? '
                        'The thumbnail images that are extracted depend on the '
                        'input parameters [trans], [ref] and [fullsource]:'
                        'reduced image if [fullsource] is True; '
                        'reference image if [ref] is True; '
                        'difference and significance images if [trans] is True; '
                        'default=False')

    parser.add_argument('--size_thumbnails', type=int, default=100,
                        help='size of square thumbnail images in pixels; '
                        'default=100')

    parser.add_argument('--dtime_max', type=float, default=1,
                        help='[hr] maximum time difference between the input '
                        'date of observation in [radecs] and the filename date '
                        'of observation; [dtime_max] is not used if input date '
                        'of observation is not provided; default=1')

    parser.add_argument('--nepochs_min', type=int, default=1,
                        help='minimum number of epochs required for a set of '
                        'coordinates to be processed and feature in the output '
                        'table; default=1')

    par_default = ('MJD-OBS,OBJECT,FILTER,EXPTIME,S-SEEING,AIRMASS,PC-ZP,'
                   'PC-ZPSTD,PC-ZPERR,QC-FLAG')
    parser.add_argument('--keys2add', type=str, default=par_default,
                        help='header keyword values to add to output '
                        'table; default={}'.format(par_default))

    par_default = 'float,U5,U1,float32,float32,float32,float32,float32,float32,U6'
    parser.add_argument('--keys2add_dtypes', type=str, default=par_default,
                        help='corresponding header keyword dtypes; default={}'
                        .format(par_default))

    parser.add_argument('--fov_deg', type=float, default=1.655,
                        help='[deg] instrument field-of-view (FOV); '
                        'default=1.655 for MeerLICHT/BlackGEM')

    parser.add_argument('--ncpus', type=int, default=None,
                        help='number of CPUs to use; if None, the number of '
                        'CPUs available as defined by environment variables '
                        'SLURM_CPUS_PER_TASK or OMP_NUM_THREADS will be used; '
                        'default=None')

    parser.add_argument('--logfile', type=str, default=None,
                        help='if name is provided, an output logfile is created; '
                        'default=None')

    parser.add_argument('--proc_env', choices=['test', 'staging', 'production'],
                        default='production',
                        help='processing environment (test, staging or '
                        'production) used for BlackGEM; this determines the '
                        'bucket name that files will be read from; default='
                        'production, referring to the bucket gs://blackgem-red; '
                        'test will refer to gs://blackgem-test-env/blackgem-red,'
                        ' and staging to gs://blackgem-staging-env/blackgem-red')


    args = parser.parse_args()
    tel = args.telescope


    # for timing
    t0 = time.time()


    # create logfile
    if args.logfile is not None:

        # since logfile is defined, change StreamHandler loglevel to
        # ERROR so that not too much info is sent to stdout
        if False:
            for handler in log.handlers[:]:
                if 'Stream' in str(handler):
                    handler.setLevel(logging.WARNING)
                    #handler.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(args.logfile, 'w')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel('INFO')
        log.addHandler(fileHandler)
        log.info ('logfile created: {}'.format(args.logfile))


    # define number of CPUs to use [ncpus]; if input parameter [npcus]
    # is defined, that value is used. If not and if the ilifu cluster
    # environment variable SLURM_CPUS_PER_TASK is set, either through
    # the --cpus-per-task option in the sbatch script, or when opening
    # an interactive node with multiple processors, that value is
    # adopted. If not, the environment variable OMP_NUM_THREADS is
    # looked for and used if defined.  If none of the above are
    # defined, npcus=1 is used.
    slurm_ncpus = os.environ.get('SLURM_CPUS_PER_TASK')
    omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    if args.ncpus is not None:
        ncpus = args.ncpus
        if slurm_ncpus is not None and ncpus > int(slurm_ncpus):
            log.warning ('number of CPUs defined ({}) is larger than the number '
                         'available ({})'.format(ncpus, slurm_ncpus))
        elif omp_num_threads is not None and ncpus > int(omp_num_threads):
            log.warning ('number of CPUs defined ({}) is larger than the number '
                         'available ({})'.format(ncpus, omp_num_threads))
    else:
        if slurm_ncpus is not None:
            ncpus = int(slurm_ncpus)
        elif omp_num_threads is not None:
            ncpus = int(omp_num_threads)
        else:
            ncpus = 1


    log.info ('number of CPUs used: {}'.format(ncpus))


    # infer RAs and DECs to go through from [args.radecs]
    # ---------------------------------------------------
    mjds_in = None
    if zogy.isfile (args.radecs):

        # if [args.radecs] is a file, read it into a table
        if args.radecs_file_format is not None:
            table_in = Table.read(args.radecs, format=args.radecs_file_format)
        else:
            table_in = Table.read(args.radecs)

        colnames = table_in.colnames
        log.info ('input table column names: {}'.format(colnames))

        if len(table_in) == 0:
            log.error ('no input coordinates found; exiting')
            raise SystemExit
        else:
            log.info ('{} lines in input file {}'
                      .format(len(table_in), args.radecs))


        # add column identifying (line) number of input coordinate;
        # this needs to be done here before any potential masking is
        # done with e.g. mask_dtime below
        table_in['NUMBER_IN'] = np.arange(1,len(table_in)+1, dtype='int32')



        # convert column [date_col] to list mjds_in
        if args.date_col is not None and args.date_col in colnames:

            # if format is in mjd, use the args.date_col directly
            if args.date_format == 'mjd':

                # rename it to MJD_IN as that name is used inside;
                # at the end it will be renamed to args.date_col
                table_in[args.date_col].name = 'MJD_IN'
                mjds_in = table_in['MJD_IN']

            else:

                # convert input dates to mjds
                dates_in = list(table_in[args.date_col])
                mjds_in = Time(dates_in, format=args.date_format).mjd

                # add these to table
                table_in['MJD_IN'] = mjds_in


            # filter table_in by comparing mjds_in with
            # args.date_start and args.date_end
            mask_dtime = get_mjd_mask (mjds_in, args.date_start, args.date_end,
                                       args.date_format)
            table_in = table_in[mask_dtime]
            log.info ('{} lines in input file {} after filtering on input '
                      '[date_start] and [date_end]'
                      .format(args.radecs, len(table_in)))

        else:
            # if not provided, set mjds_in to None
            mjds_in = None



        # check if [args.ra_col] exists
        if args.ra_col in colnames:
            # convert sexagesimal to decimal degrees
            ras_in = list(table_in[args.ra_col])
            if not isinstance(ras_in[0], float) and ':' in ras_in[0]:
                ras_in = Angle(ras_in, unit=u.hour).degree
                table_in[args.ra_col] = ras_in
        else:
            log.critical ('column {} not present in {}; exiting'
                          .format(args.ra_col, args.radecs))
            raise SystemExit

        # same for declination
        if args.dec_col in colnames:
            decs_in = list(table_in[args.dec_col])
            if not isinstance(decs_in[0], float) and ':' in decs_in[0]:
                decs_in = Angle(decs_in, unit=u.deg).degree
                table_in[args.dec_col] = decs_in
        else:
            log.critical ('column {} not present in {}; exiting'
                          .format(args.dec_col, args.radecs))
            raise SystemExit


    else:

        log.info ('input parameter [radecs] {} is apparently not a file; '
                  'assuming it is a comma-separated list of RA,DEC coordinates'
                  .format(args.radecs))

        # split input [radecs] into list of strings ['ra1', 'dec1', ...]
        radecs_list0 = re.sub(r'\(|\)|\[|\]|\{|\}', '', args.radecs)
        radecs_list = re.sub(';', ',', radecs_list0).split(',')

        # remove potential empty entries and check for an even number
        remove_empty (radecs_list)
        if len(radecs_list) % 2 != 0:
            log.critical ('number of coordinates in [radecs] is not even; '
                          'exiting')
            raise SystemExit

        ras_in = []
        decs_in = []
        mjds_in = None
        log.info('radecs_list: {}'.format(radecs_list))
        for i, s in enumerate(radecs_list):
            if i % 2 == 0:
                # RAs:
                if ':' in s:
                    ras_in.append(Angle(s, unit=u.hour).degree)
                else:
                    ras_in.append(float(s))

            else:
                # DECs
                if ':' in s:
                    decs_in.append(Angle(s, unit=u.deg).degree)
                else:
                    decs_in.append(float(s))


        # create table as if input file was provided with only RAs and DECs
        table_in = Table()
        table_in[args.ra_col] = ras_in
        table_in[args.dec_col] = decs_in


        # add column identifying (line) number of input coordinate, for
        # easy mapping between input and output tables
        table_in['NUMBER_IN'] = np.arange(1,len(table_in)+1, dtype='int32')



    # rename input columns [args.ra_col] and [args.dec_col]
    table_in[args.ra_col].name = 'RA_IN'
    table_in[args.dec_col].name = 'DEC_IN'


    # rename proper motion columns if they are present
    colnames = table_in.colnames
    if args.pmra_col in colnames and args.pmdec_col in colnames:
        table_in[args.pmra_col].name = 'PMRA_IN'
        table_in[args.pmdec_col].name = 'PMDEC_IN'


    # only keep columns that are needed; note that MJD_IN, PMRA_IN and
    # PMDEC_IN may not exist, but that is checked a bit further below
    cols2keep = ['RA_IN', 'DEC_IN', 'MJD_IN', 'NUMBER_IN', 'PMRA_IN', 'PMDEC_IN']

    # add the ones from args.input_cols2copy
    if args.input_cols2copy is not None:
        cols2keep += args.input_cols2copy.split(',')

    # could in principle do the following in a single line with
    # Table.keep_columns(cols2keep), but need to check if MJD_IN,
    # PMRA_IN and PMDEC_IN are in table_in - which need not be the
    # case - and also it would lead to an issue if names in
    # [args.input_cols2copy] are mispelled
    for colname in table_in.colnames:
        if colname not in cols2keep:
            del table_in[colname]


    # unique elements based on RA_IN and DEC_IN
    table_in = unique(table_in, keys=['RA_IN', 'DEC_IN'])
    log.info('{} unique RADECs (and MJDs) in input list/file [radecs]'
             .format(len(table_in)))


    # check if all filters are needed
    filts_all = np.all([filt in args.filters for filt in 'ugqriz'])



    # read list of tables with filenames and relevant header keywords
    # ---------------------------------------------------------------
    # previously also included possiblity to prepare this table, but
    # not needed anymore as header tables are ready to be used


    if args.specific_files is not None:

        # create header table from input [specific_files]
        if zogy.isfile(args.specific_files):

            # if the input is a single file, it could be a single
            # image or an ASCII file containing multiple files; to
            # test between the two, try to read the file as a table,
            # which will cause an exception if it is an image
            try:

                # read ASCII file
                table_files = Table.read(args.specific_files, format='ascii',
                                         data_start=0, names=['FILENAME'])
                files = list(table_files['FILENAME'])

            except:

                # apparently the input is not an ASCII file, so assume
                # it is a single image and put it into a list
                files = args.specific_files.split(',')

        else:

            # if not a file, it should be a string with the filenames
            # separated by commas
            files = args.specific_files.split(',')



        # convert files to table_hdr with a handful of crucial
        # keywords
        len_max = np.max([len(f) for f in files])
        objects, mjds_obs, filts, ras_cntr, decs_cntr = get_headkeys(files)
        table_hdr = Table(
            [files, objects, mjds_obs, filts, ras_cntr, decs_cntr],
            names=['FILENAME','OBJECT','MJD-OBS','FILTER','RA-CNTR','DEC-CNTR'],
            dtype=['U{}'.format(len_max), 'int16', float, 'U1', float, float])


        # if all files are ML1 files, force tel to be 'ML1'; the
        # default input telescope parameter is 'BG', but if it is
        # forgotten for ML1 data, the routine will otherwise attempt
        # to read Google Cloud files, leading to a missing Google
        # credentials error, which is unexpected when working with ML1
        # files
        if np.all([f.split('/')[-1][0:3]=='ML1' for f in files]):
            tel = 'ML1'



    # refer to the existing header tables for both ML and BG; this
    # needs to be done even when specific_files are specified, to be
    # able to infer the table description used futher down below
    if tel == 'ML1':

        fits_hdrtable_list = ['/idia/projects/meerlicht/Headers/'
                              'ML1_headers_cat.fits']
    else:

        # for BG, loop telescopes and add header table if needed;
        # exact location depends on input parameter proc_env
        bucket_env = {'test': 'blackgem-test-env/',
                      'staging': 'blackgem-staging-env/',
                      'production': ''}
        fits_hdrtable_list = []
        for tel_tmp in ['BG2', 'BG3', 'BG4']:
            if tel in tel_tmp:
                fits_hdrtable_list.append(
                    'gs://{}blackgem-hdrtables/{}/{}_headers_cat.fits'
                    .format(bucket_env[args.proc_env], tel_tmp, tel_tmp))



    # if specific_files were not specified, read header tables into
    # single table_hdr
    if args.specific_files is None:

        # read header fits files into table
        for it, fits_table in enumerate(fits_hdrtable_list):

            try:

                if zogy.isfile(fits_table):

                    log.info ('reading header table: {}'.format(fits_table))
                    table_tmp = Table.read(fits_table)
                    if it==0:
                        table_hdr = table_tmp
                    else:
                        # stack tables
                        table_hdr = vstack([table_hdr, table_tmp])

                else:
                    log.warning ('{} not found'.format(fits_table))

            except Exception as e:

                if 'Credentials' in str(e):
                    log.exception(e)
                    log.error ('Google Cloud credentials error; if working '
                               'with MeerLICHT data, make sure to specify '
                               'input parameter telescope to be \'ML1\'')
                    logging.shutdown()
                    raise SystemExit

                else:
                    log.error ('exception raised when attempting to read header '
                               'table {}: {}'.format(fits_table, e))



    # check if table contains any entries
    if len(table_hdr)==0:
        log.error ('no input files; exiting')
        logging.shutdown()
        raise SystemExit
    else:
        log.info ('{} input files'.format(len(table_hdr)))



    # infer list of filenames to consider
    # -----------------------------------
    radecs_cntr = None

    # get central coordinates of filenames
    colnames = table_hdr.colnames
    if 'RA-CNTR' in colnames and 'DEC-CNTR' in colnames:

        # mask with files with a WCS solution
        mask_WCS = (np.isfinite(table_hdr['RA-CNTR']) &
                    np.isfinite(table_hdr['DEC-CNTR']) &
                    (table_hdr['RA-CNTR'] > 0) &
                    (table_hdr['RA-CNTR'] < 360) &
                    (np.abs(table_hdr['DEC-CNTR']) < 90))

        log.info ('{} filename(s) with valid CNTR coordinates'
                  .format(np.sum(mask_WCS)))
        table_hdr = table_hdr[mask_WCS]


        # define list of (ra_cntr,dec_cntr) tuples to be used in
        # function [index_images]
        radecs_cntr = np.array(list(zip(table_hdr['RA-CNTR'],
                                        table_hdr['DEC-CNTR'])))

    else:
        log.error ('RA-CNTR and/or DEC-CNTR column not in header table; exiting')
        logging.shutdown()
        raise SystemExit


    # define list of filenames
    filenames = list(table_hdr['FILENAME'])


    # define objects and mjds_obs arrays
    if 'OBJECT' in colnames and 'MJD-OBS' in colnames:
        objects = np.array(table_hdr['OBJECT']).astype(int)
        mjds_obs = np.array(table_hdr['MJD-OBS'])


    # and filter array if needed
    if not filts_all:
        if 'FILTER' in colnames:
            filts = np.array(table_hdr['FILTER']).astype(str)



    # filtering of filenames
    # ----------------------
    nfiles = len(filenames)
    mask_keep = np.ones(nfiles, dtype=bool)


    # filter by input args.date_start and args.date_end
    if args.date_start is not None or args.date_end is not None:
        mask_keep &= get_mjd_mask (mjds_obs, args.date_start, args.date_end,
                                   args.date_format)


    # filter by minimum and maximum [mjds_in] if it is not None
    if mjds_in is not None:
        mjd_start = min(mjds_in)-args.dtime_max
        mjd_end = max(mjds_in)+args.dtime_max
        mask_keep &= get_mjd_mask (mjds_obs, mjd_start, mjd_end, 'mjd')


    # filter by filters specified in args.filters
    if not filts_all:
        mask_filts = np.zeros(nfiles, dtype=bool)
        for filt in args.filters:
            mask_filts |= (filts == filt)

        # keep the relevant filters
        mask_keep &= mask_filts


    # update filenames, mjds_obs, objects and radecs_cntr
    if np.any(~mask_keep):
        filenames = list(np.array(filenames)[mask_keep].astype(str))
        nfiles = len(filenames)
        mjds_obs = mjds_obs[mask_keep]
        objects = objects[mask_keep]
        if radecs_cntr is not None:
            radecs_cntr = radecs_cntr[mask_keep]

        log.info ('{} filename(s) left to search through after filtering '
                  'on input [date_start] and [date_end] and/or on potential '
                  'input dates provided in input file [radecs] and/or on '
                  'filters provided in [filters]'.format(nfiles))



    # determine basenames now that [mask_keep] was applied to filenames
    basenames = ['{}_red'.format(fn.split('_red')[0]) for fn in filenames]


    # if [radecs_cntr] is not defined yet because the default fits
    # header was not used as args.filenames, exit
    if radecs_cntr is None:
        log.error ('radecs_cntr is None; exiting')
        raise SystemExit


    # limit the input coordinates to be within the range of observed
    # central image DECs plus a band of 1 degree
    dec_cntr_min = np.min(radecs_cntr[:,1])
    dec_cntr_max = np.max(radecs_cntr[:,1])
    mask_coords = ((table_in['DEC_IN'] >= dec_cntr_min-1) &
                   (table_in['DEC_IN'] <= dec_cntr_max+1))

    if np.any(~mask_coords):
        table_in = table_in[mask_coords]
        log.info ('{} entries in [table_in] after filtering on range of '
                  'observed declinations'.format(len(table_in)))



    # create {table_index1: image_list1, table_index2: imagelist2
    # dictionary using multiprocessing of function [index_images];
    # to optimize the multiprocessing, split the input list of table
    # indices into [ncpus] lists so that each worker processes a
    # number of indices at a time, rather than a single one
    table_indices = np.arange(len(table_in))
    table_indices_cpu = []
    index = np.linspace(0,len(table_in),num=ncpus+1).astype(int)
    for i in range(ncpus):
        table_indices_cpu.append(table_indices[index[i]:index[i+1]])


    # run multiprocessing
    results = zogy.pool_func (index_images, table_indices_cpu, table_in,
                              mjds_obs, args.dtime_max, basenames, radecs_cntr,
                              args.fov_deg, args.pm_epoch, nproc=ncpus)

    # convert results, which is a list of dictionaries, to a single
    # dictionary
    log.info ('merging list of dictionaries from [index_images] into a single '
              'dictionary')
    index_images_dict = {}
    for i, d in enumerate(results):
        index_images_dict.update(d)


    if False:
        for k in index_images_dict.keys():
            log.info ('k: {}, len(index_images_dict[k]): {}'
                      .format(k, len(index_images_dict[k])))
            # write file with list of images
            t_tmp = Table()
            t_tmp['filenames'] = index_images_dict[k]
            ascii.write (t_tmp, 'filenames_{}.dat'.format(k),
                         format='fixed_width_no_header',
                         delimiter=' ', overwrite=True)

            raise SystemExit



    if False:
        for k in index_images_dict.keys():
            log.info ('index_images_dict key: {}, value: {}'
                      .format(k, index_images_dict[k]))



    # could limit [table_in] to indices present in [index_images_dict],
    # but then would also have to update the keys
    if False:
        table_in = table_in[np.array(list(index_images_dict.keys()))]
        # need to update the indices in index_images_dict accordingly
        index_images_dict_new = {}
        for i_new, i_old in zip(range(len(table_in)), index_images_dict.keys()):
            index_images_dict_new[i_new] = index_images_dict[i_old]

        # replace [index_images_dict]
        del index_images_dict
        index_images_dict = index_images_dict_new


    # convert input dictionary {index1: image_list1, index2:
    # image_list2, ...}  to {image1: index_list1, image2: index_list2,
    # ...}, which is the input form to [force_phot]; this way if there
    # are many sources on the same images, it will be faster to
    # process
    log.info ('converting dictionary from '
              '{index1: image_list1, index2: image_list2, ...} '
              'to {image1: index_list1, image2: index_list2, ...}')
    image_indices_dict = {}
    for k, vlist in index_images_dict.items():
        for v in vlist:
            image_indices_dict[v] = image_indices_dict.get(v, []) + [k]


    # timing so far
    log.info ('time spent to select relevant images: {:.1f}s'
              .format(time.time()-t0))
    t1 = time.time()


    if False:
        # now that relevant images have been selected, save these to an
        # ASCII file that contains the date
        matching_images_list = ['{}_red.fits.fz'.format(fn.split('_red')[0])
                                for fn in image_indices_dict]
        table_images = Table([matching_images_list])
        name_tmp = 'matching_images_{}.txt'.format(Time.now().isot.split('T')[0])
        table_images.write(name_tmp, format='ascii.no_header', overwrite=True)


    # could cut up dictionary in pieces if it turns out to be
    # too large, and loop over [force_phot] various times


    # convert input keys2add and corresponding types to lists
    if args.keys2add is not None:
        keys2add = args.keys2add.upper().split(',')
    else:
        keys2add = None

    if args.keys2add_dtypes is not None:
        keys2add_dtypes = args.keys2add_dtypes.split(',')
    else:
        keys2add_dtypes = None

    verify_lengths (keys2add, keys2add_dtypes)


    # change input [apphot_radii] from string list of floats/ints
    if args.apphot_radii is not None:
        apphot_radii = [float(rad) if '.' in rad else int(rad)
                        for rad in args.apphot_radii.split(',')]
        log.info ('apphot_radii: {}'.format(apphot_radii))
    else:
        # or from 'None' to None
        apphot_radii = None


    # change input [bkg_local_radii] from string to list of floats/ints
    if args.bkg_local_radii is not None:
        bkg_radii = [float(rad) if '.' in rad else int(rad)
                     for rad in args.bkg_local_radii.split(',')]
        if len(bkg_radii) != 2:
            log.error ('2 background radii are required, while {} have been '
                       'specified: {}; exiting'
                       .format(len(bkg_radii), bkg_radii))
            raise SystemExit
        else:
            log.info ('bkg_radii: {}'.format(bkg_radii))
    else:
        # or from 'None' to None
        bkg_radii = None



    # prepare list of masks
    mask_list = ['{}_mask.fits.fz'.format(fn.split('_red')[0])
                 for fn in image_indices_dict]


    log.info ('table_in.colnames: {}'.format(table_in.colnames))
    log.info ('keys2add: {}'.format(keys2add))

    # call [force_phot]
    table_out = force_phot (
        table_in, image_indices_dict, mask_list=mask_list, trans=args.trans,
        ref=args.ref, fullsource=args.fullsource, nsigma=args.nsigma,
        bkg_global=args.bkg_global, apphot_radii=apphot_radii,
        bkg_radii=bkg_radii, bkg_objmask=args.bkg_local_objmask,
        bkg_limfrac=args.bkg_local_limfrac, pm_epoch=args.pm_epoch,
        keys2add=keys2add, keys2add_dtypes=keys2add_dtypes,
        thumbnails=args.thumbnails, size_thumbnails=args.size_thumbnails,
        tel=tel, ncpus=ncpus)



    # copy columns from the input to the output table; even if
    # [args.input_cols2copy] was not defined but [args.date_col] is
    # defined, let's copy over at least [args.date_col]
    # N.B.: this is now done by providing an input table to
    # [force_phot] that already includes these columns, so do not
    # have to be added here anymore
    if table_out is not None:

        # delete flux columns if not wanted
        if not args.include_fluxes:
            log.warning ('input parameter include_fluxes is set to False')
            for col in table_out.colnames:
                if 'FNU' in col:
                    log.warning ('deleting {}'.format(col))
                    del table_out[col]


        # rename columns RA and DEC to the input column names; if
        # the input columns were in sexagesimal notation, these
        # will be decimal degrees
        table_out['RA_IN'].name = args.ra_col
        table_out['DEC_IN'].name = args.dec_col


        # if [date_col] was provided and the MJD-OBS column is present
        # in the output table, the delta time between it and the image
        # date of observations can be determined
        colnames = table_out.colnames
        if (args.date_col is not None and 'MJD-IN' in colnames and
            'MJD-OBS' in colnames):

            mjds_in = table_out['MJD_IN'].value
            dtime_days = np.abs(mjds_in - table_out['MJD-OBS'])
            table_out.add_column(dtime_days, name='DELTA_MJD',
                                 index=colnames.index('MJD-OBS')+1)

            # if args.date_format is mjd, then rename 'MJD_IN' back to
            # the original name, unless the original name is the same
            # as MJD-OBS - the MJD of the images
            if args.date_format == 'mjd' and args.date_col != 'MJD-OBS':
                table_out['MJD_IN'].name = args.date_col


        # rename proper motion columns if needed
        if 'PMRA_IN' in colnames and 'PMDEC_IN' in colnames:
            table_out['PMRA_IN'].name = args.pmra_col
            table_out['PMDEC_IN'].name = args.pmdec_col


        # order the output table by original row number
        try:
            log.info ('sorting by NUMBER_IN and FILENAME')
            table_out.sort(['NUMBER_IN','FILENAME'])
        except Exception as e:
            log.warning ('unable to sort table_out due to exception: {}'
                         .format(e))



        # write output table to fits
        log.info ('writing output file {}'.format(args.file_out))
        try:
            # let format be auto-identified through the extension of
            # file_out
            table_out.write(args.file_out, overwrite=True)
        except:
            # otherwise, force output file to be fits
            table_out.write('{}.fits'.format(args.file_out),
                            format='fits', overwrite=True)



        # read header of trans table for description of keywords
        #header_transtable = zogy.read_hdulist(
        #    fits_hdrtable_list[0].replace('_cat.fits', '_trans.fits'),
        #    get_data=False, get_header=True)
        fn_tmp = fits_hdrtable_list[0].replace('_cat.fits', '_trans.fits')
        with fits.open(fn_tmp) as hdulist:
            header_transtable = hdulist[-1].header



        col_descr_dict = create_col_descr (keys2add, header_transtable,
                                           args.ra_col, args.dec_col)
        with fits.open(args.file_out, mode='update') as hdulist:
            for i, col0 in enumerate(table_out.colnames):
                ncol = i + 1
                # edit aperture names
                if 'APER' in col0:
                    col = '_'.join(col0.split('_')[:2])
                    if 'REF' in col0:
                        col += '_REF'
                    if 'RED' in col0:
                        col += '_RED'
                else:
                    col = col0

                # check if col is present in col_descr_dict
                if col in col_descr_dict:
                    descr = col_descr_dict[col]
                else:
                    log.warning ('{} not in column description dictionary'
                                 .format(col))
                    descr = ''

                # update relevant header keyword COMM
                hdulist[-1].header['TCOMM{}'.format(ncol)] = descr


    else:
        log.warning ('empty output table; no output file to create')


    log.info ('time spent in [force_phot]: {:.1f}s'.format(time.time()-t1))
    log.info ('time spent in total:        {:.1f}s'.format(time.time()-t0))


    # list memory used
    zogy.mem_use ('at very end')
