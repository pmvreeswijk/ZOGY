
import os
import argparse
import re
from datetime import datetime
from dateutil.tz import gettz
import itertools
from random import choice
from string import ascii_uppercase
import glob
import sys

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

import zogy
import set_zogy
set_zogy.verbose=False

import numpy as np

import astropy.io.fits as fits
from astropy.coordinates import Angle, SkyOffsetFrame, SkyCoord
from astropy.table import Table, hstack, vstack, unique
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import SigmaClip

from fitsio import FITS

#import sep as SEP
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.aperture import CircularAnnulus, ApertureStats
from photutils.utils import calc_total_error

from multiprocessing import Pool

# since version 0.9.3 (Feb 2023) this module was moved over from
# BlackBOX to ZOGY to be able to perform forced photometry on an input
# (Gaia) catalog inside ZOGY
__version__ = '0.9.4'


################################################################################

def force_phot (table_in, image_indices_dict, mask_list=None, trans=True,
                ref=True, fullsource=False, nsigma=3, apphot_radii=None,
                apphot_sky_inout=None, apphot_att2add=None, pm_epoch=2016.0,
                include_fluxes=False, keys2add=None, keys2add_dtypes=None,
                bkg_global=True, thumbnails=False, size_thumbnails=None,
                tel=None, ncpus=1):


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

    apphot_sky_inout: 2-element list or numpy array (default=None)
                      indicating the inner and outer radius of the sky
                      annulus used for the aperture photometry sky
                      background determination
                 
    apphot_att2add: list of strings, indicating any float or integer
                    attribute of photutils.aperture.ApertureStats

    pm_epoch: float (default=2016.0), proper motion reference epoch;
              if [table_in] includes the columns PMRA_IN and PMDEC_IN
              (units: mas/yr), the proper motion is taken into account
              using this reference epoch

    include_fluxes: boolean (default=False) deciding whether the
                    electron fluxes (e-/s) corresponding to the
                    magnitudes are included in the output table

    keys2add: list of strings (default=None); header keywords that
              will be added as columns to the output table

    keys2add_dtypes: list of dtypes (default=None); corresponding
                     dtypes of the header keywords provided in
                     [keys2add]

    bkg_global: boolean (default=True) determining whether to use the
                global or local background determination in the
                photometry
    
    ncpus: int (default=1); number of processes/tasks to use

    thumbnails: boolean (default=None) determining whether to include
                the thumbnail images of size [size_thumbnails] to the
                output catalog. Which thumbnails are included depends
                on [trans], [ref] and [fullsource]

    tel: str (default=None) indicating telescope (e.g. ML1, BG2)

    size_thumbnails: int (default=100), size in pixels of thumbnails

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
    if tel not in ['ML1', 'BG2', 'BG3', 'BG4']:
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
    names += ['FILENAME', 'X_POS', 'Y_POS']
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


    # add FLAGS_MASK, which is determined irrespective of a match with
    # a full-source catalog source
    if 'FLAGS_MASK' not in names:
        names += ['FLAGS_MASK']
        dtypes += ['uint8']


    # initialize columns to be determined below
    if fullsource:

        # optimal photometry columns
        names_fullsource = ['MAG_OPT', 'MAGERR_OPT', 'SNR_OPT', 'LIMMAG_OPT']
        names += names_fullsource
        dtypes += ['float32', 'float32', 'float32', 'float32']


        # add corresponding fluxes
        if include_fluxes:
            names += ['E_FLUX_OPT', 'E_FLUXERR_OPT']
            dtypes += ['float32', 'float32']


        # add aperture photometry columns if needed
        if apphot_radii is not None:       

            for radius in apphot_radii:
                names += ['MAG_APER_R{}xFWHM'.format(radius),
                          'MAGERR_APER_R{}xFWHM'.format(radius),
                          'SNR_APER_R{}xFWHM'.format(radius)]
                dtypes += ['float32', 'float32', 'float32']


                # add corresponding fluxes
                if include_fluxes:
                    names += ['E_FLUX_APER_R{}xFWHM'.format(radius),
                              'E_FLUXERR_APER_R{}xFWHM'.format(radius)]
                    dtypes += ['float32', 'float32']

                
                # and columns corresponding to the apphot attributes
                if apphot_att2add is not None:
                    for att in apphot_att2add:
                        names += ['{}_APER_R{}xFWHM'.format(att.upper(), radius)]
                        dtypes += ['float32']


        # add thumbnail if relevant
        if thumbnails:
            names += ['THUMBNAIL_RED']
            dtypes += ['float32']



    if trans:

        names_trans = ['MAG_ZOGY', 'MAGERR_ZOGY', 'SNR_ZOGY', 'LIMMAG_ZOGY']
        names += names_trans
        dtypes += ['float32', 'float32', 'float32', 'float32']


        # add corresponding fluxes
        if include_fluxes:
            names += ['E_FLUX_ZOGY', 'E_FLUXERR_ZOGY']
            dtypes += ['float32', 'float32']

        
        # add thumbnails if relevant
        if thumbnails:
            names += ['THUMBNAIL_D', 'THUMBNAIL_SCORR']
            dtypes += ['float32', 'float32']



    if ref:

        # add pixelcoordinates corresponding to input RA/DEC to table
        names += ['X_POS_REF', 'Y_POS_REF']
        dtypes += [float, float]


        # add FLAGS_MASK for the reference image
        if 'FLAGS_MASK_REF' not in names:
            names += ['FLAGS_MASK_REF']
            dtypes += ['uint8']


        # magnitude, snr and limiting magnitude columns
        names_ref = ['MAG_OPT_REF', 'MAGERR_OPT_REF', 'SNR_OPT_REF',
                     'LIMMAG_OPT_REF']
        names += names_ref
        dtypes += ['float32', 'float32', 'float32', 'float32']


        # add corresponding fluxes
        if include_fluxes:
            names += ['E_FLUX_OPT_REF', 'E_FLUXERR_OPT_REF']
            dtypes += ['float32', 'float32']
            
        
        # add aperture photometry columns if needed
        if apphot_radii is not None:

            for radius in apphot_radii:
                names += ['MAG_APER_R{}xFWHM_REF'.format(radius),
                          'MAGERR_APER_R{}xFWHM_REF'.format(radius),
                          'SNR_APER_R{}xFWHM_REF'.format(radius)]
                dtypes += ['float32', 'float32', 'float32']


                # add corresponding fluxes
                if include_fluxes:
                    names += ['E_FLUX_APER_R{}xFWHM_REF'.format(radius),
                              'E_FLUXERR_APER_R{}xFWHM_REF'.format(radius)]
                    dtypes += ['float32', 'float32']


                # and columns corresponding to the apphot attributes
                if apphot_att2add is not None:
                    for att in apphot_att2add:
                        names += ['{}_APER_R{}xFWHM_REF'.format(att.upper(),
                                                                radius)]
                        dtypes += ['float32']


        # in case trans==True, add these ZOGY+REF columns
        if trans:
            names += ['MAG_ZOGY_PLUSREF',
                      'MAGERR_ZOGY_PLUSREF',
                      'SNR_ZOGY_PLUSREF']
            dtypes += ['float32', 'float32', 'float32']


        # add thumbnail if relevant
        if thumbnails:
            names += ['THUMBNAIL_REF']
            dtypes += ['float32']




    # convert [image_indices_dict] to a list of [image, indices]
    # lists, so it is more easily processed by pool_func
    image_indices_list = []
    for k, v in image_indices_dict.items():
        image_indices_list.append([k,v])


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
            apphot_sky_inout, apphot_att2add, pm_epoch, include_fluxes, keys2add,
            add_keys, names, dtypes, bkg_global, thumbnails, size_thumbnails]

    if nimages == 1:
        # for single image, execute [get_rows] without pool_func, and
        # provide ncpus as input parameter, which will multiprocess
        # the function zogy.get_psfoptflux_mp such that the different cpus
        # process different lists of objects on the same image
        table = get_rows (image_indices_list[0], *pars, ncpus=ncpus)
        table_list = [table]


    else:
        # for multiple images, execute [get_rows] with pool_func, such
        # that the different cpus process different images
        table_list = zogy.pool_func (get_rows, image_indices_list, *pars,
                                     nproc=ncpus)


    # remove None entries, e.g. due to coordinates off the field
    log.info ('removing None entries in [table_list]')
    while True:
        try:
            table_list.pop(table_list.index(None))
        except:
            break


    # finished multi-processing filenames
    log.info ('stacking individual tables into output table')
    ntables = len(table_list)
    if ntables > 0:
        # old simple method
        if True:
            table = vstack(table_list)

        else:
            # new method: keep adding tables in pairs until there is a
            # single table left
            while ntables > 1:
                # number of tables left
                ntables = len(table_list)
                log.info ('ntables: {}'.format(ntables))
                # initialize new table_list
                table_list_new = []
                # loop list in steps of 2
                for i in range(0,ntables,2):
                    # make sure i is within range of list index
                    if i+1 <= ntables-1:
                        # if so, stack the two tables
                        table_tmp = vstack([table_list[i],
                                            table_list[i+1]])
                    else:
                        # otherwise, table_tmp is the last table
                        table_tmp = table_list[i]

                    # add table_tmp to new table_list
                    table_list_new.append(table_tmp)

                # after loop is finished, replace the old table_list
                # with the new one
                table_list = table_list_new

            # the single table left is the final table
            table = table_list[0]

    else:
        return None


    # sort in time
    log.info ('sorting by FILENAME')
    index_sort = np.argsort(table['FILENAME'])

    zogy.mem_use ('at end of [force_phot]')

    # return table
    return table[index_sort]


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

    if os.path.exists(filename):
        return filename
    elif os.path.exists('{}.fz'.format(filename)):
        return '{}.fz'.format(filename)
    elif os.path.exists(filename.split('.fz')[0]):
        return filename.split('.fz')[0]
    else:
        return None


################################################################################

def get_rows (image_indices, table_in, trans, ref, fullsource, nsigma,
              apphot_radii, apphot_sky_inout, apphot_att2add, pm_epoch,
              include_fluxes, keys2add, add_keys, names, dtypes, bkg_global,
              thumbnails, size_thumbnails, ncpus=None):


    # extract filenames and table indices from input list
    # [image_indices] consisting of [filename, [index1, index2, ..]]
    # with possibly fits_mask as a 3rd element
    if len(image_indices)==3:
        filename, indices, fits_mask = image_indices
    else:
        filename, indices = image_indices
        fits_mask = None
        

    log.info ('processing {}'.format(filename))


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
    fits_red = add_drop_fz ('{}.fits.fz'.format(basename))
    fits_cat = '{}_cat.fits'.format(basename)
    fits_trans = '{}_trans.fits'.format(basename)
    # try to read transient catalog header, as it is more complete
    # than the full-source catalog header
    if os.path.exists(fits_trans):
        fits2read = fits_trans
    elif os.path.exists(fits_cat):
        fits2read = fits_cat
    elif os.path.exists(fits_red):
        fits2read = fits_red
    else:
        log.warning ('reduced image, full-source and transient catalog all '
                     'do not exist for {}; skipping its extraction'
                     .format(basename))
        return [None]


    # read header
    try:
        log.info ('reading header of {}'.format(fits2read))
        header = FITS(fits2read)[-1].read_header()
    except:
        log.exception ('trouble reading header of {}; skipping its extraction'
                       .format(fits2read))
        return [None]


    # if proper motion need to be corrected for (if [pm_epoch] is not
    # None and [table_in] columns PMRA_IN and PMDEC_IN exist),
    # extract the image date of observation and apply the proper motion
    # correction
    if (pm_epoch is not None and 'PMRA_IN' in colnames and 'PMDEC_IN' in colnames
        and 'DATE-OBS' in header):

        obsdate = header['DATE-OBS']
        table = zogy.apply_gaia_pm (
            table, obsdate, epoch=pm_epoch, return_table=True, ra_col='RA_IN',
            dec_col='DEC_IN', pmra_col='PMRA_IN', pmdec_col='PMDEC_IN')


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
                             .format(key, basename))

            if key=='QC-FLAG' and 'TQC-FLAG' not in keys2add and trans:

                try:
                    table['TQC-FLAG'] = header['TQC-FLAG']
                except:
                    table['TQC-FLAG'] = None
                    log.warning ('keyword TQC-FLAG not in header of {}'
                                 .format(basename))




    # full-source; determining optimal flux
    # -------------------------------------
    if fullsource:              

        # infer full-source magnitudes and S/N
        table = infer_mags (table, basename, fits_mask, nsigma, apphot_radii,
                            apphot_sky_inout, apphot_att2add, include_fluxes,
                            keys2add, add_keys, bkg_global, thumbnails, size_tn,
                            imtype='new', tel=tel, ncpus=ncpus)



    # transient; extracting ZOGY fluxes
    # ---------------------------------
    if trans:

        # infer transient magnitudes and S/N
        table = infer_mags (table, basename, fits_mask, nsigma, apphot_radii,
                            apphot_sky_inout, apphot_att2add, include_fluxes,
                            keys2add, add_keys, bkg_global, thumbnails, size_tn,
                            imtype='trans', tel=tel, ncpus=ncpus)



    # reference; determining optimal fluxes
    # -------------------------------------
    if ref:

        # infer path to ref folder from basename 
        ref_dir = '{}/ref'.format(basename.split('/red/')[0])
        
        # read field ID from header
        obj, filt = header['OBJECT'], header['FILTER']

        # reference image and catalog names including full path
        basename = '{}/{}/{}_{}_red'.format(ref_dir, obj, tel, filt)

        # reference mask image; input parameter fits_mask refers to
        # the reduced image mask
        fits_mask = '{}/{}/{}_{}_mask.fits.fz'.format(ref_dir, obj, tel, filt)


        # infer reference magnitudes and S/N
        table = infer_mags (table, basename, fits_mask, nsigma, apphot_radii,
                            apphot_sky_inout, apphot_att2add, include_fluxes,
                            keys2add, add_keys, bkg_global, thumbnails, size_tn,
                            imtype='ref', tel=tel, ncpus=ncpus)


        # for transients, add any potential source in the reference
        # image to the transient flux and save the result in the
        # column MAG_ZOGY_PLUSREF
        if trans:

            # start off with MAG_ZOGY
            mag_zogy = np.array(table['MAG_ZOGY'])

            # transient flux in arbitrary flux units; mag_zogy is
            # always positive, even in case of a negative transient,
            # i.e. when the flux in the reference image was higher, so
            # include sign of [snr_zogy]
            snr_zogy = np.array(table['SNR_ZOGY'])

            # initially: set insignificant transients to zero, but
            # decided to not do so anymore - just add transient and
            # reference flux irrespective of whether they are
            # significant or not
            # mask_snr_trans = (np.abs(snr_zogy) >= nsigma)
            # mag_zogy[~mask_snr_trans] = 100
            flux_zogy = np.sign(snr_zogy) * 10**(-0.4*mag_zogy)


            # reference flux in arbitrary units (but same as flux_zogy
            # above)
            mag_opt_ref = np.array(table['MAG_OPT_REF'])
            # corresponding S/N
            snr_opt_ref = np.array(table['SNR_OPT_REF'])
            # require reference source to be positive
            mask_snr_ref = (snr_opt_ref > 0)
            # set magnitudes of non-positive sources to negligibly faint
            mag_opt_ref[~mask_snr_ref] = 99
            flux_ref = 10**(-0.4*mag_opt_ref)

            
            # corrected flux and magnitude
            flux_corr = (flux_zogy + flux_ref)
            mag_corr = np.zeros_like(flux_corr, dtype='float32') + 99
            mask_pos = (flux_corr > 0)
            mag_corr[mask_pos] = -2.5 * np.log10(flux_corr[mask_pos])
            mask_99 = (~mask_pos | (mag_corr >= 99))
            mag_corr[mask_99] = 99
            table['MAG_ZOGY_PLUSREF'] = mag_corr.astype('float32')


            # the corresponding error
            pogson = 2.5 / np.log(10)
            magerr_zogy = np.array(table['MAGERR_ZOGY'])
            fluxerr_zogy = np.abs(flux_zogy) * magerr_zogy / pogson
            magerr_opt_ref = np.array(table['MAGERR_OPT_REF'])
            fluxerr_opt_ref = np.abs(flux_ref) * magerr_opt_ref / pogson
            fluxerr_corr = np.sqrt(fluxerr_zogy**2 + fluxerr_opt_ref**2)
            magerr_corr = np.zeros_like(flux_corr, dtype='float32') + 99
            magerr_corr[mask_pos] = pogson * (fluxerr_corr[mask_pos]
                                              / flux_corr[mask_pos])
            magerr_corr[mask_99] = 99
            table['MAGERR_ZOGY_PLUSREF'] = magerr_corr.astype('float32')

            # S/N
            mask_nonzero = (fluxerr_corr != 0)
            snr_corr = np.zeros_like(flux_corr, dtype='float32')
            snr_corr[mask_nonzero] = (flux_corr[mask_nonzero] /
                                      fluxerr_corr[mask_nonzero])            
            table['SNR_ZOGY_PLUSREF'] = snr_corr




    zogy.mem_use ('at end of [get_rows]')

    return table


################################################################################

def infer_mags (table, basename, fits_mask, nsigma, apphot_radii,
                apphot_sky_inout, apphot_att2add, include_fluxes, keys2add,
                add_keys, bkg_global, thumbnails, size_tn, imtype='new',
                tel='ML1', ncpus=None):


    # label in logging corresponding to 'new', 'ref' and 'trans' imtypes
    label_dict = {'new': 'full-source', 'ref': 'reference', 'trans': 'transient'}
    label = label_dict[imtype]

    # similar dictionary for string to add to output table colnames
    s2add_dict = {'new': '', 'ref': '_REF', 'trans': '_TRANS'}
    s2add = s2add_dict[imtype]

    # shorthand
    new = (imtype == 'new')
    trans = (imtype == 'trans')
    ref = (imtype == 'ref')


    # filenames relevant for magtype 'full-source' and 'reference'
    fits_red = add_drop_fz ('{}.fits.fz'.format(basename))
    fits_cat = '{}_cat.fits'.format(basename)
    fits_limmag = add_drop_fz ('{}_limmag.fits.fz'.format(basename))
    psfex_bintable = '{}_psf.fits'.format(basename)
    fits_objmask = add_drop_fz ('{}_objmask.fits.fz'.format(basename))


    # filenames relevant for magtypes 'trans'
    fits_Fpsf = add_drop_fz ('{}_Fpsf.fits.fz'.format(basename))
    fits_trans = '{}_trans.fits'.format(basename)
    fits_tlimmag = add_drop_fz ('{}_trans_limmag.fits.fz'.format(basename))
    fits_Scorr = add_drop_fz ('{}_Scorr.fits.fz'.format(basename))
    fits_D = add_drop_fz ('{}_D.fits.fz'.format(basename))


    if trans:
        list2check = [fits_Fpsf, fits_trans, fits_tlimmag, fits_Scorr]
    else:
        if fits_mask is not None:
            list2check = [fits_red, fits_mask, psfex_bintable]
        else:
            list2check = [fits_red, psfex_bintable]


    # check if required images/catalogs are available
    for fn in list2check:
        if not os.path.exists (fn):
            log.warning ('{} not found; skipping extraction of {} magnitudes '
                         'for {}'.format(fn, label, basename))
            return table


    # read header
    try:
        if trans:
            fits2read = fits_trans
        else:
            fits2read = fits_red

        header = FITS(fits2read)[-1].read_header()

    except:
        log.exception ('trouble reading header of {}; skipping extraction of {} '
                       'magnitudes for {}'
                       .format(fits2read, label, basename))
        return table



    # read FWHM from the header
    if 'PSF-FWHM' in header:
        fwhm = header['PSF-FWHM']
    elif 'S-FWHM' in header:
        fwhm = header['S-FWHM']
    else:
        fwhm = 5
        log.warning ('keywords PSF-FWHM nor S-FWHM present in the header '
                     'for {}; assuming fwhm=5 pix'.format(basename))


    # data_shape from header
    if ref:
        data_shape = (header['ZNAXIS2'], header['ZNAXIS1'])
    else:
        data_shape = zogy.get_par(set_zogy.shape_new,tel)

    ysize, xsize = data_shape
    
    
    # convert input RA/DEC from table to pixel coordinates; needs to
    # be done from table as it may shrink in size between different
    # calls to [infer_mags]
    xcoords, ycoords = WCS(header).all_world2pix(table['RA_IN'],
                                                 table['DEC_IN'], 1)


    # discard entries that were not finite or off the image NB; this
    # means that any source that is off the reduced image, but present
    # in the reference image will not appear in the reference part of
    # the output table. A way around this is to set both [fullsource]
    # and [trans] to False. Could try to keep all coordinates, but
    # then would have to juggle with masks below, prone to mistakes.

    # make sure xcoords and ycoords are finite
    mask_finite = (np.isfinite(xcoords) & np.isfinite(ycoords))

    # and on the image
    dpix_edge = 10
    mask_on = ((xcoords > dpix_edge) & (xcoords < xsize-dpix_edge) &
               (ycoords > dpix_edge) & (ycoords < ysize-dpix_edge))

    # combination of finite/on-image masks; return if no coordinates
    # left
    mask_ok = mask_finite & mask_on
    if np.sum(mask_ok)==0:
        log.warning ('all of the inferred pixel coordinates are infinite/nan '
                     'and/or off the image for {}; skipping extraction of {} '
                     'magnitudes for {}'.format(fits_red, label, basename))
        return table


    ncoords_ok = np.sum(mask_ok)
    if np.sum(~mask_ok) != 0:
        log.info ('{} off-image or non-finite coordinates for {} extraction of '
                  '{}'.format(np.sum(~mask_ok), label, basename))
        xcoords = xcoords[mask_ok]
        ycoords = ycoords[mask_ok]
        table = table[mask_ok]


    # update table with coordinates
    if ref:
        table['X_POS_REF'] = xcoords
        table['Y_POS_REF'] = ycoords
    else:
        table['X_POS'] = xcoords
        table['Y_POS'] = ycoords


    # indices of pixel coordinates; need to be defined after
    # discarding coordinates off the image
    x_indices = (xcoords-0.5).astype(int)
    y_indices = (ycoords-0.5).astype(int)


    # determine several other header keyword values; NB: use of
    # mask_ok, which narrows the table down to valid coordinates
    exptime, filt, zp, airmass, ext_coeff = get_keys (
        header, table['RA_IN'], table['DEC_IN'], tel)

    if False:
        log.info ('exptime:   {}'.format(exptime))
        log.info ('filt:      {}'.format(filt))
        log.info ('zp:        {}'.format(zp))
        log.info ('airmass:   {}'.format(airmass))
        log.info ('ext_coeff: {}'.format(ext_coeff))

    
    # split between new/ref and transient extraction
    if not trans:

        # determine background standard deviation using [get_bkg_std]
        data_bkg_std = get_bkg_std (basename, xcoords, ycoords, data_shape,
                                    imtype, tel)

        # object mask - not always available, so first check if it
        # exists
        if fits_objmask is not None and os.path.exists(fits_objmask):
            objmask = zogy.read_hdulist (fits_objmask, dtype=bool)

        else:
            # if it does not exist, create an object masking depending
            # on input parameter [bkg_global]
            if bkg_global:
                # if True, global background is used, i.e. any local flux
                # due to nearby sources or galaxy is not taken into
                # account
                objmask = np.ones (data_shape, dtype=bool)
            else:
                # if False, a circular annulus around each object is used
                # to estimate the sky background
                objmask = np.zeros (data_shape, dtype=bool)


        # read reduced image; need to use astropy method, as otherwise
        # this will lead to an exception in [zogy.get_psfoptflux] as
        # (probably) the shape attribute is not available when data is
        # read through fitsio.FITS
        data = zogy.read_hdulist (fits_red)

        # corresponding mask may not be available, so first check if
        # it exists
        if fits_mask is not None and os.path.exists(fits_mask):
            log.info ('fits_mask used: {}'.format(fits_mask))
            data_mask = zogy.read_hdulist (fits_mask)
            # mask can be read using fitsio.FITS, but only little bit
            # faster than astropy.io.fits
            #data_mask = FITS(fits_mask)[-1][:,:]
        else:
            log.warning ('fits_mask {} does not exist; assuming that none of '
                         'the pixels are flagged'.format(fits_mask))
            data_mask = np.zeros (data_shape, dtype='uint8')


        # add combined FLAGS_MASK column to output table using
        # [get_flags_mask_comb]
        table['FLAGS_MASK{}'.format(s2add)] = (
            get_flags_mask_comb(data_mask, xcoords, ycoords, fwhm, xsize, ysize))


        try:
            # determine optimal fluxes at pixel coordinates
            # !!! CHECK !!! _mp or not
            if ncpus is None:
                # submit to [get_psfoptflux_mp] with single thread,
                # as the multiprocessing is done on the image level,
                # i.e. each cpu is processing a different image
                flux_opt, fluxerr_opt = zogy.get_psfoptflux_mp (
                    psfex_bintable, data, data_bkg_std**2, data_mask, xcoords,
                    ycoords, imtype=imtype, fwhm=fwhm, D_objmask=objmask,
                    set_zogy=set_zogy, tel=tel, nthreads=1)
            else:
                # submit to [get_psfoptflux_mp] with [ncpu] threads as
                # this concerns a single image and the multiprocessing
                # should be done on a the object level, i.e. each cpu
                # processes different objects on the same image. The
                # [force_phot] function only provides ncpus to
                # [get_rows] and [infer_mags] in case of a single
                # image.
                flux_opt, fluxerr_opt = zogy.get_psfoptflux_mp (
                    psfex_bintable, data, data_bkg_std**2, data_mask, xcoords,
                    ycoords, imtype=imtype, fwhm=fwhm, D_objmask=objmask,
                    set_zogy=set_zogy, tel=tel, nthreads=ncpus)


        except Exception as e:
            log.error ('exception was raised while executing '
                       '[zogy.get_psfoptflux]; skipping extraction of {} '
                       'magnitudes for {}: {}'.format(label, basename, e))
            return table


        if zp is not None:
            # infer calibrated magnitudes using the zeropoint
            mag_opt, magerr_opt = zogy.apply_zp (flux_opt, zp, airmass, exptime,
                                                 filt, ext_coeff,
                                                 fluxerr=fluxerr_opt)
            mask_pos = (flux_opt > 0)
            mag_opt[~mask_pos] = 99
            magerr_opt[~mask_pos] = 99

        else:
            log.warning ('keyword PC-ZP not in header; unable to infer {} '
                         'magnitudes for {}'.format(label, basename))


        # infer limiting magnitudes
        limmags = get_limmags (fits_limmag, y_indices, x_indices, header, nsigma,
                               nsigma_orig=5, label=label)


        # calculate signal-to-noise ratio; applies to either the SNR
        # of the limit or the matched source in the catalog
        snr_opt = np.zeros(ncoords_ok)
        mask_nonzero = (fluxerr_opt != 0)
        snr_opt[mask_nonzero] = (flux_opt[mask_nonzero] /
                                 fluxerr_opt[mask_nonzero])


        # update table
        table['MAG_OPT{}'.format(s2add)] = mag_opt.astype('float32')
        table['MAGERR_OPT{}'.format(s2add)] = magerr_opt.astype('float32')
        table['SNR_OPT{}'.format(s2add)] = snr_opt.astype('float32')
        table['LIMMAG_OPT{}'.format(s2add)] = (limmags.astype('float32'))


        # add fluxes if needed
        if include_fluxes:
            col_tmp = 'E_FLUX_OPT{}'.format(s2add)
            table[col_tmp] = (flux_opt/exptime).astype('float32')
            col_tmp = 'E_FLUXERR_OPT{}'.format(s2add)
            table[col_tmp] = (fluxerr_opt/exptime).astype('float32')



        # add aperture measurements
        if apphot_radii is not None:

            t_ap = time.time()

            # update type of background annulus
            if bkg_global or apphot_sky_inout is None:
                bkgann = None
            else:
                bkgann = tuple(np.array(apphot_sky_inout).astype(float) * fwhm)

            apphot_radii_xfwhm = np.array(apphot_radii).astype(float) * fwhm
            nrad = len(apphot_radii)


            if False:
                # use Source Extraction and Photometry
                flux_ap, fluxerr_ap, flags_ap = SEP.sum_circle (
                    data, xcoords-1, ycoords-1,
                    np.expand_dims(apphot_radii_xfwhm, 1),
                    err=data_bkg_std, gain=1.0, bkgann=bkgann, mask=data_mask)


            # use photutils instead, as it has more freedom regarding
            # background annulus subtraction (SEP only provides the
            # mean value), and it can also provide Source Extractor
            # quanities like FWHM and ELONGATION

            xycoords = list(zip(xcoords, ycoords))
            apertures = [CircularAperture(xycoords, r=rad)
                         for rad in apphot_radii_xfwhm]

            #data_err = calc_total_error(data, data_bkg_std, 1.0)
            data_err = np.sqrt(np.abs(data) + data_bkg_std**2)

            if bkgann is not None:
                r_in, r_out = bkgann 
                bkg_annuli = CircularAnnulus(xycoords, r_in=r_in, r_out=r_out)
                sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
                bkg_stats = ApertureStats(data, bkg_annuli, error=data_err,
                                          sigma_clip=sigma_clip)
                local_bkg = bkg_stats.mean
            else:
                local_bkg = None


            # perform aperture measurements (many attributes
            # available, e.g. covar_sigx2 and y2, fwhm, elongation);
            # unfortunately only single aperture can be processed at a
            # time
            for i, radius in enumerate(apphot_radii):
                aper_stats = ApertureStats(data, apertures[i],
                                           error=data_err,
                                           mask=data_mask.astype(bool),
                                           local_bkg=local_bkg)

                # flux sum and corresponding error for this radius
                flux_ap = aper_stats.sum
                fluxerr_ap = aper_stats.sum_err


                # infer calibrated magnitudes using the zeropoint
                if zp is not None:
                    mag_ap, magerr_ap = zogy.apply_zp (flux_ap, zp, airmass,
                                                       exptime, filt, ext_coeff,
                                                       fluxerr=fluxerr_ap)
                    mask_pos = (flux_ap > 0)
                    mag_ap[~mask_pos] = 99
                    magerr_ap[~mask_pos] = 99

                    # S/N
                    mask_nonzero = (fluxerr_ap != 0)
                    snr_ap = np.zeros_like(flux_ap, dtype='float32')
                    snr_ap[mask_nonzero] = (flux_ap[mask_nonzero] /
                                            fluxerr_ap[mask_nonzero])

                    col_tmp = 'MAG_APER_R{}xFWHM{}'.format(radius, s2add)
                    table[col_tmp] = mag_ap.astype('float32')
                    col_tmp = 'MAGERR_APER_R{}xFWHM{}'.format(radius, s2add)
                    table[col_tmp] = magerr_ap.astype('float32')
                    col_tmp = 'SNR_APER_R{}xFWHM{}'.format(radius, s2add)
                    table[col_tmp] = snr_ap

                else:
                    log.warning ('keyword PC-ZP not in header; unable to infer {} '
                                 'magnitudes for {}'.format(label, basename))


                # add fluxes if needed
                if include_fluxes:
                    col_tmp = 'E_FLUX_APER_R{}xFWHM{}'.format(radius, s2add)
                    table[col_tmp] = (flux_ap/exptime).astype('float32')
                    col_tmp = 'E_FLUXERR_APER_R{}xFWHM{}'.format(radius, s2add)
                    table[col_tmp] = (fluxerr_ap/exptime).astype('float32')
                    


                # add ApertureStats attribute(s) provided in [[apphot_att2add]
                if apphot_att2add is not None:
                    for att in apphot_att2add:
                        vals_att = eval('aper_stats.{}'.format(att.lower()))
                        col_tmp = '{}_APER_R{}xFWHM{}'.format(att.upper(),
                                                              radius, s2add)
                        table[col_tmp] = vals_att


            log.info ('{} source(s), {} aperture(s), apphot time: {:.3f}s'
                      .format(ncoords_ok, nrad, time.time()-t_ap))


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


        
    else:

        # read flux values at xcoords, ycoords
        Fpsf = get_fitsio_values (fits_Fpsf, y_indices, x_indices)

        # get transient limiting magnitude at xcoord, ycoord
        # and convert it back to Fpsferr

        # read limiting magnitude at pixel coordinates
        nsigma_trans_orig = 6
        tlimmags = get_limmags (fits_tlimmag, y_indices, x_indices, header,
                                nsigma, nsigma_orig=nsigma_trans_orig,
                                label=label)


        # zp, object airmass, ext_coeff and exptime were
        # determined above; for conversion from transient
        # limiting magnitude to Fpsferr the airmass at image
        # centre was used
        airmassc = header['AIRMASSC']
        Fpsferr = (10**(-0.4*(tlimmags - zp + airmassc * ext_coeff))
                   * exptime / zogy.get_par(set_zogy.transient_nsigma,tel))


        # read off transient S/N from Scorr image
        snr_zogy = get_fitsio_values (fits_Scorr, y_indices, x_indices)


        if zp is not None:
            # infer calibrated magnitudes using the zeropoint
            mag_zogy, magerr_zogy = zogy.apply_zp (np.abs(Fpsf), zp, airmass,
                                                   exptime, filt, ext_coeff,
                                                   fluxerr=Fpsferr)
            mask_zero = (Fpsf==0)
            mag_zogy[mask_zero] = 99
            magerr_zogy[mask_zero] = 99

        else:
            mag_zogy = np.zeros(ncoords_ok, dtype='float32') + 99
            magerr_zogy = np.zeros(ncoords_ok, dtype='float32') + 99
            log.warning ('keyword PC-ZP not in header; unable to infer {} '
                         'magnitudes for {}'.format(label, basename))



        # update table
        table['MAG_ZOGY'] = mag_zogy.astype('float32')
        table['MAGERR_ZOGY'] = magerr_zogy.astype('float32')
        table['SNR_ZOGY'] = snr_zogy.astype('float32')
        table['LIMMAG_ZOGY'] = tlimmags.astype('float32')


        # add fluxes if needed
        if include_fluxes:
            table['E_FLUX_ZOGY'] = (Fpsf/exptime).astype('float32')
            table['E_FLUXERR_ZOGY'] = (Fpsferr/exptime).astype('float32')


        # add transient thumbnail images
        if thumbnails:

            fits_dict = {'D': fits_D, 'SCORR': fits_Scorr}
            for key in ['D', 'SCORR']:
                
                # shorthand
                key_tn = 'THUMBNAIL_{}'.format(key)
                fn = fits_dict[key]
                
                # check if file exists
                if os.path.exists(fn):
                    # read data using fitsio.FITS
                    data = FITS(fn)[-1]
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
        index_full, index_tn = (zogy.get_index_around_xy(ysize, xsize, y, x,
                                                         size_tn))

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

def get_limmags (fits_limmag, y_indices, x_indices, header, nsigma,
                 nsigma_orig=5, label='full-source'):

    
    # read limiting magnitude at pixel coordinates
    if os.path.exists(fits_limmag):

        # infer limiting magnitudes
        limmags = get_fitsio_values (fits_limmag, y_indices, x_indices)

        # convert limmag from number of sigma listed
        # in the image header to input [nsigma]
        if ('NSIGMA' in header and
            isinstance(header['NSIGMA'], (float, int)) and
            header['NSIGMA'] != 0):
            nsigma_orig = header['NSIGMA']

        if nsigma_orig != nsigma:
            limmags += -2.5*np.log10(nsigma/nsigma_orig)

    else:
        log.warning ('{} not found; no {} limiting magnitude(s) '
                     'available'.format(fits_limmag, label))
        ncoords = len(y_indices)
        limmags = np.zeros(ncoords, dtype='float32')


    return limmags


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


    # determine object airmass, unless input image is a combined
    # image
    if 'R-V' in header or 'R-COMB-M' in header:
        airmass = 1.0
    else:
        lat = zogy.get_par(set_zogy.obs_lat,tel)
        lon = zogy.get_par(set_zogy.obs_lon,tel)
        height = zogy.get_par(set_zogy.obs_height,tel)
        airmass = zogy.get_airmass(ra_in, dec_in, obsdate, lat, lon, height)


    # extinction coefficient
    ext_coeff = zogy.get_par(set_zogy.ext_coeff,tel)[filt]       


    return exptime, filt, zp, airmass, ext_coeff


################################################################################

def get_bkg_std (basename, xcoords, ycoords, data_shape, imtype, tel):

    # background STD
    fits_bkg_std = '{}_bkg_std.fits.fz'.format(basename)
    if os.path.exists(fits_bkg_std):
        data_bkg_std = zogy.read_hdulist (fits_bkg_std, dtype='float32')
        # only little bit faster with fitsio.FITS
        #data_bkg_std = FITS(fits_bkg_std)[-1][:,:]
    else:
        # if it does not exist, create it from the background mesh
        fits_bkg_std_mini = '{}_bkg_std_mini.fits'.format(basename)
        data_bkg_std_mini, header_mini = zogy.read_hdulist (
            fits_bkg_std_mini, get_header=True, dtype='float32')

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
            data_bkg_std = zogy.mini2back (
                data_bkg_std_mini, data_shape, order_interp=1,
                bkg_boxsize=bkg_size, interp_Xchan=interp_Xchan_std,
                timing=zogy.get_par(set_zogy.timing,tel))

            
    return data_bkg_std


################################################################################

def get_fitsio_values (filename, y_indices=None, x_indices=None):

    # read data using fitsio.FITS
    data = FITS(filename)[-1]

    # infer data values at indices
    if y_indices is None or x_indices is None:
        values = data[:,:]
    else:
        nvalues = len(y_indices)
        values = np.zeros(nvalues)
        for i in range(nvalues):
            values[i] = data[y_indices[i]:y_indices[i]+1,
                             x_indices[i]:x_indices[i]+1]
            
    return values


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
    flags_mask_comb = np.zeros(ncoords, dtype='uint')

    # loop coordinates
    for m in range(ncoords):

        # get index around x,y position using function
        # [zogy.get_index_around_xy]
        index_full, index_tn = (zogy.get_index_around_xy(ysize, xsize,
                                                         ycoords[m], xcoords[m],
                                                         size_4fwhm))

        if np.sum(data_mask[index_full]) != 0:

            # create zero-valued temporary thumbnail data_mask
            data_mask_tn = np.zeros((size_4fwhm, size_4fwhm), dtype='uint')

            # fill temporary thumbnail with values from full data_mask
            data_mask_tn[index_tn] = data_mask[index_full]

            # define mask of pixels within 2*fwhm of the pixel coordinates
            mask_central = (np.sqrt(
                (xx - (xcoords[m] - int(xcoords[m]) + hsize))**2 +
                (yy - (ycoords[m] - int(ycoords[m]) + hsize))**2) < 2*fwhm)

            # add sum of unique values in the thumbnail data mask
            # to the output array flags_mask_comb
            flags_mask_comb[m] = np.sum(np.unique(data_mask_tn[mask_central]))



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
        log.info ('proper motion buffer: {:.2f} arcmin'.format(pm_buffer * 60))
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
        #with fits.open('{}_hdr.fits'.format(basename)) as hdulist:
        #    header = hdulist[-1].header
        header = FITS('{}_hdr.fits'.format(basename))[-1].read_header()


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

# from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
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
                        '[date_format]; RA and DEC can either be both in decimal '
                        'degrees or in colon-separated sexagesimal hours and '
                        'degrees for RA and DEC, respectively')

    parser.add_argument('fits_out', type=str,
                        help='output fits table with resulting magnitudes')

    parser.add_argument('--radecs_file_format', type=str, default='fits',
                        help='astropy file format of [radecs] in case it points '
                        'to a file, e.g. ascii or csv; default=fits')

    parser.add_argument('--ra_col', type=str, default='RA',
                        help='name of input RA column in [radecs]; default=RA')

    parser.add_argument('--dec_col', type=str, default='DEC',
                        help='name of input DEC column in [radecs]; default=DEC')

    parser.add_argument('--date_col', type=str, default='DATE-OBS',
                        help='name of input date of observation in [radecs]; '
                        'only needed in case you want to limit the images '
                        'defined in [filenames] to be observed within a time '
                        'window with size [dtime_max] around this input date; '
                        'default=DATE-OBS')

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

    parser.add_argument('--filenames', type=str,
                        default='/idia/projects/meerlicht/Headers/ML1_headers_cat.fits',
                        help='comma-separated list with the filenames of the '
                        'reduced images to be used or the name of a file with '
                        'format [filenames_file_format] containing the filenames '
                        'with the column name [filenames_col]; '
                        'default=/idia/projects/meerlicht/Headers/ML1_headers_cat.fits')

    parser.add_argument('--filenames_col', type=str, default='FILENAME',
                        help='name of input filename column in [filenames]; '
                        'default=FILENAME')

    parser.add_argument('--filenames_file_format', type=str, default='fits',
                        help='astropy file format of [filenames] in case it '
                        'points to a file; default=fits')

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

    parser.add_argument('--bkg_global', type=str2bool, default=False,
                        help='for full-source and ref cases: use global '
                        'background estimate (T) or estimate local background '
                        'from annulus around the coordinates (F); default=False')

    parser.add_argument('--nsigma', type=int, default=3,
                        help='significance threshold for a detection; default=3')

    parser.add_argument('--apphot_radii', type=str, default='0.66,1.5,5',
                        help='radii in units of the image FWHM at which to '
                        'extract aperture photometry; default=0.66,1.5,5, '
                        'if set to None, no aperture photometry is performed')

    parser.add_argument('--apphot_sky_inout', type=str, default='4,5',
                        help='inner and outer radii in units of the image FWHM '
                        'of the sky annulus used to determine the aperture '
                        'photometry local background; default=4,5; if set to '
                        'None, no local background is subtracted (this also '
                        'depends on [bkg_global], i.e. if these sky radii are '
                        'defined but [bkg_global]=True, no background is '
                        'subtracted either')

    parser.add_argument('--apphot_att2add', type=str, default=None,
                        help='comma-separated list of any additional float or '
                        'integer attribute of photutils.aperture.ApertureStats '
                        'to add to the output table (for a complete list, see '
                        'https://photutils.readthedocs.io/en/stable/api/'
                        'photutils.aperture.ApertureStats.html#photutils'
                        '.aperture.ApertureStats); N.B.: can take several '
                        'seconds per source!; default=None')

    parser.add_argument('--include_fluxes', type=str2bool, default=False,
                        help='besides the optimal/aperture magnitudes, also '
                        'include the corresponding fluxes and their errors, '
                        'in units of e-/s, in the output table; default=False')

    parser.add_argument('--thumbnails', type=str2bool, default=False,
                        help='extract thumbnail images around input coordinates? '
                        'The thumbnail images that are extracted depends on the '
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

    par_default = 'MJD-OBS,OBJECT,FILTER,EXPTIME,S-SEEING,AIRMASS,PC-ZP,' \
        'PC-ZPSTD,QC-FLAG'
    parser.add_argument('--keys2add', type=str, default=par_default,
                        help='header keyword values to add to output '
                        'table; default={}'.format(par_default))

    par_default = 'float,U5,U1,float32,float32,float32,float32,float32,U6'
    parser.add_argument('--keys2add_dtypes', type=str, default=par_default,
                        help='corresponding header keyword dtypes; default={}'
                        .format(par_default))

    parser.add_argument('--telescope', type=str, default='ML1',
                        help='telescope; default=ML1')
    
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
    
    args = parser.parse_args()


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
    if os.path.isfile(args.radecs):

        # if [args.radecs] is a file, read it into a table
        table_in = Table.read(args.radecs, format=args.radecs_file_format)
        colnames = table_in.colnames
        log.info ('input table column names: {}'.format(colnames))


        if len(table_in) == 0:
            log.critical('no input coordinates found; if the input is meant to '
                         'be a file, check whether its format provided through '
                         'the input parameter [radecs_file_format] is correct')
            raise SystemExit
        else:
            log.info ('{} lines in input file {}'
                      .format(len(table_in), args.radecs))


        # add column identifying (line) number of input coordinate;
        # this needs to be done here before any potential masking is
        # done with e.g. mask_dtime below
        table_in['NUMBER_IN'] = np.arange(1,len(table_in)+1)


        # convert column [date_col] to list mjds_in
        if str(args.date_col) != 'None' and args.date_col in colnames:

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
            #log.warning ('no info on {} is found in the input, so the '
            #             'coordinates in [radecs] will be searched for in all '
            #             '[filenames] rather than a subset within a time window '
            #             'centered on {} with a total width of [dtime] hours'
            #             .format(args.date_col, args.date_col))



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

        # split input [radecs] into list of strings ['ra1', 'dec1', ...]
        radecs_list0 = re.sub('\(|\)|\[|\]|\{|\}', '', args.radecs)
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
        table_in['NUMBER_IN'] = np.arange(1,len(table_in)+1)



    # rename input columns [args.ra_col] and [args.dec_col]
    table_in[args.ra_col].name = 'RA_IN'
    table_in[args.dec_col].name = 'DEC_IN'


    # rename proper motion columns if they are present
    colnames = table_in.colnames
    if args.pmra_col in colnames and args.pmdec_col in colnames:
        table_in[args.pmra_col].name = 'PMRA_IN'
        table_in[args.pmdec_col].name = 'PMDEC_IN'


    # only keep columns that are needed; note that MJD_IN, PMRA_IN and
    # PMDEC_IN may not exist, but this is checked a bit further below
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

        
    # infer list of filenames to consider from [args.filenames]
    # ---------------------------------------------------------
    radecs_cntr = None

    if os.path.isfile(args.filenames):

        # if the input is a single file, it could be a single image or
        # a fits table/file containing multiple files; to test between
        # the two, try to read the file as a table, which will cause
        # an exception if it is an image
        try:
            # read it into a table using the format
            # [args.filenames_file_format]
            table_filenames = Table.read(args.filenames,
                                         format=args.filenames_file_format)
            log.info ('{} line(s) in input file {}'
                      .format(len(table_filenames), args.filenames))


            # get central coordinates of filenames
            colnames = table_filenames.colnames
            if 'RA-CNTR' in colnames and 'DEC-CNTR' in colnames:

                # mask with files with a WCS solution
                mask_WCS = (np.isfinite(table_filenames['RA-CNTR']) &
                            np.isfinite(table_filenames['DEC-CNTR']) &
                            (table_filenames['RA-CNTR'] > 0) &
                            (table_filenames['RA-CNTR'] < 360) &
                            (np.abs(table_filenames['DEC-CNTR']) < 90))

                log.info ('{} filename(s) with valid CNTR coordinates'
                          .format(np.sum(mask_WCS)))
                table_filenames = table_filenames[mask_WCS]


                # define list of (ra_cntr,dec_cntr) tuples to be used in
                # function [index_images]
                radecs_cntr = np.array(list(zip(table_filenames['RA-CNTR'],
                                                table_filenames['DEC-CNTR'])))


            # define list of filenames
            filenames = list(table_filenames[args.filenames_col])


            # define objects and mjds_obs arrays
            if 'OBJECT' in colnames and 'MJD-OBS' in colnames:
                objects = np.array(table_filenames['OBJECT']).astype(int)
                mjds_obs = np.array(table_filenames['MJD-OBS'])


            # and filter array if needed
            if not filts_all:
                if 'FILTER' in colnames:
                    filts = np.array(table_filenames['FILTER']).astype(str)


        except:
            # apparently the input is not a fits table, so assume it
            # is a single image and put it into a list
            filenames = args.filenames.split(',')
            remove_empty (filenames)
            
    else:

        # filenames were provided as a comma-separated list; put them
        # into a list
        filenames = args.filenames.split(',')
        remove_empty (filenames)



    # in case the dates of observation [mjds_obs] or field IDs
    # [objects] have not yet been defined, need to infer them from the
    # headers; also include RA-CNTR and DEC-CNTR
    if ('mjds_obs' not in locals() or 'objects' not in locals() or
        (not filts_all and 'filts' not in locals())): 
        objects, mjds_obs, filts, ras_cntr, decs_cntr = get_headkeys (filenames)
        radecs_cntr = np.array(list(zip(ras_cntr, decs_cntr)))



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
    # header was not used as args.filenames, 
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
    for d in results:
        index_images_dict.update(d)

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


    # could cut up dictionary in pieces if it turns out to be
    # too large, and loop over [force_phot] various times


    # convert input keys2add and corresponding types to lists
    if str(args.keys2add) != 'None':
        keys2add = args.keys2add.upper().split(',')
    else:
        keys2add = None

    if str(args.keys2add_dtypes) != 'None':
        keys2add_dtypes = args.keys2add_dtypes.split(',')
    else:
        keys2add_dtypes = None

    verify_lengths (keys2add, keys2add_dtypes)


    # change input [apphot_radii] from string to numpy array
    apphot_radii = args.apphot_radii
    if str(apphot_radii) != 'None':
        apphot_radii = apphot_radii.split(',')
        log.info ('apphot_radii: {}'.format(apphot_radii))
    else:
        # or from 'None' to None
        apphot_radii = None

    # change input [apphot_sky_inout] from string to numpy array
    apphot_sky_inout = args.apphot_sky_inout
    if str(apphot_sky_inout) != 'None':
        apphot_sky_inout = apphot_sky_inout.split(',')
        log.info ('apphot_sky_inout: {}'.format(apphot_sky_inout))
    else:
        # or from 'None' to None
        apphot_sky_inout = None

    # change input [apphot_att2add] from string to list of
    # attributes
    apphot_att2add = args.apphot_att2add
    if str(apphot_att2add) != 'None':
        apphot_att2add = apphot_att2add.split(',')
        log.info ('apphot_att2add: {}'.format(apphot_att2add))
    else:
        apphot_att2add = None


    # prepare list of masks
    mask_list = ['{}_mask.fits.fz'.format(fn.split('_red')[0])
                 for fn in image_indices_dict]


    # call [force_phot]
    table_out = force_phot (
        table_in, image_indices_dict, mask_list=mask_list, trans=args.trans,
        ref=args.ref, fullsource=args.fullsource, nsigma=args.nsigma,
        apphot_radii=apphot_radii, apphot_sky_inout=apphot_sky_inout,
        apphot_att2add=apphot_att2add, pm_epoch=args.pm_epoch,
        include_fluxes=args.include_fluxes, keys2add=keys2add,
        keys2add_dtypes=keys2add_dtypes, bkg_global=args.bkg_global,
        thumbnails=args.thumbnails, size_thumbnails=args.size_thumbnails,
        tel=args.telescope, ncpus=ncpus)



    # copy columns from the input to the output table; even if
    # [args.input_cols2copy] was not defined but [args.date_col] is
    # defined, let's copy over at least [args.date_col]
    # N.B.: this is now done by providing an input table to
    # [force_phot] that already includes these columns, so do not
    # have to be added here anymore
    if table_out is not None:

        # rename columns RA and DEC to the input column names; if
        # the input columns were in sexagesimal notation, these
        # will be decimal degrees
        table_out['RA_IN'].name = args.ra_col
        table_out['DEC_IN'].name = args.dec_col


        # if [date_col] was provided and the MJD-OBS column is present
        # in the output table, the delta time between it and the image
        # date of observations can be determined
        if str(args.date_col) != 'None' and 'MJD-OBS' in table_out.colnames:
            mjds_in = table_out['MJD_IN'].value
            dtime_days = np.abs(mjds_in - table_out['MJD-OBS'])
            table_out.add_column(dtime_days, name='DELTA_MJD',
                                 index=table_out.colnames.index('MJD-OBS')+1)
            
            # if args.date_format is mjd, then can rename 'MJD_IN'
            # back to the original name, unless the original name is
            # the same as MJD-OBS - the MJD of the images
            if args.date_format == 'mjd' and args.date_col != 'MJD-OBS':
                table_out['MJD_IN'].name = args.date_col



        # order the output table by original row number
        indices_sorted = np.argsort(table_out, order=(('NUMBER_IN','FILENAME')))
        table_out = table_out[indices_sorted]



    log.info ('time spent in [force_phot]: {:.1f}s'.format(time.time()-t1))
    log.info ('time spent in total:        {:.1f}s'.format(time.time()-t0))


    # write output table to fits
    log.info ('writing output table {}'.format(args.fits_out))
    if table_out is not None:
        table_out.write(args.fits_out, format='fits', overwrite=True)


    # list memory used
    zogy.mem_use ('at very end')
    
