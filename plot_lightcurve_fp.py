import os
import sys
import argparse

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
#log.propagate = False

import numpy as np

import astropy.io.fits as fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import Angle
from astropy import units as u

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
#matplotlib.use('MacOSX')


# filter colours
colours = {'u': 'darkviolet', 'g':'forestgreen', 'q':'darkorange',
           'r': 'orangered', 'i':'crimson', 'z':'dimgrey'}


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

def main():

    parser = argparse.ArgumentParser(description='Plot lightcurve from input file')
    parser.add_argument('inputfile', help='input fits or csv file with fields '
                        'MJD-OBS, FILTER and fluxes or magnitudes')
    parser.add_argument('colname_magflux', help='magnitude or flux column to plot')
    parser.add_argument('colname_magfluxerr', help='magnitude or flux error column '
                        'to plot')
    parser.add_argument('--filters', default='ugqriz',
                        help='filters to plot (default=ugqriz)')
    parser.add_argument('--date_start', default=None, help='start date of '
                        'observation in isot format to plot (default=None)')
    parser.add_argument('--date_end', default=None, help='end date of '
                        'observation in isot format to plot (default=None)')
    parser.add_argument('--split_fields', type=str2bool, default=False,
                        help='plot different fields with different symbols? '
                        'default=False')
    parser.add_argument('--plot_fwhm', type=str2bool, default=False,
                        help='plot the FWHM in an additional panel; default=False')
    parser.add_argument('--qcflag_max', type=str, default='orange',
                        choices=['green', 'yellow', 'orange', 'red'],
                        help='worst QC flag to plot; default=\'orange\'')


    args = parser.parse_args()
    inputfile = args.inputfile
    filters = args.filters
    date_start = args.date_start
    date_end = args.date_end
    split_fields = args.split_fields
    plot_fwhm = args.plot_fwhm
    qcflag_max = args.qcflag_max
    colname_magflux = args.colname_magflux
    colname_magfluxerr = args.colname_magfluxerr


    # read input file
    t = Table.read(args.inputfile)

    telescope = {'ML': 'MeerLICHT', 'BG': 'BlackGEM'}
    if 'FILENAME' in t.colnames:
        tel = t['FILENAME'][0].split('/')[-1].split('_')[0][0:2]
    else:
        tel = inputfile.split('.')[-2].split('_')[-1]

    tel = telescope[tel]


    # mask out filters
    mask_filt = np.array([t['FILTER'][i] in filters for i in range(len(t))])
    t = t[mask_filt]

    # initialize mask_date to be all True
    mask_date = np.ones(len(t), dtype=bool)

    if date_start is not None:
        mask_date &= (t['MJD-OBS'] >= Time(date_start).mjd)

    if date_end is not None:
        mask_date &= (t['MJD-OBS'] <= Time(date_end).mjd)

    # apply mask_date
    t = t[mask_date]


    # if qcflag_max is specified, select only images with QC-FLAG of
    # qcflag_max and better
    if len(t)>0 and qcflag_max is not None:
        qc_col = ['green', 'yellow', 'orange', 'red']
        # redefine qc_col up to and including qcflag_max
        qc_col = qc_col[0:qc_col.index(qcflag_max)+1]

        if ('ZOGY' in colname_magflux or 'PSF_D' in colname_magflux):
            qcflag = t['TQC-FLAG']
        else:
            qcflag = t['QC-FLAG']


        mask_green =  [qcflag[i].strip()=='green'  for i in range(len(t))]
        mask_yellow = [qcflag[i].strip()=='yellow' for i in range(len(t))]
        mask_orange = [qcflag[i].strip()=='orange' for i in range(len(t))]
        mask_red =    [qcflag[i].strip()=='red'    for i in range(len(t))]
        log.info ('number of green: {}, yellow: {}, orange: {}, red: {}'
                  .format(np.sum(mask_green), np.sum(mask_yellow),
                          np.sum(mask_orange), np.sum(mask_red)))

        # strip table color from spaces
        mask_qc = [qcflag[i].strip() in qc_col for i in range(len(t))]
        t = t[mask_qc]



    if 'index' not in t.colnames and 'NUMBER_IN' in t.colnames:
        t['index'] = t['NUMBER_IN'] - 1


    # plot different source_ids
    for i in np.unique(t['index']):
        mask_id = (t['index'] == i)
        plot_source (t[mask_id], inputfile, colname_magflux,
                     colname_magfluxerr, tel, split_fields, plot_fwhm)


def times2mjd (times):
    return Time(times, format='datetime').mjd

def mjd2times (mjd):
    return Time(mjd, format='mjd').datetime


def plot_source (t, inputfile, colname_magflux, colname_magfluxerr, tel,
                 split_fields, plot_fwhm):

    magflux = t[colname_magflux].value
    magfluxerr = t[colname_magfluxerr].value
    mjd_obs = t['MJD-OBS'].value
    filters = t['FILTER']
    times = Time(mjd_obs, format='mjd').datetime
    if 'OBJECT' in t.colnames:
        fields = t['OBJECT']
        fields_uniq, idx_inverse = np.unique(fields, return_inverse=True)
    else:
        if split_fields:
            log.error ('no OBJECT keyword among table columns, so cannot plot '
                       'fields with different symbols')


    # mask indicating FLAGS RED and/or REF are zero
    mask_ok = np.ones(len(t), dtype=bool)
    if 'FLAGS_MASK_RED' in t.colnames:
        mask_ok &= (t['FLAGS_MASK_RED']==0)

    if 'FLAGS_MASK_REF' in t.colnames:
        mask_ok &= (t['FLAGS_MASK_REF']==0)


    # relevant QC-FLAG
    if ('ZOGY' in colname_magflux or 'PSF_D' in colname_magflux):
        label = 'TQC-FLAG'
        qcflag = t['TQC-FLAG']
    else:
        label = 'QC-FLAG'
        qcflag = t['QC-FLAG']



    if 'MAG' in colname_magflux:
        try:
            limmag = t[colname_magflux.replace('MAG', 'LIMMAG')].value
        except:
            # not all magnitude columns have a corresponding limiting
            # magnitude column; try to infer it from the corresponding
            # FNU error columns
            fnuerr = t[colname_magflux.replace('MAG', 'FNUERR')].value
            limmag = -2.5*np.log10(3*fnuerr) + 23.9

        snr_opt = t[colname_magflux.replace('MAG', 'SNR')].value
        mask_detections = (snr_opt >= 3) & (limmag != 99) & (qcflag != 'red')
        mask_limits = (snr_opt < 3) & (limmag != 99) & (qcflag != 'red')
    else:
        limmag = None
        mask_detections = np.ones(len(magflux), dtype=bool)
        mask_limits = ~mask_detections


    # plot
    if plot_fwhm:
        fig, (ax, ax_fwhm, ax_qc) = plt.subplots(3, 1, sharex=True,
                                                 height_ratios=[10,4,1],
                                                 figsize=(12,8))
    else:
        fig, (ax, ax_qc) = plt.subplots(2, 1, sharex=True,
                                        height_ratios=[15,1],
                                        figsize=(12,8))
    fig.subplots_adjust(hspace=0, left=0.07, bottom=0.14)


    marker_list = ['o', 's', 'p', 'P', '*']
    if not split_fields or 'fields' not in locals():
        marker = 'o'
    else:
        marker = np.array(marker_list)[idx_inverse]


    for filt in np.unique(filters):

        mask_filt = (filters == filt)
        mask_lim = mask_filt & mask_limits
        mask_det = mask_filt & mask_detections
        mask_det_ok = mask_det & mask_ok
        mask_det_notok = mask_det & ~mask_ok


        # plot detections
        if np.sum(mask_det) > 0:
            ax.errorbar(times[mask_det], magflux[mask_det], ms=5,
                        yerr=magfluxerr[mask_det], color=colours[filt],
                        linestyle="None", capsize=2)

            if not split_fields or 'fields' not in locals():
                # solid symbols
                ax.plot(times[mask_det_ok], magflux[mask_det_ok],
                        'o', ms=5, mec=colours[filt],
                        mfc=colours[filt], label=filt)
                # open symbols
                ax.plot(times[mask_det_notok], magflux[mask_det_notok],
                        'o', ms=5, mec=colours[filt],
                        mfc='none', markeredgewidth=1)
            else:
                # solid symbols
                for i in range(np.sum(mask_det_ok)):
                    ax.plot(times[mask_det_ok][i], magflux[mask_det_ok][i],
                            marker[mask_det_ok][i], ms=5, mec=colours[filt],
                            mfc=colours[filt])
                # open symbols
                for i in range(np.sum(mask_det_notok)):
                    ax.plot(times[mask_det_notok][i], magflux[mask_det_notok][i],
                            marker[mask_det_notok][i], ms=5, mec=colours[filt],
                            mfc='none', markeredgewidth=1)



        # plot limits
        nlimits = np.sum(mask_lim)
        print ('filter: {}, nlimits: {}'.format(filt, nlimits))
        if limmag is not None and nlimits > 0:
            limplot = True
            ax.plot(times[mask_lim], limmag[mask_lim], 'v', ms=7,
                    mec=colours[filt], mfc='none', markeredgewidth=1)


        if plot_fwhm and 'S-SEEING' in t.colnames:
            seeing = t['S-SEEING']
            m = mask_det | mask_lim
            ax_fwhm.plot(times[m], seeing[m], 'o', ms=5, mec=colours[filt],
                         mfc=colours[filt], label=filt)
            #ax_fwhm.set_ylim([1,5])


    if plot_fwhm and 'S-SEEING' in t.colnames:
        ax_fwhm.set_ylabel('FWHM', size=14, labelpad=10)


    ax.set_ylabel(colname_magflux, size=14, labelpad=10)
    ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    ax.yaxis.set_ticks_position('both')
    ax.invert_yaxis()
    ax.spines[['bottom']].set_visible(False)


    # indicate QC flag
    m = mask_detections | mask_limits
    for i in range(np.sum(m)):
        ax_qc.plot([times[m][i],times[m][i]], [0,1], color=qcflag[m][i])

    ax_qc.spines[['top']].set_visible(False)
    ax_qc.tick_params(left=False, right=False, top=False)
    ax_qc.tick_params(labelleft=False, labelright=False, labeltop=False)

    ax_qc.set_xlabel('Date of observation (UTC)', size=15, labelpad=10)
    ax_qc.set_ylabel(label, size=15, labelpad=40)
    for tick in ax_qc.get_xticklabels():
        tick.set_rotation(30)



    if 'SOURCE_ID' in t.colnames:

        source_id = t['SOURCE_ID'][0]
        title = ('Gaia DR3 {}_{}'.format(inputfile.split('.')[-2].split('_')[-1],
                                         source_id))
        figname = '{}_{}.pdf'.format(inputfile.split('.')[-2], source_id)

    elif ('RA' in t.colnames and 'DEC' in t.colnames or
          'ra' in t.colnames and 'dec' in t.colnames):

        if 'RA' in t.colnames:
            ra = t['RA'][0]
            dec = t['DEC'][0]
        else:
            ra = t['ra'][0]
            dec = t['dec'][0]


        ra_sexa = Angle(ra, unit='deg').to_string(u.hour, precision=3)
        dec_sexa = Angle(dec, unit='deg').to_string(u.deg, precision=2)

        title = ('{} forced photometry at RA={}, DEC={}'
                 .format(tel, ra_sexa, dec_sexa))
        figname = '{}_RA={}_DEC={}.pdf'.format(inputfile.split('.')[-2],
                                               ra_sexa, dec_sexa)



    ax2 = ax.twiny()
    ax2.set_xlabel('Modified Julian Date (days)', size=14, labelpad=10)
    t1, t2 = ax.get_xlim()
    d1, d2 = matplotlib.dates.num2date([t1,t2])
    ax2.set_xlim(times2mjd([d1,d2]))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title, size=18)

    ax.legend(bbox_to_anchor=(1.11, 1), loc='upper right')
    plt.savefig(figname)
    plt.show()
    plt.close()


################################################################################

if __name__ == "__main__":
    main()
