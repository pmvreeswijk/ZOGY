
import argparse
import astropy.io.fits as pyfits
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

from photutils import CircularAperture

from sip_to_pv import *

# some global parameter settings

# optimal subtraction parameters
subimage_size = 1024     # size of subimages
subimage_border = 28     # border around subimage to avoid edge effects
bkg_sex = False          # background: use Sextractor image (T) or simple median (F)
bkg_mode = False         # background: use mode rather than median
bkg_nsigma = 3.          # background: data outside mean +- nsigma * stddev are clipped
use_single_PSF = False   # use the same central PSF for all subimages
skip_wcs = True          # 
nfakestars = 1           # number of fake stars to be added to each subimage
                         # if 1: star will be at the center, if > 1: randomly distributed                       # if 1: star will be at the center, if > 1: randomly distributed
fakestar_s2n = 5.        # required signal-to-noise ratio of the fake stars    
fratio_local = True      # determine fratio (Fn/Fr) from subimage (T) or full frame (F)
dxdy_local = False       # determine dx and dy from subimage (T) or full frame (F)
transient_nsigma = 5     # required significance in Scorr for transient detection

# switch on/off different functions
dosex = False            # do extra SExtractor run (already done inside Astrometry.net)
dosex_psffit = False     # do extra SExtractor run with PSF fitting

# for seeing estimate
fwhm_imafrac = 0.25      # fraction of image area that will be used
                         # for initial seeing estimate
fwhm_detect_thresh = 10. # detection threshold for fwhm SExtractor run
fwhm_class_sort = False  # sort objects according to CLASS_STAR (T)
                         # or by FLUX_AUTO (F)
fwhm_frac = 0.25         # fraction of objects, sorted in brightness
                         # or class_star, used for fwhm estimate
psf_radius = 8           # PSF radius in units of FWHM used to build the PSF
                         # this determines the PSF_SIZE in psfex.config
                         # and size of the VIGNET in sex.params
psf_sampling = 2.        # sampling factor used in PSFex - if zero, it
                         # is automatically determined for the new and
                         # ref image (~FWHM/4.5); if non-zero, it is
                         # fixed to the same sampling for both images
                                                  
# path and names of configuration files
cfg_dir = '/Users/pmv/BlackGem/PipeLine/ImageSubtraction/Config/'
sex_cfg = cfg_dir+'sex.config'     # SExtractor configuration file
sex_cfg_psffit = cfg_dir+'sex_psffit.config' # same for PSF-fitting version
sex_par = cfg_dir+'sex.params'     # SExtractor output parameters definition file
sex_par_psffit = cfg_dir+'sex_psffit.params' # same for PSF-fitting version
psfex_cfg = cfg_dir+'psfex.config' # PSFex configuration file
swarp_cfg = cfg_dir+'swarp.config' # SWarp configuration file

apphot_radii = [0.5, 1, 1.5, 2, 3, 5, 7, 10] # list of radii in units
                                             # of FWHM used for
                                             # aperture photometry in
                                             # SExtractor
# general
redo = True              # execute functions even if output file exist
verbose = True           # print out extra info
timing = True            # (wall-)time the different functions
display = False          # show intermediate fits images
makeplots = False        # show various diagnostic plots and save as pdf

pixelscale = 0.4

################################################################################

def optimal_subtraction(new_fits, ref_fits, ref_fits_remap=None, telescope=None, log=None, subpipe=False):
    
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
    from Barak Zackay and Eran Ofek.
    Adapted by Kerry Paterson for intergration into pipeline for MeerLICHT (ptrker004@myuct.ac.za)
    """

    start_time1 = os.times()

    if telescope is not None:
        Constants = importlib.import_module('Utils.Constants_'+telescope)
        # some global parameter settings
        global subimage_size, subimage_border, bkg_sex, bkg_mode, bkg_nsigma, skip_wcs, nfakestars, fakestar_s2n, fratio_local, dxdy_local, transient_nsigma, dosex, dosex_psffit, pixelscale, fwhm_imafrac, fwhm_detect_thresh, fwhm_class_sort, fwhm_frac, psf_radius, psf_sampling, cfg_dir, sex_cfg, sex_cfg_psffit, sex_par, sex_par_psffit, psfex_cfg, swarp_cfg, apphot_radii, redo, timing, display, makeplots, verbose
        # optimal subtraction parameters
        subimage_size = Constants.Imager.subimage_size      # size of subimages
        subimage_border = Constants.Imager.subimage_border     # border around subimage to avoid edge effects
        bkd_sex = Constants.bkg_sex   # background: use Sextractor image or simple median 
        bkg_mode = Constants.bkg_mode        # background: use mode rather than median
        bkg_nsigma = Constants.bkg_nsigma
        use_single_PSF = Constants.use_single_PSF
        skip_wcs = Constants.skip_wcs
        nfakestars = Constants.nfakestars           # number of fake stars to be added to each subimage
        fakestar_s2n = Constants.fakestar_s2n        # required signal-to-noise ratio of the fake stars 
        fratio_local  = Constants.fratio_local     # determine fratio (Fn/Fr) from subimage
        dxdy_local = Constants.dxdy_local       # determine dx and dy (sigma_x and sigma_y) from subimage
        transient_nsigma = Constants.transient_nsigma     # required significance in Scorr for transient detection

        # switch on/off different functions
        dosex = Constants.dosex
        dosex_psffit = Constants.dosex_psffit    # do extra SExtractor run with PSF fitting

        pixelscale = Constants.Imager.pixel_scale

        fwhm_imafrac = Constants.fwhm_imafrac      # fraction of image area that will be used
                                 # for initial seeing estimate
        fwhm_detect_thresh = Constants.fwhm_detect_thresh # detection threshold for fwhm SExtractor run
        fwhm_class_sort = Constants.fwhm_class_sort  # sort objects according to CLASS_STAR (T)
                                 # or by FLUX_AUTO (F)
        fwhm_frac = Constants.fwhm_frac        # fraction of objects, sorted in brightness
                                 # or class_star, used for fwhm estimate
        psf_radius = Constants.psf_radius           # PSF radius in units of FWHM used to build the PSF
                                 # this determines the PSF_SIZE in psfex.config
                                 # and size of the VIGNET in sex.params
        psf_sampling = Constants.psf_sampling        # sampling factor used in PSFex - if zero, it
                                 # is automatically determined for the new and
                                 # ref image (~FWHM/4.5); if non-zero, it is
                                 # fixed to the same sampling for both images
                                                          
        # path and names of configuration files
        cfg_dir = Constants.cfg_dir
        sex_cfg = Constants.sex_cfg   # SExtractor configuration file
        sex_cfg_psffit = Constants.sex_cfg_psffit # same for PSF-fitting version
        sex_par = Constants.sex_par     # SExtractor output parameters definition file
        sex_par_psffit = Constants.sex_par_psffit # same for PSF-fitting version
        psfex_cfg = Constants.psfex_cfg # PSFex configuration file
        swarp_cfg = Constants.swarp_cfg # SWarp configuration file

        apphot_radii = Constants.apphot_radii # list of radii in units
                                                     # of FWHM used for
                                                     # aperture photometry in
                                                     # SExtractor
        # general
        redo = Constants.redo              # execute functions even if output file exist
        timing = Constants.timing           # (wall-)time the different functions
        display = Constants.display          # show intermediate images
        makeplots = Constants.makeplots       # produce astrometry plots
        verbose = Constants.verbose

    # define the base names of input fits files, base_new and
    # base_ref, as global so they can be used in any function in this
    # module
    global base_new, base_ref
    base_new = new_fits.split('.')[0]
    base_ref = ref_fits.split('.')[0]
    
    # read in header of new_fits
    t = time.time()
    with pyfits.open(new_fits) as hdulist:
        header_new = hdulist[0].header
    keywords = ['NAXIS2','NAXIS1','GAIN','RDNOISE','SATURATE','RA','DEC']
    ysize_new, xsize_new, gain_new, readnoise_new, satlevel_new, ra_new, dec_new = read_header(header_new, keywords)
    if verbose:
        print keywords
        print read_header(header_new, keywords)

    # read in header of ref_fits
    with pyfits.open(ref_fits) as hdulist:
        header_ref = hdulist[0].header
    ysize_ref, xsize_ref, gain_ref, readnoise_ref, satlevel_ref, ra_ref, dec_ref = read_header(header_ref, keywords)
    if verbose:
        print keywords
        print read_header(header_ref, keywords)
        
    if not subpipe:    
        # run SExtractor for seeing estimate of new_fits:
        sexcat_new = base_new+'.sexcat'
        fwhm_new, fwhm_std_new = run_sextractor(base_new+'.fits', sexcat_new, sex_cfg,
                                                sex_par, fraction=fwhm_imafrac)
        print 'fwhm_new, fwhm_std_new', fwhm_new, fwhm_std_new

        # write FWHM to header
        #fwhm_new_str = str('{:.2f}'.format(fwhm_new))
        #header_new['FWHM'] = (fwhm_new_str, '[pix] FWHM estimated from central '+str(fwhm_imafrac))

        # determine WCS solution of new_fits
        new_fits_wcs = base_new+'_wcs.fits'
        if not os.path.isfile(new_fits_wcs) or redo:
            result = run_wcs(base_new+'.fits', new_fits_wcs, ra_new, dec_new,
                             gain_new, readnoise_new, fwhm_new)

        # run SExtractor for seeing estimate of ref_fits:
        sexcat_ref = base_ref+'.sexcat'
        fwhm_ref, fwhm_std_ref = run_sextractor(base_ref+'.fits', sexcat_ref, sex_cfg,
                                                sex_par, fraction=fwhm_imafrac)
        print 'fwhm_ref, fwhm_std_ref', fwhm_ref, fwhm_std_ref

        # determine WCS solution of ref_fits
        ref_fits_wcs = base_ref+'_wcs.fits'
        if not os.path.isfile(ref_fits_wcs) or redo:
            result = run_wcs(base_ref+'.fits', ref_fits_wcs, ra_ref, dec_ref,
                             gain_ref, readnoise_ref, fwhm_ref)


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
    data_new_bkg_full = np.ndarray((ysize_new, xsize_new), dtype='float32')
    data_ref_bkg_full = np.ndarray((ysize_new, xsize_new), dtype='float32')

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size,
                                                                       ysize_new, xsize_new)
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
    if not subpipe:
        data_new, psf_new, psf_orig_new, data_new_bkg = prep_optimal_subtraction(base_new+'_wcs.fits',
                                                                                 nsubs, 'new', fwhm_new)
        data_ref, psf_ref, psf_orig_ref, data_ref_bkg = prep_optimal_subtraction(base_ref+'_wcs.fits',
                                                                                 nsubs, 'ref', fwhm_ref)
    if subpipe:
        data_new, psf_new, psf_orig_new, data_new_bkg = prep_optimal_subtraction(new_fits,
                                                                                 nsubs, 'new', fwhm_new)
        data_ref, psf_ref, psf_orig_ref, data_ref_bkg = prep_optimal_subtraction(ref_fits,
                                                                                 nsubs, 'ref', fwhm_ref, remap=ref_fits_remap)

    # determine corresponding variance images
    var_new = data_new + readnoise_new**2 
    var_ref = data_ref + readnoise_ref**2
    
    if verbose:
        print 'readnoise_new, readnoise_ref', readnoise_new, readnoise_ref

    # get x, y and fratios from matching PSFex stars across entire frame
    if not subpipe:
        x_fratio, y_fratio, fratio, dra, ddec = get_fratio_radec(base_new+'_wcs.psfexcat',
                                                                 base_ref+'_wcs.psfexcat',
                                                                 base_new+'_wcs.sexcat',
                                                                 base_ref+'_wcs.sexcat')
    if subpipe:
        x_fratio, y_fratio, fratio, dra, ddec = get_fratio_radec(base_new+'.psfexcat',
                                                                 base_ref+'.psfexcat',
                                                                 base_new+'.sexcat',
                                                                 base_ref+'.sexcat')
    dx = dra / pixelscale
    dy = ddec / pixelscale

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
    #dy_full = np.std(dy)
    if verbose:
        print 'np.median(dr), np.std(dr)', np.median(dr), np.std(dr)
        print 'np.median(dx), np.std(dx)', np.median(dx), np.std(dx)
        print 'np.median(dy), np.std(dy)', np.median(dy), np.std(dy)
        print 'dr_full, dx_full, dy_full', dr_full, dx_full, dy_full
    
    #fratio_median, fratio_std = np.median(fratio), np.std(fratio)
    fratio_mean, fratio_std, fratio_median = clipped_stats(fratio, nsigma=2)
    if verbose:
        print 'fratio_mean, fratio_std, fratio_median', fratio_mean, fratio_std, fratio_median
    
    if makeplots:
        # plot dy vs dx
        plt.axis((-1,1,-1,1))
        plt.plot(dx, dy, 'go') 
        plt.xlabel('dx (pixels)')
        plt.ylabel('dy (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dxdy.pdf')
        plt.show()
        plt.close()
        
        # plot dr vs x_fratio
        plt.axis((0,xsize_new,0,1))
        plt.plot(x_fratio, dr, 'go')
        plt.xlabel('x (pixels)')
        plt.ylabel('dr (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('drx.pdf')
        plt.show()
        plt.close()

        # plot dr vs y_fratio
        plt.axis((0,ysize_new,0,1))
        plt.plot(y_fratio, dr, 'go')
        plt.xlabel('y (pixels)')
        plt.ylabel('dr (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dry.pdf')
        plt.show()
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
        plt.show()
        plt.close()
                
        # plot dx vs x_fratio
        plt.axis((0,xsize_new,-1,1))
        plt.plot(x_fratio, dx, 'go')
        plt.xlabel('x (pixels)')
        plt.ylabel('dx (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dxx.pdf')
        plt.show()
        plt.close()

        # plot dy vs y_fratio
        plt.axis((0,ysize_new,-1,1))
        plt.plot(y_fratio, dy, 'go')
        plt.xlabel('y (pixels)')
        plt.ylabel('dy (pixels)')
        plt.title(new_fits+'\n vs '+ref_fits, fontsize=12)
        plt.savefig('dyy.pdf')
        plt.show()
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
            
        # determine clipped mean, median and std
        if timing: t1 = time.time()

        if not bkg_mode:
            mean_new, std_new, bkg_new = clipped_stats(data_new[nsub], nsigma=bkg_nsigma,
                                                       show_hist=display, verbose=True)
            mean_ref, std_ref, bkg_ref = clipped_stats(data_ref[nsub], nsigma=bkg_nsigma,
                                                       show_hist=display, verbose=True)
        else:
            mean_new, std_new, bkg_new = clipped_stats(data_new[nsub], nsigma=bkg_nsigma,
                                                       get_median=False,
                                                       get_mode=True)
            mean_ref, std_ref, bkg_ref = clipped_stats(data_ref[nsub], nsigma=bkg_nsigma,
                                                       get_median=False,
                                                       get_mode=True)
        if timing: print 'clipped stats timing', time.time()-t1

        if verbose:
            print 'mean_new, std_new, bkg_new', mean_new, std_new, bkg_new
            print 'mean_ref, std_ref, bkg_ref', mean_ref, std_ref, bkg_ref

        show = False
        if makeplots and show:
            range_new = (bkg_new-5.*std_new, bkg_new+5.*std_new)
            bins = np.linspace(range_new[0], range_new[1], 100)
            plt.hist(np.ravel(data_new[nsub]), bins, color='green') 
            plt.xlabel('pixel value (e-)')
            plt.ylabel('number')
            plt.title('subsection of '+new_fits)
            plt.show()
            plt.close()

            range_ref = (bkg_ref-5.*std_ref, bkg_ref+5.*std_ref)
            bins = np.linspace(range_ref[0], range_ref[1], 100)
            plt.hist(np.ravel(data_ref[nsub]), bins, color='green') 
            plt.xlabel('pixel value (e-)')
            plt.ylabel('number')
            plt.title('subsection of '+ref_fits)
            plt.show()
            plt.close()
            
        # replace low values in subimages
        data_new[nsub][data_new[nsub] <= 0.] = bkg_new
        data_ref[nsub][data_ref[nsub] <= 0.] = bkg_ref

        # replace low values in variance subimages
        #var_new[nsub][var_new[nsub] < std_new**2] = std_new**2
        #var_ref[nsub][var_ref[nsub] < std_ref**2] = std_ref**2

        if nfakestars>0:
            # add fake star(s) to new image
            
            s2n = fakestar_s2n

            # use function [flux_optimal_s2n] to estimate flux needed
            # for star with S/N of [fakestar_s2n]
            psf_orig_new[nsub] /= np.sum(psf_orig_new[nsub])
            fakestar_flux, fakestar_data = flux_optimal_s2n (psf_orig_new[nsub], bkg_new,
                                                             readnoise_new, s2n, fwhm_new)

            # for plot of input vs. output flux
            fakestar_flux_input[nsub] = fakestar_flux

            if verbose:
                print 'fakestar_flux: {} e-'.format(fakestar_flux)
                flux, fluxerr = flux_optimal(psf_orig_new[nsub], fakestar_data, bkg_new,
                                             readnoise_new)
                print 'recovered flux, fluxerr, S/N', flux, fluxerr, flux/fluxerr
                
                # check S/N with Eq. 51 from Zackay & Ofek 2017, ApJ, 836, 187
                print 'S/N check', get_s2n_ZO(psf_orig_new[nsub], fakestar_data, bkg_new,
                                              readnoise_new)

                flux, fluxerr = get_optflux_Naylor(psf_orig_new[nsub], fakestar_data, bkg_new,
                                                   fakestar_data+readnoise_new**2)
                print 'Naylor recovered flux, fluxerr, S/N', flux, fluxerr, flux/fluxerr
                
                
            # normalize psf_orig_new to contain fakestar_flux
            psf_fakestar = psf_orig_new[nsub] * (fakestar_flux/np.sum(psf_orig_new[nsub]))
            
            if nfakestars==1:
                # place it at the center of the new subimage
                xpos = xsize_fft/2
                ypos = ysize_fft/2
                data_new[nsub][ypos-psf_size_new/2:ypos+psf_size_new/2,
                               xpos-psf_size_new/2:xpos+psf_size_new/2] += psf_fakestar
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
                    data_new[nsub][ypos-psf_size_new/2:ypos+psf_size_new/2,
                                   xpos-psf_size_new/2:xpos+psf_size_new/2] += psf_fakestar                
                
        if bkg_sex:
            # use background subimages
            bkg_new = data_new_bkg[nsub]
            bkg_ref = data_ref_bkg[nsub]

        # subtract the background
        data_new[nsub] -= bkg_new
        data_ref[nsub] -= bkg_ref

        # replace saturated pixel values with zero
        #data_new[nsub][data_new[nsub] > 0.95*satlevel_new] = 0.
        #data_ref[nsub][data_ref[nsub] > 0.95*satlevel_ref] = 0.

        # get median fratio from PSFex stars across subimage
        subcut = cuts_ima[nsub]
        index_sub = ((y_fratio > subcut[0]) & (y_fratio < subcut[1]) & 
                     (x_fratio > subcut[2]) & (x_fratio < subcut[3]))

        # take local or full-frame values for fratio
        if fratio_local and any(index_sub):
            # replace the full-frame values defined before the nsub loop
            fratio_mean, fratio_std, fratio_median = clipped_stats(fratio[index_sub])
        # and the same for dx and dy
        if dxdy_local and any(index_sub):
            dx_sub = np.sqrt(np.median(dx[index_sub])**2 + np.std(dx[index_sub])**2)
            dy_sub = np.sqrt(np.median(dy[index_sub])**2 + np.std(dy[index_sub])**2)
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

            
        # call Barak's function
        data_D, data_S, data_Scorr, data_Fpsf, data_Fpsferr = run_ZOGY(data_ref[nsub], data_new[nsub], 
                                              psf_ref[nsub], psf_new[nsub], 
                                              std_ref, std_new, 
                                              f_ref, f_new,
                                              var_ref[nsub], var_new[nsub],
                                              dx_sub, dy_sub)

        # check that robust std of Scorr is around unity
        if verbose:
            mean_Scorr, std_Scorr, median_Scorr = clipped_stats(data_Scorr, clip_zeros=False)
            print 'mean_Scorr, median_Scorr, std_Scorr', mean_Scorr, median_Scorr, std_Scorr

        # if fake star(s) was (were) added to the subimages, compare
        # the input flux (the same for the entire subimage) with the
        # PSF flux determined by run_ZOGY. If multiple stars were
        # added, then this comparison is done for the last of them.
        if nfakestars>0:
            fakestar_flux_output[nsub] = data_Fpsf[xpos, ypos]
            fakestar_fluxerr_output[nsub] = data_Fpsferr[xpos, ypos]
            # and S/N from Scorr
            fakestar_s2n_output[nsub] = data_Scorr[xpos, ypos]

        # put sub images into output frames
        subcut = cuts_ima[nsub]
        fftcut = cuts_fft[nsub]
        y1 = subimage_border
        x1 = subimage_border
        y2 = subimage_border+subimage_size
        x2 = subimage_border+subimage_size
        data_D_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_D[y1:y2,x1:x2] / gain_new
        data_S_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_S[y1:y2,x1:x2]
        data_Scorr_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_Scorr[y1:y2,x1:x2]
        data_Fpsf_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_Fpsf[y1:y2,x1:x2]
        data_Fpsferr_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_Fpsferr[y1:y2,x1:x2]

        if bkg_sex:
            data_new_bkg_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = bkg_new[y1:y2,x1:x2] / gain_new
            data_ref_bkg_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = bkg_ref[y1:y2,x1:x2] / gain_ref
        else:
            data_new_bkg_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = bkg_new / gain_new
            data_ref_bkg_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = bkg_ref / gain_ref
            
        if nfakestars>0:
            if bkg_sex:
                  data_new_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = (data_new[nsub][y1:y2,x1:x2]
                                                                            + bkg_new[y1:y2,x1:x2]) / gain_new
                  data_ref_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = (data_ref[nsub][y1:y2,x1:x2]
                                                                           + bkg_ref[y1:y2,x1:x2]) / gain_ref
            else:
                data_new_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = (data_new[nsub][y1:y2,x1:x2]
                                                                           + bkg_new) / gain_new
                data_ref_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = (data_ref[nsub][y1:y2,x1:x2]
                                                                           + bkg_ref) / gain_ref      
        
        if display and (nsub == 65 or nsub==0):
            # just for displaying purpose:
            pyfits.writeto('D.fits', data_D, clobber=True)
            pyfits.writeto('S.fits', data_S, clobber=True)
            pyfits.writeto('Scorr.fits', data_Scorr, clobber=True)
            pyfits.writeto('Scorr_abs.fits', np.abs(data_Scorr), clobber=True)
            #pyfits.writeto('Scorr_1sigma.fits', data_Scorr_1sigma, clobber=True)
        
            # write new and ref subimages to fits
            subname = '_sub'+str(nsub)
            newname = base_new+'_wcs'+subname+'.fits'
            pyfits.writeto(newname, data_new[nsub]+bkg_new, clobber=True)
            refname = base_ref+'_wcs'+subname+'.fits'
            pyfits.writeto(refname, data_ref[nsub]+bkg_ref, clobber=True)
            # variance images
            pyfits.writeto('Vnew.fits', var_new[nsub], clobber=True)
            pyfits.writeto('Vref.fits', var_ref[nsub], clobber=True)
            
            # and display
            cmd = ['ds9','-zscale',newname,refname,'D.fits','S.fits','Scorr.fits']
            cmd = ['ds9','-zscale',newname,refname,'D.fits','S.fits','Scorr.fits',
                   'Vnew.fits', 'Vref.fits', 'VSn.fits', 'VSr.fits', 
                   'VSn_ast.fits', 'VSr_ast.fits', 'Sn.fits', 'Sr.fits', 'kn.fits', 'kr.fits']
            result = call(cmd)

        if timing: print 'wall-time spent in nsub loop', time.time()-tloop
    
    if timing:        
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
        pyfits.writeto('new.fits', data_new_full, header_new, clobber=True)
        pyfits.writeto('ref.fits', data_ref_full, header_ref, clobber=True)
    pyfits.writeto('D.fits', data_D_full, clobber=True)
    pyfits.writeto('S.fits', data_S_full, clobber=True)
    pyfits.writeto('Scorr.fits', data_Scorr_full, clobber=True)
    pyfits.writeto('Scorr_abs.fits', np.abs(data_Scorr_full), clobber=True)
    pyfits.writeto('Fpsf.fits', data_Fpsf_full, clobber=True)
    pyfits.writeto('Fpsferr.fits', data_Fpsferr_full, clobber=True)
    pyfits.writeto(base_new+'_bkg.fits', data_new_bkg_full, clobber=True)
    pyfits.writeto(base_ref+'_bkg.fits', data_ref_bkg_full, clobber=True)
    
    # make comparison plot of flux input and output
    if makeplots and nfakestars>0:
        #plt.axis((0,nsubs,0,2))
        x = np.arange(nsubs)+1
        y = (fakestar_flux_input - fakestar_flux_output) / fakestar_flux_input
        yerr = fakestar_fluxerr_output / fakestar_flux_input
        plt.errorbar(x, y, yerr=yerr, fmt='ko')
        plt.xlabel('subimage number')
        plt.ylabel('(true flux - ZOGY flux) / true flux')
        plt.title('fake stars true input flux vs. ZOGY Fpsf output flux')
        plt.savefig('fakestar_flux_input_vs_ZOGYoutput.pdf')
        plt.show()
        plt.close()

        mean, std, median = clipped_stats(y)
        print 'mean, std, median', mean, median, std
        print 'chi2', np.sum((fakestar_flux_input - fakestar_flux_output)**2 / fakestar_fluxerr_output**2)
        
        # same for S/N as determined by Scorr
        y = fakestar_s2n_output
        plt.plot(x, y, 'ko')
        plt.xlabel('subimage number')
        plt.ylabel('S/N from Scorr')
        plt.title('signal-to-noise ratio from Scorr')
        plt.savefig('fakestar_S2N_ZOGYoutput.pdf')
        plt.show()
        plt.close()
   
    # and display
    if display:
        if nfakestars>0:
            cmd = ['ds9','-zscale','new.fits','ref.fits','D.fits','S.fits','Scorr.fits',
               'Fpsf.fits', 'Fpsferr.fits']
        else:
            cmd = ['ds9','-zscale',new_fits,ref_fits_remap,'D.fits','S.fits','Scorr.fits',
               'Fpsf.fits', 'Fpsferr.fits']
        result = call(cmd)

################################################################################
    
def get_optflux (P, D, S, V):

    """Function that calculates optimal flux and corresponding error
    based on the PSF [P], data [D], sky [S] and variance [V].
    All are assumed to be in electrons rather than counts. These can
    be 1- or 2-dimensional lists, while the sky is also allowed to be
    a scalar. See Horne 1986, PASP, 98, 609 and Naylor 1998, MNRAS,
    296, 339."""

    # and optimal flux and its error
    denominator = np.sum(P**2/V)
    optflux = np.sum((P*(D-S)/V)) / denominator
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
    s2n = np.sqrt(np.sum( (D-S)**2 / V ))
    s2n = np.sqrt(np.sum( (T0*P)**2 / V ))

    return s2n

################################################################################

def flux_optimal (P, D, S, RON, nsigma=5):
    
    """Function that calculates optimal flux and corresponding error based
    on the PSF [P], data [D], sky [S] and read-out noise [RON].  This
    makes use of function [get_optflux].

    """

    # if S(ky) is a scalar, make it an array
    if np.isscalar(S):
        sky = np.ndarray(P.shape)
        sky[:] = S
        
    # replace negative values in D with the sky
    D[D<0] = sky[D<0]

    # mask with same shape as D; set all elements to True
    mask = np.ones(D.shape, dtype=bool)
    
    # loop
    for i in range(10):

        if i==0:
            # initial variance estimate (Eq. 12 from Horne 1986)
            V = RON**2 + D
        else:
            # improved variance (Eq. 13 from Horne 1986)
            V = RON**2 + flux_opt * P + sky
            
        # optimal flux
        flux_opt, fluxerr_opt = get_optflux(P[mask], D[mask], sky[mask], V[mask])    

        # reject any discrepant values
        index_rej = (((D - flux_opt * P - sky) / V) > nsigma**2)
        mask[index_rej] = False
        
    return flux_opt, fluxerr_opt
    

################################################################################

def flux_optimal_s2n (P, S, RON, s2n, fwhm=5.):
    
    """Similar to function [flux_optimal] above, but this function returns
    the total flux sum(D-S) required for the point source to have a
    particular signal-to-noise ratio [s2n]. This function is used to
    estimate the flux of the fake stars that are being added to the
    image with a required S/N [fakestar_s2n].

    """
    
    for i in range(10):
        if i==0:
            # initial estimate of variance (scalar)
            V = RON**2 + S
            # and flux (see Eq. 13 of Naylor 1998)
            flux = s2n * fwhm * np.sqrt(V) / np.sqrt(2*np.log(2)/np.pi)
        else:
            # estimate new flux based on fluxerr_opt of previous iteration
            flux = s2n * fluxerr_opt
            # improved estimate of variance (2D list)
            V = RON**2 + S + flux * P
            
        # estimate of D
        D = S + flux * P
        # and optimal flux
        flux_opt, fluxerr_opt = get_optflux(P, D, S, V)

        # break out of loop if S/N sufficiently close
        if abs(flux_opt/fluxerr_opt - s2n) / s2n < 1e-2:
            break
        
    return flux_opt, D
    

################################################################################

def clipped_stats(array, nsigma=3, max_iters=10, epsilon=1e-6, clip_upper10=False,
                  clip_zeros=True, get_median=True, get_mode=False, mode_binsize=0.1,
                  verbose=False, show_hist=False):
    
    # remove zeros
    if clip_zeros:
        array = np.copy(array[array.nonzero()])
        
    if clip_upper10:
        index_upper = int(0.9*len(array)+0.5)
        array = np.sort(np.flatten(array))[:index_upper]

    mean_old = float('inf')
    for i in range(max_iters):
        mean = array.mean()
        std = array.std()
        if verbose:
            print 'mean, std', mean, std
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
        plt.show()
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
        values.append(header[keywords[i]])
    return values

################################################################################
    
def prep_optimal_subtraction(input_fits, nsubs, imtype, fwhm, remap=None):

    print '\nexecuting prep_optimal_subtraction ...'
    t = time.time()
    
    # read in header and data; in case of the reference image, the
    # remapped image should be read into data, but the PSF
    # determination should be done using the image before
    # remapping
    read_fits = input_fits
    if imtype == 'ref':
        read_fits = input_fits.replace('.fits', '_remap.fits')
    if remap is not None:
        read_fits = remap
    with pyfits.open(read_fits) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data
    # replace NANs with zero, and +-infinity with large +-numbers
    data = np.nan_to_num(data)
    # get gain and readnoise from header
    gain = header['GAIN']
    readnoise = header['RDNOISE']
    # convert counts to electrons
    data *= gain

    # determine psf of input image with get_psf function
    psf, psf_orig = get_psf(input_fits, header, nsubs, imtype, fwhm)

    # read background image produced by sextractor
    if bkg_sex:
        bkg_fits = input_fits.replace('_wcs.fits', '_bkg.fits')
        with pyfits.open(bkg_fits) as hdulist:
            data_bkg = hdulist[0].data
        # convert counts to electrons
        data_bkg *= gain
    else:
        # return zero array with same shape as data
        # 
        data_bkg = np.zeros(data.shape)
        
    # split full image into subimages
    ysize, xsize = header['NAXIS2'], header['NAXIS1']
    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size, ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border
    
    fftdata = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    fftdata_bkg = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    for nsub in range(nsubs):
        subcutfft = cuts_ima_fft[nsub]
        fftcut = cuts_fft[nsub]
        fftdata[nsub][fftcut[0]:fftcut[1],fftcut[2]:fftcut[3]] = data[subcutfft[0]:subcutfft[1],
                                                                      subcutfft[2]:subcutfft[3]]
        fftdata_bkg[nsub][fftcut[0]:fftcut[1],fftcut[2]:fftcut[3]] = data_bkg[subcutfft[0]:subcutfft[1],
                                                                            subcutfft[2]:subcutfft[3]]
        
    if timing: print 'wall-time spent in prep_optimal_subtraction', time.time()-t
    return fftdata, psf, psf_orig, fftdata_bkg
    
                     
################################################################################

def get_psf(image, ima_header, nsubs, imtype, fwhm):

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
    # done inside Astrometry.net, producing the same catalog was an
    # independent SExtractor run would.
    sexcat = image.replace('.fits', '.sexcat')
    if (not os.path.isfile(sexcat) or redo) and dosex:
        result = run_sextractor(image, sexcat, sex_cfg, sex_par, fwhm=fwhm)
        
    # run psfex on SExtractor output catalog
    psfexcat = image.replace('.fits', '.psfexcat')
    if not os.path.isfile(psfexcat) or redo:
        print 'sexcat', sexcat
        print 'psfexcat', psfexcat
        result = run_psfex(sexcat, psfex_cfg, psfexcat)

    # again run SExtractor, but now using output PSF from PSFex, so
    # that PSF-fitting can be performed for all objects. The output
    # columns defined in [sex_par_psffit] include several new columns
    # related to the PSF fitting.
    if dosex_psffit:
        result = run_sextractor(image, sexcat+'_psffit', sex_cfg_psffit,
                                sex_par_psffit, fitpsf=True, fwhm=fwhm)

    # read in PSF output binary table from psfex
    psfex_bintable = image.replace('.fits', '.psf')
    with pyfits.open(psfex_bintable) as hdulist:
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
    # 4-5 pixels per FWHM) which is automatically determined by PSFex
    # (PSF_SAMPLING parameter in PSFex config file set to zero)
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    # if this is odd, make it even - for the moment this is because
    # the index range of the bigger image in which this psf is put
    # ([psf_ima_center]) assumes this is even
    if psf_size % 2 != 0:
        psf_size += 1
    # now change psf_samp slightly:
    psf_samp_update = float(psf_size) / float(psf_size_config)
    if imtype == 'new': psf_size_new = psf_size
    # [psf_ima] is the corresponding cube of PSF subimages
    psf_ima = np.ndarray((nsubs,psf_size,psf_size), dtype='float32')
    # [psf_ima_center] is [psf_ima] broadcast into images of xsize_fft
    # x ysize_fft
    psf_ima_center = np.ndarray((nsubs,ysize_fft,xsize_fft), dtype='float32')
    # [psf_ima_shift] is [psf_ima_center] shifted - this is
    # the input PSF image needed in the zogy function
    psf_ima_shift = np.ndarray((nsubs,ysize_fft,xsize_fft), dtype='float32')
    
    # loop through nsubs and construct psf at the center of each
    # subimage, using the output from PSFex that was run on the full
    # image
    for nsub in range(nsubs):
        
        x = (centers[nsub,1] - polzero1) / polscal1
        y = (centers[nsub,0] - polzero2) / polscal2

        
        if nsubs==1 or use_single_PSF:
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + \
                          data[3] * y + data[4] * x * y + data[5] * y**2
            elif poldeg==3:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 + \
                          data[4] * y + data[5] * x * y + data[6] * x**2 * y + \
                          data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3

        if display:
            # write this psf to fits
            pyfits.writeto('psf_'+imtype+'_sub'+str(nsub)+'.fits', psf_ima_config, clobber=True)
            #result = show_image(psf_ima_config)

        # resample PSF image at image pixel scale
        psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update)
        psf_ima[nsub] = psf_ima_resized
        if verbose and nsub==1:
            print 'psf_samp, psf_samp_update', psf_samp, psf_samp_update
            print 'np.shape(psf_ima_config)', np.shape(psf_ima_config)
            print 'np.shape(psf_ima)', np.shape(psf_ima)
            print 'np.shape(psf_ima_resized)', np.shape(psf_ima_resized)
            print 'psf_size ', psf_size
        if display:
            # write this psf to fits
            pyfits.writeto('psf_resized_'+imtype+'_sub'+str(nsub)+'.fits',
                           psf_ima_resized, clobber=True)
            #result = show_image(psf_ima_resized)

        # normalize to unity
        psf_ima_resized_norm = psf_ima_resized / np.sum(psf_ima_resized)
            
        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            print 'WARNING: image not even in both dimensions!'
            
        xcenter_fft, ycenter_fft = xsize_fft/2, ysize_fft/2
        if verbose and nsub==1:
            print 'xcenter_fft, ycenter_fft ', xcenter_fft, ycenter_fft
        psf_ima_center[nsub, ycenter_fft-psf_size/2:ycenter_fft+psf_size/2, 
                       xcenter_fft-psf_size/2:xcenter_fft+psf_size/2] = psf_ima_resized_norm

        if display:
            pyfits.writeto('psf_center_'+imtype+'_sub'+str(nsub)+'.fits',
                           psf_ima_center[nsub], clobber=True)            
            #result = show_image(psf_ima_center[nsub])

        # perform fft shift
        psf_ima_shift[nsub] = fft.fftshift(psf_ima_center[nsub])
        # Eran's function:
        #print np.shape(image_shift_fft(psf_ima_center[nsub], 1., 1.))
        #psf_ima_shift[nsub] = image_shift_fft(psf_ima_center[nsub], 0., 0.)

        #result = show_image(psf_ima_shift[nsub])

    if timing: print 'wall-time spent in get_psf', time.time() - t

    return psf_ima_shift, psf_ima

################################################################################

def get_fratio_radec(psfcat_new, psfcat_ref, sexcat_new, sexcat_ref):

    """Function that takes in output catalogs of stars used in the PSFex
    runs on the new and the ref image, and returns the arrays x, y (in
    the new frame) and fratios for the matching stars. In addition, it
    provides the difference in stars' RAs and DECs in arcseconds
    between the two catalogs.

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
        # read SExctractor fits table
        with pyfits.open(sexcat) as hdulist:
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
        dra = 3600.*(ra_new[i_new]-ra_ref)*np.cos(dec_ref[i_new]*np.pi/180.)
        ddec = 3600.*(dec_new[i_new]-dec_ref)
        dist = np.sqrt(dra**2 + ddec**2)
        # minimum distance and its index
        dist_min, i_ref = np.amin(dist), np.argmin(dist)
        if dist_min < 1.:
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

def centers_cutouts(subsize, ysize, xsize):

    """Function that determines the input image indices (!) of the centers
    (list of nsubs x 2 elements) and cut-out regions (list of nsubs x
    4 elements) of image with the size xsize x ysize. Subsize is the
    fixed size of the subimages, e.g. 512 or 1024. The routine will
    fit as many of these in the full frames, and will calculate the
    remaining subimages.

    """

    nxsubs = xsize / subsize
    nysubs = ysize / subsize
    if xsize % subsize != 0 and ysize % subsize != 0:
        nxsubs += 1
        nysubs += 1
        remainder = True
    else:
        remainder = False
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
        if i == nxsubs-1 and remainder:
            nx = xsize % subsize
        for j in range(nysubs):
            ny = subsize
            if j == nysubs-1 and remainder:
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

def run_wcs(image_in, image_out, ra, dec, gain, readnoise, fwhm):

    if timing: t = time.time()
    print '\nexecuting run_wcs ...'
    
    scale_low = 0.999 * pixscale
    scale_high = 1.001 * pixscale

    # round fwhm to 2 decimals
    fwhm = float('{:.2f}'.format(fwhm))
    # determine seeing
    seeing = fwhm * pixelscale
    
    sexcat = image_out.replace('.fits','.sexcat')

    # if psf_sampling is zero, scale the size of the VIGNET output
    # in the output catalog with [psf_radius]*[fwhm]
    if psf_sampling == 0.:
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
        # point sex_par to _temp file
        if verbose:
            print 'VIGNET size:', size_vignet_str
    # if psf_sampling is non-zero, the default VIGNET size is used: (99,99)
    else:
        sex_par_temp = sex_par
            
    #scampcat = image_in.replace('.fits','.scamp')
    cmd = ['solve-field', '--no-plots', '--no-fits2fits',
           '--sextractor-config', sex_cfg,
           '--x-column', 'XWIN_IMAGE', '--y-column', 'YWIN_IMAGE',
           '--sort-column', 'FLUX_AUTO',
           '--no-remove-lines',
           '--keep-xylist', sexcat,
           #'--scamp', scampcat,
           image_in,
           '--tweak-order', str(3), '--scale-low', str(scale_low),
           '--scale-high', str(scale_high), '--scale-units', 'app',
           '--ra', str(ra), '--dec', str(dec), '--radius', str(2.),
           '--new-fits', image_out, '--overwrite']

    # prepare aperture radii string 
    apphot_diams = np.array(apphot_radii) * 2 * fwhm
    apphot_diams_str = ",".join(apphot_diams.astype(str))
    if verbose:
        print 'aperture diameters used for PHOT_APERTURES', apphot_diams_str
    
    cmd_sex = 'sex -SEEING_FWHM '+str(seeing)+' -PARAMETERS_NAME '+sex_par_temp+' -PHOT_APERTURES '+apphot_diams_str
    
    if bkg_sex:
        bkg = image_in.replace('.fits','_bkg.fits')
        obj = image_in.replace('.fits','_obj.fits')
        cmd_sex += ' -CHECKIMAGE_TYPE BACKGROUND,OBJECTS -CHECKIMAGE_NAME '+bkg+','+obj
        
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
        with pyfits.open(radecfile) as hdulist:
            data_newradec = hdulist[1].data
        newra = data_newradec['RA']
        newdec = data_newradec['DEC']

    # convert SIP header keywords from Astrometry.net to PV keywords
    # that swarp, scamp (and sextractor) understand using this module
    # from David Shupe:
    sip_to_pv(image_out, image_out, tpv_format=False)

    # read data from SExtractor catalog produced in Astrometry.net
    with pyfits.open(sexcat) as hdulist:
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
    with pyfits.open(image_out) as hdulist:
        header_wcsimage = hdulist[0].header

    # add header of .axy extension as the SExtractor keywords are there,
    # although PSFex only seems to use 2 of them: SEXGAIN and SEXBKDEV.
    # Astrometry.net does not provide these values (zeros), so their
    # values need to be set.
    axycat = image_in.replace('.fits','.axy')
    with pyfits.open(axycat) as hdulist:
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
    col1 = pyfits.Column(name='Field Header Card', array=ext2_data, format=formatstr)
    cols = pyfits.ColDefs([col1])
    ext2 = pyfits.BinTableHDU.from_columns(cols)
    # make sure these keywords are in the header
    ext2.header['EXTNAME'] = 'LDAC_IMHEAD'
    ext2.header['TDIM1'] = '(80, {0})'.format(len(ext2_str)/80)

    # simply create extension 3 from [data4ext3]
    ext3 = pyfits.BinTableHDU(data4ext3)
    # extname needs to be as follows
    ext3.header['EXTNAME'] = 'LDAC_OBJECTS'

    # sort output table by number column if needed
    if doSort:
        index_sort = np.argsort(ext3.data['NUMBER'])
        ext3.data = ext3.data[index_sort]
    
    # create primary HDU
    prihdr = pyfits.Header()
    prihdu = pyfits.PrimaryHDU(header=prihdr)
    prihdu.header['EXPTIME'] = header4ext2['EXPTIME']
    prihdu.header['FILTNAME'] = header4ext2['FILTNAME']
    prihdu.header['SEEING'] = header4ext2['SEEING']
    prihdu.header['BKGSIG'] = header4ext2['SEXBKDEV']

    
    # write hdulist to output LDAC fits table
    hdulist = pyfits.HDUList([prihdu, ext2, ext3])
    hdulist.writeto(fits_ldac_out, clobber=True)
    hdulist.close()
    
################################################################################
    
def run_remap(image_new, image_ref, image_out,
              image_out_size, gain, config=swarp_cfg):
        
    """Function that remaps [image_ref] onto the coordinate grid of
       [image_new] and saves the resulting image in [image_out] with
       size [image_size].
    """

    if timing: t = time.time()
    print '\nexecuting run_remap ...'

    # read headers
    t = time.time()
    with pyfits.open(image_new) as hdulist:
        header_new = hdulist[0].header
    with pyfits.open(image_ref) as hdulist:
        header_ref = hdulist[0].header
        
    # create .head file with header info from [image_new]
    header_out = header_new[:]
    # copy some keywords from header_ref
    for key in ['EXPTIME','SATURATE','GAIN','RDNOISE','SEEING']:
        header_out[key] = header_ref[key]
    # delete some others
    for key in ['WCSAXES','NAXIS1', 'NAXIS2']:
        del header_out[key]
    # write to .head file
    with open(image_out.replace('.fits','.head'),'w') as newrefhdr:
        for card in header_out.cards:
            newrefhdr.write(str(card)+'\n')

    size_str = str(image_out_size[1]) + ',' + str(image_out_size[0]) 
    cmd = ['swarp', image_ref, '-c', config, '-IMAGEOUT_NAME', image_out, 
           '-IMAGE_SIZE', size_str, '-GAIN_DEFAULT', str(gain)]
    result = call(cmd)
    
    if timing: print 'wall-time spent in run_remap', time.time()-t

################################################################################

def run_sextractor(image, cat_out, file_config, file_params, fitpsf=False,
                   fraction=1.0, fwhm=5.0, detect_thresh=10.0, fwhm_frac=0.25,
                   fwhm_class_sort=False):

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
        with pyfits.open(image) as hdulist:
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
        pyfits.writeto(image_fraction, data_fraction.astype(np.float32), header, clobber=True)

        # make image point to image_fraction
        image = image_fraction
        cat_out = cat_out+'_fraction'


    # the input fwhm determines the SEEING_FWHM (important for
    # star/galaxy separation) and the radii of the apertures used for
    # aperture photometry. If fwhm is not provided as input, it will
    # assume fwhm=5.0 pixels.
    fwhm = float('{:.2f}'.format(fwhm))
    # determine seeing
    seeing = fwhm * pixelscale
    # prepare aperture diameter string to provide to SExtractor 
    apphot_diams = np.array(apphot_radii) * 2 * fwhm
    apphot_diams_str = ",".join(apphot_diams.astype(str))
    
    # run sextractor from the unix command line
    cmd = ['sex', image, '-c', file_config, '-CATALOG_NAME', cat_out, 
           '-PARAMETERS_NAME', file_params, '-PIXEL_SCALE', str(pixelscale),
           '-SEEING_FWHM', str(seeing),'-PHOT_APERTURES',apphot_diams_str]

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

def get_fwhm (cat_ldac, fraction, class_Sort = False):

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

    with pyfits.open(cat_ldac) as hdulist:
        data = hdulist[2].data

    # these arrays correspond to objecst with flag==0 and flux_auto>0.
    index = (data['FLAGS']==0) & (data['FLUX_AUTO']>0.)
    fwhm = data['FWHM_IMAGE'][index]
    class_star = data['CLASS_STAR'][index]
    flux_auto = data['FLUX_AUTO'][index]
    mag_auto = -2.5*np.log10(flux_auto)

    if class_Sort:
        # sort by CLASS_STAR
        index_sort = np.argsort(class_star)
    else:
        # sort by FLUX_AUTO
        index_sort = np.argsort(flux_auto)

    # select fraction of targets
    index_select = np.arange(-np.int(len(index_sort)*fraction+0.5),-1)
    fwhm_select = fwhm[index_sort][index_select] 

    # print warning if few stars are selected
    if len(fwhm_select) < 10:
        print 'WARNING: fewer than 10 objects are selected for FWHM determination'
    
    # determine mean, median and standard deviation through sigma clipping
    fwhm_mean, fwhm_std, fwhm_median = clipped_stats(fwhm_select)
    if verbose:
        print 'catalog', cat_ldac
        print 'fwhm_mean, fwhm_median, fwhm_std', fwhm_mean, fwhm_median, fwhm_std
            
    if makeplots:

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
        plt.show()
        plt.close()

    if timing: print 'wall-time spent in get_fwhm', time.time()-t
    return fwhm_median, fwhm_std


################################################################################

def run_psfex(cat_in, file_config, cat_out):
    
    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""

    if timing: t = time.time()

    if psf_sampling == 0:
        # provide new PSF_SIZE based on psf_radius, which is 2 *
        # [psf_radius] * FWHM / sampling factor. The sampling factor is
        # automatically determined in PSFex, and is such that FWHM /
        # sampling factor ~ 4-5, so:
        size = np.int(psf_radius*9+0.5)
        # make sure it's odd
        if size % 2 == 0: size += 1
        psf_size_config = str(size)+','+str(size)
    else:
        # use some reasonable default size
        psf_size_config = '45,45'

    if verbose:
        print 'psf_size_config', psf_size_config
        
    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config, '-PSF_SAMPLING', str(psf_sampling)]
    result = call(cmd)    

    if timing: print 'wall-time spent in run_psfex', time.time()-t
    
################################################################################

def run_ZOGY(R,N,Pr,Pn,sr,sn,fr,fn,Vr,Vn,dx,dy):

# edited Barak's original code to include variances sigma_n**2 and
# sigma_r**2 (see Eq. 9, here sn and sr) and Fn and Fr which are
# assumed to be unity in Barak's code.
    
    if timing: t = time.time()

    R_hat = fft.fft2(R)
    N_hat = fft.fft2(N)
    Pn_hat = fft.fft2(Pn)
    Pn_hat2_abs = np.abs(Pn_hat**2)
    Pr_hat = fft.fft2(Pr)
    Pr_hat2_abs = np.abs(Pr_hat**2)

    sn2 = sn**2
    sr2 = sr**2
    #beta = fn/fr
    #beta2 = beta**2
    fn2 = fn**2
    fr2 = fr**2
    fD = fr*fn / np.sqrt(sn2*fr2+sr2*fn2)
    
    denominator = sn2*fr2*Pr_hat2_abs + sr2*fn2*Pn_hat2_abs
    #denominator_beta = sn2*Pr_hat2_abs + beta2*sr2*Pn_hat2_abs

    D_hat = (fr*Pr_hat*N_hat - fn*Pn_hat*R_hat) / np.sqrt(denominator)
    # alternatively using beta:
    #D_hat = (Pr_hat*N_hat - beta*Pn_hat*R_hat) / np.sqrt(denominator_beta)

    D = np.real(fft.ifft2(D_hat)) / fD
    
    P_D_hat = (fr*fn/fD) * (Pr_hat*Pn_hat) / np.sqrt(denominator)
    #alternatively using beta:
    #P_D_hat = np.sqrt(sn2+beta2*sr2)*(Pr_hat*Pn_hat) / np.sqrt(denominator_beta)

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
    dSndy = Sn - np.roll(Sn,1,axis=1)
    dSndx = Sn - np.roll(Sn,1,axis=0)
    VSn_ast = dx2 * dSndx**2 + dy2 * dSndy**2
    
    Sr = np.real(fft.ifft2(kr_hat*R_hat))
    dSrdy = Sr - np.roll(Sr,1,axis=1)
    dSrdx = Sr - np.roll(Sr,1,axis=0)
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
        pyfits.writeto('kr.fits', np.real(kr), clobber=True)
        pyfits.writeto('kn.fits', np.real(kn), clobber=True)
        pyfits.writeto('Sr.fits', Sr, clobber=True)
        pyfits.writeto('Sn.fits', Sn, clobber=True)
        pyfits.writeto('VSr.fits', VSr, clobber=True)
        pyfits.writeto('VSn.fits', VSn, clobber=True)
        pyfits.writeto('VSr_ast.fits', VSr_ast, clobber=True)
        pyfits.writeto('VSn_ast.fits', VSn_ast, clobber=True)

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
    
    if timing: print 'wall-time spent in optimal subtraction', time.time()-t

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
    parser.add_argument('ref_fits_remap', default=None, help='remapped ref imaage')
    parser.add_argument('telescope', default=None, help='telescope')
    parser.add_argument('log', default=None, help='help')
    parser.add_argument('subpipe', default=False, help='subpipe')
    args = parser.parse_args()
    optimal_subtraction(args.new_fits, args.ref_fits, args.ref_fits_remap, args.telescope, args.log)
        
if __name__ == "__main__":
    main()
