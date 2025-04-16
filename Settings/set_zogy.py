import os

# folder with calibration files such as the real/bogus machine
# learning model and (Gaia) photometric calibration and forced
# photometry input catalogs referenced further down below; use
# environment variable MLBG_CALDIR if that is defined
cal_dir = os.environ.get('MLBG_CALDIR')
if cal_dir is None:
    # in case MLBG_CALDIR environment variable is not set, use:
    cal_dir = '/idia/projects/meerlicht/CalFiles'


# string to append to input filename(s) before the .fits extension to
# use for all output products; if left empty '' or None, the input
# images will be overwritten unless they are compressed files
output_str2add = None


#===============================================================================
# ZOGY
#===============================================================================
# the full image will be subdivided in a number of square subimages
# defined by [subimage_size]; the FFTs for the image subtraction will
# be performed on images with size [subimage_size]+2xsubimage_border,
# where only the central [subimage_size] pixels will be used for the
# full output image
# CHECK!!!
#subimage_size = 660      # size of subimages
#subimage_border = 30     # border around subimage to avoid edge effects
subimage_size = 1320      # size of subimages
subimage_border = 40      # border around subimage to avoid edge effects
shape_new = (10560, 10560) # shape new image, ref image can have any shape


# ZOGY parameters
fratio_local = False     # determine flux ratio (Fn/Fr) from subimage (T) or
                         # full frame (F)
fratio_optflux = True    # use optimal flux (T) or FLUX_AUTO (F) for flux ratio
dxdy_local = False       # determine dx,dy from subimage (T) or full frame (F)
transient_nsigma = 6     # required Scorr significance for transient detection
chi2_max = 10            # maximum reduced chi2 in PSF/Gauss fit to D to filter
                         # transients; float('inf') for infinity
chi2_snr_limit = 50      # transient signal-to-noise ratio limit above
                         # which the reduced chi2 cut is not applied
                         # (to avoid very bright targets possibly
                         # being falsely discarded based on a poor fit)


# transient with any FLAGS_MASK value (see parameter mask_value
# further down below) contained in transient_mask_discard are
# discarded; e.g. 63 implies that any transient containing at least
# one pixel in its inner profile with FLAGS_MASK 1 | 2 | 4 | 8 | 16 |
# 32 (=63) does not appear in transient catalog
transient_flagsmask_discard = 63 # = 1 | 2 | 4 | 8 | 16 | 32; allow crosstalk
# same for FLAGS; 255 implies discard transients with any non-zero
# Source Extractor FLAGS
transient_flags_discard = 255 # = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128


save_thumbnails = True   # save thumbnails of reduced image, remapped reference
                         # image and ZOGY products D and Scorr in transient catalog
size_thumbnails = 100    # size of square thumbnail arrays in (new) image pixels
orient_thumbnails = True # orient thumbnails in North up, East left orientation?


# switch to use old or new (~June 2024) transient catalog definition
use_new_transcat = {'ML1': False, 'BG': True}


# use "flattened" new image with subimage zeropoints (as defined with
# zp_nsubs_shape_new below) in optimal image subraction; if reference
# images have not been flattened (set with set_br.scale_zps), then
# (probably) best to set scale_zps_sub to False as well
scale_zps_sub = {'ML1': False, 'BG': True}


#===============================================================================
# Replacement of saturated pixels to avoid artefacts in image subtraction
#===============================================================================

# perform interpolation to replace saturated+connected pixels
interp_sat = True
# interpolating function; 'spline' (cubic) or 'gauss'
interp_func = 'spline'
# along pixel row (True) or column (False)
interp_along_row = True
# number of pixels to use on either side of saturated pixels
interp_dpix = 7

# replace saturated+connected pixels with PSF values; if [interp_sat]
# is also set, this PSF replacement will have precedence; only pixels
# within the PSF footprint will be replaced
replace_sat_psf = False
# maximum number of saturated pixels in object before PSF replacement
# is applied; if object is very saturated, PSF replacement will be
# less reliable
replace_sat_nmax = float('inf')


#===============================================================================
# Injection of fake stars/transients into new image
#===============================================================================
nfakestars = 0           # number of fake stars to be added to each subimage;
                         # first star is at the center, the rest (if any) is
                         # randomly distributed
fakestar_radec = (199.7506, -21.0621) # [deg] if defined, the first fake star is
                         # inserted at these coordinates
fakestar_s2n = 10.       # required signal-to-noise ratio of the fake stars


#===============================================================================
# Machine Learning real/bogus classification - trained on MeerLICHT data!
#===============================================================================
ML_calc_prob = True
# version to use: '1'=Zafiirah (obsolete); '2'=Diederik; '3'=Diederik August 2024
ML_version = '3'
# dictionary of ML_models corresponding to ML_version
ML_models = {
    #'1': '{}/meerCRAB_model/NET3_threshold_9_NRD'
    #.format(os.environ['MEERCRABHOME']),
    '2': '{}/model270123.h5'.format(cal_dir),
    '3': '{}/supervised_CNN.h5'.format(cal_dir)}


#===============================================================================
# Background
#===============================================================================
# background estimation: these are the optional methods to estimate the
# backbround and its standard deviation (STD):
# (1) background and STD/RMS map determined by SExtractor (fast)
# (2) improved background and STD map using masking of all sources (recommended)
bkg_method = 2           # background method to use
bkg_nsigma = 3           # data outside mean +- nsigma * stddev are
                         # clipped (method 2 only)
bkg_boxsize = 60         # size of region used to determine
                         # background (both methods)
bkg_filtersize = 3       # size of filter used for smoothing the above
                         # regions (both methods)


# these parameters are related to MeerLICHT/BlackGEM images only
MLBG_use2Dfit = True     # use 2D polynomial fit in background estimation
# apply channel correction factor estimated while
# performing a 2D polynomial fit to the background
MLBG_chancorr = {'ML1': True, 'BG': False}
MLBG_chancorr_limdev = 0.05 # single channel correction limiting deviation
                            # above which none of the corrections are applied


#===============================================================================
# Header keywords
#===============================================================================
# Definition of some required variables, or alternatively, their
# corresponding keyword names in the fits header of the input
# image(s). If it is defined, the variable value is adopted. If it is
# not defined, the header is checked for the keyword defined, and if
# it exists, the keyword value is adopted. If neither variable nor
# header keyword exists, an error is raised. Note that this checking
# of the fits header only goes for these specific header keyword
# variables; this is not done for other variables defined in this
# file/module.
key_naxis1 = 'NAXIS1'
key_naxis2 = 'NAXIS2'
key_gain = 'GAIN'
gain = 1.0
key_satlevel = 'SATURATE'
key_ra = 'RA'
key_dec = 'DEC'
key_pixscale = 'A-PSCALE'
pixscale = 0.564	# in arcsec/pixel
key_exptime = 'EXPTIME'
key_filter = 'FILTER'
key_obsdate = 'DATE-OBS'
key_rdnoise = 'RDNOISE'


#===============================================================================
# Initial seeing estimate
#===============================================================================
fwhm_imafrac = 0.90      # fraction of image area that will be used
                         # for initial seeing estimate
fwhm_detect_thresh = 20  # detection threshold for fwhm SExtractor run
fwhm_class_sort = False  # sort objects according to CLASS_STAR (T)
                         # or by FLUX_AUTO (F)
fwhm_frac = 0.33         # fraction of objects, sorted in brightness
                         # or class_star, used for fwhm estimate


#===============================================================================
# PSF parameters
#===============================================================================
psf_poldeg = 2           # polynomial degree PSF across image (0=constant PSF)
psf_clean_factor = 0     # pixels with values below (PSF peak * this
                         # factor) are set to zero; if this parameter
                         # is zero, no cleaning is done
psf_rad_phot = 5.0       # PSF radius in units of FWHM used in optimal photometry
psf_rad_zogy = 4.0       # PSF radius in units of FWHM used in optimal subtraction
psf_sampling = 0.0       # PSF sampling step in image pixels used in PSFex
                         # If zero, it is automatically determined for the
                         # new and ref image as follows:
                         #    psf_sampling = FWHM * [psf_samp_fwhmfrac]
                         # If non-zero, its value is adopted for the sampling
                         # step in both images.
psf_samp_fwhmfrac = 1/4.5 # PSF sampling step in units of FWHM
                         # this is only used if [psf_sampling]=0.
#size_vignet = 99         # size of the square VIGNETs saved in the SExtractor
size_vignet = {'ML1': 99, 'BG': 49} # size of the square VIGNETs saved in the
                         # SExtractor LDAC catalog used by PSFEx; its value should
                         # be set to ~ 2 * max(psf_rad_phot,psf_rad_zogy) *
                         # maximum expected FWHM in any of the images.
psf_stars_s2n_min = 20   # minimum signal-to-noise ratio for PSF stars
                         # (do not set this too high as otherwise the PSF
                         #  will be mainly based on bright stars)
psf_stars_nmax = 15000   # maximum number of stars to provide as input to PSFEx,
                         # in case the image is very crowded


#===============================================================================
# Astrometry
#===============================================================================
# WCS
skip_wcs = False         # skip Astrometry.net step if image already
                         # contains a reliable WCS solution
# Astrometry.net's tweak order
astronet_tweak_order = 3
# Astrometry.net configuration file that also defines the location of
# the index files
astronet_config = '{}/astrometry/astrometry.cfg'.format(cal_dir)
# only search in Astrometry.net index files within this radius of the
# header RA and DEC [deg]
astronet_radius = 30.
pixscale_varyfrac = 0.0015 # pixscale solution found by Astrometry.net will
                           # be within this fraction of the assumed pixscale
# solve-field depth parameter step; this determines how many of the
# brightest sources are used to find a solution before moving on to
# the next index file; if no solution is found for any index file, it
# will use the next batch of [astronet_depth_step] sources up to
# [ast_nbright]/3
astronet_depth_step = 50


# calibration catalog used for photometry and to check the astrometry
#cal_cat = ('{}/CalFiles/ML_calcat_kur_allsky_ext1deg_20181115.fits'
#           .format(os.environ['ZOGYHOME']))
#cal_epoch = 2015.5
# new catalog based on DR3
#cal_cat = ('{}/GaiaDR3_calcat_MLBG_HP3_highPM_g10-17_HPfine12.fits'
#           .format(cal_dir))
cal_cat = ('{}/GaiaDR3_calcat_MLBG_HP3_highPM_g10-17_HPfine12'
           '_Cstar_blendcont_npick2.fits'.format(cal_dir))
cal_epoch = 2016.0

ast_nbright = 1500       # brightest no. of objects in the field to
                         # use for astrometric solution and crosscheck
                         # of positions obtained against calibration
                         # catalog


#===============================================================================
# Photometry
#===============================================================================

# if force_phot_gaia is True, forced photometry will be performed on
# all sources from the input catalog - provided in parameter gaia_cat
# - whose positions are within the input image(s); proper motion is
# taken into account if columns pmra and pmdec are available in
# gaia_cat and adopting gaia_epoch for the input catalog EPOCH. For
# quicker reading, the catalog is assumed to be split into multiple
# extensions, where the extensions contain the entries corresponding
# to healpixel of particular level (infer from name or LEVEL header
# keyword), where nside=2**level and n_healpix = 12*nside**2 =
# 12*4**level. The output catalog will contain the same number of
# entries as the number of sources in gaia_cat within the image(s).
#
# if force_phot_gaia is False, the number of sources is determined by
# the significant sources detected by Source Extractor, where the
# threshold can be adjusted in the configuration file (see sex_cfg
# further below)
force_phot_gaia = {'ML1': False, 'BG': True}
gaia_cat = '{}/GaiaDR3_all_HP4_highPM.fits'.format(cal_dir)
gaia_epoch = 2016.0

# in case of Gaia forced photometry (force_phot_gaia is True): perform
# the photometry on image sources in order of Gaia magnitude
# brightness and remove the inferred PSF from the image before
# continuing with the next source
remove_psf = True


# switch to record fluxes in microJy instead of AB mags in full-source
# catalog; only relevant if force_phot_Gaia is True
record_fnu = {'ML1': False, 'BG': True}


# aperture photometry radii in units of FWHM
apphot_radii = [0.66, 1.5, 5]

# use local or global background for photometry
bkg_phototype = 'local'

# in case bkg_phototype = 'local':
# if force_phot_gaia=False: aperture photometry is performed by source
# extractor with thickness of the background LOCAL annulus:
bkg_photothick = 24

# if force_phot_gaia=True: apeture photometry is performed using
# function [get_apflux] with the following parameters: background
# annulus radii in units of FWHM
bkg_radii = [5,7]
# discard pixels in the local background annulus affected by objects
# (stars or galaxies, as detected by source extractor) to improve the
# local background determination; if set to False, all pixels that
# have not been identified as bad/saturated/etc. in the input mask
# will be used
bkg_objmask = True
# if the fraction of background annulus pixels affected by objects
# and/or masked pixels is higher than [bkg_limfrac], the global
# background is adopted instead of the local one
bkg_limfrac = 0.5


# integer subsampling factor to be able to sum over fraction of a
# pixel; a value of n will split a pixel into n**2 subpixels
apphot_fzoom = 5


source_nsigma = 5         # signal-to-noise ratio used in calculating
                          # limiting magnitudes and - only in case
                          # [force_phot_gaia] is False - for source to be
                          # included in output catalog
source_minpixfrac = 0.67  # required fraction of good pixels in inner/central
                          # footprint for source to have non-zero flux
                          # in output catalog
inner_psflim = 0.01       # PSF profile pixels with values above this limit
                          # times the peak value are considered part of the
                          # central/inner PSF profile


# PSF fitting
psffit = False            # perform PSF fitting using own function
psffit_sex = False        # perform PSF fitting using SExtractor


# Photometric calibration
# telescope latitude in degrees (North); adopting the (middle)
# longitude of BG3 for all three BGs; see email from PaulG on 19 March
# 2019, 0.0001 degrees is accurate to 5-10m
obs_lat = {'ML1': -32.3799, 'BG': -29.2575}
# telescope longitude in degrees (East)
obs_lon = {'ML1': 20.8112, 'BG': -70.7380}
# telescope height in meters above sealevel
obs_height = {'ML1': 1802, 'BG': 2383}
# observatory time zone (see /usr/share/zoneinfo); BlackGEM: 'America/Santiago'
obs_timezone = {'ML1': 'Africa/Johannesburg', 'BG': 'America/Santiago'}
# these [ext_coeff] are mean extinction estimates for Sutherland in
# the MeerLICHT filters:
ext_coeff = {'ML1': {'u':0.52, 'g':0.23, 'q':0.15, 'r':0.12, 'i':0.08, 'z':0.06},
             # and the same for La Silla in the BlackGEM filters:
             'BG':  {'u':0.38, 'g':0.16, 'q':0.09, 'r':0.07, 'i':0.02, 'z':0.01}}

# name of the photometric calibration catalog (in binary fits format)
# with the stars' magnitudes converted to the same filter(s) as the
# observations (in this case the MeerLICHT/BlackGEM filter set):
# this is now the same as the astrometric catalog: [cal_cat] defined above


# default zeropoints used if no photometric calibration catalog is
# provided or a particular field does not contain any calibration stars
zp_default = {'ML1': {'u':22.4, 'g':23.3, 'q':23.8, 'r':22.9, 'i':22.3, 'z':21.4},
              'BG':  {'u':22.4, 'g':23.3, 'q':23.8, 'r':22.9, 'i':22.3, 'z':21.4}}


# number of subimages, along the Y- and X-axis, that determines the
# scale at which the zeropoints are determined and applied; (1,1)
# means the full-image zeropoint is used, whereas e.g. (2,4) for an
# image with shape (1024,2048), the zeropoints are determined on
# subimages with shape (512,512); the zeropoint subimage sizes should
# fit integer times into the full image
zp_nsubs_shape_new = {'ML1': (1,1), 'BG': (16,16)}
zp_nsubs_shape_ref = {'ML1': (1,1), 'BG': (1,1)}

# minimum number of non-saturated stars required (after sigma
# clipping) to determine zeropoint(s) at any resolution
phot_ncal_min = 7


#===============================================================================
# Configuration
#===============================================================================

# path and names of configuration files
cfg_dir = os.environ['ZOGYHOME']+'/Config/'
sex_cfg = cfg_dir+'sex.config'               # SExtractor configuration file
sex_cfg_psffit = cfg_dir+'sex_psffit.config' # same for PSF-fitting version
sex_det_filt = cfg_dir+'default.conv'        # SExtractor detection filter file
sex_par = cfg_dir+'sex.params'               # SExtractor output parameters file
sex_par_psffit = cfg_dir+'sex_psffit.params' # same for PSF-fitting version
sex_par_ref = cfg_dir+'sex_ref.params'       # same for ref image output version
psfex_cfg = cfg_dir+'psfex.config'           # PSFex configuration file
swarp_cfg = cfg_dir+'swarp.config'           # SWarp configuration file

# if a mask image is provided, the mask values can be associated to
# the type of masked pixel with this dictionary:
mask_value = {'bad': 1, 'cosmic ray': 2, 'saturated': 4,
              'saturated-connected': 8, 'satellite trail': 16, 'edge': 32,
              'crosstalk': 64}

# definition of flags recorded in FLAGS_OPT column in output catalog
flags_opt_dict = {
    'bkg_annulus':        0, # sky background determined from annulus
    'bkg_localfit':       1, # sky determined by performing local fit
    'bkg_global':         2, # local fit unsuccessful, using global sky
    'calstar':            4, # entry in catalog is a calibration star
    'source_minpixfrac':  8, # too few inner PSF pixels to infer flux_opt
    'exception':         16, # exception occurred in flux_opt determination
}


# subfolder to save the many temporary numpy binary files
dir_numpy = 'NumpyFiles'

# switch to keep intermediate/temporary files
keep_tmp = False

# switch on/off different functions
redo_new = False         # execute SExtractor, astrometry.net, PSFEx, optimal flux
redo_ref = False         # determination even if new/ref products already present
verbose = True           # print out extra info
timing = True            # (wall-)time the different functions
display = False          # show intermediate fits images (centre and 4 corners)
make_plots = False       # make diagnostic plots and save them as pdf
show_plots = False       # show diagnostic plots
