import os

#===============================================================================
# ZOGY
#===============================================================================
# the full image will be subdivided in a number of square subimages
# defined by [subimage_size]; the FFTs for the image subtraction will
# be performed on images with size [subimage_size]+2xsubimage_border,
# where only the central [subimage_size] pixels will be used for the
# full output image
subimage_size = 1320     # size of subimages
subimage_border = 40     # border around subimage to avoid edge effects
#subimage_size = 960      # size of subimages
#subimage_border = 32     # border around subimage to avoid edge effects

# ZOGY parameters
fratio_local = False     # determine flux ratio (Fn/Fr) from subimage (T) or full
                         # frame (F)
fratio_optflux = True    # use optimal flux (T) or FLUX_AUTO (F) for flux ratio
dxdy_local = False       # determine dx and dy from subimage (T) or full frame (F)
transient_nsigma = 6     # required significance in Scorr for transient detection
chi2_max = 5             # maximum reduced chi2 in PSF/Gauss fit to D to filter
                         # transients
                         
# maximum number of flagged pixels of particular type (corresponding
# to [mask_value] below) in the vicinity of the transient to filter
transient_mask_max = {'bad': 0, 'cosmic ray': 0, 'saturated': 0,
                      'saturated-connected': 0, 'satellite trail': 0, 'edge': 0}

save_thumbnails = True   # save thumbnails of reduced image, remapped reference
                         # image and ZOGY products D and Scorr in transient catalog
size_thumbnails = 100    # size of square thumbnail arrays in (new) image pixels
orient_thumbnails = True # orient thumbnails in North up, East left orientation?


# add optional fake stars for testing purposes
nfakestars = 0           # number of fake stars to be added to each subimage; first star
                         # is at the center, the rest (if any) is randomly distributed
fakestar_s2n = 10.       # required signal-to-noise ratio of the fake stars


#===============================================================================
# Machine Learning - trained on MeerLICHT data!
#===============================================================================
ML_calc_prob = True
ML_model = ('{}/meerCRAB_model/NET3_threshold_9_NRD'
            .format(os.environ['MEERCRABHOME']))


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
MLBG_chancorr = True     # apply channel correction factor estimated while
                         # performing a 2D polynomial fit to the background
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
pixscale = 0.5642	# in arcsec/pixel
key_exptime = 'EXPTIME'
key_filter = 'FILTER'
key_obsdate = 'DATE-OBS'
key_rdnoise = 'RDNOISE'


#===============================================================================
# Initial seeing estimate
#===============================================================================
fwhm_imafrac = 0.9       # fraction of image area that will be used
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
psf_rad_zogy = 5.0       # PSF radius in units of FWHM used in optimal subtraction
psf_sampling = 0.0       # PSF sampling step in image pixels used in PSFex
                         # If zero, it is automatically determined for the
                         # new and ref image as follows:
                         #    psf_sampling = FWHM * [psf_samp_fwhmfrac]
                         # If non-zero, its value is adopted for the sampling
                         # step in both images.
psf_samp_fwhmfrac = 1/4.5 # PSF sampling step in units of FWHM
                         # this is only used if [psf_sampling]=0.
size_vignet = 99         # size of the square VIGNETs saved in the SExtractor
                         # LDAC catalog used by PSFEx; its value should be set to
                         # ~ 2 * max(psf_rad_phot,psf_rad_zogy) * maximum
                         # expected FWHM in any of the images.
psf_stars_s2n_min = 20   # minimum signal-to-noise ratio for PSF stars
                         # (don't set this too high as otherwise the PSF
                         #  will be mainly based on bright stars)


#===============================================================================
# Astrometry
#===============================================================================
# WCS
skip_wcs = False         # skip Astrometry.net step if image already
                         # contains a reliable WCS solution
# Astrometry.net's tweak order
astronet_tweak_order = 3
# only search in Astrometry.net index files within this radius of the
# header RA and DEC [deg]
astronet_radius = 30.
pixscale_varyfrac = 0.001  # pixscale solution found by Astrometry.net will
                           # be within this fraction of the assumed pixscale
# calibration catalog used for both astrometry and photometry

cal_cat = {'ML1': '{}/CalFiles/ML_calcat_kur_allsky_ext1deg_20181115.fits'
           .format(os.environ['ZOGYHOME']),
           'BG': '{}/CalFiles/ML_calcat_kur_allsky_ext1deg_20181115.fits'
           .format(os.environ['ZOGYHOME']),
           }

ast_nbright = 1000       # brightest no. of objects in the field to
                         # use for astrometric solution and crosscheck
                         # of positions obtained against calibration
                         # catalog
ast_filter = 'r'         # magnitude column to sort in brightness


#===============================================================================
# Photometry
#===============================================================================
# aperture radii in units of FWHM
apphot_radii = [0.66, 1.5, 5] # list of radii in units of FWHM used
                              # for aperture photometry in SExtractor

# PSF fitting
psffit = False                # perform PSF fitting using own function
psffit_sex = False            # perform PSF fitting using SExtractor

source_nsigma = 5             # required S/N in total flux (optimal or psffit)
                              # for source to be included in output catalog
                              # this also determines level of limiting flux
                              # and magnitudes of images
source_minpixfrac = 0.67      # required fraction of good pixels in footprint
                              # for source to be included in output catalog
                              
# Photometric calibration
# telescope latitude in degrees (North)
obs_lat = {'ML1': -32.3799, 'BG': -29.2575}  
# telescope longitude in degrees (East); BlackGEM: -70.73797
obs_lon = {'ML1': 20.8112, 'BG': -70.7380}
# telescope height in meters above sealevel; BlackGEM: 2343.
obs_height = {'ML1': 1803, 'BG': 2348}
# observatory time zone (see /usr/share/zoneinfo); BlackGEM: 'America/Santiago'
obs_timezone = {'ML1': 'Africa/Johannesburg', 'BG': 'America/Santiago'}
# these [ext_coeff] are mean extinction estimates for Sutherland in
# the MeerLICHT filters:
ext_coeff = {'ML1': {'u':0.52, 'g':0.23, 'q':0.15, 'r':0.12, 'i':0.08, 'z':0.06},
             'BG':  {'u':0.52, 'g':0.23, 'q':0.15, 'r':0.12, 'i':0.08, 'z':0.06}}
# and the same for La Silla in the BlackGEM filters:
#ext_coeff = {'u':0.38, 'g':0.16, 'q':0.09, 'r':0.07, 'i':0.02, 'z':0.01}
# name of the photometric calibration catalog (in binary fits format)
# with the stars' magnitudes converted to the same filter(s) as the
# observations (in this case the MeerLICHT/BlackGEM filter set):
# this is now the same as the astrometric catalog: [cal_cat] defined above
phot_ncal_max = 100 # max no. of calibration stars used for a given field 
phot_ncal_min = 10  # min no. of stars below which filter requirements are dropped
# default zeropoints used if no photometric calibration catalog is
# provided or a particular field does not contain any calibration stars
zp_default = {'ML1': {'u':22.4, 'g':23.3, 'q':23.8, 'r':22.9, 'i':22.3, 'z':21.4},
              'BG':  {'u':22.4, 'g':23.3, 'q':23.8, 'r':22.9, 'i':22.3, 'z':21.4}}


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
              'saturated-connected': 8, 'satellite trail': 16, 'edge': 32}

# subfolder to save the many temporary numpy binary files
dir_numpy = 'NumpyFiles'

# switch to keep intermediate/temporary files
keep_tmp = True

# switch on/off different functions
redo_new = False         # execute SExtractor, astrometry.net, PSFEx, optimal flux
redo_ref = False         # determination even if new/ref products already present
verbose = True           # print out extra info
timing = True            # (wall-)time the different functions
display = False          # show intermediate fits images (centre and 4 corners)
make_plots = True        # make diagnostic plots and save them as pdf
show_plots = False       # show diagnostic plots
