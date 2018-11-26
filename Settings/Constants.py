
__version__ = '0.6'

#===============================================================================
# ZOGY
#===============================================================================
# the full image will be subdivided in a number of square subimages
# defined by [subimage_size]; the FFTs for the image subraction will
# be performed on images with size [subimage_size]+2xsubimage_border,
# where only the central [subimage_size] pixels will be used for the
# full output image
subimage_size = 960      # size of subimages
subimage_border = 32     # border around subimage to avoid edge effects

# ZOGY parameters
fratio_local = False     # determine fratio (Fn/Fr) from subimage (T) or full frame (F)
dxdy_local = False       # determine dx and dy from subimage (T) or full frame (F)
transient_nsigma = 6     # required significance in Scorr for transient detection

# add optional fake stars for testing purposes
nfakestars = 1           # number of fake stars to be added to each subimage; first star
                         # is at the center, the rest (if any) is randomly distributed
fakestar_s2n = 10        # required signal-to-noise ratio of the fake stars    

#===============================================================================
# Background
#===============================================================================
# background estimation: these are the optional methods to estimate the
# backbround and its standard deviation (STD):
# (1) background and STD/RMS map determined by SExtractor (fastest)
# (2) improved background and STD map using masking of all sources (recommended)
# (3) similar to 2 but using photutils' Background2D (very very slow!)
bkg_method = 2           # background method to use
bkg_nsigma = 3           # data outside mean +- nsigma * stddev are
                         # clipped (methods 2 and 3)
bkg_boxsize = 240        # size of region used to determine
                         # background (all methods)
bkg_filtersize = 5       # size of filter used for smoothing the above
                         # regions (all methods)

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
key_ron = 'RDNOISE'
key_satlevel = 'SATURATE'
key_ra = 'RA'
key_dec = 'DEC'
#key_pixscale = 'PIXSCALE'
pixscale = 0.563
key_exptime = 'EXPTIME'
key_filter = 'FILTER'
key_obsdate = 'DATE-OBS'

#===============================================================================
# initial seeing estimate
#===============================================================================
fwhm_imafrac = 0.25      # fraction of image area that will be used
                         # for initial seeing estimate
fwhm_detect_thresh = 10. # detection threshold for fwhm SExtractor run
fwhm_class_sort = False  # sort objects according to CLASS_STAR (T)
                         # or by FLUX_AUTO (F)
fwhm_frac = 0.25         # fraction of objects, sorted in brightness
                         # or class_star, used for fwhm estimate

#===============================================================================
# PSF parameters
#===============================================================================
use_single_psf = False   # use the same central PSF for all subimages
psf_clean_factor = 0     # pixels with values below (PSF peak * this
                         # factor) are set to zero; if this parameter
                         # is zero, no cleaning is done
psf_radius = 5           # PSF radius in units of FWHM used to build the PSF
                         # this determines the PSF_SIZE in psfex.config
                         # and size of the VIGNET in sex.params
psf_sampling = 0.0       # PSF sampling step in image pixels used in PSFex
                         # If zero, it is automatically determined for the
                         # new and ref image as:
                         #    psf_sampling = FWHM * [psf_samp_fwhmfrac]
                         # If non-zero, its value is adopted for the sampling
                         # step in both images.
psf_samp_fwhmfrac = 1/4.5 # PSF sampling step in units of FWHM
                         # this is only used if [psf_sampling]=0.
size_vignet_ref = 71     # size of the square VIGNETs saved in the SExtractor
                         # catalog and used by PSFEx for the reference image.
                         # For the new image this value is set to
                         # ~ 2 * [psf_radius] * FWHM. This reference value
                         # should be set to ~ 2 * [psf_radius] * maximum expected
                         # FWHM in any of the new images.
psf_stars_s2n_min = 20   # minimum signal-to-noise ratio for PSF stars
                         # (don't set this too high as otherwise the PSF
                         #  will be mainly based on bright stars)
                         
#===============================================================================
# Astrometry
#===============================================================================
# WCS
skip_wcs = Talse         # skip Astrometry.net step if image already
                         # contains a reliable WCS solution
# Astrometry.net's tweak order
astronet_tweak_order = 3
# only search in Astrometry.net index files within this radius of the
# header RA and DEC [deg]
astronet_radius = 1.5
pixscale_varyfrac = 0.02 # pixscale solution found by Astrometry.net will
                         # be within this fraction of the assumed pixscale
# calibration catalog used for both astrometry and photometry
cal_cat = '/media/data/pmv/PhotCalibration/ML_calcat_kur_allsky_ext1deg_20181115.fits'
ast_nbright = 1000       # brightest no. of objects in the field to use for astrometry
ast_filter = 'r'         # magnitude column to sort in brightness

#===============================================================================
# Photometry
#===============================================================================
# aperture radii in units of FWHM
apphot_radii = [0.66, 1.5, 5] # list of radii in units of FWHM
                              # used for aperture photometry
                              # in SExtractor general
# PSF fitting
dosex_psffit = False     # do extra SExtractor run with PSF fitting
                              
# Photometric calibration
obs_lat = -32.38722      # observatory latitude in degrees (North)
obs_long = 20.81667      # observatory longitude in degrees (East)
obs_height = 1798.       # observatory height in meters above sealevel
# these [ext_coeff] are mean extinction estimates for Sutherland in
# the MeerLICHT filters:
ext_coeff = {'u':0.52, 'g':0.23, 'q':0.15, 'r':0.12, 'i':0.08, 'z':0.06}
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
zp_default = {'u':24., 'g':24., 'q':24., 'r':24., 'i':24., 'z':24.}

#===============================================================================
# Configuration
#===============================================================================
# path and names of configuration files
cfg_dir = './Config/'
sex_cfg = cfg_dir+'sex.config'               # SExtractor configuration file
sex_cfg_psffit = cfg_dir+'sex_psffit.config' # same for PSF-fitting version
sex_par = cfg_dir+'sex.params'               # SExtractor output parameters definition file
sex_par_psffit = cfg_dir+'sex_psffit.params' # same for PSF-fitting version
sex_par_ref = cfg_dir+'sex_ref.params'       # same for reference image output version
psfex_cfg = cfg_dir+'psfex.config'           # PSFex configuration file
swarp_cfg = cfg_dir+'swarp.config'           # SWarp configuration file

# if a mask image is provided, the mask values can be associated to
# the type of masked pixel with this dictionary:
mask_value = {'bad': 1, 'cosmic': 2, 'saturated': 4, 'saturated_connected': 8,
              'satellite': 16, 'edge': 32}

# switch on/off different functions
redo = False             # execute functions even if output file exist
verbose = True           # print out extra info
timing = True            # (wall-)time the different functions
display = False          # show intermediate fits images (centre and 4 corners)
make_plots = False       # make diagnostic plots and save them as pdf
show_plots = False       # show diagnostic plots
