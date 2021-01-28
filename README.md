# ZOGY
a Python implementation of proper image subtraction (Zackay, Ofek &amp; Gal-Yam 2016, ApJ, 830, 27), with vital input from Barak Zackay and Eran Ofek; adapted by Kerry Paterson for integration into pipeline for MeerLICHT.

This module accepts a new and a reference fits image, runs SExtractor on them, finds their WCS solution using Astrometry.net, uses PSFex to infer the position-dependent PSFs of the images and SWarp to map the reference image to the new image and performs optimal image subtraction following Zackay et al. (2016) to produce the subtracted image (D), the significance image (S), the corrected significance image (Scorr), and PSF photometry image (Fpsf - alpha in the paper) and associated error image (Fpsferr). The inferred PSFs are also used to extract optimal photometry (following Horne 1986, PASP, 98, 609) of all sources detected by SExtractor. The configuration files of SExtractor, PSFex and SWarp are located in the Config directory.

It makes grateful use of the following programs:

- Astrometry.net (in particular "solve-field" and index files): http://astrometry.net 
- SExtractor: http://www.astromatic.net/software/sextractor
- SWarp: http://www.astromatic.net/software/swarp
- PSFex: http://www.astromatic.net/software/psfex
- ds9
- sip_to_pv package from David Shupe: https://github.com/stargaser/sip_tpv
- pyfftw to speed up the many FFTs performed
- the other modules imported at the top (e.g. astropy, matplotlib, etc.)

In InstallScripts you will find bash scripts to help with the full installation of ZOGY on Ubuntu and macOS; please read through them before running them to ensure they do not interfere with your current installation.

Warning: this module is still being developed and has so far been tested on KMTNet and MeerLICHT images. It is designed specifically to be included in the MeerLICHT and BlackGEM pipelines, but we hope that it will be useful to apply to images of other telescopes as well.


Suggested steps to get started:

(1) download one of the install scripts, e.g. for mac OS:

    wget https://raw.githubusercontent.com/pmvreeswijk/ZOGY/master/InstallScripts/install_zogy_macos.sh

(2) read through the install script to make sure that it won't interfere with your present set-up, e.g. the python version

(3) execute the script in a folder where you'd like the ZOGY folder with python modules and settings/configuration subfolders to be created

(4) edit the settings file located in [some path]/ZOGY/Settings/set_zogy.py to adapt it to your images and their headers. For telescope-dependent parameters, you could add your telescope name to the keys with the corresponding value for that parameter, which will then be used if the telescope input parameter (see below) is set (default: "ML1" for MeerLICHT). If a parameter is not a dictionary but a single value, that value will be used regardless of the telescope input parameter.

(5) check out the main input parameters:

    python [some path]/ZOGY/zogy.py -h

(6) some examples how to run it:

    - run it on a "new.fits" and "ref.fits" using the default MeerLICHT settings:

    python [some path]/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits
    

    - instead of MeerLICHT, use the dictionary keys corresponding to "my_tel" defined in the settings file:
    
    python [some path]/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits --telescope my_tel
      

    - instead of the default settings file (set_zogy.py), use a copy of it that was adapted to your images
      (depending on whether copy contains dictionaries for some parameters or not, the telescope input 
       parameter should be provided or not):
    
    python [some path]/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits --set_file mycopy [--telescope my_tel]


    - if you have mask images available (settings parameters [transient_mask_max] and [mask_value] should be
      updated if your masks contain different values for the mask pixel type; the keys cannot be changed):

    python [some path]/ZOGY/zogy.py --new_fits new.fits --new_fits_mask new_mask.fits --ref_fits ref.fits --ref_fits_mask ref_mask.fits --set_file mycopy [--telescope my_tel]


This project is licensed under the terms of the MIT license.
