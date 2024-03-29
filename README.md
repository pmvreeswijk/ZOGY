# ZOGY
a Python implementation of proper image subtraction (Zackay, Ofek &amp; Gal-Yam 2016, ApJ, 830, 27), with vital input from Barak Zackay and Eran Ofek; adapted by Kerry Paterson for integration into pipeline for MeerLICHT.

This module accepts a new and a reference fits image, runs SExtractor on them, finds their WCS solution using Astrometry.net, uses PSFex to infer the position-dependent PSFs of the images and SWarp to map the reference image to the new image and performs optimal image subtraction following Zackay et al. (2016) to produce the subtracted image (D), the significance image (S), the corrected significance image (Scorr), and PSF photometry image (Fpsf - alpha in the paper) and associated error image (Fpsferr). The inferred PSFs are also used to extract optimal photometry (following Horne 1986, PASP, 98, 609) of all sources detected by SExtractor. It is also possible to perform forced photometry on all Gaia DR3 sources in the field-of-view. The configuration files of SExtractor, PSFex and SWarp are located in the Config directory.

It makes grateful use of the following programs:

- Astrometry.net (in particular "solve-field" and index files): http://astrometry.net
- SExtractor: http://www.astromatic.net/software/sextractor
- SWarp: http://www.astromatic.net/software/swarp
- PSFex: http://www.astromatic.net/software/psfex
- sip_to_pv package from David Shupe: https://github.com/stargaser/sip_tpv
- pyfftw to speed up the many FFTs performed
- the other modules imported at the top (e.g. astropy, matplotlib, etc.)

Warning: this module is written specifically to be included in the MeerLICHT and BlackGEM pipelines, but we hope that it will be useful to apply to images of other telescopes as well.


Suggested steps to get started:

(1) because of the mix of software packages used by ZOGY, it is easiest to run it on a machine with singularity installed; see https://sylabs.io/docs. On a linux machine, installing singularity is fairly straightforward, see e.g. https://docs.sylabs.io/guides/4.1/user-guide/quick_start.html for the currently latest version.

(2) download the latest BlackBOX/ZOGY singularity container from https://surfdrive.surf.nl/files/index.php/s/3zxuCHbt8eIh5tk

(3) for BlackGEM and MeerLICHT we use various calibration files; you can find these at https://surfdrive.surf.nl/files/index.php/s/ShRamgz5SxbA77n. However, most of these are specific to MeerLICHT and BlackGEM, including the photometric calibration catalogue and the transient real/bogus classification. Another cautionary note: the fits table GaiaDR3_all_HP4_highPM.fits, used to be able to perform forced photometry at Gaia DR3 positions, is very large: ~72GB!

If you use (some of) them, you need to update the settings file parameter **cal_dir** to point to the folder where you put them.

Very likely you will need to use your own Astrometry.net index files. If you already have these available, you can update the settings file parameter **astronet_config** to point to the existing configuration file.


(4) open a shell in the container:

    singularity shell [path to container]/MLBG_[version].sif

(5) inside the container, the zogy software is in /Software/ZOGY and the settings file is in /Software/ZOGY/Settings/set_zogy.py. You cannot edit these files inside the container. For the settings file, you can copy it to some folder, edit it and provide it with the command-line input parameter **set_file**. Alternatively, you can also add that folder to the front of your PYTHONPATH environment variable. E.g.:

    export PYTHONPATH=[path to your zogy folder]:$PYTHONPATH

Then those modules will be be used instead of the default ones in /Software. N.B.: if you edit zogy.py itself, make sure to run python on that version rather than the one in /Software/ZOGY/zogy.py.

(6) edit the settings file to adapt it to your images and their headers. For telescope-dependent parameters, you could add your telescope name to the keys with the corresponding value for that parameter, which will then be used if the telescope input parameter (see below) is set (default: "ML1" for MeerLICHT). If a parameter is not a dictionary but a single value, that value will be used regardless of the telescope input parameter.

(7) check out the main input parameters:

    python /Software/ZOGY/zogy.py -h

(8) some examples how to run it:

- run it on a "new.fits" and "ref.fits" using the default MeerLICHT settings:

  ```
  python /Software/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits
  ```


- instead of MeerLICHT, use the dictionary keys corresponding to
  "my_tel" defined in the settings file:

  ```
  python /Software/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits --telescope my_tel
  ```


- instead of the default settings file (set_zogy.py), use a copy of it
  that was adapted to your images (depending on whether copy contains
  dictionaries for some parameters or not, the telescope input
  parameter should be provided or not):

  ```
  python /Software/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits --set_file mycopy [--telescope my_tel]
  ```


- if you have mask images available (settings parameters
  [transient_mask_max] and [mask_value] should be updated if your
  masks contain different values for the mask pixel type; the keys
  cannot be changed):

  ```
  python /Software/ZOGY/zogy.py --new_fits new.fits --new_fits_mask new_mask.fits --ref_fits ref.fits --ref_fits_mask ref_mask.fits --set_file mycopy [--telescope my_tel]
  ```

(9) instead of opening a shell in the container, you can also run commands from outside as follows:

    singularity exec [path to container]/MLBG_[version].sif python /Software/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits

To update the PYTHONPATH while using exec:

    singularity exec --env PYTHONPATH="[path to your zogy folder]:\$PYTHONPATH" [path to container]/MLBG_[version].sif python /Software/ZOGY/zogy.py --new_fits new.fits --ref_fits ref.fits


This project is licensed under the terms of the MIT license.
