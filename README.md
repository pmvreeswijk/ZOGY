# ZOGY
a Python implementation of proper image subtraction (Zackay, Ofek &amp; Gal-Yam 2016, ApJ, 830, 27)

Written by Paul Vreeswijk with vital input from Barak Zackay and Eran Ofek. Adapted by Kerry Paterson for integration into pipeline for MeerLICHT.

This module accepts a new and a reference fits image, runs SExtractor on them, finds their WCS solution using Astrometry.net, uses PSFex to infer the position-dependent PSFs of the images and SWarp to map the reference image to the new image and performs optimal image subtraction following Zackay et al. (2016) to produce the subtracted image (D), the significance image (S), the corrected significance image (Scorr), and PSF photometry image (Fpsf - alpha in the paper) and associated error image (Fpsferr). The inferred PSFs are also used to extract optimal photometry (following Horne 1986, PASP, 98, 609) of all sources detected by SExtractor. The configuration files of SExtractor, PSFex and SWarp are located in the Config directory.

It makes grateful use of the following programs that first need to be installed:

 - Astrometry.net (in particular "solve-field" and index files): http://astrometry.net 
 - SExtractor: http://www.astromatic.net/software/sextractor
 - SWarp: http://www.astromatic.net/software/swarp
 - PSFex: http://www.astromatic.net/software/psfex
 - ds9
 - sip_to_pv package from David Shupe: https://github.com/stargaser/sip_tpv
 - pyfftw to speed up the many FFTs performed
 - the other modules imported at the top (e.g. astropy, matplotlib, etc.)
 
Warning: this module is still being developed and has so far been tested on KMTNet and MeerLICHT images. It is designed specifically to be included in the MeerLICHT and BlackGEM pipelines, but we hope that it will be useful to apply to images of other telescopes as well.

This project is licensed under the terms of the MIT license.
