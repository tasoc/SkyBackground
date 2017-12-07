#!/bin/env python
# -*- coding: utf-8 -*-

#AUTHORS:
#Rasmus HANDBERG

import numpy as np
import astropy.io.fits as pyfits
from photutils import Background2D, SigmaClip, SExtractorBackground

def fit_background(fname):

	# Load file:
	if fname.endswith('.npy'):
		img = np.load(fname)
	else:
		with pyfits.open(fname, memmap=True, mode='readonly') as hdu:
			img = hdu[0].data

	# Create mask
	# TODO: Use the known locations of bright stars
	mask = ~np.isfinite(img)
	mask |= (img > 2e5)

	# Estimate the background:
	sigma_clip = SigmaClip(sigma=3.0, iters=5)
	bkg_estimator = SExtractorBackground()
	bkg = Background2D(img, (64, 64),
		filter_size=(3, 3),
		sigma_clip=sigma_clip,
		bkg_estimator=bkg_estimator,
		mask=mask)

	return bkg.background, mask
