#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function fo restimation of sky background in TESS Full Frame Images

Includes a '__main__' for independent test runs on local machines.

.. versionadded:: 1.0.0
.. versionchanged:: 1.0.2

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""
#TODO: Sigma Clip before interpolation
#TODO: Include a unity test

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import astropy.io.fits as pyfits
import sys
import glob

import numpy as np
from scipy import interpolate

from MNL_estimate import cPlaneModel
from Functions import *


def fit_background(ffi, ribsize=8, nside=10, plots_on=False):
	"""
	Estimate the background of a Full Frame Image (FFI).
	This method employs basic principles from two previous works:
	-	It uses the Kepler background estimation approach by measuring the
	background in evenly spaced squares around the image.
	-	It employs the same background estimation as Buzasi et al. 2015, by taking
	the median of the lowest 20% of the selected square.

	The background values across the FFI are then fit to with a 2D polynomial
	using the cPlaneModels class from the MNL_estimate code.


	Parameters:
		ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

		ribsize (int): Default: 8. A single integer value that determines the length
			of the sides of the boxes the backgrounds are measured in.

		nside (int): Default: 100. The number of points a side to evaluate the
			background for, not consider additional points for corners and edges.

		plots_on (bool): Default False. A boolean parameter. When True, it will show
			a plot indicating the location of the background squares across the image.

	Returns:
		ndarray: Estimated background with the same size as the input image.

	.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
	"""

	#Setting up the values required for the measurement locations
	if ffi.shape[0] < 2048:	#If FFI file is a superstamp, reduce ribsize
		ribsize = 4

	xlen = ffi.shape[1]
	ylen = ffi.shape[0]

	perc = 0.1

	lx = xlen/(nside+2)
	ly = ylen/(nside+2)

	superx = lx/2
	supery = ly/2

	nsuper = perc*nside*2
	nreg = (1-2*perc)*nside

	xend = perc*xlen
	yend = perc*xlen

	xlocs_left = np.linspace(superx, xend-superx, nsuper)
	ylocs_left = np.linspace(supery, yend-supery, nsuper)

	xlocs_right = np.linspace(xlen-xend+superx, xlen-superx, nsuper)
	ylocs_right = np.linspace(ylen-yend+supery, ylen-supery, nsuper)

	xlocs_mid = np.linspace(xend,xlen-xend,nreg)
	ylocs_mid = np.linspace(yend,ylen-yend,nreg)

	xx = np.append(xlocs_mid, [xlocs_left, xlocs_right])
	yy = np.append(ylocs_mid, [ylocs_left,ylocs_right])
	X, Y = np.meshgrid(xx, yy)

	#Setting up a mask with points considered for background estimation
	mask = np.zeros_like(ffi)
	bkg_field = np.zeros_like(X.ravel())
	hr = int(ribsize/2)

	#Calculating Buzasi+15 background inside masked areas
	for idx, (xx, yy) in enumerate(zip(X.ravel(), Y.ravel())):
		y = int(yy)
		x = int(xx)
		ffi_eval = ffi[y-hr:y+hr+1, x-hr:x+hr+1]	#Adding the +1 to make the selection even
		bkg_field[idx] = np.nanmedian(ffi_eval[ffi_eval < np.nanpercentile(ffi_eval,[20])])
		mask[y-hr:y+hr+1, x-hr:x+hr+1] = 1			#Saving the evaluated location in a mask


	#Plotting the ffi with measurement locations shown
	if plots_on:
		fig, ax = plt.subplots()
		im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
		fig.colorbar(im,label=r'$log_{10}$(Flux)')
		ax.set_title(ffi_type)
		ax.contour(mask, c='r', N=1)
		plt.show()

	print('Fitting a 2D polynomial using the cPlaneModel')
	#Preparing the data
	neighborhood = np.zeros([len(bkg_field),3])
	neighborhood[:, 0] = X.flatten()
	neighborhood[:, 1] = Y.flatten()
	neighborhood[:, 2] = bkg_field

	#Setting up the Plane Model Class
	Model = cPlaneModel(order=2, weights=None)
	Fit = Model.fit(neighborhood)
	fit_coeffs = Fit.coeff

	#Constructing the model on a grid the size of the full ffi
	Xf, Yf = np.meshgrid(np.arange(xlen), np.arange(ylen))
	bkg_est = Model.evaluate(Xf, Yf, fit_coeffs)

	return bkg_est


if __name__ == '__main__':
	plt.close('all')

	#Define parameters
	plots_on = True
	nside = 25
	npts = nside**2
	ribsize = 8

	# Load file:
	ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
	ffi_type = ffis[1]

	ffi, bkg = load_file(ffi_type)

	#Get background
	est_bkg = fit_background(ffi, ribsize, nside, plots_on)

	'''Plotting: all'''
	print('The plots are up!')
	fig, ax = plt.subplots()
	im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
	fig.colorbar(im,label=r'$log_{10}$(Flux)')
	ax.set_title(ffi_type)

	fdiff, adiff = plt.subplots()
	diff = adiff.imshow(np.log10(est_bkg) - np.log10(bkg), origin='lower')
	fdiff.colorbar(diff, label='Estimated Bkg - True Bkg (both in log10 space)')
	adiff.set_title('Estimated bkg - True bkg')

	fest, aest = plt.subplots()
	est = aest.imshow(np.log10(est_bkg), cmap='Blues_r', origin='lower')
	fest.colorbar(est, label=r'$log_{10}$(Flux)')
	aest.set_title('Background estimated with '+str(npts)+' squares of '+str(ribsize)+'x'+str(ribsize))

    cc, button = close_plots()
    button.on_clicked(close)

	plt.show('all')
	plt.close('all')
