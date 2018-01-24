#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function fo restimation of sky background in TESS Full Frame Images

Includes a '__main__' for independent test runs on local machines.

.. versionadded:: 1.0.0
.. versionchanged:: 1.2

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""
#TODO: Include a unity test

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import astropy.io.fits as pyfits
import sys
import glob
import corner

import numpy as np
from scipy import interpolate
from scipy import stats

from MNL_estimate import cPlaneModel
from MNL_estimate import fRANSAC
from Functions import *


def fit_background(ffi, ribsize=8, nside=10, itt_ransac=500, order=1, plots_on=False):
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

		itt_ransac (int): Default 500. The number of RANSAC fits to make to the
			calculated modes across the full FFI.

		order (int): Default: 1. The desired order of the polynomial to be fit
			to the estimated background points.

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

	xlocs_left = np.linspace(superx, xend-superx, int(nsuper))
	ylocs_left = np.linspace(supery, yend-supery, int(nsuper))

	xlocs_right = np.linspace(xlen-xend+superx, xlen-superx, int(nsuper))
	ylocs_right = np.linspace(ylen-yend+supery, ylen-supery, int(nsuper))

	xlocs_mid = np.linspace(xend,xlen-xend,int(nreg))
	ylocs_mid = np.linspace(yend,ylen-yend,int(nreg))

	xx = np.append(xlocs_mid, [xlocs_left, xlocs_right])
	yy = np.append(ylocs_mid, [ylocs_left,ylocs_right])
	X, Y = np.meshgrid(xx, yy)

	#Setting up a mask with points considered for background estimation
	mask = np.zeros_like(ffi)
	bkg_field = np.zeros_like(X.ravel())
	hr = int(ribsize/2)

	#Calculating the KDE and consequent mode inside masked ares
	for idx, (xx, yy) in enumerate(zip(X.ravel(), Y.ravel())):
		y = int(yy)
		x = int(xx)
		ffi_eval = ffi[y-hr:y+hr+1, x-hr:x+hr+1] #Adding the +1 to make the selection even

		#Building a KDE on the data
		kernel = stats.gaussian_kde(ffi_eval.flatten(),bw_method='scott')
		alpha = np.linspace(ffi_eval.min(), ffi_eval.max(), 10000)

		#Calculate the optimal value of the mode from the KDE
		bkg_field[idx] = alpha[np.argmax(kernel(alpha))]
		mask[y-hr:y+hr+1, x-hr:x+hr+1] = 1			#Saving the evaluated location in a mask

	#Plotting the ffi with measurement locations shown
	if plots_on:
		fig, ax = plt.subplots()
		im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
		fig.colorbar(im,label=r'$log_{10}$(Flux)')
		ax.contour(mask, c='r', N=1)
		plt.show()

	print('Fitting a 2D polynomial using the cPlaneModel')
	#Preparing the data
	neighborhood = np.zeros([len(bkg_field),3])
	neighborhood[:, 0] = X.flatten()
	neighborhood[:, 1] = Y.flatten()
	neighborhood[:, 2] = bkg_field

	#Getting the inlier masks with RANSAC to expel outliers
	inlier_masks, coeffs = fRANSAC(bkg_field, neighborhood, itt_ransac)

	if plots_on:
		fig = corner.corner(coeffs, labels=['m','c'])
		plt.show()

	#Setting up the Plane Model Class
	Model = cPlaneModel(order=order, weights=inlier_masks)
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
	itt_ransac = 500
	order = 1

	# Load file:
	ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
	ffi_type = ffis[0]

	# ffi, bkg = load_files(ffi_type)
	ffi, bkg = get_sim()

	#Get background
	est_bkg = fit_background(ffi, ribsize, nside, itt_ransac, order, plots_on)

	'''Plotting: all'''
	print('The plots are up!')
	fig, ax = plt.subplots()
	im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
	fig.colorbar(im,label=r'$log_{10}$(Flux)')

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

	resOJH = est_bkg - bkg
	medOH = np.median(100*resOJH/bkg)
	stdOH = np.std(100*resOJH/bkg)
	print('OJH offset: '+str(np.round(medOH,3))+r"% $\pm$ "+str(np.round(stdOH,3))+'%')

	plt.show('all')
	plt.close('all')
