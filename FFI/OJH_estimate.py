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
#TODO: Increase density of points near ages in style of Kepler
#TODO: Include a unity test

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import sys
import glob

import numpy as np
from scipy import interpolate

def fit_background(ffi, ribsize=8, npts=100, plots_on=False):
	"""
	Estimate the background of a Full Frame Image (FFI).
	This method employs basic principles from two previous works:
	-	It uses the Kepler background estimation approach by measuring the
	background in evenly spaced squares around the image.
	-	It employs the same background estimation as Buzasi et al. 2015, by taking
	the median of the lowest 20% of the selected square.

	The background values across the FFI are then interpolated over using the
	gaussian scipy.interpolate.rbf function with smoothing turned on.


	Parameters:
		ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

		ribsize (int): Default: 8. A single integer value that determines the length
			of the sides of the boxes the backgrounds are measured in.

		npts (int): Default: 100. A single integer value determining the number of
			squares at which the background is to be measured across the entire image.

		plots_on (bool): Default False. A boolean parameter. When True, it will show
			a plot indicating the location of the background squares across the image.

	Returns:
		ndarray: Estimated background with the same size as the input image.

	.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
	"""

	#Setting up the values required for the measurement locations
	if ffi.shape[0] < 2048:	#If FFI file is a superstamp, reduce ribsize
		ribsize = 4
	nside = np.round(np.sqrt(npts))	#The number of points per side
	xlen = ffi.shape[1]
	ylen = ffi.shape[0]

	# #Getting the spacing of points
	# lx = xlen/(nside+2)
	# ly = ylen/(nside+2)

	# #Generating the points meshgrid
	# xlocs = np.round(np.linspace(lx, xlen-lx, nside))
	# ylocs = np.round(np.linspace(ly, ylen-ly, nside))
	# X, Y = np.meshgrid(xlocs, ylocs)

	perc = 0.2

	superx = perc*xlen
	supery = perc*ylen

	nsuper = perc*nside
	nreg = (1-perc)*nside

	lx = superx/(nsuper+2)
	ly = supery/(nsuper+2)

	xlocs_left = np.linspace(lx, superx-lx, nsuper)
	ylocs_left = np.linspace(ly, supery-ly, nsuper)

	xlocs_right = np.linspace(xlen-superx+lx, xlen-lx, nsuper)
	ylocs_right = np.linspace(ylen-supery+ly, ylen-ly, nsuper)

	xlocs_mid = np.linspace(superx,xlen-superx,nreg)
	ylocs_mid = np.linspace(supery,ylen-supery,nreg)

	xx = np.append(xlocs_left, xlocs_mid)
	xx = np.append(xx, xlocs_right)
	yy = np.append(ylocs_left, ylocs_mid)
	yy = np.append(yy, ylocs_right)
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

	#Interpolating to draw the background
	xx = np.arange(xlen)
	yy = np.arange(ylen)
	Xf, Yf = np.meshgrid(xx, yy)

	#Plotting the ffi with measurement locations shown
	if plots_on:
		fig, ax = plt.subplots()
		im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
		fig.colorbar(im,label=r'$log_{10}$(Flux)')
		ax.set_title(ffi_type)
		ax.contour(mask, c='r', N=1)
		plt.show()

	print('Starting interpolation (CPU heavy)...')
	fn = interpolate.Rbf(X.ravel(), Y.ravel(), bkg_field, function='gaussian')
	bkg_est = fn(Xf, Yf)
	print('Interpolation complete!')


	return bkg_est


if __name__ == '__main__':
	plt.close('all')

	#Define parameters
	plots_on = True
	npts = 100
	ribsize = 8

	# Load file:
	ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
	ffi_type = ffis[1]
	sfile = glob.glob('../data/FFI/'+ffi_type+'.fits')[0]
	bgfile = glob.glob('../data/FFI/backgrounds_'+ffi_type+'.fits')[0]

	try:
	    hdulist = pyfits.open(sfile)
	    bkglist = pyfits.open(bgfile)

	except IOError:
	    print('File not located correctly.')
	    exit()

	ffi = hdulist[0].data
	bkg = bkglist[0].data

	#Get background
	est_bkg = fit_background(ffi, ribsize, npts, plots_on)

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

	plt.show('all')
	plt.close('all')
