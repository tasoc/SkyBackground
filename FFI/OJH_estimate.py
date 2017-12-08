#!/bin/env python
# -*- coding: utf-8 -*-

#AUTHORS:
#Oliver James HALL

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy import interpolate

import sys
import glob
'''
Estimate the background of a Full Frame Image (FFI).

Includes a '__main__' for independent test runs on local machines.

Parameters
-----------

-----------

Output
-----------

'''

def fit_background(ffi, plots_on = False):
	#Setting up the values required for the measurement locations
	ribsize = 8 #Has to be even
	npts = 100  #Root must be whole number
	nside = np.round(np.sqrt(100))
	xlen = ffi.shape[1]
	ylen = ffi.shape[0]

	#Getting the spacing of points
	lx = xlen/(nside+2)
	ly = ylen/(nside+2)

	#Generating the points meshgrid
	xlocs = np.round(np.linspace(lx, xlen-lx, nside))
	ylocs = np.round(np.linspace(ly, ylen-ly, nside))
	X, Y = np.meshgrid(xlocs, ylocs)

	#Setting up a mask with points considered for background estimation
	mask = np.zeros_like(ffi)
	bkg_field = np.zeros_like(X.ravel())
	hr = ribsize/2

	#Calculating Buzasi+15 background inside masked areas
	for idx, (xx, yy) in enumerate(zip(X.ravel(), Y.ravel())):
		y = int(yy)
		x = int(xx)
		ffi_eval = ffi[y-hr:y+hr, x-hr:x+hr]
		bkg_field[idx] = np.nanmedian(ffi_eval[ffi_eval < np.nanpercentile(ffi_eval,[20])])
		mask[y-hr:y+hr, x-hr:x+hr] = 1

	#Interpolating to draw the background
	xx = np.arange(0,ffi.shape[1],1)
	yy = np.arange(0,ffi.shape[0],1)
	Xf, Yf = np.meshgrid(xx, yy)

	fn = interpolate.Rbf(X.ravel(), Y.ravel(), bkg_field, function='cubic', smooth=0)
	bkg_est = fn(Xf, Yf)

	#Plotting the ffi with measurement locations shown
	if plots_on:
		fig, ax = plt.subplots()
		im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
		fig.colorbar(im,label=r'$log_{10}$(Flux)')
		ax.set_title(ffi_type)
		ax.contour(mask, c='r', N=1)
		plt.show()

		fig, ax = plt.subplots()
		im = ax.imshow(np.log10(bkg), cmap='Blues_r', origin='lower')
		fig.colorbar(im, label=r'$log_{10}$(Flux)')

		f2, ax2 = plt.subplots()

		plt.show()

	return bkg_est


if __name__ == '__main__':
	plt.close('all')
	plots_on = False

	# Load file:
	ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
	ffi_type = ffis[0]
	sfile = glob.glob('../../data/'+ffi_type+'/simulated/*.fits')[0]
	bgfile = glob.glob('../../data/'+ffi_type+'/backgrounds.fits')[0]

	try:
	    hdulist = pyfits.open(sfile)
	    bkglist = pyfits.open(bgfile)

	except IOError:
	    print('File not located correctly.')
	    exit()

	ffi = hdulist[0].data
	bkg = bkglist[0].data

	est_bkg = fit_background(ffi)


	'''Plotting: all'''
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
