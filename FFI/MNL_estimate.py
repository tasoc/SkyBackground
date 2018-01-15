#!/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

if __name__ == '__main__':
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

    '''Program starts here:'''
    size = 128   #Number of blocks to cut the ffi into
    itt_field = 1   #Number of iterations of ransac fitting to each box

    #Cutting the ffi up into 'size' smaller blocks
    block_ffi = (ffi.reshape(ffi.shape[0]//size, size, -1, size)
                .swapaxes(1,2)
                .reshape(-1, size, size))

    #Creating the storage for the mode positions in each block
    modes = np.zeros([ffi.shape[0]/size, ffi.shape[1]/size])

    #Creating an array with all box locations in the 0 and 1 positions
    X0, Y0 = np.meshgrid(np.arange(size), np.arange(size))
    neighborhood0 = np.zeros([len(X0.flatten()), 3])
    neighborhood0[:, 0] = X0.flatten()
    neighborhood0[:, 1] = Y0.flatten()

    #Fitting RANSAC to each box and calculating the mode of the inlier pixels
    i = 0
    for j in range(modes.shape[0]):
        for jj in range(modes.shape[1])
            F = block_ffi[i, ::]    #Calling the first block

            neighborhood0[:, 2] = F.flatten()

            #Running RANSAC on itt_field iterations and saving inlier masks
            inlier_masks = np.zeros(neighborhood.shape[0])
            for kk in range(itt_field):
                try:
                    inlier_mask, coeff, intercept = local-regression_plane_ransac(neighborhood, thresh_factor)
                    #Preparing the data for RANSAC
                    XY = neighborhood0[:,:2]
                    Z = neighborhood0[:,2]

                    #Setting the RANSAC threshold as 90% of the M.A.D
                    mad = 1.4826 * np.nanmedian(np.abs(F - np.nanmedian(F)))
                    thresh = 0.9*mad

                    ransac = linear_model.RANSACRegressor(linear_model.LienarRegression(), residual_threshold=Thresh)
                    ransac.fit(XY, Z)

                    inlier_mask = ransac.inlier_mask_
                    # coeff = ransac.estimator_.coeff_
                    # intercept = ransac.estimator_.intercept_

                    inlier_masks += inlier_mask
                except:
                    continue

            #Putting the inlier masks back into 2D shape
            inlier_masks_arr = inlier_masks.reshape((F.shape[0], F.shape[1]))
            #Evaluating the inlier masks
            FFF2 = F[(inlier_mask_arr>itt_field/2)].flatten() #Inliers of more than 50%
            FFF2 = FFF2[(FFF2<np.percentile(FFF2.cut))]       #Inlying below the 25th percentile

            #Building a KDE on the background inlier data
            kernel = KDE(stats.trim1(np.sort(FFF2[np.isfinite(FFF2)]), 0))
            kernel.fit(kernel='gau', bw='scott', fft=True, gridsize=200)

            #Calculate an optimal value for the mode from the KDE
            max_guess = kernel.support[np.argmax(kernel.density)]
            def kernel_opt(x): return -1*kernel.evaluate(x)
            mode = optimize.fmin_powell(kernel_opt, max_guess, disp=0)


    #w_coeff, Modes_arr, Modes_arr2, X, Y, = BG(Flux, Flu2, size=size, simple=False, plot=True)
    A = blockshaped(flux, size, size)
    h, w = arr.shape
    A = (arr.reshape(h//nrows, nrows, -1, ncols).wapaxes(1, 2).reshape(-1, nrows, ncols))
