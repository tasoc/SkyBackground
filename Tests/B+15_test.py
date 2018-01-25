#!/bin/env python
# -*- coding: utf-8 -*-

'''
A simple code to see how the Buzasi+15 method approximates a synthetic
background on a synthetic image of a point source with incresing image size.

The method involves taking the median of the lowest 20% of the image as the
background estimate.

The point source is approximated as  2D gaussian centered at the rounded center
pixel of the image, with a standard deviation of 1.5 pixels.

The background level is assumed to be 0.1% of the stellar flux as the location
of the point source.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
'''

import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy import stats

if __name__ == "__main__":
    plt.close('all')
    plots_on = True
    size = np.arange(5,20,1)

    numpix = np.ones_like(size)
    bkg_est = np.ones_like(size)
    for idx, rib in enumerate(size):
        #Drawing the frame
        xx = np.arange(0,rib)
        yy = np.arange(0,rib)
        X, Y = np.meshgrid(xx,yy)
        numpix[idx] = rib**2

        #True Peak location
        pt = np.array([np.round(rib/2),np.round(rib/2)])

        #Setting the background and the consequent point source flux
        bkg = 100
        bkg_arr = np.random.poisson(np.ones_like(X)*bkg, X.shape)
        flux = 100 * (1/0.001)
        sig = 1.5

        #Draw PSF from peaks with 1.5 pixel falloff, and add background
        d = flux * np.exp( (-(pt[0] - Y)**2 - (pt[1] - X)**2)/sig**2)
        d += bkg_arr   #Adding poissoninan noise to bkg

        '''Estimate background instead using the KDE method'''
        dd = d[d < np.nanpercentile(d,[75])]
        kernel = stats.gaussian_kde(dd.flatten(),bw_method='scott')
        alpha = np.linspace(dd.min(), dd.max(), 10000)
        bkg_est[idx] = alpha[np.argmax(kernel(alpha))]

        # bkg_est[idx] = np.median(d[d < np.nanpercentile(d,[20])])

        if plots_on:
            if idx == 0 or idx == 3:
                fig, ax = plt.subplots()
                c = ax.imshow(d)
                fig.colorbar(c, label = 'Flux (arbitrary units)')
                ax.set_xlabel('Pixel #')
                ax.set_ylabel('Pixel #')
                ax.set_title('Total pixels: '+str(numpix[idx]))
                plt.show()


    fig, ax = plt.subplots()
    ax.plot(np.sqrt(numpix), bkg_est, label='B+15 estimate')
    ax.axhline(np.median(bkg_arr), c='r', label='True background (0.1%)')
    ax.set_xlabel('Number of pixels per side')
    ax.set_ylabel('Background level (Flux)')
    ax.set_title('Improvement in background estimation with postage stamp size')
    ax.legend(loc='best',fancybox=True)
    plt.show()
