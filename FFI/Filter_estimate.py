#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function for estimation of sky background in TESS Full Frame Images

Includes a '__main__' for independent test runs on local machines.

.. versionadded:: 1.0.0

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import sys
import glob

import numpy as np
import scipy.ndimage as nd

from Functions import *
    

if __name__ == '__main__':

    filters = ['minimum', 'percentile']
    filter_type = filters[1]

    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi_type = ffis[0]

    ffi, bkg = load_files(ffi_type)

    #Cutting up a small bit of data for testing
    smol = ffi[:500,:500]
    bbkg = bkg[:500,:500]

    #Building the footprint
    diam = 50
    percentile = 10

    if diam%2 == 0: diam+=1
    core = int(diam/2)
    X, Y = np.meshgrid(np.arange(diam),np.arange(diam))
    circle = (X - core)**2 + (Y - core)**2
    lim = circle[np.where(circle==0)[0]][:,0]
    circle[circle <= lim] = 1
    circle[circle > lim] = 0

    if filter_type == 'percentile':
        filt = nd.filters.percentile_filter(smol, percentile=percentile,\
                footprint=circle)


    if filter_type == 'minimum':
        filt = nd.filters.minimum_filter(smol, footprint=circle)


    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(np.log10(bbkg))
    ax[0].set_title('True background')
    ax[1].imshow(np.log10(filt))
    ax[1].set_title('Filtered')
    fig.tight_layout()

    fig2, ax2 = plt.subplots()
    bins = int(np.sqrt(bbkg.size))
    ax2.hist(bbkg.flatten(),histtype='step',bins=bins,label='bkg', normed=True)
    ax2.hist(filt.flatten(),histtype='step',bins=bins,label='filt', normed=True)
    ax2.legend(loc='best',fancybox=True)
    plt.show()
