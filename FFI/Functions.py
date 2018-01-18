#!/bin/env python
# -*- coding: utf-8 -*-

"""
A code containing some multi-use functions.

.. versionadded:: 1.3

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.mlab as mlab
from matplotlib.widgets import Button
import astropy.io.fits as pyfits
import scipy.ndimage as nd


def circular_filter(data, diam=10, percentile=10, filter_type='percentile'):
    '''
    A function that runs a filter of choice using a circular footprint of a
    diameter determined by the user.

    Parameters:
        data (ndarray): An array containing the unsmoothed data.

        diam (int): Default: 10. The desired diameter in pixels of the circular
            footprint.

        percentile (int): Default: 10. The desired percentile to use on the percentile
            filter.

        filter_type (str): Default 'percentile'. Call 'minimum' for a minimum filter
            or 'percentile' for a percentile filter.

    Returns:
        ndarray: An array of the same shape as the input data containing the data
            smoothed using the filter of choice.
    '''

    if diam%2 == 0: diam+=1 #Make sure the diameter is uneven for symmetry
    core = int(diam/2)      #Finding the centre of the circle
    X, Y = np.meshgrid(np.arange(diam), np.arange(diam))    #Creating a meshgrid
    circle = (X - core)**2 + (Y-core)**2        #Building the circle in the meshgrid
    lim = circle[np.where(circle==0)[0]][:,0]   #Finding the value at the diameter edge
    circle[circle <= lim] = 1  #Setting all values inside of the circle to 1
    circle[circle > lim] = 0    #Setting all values outside of the circle to 0

    if filter_type == 'percentile':
        filt = nd.filters.percentile_filter(data, percentile=percentile,\
                footprint=circle)

    if filter_type == 'minimum':
        filt = nd.filters.minimum_filter(smol, footprint=circle)

    return filt


def get_sim():
    '''
    A function that creates a simple testing backround.
    Returns:
        ndarray: Simulated FFI of given shape.

        ndarray: The background of the simulated FFI of given shape.
    '''
    shape = (2048,2048)

    X, Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    z = 1000.
    sigma = 100.

    sim = np.random.normal(z, 10., shape)

    return sim, np.ones(shape)*z


def load_files(ffi_type):
    '''
    A function that reads in the FFI testing data from inside the git repo.

    Parameters:
        ffi_type (str): The name of the type of ffi. ffi_north, ffi_south, or
            ffi_cluster.

    Returns:
        ndarray: The read-in simulated FFI

        ndarray: The read-in simulated background for the FFI

    '''

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

    return ffi, bkg

def close_plots():
    '''
    A function that plots a button to instantly close all subplots. Useful when
    plotting a large number of comparisons.

    Returns:
        matplotlib.figure.Figure: 1 by 1 plot containing a 'close all' button.

        matplotlib.widgets.Button: A button widget required for button function.

    Note: Must be called with the line:
        button.on_clicked(close)

    Note: The close function must also be imported. Best to use
        from Functions import *

    '''
    fig, ax = plt.subplots(figsize=(1,1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    closeax =plt.axes([0.1,0.1,0.8,0.8])

    button = Button(closeax, 'Close Plots', color='white', hovercolor='r')
    return fig, button

def close(event):
    ''' A simple plt.close('all') function for use with close_plots().'''
    plt.close('all')
