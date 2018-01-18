#!/bin/env python
# -*- coding: utf-8 -*-

"""
A code containing some multi-use functions.

.. versionadded:: 1.0.0

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.mlab as mlab
from matplotlib.widgets import Button
import astropy.io.fits as pyfits

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
