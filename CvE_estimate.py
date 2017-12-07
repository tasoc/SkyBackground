#AUTHORS:
#T'DA1 | Carolina VAN ESSEN  | 2016
#T'DA3 | Oliver James HALL | 2017


import numpy as np
import matplotlib.pylab as plt
from numpy import *
import scipy
import scipy.ndimage
import scipy.integrate
import pyfits
import operator
from PyAstronomy import funcFit as fuf
from PyAstronomy import pyasl
from PyAstronomy.pyaC import pyaErrors as PE
from itertools import combinations
from PyAstronomy.modelSuite import forTrans as ft
from scipy.interpolate import UnivariateSpline
import random
import glob
from PyAstronomy.modelSuite.XTran import _ZList
import scipy as sp
from matplotlib.colors import LogNorm
import math

#############################################################
#############################################################

## BACKGROUND INPUT
#hdu_list = pyfits.open('backgrounds.fits')
#background = hdu_list[0].data
#plt.imshow(background, cmap='gray', origin = 'lower')
#plt.plot(background[0,::])
#plt.show()
#exit()

'''NOTE TO THE USER:
This code is not generalised for use on all platforms. The user will have to either
conform to the data locations used to read in the ffi data, or re-write the readin
to suit their needs.'''


if __name__ == '__main__':
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi = ffis[0]
    sfile = glob.glob('../data/'+ffi+'/simulated/*.fits')[0]

    '''Setting up image dimensions'''
    ndim = 2048
    nbin = int(ndim/60)
    ndim_vec = np.linspace(0,ndim,ndim)
    nbin_vec = np.linspace(0,ndim,nbin)
    int_nbin_vec = np.zeros(nbin, dtype=np.int)
    for j in range(nbin):
      int_nbin_vec[j] = int(nbin_vec[j])

    '''Setting up filter'''
    background_map = np.zeros(ndim*ndim).reshape(ndim, ndim)
    hamming_filter = ndim/2+1

    '''Reading in data'''
    hdu_list = pyfits.open(sfile)
    image_data = hdu_list[0].data

    fig, ax = plt.subplots()
    ax.imshow(np.log10(image_data),cmap='Blues_r')
    ax.set_title(ffi)
    plt.show()
    plt.close('all')

    '''Calculating data metadata'''
    data_min = np.min(image_data)
    data_max = np.max(image_data)
    data_mean_img = np.mean(image_data)

    '''Smoothing the data'''
    image_data_smooth_1 = np.zeros(ndim*ndim).reshape(ndim,ndim)
    image_data_smooth_2 = np.zeros(ndim*ndim).reshape(ndim,ndim)

    '''Cutting line by line'''
    for i in range(ndim):
        min_vec = np.zeros(nbin)
        msky = image_data[i,::]

        for h in range(nbin-1):
            min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])


        nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
        min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

        min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

        image_data_smooth_1[i,::] = min_vec_int_smooth

        if (i == 0):
          plt.ylim(90000.,250000.)
          plt.plot(ndim_vec, msky, 'r-')
          plt.show()
          plt.ylim(250000., 90000.)
          plt.plot(ndim_vec, msky, 'r-')
          plt.show()
          plt.ylim(130000., 90000.)
          plt.plot(ndim_vec, msky, 'r-')
          plt.show()
          plt.ylim(130000., 90000.)
          plt.plot(ndim_vec, msky, 'r-')
          plt.plot(ndim_vec, min_vec_int, 'b-')
          plt.plot(ndim_vec, min_vec_int_smooth, 'k-')
          plt.show()

    kern_px = 100
    image_data_smooth_11 = scipy.ndimage.gaussian_filter(image_data_smooth_1, kern_px/(2*math.sqrt(2*math.log(2))))

    plt.imshow(image_data_smooth_11, cmap='gray', origin = 'lower', vmin=80000., vmax=110000.)
    plt.show()
    sys.exit() 
    for i in range(ndim):
        min_vec = np.zeros(nbin)
        msky = image_data[::,i]
        for h in range(nbin-1):
            min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])

        nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
        min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

        min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

        image_data_smooth_2[::,i] = min_vec_int_smooth

        plt.ylim(80000.,105000.)
        plt.plot(ndim_vec, min_vec_int)
        plt.plot(ndim_vec, msky, 'r-')
        plt.plot(ndim_vec, min_vec_int_smooth)
        plt.show()

    image_data_smooth_22 = scipy.ndimage.gaussian_filter(image_data_smooth_2, kern_px/(2*math.sqrt(2*math.log(2))))

    plt.imshow(image_data_smooth_22, cmap='gray', origin = 'lower', vmin=80000., vmax=110000.)
    plt.show()

    final = (image_data_smooth_11 + image_data_smooth_22)/2.

    plt.imshow((image_data_smooth_11+image_data_smooth_22)/2., cmap='gray', origin = 'lower')
    plt.show()
    #
    # pyfits.writeto('backgrounds_PRO.fits', final)
    # pyfits.writeto(obj_names_PRO[j], image_data - final)
    # exit()
