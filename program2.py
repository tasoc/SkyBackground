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
import pymc
from PyAstronomy.modelSuite import forTrans as ft
from scipy.interpolate import UnivariateSpline
import random
from PyAstronomy.modelSuite.XTran import _ZList
import scipy as sp
from matplotlib.colors import LogNorm
import math

#############################################################
#############################################################

with open('short.lst') as f:
  obj_names = f.read().splitlines()
f.close()

with open('short_PRO.lst') as f:
  obj_names_PRO = f.read().splitlines()
f.close()


## BACKGROUND INPUT
#hdu_list = pyfits.open('backgrounds.fits')
#background = hdu_list[0].data
#plt.imshow(background, cmap='gray', origin = 'lower')
#plt.plot(background[0,::])
#plt.show()
#exit()

ndim = 2048
nbin = int(ndim/60)
ndim_vec = np.linspace(0,ndim,ndim)
nbin_vec = np.linspace(0,ndim,nbin)
int_nbin_vec = np.zeros(nbin)
for j in range(nbin):
  int_nbin_vec[j] = int(nbin_vec[j])

background_map = np.zeros(ndim*ndim).reshape(ndim, ndim)
hamming_filter = ndim/2+1

for j in range(len(obj_names)):
  hdu_list = pyfits.open(obj_names[j])
  image_data = hdu_list[0].data

  data_min = np.min(image_data)
  data_max = np.max(image_data)
  data_mean_img = np.mean(image_data)

  image_data_smooth_1 = np.zeros(ndim*ndim).reshape(ndim,ndim)
  image_data_smooth_2 = np.zeros(ndim*ndim).reshape(ndim,ndim)

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

  for i in range(ndim):
    min_vec = np.zeros(nbin)
    msky = image_data[::,i]
    for h in range(nbin-1):
      min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])

    nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
    min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

    min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

    image_data_smooth_2[::,i] = min_vec_int_smooth

    #plt.ylim(80000.,105000.)
    #plt.plot(ndim_vec, min_vec_int)
    #plt.plot(ndim_vec, msky, 'r-')
    #plt.plot(ndim_vec, min_vec_int_smooth)
    #plt.show()

  image_data_smooth_22 = scipy.ndimage.gaussian_filter(image_data_smooth_2, kern_px/(2*math.sqrt(2*math.log(2))))

  plt.imshow(image_data_smooth_22, cmap='gray', origin = 'lower', vmin=80000., vmax=110000.)
  plt.show()

  final = (image_data_smooth_11 + image_data_smooth_22)/2.

  plt.imshow((image_data_smooth_11+image_data_smooth_22)/2., cmap='gray', origin = 'lower')
  plt.show()

  pyfits.writeto('backgrounds_PRO.fits', final)
  pyfits.writeto(obj_names_PRO[j], image_data - final)
  exit()
