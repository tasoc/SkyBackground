#AUTHORS:
#T'DA1 | Carolina VAN ESSEN  | 2016
#T'DA3 | Oliver James HALL | 2017


import numpy as np
import matplotlib.pylab as plt
import scipy
import pyfits
import glob

import scipy.ndimage
import scipy.integrate
from PyAstronomy import pyasl

'''NOTE TO THE USER:
This code is not generalised for use on all platforms. The user will have to either
conform to the data locations used to read in the ffi data, or re-write the readin
to suit their needs.'''


if __name__ == '__main__':
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi = ffis[0]
    sfile = glob.glob('../data/'+ffi+'/simulated/*.fits')[0]
    plots_on = False

    '''Reading in data'''
    hdu_list = pyfits.open(sfile)
    image_data = hdu_list[0].data

    '''Setting up image dimensions'''
    ndim = image_data.shape[0]
    nbin = int(ndim/60)
    ndim_vec = np.linspace(0,ndim,ndim)
    nbin_vec = np.linspace(0,ndim,nbin)
    int_nbin_vec = np.zeros(nbin, dtype=np.int)
    for j in range(nbin):
      int_nbin_vec[j] = int(nbin_vec[j])

    '''Setting up filter'''
    background_map = np.zeros(ndim*ndim).reshape(ndim, ndim)
    hamming_filter = ndim/2+1

    '''Calculating data metadata'''
    data_min = np.min(image_data)
    data_max = np.max(image_data)
    data_mean_img = np.mean(image_data)

    '''Smoothing the data'''
    image_data_smooth_1 = np.zeros(ndim*ndim).reshape(ndim,ndim)
    image_data_smooth_2 = np.zeros(ndim*ndim).reshape(ndim,ndim)

    '''Cutting line by line in y'''
    for i in range(ndim):
        min_vec = np.zeros(nbin)
        msky = image_data[i,::]

        for h in range(nbin-1):
            min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])


        nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
        min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

        min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

        image_data_smooth_1[i,::] = min_vec_int_smooth

        if plots_on:
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
    image_data_smooth_11 = scipy.ndimage.gaussian_filter(image_data_smooth_1, kern_px/(2*np.sqrt(2*np.log(2))))

    '''Repeating the process in x'''
    for i in range(ndim):
        min_vec = np.zeros(nbin)
        msky = image_data[::,i]
        for h in range(nbin-1):
            min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])

        nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
        min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

        min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

        image_data_smooth_2[::,i] = min_vec_int_smooth

    image_data_smooth_22 = scipy.ndimage.gaussian_filter(image_data_smooth_2, kern_px/(2*np.sqrt(2*np.log(2))))


    '''Taking the average between the two images.'''
    final = (image_data_smooth_11 + image_data_smooth_22)/2.

    '''Plotting: all'''
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(image_data),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Bin')
    ax.set_title(ffi)

    fb, ab = plt.subplots(2)
    bgy = ab[0].imshow(np.log10(image_data_smooth_11), cmap='Blues_r', origin = 'lower')
    fb.colorbar(bgy,label=r'$log_{10}$(Flux)')
    ab[0].set_xlabel('Bin')
    ab[0].set_ylabel('Bin')
    ab[0].set_title('Background evaluated in y direction')

    bgx = ab[1].imshow(np.log10(image_data_smooth_22), cmap='Blues_r', origin = 'lower')
    fb[1].colorbar(bgx,label=r'$log_{10}$(Flux)')
    ab[1].set_xlabel('Bin')
    ab[1].set_ylabel('Bin')
    ab[1].set_title('Background evaluated in x direction')

    fbf, abf = plt.subplots()
    bgf = abf.imshow(np.log10(final), cmap='Blues_r', origin='lower')
    fbf.colorbar(bgf, label=r'$log_{10}$(Flux)')
    abf.set_xlabel('Bin')
    abf.set_ylabel('Bin')
    abf.set_title('Background averaged across evaluations in x and y')

    plt.show('all')
    plt.close('all')
