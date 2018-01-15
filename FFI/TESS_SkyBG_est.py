# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:24:36 2016

@author: Dr. Mikkel N. Lund
"""
#===============================================================================
# Packages
#===============================================================================

from __future__ import division
import numpy as np
import sys, os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
mpl.rcParams['font.family'] = 'serif'
import pyfits as pf
import cv2
from scipy import optimize as OP
from scipy import stats
import itertools
import scipy
import copy
import gzip
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
plt.ioff()
import itertools
import corner
from sklearn import linear_model
from astropy.modeling import models, fitting
import warnings
disk_path = '/media/mikkelnl/Elements/'
#disk_path = '/media/Elements/'
#===============================================================================
# Code
#===============================================================================

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def load_file(path, name):
    pixfilename = path + name
    file = gzip.open(pixfilename, 'rb')
    pixfile = pf.open(file, memmap=True, mode='readonly')
    flux = read_pixel_file3(pixfile)
    flux = np.array(flux, dtype=np.float64)
    return flux, flux.flatten()

def read_pixel_file3(pixfile):
    NX=pixfile[0].header['NAXIS1']  # Number of columns in frame
    NY=pixfile[0].header['NAXIS2']
    temp=np.array(pixfile[0].data, dtype='float64')
    dims = len(np.shape(temp))
    if dims==1: #pyfits <3.0
        flux = temp.reshape(NY, NX)
    elif dims==2: #pyfits >=3.0
        flux = temp
    return flux

def plot_frame(flux_ref0, cmap=cm.hot, origin='upper'):
    flux_ref = copy.deepcopy(flux_ref0)
    color_max=np.nanmax(flux_ref.ravel())
    color_min=np.nanmin(flux_ref.ravel())
    flux_ref += np.abs(color_min)
    Flux_mat = np.log10(flux_ref)/np.log10((color_max-color_min))*1.5


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.matshow(Flux_mat, origin=origin, cmap=cmap, interpolation='none')
    ax.set_xlim([-0.5, flux_ref.shape[1]-1+0.5])
    ax.set_ylim([-0.5, flux_ref.shape[0]-1+0.5])
#    ax.contourf(Flux_mat, cmap=cmap, interpolation='none')
#    ax.set_xlim([X.min()-0.5, X.max()+0.5])
#    ax.set_ylim([Y.min()-0.5, Y.max()+0.5])
    fig.patch.set_visible(False)

    ax.tick_params(direction='out', which='both', length=4)
    ax.tick_params( which='major', pad=7, length=6,labelsize='15')

    ax.xaxis.tick_bottom()
    ax.set_xlabel(r'$\rm Pixels$', fontsize=16)
    ax.set_ylabel(r'$\rm Pixels$', fontsize=16)
#    ax.xaxis.set_major_locator(MultipleLocator(500))
#    ax.xaxis.set_minor_locator(MultipleLocator(250))
#    ax.yaxis.set_major_locator(MultipleLocator(500))
#    ax.yaxis.set_minor_locator(MultipleLocator(250))


def Kernel(Flux_flat, trim=0):
    """
    Distribution mode calculation
    """
    kernel = KDE(stats.trim1(np.sort(Flux_flat[np.isfinite(Flux_flat)]), trim))
    kernel.fit(kernel='gau', bw='scott', fft=True, gridsize=200)
    return kernel

def mode_calc(kernel):
    max_guess = kernel.support[np.argmax(kernel.density)]
    def kernel_opt(x): return -1*kernel.evaluate(x)
    MODE3 = OP.fmin_powell(kernel_opt, max_guess, disp=0)
    return MODE3

def MAD(F, relative_to='median'):
    if relative_to=='median':
        MAD = 1.4826 * stats.nanmedian( np.abs( F - stats.nanmedian(F) ) )
    else:
        MAD = 1.4826 * stats.nanmedian( np.abs( F - relative_to ) )
    return MAD

def Sum_stats(Flux_flat):
    MED = stats.nanmedian(Flux_flat)
    MEAN1 = stats.nanmean(Flux_flat)
    MEAN2 = stats.nanmean(stats.trim1(np.sort(Flux_flat), 0.25))

    MODE1 = 3*MED - 2*MEAN1
    MODE2 = 3*MED - 2*MEAN2

    kernel = Kernel(Flux_flat)
    max_guess = kernel.support[np.argmax(kernel.density)]
    def kernel_opt(x): return -1*kernel.evaluate(x)
    MODE3 = OP.fmin_powell(kernel_opt, max_guess, disp=0)

    MAD1 = 1.4826 * stats.nanmedian( np.abs( Flux_flat[(Flux_flat < MODE1)] - MODE1 ) )
    MAD2 = 1.4826 * stats.nanmedian( np.abs( Flux_flat[(Flux_flat < MODE2)] - MODE2 ) )
    MAD3 = 1.4826 * stats.nanmedian( np.abs( Flux_flat[(Flux_flat < MODE3)] - MODE3 ) )

    STD1 = stats.nanstd(Flux_flat)
    STD2 = stats.nanstd(stats.trim1(np.sort(Flux_flat), 0.25))

    return MED, MEAN1, MEAN2, MODE1, MODE2, MODE3, MAD1, MAD2, MAD3, STD1, STD2


def local_regression_plane_ransac(neighborhood, thresh_factor=1):
    """
    Computes parameters for a local regression plane using RANSAC
    """
    XY = neighborhood[:,:2]
    Z  = neighborhood[:,2]
    Thresh = thresh_factor*MAD(Z.flatten())
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=Thresh)
    ransac.fit(XY, Z)

    inlier_mask = ransac.inlier_mask_
    coeff = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_
    return inlier_mask, coeff, intercept

def sigma_clip(k, F, itt=10):
    for j in range(itt):
        kernel = Kernel(F, trim=0)
        M = mode_calc(kernel)
        S = np.median(np.abs(F-M))*1.48
        F=F[(F<M+k*S)]
    return F


class PlaneModel:
    """linear system solved using linear least squares

    """
    def __init__(self,debug=False,order=2):
        self.order = order
        self.debug = debug

    def fit(self, data):
        # Fit the data using astropy.modeling
        p_init = models.Polynomial2D(degree=self.order)
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            p = fit_p(p_init, data[:,0], data[:,1], data[:,2])
        return p

    def get_error(self, data, p):
        D = (data[:,2] - p(data[:,0], data[:,1]))**2
        err_per_point = D.flatten() # sum squared error per row
        return err_per_point

class PlaneModel2:
    """linear system solved using linear least squares

    """
    def __init__(self,debug=False,order=2,weights=None):
        self.order = order
        self.debug = debug
        self.weights = weights
        self.coeff = None
        self.r = None
        self.rank = None
        self.s = None

    def Amet(self, X2, Y2):
        if self.order==0:
            A = np.array([X2*0+1,]).T
        if self.order==1:
            A = np.array([X2*0+1, X2, Y2]).T
        if self.order==2:
            A = np.array([X2*0+1, X2, X2**2,Y2, Y2**2, X2*Y2]).T
        if self.order==3:
            A = np.array([X2*0+1, X2, X2**2, X2**3, Y2, Y2**2, Y2**3, X2*Y2, X2**2*Y2, X2*Y2**2]).T
        return A

    def evaluate(self, X, Y, m):
        A = PlaneModel2.Amet(self, X, Y)
        B = np.dot(A, m)
        return B

    def fit(self, data):
        X2 = data[:,0]#.flatten()
        Y2 = data[:,1]#.flatten()
        A = PlaneModel2.Amet(self, X2, Y2)
        B = data[:,2]#.flatten()
        if not self.weights is None:
            W = self.weights/np.sum(self.weights)
            W = np.diag(np.sqrt(W))
            Aw = np.dot(W,A)
            Bw = np.dot(B,W)
            coeff, r, rank, s = np.linalg.lstsq(Aw, Bw)
        else:
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        self.coeff = coeff
        self.r = r
        self.rank = rank
        self.s = s

        return self

def simple_mode(F, cut=0.75):
    MED = stats.nanmedian(stats.trim1(np.sort(F.flatten()), cut))
    MEAN = stats.nanmean(stats.trim1(np.sort(F.flatten()), cut))
    MODE = 3*MED - 2*MEAN
    return MODE

def non_simple_mode(F, neighborhood, itt_field, cut=25, thresh_factor=0.5):
    inlier_masks = np.zeros(neighborhood.shape[0])
    for kk in range(itt_field):
        try:
            inlier_mask, coeff, intercept = local_regression_plane_ransac(neighborhood, thresh_factor=thresh_factor)
            inlier_masks += inlier_mask
        except:
            continue

    inlier_mask_arr = inlier_masks.reshape((F.shape[0], F.shape[1]))

    FFF2 = F[(inlier_mask_arr>itt_field/2)].flatten()
    FFF2 = FFF2[(FFF2<np.percentile(FFF2,cut))]

#    print len(FFF2)
#    FFF3 = F.copy()
#    plot_frame(F)
#    FFF3[(inlier_mask_arr<itt_field/2)]=0
#    plot_frame(FFF3)
#    plt.show()

    kernel = Kernel(FFF2, trim=0)
    MODE = mode_calc(kernel)
    return MODE, kernel



def BG(Flux, Flux2=None, size=64, itt_field=1, itt_ransac=200, plot=False, simple=True):

    plot_frame(Flux)
    plt.show()
    A = blockshaped(Flux, size, size)
    Modes = np.zeros([Flux.shape[0]/size, Flux.shape[0]/size])

    if not Flux2 is None:
        A2 = blockshaped(Flux2, size, size)
        Modes2 = np.zeros_like(Modes)


    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)


    if not simple:
        X0, Y0 = np.meshgrid(np.arange(size), np.arange(size))
        neighborhood0 = np.zeros([len(X0.flatten()), 3])
        neighborhood0[:, 0] = X0.flatten()
        neighborhood0[:, 1] = Y0.flatten()


    i =0
    for j in range(Modes.shape[0]):
        for jj in range(Modes.shape[0]):
            F = A[i, ::]


            if simple:
                Modes[j,jj] = simple_mode(F, cut=0.75)
            else:
                neighborhood0[:, 2] = F.flatten()
                Modes[j,jj], kernel = non_simple_mode(F, neighborhood0, itt_field, cut=90, thresh_factor=0.5)


            if not Flux2 is None:
                F2 = A2[i, ::]
                kernel2 = Kernel(F2.flatten(), trim=0)
                Modes2[j,jj] = mode_calc(kernel2)
                print i, j, jj, Modes[j,jj], Modes2[j,jj], Modes[j,jj]/Modes2[j,jj]
            else:
                pass
    #            print i, Modes[i], MODE2, Modes[i]/MODE2


            if not simple:
                if plot:
                    ax.plot(kernel.support, kernel.density, 'k', alpha=0.2)
                    ax.scatter(Modes[j,jj], kernel.evaluate(Modes[j,jj]), marker='o', color='r', alpha=0.2)

                    ax.set_xlabel('Flux')
                    ax.set_ylabel('KDE')
                    ax.set_ylim(ymin=0)
                    ax.tick_params(direction='out', which='both', pad=5)


            i += 1



    X, Y = np.meshgrid(np.arange(Modes.shape[1]), np.arange(Modes.shape[0]))
    pixel_factor = (size-1)
    X = (X+0.5)*pixel_factor
    Y = (Y+0.5)*pixel_factor

    neighborhood = np.zeros([len(Modes.flatten()), 3])
    neighborhood[:, 0] = X.flatten()
    neighborhood[:, 1] = Y.flatten()
    neighborhood[:, 2] = Modes.flatten()


#    X, Y = np.meshgrid(np.arange(Flux.shape[1]), np.arange(Flux.shape[0]))
#    neighborhood = np.zeros([len(Flux.flatten()), 3])
#    neighborhood[:, 0] = X.flatten()
#    neighborhood[:, 1] = Y.flatten()
#    neighborhood[:, 2] = Flux.flatten()


    ############### RANSAC ON MODES ######################
    inlier_masks = np.zeros(neighborhood.shape[0])
    for k in range(itt_ransac):
        inlier_mask, coeff, intercept = local_regression_plane_ransac(neighborhood, thresh_factor=1)
        inlier_masks += inlier_mask

    if plot:
        inlier_mask_arr = inlier_masks.reshape((Flux.shape[0]/size,Flux.shape[0]/size))
#        inlier_mask_arr = inlier_masks.reshape((Flux.shape[0],Flux.shape[0]))
        plt.matshow(inlier_mask_arr, origin='lower', cmap=cm.bone)
#        plt.show()
    ######################################################

    ############### 2D Polyfit ########################
#    maybeinliers = neighborhood[inlier_mask]
#    maybemodel = PlaneModel(order=2).fit(maybeinliers)
#    print(maybemodel.parameters)

    maybemodel2 = PlaneModel2(order=2,weights=inlier_masks).fit(neighborhood)
    w_coeff = maybemodel2.coeff
#    print(maybemodel2.coeff)

#    print(Modes)

#    maybemodel3 = PlaneModel2().fit(maybeinliers)
#    w_coeff = maybemodel3.coeff
#    print(maybemodel3.coeff)

    if not Flux2 is None:
#        return w_coeff, Modes_arr, Modes_arr2, X, Y
        return w_coeff, Modes, Modes2, X, Y
    else:
#        return w_coeff, Modes_arr, X, Y
        return w_coeff, Modes, X, Y

#==============================================================================
#
#==============================================================================

def make_sumimage(path, NO=None):

    if NO is None:
        NO = len(os.listdir(path))

    num = 0
    for h in range(NO):
        no = str(h).zfill(6)
#        name = 'download_conference.php?file=TDA1%2Fdata%2Fffi_south%2Fsimulated%2Fsimulated_06h00m00s-66d33m39s_sub2048x2048_'+no+'.fits.gz'
        name = 'simulated_18h00m00s+66d33m39s_sub2048x2048_'+no+'.fits.gz'
        print(h, NO)
        if h==0:
            Flux, Flux_flat = load_file(path, name)
            num += 1
        else:
            try:
                Flux0, Flux_flat0 = load_file(path, name)
                Flux += Flux0
                num += 1
            except:
                continue


    return Flux/num


#==============================================================================
#
#==============================================================================

#path = '/media/mikkelnl/Elements/TESS/TDA1_analysis/South_FFI/'
path = '/media/mikkelnl/Elements/TESS/TDA1_analysis/North_FFI/files/'


#path2 = '/media/mikkelnl/Elements/TESS/TDA1_analysis/'
path2 = '/media/mikkelnl/Elements/TESS/TDA1_analysis/North_FFI/'
name2 = 'backgrounds.fits.gz'

Flux2, Flux_flat2 = load_file(path2, name2)
#Flux2 = Flux2[0:1024, 0:1024]
#Flux2 = Flux2[::-1, :]
#NO = 1
size=128
run_sumim=False

#Coeffs = np.zeros([NO, 7])
#for h in range(NO):
#    no = str(h).zfill(6)
##    try:
#    name = 'download_conference.php?file=TDA1%2Fdata%2Fffi_south%2Fsimulated%2Fsimulated_06h00m00s-66d33m39s_sub2048x2048_'+no+'.fits.gz'
##    name = 'simulated_18h00m00s+66d33m39s_sub2048x2048_'+no+'.fits.gz'
#    Flux, Flux_flat = load_file(path, name)
#    Flux = Flux[0:1024, 0:1024]
#
#    if h==0:
#        w_coeff, Modes_arr, Modes_arr2, X, Y = BG(Flux, Flux2, size=size, simple=False, plot=True)
#    else:
#        w_coeff, Modes_arr, X, Y = BG(Flux)
#
#    Coeffs[h,0] = h
#    Coeffs[h,1::] = w_coeff
##    except:
##        continue
#
#    print(h, w_coeff)

w_coeff, Modes_arr, Modes_arr2, X, Y = BG(Flux, Flux2, size=size, simple=False, plot=True)


#####################################################

#idx = (Coeffs[:, 1]!=0)
#corner.corner(Coeffs[idx, 1::], labels=['V1', 'V2', 'V3', 'V4', 'V5','V6'])

Modes = Modes_arr.flatten()
Modes2 = Modes_arr2.flatten()

Xx, Yy = np.meshgrid(np.arange(Flux.shape[0]), np.arange(Flux.shape[0]))

M = PlaneModel2().evaluate(Y,X,w_coeff)
M2 = PlaneModel2().evaluate(Yy,Xx,w_coeff)

plot_frame(Modes_arr)
plot_frame(Modes_arr2)


plot_frame(M)

plot_frame(Modes_arr2/M)
plot_frame(Modes_arr/M)

plot_frame(Flux, cmap=cm.seismic)
plot_frame(Flux2, cmap=cm.hot)
plot_frame(Flux-M2, cmap=cm.seismic)
plot_frame(Flux2-M2, cmap=cm.seismic)
plot_frame(Flux-Flux2, cmap=cm.seismic)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Modes_arr, rstride=1, cstride=1, alpha=0.2)
ax.plot_surface(X, Y, Modes_arr2, rstride=1, cstride=1, alpha=0.2, color='r')
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')


fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.plot_surface(X, Y, Modes_arr-M, rstride=1, cstride=1, alpha=0.2)
plt.xlabel('X')
plt.ylabel('Y')
ax2.set_zlabel('Z')
ax2.axis('equal')
ax2.axis('tight')


fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
ax3.plot_surface(X, Y, Modes_arr2-M, rstride=1, cstride=1, alpha=0.2)
plt.xlabel('X')
plt.ylabel('Y')
ax3.set_zlabel('Z')
ax3.axis('equal')
ax3.axis('tight')

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
ax4.plot_surface(X, Y, Modes_arr-Modes_arr2, rstride=1, cstride=1, alpha=0.2)
plt.xlabel('X')
plt.ylabel('Y')
ax4.set_zlabel('Z')
ax4.axis('equal')
ax4.axis('tight')

plt.figure()
plt.hist((Modes_arr-M).flatten(), 500)

plt.figure()
plt.hist(Modes/Modes2, 500)

plt.figure()
plt.hist(Modes, 500)

plt.figure()
plt.hist(Modes2, 500)


plt.show()











#==============================================================================
#
#==============================================================================



#def ransac(data,model,n,k,t,d,debug=False,return_all=False):
#    """fit model parameters to data using the RANSAC algorithm
#
#   This implementation written from pseudocode found at
#   http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
#
#   {{{
#   Given:
#       data - a set of observed data points
#       model - a model that can be fitted to data points
#  45     n - the minimum number of data values required to fit the model
#  46     k - the maximum number of iterations allowed in the algorithm
#  47     t - a threshold value for determining when a data point fits a model
#  48     d - the number of close data values required to assert that a model fits well to data
#  49 Return:
#  50     bestfit - model parameters which best fit the data (or nil if no good model is found)
#  51 iterations = 0
#  52 bestfit = nil
#  53 besterr = something really large
#  54 while iterations < k {
#  55     maybeinliers = n randomly selected values from data
#  56     maybemodel = model parameters fitted to maybeinliers
#  57     alsoinliers = empty set
#  58     for every point in data not in maybeinliers {
#  59         if point fits maybemodel with an error smaller than t
#  60              add point to alsoinliers
#  61     }
#  62     if the number of elements in alsoinliers is > d {
#  63         % this implies that we may have found a good model
#  64         % now test how good it is
#  65         bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
#  66         thiserr = a measure of how well model fits these points
#  67         if thiserr < besterr {
#  68             bestfit = bettermodel
#  69             besterr = thiserr
#  70         }
#  71     }
#  72     increment iterations
#  73 }
#  74 return bestfit
#  75 }}}
#    """
#    iterations = 0
#    bestfit = None
#    besterr = np.inf
#    best_inlier_idxs = None
#    while iterations < k:
#        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
#        maybeinliers = data[maybe_idxs]
##        print maybeinliers.shape
#        test_points = data[test_idxs]
##        print test_points.shape
#        maybemodel = model.fit(maybeinliers)
##        print maybemodel.parameters
#        test_err = model.get_error( test_points, maybemodel)
#        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
#        alsoinliers = data[also_idxs,:]
#        if debug:
#            print 'test_err.min()',test_err.min()
#            print 'test_err.max()',test_err.max()
#            print 'np.mean(test_err)',np.mean(test_err)
#            print 'iteration %d:len(alsoinliers) = %d'%(iterations,len(alsoinliers))
#        if len(alsoinliers) > d:
#            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
#            bettermodel = model.fit(betterdata)
#            better_errs = model.get_error( betterdata, bettermodel)
#            thiserr = np.mean( better_errs )
#            if thiserr < besterr:
#                bestfit = bettermodel
#                besterr = thiserr
#                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )
#        iterations+=1
#    if bestfit is None:
#        raise ValueError("did not meet fit acceptance criteria")
#    if return_all:
#        return bestfit.parameters, {'inliers':best_inlier_idxs}
#    else:
#        return bestfit.parameters
#
#def random_partition(n,n_data):
#    """return n random rows of data (and also the other len(data)-n rows)"""
#    all_idxs = np.arange( n_data )
#    np.random.shuffle(all_idxs)
#    idxs1 = all_idxs[:n]
#    idxs2 = all_idxs[n:]
#    return idxs1, idxs2
#
#


#
#
#
#def local_regression_poly_ransac(neighborhood):
#    """
#    Computes parameters for a local regression plane using RANSAC
#    """
#    order = 1
#    debug = True
#    model = PlaneModel(debug=debug,order=order)
#
#    # run RANSAC algorithm
#    ransac_fit, ransac_data = ransac(neighborhood, model, 400, 2000, 1000, 2, debug=debug, return_all=True)# misc. parameters
#
#    return ransac_fit, ransac_data
