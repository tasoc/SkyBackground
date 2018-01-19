#!/bin/env python
# -*- coding: utf-8 -*-

"""
Tests the 4 different types of background estimation on simulated data of
increasing complexity, while outputting useful metrics.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy import interpolate

import sys
import glob

sys.path.append('../FFI/')
from CvE_estimate import fit_background as CEfit_bkg
from RH_estimate import fit_background as RHfit_bkg
from OJH_estimate import fit_background as OHfit_bkg
from MNL_estimate import fit_background as MLfit_bkg
from Functions import *


if __name__ == "__main__":
    plt.close('all')

    ffi, bkg = get_sim(style='complex')

    print('fitting ML')
    ML = MLfit_bkg(ffi,order=3)
    print('fitting OJH')
    OJH = OHfit_bkg(ffi,order=3)
    print('fitting RH')
    RH  = RHfit_bkg(ffi)
    print('fitting CvE')
    CvE = CEfit_bkg(ffi, percentile=50)

    resML = ML-bkg
    resOJH = OJH-bkg
    resRH  = RH-bkg
    resCvE = CvE-bkg

    '''Plotting: all'''
    plt.close('all')
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')
    ax.set_title('simulated background')

    fC, aC = plt.subplots()
    diffC = aC.imshow(np.log10(CvE) - np.log10(bkg), origin='lower')
    fC.colorbar(diffC, label='Estimated Bkg - True Bkg (both in log10 space)')
    aC.set_title('CvE_estimate | Estimated bkg - True bkg')

    fH, aH = plt.subplots()
    diffH = aH.imshow(np.log10(OJH) - np.log10(bkg),origin='lower')
    fH.colorbar(diffH, label=r'$log_{10}$(Flux)')
    aH.set_title('OJH_estimate | Estimated bkg - True bkg')

    fRH, aRH = plt.subplots()
    diffRH = aRH.imshow(np.log10(RH) - np.log10(bkg),origin='lower')
    fRH.colorbar(diffRH, label=r'$log_{10}$(Flux)')
    aRH.set_title('RH_estimate | Estimated bkg - True bkg')

    fML, aML = plt.subplots()
    diffML = aML.imshow(np.log10(ML) - np.log10(bkg),origin='lower')
    fML.colorbar(diffML, label=r'$log_{10}$(Flux)')
    aML.set_title('ML_estimate | Estimated bkg - True bkg')

    fres, ares = plt.subplots(4,sharex=True)
    ares[0].plot(resML.ravel()[::2048],alpha=.4,linewidth=1,marker='v')
    ares[0].set_title('ML method')
    ares[0].axhline(0.,c='r')
    ares[1].plot(resCvE.ravel()[::2048],alpha=.4,linewidth=1,marker='v')
    ares[1].set_title('CvE method')
    ares[1].axhline(0.,c='r')
    ares[2].plot(resOJH.ravel()[::2048],alpha=.4,linewidth=1,marker='v')
    ares[2].set_title('OJH method')
    ares[2].axhline(0.,c='r')
    ares[3].plot(resRH.ravel()[::2048],alpha=.4,linewidth=1,marker='v')
    ares[3].set_title('RH method')
    ares[3].axhline(0.,c='r')
    fres.tight_layout()

    fhist, ahist = plt.subplots(2,2, sharex=True)
    ahist[0,0].hist((CvE-bkg).ravel(),histtype='step',bins=int(np.sqrt(len(bkg.ravel()))))
    ahist[1,0].hist((OJH-bkg).ravel(),histtype='step',bins=int(np.sqrt(len(bkg.ravel()))))
    ahist[0,1].hist((RH-bkg).ravel(),histtype='step',bins=int(np.sqrt(len(bkg.ravel()))))
    ahist[1,1].hist((ML-bkg).ravel(),histtype='step',bins=int(np.sqrt(len(bkg.ravel()))))

    ahist[0,0].set_title('Histogram of residuals for CvE')
    ahist[1,0].set_title('Histogram of residuals for OJH')
    ahist[0,1].set_title('Histogram of residuals for RH')
    ahist[1,1].set_title('Histogram of residuals for ML')
    ahist[1,1].set_xlabel('Background Estimate - True Background')

    ahist[0,0].axvline(0.,c='r')
    ahist[1,0].axvline(0.,c='r')
    ahist[0,1].axvline(0.,c='r')
    ahist[1,1].axvline(0.,c='r')
    fhist.tight_layout()

    percoffset = 100*resCvE/bkg
    medCvE = np.median(100*resCvE/bkg)
    stdCvE = np.std(100*resCvE/bkg)
    medOH = np.median(100*resOJH/bkg)
    stdOH = np.std(100*resOJH/bkg)
    medRH = np.median(100*resRH/bkg)
    stdRH = np.std(100*resRH/bkg)
    medML = np.median(100*resML/bkg)
    stdML = np.std(100*resML/bkg)

    print('Median offset & standard deviation on residuals:')
    print('CvE offset: '+str(np.round(medCvE,3))+r"% $\pm$ "+str(np.round(stdCvE,3))+'%')
    print('OJH offset: '+str(np.round(medOH,3))+r"% $\pm$ "+str(np.round(stdOH,3))+'%')
    print('ML offset: '+str(np.round(medML,3))+r"% $\pm$ "+str(np.round(stdML,3))+'%')
    print('RH offset: '+str(np.round(medRH,3))+r"% $\pm$ "+str(np.round(stdRH,3))+'%')
    cc, button = close_plots()
    button.on_clicked(close)

    plt.show('all')
    plt.close('all')
