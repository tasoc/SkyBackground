#!/bin/env python
# -*- coding: utf-8 -*-

#AUTHORS:
#T'DA3 | Oliver James HALL | 2017

import numpy as np
from matplotlib import pyplot as plt
import sys

'''
A simple code to see how the Buzasi+15 method approximates a synthetic
background on a synthetic image of a point source with incresing image size.

The method involves taking the median of the lowest 20% of the image as the
background estimate.

The point source is approximated as  2D gaussian centered at the rounded center
pixel of the image, with a standard deviation of 1.5 pixels.

The background level is assumed to be 0.1% of the stellar flux as the location
of the point source.
'''

if __name__ == "__main__":
    plt.close('all')
    plots_on = True
    size = np.arange(5,15,1)

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
        flux = 100 * (1/0.0001)
        sig = 1.5

        #Draw PSF from peaks with 1.5 pixel falloff, and add background
        d = flux * np.exp( (-(pt[0] - Y)**2 - (pt[1] - X)**2)/sig**2)
        d += bkg

        bkg_est[idx] = np.median(d[d < np.nanpercentile(d,[20])])

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
    ax.axhline(bkg, c='r', label='True background (0.1%)')
    ax.set_xlabel('Number of pixels per side')
    ax.set_ylabel('Background level (Flux)')
    ax.set_title('Improvement in background estimation with postage stamp size')
    ax.legend(loc='best',fancybox=True)
    plt.show()
















    sys.exit()

    '''
    Stellar radius: Solar
    Planet radius: 0.5Rearth - 2Rjup
    '''
    pRii = np.linspace(0.5*6371, 69911,1000)
    sR = 695700

    delta = (pRii/sR)**2
    Kep2R = 16.71 * 6371 #km
    Kep2S = 1.84 * 695700 #km
    delta_hat = (pRii/Kep2S)**2

    # fig, ax = plt.subplots()
    # ax.plot(pRii/69911,delta*100)
    # ax.set_xlabel('Planetary Radius (Rjup)')
    # ax.set_ylabel('Fractional depth (%)')
    # ax.set_title('Range of transit depths for planets around a solar radius star')
    # ax.scatter(Kep2R/69911, (Kep2R/Kep2S)**2*100,c='r',label='HAT-P-7b')
    # ax.plot(pRii/69911,delta_hat*100,c='r',linestyle='--')
    # ax.legend(loc='best',fancybox=True)
    # plt.show()

    '''
    1] The smaller the star, the larger the impact of planetary size.
    A] We want a low-luminosity target, ideally, with a small transit.

    Need to figure out:
        -Baseline flux of a faint target (through magnitude?)
        -Estimate a background level---
            +Estimate from Kepler value and impose 25% offset
        -Plot radius against depth offset (new - old)
        -Determine normal distributed noise on my fits
        -Determine minimum transit required for noise threshold
    '''

    '''The following is for HAT-P-7b'''
    baseline = 1000.  #Electrons per second
    offset_f = 1.     #Electrons per second
    offset_p = offset_f/baseline

    offset = np.linspace(0.001,0.1,10)
    transits = np.linspace(2e-5,1e-2,1000) #Fractional depth
    radii = np.sqrt(transits) * Kep2S / 69911

    fig, ax = plt.subplots()
    ax.plot(radii,transits*100,c='k',linestyle='--',zorder=999,label='1-to-1 relation')

    for idx, off in enumerate(offset):
        Fot = baseline
        sB = off * Fot
        Fmt = baseline - transits*baseline
        delta = 1-(Fmt + sB)/(Fot + sB)
        ax.plot(radii, delta*100,zorder=100, label=str(off*100)+'\%')

    ax.set_xlabel('Planetary Radius (Rjup) around HAT-P-7')
    ax.set_ylabel(r'Perturbed fractional depth (HAT-P-7 based) (\%)')
    ax.set_title(r'Change in fractional depth due to background perturbation')
    ax.grid()
    ax.legend(loc='best',fancybox=True)
    plt.show()

    '''Lets type up the algebra... [HAT-P-7b]'''
    Rsun = 695700 #km
    Rjup = 69911. #km

    '''CASE 1: B{0-100} v F{200-1000} v dR'''
    B = np.linspace(0.001,10,20) #0
    F = np.linspace(200,1000,20)
    BB, FF = np.meshgrid(B, F)
    d = (Rjup/Rsun)**2

    Rp2 = Rsun**2 - Rsun**2 * (((1-d)*FF-BB)/(FF-BB))
    Rp = np.sqrt(Rp2)
    dR = (Rp - Rjup)*100/Rjup

    plt.close()
    fig, ax = plt.subplots(2)
    im = ax[0].contourf(BB, FF, dR, 100, cmap='viridis')
    ax[0].set_xlabel(r'Background level')
    ax[0].set_ylabel('Flux level')
    fig.colorbar(im,label=r'$\Delta$R (\%)',ax=ax[0])

    bg = ax[1].contourf(BB,FF,(BB*100/FF), 100, cmap='Blues_r')
    ax[1].set_xlabel(r'Background level')
    ax[1].set_ylabel('Flux level')
    fig.colorbar(bg,label=r'Background level as a \% of the stellar flux',ax=ax[1])

    fig.tight_layout()
    plt.show()

    '''CASE 2: F{B/0.1% - B/1.0%} V \delta{2e-4,2e-1}'''
    x = np.linspace(0.001,0.01,20) #Background offset as a fractio of total flux
    d = np.linspace(2e-4,1e-1,20)     #Fractional depth of transit
    xx, dd = np.meshgrid(x, d)

    dn = 1 - ( ((1-dd)-xx) / (1-xx) )   #New fractional depth of transit
    Ddn_per = (dn - dd)*100/dd
    Ddn_mag = (dn-dd)

    plt.close()
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_axes([0.1,0.2,0.35,0.6])
    ax2 = fig.add_axes([0.55,0.2,0.35,0.6],sharex=ax1,sharey=ax1)

    im0 = ax1.contourf(xx*100,dd,Ddn_per, 100)
    ax1.set_xlabel(r'Background offset as a fraction of the signal flux (\%)')
    ax1.set_ylabel(r'Original fractional depth of transit')
    im1 = ax2.contourf(xx*100,dd,Ddn_mag,100)

    ax1.hlines(y=.06, xmin=0.1, xmax=0.22, color='r',alpha=.5)
    ax1.vlines(x=0.22, ymin = 2e-4, ymax=0.06, color='r',alpha=.5)
    ax1.scatter(0.22,0.06,c='r',marker='.', alpha=.5,label='HAT-P-7b w Hall+17')
    ax2.hlines(y=.06, xmin=0.1, xmax=0.22, color='r',alpha=.5)
    ax2.vlines(x=0.22, ymin = 2e-4, ymax=0.06, color='r',alpha=.5)
    ax2.scatter(0.22,0.06,c='r',marker='.', alpha=.5,label='HAT-P-7b w Hall+17')

    ax1.legend(loc='best',fancybox=True)
    ax2.legend(loc='best',fancybox=True)
    fig.colorbar(im0,extend='both',label=r'Percentage difference in fractional depth (\%)',ax=ax1)
    fig.colorbar(im1,extend='both',label=r'Difference in fractional depth',ax=ax2)
    fig.suptitle(r'Difference in $\delta$ with changing $\delta$ and target flux level at constant background offset')

    plt.show()
