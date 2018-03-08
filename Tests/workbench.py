import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
from tqdm import tqdm
sys.path.append('../FFI/')
from Functions import *
import ClosePlots as cp

def get_gaussian(X, Y, A, (mux, muy), (sigma_x, sigma_y)):
    return  A   * np.exp(-0.5 * (mux - X)**2 / sigma_x**2) \
                * np.exp(-0.5 * (muy- Y)**2 / sigma_y**2)

if __name__ == '__main__':
    shape = (2048,2048)
    X, Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))

    ffi, bkg = load_files('ffi_north')

    starraw = ffi - bkg

    if os.path.isfile('complex_sim_bkg.fits'):
        bkg = pyfits.open('../Tests/complex_sim_bkg.fits')[0].data

    # nstars = 5000
    # stars  = np.zeros(ffi.shape)
    # locx = np.random.rand(nstars) * (shape[1]-1)
    # locy = np.random.rand(nstars) * (shape[0]-1)
    # heights = np.random.exponential(size=nstars)
    # ffimed =  5058  #The median of the ffi data
    # ffiheights = np.median(heights)
    # heights *= (ffimed/ffiheights)  #Scaling the heights to something ffi-like
    # ws = np.random.rand(nstars) * 2
    #
    # for s in tqdm(range(nstars)):
    #     stars += get_gaussian(X, Y, heights[s], (locx[s], locy[s]), (ws[s],ws[s]))

    fig, ax = plt.subplots()
    semisim = np.random.normal(starraw+bkg,2000,shape)
    ax.imshow(np.log10(semisim), origin='lower',cmap='Blues_r')

    f2, ax2 = plt.subplots()
    ax2.imshow(np.log10(ffi), origin='lower',cmap='Blues_r')

    sim = np.random.normal(stars*10+bkg, 2000, shape)
    f3, a3 = plt.subplots()
    a3.imshow(np.log10(sim), origin='lower', cmap='Blues_r')


    cp.show()

    sys.exit()

    shape = (2048,2048)
    xx = np.arange(shape[1])
    yy = np.arange(shape[0])
    X, Y = np.meshgrid(xx, yy)
    a = -1.05    #Slope on spiffy bkg
    b = -3.55    #Slope on spiffy bkg

    z = 90000    #Median height of spiffy background
    sigma = 2000 #Sigma is std on spiffy bkg (2% spread)

    nstars = 2000
    orig_stars = np.zeros(X.shape)
    locx = np.random.rand(nstars) * (shape[1]-1)
    locy = np.random.rand(nstars) * (shape[0]-1)
    for s in tqdm(range(nstars)):
        height = np.random.exponential()*1
        w = 20
        orig_stars +=  get_gaussian(X, Y, 1, (locx[s], locy[s]),(w, w))

    #Normalising the bkg_stars component to be a 10% fraction
    bkg_stars = orig_stars/orig_stars.max()
    bkg_stars *= 0.05
    bkg_stars += 1

    #Defining the other components
    bkg_slope = a*X + b*Y + z
    bkg_gauss = get_gaussian(X, Y, 0.2, (650,1650), (600,400)) + 1

    fig, ax = plt.subplots(1,3,figsize=(15,4))
    c0 = ax[0].imshow(bkg_slope)
    c1 = ax[1].imshow(bkg_gauss)
    c2 = ax[2].imshow(bkg_stars)

    conv = bkg_slope * bkg_gauss * bkg_stars
    f2, a2 = plt.subplots()
    c= a2.imshow(np.log10(conv))
    f2.colorbar(c)

    cc, button = close_plots()
    button.on_clicked(close)

    sim = np.random.normal(conv, sigma, shape)
    ff, aa = plt.subplots()
    cc = aa.imshow(np.log10(sim), cmap='Blues_r')
    ff.colorbar(cc)
    plt.show()

    sys.exit()
    '''My data goes from 0 to 10
    We want first and last 20% superspaced

    '''
    xlen = 2048     #Pixels in x and y
    ylen = 2048
    perc = 0.2     #Percentage of image to be double spaced
    nside = 10     #Number of points per side without increased density

    lx = xlen/(nside+2)
    ly = ylen/(nside+2)

    superx = lx/2
    supery = ly/2

    nsuper = perc*nside*2
    nreg = (1-2*perc)*nside

    xx = perc*xlen
    yy = perc*xlen

    xlocs_left = np.linspace(superx, xx-superx, nsuper)
    ylocs_left = np.linspace(supery, yy-supery, nsuper)

    xlocs_right = np.linspace(xlen-xx+superx, xlen-superx, nsuper)
    ylocs_right = np.linspace(ylen-yy+supery, ylen-supery, nsuper)

    xlocs_mid = np.linspace(xx,xlen-xx,nreg)
    ylocs_mid = np.linspace(yy,ylen-yy,nreg)


    xx = np.append(xlocs_mid, [xlocs_left, xlocs_right])
    yy = np.append(ylocs_mid, [ylocs_left,ylocs_right])
    X, Y = np.meshgrid(xx, yy)

    plt.scatter(X, Y, s=1)
    plt.show()
