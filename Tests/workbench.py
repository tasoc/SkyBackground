import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':







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
