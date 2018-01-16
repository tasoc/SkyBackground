import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''My data goes from 0 to 10
    We want first and last 20% superspaced

    '''
    lenx = 10
    leny = 10

    superx = 0.1*10
    supery = 0.1*10

    nsuper = 7 #Number of supergrid points
    nreg = 14

    lx = superx/(nsuper+2)
    ly = supery/(nsuper+2)

    xlocs_left = np.linspace(lx, superx-lx, nsuper)
    ylocs_left = np.linspace(ly, supery-ly, nsuper)

    xlocs_right = np.linspace(lenx-superx+lx, lenx-lx, nsuper)
    ylocs_right = np.linspace(leny-supery+ly, leny-ly, nsuper)

    xlocs_mid = np.linspace(superx,lenx-superx,nreg)
    ylocs_mid = np.linspace(supery,leny-supery,nreg)


    xx = np.append(xlocs_left, xlocs_mid)
    xx = np.append(xx, xlocs_right)
    yy = np.append(ylocs_left, ylocs_mid)
    yy = np.append(yy, ylocs_right)
    X, Y = np.meshgrid(xx, yy)

    plt.scatter(X, Y, s=1)
    plt.show()
