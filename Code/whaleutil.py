"""

2015 September 30

Shane Bussmann

Find the whale.  Model the whale as a rectangle with aspect ratio = 3.0.

"""

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from scipy.ndimage import gaussian_filter

def xy_rotate(x, y, x0, y0, phi):
    phirad = np.deg2rad(phi)
    xnew = (x - x0) * np.cos(phirad) + (y - y0) * np.sin(phirad)
    ynew = (y - y0) * np.cos(phirad) - (x - x0) * np.sin(phirad)
    return (xnew,ynew)

def ellipse_2d(x, y, par):
    (xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
    r_ell_sq = ((xnew**2)*par[4] + (ynew**2)/par[4]) / np.abs(par[1])**2
    ellipse = r_ell_sq.copy()
    ellipse[:] = 0.
    inside = r_ell_sq < 1
    ellipse[inside] = par[0]

    #import matplotlib.pyplot as plt
    #plt.imshow(r_ell_sq, origin='lower', vmax=10*par[1])
    #plt.colorbar()
    #plt.contour(ellipse)
    #plt.show()
    return ellipse

def whale_2d(x, y, par):
    # the head and body of the whale
    e1 = ellipse_2d(x, y, par)

    ## the tail of the whale
    #r1 = par[1] / 3.
    #q1 = 0.5
    #b1 = r1 * np.sqrt(q1)
    #a0 = par[1] / np.sqrt(par[4])
    #d = a0 + b1
    #dx = d * np.cos(par[5])
    #dy = d * np.sin(par[5])
    #x1 = par[2] - dx
    #y1 = par[3] - dy
    #phi1 = par[5] - 90.
    #par2 = [par[0], r1, x1, y1, q1, phi1]
    #e2 = ellipse_2d(x, y, par2)
    #import matplotlib.pyplot as plt
    #plt.contour(e1)
    #plt.contour(e2)
    #plt.show()
    #import pdb; pdb.set_trace()
    #print(par)
    #print(par2)

    return e1# + e2

def color(im):
    diff = 2 * im[:, :, 0] - im[:, :, 1] - im[:, :, 2]
    invdiff = diff.max() / diff
    uhoh = invdiff * 0 != 0
    invdiff[uhoh] = 0
    invdiff = gaussian_filter(invdiff, 20)
    return invdiff

def lumin(im):
    diff = im[:, :, 0] + im[:, :, 1] + im[:, :, 2]
    return diff

def colorlumin(im):
    #diff = rgb2hsv(im)
    #diff = diff[:, :, 0]#
    im = np.array(im).astype('float')
    diff = 2 * im[:, :, 0] - im[:, :, 1] - im[:, :, 2]
    print(np.median(diff))
    imcolor = diff + np.median(diff)
    colorthresh = np.percentile(imcolor, 97)
    print("Found color threshold of " + str(colorthresh))
    #invdiff = diff.max() / diff
    #uhoh = invdiff * 0 != 0
    #invdiff[uhoh] = 0
    #invdiff = gaussian_filter(diff, 2)
    #import matplotlib.pyplot as plt
    #plt.hist(imcolor.flatten(), bins=100)
    #plt.show()
    #import pdb; pdb.set_trace()

    diff = rgb2gray(im)
    imlumin = diff.copy()
    imlumin /= imlumin.max()
    #plt.imshow(imlumin)
    #plt.colorbar()
    #plt.show()

    # mask regions with a strong wave signature
    waveindex = imlumin > 0.9
    imcolor[waveindex] = imcolor.min()
    #plt.imshow(imcolor)
    #plt.colorbar()
    #plt.show()

    # first guess at whale region
    #import matplotlib.pyplot as plt
    #plt.imshow(imcolor)
    #plt.colorbar()
    #plt.show()
    hicol = imcolor >= colorthresh
    imcolor[hicol] = np.abs(colorthresh)
    locol = imcolor < colorthresh
    imcolor[locol] = colorthresh - 10
    #plt.imshow(imcolor)
    #plt.colorbar()
    #plt.show()
    #print(smallim.mean())

    return (imcolor, imlumin, colorthresh)

