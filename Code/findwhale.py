"""

2015 September 30

Shane Bussmann

Find the whale.  Model the whale as a rectangle with aspect ratio = 3.0.

"""

import numpy as np
from skimage.io import imread
import emcee
import time
import pandas as pd
import whaleutil
from glob import glob
import os


def lnprior(pzero, xylim):
    # penalty for size being too small or too large
    #size = np.sqrt(pzero[5] ** 2 + pzero[1] ** 2)
    #expectedsize = 600/4
    #lp = -(size - expectedsize) ** 2 / 2. / 20. ** 2
    lp = 0
    if pzero[1] > 1200/4 or pzero[1] < 300/4:
        #print('pzero1')
        lp = -np.inf
    if pzero[2] < xylim[0] or pzero[2] > xylim[1]:
        #print('pzero2')
        lp = -np.inf
        #print('pzero3')
    if pzero[3] < xylim[2] or pzero[3] > xylim[3]:
        lp = -np.inf
    if pzero[4] < 0.1 or pzero[4] > 1:
        #print('pzero4')
        lp = -np.inf
    if pzero[5] < 0 or pzero[5] > 180:
        #print('pzero5')
        lp = -np.inf

    return lp

def lnlike(pzero, imcolor, imluminmask, x, y):
    flux = pzero[0]
    rotation_angle = pzero[5]#np.arctan2(pzero[5], pzero[1]) * 180 / np.pi
    intermediate_axis = pzero[1]#) / np.cos(rotation_angle / 180 * np.pi)
    xcenter = pzero[2]
    ycenter = pzero[3]
    aspect_ratio = pzero[4]
    if xcenter < 0 or ycenter < 0 or xcenter > nx or ycenter > ny:
        print("Bad location: {} {}".format(xcenter, ycenter))
        return -np.inf
    #height = width * aspect_ratio
    #height = pzero[3]
    #x1 = xcenter - width
    #x2 = xcenter + width
    #y1 = ycenter - height
    #y2 = ycenter + height
    #if x1 > 0 and x2 < im[0, :].size and y1 > 0 and y2 < im[:, 0].size:
    par = [flux, intermediate_axis, xcenter, ycenter, aspect_ratio, 
            rotation_angle]
    whale_model = whaleutil.whale_2d(x, y, par)

    #whale_model[imluminmask] = 0
    #modelloc = whale_model > 0
    #cost = -imcolor[modelloc].sum()
    #luminindex = whale_model > 0
    #penalty_lumin = imlumin[luminindex].sum()
    resid_color = np.abs(imcolor - whale_model)
    cost = resid_color[imluminmask].sum()# + 0.1 * penalty_lumin
    #if cost < 4.5e3:
    #    #print(par, cost)
    #    import matplotlib.pyplot as plt
    #    #plt.imshow(imlumin, origin='lower')
    #    #plt.colorbar()
    #    #plt.contour(whale_model)
    #    #plt.show()
    #    print(par)
    #    plt.imshow(imcolor - whale_model, origin='lower')
    #    plt.colorbar()
    #    try:
    #        plt.contour(whale_model)
    #    except:
    #        import pdb; pdb.set_trace()
    #    plt.title(str(cost))
    #    plt.show()
    #    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    return -cost


def lnprob(pzero, imcolor, imluminmask, x, y, xylim):
    likeln = lnlike(pzero, imcolor, imluminmask, x, y)
    priorln = lnprior(pzero, xylim)
    #if likeln > -3e3:
    #print(likeln, priorln, pzero)
    return likeln + priorln


Nthreads = 2

pool = ''

nwalkers = 128
nparams = 6

whaleids = glob('w_*.jpg')
whaleids = whaleids[-2:]
print(whaleids)
for whaleid in whaleids:
    whalenum = os.path.splitext(whaleid)[0]
    im3 = imread(whaleid)
    im3 = np.array(im3).astype('float')
    diffim = 2 * im3[:, :, 0] - im3[:, :, 1] - im3[:, :, 2]
    #diffim = diffim.max() / diffim
    #toohigh = diffim > 5
    #diffim[toohigh] = 5.
    #toolow = diffim < 1
    #diffim[toolow] = 0.
    #print(diffim.mean())

    from scipy.misc import imresize
    rebin = 4.0
    ny, nx = diffim.shape
    #print(nx, ny)
    smallim = np.zeros((ny/rebin, nx/rebin, 3))
    for i in range(3):
        smallim[:, :, i] = imresize(im3[:, :, i], 1/rebin)

    # get the color and luminesence of the binned RGB image
    #colorthresh = -60.0
    imcolor, imlumin, colorthresh = whaleutil.colorlumin(smallim)

    imluminmask = imlumin < 0.9
    # mask regions with a strong wave signature
    #waveindex = imlumin > 300
    #imcolor[waveindex] = 0

    # first guess at whale region
    #hicol = imcolor >= 5
    #imcolor[hicol] = 5.
    #toolow = imcolor < 1
    #imcolor[toolow] = 0.
    #print(smallim.mean())

    ny, nx = imcolor.shape
    #print(nx, ny)
    xvec = np.arange(nx)
    yvec = np.arange(ny)
    x, y = np.meshgrid(xvec, yvec)
    #plt.imshow(x)
    #plt.show()
    #plt.imshow(y)
    #plt.show()

    hicol = imcolor > colorthresh - 10
    xguess = x[hicol].mean()
    yguess = y[hicol].mean()
    dguess = 100
    xylim = (xguess-dguess, xguess + dguess, yguess - dguess, yguess + dguess)
    rguess = 100.#np.sqrt(imcolor[hicol].size / np.pi)
    print(xguess*4, yguess*4, rguess*4)

    import matplotlib.pyplot as plt
    plt.imshow(imcolor)
    plt.colorbar()
    plt.plot([xguess], [yguess], 'o')
    plt.show()
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, \
        args=[imcolor, imluminmask, x, y, xylim], threads=Nthreads)


    # initialize with rectangles located within the inner quadrant
    ylength, xlength = diffim.shape
    #print(ylength, xlength)
    fluxinit = np.random.uniform(np.abs(colorthresh), np.abs(colorthresh), nwalkers)
    sizeinit = np.random.uniform(0.8*rguess, 1.5*rguess, nwalkers)
    #sizeinit[::2] *= -1
    #np.random.shuffle(sizeinit)
    #heightinit = np.random.uniform(100, 700, nwalkers)
    xinit = np.random.uniform(xguess-dguess, xguess+dguess, nwalkers)
    yinit = np.random.uniform(yguess-dguess, yguess+dguess, nwalkers)
    arinit = np.random.uniform(0.2, 0.5, nwalkers)
    phiinit = np.random.uniform(0, 180, nwalkers)
    #phiinit[::2] *= -1
    #np.random.shuffle(phiinit)
    #pzero = np.array([xinit, yinit, widthinit, heightinit, fluxinit]).transpose()
    pzero = np.array([fluxinit, sizeinit, xinit, yinit, arinit, \
            phiinit]).transpose()

    currenttime = time.time()

    cols = ['lnprob', 'fluxinit', 'size', 'xcenter', 'ycenter', \
            'aspect_ratio', 'rotation_angle']
    dfdict = {'lnprob': [], 'xcenter': [], 'ycenter': [], 'size': [], 
            'aspect_ratio': [], 'fluxinit': [], 'rotation_angle': []}
    dfposterior = pd.DataFrame(dfdict)

    # pos is the position of the sampler
    # prob the ln probability
    # state the random number generator state
    # amp the metadata 'blobs' associated with the current position
    for pos, prob, state in sampler.sample(pzero, iterations=100):

        print("Mean acceptance fraction: {:5.3f}\n".
                format(np.mean(sampler.acceptance_fraction)), 
                "Mean lnprob and Max lnprob values: {:8.2f} {:8.2f}\n".
                format(np.mean(prob), np.max(prob)),
                #"\nModel parameters: {:f} {:f} {:f} {:f}".
                #format(np.mean(pos, axis=0)),
                "\nTime to run previous set of walkers (seconds): {:6.3f}".
                format(time.time() - currenttime))
        currenttime = time.time()
        #ff.write(str(prob))
        superpos = np.zeros((1, 1 + nparams,))

        for wi in range(nwalkers):
            superpos[0, 0] = prob[wi]
            superpos[0, 1:nparams + 1] = pos[wi]
            dfsuperpos = pd.DataFrame(superpos)
            dfsuperpos.columns = cols
            dfposterior = dfposterior.append(dfsuperpos)
        dfposterior.to_csv('posteriorpdf_' + whalenum + '.csv')

