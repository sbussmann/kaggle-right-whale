"""

Inspect results of MCMC simulation.

"""

import pandas as pd
from skimage.io import imread, imshow, imsave
import numpy as np
from whaleutil import whale_2d, xy_rotate, colorlumin
import whaleutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from glob import glob
import emcee
import time


def lnprior(pzero, xylim):
    # penalty for size being too small or too large
    #size = np.sqrt(pzero[5] ** 2 + pzero[1] ** 2)
    #expectedsize = 600/4
    #lp = -(size - expectedsize) ** 2 / 2. / 20. ** 2
    lp = 0
    if pzero[0] < xylim[0] or pzero[0] > xylim[1]:
        lp = -np.inf
    if pzero[1] < xylim[2] or pzero[1] > xylim[3]:
        lp = -np.inf
    if pzero[2] > 300 or pzero[2] < 200:
        lp = -np.inf

    return lp

def lnlike(pzero, im, imluminmask):
    ny, nx = im.shape
    xcenter = nx / 2 + int(pzero[0])
    ycenter = ny / 2 + int(pzero[1])
    size = int(pzero[2])
    if xcenter < 0 or ycenter < 0 or xcenter > nx or ycenter > ny:
        #print("Bad location: {} {}".format(xcenter, ycenter))
        return -np.inf
    x1 = xcenter - size
    x2 = xcenter + size
    y1 = ycenter - size
    y2 = ycenter + size
    #if x1 < xylim[0]-size or x2 > xylim[1]+size:
    #    return -np.inf
    #if y1 < xylim[2]-size or y2 > xylim[3]+size:
    #    return -np.inf

    #if x1 >= x2 or y1 >= y2:
    #    print(x1, x2, y1, y2)
    #    return -np.inf
    #import matplotlib.pyplot as plt
    imhead = im[y1:y2, x1:x2]
    #plt.imshow(imhead)
    #plt.colorbar()
    #plt.show()
    imhead = whaleutil.extract_asymm(imhead)
    #plt.imshow(imhead)
    #plt.colorbar()
    #plt.show()
    maskhead = imluminmask[y1:y2, x1:x2]
    revmaskhead = maskhead[:,::-1]
    fullmaskhead = 1 - maskhead + 1 - revmaskhead
    #plt.imshow(imluminmask)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(fullmaskhead)
    #plt.colorbar()
    #plt.show()
    nyhead, nxhead = imhead.shape
    ylow = int(nyhead * 1.5 / 4)
    yhigh = int(nyhead * 2.5 / 4)
    #npixels = (yhigh - ylow) * nxhead
    maskbar = fullmaskhead[ylow:yhigh, :]
    imbar = imhead[ylow:yhigh,:]
    notmasked = maskbar == 0
    npixels = imbar[notmasked].size
    asymmetry = np.abs(imbar[notmasked]).sum() / np.float(npixels)
    #print(asymmetry)
    return asymmetry


def lnprob(pzero, im, imluminmask, xylim):
    likeln = lnlike(pzero, im, imluminmask)
    priorln = lnprior(pzero, xylim)
    #if likeln > -3e3:
    #print(likeln, priorln, pzero)
    return likeln + priorln


def gini(list_of_values):
  sorted_list = sorted(list_of_values)
  n = np.float(len(list_of_values))
  numer = 0.
  denom = 0.
  for i, value in enumerate(sorted_list):
      numer += i * value
      denom += value
  num = 1 + 1 / n + 2 / n * numer / denom
  return num

cwd = os.getcwd()
datadir = '../../BigData/kaggle-right-whale/right_whale_hunt/imgs/'
whaledirs = glob(datadir + 'whale_*')
whaledirs = whaledirs[0:1]

nwalkers = 64
nparams = 3
Nthreads = 1

for whaledir in whaledirs:

    print(whaledir)
    os.chdir(whaledir)
    imageids = glob('w_*.jpg')
    for imageid in imageids[0:1]:
        #if imageid == 'w_3781.jpg':
        #    continue
        print(imageid)
        imagenum = os.path.splitext(imageid)[0]
        df = pd.read_csv('posteriorpdf_rebin16_64walkers_' + imagenum + '.csv')

        bestfitloc = df['lnprob'] == df['lnprob'].max()

        bestfit = df[bestfitloc]

        im = imread(imageid)

        im = np.array(im).astype('float')


        parcols = ['fluxinit', 'size', 'xcenter', 'ycenter', \
                'aspect_ratio', 'rotation_angle']
        par = bestfit[parcols]

        binned = 16
        rebin = binned

        par['size'] *= binned
        par['xcenter'] *= binned
        par['ycenter'] *= binned
        parvalues = par.values[0]

        xcenter = par['xcenter'].values[0]
        ycenter = par['ycenter'].values[0]
        xycenter = (xcenter, ycenter)
        #for i in range(3):
        #    im[:, :, i] = rotate(im[:, :, i], par['rotation_angle'].values[0],
        #            center=xycenter)

        #plt.imshow(im[:, :, 0])
        #plt.colorbar()
        #plt.show()
        #import pdb; pdb.set_trace()

        ny, nx = im[:, :, 0].shape
        #print(nx, ny)
        xvec = np.arange(nx)
        yvec = np.arange(ny)
        x, y = np.meshgrid(xvec, yvec)
        #plt.imshow(x)
        #plt.show()

        #print(parvalues)

        whale_model = whale_2d(x, y, parvalues)

        # identify the head
        #rotation_angle = np.arctan2(par['rotation_angle'].values[0], \
        #        par['size'].values[0]) * 180 / np.pi
        phi0 = par['rotation_angle'].values[0]
        a0 = np.abs(par['size'].values[0]) / np.sqrt(par['aspect_ratio'].values[0])
        b0 = a0 * par['aspect_ratio'].values[0]
        x0 = par['xcenter'].values[0]
        y0 = par['ycenter'].values[0]
        xhead, yhead = xy_rotate(a0, 0, 0, 0, phi0)
        xhead *= -1
        xhead += x0
        yhead += y0
        xhead1 = xhead
        yhead1 = yhead
        headcent = (yhead, xhead)
        print(headcent)
        headradius = 512
        y1 = yhead - headradius
        y2 = yhead + headradius
        x1 = xhead - headradius
        x2 = xhead + headradius
        if x1 < 0: x1 = 0 
        if x2 > nx: x2 = nx 
        if y1 < 0: y1 = 0 
        if y2 > ny: y2 = ny 
        #print(x1, x2, y1, y2)
        imcolor, imlumin, colorthresh, luminthresh = colorlumin(im)
        if colorthresh == 0:
            print("Color threshold of 0 was found, uh oh!")
            continue
        dguess = 50
        xguess = xhead
        yguess = yhead
        xylim = (-dguess, dguess, -dguess, dguess)
        rguess = headradius / 2
        #if y1 < ny and x1 < nx and x2 > 0 and y2 > 0:
        if y1 > 0 and x1 > 0 and x2 < nx and y2 < ny:
            xinit = np.random.uniform(xguess-dguess, xguess+dguess, nwalkers)
            yinit = np.random.uniform(yguess-dguess, yguess+dguess, nwalkers)
            sizeinit = np.random.uniform(200, 300, nwalkers)
            pzero = np.array([xinit, yinit, sizeinit]).transpose()
            currenttime = time.time()
            
            imdiff = im[:, :, 0] - im[:, :, 1]
            imdiff = whaleutil.extract_head(imdiff, x1, x2, y1, y2, phi0)
            imluminmask = imlumin < 0.9
            imluminmask = whaleutil.extract_head(imluminmask, 
                    x1, x2, y1, y2, phi0)
            #imsave('whalehead_' + imageid + '.png', imhead)

            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, \
                args=[imdiff, imluminmask, xylim], threads=Nthreads)
            counter = 0
            dfposterior = pd.DataFrame({})
            for pos, prob, state in sampler.sample(pzero, iterations=2):

                if counter % 1 == 0:
                    print("Mean acceptance fraction: {:5.3f}".
                            format(np.mean(sampler.acceptance_fraction))) 
                    print("Mean lnprob and Max lnprob values: {:8.2f} {:8.2f}".
                            format(np.mean(prob), np.max(prob)))
                            #"\nModel parameters: {:f} {:f} {:f} {:f}".
                            #format(np.mean(pos, axis=0)),
                    print("Time to run previous set of walkers (seconds): {:6.3f}".
                            format(time.time() - currenttime))
                currenttime = time.time()
                #ff.write(str(prob))
                superpos = np.zeros((1, 1 + nparams,))
                dfpdf = pd.DataFrame({})
                dfpdf['lnprob'] = prob
                dfpdf['xcenter'] = pos[:, 0]
                dfpdf['ycenter'] = pos[:, 1]
                dfpdf['size'] = pos[:, 2]

                dfposterior = dfposterior.append(dfpdf)
                counter += 1
                dfposterior.to_csv('headpdf_rebin16_64walkers_' + imagenum + '.csv')
        #import pdb; pdb.set_trace()

        xhead, yhead = xy_rotate(-a0, -0, 0, 0, phi0)
        xhead *= -1
        xhead += x0
        yhead += y0
        xhead2 = xhead
        yhead2 = yhead
        headcent = (yhead, xhead)
        #print(headcent)
        y1 = yhead - headradius
        y2 = yhead + headradius
        x1 = xhead - headradius
        x2 = xhead + headradius
        if x1 < 0: x1 = 0 
        if x2 > nx: x2 = nx 
        if y1 < 0: y1 = 0 
        if y2 > ny: y2 = ny 
        xguess = xhead
        yguess = yhead
        xylim = (-dguess, dguess, -dguess, dguess)
        print(x1, x2, y1, y2)
        if y1 < nx and x1 < ny and x2 > 0 and y2 > 0:
            #imhead = im[y1:y2, x1:x2, :]
            #imsave('whaletail_' + imageid + '.png', imhead)
            xinit = np.random.uniform(-dguess, dguess, nwalkers)
            yinit = np.random.uniform(-dguess, dguess, nwalkers)
            sizeinit = np.random.uniform(200, 300, nwalkers)
            pzero = np.array([xinit, yinit, sizeinit]).transpose()
            currenttime = time.time()
            
            imdiff = im[:, :, 0] - im[:, :, 1]
            imdiff = whaleutil.extract_head(imdiff, x1, x2, y1, y2, phi0)
            imluminmask = imlumin < 0.9
            imluminmask = whaleutil.extract_head(imluminmask, 
                    x1, x2, y1, y2, phi0)
            #imsave('whalehead_' + imageid + '.png', imhead)

            sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, \
                args=[imdiff, imluminmask, xylim], threads=Nthreads)
            counter = 0
            dfposterior = pd.DataFrame({})
            for pos, prob, state in sampler.sample(pzero, iterations=20):

                if counter % 1 == 0:
                    print("Mean acceptance fraction: {:5.3f}".
                            format(np.mean(sampler.acceptance_fraction))) 
                    print("Mean lnprob and Max lnprob values: {:8.2f} {:8.2f}".
                            format(np.mean(prob), np.max(prob)))
                            #"\nModel parameters: {:f} {:f} {:f} {:f}".
                            #format(np.mean(pos, axis=0)),
                    print("Time to run previous set of walkers (seconds): {:6.3f}".
                            format(time.time() - currenttime))
                currenttime = time.time()
                #ff.write(str(prob))
                superpos = np.zeros((1, 1 + nparams,))
                dfpdf = pd.DataFrame({})
                dfpdf['lnprob'] = prob
                dfpdf['xcenter'] = pos[:, 0]
                dfpdf['ycenter'] = pos[:, 1]
                dfpdf['size'] = pos[:, 2]

                dfposterior = dfposterior.append(dfpdf)
                counter += 1
                dfposterior.to_csv('headpdf_rebin16_64walkers_' + imagenum + '.csv')


        def differ(im, ch1, ch2):
            diffim = im[:, :, 0] + im[:, :, 1] + im[:, :, 2]
            normdiff =  diffim.max() / diffim
            smoothdiff = gaussian_filter(normdiff, 20)
            return smoothdiff



        diffim = 2 * im[:, :, 0] - im[:, :, 1] - im[:, :, 2]
        #plt.clf()
        plt.clf()
        plt.imshow(imcolor)
        plt.contour(whale_model)
        plt.colorbar()

        plt.plot([xhead1], [yhead1], 'o')
        plt.plot([xhead2], [yhead2], 'o')

        plt.savefig('whalemodel_' + imagenum + '.png')
        #imluminmask = imlumin < 0.8
        #imdiff = imcolor - whale_model
        #print(imdiff[imluminmask].sum())
        #plt.imshow(imcolor, origin='lower')
        #plt.colorbar()
        #plt.contour(whale_model)

        #plt.plot([xhead1], [yhead1], 'o')
        #plt.plot([xhead2], [yhead2], 'o')

        #plt.savefig('whalecolor' + imageid + '.png')

        #plt.clf()
        #plt.imshow(imlumin, origin='lower')
        #plt.colorbar()
        #plt.contour(whale_model)

        #plt.plot([xhead1], [yhead1], 'o')
        #plt.plot([xhead2], [yhead2], 'o')

        #plt.savefig('whalelumin' + imageid + '.png')

        plt.clf()
        nheads = 16
        nrows = int(np.sqrt(nheads))
        f, axarr = plt.subplots(nrows, nrows)

        xheads = np.linspace(xhead1, xhead2, nheads)

        yheads = np.linspace(yhead1, yhead2, nheads)
        row = -1
        imdiff = im[:, :, 0] - im[:, :, 1]
        mincol01 = imdiff.max()
        summedmarks = []
        xmarks = []
        ymarks = []

        # parameters of the small mask used to identify the whale's head
        headsize = 64
        headq = 0.15
        flux = 5
        maxasymm = 0
        for i in range(len(xheads)):

            if i % nrows == 0:
                row += 1
                col = 0
            y1 = yheads[i] - headradius
            y2 = yheads[i] + headradius
            x1 = xheads[i] - headradius
            x2 = xheads[i] + headradius
            if x1 < 0: x1 = 0 
            if x2 > nx: x2 = nx 
            if y1 < 0: y1 = 0 
            if y2 > ny: y2 = ny 
            #print(x1, x2, y1, y2)
            if y1 < ny and x1 < nx and x2 > 0 and y2 > 0:
                #imhead = imcolor[y1:y2, x1:x2]
                imhead = whaleutil.extract_head(imdiff, x1, x2, y1, y2, phi0)
                maskhead = whaleutil.extract_head(imlumin, x1, x2, y1, y2, phi0)
                #imhead = rotate(imhead, phi0)
                imhead = whaleutil.extract_asymm(imhead)
                if imhead.min() < mincol01:
                    mincol01 = imhead.min()
                fig = axarr[row, col].imshow(imhead)
                #gininum = gini(imhead.flatten())
                #print(gininum)
                divider3 = make_axes_locatable(axarr[row, col])
                cax3 = divider3.append_axes("right", size="20%", pad=0.05)
                #axarr[row, col].text(30,70, str(gininum))
                cbar = plt.colorbar(fig, cax=cax3)
                nyhead, nxhead = imhead.shape
                xvec = np.arange(nxhead)
                yvec = np.arange(nyhead)
                #xmatrix, ymatrix = np.meshgrid(yvec, xvec)
                
                #axarr[row, col].text(30,70, str(maxmark))
                ylow = int(nyhead * 1.5 / 4)
                yhigh = int(nyhead * 2.5 / 4)
                npixels = (yhigh - ylow) * nxhead
                maskbar = maskhead[ylow:yhigh, :]
                imbar = imhead[ylow:yhigh,:]
                notmasked = maskbar < luminthresh
                asymmetry = np.abs(imbar[notmasked]).sum() / np.float(npixels)
                if asymmetry > maxasymm:
                    besti = i
                    bestx = xheads[besti]
                    besty = yheads[besti]
                    maxasymm = asymmetry
                axarr[row, col].text(30,70, str(asymmetry))
                #imsave('whalecut_' + imageid + str(i) + '.png', imhead)
            col += 1

        y1 = yheads[besti] - headradius
        y2 = yheads[besti] + headradius
        x1 = xheads[besti] - headradius
        x2 = xheads[besti] + headradius
        if x1 < 0: x1 = 0 
        if x2 > nx: x2 = nx 
        if y1 < 0: y1 = 0 
        if y2 > ny: y2 = ny 
        imhead = whaleutil.extract_head(imdiff, x1, x2, y1, y2, phi0)
        imhead -= np.median(imhead)
        #plt.clf()
        #plt.imshow(imhead)
        #plt.colorbar()
        #plt.show()
        maxmark = 0
        for ix in range(nxhead/2, nxhead/4*3):
            for iy in range(nyhead/4, nyhead/4*3):
                #par = [flux, headsize, ix, iy, headq, phi0]
                #whale_model = whaleutil.whale_2d(xmatrix, ymatrix, par)
                #headmark = whale_model == flux
                x1 = ix - 50
                x2 = ix + 50
                y1 = iy - 15
                y2 = iy + 15
                imheadsum = imhead[y1:y2, x1:x2].sum()
                if imheadsum > maxmark:
                    maxmark = imheadsum
                    xmark = ix
                    ymark = iy
                    #print(ix, iy, maxmark)
                #summedmarks.append(imheadsum)
                #xmarks.append(ix)
                #ymarks.append(iy)

        fig = plt.gcf()
        fig.set_size_inches(14.5, 10.5)
        plt.savefig('whalecutscolor01' + imagenum + '.png')

        print(besti, xmark, ymark)
        y1 = besty + ymark - nyhead / 2 - headradius
        y2 = besty + ymark - nyhead / 2 + headradius
        x1 = bestx + xmark - nxhead / 2 - headradius
        x2 = bestx + xmark - nxhead / 2 + headradius
        if x1 < 0: x1 = 0 
        if x2 > nx: x2 = nx 
        if y1 < 0: y1 = 0 
        if y2 > ny: y2 = ny 
        imhead = im[y1:y2, x1:x2, :]
        imhead = imhead.astype('uint8')
        plt.clf()
        imshow(imhead)
        imsave('whalehead' + imageid, imhead)
        plt.savefig('whalehead' + imagenum + '.png')

    os.chdir(cwd)
import pdb; pdb.set_trace()

