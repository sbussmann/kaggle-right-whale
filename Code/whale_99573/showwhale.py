"""

Inspect results of MCMC simulation.

"""

import pandas as pd
from skimage.io import imread, imshow, imsave
import numpy as np
from whaleutil import whale_2d, xy_rotate, colorlumin
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

whaleid = 'w_2234.jpg'
df = pd.read_csv('posteriorpdf_' + whaleid + '.csv')

bestfitloc = df['lnprob'] == df['lnprob'].max()

bestfit = df[bestfitloc]

im = imread(whaleid)


parcols = ['fluxinit', 'size', 'xcenter', 'ycenter', \
        'aspect_ratio', 'rotation_angle']
par = bestfit[parcols]

binned = 4

par['size'] *= binned
par['xcenter'] *= binned
par['ycenter'] *= binned
parvalues = par.values[0]

nx, ny = im[:, :, 0].shape
#print(nx, ny)
xvec = np.arange(nx)
yvec = np.arange(ny)
x, y = np.meshgrid(yvec, xvec)

print(parvalues)

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
headradius = 300
y1 = yhead - headradius
y2 = yhead + headradius
x1 = xhead - headradius
x2 = xhead + headradius
if x1 < 0: x1 = 0 
if x2 > nx: x2 = nx 
if y1 < 0: y1 = 0 
if y2 > ny: y2 = ny 
print(x1, x2, y1, y2)
if y1 < ny and x1 < nx and x2 > 0 and y2 > 0:
    imhead = im[y1:y2, x1:x2, :]
    imsave('whalehead_' + whaleid + '.png', imhead)

xhead, yhead = xy_rotate(-a0, -0, 0, 0, phi0)
xhead *= -1
xhead += x0
yhead += y0
xhead2 = xhead
yhead2 = yhead
headcent = (yhead, xhead)
print(headcent)
y1 = yhead - headradius
y2 = yhead + headradius
x1 = xhead - headradius
x2 = xhead + headradius
if x1 < 0: x1 = 0 
if x2 > nx: x2 = nx 
if y1 < 0: y1 = 0 
if y2 > ny: y2 = ny 
print(x1, x2, y1, y2)
if y1 < nx and x1 < ny and x2 > 0 and y2 > 0:
    imhead = im[y1:y2, x1:x2, :]
    imsave('whaletail_' + whaleid + '.png', imhead)


def differ(im, ch1, ch2):
    diffim = im[:, :, 0] + im[:, :, 1] + im[:, :, 2]
    normdiff =  diffim.max() / diffim
    smoothdiff = gaussian_filter(normdiff, 20)
    return smoothdiff


plt. clf()
imshow(im, origin='lower')
plt.contour(whale_model)

plt.plot([xhead1], [yhead1], 'o')
plt.plot([xhead2], [yhead2], 'o')

plt.savefig('whalemodel_' + whaleid + '.png')

plt.clf()
colorthresh = 5
imcolor, imlumin = colorlumin(im, colorthresh)
imluminmask = imlumin < 0.8
imdiff = imcolor - whale_model
print(imdiff[imluminmask].sum())
plt.imshow(imcolor, origin='lower')
plt.colorbar()
plt.contour(whale_model)

plt.plot([xhead1], [yhead1], 'o')
plt.plot([xhead2], [yhead2], 'o')

plt.savefig('whalecolor' + whaleid + '.png')

plt.clf()
plt.imshow(imlumin, origin='lower')
plt.colorbar()
plt.contour(whale_model)

plt.plot([xhead1], [yhead1], 'o')
plt.plot([xhead2], [yhead2], 'o')

plt.savefig('whalelumin' + whaleid + '.png')

xheads = np.linspace(xhead1, xhead2, 5)

yheads = np.linspace(yhead1, yhead2, 5)

for i in range(len(xheads)):

    y1 = yheads[i] - headradius
    y2 = yheads[i] + headradius
    x1 = xheads[i] - headradius
    x2 = xheads[i] + headradius
    if x1 < 0: x1 = 0 
    if x2 > nx: x2 = nx 
    if y1 < 0: y1 = 0 
    if y2 > ny: y2 = ny 
    print(x1, x2, y1, y2)
    if y1 < nx and x1 < ny and x2 > 0 and y2 > 0:
        imhead = im[y1:y2, x1:x2, :]
        imsave('whaletail_' + whaleid + str(i) + '.png', imhead)

import pdb; pdb.set_trace()

