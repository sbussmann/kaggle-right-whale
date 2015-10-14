"""

Resize 256x256 whale head images into 32x32 images.

"""

from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import os
import numpy as np


# factor by which to shrink the input images
rebin = 8.

# directory containing whale images
datadir = '../../BigData/kaggle-right-whale/right_whale_hunt/imgs/'

# get the headshot images in the training set
trainlocs = glob(datadir + '*headw*jpg')
ntrain_head = len(trainlocs)

for train in trainlocs:
    traindir = os.path.splitext(train)[0]
    headloc = traindir.index('headw_') + 4
    imageid = traindir[headloc:]
    trainim = imread(train).astype('float')
    ny, nx = trainim[:, :, 0].shape
    trimim = np.zeros((64, 64, 3))
    for i in range(3):
        trimim[:, :, i] = resize(trainim[:, :, i], (64, 64))
    trimim = trimim.astype('uint8')
    imsave('../Data/headshots/test/whalehead_trim_' + imageid + '.jpg', trimim)
    
