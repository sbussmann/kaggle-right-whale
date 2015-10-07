"""

Measure the percentage of image files where the whale location and heading has
been computed.

"""

from glob import glob
import numpy as np


# directory containing whale images
datadir = '../../BigData/kaggle-right-whale/right_whale_hunt/imgs/'

# get the number of images in the training set
trainlocs = glob(datadir + 'whale_*/w_*jpg')
ntrain = len(trainlocs)

# get the number of processed images in the training set
trainlocs = glob(datadir + 'whale_*/*colorclass*png')
ntrain_processed = len(trainlocs)

# get the number of images in the test set
trainlocs = glob(datadir + 'w_*jpg')
ntest = len(trainlocs)

# get the number of processed images in the test set
trainlocs = glob(datadir + '*colorclass*png')
ntest_processed = len(trainlocs)

print('Training set statistics')
print('{} {} {:.3f}'.format(ntrain, ntrain_processed,
    np.float(ntrain_processed)/ntrain))

print('Test set statistics')
print('{} {} {:.3f}'.format(ntest, ntest_processed,
    np.float(ntest_processed)/ntest))
