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

# get the fraction of processed images in the training set
ftrain_processed = np.float(ntrain_processed) / ntrain

# get the number of headshot images in the training set
trainlocs = glob(datadir + 'whale_*/*headw*png')
ntrain_head = len(trainlocs)

# get the fraction of headshot images in the training set
ftrain_head = np.float(ntrain_head) / ntrain

# get the number of images in the test set
trainlocs = glob(datadir + 'w_*jpg')
ntest = len(trainlocs)

# get the number of processed images in the test set
trainlocs = glob(datadir + '*colorclass*png')
ntest_processed = len(trainlocs)

ftest_processed = np.float(ntest_processed) / ntest

# get the number of headshofts in the test set
trainlocs = glob(datadir + '*headw*png')
ntest_head = len(trainlocs)

# get the fraction of headshofts in the test set
ftest_head = np.float(len(trainlocs)) / ntest

print('Training set statistics: Ntotal Nheading fheading Nhead fhead')
print('{} {} {:.3f} {:.3f} {:.3f}'.format(ntrain, ntrain_processed,
    ftrain_processed, ntrain_head, ftrain_head))

print('Test set heading statistics')
print('{} {} {:.3f} {} {:.3f}'.format(ntest, ntest_processed,
    ftest_processed, ntest_head, ftest_head))
