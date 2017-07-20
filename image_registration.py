# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:50:51 2013

@author: Josh Bradley

@purpose: To perform sub-pixel image registration using the image
registration algorithm from David Madsen's thesis.
"""
from sarimage_matt_edit import SARImage
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
#from mayavi import mlab
from sirtools import *
from scipy.misc import imsave

"""
Grab some repeat pass SAR data (UHF or L) from the FOPEN collects up
Payson Canyon
"""
# define the base path
#basepath = '/home/josh/Data/20130617'
basepath = '/home/josh/Data/20130530'
basepath = '/data4/SD/2013/20130722_fopen'
#basepath = '/data4/SD/2013/20130530_f2'
#basepath = '/data4/SD/2013/20130626_fopen_f2'

# define the 1st pass and 2nd pass files
#masterfile = 'SAR_05302013_143633_downloadLHH_34500'
slavefile = 'SAR_07222013_114256LHH_121000'
#masterfile = 'SAR_05302013_142807_downloadLHH_121000'
#masterfile = 'SAR_06172013_135748_downloadLVV_121000'
#masterfile = 'SAR_06262013_135300LHH_121000'
#slavefile = 'SAR_05302013_144057_downloadLHH_34500'
masterfile = 'SAR_07222013_113902LHH_121000'
#slavefile = 'SAR_05302013_145614_downloadLHH_34500'
#slavefile = 'SAR_05302013_150035_downloadLHH_34500'
# slavefile = 'SAR_05302013_145216_downloadLHH_121000'
#slavefile = 'SAR_06172013_142919_downloadLVV_121000'
#slavefile = 'SAR_06262013_145810LHH_121000'

# load in the SAR images
sarmaster = SARImage('%s/%s'%(basepath,masterfile))
#sarmaster.data = sarmaster.data[1500:sarmaster.data.shape[0]-1500,800:sarmaster.data.shape[1]-1500]

sarslave = SARImage('%s/%s'%(basepath,slavefile))
#sarslave.data = sarslave.data
print "Starting center spectrum for the master.\n"
jazz = sarmaster.centerSpectrum()
print "Starting center spectrum for the slave.\n"
blues = sarslave.centerSpectrum(jazz[0],jazz[1])
print "Done doing center spectrum nonsense.\n"

#sarslave = SARImage('/home/josh/Data/20130625/SAR_06252013_170114LHH_121000')

"""
Let's see if the function I implemented for interpolating the data to an
arbitrary point is working properly
"""
#Nrow,Ncol = sarslave.data.shape
#T = 0.25 # m/sample
#
#rowNfft = int(2**ceil(log2(Ncol)))
#colNfft = int(2**ceil(log2(Nrow)))
#
#f = 1/T
#
## let's take the fft in one dimension
#sdataFFTrow = fft2(sarslave.data,[rowNfft],[1])
#mdataFFTrow = fft2(sarmaster.data,[rowNfft],[1])
#
#sdata = ifft(fftshift(sdataFFTrow,1))
#sdata = sdata[:,:Ncol]
#mdata = ifft(fftshift(mdataFFTrow,1))
#mdata = mdata[:,:Ncol]
#
##figure(5);imshow(abs(sdata),interpolation='nearest',cmap='gray',clim=[abs(sdata).min(), abs(sdata).max()/2])
#
##sarmaster.data = mdata
##sarslave.data = sdata
#
#del mdata, sdata, sdataFFTrow, mdataFFTrow

#colInd, rowInterData, rowInterData2 = interpolateData(rowData, 8, 50, 1)
#rowInd, colInterData, colInterData2 = interpolateData(colData, 8, 50, 1)

if True:
    """
    Need to determine a global shift for the slave image to allign with the
    master image by computing the global correlation
    """
    # compute the correlation of the entire image
    print "Let's compute the total image correlation.\n"
    Rval, ysam, xsam = computeCorrelation(sarmaster.data, sarslave.data, (0,0), sarslave.data.shape, (0,0), sarmaster.data.shape)
    
    print "Now, determine the shifts.\n"
    # determine the shift from the position of the maximum correlation
    Rmax, yShift, xShift = determineShifts(Rval, ysam, xsam)
    print 'xShift: %d, yShift: %d\n'%(xShift, yShift)
    #xShift = -1095
    #yShift = -1473
    (yPos, xPos) = nonzero(abs(Rval) == abs(Rval).max())
    
    # lets now try to visualize the results
    if True:
        figure(1);imshow(abs(Rval),interpolation='nearest',clim=[abs(Rval).min(), abs(Rval).max()],
               extent=(xsam[0],xsam[-1],ysam[-1],ysam[0]))
        title('Correlation Matrix')
        figure(10);subplot(211);plot(xsam,abs(Rval[yPos,:].squeeze()))
        subplot(212);plot(ysam,abs(Rval[:,xPos].squeeze()))
    
    """
    We need to resample the slave image into a new matrix the size of the master image
    """
    mNy, mNx = sarmaster.data.shape
    newSarSlave = sarslave.copy()
    newSarSlave.data = shiftSlaveData(sarslave, zeros((mNy,mNx),'complex64'), (0,0), (mNy,mNx), yShift, xShift)
    
    # compute the coherence between the two images
    gamma = sarmaster.coherence(newSarSlave,diam=9)
    gamma.showMag(fignum=2,scale=1)
    
    """
    Now, we need to split the image up into blocks (about 512x512) and do the same
    thing for each of the blocks before proceeding with subpixel alignment and
    warping based on sections
    """
    # bsSlaveSar = blockShift(sarmaster, newSarSlave, 2.0**9)
    bsSlaveSar = newSarSlave.copy()
    bsSlaveSar.data = bsSlaveSar.data + 0
    
    # compute the coherence between the two images
    gamma2 = sarmaster.coherence(bsSlaveSar,diam=9,name='Coher_block')
    gamma2.showMag(fignum=7,scale=1)
        
    """
    Now I need to split each chunk further into smaller sections to do sub-pixel
    alignment and interpolation
    """
    spSlaveSar = subpixelAlignment(sarmaster, bsSlaveSar, idealblocksize=2.0**9, idealSecNum=8, threshold=0.0028)
    
    # compute the coherence between the two images
    gamma3 = sarmaster.coherence(spSlaveSar,diam=9,name='Coher_subPix')
    gamma3.showMag(scale=1)
        
        
