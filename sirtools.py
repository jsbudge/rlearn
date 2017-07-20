# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:20:20 2013

@author: Josh Bradley

@purpose: To make a module that provides valuable different functions for use
    in SAR image registration (SIR) algorithms
"""
from sarimage_matt_edit import SARImage
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
#from mayavi import mlab

def determineShifts(R, corrYInd, corrXInd):
    (yPos, xPos) = nonzero(abs(R) == abs(R).max())
    yShift = corrYInd[yPos[0]]
    xShift = corrXInd[xPos[0]]
    return abs(R[yPos[0],xPos[0]])/sqrt((R*R.conj()).real.sum()), yShift, xShift
    
def computeBlockNum(imageShape, blockSize):
    numY = int(ceil(imageShape[0] / blockSize))
    numX = int(ceil(imageShape[1] / blockSize))
    sizeY = floor(imageShape[0] / numY)
    sizeX = floor(imageShape[1] / numX)
    return numY, numX, sizeY, sizeX
    
def computeSizes(begIndSlave, endIndSlave, begIndMaster, endIndMaster):
    # compute the actual size of the block along axis
    siz = int(endIndSlave - begIndSlave)
    siz2 = int(endIndMaster - begIndMaster)
    # compute the correlation length
    corrSiz = siz + siz2 - 1
    # compute the length of the padded FFT
    padSiz = int(2**ceil(log2(corrSiz)))
    return siz, siz2, corrSiz, padSiz
    
def computeCorrelation(master, slave, sBeg, sEnd, mBeg=None, mEnd=None):
    if not mBeg and not mEnd:
        mBeg = sBeg
        mEnd = sEnd
    # determine actual block size, correlation length, and FFT size in X (NOTE: the last block will usually
    # be different due to the number of blocks not dividing evenly into the size of the image)
    sNx, mNx, corrNx, padNx = computeSizes(sBeg[1], sEnd[1], mBeg[1], mEnd[1])
    # determine actual block size, correlation length, and FFT size in X
    sNy, mNy, corrNy, padNy = computeSizes(sBeg[0], sEnd[0], mBeg[0], mEnd[0])
    print "sNx = %d, mNx = %d, corrNx = %d, padNx = %d\n" % (sNx, mNx, corrNx, padNx)
    print "sNy = %d, mNy = %d, corrNy = %d, padNy = %d\n" % (sNy, mNy, corrNy, padNy)
    
    # to get the correlation to come out right without any awkward shifting,
    # I will allocate 2 matrices that will be used to hold the correctly zero-padded block
    zeroMaster = zeros((corrNy,corrNx),'complex64')
    zeroSlave = zeros((corrNy,corrNx),'complex64')
    zeroMaster[sNy-1:corrNy, sNx-1:corrNx] = master[mBeg[0]:mEnd[0],mBeg[1]:mEnd[1]]
    zeroSlave[0:sNy,0:sNx] = slave[sBeg[0]:sEnd[0],sBeg[1]:sEnd[1]] 
    freqMaster = fft2(zeroMaster, (padNy,padNx))
    freqSlave = fft2(zeroSlave, (padNy,padNx))
    
    # compute the correlation in the frequency domain
    S = freqMaster * freqSlave.conj()
    
    # compute the inverse FFT to get the correlation in the spatial domain
    R = ifft2(S)
    
    # compute the proper indices for the correlation matrix
    yind = range(-(sNy-1),mNy)
    xind = range(-(sNx-1),mNx)
    
    # only return the valid samples of the correlation matrix
    return R[0:corrNy,0:corrNx], yind, xind

def computeCorrelation2(master, slave, sBeg, sEnd, mBeg=None, mEnd=None, interpFactor=1):
    """
    This should be the same as "computeCorrelation", except that it gives the
    option of interpolating the result
    """
    if not mBeg and not mEnd:
        mBeg = sBeg
        mEnd = sEnd
    # determine actual block size, correlation length, and FFT size in X 
    sNx, mNx, corrNx, padNx = computeSizes(sBeg[1], sEnd[1], mBeg[1], mEnd[1])
    # determine actual block size, correlation length, and FFT size in X
    sNy, mNy, corrNy, padNy = computeSizes(sBeg[0], sEnd[0], mBeg[0], mEnd[0])
    
    # to get the correlation to come out right without any awkward shifting,
    # I will allocate 2 matrices that will be used to hold the correctly zero-padded block
    zeroMaster = zeros((corrNy,corrNx),'complex64')
    zeroSlave = zeros((corrNy,corrNx),'complex64')
    zeroMaster[sNy-1:corrNy, sNx-1:corrNx] = master[mBeg[0]:mEnd[0],mBeg[1]:mEnd[1]]
    zeroSlave[0:sNy,0:sNx] = slave[sBeg[0]:sEnd[0],sBeg[1]:sEnd[1]] 
    freqMaster = fft2(zeroMaster, (padNy,padNx))
    freqSlave = fft2(zeroSlave, (padNy,padNx))
    
    # compute the correlation in the frequency domain
    S = freqMaster * freqSlave.conj()
    
    # we also want and interpolated version of the correlation matrix for
    # subpixel registration (so we will zero-pad in the frequency domain)
    # interpFactor = 1
    Spad = zeros((padNx*interpFactor, padNy*interpFactor),dtype='complex64')
    # we need to insert zeros at pi in the spectrum
    Spad[0:padNx/2+1,0:padNy/2+1] = S[0:padNx/2+1,0:padNy/2+1]
    Spad[padNx/2+1+(interpFactor-1)*padNx:,0:padNy/2+1] = S[padNx/2+1:,0:padNy/2+1]
    Spad[0:padNx/2+1,padNy/2+1+(interpFactor-1)*padNy:] = S[0:padNx/2+1,padNy/2+1:]
    Spad[padNx/2+1+(interpFactor-1)*padNx:,padNy/2+1+(interpFactor-1)*padNy:] = S[padNx/2+1:,padNy/2+1:]
    del S
    
    # compute the inverse FFT to get the correlation in the spatial domain
    R = ifft2(Spad)
    
    # compute the proper indices for the correlation matrix
    yind = arange(-(sNy-1),mNy, 1.0/interpFactor)
    xind = arange(-(sNx-1),mNx, 1.0/interpFactor)
    
    # only return the valid samples of the correlation matrix
    return R[0:corrNy*interpFactor,0:corrNx*interpFactor], yind, xind
    
def computeCorrelation3(master, slave, sBeg, sEnd, mBeg=None, mEnd=None):
    if not mBeg and not mEnd:
        mBeg = sBeg
        mEnd = sEnd
    # determine actual block size, correlation length, and FFT size in X (NOTE: the last block will usually
    # be different due to the number of blocks not dividing evenly into the size of the image)
    sNx, mNx, corrNx, padNx = computeSizes(sBeg[1], sEnd[1], mBeg[1], mEnd[1])
    # determine actual block size, correlation length, and FFT size in X
    sNy, mNy, corrNy, padNy = computeSizes(sBeg[0], sEnd[0], mBeg[0], mEnd[0])
    print "sNx = %d, mNx = %d, corrNx = %d, padNx = %d\n" % (sNx, mNx, corrNx, padNx)
    print "sNy = %d, mNy = %d, corrNy = %d, padNy = %d\n" % (sNy, mNy, corrNy, padNy)
    
    # to get the correlation to come out right without any awkward shifting,
    # I will allocate 2 matrices that will be used to hold the correctly zero-padded block
    zeroMaster = zeros((corrNy,corrNx),'complex64')
    zeroSlave = zeros((corrNy,corrNx),'complex64')
    zeroMaster[sNy-1:corrNy, sNx-1:corrNx] = master[mBeg[0]:mEnd[0],mBeg[1]:mEnd[1]]
    zeroSlave[0:sNy,0:sNx] = slave[sBeg[0]:sEnd[0],sBeg[1]:sEnd[1]] 
    freqMaster = fft2(zeroMaster, (padNy,padNx))
    freqSlave = fft2(zeroSlave, (padNy,padNx))
    
    # compute the correlation in the frequency domain
    S = freqMaster * freqSlave.conj()
    
    # compute the inverse FFT to get the correlation in the spatial domain
    R = ifft2(S)
    
    # compute the proper indices for the correlation matrix
    yind = range(-(sNy-1),mNy)
    xind = range(-(sNx-1),mNx)
    
    # only return the valid samples of the correlation matrix
    return R[0:corrNy,0:corrNx], yind, xind
        
def shiftSlaveData(slave, newSlaveData, sBeg, sEnd, deltaY, deltaX):
    # need the size of the slave image
    maxY, maxX = slave.data.shape
    
    # We need to lay out a matrix of index values into the slave image
    yInd = arange(int(sBeg[0]),int(sEnd[0]))
    xInd = arange(int(sBeg[1]),int(sEnd[1]))
    X,Y = meshgrid(xInd,yInd)
    
    # then adjust them by the x and y shift values
    Xs = X - deltaX
    Ys = Y - deltaY
    
    # then I need to take the indices that are valid in the image
    indmask = logical_and(logical_and(logical_and(Xs>=0,Xs<maxX),Ys>=0),Ys<maxY)
    newSlaveData[Y[indmask],X[indmask]] = slave.data[Ys[indmask],Xs[indmask]]
    
    return newSlaveData
    
def computeSurfaceCoefficients(y, x, deltaY, deltaX, corr):
    """
    Solve for the coefficients that define the surface for the shifts in x and y
    """
    # determine length of f and g
    n = len(deltaY)
    N = n + 3
    
    # the "f" function will be for the deltaY's, and "g" function will be for the deltaX's
    f = zeros((N,1))
    f[0:n,0] = deltaY
    g = zeros((N,1))
    g[0:n,0] = deltaX
    
    # construct parts of L
    Y = array([y]).repeat(n,0)
    X = array([x]).repeat(n,0)
    rsquare = (X-X.T)**2 + (Y-Y.T)**2
    L11 = rsquare * log(rsquare)
    L11[isnan(L11)] = 0.0
    L21 = array([ones(n), x, y])
    
    # piece together L
    L = zeros((N,N))
    L[0:n,0:n] = L11
    L[n:,0:n] = L21
    L[0:n,n:] = L21.T
    
    # invert L and solve for the coefficients
    Linv = inv(L)
    a_f = Linv.dot(f)
    a_g = Linv.dot(g)
    
    return a_f, a_g
    
def findControlPoints(mData, sData, idealSecNum, thresh):
    """
    INPUTS: complex 2D numpy array, complex 2D numpy array, int
    Note that the dimension of the data in both arrays should be the same
    """
    # size in x and y of the 2D array
    Ny, Nx = mData.shape
    sizeY = floor(Ny / float(idealSecNum))
    sizeX = floor(Nx / float(idealSecNum))
    
    controlPointsX = []
    controlPointsY = []
    controlShiftsX = []
    controlShiftsY = []
    controlCorrs = []
    
    # loop through each section of the image block data calculating the interpolated correlation matrix
    for i in range(idealSecNum):
        # calculate the beginning and ending index in x
        xBeg = i * sizeX
        xEnd = min(xBeg + sizeX, Nx)
        xCen = floor(xBeg + sizeX/2)
        
        for j in range(idealSecNum):
            yBeg = j * sizeY
            yEnd = min(yBeg + sizeY, Ny)
            yCen = floor(yBeg + sizeY/2)
            
            # compute the correlation matrix and sample indices in x and y
            R, ysam, xsam = computeCorrelation2(mData, sData, (yBeg, xBeg), (yEnd, xEnd), interpFactor=2**4)
            #print "The correlation is of size {}".format(R.shape)
            
            # determine the shift
            Rmax, yShift, xShift = determineShifts(R, ysam, xsam)
            
            # record the results relating to the shifts for the section if Rmax is above the threshold
            if Rmax > thresh:
                controlPointsX.append(xCen)
                controlPointsY.append(yCen)
                controlShiftsX.append(xShift)
                controlShiftsY.append(yShift)
                controlCorrs.append(Rmax)
            
             # lets now try to visualize the results
            if i==1 and j==3 and False:
                (yPos, xPos) = nonzero(abs(R) == abs(R).max())
                chunk1 = abs(mData[yBeg:yEnd,xBeg:xEnd])
                chunk2 = abs(sData[yBeg:yEnd,xBeg:xEnd])
                figure(8);imshow(chunk1,interpolation='nearest',clim=[chunk1.min(),chunk1.max()/2],cmap='gray')
                figure(9);imshow(chunk2,interpolation='nearest',clim=[chunk2.min(),chunk2.max()/2],cmap='gray')
                figure(11);imshow(abs(R),interpolation='nearest',clim=[abs(R).min(), abs(R).max()],
                       extent=(xsam[0],xsam[-1],xsam[-1],xsam[0]))
                title('Section %d,%d Correlation Matrix' % (i,j))
                
                figure(12);subplot(211);plot(xsam,abs(R[yPos,:].squeeze()))
                subplot(212);plot(ysam,abs(R[:,xPos].squeeze()))
                
            print "For section %d,%d, shifts are x: %0.3f and y: %0.3f, and corr: %0.4f" % (i,j,xShift,yShift,Rmax)
    
    return controlPointsY, controlPointsX, controlShiftsY, controlShiftsX, controlCorrs
    
def interpolateData2Points(slave, sBeg, sEnd, yInd, xInd, Ty, Tx):
    """
    Inputs: 2d numpy array of complex sar image data, tuple of y and x beginning
        indices, tuple of y and x ending indices, 2d array of y shifts, 2d array
        of x shifts, float containing the sampling interval in y, float containing
        the sampling interval in x
    """
    # compute the size of the block in x and y
    Ny = sEnd[0] - sBeg[0]
    Nx = sEnd[1] - sBeg[1]
    
    print 'Ny = %d, Nx = %d' % (Ny,Nx)
    # for the number of samples of the sinc to use for interpolation we will 
    # try using 25
    N = 50
    
    # allocate a 2d array for storing the new values of the interpolation
    slaveInter = zeros((Ny,Nx),'complex64')
    
    # interpolate in along x first, then y
    for i in range(int(Nx)):
        for j in range(int(Ny)):
            # need to determine the values of n to sum over (x indices in the original data)
            x = round(xInd[j,i])
            xBeg = max(x-N, 0)
            xEnd = min(x+N+1, slave.shape[1])
            nx = arange(xBeg, xEnd)
            
            # need to determine the values of n to sum over (y indices in the original data)
            y = round(yInd[j,i])
            yBeg = max(y-N, 0)
            yEnd = min(y+N+1, slave.shape[0])
            ny = arange(yBeg, yEnd)
            
            # create a hanning window in x
            wx = hanning(N*2+1)
            if xEnd == slave.shape[1]:
                wx = wx[:len(nx)]
            elif xBeg == 0:
                wx = wx[-len(nx):]
                
            # create the sinc matrix to cover x for the whole range of y
            #sincXMat = (sinc(((xInd[j,i]-(sBeg[1]+i)) - (nx-(sBeg[1]+i))*Tx)/Tx) * wx).reshape(1,len(nx)).repeat(len(ny),0)
            sincXMat = (sinc(((xInd[j,i] - x) - (nx - x)*Tx)/Tx) * wx).reshape(1,len(nx)).repeat(len(ny),0)
            
            # create a hanning window in y
            wy = hanning(N*2+1)
            if yEnd == slave.shape[0]:
                wy = wy[:len(ny)]
            elif yBeg == 0:
                wy = wy[-len(ny):]
                
            # create the sinc vector for y
            # sincY = sinc(((yInd[j,i] - (sBeg[0]+j)) - (ny-(sBeg[0]+j))*Ty)/Ty) * wy
            sincY = sinc(((yInd[j,i] - y) - (ny - y)*Ty)/Ty) * wy
            
            # interpolate in x
            tempInterData = (slave[yBeg:yEnd,xBeg:xEnd] * sincXMat).sum(1)
            
            slaveInter[j,i] = (tempInterData * sincY).sum()
            
            if i==100 and j==100:
                print "len(nx) = %d, len(ny) = %d" % (len(nx),len(ny))
                print "newX=%f, round(newX)=%d, newY=%f, round(newY)=%d" % (xInd[j,i],x,yInd[j,i],y)
                figure(30);plot(xInd[j,i] - nx,sincXMat[0,:].squeeze());title('The sinc in X')
                figure(31);plot(yInd[j,i] - ny,sincY);title('The sinc in Y')
    
    return slaveInter

def interpolateData(data, interFactor=2, N=50, T=1):
    # determine the length of the data
    Ndata = data.shape[0]
    
    # create an array of indices to interpolate toj
    inds = arange(0,Ndata,1.0/interFactor)
    
    # allocate an empty array for which to save interpolated data to
    interData = zeros(inds.shape,'complex64')
    
    # loop through all of the interpolation indices and interpolate the data
    for i in range(Ndata*interFactor):
        ind = round(inds[i])
        beg = max(ind-N,0)
        end = min(ind+N+1, Ndata)
        origInds = arange(beg,end)
        
        # create a hanning window
        w = hanning(N*2+1)
        if end == Ndata:
            w = w[:len(origInds)]
        elif beg == 0:
            w = w[-len(origInds):]
        
        # calculate the sinc function
        # sincF = sinc((inds[i] - origInds*T)/T) * w
        sincF = sinc(((inds[i] - ind) - (origInds - ind)*T)/T) * w
        interData[i] = (data[beg:end] * sincF).sum()
    
    
    # try again
    Nfreq = int(2**ceil(log2(Ndata)))    
    dataFreq = fft(data, Nfreq)
    interFreq = zeros(Nfreq*interFactor,'complex64')
    interFreq[:int(Nfreq/2+1)] = dataFreq[:int(Nfreq/2+1)]
    interFreq[int(Nfreq/2+1 + (interFactor-1)*Nfreq):] = dataFreq[int(Nfreq/2+1):]
    interData2 = ifft(interFreq)
    garbage = (Nfreq - Ndata)*interFactor
    interData2 = interData2[:Nfreq*interFactor - garbage]*interFactor
    
    return inds, interData, interData2
    
    
    
def blockShift(master, slave, idealblocksize=2.0**9):
    """
    INPUTS: SARImage, SARImage, float
    Note that the dimensions of the data in both SARImages should be the same
    at this point
    """
    # NOTE: I've found that using a fixed value that is a power of 2 for the blocksize
    # leads to potentially small leftover regions at the ends of the image that don't
    # end up having enough pixel support for a good correlation matrix estimation
    # Plus, there is not really any point to making the blocksize a multiple of 2
    # since I am just going to zero-pad it anyways.
    # SO, I propose that we calculate an X and Y blocklength that makes the chunks
    # as close to uniform in size as possible
    # we don't want the blocksizes to be around 2**9 (512), and we will err on the
    # side of slightly smaller so that the FFT doesn't have to be too big (1024, as
    # apposed to 2048)
    # determine number of blocks in x and y
    
    # compute the number of blocks and their size
    Ny, Nx = master.data.shape
    numBlockY, numBlockX, blockSizeY, blockSizeX  = computeBlockNum((Ny, Nx), idealblocksize)
    
    print numBlockX
    print numBlockY
    
    # allocate a matrix the size of the master image for reconstructing the image based
    # on the shifts of each block
    newSlave = zeros((Ny,Nx), 'complex64')
    newBSsarSlave = slave.copy()
    
    # loop over both dimensions calculating shifts for each block
    for i in range(numBlockX):
        # calculate the beginning and ending index in x
        xBeg = i * blockSizeX
        xEnd = min(xBeg + blockSizeX, Nx)
        
        for j in range(numBlockY):
            # calculate the beginning and ending index in y
            yBeg = j * blockSizeY
            yEnd = min(yBeg + blockSizeY, Ny)
            
            # compute the correlation matrix and sample indices in x and y
            R, ysam, xsam = computeCorrelation(master.data, slave.data, (yBeg, xBeg), (yEnd, xEnd))
            
            # determine the shift from the position of the maximum correlation
            Rmax, ybShift, xbShift = determineShifts(R, ysam, xsam)
            
            # lets now try to visualize the results
            if i==0 and j==2 and False:
                (yPos, xPos) = nonzero(abs(R) == abs(R).max())
                chunk1 = abs(master.data[yBeg:yEnd,xBeg:xEnd])
                chunk2 = abs(slave.data[yBeg:yEnd,xBeg:xEnd])
                figure(3);imshow(chunk1,interpolation='nearest',clim=[chunk1.min(),chunk1.max()/2],cmap='gray')
                figure(4);imshow(chunk2,interpolation='nearest',clim=[chunk2.min(),chunk2.max()/2],cmap='gray')
                figure(5);imshow(abs(R),interpolation='nearest',clim=[abs(R).min(), abs(R).max()],
                       extent=(xsam[0],xsam[-1],xsam[-1],xsam[0]))
                title('Block %d,%d Correlation Matrix' % (i,j))
                
                figure(6);subplot(211);plot(xsam,abs(R[yPos,:].squeeze()))
                subplot(212);plot(ysam,abs(R[:,xPos].squeeze()))
                
            print "For block %d,%d, shifts are x: %d and y: %d" % (i,j,xbShift,ybShift)
            
            # now shift the data and reconstruct the entire image
            newBSsarSlave.data = shiftSlaveData(slave, newSlave, (yBeg, xBeg), (yEnd, xEnd), ybShift, xbShift)
    
    return newBSsarSlave
    
def subpixelAlignment(master, slave, idealblocksize=2.0**9, idealSecNum=4, threshold=0.01):
    """
    INPUTS: SARImage, SARImage, float
    Note that the dimensions of the data in both SARImages should be the same
    at this point
    """
    # NOTE: I've found that using a fixed value that is a power of 2 for the blocksize
    # leads to potentially small leftover regions at the ends of the image that don't
    # end up having enough pixel support for a good correlation matrix estimation
    # Plus, there is not really any point to making the blocksize a multiple of 2
    # since I am just going to zero-pad it anyways.
    # SO, I propose that we calculate an X and Y blocklength that makes the chunks
    # as close to uniform in size as possible
    # we don't want the blocksizes to be around 2**9 (512), and we will err on the
    # side of slightly smaller so that the FFT doesn't have to be too big (1024, as
    # apposed to 2048)
    # determine number of blocks in x and y
    
    # record the sampling interval (T) for the data in x and y
    Tx = master.paramGeo['colPixelSizeM']
    Ty = master.paramGeo['rowPixelSizeM']
    Tx = 1
    Ty = 1
    
    # compute the number of blocks and their size
    Ny, Nx = master.data.shape
    numBlockY, numBlockX, blockSizeY, blockSizeX  = computeBlockNum((Ny, Nx), idealblocksize)
    
    print numBlockX
    print numBlockY
    
    # allocate a matrix the size of the master image for reconstructing the image based
    # on the shifts of each block
    newSlave = zeros((Ny,Nx), 'complex64')
    newBSsarSlave = slave.copy()
    
    # loop through each block of the image, and then go through each section of each block
    # and determine a shift for each section, and try to find the 10 best control points for the
    # block (these will be used for the warping)
    for i in range(numBlockX):
    #for i in [2]:
        # calculate the beginning and ending index in x
        xBeg = i * blockSizeX
        xEnd = min(xBeg + blockSizeX, Nx)
        
        for j in range(numBlockY):
        #for j in [0]:
            # calculate the beginning and ending index in y
            yBeg = j * blockSizeY
            yEnd = min(yBeg + blockSizeY, Ny)
            
            print 'Computing surface shifts for block: %d x %d' % (i, j)
            
            pointsY, pointsX, shiftsY, shiftsX, corrs = findControlPoints(master.data[yBeg:yEnd,xBeg:xEnd], slave.data[yBeg:yEnd,xBeg:xEnd], idealSecNum, threshold)
            print "There were %d control points found for block %d, %d" % (len(pointsY), i, j)
            
            if len(pointsY) > 10:
                # now we need calculate the coefficients (b1, ..., bn, a0, a1, a2) that describe the surfaces for x and y shifts
                yCoef, xCoef = computeSurfaceCoefficients(pointsY, pointsX, shiftsY, shiftsX, corrs)
                
                # use the coefficients to calculate the surface for x and y shifts
                sizeX = int(xEnd - xBeg)
                sizeY = int(yEnd - yBeg)
                
                # lay out a grid of the x and y indices
                X,Y = meshgrid(range(sizeX), range(sizeY))
                # calculate f
                f = yCoef[-3] + yCoef[-2]*X + yCoef[-1]*Y
                g = xCoef[-3] + xCoef[-2]*X + xCoef[-1]*Y
                for k in range(len(shiftsY)):
                    rsquare = (X - pointsX[k])**2 + (Y - pointsY[k])**2
                    rlogprod = rsquare * log(rsquare)
                    rlogprod[isnan(rlogprod)] = 0.0           
                    f += yCoef[k] * rlogprod
                    g += xCoef[k] * rlogprod
                
                X,Y = meshgrid(arange(xBeg,xEnd),arange(yBeg,yEnd))
                
                newSlaveBlock = interpolateData2Points(slave.data, (yBeg, xBeg), (yEnd, xEnd), Y-f, X-g, Ty, Tx)
                newBSsarSlave.data[yBeg:yEnd,xBeg:xEnd] = newSlaveBlock
            
            if i == 1 and j == 4 and True:
                figure(20);imshow(f,interpolation='nearest');colorbar()
                title('Y shifts as a surface for %d,%d' % (i,j))
                figure(21);imshow(g,interpolation='nearest');colorbar()
                title('X shifts as a surface for %d,%d' % (i,j))
                figure(22);imshow(abs(newSlaveBlock),interpolation='nearest',cmap='gray',
                       clim=[abs(newSlaveBlock).min(), abs(newSlaveBlock).max()/2]);colorbar()
                title('The warped and interpolated data for block %d,%d' % (i,j))                            
            
    return newBSsarSlave


    
    