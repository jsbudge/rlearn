# -*- coding: utf-8 -*-
"""
Created on Jun 26, 2012

@author: Josh Bradley

@purpose: Provide a SAR image class that provides useful methods and 
    and functions for manipulating and using SAR image data.

Last updated on Sep 18, 2012 by JPB
"""
# Imported modules and functions used in SARImage class implementation
from xml.dom.minidom import parse
from glob import glob
from re import search
import copy
from math import atan2, atan
from numpy import fromfile,sin,cos,pi,angle,zeros,abs,sqrt,ones,isnan,ceil,linspace,log10,log,\
                  logical_and,logical_not,log2,exp,nonzero,ndarray,round,any,histogram,\
                  logical_or, arctan, arctan2, array
from matplotlib.pyplot import figure,imshow,title,show,axis
from scipy.ndimage import gaussian_filter,convolve
from scipy.fftpack import fft,fftshift,ifft
from numpy import load
from os import system
import os

class SCPCOA(object):
    
    def __init__(self,scptime,arppos, arpvel):
        self.scptime = scptime
        self.arppos = arppos
        self.arpvel = arpvel
        
    def __repr__(self):
        return "Time: {}, position: {}, velocity: {}".format(self.scptime, self.arppos, self.arpvel)

class SARImage:
    """
    Represents a complex SAR image
        Parameters:
            fileName - is the full path and filename of the .ash file
            minus the ending .ash.  An error is thrown if there is not a .asi and
            image .kml file located in the same directory as the .ash file.
    """
    # keeps count of the number of SAR image objects
    _count = 0
    # keeps a record of the names of all the SAR image objects
    _names = []
    
    def __init__(self, fileName, scale=1e10):
        """Initialize an instance of the SARImage class"""
        self.scale = scale
        self._decompose_filename(fileName)
        # the ash file contains all of the necessary parameters pertinent to image creation
        self._parse_ash()
        # the kml file contains the lat/lon of the image bounding box
        #self._parse_kml()
        # the asi file contains the complex SAR image data
        self._parse_asi()
        # record the min and max of absolute 
        self.min = abs(self.data).min()
        self.max = abs(self.data).max()
        # set the reference latitude to the latitude of the center of the image
        #self.reflat = self.paramGeo['refLat']
        self.reflat = self.paramGeo['centerY']
        # compute the longitude and latitude conversion factors to nortings/eastings
        self.latConv,self.lonConv = computeLatLonConv(self.reflat)
        # compute and record the center easting and center northing
        self.cenE = self.paramGeo['centerX'] * self.lonConv
        self.cenN = self.paramGeo['centerY'] * self.latConv
        
        # convert the SCPCOA information from ECEF to LLA
        #self.scpcoa = self.convertECEF2LLA()
        
        # update the count of number of SAR image objects instantiated
        SARImage._count += 1
        # this instance of the SAR image class is number self.number
        self.number = SARImage._count
        self.dupe = 0 # counter for number of times the object was copied with the same name
        # append the SAR image object name to the global list of SAR image object names
        SARImage._names.append(self._name)
    
    def __str__(self):
        """Return a string representation of a SARImage object"""
        string = '\tType: %s\n\tName: %s\n\tCapture date: %s\n' % (self.type, self._name, 
                                                                   self.paramFlight['startDate'])
        string += '\tCapture time: %s\n\tCenter frequency: %0.2f MHz\n' % (self.paramFlight['startTime'],
                                                                           self.paramRadar['centerFreq']/1e6)
        string += '\tBandwidth: %0.2f MHz\n\tPolarization: %s\n' % (self.paramRadar['bandwidth']/1e6,
                                                                    self.paramRadar['polarization'])
        string += '\tCicada version: %0.2f\n' % (self.paramProc['algorithmVersion']) 
        string += '\tCenter latitude/longitude: %0.2f/%0.2f deg' % (self.paramGeo['centerY'],self.paramGeo['centerX'])
        return string
        
    def _decompose_filename(self, fileName):
        """Decompose the filepath into a path and base filename"""
        # generate a list of the parts of the fileName that are delimited by '/'
        parts = fileName.split('/')
        # the file's name is the last member of the list
        self._name = parts[-1]
        # record the origin name to keep track of what file it originated from
        self._basename = parts[-1]
        # initialize the path with an empty string
        self._path = ''
        # reconstruct the entire path by combining all but the last member of the list
        for i in parts[:-1]:
            self._path = self._path + i + '/'
    
    def _parse_ash(self):    
        """
        Parse the xml header file ('ash') associated with the SAR image data to
        retrieve the relative parameters for the dataset
        """
        # instantiate an object of the W3C Document Object Model (DOM) using the ".ash" file for parsing the xml
        # The subapertures have a different name than full aperture SAR images
        index = self._name.find('_s')
        ash_name = self._name
        if index >= 0:
            ash_name = self._name[:index+5]
        dom = parse(self._path + ash_name + '.ash')
        # The SARImage data type is assigned the xml base node
        self.type           = str(dom.childNodes[0].nodeName)
        # pass _parse_xml the DOM object corresponding to each of the sub-parameter fields (or nodes) 
        # to extract key/value pairs for a python dictionary data structure.  This is done for File_Parameters,
        # Image_Parameters, Geo_Parameters, Flight_Parameters, Radar_Parameters, Processing_Parameters.
        self.paramFile      = self._parse_xml(dom.childNodes[0].childNodes[1],{})
        self.paramImage     = self._parse_xml(dom.childNodes[0].childNodes[3],{})
        self.paramGeo       = self._parse_xml(dom.childNodes[0].childNodes[5],{})
        self.paramFlight    = self._parse_xml(dom.childNodes[0].childNodes[7],{})
        self.paramRadar     = self._parse_xml(dom.childNodes[0].childNodes[9],{})
        self.paramProc      = self._parse_xml(dom.childNodes[0].childNodes[11],{})
        #self.paramSICD      = self._parse_sicd(dom.childNodes[0].childNodes[13],{})
        
    def _parse_kml(self):
        """
        Parse through xml of the kml file to extract key/value pairs.  In particular,
        this extracts the north, south, east, west bounding edges of the SAR image.
        """
        # check to see if the name contains 'image', if it does, than it is a sub-aperture image
        if self._name.__contains__('image'):
            # then it is a subaperture image
            #print "I shouldn't be coming here."
            kml = glob(self._path + self._name + '*.kml')
        else:
            #print "This is where control should be going."
            # it is not a sub-aperture image and naming is different
            # generate a list of the parts of the fileName that are delimited by '_'
            parts1 = self._name.split('_')
            # grab the date from the file name
            year = int(parts1[1][-4:])
            month = int(parts1[1][:2])
            # the kml_base path is initialized to the datapath
            kml_base = self._path
            # complete the kml file base path by concatenating on the members of parts1
            for i in parts1[:-1]:
                kml_base = kml_base + i + '_'
            # assign kml a list of the paths matching the kml_base general expression
            #print "The kml base is: {}".format(kml_base)
            if (year >= 2013) or (year == 2012 and month >= 3):
                # the kml_base path is initialized to the datapath
                kml_base = self._path
                # complete the kml file base path by concatenating on the members of parts1
                for i in parts1[:-1]:
                    kml_base = kml_base + i + '_'
                kml_base = kml_base + parts1[-1]
                # assign kml a list of the paths matching the kml_base general expression
                #print "The kml string used in glob is {}".format(kml_base + '*.kml')
                kml = glob(kml_base + '*.kml')
            else:
                # kml = glob(self._path + self._name + '*.kml')
                # kml = glob(kml_base + '?????????.kml')
                kml = glob(kml_base + '*image*.kml')
        # the kml file path should be the first one in the list
        self._kmlname = kml[0]
        #print self._kmlname
        # instantiate an object of the W3C Document Object Model (DOM) using the ".ash" file for parsing the xml
        # generate a W3C Document Object Model (DOM) for the kml file xml
        dom = parse(self._kmlname)
        # grab the node that corresponds to the "LatLonBox"
        node = dom.childNodes[0].childNodes[1].childNodes[3].childNodes[9]
        # append the key/value pairs for the bounding box lat/lons to the paramGeo dictionary
        self.paramGeo = self._parse_xml(node,self.paramGeo)
        
    def _parse_asi(self):
        """Read in the complex SAR image data and reshape it"""
        # we should really scale the data by a value
        #scale = 1e10
        #scale = 1.0
        # construct filename from the SAR image object path, name, and .asi
        fileName = self._path + self._name + '.asi'
        # number of rows in the SAR image
        nRows = self.paramImage['nRows']
        # number of columns in the SAR image
        nCols = self.paramImage['nCols']
        # open the file for reading
        fid = open(fileName,'r')
        # read in the whole file binary data as type: complex64, and reshape to 2-dimensional numpy array
        self.data = (fromfile(fid,dtype='complex64',count=-1,sep="")).reshape((nRows,nCols),order='C')/self.scale
        fid.close()
        
    def _parse_sicd(self, xmlObject, param={}):
        """Extract the pertinent information from the SICD portion of the ASH file"""
        for node in xmlObject.childNodes:
            # if the type of node is an element node, then try to extract the key/value
            if node.nodeType == node.ELEMENT_NODE:
                # I am only interested in the information relating to the SCPCOA right now
                if str(node.nodeName) == "SCPCOA":
                    # I need to get the SCP time, position, and velocity
                    param[str(node.nodeName)] = self._parse_scpcoa(node, {})
        return param
    
    def _parse_scpcoa(self, xmlObject, param={}):
        """Extract the time, position, and velocity for the scpcoa"""
        for node in xmlObject.childNodes:
            # if the type of node is an element node, then try to extract the key/value
            if node.nodeType == node.ELEMENT_NODE:
                # if it has a child node, then we will extract the data
                if node.hasChildNodes():
                    if str(node.nodeName) == "ARPPos" or str(node.nodeName) == "ARPVel" or str(node.nodeName) == "ARPAcc":
                        # there is going to be multiple childnodes
                        subparam = {}
                        for cnode in node.childNodes:
                            # if the type of node is an element node, then try to extract the key/value
                            if cnode.nodeType == cnode.ELEMENT_NODE:
                                # if it has a child node, then we will extract the data
                                if cnode.hasChildNodes():
                                    subparam[str(cnode.nodeName)] = float(str(cnode.childNodes[0].nodeValue))
                                else:
                                    subparam[str(cnode.nodeName)] = []
                        # assign the current node the dictionary subparam
                        param[str(node.nodeName)] = subparam
                    elif str(node.nodeName) == "SideOfTrack":
                        # must interpret this one as a string
                        param[str(node.nodeName)] = str(node.childNodes[0].nodeValue)
                    else:
                        # for any other node, then there is just one child
                        param[str(node.nodeName)] = float(str(node.childNodes[0].nodeValue))
                else:
                    param[str(node.nodeName)] = []
        return param
        
    def _parse_xml(self, xmlObject, param={}):
        """Extract and return key/value pairs from xml objects.  This method accepts a DOM xml object"""
        # Loop through each node of the DOM xml object and assign the key/value pairs to the dictionary
        for node in xmlObject.childNodes:
            # if the type of node is an element node, then try to extract the key/value
            if node.nodeType == node.ELEMENT_NODE:
                # if it has a child node, then we will extract the data
                if node.hasChildNodes():
                    # if the node value contains a letter A-Za-z then interpret it as a string
                    # TODO: proper detection of scientific numbers ("e")
                    if search('[A-Za-z]|:',node.childNodes[0].nodeValue):
                        param[str(node.nodeName)] = str(node.childNodes[0].nodeValue)
                    # else interpret the node value as a float
                    else:
                        param[str(node.nodeName)] = float(node.childNodes[0].nodeValue)
                # else, if there is not a child node, then assign it an empty list
                else:
                    param[str(node.nodeName)] = []
        return param
      
    def _refresh(self):
        """Refreshes the min and max values based on the current data values"""
        self.min = abs(self.data).min()
        self.max = abs(self.data).max()
    
    def convertECEF2LLA(self):
        """Converts the information in the SCPCOA from SICD stuff to ECEF to LLA"""
        # compute the constants
        a1 = 6378137.0
        f1 = 1.0 / 298.257223563
        b1 = a1*(1.0 - f1)
        e1 = sqrt((a1**2 - b1**2) / a1**2)
        e2 = sqrt((a1**2 - b1**2) / b1**2)
        N1 = a1 / sqrt(1.0 - e1**2*sin(self.reflat * pi/180.0)**2)
        
        # store the X, Y, and Z positions
        X = self.paramSICD['SCPCOA']['ARPPos']['X']
        Y = self.paramSICD['SCPCOA']['ARPPos']['Y']
        Z = self.paramSICD['SCPCOA']['ARPPos']['Z']
        
        # compute auxiliary values
        p1 = sqrt(X**2 + Y**2)
        theta1 = atan2(Z*a1, p1*b1)
        
        # now comes the computations for the position
        longitude = atan2(Y, X)
        latitude = atan2(Z + e2**2*b1*sin(theta1)**3, p1 - e1**2*a1*cos(theta1)**3)
        altitude = p1 / cos(latitude) - N1
        
        # convert the longitude and latitude to northings and eastings
        Pe = longitude * 180/pi * self.lonConv
        Pn = latitude * 180/pi * self.latConv
        
        # store the X, Y, and Z velocities
        Vx = self.paramSICD['SCPCOA']['ARPVel']['X']
        Vy = self.paramSICD['SCPCOA']['ARPVel']['Y']
        Vz = self.paramSICD['SCPCOA']['ARPVel']['Z']
        
        # and now the computations for the velocity
        Vn = -Vx*sin(latitude)*cos(longitude) - Vy*sin(latitude)*sin(longitude) + Vz*cos(latitude)
        Ve = -Vx*sin(longitude) + Vy*cos(longitude)
        Vh = -(-Vx*cos(latitude)*cos(longitude) - Vy*cos(latitude)*sin(longitude) - Vz*sin(latitude))
        
        return SCPCOA(self.paramSICD['SCPCOA']['SCPTime'], array([[Pe],[Pn],[altitude]]), array([[Ve],[Vn],[Vh]]))
        
        
    
    def copy(self, name=None):
        """
        Return a copy of the current SAR image object with an updated name
            Method usage:
                sar_copy = sar.copy(name=None)
            Parameters:
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Note:
                If a name is given, it is appended to the end of 
                sar_image_object's name.  If a name is not given, then a new
                name is automatically created based on the sar SARImage object
        """
        # assign a copy of the this current SAR image using the built-in python "copy" module
        another = copy.copy(self)
        # update the SARImage class object count
        SARImage._count += 1
        # assign the copy's number to be the curent count
        another.number = SARImage._count
        # initialize the new copy's dupe field to 0
        another.dupe = 0
        # assign the copy a name
        if name:    # if a name is passed in, then append that to the end of the current SAR image object name
            another._name = '%s_%s' % (self._name, name)
        else:   # else, increment dupe and append the dupe number to the end of the current SAR image object name
            self.dupe += 1
            another._name = '%s_c%d' % (self._name, self.dupe)     
        # append the new object's name to the _names list for the SARImage class
        SARImage._names.append(another._name)
        return another
        
    def save(self, fileName=None):
        """
        Saves out the SAR image object data into a '.asi' file with the
        associated '.ash' and '.kml' files.  Unless a user specified fileName
        is passed when calling the method, the '_name' field of the object
        will be used for naming the files.  (Maybe support for saving out the
        '.png' file will be added later)
            Note:
                This method is currently under contruction (it really is just
                sort of a 'hack' right now) and may contain some minor issues
        """
        # Determine the new kml name
        # generate a list of the parts of the fileName that are delimited by '_'
        parts1 = self._kmlname.split('_')
        # the kml_base path is initialized to the datapath
        kml_base = ''
        # complete the kml file base path by concatenating the members of parts1
        for i in parts1[:-2]:
            kml_base = kml_base + i + '_'
        # determine length of origin name
        length = self._basename.__len__()
        # add on the additional names tacked onto the end, if any
        kmlname = kml_base + self._name[length+1:]
        # now add on the original ending of the original kml file
        for i in parts1[-2:]:
            kmlname = kmlname + '_' + i
        
        # to simplify this for the time being, I am going to just make copies
        # of the original .ash and .kml files with different names, but later
        # on functionality should be added to create a completely original 
        # .ash and .kml with any information that may have changed
        prevashname = self._path + self._basename + '.ash'
        curashname = self._path + self._name + '.ash'
        cmd = 'cp %s %s' % (prevashname, curashname)
        system(cmd)
        cmd = 'cp %s %s' % (self._kmlname, kmlname)
        system(cmd)
        
        # write out the complex image data to file
        asiname = self._path + self._name + '.asi'
        # open file for binary writing
        fid = open(asiname,'wb')
        # write data to file as type 'complex64', which is equivalent to 'float32'
        # for the real and imaginary parts separately
        # this writes to file in 'C' standard order (row by row)
        (self.data.astype('complex64')).tofile(fid)
        # close file
        fid.close()
        
        kmlparts = kmlname.split('/')
        print 'Wrote the following files to %s:\n\t%s\n\t%s\n\t%s' % (self._path,self._name+'.asi',self._name+'.ash',kmlparts[-1])
        
    def mag(self):
        """Return the magnitude of the SAR image data"""
        return abs(self.data)
        
    def phase(self):
        """Return the phase of the SAR image data"""
        return angle(self.data)
    
    def showMag(self, fignum=None, scale=10):
        """
        Show a 2D visualization of the complex SAR image magnitude
            Method usage:
                SARImage_object.showMag(fignum=None,scale=10)
            Parameters:
                fignum - the figure's number
                scale - the value used to scale the max end of the colorbar
                (ie. vmax=max(data)/scale)
        """
        # create a figure
        figure(num=fignum)
        # display the amplitude of the SAR image data in the figure with
        # a gray colormap, and default interpolation
        imshow(abs(self.data),cmap='gray',interpolation='none',
            vmin=0,vmax=abs(self.data).max()/scale)
        # scale the axis limits to be equal to the data limits
        axis('image')
        # display as the title of the figure the name of the sar image
        title(self._name + ' magnitude')
        # command the figure to show itself
        show()
        
    def showPhase(self, fignum=None):
        """
        Show a 2D visualization of the complex SAR image phase
            Method usage:
                SARImage_object.showPhase(fignum=None)
            Parameters:
                fignum - the figure's number
        """
        # create a figure
        figure(num=fignum)
        # display the phase of the SAR image data in the figure with a jet
        # colormap, and default interpolation
        imshow(angle(self.data),cmap='jet',interpolation='none')
        # scale the axis limits to be equal to the data limits
        axis('image')
        # display as the title of the figure the name of the sar image
        title(self._name + ' phase')
        # command the figure to show itself
        show()
    
    def ifer(self, SARImage2, name='Ifer'):
        """
        Returns a SAR image object that is the interferogram of the calling 
        SAR image and a 2nd SAR image
            Method usage:
                ifer = sar1.ifer(sar2,name='Ifer')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                interferogram\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                ifer - a SARImage object with the interferogram of sar1 and 
                sar2 (ifer = sar1 * conj(sar2))
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SARImage object data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            # create a copy of the calling SAR image object to hold data
            interferogram = self.copy(name)
            # assign the new SAR image object's data field to hold the
            # interferometric data
            interferogram.data = interferogram.data * SARImage2.data.conj()
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR images are not the same size")
        # make sure that all the pixels where there was no data, are set to 0
        interferogram.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 0.0
        # refresh the mag and min fields of the interferogram SAR image object
        interferogram._refresh()
        # return the interferogram SAR image object
        return interferogram
        
    def computeSNR(self, SARImage2, diam=7, name='SNR'):
        """
        Returns a SAR image object of the correlation between the calling SAR 
        image and a 2nd SAR image
            Method usage:
                gamma = sar1.coherence(sar2,diam=5,name='Correl')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                coherence\n
                diam - specifies the width of the box (in pixels) used for 
                estimating the coherence (using a number too low will bias
                the coherence estimate greater than what it should be)\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                R - a SARImage object with the complex sample correlation
                between sar1 and sar2
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SARImage oject data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            # create a copy of the calling SAR image object to hold data
            R = self.copy(name)
            # create a structure to use as a summing filter kernel
            struct = ones((diam,diam))
            temp = (self.data.astype('complex128')) * (SARImage2.data.astype('complex128').conj())
            realg = convolve(temp.real,struct,mode='constant')
            R.data = realg
            realg = convolve(abs(SARImage2.data - self.data)**2,struct,mode='constant')
            R.data = 2*R.data / realg
            # for some reason there may appear some nans, and they must be set to 0
            # (this is probably a bug that can be avoided by doing these steps differently)
            R.data[isnan(R.data)] = 1e-20
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR image are not the same size")
        # make sure that all the pixels where there was no data is set to 0.0
        R.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 1e-20
        # refresh the min and max fields of the new SAR image object
        R._refresh()
        return R.data
        
    def computeNoise(self, SARImage2, diam=7, name='SNR'):
        """
        Returns a SAR image object of the correlation between the calling SAR 
        image and a 2nd SAR image
            Method usage:
                gamma = sar1.coherence(sar2,diam=5,name='Correl')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                coherence\n
                diam - specifies the width of the box (in pixels) used for 
                estimating the coherence (using a number too low will bias
                the coherence estimate greater than what it should be)\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                R - a SARImage object with the complex sample correlation
                between sar1 and sar2
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SARImage oject data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            # create a copy of the calling SAR image object to hold data
            R = self.copy(name)
            # create a structure to use as a summing filter kernel
            struct = ones((diam,diam))
            realg = convolve(abs(self.data - SARImage2.data)**2,struct,mode='constant')
            R.data = realg/(2*diam**2)
            # for some reason there may appear some nans, and they must be set to 0
            # (this is probably a bug that can be avoided by doing these steps differently)
            R.data[isnan(R.data)] = 1e-20
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR image are not the same size")
        # make sure that all the pixels where there was no data is set to 0.0
        R.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 1e-20
        # refresh the min and max fields of the new SAR image object
        R._refresh()
        return R.data
        
    def covariance(self, SARImage2, diam=7, name='Covar'):
        """
        Returns a SAR image object of the covariance between the calling SAR 
        image and a 2nd SAR image
            Method usage:
                gamma = sar1.coherence(sar2,diam=5,name='Covar')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                coherence\n
                diam - specifies the width of the box (in pixels) used for 
                estimating the coherence (using a number too low will bias
                the coherence estimate greater than what it should be)\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                R - a SARImage object with the complex sample correlation
                between sar1 and sar2
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SARImage oject data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            dind = nonzero(self.mag()>0)
            # create a copy of the calling SAR image object to hold data
            R = self.copy(name)
            # create a structure to use as a summing filter kernel
            struct = ones((diam,diam))
            # compute the interferogram
            # now I need to calculate the mean for each channel
            realg = convolve(self.data.real,struct,mode='constant')
            imagg = convolve(self.data.imag,struct,mode='constant')
            mean1 = realg + 1j*imagg
            realg = convolve(SARImage2.data.real,struct,mode='constant')
            imagg = convolve(SARImage2.data.imag,struct,mode='constant')
            mean2 = realg + 1j*imagg
            R.data = self.data * SARImage2.data.conj()
            R.data = (self.data.astype('complex128') - mean1) * (SARImage2.data.astype('complex128') - mean2).conj()
            # sum the real and imaginary parts of interferogram within the box
            # surrounding each pixel
            realg = convolve(R.data.real,struct,mode='constant')
            imagg = convolve(R.data.imag,struct,mode='constant')
            # recombine the real and imaginary parts to create complex data
            #R.data = realg + 1j*imagg - mean1*mean2.conj()
            #i1 = sqrt(convolve((self.data.real)**2 + (self.data.imag)**2,struct,mode='constant'))
            #i2 = sqrt(convolve((SARImage2.data.real)**2 + (SARImage2.data.imag)**2,struct,mode='constant'))
            # divide the interferogram by the multiplication of the sums of magnitudes
            #R.data[dind] = R.data[dind] / (i1[dind] * i2[dind])
            # for some reason there may appear some nans, and they must be set to 0
            # for some reason there may appear some nans, and they must be set to 0
            # (this is probably a bug that can be avoided by doing these steps differently)
            R.data[isnan(R.data)] = 1e-10
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR image are not the same size")
        # make sure that all the pixels where there was no data is set to 0.0
        R.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 1e-10
        # refresh the min and max fields of the new SAR image object
        R._refresh()
        return R
        
    def cfar_2param(self, gaurd_bins=5, out_bins=15):
        temp = abs(self.data.copy()) + 0.0
        R = abs(self.data.copy()) + 0.0
        
        # first calculate the gaurd cell statistics (mean)
        gaurd_struct = ones((gaurd_bins, gaurd_bins))
        gaurd_mean = convolve(temp, gaurd_struct, mode='constant')/gaurd_bins**2
        # (variance)
        R = temp * temp.conj()
        gaurd_var = convolve(R, gaurd_struct, mode='constant')/gaurd_bins**2
                
        # now calculate the reference cell statistics (mean)
        out_struct = ones((out_bins, out_bins))
        out_mean = convolve(temp, out_struct, mode='constant')/out_bins**2
        # (variance)
        R = temp * temp.conj()
        out_var = convolve(R.real, out_struct, mode='constant')/out_bins**2
        
        # Now we need to calculate the mean and variance of the reference cell
        # from the estimate of the outer cell and gaurd cell
        bin_diff = out_bins**2 - gaurd_bins**2
        ref_mean = (out_bins**2/bin_diff)*out_mean - (gaurd_bins**2/bin_diff)*gaurd_mean
        ref_var = (out_bins**2/bin_diff)*out_var - (gaurd_bins**2/bin_diff)*gaurd_var
        ref_std = sqrt(ref_var - ref_mean**2)
        
        cfar = (temp - ref_mean) / ref_std
        
        return cfar
    
    def mean_cfar(self, gaurd_bins=3, out_bins=9):
        temp = abs(self.data.copy()) + 0.0
        R = abs(self.data.copy()) + 0.0
        
        # first calculate the gaurd cell statistics (mean)
        gaurd_struct = ones((gaurd_bins, gaurd_bins))
        gaurd_mean = convolve(temp, gaurd_struct, mode='constant')/gaurd_bins**2
                
        # now calculate the reference cell statistics (mean)
        out_struct = ones((out_bins, out_bins))
        out_mean = convolve(temp, out_struct, mode='constant')/out_bins**2
        
        # Now we need to calculate the mean and variance of the reference cell
        # from the estimate of the outer cell and gaurd cell
        bin_diff = out_bins**2 - gaurd_bins**2
        ref_mean = (out_bins**2/bin_diff)*out_mean - (gaurd_bins**2/bin_diff)*gaurd_mean
        
        cfar = (temp - ref_mean)
        
        return cfar
        
    def peak_cfar(self, gaurd_bins=2, out_bins=5):
        temp = abs(self.data.copy()) + 0.0
        
        return 0
        
    def correlation(self, SARImage2, diam=7, name='Correl'):
        """
        Returns a SAR image object of the correlation between the calling SAR 
        image and a 2nd SAR image
            Method usage:
                gamma = sar1.coherence(sar2,diam=5,name='Correl')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                coherence\n
                diam - specifies the width of the box (in pixels) used for 
                estimating the coherence (using a number too low will bias
                the coherence estimate greater than what it should be)\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                R - a SARImage object with the complex sample correlation
                between sar1 and sar2
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SARImage oject data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            # create a copy of the calling SAR image object to hold data
            R = self.copy(name)
            # create a structure to use as a summing filter kernel
            struct = ones((diam,diam))
            # compute the interferogram
            R.data = self.data * SARImage2.data.conj()
            # sum the real and imaginary parts of interferogram within the box
            # surrounding each pixel
            realg = convolve(R.data.real,struct,mode='constant')/diam**2
            imagg = convolve(R.data.imag,struct,mode='constant')/diam**2
            # recombine the real and imaginary parts to create complex data
            R.data = realg + 1j*imagg
            # for some reason there may appear some nans, and they must be set to 0
            # (this is probably a bug that can be avoided by doing these steps differently)
            R.data[isnan(R.data)] = 1e-20
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR image are not the same size")
        # make sure that all the pixels where there was no data is set to 0.0
        R.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 1e-20
        # refresh the min and max fields of the new SAR image object
        R._refresh()
        return R
        
    def entropy(self, diam=5, name='Entropy'):
        """
        Returns a SAR image object of the entropy in the SAR image
            Method usage:
                gamma = sar1.coherence(diam=5,name='Entropy')
            Parameters:
                diam - specifies the width of the box (in pixels) used for 
                estimating the coherence (using a number too low will bias
                the coherence estimate greater than what it should be)\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                gamma - a SARImage object with the complex sample entropy
                between sar1 and sar2
            Note:
                If the data is not the same shape an exception will be raised
        """
        # calculate the total power P in the image
        P = (self.data*self.data.conj()).real.sum()
        # create a copy of the calling SAR image object to hold data
        H = self.copy(name)
        # create a structure to use as a summing filter kernel
        struct = ones((diam,diam))
        # compute the interferogram
        H.data = (self.data*self.data.conj()).real/P
        # sum the entropy statistic with box surrounding each pixel
        H.data = -convolve(H.data*log(H.data),struct,mode='constant')
        # for some reason there may appear some nans, and they must be set to 0
        # (this is probably a bug that can be avoided by doing these steps differently)
        H.data[isnan(H.data)] = 1e-20
        # make sure that all the pixels where there was no data is set to 0.0
        H.data[self.mag()==0] = 1e-20
        # refresh the min and max fields of the new SAR image object
        H._refresh()
        return H
            
    def coherence(self, SARImage2, diam=5, name='Coher'):
        """
        Returns a SAR image object of the coherence between the calling SAR 
        image and a 2nd SAR image
            Method usage:
                gamma = sar1.coherence(sar2,diam=5,name='Coher')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                coherence\n
                diam - specifies the width of the box (in pixels) used for 
                estimating the coherence (using a number too low will bias
                the coherence estimate greater than what it should be)\n
                name - an identifier appended to the end of the name of the
                calling SAR image object
            Returns:
                gamma - a SARImage object with the complex sample coherence
                between sar1 and sar2
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SARImage oject data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            # find the indices for the pixels with non-zero data (to avoid a divide by zero later)
            dind = nonzero(self.mag()>0)
            # create a copy of the calling SAR image object to hold data
            gamma = self.copy(name)
            # create a structure to use as a summing filter kernel
            struct = ones((diam,diam))
            # compute the interferogram
            gamma.data = self.data * SARImage2.data.conj()
            # sum the real and imaginary parts of interferogram within the box
            # surrounding each pixel
            realg = convolve(gamma.data.real,struct,mode='constant')
            imagg = convolve(gamma.data.imag,struct,mode='constant')
            # recombine the real and imaginary parts to create complex data
            gamma.data = realg + 1j*imagg
            # sum the SAR image data amplitude within the box surrounding each pixel
            # for each SAR image object
            i1 = sqrt(convolve(self.data.real**2 + self.data.imag**2,struct,mode='constant'))
            i2 = sqrt(convolve(SARImage2.data.real**2 + SARImage2.data.imag**2,struct,mode='constant'))
            # divide the interferogram by the multiplication of the sums of magnitudes
            gamma.data[dind] = gamma.data[dind] / (i1[dind] * i2[dind])
            # for some reason there may appear some nans, and they must be set to 0
            # (this is probably a bug that can be avoided by doing these steps differently)
            gamma.data[isnan(gamma.data)] = 1e-10
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR image are not the same size")
        # make sure that all the pixels where there was no data is set to 0.0
        gamma.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 1e-10
        # refresh the min and max fields of the new SAR image object
        gamma._refresh()
        return gamma
        
    def dpca(self, SARImage2, name='dpca'):
        """
        Returns a SAR image object of the DPCA of the current SAR image and a
        2nd SAR image
            Method usage:
                dpca = sar1.dpca(sar2,name='dpca')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for DPCA\n
                name - an identifier appended to the end of the name of the 
                calling SAR image object
            Returns:
                dpca - a SARImage object with the DPCA of sar1 and sar2
                (dpca = sar1 - sar2)
            Note:
                If the data is not the same shape an exception will be raised
        """
        # check to make sure the SArImage object data is the same shape
        if (self.data.shape == SARImage2.data.shape):
            # create a copy of the calling SAR image object to store data
            dpca = self.copy(name)
            # compute the DPCA as the complex subtraction of the two SAR image objects' data
            dpca.data = dpca.data - SARImage2.data
        else:
            # throw an error since the SAR image objects are not the same shape
            raise Exception("SAR images are not the same size")
        # make sure that all the pixels where there was no data is set to 0.0
        dpca.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 0.0
        # refresh the min and max fields of the new SAR image object
        dpca._refresh()
        return dpca
        
    def iferSmooth(self, SARImage2, sigma=None, name='smIfer'):
        """
        Returns a SAR image object of the smoothed interferogram of the 
        current SAR image and a 2nd SAR image
            Method usage:
                smooth_ifer = sar1.iferSmooth(sar2,sigma=None,name='smIfer')
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                smoothed interferogram\n
                sigma - sigma is the standard deviation associated with a 
                standard normal distribution that is used as a parameter for
                a guassian filter.  If a value is given the interferogram
                will be smoothed using a guassian filter, otherwise a 
                modified gaussian/circular filter with std=2 will be used 
                that provides better edge and detail preservation.\n
                name - an identifier appended to the end of the name of the 
                calling SAR image object
            Returns:
                smooth_ifer - a SARImage object with the smoothed
                interferogram of sar1 and sar2 (smooth_ifer = sar1 * conj(sar2))
            Note:
                If the data is not the same shape an exception will be raised
        """
        # attain a new SAR image object of the interferogram of the two SAR images
        interferogram = self.ifer(SARImage2,name)
        # if the interferogram was created successfully, then we will proceed to smooth
        if interferogram:
            # call the smooth method
            interferogram.smooth(sigma)
        # make sure all the pixels where there was no data is set to 0.0
        interferogram.data[logical_or(self.mag()==0,SARImage2.mag()==0)] = 0.0
        # refresh the min and max fields of the new SAR image object
        interferogram._refresh()
        return interferogram
        
    def smooth(self, sigma=None):
        """
        Smoothes the SAR image data using a modified gaussian filter with
        std=2, unless a value is specified for sigma
            Method usage:
                SARImage_object.smooth(sigma=None)
            Parameters:
                sigma - sigma is the standard deviation associated with a 
                standard normal distribution that is used as a parameter for
                a guassian filter.  If a value is given the interferogram
                will be smoothed using a guassian filter, otherwise a 
                modified gaussian/circular filter with std=2 will be used 
                that provides better edge and detail preservation.
        """
        # if the interferogram was created successfully, then we will proceed to smooth
        if (sigma):
            # smooth the real and imaginary parts separately
            reald = gaussian_filter(self.data.real,sigma)
            imagd = gaussian_filter(self.data.imag,sigma)
        else:
            # Formulate gaussian disk for 2D filter kernel
            # make a kernel of size 7x7
            kernel = zeros((7,7))
            # compute the impulse response of the gaussian filter in the 7x7 kernel by first
            # making the kernel a 2-dimensional delta function (assign the center pixel 1)
            kernel[3,3] = 1
            # now make it a circular kernel by setting the pixels with values less
            # then the 3rd pixel in from the edge, on the edge
            kernel = gaussian_filter(kernel,2)
            # convolve the interferogram with the filter kernel separately
            kernel[kernel<kernel[2,0]] = 0
            # Convolve the interferogram with the filter kernel
            reald = convolve(self.data.real,kernel)
            imagd = convolve(self.data.imag,kernel)
        # Recompose smoothed real and imaginary components
        self.data = reald + 1j*imagd
        self._refresh()
        
    def balance(self, pol, subNum, overOcean = False, sar2 = None):
        """
        Applies the channel balancing corections previously computed over
        land if overOcean = True, otherwise, calculates it's own balance
        parameters and then applies them.
            Method usage:
                SARImage_object.balance(pol, overOcean)
            Parameters:
                pol - either 'v' or 'h' designating polarization
                subNum - is the subaperture number (int)
                overOcean - a boolean specifying whether the interferometric
                pair is for over ocean.  If this is True, precalculated 
                corrections will be read in from file, otherwise, corrections
                will be calculated on fly if it is False.
        """
        if overOcean:
            # determine file name
            filename = '%s/ati_corrections/fore_%spol_corrections_s%0.2d.npy' % (os.getcwd(),pol,subNum)
            print "Balance corrections read in from %s" % (filename)
            balparam = load(filename)
            atiBias = balparam[0]
            mu = balparam[1]
            sig = balparam[2]
            # if an another sar image was specified, then they want to calculate magnitude balances with
            # the given sar image
            if sar2 and True:
                print "The program execution should not be getting here."
                throwaway, mu, sig = self.calculateBalanceParameters(sar2)
        else:
            atiBias, mu, sig = self.calculateBalanceParameters(sar2)
        
        newMags = 10.0**(((10*log10(abs(self.data[self.mag() > 0.0])) - mu) / sig)/10.0)
        newPhases = angle(self.data[self.mag() > 0.0]) - atiBias
        self.data[self.mag() > 0.0] = newMags * exp(1j*newPhases)
        self._refresh()
        
        return atiBias, mu, sig
        
    def calibrate(self, SARImage2, iterations=3):
        """
        Calibrates the current SAR image object to the 2nd SAR image object 
        data. This includes balancing the amplitude of the two images, and 
        calibrating their phases.
            Method usage:
                SARImage_object.calibrate(SARImage2,iterations=3)
            Parameters:
                SARImage2 - is the 2nd SAR image object that is used for the
                calibrating the calling SAR image object
                iterations - the calibration technique is iterative, this
                specifies the number of iterations to perform
        """
        # We will be performing a 2-dimensional FFT followed by some summing
        # and some complex multiplication
        P = self.data.shape[0]  # the height of the data matrix
        N = self.data.shape[1]  # the width of the data matrix
        zeroP = 2**(ceil(log2(P))) # length of the FFT in height
        zeroN = 2**(ceil(log2(N))) # length of the FFT in width
        # I need to define some 1D arrays of one for some matrix multiplication later
        oneP = ones((1,zeroP))  # a 1D array of ones the length of the FFT in height
        oneN = ones((1,zeroN))  # a 1D array of ones the length of the FFT in width
        
        # take the size=zeroP fft of the first SAR image data in height (0 axis), and shift it
        self.data = fftshift(fft(self.data,n=zeroP,axis=0),axes=0)
        # take the size=zeroN fft of the first SAR image data in width (1 axis) and shift it
        self.data = fftshift(fft(self.data,n=zeroN,axis=1),axes=1)
        # do the same for the second SAR image data
        afz = fftshift(fft(SARImage2.data,n=zeroP,axis=0),axes=0)
        afz = fftshift(fft(afz,n=zeroN,axis=1),axes=1)
        
        # Magnitude balance
        # compute the ratio of mean magnitude for the 2nd SAR image to the 1st SAR image
        # this will be our multiplicative calibration factor for the magnitude
        avratio = abs(afz).mean()/abs(self.data).mean()
        # mutliply the calibration factor by the 1st SAR image data
        self.data = self.data * avratio
        
        # iterative phase calibration
        k = 0
        while k < iterations:
            k = k + 1
            # balance over width (eastings) of the SAR image data
            # calculate the angles of the sums of the conjugate multiply of SAR 1 and 2 down each column
            cal0 = angle((self.data.conj() * afz).sum(0))
            # matrix multiply this by the array of ones the size of the height to get a zeroPxzeroN matrix
            # where the rows are all identical and each column contains the calibration factor for that column
            cal0 = cal0.reshape(1,cal0.size).T.dot(oneP).T
            #foz = foz * cal0
            # multiply the 2d FFT of the 1st SAR by the complex exponential of the cal0 factors
            self.data = self.data * exp(1j*cal0)
            # balance over height (northings) of the SAR image data
            # calculate the angles of the sums of the conjugate multiply of SAR 1 and 2 across each row
            cal1 = angle((self.data.conj() * afz).sum(1))
            # matrix multiply this by the array of ones the size of the width to get a zeroPxzeroN matrix
            # where the columns are all identical and each row contains the calibration factor for that column
            cal1 = cal1.reshape(cal1.size,1).dot(oneN)
            #foz = foz * cal1
            # multiply the 2d FFT of the 1st SAR by the complex exponential of the cal1 factors
            self.data = self.data * exp(1j*cal1)
        # delete these variables because they are no longer needed and are taking up resources
        del oneN, oneP, cal0, cal1
        
        # inverse FFT the calibrated 1st SAR image data first in width 
        self.data = ifft(fftshift(self.data,axes=1),axis=1)
        # We only want to keep the first N columns because (zeroN-N) columns should just be 0 
        # (we implicitly performed zero padding when we took the FFT)
        self.data = self.data[:,0:N]
        # next inverse FFT the data in the height
        self.data = ifft(fftshift(self.data,axes=0),axis=0)
        # We only want to keep the first P rows because (zeroP-P) rows should just be 0
        # (we implicitly performed zero padding when we took the FFT)
        self.data = self.data[0:P,:] 
        # refresh the min and max fields in the SAR image object
        self._refresh()

        # Zero out pixels that should contain no data
        self.data[SARImage2.mag()==0] = 0
        # change the name of the object to reflect the fact that the data has been modified
        self._name += '_balanced'
        
    def coregister(self, SARImage2):
        """
        Returns a copy of the calling SAR image object with data from the 
        referenced SAR image resampled onto its image grid
            Method call:
                sar2_coreg = sar1.coregister(sar2)
            Parameters:
                sar2 - is the SARImage object whos data will be 
                resampled onto the calling SAR image object's grid
            Returns:
                sar2_coreg - a SARImage object containing the data from
                sar2 overlapping with sar1 data resampled onto sar1 image grid
        """
        # assume originally that the two SARImage objects' data will overlap until proven otherwise
        overlap = True
        # perform basic check to make sure the image borders atleast overlap
        if self.paramGeo['north'] < SARImage2.paramGeo['south']:
            overlap = False
        elif self.paramGeo['south'] > SARImage2.paramGeo['north']:
            overlap = False
        elif self.paramGeo['east'] < SARImage2.paramGeo['west']:
            overlap = False
        elif self.paramGeo['west'] > SARImage2.paramGeo['east']:
            overlap = False
        
        # If the borders overlap, then we need to check if they have
        # overlapping data, otherwise throw an error
        if overlap:
            # create a mask of the pixels that contain data for the 1st SAR image
            sar1map = self.mag() > 0
            # create a tuple of the x and y indices of the data pixels in the 2nd SAR image
            sar2ind = nonzero(SARImage2.mag() > 0)
            # calculate the northings and eastings associated with the indices of the data in the 2nd SAR image
            # (it's important that these are computed using the reflat associated with the 1st SAR image, or mostly
            # it is just important that the same reflat is used throughout this entire process)
            sar2n,sar2e = SARImage2.pix2ne(sar2ind[0],sar2ind[1],self.reflat)
            # now convert those northings and eastings into x and y pixel coordinates in the 1st SAR image grid
            sar2y1,sar2x1 = self.ne2pix(sar2n, sar2e)
            # round to the nearest pixel edge and then convert to an integer
            sar2y1 = round(sar2y1).astype('int')
            sar2x1 = round(sar2x1).astype('int')
            # find all of the pixels indices that are within the bounds of the 1st SAR image
            tempy = logical_and(sar2y1>=0,sar2y1<sar1map.shape[0])
            tempx = logical_and(sar2x1>=0,sar2x1<sar1map.shape[1])
            # these are the valid indices
            valid = logical_and(tempy,tempx)
            # grab only the valid indices
            sar2y1 = sar2y1[valid]
            sar2x1 = sar2x1[valid]
            # create an unitialized logical matrix the size of the 1st SAR image 
            sar2map = ndarray(sar1map.shape,sar1map.dtype)
            # initialize all elements of the matrix to False
            sar2map[:] = False
            # make True the pixels of the section of the 1st SAR image that overlaps with the 2nd SAR image
            sar2map[sar2y1,sar2x1] = True
            # now check to see that their actual areas with data overlap by finding where both masks are true
            similar = logical_and(sar1map,sar2map)
            # Set our overlap flag (if there are any True pixels in similar then overlap will be True, otherwise False)
            overlap = any(similar)
        # if no overlapping data regions then throw an error
        if not overlap:
            raise Exception("SAR images share no overlapping data")
        # Now resample the second SAR images data onto the map of the 
        # overlapping region on the first SAR image
        # create a copy of the calling SAR image object to store data (and give it a descriptive name)
        sar2res1 = self.copy('with_resampled_data_from_'+SARImage2._name)
        # reassign its data so it doesn't point to the same place in memory
        sar2res1.data = ndarray(self.data.shape,self.data.dtype)
        # initialize the entire matrix to 0
        sar2res1.data[:] = 0
        # assign the pixels of overlap to contain the data from the 2nd SAR image 
        # (this is implicitly using neareast neighbor interpolation)
        # this could be improved potentially by using some other sort of interpolation, but nearest neighbor was the easiest
        # to implement for me, and since this was just meant to be a quick proof of concept thing, that was good enough
        sar2res1.data[sar2y1,sar2x1] = SARImage2.data[sar2ind[0][valid],sar2ind[1][valid]]
        # this last step seems like it might be redundant and unnecessary (setting the pixels where
        # there was no overlap to 0.  But they should already be zero, since we only changed the pixels with overlap.)
        sar2res1.data[logical_not(similar)] = 0
        return sar2res1
        
    def pix2ne(self, y, x, reflat=None):
        """
        Returns the northings/eastings of the x and y indices of pixels with 
        respect to the reflat and lat/lon values of the boundaries associated
        with the SAR image
            Method usage:
                north, east = SARImage_object.pix2ne(y_ind, x_ind, reflat)
            Parameters:
                y - a numpy array of y (or vertical) indices of pixels in the 
                image\n
                x - a numpy array of x (or horizontal) indices of pixels in 
                the image\n
                reflat - the reference latitude used to compute longitude
                and latitude conversion factors (unless specified the 
                reference latitude associated with the calling SAR image
                object will be used)
            Returns:
                north - a numpy array of northings\n
                east - a numpy array of eastings
        """
        # if a reflat is not specified, then use the precalculated lat/lon conversions for the SAR image object
        if not reflat:
            latConv = self.latConv
            lonConv = self.lonConv
        else:
            # compute new longitude and latitude conversion factors based on the reflat passed in
            latConv,lonConv = computeLatLonConv(reflat)
        # calculate the range of northing values for the SAR image object data
        nrange = (self.paramGeo['north'] - self.paramGeo['south']) * latConv
        # calculate the range of easting values for the SAR image object data
        erange = (self.paramGeo['east'] - self.paramGeo['west']) * lonConv
        # the number of rows in the image
        y_range = self.data.shape[0] - 1
        # the number of columns in the image
        x_range = self.data.shape[1] - 1
        # compute the easting values based on the x indices (note that I am interested in the eastings
        # associated with the center of the pixel, hence the + 0.5)
        east = self.paramGeo['west']*lonConv + (x+0.5)/x_range * erange
        # compute the northing values based on the y indices (note that I am interested in the northings
        # associated with the center of the pixel, hence the + 0.5)
        north = self.paramGeo['north']*latConv - (y+0.5)/y_range * nrange
        return north, east
        
    def ne2pix(self, north, east, reflat=None):
        """
        Returns the x and y pixels indices of northing/easting coordinates 
        with respect to the reflat and lat/lon values of the boundaries 
        associated with the SAR image
            Method usage:
                y_ind, x_ind = SARImage_object.ne2pix(north, east)
            Parameters:
                north - a numpy array of northings\n
                east - a numpy array of eastings\n
                reflat - the reference latitude used to compute longitude
                and latitude conversion factors (unless specified the
                reference latitude associated with the calling SAR image
                object will be used)
            Returns:
                y_ind - a numpy array of y (vertical) indices\n
                x_ind - a numpy array of x (horizontal) indices
        """
        # if a reflat is not specified, then use the precalculated lat/lon conversions for the SAR image object
        if not reflat:
            latConv = self.latConv
            lonConv = self.lonConv
        else:
            # compute new longitude and latitude conversion factors based on the reflat passed in
            latConv,lonConv = computeLatLonConv(reflat)
        # calculate the range of northing values for the SAR image object data
        nrange = (self.paramGeo['north'] - self.paramGeo['south']) * latConv
        # calculate the range of easting values for the SAR image object data
        erange = (self.paramGeo['east'] - self.paramGeo['west']) * lonConv
        # the number of rows in the image
        y_range = self.data.shape[0] - 1
        # the number of columns in the image
        x_range = self.data.shape[1] - 1
        # compute the y indices based on the northings (note that the northings should be referenced
        # to the center of the pixels, hence the - 0.5 from the pixel indices)
        y = (self.paramGeo['north']*latConv - north)/nrange * y_range - 0.5
        # compute the x indices based on the eastings (note that the eastings should be referenced
        # to the center of the pixels, hence the - 0.5 from the pixel indices)
        x = (east - self.paramGeo['west']*lonConv)/erange * x_range - 0.5
        return y, x
        
    def calculateBalanceParameters(self, sar2):
        """
        Returns the phase offset and maybe the mean and standard deviation
        adjustments to balance the magnitude of the SAR image pairs.
            Method usage:
                atiBias, mu, sig = SARImage_object.ne2pix(sar2)
            Parameters:
                sar2 - this should be the aft SAR image of the interferometric
                SAR image pair
            Returns:
                atiBias - the ati phase offset from 0 deg (given in radians)
                mu - the correction factor for the mean of the magnitude 
                distribution
                sig - the correction factor for the variance of the magnitude
                distribution
        """
        # first and foremost we need to calculate the histogram of the smoothed ati phase
        
        # calculate the ati
        ati = self.iferSmooth(sar2)
        # calculate the coherence for masking purposes
        gamma = self.coherence(sar2,diam=5)
        coherT = 0.7
        mask = logical_and(logical_and(self.mag() > 1e-10, sar2.mag() > 1e-10), gamma.mag() > coherT)
        
        # layout the bins
        bins = linspace(-pi,pi,360)
        ati_hist,edges = histogram(angle(ati.data[mask]),bins)
        
        # grab the index of the max
        ind = nonzero(ati_hist==ati_hist.max())
        atiBias = (edges[ind[0][0]+1] + edges[ind[0][0]]) / 2
        
        # we can now compute the mean and variance corrections
        # now we need to convvert the magnitude of the SAR images to log space
        # to make their measurement distribution nice and Gaussian like
        selfLogVals = 10*log10(abs(self.data[mask]))
        sar2LogVals = 10*log10(abs(sar2.data[mask]))
        
        # calculate the means and standard deviations
        selfMean = selfLogVals.mean()
        selfStd = selfLogVals.std()
        sar2Mean = sar2LogVals.mean()
        sar2Std = sar2LogVals.std()
        
        # compute the mean and std calibration factors
        sig = selfStd / sar2Std
        mu = selfMean - (sar2Mean * sig)
        
        print "Correction factors are:\n\tatiBias = %0.2f\n\tmu = %0.3f\n\tsig = %0.3f" % (atiBias * 180/pi, mu, sig)        
        return atiBias.item(), mu.item(), sig.item()

def computeLatLonConv(reflat):
    """
    Returns the conversion factors from lat/lon to northing/easting
        Function usage:
            latConv, lonConv = computeLatLonConv(reflat)
        Parameters:
            reflat - this the reference latitude value you desire to use to
            compute the longitude and latitude conversion factors
        Return values:
            latConv - a multiplicative factor to compute northings from 
            latitude values\n
            lonConv - a multiplicative factor to compute eastings from
            longitude values
    """
    a = 6378137.0           # radius of the Earth at the equator
    f = 1.0/298.257223563   # flattening of the ellipsoid
    e = sqrt(f * (2 - f))   # eccentricity of the ellipsoid
    w = sqrt(1 - e**2*sin(reflat * pi/180)**2)
    Rm = a * (1 - e**2) / w**3
    Cm = 2*pi * Rm
    Rp = a * cos(reflat * pi/180)/w
    Cp = 2*pi * Rp
    latConv = Cm / 360.0
    lonConv = Cp / 360.0
    return latConv, lonConv
     
     
     
"""
FUNCTIONS
These are functions based on the same code written up above as SARImage class
methods.  They have just been appropriately changed to accept the first SAR
image object as another argument.  I am not oging to document and comment
them as well as I did above, because I already did it above.
"""
def ifer(data1,data2):
    """Returns the interferogram of the current SAR image and a 2nd SAR image"""
    if (data1.shape == data2.shape):
        interferogram = data1 * data2.conj()
    else:
        raise Exception("SAR images are not the same size")
    return interferogram
        
def coherence(data1,data2,diam=3):
    """Returns an image of the coherence between the current SAR image and a 2nd SAR image"""
    if (data1.shape == data2.shape):
        struct = ones((diam,diam)) # summing filter kernel          
        gamma = data1 * data2.conj()
        realg = convolve(gamma.real,struct,mode='constant')
        imagg = convolve(gamma.imag,struct,mode='constant')
        gamma = realg + 1j*imagg
        i1 = sqrt(convolve(abs(data1)**2,struct,mode='constant'))
        i2 = sqrt(convolve(abs(data2)**2,struct,mode='constant'))
        gamma = gamma / (i1 * i2)
        gamma[isnan(gamma)] = 0.0
    else:
        raise Exception("SAR image are not the same size")
    return gamma
    
def iferSmooth(data1,data2,sigma=None):
    """Returns the smoothed interferogram of the current SAR image and a 2nd SAR image"""
    interferogram = ifer(data1,data2)
    if interferogram:
        if sigma:
            reald = gaussian_filter(interferogram.real,sigma)
            imagd = gaussian_filter(interferogram.imag,sigma)
        else:
            # Formulate gaussian disk for 2D filter kernel
            kernel = zeros((7,7))
            kernel[3,3] = 1
            kernel = gaussian_filter(kernel,2)
            kernel[kernel<kernel[2,0]] = 0
            # Convolve the interferogram with the filter kernel
            reald = convolve(interferogram.real,kernel)
            imagd = convolve(interferogram.imag,kernel)
        # Recompose smoothed real and imaginary components
        interferogram = reald + 1j*imagd
    return interferogram
    
def smooth(data,sigma=None):
    """Smooths the SAR image data using a modified gaussian filter with std=2"""
    if (sigma):
        reald = gaussian_filter(data.real,sigma)
        imagd = gaussian_filter(data.imag,sigma)
    else:
        kernel = zeros((7,7))
        kernel[3,3] = 1
        kernel = gaussian_filter(kernel,2)
        kernel[kernel<kernel[2,0]] = 0
        # Convolve the interferogram with the filter kernel
        reald = convolve(data.real,kernel)
        imagd = convolve(data.imag,kernel)
    # Recompose smoothed real and imaginary components
    data = reald + 1j*imagd
    return data

def parse_xml(xmlObject, param={}):
    """Extract and return key/value pairs from xml objects.  This method accepts a DOM xml object"""
    # Loop through each node of the DOM xml object and assign the key/value pairs to the dictionary
    for node in xmlObject.childNodes:
        # if the type of node is an element node, then try to extract the key/value
        if node.nodeType == node.ELEMENT_NODE:
            # if it has a child node, then we will extract the data
            if node.hasChildNodes():
                # if the node value contains a letter A-Za-z then interpret it as a string
                # TODO: proper detection of scientific numbers ("e")
                if search('[A-Za-z]|:',node.childNodes[0].nodeValue):
                    param[str(node.nodeName)] = str(node.childNodes[0].nodeValue)
                # else interpret the node value as a float
                else:
                    param[str(node.nodeName)] = float(node.childNodes[0].nodeValue)
            # else, if there is not a child node, then assign it an empty list
            else:
                param[str(node.nodeName)] = []
    return param

if __name__ == "__main__":
    # create Qt application
    foSAR = SARImage('/data4/SciFly/DATA/RAW/20130227/SAR_02272013_185855_downloadLHF_121000')
    afSAR = SARImage('/data4/SciFly/DATA/RAW/20130227/SAR_02272013_185855_downloadLHA_121000')
    ati = foSAR.iferSmooth(afSAR)
    ati.showPhase()
