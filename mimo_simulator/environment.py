from re import search
from xml.dom.minidom import parse

import numpy as np
from tqdm import tqdm

import pandas as pd
from numpy.fft import fft
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.transform import Rotation as rot
from scipy.signal import fftconvolve, find_peaks
from scipy.signal.windows import taylor
import simplekml
from scipy.ndimage import rotate
from shapely.geometry import Polygon

from rawparser import loadASHFile, loadASIFile
from simlib import getElevationMap, getElevation, enu2llh

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


class Environment(object):

    def __init__(self, ashfile, asifile, size=(50, 50)):
        ash = loadASHFile(ashfile)
        nRows = int(ash['image']['nRows'])
        nCols = int(ash['image']['nCols'])
        llr = (ash['geo']['centerY'], ash['geo']['centerX'],
               getElevation((ash['geo']['centerY'], ash['geo']['centerX'])))
        asi = loadASIFile(asifile, nRows, nCols)
        rpsz = ash['geo']['rowPixelSizeM']
        cpsz = ash['geo']['colPixelSizeM']

        latlen = 111132.92 - 559.82 * np.cos(2 * llr[0] * DTR) + 1.175 * np.cos(4 * llr[0] * DTR) - .0023 * np.cos(
            6 * llr[0] * DTR)
        lonlen = 111412.84 * np.cos(llr[0] * DTR) - 93.5 * np.cos(3 * llr[0] * DTR) + .118 * np.cos(
            5 * llr[0] * DTR)
        # Find indexes into ASI file
        cta = ash['flight']['ctrackAngle'] * DTR
        R = np.array([[np.cos(cta), -np.sin(cta)],
                      [np.sin(cta), np.cos(cta)]])

        nRowPix = int(size[0] / rpsz / 2)
        nColPix = int(size[1] / cpsz / 2)
        grid_data = abs(asi[nRows // 2 - nRowPix:nRows // 2 + nRowPix, nCols // 2 - nColPix:nCols // 2 + nColPix])

        ge = (np.arange(grid_data.shape[0]) - grid_data.shape[0] / 2) * rpsz
        gn = (np.arange(grid_data.shape[1]) - grid_data.shape[1] / 2) * cpsz
        geg, gng = np.meshgrid(ge, gn)
        gu = np.zeros(geg.flatten().shape)
        glat, glon, galt = enu2llh(geg.flatten(), gng.flatten(), gu, llr)
        galt = getElevationMap(glat, glon)

        hfunc = lambda x, y: RectBivariateSpline(ge, gn, galt.reshape(grid_data.shape).T)(x, y, grid=False) - llr[2]
        dfunc = RectBivariateSpline(ge, gn, grid_data.T)

        self._scp = llr
        self._conv = (latlen, lonlen)
        self._hfunc = hfunc
        self._dfunc = dfunc
        self._data = grid_data
        self._bounds = Polygon
        self.shape = size

    def __call__(self, x, y, is_enu=True):
        if is_enu:
            return self._hfunc(x, y), self._dfunc(x, y, grid=False)
        else:
            x_conv = (self._scp[0] - x) / self._conv[0]
            y_conv = (self._scp[1] - y) / self._conv[1]
            return self._hfunc(x_conv, y_conv), self._dfunc(x_conv, y_conv, grid=False)

    @property
    def data(self):
        return self._data

    @property
    def scp(self):
        return self._scp
