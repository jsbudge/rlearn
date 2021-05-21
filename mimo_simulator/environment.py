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

    def __init__(self, ashfile, asifile, dec_fac=4):
        ash = loadASHFile(ashfile)
        nRows = int(ash['image']['nRows'])
        nCols = int(ash['image']['nCols'])
        llr = (ash['geo']['centerY'], ash['geo']['centerX'],
               getElevation((ash['geo']['centerY'], ash['geo']['centerX'])))
        asi = loadASIFile(asifile, nRows, nCols)

        latlen = 111132.92 - 559.82 * np.cos(2 * llr[0] * DTR) + 1.175 * np.cos(4 * llr[0] * DTR) - .0023 * np.cos(
            6 * llr[0] * DTR)
        lonlen = 111412.84 * np.cos(llr[0] * DTR) - 93.5 * np.cos(3 * llr[0] * DTR) + .118 * np.cos(
            5 * llr[0] * DTR)
        # Find indexes into ASI file
        cta = ash['flight']['ctrackAngle'] * DTR

        grid_data = rotate(abs(asi), -cta / DTR)
        e_span = abs(ash['geo']['west'] - ash['geo']['east']) * lonlen
        n_span = abs(ash['geo']['north'] - ash['geo']['south']) * latlen
        ge = np.linspace(-e_span / 2, e_span / 2, grid_data.shape[0])
        gn = np.linspace(-n_span / 2, n_span / 2, grid_data.shape[1])
        geg, gng = np.meshgrid(ge[::dec_fac], gn[::dec_fac])
        gu = np.zeros(geg.flatten().shape)
        glat, glon, galt = enu2llh(geg.flatten(), gng.flatten(), gu, llr)
        galt = getElevationMap(glat, glon)

        # hfunc = lambda x, y: RectBivariateSpline(ge, gn, galt.reshape(grid_data.shape).T)(x, y, grid=False) - llr[2]
        # dfunc = RectBivariateSpline(ge, gn, grid_data)

        self._scp = llr
        self._conv = (latlen, lonlen)
        self._hdata = galt.reshape(geg.shape)
        self._data = grid_data
        self._spans = (ge, gn)
        self._origin = (ge[0], gn[0])
        self._res = (ge[1] - ge[0], gn[1] - gn[0])
        self._dec = dec_fac
        self.shape = (e_span, n_span)

    def __call__(self, x, y, is_enu=True):
        x_i = (self._scp[0] - y) / self._conv[0] if not is_enu else x
        y_i = (self._scp[1] - x) / self._conv[1] if not is_enu else y
        try:
            x0 = int(np.round((x_i - self._origin[0]) / self._res[0]))
            y0 = int(np.round((y_i - self._origin[1]) / self._res[1]))
        except TypeError:
            x0 = np.round((x_i - self._origin[0]) / self._res[0]).astype(int)
            y0 = np.round((y_i - self._origin[1]) / self._res[1]).astype(int)
        x1 = x0 + np.sign(x_i - x0).astype(int)
        xdiff = (x_i - self._origin[0]) / self._res[0] - x0
        y1 = y0 + np.sign(y_i - y0).astype(int)
        ydiff = (y_i - self._origin[1]) / self._res[1] - y0
        d_int = self._data[x1, y1] * xdiff * ydiff + self._data[x1, y0] * xdiff * (1 - ydiff) + self._data[x0, y1] * \
                (1 - xdiff) * ydiff + self._data[x0, y0] * (1 - xdiff) * (1 - ydiff)
        h_int = self._hdata[x1 // self._dec, y1 // self._dec] * xdiff * ydiff + self._hdata[x1 // self._dec, y0 // self._dec] * xdiff * (1 - ydiff) + self._hdata[x0 // self._dec, y1 // self._dec] * \
                (1 - xdiff) * ydiff + self._hdata[x0 // self._dec, y0 // self._dec] * (1 - xdiff) * (1 - ydiff)
        return h_int - self._scp[2], d_int

    @property
    def data(self):
        return self._data

    @property
    def scp(self):
        return self._scp

    @property
    def hdata(self):
        return self._hdata
