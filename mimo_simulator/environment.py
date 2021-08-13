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
from scipy.ndimage import label
import simplekml
from scipy.ndimage import rotate
from shapely.geometry import Polygon, Point

from rawparser import loadASHFile, loadASIFile
from simlib import getElevationMap, getElevation, enu2llh

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


class Environment(object):

    def __init__(self, llr=None, bg_data=None, bg_res=None, cta=None, ashfile=None, asifile=None,
                 subgrid_size=(100, 100), dec_fac=4):
        """
        Init for environment. Creates data background and parameters for proper
            indexing, and gets elevation data.
        :param ashfile: Name of .ash file to use as indexing info. Needs {asifile} to work properly.
        :param asifile: Name of .asi file to use as background. Needs {ashfile} to work properly.
        :param subgrid_size: tuple. Size of grid to actually sample from, in meters.
        :param dec_fac: int. Decimation factor of .asi file, to reduce sampling/reading times.
        """
        if bg_data is not None:

            # Get conversions for lat/lon to ENU
            if llr is None:
                raise ValueError('Lat/Lon reference point is None.')
            latlen = 111132.92 - 559.82 * np.cos(2 * llr[0] * DTR) + 1.175 * np.cos(4 * llr[0] * DTR) - .0023 * np.cos(
                6 * llr[0] * DTR)
            lonlen = 111412.84 * np.cos(llr[0] * DTR) - 93.5 * np.cos(3 * llr[0] * DTR) + .118 * np.cos(
                5 * llr[0] * DTR)

            # Load in grid information
            if bg_res is None:
                bg_res = (subgrid_size[0] / bg_data.shape[0], subgrid_size[1] / bg_data.shape[1])
            grid_data = abs(bg_data)
            e_span = bg_data.shape[0] * bg_res[0]
            n_span = bg_data.shape[1] * bg_res[1]
            ogsize = grid_data.shape
            cta = 0
        elif ashfile is not None:
            # Use the .ash and .asi files to get background
            if asifile is None:
                raise ValueError('Specify both .asi file and .ash file.')
            ash = loadASHFile(ashfile)
            nRows = int(ash['image']['nRows'])
            nCols = int(ash['image']['nCols'])
            asi = loadASIFile(asifile, nRows, nCols)

            # Use LLR to get conversions from Lat/Lon to ENU
            if llr is None:
                llr = (ash['geo']['centerY'], ash['geo']['centerX'],
                       getElevation((ash['geo']['centerY'], ash['geo']['centerX'])))
            latlen = 111132.92 - 559.82 * np.cos(2 * llr[0] * DTR) + 1.175 * np.cos(4 * llr[0] * DTR) - .0023 * np.cos(
                6 * llr[0] * DTR)
            lonlen = 111412.84 * np.cos(llr[0] * DTR) - 93.5 * np.cos(3 * llr[0] * DTR) + .118 * np.cos(
                5 * llr[0] * DTR)

            # Load in grid data and resolution stuff
            cta = ash['flight']['ctrackAngle'] * DTR
            ogsize = (nCols, nRows)
            grid_data = np.fliplr(rotate(abs(asi), cta / DTR + 90))
            grid_data = grid_data[1:-1, 1:-1]
            e_span = abs(ash['geo']['west'] - ash['geo']['east']) * lonlen
            n_span = abs(ash['geo']['north'] - ash['geo']['south']) * latlen

        else:
            raise ValueError('Specify either a .ash/.asi file pair or background array values.')

        ge = np.linspace(-e_span / 2, e_span / 2, grid_data.shape[0])
        gn = np.linspace(-n_span / 2, n_span / 2, grid_data.shape[1])
        geg, gng = np.meshgrid(ge[::dec_fac], gn[::dec_fac])
        gu = np.zeros(geg.flatten().shape)
        glat, glon, galt = enu2llh(geg.flatten(), gng.flatten(), gu, llr)
        galt = getElevationMap(glat, glon)

        # All the member variables
        self._scp = llr
        self._conv = (latlen, lonlen)
        self._hdata = galt.reshape(geg.shape).T
        self._data = grid_data
        self._spans = (ge, gn)
        self._origin = (ge[0], gn[0])
        self._res = (ge[1] - ge[0], gn[1] - gn[0])
        self._dec = dec_fac
        self.data_shape = (e_span / 2, n_span / 2)
        self.shape = subgrid_size
        self.cta = cta
        self.rot_dist = np.sin(np.pi/2 - cta) * np.cos(np.pi/2 - cta) * ogsize[0]
        self.ogsize = ogsize

    def __call__(self, x, y, is_enu=True):
        """
        When called, Environment returns the background values for a given x/y pair.
        :param x: float. E or Lat value.
        :param y: float. N or Lon value.
        :param is_enu: bool. If True, assumes x/y are given in ENU.
        :return: Relative height and background value of x/y pair.
        """
        x_i = (x - self._scp[0]) * self._conv[0] if not is_enu else x
        y_i = (y - self._scp[1]) * self._conv[1] if not is_enu else y
        x_conv = (x_i - self._origin[0]) / self._res[0]
        y_conv = (y_i - self._origin[1]) / self._res[1]
        pts = np.array([c for c in zip(x_conv.flatten(), y_conv.flatten())])
        pts = pts.dot(np.array([[np.cos(-self.cta), -np.sin(-self.cta)], [np.sin(-self.cta), np.cos(-self.cta)]]))
        pts[pts[:, 1] < self.rot_dist, 1] += self.ogsize[1]
        pts[pts[:, 1] > self.rot_dist + self.ogsize[1], 1] -= self.ogsize[1]
        pts[pts[:, 0] < -self.ogsize[0]/2, 0] += self.ogsize[0]/2
        pts[pts[:, 0] > self.ogsize[0]/2, 0] -= self.ogsize[0] / 2
        pts = pts.dot(np.array([[np.cos(self.cta), -np.sin(self.cta)], [np.sin(self.cta), np.cos(self.cta)]]))
        x_conv = pts[:, 0].reshape(x_i.shape)
        y_conv = pts[:, 1].reshape(y_i.shape)
        x0 = np.round(pts[:, 0]).reshape(x_i.shape).astype(int)
        y0 = np.round(pts[:, 1]).reshape(y_i.shape).astype(int)
        x1 = x0 + np.sign(x_i - x0).astype(int)
        y1 = y0 + np.sign(y_i - y0).astype(int)

        # Make sure that the indexing values are valid
        x1[x1 >= self._data.shape[0]] = self._data.shape[0] - 1
        x1[x1 < 0] = 0
        x0[x0 >= self._data.shape[0]] = self._data.shape[0] - 1
        x0[x0 < 0] = 0

        # Similar process with the y values
        y1[y1 >= self._data.shape[1]] = self._data.shape[1] - 1
        y1[y1 < 0] = 0
        y0[y0 >= self._data.shape[1]] = self._data.shape[1] - 1
        y0[y0 < 0] = 0

        # Get differences
        xdiff = x_conv - x0
        ydiff = y_conv - y0

        # Get the actual values
        d_int = self._data[x1, y1] * xdiff * ydiff + self._data[x1, y0] * xdiff * (1 - ydiff) + self._data[x0, y1] * \
            (1 - xdiff) * ydiff + self._data[x0, y0] * (1 - xdiff) * (1 - ydiff)
        h_int = self._hdata[x1 // self._dec, y1 // self._dec] * xdiff * ydiff + \
            self._hdata[x1 // self._dec, y0 // self._dec] * xdiff * (1 - ydiff) + \
            self._hdata[x0 // self._dec, y1 // self._dec] * \
            (1 - xdiff) * ydiff + self._hdata[x0 // self._dec, y0 // self._dec] * (1 - xdiff) * (1 - ydiff)

        d_int
        return h_int - self._scp[2], d_int

    def getOffset(self, x, y, is_enu=True):
        """
        Gets the offset from the center of the .asi file of the given x/y pair.
        :param x: float. E or Lat value.
        :param y: float. N or Lon value.
        :param is_enu: bool. If True, assumes x/y are given in ENU.
        :return: Relative x/y from center of .asi file in ENU.
        """
        x_i = (x - self._scp[0]) * self._conv[0] if not is_enu else x
        y_i = (y - self._scp[1]) * self._conv[1] if not is_enu else y
        return x_i, y_i

    @property
    def data(self):
        return self._data

    @property
    def scp(self):
        return self._scp

    @property
    def hdata(self):
        return self._hdata
