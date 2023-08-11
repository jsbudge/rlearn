import contextlib
import numpy as np
from simulation_functions import getElevationMap, llh2enu, \
    enu2llh, getElevation, db, resampleGrid
from SDRParsing import SDRParse
from scipy.spatial.transform import Rotation as rot
from scipy.spatial import Delaunay
from scipy.signal import medfilt2d
from scipy.interpolate import CubicSpline, interpn, NearestNDInterpolator, LinearNDInterpolator
import pickle

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

'''
Environment
This is a class to represent the environment of a radar. 
'''


class Environment(object):
    _altgrid = None
    _transforms = None
    _refgrid = None

    def __init__(self, rmat=None, shift=None, reflectivity=None):
        if rmat is not None:
            self.setGrid(reflectivity, rmat, shift)

    def getDistance(self, pos):
        return np.linalg.norm(self._grid - pos[None, :], axis=1)

    def getGridParams(self, pos, width, height, npts, az=0):
        shift_x, shift_y, _ = llh2enu(*pos, self.ref)
        rmat = np.array([[np.cos(az), -np.sin(az)],
                         [np.sin(az), np.cos(az)]]).dot(np.diag([width / npts[0], height / npts[1]]))

        return rmat, np.array([shift_x, shift_y])

    def getGrid(self, pos=None, width=None, height=None, npts=None, az=0):
        if pos is None:
            x, y = np.meshgrid(np.arange(self.shape[0]) - self.shape[0] / 2,
                               np.arange(self.shape[1]) - self.shape[1] / 2)
            pts = np.column_stack((x.flatten(), y.flatten()))
            pos_r = np.squeeze(np.einsum('ji, mni -> jmn', self._transforms[0], [pts])).T + self._transforms[1]
            latg, long, altg = enu2llh(pos_r[:, 0], pos_r[:, 1], np.zeros(pos_r.shape[0]), self.ref)
            gz = (getElevationMap(latg, long, interp_method='splinef2d') - self.ref[2])
            sh = self.shape
        else:
            shift_x, shift_y, _ = llh2enu(*pos, self.ref)
            gxx = np.linspace(-width / 2, width / 2, npts[0])
            gyy = np.linspace(-height / 2, height / 2, npts[1])
            gx, gy = np.meshgrid(gxx, gyy)
            pts = np.column_stack((gx.flatten(), gy.flatten()))
            rmat = np.array([[np.cos(az), -np.sin(az)],
                             [np.sin(az), np.cos(az)]])
            pos_r = np.squeeze(np.einsum('ji, mni -> jmn', rmat, [pts])).T + \
                    np.array([shift_x, shift_y])
            latg, long, altg = enu2llh(pos_r[:, 0], pos_r[:, 1], np.zeros(pos_r.shape[0]), self.ref)
            sh = gx.shape
            gz = (getElevationMap(latg, long, interp_method='splinef2d') - self.ref[2])
        return pos_r[:, 0].reshape(sh), pos_r[:, 1].reshape(sh), gz.reshape(sh)

    def setGrid(self, newgrid, rmat, shift):
        self._refgrid = newgrid
        self._transforms = (rmat, shift)

    def resample(self, pos, width, height, npts, az=0):
        rmat, shift = self.getGridParams(pos, width, height, npts, az)
        x, y, _ = self.getGrid(pos, width, height, npts, az)
        pts = np.column_stack((x.flatten(), y.flatten()))
        irmat = np.linalg.pinv(self._transforms[0])
        pos_r = np.squeeze(np.einsum('ji, mni -> jmn', irmat, [pts - self._transforms[1]])).T + \
                np.array([self.shape[0] / 2, self.shape[1] / 2])
        self.setGrid(interpn((np.arange(self.refgrid.shape[0]),
                              np.arange(self.refgrid.shape[1])), self.refgrid, pos_r, bounds_error=False,
                             fill_value=0).reshape(x.shape), rmat, shift)

    def save(self, fnme):
        with open(fnme, 'wb') as f:
            pickle.dump(self, f)

    def getPos(self, px, py):
        return self._transforms[0].dot(np.array([px - self.shape[0] / 2, py - self.shape[1] / 2])) + self._transforms[1]

    def getIndex(self, x, y):
        irmat = np.linalg.pinv(self._transforms[0])
        return irmat.dot(np.array([x, y]) - self._transforms[1]) + np.array([self.shape[0] / 2, self.shape[1] / 2])

    def interp(self, x, y):
        return interpn((np.arange(self.refgrid.shape[0]),
                 np.arange(self.refgrid.shape[1])), self.refgrid, self.getIndex(x, y)).reshape(x.shape)

    @property
    def refgrid(self):
        return self._refgrid

    @property
    def shape(self):
        return self._refgrid.shape

    @property
    def transforms(self):
        return self._transforms


class MapEnvironment(Environment):

    def __init__(self, origin, extent, npts_background=(500, 500)):
        self.origin = origin
        self.ref = origin
        gp = getGridParams(origin, origin, extent[0], extent[1], npts_background)
        super().__init__(gp[1], gp[0], np.zeros(npts_background))


class SDREnvironment(Environment):
    rps = 1
    cps = 1
    heading = 0.

    def __init__(self, sdr_file, local_grid=None, origin=None):
        # Load in the SDR file
        sdr = SDRParse(sdr_file) if isinstance(sdr_file, str) else sdr_file
        grid = None
        print('SDR loaded')
        try:
            asi = sdr.loadASI(sdr.files['asi'])
        except KeyError:
            print('ASI not found.')
            asi = np.random.rand(2000, 2000)
            asi[250, 250] = 10
            asi[750, 750] = 10
            grid = asi
        except TypeError:
            asi = sdr.loadASI(sdr.files['asi'][0])
            grid = abs(asi)
        self._sdr = sdr
        self._asi = asi
        self.heading = -np.arctan2(sdr.gps_data['ve'].values[0], sdr.gps_data['vn'].values[0])
        if sdr.ash is None:
            try:
                hght = sdr.xml['Flight_Line']['Flight_Line_Altitude_M']
                pt = ((sdr.xml['Flight_Line']['Start_Latitude_D'] + sdr.xml['Flight_Line']['Stop_Latitude_D']) / 2,
                      (sdr.xml['Flight_Line']['Start_Longitude_D'] + sdr.xml['Flight_Line']['Stop_Longitude_D']) / 2)
                alt = getElevation(pt)
            except KeyError:
                alt = sdr.gps_data['alt'].mean()
                pt = (sdr.gps_data['lat'].mean(), sdr.gps_data['lon'].mean())
                hght = alt + getElevation(pt)
            mrange = hght / np.tan(sdr.ant[0].dep_ang)
            if origin is None:
                ref_llh = origin = enu2llh(mrange * np.sin(self.heading), mrange * np.cos(self.heading), 0.,
                                           (pt[0], pt[1], alt))
            else:
                ref_llh = enu2llh(mrange * np.sin(self.heading), mrange * np.cos(self.heading), 0.,
                                           (pt[0], pt[1], alt))
        else:
            if origin is None:
                origin = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                          getElevation((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'])))
            ref_llh = (sdr.ash['geo']['refLat'], sdr.ash['geo']['refLon'],
                       sdr.ash['geo']['hRef'])
            self.rps = sdr.ash['geo']['rowPixelSizeM']
            self.cps = sdr.ash['geo']['colPixelSizeM']
            self.heading = -sdr.ash['flight']['flnHdg'] * DTR

        self.origin = origin
        self.ref = ref_llh

        if local_grid is not None:
            grid = local_grid
        else:
            # Set grid to be one meter resolution
            '''rowup = int(1 / self.rps) if self.rps < 1 else 1
            colup = int(1 / self.cps) if self.cps < 1 else 1
            grid = grid[::rowup, ::colup]
            self.rps *= rowup
            self.cps *= colup'''

        rmat, shift = self.getGridParams(self.origin, grid.shape[0] * self.rps, grid.shape[1] * self.cps, grid.shape,
                                         self.heading)

        super().__init__(rmat=rmat, shift=shift, reflectivity=grid)

    @property
    def sdr(self):
        return self._sdr


def mesh(grid, tri_err, num_vertices):
    # Generate a mesh using SVS metrics to make triangles in the right spots
    ptx = grid[0, :, :].flatten()
    pty = grid[1, :, :].flatten()
    im = grid[2, :, :].flatten()
    pts = np.array([ptx, pty]).T

    # Initial points are the four corners of the grid
    init_pts = np.array([[ptx.min(), pty.min()],
                         [ptx.min(), pty.max()],
                         [ptx.max(), pty.min()],
                         [ptx.max(), pty.max()]])
    tri = Delaunay(init_pts, incremental=True, qhull_options='QJ')
    total_err = np.inf
    its = 0
    total_its = 20
    while total_err > tri_err and its < total_its:
        pts_tri = tri.find_simplex(pts).astype(int)
        sort_args = pts_tri.argsort()
        sorted_arr = pts_tri[sort_args]
        _, cut = np.unique(sorted_arr, return_index=True)
        out = np.split(sort_args, cut)
        total_err = 0
        add_pts = []
        for simplex in out:
            # Calculate out SVS error of triangle
            if len(simplex) > 1:
                tric = im[simplex].mean()
                errors = abs(im[simplex] - tric) ** 2
                if np.mean(errors) > tri_err:
                    winner = simplex[errors == errors.max()][0]
                    add_pts.append([ptx[winner], pty[winner]])
                    total_err += sum(errors)
                    if len(add_pts) + tri.points.shape[0] >= num_vertices:
                        its = total_its
                        break
        total_err /= len(pts_tri)
        if not add_pts:
            break
        try:
            tri.add_points(np.array(add_pts))
        except Exception:
            print('Something went wrong.')
            break
        its += 1
    ptx = tri.points[:, 0]
    pty = tri.points[:, 1]
    reflectivity = tri.find_simplex(pts).reshape((grid.shape[1], grid.shape[2]))
    return ptx, pty, reflectivity, tri.simplices


def getGridParams(ref, pos, width, height, npts, az=0):
    shift_x, shift_y, _ = llh2enu(*pos, ref)
    rmat = np.array([[np.cos(az), -np.sin(az)],
                     [np.sin(az), np.cos(az)]]).dot(np.diag([width / npts[0], height / npts[1]]))

    return (shift_x, shift_y), rmat

