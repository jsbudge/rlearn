import numpy as np
from osgeo import gdal
from scipy.interpolate import RectBivariateSpline, interpn
from scipy.spatial.transform import Rotation as rot
from itertools import product
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import plotly.io as pio
import plotly.graph_objects as go
import os
from functools import reduce
from numba import jit, prange

pio.renderers.default = 'browser'

WGS_A = 6378137.0
WGS_F = 1 / 298.257223563
WGS_B = 6356752.314245179
WGS_E2 = 6.69437999014e-3
c0 = 299792458.0
DTR = np.pi / 180


def getDTEDName(lat, lon):
    """Return the path and name of the dted to load for the given lat/lon"""
    tmplat = int(np.floor(lat))
    tmplon = int(np.floor(lon))
    direw = 'w' if tmplon < 0 else 'e'
    dirns = 's' if tmplat < 0 else 'n'
    if os.name == 'nt':
        return 'Z:\\dted\\%s%03d\\%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))
    else:
        return '/data5/dted/%s%03d/%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))


def detect_local_extrema(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = filters.minimum_filter(arr, footprint=neighborhood) == arr
    # local_max = filters.maximum_filter(arr, footprint=neighborhood) == arr
    background = arr == 0
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_extrema = local_min ^ eroded_background  # + local_max ^ eroded_background
    return np.where(detected_extrema)


def db(x):
    ret = abs(x)
    ret[ret < 1e-15] = 1e-15
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2 ** (np.ceil(np.log2(x))))


def undulationEGM96(lat, lon):
    if os.name == 'nt':
        egmdatfile = "Z:\\dted\\EGM96.DAT"
    else:
        egmdatfile = "/data5/dted/EGM96.DAT"
    with open(egmdatfile, "rb") as f:
        emg96 = np.fromfile(f, 'double', 1441 * 721, '')
        eg_n = np.ceil(lat / .25) * .25
        eg_s = np.floor(lat / .25) * .25
        eg_e = np.ceil(lon / .25) * .25
        eg_w = np.floor(lon / .25) * .25
        eg1 = emg96[((eg_w + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_n + 90 + .25) / .25 - 1).astype(int)]
        eg2 = emg96[((eg_w + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_s + 90 + .25) / .25 - 1).astype(int)]
        eg3 = emg96[((eg_e + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_n + 90 + .25) / .25 - 1).astype(int)]
        eg4 = emg96[((eg_e + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_s + 90 + .25) / .25 - 1).astype(int)]
        egc = (eg2 / ((eg_e - eg_w) * (eg_n - eg_s))) * (eg_e - lon) * (eg_n - lat) + \
              (eg4 / ((eg_e - eg_w) * (eg_n - eg_s))) * (lon - eg_w) * (eg_n - lat) + \
              (eg1 / ((eg_e - eg_w) * (eg_n - eg_s))) * (eg_e - lon) * (lat - eg_s) + \
              (eg3 / ((eg_e - eg_w) * (eg_n - eg_s))) * (lon - eg_w) * (lat - eg_s)
    return egc


def getElevationMap(lats, lons, und=True, interp_method='linear'):
    """Returns the digital elevation for a latitude and longitude"""
    # First, check to see if multiple DTEDs are needed
    floor_lats = np.floor(lats)
    floor_lons = np.floor(lons)
    hght = np.zeros_like(lats)
    dteds = list(product(np.unique(np.floor(lats)), np.unique(np.floor(lons))))
    for ted in dteds:
        idx = np.logical_and(floor_lats == ted[0], floor_lons == ted[1])
        dtedName = getDTEDName(*ted)

        # open DTED file for reading
        ds = gdal.Open(dtedName)

        # grab geo transform with resolutions
        gt = ds.GetGeoTransform()

        # read in raster data
        raster = ds.GetRasterBand(1).ReadAsArray()

        # Get lats and lons as bins into raster
        bin_lat = (lats[idx] - gt[3]) / gt[-1]
        bin_lon = (lons[idx] - gt[0]) / gt[1]

        # Linear interpolation using bins
        # Supported interp methods are linear, nearest, slinear, cubic, quintic, pchip, splinef2d
        hght[idx] = interpn(np.array([np.arange(3601), np.arange(3601)]), raster, np.array([bin_lat, bin_lon]).T,
                       method=interp_method, bounds_error=False, fill_value=0) + undulationEGM96(lats[idx], lons[idx]) if und else hght

    return hght


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bilinear_interpolation(x_in, y_in, f_in, x_out, y_out):
    f_out = np.zeros((y_out.size, x_out.size))

    for i in prange(f_out.shape[1]):
        idx = np.searchsorted(x_in, x_out[i])

        x1 = x_in[idx - 1]
        x2 = x_in[idx]
        x = x_out[i]

        for j in prange(f_out.shape[0]):
            idy = np.searchsorted(y_in, y_out[j])
            y1 = y_in[idy - 1]
            y2 = y_in[idy]
            y = y_out[j]

            f11 = f_in[idy - 1, idx - 1]
            f21 = f_in[idy - 1, idx]
            f12 = f_in[idy, idx - 1]
            f22 = f_in[idy, idx]

            f_out[j, i] = ((f11 * (x2 - x) * (y2 - y) +
                            f21 * (x - x1) * (y2 - y) +
                            f12 * (x2 - x) * (y - y1) +
                            f22 * (x - x1) * (y - y1)) /
                           ((x2 - x1) * (y2 - y1)))

    return f_out


def getElevation(pt, und=True):
    lat = pt[0]
    lon = pt[1]
    """Returns the digital elevation for a latitude and longitude"""
    dtedName = getDTEDName(lat, lon)
    gdal.UseExceptions()
    # open DTED file for reading
    ds = gdal.Open(dtedName)

    # get the geo transform info for the dted
    # ulx is upper left corner longitude
    # xres is the resolution in the x-direction (in degrees/sample)
    # xskew is useless (0.0)
    # uly is the upper left corner latitude
    # yskew is useless (0.0)
    # yres is the resolution in the y-direction (in degrees/sample)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    # calculate the x and y indices into the DTED data for the lat/lon
    px = int((lon - ulx) / xres) - (1 if (lon - ulx) / xres % 1 < .5 else 0)
    py = int((lat - uly) / yres) - (1 if (lat - uly) / yres % 1 < .5 else 0)

    # only if these x and y indices are within the bounds of the DTED, get the
    # raster band and try to read in the DTED values
    elevation = -1e20
    if (0 <= px < ds.RasterXSize) and (0 <= py < ds.RasterYSize):
        rasterBand = ds.GetRasterBand(1)
        dtedData = rasterBand.ReadAsArray(px, py, 2, 2)

        # use bilinear interpolation to get the elevation for the lat/lon
        x = (lon - ulx)
        y = (lat - uly)
        x1 = int((lon - ulx) / xres) * xres
        x2 = int((lon - ulx) / xres + 1) * xres
        y1 = int((lat - uly) / yres) * yres
        y2 = int((lat - uly) / yres + 1) * yres
        elevation = 1 / ((x2 - x1) * (y2 - y1)) * \
                    dtedData.ravel().dot(np.array([[x2 * y2, -y2, -x2, 1],
                                                    [-x2 * y1, y1, x2, -1],
                                                    [-x1 * y2, y2, x1, -1],
                                                    [x1 * y1, -y1, -x1, 1]])).dot(np.array([1, x, y, x * y]))

    return elevation + undulationEGM96(lat, lon) if und else elevation


def llh2enu(lat, lon, h, refllh):
    ecef = llh2ecef(lat, lon, h)
    return ecef2enu(*ecef, refllh)


def enu2llh(e, n, u, refllh):
    ecef = enu2ecef(e, n, u, refllh)
    return ecef2llh(*ecef)


def enu2ecef(e, n, u, refllh):
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    enu = np.array([e, n, u])
    tmp_rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                        [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                        [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]]).T
    if len(enu.shape) > 1:
        sz = np.ones((enu.shape[1],))
        ecef = tmp_rot.dot(enu) + np.array([sz * rx, sz * ry, sz * rz])
    else:
        ecef = tmp_rot.dot(enu) + np.array([rx, ry, rz])
    return ecef[0], ecef[1], ecef[2]


def llh2ecef(lat, lon, h):
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y, Z
    given the Geodetic Coordinates lat, lon + Ellipsoid Height h
    """
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180
    N = WGS_A / np.sqrt(1 - WGS_E2 * np.sin(lat_rad) ** 2)
    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (WGS_B ** 2 / WGS_A ** 2 * N + h) * np.sin(lat_rad)
    return X, Y, Z


def ecef2llh(x, y, z):
    # This is the Heikkinen application of the Ferrari solution to Bowring's irrational
    # geodetic-latitude equation to get a geodetic latitude and height.
    # Longitude remains the same between the two.
    r = np.sqrt(x ** 2 + y ** 2)
    ep2 = (WGS_A ** 2 - WGS_B ** 2) / WGS_B ** 2
    F = 54 * WGS_B ** 2 * z ** 2
    G = r ** 2 + (1 - WGS_E2) * z ** 2 - WGS_E2 * (WGS_A ** 2 - WGS_B ** 2)
    c = WGS_E2 ** 2 * F * r ** 2 / G ** 3
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G ** 2)
    Q = np.sqrt(1 + 2 * WGS_E2 ** 2 * P)
    r0 = -P * WGS_E2 * r / (1 + Q) + np.sqrt(
        1 / 2 * WGS_A ** 2 * (1 + 1 / Q) - P * (1 - WGS_E2) * z ** 2 / (Q * (1 + Q)) - 1 / 2 * P * r ** 2)
    U = np.sqrt((r - WGS_E2 * r0) ** 2 + z ** 2)
    V = np.sqrt((r - WGS_E2 * r0) ** 2 + (1 - WGS_E2) * z ** 2)
    z0 = WGS_B ** 2 * z / (WGS_A * V)
    h = U * (1 - WGS_B ** 2 / (WGS_A * V))
    lat = np.arctan((z + ep2 * z0) / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon, h


def ecef2enu(x, y, z, refllh):
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                    [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                    [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]])
    enu = rot.dot(np.array([x - rx, y - ry, z - rz]))
    return enu[0], enu[1], enu[2]


def getMapLocation(p1, extent, init_llh, npts_background=500, resample=True):
    pt_enu = llh2enu(*p1, init_llh)
    lats = np.linspace(p1[0] - extent[0] / 2 / 111111, p1[0] + extent[0] / 2 / 111111, npts_background)
    lons = np.linspace(p1[1] - extent[1] / 2 / 111111, p1[1] + extent[1] / 2 / 111111, npts_background)
    lt, ln = np.meshgrid(lats, lons)
    ltp = lt.flatten()
    lnp = ln.flatten()
    e, n, u = llh2enu(ltp, lnp, getElevationMap(ltp, lnp), init_llh)
    if resample:
        nlat, nlon, nh = resampleGrid(u.reshape(lt.shape), lats, lons, int(len(u) * .8))
        e, n, u = llh2enu(nlat, nlon, nh + init_llh[2], init_llh)
    point_cloud = np.array([e, n, u]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd = pcd.translate(pt_enu)
    return pcd


def resampleGrid(grid, x, y, npts):
    gxx, gyy = np.gradient(grid / grid.max())
    gx = RectBivariateSpline(np.arange(len(x)), np.arange(len(y)), gxx)
    gy = RectBivariateSpline(np.arange(len(x)), np.arange(len(y)), gyy)
    ptx = np.random.uniform(0, len(x) - 1, npts)
    pty = np.random.uniform(0, len(y) - 1, npts)
    for n in range(4):
        dx = gx(ptx, pty, grid=False) * 1e3 / (n + 1)
        dy = gy(ptx, pty, grid=False) * 1e3 / (n + 1)
        ptx += dx
        pty += dy
        ptx[ptx > len(x) - 1] = np.random.uniform(0, len(x) - 1, sum(ptx > len(x) - 1))
        pty[pty > len(y) - 1] = np.random.uniform(0, len(y) - 1, sum(pty > len(y) - 1))
    finalgrid = RectBivariateSpline(np.arange(len(x)), np.arange(len(y)), grid)
    return np.interp(ptx, np.arange(len(x)), x), np.interp(pty, np.arange(len(y)), y), finalgrid(ptx, pty, grid=False)


def createMeshFromPoints(pcd):
    pcd = pcd.voxel_down_sample(voxel_size=np.mean(pcd.compute_nearest_neighbor_distance()) / 1.5)
    its = 0
    while np.std(pcd.compute_nearest_neighbor_distance()) > 2. and its < 30:
        dists = pcd.compute_nearest_neighbor_distance()
        pcd = pcd.voxel_down_sample(voxel_size=np.mean(dists) / 1.5)
        its += 1

    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    radius = 3 * avg_dist
    radii = [radius, radius * 2]
    pcd.estimate_normals()
    try:
        pcd.orient_normals_consistent_tangent_plane(100)
    except RuntimeError:
        pass

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    # rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    rec_mesh.remove_duplicated_vertices()
    rec_mesh.remove_duplicated_triangles()
    rec_mesh.remove_degenerate_triangles()
    rec_mesh.remove_unreferenced_vertices()
    return rec_mesh


def genPulse(phase_x, phase_y, nnr, nfs, nfc, bandw):
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * 1 / nfs))


def rotate(az, nel, rot_mat):
    return rot.from_euler('zx', [[-az, 0.],
                                 [0., nel - np.pi / 2]]).apply(rot_mat)


def azelToVec(az, el):
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])


def hornPattern(fc, width, height, theta=None, phi=None, deg_per_bin=.5, az_only=False):
    _lambda = c0 / fc
    d = _lambda / 2.
    if theta is None:
        theta = np.arange(0, np.pi, deg_per_bin * DTR)
    if phi is None:
        phi = [0] if az_only else np.arange(-np.pi / 2, np.pi / 2, deg_per_bin * DTR)
    theta, phi = np.meshgrid(theta, phi)
    lcw = np.arange(-width / 2, width / 2, d)
    lch = np.arange(-height / 2, height / 2, d)
    lch, lcw = np.meshgrid(lch, lcw)
    lchm = lch.flatten()
    lcwm = lcw.flatten()
    k = 2 * np.pi / _lambda
    locs = np.array([lcwm, np.zeros_like(lcwm), lchm]).T
    ublock = -azelToVec(theta.flatten(), phi.flatten())
    AF = np.sum(np.exp(-1j * k * locs.dot(ublock)), axis=0)
    AF = AF.flatten() if az_only else AF.reshape(theta.shape)

    # Return degree array and antenna pattern
    return theta, phi, AF


def calcSNR(p_s, ant_g, az, el, wavelength, pulse_time, bandw, dopp_mult, rng, rcs_val=None):
    sig = calcPower(p_s, ant_g, az, el, wavelength, pulse_time, bandw, dopp_mult, rng, rcs_val)
    # Boltzmans constant
    kb = 1.3806503e-23
    # the reference temperature (in Kelvin)
    T0 = 290.0
    # Noise figure of 3dB
    F = 10.0 ** (3.0 / 10.0)
    N0 = kb * T0 * F
    sigma_n = np.sqrt(N0 * bandw)
    noise = sigma_n ** 2
    return db(np.array([sig / noise]))[0]


def calcPower(p_s, ant_g, az, el, wavelength, pulse_time, bandw, dopp_mult, rng, rcs_val=None):
    if rcs_val is None:
        sig = p_s * ant_g ** 2 * np.tan(az) * np.tan(el) * wavelength ** 2 * pulse_time * bandw * dopp_mult
        sig /= (4 * np.pi) ** 3 * rng ** 2
    else:
        sig = p_s * ant_g ** 2 * rcs_val * wavelength ** 2 * pulse_time * bandw * dopp_mult
        sig /= (4 * np.pi) ** 3 * rng ** 4
    return sig


def arrayFactor(fc, pos, theta=None, phi=None, weights=None, deg_per_bin=.5, az_only=False, horn_dim=None,
                horn_pattern=None):
    use_pat = False
    if horn_pattern is not None:
        _, _, el_pat = horn_pattern
        use_pat = True
    elif horn_dim is not None:
        use_pat = True
        _, _, el_pat = hornPattern(fc, horn_dim[0], horn_dim[1], theta=theta, phi=phi,
                                   deg_per_bin=deg_per_bin, az_only=az_only)
    _lambda = c0 / fc
    if theta is None:
        theta = np.arange(0, np.pi, deg_per_bin * DTR)
    if phi is None:
        phi = [0] if az_only else np.arange(-np.pi / 2, np.pi / 2, deg_per_bin * DTR)
    theta, phi = np.meshgrid(theta, phi)
    k = 2 * np.pi / _lambda
    # az, el = np.meshgrid(theta, theta)
    ublock = -azelToVec(theta.flatten(), phi.flatten())
    AF = np.exp(-1j * k * pos.dot(ublock))
    if use_pat:
        AF *= el_pat.flatten()[None, :]
    weights = weights if weights is not None else np.ones(pos.shape[0])
    AF = AF.T.dot(weights).flatten() if az_only else AF.T.dot(weights).reshape(theta.shape)
    # Return degree array and antenna pattern
    return theta, phi, AF


'''
------------------------------------------------------------------------------------
------------------- RENDERING AND PLOTTING FUNCTIONS -------------------------------
------------------------------------------------------------------------------------
'''


def createFigureDict(xrngs=None, yrngs=None, zrngs=None):
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    figure['layout']['scene'] = dict(
        xaxis=dict(range=xrngs, autorange=False),
        yaxis=dict(range=yrngs, autorange=False),
        zaxis=dict(range=zrngs, autorange=False),
        aspectratio=dict(x=1, y=1, z=1))
    figure['layout']['updatemenus'] = [dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None]),
                 {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                  "label": "Pause",
                  "method": "animate"}
                 ])]
    sliders_dict = {
        'active': 0, 'yanchor': 'top', 'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Time: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 0},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }
    return figure, sliders_dict


class PlotWithSliders(object):

    def __init__(self, bg=None, ntraces=1):
        self._frames = []
        if bg is None:
            self._fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(color="red", size=10)))
            self._traces = [0]
        else:
            self._fig = go.Figure(data=[go.Scatter3d(x=bg[:, 0], y=bg[:, 1], z=bg[:, 2], mode="markers",
                                                     marker=dict(color="red", size=10))] +
                                       [go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                                     marker=dict(color="red", size=10)) for _ in range(ntraces)])
            self._traces = list(range(1, ntraces + 1))

    def addFrame(self, fdata, trace=None, args=None):
        t = [trace] if trace is not None else [self._traces[0]]
        frame_args = dict(data=[go.Scatter3d(x=fdata[:, 0], y=fdata[:, 1], z=fdata[:, 2], mode='lines+markers')],
                                     traces=t, name=f'frame {len(self._frames) + 1}')
        if args is not None:
            for key, val in args.items():
                frame_args[key] = val
        self._frames.append(go.Frame(**frame_args))

    def render(self):
        args = {"frame": {"duration": 0}, "mode": "immediate", "fromcurrent": True,
                "transition": {"duration": 0, "easing": "linear"}, }
        slider = [{"pad": {"b": 1, "t": 1}, "len": 0.9, "x": 0.1, "y": 0,
                   "steps": [{"args": [[f.name], args],
                              "label": str(k), "method": "animate", } for k, f in enumerate(self._fig.frames)]}]

        update_menus = [{"buttons": [{'args': [None, args], 'label': 'Play', 'method': 'animate'},
                            {'args': [[None], args], 'label': 'Pause', 'method': 'animate'}],
                         "direction": "left", "pad": {"r": 10, "t": 1}, "type": "buttons", "x": 0.1, "y": 0, }]
        self._fig.update(frames=self._frames)
        self._fig.update_layout(updatemenus=update_menus, sliders=slider)
        self._fig.show()


'''
--------------------------------------DEBUG DATA PARSER STUFF-----------------------------------------------------------
'''


def loadRawData(filename, num_pulses, start_pulse=0):
    with open(filename, 'rb') as fid:
        num_frames = np.fromfile(fid, 'uint32', 1, '')[0]
        if start_pulse + num_pulses > num_frames:
            num_pulses = num_frames - start_pulse
            print(f"Too many frames for file! Using {num_pulses} pulses instead")
        num_samples = np.fromfile(fid, 'uint16', 1, '')[0]
        attenuation = np.fromfile(fid, 'uint8', num_frames, '')
        sys_time = np.fromfile(fid, 'double', num_frames, '')
        raw_data = np.zeros((num_samples, num_pulses)).astype(np.int16)
        fid.seek(fid.tell() + start_pulse * 2 * num_samples)
        for i in range(num_pulses):
            raw_data[:, i] = np.fromfile(fid, 'int16', num_samples, '') * 10 ** (attenuation[start_pulse + i] / 20)
    return raw_data, num_pulses, attenuation, sys_time


def getRawDataGen(filename, num_pulses, num_desired_frames=None, start_pulse=0, isIQ=False):
    with open(filename, 'rb') as fid:
        num_frames = np.fromfile(fid, 'uint32', 1, '')[0]
        if isIQ:
            num_samples = np.fromfile(fid, 'uint32', 1, '')[0]
        else:
            num_samples = np.fromfile(fid, 'uint16', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', num_frames, '')
        sys_time = np.fromfile(fid, 'double', num_frames, '')
        ndf = num_frames if num_desired_frames is None else num_desired_frames
        if isIQ:
            fid.seek(fid.tell() + start_pulse * 2 * num_samples)
            for npulse in range(0, ndf, num_pulses):
                proc_pulses = num_pulses if npulse + num_pulses < ndf else ndf - npulse
                raw_data = np.zeros((num_samples, proc_pulses)).astype(np.complex128)
                pulse_range = np.arange(npulse + start_pulse, npulse + proc_pulses + start_pulse)
                for i in range(proc_pulses):
                    tmp = np.fromfile(fid, 'int16', num_samples * 2, '')
                    raw_data[:, i] = (tmp[0::2] + 1j * tmp[1::2]) * 10 ** (attenuation[pulse_range[0] + i] / 20)
                yield raw_data, pulse_range, attenuation[pulse_range], sys_time[pulse_range]
        else:
            fid.seek(fid.tell() + start_pulse * 2 * num_samples)
            for npulse in range(0, ndf, num_pulses):
                proc_pulses = num_pulses if npulse + num_pulses < ndf else ndf - npulse
                raw_data = np.zeros((num_samples, proc_pulses)).astype(np.int16)
                pulse_range = np.arange(npulse + start_pulse, npulse + proc_pulses + start_pulse)
                for i in range(proc_pulses):
                    raw_data[:, i] = np.fromfile(fid, 'int16', num_samples, '') * 10 ** (
                            attenuation[pulse_range[0] + i] / 20)
                yield raw_data, pulse_range, attenuation[pulse_range], sys_time[pulse_range]


def getRawData(filename, num_pulses, start_pulse=0, isIQ=False):
    """
    Parses raw data from an APS debug .dat file.
    :param filename: str Name of .dat file to parse.
    :param num_pulses: int Number of pulses to parse.
    :param start_pulse: int The function will start with this pulse number.
    :param isIQ: bool if True, assumes data is stored as complex numbers. Otherwise, reads data as ints.
    :return:
        raw_data: numpy array Array of pulse data, size of number_samples_per_pulse x num_pulses.
        pulse_range: numpy array List of each pulse's number in the parsed file.
        attenuation: numpy array List of attenuation factors associated with each pulse.
        sys_time: numpy array List of system times, in TAC, associated with each pulse.
    """
    with open(filename, 'rb') as fid:
        num_frames = np.fromfile(fid, 'uint32', 1, '')[0]
        if isIQ:
            num_samples = np.fromfile(fid, 'uint32', 1, '')[0]
        else:
            num_samples = np.fromfile(fid, 'uint16', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', num_frames, '')
        sys_time = np.fromfile(fid, 'double', num_frames, '')
        proc_pulses = num_pulses if start_pulse + num_pulses < num_frames else num_frames - start_pulse
        pulse_range = np.arange(start_pulse, start_pulse + proc_pulses)
        if isIQ:
            fid.seek(fid.tell() + start_pulse * 4 * num_samples)
            raw_data = np.zeros((num_samples, proc_pulses)).astype(np.complex128)
            for i in range(proc_pulses):
                tmp = np.fromfile(fid, 'int16', num_samples * 2, '')
                raw_data[:, i] = (tmp[0::2] + 1j * tmp[1::2]) * 10 ** (attenuation[start_pulse + i] / 20)
            return raw_data, pulse_range, attenuation[pulse_range], sys_time[pulse_range]
        else:
            fid.seek(fid.tell() + start_pulse * 2 * num_samples)
            raw_data = np.zeros((num_samples, proc_pulses)).astype(np.int16)
            for i in range(proc_pulses):
                raw_data[:, i] = np.fromfile(fid, 'int16', num_samples, '') * 10 ** (
                        attenuation[start_pulse + i] / 20)
            return raw_data, pulse_range, attenuation[pulse_range], sys_time[pulse_range]


def getFullRawData(filename, num_pulses, start_pulse=0, isIQ=False):
    """
    Parses raw data from an APS debug .dat file.
    :param filename: str Name of .dat file to parse.
    :param num_pulses: int Number of pulses to parse.
    :param start_pulse: int The function will start with this pulse number.
    :param isIQ: bool if True, assumes data is stored as complex numbers. Otherwise, reads data as ints.
    :return:
        raw_data: numpy array Array of pulse data, size of number_samples_per_pulse x num_pulses.
        pulse_range: numpy array List of each pulse's number in the parsed file.
        attenuation: numpy array List of attenuation factors associated with each pulse.
        sys_time: numpy array List of system times, in TAC, associated with each pulse.
    """
    with open(filename, 'rb') as fid:
        num_frames = np.fromfile(fid, 'uint32', 1, '')[0]
        if isIQ:
            num_samples = np.fromfile(fid, 'uint32', 1, '')[0]
        else:
            num_samples = np.fromfile(fid, 'uint16', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', num_frames, '')
        sys_time = np.fromfile(fid, 'double', num_frames, '')
        proc_pulses = num_pulses if start_pulse + num_pulses < num_frames else num_frames - start_pulse
        pulse_range = np.arange(start_pulse, start_pulse + proc_pulses)
        if isIQ:
            fid.seek(fid.tell() + start_pulse * 8 * num_samples)
            raw_data = np.zeros((num_samples, proc_pulses)).astype(np.complex128)
            for i in range(proc_pulses):
                tmp = np.fromfile(fid, 'complex128', num_samples, '')
                raw_data[:, i] = tmp
        else:
            fid.seek(fid.tell() + start_pulse * 2 * num_samples)
            raw_data = np.zeros((num_samples, proc_pulses)).astype(np.int16)
            for i in range(proc_pulses):
                raw_data[:, i] = np.fromfile(fid, 'int16', num_samples, '') * 10 ** (
                        attenuation[start_pulse + i] / 20)

        return raw_data, pulse_range, attenuation[pulse_range], sys_time[pulse_range]


def loadDechirpRawData(filename, num_pulses, start_pulse=0):
    with open(filename, 'rb') as fid:
        num_frames = np.fromfile(fid, 'uint32', 1, '')[0]
        if start_pulse + num_pulses > num_frames:
            num_pulses = num_frames - start_pulse
            print(f"Too many frames for file! Using {num_pulses} pulses instead")
        num_samples = np.fromfile(fid, 'uint32', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', num_frames, '')
        sys_time = np.fromfile(fid, 'double', num_frames, '')
        raw_data = np.zeros((num_samples, num_pulses)).astype(np.complex64)
        fid.seek(fid.tell() + start_pulse * 2 * num_samples)
        for i in range(num_pulses):
            raw_data[:, i] = (np.fromfile(fid, 'int16', num_samples, '') + 1j * np.fromfile(fid, 'int16', num_samples,
                                                                                            '')) * 10 ** (
                                     attenuation[i] / 20)
    return raw_data, num_pulses, attenuation, sys_time


def factors(n):
    return list(set(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0))))


def getDechirpRawDataGen(filename, numPulses, numDesiredFrames=None, start_pulse=0):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        numSamples = np.fromfile(fid, 'uint32', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', numFrames, '')
        sys_time = np.fromfile(fid, 'double', numFrames, '')
        ndf = numFrames if numDesiredFrames is None else numDesiredFrames
        fid.seek(fid.tell() + start_pulse * 2 * numSamples)
        for npulse in range(0, ndf, numPulses):
            proc_pulses = numPulses if npulse + numPulses < ndf else ndf - npulse
            raw_data = np.zeros((numSamples, proc_pulses)).astype(np.complex128)
            pulseRange = np.arange(npulse, npulse + proc_pulses)
            for i in range(proc_pulses):
                raw_data[:, i] = (np.fromfile(fid, 'int16', numSamples, '') + 1j * np.fromfile(fid, 'int16', numSamples,
                                                                                               '')) * 10 ** (
                                         attenuation[pulseRange[0] + i] / 20)
            yield raw_data, pulseRange, attenuation[pulseRange], sys_time[pulseRange]


def loadFFTData(filename, numPulses, start_pulse=0):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        if start_pulse + numPulses > numFrames:
            numPulses = numFrames - start_pulse
            print(f"Too many frames for file! Using {numPulses} pulses instead")
        numFFTSamples = np.fromfile(fid, 'uint32', 1, '')[0]
        FFTData = np.zeros((numFFTSamples, numPulses)).astype(np.complex64)
        fid.seek(fid.tell() + start_pulse * 8 * numFFTSamples)
        for i in range(numPulses):
            FFTData[:, i] = np.fromfile(fid, 'complex64', numFFTSamples, '')
    return FFTData, numPulses


def getFFTDataGen(filename, numPulses, numDesiredFrames=None, start_pulse=0):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        numFFTSamples = np.fromfile(fid, 'uint32', 1, '')[0]
        ndf = numFrames if numDesiredFrames is None else numDesiredFrames
        fid.seek(fid.tell() + start_pulse * 8 * numFFTSamples)
        for npulse in range(0, ndf, numPulses):
            proc_pulses = numPulses if npulse + numPulses < ndf else ndf - npulse
            FFTdata = np.zeros((numFFTSamples, proc_pulses)).astype(np.complex64)
            pulseRange = np.arange(npulse + start_pulse, npulse + proc_pulses + start_pulse)
            for i in range(proc_pulses):
                FFTdata[:, i] = np.fromfile(fid, 'complex64', numFFTSamples, '')
            yield FFTdata, pulseRange


def getSinglePulse(filename, pulse):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        numSamples = np.fromfile(fid, 'uint32', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', numFrames, '')
        sys_time = np.fromfile(fid, 'double', numFrames, '')
        fid.seek(fid.tell() + pulse * 2 * numSamples)
        raw_data = np.fromfile(fid, 'int16', numSamples, '') * 10 ** (attenuation[pulse] / 20)
    return raw_data, attenuation[pulse], sys_time[pulse]


def getSingleFFTPulse(filename, pulse):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        numFFTSamples = np.fromfile(fid, 'uint32', 1, '')[0]
        fid.seek(fid.tell() + pulse * 8 * numFFTSamples)
        FFTData = np.fromfile(fid, 'complex64', numFFTSamples, '')
    return FFTData


"""
GPS data format:
First 32-bits: The number of INS frames (N)
N * sizeof( double ): latitude in degrees
N * sizeof( double ): longitude in degrees
N * sizeof( double ): altitude in meters
N * sizeof( double ): north velocity in meters per second
N * sizeof( double ): east velocity in meters per second 
N * sizeof( double ): up velocity in meters per second 
N * sizeof( double ): roll in radians
N * sizeof( double ): pitch in radians
N * sizeof( double ): azimuth component in radians
N * sizeof( double ): The number of milliseconds since the start of the GPS week, unwrapped
"""


def loadGPSData(filename):
    fid = open(filename, 'rb')
    numFrames = np.fromfile(fid, 'int32', 1, '')[0]
    return {'frames': numFrames, 'lat': np.fromfile(fid, 'float64', numFrames, ''),
            'lon': np.fromfile(fid, 'float64', numFrames, ''), 'alt': np.fromfile(fid, 'float64', numFrames, ''),
            'vn': np.fromfile(fid, 'float64', numFrames, ''), 've': np.fromfile(fid, 'float64', numFrames, ''),
            'vu': np.fromfile(fid, 'float64', numFrames, ''), 'r': np.fromfile(fid, 'float64', numFrames, ''),
            'p': np.fromfile(fid, 'float64', numFrames, ''), 'azimuthX': np.fromfile(fid, 'float64', numFrames, ''),
            'azimuthY': np.fromfile(fid, 'float64', numFrames, ''),
            'gps_ms': np.fromfile(fid, 'float64', numFrames, ''), 'systime': np.fromfile(fid, 'float64', numFrames, '')}


def loadPreCorrectionsGPSData(fnme):
    # open the post-corrections
    with open(fnme, 'rb') as fid:
        # parse all of the post-correction data
        numFrames = int(np.fromfile(fid, 'uint32', 1, '')[0])
        lat = np.fromfile(fid, 'float64', numFrames, '')
        lon = np.fromfile(fid, 'float64', numFrames, '')
        alt = np.fromfile(fid, 'float64', numFrames, '')
        vn = np.fromfile(fid, 'float64', numFrames, '')
        ve = np.fromfile(fid, 'float64', numFrames, '')
        vu = np.fromfile(fid, 'float64', numFrames, '')
        r = np.fromfile(fid, 'float64', numFrames, '')
        p = np.fromfile(fid, 'float64', numFrames, '')
        az = np.fromfile(fid, 'float64', numFrames, '')
        sec = np.fromfile(fid, 'float64', numFrames, '')
    return dict(frames=numFrames, lat=lat, lon=lon, alt=alt, vn=vn, ve=ve, vu=vu, r=r, p=p, az=az, sec=sec)


def loadPostCorrectionsGPSData(fnme):
    # open the post-corrections
    with open(fnme, 'rb') as fid:
        # parse all of the post-correction data
        numPostFrames = int(np.fromfile(fid, 'uint32', 1, '')[0])
        latConv = np.fromfile(fid, 'float64', 1, '')[0]
        lonConv = np.fromfile(fid, 'float64', 1, '')[0]
        rxEastingM = np.fromfile(fid, 'float64', numPostFrames, '')
        rxEastingM[abs(rxEastingM) < lonConv] *= lonConv
        rxNorthingM = np.fromfile(fid, 'float64', numPostFrames, '')
        rxNorthingM[abs(rxNorthingM) < latConv] *= \
            latConv
        rxAltM = np.fromfile(fid, 'float64', numPostFrames, '')
        # fid.seek(numPostFrames * 8 * 3, 1)
        txEastingM = np.fromfile(fid, 'float64', numPostFrames, '')
        txEastingM[abs(txEastingM) < lonConv] *= lonConv
        txNorthingM = np.fromfile(fid, 'float64', numPostFrames, '')
        txNorthingM[abs(txNorthingM) < latConv] *= \
            latConv
        txAltM = np.fromfile(fid, 'float64', numPostFrames, '')
        aziPostR = np.fromfile(fid, 'float64', numPostFrames, '')
        sec = np.fromfile(fid, 'float64', numPostFrames, '')
    return dict(frames=numPostFrames, latConv=latConv, lonConv=lonConv, rx_lon=rxEastingM / lonConv,
                rx_lat=rxNorthingM / latConv, rx_alt=rxAltM, tx_lon=txEastingM / lonConv, tx_lat=txNorthingM / latConv,
                tx_alt=txAltM, az=aziPostR, sec=sec)


"""
Gimbal data format:
First 32-bits: The number of gimbal frames (N)
N * sizeof( double ): pan position in radians
N * sizeof( double ): tilt position in radians 
N * sizeof( double ): system time in TAC, unwrapped
"""


def loadGimbalData(filename):
    fid = open(filename, 'rb')
    numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
    ret_dict = {'pan': np.fromfile(fid, 'float64', numFrames, ''), 'tilt': np.fromfile(fid, 'float64', numFrames, ''),
                'systime': np.fromfile(fid, 'float64', numFrames, '')}
    return ret_dict


def loadMatchedFilter(filename):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        ret = np.fromfile(fid, 'complex64', numFrames, '')
    return ret


def loadReferenceChirp(filename):
    with open(filename, 'rb') as fid:
        num_samples = np.fromfile(fid, 'uint32', 1, '')[0]
        tmp = np.fromfile(fid, 'int16', num_samples * 2, '')
        ret = (tmp[0::2] + 1j * tmp[1::2]) * 10 ** (31 / 20)
    return ret


def getFFTParams(filename):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')
        numFFTSamples = np.fromfile(fid, 'uint32', 1, '')
    return numFrames[0], numFFTSamples[0]


def getRawParams(filename):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        numSamples = np.fromfile(fid, 'uint16', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', numFrames, '')
        sys_time = np.fromfile(fid, 'double', numFrames, '')
    return numFrames, numSamples, attenuation, sys_time


def getRawSDRParams(filename):
    with open(filename, 'rb') as fid:
        numFrames = np.fromfile(fid, 'uint32', 1, '')[0]
        numSamples = np.fromfile(fid, 'uint32', 1, '')[0]
        attenuation = np.fromfile(fid, 'int8', numFrames, '')
        sys_time = np.fromfile(fid, 'double', numFrames, '')
    return numFrames, numSamples, attenuation, sys_time


def createIFTMatrix(m, fs):
    D = np.ones((m, m), dtype='complex64')
    for i in range(1, m):
        D[:, m] = np.exp(1j * 2 * pi * i * fs / m * np.arange(m) * 1 / (fs / m))

    return D


def GetAdvMatchedFilter(chan, nbar=5, SLL=-35, sar=None, pulseNum=20, fft_len=None):
    # Things the PS will need to know from the configuration
    numSamples = chan.nsam
    samplingFreqHz = chan.fs
    basebandedChirpRateHzPerS = chan.chirp_rate
    # If the NCO was positive it means we will have sampled the reverse spectrum
    #   and the chirp will be flipped
    if chan.NCO_freq_Hz > 0:
        basebandedChirpRateHzPerS *= -1
    halfBandwidthHz = chan.bw / 2.0
    # Get the basebanded center, start and stop frequency of the chirp
    basebandedCenterFreqHz = chan.baseband_fc
    basebandedStartFreqHz = chan.baseband_fc - halfBandwidthHz
    basebandedStopFreqHz = chan.baseband_fc + halfBandwidthHz
    if basebandedChirpRateHzPerS < 0:
        basebandedStartFreqHz = chan.baseband_fc + halfBandwidthHz
        basebandedStopFreqHz = chan.baseband_fc - halfBandwidthHz

    # Get the reference waveform and mix it down by the NCO frequency and
    #   downsample to the sampling rate of the receive data if necessary
    # The waveform input into the DAC has already had the Hilbert transform
    #   and downsample operation performed on it by SDRParsing, so it is
    #   complex sampled data at this point at the SlimSDR base complex sampling
    #   rate.
    # Compute the decimation rate if the data has been low-pass filtered and
    #   downsampled
    decimationRate = 1
    if chan.is_lpf:
        decimationRate = int(np.floor(chan.BASE_COMPLEX_SRATE_HZ / samplingFreqHz))

    # Grab the waveform
    waveformData = chan.ref_chirp

    # Create the plot for the FFT of the waveform
    waveformLen = len(waveformData)

    # Compute the mixdown signal
    mixDown = np.exp(1j * (2 * np.pi * chan.NCO_freq_Hz * np.arange(waveformLen) / chan.BASE_COMPLEX_SRATE_HZ))
    basebandWaveform = mixDown * waveformData

    # Decimate the waveform if applicable
    if decimationRate > 1:
        basebandWaveform = basebandWaveform[:: decimationRate]
    # Calculate the updated baseband waveform length
    basebandWaveformLen = len(basebandWaveform)
    # Grab the calibration data
    calData = chan.cal_chirp + 0.0
    # Grab the pulses
    if sar:
        calData = sar.getPulse(pulseNum, channel=0).T + 0.0

    # Calculate the convolution length
    convolutionLength = numSamples + basebandWaveformLen - 1
    FFTLength = findPowerOf2(convolutionLength) if fft_len is None else fft_len

    # Calculate the inverse transfer function
    FFTCalData = np.fft.fft(calData, FFTLength)
    FFTBasebandWaveformData = np.fft.fft(basebandWaveform, FFTLength)
    inverseTransferFunction = FFTBasebandWaveformData / FFTCalData
    # NOTE! Outside of the bandwidth of the signal, the inverse transfer function
    #   is invalid and should not be viewed. Values will be enormous.

    # Generate the Taylor window
    TAYLOR_NBAR = 5
    TAYLOR_NBAR = nbar
    TAYLOR_SLL_DB = -35
    TAYLOR_SLL_DB = SLL
    windowSize = \
        int(np.floor(halfBandwidthHz * 2.0 / samplingFreqHz * FFTLength))
    taylorWindow = window_taylor(windowSize, nbar=TAYLOR_NBAR, sll=TAYLOR_SLL_DB) if SLL != 0 else np.ones(windowSize)

    # Create the matched filter and polish up the inverse transfer function
    matchedFilter = np.fft.fft(basebandWaveform, FFTLength)
    # IQ baseband vs offset video
    if np.sign(basebandedStartFreqHz) != np.sign(basebandedStopFreqHz):
        # Apply the inverse transfer function
        aboveZeroLength = int(np.ceil((basebandedCenterFreqHz + halfBandwidthHz) / samplingFreqHz * FFTLength))
        belowZeroLength = int(windowSize - aboveZeroLength)
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[int(FFTLength / 2) - aboveZeroLength:int(FFTLength / 2) - aboveZeroLength + windowSize] = \
            taylorWindow
        # Zero out the invalid part of the inverse transfer function
        inverseTransferFunction[aboveZeroLength: -belowZeroLength] = 0
        taylorWindowExtended = np.fft.fftshift(taylorWindowExtended)
    else:
        # Apply the inverse transfer function
        bandStartInd = \
            int(np.floor((basebandedCenterFreqHz - halfBandwidthHz) / samplingFreqHz * FFTLength))
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[bandStartInd: bandStartInd + windowSize] = taylorWindow
        inverseTransferFunction[: bandStartInd] = 0
        inverseTransferFunction[bandStartInd + windowSize:] = 0
    matchedFilter = matchedFilter.conj() * inverseTransferFunction * taylorWindowExtended
    return matchedFilter


def window_taylor(N, nbar=4, sll=-30):
    """Taylor tapering window
    Taylor windows allows you to make tradeoffs between the
    mainlobe width and sidelobe level (sll).
    Implemented as described by Carrara, Goodman, and Majewski
    in 'Spotlight Synthetic Aperture Radar: Signal Processing Algorithms'
    Pages 512-513
    :param N: window length
    :param float nbar:
    :param float sll:
    The default values gives equal height
    sidelobes (nbar) and maximum sidelobe level (sll).
    .. warning:: not implemented
    .. seealso:: :func:`create_window`, :class:`Window`
    """
    if sll > 0:
        sll *= -1
    B = 10 ** (-sll / 20)
    A = np.log(B + np.sqrt(B ** 2 - 1)) / np.pi
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)

    def calc_Fm(m):
        numer = (-1) ** (m + 1) \
                * np.prod(1 - m ** 2 / s2 / (A ** 2 + (ma - 0.5) ** 2))
        denom = 2 * np.prod([1 - m ** 2 / j ** 2 for j in ma if j != m])
        return numer / denom

    Fm = np.array([calc_Fm(m) for m in ma])

    def W(n):
        return 2 * np.sum(
            Fm * np.cos(2 * np.pi * ma * (n - N / 2 + 1 / 2) / N)) + 1

    w = np.array([W(n) for n in range(N)])
    # normalize (Note that this is not described in the original text)
    scale = W((N - 1) / 2)
    w /= scale
    return w


def sinc_interp(x, s, u):
    if len(x) != len(s):
        raise ValueError('x and s must be the same length')

    # Find the period
    T = s[1] - s[0]

    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    return np.dot(x, np.sinc(sincM / T))


def getDopplerLine(effAzI, rangeBins, antVel, antPos, nearRangeGrazeR, azBeamwidthHalf, PRF, wavelength, origin):
    """Compute the expected Doppler vs range for the given platform geometry"""

    # compute the grazing angle for the near range to start
    (nearRangeGrazeR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
        effAzI, nearRangeGrazeR, antPos, rangeBins[0], origin)

    # now I need to get the grazing angles across all of the range bins
    grazeOverRanges = np.arcsin((antPos[2] + origin[2] - surfaceHeight) / rangeBins)

    # this is a special version of Rvec (it is not 3x1, it is 3xNrv)
    Rvec = np.array([
        np.cos(grazeOverRanges) * np.sin(effAzI),
        np.cos(grazeOverRanges) * np.cos(effAzI),
        -np.sin(grazeOverRanges)])
    # perform the dot product and calculate the Doppler
    DopplerCen = ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = np.nonzero(DopplerCen > PRF / 2)
    DopplerCen[ind] -= PRF
    ind = np.nonzero(DopplerCen < -PRF / 2)
    DopplerCen[ind] += PRF

    # generate the radial vector for the forward beamwidth edge
    # (NOTE!!!: this is dependent
    # on the antenna pointing vector attitude with respect to the aircraft heading.
    # if on the left side, negative azimuth will be lower Doppler, and positive
    # azimuth will be higher, but on the right side, it will be the opposite, one
    # could use the sign of the cross-product to determine which it is.)
    # if (xmlData.gimbalSettings.lookSide.lower() == 'left'):
    eff_boresight = np.mean(np.array([
        np.cos(grazeOverRanges) * np.sin(effAzI),
        np.cos(grazeOverRanges) * np.cos(effAzI),
        -np.sin(grazeOverRanges)]), axis=1)
    ant_dir = np.cross(eff_boresight, antVel)
    azBeamwidthHalf *= np.sign(ant_dir[2])

    newAzI = effAzI - azBeamwidthHalf
    Rvec = np.array([
        np.cos(grazeOverRanges) * np.sin(newAzI),
        np.cos(grazeOverRanges) * np.cos(newAzI),
        -np.sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerUp = ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = np.nonzero(DopplerUp > PRF / 2)
    DopplerUp[ind] -= PRF
    ind = np.nonzero(DopplerUp < -PRF / 2)
    DopplerUp[ind] += PRF

    # generate the radial vector for the forward beamwidth edge
    newAzI = effAzI + azBeamwidthHalf
    Rvec = np.array([
        np.cos(grazeOverRanges) * np.sin(newAzI),
        np.cos(grazeOverRanges) * np.cos(newAzI),
        -np.sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerDown = \
        ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = np.nonzero(DopplerDown > PRF / 2)
    DopplerDown[ind] -= PRF
    ind = np.nonzero(DopplerDown < -PRF / 2)
    DopplerDown[ind] += PRF
    return DopplerCen, DopplerUp, DopplerDown, grazeOverRanges


def computeGrazingAngle(effAzIR, grazeIR, antPos, theRange, origin):
    # initialize the pointing vector to first range bin
    Rvec = np.array([np.cos(grazeIR) * np.sin(effAzIR),
                  np.cos(grazeIR) * np.cos(effAzIR),
                  -np.sin(grazeIR)])

    groundPoint = antPos + Rvec * theRange
    nlat, nlon, alt = enu2llh(*groundPoint, origin)
    # look up the height of the surface below the aircraft
    surfaceHeight = getElevation((nlat, nlon), False)
    # check the error in the elevation compared to what was calculated
    elevDiff = surfaceHeight - alt

    iterationThresh = 2
    heightDiffThresh = 1.0
    numIterations = 0
    newGrazeR = grazeIR + 0.0
    # iterate if the difference is greater than 1.0 m
    while abs(elevDiff) > heightDiffThresh and numIterations < iterationThresh:
        hAgl = antPos[2] + origin[2] - surfaceHeight
        newGrazeR = np.arcsin(hAgl / theRange)
        if np.isnan(newGrazeR) or np.isinf(newGrazeR):
            print('NaN or inf found.')
        Rvec = np.array([np.cos(newGrazeR) * np.sin(effAzIR),
                      np.cos(newGrazeR) * np.cos(effAzIR),
                      -np.sin(newGrazeR)])
        groundPoint = antPos + Rvec * theRange
        nlat, nlon, alt = enu2llh(*groundPoint, origin)
        surfaceHeight = getElevation((nlat, nlon), False)
        # check the error in the elevation compared to what was calculated
        elevDiff = surfaceHeight - alt
        numIterations += 1

    return newGrazeR, Rvec, surfaceHeight, numIterations

