import numpy as np
import cupy as cupy
import cupyx.scipy.signal
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
from scipy.signal import medfilt, fftconvolve, find_peaks
from scipy.ndimage import binary_erosion, binary_dilation, label, median_filter
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
from cuda_kernels import calcBounceFromMesh, checkIntersection, genRangeProfileFromMesh, findPowerOf2, db
from sklearn.covariance import oas, graphical_lasso
import open3d as o3d
from numba import cuda, njit
from tqdm import tqdm
from celluloid import Camera
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from cambiguity import amb_surf, narrow_band, wide_band
from scipy.spatial.transform import Rotation as rot
from simulation_functions import getElevationMap, getElevation, llh2enu, getMapLocation, getLivingRoom, resampleGrid

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
fs = 100e6
DTR = np.pi / 180
inch_to_m = .0254
BYTES_PER_SAMPLE = 4
BYTES_TO_MB = 1048576


def genPulse(phase_x, phase_y, nnr, t0, nfc, bandw):
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * t0 / nnr))


def rotate(az, nel, rot_mat):
    return rot.from_euler('zxy', [-az, 0, 0]).apply(rot.from_euler('zxy', [0, nel - np.pi / 2, 0]).apply(rot_mat))


def azelToVec(az, el):
    return np.array([np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), np.cos(el)])


# OPTIONS
fc = 32e9
bandwidth = 90e6
PRF = 1500.0
collect_dur = 2.
cpi_len = 32
angle_npts = 120
az_bw = 20 * DTR
el_bw = 3 * DTR
platform_spd = 89.4
platform_dir = rotate(140. * DTR, np.pi / 2, np.array([0, 1., 0]))
# platform_dir = np.array([0, 1., 0])
platform_height = 60
npts_background = 250
init_ll = (40.068287, -111.576847)
nrange = 750.
frange = 3000.
plp = .5
rot_spd = 50 * DTR
upsample = 1
mesh_reflectivity = 1.
noise_power = 1e-4
use_mesh = False
p1 = (40.060023, -111.567205)
static_points = False
num_pts_per_triangle = 2
# Rotation limits only in azimuth because it doesn't rotate in elevation
rot_lims = (-45 * DTR, 45 * DTR)
dr_wave = .07

# measure_z = 3.324265984 * inch_to_m
# measure_x1 = 1.368678437 * inch_to_m
# measure_x2 = 1.7005915629 * inch_to_m
rx_array = np.array([[0, -dr_wave / 2, 0],
                     [0, dr_wave / 2, 0],
                     [0, dr_wave * 3/2, 0],
                     [0, -dr_wave * 3/2, 0]])

# Generate nsam, nr, based on geometry
max_pl = (nrange * 2 / c0 - 1 / TAC) * .99
nr = int(max_pl * plp * fs)
nsam = int((np.ceil((2 * frange / c0 + max_pl * plp) * TAC) -
            np.floor(2 * nrange / c0 * TAC)) * fs / TAC)
MPP = c0 / fs / upsample / 2
ranges = nrange + np.arange(nsam * upsample) * MPP + c0 / fs
scaled_ranges = ranges ** 2 / np.max(ranges ** 2)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample
midrange = (nrange + frange) / 2
beam_extent = (frange - nrange) / 2

# Calculate size of horns to get desired beamwidths
# This is an empirically derived rough estimate of radiation function params
rad_cons = np.array([4, 3, 2, 1, .01])
rad_bws = np.array([4.01, 6, 8.6, 18.33, 33])
az_fac = np.interp(az_bw / DTR / 2, rad_bws, rad_cons)
el_fac = np.interp(el_bw / DTR / 2, rad_bws, rad_cons)

# Calculate out data rate for Kozak
data_rate = PRF * (2 / c0 * (frange-nrange) + max_pl * plp) * BYTES_PER_SAMPLE * (rx_array.shape[0] + 1) * \
            fs / BYTES_TO_MB

# Generate position based on lat/lon/alt
origin = (*p1, getElevation(p1))
try:
    init_llh = (*init_ll, getElevation(init_ll))
    init_pos = llh2enu(*init_ll, getElevation(init_ll) + platform_height, origin)
except RuntimeError:
    init_llh = (*init_ll, 0.)
    init_pos = np.array([0, 0., platform_height])

n_pulses = int(collect_dur * PRF)
use_elevation = np.diff(rx_array[:, 2]).mean() >= dr_wave / 4

try:
    if use_mesh:
        pcd = getLivingRoom()
    else:
        pcd = getMapLocation(origin, (8000, 8000), init_llh, npts_background=npts_background, resample=False)
except:
    forward_point = init_pos + platform_dir * (nrange + frange) / 2
    pcd = o3d.geometry.TriangleMesh.create_box(400, 400, 200)
    pcd = pcd.translate(forward_point)
    pcd = pcd.sample_points_poisson_disk(npts_background * 5)

print('Generating mesh...')
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist
radii = [radius, radius * 2]
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)

rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

rec_mesh.remove_duplicated_vertices()
rec_mesh.remove_duplicated_triangles()
rec_mesh.remove_degenerate_triangles()
rec_mesh.remove_unreferenced_vertices()

vertices = np.asarray(rec_mesh.vertices)
triangles = np.asarray(rec_mesh.triangles)
normals = np.asarray(rec_mesh.vertex_normals)
# Press n in visualizer to see normals
# o3d.visualization.draw_geometries([pcd, rec_mesh])

# First, generate the copter's flight path
tt = np.linspace(0, collect_dur, n_pulses)
p_e = medfilt(np.random.normal(0, 1e-6, (n_pulses,))) + tt * platform_dir[0] * platform_spd + init_pos[0]
p_cse = CubicSpline(tt, p_e)
p_n = medfilt(np.random.normal(0, 1e-6, (n_pulses,))) + tt * platform_dir[1] * platform_spd + init_pos[1]
p_csn = CubicSpline(tt, p_n)
p_u = medfilt(np.random.normal(0, 1e-6, (n_pulses,))) + tt * platform_dir[2] * platform_spd + init_pos[2]
p_csu = CubicSpline(tt, p_u)

# Generate pan and tilt angles as the copter moves along
t_rot = np.linspace(0, 2 * np.pi / rot_spd, n_pulses)
pan_cs = CubicSpline(t_rot,
                     np.concatenate((np.linspace(rot_lims[0], rot_lims[1], n_pulses // 2),
                                     np.linspace(rot_lims[1], rot_lims[0], n_pulses // 2))))
pan_rot = lambda t: pan_cs(t % (2 * np.pi / rot_spd))

# Get doppler bins for angle_nn
dopp_bins = np.fft.fftfreq(cpi_len, 1 / PRF)


def el(t):
    try:
        return np.ones((len(t),)) * np.pi / 2
    except TypeError:
        return np.pi / 2


p_pos = lambda t: np.array([p_cse(t), p_csn(t), p_csu(t)])

# Next, generate the environment in which the copter will be flying

gpuDevice = cuda.get_current_device()
maxThreads = gpuDevice.MAX_THREADS_PER_BLOCK // 2
sqrtMaxThreads = int(np.sqrt(maxThreads))
threads_per_block = (sqrtMaxThreads, sqrtMaxThreads)
blocks_per_grid = (max(1, triangles.shape[0] // threads_per_block[0] + 1), cpi_len // threads_per_block[1] + 1)

n_pts = triangles.shape[0] * num_pts_per_triangle
bpg_rdata = (max(1, n_pts // threads_per_block[0] + 1), cpi_len // threads_per_block[1] + 1)

tri_vert_indices = cupy.array(triangles, dtype=np.int32)
vert_xyz = cupy.array(vertices, dtype=np.float64)
vert_norms = cupy.array(normals, dtype=np.float64)
reflectivity = np.ones((vert_xyz.shape[0],)) * mesh_reflectivity
vert_reflectivity = cupy.array(reflectivity, dtype=np.float64)
return_xyz = cupy.zeros((n_pts, cpi_len, 3), dtype=np.float64)
return_pow = cupy.zeros((n_pts, cpi_len), dtype=np.float64)
bounce_xyz = cupy.zeros((n_pts, cpi_len, 3), dtype=np.float64)

taytay = taylor(up_fft_len)
tayd = np.fft.fftshift(taylor(cpi_len))
taydopp = np.fft.fftshift(np.ones((nsam * upsample, 1)).dot(tayd.reshape(1, -1)), axes=1)
chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, nr / fs, fc, bandwidth) * 1e5, up_fft_len)
mfilt = chirp.conj()
mfilt[:up_fft_len // 2] *= taytay[up_fft_len // 2:]
mfilt[up_fft_len // 2:] *= taytay[:up_fft_len // 2]
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
pts_plot = []
img_plot = []
xrngs = [min(init_pos[0], vertices[:, 0].min()), max(init_pos[0], vertices[:, 0].max())]
yrngs = [min(init_pos[1], vertices[:, 1].min()), max(init_pos[1], vertices[:, 1].max())]
zrngs = [min(init_pos[2], vertices[:, 2].min()), max(init_pos[2], vertices[:, 2].max())]
# zrngs = [min(init_pos[2], vertices[:, 2].min()), min(init_pos[2] * 2, max(init_pos[2], vertices[:, 2].max()))]

amb_func = cupy.array(narrow_band(np.fft.ifft(chirp), np.fft.ifft(chirp), np.arange(cpi_len) - cpi_len // 2)[0],
                      dtype=np.complex128)
kernel = np.ones((10, 10))
# kernel[5:8, 5:8] = 0
amb_func = cupy.array(kernel, dtype=np.complex128)
amb_func = cupy.fft.fft2(amb_func.conj().T, s=(nsam * upsample, cpi_len))

rxinit_gpu = cupy.array(rx_array, dtype=np.float64)

if use_elevation:
    rough_az_mesh, rough_el_mesh = np.meshgrid(np.linspace(-az_bw / 2, az_bw / 2, 3),
                                               np.linspace(-el_bw / 2, el_bw / 2, 3))
    rough_az_mesh = rough_az_mesh.flatten()
    rough_el_mesh = rough_el_mesh.flatten()
    fine_az_mesh, fine_el_mesh = np.meshgrid(np.linspace(-az_bw / 2, az_bw / 2, angle_npts),
                                             np.linspace(-el_bw / 2, el_bw / 2, angle_npts))
    fine_az_mesh = fine_az_mesh.flatten()
    fine_el_mesh = fine_el_mesh.flatten()
else:
    rough_az_mesh = np.linspace(-az_bw / 2, az_bw / 2, 5)
    rough_el_mesh = np.zeros_like(rough_az_mesh)
    fine_az_mesh = np.linspace(-az_bw / 2, az_bw / 2, angle_npts)
    fine_el_mesh = np.zeros_like(fine_az_mesh)
el_atten = np.ones_like(fine_el_mesh)
el_atten[fine_el_mesh != 0] = np.sin(np.pi / (el_bw * DTR) * fine_el_mesh[fine_el_mesh != 0]) / \
                              (np.pi / (el_bw * DTR) * fine_el_mesh[fine_el_mesh != 0])
az_atten = np.ones_like(fine_az_mesh)
az_atten[fine_az_mesh != 0] = np.sin(np.pi / (az_bw * DTR) * fine_az_mesh[fine_az_mesh != 0]) / \
                              (np.pi / (az_bw * DTR) * fine_az_mesh[fine_az_mesh != 0])
fine_atten = abs(el_atten) * abs(az_atten)
ublock = azelToVec(rough_az_mesh, rough_el_mesh)
rough_avec = np.exp(-1j * 2 * np.pi * fc / c0 * rx_array.dot(ublock))
ublock = azelToVec(fine_az_mesh, fine_el_mesh)
fine_avec = np.exp(-1j * 2 * np.pi * fc / c0 * rx_array.dot(ublock))

# Data animation
fig, ax = plt.subplots(1, 1)
cam = Camera(fig)
blob_pow_mean = 0.
blob_pow_var_d = [0., 0.]
blob_total_count = 0

# Plotly animation stuff
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
check_back_sz = 0.
print('Running data simulation...')
photon_data = np.zeros((nsam * upsample, n_pulses), dtype=np.complex128)

for ts_idx, ts in tqdm(enumerate([tt[pos:pos + cpi_len] for pos in range(0, len(tt), cpi_len)])):
    if len(ts) < cpi_len:
        break
    if static_points:
        uvs = np.ones((num_pts_per_triangle, 2)) / 3
    else:
        uvs = np.random.rand(num_pts_per_triangle, 2)
        if sum(np.sum(uvs, 1) > 1) > 1:
            uvs[np.sum(uvs, 1) > 1, :] = uvs[np.sum(uvs, 1) > 1, :] / np.sum(uvs[np.sum(uvs, 1) > 1, :], 1)[:, None]
        elif sum(np.sum(uvs, 1) > 1) == 1:
            uvs[np.sum(uvs, 1) > 1, :] = uvs[np.sum(uvs, 1) > 1, :] / sum(uvs[np.sum(uvs, 1) > 1, :])
    bounce_uv = cupy.array(uvs, dtype=np.float64)
    tx_pos = p_pos(ts)
    tx_vel = np.gradient(tx_pos, axis=1)
    az_pos = np.arctan2(tx_vel[0, :].mean(), tx_vel[1, :].mean())
    tx_gpu = cupy.array(tx_pos, dtype=np.float64)
    pan_tx_gpu = cupy.array(pan_rot(ts) + az_pos, dtype=np.float64)
    el_tx_gpu = cupy.array(el(ts), dtype=np.float64)
    calcBounceFromMesh[blocks_per_grid, threads_per_block](tri_vert_indices, vert_xyz, vert_norms, vert_reflectivity,
                                                           tx_gpu, bounce_uv, return_xyz, return_pow,
                                                           bounce_xyz)
    cupy.cuda.Device().synchronize()

    is_blocked = cupy.zeros((n_pts, cpi_len), dtype=bool)
    # checkIntersection[blocks_per_grid, threads_per_block](return_sph, is_blocked)
    # cupy.cuda.Device().synchronize()

    data = np.random.normal(0, noise_power, (nsam * upsample, cpi_len, rx_array.shape[0])) + \
           1j * np.random.normal(0, noise_power, (nsam * upsample, cpi_len, rx_array.shape[0]))
    deb_att = np.zeros((n_pts, cpi_len))
    for rx in range(rx_array.shape[0]):
        data_r = cupy.zeros((nsam, cpi_len), dtype=np.float64)
        data_i = cupy.zeros((nsam, cpi_len), dtype=np.float64)
        rx_pos = np.array([p_pos(t) + rotate(pan_rot(t), el(t), rx_array[rx, :]) for t in ts]).T
        rx_gpu = cupy.array(rx_pos, dtype=np.float64)
        pan_rx_gpu = cupy.array(pan_rot(ts) + az_pos, dtype=np.float64)
        el_rx_gpu = cupy.array(el(ts), dtype=np.float64)
        debug_att = cupy.zeros((n_pts, cpi_len), dtype=np.float64)
        genRangeProfileFromMesh[bpg_rdata, threads_per_block](return_xyz, bounce_xyz, rx_gpu, return_pow,
                                                              is_blocked,
                                                              pan_rx_gpu, el_rx_gpu, pan_tx_gpu, el_rx_gpu,
                                                              data_r, data_i, debug_att, c0 / fc,
                                                              nrange / c0,
                                                              fs, az_bw / 2, el_bw / 2)
        cupy.cuda.Device().synchronize()
        rtdata = cupy.fft.fft(data_r + 1j * data_i, fft_len, axis=0)
        upsample_data = cupy.zeros((up_fft_len, cpi_len), dtype=np.complex128)
        upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
        rtdata = cupy.fft.ifft(upsample_data * chirp_gpu * mfilt_gpu, axis=0)
        cupy.cuda.Device().synchronize()
        data[:, :, rx] += rtdata.get()[:nsam * upsample, :]
        deb_att += debug_att.get()

    del pan_rx_gpu
    del pan_tx_gpu
    del data_r
    del data_i
    del rtdata
    del el_rx_gpu
    del el_tx_gpu
    del tx_gpu
    del rx_gpu
    del debug_att

    pt_theta = []
    pt_rng = []
    pt_t = []
    pt_phi = []
    pt_pow = []
    pt_sz = []
    ang_block = 0
    photon_data[:, ts_idx * cpi_len:(ts_idx+1) * cpi_len] = data[:, :, 0]

    Rhats = np.zeros((rx_array.shape[0], rx_array.shape[0], cpi_len), dtype=np.complex128)
    Rinvs = np.zeros((rx_array.shape[0], rx_array.shape[0], cpi_len), dtype=np.complex128)
    plot_beams = []
    # Do a general check across the beam
    for tt in range(cpi_len):
        eig_values, Uh = np.linalg.eig(oas(data[:, tt, :])[0].T)
        eig_values[-2:] = 0.
        Rhats[:, :, tt] = Uh.dot(np.diag(eig_values)).dot(Uh.conj().T)
        # Rhats[:, :, tt] = oas(data[:, tt, :])[0].T
        Rinvs[:, :, tt] = np.linalg.pinv(Rhats[:, :, tt])

    for beam_seg in range(rough_avec.shape[1]):
        straight_beam = np.fft.fft(data.dot(rough_avec[:, beam_seg]), axis=1) * taydopp
        bf = cupy.array(straight_beam, dtype=np.complex128)
        det_gpu = cupy.fft.ifft2(cupy.fft.fft2(bf) * amb_func)
        cupy.cuda.Device().synchronize()
        thresh_pts = det_gpu.get()

        del bf
        del det_gpu
        cupy.get_default_memory_pool().free_all_blocks()

        thresh_pts = abs(thresh_pts)
        thresh = np.sum(thresh_pts, axis=1)[:nsam] * scaled_ranges
        rngs = np.where(thresh > thresh.mean() + 3 * thresh.std())[0]
        single_rngs = list(set(rngs))

        if np.any(single_rngs):
            cong_spec = []
            cong_az = []
            cong_el = []
            bin_az = np.linspace(-az_bw / 2 + pan_rot(ts[0]) + az_pos, az_bw / 2 + pan_rot(ts[-1]) + az_pos, angle_npts)
            bin_el = np.linspace(-el_bw / 2, el_bw / 2, angle_npts)
            for tidx, tknot in enumerate(ts):
                # Find angles for bin under test
                # Capon beamformer
                spectrum = abs(1 / np.sum(fine_avec.conj().T.dot(Rinvs[:, :, tidx]) * fine_avec.T, axis=1)) / fine_atten
                # Bartlett beamformer
                # spectrum = abs(np.sum(fine_avec.conj().T.dot(Rhats[:, :, tidx]) * fine_avec.T, axis=1)
                #               / np.sqrt(rx_array.shape[1]))
                cong_spec = np.concatenate((cong_spec, spectrum))
                cong_az = np.concatenate((cong_az, fine_az_mesh + pan_rot(tknot) + az_pos))
                cong_el = np.concatenate((cong_el, fine_el_mesh + el(tknot)))
            if use_elevation:
                ang_block, bin_az, bin_el = np.histogram2d(cong_az, cong_el, bins=[bin_az, bin_el], weights=cong_spec)
                labels, nlabels = label(ang_block > np.median(ang_block) + 3 * ang_block.std())
            else:
                ang_block, bin_az = np.histogram(cong_az, bins=bin_az, weights=cong_spec)
                labels = find_peaks(ang_block)[0]
                nlabels = 1
            for blob in range(1, nlabels + 1):
                if use_elevation:
                    blob_az_idx, blob_el_idx = np.where(labels == blob)
                    blob_az = bin_az[blob_az_idx]
                    blob_el = bin_el[blob_el_idx]
                else:
                    blob_az = bin_az[labels]
                    blob_el = el(ts[0]) * np.ones_like(blob_az)
                ub = azelToVec(blob_az, blob_el)
                avec_blob = np.exp(-1j * 2 * np.pi * fc / c0 * rx_array.dot(ub))
                check_rng = np.zeros_like(blob_az, dtype=int)
                check_pow = np.zeros_like(blob_az)
                for pix in range(avec_blob.shape[1]):
                    blob_data = sum(
                        [abs(data[single_rngs, blob_t, :].dot(avec_blob[:, pix])) * scaled_ranges[single_rngs]
                         for blob_t in range(cpi_len)])
                    blob_rng = single_rngs[np.where(blob_data == np.max(blob_data))[0][0]]
                    check_rng[pix] = blob_rng
                    check_pow[pix] = np.max(blob_data)
                    for n in range(len(single_rngs)):
                        blob_pow_mean = blob_pow_mean + (blob_data[n] - blob_pow_mean) / (blob_total_count + 1)
                        old_bpvd = blob_pow_var_d[0] + 0.0
                        blob_pow_var_d[0] = blob_pow_var_d[0] + (blob_data[n] - blob_pow_var_d[0]) / \
                                            (blob_total_count + 1)
                        blob_pow_var_d[1] = blob_pow_var_d[1] + (blob_data[n] - blob_pow_var_d[0]) * \
                                            (blob_data[n] - old_bpvd)
                        blob_total_count += 1
                pt_theta = np.concatenate((pt_theta, blob_az))
                pt_rng = np.concatenate((pt_rng, ranges[check_rng]))
                pt_t = np.concatenate((pt_t, ts[0] * np.ones_like(blob_az)))
                pt_phi = np.concatenate((pt_phi, blob_el))
                pt_pow = np.concatenate((pt_pow, check_pow))

            if np.any(pt_theta):
                pt_sz = (np.array(pt_pow) - blob_pow_mean) / np.sqrt(blob_pow_var_d[1] / (blob_total_count - 1))
                pt_sz += -np.min(pt_sz) + .1

    # Plotting stuff for the cones
    plot_pos = p_pos(ts[0])
    plane_or = np.array([np.sin(az_pos + pan_rot(ts[0])), np.cos(az_pos + pan_rot(ts[0])), 0.]) * 100

    # Plotting stuff for beam
    ptb_rngs = np.zeros((17, 3))
    ptb_rngs[:, 1] = [nrange, nrange, nrange, nrange, frange, frange, frange, frange, nrange, nrange, frange, frange,
                      nrange, nrange, nrange, frange, frange]
    a = az_pos + pan_rot(ts[0]) - az_bw / 2
    b = az_pos + pan_rot(ts[-1]) + az_bw / 2
    ptb_theta = np.array([a, a, b, b, b, b, a, a, a, a, a, b, b, b, a, a, b])
    a = el(ts[0]) + el_bw / 2
    b = el(ts[0]) - el_bw / 2
    ptb_phi = np.array([a, b, b, a, a, b, b, a, a, b, b, b, b, a, a, a, a])
    ptb_pos = p_pos(ts[0])[:, None] + np.array(
        [rotate(ptb_theta[n], ptb_phi[n], ptb_rngs[n, :]) for n in range(len(ptb_theta))]).T

    # Project points and add frame to plot
    frame_name = f'{ts[0]:.2f}'
    if np.any(pt_rng):
        pt_sz *= 100.
        pt_sz[pt_sz > 40] = 40
        pt_rngs = np.zeros((len(pt_rng), 3))
        pt_rngs[:, 1] = pt_rng
        pt_pos = p_pos(pt_t) + np.array(
            [rotate(pt_theta[n], pt_phi[n], pt_rngs[n, :]) for n in range(len(pt_theta))]).T
        # pt_pos = p_pos(pt_t) + pt_dvec
        xrngs = [min(xrngs[0], pt_pos[0, :].min()), max(xrngs[1], pt_pos[0, :].max())]
        yrngs = [min(yrngs[0], pt_pos[1, :].min()), max(yrngs[1], pt_pos[1, :].max())]
        zrngs = [min(zrngs[0], pt_pos[2, :].min()), max(zrngs[1], pt_pos[2, :].max())]
        figure['frames'].append(
            go.Frame(data=[go.Scatter3d(x=pt_pos[0, :], y=pt_pos[1, :], z=pt_pos[2, :], mode='markers',
                                        marker={'color': 'blue', 'size': pt_sz}, name='det_pts'),
                           go.Scatter3d(x=ptb_pos[0, :], y=ptb_pos[1, :], z=ptb_pos[2, :],
                                        marker={'color': 'green', 'size': 1}, mode='lines+markers',
                                        line={'color': 'green', 'width': 2}, name='beam_pts'),
                           go.Cone(x=[plot_pos[0]], y=[plot_pos[1]], z=[plot_pos[2]], u=[plane_or[0]],
                                   v=[plane_or[1]], w=[plane_or[2]])], name=frame_name,
                     traces=[1, 2, 3], layout={'title': f'Frame {ts[0]:.2f}, Angle {pan_rot(ts[0])}'}))
    else:
        figure['frames'].append(
            go.Frame(data=[go.Cone(x=[plot_pos[0]], y=[plot_pos[1]], z=[plot_pos[2]], u=[plane_or[0]],
                                   v=[plane_or[1]], w=[plane_or[2]]),
                           go.Scatter3d(x=ptb_pos[0, :], y=ptb_pos[1, :], z=ptb_pos[2, :],
                                        marker={'color': 'green', 'size': 1}, mode='lines+markers',
                                        line={'color': 'green', 'width': 2}, name='beam_pts')
                           ],
                     name=frame_name, traces=[1, 2],
                     layout={'title': f'Frame {ts[0]:.2f}, Angle {pan_rot(ts[0])}, No detected Points'}))
    sliders_dict['steps'].append({'args': [
        [frame_name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate',
                       'transition': {'duration': 0}}
    ],
        'label': frame_name,
        'method': 'animate'})

    back_xyz = return_xyz.get()
    back_sz = deb_att.sum(axis=1)
    check_back_sz += back_sz.max() / (n_pulses // cpi_len)
    legits = back_sz > back_sz.mean()
    back_xyz = back_xyz[legits, 0, :]
    b_xyz = bounce_xyz.get()
    b_xyz = b_xyz[legits, 0, :]
    if b_xyz.shape[0] != 0:
        back_sz = back_sz[legits] / back_sz[legits].max()
        back_sz[back_sz < 0] = 0
        back_sz *= 256

        img_plot.append(go.Frame(data=[go.Scatter3d(x=back_xyz[:, 0] + p_pos(ts[0])[0], y=back_xyz[:, 1] + p_pos(ts[0])[1],
                                                    z=back_xyz[:, 2] + p_pos(ts[0])[2], mode='markers',
                                                    marker={'color': back_sz, 'size': 10},
                                                    name='det_pts'),
                                       go.Scatter3d(x=ptb_pos[0, :], y=ptb_pos[1, :], z=ptb_pos[2, :],
                                                    marker={'color': 'green', 'size': 1}, mode='lines+markers',
                                                    line={'color': 'green', 'width': 2}, name='beam_pts'),
                                       go.Cone(x=back_xyz[:, 0] + p_pos(ts[0])[0], y=back_xyz[:, 1] + p_pos(ts[0])[1],
                                               z=back_xyz[:, 2] + p_pos(ts[0])[2], u=b_xyz[:, 0], v=b_xyz[:, 1],
                                               w=b_xyz[:, 2])],
                                 traces=[1, 2, 3], name=f'Frame {ts[0]:.2f}',
                                 layout={'title': f'Frame {ts[0]:.2f}, Angle {pan_rot(ts[0])}'}))

    if use_elevation:
        ax.imshow(ang_block, extent=[dopp_bins[-1], dopp_bins[0], ranges[-1], ranges[0]])
    else:
        ax.imshow(thresh_pts, extent=[dopp_bins[-1], dopp_bins[0], ranges[-1], ranges[0]])
    ax.text(0.5, 1.01, f'Frame {ts[0]:.2f}', transform=ax.transAxes)
    # ax.imshow(ang_block)
    ax.axis('tight')
    cam.snap()

del tri_vert_indices

print('Rendering video...')
animation = cam.animate()

# print('Displaying point cloud...')
# o3d.visualization.draw_geometries([mesh])

plane_plot_pts = p_pos(np.linspace(0, 1, 10))
if vertices.shape[0] < 5:
    background = go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='markers',
                              marker={'color': 'red', 'size': 5}, name='background')
else:
    background = go.Mesh3d(i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], x=vertices[:, 0], y=vertices[:, 1],
                           z=vertices[:, 2], intensity=vertices[:, 2], colorscale='Cividis', name='background')

figure['layout']['sliders'] = [sliders_dict]
figure['data'] = [background,
                  go.Scatter3d(x=plane_plot_pts[0, :], y=plane_plot_pts[1, :], z=plane_plot_pts[2, :], mode='markers',
                               marker={'color': 'green'}, name='helicopter'),
                  go.Scatter3d(x=plane_plot_pts[0, :], y=plane_plot_pts[1, :], z=plane_plot_pts[2, :], mode='markers',
                               marker={'color': 'green'}, name='other'),
                  go.Cone(x=[plane_plot_pts[0, 0]], y=[plane_plot_pts[1, 0]], z=[plane_plot_pts[2, 0]],
                          u=[100.], v=[0.], w=[0.])
                  ]
fig = go.Figure(data=figure['data'], layout=go.Layout(**figure['layout']), frames=figure['frames'])
fig.show()

imgfig = go.Figure(
    data=[background,
          go.Scatter3d(x=plane_plot_pts[0, :], y=plane_plot_pts[1, :], z=plane_plot_pts[2, :], mode='markers',
                       marker={'color': 'green'}, name='helicopter'),
          go.Scatter3d(x=plane_plot_pts[0, :], y=plane_plot_pts[1, :], z=plane_plot_pts[2, :], mode='markers',
                       marker={'color': 'green'}, name='other'),
          go.Cone(x=[plane_plot_pts[0, 0]], y=[plane_plot_pts[1, 0]], z=[plane_plot_pts[2, 0]],
                  u=[100.], v=[0.], w=[0.])
          ],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(range=xrngs, autorange=False),
            yaxis=dict(range=yrngs, autorange=False),
            zaxis=dict(range=zrngs, autorange=False),
            aspectratio=dict(x=1, y=1, z=1)),
        title="Scattering Point Cloud",
        updatemenus=[dict(
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
    ),
    frames=img_plot
)
imgfig.show()

meshfig = go.Figure(
    data=[background,
          go.Cone(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], u=normals[:, 0] * 100000,
                  v=normals[:, 1] * 100000, w=normals[:, 2] * 100000, name='normals')],
    layout=go.Layout(
        title='Mesh and Normals',
    )
)
meshfig.show()

print(f'Expected data rate of {data_rate:.2f} MB/s')
