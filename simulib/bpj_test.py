import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject, cpudiff
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, basinhopping
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
from scipy.signal import medfilt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm
from SDRParsing import SDRParse, load, findAllFilenames
from SARParsing import SARParse

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def spline(x, k0, y0, x0, x1, x2):
    # Piecewise spline for smooth transitioning between CPIs
    y1 = x[0]
    y2 = x[1]
    k1 = x[2]
    k2 = x[3]
    tx0 = (np.arange(x0, x1) - x0) / (x1 - x0)
    q0 = (1 - tx0) * y0 + tx0 * y1 + tx0 * (1 - tx0) * ((1 - tx0) * (k0 * (x1 - x0) - (y1 - y0)) +
                                                        tx0 * (-k1 * (x1 - x0) + (y1 - y0)))
    tx1 = (np.arange(x1, x2) - x1) / (x2 - x1)
    q1 = (1 - tx1) * y1 + tx1 * y2 + tx1 * (1 - tx1) * ((1 - tx1) * (k1 * (x2 - x1) - (y2 - y1)) +
                                                        tx1 * (-k2 * (x2 - x1) + (y2 - y1)))
    return np.concatenate((q0, q1))


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

# This is the file used to backproject data
# bg_file = '/data5/SAR_DATA/2021/05052021/SAR_05052021_112647.sar'
# bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
# bg_file = '/data5/SAR_DATA/2022/Redstone/SAR_08122022_170753.sar'
# bg_file = '/data6/SAR_DATA/2023/06202023/SAR_06202023_135617.sar'
# bg_file = '/data6/Tower_Redo_Again/tower_redo_SAR_03292023_120731.sar'
# bg_file = '/data5/SAR_DATA/2022/09272022/SAR_09272022_103053.sar'
# bg_file = '/data5/SAR_DATA/2019/08072019/SAR_08072019_100120.sar'
bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_123311.sar'
# bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_122801.sar'
# bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_123050.sar'
upsample = 4
poly_num = 1
use_rcorr = False
use_aps_debug = True
rotate_grid = True
use_ecef = True
ipr_mode = False
cpi_len = 128
plp = 0
partial_pulse_percent = .2
debug = True
pts_per_m = 1
grid_width = 200
grid_height = 200
channel = 0
fdelay = 6.3
origin = (40.136013, -111.661530, 1382)

nbpj_pts = (int(grid_width // pts_per_m), int(grid_height // pts_per_m))

files = findAllFilenames(bg_file, use_debug=True, exact_matches=False)

print('Loading SDR file...')
is_sar = False
try:
    sdr = load(bg_file)
except KeyError:
    is_sar = True
    print('Using SlimSAR parser instead.')
    sdr = SARParse(bg_file)
    sdr.ash = sdr.loadASH('/data5/SAR_DATA/2019/08072019/SAR_08072019_100120RVV_950000_95.ash')

if origin is None:
    try:
        origin = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                  getElevation((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'])))
        heading = sdr.gim.initial_course_angle
    except TypeError:
        heading = sdr.gim.initial_course_angle
        pt = (sdr.gps_data['lat'].values[0], sdr.gps_data['lon'].values[0])
        alt = getElevation(pt)
        nrange = ((sdr[channel].receive_on_TAC - sdr[channel].transmit_on_TAC) / TAC -
                  sdr[channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        frange = ((sdr[channel].receive_off_TAC - sdr[channel].transmit_on_TAC) / TAC -
                  sdr[channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        mrange = (nrange + frange) / 2
        origin = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                         (pt[0], pt[1], alt))

sdr.gimbal['systime'] += TAC * .01

bg = SDREnvironment(sdr_file=sdr)

if ipr_mode:
    print('IPR mode...', end='')
    dx_shift = np.inf
    iters = 0
    while dx_shift > .5 and iters < 10:
        oe, on, ou = llh2enu(*origin, bg.ref)
        ix, iy = bg.getIndex(oe, on).astype(int)
        minigrid = bg.refgrid[ix-int(50 / bg.rps):ix+int(50 / bg.rps), iy-int(50 / bg.cps):iy+int(50 / bg.cps)]
        nx, ny = np.where(minigrid == minigrid.max())
        origin = enu2llh(*bg.getPos(nx[0] + ix-int(50 / bg.rps), ny[0] + iy-int(50 / bg.cps)), 0, bg.ref)
        dx_shift = np.sqrt(((nx[0] - 50 / bg.rps) * bg.rps)**2 + ((ny[0] - 50 / bg.cps) * bg.cps)**2)
        iters += 1
    print(f'Origin set to {origin}')
ref_llh = bg.ref

# Generate a platform
print('Generating platform...', end='')

# Bunch of debug files used for testing. These are not necessary for backprojection.
gps_check = True
try:
    for key, val in files.items():
        if f'Channel_{channel + 1}' in key:
            if 'preCorrections' in key:
                preCorr = loadPreCorrectionsGPSData(val)
            elif 'postCorrections' in key:
                postCorr = loadPostCorrectionsGPSData(val)
        elif 'Gimbal' in key:
            gimbal_debug = val
            gimbal_data = loadGimbalData(val)
        elif 'GPSDataPreJump' in key:
            rawGPS = loadGPSData(val)
        elif 'GPSDataPostJump' in key:
            jumpCorrGPS = loadGPSData(val)
except FileNotFoundError:
    gps_check = False
    use_aps_debug = False
    print('Failed to find APS GPS debug outputs.')

if use_aps_debug:
    rp = SDRPlatform(sdr, ref_llh, channel=channel, gps_debug=postCorr, gimbal_debug=gimbal_debug,
                     gps_replace=jumpCorrGPS, use_ecef=use_ecef)
    # rp = SDRPlatform(sdr, ref_llh, channel=channel, gimbal_debug=gimbal_debug)
else:
    rp = SDRPlatform(sdr, ref_llh, channel=channel)

# rp.az_half_bw = 5 * DTR

# Atmospheric modeling params
Ns = 313
Nb = 66.65
hb = 12192
if rp.pos(rp.gpst)[2, :].mean() > hb:
    hb = rp.pos(rp.gpst)[2, :].mean() + 1000.
    Nb = 105 * np.exp(-(hb - 9000) / 7023)

# Get reference data
# flight = rp.pos(postCorr['sec'])
fs = sdr[channel].fs
bwidth = sdr[channel].bw
fc = sdr[channel].fc
print('Done.')

# Generate values needed for backprojection
print('Calculating grid parameters...')
# General calculations for slant ranges, etc.
# plat_height = rp.pos(rp.gpst)[2, :].mean()
nr = rp.calcPulseLength(fdelay, plp, use_tac=True)
nsam = rp.calcNumSamples(fdelay, plp)
ranges = rp.calcRangeBins(fdelay, upsample, plp)
granges = ranges * np.cos(rp.dep_ang)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample

# Chirp and matched filter calculations
if is_sar:
    bpj_wavelength = c0 / (fc - bwidth / 2 - 5e6)
    mfilt = sdr.genMatchedFilter(0)
else:
    try:
        bpj_wavelength = c0 / (fc - bwidth / 2 - sdr[channel].xml['DC_Offset_MHz'] * 1e6) \
            if sdr[channel].xml['Offset_Video_Enabled'] == 'True' else c0 / fc
    except KeyError as e:
        f'Could not find {e}'
        bpj_wavelength = c0 / (fc - bwidth / 2 - 5e6)

    mfilt = GetAdvMatchedFilter(sdr[channel], fft_len=fft_len)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
rbins_gpu = cupy.array(ranges, dtype=np.float64)

# Calculate out points on the ground
gx, gy, gz = bg.getGrid(origin, grid_width, grid_height, nbpj_pts, bg.heading if rotate_grid else 0)
gx_gpu = cupy.array(gx, dtype=np.float64)
gy_gpu = cupy.array(gy, dtype=np.float64)
gz_gpu = cupy.array(gz, dtype=np.float64)

if debug:
    pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    # pts_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
    # angs_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, nbpj_pts[0] // threads_per_block[0] + 1), nbpj_pts[1] // threads_per_block[1] + 1)
# rng_states = create_xoroshiro128p_states(triangles.shape[0], seed=10)

# Run through loop to get data simulated
data_t = sdr[channel].pulse_time
idx_t = sdr[channel].frame_num
test = None
freqs = np.fft.fftfreq(fft_len, 1 / sdr[0].fs)
print('Backprojecting...')
pulse_pos = 0
# Data blocks for imaging
bpj_truedata = np.zeros(nbpj_pts, dtype=np.complex128)
for tidx, frames in tqdm(enumerate(idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)),
                         total=len(data_t) // cpi_len + 1):
    ts = data_t[tidx * cpi_len + np.arange(len(frames))]
    tmp_len = len(ts)

    if not np.all(cpudiff(np.arctan2(-rp.pos(ts)[1, :], rp.pos(ts)[0, :]), rp.pan(ts)) -
                  rp.az_half_bw < 0):
        continue
    panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
    postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)
    bpj_grid = cupy.zeros(nbpj_pts, dtype=np.complex128)

    # Armin Doerry's corrections for atmospheric changes to the speed of light
    Hb = (hb - ref_llh[2]) / np.log(Ns / Nb)
    rcatmos = (1 + (Hb * 10e-6 * Ns) / rp.pos(ts)[2, :] *
               (1 - np.exp(-rp.pos(ts)[2, :] / Hb))) ** -1
    if use_rcorr:
        r_corr_gpu = cupy.array(rcatmos, dtype=np.float64)
    else:
        r_corr_gpu = cupy.array(np.ones_like(ts), dtype=np.float64)

    # Reset the grid for truth data
    rtdata = cupy.fft.fft(cupy.array(sdr.getPulses(frames, channel),
                                     dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, :tmp_len]
    upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * upsample, :]
    cupy.cuda.Device().synchronize()

    backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                            elrx_gpu, panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                            bpj_wavelength, ranges[0] / c0, rp.fs * upsample, bwidth, rp.az_half_bw,
                                            rp.el_half_bw, poly_num, pts_debug, angs_debug, debug, r_corr_gpu)
    cupy.cuda.Device().synchronize()

    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp = rp.pos(ts[-1])
        test = rtdata.get()
        angd = angs_debug.get()
        locd = pts_debug.get()

    # bpj_traces.append(go.Heatmap(z=db(bpj_grid.get())))
    bpj_truedata += bpj_grid.get()

del panrx_gpu
del postx_gpu
del posrx_gpu
del elrx_gpu
del rtdata
del upsample_data
del bpj_grid
del r_corr_gpu
# del shift

del rbins_gpu
del gx_gpu
del gy_gpu
del gz_gpu

# Apply range roll-off compensation to final image
mag_data = np.sqrt(abs(bpj_truedata))
brightness_raw = np.median(np.sqrt(abs(bpj_truedata)), axis=0)
brightness_curve = np.polyval(np.polyfit(np.arange(bpj_truedata.shape[0]), brightness_raw, 4),
                              np.arange(bpj_truedata.shape[0]))
brightness_curve /= brightness_curve.max()
brightness_curve = 1. / brightness_curve
mag_data *= np.outer(np.ones(mag_data.shape[1]), brightness_curve)

if test is not None:
    plt.figure('Doppler data')
    plt.imshow(np.fft.fftshift(db(np.fft.fft(test, axis=1)), axes=1),
               extent=[-sdr[channel].prf / 2, sdr[channel].prf / 2, ranges[-1], ranges[0]])
    plt.axis('tight')

plt.figure('IMSHOW backprojected data')
plt.imshow(db(mag_data), origin='lower')
plt.axis('tight')

try:
    if (nbpj_pts[0] * nbpj_pts[1]) < 400 ** 2:
        cx, cy, cz = bg.getGrid(origin, grid_width, grid_height, nbpj_pts)

        fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
        fig.add_scatter3d(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), mode='markers')
        fig.show()

    plt.figure('IMSHOW truth data')
    plt.imshow(db(bg.refgrid), origin='lower')
    plt.axis('tight')
except Exception as e:
    print(f'Error in generating background image: {e}')

from sklearn.preprocessing import QuantileTransformer

# mag_data = np.sqrt(abs(sdr.loadASI(sdr.files['asi'][0])))
nbits = 256
plot_data_init = QuantileTransformer(output_distribution='normal').fit(
    mag_data[mag_data > 0].reshape(-1, 1)).transform(mag_data.reshape(-1, 1)).reshape(mag_data.shape)
plot_data = plot_data_init
max_bin = 3
hist_counts, hist_bins = \
    np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
while hist_counts[-1] == 0:
    max_bin -= .2
    hist_counts, hist_bins = \
        np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
scaled_data = np.digitize(plot_data_init, hist_bins)

# px.imshow(db(mag_data), color_continuous_scale=px.colors.sequential.gray).show()
px.imshow(np.fliplr(scaled_data), color_continuous_scale=px.colors.sequential.gray, zmin=0, zmax=nbits).show()
plt.figure('Image Histogram')
plt.plot(hist_bins[1:], hist_counts)

# Get IPR cut
if ipr_mode:
    db_bpj = db(bpj_truedata).T
    db_bpj -= np.max(db_bpj)
    mx = np.where(db_bpj == db_bpj.max())
    ipr_gridsz = min(db_bpj.shape[0] - mx[0][0], mx[0][0], db_bpj.shape[1] - mx[1][0], mx[1][0])

    cutfig = make_subplots(rows=2, cols=2, subplot_titles=(f'Azimuth', 'Contour', 'Magnitude', 'Range'))
    cutfig.append_trace(
        go.Scatter(x=np.arange(ipr_gridsz * 2) * grid_width / nbpj_pts[0] - ipr_gridsz * grid_width / nbpj_pts[0],
                   y=db_bpj[mx[0][0], mx[1][0] - ipr_gridsz:mx[1][0] + ipr_gridsz] - db_bpj[mx[0][0],
                                                                                     mx[1][0] - ipr_gridsz:mx[1][
                                                                                                               0] + ipr_gridsz].max(),
                   mode='lines', showlegend=False), row=1, col=1)
    cutfig.append_trace(
        go.Scatter(x=np.arange(ipr_gridsz * 2) * grid_height / nbpj_pts[1] - ipr_gridsz * grid_height / nbpj_pts[1],
                   y=db_bpj[mx[0][0] - ipr_gridsz:mx[0][0] + ipr_gridsz, mx[1][0]] - db_bpj[mx[0][0] - ipr_gridsz:mx[0][
                                                                                                                      0] + ipr_gridsz,
                                                                                     mx[1][0]].max(),
                   mode='lines', showlegend=False), row=2, col=2)
    cutfig.append_trace(
        go.Heatmap(z=db_bpj, colorscale=px.colors.sequential.gray), row=2, col=1)
    cutfig.append_trace(
        go.Contour(z=db_bpj, contours_coloring='lines', line_width=2, contours=dict(
            start=0,
            end=-60,
            size=10,
            showlabels=True,
        ), showscale=False), row=1, col=2)
    cutfig.show()

if gps_check:
    re, rn, ru = llh2enu(rawGPS['lat'], rawGPS['lon'], rawGPS['alt'], rp.origin)
    ge, gn, gu = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], rp.origin)
    plt.figure('Raw GPS data')
    plt.subplot(2, 2, 1)
    plt.title('E')
    plt.plot(rawGPS['gps_ms'], re)
    plt.plot(sdr.gps_data.index, ge)
    plt.subplot(2, 2, 2)
    plt.title('N')
    plt.plot(rawGPS['gps_ms'], rn)
    plt.plot(sdr.gps_data.index, gn)
    plt.subplot(2, 2, 3)
    plt.title('U')
    plt.plot(rawGPS['gps_ms'], ru)
    plt.plot(sdr.gps_data.index, gu)

    re, rn, ru = llh2enu(rawGPS['lat'], rawGPS['lon'], rawGPS['alt'], rp.origin)
    ge, gn, gu = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], rp.origin)
    plt.figure('Diff Raw GPS data')
    plt.subplot(2, 2, 1)
    plt.title('E')
    plt.plot(rawGPS['gps_ms'], re - ge)
    plt.subplot(2, 2, 2)
    plt.title('N')
    plt.plot(rawGPS['gps_ms'], rn - gn)
    plt.subplot(2, 2, 3)
    plt.title('U')
    plt.plot(rawGPS['gps_ms'], ru - gu)

    rerx = rp.rxpos(postCorr['sec'])[0, :]
    rnrx = rp.rxpos(postCorr['sec'])[1, :]
    rurx = rp.rxpos(postCorr['sec'])[2, :]
    gnrx = postCorr['rx_lat'] * postCorr['latConv'] - origin[0] * postCorr['latConv']
    gerx = postCorr['rx_lon'] * postCorr['lonConv'] - origin[1] * postCorr['lonConv']
    gurx = postCorr['rx_alt'] - origin[2]
    gerx, gnrx, gurx = llh2enu(postCorr['rx_lat'], postCorr['rx_lon'], postCorr['rx_alt'], origin)
    retx = rp.txpos(postCorr['sec'])[0, :]
    rntx = rp.txpos(postCorr['sec'])[1, :]
    rutx = rp.txpos(postCorr['sec'])[2, :]
    gntx = postCorr['tx_lat'] * postCorr['latConv'] - origin[0] * postCorr['latConv']
    getx = postCorr['tx_lon'] * postCorr['lonConv'] - origin[1] * postCorr['lonConv']
    gutx = postCorr['tx_alt'] - origin[2]
    getx, gntx, gutx = llh2enu(postCorr['tx_lat'], postCorr['tx_lon'], postCorr['tx_alt'], origin)
    plt.figure('ENU diff')
    plt.subplot(2, 2, 1)
    plt.title('E')
    plt.plot(postCorr['sec'], rerx - gerx)
    plt.plot(postCorr['sec'], retx - getx)
    plt.subplot(2, 2, 2)
    plt.title('N')
    plt.plot(postCorr['sec'], rnrx - gnrx)
    plt.plot(postCorr['sec'], rntx - gntx)
    plt.subplot(2, 2, 3)
    plt.title('U')
    plt.plot(postCorr['sec'], rurx - gurx)
    plt.plot(postCorr['sec'], rutx - gutx)
    plt.legend(['Rx', 'Tx'])

    rp_r = rp.att(preCorr['sec'])[0, :]
    rp_p = rp.att(preCorr['sec'])[1, :]
    rp_y = rp.att(preCorr['sec'])[2, :]
    plt.figure('rpy')
    plt.subplot(2, 2, 1)
    plt.title('r')
    plt.plot(preCorr['sec'], rp_r)
    plt.plot(preCorr['sec'], preCorr['r'])
    plt.plot(rawGPS['gps_ms'], rawGPS['r'])
    plt.subplot(2, 2, 2)
    plt.title('p')
    plt.plot(preCorr['sec'], rp_p)
    plt.plot(preCorr['sec'], preCorr['p'])
    plt.plot(rawGPS['gps_ms'], rawGPS['p'])
    plt.subplot(2, 2, 3)
    plt.title('y')
    plt.plot(preCorr['sec'], rp_y)
    plt.plot(postCorr['sec'], postCorr['az'])
    plt.plot(preCorr['sec'], preCorr['az'])
    plt.legend(['sdr', 'interp_sdr', 'pre', 'raw'])

    times = np.interp(gimbal_data['systime'], sdr.gps_data['systime'], sdr.gps_data.index)
    plt.figure('Gimbal')
    plt.subplot(2, 1, 1)
    plt.title('Pan')
    plt.plot(times, gimbal_data['pan'])
    plt.plot(times, sdr.gimbal['pan'])
    plt.subplot(2, 1, 2)
    plt.title('Tilt')
    plt.plot(times, gimbal_data['tilt'])
    plt.plot(times, sdr.gimbal['tilt'])

    gerx, gnrx, gurx = llh2enu(postCorr['rx_lat'], postCorr['rx_lon'], postCorr['rx_alt'], rp.origin)
    gerx = np.interp(rawGPS['gps_ms'][rawGPS['gps_ms'] > postCorr['sec'][0]], postCorr['sec'], gerx)
    gnrx = np.interp(rawGPS['gps_ms'][rawGPS['gps_ms'] > postCorr['sec'][0]], postCorr['sec'], gnrx)
    gurx = np.interp(rawGPS['gps_ms'][rawGPS['gps_ms'] > postCorr['sec'][0]], postCorr['sec'], gurx)
    re, rn, ru = llh2enu(rawGPS['lat'][rawGPS['gps_ms'] > postCorr['sec'][0]],
                         rawGPS['lon'][rawGPS['gps_ms'] > postCorr['sec'][0]],
                         rawGPS['alt'][rawGPS['gps_ms'] > postCorr['sec'][0]], rp.origin)
    je, jn, ju = llh2enu(jumpCorrGPS['lat'][jumpCorrGPS['gps_ms'] > postCorr['sec'][0]],
                         jumpCorrGPS['lon'][jumpCorrGPS['gps_ms'] > postCorr['sec'][0]],
                         jumpCorrGPS['alt'][jumpCorrGPS['gps_ms'] > postCorr['sec'][0]], rp.origin)
    plt.figure('Lever arm ENU diff')
    plt.subplot(2, 2, 1)
    plt.title('E')
    plt.plot(rawGPS['gps_ms'][rawGPS['gps_ms'] > postCorr['sec'][0]], re - gerx)
    plt.plot(jumpCorrGPS['gps_ms'][jumpCorrGPS['gps_ms'] > postCorr['sec'][0]], je - gerx)
    plt.plot(rawGPS['gps_ms'], rp.pos(rawGPS['gps_ms'])[0, :] - rp.rxpos(rawGPS['gps_ms'])[0, :])
    plt.subplot(2, 2, 2)
    plt.title('N')
    plt.plot(rawGPS['gps_ms'][rawGPS['gps_ms'] > postCorr['sec'][0]], rn - gnrx)
    plt.plot(jumpCorrGPS['gps_ms'][jumpCorrGPS['gps_ms'] > postCorr['sec'][0]], jn - gnrx)
    plt.plot(rawGPS['gps_ms'], rp.pos(rawGPS['gps_ms'])[1, :] - rp.rxpos(rawGPS['gps_ms'])[1, :])
    plt.subplot(2, 2, 3)
    plt.title('U')
    plt.plot(rawGPS['gps_ms'][rawGPS['gps_ms'] > postCorr['sec'][0]], ru - gurx)
    plt.plot(jumpCorrGPS['gps_ms'][jumpCorrGPS['gps_ms'] > postCorr['sec'][0]], ju - gurx)
    plt.plot(rawGPS['gps_ms'], rp.pos(rawGPS['gps_ms'])[2, :] - rp.rxpos(rawGPS['gps_ms'])[2, :])
    plt.legend(['APS output', 'APS jump corrected', 'SAR output'])

plt.show()
