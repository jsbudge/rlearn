import cmath
import math
from numba import cuda, njit
import numpy as np

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(device=True)
def interp(x, y, tt, bg):
    x0 = int(x)
    y0 = int(y)
    x1 = int(x0 + 1 if x - x0 >= 0 else -1)
    y1 = int(y0 + 1 if y - y0 >= 0 else -1)
    xdiff = x - x0
    ydiff = y - y0
    return bg[tt, x1, y1].real * xdiff * ydiff + bg[tt, x1, y0].real * xdiff * (1 - ydiff) + bg[tt, x0, y1].real * \
         (1 - xdiff) * ydiff + bg[tt, x0, y0].real * (1 - xdiff) * (1 - ydiff)


@cuda.jit(
    'void(float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], float64[:], float64[:], ' +
    'float64[:, :], float64[:, :], float64[:])')
def genRangeProfile(pathrx, pathtx, gp, bg, pan, el, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gp.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gp[samp_point, 0]
        ty = gp[samp_point, 1]

        # Calc out wave height
        x_i = tx % params[6] / params[6] * bg.shape[1]
        y_i = ty % params[7] / params[7] * bg.shape[2]
        tz = interp(x_i, y_i, tt, bg)

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = tx - pathtx[0, tt]
        s_ty = ty - pathtx[1, tt]
        s_tz = tz - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        s_rx = tx - pathrx[0, tt]
        s_ry = ty - pathrx[1, tt]
        s_rz = tz - pathrx[2, tt]
        rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
        rng = (rngtx + rngrx)
        rng_bin = (rng / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            # Cross product of dx and dy to get normal vector
            tz_v1 = interp(x_i + .01, y_i, tt, bg)
            tz_v2 = interp(x_i, y_i + .01, tt, bg)
            # l2 norm for normalization
            nnorm = math.sqrt((.01 * tz_v1) * (.01 * tz_v1) + (.01 * tz_v2) * (.01 * tz_v2) + .0001 * .0001)
            rnorm = (-.01 * tz_v1 * s_tx - .01 * tz_v2 * s_ty + .0001 * s_tz) / \
                    (rngtx * nnorm)
            # Reflection of wave
            ref_x = 2 * rnorm * -.01 * tz_v1 / nnorm - s_tx / rngtx
            ref_y = 2 * rnorm * -.01 * tz_v2 / nnorm - s_ty / rngtx
            ref_z = 2 * rnorm * .0001 / nnorm - s_tz / rngtx
            # Dot product of wave with Rx vector
            gv = ref_x * s_rx + ref_y * s_ry + ref_z * s_rz
            if gv < 0:
                gv = 0
            el_tx = math.asin(-s_tz / rngtx)
            az_tx = math.atan2(s_tx, s_ty)
            eldiff = diff(el_tx, el[tt])
            azdiff = diff(az_tx, pan[tt])
            tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            el_rx = math.asin(-s_rz / rngrx)
            az_rx = math.atan2(s_rx, s_ry)
            eldiff = diff(el_rx, el[tt])
            azdiff = diff(az_rx, pan[tt])
            rx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
            rx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
            att = tx_elpat * tx_azpat * rx_elpat * rx_azpat
            acc_val = gv * att * cmath.exp(-1j * wavenumber * rng) * 1 / (rng * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit('void(float64[:, :], float64[:, :], float64[:, :, :], float64[:], float64[:], ' +
          'float64[:, :], float64[:, :], float64[:])')
def genSubProfile(pathrx, pathtx, subs, pan, el, pd_r, pd_i, params):
    tt, subnum = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and subnum < subs.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        sub_x = subs[subnum, tt, 1]
        sub_y = subs[subnum, tt, 2]
        sub_cos = subs[subnum, tt, 3]
        sub_sin = subs[subnum, tt, 4]
        spow = subs[subnum, tt, 0] * 2000
        sub_z = subs[subnum, tt, 0]

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        xpts = 5
        ypts = 3
        for n in range(xpts):
            for m in range(ypts):
                shift_x = sub_x + (n - xpts // 2) / (xpts // 2) * 10.0 * sub_sin
                shift_y = sub_y + 2.5 / 10.0 * math.sqrt(
                    10.0 * 10.0 - ((n - xpts // 2) / (xpts // 2) * 10.0) *
                    ((n - xpts // 2) / (xpts // 2) * 10.0)) * sub_cos * (m - ypts // 2) / (ypts // 2)
                s_tx = shift_x - pathtx[0, tt]
                s_ty = shift_y - pathtx[1, tt]
                s_tz = sub_z - pathtx[2, tt]
                rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
                s_rx = shift_x - pathrx[0, tt]
                s_ry = shift_y - pathrx[1, tt]
                s_rz = sub_z - pathrx[2, tt]
                rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
                rng = (rngtx + rngrx)
                rng_bin = (rng / c0 - 2 * params[3]) * params[4]
                but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
                if n_samples > but > 0:
                    el_tx = math.asin(-s_tz / rngtx)
                    az_tx = math.atan2(s_tx, s_ty)
                    eldiff = diff(el_tx, el[tt])
                    azdiff = diff(az_tx, pan[tt])
                    tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
                    tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
                    el_rx = math.asin(-s_rz / rngrx)
                    az_rx = math.atan2(s_rx, s_ry)
                    eldiff = diff(el_rx, el[tt])
                    azdiff = diff(az_rx, pan[tt])
                    rx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
                    rx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
                    att = tx_elpat * tx_azpat * rx_elpat * rx_azpat
                    acc_val = spow * att * cmath.exp(-1j * wavenumber * rng) * 1 / (rng * rng)
                    cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
                    cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@cuda.jit()
def getDetectionCheck(pathtx, subs, pan, el, pd_r, pd_i, params):
    tt, subnum = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and subnum < subs.shape[0]:
        # Load in all the parameters that don't change
        wavenumber = 2 * np.pi / params[2]

        sub_x = subs[subnum, tt, 1]
        sub_y = subs[subnum, tt, 2]
        sub_z = subs[subnum, tt, 0]

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = sub_x - pathtx[0, tt]
        s_ty = sub_y - pathtx[1, tt]
        s_tz = sub_z - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        el_tx = math.asin(-s_tz / rngtx)
        az_tx = math.atan2(s_tx, s_ty)
        eldiff = diff(el_tx, el[tt])
        azdiff = diff(az_tx, pan[tt])
        tx_elpat = abs(math.sin(params[0] * eldiff) / (params[0] * eldiff)) if eldiff != 0 else 1
        tx_azpat = abs(math.sin(params[1] * azdiff) / (params[1] * azdiff)) if azdiff != 0 else 1
        att = tx_elpat * tx_azpat
        acc_val = att * cmath.exp(1j * wavenumber * rngtx) * 1 / (rngtx * rngtx)

        # Find exact spot in detection chunk to place pulse
        ptt = int((params[5] + rngtx / c0) * params[4])
        pt_s = ptt // params[6]
        pt_f = int(ptt % params[6])
        cuda.atomic.add(pd_r, (pt_f, pt_s), acc_val.real)
        cuda.atomic.add(pd_i, (pt_f, pt_s), acc_val.imag)