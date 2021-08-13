import numpy as np
from SARParse import SDRParse
from useful_lib import db, findPowerOf2
from scipy.stats import kurtosis
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy


def clean(pulse, Fs=2e9, nperseg=512, noverlap=64, cb_sz=40):
    stft_data = stft(pulse, fs=Fs, nperseg=nperseg, noverlap=noverlap)[2]
    clean_data = stft_data + 0.0
    clean_data[abs(stft_data) > np.median(abs(stft_data)) + 3 * abs(stft_data).std()] = 1e-12
    clean_data[clean_data.shape[0] // 2 - cb_sz:clean_data.shape[0] // 2 + cb_sz, :] = 0
    clean_pulse = istft(clean_data, fs=Fs, input_onesided=False, nperseg=nperseg, noverlap=noverlap)[1]
    return stft_data, clean_data, clean_pulse


fnme = '/data5/SAR_DATA/2021/06232021/SAR_06232021_092747.sar'
fs = 2e9
upsample = 2
band_removal = 780e3
npulses_chunk = 256
mempool = cupy.get_default_memory_pool()

sdr = SDRParse(fnme)
fft_sz = findPowerOf2(sdr.nsam) * upsample
band_sz = int(sdr.xml['Bandwidth_Hz'] / fs * fft_sz)
stopband_sz = int(np.ceil(band_removal / fs * fft_sz))
thresh = np.zeros((sdr.nframes,))
overall_av = np.zeros((fft_sz,), dtype=np.complex128)

for pdata, pnum, att, sys_time in tqdm(sdr.getPulseGen(npulses_chunk), total=sdr.nframes//npulses_chunk):
    fft_gpu = cupy.array(pdata, dtype=np.complex128)
    fft_gpu = cupy.fft.fft(fft_gpu, axis=0, n=fft_sz)
    cupy.cuda.Device().synchronize()
    fftdata = fft_gpu.get()
    overall_av = overall_av + np.sum(db(fftdata), axis=1) / sdr.nframes
    ex = np.concatenate((db(fftdata[-band_sz // 2:-stopband_sz, :]), db(fftdata[stopband_sz:band_sz // 2, :])), axis=0)
    thresh[pnum] = kurtosis(ex, axis=0)
    del fft_gpu
mempool.free_all_blocks()

cutoff = np.median(thresh) + 4 * thresh.std()
bad_pulses = np.arange(sdr.nframes)[thresh > cutoff]
ex_pulses = np.arange(sdr.nframes)[thresh < np.median(thresh)]
ndisps = 0
for p in bad_pulses:
    pulse = sdr.getPulse(p)
    stft_data, clean_data, clean_pulse = clean(pulse)
    plt.figure('Bad Pulse {}'.format(p))
    plt.subplot(2, 2, 1)
    plt.imshow(db(stft_data))
    plt.axis('tight')
    plt.subplot(2, 2, 3)
    plt.magnitude_spectrum(pulse, pad_to=fft_sz, window=lambda x: x)
    plt.subplot(2, 2, 2)
    plt.imshow(db(clean_data))
    plt.axis('tight')
    plt.subplot(2, 2, 4)
    plt.magnitude_spectrum(clean_pulse, pad_to=fft_sz, window=lambda x: x)
    ndisps += 1
    if ndisps >= 3:
        break

ndisps = 0
for p in ex_pulses:
    pulse = sdr.getPulse(p)
    stft_data, clean_data, clean_pulse = clean(pulse)
    plt.figure('Good Pulse {}'.format(p))
    plt.subplot(2, 2, 1)
    plt.imshow(db(stft_data))
    plt.axis('tight')
    plt.subplot(2, 2, 3)
    plt.magnitude_spectrum(pulse, pad_to=fft_sz, window=lambda x: x)
    plt.subplot(2, 2, 2)
    plt.imshow(db(clean_data))
    plt.axis('tight')
    plt.subplot(2, 2, 4)
    plt.magnitude_spectrum(clean_pulse, pad_to=fft_sz, window=lambda x: x)
    ndisps += 1
    if ndisps >= 2:
        break

plt.figure('Threshold')
plt.plot(thresh)
plt.hlines(cutoff, 0, len(thresh), color='g')

plt.figure('Averages')
plt.plot(np.fft.fftshift(np.fft.fftfreq(fft_sz, 1/fs))[:-1], np.fft.fftshift(overall_av)[:-1])

cal_av = np.zeros_like(sdr.getPulse(0))
for c in sdr.cal_num:
    cal_av = cal_av + sdr.getPulse(c, is_cal=True) / sdr.cal_num[-1]

plt.figure('Cal Pulse')
plt.magnitude_spectrum(cal_av, pad_to=fft_sz, window=lambda x: x)

match_filt = np.fft.fft(cal_av, fft_sz).conj().T
match_filt[band_sz // 2:-band_sz // 2] = 0

match_pulse = np.fft.ifft(np.fft.fft(sdr.getPulse(ex_pulses[-1]), fft_sz) * match_filt)
rc_stft, clean_rc, clean_pulse = clean(match_pulse)
plt.figure('RC Pulse STFT')
plt.imshow(np.fft.fftshift(db(rc_stft), axes=0))
plt.axis('tight')
plt.figure('RC Pulse Clean STFT')
plt.imshow(np.fft.fftshift(db(clean_rc), axes=0))
plt.axis('tight')
