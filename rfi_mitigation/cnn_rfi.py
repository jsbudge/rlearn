import numpy as np
from rawparser import loadXMLFile, getRawSDRParams
from useful_lib import findAllFilenames, findPowerOf2, gaus, db
from SARParse import SDRParse
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten
import keras
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


def loadRFITrainingData(inp_fnme, inp_ch_sz=64, n_chs=None):
    with open(inp_fnme, 'rb') as f:
        inp_nframes = np.fromfile(f, 'uint32', 1, '')[0]
        nsamples = np.fromfile(f, 'uint32', 1, '')[0]
        atts = np.fromfile(f, 'int8', inp_nframes, '')
        sys_time = np.fromfile(f, 'double', inp_nframes, '')
        n_iters = inp_nframes if n_chs is None else inp_ch_sz * n_chs
        for n in range(0, n_iters, inp_ch_sz):
            inp_pdata = np.zeros((min(inp_ch_sz, inp_nframes - n), nsamples), dtype=np.complex128)
            for p in range(min(inp_ch_sz, inp_nframes - n)):
                data = np.fromfile(f, 'int16', nsamples * 2, '')
                inp_pdata[p, :] = (data[0::2] + 1j * data[1::2]) * (10 ** (atts[n * inp_ch_sz + p] / 20))
            yield inp_nframes, nsamples, inp_pdata


def loadRFILabels(inp_fnme):
    with open(inp_fnme, 'rb') as f:
        isRFI = np.fromfile(f, 'int8', -1, '')
    return isRFI.astype(bool)


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

fnme = '/data5/SAR_DATA/2021/06252021/06252021/SAR_06252021_091624.sar'
dat_fnme = './rfi_sim.dat'
net_sz = findPowerOf2(39296) * 1
ch_sz = 256
# sar = SDRParse(fnme)

# Load in the file data
yt = loadRFILabels(dat_fnme[:-4] + '_rfi.dat')
# yt = OneHotEncoder(sparse=False).fit_transform(yt.reshape(-1, 1))
dgen = loadRFITrainingData(dat_fnme, inp_ch_sz=ch_sz, n_chs=4)

print(f'{sum(yt) / len(yt) * 100}% RFI pulses.')

# Load in the model
inp = Input(shape=(net_sz,))
dense = Dense(150, activation='relu')(inp)
dense = Dense(50, activation='relu')(dense)
outp = Dense(1, activation='softmax')(dense)

model = Model(inputs=inp, outputs=outp)
model.compile(optimizer='adam', loss=['binary_crossentropy'])

for ch, (nframes, nsam, pdata) in enumerate(dgen):
    pdata = db(np.fft.fft(pdata, net_sz, axis=1))
    mhist = model.fit(pdata, yt[ch:ch+ch_sz], epochs=15)










