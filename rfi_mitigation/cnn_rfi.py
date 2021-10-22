import numpy as np
from rawparser import loadXMLFile, getRawSDRParams
from useful_lib import findAllFilenames, findPowerOf2, gaus, db
from SARParse import SDRParse
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tqdm import tqdm
# plt.switch_backend('Qt5Agg')


def msfe_loss(y_true, y_pred):

    fp = math_ops.logical_and(y_true == 0, y_pred >= .5)
    fn = math_ops.logical_and(y_true == 1, y_pred < .5)
    fp_sq = math_ops.reduce_mean(math_ops.square(y_true[fp] - y_pred[fp]), axis=-1)
    fn_sq = math_ops.reduce_mean(math_ops.square(y_true[fn] - y_pred[fn]), axis=-1)
    fp_sq = 0.0 if math_ops.is_nan(fp_sq) else fp_sq
    fn_sq = 0.0 if math_ops.is_nan(fn_sq) else fn_sq
    return fp_sq + fn_sq


def plotActivations(mdl, inp, lnum=2):
    louts = [layer.output for layer in mdl.layers]
    act_mdl = Model(inputs=mdl.input, outputs=louts)
    acts = act_mdl.predict(inp)
    plt_act = acts[lnum]
    if 'conv1d' in louts[lnum].name:
        # It's a convolution layer
        plt.figure(louts[lnum].name)
        grid_sz = int(np.sqrt(plt_act.shape[2]) + 1)
        for n in range(plt_act.shape[2]):
            plt.subplot(grid_sz, grid_sz, n+1)
            plt.plot(plt_act[0, :, n])
            plt.title(f'{n}')
    elif 'dense' in louts[lnum].name:
        plt.figure(louts[lnum].name)
        plt.plot(plt_act[0, :].flatten())
    elif 'input' in louts[lnum].name:
        plt.figure(louts[lnum].name)
        plt.plot(plt_act[0, :, :].flatten())
    else:
        print('Did not plot layer ' + louts[lnum].name)


def loadRFILabels(inp_fnme):
    with open(inp_fnme, 'rb') as f:
        isRFI = np.fromfile(f, 'int8', -1, '')
    return isRFI.astype(bool)


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

fnme = '/home/jeff/repo/mimo_simulator/SAR_09222021_163338SIM.sar'
dat_fnme = fnme[:-4] + '_rfi.dat'
ch_sz = 256

# Load in the file data
labels = loadRFILabels(dat_fnme)
labels = labels.reshape(len(labels), 1).astype(np.float32)
sdr_f = SDRParse(fnme)
net_sz = findPowerOf2(sdr_f[0].nsam) * 2
dgen = sdr_f.getPulseGen(ch_sz, 0)
pulse_percent = sum(labels) / len(labels) * 100

print(f'{pulse_percent[0]:.2f}% RFI pulses.')

# Load in the model
inp = Input(shape=(net_sz, 1,))
x = MaxPooling1D(2)(inp)
x = Conv1D(10, 256)(x)
x = AveragePooling1D(2)(x)
x = Conv1D(10, 128, activation='relu')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(.4)(x)
outp = Dense(1, activation='softmax')(x)

opt = keras.optimizers.Adam(learning_rate=1e-2)

model = Model(inputs=inp, outputs=outp)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Select our training pulses such that the class sizes are about even
dp = np.where(labels == 1)[0]
cp = np.random.permutation(np.where(labels == 0)[0])

# Remove some of the pulses for validation (~25%)
dp_s = dp[:int(len(dp) * .25)]
cp_s = cp[:int(len(dp) * .25)]
dp_t = dp[int(len(dp) * .25):]
cp_t = cp[int(len(dp) * .25):]

val_pulses = np.concatenate((dp_s, cp_s))

Xs = db(np.fft.fft(sdr_f.getPulses(val_pulses), net_sz, axis=0)).reshape((len(val_pulses), net_sz, 1))
ys = labels[val_pulses]

loss = []
val_loss = []
yp = []
ypt = []
clean_frame = None
dirty_frame = None
f_mu = None
f_std = None

n_iters = len(cp_t) // len(dp_t)

for it in tqdm(range(n_iters)):
    pulses = np.random.permutation(np.concatenate((dp_t[:64], cp_t[it * len(dp_t):it * len(dp_t) + 64])))
    Xt = db(np.fft.fft(sdr_f.getPulses(pulses), net_sz, axis=0)).reshape((len(pulses), net_sz, 1))
    if f_mu is None:
        f_mu = Xt.mean()
        f_std = Xt.std()
    Xt = (Xt - f_mu) / f_std
    yt = labels[pulses]
    if np.any(yt):
        weight = (sum(yt) / len(yt))[0]
        mhist = model.fit(Xt, yt, validation_data=(Xs, ys), epochs=15, verbose=0)
        if clean_frame is None:
            clean_frame = Xt[(yt == 0).flatten(), :, :][:1, :, :]
        if dirty_frame is None:
            dirty_frame = Xt[(yt == 1).flatten(), :, :][:1, :, :]
        yp = np.concatenate((yp, model.predict(Xs).flatten()))
        ypt = np.concatenate((ypt, ys.flatten()))
        loss.append(mhist.history['loss'][0])
        val_loss.append(mhist.history['val_loss'][0])

plt.figure('Losses')
plt.plot(loss)
plt.plot(val_loss)

print(confusion_matrix(ypt, yp))

for lay in range(len(model.layers)):
    plotActivations(model, dirty_frame, lay)
    plotActivations(model, clean_frame, lay)










