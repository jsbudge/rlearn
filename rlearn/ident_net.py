"""
Given a continuous set of sampled data, this should chop it into overlapping sections
and try to match them to any of a set of waveforms / just test if there's a signal involved
try to identify signal parameters and such
"""

import numpy as np
from tqdm import tqdm
import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    Dropout, GaussianNoise
from kapre import STFT, MagnitudeToDecibel, Magnitude
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1_l2
from wave_env import genPulse
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.signal import stft
from tftb.processing import WignerVilleDistribution
import matplotlib.pyplot as plt

c0 = 299792458.0
TAC = 125e6
fs = 2e9 / 4
DTR = np.pi / 180


def db(x):
    ret = abs(x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


def plotWeights(mdl, lnum=2, mdl_name=''):
    lw = mdl.layers[lnum].get_weights()
    if len(lw) > 0:
        lw = lw[0]
        lnm = mdl.layers[lnum].output.name
    else:
        return
    if 'conv2d' in lnm:
        # It's a convolution layer
        plt.figure(lnm.split('/')[1] + ' weights: ' + mdl_name)
        grid_sz = int(np.ceil(np.sqrt(lw.shape[3])))
        for n in range(lw.shape[3]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.imshow(lw[:, :, 0, n])
            plt.title(f'{n}')


def plotActivations(classifier, inp):
    layer_outputs = [layer.output for layer in classifier.layers]  # Extracts the outputs of the top 12 layers
    activation_model = keras.Model(inputs=classifier.input,
        outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(inp.reshape((1, len(inp), 1)))
    layer_names = []
    for layer in classifier.layers:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        if 'stft' in layer_name or 'conv' in layer_name or 'pooling' in layer_name:
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            grid_sz = int(np.ceil(np.sqrt(n_features)))
            plt.figure(layer_name)
            plt.grid(False)
            for n in range(n_features):
                plt.subplot(grid_sz, grid_sz, n+1)
                try:
                    plt.imshow(layer_activation[0, :, :, n], aspect='auto', cmap='viridis')
                except TypeError:
                    plt.imshow(db(layer_activation[0, :, :, n]), aspect='auto', cmap='viridis')
        elif 'dense' in layer_name:
            plt.figure(layer_name)
            plt.plot(layer_activation[0, ...])


# Base net params
dec_facs = [1]
batch_sz = 64
minp_sz = 40000
stft_sz = 512
band_limits = (10e6, fs / 2)
base_pl = 6.468e-6

segment_base_samp = minp_sz * dec_facs[-1]
segment_t0 = segment_base_samp / fs   # Segment time in secs
seg_sz = int(np.ceil(segment_t0 * fs))


def genModel(nsam):
    inp = Input(shape=(nsam, 1))
    lay = STFT(n_fft=stft_sz, win_length=stft_sz - (stft_sz % 100), hop_length=stft_sz // 2, window_name=None)(inp)
    lay = Magnitude()(lay)
    lay = BatchNormalization()(lay)
    lay = MaxPooling2D((2, 2))(lay)
    lay = Conv2D(30, (32, 32), activation=keras.layers.LeakyReLU(alpha=.1), activity_regularizer=l1_l2(1e-4),
                 kernel_regularizer=l1_l2(1e-3), bias_regularizer=l1_l2(1e-3))(lay)
    lay = Conv2D(30, (16, 16), activation=keras.layers.LeakyReLU(alpha=.1), activity_regularizer=l1_l2(1e-4),
                 kernel_regularizer=l1_l2(1e-3), bias_regularizer=l1_l2(1e-3))(lay)
    lay = Flatten()(lay)
    lay = Dropout(.4)(lay)
    lay = Dense(512, activation=keras.layers.LeakyReLU(alpha=.1), activity_regularizer=l1_l2(1e-4),
                kernel_regularizer=l1_l2(1e-3), bias_regularizer=l1_l2(1e-3))(lay)
    lay = GaussianNoise(1)(lay)
    outp = Dense(2, activation='softmax')(lay)
    return keras.Model(inputs=inp, outputs=outp)


# Generate models for different sampling rates
mdl = genModel(minp_sz)
mdl.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
ramp = np.linspace(0, 1, 100)
hist_loss = []
hist_val_loss = []
hist_acc = []
hist_val_acc = []

sig_on = True

for run in tqdm(range(2)):
    t0 = 0
    sig_t = 0
    Xt = []
    yt = []
    prf = np.random.rand() * 400 + 100
    # Make sure it's at least a microsecond long
    nr = int((np.random.rand() * (base_pl - 1e-6) + 1e-6) * fs)
    bw = np.random.rand() * (band_limits[1] - band_limits[0]) + band_limits[0]
    pcnt = 0
    while len(Xt) < batch_sz * 2:
        # Generate the data for this segment
        seg_data = np.random.normal(0, 1, seg_sz) + 1j * np.random.normal(0, 1, seg_sz)
        seg_truth = np.zeros(seg_sz)
        if pcnt > 10:
            prf = np.random.rand() * 400 + 100
            # Make sure it's at least a microsecond long
            nr = int((np.random.rand() * (base_pl - 1e-6) + 1e-6) * fs)
            bw = np.random.rand() * (band_limits[1] - band_limits[0]) + band_limits[0]
            pcnt = 0
        # First, get the signal we may or may not use here
        if sig_on:
            while sig_t < t0:
                sig_t += 1 / prf
            # Continue to pulse to the PRF during the segment length
            while sig_t < t0 + segment_t0:
                ns = int((sig_t - t0) * fs)
                seg_data[ns:min(seg_sz, ns + nr)] += \
                    genPulse(ramp, ramp, nr, nr / fs, 9.6e9, bw)[:min(seg_sz - ns, nr)]
                seg_truth[ns:min(seg_sz, ns + nr)] = 1
                sig_t += 1 / prf
                pcnt += 1

        # Run each model data, using different decimation factors
        bgn = seg_data
        if len(yt) == 0:
            if np.any(seg_truth):
                yt.append([True, False])
            else:
                yt.append([False, True])
            Xt.append(bgn)
        else:
            if np.any(seg_truth):
                if yt[-1][1]:
                    yt.append([True, False])
                    Xt.append(bgn)
            else:
                if yt[-1][0]:
                    yt.append([False, True])
                    Xt.append(bgn)
        t0 += segment_t0
    Xs = np.array(Xt)[batch_sz:, ...]
    ys = np.array(yt)[batch_sz:, ...]
    Xt = np.array(Xt)[:batch_sz, ...]
    yt = np.array(yt)[:batch_sz, ...]

    h = mdl.fit(Xt, yt, validation_data=(Xs, ys), epochs=20, callbacks=[TerminateOnNaN()])
    hist_loss = np.concatenate((hist_loss, h.history['loss']))
    hist_val_loss = np.concatenate((hist_val_loss, h.history['val_loss']))
    hist_acc = np.concatenate((hist_acc, h.history['accuracy']))
    hist_val_acc = np.concatenate((hist_val_acc, h.history['val_accuracy']))

plt.figure('Losses')
plt.plot(np.array(hist_loss).T)
plt.plot(np.array(hist_val_loss).T)
plt.legend([f'{d}_loss' for d in dec_facs] + [f'{d}_val_loss' for d in dec_facs])

plt.figure('Accuracy')
plt.plot(np.array(hist_acc).T)
plt.plot(np.array(hist_val_acc).T)
plt.legend([f'{d}_acc' for d in dec_facs] + [f'{d}_val_acc' for d in dec_facs])

for idx, l in enumerate(mdl.layers):
    plotWeights(mdl, idx, mdl_name=f'model')

plotActivations(mdl, Xs[ys[:, 0] == 1, :][0, :])

pos_pulses = sum(ys[:, 0])
pos_res = mdl.predict(Xs)
plt.figure(f'Pulse Found')
grid_sz = int(np.ceil(np.sqrt(pos_pulses)))
pos = 0
for n in range(Xs.shape[0]):
    if ys[n, 0]:
        pos += 1
        plt.subplot(grid_sz, grid_sz, pos)
        plt.title(f'{pos_res[n, 0] * 100:.2f}')
        plt.imshow(db(stft(Xs[n, :], return_onesided=False)[2]))
        plt.axis('tight')

 # id_model.save('./id_model')