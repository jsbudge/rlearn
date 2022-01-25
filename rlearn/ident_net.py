"""
Given a continuous set of sampled data, this should chop it into overlapping sections
and try to match them to any of a set of waveforms / just test if there's a signal involved
try to identify signal parameters and such
"""

import numpy as np
from tqdm import tqdm
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    Dropout, GaussianNoise, Concatenate
from kapre import STFT, MagnitudeToDecibel, Magnitude, Phase
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle
from wave_env import genPulse
from tqdm import tqdm
from scipy.signal.windows import taylor
from scipy.signal import stft
from tftb.processing import WignerVilleDistribution
import matplotlib.pyplot as plt

c0 = 299792458.0
TAC = 125e6
fs = 2e9 / 8
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
        plt.figure(lnm.split('/')[1] + 'layer {} weights: '.format(lnum) + mdl_name)
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
epoch_sz = 256
batch_sz = 32
neg_per_pos = 1
minp_sz = 40000
stft_sz = 512
band_limits = (10e6, fs / 2)
base_pl = 6.468e-6

segment_base_samp = minp_sz * dec_facs[-1]
segment_t0 = segment_base_samp / fs   # Segment time in secs
seg_sz = int(np.ceil(segment_t0 * fs))


def genModel(nsam):
    inp = Input(shape=(nsam, 1))
    lay = STFT(n_fft=stft_sz, win_length=stft_sz - (stft_sz % 100),
               hop_length=stft_sz // 4, window_name='hann_window')(inp)
    lay = Magnitude()(lay)
    lay = MaxPooling2D((4, 4))(lay)
    lay = BatchNormalization()(lay)
    lay = Conv2D(35, (16, 16))(lay)
    lay = Conv2D(25, (16, 16))(lay)
    lay = Flatten()(lay)
    lay = Dense(512, activation=keras.layers.LeakyReLU(alpha=.1), activity_regularizer=l1_l2(1e-4),
                kernel_regularizer=l1_l2(1e-3), bias_regularizer=l1_l2(1e-3))(lay)
    outp = Dense(1, activation='sigmoid')(lay)
    return keras.Model(inputs=inp, outputs=outp)


def genParamModel(nsam):
    inp = Input(shape=(nsam, 1))
    lay = STFT(n_fft=stft_sz, win_length=stft_sz - (stft_sz % 100),
               hop_length=stft_sz // 4, window_name='hann_window')(inp)
    lay = Magnitude()(lay)
    lay = MaxPooling2D((2, 2))(lay)
    lay = BatchNormalization()(lay)
    lay = Conv2D(25, (16, 16), activation=keras.layers.LeakyReLU(alpha=.1))(lay)
    lay = Flatten()(lay)
    lay = Dense(512, activation=keras.layers.LeakyReLU(alpha=.1), activity_regularizer=l1_l2(1e-4),
                kernel_regularizer=l1_l2(1e-3), bias_regularizer=l1_l2(1e-3))(lay)
    outp = Dense(2, kernel_constraint=NonNeg())(lay)
    return keras.Model(inputs=inp, outputs=outp)


# Generate models for detection and estimation
mdl = genModel(minp_sz)
mdl_comp_opts = dict(optimizer=Adam(learning_rate=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
mdl.compile(**mdl_comp_opts)
par_mdl = genParamModel(seg_sz)
par_comp_opts = dict(optimizer=Adam(learning_rate=1e-6), loss='huber_loss', metrics=['mean_squared_error'])
par_mdl.compile(**par_comp_opts)

ramp = np.linspace(0, 1, 100)
hist_loss = []
hist_val_loss = []
hist_acc = []
hist_val_acc = []

sig_on = True

genparams = lambda: (np.random.rand() * 400 + 100, int((np.random.rand() * (base_pl - 1e-6) + 1e-6) * fs),
                     np.random.rand() * (band_limits[1] - band_limits[0]) + band_limits[0])

for run in tqdm(range(120)):
    t0 = 0
    sig_t = 0
    count = 0
    Xt = []
    yt = []
    p_yt = []
    prf, nr, bw = genparams()
    pcnt = 0
    while len(Xt) < epoch_sz * 2:
        # Generate the data for this segment
        seg_data = np.random.normal(0, 1, seg_sz) + 1j * np.random.normal(0, 1, seg_sz)
        seg_truth = np.zeros(seg_sz)
        if pcnt > 10:
            prf, nr, bw = genparams()
            pcnt = 0
        # First, get the signal we may or may not use here
        if sig_on:
            while sig_t < t0:
                sig_t += 1 / prf
            # Continue to pulse to the PRF during the segment length
            while sig_t < t0 + segment_t0:
                ns = int((sig_t - t0) * fs)
                seg_data[ns:min(seg_sz, ns + nr)] += \
                    genPulse(ramp, np.random.rand(len(ramp)), nr, nr / fs, 9.6e9, bw)[:min(seg_sz - ns, nr)]
                seg_truth[ns:min(seg_sz, ns + nr)] = 1
                sig_t += 1 / prf
                pcnt += 1

        # Run each model data, using different decimation factors
        bgn = seg_data
        if len(yt) == 0:
            if np.any(seg_truth):
                yt.append([0])
                p_yt.append([nr / fs * 1e6, bw * 2 / fs])
            else:
                yt.append([1])
                count += 1
            Xt.append(bgn)
        else:
            if np.any(seg_truth):
                if count >= neg_per_pos:
                    yt.append([0])
                    p_yt.append([nr / fs * 1e6, bw * 2 / fs])
                    Xt.append(bgn)
                    count = 0
            else:
                if count < neg_per_pos:
                    yt.append([1])
                    Xt.append(bgn)
                    count += 1
        t0 += segment_t0
    p_Xt = np.array(Xt)[(np.array(yt) == 0).flatten(), ...]
    p_yt = np.array(p_yt)
    Xs = np.array(Xt)[:batch_sz, ...]
    ys = np.array(yt)[:batch_sz, ...]
    Xt = np.array(Xt)[batch_sz:, ...]
    yt = np.array(yt)[batch_sz:, ...]
    Xs, ys = shuffle(Xs, ys)
    Xt, yt = shuffle(Xt, yt)
    p_Xt, p_yt = shuffle(p_Xt, p_yt)

    h = mdl.fit(Xt, yt, validation_data=(Xs, ys), epochs=5, batch_size=batch_sz,
                callbacks=[ReduceLROnPlateau(), TerminateOnNaN()])
    ph = par_mdl.fit(p_Xt, p_yt, epochs=5, batch_size=batch_sz,
                callbacks=[TerminateOnNaN()])
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

plotActivations(mdl, Xs[ys[:, 0] == 0, :][0, :])

# Model analysis
test_mdl = genModel(minp_sz)
test_mdl.compile(**mdl_comp_opts)
sing_true = Xs[ys[:, 0] == 0, :][0, :].reshape((1, seg_sz))
sing_false = Xs[ys[:, 0] == 1, :][0, :].reshape((1, seg_sz))
sing_y = ys[ys[:, 0] == 0, :][0, :].reshape((1, 1))
sing_fy = ys[ys[:, 0] == 1, :][0, :].reshape((1, 1))

ms = test_mdl.fit(sing_false, sing_fy, epochs=20, verbose=0, callbacks=[TerminateOnNaN()])

plt.figure('Single Truth')
plt.subplot(2, 1, 1)
plt.title('Loss')
plt.plot(ms.history['loss'])
plt.subplot(2, 1, 2)
plt.title('Acc')
plt.plot(ms.history['accuracy'])

pos_pulses = sum(np.logical_not(ys))
pos_res = mdl.predict(Xs)
plt.figure(f'Pulse Found')
grid_sz = int(np.ceil(np.sqrt(pos_pulses)))
pos = 0
for n in range(Xs.shape[0]):
    if not ys[n]:
        pos += 1
        plt.subplot(grid_sz, grid_sz, pos)
        plt.title(f'{(1 - pos_res[n, 0]) * 100:.2f}')
        plt.imshow(db(stft(Xs[n, :], return_onesided=False)[2]))
        plt.axis('tight')

neg_pulses = sum(ys)
pos_res = mdl.predict(Xs)
plt.figure(f'No Pulse Found')
grid_sz = int(np.ceil(np.sqrt(neg_pulses)))
pos = 0
for n in range(Xs.shape[0]):
    if ys[n]:
        pos += 1
        plt.subplot(grid_sz, grid_sz, pos)
        plt.title(f'{(1 - pos_res[n, 0]) * 100:.2f}')
        plt.imshow(db(stft(Xs[n, :], return_onesided=False)[2]))
        plt.axis('tight')

# Save out the model for future use
mdl.save('./id_model')
par_mdl.save('./par_model')

# plot_model(mdl, to_file='mdl.png', show_shapes=True)
# plot_model(par_mdl, to_file='par_mdl.png', show_shapes=True)





