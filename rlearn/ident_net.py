"""
Given a continuous set of sampled data, this should chop it into overlapping sections
and try to match them to any of a set of waveforms / just test if there's a signal involved
try to identify signal parameters and such
"""

import numpy as np
from tqdm import tqdm
import keras
from tensorflow.signal import rfft
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.constraints import NonNeg
import tensorflow as tf
# from tensorflow.profiler import profile, ProfileOptionBuilder
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    Dropout, GaussianNoise, Concatenate, LSTM, Embedding, Conv1D, Lambda, MaxPooling1D, ActivityRegularization, \
    LocallyConnected2D, Normalization, LayerNormalization
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
from matplotlib import cm
from datetime import datetime

c0 = 299792458.0
TAC = 125e6
fs = 2e9 / 8
DTR = np.pi / 180


def db(x):
    ret = abs(x)
    ret[ret == 0] = 1e-9
    return 20 * np.log10(ret)


def readTrainingDataGen(fnme, chunk):
    with open(fnme + '_data.dat', 'rb') as f:
        with open(fnme + '_labels.dat', 'rb') as f_l:
            nsam = int(np.fromfile(f, 'float64', 1)[0])
            run = True
            while run:
                ret = np.fromfile(f, 'complex128', chunk * nsam)
                lab = np.fromfile(f_l, 'bool', chunk)
                try:
                    ret = ret.reshape((chunk, nsam))
                    yield ret, lab
                except ValueError:
                    run = False


def plotWeights(mdl, lnum=2, mdl_name=''):
    lw = mdl.layers[lnum].get_weights()
    if len(lw) > 0:
        lw = lw[0]
        lnm = mdl.layers[lnum].output.name
    else:
        return
    if 'conv2d' in lnm:
        # It's a convolution layer
        for m in range(1):
            plt.figure(lnm.split('/')[1] + ' layer {} channel {} weights: '.format(lnum, m) + mdl_name)
            grid_sz = int(np.ceil(np.sqrt(lw.shape[3])))
            for n in range(lw.shape[3]):
                plt.subplot(grid_sz, grid_sz, n + 1)
                plt.imshow(lw[:, :, m, n])
                plt.title(f'{n}')
    if 'conv1d' in lnm:
        for m in range(1):
            plt.figure(lnm.split('/')[1] + ' layer {} channel {} weights: '.format(lnum, m) + mdl_name)
            grid_sz = int(np.ceil(np.sqrt(lw.shape[2])))
            for n in range(lw.shape[2]):
                plt.subplot(grid_sz, grid_sz, n + 1)
                plt.plot(lw[:, m, n])
                plt.title(f'{n}')


def plotActivations(classifier, inp, is_3d=False):
    layer_outputs = [layer.output for layer in classifier.layers]  # Extracts the outputs of the top 12 layers
    activation_model = keras.Model(inputs=classifier.input,
                                   outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(inp.reshape((1, len(inp), 1)))
    layer_names = []
    for layer in classifier.layers:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        if 'stft' in layer_name or '2d' in layer_name:
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            if is_3d and n_features > 1:
                for m in range(layer_activation.shape[0]):
                    plt.figure(layer_name + f' channel {m}')
                    ax = plt.subplot(111, projection='3d')
                    ly = layer_activation[m, ...]
                    xx, yy = np.meshgrid(np.arange(ly.shape[1]), np.arange(ly.shape[0]))
                    zz = np.zeros(xx.shape)
                    for n in range(ly.shape[2]):
                        try:
                            cmap = cm.coolwarm(ly[:, :, n] / np.max(ly[:, :, n]))
                        except TypeError:
                            cmap = cm.coolwarm(db(ly[:, :, n]))
                        ax.plot_surface(xx, zz + (4 * n), yy, alpha=.3, rstride=1, cstride=1,
                                        facecolors=cmap)
                    plt.axis('off')
            else:
                grid_sz = int(np.ceil(np.sqrt(n_features)))
                for m in range(layer_activation.shape[0]):
                    plt.figure(layer_name + f' channel {m}')
                    plt.grid(False)
                    for n in range(n_features):
                        plt.subplot(grid_sz, grid_sz, n + 1)
                        try:
                            plt.imshow(layer_activation[m, :, :, n], aspect='auto', cmap='viridis')
                        except TypeError:
                            plt.imshow(db(layer_activation[m, :, :, n]), aspect='auto', cmap='viridis')
                        plt.axis('off')
        elif 'dense' in layer_name:
            for m in range(layer_activation.shape[0]):
                plt.figure(layer_name + f' channel {m}')
                if layer_activation.shape[-1] < 3:
                    plt.scatter(np.arange(layer_activation.shape[-1]), layer_activation[m, ...])
                else:
                    plt.plot(layer_activation[m, ...])
        elif '1d' in layer_name:
            n_features = layer_activation.shape[-1]
            grid_sz = int(np.ceil(np.sqrt(n_features)))
            for m in range(layer_activation.shape[0]):
                plt.figure(layer_name + f' channel {m}')
                plt.grid(False)
                for n in range(n_features):
                    plt.subplot(grid_sz, grid_sz, n + 1)
                    try:
                        plt.plot(layer_activation[m, :, n])
                    except TypeError:
                        plt.plot(db(layer_activation[m, :, n]))
        elif 'fft' in layer_name:
            plt.figure(layer_name)
            plt.plot(db(layer_activation[0, :, 0]))


# Base net params
dec_facs = [1]
epoch_sz = 256
batch_sz = 32
neg_per_pos = 1
minp_sz = 16384
stft_sz = 256
band_limits = (10e6, fs / 2)
base_pl = 6.468e-6
train_prf = 1000.
train_runs = 1
load_model = True
save_model = False
tset_data_only = True
noise_sigma = 1e-6

segment_base_samp = minp_sz * dec_facs[-1]
segment_t0 = segment_base_samp / fs  # Segment time in secs
seg_sz = int(np.ceil(segment_t0 * fs))

# Tensorboard stuff
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def genModel(nsam):
    inp = Input(shape=(nsam, 1))
    lay = STFT(n_fft=stft_sz,
               hop_length=stft_sz // 2, window_name='hann_window')(inp)
    lay = Magnitude()(lay)
    lay = BatchNormalization(center=True, scale=True, axis=1)(lay)
    lay = Conv2D(32, (16, 16), padding='same')(lay)
    lay = MaxPooling2D((2, 2))(lay)
    lay = Conv2D(32, (16, 16), padding='same')(lay)
    lay = Flatten()(lay)
    lay = Dropout(.4)(lay)
    lay = Dense(128, activation='relu')(lay)
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
    lay = Embedding(input_dim=512, output_dim=256)(lay)
    lay = LSTM(256)(lay)
    outp = Dense(2, kernel_constraint=NonNeg())(lay)
    return keras.Model(inputs=inp, outputs=outp)


# Generate models for detection and estimation
if load_model:
    mdl = keras.models.load_model('./id_model')
else:
    mdl = genModel(minp_sz)
mdl_comp_opts = dict(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
mdl_callbacks = [EarlyStopping(patience=3), TerminateOnNaN(),
                 ReduceLROnPlateau(patience=10, factor=.9, min_lr=1e-9),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
mdl.compile(**mdl_comp_opts)
par_mdl = genParamModel(seg_sz)
par_comp_opts = dict(optimizer=Adadelta(learning_rate=1.0), loss='huber_loss', metrics=['mean_squared_error'])
par_mdl.compile(**par_comp_opts)

ramp = np.linspace(0, 1, 100)
hist_loss = []
hist_val_loss = []
hist_acc = []
hist_val_acc = []

# Initial gimme dataset to learn basic things
tpt = 0
spt = 0
npt = minp_sz / fs
for tset in tqdm(range(train_runs)):
    data = np.random.normal(0, noise_sigma, (epoch_sz, minp_sz)) + 1j * np.random.normal(0, noise_sigma,
                                                                                         (epoch_sz, minp_sz))
    labels = np.zeros((epoch_sz,))
    for n in np.arange(0, epoch_sz, 2):
        fc = np.random.uniform(8e9, 12e9)
        bw = np.random.uniform(*band_limits)
        rngtx = np.random.uniform(1500, 3000)
        t0 = np.random.uniform(.45, .55) * base_pl
        pulse = np.fft.fft(genPulse(ramp, ramp, int(t0 * fs), t0, fc, bw), minp_sz)
        ndata = np.zeros((minp_sz,), dtype=np.complex128)
        while tpt > npt:
            spt += minp_sz / fs
            npt += minp_sz / fs
        while tpt < spt:
            tpt += 1 / train_prf
        while spt <= tpt < npt:
            times = np.arange(spt, npt, 1 / fs)[:minp_sz]
            tdist = abs(times - tpt)
            ndata[tdist == tdist.min()] += 10 * np.exp(1j * 2 * np.pi * c0 / fc * rngtx) * 1 / (rngtx * rngtx)
            tpt += 1 / train_prf
        data[n, :] += np.fft.ifft(np.fft.fft(ndata) * pulse)
        labels[n] = 1.
    h = mdl.fit(data, labels, validation_split=.2, epochs=15, batch_size=batch_sz,
                callbacks=mdl_callbacks)
    hist_loss = np.concatenate((hist_loss, h.history['loss']))
    hist_val_loss = np.concatenate((hist_val_loss, h.history['val_loss']))
    hist_acc = np.concatenate((hist_acc, h.history['accuracy']))
    hist_val_acc = np.concatenate((hist_val_acc, h.history['val_accuracy']))
    if tset_data_only:
        Xs = data[:batch_sz, :]
        ys = labels[:batch_sz]

if not tset_data_only:
    for data, labels in tqdm(readTrainingDataGen('/home/jeff/repo/rlearn/mdl_data/train', epoch_sz)):
        Xs = data[:batch_sz, :]
        Xt = data[batch_sz:, :]
        ys = labels[:batch_sz]
        yt = labels[batch_sz:]
        Xs, ys = shuffle(Xs, ys)
        Xt, yt = shuffle(Xt, yt)

        h = mdl.fit(Xt, yt, validation_data=(Xs, ys), epochs=15, batch_size=batch_sz,
                    callbacks=mdl_callbacks,
                    class_weight={0: sum(ys) / len(ys), 1: 1 - sum(ys) / len(ys)})
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

plotActivations(mdl, Xs[ys == 1, :][1, :], True)

# Model analysis
test_mdl = genModel(minp_sz)
test_mdl.compile(**mdl_comp_opts)
sing_true = Xs[ys == 1, :][0, :].reshape((1, seg_sz))
sing_false = Xs[ys == 0, :][0, :].reshape((1, seg_sz))
sing_y = np.array([int(ys[ys == 1][0])])
sing_fy = np.array([int(ys[ys == 0][0])])

with tf.profiler.experimental.Profile(log_dir):
    ms = test_mdl.fit(sing_false, sing_fy, epochs=20, verbose=0, callbacks=[TerminateOnNaN()])

plt.figure('Single Truth')
plt.subplot(2, 1, 1)
plt.title('Loss')
plt.plot(ms.history['loss'])
plt.subplot(2, 1, 2)
plt.title('Acc')
plt.plot(ms.history['accuracy'])

pos_pulses = sum(ys)
pos_res = mdl.predict(Xs)
plt.figure(f'Pulse Found')
grid_sz = int(np.ceil(np.sqrt(pos_pulses)))
pos = 0
for n in range(Xs.shape[0]):
    if ys[n]:
        pos += 1
        plt.subplot(grid_sz, grid_sz, pos)
        plt.title(f'{(1 - pos_res[n, 0]) * 100:.2f}')
        plt.imshow(db(stft(Xs[n, :], return_onesided=False)[2]))
        plt.axis('tight')

neg_pulses = sum(np.logical_not(ys))
pos_res = mdl.predict(Xs)
plt.figure(f'No Pulse Found')
grid_sz = int(np.ceil(np.sqrt(neg_pulses)))
pos = 0
for n in range(Xs.shape[0]):
    if not ys[n]:
        pos += 1
        plt.subplot(grid_sz, grid_sz, pos)
        plt.title(f'{(1 - pos_res[n, 0]) * 100:.2f}')
        plt.imshow(db(stft(Xs[n, :], return_onesided=False)[2]))
        plt.axis('tight')

# Save out the model for future use
if save_model:
    mdl.save('./id_model')
    plot_model(mdl, to_file='mdl.png', show_shapes=True)
    # par_mdl.save('./par_model')

mdl_graph = tf.profiler.experimental.Profile(mdl)

# num_flops = profile(mdl, options=ProfileOptionBuilder.float_operation())
# print(f'FLOPs of model: {flops.total_float_ops}')