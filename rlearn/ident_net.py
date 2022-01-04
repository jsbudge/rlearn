"""
Given a continuous set of sampled data, this should chop it into overlapping sections
and try to match them to any of a set of waveforms / just test if there's a signal involved
try to identify signal parameters and such
"""

import numpy as np
from tqdm import tqdm
import keras
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    Dropout, GaussianNoise
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from wave_env import genPulse
from tqdm import tqdm
from scipy.signal.windows import taylor
from tftb.processing import WignerVilleDistribution
import matplotlib.pyplot as plt

c0 = 299792458.0
TAC = 125e6
fs = 2e9 / 4
DTR = np.pi / 180


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


# Base net params
dec_facs = [1, 2, 4]
batch_sz = 64
minp_sz = 1600
band_limits = (10e6, fs / 2)
base_pl = 6.468e-6

segment_base_samp = minp_sz * dec_facs[-1]
segment_t0 = segment_base_samp / fs   # Segment time in secs
run_time = segment_t0 * 10
seg_sz = int(np.ceil(segment_t0 * fs))


def genModel(m_sz):
    inp = Input(shape=(m_sz, m_sz, 1))
    lay = BatchNormalization()(inp)
    lay = MaxPooling2D((4, 4))(lay)
    lay = Conv2D(10, (16, 16))(lay)
    lay = MaxPooling2D((4, 4))(lay)
    lay = Conv2D(10, (16, 16), activation=keras.layers.LeakyReLU(alpha=.3))(lay)
    lay = Flatten()(lay)
    lay = Dropout(.4)(lay)
    lay = Dense(512, activation=keras.layers.LeakyReLU(alpha=.3))(lay)
    lay = GaussianNoise(1)(lay)
    outp = Dense(2, activation='softmax')(lay)
    return keras.Model(inputs=inp, outputs=outp)


# Generate models for different sampling rates
mdls = []
for d in dec_facs:
    mdl = genModel(minp_sz)
    mdl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mdls.append(mdl)
ramp = np.linspace(0, 1, 100)
hist_loss = [[] for d in dec_facs]
hist_val_loss = [[] for d in dec_facs]
hist_acc = [[] for d in dec_facs]
hist_val_acc = [[] for d in dec_facs]

sig_on = True

for run in tqdm(range(400)):
    t0 = 0
    sig_t = 0
    Xsplt = [[] for d in dec_facs]
    ysplt = [[] for d in dec_facs]
    prf = np.random.rand() * 400 + 100
    # Make sure it's at least a microsecond long
    nr = int((np.random.rand() * (base_pl - 1e-6) + 1e-6) * fs)
    pcnt = 0
    while np.any([len(X) < batch_sz for X in Xsplt]):
        # Generate the data for this segment
        seg_data = np.random.normal(0, 1, seg_sz) + 1j * np.random.normal(0, 1, seg_sz)
        seg_truth = np.zeros(seg_sz)
        if pcnt > 10:
            prf = np.random.rand() * 400 + 100
            # Make sure it's at least a microsecond long
            nr = int((np.random.rand() * (base_pl - 1e-6) + 1e-6) * fs)
            pcnt = 0
        # First, get the signal we may or may not use here
        if sig_on:
            while sig_t < t0:
                sig_t += 1 / prf
            # Continue to pulse to the PRF during the segment length
            while sig_t < t0 + segment_t0:
                ns = int((sig_t - t0) * fs)
                bw = np.random.rand() * (band_limits[1] - band_limits[0]) + band_limits[0]
                seg_data[ns:min(seg_sz, ns + nr)] += \
                    genPulse(ramp, ramp, nr, nr / fs, 9.6e9, bw)[:min(seg_sz - ns, nr)]
                seg_truth[ns:min(seg_sz, ns + nr)] = 1
                sig_t += 1 / prf
                pcnt += 1

        # Run each model data, using different decimation factors
        for idx, dfac in enumerate(dec_facs):
            for b in range(0, seg_sz // minp_sz, dfac):
                if (b+dfac) * minp_sz > seg_sz:
                    break
                bgn = seg_data[b * minp_sz:min(seg_sz, (b+dfac) * minp_sz):dfac]
                if len(ysplt[idx]) == 0:
                    if np.any(seg_truth[b * minp_sz:min(seg_sz, (b + dfac) * minp_sz):dfac]):
                        ysplt[idx].append([True, False])
                    else:
                        ysplt[idx].append([False, True])
                    Xsplt[idx].append(WignerVilleDistribution(bgn).run()[0])
                else:
                    if np.any(seg_truth[b * minp_sz:min(seg_sz, (b+dfac) * minp_sz):dfac]):
                        if ysplt[idx][-1][1]:
                            ysplt[idx].append([True, False])
                            Xsplt[idx].append(WignerVilleDistribution(bgn).run()[0])
                    else:
                        if ysplt[idx][-1][0]:
                            ysplt[idx].append([False, True])
                            Xsplt[idx].append(WignerVilleDistribution(bgn).run()[0])
        t0 += segment_t0
    for dec_idx in range(len(dec_facs)):
        Xt = np.array(Xsplt[dec_idx])
        yt = np.array(ysplt[dec_idx])
        Xs = Xt
        ys = yt

        h = mdls[dec_idx].fit(Xt, yt, validation_data=(Xs, ys), epochs=2, callbacks=[TerminateOnNaN()])
        hist_loss[dec_idx] = np.concatenate((hist_loss[dec_idx], h.history['loss']))
        hist_val_loss[dec_idx] = np.concatenate((hist_val_loss[dec_idx], h.history['val_loss']))
        hist_acc[dec_idx] = np.concatenate((hist_acc[dec_idx], h.history['accuracy']))
        hist_val_acc[dec_idx] = np.concatenate((hist_val_acc[dec_idx], h.history['val_accuracy']))
for m_idx in range(len(dec_facs)):
    Xsplt[m_idx] = np.array(Xsplt[m_idx])
    ysplt[m_idx] = np.array(ysplt[m_idx])

plt.figure('Losses')
plt.plot(np.array(hist_loss).T)
plt.plot(np.array(hist_val_loss).T)
plt.legend([f'{d}' for d in dec_facs])

plt.figure('Accuracy')
plt.plot(np.array(hist_acc).T)
plt.plot(np.array(hist_val_acc).T)
plt.legend([f'{d}' for d in dec_facs])

for m_idx, id_model in enumerate(mdls):
    #for idx, l in enumerate(id_model.layers):
    #    plotWeights(id_model, idx, mdl_name=f'dec_fac_{dec_facs[m_idx]}')

    pos_pulses = sum(ysplt[m_idx][:, 0])
    pos_res = id_model.predict(Xsplt[m_idx])
    plt.figure(f'Pulse Found DF_{dec_facs[m_idx]}')
    grid_sz = int(np.ceil(np.sqrt(pos_pulses)))
    pos = 0
    for n in range(Xsplt[m_idx].shape[0]):
        if ysplt[m_idx][n, 0]:
            pos += 1
            plt.subplot(grid_sz, grid_sz, pos)
            plt.title(f'{pos_res[n, 0] * 100:.2f}')
            plt.imshow(Xsplt[m_idx][n, :, :])

 # id_model.save('./id_model')