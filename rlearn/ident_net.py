import numpy as np
from tqdm import tqdm
import keras
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from wave_env import genPulse
from tqdm import tqdm
from scipy.signal.windows import taylor
from tftb.processing import WignerVilleDistribution
import matplotlib.pyplot as plt

c0 = 299792458.0
TAC = 125e6
fs = 2e9 / 8
DTR = np.pi / 180


def plotWeights(mdl, lnum=2):
    lw = mdl.layers[lnum].get_weights()
    if len(lw) > 0:
        lw = lw[0]
        lnm = mdl.layers[lnum].output.name
    else:
        return
    if 'conv2d' in lnm:
        # It's a convolution layer
        plt.figure(lnm.split('/')[1] + ' weights')
        grid_sz = int(np.ceil(np.sqrt(lw.shape[3])))
        for n in range(lw.shape[3]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.imshow(lw[:, :, 0, n])
            plt.title(f'{n}')


batch_sz = 64
runs = 10
segment_t0 = 6e-6
pulse_limits = (500e-9, 5e-6)
band_limits = (10e6, fs / 2)
segment_sz = int(np.ceil(segment_t0 * fs))

inp = Input(shape=(segment_sz, segment_sz, 1))
lay = BatchNormalization()(inp)
lay = MaxPooling2D((4, 4))(lay)
lay = Conv2D(10, (16, 16))(lay)
lay = MaxPooling2D((4, 4))(lay)
lay = Conv2D(10, (16, 16), activation=keras.layers.LeakyReLU(alpha=.3))(lay)
lay = Flatten()(lay)
lay = Dense(512, activation=keras.layers.LeakyReLU(alpha=.3))(lay)
outp = Dense(2, activation='softmax')(lay)

id_model = keras.Model(inputs=inp, outputs=outp)

id_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ramp = np.linspace(0, 1, 100)
hist_loss = []
hist_val_loss = []
hist_acc = []
hist_val_acc = []

for rnd in range(runs):
    Xt = []
    yt = []
    Xs = []
    ys = []
    for b in tqdm(range(batch_sz)):
        # Generate random data for training
        bgn = np.random.rand(segment_sz) + 1j * np.random.rand(segment_sz)
        if np.random.rand() < .5:
            nr = int((np.random.rand() * (pulse_limits[1] - pulse_limits[0]) + pulse_limits[0]) * fs)
            bw = np.random.rand() * (band_limits[1] - band_limits[0]) + band_limits[0]
            start_loc = np.random.randint(0, segment_sz - 10)
            yt.append([True, False])
            pulse = genPulse(ramp, np.random.rand(100), nr, nr / fs, 9.6e9, bw) * \
                    (np.random.rand() * (1 - np.mean(abs(bgn))) + np.mean(abs(bgn)))
            bgn[start_loc:min(start_loc+nr, segment_sz)] += pulse[:min(nr, segment_sz - start_loc)]
        else:
            yt.append([False, True])
        Xt.append(WignerVilleDistribution(bgn).run()[0])

        # Repeat for validation
        if b % 2 == 0:
            bgn = np.random.rand(segment_sz) + 1j * np.random.rand(segment_sz)
            if np.random.rand() < .5:
                nr = int((np.random.rand() * (pulse_limits[1] - pulse_limits[0]) + pulse_limits[0]) * fs)
                bw = np.random.rand() * (band_limits[1] - band_limits[0]) + band_limits[0]
                start_loc = np.random.randint(0, segment_sz - 10)
                ys.append([True, False])
                pulse = genPulse(ramp, np.random.rand(100), nr, nr / fs, 9.6e9, bw) * \
                        (np.random.rand() * (1 - np.mean(abs(bgn))) + np.mean(abs(bgn)))
                bgn[start_loc:min(start_loc+nr, segment_sz)] += pulse[:min(nr, segment_sz - start_loc)]
            else:
                ys.append([False, True])
            Xs.append(WignerVilleDistribution(bgn).run()[0])
    Xs = np.array(Xs)
    Xt = np.array(Xt)
    yt = np.array(yt)
    ys = np.array(ys)

    h = id_model.fit(Xt, yt, validation_data=(Xs, ys), epochs=2, callbacks=[TerminateOnNaN()])
    hist_loss = np.concatenate((hist_loss, h.history['loss']))
    hist_val_loss = np.concatenate((hist_val_loss, h.history['val_loss']))
    hist_acc = np.concatenate((hist_acc, h.history['accuracy']))
    hist_val_acc = np.concatenate((hist_val_acc, h.history['val_accuracy']))

plt.figure('Losses')
plt.plot(hist_loss)
plt.plot(hist_val_loss)

plt.figure('Accuracy')
plt.plot(hist_acc)
plt.plot(hist_val_acc)

for idx, l in enumerate(id_model.layers):
    plotWeights(id_model, idx)

pos_pulses = sum(ys[:, 0])
pos_res = id_model.predict(Xs)
plt.figure('Pulse Found')
grid_sz = int(np.ceil(np.sqrt(pos_pulses)))
pos = 0
for n in range(Xs.shape[0]):
    if ys[n, 0]:
        pos += 1
        plt.subplot(grid_sz, grid_sz, pos)
        plt.title(f'{pos_res[n, 0] * 100:.2f}')
        plt.imshow(Xs[n, :, :])

id_model.save('./id_model')