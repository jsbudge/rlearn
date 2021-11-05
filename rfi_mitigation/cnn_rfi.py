import numpy as np
from rawparser import loadXMLFile, getRawSDRParams
from useful_lib import findAllFilenames, findPowerOf2, gaus, db
from SARParse import SDRParse
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from tensorflow import keras
from keras.models import Model
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, InputLayer, \
    BatchNormalization
from keras.regularizers import l2
from keras.layers import preprocessing
from keras.utils.vis_utils import plot_model
from keras.optimizer_v2.adam import Adam
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tqdm import tqdm
import keras_tuner as kt


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, a_sdr_f: SDRParse, list_ids, a_labels, batch_size=64, nsz=48, shuffle=True):
        """Initialization"""
        self.dim = (nsz, 1)
        self.data = a_sdr_f
        self.batch_size = batch_size
        self.labels = a_labels
        self.list_IDs = list_ids
        self.idx = np.arange(len(list_ids))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Find list of IDs
        idxes = self.idx[index * self.batch_size:(index + 1) * self.batch_size]
        d_id = self.list_IDs[idxes]
        # Store sample
        X = db(np.fft.fft(sdr_f.getPulses(d_id), self.dim[0], axis=0)).T.reshape(
            (self.batch_size, *self.dim))

        # Store class
        y = self.labels[idxes]
        return X, keras.utils.to_categorical(y)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.idx)


def plotActivations(mdl, inp, lnum=2):
    louts = [layer.output for layer in mdl.layers]
    act_mdl = Model(inputs=mdl.input, outputs=louts)
    acts = act_mdl.predict(inp)
    plt_act = acts[lnum]
    if 'conv1d' in louts[lnum].name or 'max_pooling' in louts[lnum].name:
        # It's a convolution layer
        plt.figure(louts[lnum].name)
        grid_sz = int(np.ceil(np.sqrt(plt_act.shape[2])))
        for n in range(plt_act.shape[2]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.plot(plt_act[0, :, n])
            plt.title(f'{n}')
    elif 'dense' in louts[lnum].name:
        plt.figure(louts[lnum].name)
        plt.scatter(np.arange(plt_act.shape[1]), plt_act[0, :].flatten())
    elif 'input' in louts[lnum].name:
        plt.figure(louts[lnum].name)
        try:
            plt.plot(plt_act[0, :, :].flatten())
        except IndexError:
            plt.plot(plt_act[0, :].flatten())
    else:
        print('Did not plot layer ' + louts[lnum].name)


def plotWeights(mdl, lnum=2):
    lw = mdl.layers[lnum].get_weights()
    if len(lw) > 0:
        lw = lw[0]
        lnm = model.layers[lnum].output.name
    else:
        return
    if 'conv1d' in lnm:
        # It's a convolution layer
        plt.figure(lnm.split('/')[1] + ' weights')
        grid_sz = int(np.ceil(np.sqrt(lw.shape[2])))
        for n in range(lw.shape[2]):
            plt.subplot(grid_sz, grid_sz, n + 1)
            plt.plot(lw[:, 0, n])
            plt.title(f'{n}')


def loadRFILabels(inp_fnme):
    with open(inp_fnme, 'rb') as f:
        isRFI = np.fromfile(f, 'int8', -1, '')
    return isRFI.astype(bool)


fnme = '/home/jeff/repo/mimo_simulator/SAR_09222021_163338SIM.sar'
model_fpath = './mdl'
dat_fnme = fnme[:-4] + '_rfi.dat'
total_iters = 100
batch_sz = 64
val_perc = .2

# Load in the file data
labels = loadRFILabels(dat_fnme)
labels = labels.reshape(len(labels), 1).astype(np.float32)
sdr_f = SDRParse(fnme)
net_sz = findPowerOf2(sdr_f[0].nsam) * 2

print(f'{sum(labels)[0] / len(labels) * 100:.2f}% RFI pulses.')


# Load in the model
def build_model(hp):
    mdl = keras.Sequential()
    mdl.add(InputLayer(input_shape=(net_sz, 1)))
    mdl.add(MaxPooling1D(32))
    mdl.add(BatchNormalization())
    mdl.add(Conv1D(filters=4, kernel_size=128, activation=keras.layers.LeakyReLU(alpha=.3),
                   kernel_regularizer=l2(.33), bias_regularizer=l2(.33), activity_regularizer=l2(.33)))
    mdl.add(MaxPooling1D(8))
    mdl.add(Dropout(.4))
    mdl.add(Flatten())
    hp_dlaysz = hp.Int('units', min_value=2, max_value=512)
    mdl.add(Dense(units=hp_dlaysz, activation=keras.layers.LeakyReLU(alpha=.3)))
    mdl.add(Dense(2, activation='softmax'))

    hp_lr = hp.Choice('learning_rate', values=[1e-4, 1e-6, 1e-8])
    opt = keras.optimizers.Adam(learning_rate=hp_lr)

    mdl.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return mdl


# Specify a tuner
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=2,
                     directory='tuning',
                     project_name='rfi_mit',
                     overwrite=True)

# Select our training pulses such that the class sizes are about even
dp = np.where(labels == 1)[0] + 1
cp = np.where(labels == 0)[0] + 1

if len(dp) > len(cp):
    dp = dp[:len(cp)]
elif len(cp) > len(dp):
    cp = cp[:len(dp)]

# Remove some of the pulses for validation (~25%)
dp_s = dp[:int(len(dp) * val_perc)]
cp_s = cp[:int(len(dp) * val_perc)]
dp_t = dp[int(len(dp) * val_perc):]
cp_t = cp[int(len(dp) * val_perc):]

val_pulses = np.concatenate((dp_s, cp_s))
train_pulses = np.concatenate((dp_t, cp_t))

# Callbacks definition
callbacks = [EarlyStopping(monitor='val_loss', patience=5), TerminateOnNaN()]

# Generate dataset for hyperparameter tuning and normalization
pulses = np.random.permutation(np.concatenate((dp_t[:batch_sz],
                                               cp_t[:batch_sz])))
Xt = db(np.fft.fft(sdr_f.getPulses(pulses), net_sz, axis=0)).T.reshape((len(pulses), net_sz, 1))
yt = keras.utils.to_categorical(labels[pulses - 1])
print(f'Tuner training data has {sum(labels[pulses - 1])[0] / len(pulses) * 100:.2f}% RFI pulses.')

# Search for best hyperparams and build model based on that
tuner.search(Xt, yt, epochs=1, validation_split=.2, callbacks=callbacks)
hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(hps)

# Create data generators as pipeline for fitting
Xt_gen = DataGenerator(sdr_f, train_pulses, labels[train_pulses - 1], batch_size=batch_sz, nsz=net_sz)
Xs_gen = DataGenerator(sdr_f, val_pulses, labels[val_pulses - 1], batch_size=batch_sz, nsz=net_sz)

# Fit model to data using parallel processing to speed things up
mhist = model.fit_generator(generator=Xt_gen, validation_data=Xs_gen, use_multiprocessing=True,
                            workers=6, epochs=2, callbacks=callbacks)

model.save(model_fpath)

# Example frames for some plots
clean_frame = db(np.fft.fft(sdr_f.getPulse(cp_t[0]), net_sz)).reshape((1, net_sz, 1))
dirty_frame = db(np.fft.fft(sdr_f.getPulse(dp_t[0]), net_sz)).reshape((1, net_sz, 1))

try:
    plot_model(model, to_file='./mdl.png', show_shapes=True)
except AssertionError:
    print('Model does not want to print.')

plt.figure('Losses')
plt.plot(mhist.history['loss'])
plt.plot(mhist.history['val_loss'])

plt.figure('Spectrum')
plt.subplot(2, 1, 1)
plt.title('Clean')
plt.plot(clean_frame.flatten())
plt.subplot(2, 1, 2)
plt.title('Dirty')
plt.plot(dirty_frame.flatten())

plt.figure('Xt')
plt.imshow(Xt)
plt.axis('tight')

for lay in range(len(model.layers)):
    plotActivations(model, dirty_frame, lay)
    plotActivations(model, clean_frame, lay)
    plotWeights(model, lay)
