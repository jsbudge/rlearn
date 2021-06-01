"""
parsing

Implements MiniParsers for both SAR and SDR data.
Both of them are .sar files.

To be used mostly with the parser included in this folder.
"""

import numpy as np
import mmap

START_WORD = b'\x0A\xB0\xFF\x18'
STOP_WORD = b'\xAA\xFF\x11\x55'
COLLECTION_MODE_DIGITAL_CHANNEL_MASK = 0x000003E000000000
COLLECTION_MODE_OPERATION_MASK = 0x1800000000000000
COLLECTION_MODE_DIGITAL_CHANNEL_SHIFT = 37
COLLECTION_MODE_OPERATION_SHIFT = 59


class SARMiniParser(object):
    def __init__(self, fnme=None, is_dechirp=False):

        # Get the file we'll be parsing, and some extra data for it
        self.fnme = fnme

        self.isDechirp = is_dechirp
        # Read the points in the file where the packets we care about lie hidden
        self.packet_points = self.readPacketPoints()
        self.cal, self.n_samples, tmp_times, self.frameNum, self.att, self.frameType = \
            self.readDataParams()

        # There may be a few stragglers that accidentally have the matching words.
        # Let's remove them.
        true_nsamples = self.n_samples[0]
        weirdos_mask = self.n_samples == true_nsamples
        self.packet_points['data'] = self.packet_points['data'][weirdos_mask]
        self.cal = self.cal[weirdos_mask].astype(bool)
        self.n_samples = self.n_samples[weirdos_mask]
        tmp_times = tmp_times[weirdos_mask]
        self.sys_time = np.zeros((len(tmp_times),))
        self.frameNum = np.arange(len(self.packet_points['data'])) + 1
        self.att = self.att[weirdos_mask]

    '''
    getPulse
    Returns a single pulse.

    pulse_num - the pulse number or frame number from the file.
    '''

    def getPulse(self, pulse_num):
        pp = self.packet_points['data'][pulse_num - 1]
        n_samples = self.n_samples[pulse_num - 1]
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            mm.seek(pp)
            data = mm.read(28)
            if not self.isDechirp:
                tmp_data = np.zeros((n_samples,), dtype=np.int16)
                data = mm.read(2 * n_samples)
                for n in range(n_samples):
                    tmp_data[n] = int.from_bytes(data[2 * n:2 * n + 2], byteorder='big', signed=True) * 10 ** (
                            self.att[pulse_num - 1] / 20)
            else:
                tmp_data = np.zeros((n_samples,), dtype=np.complex64)
                data = mm.read(4 * n_samples)
                for n in range(n_samples):
                    tmp_data[n] = int.from_bytes(data[2 * n:2 * n + 2], byteorder='big', signed=True) * 10 ** (
                            self.att[pulse_num - 1] / 20)
                for n in range(n_samples, 2 * n_samples):
                    tmp_data[n] += 1j * int.from_bytes(data[2 * n:2 * n + 2], byteorder='big', signed=True) * 10 ** (
                            self.att[pulse_num - 1] / 20)
        return tmp_data

    def getPulses(self, pulse_nums, verbose=False):
        pdata = np.zeros((len(pulse_nums), self.n_samples[0]))
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            if verbose:
                for idx, p in enumerate(pulse_nums):
                    try:
                        pp = self.packet_points['data'][p - 1]
                    except:
                        print('IDX is {} and p is {}'.format(idx, p))
                        pp = self.packet_points['data'][p - 1]
                    mm.seek(pp)
                    mm.read(28)
                    tmp_data = np.zeros((pdata.shape[1],))
                    data = mm.read(2 * pdata.shape[1])
                    for n in range(pdata.shape[1]):
                        tmp_data[n] = int.from_bytes(data[2 * n:2 * n + 2], byteorder='big', signed=True)
                    pdata[idx, :] = tmp_data
            else:
                for idx, p in enumerate(pulse_nums):
                    try:
                        pp = self.packet_points['data'][p - 1]
                    except:
                        print('IDX is {} and p is {}'.format(idx, p))
                        pp = self.packet_points['data'][p - 1]
                    mm.seek(pp)
                    data = mm.read(28)
                    tmp_data = np.zeros((pdata.shape[1],))
                    data = mm.read(2 * pdata.shape[1])
                    for n in range(pdata.shape[1]):
                        tmp_data[n] = int.from_bytes(data[2 * n:2 * n + 2], byteorder='big', signed=True)#  * \
                                      # 10 ** (self.att[pulse_nums[n - 1]] / 20)
                    pdata[idx, :] = tmp_data
        return pdata

    def readPacketPoints(self):
        file_pos = 0
        packet_types = [('data', b'\x0A\xB0\xFF\x18')]
        packet_points = {'data': []}
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            for pt in packet_types:
                file_pos = 0
                stf = mm.find(pt[1], file_pos)
                while stf != -1:
                    packet_points[pt[0]].append(stf)
                    file_pos = stf + 1
                    stf = mm.find(pt[1], file_pos)
        packet_points['data'] = np.array(packet_points['data'])
        return packet_points

    def readDataParams(self):
        nframes = len(self.packet_points['data'])
        cals = np.zeros((nframes,))
        nsamples = np.zeros((nframes,), dtype=int)
        sys_times = np.zeros((nframes,))
        frame_num = np.zeros((nframes,), dtype=int)
        atts = np.zeros((nframes,), dtype=int)
        frame_type = np.zeros((nframes,), dtype=int)
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            for idx, pulse in enumerate(self.packet_points['data']):
                mm.seek(pulse)
                data = mm.read(28)
                cals[idx] = True if (data[12] & 14) == 0 else False
                nsamples[idx] = int.from_bytes(data[14:16], byteorder='big', signed=False)
                sys_times[idx] = int.from_bytes(data[8:12], byteorder='big', signed=False)
                frame_num[idx] = int.from_bytes(data[4:8], byteorder='big', signed=False)
                atts[idx] = int.from_bytes(data[25:26], byteorder='big', signed=False)
                mm.seek(pulse - 1)
                data = mm.read(1)
                frame_type[idx] = int.from_bytes(data, byteorder='big', signed=False)
        return cals, nsamples, sys_times, frame_num, atts, frame_type


class SDRMiniParser(object):
    def __init__(self, fnme=None):

        # Get the file we'll be parsing, and some extra data for it
        self.fnme = fnme

        # Read the points in the file where the packets we care about lie hidden
        self.packet_points = self.readPacketPoints()
        self.cal, self.n_samples, tmp_times, self.frameNum, self.att = \
            self.readDataParams()

        # There may be a few stragglers that accidentally have the matching words.
        # Let's remove them.
        true_nsamples = self.n_samples[0]
        weirdos_mask = self.n_samples == true_nsamples
        self.packet_points['data'] = self.packet_points['data'][weirdos_mask]
        self.cal = self.cal[weirdos_mask].astype(bool)
        self.n_samples = self.n_samples[weirdos_mask]
        tmp_times = tmp_times[weirdos_mask]
        self.sys_time = np.zeros((len(tmp_times),))
        self.frameNum = np.arange(len(self.packet_points['data'])) + 1
        self.att = self.att[weirdos_mask]

    '''
    getPulse
    Returns a single pulse.

    pulse_num - the pulse number or frame number from the file.
    '''

    def getPulse(self, pulse_num):
        pp = self.packet_points['data'][pulse_num - 1]
        n_samples = self.n_samples[pulse_num - 1]
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            mm.seek(pp + 32)
            tmp_data = np.zeros((n_samples,), dtype=np.complex64)
            for n in range(n_samples):
                tmp_data[n] = (int.from_bytes(mm.read(4), byteorder='big', signed=True) + \
                               1j * int.from_bytes(mm.read(4), byteorder='big', signed=True)) * \
                              10 ** (self.att[pulse_num - 1] / 20)
        return tmp_data

    def getPulses(self, pulse_nums, verbose=False):
        pdata = np.zeros((len(pulse_nums), self.n_samples[0]), dtype=np.complex64)
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            for idx, p in enumerate(pulse_nums):
                try:
                    pp = self.packet_points['data'][p - 1]
                except:
                    print('IDX is {} and p is {}'.format(idx, p))
                    pp = self.packet_points['data'][p - 1]
                mm.seek(pp + 32)
                tmp_data = np.zeros((pdata.shape[1],), dtype=np.complex64)
                for n in range(pdata.shape[1]):
                    tmp_data[n] = int.from_bytes(mm.read(4), byteorder='big', signed=True) + \
                        1j * int.from_bytes(mm.read(4), byteorder='big', signed=True)
                pdata[idx, :] = tmp_data * 10 ** (self.att[p - 1] / 20)
        return pdata

    def readPacketPoints(self):
        file_pos = 0
        packet_types = [('data', b'\x0A\xB0\xFF\x18')]
        packet_points = {'data': []}
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            for pt in packet_types:
                file_pos = 0
                stf = mm.find(pt[1], file_pos)
                while stf != -1:
                    packet_points[pt[0]].append(stf)
                    file_pos = stf + 1
                    stf = mm.find(pt[1], file_pos)
        packet_points['data'] = np.array(packet_points['data'])
        return packet_points

    def readDataParams(self):
        nframes = len(self.packet_points['data'])
        cals = np.zeros((nframes,))
        nsamples = np.zeros((nframes,), dtype=int)
        sys_times = np.zeros((nframes,))
        frame_num = np.zeros((nframes,), dtype=int)
        atts = np.zeros((nframes,), dtype=int)
        with open(self.fnme, 'r+b') as fin:
            mm = mmap.mmap(fin.fileno(), 0)
            for idx, pulse in enumerate(self.packet_points['data']):
                mm.seek(pulse)
                # Skip start word
                mm.read(4)
                frame_num[idx] = int.from_bytes(mm.read(4), byteorder='big', signed=False)
                sys_times[idx] = int.from_bytes(mm.read(4), byteorder='big', signed=False)
                mode = int.from_bytes(mm.read(8), byteorder='big', signed=False)
                # Calc channel number
                channel = mode & COLLECTION_MODE_DIGITAL_CHANNEL_MASK >> COLLECTION_MODE_DIGITAL_CHANNEL_SHIFT
                is_cal = mode & COLLECTION_MODE_OPERATION_MASK >> COLLECTION_MODE_OPERATION_SHIFT
                cals[idx] = True if is_cal == 1 else False
                nsamples[idx] = int.from_bytes(mm.read(4), byteorder='big', signed=False)
                # Att is only 5 bits so we mask it
                atts[idx] = int.from_bytes(mm.read(1), byteorder='big', signed=False) & 0x1f
                # Skip AGC/reserved data stuff
                mm.read(7)
        return cals, nsamples, sys_times, frame_num, atts


def readPulseFromStream(data):
    pt = data.find(START_WORD)

