import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from tqdm import tqdm

TAC = 125e6
c0 = 299792458
fs = 2e9
DTR = np.pi / 180
BYTES_PER_SAMPLE = 4
BYTES_TO_MB = 1048576
GRID_SZ = 80
MAX_PRFB = 20
MIN_PRFB = 1.5
MAX_DR = 600
MAX_DC = .25


class Channel(object):
    def __init__(self, alt, near_r, far_r, doppPRF, tonTAC):
        self.doppPRF = doppPRF
        self.slmin = alt / np.sin(near_r * DTR)
        self.slmax = alt / np.sin(far_r * DTR)
        self.prfB = 1.5
        self.presum = 1
        self.plp = .01
        tp = 2 * self.slmin / c0 * .99
        # Ensure that SlimSDR can handle pulse length
        if (int(2 * self.slmin / c0 * TAC) - int(tp * TAC)) < 512:
            tp -= 512 / TAC
        self.tp = tp
        self.recON = np.floor(2 * self.slmin / c0 * TAC + tonTAC)
        self.tonTAC = tonTAC
        self.minPRFB = MIN_PRFB
        self.minPresum = 1
        self.minPLP = .01

    def calcPRI(self):
        return np.ceil(TAC / (self.doppPRF * self.prfB * self.presum))

    def calcSampleDepth(self):
        recOFF = np.ceil((2 * self.slmax / c0 + self.tp * self.plp) * TAC + self.tonTAC)
        return (recOFF - self.recON) * fs / TAC

    def calcDutyCycle(self):
        return (1.025e-6 + self.tp * self.plp) * self.doppPRF * self.prfB * self.presum

    def calcDataRate(self):
        recOFF = np.ceil((2 * self.slmax / c0 + self.tp * self.plp) * TAC + self.tonTAC)
        return self.doppPRF * self.prfB * ((recOFF - self.recON) * fs / TAC) * BYTES_PER_SAMPLE / BYTES_TO_MB

    def calcRecOFF(self):
        return np.ceil((2 * self.slmax / c0 + self.tp * self.plp) * TAC + self.tonTAC)

    def getOpt(self):
        return (MAX_PRFB - self.prfB) / MAX_PRFB * .01 + (1 - self.plp) * .15 + (MAX_DC - self.calcDutyCycle()) / MAX_DC * .75

    def setParams(self, prfB, plp, presum):
        self.prfB = prfB
        self.plp = plp
        self.presum = presum

    def setMinParams(self, prfB, plp, presum):
        self.minPRFB = prfB
        self.minPLP = plp
        self.minPresum = presum


ch1 = Channel(5000 / 3.2808, 39, 32, 1604.02483246, 1000)
ch2 = Channel(5000 / 3.2808, 39, 32, 1432.33210926, 1000)
channels = [ch1, ch2]

# Constraints
pri_con = lambda x: x.calcPRI() > x.calcRecOFF()
dc_con = lambda x: 0 < x.calcDutyCycle() < MAX_DC
ps_con = lambda x: 0 < x.plp < 1
pb_con = lambda x: MIN_PRFB < x.prfB < MAX_PRFB
dr_con = lambda x: 0 < sum([c.calcDataRate() for c in x]) < MAX_DR
samp_con = lambda x: x.calcSampleDepth() < 65536

cons = [pri_con, dc_con, ps_con, samp_con]
abs_par = [0] + [0 for c in channels]
abs_pfs = []
abs_min = 1e9
total_presums = [c for c in combinations_with_replacement(np.arange(1, 12), len(channels))]
poss_presums = total_presums

for pfs in tqdm(poss_presums):
    curr_par = [ch1.minPRFB] + [c.minPLP for c in channels]
    tmp_par = [ch1.minPRFB] + [c.minPLP for c in channels]
    cm = 1e9
    sX = (MAX_PRFB - MIN_PRFB) / GRID_SZ
    sY = (1 - .01) / GRID_SZ
    ch_i = [0] + [0 for c in channels]
    dopt = 1
    iters = 0
    while (dopt > .00001 and dopt != 0) or iters < 5:
        curr_par = tmp_par
        while ch_i[0] < GRID_SZ:
            for i in range(len(ch_i) - 1, -1, -1):
                if ch_i[i] >= GRID_SZ:
                    ch_i[i] = 0
                else:
                    ch_i[i] += 1
                    break
            # print(ch_i)
            currPRFB = curr_par[0] + (GRID_SZ / 2 - ch_i[0]) * sX
            if MIN_PRFB < currPRFB < MAX_PRFB:
                currPLP = [curr_par[n] + (GRID_SZ / 2 - ch_i[n]) * sY for n in range(1, len(ch_i))]
                if np.all([0 < c < 1 for c in currPLP]):
                    ch1.setParams(currPRFB, currPLP[0], pfs[0])
                    ch2.setParams(currPRFB, currPLP[1], pfs[1])
                    pcon = True
                    for c in cons:
                        for ch in channels:
                            if pcon:
                                pcon = c(ch)
                        if not pcon:
                            break
                    if pcon:
                        pcon = dr_con(channels)
                    if pcon:
                        currMin = sum([c.getOpt() for c in channels])
                        if currMin < cm:
                            cm = currMin
                            tmp_par = [currPRFB] + currPLP
        dopt = np.linalg.norm([tmp_par[n] - curr_par[n] for n in range(len(curr_par))])
        if dopt == 0:
            sX *= .3
            sY *= .3
        iters += 1
    if cm < abs_min:
        abs_min = cm
        abs_par = tmp_par
        abs_pfs = pfs
ch1.setParams(abs_par[0], abs_par[1], abs_pfs[0])
ch2.setParams(abs_par[0], abs_par[2], abs_pfs[1])
ch1.setMinParams(abs_par[0], abs_par[1], abs_pfs[0])
ch2.setMinParams(abs_par[0], abs_par[2], abs_pfs[1])

print('Channel 0')
print('Data Rate\tDutyCycle\tPRI')
print(f'{ch1.calcDataRate()}\t{ch1.calcDutyCycle()}\t{ch1.calcPRI()}')
print('Channel 1')
print('Data Rate\tDutyCycle\tPRI')
print(f'{ch2.calcDataRate()}\t{ch2.calcDutyCycle()}\t{ch2.calcPRI()}')






