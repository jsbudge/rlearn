# Author: Eric Bezzam
# Date: July 15, 2016

from doa import *

class SRP(DOA):
    """
    Class to apply Steered Response Power (SRP) direction-of-arrival (DoA) for 
    a particular microphone array.

    .. note:: Run locate_source() to apply the SRP-PHAT algorithm.

    :param L: Microphone array positions. Each column should correspond to the 
    cartesian coordinates of a single microphone.
    :type L: numpy array
    :param fs: Sampling frequency.
    :type fs: float
    :param nfft: FFT length.
    :type nfft: int
    :param c: Speed of sound. Default: 343 m/s
    :type c: float
    :param num_src: Number of sources to detect. Default: 1
    :type num_src: int
    :param mode: 'far' or 'near' for far-field or near-field detection 
    respectively. Default: 'far'
    :type mode: str
    :param r: Candidate distances from the origin. Default: np.ones(1)
    :type r: numpy array
    :param theta: Candidate azimuth angles (in radians) with respect to x-axis.
    Default: np.linspace(-180.,180.,30)*np.pi/180
    :type theta: numpy array
    :param phi: Candidate elevation angles (in radians) with respect to z-axis.
    Default is x-y plane search: np.pi/2*np.ones(1)
    :type phi: numpy array
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None, 
        theta=None, phi=None, **kwargs):
        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, theta=theta, phi=phi)
        self.num_pairs = self.M*(self.M-1)/2
        self.mode_vec = np.conjugate(self.mode_vec)

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response 
        spectrum.
        """
        # average over snapshots
        for s in range(self.num_snap):
            X_s = X[:,self.freq_bins,s]
            absX = abs(X_s)
            absX[absX<tol] = tol 
            pX = X_s/absX
            # grid search
            for k in range(self.num_loc):
                Yk = pX.T * self.mode_vec[self.freq_bins,:,k]
                CC = np.dot(np.conj(Yk).T, Yk)
                self.P[k] = self.P[k]+abs(np.sum(np.triu(CC,1)))/ \
                    self.num_snap/self.num_freq/self.num_pairs
