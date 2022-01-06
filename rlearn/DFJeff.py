import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d
from numba import cuda
import numba
from cmath import exp


def MUSICSinglePoint(
        a_receivedSignals,
        a_targetNumber=1,
        a_antennaElementPositions=None,
        a_receiveSignalWavelength=None,
        a_gridSpacing=0.01,
        a_azimuthRange=None,
        a_elevationRange=None,
        a_plotSingleDFResults=False,
        a_scaleElevation=True):
    """Uses a 2D version of the MUSIC algorithm to perform direction 
    finding. Estimates an azimuth and elevation angle to a target emitter.
    
    param: a_receivedSignals: This is an matrix containing the sampled data
        from each array element. Each row contains the N samples of data 
        for a single element.
    param: a_targetNumber: The number of targets present. If 'None' the
        number of targets is estimated from the data.
    param: a_antennaElementPositions: The positions of the antenna elements
        relative to one another. The shape is 
        [antenna element number, XYZ position (meters)]. The code only 
        cares about relative positions to each other, so the array can be 
        rotated in whatever orientation is easiest (eg alligned with the x 
        axis).
    param: a_receiveSignalWavelent: The wavelength of the received signal 
        that you want to perform direction finding on
    param: a_gridSpacing: How fine the azimuth and elevation search grid
        will be. Units are in radians.
    param: a_azimuthRange: Custom bounds for the azimuth search grid. 
        Format is np.array([lowerBound, upperBound]).
    param: a_elevationRange: Same as a_azimuthRange, but in elevation.
    param: a_plotSingleDFResults: Set to True to plot the azimuth slice,
        elevation slice, and 3D MUSIC Spectrum
    param: a_scaleElevation: The elevation angle has to be scaled by the
        emitter wavelength. In the interest of keeping this function self
        contained, the scaling can be done here for singular points. 
        However, for full trajectories, this scaling is better done outside
        this function after target association. This flag allows you to 
        choose whether or not scaling happens here or elsewhere.
    """

    # Some error checking
    if a_antennaElementPositions is None:
        raise Exception("ERROR: Antenna Positions must be provided.")
    if a_receiveSignalWavelength is None:
        raise Exception("ERROR: The wavelength of the received signal " + \
                        "must be provided.")

    wavelength = a_receiveSignalWavelength
    relativePositionBody = a_antennaElementPositions

    # Perform a check on the target number vs the antenna number.
    targetNumber = a_targetNumber
    antennaNumber = relativePositionBody.shape[0]
    if targetNumber > (antennaNumber - 1):
        print("WARNING: The number of targets is greater than the " + \
              "max number of detectable targets for given array.")

    # Create the sample covariance from the data
    numberSamples = a_receivedSignals.shape[1]
    sampleCovariance = (1 / numberSamples) * np.matmul(a_receivedSignals,
                                                       np.transpose(np.conjugate(a_receivedSignals)))

    # Find the Nullspace matrix
    eigenValues, eigenVectors = np.linalg.eig(sampleCovariance)
    nullSpace = eigenVectors[:, targetNumber:antennaNumber]

    # Create search grids
    if a_azimuthRange is None:
        azimuthRange = np.array([0, 2 * np.pi])
    else:
        azimuthRange = a_azimuthRange
    if a_elevationRange is None:
        elevationRange = np.array([0, np.pi / 2])
    else:
        elevationRange = a_elevationRange
    if (np.size(azimuthRange) != 2) or (np.size(elevationRange) != 2):
        raise Exception("Specified range is not in the proper format. " +
                        "Use 'np.array([lowerBound, upperBound])'.")
    gridSpacingRad = a_gridSpacing
    azimuthGrid = np.arange(
        azimuthRange[0], azimuthRange[1], gridSpacingRad)
    elevationGrid = np.arange(
        elevationRange[0], elevationRange[1], gridSpacingRad)
    spectrumMUSIC = np.zeros(
        [np.size(azimuthGrid), np.size(elevationGrid)])

    # Check for an available GPU. If available, perform grid search on GPU.
    if cuda.is_available():
        if antennaNumber > 16:
            print("WARNING: GPU detected, but not used. CUDA " + \
                  "requires hard coding thread-local array " + \
                  "sizes. This warning is raised if the " + \
                  "number of antenna elements exceeds the " + \
                  "hard coded array size in the CUDA kernel. " + \
                  "To use the GPU, manually change the array " + \
                  "size in the _musicSearchGPU function and " + \
                  "the 'if' statement triggering this warning.")
        else:
            spectrumMUSIC, peakIndices, estimatedAngles = \
                _runMUSICGPU(azimuthGrid, elevationGrid,
                             spectrumMUSIC, relativePositionBody, targetNumber,
                             antennaNumber, wavelength, nullSpace)
    else:
        raise Exception("ERROR: No GPU found for MUSIC processing.")

    # Optionally plot the results
    if a_plotSingleDFResults:
        _createMUSICPlots(True, True, True, azimuthGrid,
                          elevationGrid, spectrumMUSIC, peakIndices, estimatedAngles)

    return estimatedAngles.T


def _createMUSICPlots(a_plotSpectrumFlag, a_plotAzimuthSlicesFlag,
                      a_plotElevationSlicesFlag, a_azimuthGrid, a_elevationGrid,
                      a_spectrumMUSIC, a_peakIndices, a_estimatedAngles):
    """Handles plotting of all MUSIC related information

    param: a_plotSpectrumFlag: Set to true to plot the MUSIC spectrum.
    param: a_plotAzimuthSlicesFlag: Set to true to plot spectrum peak by
        sweeping through elevation at a constant azimuth angle.
    param: a_plotElevationSlicesFlag: Set to true to plot spectrum peak by
        sweeping through azimuth at a constant elevation angle.
    param: azimuthGrid: an array containing the azimuth values in the MUSIC
        grid search
    param: elevationGrid: an array containing the elevation values in the
        MUSIC grid search
    param: spectrumMUSIC: The result of the MUSIC algorithm
    param: peakIndices: The result of the peak extraction function
    param: estimatedAngles: Angles based on the peaks in the MUSIC spectrum
    """

    # Plot the MUSIC spectrum.
    if a_plotSpectrumFlag:
        phi, theta = np.meshgrid(
            a_azimuthGrid * 180 / np.pi, a_elevationGrid * 180 / np.pi)
        fig3D = plt.figure()
        ax3D = plt.axes(projection='3d')
        ax3D.plot_surface(phi, theta, a_spectrumMUSIC.T)
        ax3D.set_xlabel("Azimuth Angle (deg)")
        ax3D.set_ylabel("Elevation Angle (deg)")
        ax3D.set_zlabel("MUSIC Magnitude")
        plt.show(block=False)

    # Plot 2D azimuth slices
    if a_plotAzimuthSlicesFlag:
        peakNumber = np.size(a_peakIndices, axis=0)
        fig2D = plt.figure()
        for peak in range(peakNumber):
            ax = fig2D.add_subplot(peakNumber, 1, peak + 1)
            ax.plot(a_elevationGrid * 180 / np.pi,
                    a_spectrumMUSIC[a_peakIndices[peak, 0], :])
            ax.set_title(("%.2f" % a_estimatedAngles[peak, 0]) + \
                         " Degrees Azimuth")
            ax.set_xlabel("Elevation")
        plt.show(block=False)

    # Plot 2D elevation slices
    if a_plotElevationSlicesFlag:
        peakNumber = np.size(a_peakIndices, axis=0)
        fig2D = plt.figure()
        for peak in range(peakNumber):
            ax = fig2D.add_subplot(peakNumber, 1, peak + 1)
            ax.plot(a_azimuthGrid * 180 / np.pi,
                    a_spectrumMUSIC[:, a_peakIndices[peak, 1]])
            ax.set_title(("%.2f" % a_estimatedAngles[peak, 1]) + \
                         " Degrees Elevation")
            ax.set_xlabel("Azimuth")
        plt.show(block=False)


def _runMUSICGPU(a_azimuthGrid, a_elevationGrid, a_spectrumMUSIC,
                 a_relativePositionBody, a_targetNumber, a_antennaNumber,
                 a_waveLength, a_nullSpace):
    """Handler function that interfaces the Python MUSIC code with CUDA
    code using numba to process MUSIC on the GPU

    param: a_azimuthGrid: An array of azimuth values to search through
    param: a_elevationGrid: An array of elevation values to search through
    param: a_spectrumMUSIC: A matrix of zeros to hold the results of the
        MUSIC computations.
    param: a_relativePositionBody: A matrix of the positions of the antenna
        elements relative to each other. Shape is [xyz, antenna number]
    param: a_targetNumber: The number of peaks you want to pull out of TDOA
    param: a_antennaNumber: The number of elements in the antenna array
    param: a_waveLength: The wavelength of the target signal
    param: a_nullSpace: A matrix representing the nullspace (a.k.a. noise
        space) of the signal.
    """
    # Precalculations for GPU processing
    nullMatrix = a_nullSpace @ np.conjugate(a_nullSpace.T)
    azimuthSin = np.sin(a_azimuthGrid)
    azimuthCos = np.cos(a_azimuthGrid)
    elevationSin = np.sin(a_elevationGrid)
    elevationCos = np.cos(a_elevationGrid)
    spectrumSize = np.size(a_azimuthGrid) * np.size(a_elevationGrid)

    # Use Numba to interface with GPU
    # Calculate the number of threads per block
    gpuDevice = cuda.get_current_device()
    maxThreads = gpuDevice.MAX_THREADS_PER_BLOCK
    sqrtMaxThreads = int(np.sqrt(maxThreads))
    threadsPerBlock = (sqrtMaxThreads, sqrtMaxThreads)

    # Calculate the number of blocks per grid
    blocksPerGridAz = (a_spectrumMUSIC.shape[0] + \
                       (threadsPerBlock[0] - 1)) // threadsPerBlock[0]
    blocksPerGridEl = (a_spectrumMUSIC.shape[1] + \
                       (threadsPerBlock[1] - 1)) // threadsPerBlock[1]
    blocksPerGrid = (blocksPerGridAz, blocksPerGridEl)

    # Call the CUDA Kernal
    _musicSearchGPU[blocksPerGrid, threadsPerBlock](
        nullMatrix, a_waveLength, a_antennaNumber,
        a_relativePositionBody, azimuthSin, azimuthCos,
        elevationSin, elevationCos, a_spectrumMUSIC)
    cuda.synchronize()

    # Peak Estimation on the GPU
    isPeak = np.zeros(np.shape(a_spectrumMUSIC), dtype=int)
    _estimatePeaksGPU[blocksPerGrid, threadsPerBlock](
        isPeak, a_spectrumMUSIC, a_targetNumber)
    cuda.synchronize()

    # Extract peak indices
    peakIndices = _extractPeaks(isPeak, a_spectrumMUSIC,
                                a_targetNumber)
    estimatedAngles = np.array(
        [a_azimuthGrid[peakIndices[:, 0]], \
         a_elevationGrid[peakIndices[:, 1]]]).T

    return a_spectrumMUSIC, peakIndices, estimatedAngles


@cuda.jit
def _estimatePeaksGPU(a_isPeak, a_spectrumMUSIC, a_targetNumber):
    """Moves the peak estimation to the GPU. Algorithm isn't super
    complicated. Just checks that no surrounding points are higher than the
    point under test

    param: a_isPeak: matrix of zeros the same size as a_spectrumMUSIC. 
        Zeros are overridden with a one when a peak is detected.
    param: a_spectrumMUSIC: the 2D MUSIC spectrum produced by the
        _musicSearchGPU function.
    param: a_targetNumber: The number of targets being tracked. This is
        either provided by the user or estimated using the 
        estimateTargetNumber function.
    """

    azIdx, elIdx = cuda.grid(2)
    # Check for valid index and avoid all points on the edge
    if (azIdx > 0) and (elIdx > 0) and \
            (azIdx < (a_spectrumMUSIC.shape[0] - 1)) and \
            (elIdx < (a_spectrumMUSIC.shape[1] - 1)):
        candidate = a_spectrumMUSIC[azIdx, elIdx]
        if (candidate > a_spectrumMUSIC[azIdx + 1, elIdx]) and \
                (candidate > a_spectrumMUSIC[azIdx + 1, elIdx + 1]) and \
                (candidate > a_spectrumMUSIC[azIdx + 1, elIdx - 1]) and \
                (candidate > a_spectrumMUSIC[azIdx - 1, elIdx]) and \
                (candidate > a_spectrumMUSIC[azIdx - 1, elIdx + 1]) and \
                (candidate > a_spectrumMUSIC[azIdx - 1, elIdx - 1]) and \
                (candidate > a_spectrumMUSIC[azIdx, elIdx + 1]) and \
                (candidate > a_spectrumMUSIC[azIdx, elIdx - 1]):
            a_isPeak[azIdx, elIdx] = 1


def _extractPeaks(a_isPeak, a_spectrum, a_targetNumber):
    """Uses the isPeak matrix produced by various functions in this class 
    to pull out indices for the peaks in the MUSIC spectrum. This function
    also tries to disregard false peaks and remedy insufficient peaks.

    param: a_isPeak: A matrix of ones and zeros produced by the 
        estimatePeaksGPU function.
    param: a_spectrum: The MUSIC spectrum matrix produced by the
        _musicSearchGPU function.
    param: a_targetNumber: The number of targets being tracked. This is
        either provided by the user or estimated using the 
        estimateTargetNumber function.
    """
    allPeakIndices = np.argwhere(a_isPeak)
    peakNumber = np.size(allPeakIndices, axis=0)
    falsePeakBool = np.zeros(peakNumber, dtype=bool)

    # Locate peaks that are too close to each other as likely false peaks.
    for peak in range(peakNumber):
        # Calculate the 1-norm between peaks
        peakTest = allPeakIndices[peak, :]
        peakDiff = peakTest - allPeakIndices
        peakDist = np.linalg.norm(peakDiff, ord=1, axis=1)

        # False peaks are too close to each other
        falsePeakIndices = allPeakIndices[peakDist < 5.0]
        falsePeakValues = a_spectrum[
            falsePeakIndices[:, 0], falsePeakIndices[:, 1]]

        # Take the true peak to be the largest of all the close peaks
        truePeakValue = np.max(falsePeakValues)

        # Update a boolean array to track the indices of the false peaks.
        falsePeakIndices = falsePeakIndices[
                           falsePeakValues != truePeakValue, :]
        falsePeakValuesNew = a_spectrum[
            falsePeakIndices[:, 0], falsePeakIndices[:, 1]]
        allPeakValues = a_spectrum[
            allPeakIndices[:, 0], allPeakIndices[:, 1]]
        falsePeakBool[allPeakValues == falsePeakValuesNew] = True

    # Remove false peaks using boolean array
    allPeakIndices = (allPeakIndices[np.invert(falsePeakBool), :])

    # Find the peaks with maximum value
    if a_targetNumber > peakNumber:
        a_targetNumber = peakNumber
    maxValueIndices = np.argpartition(
        a_spectrum[allPeakIndices[:, 0], allPeakIndices[:, 1]],
        -a_targetNumber)[-a_targetNumber:]
    peakIndices = allPeakIndices[maxValueIndices, :]

    return peakIndices


@cuda.jit
def _musicSearchGPU(a_nullMatrix, a_waveLength, a_antennaNumber,
                    a_relativePosition, a_azimuthSin, a_azimuthCos, a_elevationSin,
                    a_elevationCos, a_MUSICGridGPU):
    """CUDA kernel to calculate the MUSIC spectrum on the GPU (if an 
    available device is found).

    Note: In order to allocate an array to hold the steering vector, local
        memory was used (cuda.local.array(...)). One peculiarity about 
        local arrays is that the array size has to be hard coded as an
        integer literal rather than a variable. The size is currently set
        at 16. If the number of antenna elements in the MUSIC array ever
        exceeds 16, it will have to be changed manually to a larger value.
    Note: I wasn't able to pass 'self' into the kernel, so some of the
        attributes stored in 'self' had to be passed in manually. 
        e.g. self.waveLength or self.relativePosition.

    param: a_nullMatrix: This is equal to 
        nullMatrix = nullSpace @ np.conjugate(nullSpace.T). This is done
        outside the CUDA kernel to avoid recomputations.
    param: a_waveLength: equal to self.waveLength, which is the wave length
        used when calculating the spacing of antenna elements. This is also
        the wavelength of the emitters you are tracking.
    param: a_relativePosition: equal to self.relativePositionBody, which
        contains information about how the antenna elements are located
        relative to one another.
    param: a_azimuthSin: equal to np.sin(azimuthGrid). CUDA didn't like
        calculating np.sin(...), so I did it outside the kernel.
    param: a_azimuthCos: equal to np.cos(azimuthGrid)
    param: a_elevationSin: equal to np.sin(elevationGrid)
    param: a_elevationCos: equal to np.cos(elevationGrid)
    param: a_MUSICGridGPU: empty 2D array to hold music spectrum 
        calculations.
    """

    # Get and check thread indices
    azimuthIdx, elevationIdx = cuda.grid(2)
    if (azimuthIdx < a_azimuthSin.size) and \
            (elevationIdx < a_elevationSin.size):

        # Pre-allocate local memory of complex data
        steeringVector = cuda.local.array(16, dtype=numba.complex128)

        # Precalculations
        waveNumberRad = 2 * np.pi / a_waveLength

        # Calculate the steering vector for each antenna.
        a_sinTheta = a_elevationSin[elevationIdx]
        a_cosTheta = a_elevationCos[elevationIdx]
        a_cosPhi = a_azimuthCos[azimuthIdx]
        a_sinPhi = a_azimuthSin[azimuthIdx]
        for ant in range(a_antennaNumber):
            phase = waveNumberRad * (a_sinTheta \
                                     * (a_relativePosition[ant, 0] * a_cosPhi \
                                        + a_relativePosition[ant, 1] * a_sinPhi)
                                     + a_relativePosition[ant, 2] * a_cosTheta)
            steeringVector[ant] = exp(1j * phase)

        # Perform the weighted inner product
        a = steeringVector
        magnitudeMUSIC = 0.0
        for i in range(a_antennaNumber):
            for j in range(a_antennaNumber):
                magnitudeMUSIC += a[i].conjugate() * a_nullMatrix[i, j] * a[j]

        # Calculate and save MUSIC spectrum as inverse of MUSIC magnitude
        a_MUSICGridGPU[azimuthIdx, elevationIdx] = 1 / abs(magnitudeMUSIC)
