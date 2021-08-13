import sys
from socket import socket, AF_INET, SOCK_STREAM, create_connection
from scipy.signal.windows import taylor

from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QMainWindow, QCheckBox
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QToolBar, QWidget, QSpinBox
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pylab as plab
# from celluloid import Camera
from parserGUI import db, findPowerOf2
from parsing import SARMiniParser, SDRMiniParser, readPulseFromStream
import threading

START_WORD = b'\x0A\xB0\xFF\x18'
STOP_WORD = b'\xAA\xFF\x11\x55'
COLLECTION_MODE_DIGITAL_CHANNEL_MASK = 0x000003E000000000
COLLECTION_MODE_OPERATION_MASK = 0x1800000000000000
COLLECTION_MODE_DIGITAL_CHANNEL_SHIFT = 37
COLLECTION_MODE_OPERATION_SHIFT = 59


class QuickView(QWidget):
    def __init__(self, pulse, mf=None):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Single Pulse Check")
        self.disp = MatplotlibWidget()
        layout.addWidget(self.label)
        layout.addWidget(self.disp)
        buttons = QHBoxLayout()
        spec_but = QPushButton('Spectrum')
        rc_but = QPushButton('Range Compression')
        if mf is None:
            rc_but.setVisible(False)
        spec_but.clicked.connect(lambda: self.drawSpectrum(pulse))
        rc_but.clicked.connect(lambda: self.drawRC(pulse, mf))
        buttons.addWidget(spec_but)
        buttons.addWidget(rc_but)
        layout.addLayout(buttons)
        self.setLayout(layout)
        xes = np.arange(len(pulse))
        self.disp.plot(xes, np.real(pulse), y2=np.imag(pulse))

    def redraw(self, pulse):
        xes = np.arange(len(pulse))
        self.disp.plot(xes, np.real(pulse), y2=np.imag(pulse))

    def drawSpectrum(self, pulse):
        xes = np.fft.fftfreq(findPowerOf2(len(pulse)) * 4)
        self.disp.plot(xes, db(np.fft.fft(pulse, n=findPowerOf2(len(pulse)) * 4)))

    def drawRC(self, pulse, filt):
        xes = np.arange(len(filt) * 4)
        self.disp.plot(xes, db(np.fft.ifft(np.fft.fft(pulse, n=len(filt)) * filt, n=len(filt) * 4)))


class CPI_Window(QMainWindow):
    _signal_statusbar = pyqtSignal(str)
    _signal_datareceived = pyqtSignal(bytes)
    _signal_controlreceived = pyqtSignal(bytes)
    _signal_dataupdate = pyqtSignal(str)
    _signal_controlupdate = pyqtSignal(str)
    _signal_singlepulse = pyqtSignal()
    _signal_filterdone = pyqtSignal()
    _signal_stopprocess = pyqtSignal()
    _signal_plotcpi = pyqtSignal()
    """Main Window."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)

        self.listen = False
        self.port_address = 11788
        self.data_address = 11789
        self.data_receiving = False
        self.error = None
        self.matched_filter = None
        self.mfcount = 0
        self._hasFilter = False
        self.load_filter = False
        self.plot_flag = False
        self.shutdown = False
        self.fft_len = 0
        self.current_cpi = []
        self.cpi_len = 128
        self.win = None

        # Connect all the signals to their slots
        self._signal_statusbar.connect(self._slotStatusMessage)
        self._signal_datareceived.connect(self._slotParseData)
        self._signal_controlreceived.connect(self._slotReceiveData)
        self._signal_dataupdate.connect(self._slotUpdateDataInfo)
        self._signal_controlupdate.connect(self._slotUpdateControlInfo)
        self._signal_singlepulse.connect(self._slotPlotPulse)
        self._signal_filterdone.connect(self._slotGenMatchedFilter)
        self._signal_stopprocess.connect(self._slotStopProcess)
        self._signal_plotcpi.connect(self._slotPlotCPI)

        # Parse Layout widgets
        socket_layout = QHBoxLayout()
        self.port_box = QSpinBox()
        self.port_box.setMaximum(50000)
        self.port_box.setMinimum(0)
        self.port_box.setValue(self.port_address)
        listen_but = QPushButton('Change Port')
        first_but = QPushButton('Show CPI')
        last_but = QPushButton('Show Single Pulse')
        stop_but = QPushButton('Stop')

        # Add widgets to layout
        socket_layout.addWidget(QLabel('Port:'))
        socket_layout.addWidget(self.port_box)
        socket_layout.addWidget(listen_but)
        socket_layout.addWidget(first_but)
        socket_layout.addWidget(last_but)
        socket_layout.addWidget(stop_but)

        # self.data_listener = threading.Thread(target=self._data_listen, args=(self.data_address,))
        # self.data_listener.start()
        self.data_listener = None
        self.control_listener = threading.Thread(target=self._listen, args=(self.port_address,))
        self.control_listener.start()
        self.plotter = threading.Thread(target=self._plot_thread)
        self.plotter.start()
        self.statusBar().showMessage('Listening on port {}'.format(self.port_address))

        listen_but.clicked.connect(lambda: self._slotChangePort())
        first_but.clicked.connect(lambda: self._signal_plotcpi.emit())
        last_but.clicked.connect(lambda: self._signal_singlepulse.emit())
        stop_but.clicked.connect(lambda: self._signal_stopprocess.emit())

        # Opts Layout widgets
        opts_layout = QHBoxLayout()
        self.cpi_len_box = QSpinBox()
        self.cpi_len_box.setMaximum(15000)
        self.cpi_len_box.setMinimum(16)
        self.cpi_len_box.setValue(self.cpi_len)
        self.cpi_len_box.valueChanged.connect(self._setCPILength)

        # Opts layout add widgets
        opts_layout.addWidget(QLabel('CPI Length:'))
        opts_layout.addWidget(self.cpi_len_box)

        # Info layout
        info_layout = QHBoxLayout()
        self.control_info = QLabel('No control server connected.')
        self.data_info = QLabel('No data server connected.')

        info_layout.addWidget(QLabel('Control:'))
        info_layout.addWidget(self.control_info)
        info_layout.addWidget(QLabel('Data:'))
        info_layout.addWidget(self.data_info)

        # Main window
        window = QWidget()
        window.setWindowTitle('Spectrum Analyzer')
        layout = QVBoxLayout()
        self.display = MatplotlibWidget()

        # Add all the other layouts to the main display
        layout.addWidget(self.display)
        layout.addLayout(socket_layout)
        layout.addLayout(opts_layout)
        layout.addLayout(info_layout)
        window.setLayout(layout)

        # Menus and toolbars for the main window
        self.setCentralWidget(window)
        self._createMenu()
        self._createToolBar()

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Exit', self.close)

    def _createToolBar(self):
        tools = QToolBar()
        self.addToolBar(tools)
        tools.addAction('Exit', self.close)

    def _listen(self, pno):
        sock = socket(AF_INET, SOCK_STREAM)
        try:
            sock.bind(('localhost', pno))
        except PermissionError:
            self._signal_controlupdate.emit("Permission Denied.")
            return
        except OSError:
            self._signal_controlupdate.emit("Address already in use.")
            return
        sock.listen(1)
        self._signal_controlupdate.emit("Listening.".format(self.port_address))
        while True:
            connection, client_address = sock.accept()
            connection.send(bytes([0, 0, 0, 1]))
            break
        while True:
            try:
                while True:
                    data = connection.recv(5)
                    # print(str(data))
                    if not self.data_receiving:
                        self._signal_controlreceived.emit(data)
                    connection.send(bytes([0, 1, 0, 1]))
                    if not data:
                        break
            finally:
                connection.close()
                self._signal_controlupdate.emit("Listening.".format(self.port_address))

    def _data_listen(self, pno):
        try:
            sock = create_connection(('localhost', pno))
        except PermissionError:
            self._signal_dataupdate.emit("Permission Denied.")
            return
        except OSError:
            self._signal_dataupdate.emit("Address already in use.")
            return
        # sock.send(bytes([1, 0, 1, 0]))
        try:
            self._signal_dataupdate.emit("Receiving.".format(self.data_address))
            self.data_receiving = True
            prev_search = bytes([0, 0, 0, 0])
            while not self.shutdown:
                search_data = sock.recv(4)
                sd_comb = prev_search + search_data
                st_word = sd_comb.find(START_WORD)
                if st_word != -1:
                    # Receive the header data
                    data = sd_comb[st_word + 4:] + sock.recv(28)
                    frame = int.from_bytes(data[:4], byteorder='big', signed=False)
                    systime = int.from_bytes(data[4:8], byteorder='big', signed=False)
                    mode = int.from_bytes(data[8:16], byteorder='big', signed=False)
                    is_cal = (mode & COLLECTION_MODE_OPERATION_MASK) >> COLLECTION_MODE_OPERATION_SHIFT
                    nsam = int.from_bytes(data[16:20], byteorder='big', signed=False)
                    att = data[20] & 0x1f
                    self._signal_statusbar.emit('Frame {}, cal: {}, nsam: {}, att: {}, systime: {}'.format(frame,
                                                                                                           is_cal,
                                                                                                           nsam,
                                                                                                           att,
                                                                                                           systime))
                    pulse = np.zeros((nsam,), dtype=np.complex128)
                    data = sock.recv(nsam * 4)
                    for i, n in enumerate(range(0, len(data), 4)):
                        pulse[i] = int.from_bytes(data[n:n + 2], byteorder='big', signed=True) + 1j * \
                                   int.from_bytes(data[n + 2:n + 4], byteorder='big', signed=True) * \
                                   (10 ** (att / 20))
                    if is_cal:
                        if self.matched_filter is None:
                            self.matched_filter = pulse
                        elif not self._hasFilter:
                            self.matched_filter += pulse
                        self.mfcount += 1
                    if len(self.current_cpi) >= self.cpi_len:
                        self.current_cpi = self.current_cpi[-(self.cpi_len + 1):]
                    self.current_cpi.append(pulse)
                    if (self.mfcount > 100 and not is_cal) and not self._hasFilter:
                        self._signal_filterdone.emit()
                    if self.fft_len == 0:
                        self.fft_len = findPowerOf2(len(pulse))
                prev_search = search_data
                # self._signal_datareceived.emit(data)
                if not search_data:
                    break
        finally:
            sock.close()
            self.data_receiving = False

    @pyqtSlot()
    def _slotPlotCPI(self):
        self.plot_flag = True

    def _plot_thread(self):
        while True:
            if self.load_filter:
                self.matched_filter = np.fft.fft(self.matched_filter / self.mfcount, self.fft_len) * taylor(self.fft_len, nbar=6, sll=30)
                self._hasFilter = True
                self._signal_statusbar.emit('Matched Filter generated.')
                self.load_filter = False
            if self.plot_flag:
                disp_cpi = np.array(self.current_cpi)
                fft_cpi = np.fft.fft(disp_cpi, n=self.fft_len, axis=1)
                if self._hasFilter:
                    fft_cpi = np.fft.fft(fft_cpi * self.matched_filter[None, :], axis=1)
                self.display.imshow(db(fft_cpi).T)
                self.plot_flag = False

    def _setCPILength(self):
        self.cpi_len = self.cpi_len_box.value()

    def closeEvent(self, event):
        if self.data_listener is not None:
            self.data_listener.join(timeout=.00001)
        self.control_listener.join(timeout=.00001)
        self.plotter.join(timeout=.00001)
        event.accept()

    @pyqtSlot()
    def _slotGenMatchedFilter(self):
        self.load_filter = True

    @pyqtSlot()
    def _slotPlotPulse(self):
        if self.win is None:
            self.win = QuickView(self.current_cpi[0], self.matched_filter)
        else:
            self.win.redraw(self.current_cpi[0])
        self.win.show()

    @pyqtSlot(str)
    def _slotStatusMessage(self, message):
        self.statusBar().showMessage(message)

    @pyqtSlot(str)
    def _slotUpdateDataInfo(self, message):
        message = 'Port {}, '.format(self.data_address) + message
        self.data_info.setText(message)

    @pyqtSlot(str)
    def _slotUpdateControlInfo(self, message):
        message = 'Port {}, '.format(self.port_address) + message
        self.control_info.setText(message)

    @pyqtSlot(bytes)
    def _slotReceiveData(self, rec_packet):
        pno = int.from_bytes(rec_packet[3:], 'big', signed=False)
        if pno != self.data_address:
            if self.data_listener is not None:
                self.data_listener.join(timeout=.00001)
            self.data_listener = threading.Thread(target=self._data_listen, args=(pno,))
            self.data_listener.start()
            self.data_address = pno
            self._signal_dataupdate.emit('Listening.')

    @pyqtSlot(str)
    def _slotChangePort(self):
        self.control_listener.join(timeout=.00001)
        self.port_address = self.port_box.value()
        self.control_listener = threading.Thread(target=self._listen, args=(self.port_address,))
        self.control_listener.start()
        self._signal_controlupdate.emit("Listening.")

    @pyqtSlot()
    def _slotStopProcess(self):
        self.shutdown = True

    @pyqtSlot(bytes)
    def _slotParseData(self, data):
        print('sup')


class MatplotlibWidget(QWidget):

    def __init__(self, size=(5.0, 4.0), dpi=100):
        QWidget.__init__(self)
        plt.ion()
        self.fig = Figure(size, dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)
        self.ylim = [100, 1]

        self.setLayout(self.vbox)

    def set_ylim(self, min_lim, max_lim):
        self.ylim = [min_lim, max_lim]

    def plot(self, x, y, y2=None, title=''):
        self.ax.cla()
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time')
        self.ax.plot(x, y, 'b-')
        if y2 is not None:
            self.ax.plot(x, y2, 'r-')
        self.ax.set_title(title)
        # self.ax.set_ylim(self.ylim)

        self.canvas.draw()
        self.canvas.flush_events()

    def imshow(self, data):
        self.ax.cla()
        self.ax.imshow(data)
        self.ax.axis('tight')
        self.canvas.draw()
        self.canvas.flush_events()


if __name__ == '__main__':
    # Instantiate the window
    app = QApplication(sys.argv)
    win = CPI_Window()
    win.show()

    # Run main loop
    sys.exit(app.exec_())

