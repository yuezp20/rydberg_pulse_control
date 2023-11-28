import sys

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton
from functools import partial
import PyQt5.QtCore as qc
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from rydberg_awg import find_waveform, find_Dwaveform
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt

sys.path.append("..")
from res.sequence import Ui_sequence


class Sequence(QMainWindow, Ui_sequence):
    def __init__(self, awg_control, config_name):
        super(Sequence, self).__init__()
        self.setupUi(self)
        self.config_name = config_name
        self.awg_control = awg_control
        self.omega_420_t = np.zeros(1)
        self.omega_1013_t = np.zeros(1)
        self.frequency_420_t = np.zeros(1)
        self.frequency_1013_t = np.zeros(1)

        self.x0_t = np.zeros(1)
        self.x1_t = np.zeros(1)

        self.settings = qc.QSettings("../initialize/config.ini", qc.QSettings.IniFormat)
        self.load_config()

        # self.done.clicked.connect(partial(awg_control.sequence, self))
        self.stop_card.clicked.connect(partial(awg_control.stop_card, self))

        self.p1, self.p2 = self.set_graph_ui()
        # self.btn.clicked.connect(self.plot_sin_cos)
        # self.write_config("../initialize/initial_config.txt")

        # self.waveform_420_1.currentIndexChanged.connect(
        # lambda: self.waveform_changed(self.waveform_420_1.currentText()))

    def set_graph_ui(self):
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', '#F0F0F0')
        pg.setConfigOption('foreground', 'k')

        win = pg.GraphicsLayoutWidget()

        self.graph_layout.addWidget(win)
        p1 = win.addPlot(title="omega")
        p1.showGrid(x=False, y=False)
        p1.setLogMode(x=False, y=False)
        p1.setLabel('bottom', text='time', units='us')
        # p1.addLegend()

        win.nextRow()
        p2 = win.addPlot(title="frequency")
        p2.setLogMode(x=False, y=False)
        p2.setLogMode(x=False, y=False)
        p2.setLabel('bottom', text='time', units='us')
        # p2.addLegend()

        return p1, p2

    def waveform_changed(self, text):
        btn_name = self.sender().objectName()
        print(btn_name)

        if text == "flat pulse":
            self.amp_420_1_1.setVisible(False)
            self.amp_420_1_1.setEnabled(False)
            self.amp_420_1_2.setVisible(False)
            self.amp_420_1_2.setEnabled(False)

        elif text == "blackman pulse":
            self.amp_420_1_1.setVisible(False)
            self.amp_420_1_1.setEnabled(False)
            self.amp_420_1_2.setVisible(False)
            self.amp_420_1_2.setEnabled(False)

        elif text == "ramp frequency":
            self.amp_420_1_1.setVisible(False)
            self.amp_420_1_1.setEnabled(False)
            self.amp_420_1_2.setVisible(False)
            self.amp_420_1_2.setEnabled(False)

        elif text == "multi component":
            self.amp_420_1_1.setVisible(True)
            self.amp_420_1_1.setEnabled(True)
            self.amp_420_1_2.setVisible(True)
            self.amp_420_1_2.setEnabled(True)

    def plot_sin_cos(self):
        t = np.linspace(0, 20, 200)
        y_sin = np.sin(t)
        y_cos = np.cos(t)
        self.p1.plot(t, y_sin, pen='g', name='sin(x)', clear=True)
        self.p2.plot(t, y_cos, pen='g', name='con(x)', clear=True)
        # self.p1.legend = None  # 重新绘图
        # self.p2.legend = None

    # def mousePressEvent(self, event):
    #     if event.button() == Qt.RightButton:
    #         self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.omega_420_t, self.omega_1013_t, self.frequency_420_t, self.frequency_1013_t = self.find_omega_fre_t()
            self.x0_t, self.x1_t = self.find_x_t()
            self.plot_omega_fre_t()
            self.awg_control.sequence(self)

    def done(self):
        self.omega_420_t, self.omega_1013_t, self.frequency_420_t, self.frequency_1013_t = self.find_omega_fre_t()
        self.x0_t, self.x1_t = self.find_x_t()
        self.plot_omega_fre_t()
        self.awg_control.sequence(self)

    def find_x_t(self):
        x0, x1 = find_Dwaveform(self)
        x0_t = np.zeros(int(1000 * x0.time))
        x1_t = np.zeros(int(1000 * x1.time))

        point_formulated = 0
        for i in range(5):
            single_digital_wave_point_num = int(1000 * (x0[2 * i + 1] + x0[2 * i + 2]))
            for point_idx in range(single_digital_wave_point_num):
                if point_idx < int(1000 * x0[2 * i + 1]):
                    x0_t[point_formulated + point_idx] = 1.2
                else:
                    x0_t[point_formulated + point_idx] = 0

            point_formulated = point_formulated + single_digital_wave_point_num

        point_formulated = 0
        for i in range(5):
            single_digital_wave_point_num = int(1000 * (x1[2 * i + 1] + x1[2 * i + 2]))
            for point_idx in range(single_digital_wave_point_num):
                if point_idx < int(1000 * x1[2 * i + 1]):
                    x1_t[point_formulated + point_idx] = 1.2
                else:
                    x1_t[point_formulated + point_idx] = 0

            point_formulated = point_formulated + single_digital_wave_point_num

        return x0_t, x1_t

    def find_omega_fre_t(self):
        waveform_cont_420, waveform_cont_1013 = find_waveform(self)

        omega_420_t = []
        omega_1013_t = []
        frequency_1013_t = []
        frequency_420_t = []

        for single_wave in waveform_cont_420:
            if single_wave.form == "multi component":
                continue
            amp = single_wave.start_amp * 0.01
            start_frequency = single_wave.start_frequency
            end_frequency = single_wave.end_frequency
            time = single_wave.time
            wait_time = single_wave.wait_time

            single_wave_omega_t = np.zeros(int(1000 * (time + wait_time)))
            single_wave_frequency_t = np.zeros(int(1000 * (time + wait_time)))

            if single_wave.form == "flat pulse":
                single_wave_omega_t[:int(1000 * time)] = amp
                single_wave_frequency_t[:] = start_frequency

            elif single_wave.form == "ramp frequency":
                single_wave_omega_t[:int(1000 * time)] = amp
                for i in range(int(1000 * time)):
                    single_wave_frequency_t[i] = start_frequency + (end_frequency - start_frequency) / time * i / 1000

            elif single_wave.form == "blackman pulse":
                for i in range(int(1000 * time)):
                    single_wave_omega_t[i] = amp * (-0.5 * np.cos(2 * np.pi * i / 1000 / time) + 0.08 * np.cos(
                        4 * np.pi * i / 1000 / time) + 0.42)
                single_wave_frequency_t[:] = start_frequency

            omega_420_t = np.append(omega_420_t, single_wave_omega_t)
            omega_420_t = np.asarray(omega_420_t)
            frequency_420_t = np.append(frequency_420_t, single_wave_frequency_t)
            frequency_420_t = np.asarray(frequency_420_t)

        for single_wave in waveform_cont_1013:
            if single_wave.form == "multi component":
                continue

            amp = single_wave.start_amp * 0.01
            start_frequency = single_wave.start_frequency
            end_frequency = single_wave.end_frequency
            time = single_wave.time
            wait_time = single_wave.wait_time

            single_wave_omega_t = np.zeros(int(1000 * (time + wait_time)))
            single_wave_frequency_t = np.zeros(int(1000 * (time + wait_time)))

            if single_wave.form == "flat pulse":
                single_wave_omega_t[:int(1000 * time)] = amp
                single_wave_frequency_t[:] = start_frequency

            elif single_wave.form == "ramp frequency":
                single_wave_omega_t[:int(1000 * time)] = amp
                for i in range(int(1000 * time)):
                    single_wave_frequency_t[i] = start_frequency + (end_frequency - start_frequency) / time * i / 1000

            elif single_wave.form == "blackman pulse":
                for i in range(int(1000 * time)):
                    single_wave_omega_t[i] = amp * (-0.5 * np.cos(2 * np.pi * i / 1000 / time) + 0.08 * np.cos(
                        4 * np.pi * i / 1000 / time) + 0.42)

                single_wave_frequency_t[:] = start_frequency

            omega_1013_t = np.append(omega_1013_t, single_wave_omega_t)
            frequency_1013_t = np.append(frequency_1013_t, single_wave_frequency_t)
            omega_1013_t = np.asarray(omega_1013_t)
            frequency_1013_t = np.asarray(frequency_1013_t)

        return omega_420_t, omega_1013_t, frequency_420_t, frequency_1013_t

    def plot_omega_fre_t(self):
        self.p1.legend = None
        self.p2.legend = None

        point_num = max(len(self.omega_420_t), len(self.omega_1013_t), len(self.x0_t), len(self.x1_t))
        t_total = point_num / 1000
        t = np.linspace(0, t_total, point_num)

        self.omega_420_t = np.pad(self.omega_420_t, (0, point_num - len(self.omega_420_t)),
                                  'constant', constant_values=(0, 0))
        self.omega_1013_t = np.pad(self.omega_1013_t, (0, point_num - len(self.omega_1013_t)),
                                   'constant', constant_values=(0, 0))
        self.x0_t = np.pad(self.x0_t, (0, point_num - len(self.x0_t)),
                           'constant', constant_values=(0, 0))
        self.x1_t = np.pad(self.x1_t, (0, point_num - len(self.x1_t)),
                           'constant', constant_values=(0, 0))

        if len(self.frequency_1013_t) > 0:
            self.frequency_1013_t = np.pad(self.frequency_1013_t,
                                           (0, point_num - len(self.frequency_1013_t)),
                                           'constant', constant_values=(0, self.frequency_1013_t[-1]))
        else:
            self.frequency_1013_t = np.pad(self.frequency_1013_t,
                                           (0, point_num - len(self.frequency_1013_t)),
                                           'constant', constant_values=(0, 0))
        if len(self.frequency_420_t) > 0:
            self.frequency_420_t = np.pad(self.frequency_420_t,
                                          (0, point_num - len(self.frequency_420_t)),
                                          'constant', constant_values=(0, self.frequency_420_t[-1]))
        else:
            self.frequency_420_t = np.pad(self.frequency_420_t,
                                          (0, point_num - len(self.frequency_420_t)),
                                          'constant', constant_values=(0, 0))

        # if len(self.omega_420_t) > len(self.omega_1013_t):
        #     t_total = len(self.omega_420_t) / 1000
        #     t = np.linspace(0, t_total, len(self.omega_420_t))
        #     self.omega_1013_t = np.pad(self.omega_1013_t, (0, len(self.omega_420_t) - len(self.omega_1013_t)),
        #                                'constant', constant_values=(0, 0))
        #
        #     if len(self.frequency_1013_t) > 0:
        #         self.frequency_1013_t = np.pad(self.frequency_1013_t,
        #                                        (0, len(self.omega_420_t) - len(self.frequency_1013_t)),
        #                                        'constant', constant_values=(0, self.frequency_1013_t[-1]))
        #     else:
        #         self.frequency_1013_t = np.pad(self.frequency_1013_t,
        #                                        (0, len(self.omega_420_t) - len(self.frequency_1013_t)),
        #                                        'constant', constant_values=(0, 0))
        # else:
        #     t_total = len(self.omega_1013_t) / 1000
        #     t = np.linspace(0, t_total, len(self.omega_1013_t))
        #     self.omega_420_t = np.pad(self.omega_420_t, (0, len(self.omega_1013_t) - len(self.omega_420_t)),
        #                               'constant', constant_values=(0, 0))
        #
        #     if len(self.frequency_420_t) > 0:
        #         self.frequency_420_t = np.pad(self.frequency_420_t,
        #                                       (0, len(self.omega_1013_t) - len(self.frequency_420_t)),
        #                                       'constant', constant_values=(0, self.frequency_420_t[-1]))
        #     else:
        #         self.frequency_420_t = np.pad(self.frequency_420_t,
        #                                       (0, len(self.omega_1013_t) - len(self.frequency_420_t)),
        #                                       'constant', constant_values=(0, 0))

        self.p1.plot(t, self.omega_420_t, pen=(0, 0, 255), name='omega_420', clear=True)
        self.p1.plot(t, self.omega_1013_t, pen=(255, 0, 0), name='omega_1013')
        # self.p1.plot(t, self.x0_t, pen=(0, 255, 0), linestyle='-.', linewidth=10.0, name='x0_t')
        # self.p1.plot(t, self.x1_t, pen=(0, 0, 0), linewidth=10.0, name='x1_t')

        if self.show_x_0.isChecked() and not self.show_x_1.isChecked():
            self.p1.plot(t, self.omega_420_t, pen=(0, 0, 255), name='omega_420', clear=True)
            self.p1.plot(t, self.omega_1013_t, pen=(255, 0, 0), name='omega_1013')
            self.p1.plot(t, self.x0_t, pen=(0, 255, 0), linestyle='-.', linewidth=10.0, name='x0_t')
        if not self.show_x_0.isChecked() and self.show_x_1.isChecked():
            self.p1.plot(t, self.omega_420_t, pen=(0, 0, 255), name='omega_420', clear=True)
            self.p1.plot(t, self.omega_1013_t, pen=(255, 0, 0), name='omega_1013')
            self.p1.plot(t, self.x1_t, pen=(0, 0, 0), linewidth=10.0, name='x1_t')
        if self.show_x_0.isChecked() and self.show_x_1.isChecked():
            self.p1.plot(t, self.omega_420_t, pen=(0, 0, 255), name='omega_420', clear=True)
            self.p1.plot(t, self.omega_1013_t, pen=(255, 0, 0), name='omega_1013')
            self.p1.plot(t, self.x0_t, pen=(0, 255, 0), linestyle='-.', linewidth=10.0, name='x0_t')
            self.p1.plot(t, self.x1_t, pen=(0, 0, 0), linewidth=10.0, name='x1_t')
        if not self.show_x_0.isChecked() and not self.show_x_1.isChecked():
            self.p1.plot(t, self.omega_420_t, pen=(0, 0, 255), name='omega_420', clear=True)
            self.p1.plot(t, self.omega_1013_t, pen=(255, 0, 0), name='omega_1013')

        if self.show_frequency_420.isChecked() and not self.show_frequency_1013.isChecked():
            self.p2.plot(t, self.frequency_420_t, pen='b', name='frequency_420', clear=True)
        if self.show_frequency_1013.isChecked() and not self.show_frequency_420.isChecked():
            self.p2.plot(t, self.frequency_1013_t, pen='r', name='frequency_1013', clear=True)
        if self.show_frequency_420.isChecked() and self.show_frequency_1013.isChecked():
            self.p2.plot(t, self.frequency_420_t, pen='b', name='frequency_420', clear=True)
            self.p2.plot(t, self.frequency_1013_t, pen='r', name='frequency_1013')
        if not self.show_frequency_420.isChecked() and not self.show_frequency_1013.isChecked():
            pass

    def load_config(self):
        # load config
        clock_frequency_value = int(self.settings.value("{}/CLOCK_FREQUENCY".format(self.config_name)))
        output_mode_value = int(self.settings.value("{}/OUTPUT_MODE".format(self.config_name)))
        trigger_value = int(self.settings.value("{}/TRIGGER".format(self.config_name)))

        checkBox_420_1_value = int(self.settings.value("{}/CHECKBOX_420_1".format(self.config_name)))
        checkBox_420_2_value = int(self.settings.value("{}/CHECKBOX_420_2".format(self.config_name)))
        checkBox_420_3_value = int(self.settings.value("{}/CHECKBOX_420_3".format(self.config_name)))
        checkBox_420_4_value = int(self.settings.value("{}/CHECKBOX_420_4".format(self.config_name)))
        checkBox_420_5_value = int(self.settings.value("{}/CHECKBOX_420_5".format(self.config_name)))
        checkBox_420_6_value = int(self.settings.value("{}/CHECKBOX_420_6".format(self.config_name)))
        checkBox_1013_1_value = int(self.settings.value("{}/CHECKBOX_1013_1".format(self.config_name)))
        checkBox_1013_2_value = int(self.settings.value("{}/CHECKBOX_1013_2".format(self.config_name)))
        checkBox_1013_3_value = int(self.settings.value("{}/CHECKBOX_1013_3".format(self.config_name)))
        checkBox_1013_4_value = int(self.settings.value("{}/CHECKBOX_1013_4".format(self.config_name)))
        checkBox_1013_5_value = int(self.settings.value("{}/CHECKBOX_1013_5".format(self.config_name)))
        checkBox_1013_6_value = int(self.settings.value("{}/CHECKBOX_1013_6".format(self.config_name)))
        show_x0_value = int(self.settings.value("{}/SHOW_X0".format(self.config_name)))
        show_x1_value = int(self.settings.value("{}/SHOW_X1".format(self.config_name)))
        show_frequency_1013_value = int(self.settings.value("{}/SHOW_FREQUENCY_1013".format(self.config_name)))
        show_frequency_420_value = int(self.settings.value("{}/SHOW_FREQUENCY_420".format(self.config_name)))

        waveform_420_1_value = int(self.settings.value("{}/WAVEFORM_420_1".format(self.config_name)))
        waveform_420_2_value = int(self.settings.value("{}/WAVEFORM_420_2".format(self.config_name)))
        waveform_420_3_value = int(self.settings.value("{}/WAVEFORM_420_3".format(self.config_name)))
        waveform_420_4_value = int(self.settings.value("{}/WAVEFORM_420_4".format(self.config_name)))
        waveform_420_5_value = int(self.settings.value("{}/WAVEFORM_420_5".format(self.config_name)))
        waveform_420_6_value = int(self.settings.value("{}/WAVEFORM_420_6".format(self.config_name)))

        waveform_1013_1_value = int(self.settings.value("{}/WAVEFORM_1013_1".format(self.config_name)))
        waveform_1013_2_value = int(self.settings.value("{}/WAVEFORM_1013_2".format(self.config_name)))
        waveform_1013_3_value = int(self.settings.value("{}/WAVEFORM_1013_3".format(self.config_name)))
        waveform_1013_4_value = int(self.settings.value("{}/WAVEFORM_1013_4".format(self.config_name)))
        waveform_1013_5_value = int(self.settings.value("{}/WAVEFORM_1013_5".format(self.config_name)))
        waveform_1013_6_value = int(self.settings.value("{}/WAVEFORM_1013_6".format(self.config_name)))

        start_amp_420_1 = self.settings.value("{}/START_AMP_420_1".format(self.config_name))
        end_amp_420_1 = self.settings.value("{}/END_AMP_420_1".format(self.config_name))
        num_amp_420_1 = self.settings.value("{}/NUM_AMP_420_1".format(self.config_name))
        start_frequency_420_1 = self.settings.value("{}/start_frequency_420_1".format(self.config_name))
        end_frequency_420_1 = self.settings.value("{}/end_frequency_420_1".format(self.config_name))
        time_420_1 = self.settings.value("{}/time_420_1".format(self.config_name))
        wait_time_420_1 = self.settings.value("{}/wait_time_420_1".format(self.config_name))

        start_amp_420_2 = self.settings.value("{}/START_AMP_420_2".format(self.config_name))
        end_amp_420_2 = self.settings.value("{}/END_AMP_420_2".format(self.config_name))
        num_amp_420_2 = self.settings.value("{}/NUM_AMP_420_2".format(self.config_name))
        start_frequency_420_2 = self.settings.value("{}/start_frequency_420_2".format(self.config_name))
        end_frequency_420_2 = self.settings.value("{}/end_frequency_420_2".format(self.config_name))
        time_420_2 = self.settings.value("{}/time_420_2".format(self.config_name))
        wait_time_420_2 = self.settings.value("{}/wait_time_420_2".format(self.config_name))

        start_amp_420_3 = self.settings.value("{}/START_AMP_420_3".format(self.config_name))
        end_amp_420_3 = self.settings.value("{}/END_AMP_420_3".format(self.config_name))
        num_amp_420_3 = self.settings.value("{}/NUM_AMP_420_3".format(self.config_name))
        start_frequency_420_3 = self.settings.value("{}/start_frequency_420_3".format(self.config_name))
        end_frequency_420_3 = self.settings.value("{}/end_frequency_420_3".format(self.config_name))
        time_420_3 = self.settings.value("{}/time_420_3".format(self.config_name))
        wait_time_420_3 = self.settings.value("{}/wait_time_420_3".format(self.config_name))

        start_amp_420_4 = self.settings.value("{}/START_AMP_420_4".format(self.config_name))
        end_amp_420_4 = self.settings.value("{}/END_AMP_420_4".format(self.config_name))
        num_amp_420_4 = self.settings.value("{}/NUM_AMP_420_4".format(self.config_name))
        start_frequency_420_4 = self.settings.value("{}/start_frequency_420_4".format(self.config_name))
        end_frequency_420_4 = self.settings.value("{}/end_frequency_420_4".format(self.config_name))
        time_420_4 = self.settings.value("{}/time_420_4".format(self.config_name))
        wait_time_420_4 = self.settings.value("{}/wait_time_420_4".format(self.config_name))

        start_amp_420_5 = self.settings.value("{}/START_AMP_420_5".format(self.config_name))
        end_amp_420_5 = self.settings.value("{}/END_AMP_420_5".format(self.config_name))
        num_amp_420_5 = self.settings.value("{}/NUM_AMP_420_5".format(self.config_name))
        start_frequency_420_5 = self.settings.value("{}/start_frequency_420_5".format(self.config_name))
        end_frequency_420_5 = self.settings.value("{}/end_frequency_420_5".format(self.config_name))
        time_420_5 = self.settings.value("{}/time_420_5".format(self.config_name))
        wait_time_420_5 = self.settings.value("{}/wait_time_420_5".format(self.config_name))

        start_amp_420_6 = self.settings.value("{}/START_AMP_420_6".format(self.config_name))
        end_amp_420_6 = self.settings.value("{}/END_AMP_420_6".format(self.config_name))
        num_amp_420_6 = self.settings.value("{}/NUM_AMP_420_6".format(self.config_name))
        start_frequency_420_6 = self.settings.value("{}/start_frequency_420_6".format(self.config_name))
        end_frequency_420_6 = self.settings.value("{}/end_frequency_420_6".format(self.config_name))
        time_420_6 = self.settings.value("{}/time_420_6".format(self.config_name))
        wait_time_420_6 = self.settings.value("{}/wait_time_420_6".format(self.config_name))

        start_amp_1013_1 = self.settings.value("{}/START_AMP_1013_1".format(self.config_name))
        end_amp_1013_1 = self.settings.value("{}/END_AMP_1013_1".format(self.config_name))
        num_amp_1013_1 = self.settings.value("{}/NUM_AMP_1013_1".format(self.config_name))
        start_frequency_1013_1 = self.settings.value("{}/start_frequency_1013_1".format(self.config_name))
        end_frequency_1013_1 = self.settings.value("{}/end_frequency_1013_1".format(self.config_name))
        time_1013_1 = self.settings.value("{}/time_1013_1".format(self.config_name))
        wait_time_1013_1 = self.settings.value("{}/wait_time_1013_1".format(self.config_name))

        start_amp_1013_2 = self.settings.value("{}/START_AMP_1013_2".format(self.config_name))
        end_amp_1013_2 = self.settings.value("{}/END_AMP_1013_2".format(self.config_name))
        num_amp_1013_2 = self.settings.value("{}/NUM_AMP_1013_2".format(self.config_name))
        start_frequency_1013_2 = self.settings.value("{}/start_frequency_1013_2".format(self.config_name))
        end_frequency_1013_2 = self.settings.value("{}/end_frequency_1013_2".format(self.config_name))
        time_1013_2 = self.settings.value("{}/time_1013_2".format(self.config_name))
        wait_time_1013_2 = self.settings.value("{}/wait_time_1013_2".format(self.config_name))

        start_amp_1013_3 = self.settings.value("{}/START_AMP_1013_3".format(self.config_name))
        end_amp_1013_3 = self.settings.value("{}/END_AMP_1013_3".format(self.config_name))
        num_amp_1013_3 = self.settings.value("{}/NUM_AMP_1013_3".format(self.config_name))
        start_frequency_1013_3 = self.settings.value("{}/start_frequency_1013_3".format(self.config_name))
        end_frequency_1013_3 = self.settings.value("{}/end_frequency_1013_3".format(self.config_name))
        time_1013_3 = self.settings.value("{}/time_1013_3".format(self.config_name))
        wait_time_1013_3 = self.settings.value("{}/wait_time_1013_3".format(self.config_name))

        start_amp_1013_4 = self.settings.value("{}/START_AMP_1013_4".format(self.config_name))
        end_amp_1013_4 = self.settings.value("{}/END_AMP_1013_4".format(self.config_name))
        num_amp_1013_4 = self.settings.value("{}/NUM_AMP_1013_4".format(self.config_name))
        start_frequency_1013_4 = self.settings.value("{}/start_frequency_1013_4".format(self.config_name))
        end_frequency_1013_4 = self.settings.value("{}/end_frequency_1013_4".format(self.config_name))
        time_1013_4 = self.settings.value("{}/time_1013_4".format(self.config_name))
        wait_time_1013_4 = self.settings.value("{}/wait_time_1013_4".format(self.config_name))

        start_amp_1013_5 = self.settings.value("{}/START_AMP_1013_5".format(self.config_name))
        end_amp_1013_5 = self.settings.value("{}/END_AMP_1013_5".format(self.config_name))
        num_amp_1013_5 = self.settings.value("{}/NUM_AMP_1013_5".format(self.config_name))
        start_frequency_1013_5 = self.settings.value("{}/start_frequency_1013_5".format(self.config_name))
        end_frequency_1013_5 = self.settings.value("{}/end_frequency_1013_5".format(self.config_name))
        time_1013_5 = self.settings.value("{}/time_1013_5".format(self.config_name))
        wait_time_1013_5 = self.settings.value("{}/wait_time_1013_5".format(self.config_name))

        start_amp_1013_6 = self.settings.value("{}/START_AMP_1013_6".format(self.config_name))
        end_amp_1013_6 = self.settings.value("{}/END_AMP_1013_6".format(self.config_name))
        num_amp_1013_6 = self.settings.value("{}/NUM_AMP_1013_6".format(self.config_name))
        start_frequency_1013_6 = self.settings.value("{}/start_frequency_1013_6".format(self.config_name))
        end_frequency_1013_6 = self.settings.value("{}/end_frequency_1013_6".format(self.config_name))
        time_1013_6 = self.settings.value("{}/time_1013_6".format(self.config_name))
        wait_time_1013_6 = self.settings.value("{}/wait_time_1013_6".format(self.config_name))

        x0_high0 = self.settings.value("{}/X0_HIGH0".format(self.config_name))
        x0_low0 = self.settings.value("{}/X0_LOW0".format(self.config_name))
        x0_high1 = self.settings.value("{}/X0_HIGH1".format(self.config_name))
        x0_low1 = self.settings.value("{}/X0_LOW1".format(self.config_name))
        x0_high2 = self.settings.value("{}/X0_HIGH2".format(self.config_name))
        x0_low2 = self.settings.value("{}/X0_LOW2".format(self.config_name))
        x0_high3 = self.settings.value("{}/X0_HIGH3".format(self.config_name))
        x0_low3 = self.settings.value("{}/X0_LOW3".format(self.config_name))
        x0_high4 = self.settings.value("{}/X0_HIGH4".format(self.config_name))
        x0_low4 = self.settings.value("{}/X0_LOW4".format(self.config_name))

        x1_high0 = self.settings.value("{}/X1_HIGH0".format(self.config_name))
        x1_low0 = self.settings.value("{}/X1_LOW0".format(self.config_name))
        x1_high1 = self.settings.value("{}/X1_HIGH1".format(self.config_name))
        x1_low1 = self.settings.value("{}/X1_LOW1".format(self.config_name))
        x1_high2 = self.settings.value("{}/X1_HIGH2".format(self.config_name))
        x1_low2 = self.settings.value("{}/X1_LOW2".format(self.config_name))
        x1_high3 = self.settings.value("{}/X1_HIGH3".format(self.config_name))
        x1_low3 = self.settings.value("{}/X1_LOW3".format(self.config_name))
        x1_high4 = self.settings.value("{}/X1_HIGH4".format(self.config_name))
        x1_low4 = self.settings.value("{}/X1_LOW4".format(self.config_name))

        # init gui
        self.clock_frequency.setCurrentIndex(clock_frequency_value)
        self.output_mode.setCurrentIndex(output_mode_value)
        self.trigger.setCurrentIndex(trigger_value)

        self.checkBox_420_1.setChecked(checkBox_420_1_value)
        self.checkBox_420_2.setChecked(checkBox_420_2_value)
        self.checkBox_420_3.setChecked(checkBox_420_3_value)
        self.checkBox_420_4.setChecked(checkBox_420_4_value)
        self.checkBox_420_5.setChecked(checkBox_420_5_value)
        self.checkBox_420_6.setChecked(checkBox_420_6_value)
        self.checkBox_1013_1.setChecked(checkBox_1013_1_value)
        self.checkBox_1013_2.setChecked(checkBox_1013_2_value)
        self.checkBox_1013_3.setChecked(checkBox_1013_3_value)
        self.checkBox_1013_4.setChecked(checkBox_1013_4_value)
        self.checkBox_1013_5.setChecked(checkBox_1013_5_value)
        self.checkBox_1013_6.setChecked(checkBox_1013_6_value)

        self.show_frequency_420.setChecked(show_frequency_420_value)
        self.show_frequency_1013.setChecked(show_frequency_1013_value)

        self.show_x_0.setChecked(show_x0_value)
        self.show_x_1.setChecked(show_x1_value)

        self.waveform_420_1.setCurrentIndex(waveform_420_1_value)
        self.waveform_420_2.setCurrentIndex(waveform_420_2_value)
        self.waveform_420_3.setCurrentIndex(waveform_420_3_value)
        self.waveform_420_4.setCurrentIndex(waveform_420_4_value)
        self.waveform_420_5.setCurrentIndex(waveform_420_5_value)
        self.waveform_420_6.setCurrentIndex(waveform_420_6_value)
        self.waveform_1013_1.setCurrentIndex(waveform_1013_1_value)
        self.waveform_1013_2.setCurrentIndex(waveform_1013_2_value)
        self.waveform_1013_3.setCurrentIndex(waveform_1013_3_value)
        self.waveform_1013_4.setCurrentIndex(waveform_1013_4_value)
        self.waveform_1013_5.setCurrentIndex(waveform_1013_5_value)
        self.waveform_1013_6.setCurrentIndex(waveform_1013_6_value)

        self.start_amp_420_1.setText(start_amp_420_1)
        self.end_amp_420_1.setText(end_amp_420_1)
        self.num_amp_420_1.setText(num_amp_420_1)
        self.start_frequency_420_1.setText(start_frequency_420_1)
        self.end_frequency_420_1.setText(end_frequency_420_1)
        self.time_420_1.setText(time_420_1)
        self.wait_time_420_1.setText(wait_time_420_1)

        self.start_amp_420_2.setText(start_amp_420_2)
        self.end_amp_420_2.setText(end_amp_420_2)
        self.num_amp_420_2.setText(num_amp_420_2)
        self.start_frequency_420_2.setText(start_frequency_420_2)
        self.end_frequency_420_2.setText(end_frequency_420_2)
        self.time_420_2.setText(time_420_2)
        self.wait_time_420_2.setText(wait_time_420_2)

        self.start_amp_420_3.setText(start_amp_420_3)
        self.end_amp_420_3.setText(end_amp_420_3)
        self.num_amp_420_3.setText(num_amp_420_3)
        self.start_frequency_420_3.setText(start_frequency_420_3)
        self.end_frequency_420_3.setText(end_frequency_420_3)
        self.time_420_3.setText(time_420_3)
        self.wait_time_420_3.setText(wait_time_420_3)

        self.start_amp_420_4.setText(start_amp_420_4)
        self.end_amp_420_4.setText(end_amp_420_4)
        self.num_amp_420_4.setText(num_amp_420_4)
        self.start_frequency_420_4.setText(start_frequency_420_4)
        self.end_frequency_420_4.setText(end_frequency_420_4)
        self.time_420_4.setText(time_420_4)
        self.wait_time_420_4.setText(wait_time_420_4)

        self.start_amp_420_5.setText(start_amp_420_5)
        self.end_amp_420_5.setText(end_amp_420_5)
        self.num_amp_420_5.setText(num_amp_420_5)
        self.start_frequency_420_5.setText(start_frequency_420_5)
        self.end_frequency_420_5.setText(end_frequency_420_5)
        self.time_420_5.setText(time_420_5)
        self.wait_time_420_5.setText(wait_time_420_5)

        self.start_amp_420_6.setText(start_amp_420_6)
        self.end_amp_420_6.setText(end_amp_420_6)
        self.num_amp_420_6.setText(num_amp_420_6)
        self.start_frequency_420_6.setText(start_frequency_420_6)
        self.end_frequency_420_6.setText(end_frequency_420_6)
        self.time_420_6.setText(time_420_6)
        self.wait_time_420_6.setText(wait_time_420_6)

        self.start_amp_1013_1.setText(start_amp_1013_1)
        self.end_amp_1013_1.setText(end_amp_1013_1)
        self.num_amp_1013_1.setText(num_amp_1013_1)
        self.start_frequency_1013_1.setText(start_frequency_1013_1)
        self.end_frequency_1013_1.setText(end_frequency_1013_1)
        self.time_1013_1.setText(time_1013_1)
        self.wait_time_1013_1.setText(wait_time_1013_1)

        self.start_amp_1013_2.setText(start_amp_1013_2)
        self.end_amp_1013_2.setText(end_amp_1013_2)
        self.num_amp_1013_2.setText(num_amp_1013_2)
        self.start_frequency_1013_2.setText(start_frequency_1013_2)
        self.end_frequency_1013_2.setText(end_frequency_1013_2)
        self.time_1013_2.setText(time_1013_2)
        self.wait_time_1013_2.setText(wait_time_1013_2)

        self.start_amp_1013_3.setText(start_amp_1013_3)
        self.end_amp_1013_3.setText(end_amp_1013_3)
        self.num_amp_1013_3.setText(num_amp_1013_3)
        self.start_frequency_1013_3.setText(start_frequency_1013_3)
        self.end_frequency_1013_3.setText(end_frequency_1013_3)
        self.time_1013_3.setText(time_1013_3)
        self.wait_time_1013_3.setText(wait_time_1013_3)

        self.start_amp_1013_4.setText(start_amp_1013_4)
        self.end_amp_1013_4.setText(end_amp_1013_4)
        self.num_amp_1013_4.setText(num_amp_1013_4)
        self.start_frequency_1013_4.setText(start_frequency_1013_4)
        self.end_frequency_1013_4.setText(end_frequency_1013_4)
        self.time_1013_4.setText(time_1013_4)
        self.wait_time_1013_4.setText(wait_time_1013_4)

        self.start_amp_1013_5.setText(start_amp_1013_5)
        self.end_amp_1013_5.setText(end_amp_1013_5)
        self.num_amp_1013_5.setText(num_amp_1013_5)
        self.start_frequency_1013_5.setText(start_frequency_1013_5)
        self.end_frequency_1013_5.setText(end_frequency_1013_5)
        self.time_1013_5.setText(time_1013_5)
        self.wait_time_1013_5.setText(wait_time_1013_5)

        self.start_amp_1013_6.setText(start_amp_1013_6)
        self.end_amp_1013_6.setText(end_amp_1013_6)
        self.num_amp_1013_6.setText(num_amp_1013_6)
        self.start_frequency_1013_6.setText(start_frequency_1013_6)
        self.end_frequency_1013_6.setText(end_frequency_1013_6)
        self.time_1013_6.setText(time_1013_6)
        self.wait_time_1013_6.setText(wait_time_1013_6)

        self.x0_high0.setText(x0_high0)
        self.x0_low0.setText(x0_low0)
        self.x0_high1.setText(x0_high1)
        self.x0_low1.setText(x0_low1)
        self.x0_high2.setText(x0_high2)
        self.x0_low2.setText(x0_low2)
        self.x0_high3.setText(x0_high3)
        self.x0_low3.setText(x0_low3)
        self.x0_high4.setText(x0_high4)
        self.x0_low4.setText(x0_low4)

        self.x1_high0.setText(x1_high0)
        self.x1_low0.setText(x1_low0)
        self.x1_high1.setText(x1_high1)
        self.x1_low1.setText(x1_low1)
        self.x1_high2.setText(x1_high2)
        self.x1_low2.setText(x1_low2)
        self.x1_high3.setText(x1_high3)
        self.x1_low3.setText(x1_low3)
        self.x1_high4.setText(x1_high4)
        self.x1_low4.setText(x1_low4)

    # def write_config(self, filename):
    #     file = open(filename)
    #     for line in file.readlines():
    #         line_split = line.split(' ')
    #         object_name = line_split[0]
    #         object_value = line_split[1]
    #         object_type = line_split[2]
    #         target_object = self.findChildren(QWidget, object_name)[0]
    #         if object_type == "QComboBox":
    #             target_object.setCurrentIndex(int(object_value))
    #         elif object_type == "QCheckBox":
    #             if object_value == "True":
    #                 target_object.setChecked(True)
    #             else:
    #                 target_object.setChecked(False)
    #         elif object_type == "QLineEdit":
    #             target_object.setText(object_value)
    #     file.close()


class STIRAP(Sequence):
    def __init__(self, awg_controller):
        super(STIRAP, self).__init__(awg_control=awg_controller)
        self.setObjectName("STIRAP")
        # self.setupUi(self)
        # self.done.clicked.connect(partial(awg_controller.sequence, self))
        # self.amp_420_1.setText(self.amp_420_1_value)


class LandauZener(Sequence):
    def __init__(self, awg_controller):
        super(LandauZener, self).__init__(awg_control=awg_controller)
        self.setupUi(self)
        self.done.clicked.connect(partial(awg_controller.sequence, self))


class RabiFlopping(Sequence):
    def __init__(self, awg_controller):
        super(RabiFlopping, self).__init__(awg_control=awg_controller)
        self.setupUi(self)
        self.done.clicked.connect(partial(awg_controller.sequence, self))
