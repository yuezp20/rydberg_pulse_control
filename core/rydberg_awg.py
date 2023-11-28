import numpy as np
from pyspcm import *
import time
from spcm_tools import *
import sys
from time import sleep
from PyQt5.QtWidgets import QCheckBox, QLineEdit, QComboBox
from collections import namedtuple
from decimal import ROUND_HALF_UP, Decimal
import matplotlib.pyplot as plt

waveform = namedtuple("waveform", ["form", "start_amp", "end_amp", "num_amp", "start_frequency",
                                   "end_frequency", "time", "wait_time", "file_name"])

Dwaveform = namedtuple("Dwaveform",
                       ["time", "HIGH0", "LOW0", "HIGH1", "LOW1", "HIGH2", "LOW2", "HIGH3", "LOW3", "HIGH4", "LOW4"])


def to_decimal(num, exp="0.000", rounding=ROUND_HALF_UP) -> Decimal:
    """
       转 Decimal，四舍五入
       :param num: int,str,float,Decimal
       :param exp: 精度
       :param rounding: 圆整模式
       :return: Decimal
       """
    if not num:
        return Decimal("0").quantize(exp=Decimal(exp), rounding=rounding)
    if not isinstance(num, str):
        num = str(num)
    return Decimal(num).quantize(exp=Decimal(exp), rounding=rounding)


class RydbergPulseAwg(object):
    def __init__(self):
        # open card
        # uncomment the second line and replace the IP address to use remote
        # cards like in a generatorNETBOX
        h_card = spcm_hOpen(create_string_buffer(b'/dev/spcm1'))  # spcm0 for AOD AWG # spcm1 for Rydberg AWG
        if h_card is None:
            print("no card found...")
            exit(1)

        # read type, function and sn and check for D/A card
        lCardType = int32(0)
        spcm_dwGetParam_i32(h_card, SPC_PCITYP, byref(lCardType))
        lSerialNumber = int32(0)
        spcm_dwGetParam_i32(h_card, SPC_PCISERIALNO, byref(lSerialNumber))
        lFncType = int32(0)
        spcm_dwGetParam_i32(h_card, SPC_FNCTYPE, byref(lFncType))

        sCardName = szTypeToName(lCardType.value)

        print("Found: {0} sn {1:05d}".format(sCardName, lSerialNumber.value))

        self.card = h_card

    def setup_card(self, ui_sequence):

        # -------------------------------------------------------------------------------#
        # setup channels we need to use
        # -------------------------------------------------------------------------------#
        qwChEnable = int64(CHANNEL0 | CHANNEL1)
        spcm_dwSetParam_i64(self.card, SPC_CHENABLE, qwChEnable)
        lSetChannels = int32(0)
        spcm_dwGetParam_i32(self.card, SPC_CHCOUNT, byref(lSetChannels))
        lBytesPerSample = int32(0)
        spcm_dwGetParam_i32(self.card, SPC_MIINST_BYTESPERSAMPLE, byref(lBytesPerSample))
        for lChannel in range(0, lSetChannels.value, 1):
            spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT0 + lChannel * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1)
            spcm_dwSetParam_i32(self.card, SPC_AMP0 + lChannel * (SPC_AMP1 - SPC_AMP0), 1000)
            spcm_dwSetParam_i32(self.card, SPC_CH0_STOPLEVEL + lChannel * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL),
                                SPCM_STOPLVL_HOLDLAST)

        # -------------------------------------------------------------------------------#
        # setup clock frequency and no clock output
        # -------------------------------------------------------------------------------#
        spcm_dwSetParam_i64(self.card, SPC_SAMPLERATE, MEGA(1000))
        samplerate = MEGA(1000)
        print("samplerate has been set to 1000MHz!")
        spcm_dwSetParam_i32(self.card, SPC_CLOCKOUT, 0)

        # -------------------------------------------------------------------------------#
        # setup output mode
        # -------------------------------------------------------------------------------#
        if ui_sequence.output_mode.currentIndex() == 0:
            llLoops = int64(0)
            spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_SINGLERESTART)
            spcm_dwSetParam_i64(self.card, SPC_LOOPS, llLoops)
        elif ui_sequence.output_mode.currentIndex() == 1:
            llLoops = int64(1)
            spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_SINGLE)
            spcm_dwSetParam_i64(self.card, SPC_LOOPS, llLoops)
        elif ui_sequence.output_mode.currentIndex() == 2:
            llLoops = int64(0)
            spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_CONTINUOUS)
            spcm_dwSetParam_i64(self.card, SPC_LOOPS, llLoops)

        # -------------------------------------------------------------------------------#
        # setup trigger
        # -------------------------------------------------------------------------------#
        if ui_sequence.trigger.currentIndex() == 0:
            spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT0_LEVEL0, 2500)
            spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT0_MODE, SPC_TM_POS)
            spcm_dwSetParam_i32(self.card, SPC_TRIG_ORMASK, SPC_TMASK_EXT0)
        else:
            spcm_dwSetParam_i32(self.card, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
        spcm_dwSetParam_i32(self.card, SPC_TRIG_ANDMASK, 0)
        spcm_dwSetParam_i32(self.card, SPC_TRIG_CH_ORMASK0, 0)
        spcm_dwSetParam_i32(self.card, SPC_TRIG_CH_ORMASK1, 0)
        spcm_dwSetParam_i32(self.card, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32(self.card, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32(self.card, SPC_TRIGGEROUT, 0)

        # -------------------------------------------------------------------------------#
        # setup digital output
        # -------------------------------------------------------------------------------#
        dwXMode_X0 = SPCM_XMODE_DIGOUT | SPCM_XMODE_DIGOUTSRC_CH0 | SPCM_XMODE_DIGOUTSRC_BIT15
        spcm_dwSetParam_i32(self.card, SPCM_X0_MODE, dwXMode_X0)
        dwXMode_X2 = SPCM_XMODE_DIGOUT | SPCM_XMODE_DIGOUTSRC_CH1 | SPCM_XMODE_DIGOUTSRC_BIT15
        spcm_dwSetParam_i32(self.card, SPCM_X2_MODE, dwXMode_X2)

        return samplerate, lSetChannels, lBytesPerSample

    def sequence(self, ui_sequence):
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        # -------------------------------------------------------------------------------#
        # setup clock frequency and no clock output
        # -------------------------------------------------------------------------------#
        samplerate, lSetChannels, lBytesPerSample = RydbergPulseAwg.setup_card(self, ui_sequence)

        # -------------------------------------------------------------------------------#
        # find out the waveform we have selected
        # -------------------------------------------------------------------------------#
        waveform_cont_420, waveform_cont_1013 = find_waveform(ui_sequence)
        x0, x1 = find_Dwaveform(ui_sequence)
        # print("waveform_cont_420:", waveform_cont_420)
        # print("waveform_cont_1013:", waveform_cont_1013)

        # -------------------------------------------------------------------------------#
        # calculate the Analog waveform and restore them in np.array
        # -------------------------------------------------------------------------------#
        wave_point, wave_point_420, wave_point_1013 = wave_calculate(samplerate, waveform_cont_420, waveform_cont_1013,
                                                                     x0, x1)

        # -------------------------------------------------------------------------------#
        # calculate mem_size we need to use
        # -------------------------------------------------------------------------------#
        llMemSamples = mem_size_length(len(wave_point_420))
        spcm_dwSetParam_i64(self.card, SPC_MEMSIZE, llMemSamples)

        # -------------------------------------------------------------------------------#
        # setup software buffer
        # -------------------------------------------------------------------------------#
        qwBufferSize = uint64(llMemSamples * lBytesPerSample.value * lSetChannels.value)
        pvBuffer = pvAllocMemPageAligned(qwBufferSize.value)

        # calculate the data
        pnBuffer = cast(pvBuffer, ptr16)
        # for i in range(0, llMemSamples.value, 1):
        #     pnBuffer[i] = i
        for i in range(0, len(wave_point), 1):
            pnBuffer[i] = wave_point[i]
        # we define the buffer for transfer and start the DMA transfer
        sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
        spcm_dwDefTransfer_i64(self.card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), pvBuffer, uint64(0),
                               qwBufferSize)
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        sys.stdout.write("... data has been transferred to board memory\n")
        # We'll start and wait until the card has finished or until a timeout occurs
        # spcm_dwSetParam_i32(self.card, SPC_TIMEOUT, 1000000)
        sys.stdout.write(
            "\nStarting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)\n")

        if ui_sequence.output_mode.currentIndex == 0:
            dwError = spcm_dwSetParam_i32(self.card, SPC_M2CMD,
                                          M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
        else:
            dwError = spcm_dwSetParam_i32(self.card, SPC_M2CMD,
                                          M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        # if dwError == ERR_TIMEOUT:
        #     spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)

        write_config(ui_sequence)

    def open_card(self):
        h_card = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
        if h_card is None:
            print("no card found...")
            exit(1)
        print("card has been opened!")

        self.card = h_card

    def close_card(self):
        spcm_vClose(self.card)
        print("rydberg awg has been closed!")

    def stop_card(self, ui_sequence):
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        # ui_sequence.status_info.setText("The card has been stopped ...")
        print("card has been stopped!")


def find_waveform(ui_sequence):
    checkbox_list = ui_sequence.findChildren(QCheckBox)
    waveform_cont_420 = []
    waveform_cont_1013 = []
    for i in range(len(checkbox_list)):
        if checkbox_list[i].isChecked():

            check_box_name = checkbox_list[i].text()
            check_box_name_split = check_box_name.split('_')

            if check_box_name_split[0] != "frequency" and check_box_name_split[0] != "x":
                form_i = ui_sequence.findChildren(QComboBox, "waveform_{}".format(check_box_name))[0].currentText()
                start_amp_i = float(
                    ui_sequence.findChildren(QLineEdit, "start_amp_{}".format(check_box_name))[0].text())
                end_amp_i = float(ui_sequence.findChildren(QLineEdit, "end_amp_{}".format(check_box_name))[0].text())
                num_amp_i = int(ui_sequence.findChildren(QLineEdit, "num_amp_{}".format(check_box_name))[0].text())
                start_frequency_i = float(
                    ui_sequence.findChildren(QLineEdit, "start_frequency_{}".format(check_box_name))[0].text())
                end_frequency_i = float(
                    ui_sequence.findChildren(QLineEdit, "end_frequency_{}".format(check_box_name))[0].text())
                time_i = float(ui_sequence.findChildren(QLineEdit, "time_{}".format(check_box_name))[0].text())
                wait_time_i = float(
                    ui_sequence.findChildren(QLineEdit, "wait_time_{}".format(check_box_name))[0].text())
                if check_box_name_split[0] == "420":
                    waveform_cont_420.append(
                        waveform(form=form_i, start_amp=start_amp_i, end_amp=end_amp_i, num_amp=num_amp_i,
                                 start_frequency=start_frequency_i,
                                 end_frequency=end_frequency_i, time=time_i,
                                 wait_time=wait_time_i))
                else:
                    waveform_cont_1013.append(
                        waveform(form=form_i, start_amp=start_amp_i, end_amp=end_amp_i, num_amp=num_amp_i,
                                 start_frequency=start_frequency_i,
                                 end_frequency=end_frequency_i, time=time_i,
                                 wait_time=wait_time_i))

    return waveform_cont_420, waveform_cont_1013


def find_Dwaveform(ui_sequence):
    x0 = Dwaveform(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    x1 = Dwaveform(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # x2 = Dwaveform(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    x0 = x0._replace(HIGH0=float(ui_sequence.x0_high0.text()))
    x0 = x0._replace(LOW0=float(ui_sequence.x0_low0.text()))
    x0 = x0._replace(HIGH1=float(ui_sequence.x0_high1.text()))
    x0 = x0._replace(LOW1=float(ui_sequence.x0_low1.text()))
    x0 = x0._replace(HIGH2=float(ui_sequence.x0_high2.text()))
    x0 = x0._replace(LOW2=float(ui_sequence.x0_low2.text()))
    x0 = x0._replace(HIGH3=float(ui_sequence.x0_high3.text()))
    x0 = x0._replace(LOW3=float(ui_sequence.x0_low3.text()))
    x0 = x0._replace(HIGH4=float(ui_sequence.x0_high4.text()))
    x0 = x0._replace(LOW4=float(ui_sequence.x0_low4.text()))

    x1 = x1._replace(HIGH0=float(ui_sequence.x1_high0.text()))
    x1 = x1._replace(LOW0=float(ui_sequence.x1_low0.text()))
    x1 = x1._replace(HIGH1=float(ui_sequence.x1_high1.text()))
    x1 = x1._replace(LOW1=float(ui_sequence.x1_low1.text()))
    x1 = x1._replace(HIGH2=float(ui_sequence.x1_high2.text()))
    x1 = x1._replace(LOW2=float(ui_sequence.x1_low2.text()))
    x1 = x1._replace(HIGH3=float(ui_sequence.x1_high3.text()))
    x1 = x1._replace(LOW3=float(ui_sequence.x1_low3.text()))
    x1 = x1._replace(HIGH4=float(ui_sequence.x1_high4.text()))
    x1 = x1._replace(LOW4=float(ui_sequence.x1_low4.text()))

    # x2 = x2._replace(HIGH0=float(ui_sequence.X2_HIGH0.text()))
    # x2 = x2._replace(LOW0=float(ui_sequence.X2_LOW0.text()))
    # x2 = x2._replace(HIGH1=float(ui_sequence.X2_HIGH1.text()))
    # x2 = x2._replace(LOW1=float(ui_sequence.X2_LOW1.text()))
    # x2 = x2._replace(HIGH2=float(ui_sequence.X2_HIGH2.text()))
    # x2 = x2._replace(LOW2=float(ui_sequence.X2_LOW2.text()))
    # x2 = x2._replace(HIGH3=float(ui_sequence.X2_HIGH3.text()))
    # x2 = x2._replace(LOW3=float(ui_sequence.X2_LOW3.text()))
    # x2 = x2._replace(HIGH4=float(ui_sequence.X2_HIGH4.text()))
    # x2 = x2._replace(LOW4=float(ui_sequence.X2_LOW4.text()))

    x0 = x0._replace(
        time=to_decimal(
            x0.HIGH0 + x0.HIGH1 + x0.HIGH2 + x0.HIGH3 + x0.HIGH4 + x0.LOW0 + x0.LOW1 + x0.LOW2 + x0.LOW3 + x0.LOW4))
    x1 = x1._replace(
        time=to_decimal(
            x1.HIGH0 + x1.HIGH1 + x1.HIGH2 + x1.HIGH3 + x1.HIGH4 + x1.LOW0 + x1.LOW1 + x1.LOW2 + x1.LOW3 + x1.LOW4))
    # x2 = x2._replace(
    #     time=x2.HIGH0 + x2.HIGH1 + x2.HIGH2 + x2.HIGH3 + x2.HIGH4 + x2.LOW0 + x2.LOW1 + x2.LOW2 + x2.LOW3 + x2.LOW4)

    return x0, x1


def wave_calculate(samplerate, waveform_cont_420, waveform_cont_1013, x0, x1):
    amp_std = 10000  # for 0dbm

    # first we compare time consume of 420 and 1013
    wave_time_420 = 0
    for single_wave_420 in waveform_cont_420:
        wave_time_420 += single_wave_420.time
        wave_time_420 += single_wave_420.wait_time
    wave_time_1013 = 0
    for single_wave_1013 in waveform_cont_1013:
        wave_time_1013 += single_wave_1013.time
        wave_time_1013 += single_wave_1013.wait_time

    wave_time = max(wave_time_420, wave_time_1013, x0.time, x1.time)
    wave_point_num = int(round(samplerate * wave_time * 10 ** -6))

    # wave_point_num = int(round(samplerate * wave_time_420 * 10 ** -6))
    # if wave_time_420 < wave_time_1013:
    #     wave_point_num = int(round(samplerate * wave_time_1013 * 10 ** -6))
    wave_point_420 = []
    wave_point_1013 = []

    # then we need to calculate each wave point according to wave_form_cont
    phi = 0
    for single_wave in waveform_cont_420:
        amp_ratio = single_wave.start_amp * 0.01
        start_frequency = MEGA(single_wave.start_frequency)
        end_frequency = MEGA(single_wave.end_frequency)
        time = single_wave.time * 10 ** -6
        signal_point = int(samplerate * single_wave.time * 10 ** -6)
        wait_point = int(samplerate * single_wave.wait_time * 10 ** -6)
        single_wave_point = np.zeros(int(round(samplerate * (single_wave.time + single_wave.wait_time) * 10 ** -6)),
                                     dtype='int16')
        if single_wave.form == "flat pulse":
            # -----------------------------------------------------------------------#
            # here we calculate flat pulse
            # -----------------------------------------------------------------------#
            for i in range(signal_point):
                phi += 2 * np.pi * start_frequency / samplerate
                single_wave_point[i] = int(amp_std * amp_ratio * np.sin(phi))
                # single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * start_frequency / samplerate * i))

            for i in range(wait_point):
                phi += 2 * np.pi * start_frequency / samplerate

        elif single_wave.form == "blackman pulse":
            # -----------------------------------------------------------------------#
            # here we calculate black_man pulse
            # -----------------------------------------------------------------------#
            for i in range(signal_point):
                amp = -0.5 * np.cos(2 * np.pi * i / samplerate / time) + 0.08 * np.cos(
                    4 * np.pi * i / samplerate / time) + 0.42
                phi += 2 * np.pi * start_frequency / samplerate
                single_wave_point[i] = int(amp_std * amp_ratio * amp * np.sin(phi))

            for i in range(wait_point):
                phi += 2 * np.pi * start_frequency / samplerate

                # amp = 2.381 * amp
                # single_wave_point[i] = int(
                #     amp_std * amp_ratio * amp * np.sin(2 * np.pi * start_frequency / samplerate * i))

        elif single_wave.form == "ramp frequency":
            # -----------------------------------------------------------------------#
            # here we calculate ramp frequency pulse
            # -----------------------------------------------------------------------#
            for i in range(signal_point):
                amp = amp_std * amp_ratio
                phi += 2 * np.pi * (
                        start_frequency + (end_frequency - start_frequency) / time * i / samplerate) / samplerate
                single_wave_point[i] = int(amp * np.sin(phi))

                # single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * (
                #         start_frequency * i / samplerate + 0.5 * (
                #         end_frequency - start_frequency) / time * i / samplerate * i / samplerate)))

            for i in range(wait_point):
                phi += 2 * np.pi * end_frequency / samplerate

        elif single_wave.form == "multi component":
            # -----------------------------------------------------------------------#
            # here we calculate multi component pulse
            # -----------------------------------------------------------------------#
            amp_std_split = amp_std / single_wave.num_amp
            amp_list = 0.01 * np.linspace(single_wave.start_amp, single_wave.end_amp, single_wave.num_amp)
            fre_list = np.linspace(single_wave.start_frequency, single_wave.end_frequency,
                                   single_wave.num_amp) * 10 ** 6

            for i in range(signal_point):
                for idx in range(single_wave.num_amp):
                    # single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * start_frequency / samplerate * i))
                    single_wave_point[i] = single_wave_point[i] + int(
                        amp_std_split * amp_list[idx] * np.sin(2 * np.pi * fre_list[idx] / samplerate * i))

            for i in range(wait_point):
                phi += 2 * np.pi * end_frequency / samplerate

        wave_point_420 = np.append(wave_point_420, single_wave_point)
    wave_point_420 = np.asarray(wave_point_420, dtype='int16')

    phi = 0
    for single_wave in waveform_cont_1013:

        amp_ratio = single_wave.start_amp * 0.01
        start_frequency = MEGA(single_wave.start_frequency)
        end_frequency = MEGA(single_wave.end_frequency)
        time = single_wave.time * 10 ** -6
        signal_point = int(samplerate * single_wave.time * 10 ** -6)
        wait_point = int(samplerate * single_wave.wait_time * 10 ** -6)
        single_wave_point = np.zeros(int(samplerate * (single_wave.time + single_wave.wait_time) * 10 ** -6),
                                     dtype='int16')
        if single_wave.form == "flat pulse":
            # -----------------------------------------------------------------------#
            # here we calculate flat pulse
            # -----------------------------------------------------------------------#
            for i in range(signal_point):
                phi += 2 * np.pi * start_frequency / samplerate
                single_wave_point[i] = int(amp_std * amp_ratio * np.sin(phi))
                # single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * start_frequency / samplerate * i))

            for i in range(wait_point):
                phi += 2 * np.pi * start_frequency / samplerate
        elif single_wave.form == "blackman pulse":
            # -----------------------------------------------------------------------#
            # here we calculate black_man pulse
            # -----------------------------------------------------------------------#
            for i in range(signal_point):
                amp = -0.5 * np.cos(2 * np.pi * i / samplerate / time) + 0.08 * np.cos(
                    4 * np.pi * i / samplerate / time) + 0.42
                phi += 2 * np.pi * start_frequency / samplerate
                single_wave_point[i] = int(amp_std * amp_ratio * amp * np.sin(phi))

            for i in range(wait_point):
                phi += 2 * np.pi * start_frequency / samplerate
        elif single_wave.form == "ramp frequency":
            # -----------------------------------------------------------------------#
            # here we calculate ramp frequency pulse
            # -----------------------------------------------------------------------#
            for i in range(signal_point):
                amp = amp_std * amp_ratio
                phi += 2 * np.pi * (
                        start_frequency + (end_frequency - start_frequency) / time * i / samplerate) / samplerate
                single_wave_point[i] = int(amp * np.sin(phi))

            for i in range(wait_point):
                phi += 2 * np.pi * end_frequency / samplerate

        elif single_wave.form == "multi component":
            # -----------------------------------------------------------------------#
            # here we calculate multi component pulse
            # -----------------------------------------------------------------------#
            amp_std_split = amp_std / single_wave.num_amp
            amp_list = 0.01 * np.linspace(single_wave.start_amp, single_wave.end_amp, single_wave.num_amp)
            fre_list = np.linspace(single_wave.start_frequency, single_wave.end_frequency, single_wave.num_amp)

            for i in range(signal_point):
                for idx in range(single_wave.num_amp):
                    # single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * start_frequency / samplerate * i))
                    single_wave_point[i] = single_wave_point[i] + int(
                        amp_std_split * amp_list[idx] * np.sin(2 * np.pi * fre_list[idx] / samplerate * i))

            for i in range(wait_point):
                phi += 2 * np.pi * end_frequency / samplerate

        wave_point_1013 = np.append(wave_point_1013, single_wave_point)
    wave_point_1013 = np.asarray(wave_point_1013, dtype='int16')

    wave_point_420 = np.pad(wave_point_420, [(0, wave_point_num - len(wave_point_420))])
    wave_point_1013 = np.pad(wave_point_1013, [(0, wave_point_num - len(wave_point_1013))])
    idx = tuple(range(1, wave_point_num + 1))

    # start add digital signal to analog channels

    # -----------------------------------------------------------------------#
    # first we calculate new waveform for 420
    # -----------------------------------------------------------------------#
    point_formulated = 0
    for i in range(5):
        single_digital_wave_point_num = int(round(samplerate * (x0[2 * i + 1] + x0[2 * i + 2]) * 10 ** -6))
        for point_idx in range(single_digital_wave_point_num):
            wave_point_420[point_formulated + point_idx] >>= 1
            if point_idx < int(round(samplerate * x0[2 * i + 1] * 10 ** -6)):
                wave_point_420[point_formulated + point_idx] |= 0x8000
            else:
                wave_point_420[point_formulated + point_idx] &= 0x7FFF

        point_formulated = point_formulated + single_digital_wave_point_num

    for i in range(point_formulated, wave_point_num):
        wave_point_420[i] >>= 1
        wave_point_420[i] &= 0x7FFF

    # -----------------------------------------------------------------------#
    # then we calculate new waveform for 1013
    # -----------------------------------------------------------------------#
    point_formulated = 0
    for i in range(5):
        single_digital_wave_point_num = int(round(samplerate * (x1[2 * i + 1] + x1[2 * i + 2]) * 10 ** -6))
        for point_idx in range(single_digital_wave_point_num):
            wave_point_1013[point_formulated + point_idx] >>= 1
            if point_idx < int(round(samplerate * x1[2 * i + 1] * 10 ** -6)):
                wave_point_1013[point_formulated + point_idx] |= 0x8000
            else:
                wave_point_1013[point_formulated + point_idx] &= 0x7FFF

        point_formulated = point_formulated + single_digital_wave_point_num

    for i in range(point_formulated, wave_point_num):
        wave_point_1013[i] >>= 1
        wave_point_1013[i] &= 0x7FFF

    wave_point = np.insert(wave_point_420, idx, wave_point_1013)

    return wave_point, wave_point_420, wave_point_1013


def write_config(ui_sequence):
    settings = ui_sequence.settings
    config_name = ui_sequence.config_name

    settings.setValue("{}/CLOCK_FREQUENCY".format(config_name), ui_sequence.clock_frequency.currentIndex())
    settings.setValue("{}/OUTPUT_MODE".format(config_name), ui_sequence.output_mode.currentIndex())
    settings.setValue("{}/TRIGGER".format(config_name), ui_sequence.trigger.currentIndex())

    settings.setValue("{}/CHECKBOX_420_1".format(config_name), int(ui_sequence.checkBox_420_1.isChecked()))
    settings.setValue("{}/CHECKBOX_420_2".format(config_name), int(ui_sequence.checkBox_420_2.isChecked()))
    settings.setValue("{}/CHECKBOX_420_3".format(config_name), int(ui_sequence.checkBox_420_3.isChecked()))
    settings.setValue("{}/CHECKBOX_420_4".format(config_name), int(ui_sequence.checkBox_420_4.isChecked()))
    settings.setValue("{}/CHECKBOX_420_5".format(config_name), int(ui_sequence.checkBox_420_5.isChecked()))
    settings.setValue("{}/CHECKBOX_420_6".format(config_name), int(ui_sequence.checkBox_420_6.isChecked()))

    settings.setValue("{}/CHECKBOX_1013_1".format(config_name), int(ui_sequence.checkBox_1013_1.isChecked()))
    settings.setValue("{}/CHECKBOX_1013_2".format(config_name), int(ui_sequence.checkBox_1013_2.isChecked()))
    settings.setValue("{}/CHECKBOX_1013_3".format(config_name), int(ui_sequence.checkBox_1013_3.isChecked()))
    settings.setValue("{}/CHECKBOX_1013_4".format(config_name), int(ui_sequence.checkBox_1013_4.isChecked()))
    settings.setValue("{}/CHECKBOX_1013_5".format(config_name), int(ui_sequence.checkBox_1013_5.isChecked()))
    settings.setValue("{}/CHECKBOX_1013_6".format(config_name), int(ui_sequence.checkBox_1013_6.isChecked()))

    settings.setValue("{}/SHOW_FREQUENCY_1013".format(config_name), int(ui_sequence.show_frequency_1013.isChecked()))
    settings.setValue("{}/SHOW_FREQUENCY_420".format(config_name), int(ui_sequence.show_frequency_420.isChecked()))

    settings.setValue("{}/SHOW_X0".format(config_name), int(ui_sequence.show_x_0.isChecked()))
    settings.setValue("{}/SHOW_X1".format(config_name), int(ui_sequence.show_x_1.isChecked()))

    settings.setValue("{}/WAVEFORM_420_1".format(config_name), ui_sequence.waveform_420_1.currentIndex())
    settings.setValue("{}/WAVEFORM_420_2".format(config_name), ui_sequence.waveform_420_2.currentIndex())
    settings.setValue("{}/WAVEFORM_420_3".format(config_name), ui_sequence.waveform_420_3.currentIndex())
    settings.setValue("{}/WAVEFORM_420_4".format(config_name), ui_sequence.waveform_420_4.currentIndex())
    settings.setValue("{}/WAVEFORM_420_5".format(config_name), ui_sequence.waveform_420_5.currentIndex())
    settings.setValue("{}/WAVEFORM_420_6".format(config_name), ui_sequence.waveform_420_6.currentIndex())
    settings.setValue("{}/WAVEFORM_1013_1".format(config_name), ui_sequence.waveform_1013_1.currentIndex())
    settings.setValue("{}/WAVEFORM_1013_2".format(config_name), ui_sequence.waveform_1013_2.currentIndex())
    settings.setValue("{}/WAVEFORM_1013_3".format(config_name), ui_sequence.waveform_1013_3.currentIndex())
    settings.setValue("{}/WAVEFORM_1013_4".format(config_name), ui_sequence.waveform_1013_4.currentIndex())
    settings.setValue("{}/WAVEFORM_1013_5".format(config_name), ui_sequence.waveform_1013_5.currentIndex())
    settings.setValue("{}/WAVEFORM_1013_6".format(config_name), ui_sequence.waveform_1013_6.currentIndex())

    settings.setValue("{}/START_AMP_420_1".format(config_name), ui_sequence.start_amp_420_1.text())
    settings.setValue("{}/END_AMP_420_1".format(config_name), ui_sequence.end_amp_420_1.text())
    settings.setValue("{}/NUM_AMP_420_1".format(config_name), ui_sequence.num_amp_420_1.text())
    settings.setValue("{}/START_FREQUENCY_420_1".format(config_name), ui_sequence.start_frequency_420_1.text())
    settings.setValue("{}/END_FREQUENCY_420_1".format(config_name), ui_sequence.end_frequency_420_1.text())
    settings.setValue("{}/TIME_420_1".format(config_name), ui_sequence.time_420_1.text())
    settings.setValue("{}/WAIT_TIME_420_1".format(config_name), ui_sequence.wait_time_420_1.text())

    settings.setValue("{}/START_AMP_420_2".format(config_name), ui_sequence.start_amp_420_2.text())
    settings.setValue("{}/END_AMP_420_2".format(config_name), ui_sequence.end_amp_420_2.text())
    settings.setValue("{}/NUM_AMP_420_2".format(config_name), ui_sequence.num_amp_420_2.text())
    settings.setValue("{}/START_FREQUENCY_420_2".format(config_name), ui_sequence.start_frequency_420_2.text())
    settings.setValue("{}/END_FREQUENCY_420_2".format(config_name), ui_sequence.end_frequency_420_2.text())
    settings.setValue("{}/TIME_420_2".format(config_name), ui_sequence.time_420_2.text())
    settings.setValue("{}/WAIT_TIME_420_2".format(config_name), ui_sequence.wait_time_420_2.text())

    settings.setValue("{}/START_AMP_420_3".format(config_name), ui_sequence.start_amp_420_3.text())
    settings.setValue("{}/END_AMP_420_3".format(config_name), ui_sequence.end_amp_420_3.text())
    settings.setValue("{}/NUM_AMP_420_3".format(config_name), ui_sequence.num_amp_420_3.text())
    settings.setValue("{}/START_FREQUENCY_420_3".format(config_name), ui_sequence.start_frequency_420_3.text())
    settings.setValue("{}/END_FREQUENCY_420_3".format(config_name), ui_sequence.end_frequency_420_3.text())
    settings.setValue("{}/TIME_420_3".format(config_name), ui_sequence.time_420_3.text())
    settings.setValue("{}/WAIT_TIME_420_3".format(config_name), ui_sequence.wait_time_420_3.text())

    settings.setValue("{}/START_AMP_420_4".format(config_name), ui_sequence.start_amp_420_4.text())
    settings.setValue("{}/END_AMP_420_4".format(config_name), ui_sequence.end_amp_420_4.text())
    settings.setValue("{}/NUM_AMP_420_4".format(config_name), ui_sequence.num_amp_420_4.text())
    settings.setValue("{}/START_FREQUENCY_420_4".format(config_name), ui_sequence.start_frequency_420_4.text())
    settings.setValue("{}/END_FREQUENCY_420_4".format(config_name), ui_sequence.end_frequency_420_4.text())
    settings.setValue("{}/TIME_420_4".format(config_name), ui_sequence.time_420_4.text())
    settings.setValue("{}/WAIT_TIME_420_4".format(config_name), ui_sequence.wait_time_420_4.text())

    settings.setValue("{}/START_AMP_420_5".format(config_name), ui_sequence.start_amp_420_5.text())
    settings.setValue("{}/END_AMP_420_5".format(config_name), ui_sequence.end_amp_420_5.text())
    settings.setValue("{}/NUM_AMP_420_5".format(config_name), ui_sequence.num_amp_420_5.text())
    settings.setValue("{}/START_FREQUENCY_420_5".format(config_name), ui_sequence.start_frequency_420_5.text())
    settings.setValue("{}/END_FREQUENCY_420_5".format(config_name), ui_sequence.end_frequency_420_5.text())
    settings.setValue("{}/TIME_420_5".format(config_name), ui_sequence.time_420_5.text())
    settings.setValue("{}/WAIT_TIME_420_5".format(config_name), ui_sequence.wait_time_420_5.text())

    settings.setValue("{}/START_AMP_420_6".format(config_name), ui_sequence.start_amp_420_6.text())
    settings.setValue("{}/END_AMP_420_6".format(config_name), ui_sequence.end_amp_420_6.text())
    settings.setValue("{}/NUM_AMP_420_6".format(config_name), ui_sequence.num_amp_420_6.text())
    settings.setValue("{}/START_FREQUENCY_420_6".format(config_name), ui_sequence.start_frequency_420_6.text())
    settings.setValue("{}/END_FREQUENCY_420_6".format(config_name), ui_sequence.end_frequency_420_6.text())
    settings.setValue("{}/TIME_420_6".format(config_name), ui_sequence.time_420_6.text())
    settings.setValue("{}/WAIT_TIME_420_6".format(config_name), ui_sequence.wait_time_420_6.text())

    settings.setValue("{}/START_AMP_1013_1".format(config_name), ui_sequence.start_amp_1013_1.text())
    settings.setValue("{}/END_AMP_1013_1".format(config_name), ui_sequence.end_amp_1013_1.text())
    settings.setValue("{}/NUM_AMP_1013_1".format(config_name), ui_sequence.num_amp_1013_1.text())
    settings.setValue("{}/START_FREQUENCY_1013_1".format(config_name), ui_sequence.start_frequency_1013_1.text())
    settings.setValue("{}/END_FREQUENCY_1013_1".format(config_name), ui_sequence.end_frequency_1013_1.text())
    settings.setValue("{}/TIME_1013_1".format(config_name), ui_sequence.time_1013_1.text())
    settings.setValue("{}/WAIT_TIME_1013_1".format(config_name), ui_sequence.wait_time_1013_1.text())

    settings.setValue("{}/START_AMP_1013_2".format(config_name), ui_sequence.start_amp_1013_2.text())
    settings.setValue("{}/END_AMP_1013_2".format(config_name), ui_sequence.end_amp_1013_2.text())
    settings.setValue("{}/NUM_AMP_1013_2".format(config_name), ui_sequence.num_amp_1013_2.text())
    settings.setValue("{}/START_FREQUENCY_1013_2".format(config_name), ui_sequence.start_frequency_1013_2.text())
    settings.setValue("{}/END_FREQUENCY_1013_2".format(config_name), ui_sequence.end_frequency_1013_2.text())
    settings.setValue("{}/TIME_1013_2".format(config_name), ui_sequence.time_1013_2.text())
    settings.setValue("{}/WAIT_TIME_1013_2".format(config_name), ui_sequence.wait_time_1013_2.text())

    settings.setValue("{}/START_AMP_1013_3".format(config_name), ui_sequence.start_amp_1013_3.text())
    settings.setValue("{}/END_AMP_1013_3".format(config_name), ui_sequence.end_amp_1013_3.text())
    settings.setValue("{}/NUM_AMP_1013_3".format(config_name), ui_sequence.num_amp_1013_3.text())
    settings.setValue("{}/START_FREQUENCY_1013_3".format(config_name), ui_sequence.start_frequency_1013_3.text())
    settings.setValue("{}/END_FREQUENCY_1013_3".format(config_name), ui_sequence.end_frequency_1013_3.text())
    settings.setValue("{}/TIME_1013_3".format(config_name), ui_sequence.time_1013_3.text())
    settings.setValue("{}/WAIT_TIME_1013_3".format(config_name), ui_sequence.wait_time_1013_3.text())

    settings.setValue("{}/START_AMP_1013_4".format(config_name), ui_sequence.start_amp_1013_4.text())
    settings.setValue("{}/END_AMP_1013_4".format(config_name), ui_sequence.end_amp_1013_4.text())
    settings.setValue("{}/NUM_AMP_1013_4".format(config_name), ui_sequence.num_amp_1013_4.text())
    settings.setValue("{}/START_FREQUENCY_1013_4".format(config_name), ui_sequence.start_frequency_1013_4.text())
    settings.setValue("{}/END_FREQUENCY_1013_4".format(config_name), ui_sequence.end_frequency_1013_4.text())
    settings.setValue("{}/TIME_1013_4".format(config_name), ui_sequence.time_1013_4.text())
    settings.setValue("{}/WAIT_TIME_1013_4".format(config_name), ui_sequence.wait_time_1013_4.text())

    settings.setValue("{}/START_AMP_1013_5".format(config_name), ui_sequence.start_amp_1013_5.text())
    settings.setValue("{}/END_AMP_1013_5".format(config_name), ui_sequence.end_amp_1013_5.text())
    settings.setValue("{}/NUM_AMP_1013_5".format(config_name), ui_sequence.num_amp_1013_5.text())
    settings.setValue("{}/START_FREQUENCY_1013_5".format(config_name), ui_sequence.start_frequency_1013_5.text())
    settings.setValue("{}/END_FREQUENCY_1013_5".format(config_name), ui_sequence.end_frequency_1013_5.text())
    settings.setValue("{}/TIME_1013_5".format(config_name), ui_sequence.time_1013_5.text())
    settings.setValue("{}/WAIT_TIME_1013_5".format(config_name), ui_sequence.wait_time_1013_5.text())

    settings.setValue("{}/START_AMP_1013_6".format(config_name), ui_sequence.start_amp_1013_6.text())
    settings.setValue("{}/END_AMP_1013_6".format(config_name), ui_sequence.end_amp_1013_6.text())
    settings.setValue("{}/NUM_AMP_1013_6".format(config_name), ui_sequence.num_amp_1013_6.text())
    settings.setValue("{}/START_FREQUENCY_1013_6".format(config_name), ui_sequence.start_frequency_1013_6.text())
    settings.setValue("{}/END_FREQUENCY_1013_6".format(config_name), ui_sequence.end_frequency_1013_6.text())
    settings.setValue("{}/TIME_1013_6".format(config_name), ui_sequence.time_1013_6.text())
    settings.setValue("{}/WAIT_TIME_1013_6".format(config_name), ui_sequence.wait_time_1013_6.text())

    settings.setValue("{}/X0_HIGH0".format(config_name), ui_sequence.x0_high0.text())
    settings.setValue("{}/X0_LOW0".format(config_name), ui_sequence.x0_low0.text())
    settings.setValue("{}/X0_HIGH1".format(config_name), ui_sequence.x0_high1.text())
    settings.setValue("{}/X0_LOW1".format(config_name), ui_sequence.x0_low1.text())
    settings.setValue("{}/X0_HIGH2".format(config_name), ui_sequence.x0_high2.text())
    settings.setValue("{}/X0_LOW2".format(config_name), ui_sequence.x0_low2.text())
    settings.setValue("{}/X0_HIGH3".format(config_name), ui_sequence.x0_high3.text())
    settings.setValue("{}/X0_LOW3".format(config_name), ui_sequence.x0_low3.text())
    settings.setValue("{}/X0_HIGH4".format(config_name), ui_sequence.x0_high4.text())
    settings.setValue("{}/X0_LOW4".format(config_name), ui_sequence.x0_low4.text())

    settings.setValue("{}/X1_HIGH0".format(config_name), ui_sequence.x1_high0.text())
    settings.setValue("{}/X1_LOW0".format(config_name), ui_sequence.x1_low0.text())
    settings.setValue("{}/X1_HIGH1".format(config_name), ui_sequence.x1_high1.text())
    settings.setValue("{}/X1_LOW1".format(config_name), ui_sequence.x1_low1.text())
    settings.setValue("{}/X1_HIGH2".format(config_name), ui_sequence.x1_high2.text())
    settings.setValue("{}/X1_LOW2".format(config_name), ui_sequence.x1_low2.text())
    settings.setValue("{}/X1_HIGH3".format(config_name), ui_sequence.x1_high3.text())
    settings.setValue("{}/X1_LOW3".format(config_name), ui_sequence.x1_low3.text())
    settings.setValue("{}/X1_HIGH4".format(config_name), ui_sequence.x1_high4.text())
    settings.setValue("{}/X1_LOW4".format(config_name), ui_sequence.x1_low4.text())


def mem_size_length(length_wave):
    if length_wave % 32 == 0:
        return int(length_wave)
    else:
        return int((length_wave // 32 + 1) * 32)
