from pyspcm import *
from spcm_tools import *
import numpy as np
# my_awg = RydbergPulseAwg()c
# my_awg.close_card()
import matplotlib.pyplot as plt

from rydberg_awg import waveform


def wave_calculate(samplerate, waveform_cont_420, waveform_cont_1013):
    amp_std = 30000

    # first we compare time consume of 420 and 1013
    wave_time_420 = 0
    for single_wave_420 in waveform_cont_420:
        print(single_wave_420)
        wave_time_420 += single_wave_420.time
        wave_time_420 += single_wave_420.wait_time
    wave_time_1013 = 0
    for single_wave_1013 in waveform_cont_1013:
        print(single_wave_1013)
        wave_time_1013 += single_wave_1013.time
        wave_time_1013 += single_wave_1013.wait_time

    wave_point_num = int(samplerate * wave_time_420 * 10 ** -6)
    if wave_time_420 < wave_time_1013:
        wave_point_num = int(samplerate * wave_time_1013 * 10 ** -6)
    wave_point_420 = []
    wave_point_1013 = []
    # then we need to calculate each wave point according to wave_form_cont
    for single_wave in waveform_cont_420:
        single_wave_point = np.zeros(int(samplerate * (single_wave.time + single_wave.wait_time) * 10 ** -6),
                                     dtype='int')
        if single_wave.form == "flat pulse":

            amp_ratio = single_wave.amp * 0.01
            frequency = MEGA(single_wave.start_frequency)
            signal_point = int(samplerate * single_wave.time * 10 ** -6)
            wait_point = samplerate * single_wave.wait_time * 10 ** -6
            for i in range(signal_point):
                single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * frequency / samplerate * i))
        elif single_wave.form == "blackman pulse":
            pass
        elif single_wave.form == "ramp frequency":
            pass

        wave_point_420 = np.append(wave_point_420, single_wave_point)
    wave_point_420 = np.asarray(wave_point_420, dtype='int')
    print(type(wave_point_420[0]))

    for single_wave in waveform_cont_1013:
        single_wave_point = np.zeros(int(samplerate * (single_wave.time + single_wave.wait_time) * 10 ** -6),
                                     dtype='int')
        if single_wave.form == "flat pulse":
            amp_ratio = single_wave.amp * 0.01
            frequency = MEGA(single_wave.start_frequency)
            signal_point = int(samplerate * single_wave.time * 10 ** -6)
            wait_point = samplerate * single_wave.wait_time * 10 ** -6
            for i in range(signal_point):
                single_wave_point[i] = int(amp_std * amp_ratio * np.sin(2 * np.pi * frequency / samplerate * i))
        elif single_wave.form == "blackman pulse":
            pass
        elif single_wave.form == "ramp frequency":
            pass
        wave_point_1013 = np.append(wave_point_1013, single_wave_point)
    wave_point_1013 = np.asarray(wave_point_1013, dtype='int')
    print(type(wave_point_1013[0]))
    if wave_time_420 > wave_time_1013:
        wave_point_1013 = np.pad(wave_point_1013, [(0, wave_point_num - len(wave_point_1013))])
    else:
        wave_point_420 = np.pad(wave_point_420, [(0, wave_point_num - len(wave_point_420))])

    # then we insert wave_point_1013 into wave_point_420 to construct wave_point
    idx = tuple(range(1, wave_point_num + 1))
    wave_point = np.insert(wave_point_420, idx, wave_point_1013)
    print(type(wave_point[0]))
    return wave_point, wave_point_420, wave_point_1013


waveform1 = waveform(form="flat pulse", amp=100, start_frequency=1, end_frequency=1, time=1, wait_time=0)
waveform2 = waveform(form="flat pulse", amp=70, start_frequency=1.5, end_frequency=1, time=2, wait_time=4)
waveform3 = waveform(form="flat pulse", amp=50, start_frequency=2, end_frequency=2, time=7, wait_time=3)
waveform4 = waveform(form="flat pulse", amp=75, start_frequency=2, end_frequency=2, time=5, wait_time=0)
waveform_cont_420 = [waveform1,waveform2]
waveform_cont_1013 = [waveform3,waveform4]

wave_point, wave_point_420, wave_point_1013 = wave_calculate(MEGA(1250), waveform_cont_420, waveform_cont_1013)
wave1_idx = np.arange(0, len(wave_point), 2)
wave1 = wave_point[wave1_idx]
plt.plot(wave1)
plt.title("fenjie")
plt.show()

# open card
# uncomment the second line and replace the IP address to use remote
# cards like in a generatorNETBOX
hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
# read type, function and sn and check for D/A card
lCardType = int32(0)
spcm_dwGetParam_i32(hCard, SPC_PCITYP, byref(lCardType))
lSerialNumber = int32(0)
spcm_dwGetParam_i32(hCard, SPC_PCISERIALNO, byref(lSerialNumber))
lFncType = int32(0)
spcm_dwGetParam_i32(hCard, SPC_FNCTYPE, byref(lFncType))
sCardName = szTypeToName(lCardType.value)
sys.stdout.write("Found: {0} sn {1:05d}\n".format(sCardName, lSerialNumber.value))

spcm_dwSetParam_i64(hCard, SPC_SAMPLERATE, MEGA(1250))

# setup the mode
qwChEnable = int64(CHANNEL0)
qwChEnable = int64 (CHANNEL0 | CHANNEL1) # uncomment to enable two channels
llMemSamples = int64(KILO_B(64))
llLoops = int64(1)  # loop continuously
spcm_dwSetParam_i32(hCard, SPC_CARDMODE, SPC_REP_STD_SINGLE)
spcm_dwSetParam_i64(hCard, SPC_CHENABLE, qwChEnable)
spcm_dwSetParam_i64(hCard, SPC_MEMSIZE, llMemSamples)
spcm_dwSetParam_i64(hCard, SPC_LOOPS, llLoops)
spcm_dwSetParam_i64(hCard, SPC_ENABLEOUT0, 1)

# setup the channels
lSetChannels = int32(0)
spcm_dwGetParam_i32(hCard, SPC_CHCOUNT, byref(lSetChannels))
lBytesPerSample = int32(0)
spcm_dwGetParam_i32(hCard, SPC_MIINST_BYTESPERSAMPLE, byref(lBytesPerSample))
for lChannel in range(0, lSetChannels.value, 1):
    spcm_dwSetParam_i32(hCard, SPC_ENABLEOUT0 + lChannel * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1)
    spcm_dwSetParam_i32(hCard, SPC_AMP0 + lChannel * (SPC_AMP1 - SPC_AMP0), 1000)
    spcm_dwSetParam_i32(hCard, SPC_CH0_STOPLEVEL + lChannel * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL),
                        SPCM_STOPLVL_HOLDLAST)

# (SW trigger, no output)
spcm_dwSetParam_i32(hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
spcm_dwSetParam_i32(hCard, SPC_TRIG_ANDMASK, 0)
spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ORMASK0, 0)
spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ORMASK1, 0)
spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ANDMASK0, 0)
spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ANDMASK1, 0)
spcm_dwSetParam_i32(hCard, SPC_TRIGGEROUT, 0)

# setup software buffer
qwBufferSize = uint64(llMemSamples.value * lBytesPerSample.value * lSetChannels.value)
# we try to use continuous memory if available and big enough
pvBuffer = pvAllocMemPageAligned(qwBufferSize.value)

# calculate the data
pnBuffer = cast(pvBuffer, ptr16)
for i in range(0, len(wave_point), 1):
    pnBuffer[i] = wave_point[i]

# we define the buffer for transfer and start the DMA transfer
sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
spcm_dwDefTransfer_i64(hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), pvBuffer, uint64(0), qwBufferSize)
spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
sys.stdout.write("... data has been transferred to board memory\n")

# We'll start and wait until the card has finished or until a timeout occurs
spcm_dwSetParam_i32(hCard, SPC_TIMEOUT, 5000)
sys.stdout.write(
    "\nStarting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)\n")
dwError = spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
if dwError == ERR_TIMEOUT:
    spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_STOP)

spcm_vClose(hCard);
