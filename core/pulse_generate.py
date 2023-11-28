import numpy as np
import matplotlib.pyplot as plt


def output_as_file(timeslot, omega_t, detun_t, filename):
    data = np.array(np.vstack((timeslot, omega_t, detun_t)).T)
    fig, axes = plt.subplots(2)
    axes[0].plot(timeslot, omega_t)
    axes[0].set_title("omega")
    axes[1].plot(timeslot, detun_t)
    axes[1].set_title("detuning")
    fig.savefig(filename + '.png')
    plt.close()
    print(data.shape)
    np.savetxt(filename + '.csv', data, delimiter=',', header='time,omega,detuning', comments='')


def flat_gen(detuning, amplitude, signal_t, filename):
    timeslot = np.linspace(0, signal_t, 2)
    omega_t = np.array([amplitude] *2)
    detun_t = np.array([detuning] * 2)
    output_as_file(timeslot, omega_t, detun_t, filename)


def blackman_gen(detuning, amplitude, signal_t, filename):
    timeslot = np.linspace(0, signal_t, 10)
    omega_t = (-0.5 * np.cos(2 * np.pi * timeslot / signal_t) + 0.08 * np.cos(
        4 * np.pi * timeslot / signal_t) + 0.42) * amplitude
    detun_t = np.array([detuning] * 10)
    output_as_file(timeslot, omega_t, detun_t, filename)


def ramp_gen(start_detuning, end_detuning, amplitude, signal_t, filename):
    timeslot = np.linspace(0, signal_t, 1000)
    omega_t = np.array([amplitude] * 1000)
    detun_t = np.linspace(start_detuning, end_detuning, 1000)
    output_as_file(timeslot, omega_t, detun_t, filename)


flat_gen(50, 1, 1, '../waveform/flat')
blackman_gen(50, 0.8, 5, "../waveform/blackman")
