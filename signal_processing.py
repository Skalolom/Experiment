import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import signal

# define constant string for searching in text files
FREQUENCY_SUBSTRING = 'kHz'
TIME_SUBSTRING = 'Sec'
MAXIMUM_LOSS_IN_PASS_BAND = 1e-1
MINIMUM_ATTENUATION_IN_STOP_BAND = 40


def get_text_files_from_directory(directory_path: str) -> list:

    """
    This function returns list of text files in directory 'path'

    :param directory_path path to directory

    :return text_files list of text files in directory_path

    """

    text_files = []
    for root, directory, files in os.walk(directory_path):
        for file in files:
            text_files.append(os.path.join(root, file))
    text_files.sort()

    return text_files


def get_directories_from_root(root_path: str) -> list:
    """
    This function returns list of directories in root directory root_path

    :param root_path path to root directory

    :return directories_list list of directories in root_path

    """
    directories_list = []

    for root, directories, files in os.walk(root_path):
        for directory in directories:
            directories_list.append(os.path.join(root, directory))

    return directories_list


def get_pressure_and_time_from_text_file(text_file_path):
    """

    This function returns vectors of pressure and time values from text file text_file_path

    :param text_file_path: path to text file with results of experiment

    :return: data_from_file (pressure_vector, time_vector) - tuple of time and pressure vectors

    """

    # time and pressure vectors
    pressure_vector = time_vector = np.array([])
    # duration of experiment and working frequency of pressure sensor
    experiment_duration, sensor_frequency = 0, 0

    with open(text_file_path, 'r') as text_file:
        for line in text_file:
            # if line consists of only 2 elements (which means that first is time stamp, and second is
            # pressure value)
            try:
                elements_in_current_line = [float(string) for string in line.split()]
            except ValueError:
                continue
            else:
                if len(elements_in_current_line) == 2:
                    time_vector = np.append(time_vector, float(elements_in_current_line[0]))
                    pressure_vector = np.append(pressure_vector, float(elements_in_current_line[1]))

    return time_vector, pressure_vector


def get_signal_frequency(time_vector, pressure_vector):
    """

    This function calculates signal frequency wo high-frequency noise

    :param time_vector: vector of time stamps
    :param pressure_vector: vector of pressure samples from the sensor
    :return: signal frequency, which has to be used in filter design
    """

    # evaluate sampling frequency (number of sample per second)
    sampling_frequency = (len(time_vector)/time_vector[-1])*1e3
    # evaluate Nyquist frequency accordingly (1/2 of sampling_frequency)
    nyquist_frequency = 0.5*sampling_frequency
    # now evaluate window function
    window = signal.get_window('blackmanharris', Nx=2**11)
    # evaluate power spectral density
    sampling_frequencies, power_spectral_density = signal.welch(
        x=pressure_vector, fs=sampling_frequency
    )
    # now evaluate peaks of power spectral density
    peaks, _ = signal.find_peaks(x=power_spectral_density[100:], height=1e-1, distance=150)
    # now we can get frequency of carrier signal wo high-frequency noise
    signal_frequency = sampling_frequencies[peaks][1]

    print(signal_frequency)

    plt.figure(1)
    plt.semilogy(sampling_frequencies, power_spectral_density,
                 sampling_frequencies[peaks], power_spectral_density[peaks], 'x')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [cm**2/Hz]')
    plt.show()

    return signal_frequency


def filter_pressure_vector(time_vector, pressure_vector):
    """

    This function filters pressure vector and returns smoothed vector of pressure without
    high-frequency noise

    :param time_vector: vector of time stamps

    :param pressure_vector: vector of pressure values from the sensor

    :return: pressure_vector_filtered: vector of filtered values of pressure
    """
    # evaluate filter design using signal_frequency
    # evaluate sampling frequency (number of sample per second)
    sampling_frequency = (len(time_vector)/time_vector[-1])*1e3
    # evaluate optimal filter parameters
    pass_stop = 0.2
    pass_band = 0.5*pass_stop
    minimal_butter_order, butterworth_natural_frequency = signal.buttord(
        wp=pass_band, ws=pass_stop,
        gpass=MAXIMUM_LOSS_IN_PASS_BAND, gstop=MINIMUM_ATTENUATION_IN_STOP_BAND,
        fs=sampling_frequency, analog=False
    )

    # # filter signal
    # second_order_section = signal.butter(N=minimal_butter_order, Wn=butterworth_natural_frequency,
    #                                      analog=False, output='sos', fs=sampling_frequency)
    #
    # # apply filter to pressure_vector
    # pressure_vector_filtered = signal.sosfiltfilt(sos=second_order_section, x=pressure_vector)

    pressure_vector_filtered = signal.savgol_filter(x=pressure_vector, window_length=333, polyorder=5,
                                                    mode='nearest')

    plt.figure(2)
    plt.plot(time_vector, pressure_vector, 'b',
             time_vector, pressure_vector_filtered, 'r')
    plt.ylim([-100, 100])
    plt.show()

    return pressure_vector_filtered
    

def approximate_pressure_with_polynomial(time_vector, pressure_vector, pressure_vector_filtered):
    """

    :param time_vector: vector of time stamps
    :param pressure_vector: vector of pressure values
    :param pressure_vector_filtered: vector of filtered values of pressure

    :return: approximated polynomial
    """

    r_0 = 5.0
    # calculate polynomial coefficients
    polynomial_coefficients = np.polyfit(time_vector, pressure_vector_filtered, 2)
    # construct polynomial according to this coeffs
    polynomial_pressure_function = np.poly1d(polynomial_coefficients)
    polynomial_pressure_vector = polynomial_pressure_function(time_vector)
    
    # вычисляем расход
    dt, dh = np.abs(np.diff(time_vector)), np.abs(np.diff(polynomial_pressure_vector))
    dt = np.append(dt, dt[-1])
    dh = np.append(dh, dh[-1])
    consumption = (np.pi*(10e-2*r_0)**2) * (dh/dt)

    plt.figure(3)
    plt.plot(polynomial_pressure_vector, consumption)
    plt.show()



def test_function():
    time_vector, pressure_vector = get_pressure_and_time_from_text_file(r'Data/5mm/c_sh_dmpJ.txt')
    pressure_vector_filtered = filter_pressure_vector(time_vector=time_vector,
                                                      pressure_vector=pressure_vector)
    approximate_pressure_with_polynomial(time_vector=time_vector,
                                         pressure_vector=pressure_vector,
                                         pressure_vector_filtered=pressure_vector_filtered)


class SignalProcessing:
    def __init__(self, path, log_path=r'path/log.txt'):
        self.files, self.path, self.log_path = [], path, log_path
        # считаем конфигурационный файл config.json в аттрибут класса config_data
        with open('config.json', 'a') as read_json:
            self.config_data = json.load(read_json)

        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.txt' in file or '.dat' in file:
                    self.files.append(os.path.join(r, file))

        self.files.sort()

    def process_data(self):
        sub_freq, sub_time, freq_rate, full_time = 'kHz', 'Sec', 1, 1
        time_vector = press_vector = names = bars = bar_colors = []
        press_min, press_max = 5, 45
        r_0, r_current, i, r_previous = 5, 5, 0, 0

        for filename in self.files:
            # вычисляем значение радиуса и цвета графика
            r_current, bar_color = self.evaluate_radius_and_bar_color(filename)

            # считываем значения моментов времени и высоты столба из файла
            press_vector, time_vector = SignalProcessing.read_time_and_pressure(
                filename, sub_freq, sub_time, r_current, r_0)

            # if name of the experiment has changed, add whitespace between bars
            if r_previous != r_current:
                bars.append(0)
                bar_colors.append(bar_color)
                # parse the filename for name of the group of experiments (for example, c_sh, c_sm)
                experiment_group_name = filename.split('/')[-1].split('.')[0].split('_')[0:2]
                names.append(experiment_group_name[0] + '_' + experiment_group_name[1])
            # save value of the r_current in r_previous
            r_previous = r_current

    @staticmethod
    # define function for segmenting input vectors according to min and max values. It returns min and max indexes
    def segment_vector(vector, min_value, max_value):
        max_index = np.argmin([abs(e - max_value) for e in vector])
        min_index = np.argmin([abs(e - min_value) for e in vector])
        return min_index, max_index

    @staticmethod
    def read_time_and_pressure(filename, sub_freq, sub_time, r_current, r_0):
        press_vector, time_vector = [], []
        full_time = 1
        # fill time and pressure vect
        for line in open(filename, 'r'):

            index = line.find(sub_freq)
            if index != -1:
                # read rate
                freq_rate = float(line[index + 5:]) * 1e3

            index = line.find(sub_time)
            if index != -1:
                # read time
                full_time = float(line[index + 5:])
                # if line contains only digits
            try:
                temp_string = [float(s) for s in line.split()]
                # if tempString has 2 elements
                if len(temp_string) == 2:
                    press_vector.append(temp_string[1])
                    time_vector.append(temp_string[0])
            except Exception:
                continue

        # нормируем значение времени в зависимости от полного времени истечения
        time_vector = [t / (time_vector[-1] / full_time) for t in time_vector]
        # нормируем значение времени по радиусу выходного отверстия
        k = r_current ** 2 / r_0 ** 2
        time_vector = [t * k for t in time_vector]

        return press_vector, time_vector

    def evaluate_radius_and_bar_color(self, filename):
        # считываем значения радиусов из конфигурационного словаря
        radius_dict = self.config_data['radius']
        if filename.find('c_sh') != -1:
            r_current = radius_dict['r_sh']
            if filename.find('J') != -1:
                bar_color = 'purple'
            else:
                bar_color = 'black'
        if filename.find('c_sm') != -1:
            r_current = radius_dict['r_sm']
            if filename.find('J') != -1:
                bar_color = 'purple'
            else:
                bar_color = 'magenta'
        if filename.find('r_1') != -1:
            r_current = radius_dict['r_1']
            if filename.find('J') != -1:
                bar_color = 'purple'
            else:
                bar_color = 'blue'
        if filename.find('r_2') != -1:
            r_current = radius_dict['r_2']
            if filename.find('J') != -1:
                bar_color = 'purple'
            else:
                bar_color = 'green'
        else:
            r_current = radius_dict['r_3']
            if filename.find('J') != -1:
                bar_color = 'purple'
            else:
                bar_color = 'red'

        return r_current, bar_color
