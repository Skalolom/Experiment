import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import time
import pandas as pd

# define constant string for searching in text files
FREQUENCY_SUBSTRING = 'kHz'
TIME_SUBSTRING = 'Sec'
MAXIMUM_LOSS_IN_PASS_BAND = 1e-1
MINIMUM_ATTENUATION_IN_STOP_BAND = 40
# ratio between number of samples in pressure vector and median filter window size (e.g. window size calculated
# as len(pressure_vector)//ratio
FACTOR_PRESSURE_TO_WINDOW = 500


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


def get_pressure_and_time_from_text_file(text_file_path, number_of_info_rows):
    """

    This function returns vectors of pressure and time values from text file text_file_path

    :param text_file_path: path to text file with results of experiment
    :param number_of_info_rows: number of information string in th file

    :return: data_from_file (pressure_vector, time_vector) - tuple of time and pressure vectors

    """

    # here we read text file from 21th row (where information part ends). Then we call columns 't' for time samples
    # and 'p' for pressure samples respectively. Finally, we write vectors of time and pressure to np.ndarrays.
    with open(text_file_path, 'r') as text_file:
        df = pd.read_csv(text_file_path, skiprows=number_of_info_rows, header=None, names=['t', 'p'],
                         delim_whitespace=True)
    pressure_vector = df['p'].to_numpy()
    time_vector = df['t'].to_numpy()

    # now we cover the case when time_vector given in seconds instead of ms
    if max(time_vector) >= 1e4:
        time_vector = time_vector*1e-3

    # now we make amendments according to radius deviations in different geometries
    # first we deserialize config.json
    with open('config.json', 'r') as json_file:
        config_dict = json.load(json_file)
    # normalize time_vector to compensate radius deviation
    radius_dict = config_dict['radius']
    r_0 = radius_dict['r_0']
    r_current = r_0
    if 'c_sh' in text_file_path:
        r_current = radius_dict['r_sh']
    if 'c_sm' in text_file_path:
        r_current = radius_dict['r_sm']
    if 'r_1' in text_file_path:
        r_current = radius_dict['r_1']
    if 'r_2' in text_file_path:
        r_current = radius_dict['r_2']
    if 'r_3' in text_file_path:
        r_current = radius_dict['r_3']

    k = r_current ** 2 / r_0 ** 2
    time_vector = k*time_vector

    return time_vector, pressure_vector


def get_signal_frequency(time_vector, pressure_vector):
    """

    This function calculates signal frequency wo high-frequency noise

    :param time_vector: vector of time stamps
    :param pressure_vector: vector of pressure samples from the sensor
    :return: signal frequency, which has to be used in filter design
    """

    # evaluate sampling frequency (number of sample per second)
    sampling_frequency = (len(time_vector) / time_vector[-1]) * 1e3
    # evaluate Nyquist frequency accordingly (1/2 of sampling_frequency)
    nyquist_frequency = 0.5 * sampling_frequency
    # now evaluate window function
    window = signal.get_window('blackmanharris', Nx=2 ** 11)
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


def filter_pressure_vector(time_vector, pressure_vector, width_of_convolve_window,
                           number_of_convolution):
    """

    This function filters pressure vector and returns smoothed vector of pressure without
    high-frequency noise

    :param time_vector: vector of time stamps

    :param pressure_vector: vector of pressure values from the sensor

    :param width_of_convolve_window: number of elements in window function vector

    :param num_of_convolution: number of consecutive convolutions

    :return: pressure_vector_filtered: vector of filtered values of pressure
    """
    # evaluate filter design using signal_frequency
    # evaluate sampling frequency (number of sample per second)
    sampling_frequency = (len(time_vector) / time_vector[-1]) * 1e3

    # this fragment smooths pressure_vector by convolving it with window window_vector

    # cut slices (5%) from left and right sides of the pressure_vector
    # number_of_erased_elements = len(pressure_vector)//10
    # pressure_vector = pressure_vector[number_of_erased_elements-1:-number_of_erased_elements]
    # time_vector = time_vector[number_of_erased_elements-1:-number_of_erased_elements]

    # define vector of window function elements
    #window_vector = signal.windows.blackmanharris(width_of_convolve_window)
    window_vector = signal.windows.flattop(width_of_convolve_window, sym=False)
    #window_vector = signal.windows.gaussian(width_of_convolve_window, std=17.0, sym=False)

    pressure_vector_filtered = signal.decimate(x=pressure_vector, q=11)
    pressure_vector_filtered = signal.decimate(x=pressure_vector_filtered, q=11)
    pressure_vector_filtered = signal.decimate(x=pressure_vector_filtered, q=5)
    #time_vector = np.linspace(time_vector[0], time_vector[-1], len(pressure_vector))

    """
    add reflected copy from the beginning of pressure_vector to the left side of new filtered pressure vector (length
    of additive vector is equal to width_of_convolve window) and reflected copy of the pressure_vector ending to the 
    right side of the new filtered pressure vector to prevent convolve errors at sides of pressure vector
    """
    # # calculate left side of the pressure_vector_filtered
    # pressure_vector_filtered = np.append(pressure_vector[width_of_convolve_window-1::-1],
    #                                      pressure_vector)
    # # calculate right side of the pressure_vector_filtered
    # pressure_vector_filtered = np.append(pressure_vector_filtered,
    #                                      pressure_vector[-1:-width_of_convolve_window-1:-1])
    #
    # # now convolve pressure_vector_filtered with values of calculated window function
    # for i in range(number_of_convolution):
    #     pressure_vector_filtered = signal.oaconvolve(pressure_vector_filtered, window_vector/window_vector.sum(),
    #                                                  mode='valid')
    #
    # # calculate number of elements, that needs to be erased at both sides of pressure_vector_filtered
    # # first we evaluate difference between length of pressure_vector and pressure_vector_filtered
    # length_difference = np.abs(len(pressure_vector_filtered) - len(pressure_vector))
    # # now we evaluate number of erased elements as integer half of length_difference
    # number_of_erased_elements = length_difference//2
    # # now we evaluate resulting vector
    # pressure_vector_filtered = pressure_vector_filtered[number_of_erased_elements:-number_of_erased_elements]

    # # apply this window vector to initial signal (pressure_vector) number_of_convolution times
    # for i in range(number_of_convolution):
    #     pressure_vector_filtered = signal.convolve(pressure_vector_filtered, window_vector / window_vector.sum(),
    #                                                mode='valid')
    # pressure_vector_filtered = pressure_vector_filtered[width_of_convolve_window-1:]
    # length of pressure_vector_filtered due to filter operations differs from time_vector length.
    # so we create new object, time_vector_filtered with equal length
    time_vector_filtered = np.linspace(start=time_vector[0], stop=time_vector[-1],
                                       num=len(pressure_vector_filtered))
    plt.figure(2)
    plt.plot(time_vector, pressure_vector, 'b',
             time_vector_filtered, pressure_vector_filtered, 'r')
    plt.ylim([-100, 100])
    plt.show()

    return pressure_vector_filtered


def downsample_pressure_vector(pressure_vector, downsampling_factor, number_of_decimations):

    """
    This function calculates consecutive decimations of input pressure. Number of decimations given by the value
    of parameter number_of_decimations

    :param pressure_vector: input pressure vector
    :param downsampling_factor: downsampling factor
    :param number_of_decimations: number of consecutive decimations
    :return: downsampled_signal: input signal affected by consecutive decimations
    """
    pressure_vector_downsampled = pressure_vector
    for i in range(number_of_decimations):
        pressure_vector_downsampled = signal.decimate(x=pressure_vector_downsampled,
                                                      q=downsampling_factor)
    return pressure_vector_downsampled


def segment_vector(vector, min_value, max_value):
    max_index = np.argmin([abs(e - max_value) for e in vector])
    min_index = np.argmin([abs(e - min_value) for e in vector])
    return min_index, max_index


def approximate_pressure_with_polynomial(time_vector, pressure_vector, filter_window_factor, polynomial_degree,
                                         pressure_min, pressure_max):
    """
    This function applies median filter to input pressure vector and then approximates filtered pressure vector
    with polynomial of order polynomial_order

    :param polynomial_degree: degree of approximation polynomial
    :param pressure_min: minimal value of pressure
    :param pressure_max: maximal value of pressure
    :param time_vector: vector of time stamps
    :param pressure_vector: vector of pressure values
    :param filter_window_factor: length of pressure_vector//size in elements of median filter

    :return: time_vector_filtered: vector of time stamps relative to input pressure max and pressure min
    :return: polynomial_pressure_vector: approximated polynomial
    """

    # apply median filter to input pressure vector
    pressure_vector_filtered = ndimage.median_filter(pressure_vector,
                                                     size=len(pressure_vector)//filter_window_factor)

    # segment pressure_vector according to values pressure_min and pressure_max
    min_index, max_index = segment_vector(pressure_vector_filtered, pressure_min, pressure_max)
    pressure_vector_filtered = pressure_vector_filtered[max_index:min_index]
    time_vector_filtered = time_vector[max_index:min_index]
    # calculate polynomial coefficients
    polynomial_coefficients = np.polyfit(x=time_vector_filtered, y=pressure_vector_filtered, deg=polynomial_degree)
    # construct polynomial according to this coefficients
    pressure_vector_filtered = np.polyval(polynomial_coefficients, time_vector_filtered)

    return time_vector_filtered, pressure_vector_filtered


def calculate_consumption_ratio(time_vector, pressure_vector, radius_of_barrel, radius_of_hole):
    """

    that function evaluates consumption ration mu for emptying the barrel with constant radius (radius_of_barrel)
    through the hole with radius = radius_of_hole

    :param time_vector: vector of timestamps
    :param pressure_vector: vector of liquid height values
    :param radius_of_barrel: radius of the cross-section of emptied barrel
    :param radius_of_hole: radius of the cross-section of the hole, that empties the barrel
    :return: mu: consumption ration according to given parameters
    """

    # here we evaluate consumption ratio assuming that velocity in the hole is equal to sqrt(2gh)
    # here we calculate cross-section of the barrel
    s_barrel = np.pi*(radius_of_barrel**2)
    # here we calculate cross-section of the hole
    s_hole = np.pi*(radius_of_hole**2)
    h_c = 0.12
    h1, h2 = max(pressure_vector), min(pressure_vector)

    # here we calculate full experimental time, that needed to empty the barrel
    time_experimental = max(time_vector) - min(time_vector)
    time_ideal = (2*s_barrel*(np.sqrt(h1+h_c) - np.sqrt(h2+h_c)))/(s_hole*np.sqrt(2*9.8))
    mu = (2*s_barrel*(np.sqrt(h1+h_c) - np.sqrt(h2+h_c)))/(time_experimental*s_hole*np.sqrt(2*9.8))

    return mu


def test_function():
    i = 0
    start_time = time.time()
    mu_dict = {}
    # time_vector, pressure_vector = get_pressure_and_time_from_text_file(r'Data/Spiral/5mm/c_sh_flt.txt', 20)
    # time_vector_filtered, pressure_vector_filtered = approximate_pressure_with_polynomial(
    #     time_vector=time_vector, pressure_vector=pressure_vector,
    #     filter_window_factor=FACTOR_PRESSURE_TO_WINDOW, polynomial_degree=2,
    #     pressure_min=5, pressure_max=45)
    # mu = calculate_consumption_ratio(time_vector=time_vector_filtered, pressure_vector=1e-2*pressure_vector_filtered,
    #                                  radius_of_barrel=3e-1, radius_of_hole=5e-3)
    for directory in get_directories_from_root(r'Data/Spiral'):
        geometry_name = directory.split('/')[-1]
        mu_dict[geometry_name] = {}
        for file in get_text_files_from_directory(directory):
            time_vector, pressure_vector = get_pressure_and_time_from_text_file(file, 20)
            time_vector_filtered, pressure_vector_filtered = approximate_pressure_with_polynomial(
                 time_vector=time_vector, pressure_vector=pressure_vector,
                 filter_window_factor=FACTOR_PRESSURE_TO_WINDOW, polynomial_degree=2,
                 pressure_min=5, pressure_max=45)
            mu = calculate_consumption_ratio(time_vector=time_vector_filtered,
                                             pressure_vector=1e-2 * pressure_vector_filtered,
                                             radius_of_barrel=3e-1, radius_of_hole=5e-3)
            # write mu into log.txt
            label = file.split('/')[-1].split('.')[0]
            # write mu to the mu_dict
            # first we write geometry name
            mu_dict[geometry_name][label] = np.round(mu, 2)
            print('#', sep='', end='')
    # #         i += 1
    with open('mu_values.json', 'w') as json_file:
        json.dump(obj=mu_dict, fp=json_file, indent=4)
    finish_time = time.time()
    print('\naverage time = {0:.1f} ms'.format(1e3*(finish_time - start_time)))
    # plt.figure(1)
    # plt.rcParams.update({'font.size': 22})
    # plt.plot(time_vector, pressure_vector, 'b', time_vector_filtered, pressure_vector_filtered, 'r')
    # plt.ylabel('pressure, sm. H2O')
    # plt.xlabel('time, sec')
    # plt.grid()
    # plt.show()


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
