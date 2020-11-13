import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
import time
from scipy import signal
from scipy.fft import fft
from scipy.signal import savgol_filter, decimate
from scipy.ndimage import gaussian_filter1d
import signal_processing as sp
import json
import os

start = time.time()
# read lines in file and search for 'kHz' in line
sub_freq, sub_time, index, freq_rate, full_time = 'kHz', 'Sec', 0, 1, 1
time_vector = press_vector = []
bar_colors = []
bars = []
names = []
press_min, press_max = 5, 45
r_0, r_current, i, r_previous = 5, 5, 0, 0

path, log_path = r'Data/5mm', r'Data/log.txt'
log_file = open(log_path, 'a')
# write type of experiment
experiment_name = path.split(r'/')[-1] + '\n'
log_file.write(experiment_name)

files, path, log_path = [], path, log_path
# считаем конфигурационный файл config.json в аттрибут класса config_data
with open('config.json', 'r') as read_json:
    config_data = json.load(read_json)

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file or '.dat' in file:
            files.append(os.path.join(r, file))

files.sort()

for filename in files:
    press_vector = []
    time_vector = []

    # if file extension is .dat
    if filename.find('.dat') != -1:
        print(filename)
        for line in open(filename, 'r'):
            press_vector.append(line)
    else:
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
                tempString = [float(s) for s in line.split()]
                # if tempString has 2 elements
                if len(tempString) == 2:
                    press_vector.append(tempString[1])
                    time_vector.append(tempString[0])
            except Exception:
                continue

    time_vector = [t/(time_vector[-1]/full_time) for t in time_vector]  # eval real time_vector based on frequency

    # normalize time_vector to compensate radius deviation
    if filename.find('c_sh') != -1:
        r_current = 9.886 / 2
        if filename.find('J') != -1:
            bar_color = 'purple'
        else:
            bar_color = 'black'
    if filename.find('c_sm') != -1:
        r_current = 9.875 / 2
        if filename.find('J') != -1:
            bar_color = 'purple'
        else:
            bar_color = 'magenta'
    if filename.find('r_1') != -1:
        r_current = 10.039 / 2
        if filename.find('J') != -1:
            bar_color = 'purple'
        else:
            bar_color = 'blue'
    if filename.find('r_2') != -1:
        r_current = 10.347 / 2
        if filename.find('J') != -1:
            bar_color = 'purple'
        else:
            bar_color = 'green'
    if filename.find('r_3') != -1:
        r_current = 10.260 / 2
        if filename.find('J') != -1:
            bar_color = 'purple'
        else:
            bar_color = 'red'

    # if name of the experiment has changed, add whitespace between bars
    if r_previous != r_current:
        bars.append(0)
        bar_colors.append(bar_color)
        # parse the filename for name of the group of experiments (for example, c_sh, c_sm)
        experimentGroupName = filename.split('/')[-1].split('.')[0].split('_')[0:2]
        names.append(experimentGroupName[0] + '_' + experimentGroupName[1])
    # save value of the r_current in r_previous
    r_previous = r_current
    k = r_current ** 2 / r_0 ** 2

    # time_vector = [t / 1000 for t in time_vector]
    time_vector = [t * k for t in time_vector]

    time_vector_flt, press_vector_flt = sp.approximate_pressure_with_polynomial(time_vector=time_vector,
                                                                                pressure_vector=press_vector,
                                                                                filter_window_factor=50,
                                                                                polynomial_degree=2,
                                                                                pressure_min=press_min,
                                                                                pressure_max=press_max)

    timeStart = time_vector_flt[0]
    timeEnd = time_vector_flt[-1]
    expTime = timeEnd - timeStart

    # вычисляем расход
    dt, dh = np.abs(np.diff(time_vector_flt)), np.abs(np.diff(press_vector_flt))
    dt = np.append(dt, dt[-1])
    dh = np.append(dh, dh[-1])
    consumption = (np.pi*(10e-2*r_current)**2) * (dh/dt)

    # write expTime and name of the graph
    bars.append(expTime)
    bar_colors.append(bar_color)
    current_name = filename.split(r'/')[-1].split('.')[0].split('_')[2]
    names.append(current_name)
    i += 1
    print('{0:.1f}% progress...'.format(100 * (i / len(files))), end=' ')
    log_file.write(names[-1] + ' : ' + str(expTime) + '\n')
    figure_label = filename.split('/')[-1].split('.')[0]
    # вычисляем спектр сигнала
    # plt.figure(i)
    # plt.title(figure_label)
    # plt.plot(time_vector, press_vector, 'b', time_vector_flt, press_vector_flt, 'r')

    # выводим на одном графике зависимости p(t)
    # plt.plot(time_vector_flt, press_vector_flt, bar_color,
    #          label=figure_label)

    # выводим расход от времени
    plt.plot(press_vector_flt, consumption, bar_color, label=figure_label)
    plt.xlabel('pressure, sm h2o')
    plt.ylabel('c, m3/sec')


log_file.close()

font = {'family': 'normal', 'weight': 'bold', 'size': 20}

plt.rcParams.update({'font.size': 22})
plt.figure(1)
# spaceBars = np.linspace(0, 0.6 * len(bars), len(bars))
# plt.bar(spaceBars, bars, width=0.6, color=bar_colors)
# plt.xticks(spaceBars, names)
# plt.tight_layout()
# plt.ylim([0.7*max(bars), max(bars) + 10])
# plt.ylabel('time, s')
# plt.grid()
# plt.show()
# i = 0
# for bar in bars:
#     if bar != 0:
#         plt.annotate(int(bar), xy=(spaceBars[i] - 0.2, bar + 1))
#     i += 1

end = time.time()
print('evaluation time = {0:.2f} sec'.format(end - start))

plt.title(experiment_name)
plt.legend(loc='upper right')
plt.grid()
plt.show()

"""

"""
