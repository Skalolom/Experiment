import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
import time
import os
from scipy import signal
import os
from scipy.fft import fft
from scipy.signal import savgol_filter
dn = os.path.dirname(os.path.realpath(__file__))

start = time.time()
# read lines in file and search for 'kHz' in line
subFreq = 'kHz'
subTime = 'Sec'
index = 0
freqRate = 0
fullTime = 0
dataIndex = -1
timeVect = []
pressVect = []
names = []
bars = []
barColors = []
pressMin = 5
pressMax = 45
# define r0
r0 = 5
rCurrent = r0
i = 0
radiusPrevious = 0

#path = r'/home/bathory/PycharmProjects/Experiment/Data/spiral/5mm/'
path = r'Data/5mm'
# define output file
#logPath = r'/home/bathory/PycharmProjects/Experiment/Data/log.txt'
logPath = r'Data/log.txt'
logFile = open(logPath, 'a')
# write type of experiment
logFile.write(path.split(r'/')[-2] + '\n')

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file or '.dat' in file:
            files.append(os.path.join(r, file))

files.sort()


# define function for segmenting input vectors according to min and max values. It returns min and max indexes
def segment_vector(vector, min_value, max_value):
    max_index = np.argmin([abs(e - max_value) for e in vector])
    min_index = np.argmin([abs(e - min_value) for e in vector])
    return min_index, max_index


for filename in files:
    pressVect = []
    timeVect = []

    # if file extension is .dat
    if filename.find('.dat') != -1:
        print(filename)
        for line in open(filename, 'r'):
            pressVect.append(line)
    else:
        # fill time and pressure vect
        for line in open(filename, 'r'):

            index = line.find(subFreq)
            if index != -1:
                # read rate
                freqRate = float(line[index + 5:]) * 1e3

            index = line.find(subTime)
            if index != -1:
                # read time
                fullTime = float(line[index + 5:])
                # if line contains only digits
            try:
                tempString = [float(s) for s in line.split()]
                # if tempString has 2 elements
                if len(tempString) == 2:
                    pressVect.append(tempString[1])
                    timeVect.append(tempString[0])
            except Exception:
                continue

    # if max(pressureVect) is above 60, we need to calibrate its values

    timeVect = [t/(timeVect[-1]/fullTime) for t in timeVect]  # eval real timeVect based on frequency

    # normalize timeVect to compensate radius deviation
    if filename.find('c_sh') != -1:
        rCurrent = 9.886 / 2
        barColor = 'black'
    if filename.find('c_sm') != -1:
        rCurrent = 9.875 / 2
        barColor = 'magenta'
    if filename.find('r_1') != -1:
        rCurrent = 10.039 / 2
        barColor = 'blue'
    if filename.find('r_2') != -1:
        rCurrent = 10.347 / 2
        barColor = 'green'
    if filename.find('r_3') != -1:
        rCurrent = 10.260 / 2
        barColor = 'red'

    # if name of the experiment has changed, add whitespace between bars
    if radiusPrevious != rCurrent:
        bars.append(0)
        barColors.append(barColor)
        # parse the filename for name of the group of experiments (for example, c_sh, c_sm)
        experimentGroupName = filename.split('/')[-1].split('.')[0].split('_')[0:2]
        names.append(experimentGroupName[0] + '_' + experimentGroupName[1])
    # save value of the rCurrent in radiusPrevious
    radiusPrevious = rCurrent
    k = rCurrent ** 2 / r0 ** 2

    # timeVect = [t / 1000 for t in timeVect]
    timeVect = [t * k for t in timeVect]

    # making the low-pass filter
    # b, a = signal.butter(8, 0.01)
    b, a = signal.butter(5, 0.001)
    pressVect_flt = list(signal.filtfilt(b, a, pressVect))
    pressVect_flt = savgol_filter(pressVect_flt,
                                  2501, 5)
    # save current values of pressureMax and pressureMin for calibration
    calibrationPressMax = max(pressVect_flt)
    calibrationPressMin = min(pressVect_flt)

    # segmenting filtered vector from pressure 2 to 48
    minIndex, maxIndex = segment_vector(pressVect_flt, pressMin, pressMax)
    pressVect_flt = pressVect_flt[maxIndex:minIndex]
    timeVect_flt = timeVect[maxIndex:minIndex]

    # eval polynomial coeffs
    #polyCoeff = polyfit(timeVect_flt, pressVect_flt, 5)
    #pressVect_poly = polyval(timeVect_flt, polyCoeff)

    # evaluate indexes of pressure = 45 and 5
    minIndex, maxIndex = segment_vector(pressVect_flt, pressMin, pressMax)

    # evaluate array of difference between filtered pressVect and pressMax|pressMin
    # maxIndex = np.argmin([abs(d - pressMax) for d in pressVect_poly])
    # minIndex = np.argmin([abs(d - pressMin) for d in pressVect_poly])

    timeStart = timeVect[maxIndex]
    timeEnd = timeVect[minIndex]
    expTime = timeEnd - timeStart

    # write expTime and name of the graph
    bars.append(expTime)
    barColors.append(barColor)
    # names.append(filename.split(r'/')[-1][0:-4])
    names.append(filename.split(r'/')[-1].split('.')[0].split('_')[2])
    i += 1
    print('{0:.1f}% progress...'.format(100 * (i / len(files))), end=' ')
    # logFile.write(names[-1] + ' : ' + str(expTime) + '\n')

    # # вычисляем спектр сигнала
    # plt.figure(i)
    # plt.title('Raw signal')
    # plt.plot(timeVect, pressVect, 'b', timeVect_flt, pressVect_flt, 'r')


logFile.close()

font = {'family': 'normal', 'weight': 'bold', 'size': 20}

plt.rcParams.update({'font.size': 22})
plt.figure(1)
# plt.bar(names, bars, width=0.6)
# define space between bars
spaceBars = np.linspace(0, 0.6 * len(bars), len(bars))
plt.bar(spaceBars, bars, width=0.6, color=barColors)
plt.xticks(spaceBars, names)
plt.tight_layout()
plt.ylim([550, max(bars) + 10])

i = 0
for bar in bars:
    if bar != 0:
        plt.annotate(int(bar), xy=(spaceBars[i], bar + 1))
    i += 1

end = time.time()
print('evaluation time = {0:.2f} sec'.format(end - start))

# plt.figure(1)
# plt.plot(timeVect, pressVect, 'b', timeVect_flt, pressVect_poly, 'r')
# plt.xlabel('time')
# plt.ylabel('pressure')

plt.show()

"""

"""
