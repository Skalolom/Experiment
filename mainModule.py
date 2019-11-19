import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
import time
import os
from scipy import signal

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
pressMin = 5
pressMax = 45
# define r0
r0 = 5
rCurrent = r0
i = 0
radiusPrevious = 0

path = r'/home/bathory/PycharmProjects/Experiment/Data/8mm/'
# define output file
logPath = r'/home/bathory/PycharmProjects/Experiment/Data/log.txt'
logFile = open(logPath, 'a')
# write type of experiment
logFile.write(path.split(r'/')[-2] + '\n')

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

files.sort()

for filename in files:
    pressVect = []
    timeVect = []

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

    timeVect = [t/(timeVect[-1]/fullTime) for t in timeVect]  # eval real timeVect based on frequency

    # normalize timeVect to compensate radius deviation
    if filename.find('c_sh') != -1:
        rCurrent = 9.886 / 2
    if filename.find('c_sm') != -1:
        rCurrent = 9.875 / 2
    if filename.find('r_1') != -1:
        rCurrent = 10.039 / 2
    if filename.find('r_2') != -1:
        rCurrent = 10.347 / 2
    if filename.find('r_3') != -1:
        rCurrent = 10.260 / 2

    # if name of the experiment has changed, add whitespace between bars
    if radiusPrevious != rCurrent:
        bars.append(0)

        # parse the filename for name of the group of experiments (for example, c_sh, c_sm)
        experimentGroupName = filename.split('/')[-1].split('.')[0].split('_')[0:2]
        names.append(experimentGroupName[0] + '_' + experimentGroupName[1])
    # save value of the rCurrent in radiusPrevious
    radiusPrevious = rCurrent
    k = rCurrent ** 2 / r0 ** 2

    # timeVect = [t / 1000 for t in timeVect]
    timeVect = [t * k for t in timeVect]

    # making the low-pass filter
    b, a = signal.butter(8, 0.01)
    pressVect_flt = list(signal.filtfilt(b, a, pressVect))

    # eval polynomial coeffs
    polyCoeff = polyfit(timeVect, pressVect_flt, 5)
    pressVect_poly = polyval(timeVect, polyCoeff)

    # evaluate array of difference between filtered pressVect and pressMax|pressMin
    maxIndex = np.argmin([abs(d - pressMax) for d in pressVect_poly])
    minIndex = np.argmin([abs(d - pressMin) for d in pressVect_poly])

    timeStart = timeVect[maxIndex]
    timeEnd = timeVect[minIndex]
    expTime = timeEnd - timeStart

    # write expTime and name of the graph
    bars.append(expTime)
    names.append(filename.split(r'/')[-1][0:-4])
    # names.append(filename.split(r'/')[-1].split('.')[0].split('_')[2])
    i += 1
    print('{0:.1f}% progress...'.format(100 * (i / len(files))), end=' ')
    logFile.write(names[-1] + ' : ' + str(expTime) + '\n')

"""
plt.figure(1)
plt.plot(timeVect, pressVect, 'b', timeVect, pressVect_poly, 'r')
plt.xlabel('time')
plt.ylabel('pressure')

"""
font = {'family': 'normal', 'weight': 'bold', 'size': 12}

plt.rcParams.update({'font.size': 9})
plt.figure(1)
# plt.bar(names, bars, width=0.6)
# define space between bars
spaceBars = np.linspace(0, 0.7 * len(bars), len(bars))
plt.bar(spaceBars, bars, width=0.6)
plt.xticks(spaceBars, names)
plt.tight_layout()

i = 0
for bar in bars:
    if bar != 0:
        plt.annotate(int(bar), xy=(spaceBars[i], bar + 1))
    i += 1

plt.show()
logFile.close()
end = time.time()
print('evaluation time = {0:.2f} sec'.format(end - start))
