import os
import numpy as np
import json


class SignalProcessing:
    def __init__(self, path, log_path):
        self.files, self.path, self.log_path = [], path, log_path
        self.log_file = open(log_path, 'a')
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
        sub_freq, sub_time, index, freq_rate, full_time = 'kHz', 'Sec', 0, 1, 1
        time_vector = press_vector = names = bars = bar_colors = []
        press_min, press_max = 5, 45
        r_0, r_current, i, r_previous = 5, 5, 0, 0

        experiment_name = self.path.split(r'/')[-1] + '\n'
        self.log_file.write(experiment_name)

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
