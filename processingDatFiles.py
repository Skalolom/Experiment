import numpy as np
from scipy import io

path = r'/home/bathory/PycharmProjects/Experiment/Data/dat/r_2_sph.dat'

byteString = open(path, 'rb').readline()
pressureVector = byteString.decode('utf-16')
# print(encodedArray)




