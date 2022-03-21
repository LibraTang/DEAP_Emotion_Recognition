import numpy as np
import pyeeg as pe
import pickle as pickle
import pandas as pd
import math

from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

import os
import time

channel = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]  # 14 Channels chosen to fit Emotive Epoch+
band = [4, 8, 12, 16, 25, 45]  # 5 bands
window_size = 256  # Averaging band power of 2 sec
step_size = 16  # Each 0.125 sec update once
sample_rate = 128  # Sampling rate of 128 Hz
subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']  # List of subjects


