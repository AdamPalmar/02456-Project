import gym
import time
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D, LSTM
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize
from sklearn.preprocessing import normalize
import datetime
import file_writer

env_string = 'Breakout-v0'
env = gym.make(env_string)
env.reset()

# Environment variables#
score = 0
counter = 0
filter_size = 4
image_size = 128

###model
model = Sequential()
model.add(LSTM(200,input_shape=(3,7000)))

model.summary()
