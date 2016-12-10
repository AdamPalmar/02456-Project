import gym
import time
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D, Activation
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
filter_size = 2
image_size = 128
border = 'same'
timestamp = str(datetime.datetime.now())
path_to_save_image = "pictures_log/" + env_string + "/" + timestamp
path_to_folder_save_images = "pictures_log/" + env_string
H=image_size
L=image_size*4


f, axarr = plt.subplots(1, 3)
first_image = True

###     Defining the network     ###
activation = LeakyReLU(alpha=0.15)


autoencoder_model = Sequential()
autoencoder_model.add(Convolution2D(16, filter_size, filter_size, border_mode=border, batch_input_shape=(1,H, H, 4)))
autoencoder_model.add(Activation(activation))
autoencoder_model.add(MaxPooling2D((2, 2), name='encoded_latent_state'))
autoencoder_model.add(Convolution2D(16, filter_size, filter_size, border_mode=border))
autoencoder_model.add(Activation(activation))
autoencoder_model.add(UpSampling2D((2, 2)))
autoencoder_model.add(Convolution2D(8, filter_size, filter_size, border_mode=border))
autoencoder_model.add(Activation(activation))

# ###     Network setup end         ###


opt_adam = Adam()
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error')

###summary = autoencoder_model.summary_to_txt()
autoencoder_model.summary()


#file_writer.write_model_conf_to_file(model_summary=summary, timestamp=path_to_save_image,
#                                     path_to_file=env_string+"-model_configs.txt",
#                                     path_to_folder=path_to_folder_save_images)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def reToG(obs, shape):
    rez = imresize(obs, size=(shape, shape, 3))
    rez = rgb2gray(rez)
    rez = np.reshape(rez, (shape, shape))
    return rez


def negToZero(diff):
    low_values_indices = diff <= 0  # Where values are low
    high_values_indices = diff > 0  # Where values are high
    diff[low_values_indices] = 0  # All low values set to 0
    diff[high_values_indices] = 255
    return diff

def addDiff(obs, shape):
    s=shape**2
    scale=0.02
    obs1=obs[range(s)]
    obs2=obs[range(s,2*s)]
    obs3=obs[range(2*s,3*s)]
    diff = obs3-obs2-obs1
    diff=negToZero(diff)
    obss= np.zeros(shape=(shape, shape, 4))
    obss[:,:,0]=obs1.reshape((shape,shape))*scale
    obss[:,:,1]=obs2.reshape((shape,shape))*scale
    obss[:,:,2]=obs3.reshape((shape,shape))*scale
    obss[:,:,3]=diff.reshape((shape,shape))
    diff=negToZero(diff)
    plt.title("Cur/diff/pred")
    axarr[0].imshow(obs1.reshape((shape,shape)), cmap='Greys_r')
    axarr[1].imshow(obs3.reshape((shape,shape)), cmap='Greys_r')
    axarr[2].imshow(diff.reshape((shape,shape)), cmap='Greys_r')
    plt.savefig("tmp.png")
    return obss


action=env.action_space.sample()

while True:
    env.render()

    observation, reward, done, info = env.step(action)  # take a random action
    observation = reToG(observation, image_size)
    for x in range(2):
        action=env.action_space.sample()
        obs, _, _, _ = env.step(action)  # take a random action
        obs = reToG(obs, image_size)
        observation = np.append(observation,obs)
        print observation.shape

    action = env.action_space.sample() # future action will be reused as first action
    futObs, _, _, _ = env.step(action)  # take a random action
    futObs = reToG(futObs, image_size)
    #observation = np.append(observation, action) # add future action to vec
    score += reward
    counter += 1
    observation=addDiff(observation,image_size)
    print observation.shape
    #observation=observation.reshape(128,128,3)
    # Init image arrays
    resized_img = np.zeros(shape=( 1, H, H, 4))
    fut_img = np.zeros(shape=( 1, 128, 128, 1))
        #This is to make sure that the input has the correct size
    resized_img[0] = observation.reshape((H, H, 4))
    fut_img[0] = futObs.reshape((128, 128, 1))
    autoencoder_model.fit(resized_img,
                          fut_img,
                          batch_size=1,
                          nb_epoch=1,
                          shuffle=False)

    prediction_img = autoencoder_model.predict(resized_img)
    print prediction_img.shape
    prediction_resized = prediction_img.reshape((image_size, image_size))
    col= np.zeros((128,128,3), 'uint8')
    col[..., 0] = prediction_resized*-1
    col[..., 1] = futObs*-1
    col[..., 2] = 0
    plt.title("Cur/diff/pred")
    axarr[0].imshow(futObs, cmap='Greys_r')
    axarr[1].imshow(prediction_resized, cmap='Greys_r')
    axarr[2].imshow(col)

    plt.savefig(path_to_save_image + ".png")

    if done:
        print done, score
        score = 0
        env.reset()
        counter = 0
