import gym
import time
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

# Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('gym/cartpole-0')
env.reset()
score = 0
counter = 0
print(env.action_space)
print(env.observation_space)
filter_size = 2
border = 'same'
input_img_observation = Input(shape=(3, 480, 640))

encoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(input_img_observation)
encoded_state = MaxPooling2D((2, 2), border_mode=border, name='encoded_latent_state')(encoder)

decoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(encoded_state)
decoder = UpSampling2D((2, 2))(decoder)

output_layer = Convolution2D(3, filter_size, filter_size, activation='relu', border_mode=border)(decoder)

autoencoder_model = Model(input=input_img_observation, output=output_layer)

opt_adam = Adam()
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error')

print(autoencoder_model.input_shape, "Input shape of model")
print(autoencoder_model.output_shape, "Output shape of model")

autoencoder_model.summary()

plt.figure(figsize=(1, 1))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

first = True
while True:
    env.render()
    if not first:
        o2=o1

    else:
        o2=np.zeros((480,640,3))
    o1, reward, done, info = env.step(env.action_space.sample())  # take a random action
    score += reward
    counter += 1
    first=False

    o=abs(o1-o2)
   # low_values_indices = o < 0  # Where values are low
   # high_values_indices = o > 0  # Where values are low
   # o[low_values_indices] = 0  # All low values set to 0
   # o[high_values_indices] = 10000
   # print(o)
    observation=o.resize(3,480,640)
    print(o1.shape)
    resized_img = np.empty(shape=(1, 3, 480, 640))
    #print(resized_img.shape)
    #grey = rgb2gray(observation)
    resized_img[0] = observation
    #plt.imshow(grey, cmap='Greys_r')
    #plt.savefig("breakout-greyscale-diff.png")
    print(resized_img.shape)

    autoencoder_model.fit(resized_img, resized_img,
                          batch_size=1,
                          nb_epoch=1,
                          shuffle=False)

    prediction_img = autoencoder_model.predict(resized_img)

    # print(prediction_img.shape)
    prediction_resized = prediction_img[0][0].reshape((480, 640))
    print(prediction_resized.shape)
    # prediction_resized = normalize(prediction_resized, axis=1, norm='l1')
    # print(prediction_resized.max(),"Max value")
    # print(prediction_resized.min(), "min value")

    plt.imshow(prediction_resized, cmap='Greys_r')
    plt.savefig("breakout-greyscale.png")

    if done:
        print done, score
        score = 0
        env.reset()
        counter = 0
