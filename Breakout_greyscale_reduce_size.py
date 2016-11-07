import gym
import time
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize
from sklearn.preprocessing import normalize

env = gym.make('Breakout-v0')
env.reset()
score = 0
counter = 0
print(env.action_space)
print(env.observation_space)
filter_size = 5
image_size = 256
border = 'same'

###     Defining the network     ###
input_img_observation = Input(shape=(1, image_size, image_size))

encoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(input_img_observation)
# encoder = MaxPooling2D((2, 2), border_mode=border)(encoder)
# encoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(encoder)

encoded_state = MaxPooling2D((2, 2), border_mode=border, name='encoded_latent_state')(encoder)

decoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(encoded_state)
decoder = UpSampling2D((2, 2))(decoder)

# decoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(decoder)
# decoder = UpSampling2D((2, 2))(decoder)


output_layer = Convolution2D(1, 5, 5, activation='relu', border_mode=border)(decoder)
###     Network setup end         ###



autoencoder_model = Model(input=input_img_observation, output=output_layer)

opt_adam = Adam()
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error')

print(autoencoder_model.input_shape, "Input shape of model")
print(autoencoder_model.output_shape, "Output shape of model")

autoencoder_model.summary()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


f, axarr = plt.subplots(1, 3)
last_frame = np.empty(shape=(1, 1, image_size, image_size))
first_image = True

while True:
    env.render()

    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    score += reward
    counter += 1
    # Resizeing image
    observation = imresize(observation, size=(image_size, image_size, 3))

    if not first_image:

        resized_img = np.empty(shape=(1, 1, image_size, image_size))
        diff_image = np.empty(shape=(1, 1, image_size, image_size))
        current_image_grey = rgb2gray(observation)

        current_resized = imresize(current_image_grey, size=(image_size, image_size))

        diff_image = (current_image_grey - last_frame)
        resized_img[0] = diff_image.reshape((1, image_size, image_size))

        autoencoder_model.fit(resized_img, resized_img,
                              batch_size=1,
                              nb_epoch=1,
                              shuffle=False)

        prediction_img = autoencoder_model.predict(resized_img)
        prediction_resized = prediction_img[0][0].reshape((image_size, image_size))

        if counter % 2:
            last_frame = current_image_grey

        axarr[0].imshow(current_image_grey, cmap='Greys_r')
        axarr[1].imshow(prediction_resized, cmap='Greys_r')
        axarr[2].imshow(current_resized, cmap='Greys_r')

        plt.savefig("breakout-greyscale-difference.png")

    else:
        last_frame = rgb2gray(observation)
        first_image = False

    if done:
        print done, score
        score = 0
        env.reset()
        counter = 0
