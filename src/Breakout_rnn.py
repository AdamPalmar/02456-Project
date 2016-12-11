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
filter_size = 5
image_size = 128
border = 'same'
timestamp = str(datetime.datetime.now())
path_to_save_image = "pictures_log/" + env_string + "/" + timestamp
path_to_folder_save_images = "pictures_log/" + env_string



f, axarr = plt.subplots(1, 3)
last_frame = np.empty(shape=( 1, image_size, image_size,1))
first_image = True

###     Defining the network     ###
activation = LeakyReLU(alpha=0.15)


input_img_observation = Input(shape=(image_size, image_size, 1))

encoder = Convolution2D(8, filter_size, filter_size, activation=activation, border_mode=border)(input_img_observation)
encoder = MaxPooling2D((4, 4), border_mode=border)(encoder)
encoder = Convolution2D(8, filter_size, filter_size, activation=activation,border_mode=border)(encoder)

encoded_state = MaxPooling2D((4, 4), border_mode=border, name='encoded_latent_state')(encoder)

decoder = Convolution2D(8, filter_size, filter_size, activation=activation,border_mode=border)(encoded_state)
decoder = UpSampling2D((4, 4))(decoder)
decoder = Convolution2D(8, filter_size, filter_size, activation=activation, border_mode=border)(decoder)
decoder = UpSampling2D((4, 4))(decoder)
#decoder = Convolution2D(8, filter_size, filter_size, activation=activation, border_mode=border)(decoder)

output_layer = Convolution2D(1, 7, 7, activation=activation, border_mode=border)(decoder)

###     Network setup end         ###



autoencoder_model = Model(input=input_img_observation, output=output_layer)


opt_adam = Adam()
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error')

###summary = autoencoder_model.summary_to_txt()
autoencoder_model.summary()


#file_writer.write_model_conf_to_file(model_summary=summary, timestamp=path_to_save_image,
#                                     path_to_file=env_string+"-model_configs.txt",
#                                     path_to_folder=path_to_folder_save_images)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])



for k in range(10):
    #env.render()

    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    score += reward
    counter += 1
    # Resizeing image
    observation = imresize(observation, size=(image_size, image_size, 3))

    if not first_image:

        # Init image arrays
        resized_img = np.zeros(shape=( 1, image_size, image_size,1))
        diff_image = np.zeros(shape=( 1, image_size, image_size,1))
        # Converting to greyscale
        current_image_grey = rgb2gray(observation)
        # Resizing the image
        current_image_grey = np.reshape(current_image_grey, (image_size, image_size))
        # Finding difference between current frame and last frame
        # Use if running diff
        diff_image = current_image_grey-last_frame
        # resized_img[0] = diff_image.reshape((1, image_size, image_size))
        # This is the improvement

        low_values_indices = diff_image <= 0  # Where values are low
        high_values_indices = diff_image > 0  # Where values are high
        diff_image[low_values_indices] = 0  # All low values set to 0
        diff_image[high_values_indices] = 255

        current_plus_diff = np.empty(shape=(image_size,image_size,1))
        current_plus_diff = current_image_grey * 0.4 + diff_image

        #This is to make sure that the input has the correct size
        resized_img[0] = current_plus_diff.reshape((image_size, image_size,1))

        autoencoder_model.fit(resized_img, resized_img,
                              batch_size=1,
                              nb_epoch=1,
                              shuffle=False)

        prediction_img = autoencoder_model.predict(resized_img)
        prediction_resized = prediction_img.reshape((image_size, image_size))

        if counter % 2:
            last_frame = current_image_grey

        plt.title("Cur/diff/pred")
        axarr[0].imshow(current_image_grey*0.8, cmap='Greys_r')
        axarr[1].imshow(diff_image, cmap='Greys_r')
        axarr[2].imshow(prediction_resized, cmap='Greys_r')

        #plt.savefig(path_to_save_image + ".png")

    else:
        last_frame = rgb2gray(observation)
        last_frame = imresize(last_frame, size=(image_size, image_size))
        first_image = False

    if done:
        print done, score
        score = 0
        env.reset()
        counter = 0

#print len(autoencoder_model.layers[1].get_weights())
#impl rnn
encoder2 = Convolution2D(8, filter_size, filter_size, activation=activation, border_mode=border, weights=autoencoder_model.layers[1].get_weights())(input_img_observation)
encoder2 = MaxPooling2D((4, 4), border_mode=border)(encoder)
encoder2 = Convolution2D(8, filter_size, filter_size, activation=activation,border_mode=border, weights=autoencoder_model.layers[3].get_weights())(encoder)
encoded_state2 = MaxPooling2D((4, 4), border_mode=border)(encoder)

encoded_model = Model(input=input_img_observation, output=encoded_state2)
encoded_model.compile(optimizer=opt_adam, loss='mean_squared_error')


N = 1000
enc_size = 8

#rnn_obs=np.empty(shape=(N,image_size,image_size))
encoded_obs = np.empty(shape=(N,1,512))
for i in range(N):
    #env.render()
    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    observation = imresize(observation, size=(image_size, image_size, 3))
    # Resizeing image
    # Init image arrays
    resized_img = np.zeros(shape=( 1, image_size, image_size, 1))
    # Converting to greyscale
    current_image_grey = rgb2gray(observation)
    # Resizing the image
    #prediction_img = autoencoder_model.predict(resized_img)
    #prediction_resized = prediction_img.reshape((image_size, image_size))

    img_resized = np.reshape(current_image_grey, (1,128,128,1))
    encoded_img = encoded_model.predict(img_resized)
    encoded_img_flatten = np.reshape(encoded_img, (1,512))
    encoded_obs[i] = encoded_img_flatten

    if done:
        print done, score
        print encoded_obs[0].shape
        env.reset()


rnn_model = Sequential()
rnn_model.add( LSTM(512, input_dim=512,input_length=1) )

rnn_model.compile(optimizer=opt_adam, loss='mean_squared_error')
rnn_model.summary()

## CREATE DATASET
#x_train = np.empty(shape=(1,512))
#y_train = np.empty(shape=(1,512))
x_train = []
y_train = []
for i in range( len(encoded_obs)-1 ):
    x_train.append(encoded_obs[i])
    y_train.append(encoded_obs[i+1])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

rnn_model.fit(x_train, y_train, nb_epoch=100, batch_size=1)
