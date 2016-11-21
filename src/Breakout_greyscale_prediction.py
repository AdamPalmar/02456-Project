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
import datetime
import file_writer

env_string = 'Breakout-v0'
env = gym.make(env_string)
env.reset()

# Environment variables#
score = 0
counter = 0
frame_counter = 0
filter_size = 5
image_size = 128

frame_stack_size = 8
num_frames_to_predict_future = 2
train_stack_size = 3
border = 'same'
timestamp = str(datetime.datetime.now())
path_to_save_image = "pictures_log/" + env_string + "/" + timestamp
path_to_folder_save_images = "pictures_log/" + env_string

f, axarr = plt.subplots(1, 4)
last_frame = np.empty(shape=(1, image_size, image_size, 1))
three_frame_stack = np.empty(shape=(1, image_size, image_size, frame_stack_size))
first_image = True
# -------------------------#



###     Defining the network     ###
activation = LeakyReLU(alpha=0.15)

# Input size is now 128*128*3
input_img_observation = Input(shape=(image_size, image_size, train_stack_size))

encoder = Convolution2D(32, filter_size, filter_size, activation=activation, border_mode=border)(input_img_observation)
# encoder = MaxPooling2D((4, 4), border_mode=border)(encoder)
# encoder = Convolution2D(4, filter_size, filter_size, activation=activation, border_mode=border)(encoder)

encoded_state = MaxPooling2D((2, 2), border_mode=border, name='encoded_latent_state')(encoder)

# decoder = Convolution2D(16, filter_size, filter_size, activation=activation, border_mode=border)(encoded_state)
# decoder = UpSampling2D((4, 4))(decoder)
decoder = Convolution2D(32, filter_size, filter_size, activation=activation, border_mode=border)(encoded_state)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Convolution2D(16, filter_size, filter_size, activation=activation, border_mode=border)(decoder)

output_layer = Convolution2D(1, 7, 7, activation=activation, border_mode=border)(decoder)

###     Network setup end         ###



autoencoder_model = Model(input=input_img_observation, output=output_layer)

opt_adam = Adam()
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error')

###summary = autoencoder_model.summary_to_txt()
autoencoder_model.summary()


# file_writer.write_model_conf_to_file(model_summary=summary, timestamp=path_to_save_image,
#                                     path_to_file=env_string+"-model_configs.txt",
#                                     path_to_folder=path_to_folder_save_images)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Fire
env.step(1)
while True:
    env.render()

    # env.action_space.sample()
    observation, reward, done, info = env.step(1)  # take a random action
    score += reward
    counter += 1
    # Resizeing image
    observation = imresize(observation, size=(image_size, image_size, 3))

    if frame_counter == frame_stack_size + 1:

        # Init image arrays
        resized_img = np.zeros(shape=(1, image_size, image_size, 1))
        diff_image = np.zeros(shape=(1, image_size, image_size, 1))
        # Converting to greyscale
        current_image_grey = rgb2gray(observation)
        # Resizing the image
        current_image_grey = np.reshape(current_image_grey, (image_size, image_size))

        # Finding difference between current frame and last frame
        # Use if running diff
        diff_image = current_image_grey - last_frame
        # resized_img[0] = diff_image.reshape((1, image_size, image_size))
        # This is the improvement

        low_values_indices = diff_image <= 0  # Where values are low
        high_values_indices = diff_image > 0  # Where values are high
        diff_image[low_values_indices] = 0  # All low values set to 0
        diff_image[high_values_indices] = 255

        current_plus_diff = np.empty(shape=(image_size, image_size, 1))
        current_plus_diff = current_image_grey * 0.1 + diff_image

        # This is to make sure that the input has the correct size
        resized_img[0] = current_plus_diff.reshape((image_size, image_size, 1))

        #
        # train input 3 stacked frame
        # train output 1 frame future
        print(three_frame_stack[:, :, :, frame_stack_size - train_stack_size:frame_stack_size].shape)
        autoencoder_model.fit(three_frame_stack[:, :, :,
                              frame_stack_size - train_stack_size:frame_stack_size],
                              resized_img,
                              batch_size=1,
                              nb_epoch=1,
                              shuffle=False)

        three_frame_stack[:, :, :, :frame_stack_size - 1] = three_frame_stack[:, :, :, 1:frame_stack_size]
        three_frame_stack[:, :, :, frame_stack_size - 1] = np.reshape(current_image_grey, (1, image_size, image_size))
        # three_frame_stack[:, :, 2] = np.reshape(current_image_grey, (image_size, image_size, 1))

        prediction_img = autoencoder_model.predict(three_frame_stack[:, :, :,
                                                   frame_stack_size - train_stack_size:frame_stack_size])
        prediction_resized = prediction_img.reshape((image_size, image_size))

        if counter % 2:
            last_frame = current_image_grey

        plt.title("Cur/diff/pred")
        image_frame_in_past = np.reshape(three_frame_stack[:, :, :, frame_stack_size - train_stack_size], (image_size, image_size))
        axarr[0].imshow(image_frame_in_past, cmap='Greys_r')
        axarr[1].imshow(current_image_grey * 0.8, cmap='Greys_r')
        axarr[2].imshow(diff_image, cmap='Greys_r')
        axarr[3].imshow(prediction_resized , cmap='Greys_r')

        plt.savefig(path_to_save_image + ".png")

    elif frame_counter < frame_stack_size + 1:
        if frame_counter < frame_stack_size:
            last_frame = rgb2gray(observation)
            last_frame = np.reshape(last_frame, (image_size, image_size))
            three_frame_stack[:, :, :, frame_counter] = np.reshape(last_frame, (1, image_size, image_size))
            frame_counter += 1

        else:
            frame_counter += 1

    if done:
        print done, score
        score = 0
        env.reset()
        counter = 0
        frame_counter = 0
        # Fire
        env.step(1)
