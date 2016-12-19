import gym
import time
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import datetime
import file_writer

env_string = 'SpaceInvaders-v0'
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
last_frame = np.empty(shape=(1, image_size, image_size, 1))
first_image = True

train_history = list()
test_history = list()

num_epochs = 1000
resized_img = np.zeros(shape=(num_epochs, image_size, image_size, 1))

###     Defining the network     ###
activation = LeakyReLU(alpha=0.15)

input_img_observation = Input(shape=(image_size, image_size, 1))

encoder = Convolution2D(32, 7, 7, activation=activation, border_mode=border)(input_img_observation)
encoder = MaxPooling2D((4, 4), border_mode=border)(encoder)
encoder = Convolution2D(8, filter_size, filter_size, activation=activation, border_mode=border)(encoder)
# encoder = MaxPooling2D((2, 2), border_mode=border)(encoder)
# encoder = Convolution2D(8, filter_size, filter_size, activation=activation,border_mode=border)(encoder)


encoded_state = MaxPooling2D((2, 2), border_mode=border, name='encoded_latent_state')(encoder)
#
# decoder = Convolution2D(8, filter_size, filter_size, activation=activation,border_mode=border)(encoded_state)
# decoder = UpSampling2D((2, 2))(decoder)
decoder = Convolution2D(8, filter_size, filter_size, activation=activation, border_mode=border)(encoded_state)
decoder = UpSampling2D((4, 4))(decoder)
decoder = Convolution2D(32, filter_size, filter_size, activation=activation, border_mode=border)(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Convolution2D(64, 7, 7, activation=activation, border_mode=border)(decoder)

output_layer = Convolution2D(1, 7, 7, activation=activation, border_mode=border)(decoder)

###     Network setup end         ###



autoencoder_model = Model(input=input_img_observation, output=output_layer)

opt_adam = Adam(lr=0.00003)
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error',metrics=['accuracy'])

###summary = autoencoder_model.summary_to_txt()
autoencoder_model.summary()


# file_writer.write_model_conf_to_file(model_summary=summary, timestamp=path_to_save_image,
#                                     path_to_file=env_string+"-model_configs.txt",
#                                     path_to_folder=path_to_folder_save_images)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

image_counter = 0

for _ in range(100):
    print("epoch num", _)
    for i in range(num_epochs):
        # env.render()

        observation, reward, done, info = env.step(env.action_space.sample())  # take a random action



        score += reward
        counter += 1
        # Resizeing image


        observation = imresize(observation, size=(image_size, image_size, 3))
        # env.render()

        if not first_image:

            # Init image arrays
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
            current_plus_diff = current_image_grey * 0.75 + diff_image

            # This is to make sure that the input has the correct size
            resized_img[image_counter] = current_plus_diff.reshape((image_size, image_size, 1))
            image_counter += 1

            if counter % 2:
                last_frame = current_image_grey




        else:
            last_frame = rgb2gray(observation)
            last_frame = imresize(last_frame, size=(image_size, image_size))
            first_image = False

        if done:
            print(done, score)
            score = 0
            env.reset()
            counter = 0

    history = autoencoder_model.fit(resized_img, resized_img,
                                    validation_split=0.5,
                                    batch_size=1,
                                    nb_epoch=1,
                                    shuffle=True)

    # summarize history for loss
    train_history.append(history.history['loss'])
    test_history.append(history.history['val_loss'])
    image_counter = 0

    prediction_img = autoencoder_model.predict(resized_img)
    prediction_resized = prediction_img[-1].reshape((image_size, image_size))


    plt.title("Cur/diff/pred")
    axarr[0].imshow(current_image_grey * 0.75, cmap='Greys_r')
    axarr[1].imshow(diff_image, cmap='Greys_r')
    axarr[2].imshow(prediction_resized, cmap='Greys_r')

    plt.savefig(path_to_save_image + ".png")


    resized_img = np.zeros(shape=(num_epochs, image_size, image_size, 1))






plt.figure()
plt.plot(train_history)
plt.plot(test_history)
plt.title('Training loss vs validation loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
