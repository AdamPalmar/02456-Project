import gym
import time
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt




env = gym.make('Breakout-v0')
env.reset()
score = 0
counter = 0
print(env.action_space)
print(env.observation_space)
filter_size = 5
border = 'same'
input_img_observation = Input(shape=(3, 210, 160))

encoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(input_img_observation)
encoded_state = MaxPooling2D((2, 2), border_mode=border, name='encoded_latent_state')(encoder)

decoder = Convolution2D(8, filter_size, filter_size, activation='relu', border_mode=border)(encoded_state)
decoder = UpSampling2D((2, 2))(decoder)

output_layer = Convolution2D(3, 5, 5, activation='relu', border_mode=border)(decoder)

autoencoder_model = Model(input=input_img_observation, output=output_layer)

opt_adam = Adam()
autoencoder_model.compile(optimizer=opt_adam, loss='mean_squared_error')

print(autoencoder_model.input_shape, "Input shape of model")
print(autoencoder_model.output_shape, "Output shape of model")

autoencoder_model.summary()

plt.figure(figsize=(1,1))


while True:
    env.render()

    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    score += reward
    counter += 1
    # print(observation.shape)
    # print(abs(observation[1]))
    # print("Reward",reward)
    # print(counter)
    # print(observation)
    resized_img = np.empty(shape=(1, 3, 210, 160))
    resized_img[0] = observation.reshape((3, 210, 160))
    # print(resized_img[0])
    # print(resized_img.shape)
    # print(resized_img.shape)
    autoencoder_model.fit(resized_img, resized_img,
                          batch_size=1,
                          nb_epoch=1,
                          shuffle=False)

    prediction_img = autoencoder_model.predict(resized_img)

    prediction_resized = prediction_img[0].reshape((210, 160, 3))
    plt.imshow(prediction_resized)
    plt.savefig("breakout-8-8   .png")


    if done:
        print done, score
        score = 0
        env.reset()
        counter = 0
