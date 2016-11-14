import gym
import time
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

env = gym.make('CartPole-v0')
env.reset()
score = 0
counter = 0
size = 13
print(env.action_space)
print(env.observation_space)

filter_size = 3

input_obs = Input(shape=(size,))

encoded = Dense(filter_size, activation='relu')(input_obs)

decoded = Dense(4, activation='linear')(encoded)

autoencoder_model = Model(input=input_obs, output=decoded)

encoder_model = Model(input=input_obs, output=encoded)

opt_adam = Adam()
autoencoder_model.compile(optimizer='adadelta', loss='mean_squared_error')

print(autoencoder_model.input_shape, "Input shape of model")
print(autoencoder_model.output_shape, "Output shape of model")

autoencoder_model.summary()


first = True
action = env.action_space.sample()

while True:
    env.render()
    o1, reward, done, info = env.step(action)  # take a random action
    action = env.action_space.sample()
    o2, _, _, _ = env.step(action)  # take a random action
    obs=o2-o1
    obs=np.append(np.append(obs,o1),o2)
    #for x in range(2):
    #    action = env.action_space.sample()
    #    o,_,_,_ = env.step(action)  # take a random action
    #    obs=np.append(obs,o)
    action = env.action_space.sample()
    #obs=normalize(obs, axis=1, norm='l1')
    obs=np.append(obs,action)
    fut, _, _, _ = env.step(action)  # take a random action

    #fut=normalize(fut, axis=1, norm='l1')
    #obs=np.append(obs, reward)
    #obs=np.append(obs, action)
    #time.sleep(1)
    #print(obs.shape)
   # score += reward
    counter += 1
    first=False
    observation = np.empty(shape=(1, size))
    observation[0]=obs
    fobservation = np.empty(shape=(1, 4))
    fobservation[0]=fut
    autoencoder_model.fit(observation, fobservation,
                          batch_size=1,
                          nb_epoch=1,
                          shuffle=False)

    prediction = autoencoder_model.predict(observation)
    print(obs)
    print(fut)
    print(prediction)


    if done:
        print(done, score)
        score = 0
        env.reset()
        counter = 0
