import gym
import numpy as np

env = gym.make('Breakout-v0')
observation = env.reset()

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

re=np.zeros((210*160,1))
for t in range(2000):
    env.render()
    print(observation)
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    grey = rgb2gray(observation)
    grey = grey.reshape(210*160,1)
    re=np.concatenate((re, grey),1)
    print(re.shape)
