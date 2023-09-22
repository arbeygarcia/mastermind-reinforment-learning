import torch
import time
import collections
import numpy as np
from envMastermind import envMastermind
from dqnModel import DQN

env = envMastermind()
net = DQN(env.state.shape,len(env.actions))
state = torch.load('RLMasterMind-best_80.dat', map_location=lambda stg, _: stg)
net.load_state_dict(state)
state = env.reset()
total_reward = 0.0
c = collections.Counter()
while True:
    tart_ts = time.time()
    state_v = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)
    c[action] += 1
    state, reward, done = env.step(action)
    print(state)
    total_reward += reward
    if done:
      break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)