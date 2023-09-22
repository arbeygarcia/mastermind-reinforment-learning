import numpy as np
import collections
import torch
import torch.nn as nn
import argparse
import time
import torch.optim as optim
import random
from envMastermind import envMastermind
from dqnModel import DQN

MEAN_REWARD_BOUND = 80
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 2000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 100000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ExperienceBuffer:
  def __init__(self,capacity):
    self.buffer = collections.deque(maxlen=capacity)
  def __len__(self):
    return len(self.buffer)
  def append(self, experience):
    self.buffer.append(experience)
  def sample(self,batch_size):
    indices = np.random.choice(len(self.buffer),batch_size,replace=False)
    states,actions,rewards,dones,next_states = \
    zip(*[self.buffer[idx] for idx in indices])
    return np.array(states), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(next_states)
  
class Agent:
  def __init__(self, env,exp_buffer):
    self.env = env
    self.exp_buffer = exp_buffer
    self._reset()
    self.action_space = env.actions
  def _reset(self):
    self.state = env.reset()
    self.total_reward = 0.0
  @torch.no_grad()
  def play_step(self,net,epsilon=0.0,device="cpu", printState=False):
    done_reward =None

    if np.random.random() < epsilon:
      action = random.randint(0, 26)
    else:
      state_a = np.array([self.state], copy = False)
      state_v = torch.tensor(state_a).to(device)
      q_vals_v = net(state_v)
      _, act_v = torch.max(q_vals_v,dim=1)  ## acha o valor maximo
      action = int(act_v.item())  ## seleciona elemento de dentro act_v
    new_state, reward, is_done = self.env.step(action)
    self.total_reward += reward
    exp = Experience(self.state.copy(),action, reward,is_done,new_state.copy())
    self.exp_buffer.append(exp)
    self.state = new_state.copy()
    if is_done:
      done_reward = self.total_reward
      self._reset()
    return done_reward
  
def calc_loss(batch,net,tgt_net,device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v      = torch.tensor(np.array(states,copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states,copy=False)).to(device)
    actions_v     = torch.tensor(np.array(actions)).to(device).long()
    rewards_v     = torch.tensor(np.array(rewards)).to(device)
    done_mask     = torch.tensor(np.array(dones)).to(device)
    
    state_action_values = net(states_v).gather(1,actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
      next_state_values = tgt_net(next_states_v).max(1)[0]
      next_state_values[done_mask.bool()] = 0.0
      next_state_values =  next_state_values.detach()
    
    expected_state_action_values = next_state_values*GAMMA + rewards_v
    return nn.MSELoss()(state_action_values,expected_state_action_values)

device = torch.device("cpu")
env = envMastermind()
net = DQN(env.state.shape,len(env.actions)).to(device)
tgt_net = DQN(env.state.shape,len(env.actions)).to(device)

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env,buffer)
epsilon = EPSILON_START

optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)
total_rewards = []
frame_idx=0
ts_frame =0
ts = time.time()
time.sleep(10)
best_m_reward = None
printState=False
while True:
  frame_idx+=1
  epsilon = max(EPSILON_FINAL,EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
  reward = agent.play_step(net,epsilon,device=device,printState=False)
  if reward is not None:
    printState=False
    total_rewards.append(reward)

    if time.time()==ts:
      speed = 0.0
    else:
        speed = (frame_idx -  ts_frame)/(time.time() -ts)

    ts_frame = frame_idx
    ts = time.time()
    m_reward = np.mean(total_rewards[-100:])
    print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
    if best_m_reward is None or best_m_reward<m_reward:
      torch.save(net.state_dict(),"RLMasterMind-best_%.0f.dat" % m_reward)
      printState=True
      best_m_reward = m_reward
    if m_reward> MEAN_REWARD_BOUND:
      print("solved in %d frames!" % frame_idx)
      break
  if len(buffer)<REPLAY_START_SIZE:
    continue
  if frame_idx % SYNC_TARGET_FRAMES == 0:
    tgt_net.load_state_dict(net.state_dict())
  optimizer.zero_grad()
  batch = buffer.sample(BATCH_SIZE)
  loss_t = calc_loss(batch,net,tgt_net, device=device)
  loss_t.backward()
  optimizer.step()