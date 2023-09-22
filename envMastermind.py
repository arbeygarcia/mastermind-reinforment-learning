import numpy as np
import random
from itertools import product

class envMastermind:
    def __init__(self,n=3, num_max_steps=10):
        self.n = n 
        self.num_max_steps= num_max_steps
        self.reset()
        self.actions= list(product(range(1,4), repeat=n))
        self.last_done= False

    def reset(self):
        self.secrect_code = self.generate_secret_code(self.n,self.n)
        self.state = np.float32(np.zeros((self.num_max_steps,5)))
        self.n_step = 0
        return self.state.copy()

    def step(self, action):
        reward=0
        done = False
        state_action = list(self.actions[action])
        feedback = self.verify(state_action, self.secrect_code)
        state_action.extend(feedback)
        if self.n_step < self.num_max_steps:
            self.state[self.n_step] = state_action
        else:
            done = True
        self.n_step += 1

        if self.last_done:
          reward = feedback[0]*3 + feedback[1]
        else:
          reward = feedback[0]*2 + feedback[1]

        if feedback[0] >= 3:
          self.last_done = True
        else:
          self.last_done = False

        return np.float32(self.state), reward, done
   
    @staticmethod
    def generate_secret_code(n, m=3):
        secret_code  = random.sample(range(1,m+1), n)
        return secret_code
    
    @staticmethod
    def verify(guess, secrect_code):
        n =  sum(g == s for g, s in zip(guess, secrect_code))
        m = len(set(guess) & set(secrect_code)) - n
        return [n,m]