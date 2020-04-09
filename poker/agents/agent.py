from models.networks import Baseline,Dueling_QNetwork,CardClassification
from models.buffers import PriorityReplayBuffer
from torch.autograd import Variable as V
import torch.nn.functional as F
import random
import torch
import os
import numpy as np
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def return_agent(agent_type,nS,nO,nA,seed,params):
    if agent_type == 'baseline':
        agent = Agent(nS,nO,nA,seed,params)
    elif agent_type == 'dqn':
        agent = Priority_DQN(nS,nO,nA,seed,params)
    else:
        raise ValueError(f'Agent not supported {agent_type}')
    return agent

class Agent(object):
    def __init__(self,nS,nO,nA,seed,params):
        super().__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.seed = seed
        self.network = Baseline(seed,nS,nA,params)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        
    def __call__(self,x,mask):
        return self.network(x,mask)
    
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            action_probs = player_data[position]['action_probs']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
            # actions = player_data[position]['actions']
            # observations = player_data[position]['observations']
            # dones = player_data[position]['dones']
            # indexes = player_data[position]['indexes']
            if len(game_states):
                self.backward(action_probs,rewards)
            
    def backward(self,action_probs,rewards):
        policy_loss = (-action_probs * rewards).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def load_weights(self,path):
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.network.state_dict(), path)
        
"""
DQN with Priority Replay, DDQN, and Dueling DQN.
"""
class Priority_DQN(object):
    def __init__(self,state_space,obs_space,action_space,seed,params):
        self.action_space = action_space
        self.state_space = state_space
        self.seed = random.seed(seed)
        self.batch_size = params['BATCH_SIZE']
        self.buffer_size = params['BUFFER_SIZE']
        self.min_buffer_size = params['MIN_BUFFER_SIZE']
        self.learning_rate = params['LEARNING_RATE']
        self.update_every = params['UPDATE_EVERY']
        self.GAMMA = params['GAMMA']
        self.alpha = params['ALPHA']
        self.tau = params['TAU']
        self.clip_norm = params['CLIP_NORM']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index = 0
        self.action_probs = V(torch.tensor([0.]),requires_grad=True)
        self.qnetwork_local = Dueling_QNetwork(seed,state_space,action_space).to(device)
        self.qnetwork_target = Dueling_QNetwork(seed,state_space,action_space).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=self.learning_rate)
        # Initialize replaybuffer
        self.memory = PriorityReplayBuffer(self.action_space,self.buffer_size,self.batch_size,self.seed,self.alpha)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def __call__(self,state,mask,eps=5.):
        state = state.float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            masked_actions = (action_values + torch.min(action_values)) * mask
            action = masked_actions.max(1)[1]
            return action,self.action_probs
        else:
            eps *= 0.98
            prob_mask = mask / mask.sum()
            action = torch.tensor(np.random.choice(np.arange(self.action_space),p=prob_mask.numpy()))
            return action,self.action_probs
        
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            # action_probs = player_data[position]['action_probs']
            # observations = player_data[position]['observations']
            actions = player_data[position]['actions']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
        self.store_trajectories(game_states,actions,rewards)
        
    def store_trajectories(self,states,actions,rewards):
        next_states = states[1:]
        states = states[:-1]
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = 0
            # Calculate TD error
            # Target
            current_network_action = self.qnetwork_local(next_state).max(1)[1]
            # initial state comes in as (1,4), squeeze to get (4)
            target = reward + self.GAMMA*(self.qnetwork_target(next_state).squeeze(0)[current_network_action])
            # Local. same rational for squeezing
            local = self.qnetwork_local(state).squeeze(0)[action]
            TD_error = reward + target - local
            # Save the experience
            self.memory.add(state,action,reward,next_state,done,TD_error,self.index)
            
            # learn from the experience
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if len(self.memory) > self.min_buffer_size:
                    experiences,indicies,weights = self.memory.sample(self.index)
                    self.backward(experiences,indicies,weights)
            self.index += 1

    def backward(self,experiences,indicies,weights):
        states,actions,rewards,next_states,dones = experiences
        # Local max action
        local_next_state_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        # Target
        target_values = self.qnetwork_target(next_states).detach()
        max_target = target_values.gather(1,local_next_state_actions)
        max_target *= (1-dones) 
        targets = rewards + (self.GAMMA*max_target)
#         targets = rewards + self.GAMMA*(target_values.gather(1,local_next_state_actions))
        # Local
        local = self.qnetwork_local(states).gather(1,actions)
        TD_error = local - targets
        if TD_error.is_cuda:
            loss = ((torch.tensor(weights).cuda() * TD_error)**2*0.5).mean()
        else:
            loss = ((torch.tensor(weights) * TD_error)**2*0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(),self.clip_norm)
        self.optimizer.step()
        # Update the priorities
        TD_errors = np.abs(TD_error.squeeze(1).detach().cpu().numpy())
        self.memory.sum_tree.update_priorities(TD_errors,indicies)
        self.update_target()
        
    # Polyak averaging  
    def update_target(self):
        for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)

"""
Understanding cards
"""

class CardAgent(object):
    def __init__(self,nS,seed,params):
        self.nS = nS
        self.seed = seed
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CardClassification(seed,nS).to(device)
        self.optimizer = optim.Adam(self.network.parameters(),lr=self.params['learning_rate'])

    def __call__(self,x):
        out = self.network(x)
        return out

    def predict(self,x):
        self.network.eval()
        out = self.network(x)
        self.network.train()
        return out

    def backward(self,preds,targets):
        loss = F.smooth_l1_loss(preds,targets).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def load_weights(self,path):
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.network.state_dict(), path)