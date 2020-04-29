
import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import optim

from models.networks import Baseline,Dueling_QNetwork,HoldemBaselineCritic,HoldemBaseline,BaselineKuhnCritic,BaselineCritic,hard_update
from models.buffers import PriorityReplayBuffer
import poker.datatypes as pdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def return_agent(agent_type,nS,nO,nA,seed,params):
    if agent_type == pdt.AgentTypes.ACTOR:
        agent = BaselineAgent(nS,nO,nA,seed,params)
    elif agent_type == pdt.AgentTypes.ACTOR_CRITIC:
        agent = Agent(nS,nO,nA,seed,params)
    else:
        raise ValueError(f'Agent not supported {agent_type}')
    return agent

class BaselineAgent(object):
    def __init__(self,nS,nO,nA,seed,params):
        super().__init__()
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.seed = seed
        self.network = params['network'](seed,nS,nA,params)
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

class Agent(object):
    def __init__(self,nS,nO,nA,seed,params):
        super().__init__()
        print('Actor critic')
        self.nS = nS
        self.nO = nO
        self.nA = nA
        self.seed = seed
        self.epochs = params['epochs']+1
        self.tau = params['TAU']
        self.max_reward = params['max_reward']
        self.gradient_clip = params['CLIP_NORM']
        self.critic_type = params['critic_type']
        self.local_actor = params['actor_network'](seed,nS,nA,params)
        self.target_actor = params['actor_network'](seed,nS,nA,params)
        self.local_critic = params['critic_network'](seed,nO,nA,params)
        self.target_critic = params['critic_network'](seed,nO,nA,params)
        if self.critic_type == 'Q':
            self.critic_backward = self.reg_critic_backward
            self.actor_backward = self.reg_actor_backward
            self.critique = self.reg_critique
        else:
            self.critic_backward = self.qcritic_backward
            self.actor_backward = self.qactor_backward
            self.critique = self.qcritique

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=1e-4,weight_decay=params['L2'])
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=1e-4)
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)
        
    def __call__(self,x,mask):
        return self.local_actor(x,mask)

    def reg_critique(self,obs,action):
        return self.target_critic(obs,action)

    def qcritique(self,obs,action):
        return self.target_critic(obs)
    
    def learn(self,player_data):
        positions = player_data.keys()
        for position in positions:
            action_probs = player_data[position]['action_probs']
            complete_probs = player_data[position]['complete_probs']
            game_states = player_data[position]['game_states']
            rewards = player_data[position]['rewards']
            actions = player_data[position]['actions'].view(-1)
            observations = player_data[position]['observations']
            # dones = player_data[position]['dones']
            # indexes = player_data[position]['indexes']
            if len(game_states):
                self.critic_backward(rewards,observations,actions)
                self.actor_backward(action_probs,observations,actions,rewards,complete_probs)

    def reg_critic_backward(self,rewards,observations,actions):
        values = self.local_critic(observations,actions)
        scaled_rewards = rewards/self.max_reward
        critic_loss = F.smooth_l1_loss(scaled_rewards,values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)
            
    def reg_actor_backward(self,action_probs,observations,actions,rewards,complete_probs):
        values = self.target_critic(observations,actions)
        policy_loss = (-action_probs * values).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        Agent.soft_update(self.local_actor,self.target_actor,self.tau)

    def return_value_mask(self,actions):
        M = actions.size(0)
        value_mask = torch.zeros(M,self.nA)
        if actions.dim() > 1:
            actions = actions.squeeze(1)
        value_mask[torch.arange(M),actions] = 1
        return value_mask.bool()

    def qcritic_backward(self,rewards,observations,actions):
        values = self.local_critic(observations)
        value_mask = self.return_value_mask(actions)
        scaled_rewards = rewards/self.max_reward
        # print('scaled_rewards',scaled_rewards,' value',values[value_mask].detach(),'values',values.detach(),' action',actions)
        critic_loss = F.smooth_l1_loss(scaled_rewards.squeeze(1),values[value_mask])
        # print('critic_loss',critic_loss)
        # critic_loss = (scaled_rewards - values[value_mask]).sum()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()
        Agent.soft_update(self.local_critic,self.target_critic,self.tau)
            
    def qactor_backward(self,action_probs,observations,actions,rewards,complete_probs):
        values = self.target_critic(observations)
        value_mask = self.return_value_mask(actions)
        expected_value = (complete_probs * values).detach().sum(-1)
        advantages = values[value_mask] - expected_value
        policy_loss = (-action_probs.view(-1) * advantages).sum()
        # print('')
        # print('observations',observations)
        # print('values',values)
        # print('expected_value',expected_value)
        # print('advantages',advantages)
        # print('Q value',values[value_mask])
        # print('complete_probs',complete_probs)
        # print('actions',actions)
        # print('rewards',rewards)
        # print('action_probs',action_probs)
        # print('policy_loss',policy_loss)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        Agent.soft_update(self.local_actor,self.target_actor,self.tau)

    def load_weights(self,path):
        self.local_actor.load_state_dict(torch.load(path))
        self.local_critic.load_state_dict(torch.load(path))
        self.local_actor.eval()
        self.local_critic.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.local_actor.state_dict(), path)
        torch.save(self.local_critic.state_dict(), path)

    
    def update_networks(self):
        self.target_critic = Agent.soft_update_target(self.local_critic,self.target_critic,self.tau)
        self.target_actor = Agent.soft_update_target(self.local_actor,self.target_actor,self.tau)

    @staticmethod
    def soft_update(local,target,tau):
        for local_param,target_param in zip(local.parameters(),target.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
        
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
    def __init__(self,params):
        self.params = params
        self.network_params = params['network_params']
        self.agent_name = params['save_path']
        self.save_dir = params['save_dir']
        self.save_path = os.path.join(params['save_dir'],params['save_path'])
        self.load_path = os.path.join(params['save_dir'],params['load_path'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = params['network'](params['network_params']).to(device)
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

    def load_weights(self,path=None):
        if path == None:
            path = self.load_path
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

    def save_weights(self,path=None):
        if path == None:
            path = self.save_path
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.network.state_dict(), path)