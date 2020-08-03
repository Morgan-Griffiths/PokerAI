from models.networks import OmahaActor,OmahaQCritic
from poker.config import Config
import poker.datatypes as pdt
import os
from poker.env import Poker
import torch
from pymongo import MongoClient
from models.model_utils import combined_masks,scale_rewards
from collections import defaultdict
from train import return_value_mask 
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss,NLLLoss


def find_strength(strength):
    # 7462-6185 High card
    # 6185-3325 Pair
    # 3325-2467 2Pair
    # 2467-1609 Trips
    # 1609-1599  Stright
    # 1599-322 Flush
    # 322-166  FH
    # 166-10 Quads
    # 10-0 Str8 flush
    if strength > 6185:
        return 8
    if strength > 3325:
        return 7
    if strength > 2467:
        return 6
    if strength > 1609:
        return 5
    if strength > 1599:
        return 4
    if strength > 322:
        return 3
    if strength > 166:
        return 2
    if strength > 10:
        return 1
    return 0


def examine_actor_critic(actor,critic,data):
    actor.eval()
    critic.eval()
    actor_probs = defaultdict(lambda:[])
    critic_values = defaultdict(lambda:[])
    for i,point in enumerate(data):
        hand_strength = point['hand_strength']
        if hand_strength is not None:
            hand_strength = find_strength(hand_strength)
            betsize_mask = torch.tensor(point['betsize_mask']).long()
            action_mask = torch.tensor(point['action_mask']).long()
            flat_mask = combined_masks(action_mask,betsize_mask)
            state = torch.tensor(point['state'],dtype=torch.float32).to(device)
            step = point['step']
            values = critic(state)['value'].detach() * flat_mask
            # values /= sum(values)
            critic_values[hand_strength].append(values)
            actor_out = actor(state,action_mask,betsize_mask)
            actor_probs[hand_strength].append(actor_out['action_probs'])
        if i == 100:
            break
    # print('0th hands',critic_values[0])
    # print('1th hands',critic_values[1])
    # print('2th values',critic_values[2])
    # print('2th probs',actor_probs[2])
    # print('3th hands',critic_values[3])
    for i in range(len(critic_values[2])):
        print(f'values {critic_values[2][i]}')
        print(f'probs {actor_probs[2][i]}')
        if i == 10:
            break

def examine_learning(actor,critic,data,examine_params):
    # analyse learning step
    actor.train()
    critic.train()
    actor_probs = defaultdict(lambda:[])
    critic_values = defaultdict(lambda:[])
    for i,point in enumerate(data):
        hand_strength = point['hand_strength']
        if hand_strength is not None:
            print('\n### new round ###')
            hand_strength = find_strength(hand_strength)
            betsize_mask = torch.tensor(point['betsize_mask']).long()
            action_mask = torch.tensor(point['action_mask']).long()
            flat_mask = combined_masks(action_mask,betsize_mask)
            action = np.array([point['action']])
            state = torch.tensor(point['state'],dtype=torch.float32).to(device)
            reward = torch.tensor(point['reward'],dtype=torch.float32).to(device)
            scale_reward = scale_rewards(reward,-1,2)
            step = point['step']
            local_values = critic(state)['value']
            # values /= sum(values)
            critic_values[hand_strength].append(local_values.detach() * flat_mask)
            actor_out = actor(state,action_mask,betsize_mask)
            actor_probs[hand_strength].append(actor_out['action_probs'])

            value_mask = return_value_mask(action)
            TD_error = local_values[value_mask] - scale_reward
            critic_loss = (TD_error**2*0.5).mean()
            # critic_loss = F.smooth_l1_loss(scale_reward,TD_error,reduction='sum')
            examine_params['critic_optimizer'].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), examine_params['gradient_clip'])
            examine_params['critic_optimizer'].step()
            # print(f'available actions {flat_mask}')
            print(f'reward {reward}')
            print(f'scale_reward {scale_reward}')
            print(f'local_values[value_mask] {local_values[value_mask]}')
            print(f'TD_error {TD_error}')
            print(f'pre critic update {local_values}')
            values = critic(state)['value'].detach() * flat_mask
            target_values = critic(state)['value']
            print(f'post critic update {target_values}')

            print(f'action {action}')
            print(f'critic best action {((values - values.min()) * flat_mask).argmax(dim=-1)}')
            # Actor update #
            expected_value = (actor_out['action_probs'].view(-1) * target_values.view(-1)).view(value_mask.size()).detach().sum(-1)
            advantages = (target_values[value_mask] - expected_value).view(-1)
            policy_loss = (-actor_out['action_prob'].view(-1) * advantages).mean()
            print('loss',(-actor_out['action_prob'].view(-1) * advantages).mean())
            # loss = NLLLoss()
            # policy_loss = loss(actor_out['action_probs'],values.argmax().unsqueeze(0))
            examine_params['actor_optimizer'].zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), examine_params['gradient_clip'])
            examine_params['actor_optimizer'].step()
            print('expected_value',expected_value)
            print('advantages',advantages)
            print(f"pre actor update {actor_out['action_probs']}")
            actor_out = actor(state,action_mask,betsize_mask)
            print(f'post actor update {actor_out["action_probs"]}')
        if i == 50:
            break

if __name__ == "__main__":
    import argparse

    config = Config()
    game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]
    env_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'betsizes': game_object.rule_params['betsizes'],
        'bet_type': game_object.rule_params['bettype'],
        'n_players': 2,
        'pot':1,
        'stacksize': game_object.state_params['stacksize'],
        'cards_per_player': game_object.state_params['cards_per_player'],
        'starting_street': 3,
        'global_mapping':config.global_mapping,
        'state_mapping':config.state_mapping,
        'obs_mapping':config.obs_mapping,
        'shuffle':True
    }
    env = Poker(env_params)

    nS = env.state_space
    nA = env.action_space
    nB = env.betsize_space
    seed = 1235
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network_params = {
        'game':pdt.GameTypes.OMAHAHI,
        'maxlen':config.maxlen,
        'state_mapping':config.state_mapping,
        'embedding_size':config.agent_params['embedding_size'],
        'device':device,
        'frozen_layer_path':os.path.join(os.getcwd(),'checkpoints/regression/PartialHandRegression')
    }

    actor = OmahaActor(seed,nS,nA,nB,network_params).to(device)
    critic = OmahaQCritic(seed,nS,nA,nB,network_params).to(device)

    actor.load_state_dict(torch.load(config.agent_params['actor_path']))
    critic.load_state_dict(torch.load(config.agent_params['critic_path']))

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.agent_params['actor_lr'],weight_decay=config.agent_params['L2'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.agent_params['critic_lr'])

    examine_params = {
        'actor_optimizer':actor_optimizer,
        'critic_optimizer':critic_optimizer,
        'gradient_clip':config.agent_params['CLIP_NORM'],
    }

    query = {
        'step':0
    }
    projection ={'state':1,'betsize_mask':1,'action_mask':1,'action':1,'step':1,'hand_strength':1,'reward':1,'_id':0}
    client = MongoClient('localhost', 27017,maxPoolSize=10000)
    db = client['poker']
    data = list(db['game_data'].find(query,projection))

    examine_learning(actor,critic,data,examine_params)