import os
import copy
import json
import logging
import pymongo
import numpy as np
from torch import load
from pymongo import MongoClient
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS

from poker.env import Poker
import poker.datatypes as pdt
from poker.config import Config
from models.networks import OmahaActor

"""
API for connecting the Poker Env with Alex's frontend client for baseline testing the trained bot.
"""

class API(object):
    def __init__(self):
        self.increment_position = {'SB':'BB','BB':'SB'}
        self.seed = 1458
        self.connect()
        self.game_object = pdt.Globals.GameTypeDict[pdt.GameTypes.OMAHAHI]
        self.config = Config()
        self.env_params = {
            'game':pdt.GameTypes.OMAHAHI,
            'betsizes': self.game_object.rule_params['betsizes'],
            'bet_type': self.game_object.rule_params['bettype'],
            'n_players': 2,
            'pot':1,
            'stacksize': self.game_object.state_params['stacksize'],
            'cards_per_player': self.game_object.state_params['cards_per_player'],
            'starting_street': self.game_object.starting_street,
            'global_mapping':self.config.global_mapping,
            'state_mapping':self.config.state_mapping,
            'obs_mapping':self.config.obs_mapping,
            'shuffle':True
        }
        self.env = Poker(self.env_params)
        self.network_params = self.instantiate_network_params()
        self.model = OmahaActor(self.seed,self.env.state_space,self.env.action_space,self.env.betsize_space,self.network_params)
        self.load_model(self.config.agent_params['actor_path'])
        self.player = {'name':None,'position':'BB'}
        self.reset_trajectories()
        self.num_hands = 0
        
    def reset_trajectories(self):
        self.trajectories = defaultdict(lambda:[])
        self.trajectory = defaultdict(lambda:{'states':[],'obs':[],'betsize_masks':[],'action_masks':[], 'actions':[],'action_category':[],'action_probs':[],'action_prob':[],'betsize':[],'rewards':[]})

    def instantiate_network_params(self):
        network_params = copy.deepcopy(self.env_params)
        network_params['maxlen'] = 10
        network_params['embedding_size'] = 128
        return network_params

    def load_model(self,path):
        if os.path.isfile(path):
            self.model.load_state_dict(load(path))
            self.model.eval()
        else:
            raise ValueError('File does not exist')

    def connect(self):
        client = MongoClient('localhost', 27017,maxPoolSize=10000)
        self.db = client.baseline

    def update_num_hands(self,value):
        self.num_hands = value

    def update_player_name(self,name:str):
        """updates player name"""
        self.player['name'] = name

    def update_player_position(self,position):
        self.player['position'] = position

    def insert_into_db(self,training_data:dict):
        """
        takes trajectories and inserts them into db for data analysis and learning.
        """
        keys = training_data.keys()
        positions = [position for position in keys if position in ['SB','BB']]   
        for position in positions:
            for i,poker_round in enumerate(training_data[position]):
                states = poker_round['states']
                observations = poker_round['obs']
                actions = poker_round['actions']
                action_prob = poker_round['action_prob']
                action_probs = poker_round['action_probs']
                action_categories = poker_round['action_category']
                betsize_masks = poker_round['betsize_masks']
                action_masks = poker_round['action_masks']
                rewards = poker_round['rewards']
                betsizes = poker_round['betsize']
                assert(isinstance(rewards,list))
                assert(isinstance(actions,list))
                assert(isinstance(action_prob,list))
                assert(isinstance(action_probs,list))
                assert(isinstance(states,list))
                print(len(states))
                print(len(actions))
                print(len(action_prob))
                print(len(action_probs))
                print(len(rewards))
                for step,state in enumerate(states):
                    state_json = {
                        'game':self.env.game,
                        'player':self.player['name'],
                        'hand_num':self.num_hands,
                        'poker_round':step,
                        'state':state.tolist(),
                        'action_probs':action_probs[step].tolist(),
                        'action_prob':action_prob[step].tolist(),
                        'action':actions[step],
                        'action_category':action_categories[step],
                        'betsize_mask':betsize_masks[step].tolist(),
                        'action_mask':action_masks[step].tolist(),
                        'betsize':betsizes[step],
                        'reward':rewards[step]
                    }
                    self.db['game_data'].insert_one(state_json)

    def return_player_stats(self):
        """Returns dict of current player stats against the bot."""
        query = {
            'player':self.player['name'],
            'poker_round': 0
        }
        projection ={'reward':1,'hand_num':1,'_id':0}
        player_results = self.db['game_data'].find(query)
        results = []
        # total_hands = 0
        for result in player_results:
            results.append(result['reward'])
            # total_hands = max(result['hand_num'],total_hands)
        total_hands = len(results)
        bb_per_hand = sum(results) / total_hands
        player_stats = {
            'results':sum(results),
            'bb_per_hand':bb_per_hand,
            'total_hands':total_hands
        }
        self.update_num_hands(total_hands)
        return player_stats

    def parse_env_outputs(self,state,action_mask,betsize_mask):
        reward = state[:,-1][:,self.env.state_mapping['hero_stacksize']] - self.env.starting_stack
        # cards go in a list
        state_object = {
            'hero_stack'            :state[:,-1][:,self.env.state_mapping['hero_stacksize']][0],
            'hero_position'         :state[:,-1][:,self.env.state_mapping['hero_position']][0],
            'hero_cards'            :state[:,-1][:,self.env.state_mapping['hero_hand']][0].tolist(),
            'villain_stack'         :state[:,-1][:,self.env.state_mapping['player2_stacksize']][0],
            'villain_position'      :state[:,-1][:,self.env.state_mapping['player2_position']][0],
            'pot'                   :state[:,-1][:,self.env.state_mapping['pot']][0],
            'board_cards'           :state[:,-1][:,self.env.state_mapping['board']][0].tolist(),
            'hero_street_total'     :state[:,-1][:,self.env.state_mapping['player1_street_total']][0],
            'villain_street_total'  :state[:,-1][:,self.env.state_mapping['player2_street_total']][0],
            'last_action'           :state[:,-1][:,self.env.state_mapping['last_aggressive_action']][0],
            'last_betsize'          :state[:,-1][:,self.env.state_mapping['last_aggressive_betsize']][0],
            'last_position'         :state[:,-1][:,self.env.state_mapping['last_aggressive_position']][0],
            'done'                  :state[:,-1][:,self.env.state_mapping['last_aggressive_position']][0],
            'action_mask'           :action_mask.tolist(),
            'betsize_mask'          :betsize_mask.tolist(),
            'street'                :state[:,-1][:,self.env.state_mapping['street']][0],
        }
        outcome_object = {
            'player1_stack':self.env.players['SB'].stack,
            'player1_reward':self.env.players['SB'].stack - self.env.starting_stack,
            'player1_hand':[item for sublist in self.env.players['SB'].hand for item in sublist],
            'player2_stack':self.env.players['BB'].stack,
            'player2_reward':self.env.players['BB'].stack - self.env.starting_stack,
            'player2_hand':[item for sublist in self.env.players['BB'].hand for item in sublist],
        }
        json_obj = {'state':state_object,'outcome':outcome_object}
        return json.dumps(json_obj)

    def increment_hand(self):
        self.num_hands += 1

    def store_state(self,state,obs,action_mask,betsize_mask):
        cur_player = self.env.current_player
        print('storing state for ',cur_player)
        self.trajectory[cur_player]['states'].append(copy.copy(state))
        self.trajectory[cur_player]['action_masks'].append(copy.copy(action_mask))
        self.trajectory[cur_player]['betsize_masks'].append(copy.copy(betsize_mask))

    def store_actions(self,actor_outputs):
        cur_player = self.env.current_player
        print('storing state for ',cur_player)
        self.trajectory[cur_player]['actions'].append(actor_outputs['action'])
        self.trajectory[cur_player]['action_category'].append(actor_outputs['action_category'])
        self.trajectory[cur_player]['action_prob'].append(actor_outputs['action_prob'])
        self.trajectory[cur_player]['action_probs'].append(actor_outputs['action_probs'])
        self.trajectory[cur_player]['betsize'].append(actor_outputs['betsize'])

    def query_bot(self,state,obs,action_mask,betsize_mask):
        while self.env.current_player != self.player['position']:
            outputs = self.model(state,action_mask,betsize_mask)
            self.store_actions(outputs)
            state,obs,done,action_mask,betsize_mask = self.env.step(outputs)
            if not done:
                self.store_state(state,obs,action_mask,betsize_mask)
        return state,obs,done,action_mask,betsize_mask

    def reset(self):
        assert self.player['name'] is not None
        assert isinstance(self.player['position'],str)
        self.increment_hand()
        self.update_player_position(self.increment_position[self.player['position']])
        state,obs,done,action_mask,betsize_mask = self.env.reset()
        self.store_state(state,obs,action_mask,betsize_mask)
        if self.env.current_player != self.player['position']:
            state,obs,done,action_mask,betsize_mask = self.query_bot(state,obs,action_mask,betsize_mask)
        return self.parse_env_outputs(state,action_mask,betsize_mask)

    def step(self,action:str,betsize:float):
        """Maps action + betsize -> to a flat action category"""
        assert self.player['name'] is not None
        assert isinstance(self.player['position'],str)
        if isinstance(betsize,str):
            betsize = float(betsize)
        # print('action,betsize',action,betsize)
        action_type = pdt.REVERSE_ACTION_DICT[action]
        action_category,betsize_category = self.env.convert_to_category(action_type,betsize)
        assert isinstance(action_category,int)
        player_outputs = {
            'action':action_category,
            'action_category':action_type,
            'betsize':betsize_category,
            'action_prob':np.array([0]),
            'action_probs':np.zeros(self.env.betsize_space)
        }
        print('player_outputs',player_outputs)
        self.store_actions(player_outputs)
        state,obs,done,action_mask,betsize_mask = self.env.step(player_outputs)
        if not done:
            self.store_state(state,obs,action_mask,betsize_mask)
            if self.env.current_player != self.player['position']:
                state,obs,done,action_mask,betsize_mask = self.query_bot(state,obs,action_mask,betsize_mask)
        if done:
            rewards = self.env.player_rewards()
            for position in self.trajectory.keys():
                N = len(self.trajectory[position]['betsize_masks'])
                self.trajectory[position]['rewards'] = [rewards[position]] * N
                self.trajectories[position].append(self.trajectory[position])
            self.insert_into_db(self.trajectories)
            self.reset_trajectories()
        return self.parse_env_outputs(state,action_mask,betsize_mask)

    @property
    def current_player(self):
        return self.player

# instantiate env
api = API()

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:*"}})

logging.basicConfig(level=logging.DEBUG)

@app.route('/health')
def home():
    return 'Server is up and running'

@app.route('/api/player/name',methods=['POST'])
def player():
    req_data = json.loads(request.get_data())
    api.update_player_name(req_data.get('name'))
    return 'Updated Name'

@app.route('/api/player/stats')
def player_stats():
    return json.dumps(api.return_player_stats())

@app.route('/api/model/load',methods=['POST'])
def load_model():
    req_data = json.loads(request.get_data())
    api.load_model(req_data.get('path'))
    return 'Loaded Model'

@app.route('/api/reset')
def reset():
    return api.reset()

@app.route('/api/step', methods=['POST'])
def gen_routes():
    log = logging.getLogger(__name__)
    print(request.get_data())
    req_data = json.loads(request.get_data())
    action = req_data.get('action')
    betsize = req_data.get('betsize')
    log.info(f'action {action}')
    log.info(f'betsize {betsize}')
    return api.step(action,betsize)

if __name__ == '__main__':
    app.run(debug=True, port=4000)