from flask import Flask, jsonify, request
import json
import os
import logging
import pymongo
from pymongo import MongoClient
from poker.env import Poker
from poker.config import Config
import poker.datatypes as pdt

"""
API for connecting the Poker Env with Alex's frontend client for baseline testing the trained bot.
"""

class API(object):
    def __init__(self):
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
        self.player = None

    def connect(self):
        client = MongoClient('localhost', 27017,maxPoolSize=10000)
        self.db = client.baseline

    def update_player(self,player):
        self.player = player

    def store_state(self,state,action_mask,betsize_mask):
        self.db['states'].insert_one(state.tolist())
        self.db['action_masks'].insert_one(action_mask.tolist())
        self.db['betsize_masks'].insert_one(betsize_mask.tolist())

    def store_result(self,outcome):
        self.db['results'].insert_one(outcome)

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
        }
        outcome_object = {
            'player1_stack':self.env.players['SB'].stack,
            'player1_reward':self.env.players['SB'].stack - self.env.starting_stack,
            'player1_hand':self.env.players['SB'].hand,
            'player2_stack':self.env.players['BB'].stack,
            'player2_reward':self.env.players['BB'].stack - self.env.starting_stack,
            'player2_hand':self.env.players['BB'].hand,
        }
        json_obj = {'state':state_object,'outcome':outcome_object}
        return json.dumps(json_obj)

    def reset(self):
        state,obs,done,mask,betsize_mask = self.env.reset()
        return self.parse_env_outputs(state,mask,betsize_mask)

    def step(self,action,betsize):
        """Maps action + betsize -> to a flat action category"""
        # print('action,betsize',action,betsize)
        action_type = pdt.REVERSE_ACTION_DICT[action]
        action_category = self.env.convert_to_category(action_type,betsize)
        player_outputs = {
            'action':action_category,
            'action_category':action_type,
            'betsize':betsize
        }
        print('player_outputs',player_outputs)
        state,obs,done,mask,betsize_mask = self.env.step(player_outputs)
        print('done',done)
        return self.parse_env_outputs(state,mask,betsize_mask)

    @property
    def current_player(self):
        return self.player

# instantiate env
api = API()

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/health')
def home():
    return 'Server is up and running'

@app.route('/api/player',methods=['POST'])
def player():
    req_data = json.loads(request.get_data())
    api.update_player(req_data.get('player'))

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
    app.run(debug=True, port=5000)