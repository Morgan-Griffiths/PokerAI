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


# instantiate env
api = API()

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/health')
def home():
    return 'Server is up and running'

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
    # log.info(f'request.args {request.args}')
    return api.step(action,betsize)
    # return api.step()
    # config = Config('math')
    # db = connect()
    # # Pull the route and plan data
    # routes, fields, goals, settings, plan, dblocations, gymId, gym = get_collections(
    #     db, plan_id, printing=False)
    # log.info(f'gym {gym}')
    # historical_routes, active_routes = separate_routes(routes)
    # # Restrict historical routes to the previous 6 months.
    # now = date.today()
    # six_months = now - relativedelta(months=+6)
    # six_months_routes = restrict_history_by_date(historical_routes, six_months)
    # # Restrict historical routes to the previous N routes (sorted according to date)
    # N_historical_routes = historical_routes[-goals['totalGymRoutes']:]
    # # Max grade to be suggested by engine
    # setters = plan['setters']
    # discipline = plan['discipline']
    # # Instantiate the utils class - this converts routes into arrays and stores them locally in the utils
    # utils = ServerUtilities(active_routes, six_months_routes,
    #                         fields, plan_id, discipline, config, gymId)
    # utils.parse_settings(settings)
    # utils.convert_goals(goals)
    # utils.parse_locations(dblocations)
    # max_setting, setting_time, setter_nicknames, relative_time, setting_mask, num_grades, max_grade, grade_index = get_setter_attributes(
    #     setters, utils)
    # # Set max grade available to set
    # utils.update_max_grade(max_grade)
    # # Two setting styles -> By Location, By Route
    # # Two climbing disciplines -> Bouldering and Roped Climbing
    # # if plan['discipline'] == 'Bouldering':
    # locations = plan['locations']
    # # Change this based on location flag. To take account of when they set by route.
    # # Find all the routes we are about to strip
    # readable_stripped_routes = return_stripped(active_routes, locations)
    # # update config based on settings and goals. Update tehcnique mask, grade mask, novelty weights, routes by location. terrain types.
    # # Update num reset routes to the num desired routes by location.
    # utils.num_reset_routes = len(readable_stripped_routes)
    # utils.bulk_strip_by_location(locations)
    # routes, readable_routes = utils.return_location_suggestions()

    # # Distribute routes
    # log.debug(('readable_routes', readable_routes))
    # info_array = utils.return_info_array(
    #     readable_stripped_routes, setter_nicknames)
    # log.debug(('info_array', info_array.shape))
    # # Distribute the routes among the setters
    # # distributed_routes = distribute_routes(routes,readable_routes,setting_time,setter_nicknames,relative_time,setting_mask,num_grades,grade_index)
    # dist = Distribution(routes, readable_routes, setting_time, setter_nicknames,
    #                     relative_time, setting_mask, num_grades, grade_index, info_array, utils)
    # distributed_routes = dist.distribute(constraint=utils.distribution_method)
    # print(dist.setting_time, flush=True)
    # update_plan(db, distributed_routes, plan_id)
    # return 'Donezors!'


def connect():
    try:
        mongo_path = os.environ['MONGODB_URI']
    except:
        mongo_path = 'mongodb://' + os.environ['MONGODB_HOST'] + ':27017/'
    client = MongoClient(mongo_path)
    db = client.setting
    return db

def get_collections(db, id, printing=False):
    plan = db['plans'].find_one({'_id': ObjectId(id)})
    gymId = plan['gymId']
    gym = db['gyms'].find_one({'_id': ObjectId(gymId)})
    goals = db['goals'].find({"active": True, "gymId": gymId})[0]
    settings = db['settings'].find({"gymId": gymId})[0]
    routes = db['routes'].find({"gymId": gymId})
    fields = db['fields'].find({"gymId": gymId})
    locations = db['locations'].find({"gymId": gymId})
    if printing == True:
        print('id plan', plan, flush=True)
        print('gymId', gymId, flush=True)
        print('gym', gym, flush=True)
        print('goals', goals, flush=True)
        print('routes', routes, flush=True)
        print('fields', fields, flush=True)
        print('locations', locations, flush=True)
        print('settings', settings, flush=True)
    return routes, fields, goals, settings, plan, locations, gymId, gym

if __name__ == '__main__':
    app.run(debug=True, port=5000)