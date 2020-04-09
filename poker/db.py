from pymongo import MongoClient
import numpy as np
import torch

class MongoDB(object):
    def __init__(self):
        self.connect()

    def connect(self):
        client = MongoClient('localhost', 27017)
        self.db = client['poker']

    def store_data(self,training_data:dict,mapping:dict,training_round:int):
        """
        training_data, contains all positions
        Poker db;
        state,obs,action,log_prob,reward collections
        State:
        training_run,round,step,p1_hand,previous_action
        *** future ***
        p1_position,p1_hand,p1_stack
        p2_position,p2_stack
        pot,board_cards,previous_action
        Action:
        training_run,round,step,p1_action,log_prob
        Reward:
        training_run,round,step,reward
        """
        positions = training_data.keys()
        for position in positions:
            for i,poker_round in enumerate(training_data[position]):
                game_states = poker_round['game_states']
                observations = poker_round['observations']
                actions = poker_round['actions']
                action_probs = poker_round['action_probs']
                rewards = poker_round['rewards']
                assert(isinstance(rewards,torch.Tensor))
                assert(isinstance(action_probs,torch.Tensor))
                assert(isinstance(actions,torch.Tensor))
                assert(isinstance(observations,torch.Tensor))
                assert(isinstance(game_states,torch.Tensor))
                for step,game_state in enumerate(game_states):
                    hand = int(game_state[mapping['state']['hand']])
                    previous_action = int(game_state[mapping['state']['previous_action']])
                    state_json = {
                        'position':position,
                        'hand':hand,
                        'reward':float(rewards[step]),
                        'action':int(actions[step]),
                        'action_probs':float(action_probs[step].detach()),
                        'previous_action':previous_action,
                        'training_round':training_round,
                        'poker_round':i,
                        'step':step
                        }
                    self.db['game_data'].insert_one(state_json)
                    
    def get_data(self,query:dict,projection:dict):
        print(f'query {query}, projection {projection}')
        data = self.db['game_data'].find(query,projection)
        return data

    @staticmethod
    def pad_inputs(inputs:list,N:int):
        assert(isinstance(inputs,list))
        for i,data in enumerate(inputs):
            if data.shape[0] < N:
                padding = np.full(N-data.shape[0],data[0])
                inputs[i] = np.concatenate([padding,data])
        return inputs

    @staticmethod
    def return_frequency(inputs:list,interval:int,num_features:int):
        assert(isinstance(inputs,list))
        assert(isinstance(inputs[0],np.ndarray))
        assert(len(inputs[0]) > interval)
        percentages = []
        for data_type in inputs:
            percentage_type = [[] for _ in range(num_features)]
            for i in range(data_type.shape[0] - interval):
                group = data_type[i:i+interval]
                uniques,counts = np.lib.arraysetops.unique(group,return_counts=True)
                frequencies = counts / np.sum(counts)
                if frequencies.shape[0] < num_features:
                    base = np.zeros(num_features)
                    mask = set(np.arange(num_features))&set(uniques)
                    for i,loc in enumerate(mask):
                        base[loc] = frequencies[i]
                    frequencies = base
                for j in range(len(frequencies)):
                    percentage_type[j].append(frequencies[j])
            percentages.append(percentage_type)
        return percentages


    def byState(self,params:dict,pad=True):
        data = self.get_data(params)
        states = []
        for point in data:
            states.append(np.array([point['hand'],point['position']]))
        return states

    def actionByHand(self,data:'pymongo.cursor',params,pad=True):
        hands = []
        actions = []
        for point in data:
            hands.append(np.array(point['hand']))
            actions.append(np.array(point['action']))
        hands = np.stack(hands)
        actions = np.stack(actions)
        R = hands.shape[0]
        hands = hands.reshape(R,1)
        actions = actions.reshape(R,1)
        unique_hands,hand_counts = np.lib.arraysetops.unique(hands,return_counts=True)
        unique_actions,action_counts = np.lib.arraysetops.unique(actions,return_counts=True)
        num_features = len(action_counts)
        print('unique_hands and counts',unique_hands,hand_counts)
        print('unique_actions and counts',unique_actions,action_counts)
        return_data = []
        N = 0
        for hand_type in unique_hands:
            mask = np.where(hands == hand_type)
            return_data.append(actions[mask])
            N = max(N,actions[mask].shape[0])
        if pad == True:
            return_data = MongoDB.pad_inputs(return_data,N)
        return_data = MongoDB.return_frequency(return_data,params['interval'],num_features)
        return return_data,unique_hands,unique_actions

    def byActions(self,params:dict,pad=True,action_only=False):
        data = self.get_data(params)
        actions = []
        probs = []
        for point in data:
            actions.append(point['action'])
            probs.append(point['action_probs'])

        actions = np.array(actions)
        if action_only == False:
            probs = np.exp(np.array(probs))
            # sort according to action types
            N = 0
            action_types = unique(actions)
            action_data = []
            for act in action_types:
                mask = np.where(actions == act)[0]
                action_data.append(probs[mask])
                N = max(N,probs[mask].shape[0])
                # action_data.append(np.stack([actions[mask],probs[mask]]))
            if pad == True:
                action_data = MongoDB.pad_inputs(action_data,N)
        else:
            action_data = [actions]
        return action_data

    def byObservations(self,params,pad=True):
        data = self.get_data(params)
        observations = []
        for point in data:
            observations.append(np.array([point['hand'],point['position']]))
        return observations

    def byRewards(self,params,pad=True):
        data = self.get_data(params)
        rewards = []
        for point in data:
            rewards.append(point['reward'])
        return np.array(rewards)

    def close(self):
        self.client.close()

    def clean_db(self):
        self.db['game_data'].delete_many({})

    def clear_collection(self,collection):
        collection.delete_many({})

    def update_training_round(self,num):
        self.current_training_round = num

    def insert_values(self,data,collection):
        collection.insert_one(data)