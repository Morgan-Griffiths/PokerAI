from pymongo import MongoClient
import numpy as np
import torch
import poker.datatypes as pdt

class MongoDB(object):
    def __init__(self):
        self.connect()

    def connect(self):
        client = MongoClient('localhost', 27017)
        self.db = client['poker']

    def store_data(self,training_data:dict,mapping:dict,training_round:int,gametype):
        if gametype == pdt.GameTypes.COMPLEXKUHN or gametype == pdt.GameTypes.KUHN or gametype == pdt.GameTypes.BETSIZEKUHN:
            self.store_kuhn_data(training_data,mapping,training_round,gametype)
        elif gametype == pdt.Holdem:
            self.store_holdem_data(training_data,mapping,training_round,gametype)

    def store_kuhn_data(self,training_data:dict,mapping:dict,training_round:int,gametype:str):
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
                action_prob = poker_round['action_prob']
                action_probs = poker_round['action_probs']
                rewards = poker_round['rewards']
                values = poker_round['values']
                betsizes = poker_round['betsizes']
                betsize_prob = poker_round['betsize_prob']
                betsize_probs = poker_round['betsize_probs']
                assert(isinstance(rewards,torch.Tensor))
                assert(isinstance(actions,torch.Tensor))
                assert(isinstance(action_prob,torch.Tensor))
                assert(isinstance(action_probs,torch.Tensor))
                assert(isinstance(observations,torch.Tensor))
                assert(isinstance(game_states,torch.Tensor))
                for step,game_state in enumerate(game_states):
                    hand = int(game_state[mapping['state']['hand']])
                    vil_hand = int(game_state[mapping['observation']['vil_hand']])
                    previous_action = int(game_state[mapping['state']['previous_action']])
                    state_json = {
                        'position':position,
                        'hand':hand,
                        'vil_hand':vil_hand,
                        'reward':float(rewards[step]),
                        'action':int(actions[step]),
                        'action_probs':action_probs[step].detach().tolist(),
                        'previous_action':previous_action,
                        'training_round':training_round,
                        'poker_round':i,
                        'step':step,
                        'game':gametype
                        }
                    if len(betsizes) > 0:
                        if betsizes[step][0].dim() > 1:
                            index = torch.arange(betsizes[step].size(0))
                            state_json['betsizes'] = float(betsizes[step][index,actions[step]].detach())
                            if len(betsize_prob) > 0:
                                state_json['betsize_prob'] = float(betsize_prob[step][index,actions[step]].detach())
                                state_json['betsize_probs'] = betsize_probs[step][index,actions[step]].detach().tolist()
                        else:
                            state_json['betsizes'] = float(betsizes[step].detach())
                            if len(betsize_prob) > 0:
                                state_json['betsize_prob'] = float(betsize_prob[step].detach())
                                state_json['betsize_probs'] = betsize_probs[step].detach().tolist()
                    if len(values) > 0:
                        if len(values[step][0]) > 1:
                            index = torch.arange(values[step].size(0))
                            state_json['value'] = values[step][index,actions[step]].detach().tolist()
                        else:
                            state_json['value'] = float(values[step].detach())
                    self.db['game_data'].insert_one(state_json)

    def store_holdem_data(self,training_data:dict,mapping:dict,training_round:int,gametype:str):
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
        print('holdem')
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
                    hand = game_state[mapping['state']['hand']].long().numpy()
                    vil_hand = game_state[mapping['observation']['vil_hand']].long().numpy()
                    board = game_state[mapping['state']['board']].long().numpy()
                    previous_action = int(game_state[mapping['state']['previous_action']])
                    state_json = {
                        'position':position,
                        'hand':hand.tolist(),
                        'vil_hand':vil_hand.tolist(),
                        'board':board.tolist(),
                        'reward':float(rewards[step]),
                        'action':int(actions[step]),
                        'action_probs':float(action_probs[step].detach()),
                        'unit_vector':float(MongoDB.unit_vector(action_probs[step].detach())),
                        'previous_action':previous_action,
                        'training_round':training_round,
                        'poker_round':i,
                        'step':step,
                        'game':gametype
                        }
                    self.db['game_data'].insert_one(state_json)
                    
    def get_data(self,query:dict,projection:dict):
        print(f'query {query}, projection {projection}')
        data = self.db['game_data'].find(query,projection)
        return data

    @staticmethod
    def unit_vector(vector):
        return vector / torch.sqrt((vector**2).sum())

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
        if len(inputs[0]) < interval:
            raise ValueError("Number of samples < the interval") 
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
                        base[int(loc)] = frequencies[i]
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

    def betsizeByHand(self,data:'pymongo.cursor',params,pad=True):
        hands = []
        betsizes = []
        for point in data:
            hands.append(np.array(point['hand']))
            betsizes.append(np.array(point['betsizes']))
        hands = np.stack(hands)
        betsizes = np.stack(betsizes)
        betsize_mask = betsizes > -1
        hands = hands[betsize_mask]
        betsizes = betsizes[betsize_mask]
        R = hands.shape[0]
        hands = hands.reshape(R,1)
        betsizes = betsizes.reshape(R,1)
        unique_hands,hand_counts = np.lib.arraysetops.unique(hands,return_counts=True)
        unique_betsizes,action_counts = np.lib.arraysetops.unique(betsizes,return_counts=True)
        num_features = len(action_counts)
        print('unique_hands and counts',unique_hands,hand_counts)
        print('unique_betsizes and counts',unique_betsizes,action_counts)
        return_data = []
        N = 0
        for hand_type in unique_hands:
            mask = np.where(hands == hand_type)
            return_data.append(betsizes[mask])
            N = max(N,betsizes[mask].shape[0])
        if pad == True:
            return_data = MongoDB.pad_inputs(return_data,N)
        return_data = MongoDB.return_frequency(return_data,params['interval'],num_features)
        return return_data,unique_hands,unique_betsizes

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