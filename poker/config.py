from torch import Tensor as T

class Config(object):
    def __init__(self,game_type='simple',n_players=2):
        self.action_dict = {0:'check',1:'bet',2:'call',3:'fold',4:'raise',5:'unopened'}
        self.betsize_dict = {0:0,1:1,2:1,3:0,4:2}
        self.position_dict = {2:['SB','BB'],3:['SB','BB','BTN'],4:['SB','BB','CO','BTN'],5:['SB','BB','MP','CO','BTN'],6:['SB','BB','UTG','MP','CO','BTN']}
        self.act_dict = {'SB':0,'BB':1}
        self.agent = 'baseline'
        self.params = {
            'game':'kuhn'
        }
        self.rule_params = {
            'unopened_action':5,
            'action_index':1,
            'state_index':0,
            'mapping':{'state':{'previous_action':1,'hand':0},
                'observation':{'previous_action':1,'hand':0},
                },
            'mask_dict' : {
                    0:T([0,0,0,0]),
                    1:T([0,0,1,1]),
                    2:T([0,0,0,0]),
                    3:T([0,0,0,0]),
                    4:T([0,0,1,1]),
                    5:T([1,1,0,1])
                    },
            'action_dict' : {0:'check',1:'bet',2:'call',3:'fold'},
            'betsize_dict' : {0:0,1:1,2:1,3:0},
            'bets_per_street' : 1,
            'raises_per_street' : 0
        }
        self.state_params = {
                'ranks':list(range(1,4)),
                'suits':None,
                'n_players':2,
                'n_rounds':1,
                'stacksize':1,
                'pot':1,
                'game_turn':0,
                'cards_per_player':1,
                'to_act':'SB',
            }
        self.training_params = {
                'epochs':2500,
                'action_index':1,
                'training_round':0,
                'save_dir':'checkpoints',
                'agent_name':'baseline'
            }
        self.agent_params = {
            'BUFFER_SIZE':10000,
            'MIN_BUFFER_SIZE':200,
            'BATCH_SIZE':50,
            'ALPHA':0.6, # 0.7 or 0.6,
            'START_BETA':0.5, # from 0.5-1,
            'END_BETA':1,
            'LEARNING_RATE':0.00025,
            'EPSILON':1,
            'MIN_EPSILON':0.01,
            'GAMMA':0.99,
            'TAU':0.01,
            'UPDATE_EVERY':4,
            'CLIP_NORM':10,
            'embedding':True,
            'mapping':{'hand':0,'action':1}
        }
        if game_type == 'complex':
            print('Complex')
            self.state_params['stacksize']=2
            self.rule_params['mask_dict'] = {
                    0:T([1,1,0,0,0]),
                    1:T([0,0,1,1,1]),
                    2:T([0,0,0,0,0]),
                    3:T([0,0,0,0,0]),
                    4:T([0,0,1,1,0]),
                    5:T([1,1,0,1,0])
                    }
            self.rule_params['action_dict'] = {0:'check',1:'bet',2:'call',3:'fold',4:'raise'}
            self.rule_params['betsize_dict'] = {0:0,1:1,2:1,3:0,4:2}
            self.rule_params['bets_per_street'] = 2
            self.rule_params['raises_per_street'] = 1
            
        self.params['rule_params'] = self.rule_params
        self.params['state_params'] = self.state_params