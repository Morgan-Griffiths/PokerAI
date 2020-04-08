class Config(object):
    def __init__(self,n_players=2):
        self.action_dict = {0:'check',1:'bet',2:'call',3:'fold',4:'raise',5:'unopened'}
        self.betsize_dict = {0:0,1:1,2:1,3:0,4:2}
        self.position_dict = {2:['SB','BB'],3:['SB','BB','BTN'],4:['SB','BB','CO','BTN'],5:['SB','BB','MP','CO','BTN'],6:['SB','BB','UTG','MP','CO','BTN']}
        self.act_dict = {'SB':0,'BB':1}
        self.agent = 'baseline'
        self.params = {
            'game':'kuhn',
            'save_path':'/Users/morgan/Code/Data/Poker',
            'unopened_action':5,
            'state_params':{
                'ranks':list(range(1,4)),
                'suits':None,
                'n_players':2,
                'n_rounds':1,
                'n_round_bets':1,
                'n_route_raises':0,
                'stacksize':1,
                'pot':1,
                'game_turn':0,
                'cards_per_player':1,
                'to_act':'SB'
            }
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