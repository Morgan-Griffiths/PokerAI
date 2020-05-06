from torch import Tensor as T

ACTION_DICT = {0:'check',1:'fold',2:'call',3:'bet',4:'raise',5:'unopened'}
ACTION_ORDER = {0:'check',1:'fold',2:'call',3:'bet',4:'raise'}
ACTION_MASKS = {
    4:  {
        0:T([1,0,0,1]),
        1:T([0,0,0,0]),
        2:T([0,0,0,0]),
        3:T([0,1,1,0]),
        4:T([0,1,1,0]),
        5:T([1,0,0,1])
        },
    5: {
        0:T([1,0,0,1,0]),
        1:T([0,0,0,0,0]),
        2:T([0,0,0,0,0]),
        3:T([0,1,1,0,1]),
        4:T([0,1,1,0,1]),
        5:T([1,0,0,1,0])
        }}
class LimitTypes:
    POT_LIMIT = 'pot_limit'
    NO_LIMIT = 'no_limit'
    LIMIT = 'limit'

class GameTypes:
    KUHN = 'kuhn'
    COMPLEXKUHN = 'complexkuhn'
    BETSIZEKUHN = 'betsizekuhn'
    HOLDEM = 'holdem'
    OMAHAHI = 'omaha_hi'

class Positions:
    SB = 'SB'
    BB = 'BB'

BLIND_DICT = {
    Positions.BB : 1,
    Positions.SB : 0.5
}

class Actions:
    CHECK = 'check'
    BET = 'bet'
    CALL = 'call'
    FOLD = 'fold'
    RAISE = 'raise'
    UNOPENED = 'unopened'

class AgentTypes:
    ACTOR = 'actor'
    ACTOR_CRITIC = 'actor_critic'

class Kuhn(object):
    def __init__(self):
        self.rule_params = {
            'unopened_action':5,
            'action_index':1,
            'state_index':0,
            'mapping':{'state':{'previous_action':1,'rank':0,'hand':0},
                'observation':{'previous_action':2,'vil_rank':1,'rank':0,'vil_hand':1,'hand':0},
                },
            'mask_dict' :  ACTION_MASKS[4],
            'action_dict' : {0:'check',1:'fold',2:'call',3:'bet'},
            'betsize_dict' : {0:T([0]),1:T([0]),2:T([1]),3:T([1])},
            'bets_per_street' : 1,
            'num_betsizes': 1,
            'betsize' : False,
            'betsizes' : T([1]),
            'blinds': BLIND_DICT,
            'bettype' : LimitTypes.LIMIT
            }
        self.state_params = {
                'ranks':list(range(1,4)),
                'suits':None,
                'n_players':2,
                'n_rounds':1,
                'stacksize':2,
                'pot':2.,
                'game_turn':0,
                'cards_per_player':1,
                'to_act':'SB',
            }

class ComplexKuhn:
    def __init__(self):
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.rule_params['mask_dict'] = ACTION_MASKS[5]
        self.rule_params['action_dict'] = ACTION_ORDER
        self.rule_params['betsize_dict'] = {0:T([0]),1:T([0]),2:T([1]),3:T([1]),4:T([2])}
        self.rule_params['bets_per_street'] = 2

class BetsizeKuhn:
    def __init__(self):
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.state_params['stacksize'] = 3
        self.rule_params['betsize'] = True
        self.rule_params['action_index'] = 1
        self.rule_params['state_index'] = 0
        self.rule_params['mapping'] = {
                'state':{'previous_betsize':2,'previous_action':1,'rank':0,'hand':0},
                'observation':{'previous_betsize':3,'previous_action':2,'vil_rank':1,'vil_hand':1,'rank':0,'hand':0},
                }
        self.rule_params['mask_dict'] = ACTION_MASKS[5]
        self.rule_params['action_dict'] = ACTION_ORDER
        self.rule_params['bets_per_street'] = 2
        self.rule_params['betsizes'] = T([0.5,1])
        self.rule_params['bettype'] = LimitTypes.NO_LIMIT

class Holdem(object):
    def __init__(self):
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.state_params['ranks'] = list(range(2,15))
        self.state_params['suits'] = list(range(0,4))
        self.state_params['cards_per_player'] = 2
        self.rule_params['action_index'] = 14
        self.rule_params['mapping'] = {
            'state':{
                'previous_action':14,
                'suit':T([1,3]).long(),
                'rank':T([0,2]).long(),
                'hand':T([0,1,2,3]).long(),
                'board':T([4,5,6,7,8,9,10,11,12,13]).long(),
                'board_ranks':T([4,6,8,10,12]).long(),
                'board_suits':T([5,7,9,11,13]).long()
                },
            'observation':{'previous_action':18,
                'suit':T([1,3]).long(),
                'rank':T([0,2]).long(),
                'hand':T([0,1,2,3]).long(),
                'vil_hand':T([4,5,6,7]).long(),
                'vil_ranks':T([4,6]).long(),
                'vil_suits':T([5,7]).long(),
                'board':T([8,9,10,11,12,13,14,15,16,17]).long(),
                'board_ranks':T([8,10,12,14,16]).long(),
                'board_suits':T([9,11,13,15,17]).long()
                }
            }
        self.rule_params['bettype'] = LimitTypes.NO_LIMIT
        self.rule_params['mask_dict'] = ACTION_MASKS[5]
        self.rule_params['action_dict'] = ACTION_ORDER
        self.rule_params['betsize_dict'] = {0:T([0]),1:T([1]),2:T([1]),3:T([0]),4:T([2])}
        self.rule_params['bets_per_street'] = 2

class OmahaHI(object):
    def __init__(self):
        ## NOT IMPLEMENTED ##
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.rule_params['bettype'] = LimitTypes.POT_LIMIT

class Globals:
    GameTypeDict = {
        GameTypes.KUHN:Kuhn(),
        GameTypes.COMPLEXKUHN:ComplexKuhn(),
        GameTypes.BETSIZEKUHN:BetsizeKuhn(),
        GameTypes.HOLDEM:Holdem(),
        GameTypes.OMAHAHI:OmahaHI(),
    }
    POSITION_DICT = {Positions.SB:Positions.SB,Positions.BB:Positions.BB}
    PLAYERS_POSITIONS_DICT = {2:['SB','BB'],3:['SB','BB','BTN'],4:['SB','BB','CO','BTN'],5:['SB','BB','MP','CO','BTN'],6:['SB','BB','UTG','MP','CO','BTN']}
    ACTION_DICT = {0:'check',1:'fold',2:'call',3:'bet',4:'raise',5:'unopened'}
    ACTION_ORDER = {0:'check',1:'fold',2:'call',3:'bet',4:'raise'}
    REVERSE_ACTION_ORDER = {v:k for k,v in ACTION_ORDER.items()}
    ACTION_MASKS = ACTION_MASKS
    KUHN_CARD_DICT = {0:'?',1:'Q',2:'K',3:'A'}
    HOLDEM_RANK_DICT = {v:v for v in range(2,11)}
    BROADWAY = {11:'J',12:'Q',13:'K',14:'A'}
    for k,v in BROADWAY.items():
        HOLDEM_RANK_DICT[k] = v
    HOLDEM_SUIT_DICT = {0:'s',1:'h',2:'d',3:'c'}
    BLIND_DICT = BLIND_DICT