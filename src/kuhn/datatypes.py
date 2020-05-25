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
        2:T([1,0,0,0,1]),
        3:T([0,1,1,0,1]),
        4:T([0,1,1,0,1]),
        5:T([1,0,0,1,0])
        }}
class VisualCategories:
    BETSIZE='betsize'
    ACTION='action'
    REWARD='reward'
    ALL=[BETSIZE,ACTION,REWARD]

class VisualActionTypes:
    FREQUENCY='frequency'
    PROBABILITY='probability'
    ALL=[FREQUENCY,PROBABILITY]

class VisualHandTypes:
    HAND='hand'
    HANDSTRENGTH='handstrength'
    ALL=[HAND,HANDSTRENGTH]

class LimitTypes:
    POT_LIMIT = 'pot_limit'
    NO_LIMIT = 'no_limit'
    LIMIT = 'limit'

class OutputTypes:
    TIERED = 'tiered'
    FLAT = 'flat'

class GameTypes:
    KUHN = 'kuhn'
    COMPLEXKUHN = 'complexkuhn'
    BETSIZEKUHN = 'betsizekuhn'
    HISTORICALKUHN = 'historicalkuhn'
    HOLDEM = 'holdem'
    OMAHAHI = 'omaha_hi'
    OMAHAHILO = 'omaha_hi_lo'

class Positions:
    SB = 'SB'
    BB = 'BB'
    ALL = ['SB','BB']

BLIND_DICT = {
    Positions.BB : T([1]),
    Positions.SB : T([0.5])
}

class Street:
    RIVER = 'river'
    TURN = 'turn'
    FLOP = 'flop'
    PREFLOP = 'preflop'

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
    COMBINED_ACTOR_CRITIC = 'combined_actor_critic'

class Kuhn(object):
    def __init__(self):
        self.starting_street = 1
        self.rule_params = {
            'unopened_action':T([5]),
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
                'cards_per_player':1
            }

class ComplexKuhn:
    def __init__(self):
        self.starting_street = 1
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.rule_params['mask_dict'] = ACTION_MASKS[5]
        self.rule_params['action_dict'] = ACTION_ORDER
        self.rule_params['bets_per_street'] = 2
        self.rule_params['betsize_dict'] = {0:T([0]),1:T([0]),2:T([1]),3:T([1]),4:T([2])}

class BetsizeKuhn:
    def __init__(self):
        self.starting_street = 1
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.state_params['stacksize'] = 3
        self.rule_params['betsize'] = True
        self.rule_params['state_index'] = 0
        self.rule_params['mapping'] = {
                'state':{'previous_betsize':2,'previous_action':1,'rank':0,'hand':0},
                'observation':{'previous_betsize':3,'previous_action':2,'vil_rank':1,'vil_hand':1,'rank':0,'hand':0},
                }
        self.rule_params['mask_dict'] = ACTION_MASKS[5]
        self.rule_params['action_dict'] = ACTION_ORDER
        self.rule_params['bets_per_street'] = 2
        self.rule_params['betsizes'] = T([0.5,1.])
        self.rule_params['bettype'] = LimitTypes.NO_LIMIT

class HistoricalKuhn:
    def __init__(self):
        self.starting_street = 1
        K = BetsizeKuhn()
        self.rule_params = K.rule_params
        self.rule_params['betsizes'] = T([0.5,1.])
        self.state_params = K.state_params
        self.state_params['stacksize'] = 5
        self.rule_params['bets_per_street'] = 2

class Globals:
    GameTypeDict = {
        GameTypes.KUHN:Kuhn(),
        GameTypes.COMPLEXKUHN:ComplexKuhn(),
        GameTypes.BETSIZEKUHN:BetsizeKuhn(),
        GameTypes.HISTORICALKUHN:HistoricalKuhn()
    }
    POSITION_DICT = {Positions.SB:Positions.SB,Positions.BB:Positions.BB}
    POSITION_MAPPING = {'SB':0,'BB':1}
    PLAYERS_POSITIONS_DICT = {2:['SB','BB'],3:['SB','BB','BTN'],4:['SB','BB','CO','BTN'],5:['SB','BB','MP','CO','BTN'],6:['SB','BB','UTG','MP','CO','BTN']}
    ACTION_DICT = {0:'check',1:'fold',2:'call',3:'bet',4:'raise',5:'unopened'}
    ACTION_ORDER = {0:'check',1:'fold',2:'call',3:'bet',4:'raise'}
    REVERSE_ACTION_ORDER = {v:k for k,v in ACTION_ORDER.items()}
    ACTION_MASKS = ACTION_MASKS
    KUHN_CARD_DICT = {0:'?',1:'Q',2:'K',3:'A'}
    BLIND_DICT = BLIND_DICT
    BETSIZE_DICT = {
        1 : T([1.]),
        2: T([0.5,1.]),
        3: T([0.8,0.9,1.]),
        4: T([0.25,0.5,0.75,1.]),
        5: T([0.2,0.4,0.6,0.8,1.]),
        11: T([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    }