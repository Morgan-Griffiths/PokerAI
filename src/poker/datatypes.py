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
        self.rule_params['betsize_dict'] = {0:T([0]),1:T([0]),2:T([1]),3:T([1]),4:T([2])}
        self.rule_params['bets_per_street'] = 2

class BetsizeKuhn:
    def __init__(self):
        self.starting_street = 1
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
        self.rule_params['betsizes'] = T([0.5,1.])
        self.rule_params['bettype'] = LimitTypes.NO_LIMIT

class HistoricalKuhn:
    def __init__(self):
        self.starting_street = 1
        K = BetsizeKuhn()
        self.rule_params = K.rule_params
        self.rule_params['betsizes'] = T([0.5,1.])
        self.state_params = K.state_params

class BaseHoldem(object):
    def __init__(self):
        self.starting_street = 0
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.state_params['ranks'] = list(range(2,15))
        self.state_params['suits'] = list(range(0,4))
        self.state_params['cards_per_player'] = 2
        self.rule_params['mapping'] = {
            'state':{
                'suit':T([1,3]).long(),
                'rank':T([0,2]).long(),
                'hand':T([0,1,2,3]).long(),
                'board':T([4,5,6,7,8,9,10,11,12,13]).long(),
                'board_ranks':T([4,6,8,10,12]).long(),
                'board_suits':T([5,7,9,11,13]).long(),
                'street':T([14]).long(),
                'hero_position':T([15]).long(),
                'vil_position':T([16]).long(),
                'previous_action':T([17]).long(),
                'previous_betsize':T([18]).long(),
                'hero_stack':T([19]).long(),
                'villain_stack':T([20]).long(),
                'amnt_to_call':T([21]).long(),
                'pot_odds':T([22]).long(),
                'hand_board':T([0,1,2,3,4,5,6,7,8,9,10,11,12,13]).long(),
                'ordinal':T([14,15,16,17]).long(),
                'continuous':T([18,19,20,21,22]).long()
                },
            'observation':{
                'suit':T([1,3]).long(),
                'rank':T([0,2]).long(),
                'hand':T([0,1,2,3]).long(),
                'vil_hand':T([4,5,6,7]).long(),
                'vil_ranks':T([4,6]).long(),
                'vil_suits':T([5,7]).long(),
                'board':T([8,9,10,11,12,13,14,15,16,17]).long(),
                'board_ranks':T([8,10,12,14,16]).long(),
                'board_suits':T([9,11,13,15,17]).long(),
                'street':T([18]).long(),
                'hero_position':T([19]).long(),
                'vil_position':T([20]).long(),
                'previous_action':T([21]).long(),
                'previous_betsize':T([22]).long(),
                'hero_stack':T([23]).long(),
                'villain_stack':T([24]).long(),
                'amnt_to_call':T([25]).long(),
                'pot_odds':T([26]).long(),
                'hand_board':T([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).long(),
                'ordinal':T([18,19,20,21]).long(),
                'continuous':T([22,23,24,25,26]).long()
                }
            }
        self.rule_params['betsize'] = True
        self.rule_params['bettype'] = LimitTypes.NO_LIMIT
        self.rule_params['mask_dict'] = ACTION_MASKS[5]
        self.rule_params['action_dict'] = ACTION_ORDER
        self.rule_params['bets_per_street'] = 4

class Holdem(object):
    def __init__(self):
        H = BaseHoldem()
        self.rule_params = H.rule_params
        self.state_params = H.state_params
        self.state_params['stacksize'] = 5.
        self.state_params['pot']= 2.
        self.rule_params['betsizes'] = T([0.5,1.])
        self.starting_street = 3

class OmahaHI(object):
    def __init__(self):
        ## NOT IMPLEMENTED ##
        self.starting_street = 0
        K = Kuhn()
        self.rule_params = K.rule_params
        self.state_params = K.state_params
        self.rule_params['bettype'] = LimitTypes.POT_LIMIT

class Globals:
    GameTypeDict = {
        GameTypes.KUHN:Kuhn(),
        GameTypes.COMPLEXKUHN:ComplexKuhn(),
        GameTypes.BETSIZEKUHN:BetsizeKuhn(),
        GameTypes.HISTORICALKUHN:HistoricalKuhn(),
        GameTypes.HOLDEM:Holdem(),
        GameTypes.OMAHAHI:OmahaHI(),
    }
    POSITION_DICT = {Positions.SB:Positions.SB,Positions.BB:Positions.BB}
    POSITION_MAPPING = {'SB':0,'BB':1}
    PLAYERS_POSITIONS_DICT = {2:['SB','BB'],3:['SB','BB','BTN'],4:['SB','BB','CO','BTN'],5:['SB','BB','MP','CO','BTN'],6:['SB','BB','UTG','MP','CO','BTN']}
    HEADSUP_POSITION_DICT = {
        0:['SB','BB'],
        1:['BB','SB'],
        2:['BB','SB'],
        3:['BB','SB']
    }
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
    BETSIZE_DICT = {
        1 : T([1.]),
        2: T([0.5,1.]),
        3: T([0.8,0.9,1.]),
        4: T([0.25,0.5,0.75,1.]),
        5: T([0.2,0.4,0.6,0.8,1.]),
        11: T([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    }
    # Takes street as key
    ADDITIONAL_BOARD_CARDS = {
        0 : 0,
        1 : 3,
        2 : 1,
        3 : 1
    }
    INITIALIZE_BOARD_CARDS = {
        0 : 0,
        1 : 3,
        2 : 4,
        3 : 5
    }
    STREET_DICT = {
        0:Street.PREFLOP,
        1:Street.FLOP,
        2:Street.TURN,
        3:Street.RIVER
    }
    REVERSE_STREET_DICT = {v:k for k,v in STREET_DICT.items()}