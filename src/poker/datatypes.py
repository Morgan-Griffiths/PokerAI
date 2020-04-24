from torch import Tensor as T

class GameTypes:
    KUHN = 'kuhn'
    COMPLEXKUHN = 'ComplexKuhn'
    HOLDEM = 'holdem'
    OMAHAHI = 'omaha_hi'

class Kuhn:
    rule_params = {
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
        'raises_per_street' :0
        }

class ComplexKuhn:
    rule_params = Kuhn.rule_params
    rule_params['mask_dict'] = {
            0:T([1,1,0,0,0]),
            1:T([0,0,1,1,1]),
            2:T([0,0,0,0,0]),
            3:T([0,0,0,0,0]),
            4:T([0,0,1,1,0]),
            5:T([1,1,0,1,0])
            }
    rule_params['action_dict'] = {0:'check',1:'bet',2:'call',3:'fold',4:'raise'}
    rule_params['betsize_dict'] = {0:0,1:1,2:1,3:0,4:2}
    rule_params['bets_per_street'] = 2
    rule_params['raises_per_street'] = 1

class Holdem:
    pass

class OmahaHI:
    pass


class Globals:
    GameTypeDict = {
        GameTypes.KUHN:Kuhn,
        GameTypes.COMPLEXKUHN:ComplexKuhn,
        GameTypes.HOLDEM:Holdem,
        GameTypes.OMAHAHI:OmahaHI,
    }
