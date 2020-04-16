
class Modes(object):
    TRAIN = 'train'
    BUILD = 'build'
    EXAMINE = 'examine'

class DataTypes(object):
    HANDTYPE = 'handtype'
    RANDOM = 'random'
    FIVECARD = 'fivecard'

class Globals(object):
    HAND_TYPE_DICT = {
                0:'Straight_flush',
                1:'Four_of_a_kind',
                2:'Full_house',
                3:'Flush',
                4:'Straight',
                5:'Three_of_a_kind',
                6:'Two_pair',
                7:'One_pair',
                8:'High_card'
            }
    HAND_TYPE_FILE_DICT = {'Hand_type_'+v:k for k,v in HAND_TYPE_DICT.items()}