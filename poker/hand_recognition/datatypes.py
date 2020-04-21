import numpy as np

class Modes(object):
    TRAIN = 'train'
    BUILD = 'build'
    EXAMINE = 'examine'

class DataTypes(object):
    THIRTEENCARD = 'thirteencard'
    NINECARD = 'ninecard'
    FIVECARD = 'fivecard'
    TENCARD = 'tencard'
    BLOCKERS = 'blockers'
    PARTIAL = 'partial'
    HANDRANKS = 'handranks'

class Encodings(object):
    TWO_DIMENSIONAL = '2d'
    THREE_DIMENSIONAL = '3d'

class LearningCategories(object):
    MULTICLASS_CATEGORIZATION = 'multiclass_categorization'
    BINARY_CATEGORIZATION = 'binary_categorization'
    REGRESSION = 'regression'

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
    DatasetCategories = {
        DataTypes.FIVECARD : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.NINECARD : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.HANDRANKS : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.TENCARD : LearningCategories.REGRESSION,
        DataTypes.THIRTEENCARD : LearningCategories.REGRESSION,
        DataTypes.PARTIAL : LearningCategories.REGRESSION,
        DataTypes.BLOCKERS : LearningCategories.BINARY_CATEGORIZATION
    }
    SUIT_DICT = {
        0:'s',
        1:'h',
        2:'d',
        3:'c'
    }
    REVERSE_SUIT_DICT = {v:k for k,v in SUIT_DICT.items()}
    INPUT_SET_DICT = {
        'train' : 'train_set_size',
        'test' : 'test_set_size',
        'val' : 'val_set_size'
    }
    ACTION_SPACES = {
        DataTypes.FIVECARD:9,
        DataTypes.NINECARD:9,
        DataTypes.HANDRANKS:7463,
        DataTypes.BLOCKERS:1,
        DataTypes.THIRTEENCARD:1,
        DataTypes.TENCARD:1,
        DataTypes.PARTIAL:1
    }
    LABEL_DICT = {
        DataTypes.BLOCKERS :{0:'No Blocker',1:'Blocker'},
        DataTypes.FIVECARD : HAND_TYPE_DICT,
        DataTypes.NINECARD : HAND_TYPE_DICT,
        DataTypes.HANDRANKS : {i:i for i in range(7463)},
        DataTypes.TENCARD : {-1:'Player 2 wins',0:'Tie',1:'Player 1 wins'},
        DataTypes.THIRTEENCARD : {-1:'Player 2 wins',0:'Tie',1:'Player 1 wins'},
        DataTypes.PARTIAL : {-1:'Player 2 wins',0:'Tie',1:'Player 1 wins'}
    }
    # 7462-6185 High card
    # 6185-3325 Pair
    # 3325-2467 2Pair
    # 2467-1609 Trips
    # 1609-1599  Stright
    # 1599-322 Flush
    # 322-166  FH
    # 166-10 Quads
    # 10-0 Str8 flush
    HAND_STRENGTH_SAMPLING = {
        0:np.random.choice(np.arange(10)),
        1:np.random.choice(np.arange(10,166)),
        2:np.random.choice(np.arange(166,322)),
        3:np.random.choice(np.arange(322,1599)),
        4:np.random.choice(np.arange(1599,1609)),
        5:np.random.choice(np.arange(1609,2467)),
        6:np.random.choice(np.arange(2467,3325)),
        7:np.random.choice(np.arange(3325,6185)),
        8:np.random.choice(np.arange(6185,7463)),
    }
    HAND_CATEGORY_EXAMPLES = {
                0:10,
                1:156,
                2:156,
                3:1277,
                4:10,
                5:858,
                6:858,
                7:2860,
                8:1277
            }
"""
High is noninclusive
"""
class RANKS(object):
    HIGH = 15
    LOW = 2

class SUITS(object):
    HIGH = 4
    LOW = 0