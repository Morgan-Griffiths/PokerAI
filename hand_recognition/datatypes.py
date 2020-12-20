import numpy as np

class Modes(object):
    TRAIN = 'train'
    EXAMINE = 'examine'
    VALIDATE = 'validate'
    MULTITRAIN = 'multitrain'

class DataTypes(object):
    THIRTEENCARD = 'thirteencard'
    NINECARD = 'ninecard'
    FIVECARD = 'fivecard'
    TENCARD = 'tencard'
    BLOCKERS = 'blockers'
    PARTIAL = 'partial'
    HANDRANKSFIVE = 'handranksfive'
    HANDRANKSNINE = 'handranksnine'
    HANDRANKSTHIRTEEN = 'handranksthirteen'
    FLATDECK = 'flatdeck'
    SMALLDECK = 'smalldeck'
    FLUSH = 'flush'

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
        DataTypes.HANDRANKSNINE : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.HANDRANKSFIVE : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.HANDRANKSTHIRTEEN : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.FLATDECK : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.SMALLDECK : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.FLUSH : LearningCategories.MULTICLASS_CATEGORIZATION,
        DataTypes.TENCARD : LearningCategories.REGRESSION,
        DataTypes.THIRTEENCARD : LearningCategories.REGRESSION,
        DataTypes.PARTIAL : LearningCategories.REGRESSION,
        DataTypes.BLOCKERS : LearningCategories.BINARY_CATEGORIZATION
    }
    SUIT_DICT = {
        1:'s',
        2:'h',
        3:'d',
        4:'c'
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
        DataTypes.HANDRANKSFIVE:7463,
        DataTypes.HANDRANKSNINE:7463,
        DataTypes.HANDRANKSTHIRTEEN:7463,
        DataTypes.SMALLDECK:1820,
        DataTypes.FLATDECK:7463,
        DataTypes.FLUSH:7463,
        DataTypes.BLOCKERS:1,
        DataTypes.THIRTEENCARD:1,
        DataTypes.TENCARD:1,
        DataTypes.PARTIAL:1
    }
    LABEL_DICT = {
        DataTypes.BLOCKERS :{0:'No Blocker',1:'Blocker'},
        DataTypes.FIVECARD : HAND_TYPE_DICT,
        DataTypes.NINECARD : HAND_TYPE_DICT,
        DataTypes.HANDRANKSNINE : {i:i for i in range(1,7463)},
        DataTypes.HANDRANKSFIVE : {i:i for i in range(1,7463)},
        DataTypes.HANDRANKSTHIRTEEN : {i:i for i in range(1,7463)},
        DataTypes.TENCARD : {-1:'Player 2 wins',0:'Tie',1:'Player 1 wins'},
        DataTypes.THIRTEENCARD : {-1:'Player 2 wins',0:'Tie',1:'Player 1 wins'},
        DataTypes.PARTIAL : {-1:'Player 2 wins',0:'Tie',1:'Player 1 wins'},
        DataTypes.SMALLDECK: {i:i for i in range(0,1820)},
        DataTypes.FLATDECK: {i:i for i in range(1,7463)},
        DataTypes.FLUSH: {i:i for i in range(1,7463)},
    }
    TARGET_SET = {
        DataTypes.THIRTEENCARD:set(range(-1,2)),
        DataTypes.TENCARD:set(range(-1,2)),
        DataTypes.PARTIAL:set(range(-1,2)),
        DataTypes.HANDRANKSFIVE:set(range(1,7463)),
        DataTypes.HANDRANKSNINE:set(range(1,7463)),
        DataTypes.HANDRANKSTHIRTEEN:set(range(1,7463)),
        DataTypes.NINECARD:set(range(9)),
        DataTypes.FIVECARD:set(range(9)),
        DataTypes.BLOCKERS:set(range(2)),
        DataTypes.SMALLDECK:set(range(0,1820)),
        DataTypes.FLATDECK:set(range(1,7463)),
        DataTypes.FLUSH:set(range(1,7463)),
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
        0:lambda : np.random.choice(np.arange(1,11)),
        1:lambda : np.random.choice(np.arange(11,167)),
        2:lambda : np.random.choice(np.arange(167,323)),
        3:lambda : np.random.choice(np.arange(323,1600)),
        4:lambda : np.random.choice(np.arange(1600,1610)),
        5:lambda : np.random.choice(np.arange(1610,2468)),
        6:lambda : np.random.choice(np.arange(2468,3326)),
        7:lambda : np.random.choice(np.arange(3326,6186)),
        8:lambda : np.random.choice(np.arange(6186,7463)),
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
    HIGH = 5
    LOW = 1