import os
import time
import datatypes as dt
from build import CardDataset
from data_utils import load_data,save_all,save_trainset,save_valset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Train and evaluate networks on card representations\n\n
        use ```python cards.py -M examine``` to check handtype probabilities
        use ```python cards.py -d random``` to train on predicting winners
        use ```python cards.py``` to train on predicting handtypes
        """)

    parser.add_argument('-d','--datatype',
                        default='handtype',type=str,
                        metavar=f"[{dt.DataTypes.THIRTEENCARD},{dt.DataTypes.TENCARD},{dt.DataTypes.NINECARD},{dt.DataTypes.FIVECARD},{dt.DataTypes.PARTIAL},{dt.DataTypes.BLOCKERS},{dt.DataTypes.HANDRANKSFIVE},{dt.DataTypes.HANDRANKSNINE}]",
                        help='Which dataset to train or build')
    parser.add_argument('-O','--datapath',
                        help='Local path to save data',
                        default=None,type=str)
    parser.add_argument('--trainsize',
                        help='Size of train set',
                        default=45000,type=int)
    parser.add_argument('--valsize',
                        help='Size of val set',
                        default=9000,type=int)
    parser.add_argument('--testsize',
                        help='Size of test set',
                        default=9000,type=int)
    parser.add_argument('-r','--random',dest='randomize',
                        help='Randomize the dataset. (False -> the data is sorted)',
                        default=True,type=bool)
    parser.add_argument('--encode',metavar=[dt.Encodings.TWO_DIMENSIONAL,dt.Encodings.THREE_DIMENSIONAL],
                        help='Encoding of the cards: 2d -> Hand (4,2). 3d -> Hand (4,13,4)',
                        default=dt.Encodings.TWO_DIMENSIONAL,type=str)

    args = parser.parse_args()

    print('OPTIONS',args)
    
    print(f'Building {args.datatype} dataset')
    learning_category = dt.Globals.DatasetCategories[args.datatype]
    if args.datapath == None:
        save_dir = os.path.join('data',learning_category,args.datatype)
    else:
        save_dir = args.datapath

    dataset_params = {
        'train_set_size':args.trainsize,
        'val_set_size':args.valsize,
        'test_set_size':args.testsize,
        'encoding':args.encode,
        'save_dir':save_dir,
        'datatype':args.datatype,
        'randomize':args.randomize,
        'learning_category':learning_category
    }

    # Check hand generation
    # dataset = CardDataset(dataset_params)
    # while 1:
    #     handtype = input('Enter in int 0-8 to pick handtype')
    #     hand = dataset.create_handtypes(int(handtype))
    #     print(f'Hand {hand}, Category {handtype}')
    tic = time.time()
    dataset = CardDataset(dataset_params)
    if dataset_params['datatype'] == dt.DataTypes.SMALLDECK:
        trainX,trainY = dataset.build_smalldeck()
        valX,valY = dataset.build_smalldeck()
        save_all(trainX,trainY,valX,valY,dataset_params['save_dir'],y_dtype='int32')
    elif dataset_params['datatype'] == dt.DataTypes.HANDRANKSNINE:
        trainX,trainY = dataset.build_hand_ranks_nine(200)
        valX,valY = dataset.build_hand_ranks_nine(20)
        save_all(trainX,trainY,valX,valY,dataset_params['save_dir'],y_dtype='int32')
    elif dataset_params['datatype'] == dt.DataTypes.HANDRANKSFIVE:
        trainX,trainY = dataset.build_hand_ranks_five()
        save_trainset(trainX,trainY,dataset_params['save_dir'],y_dtype='int32')
        del trainX
        del trainY
        valX,valY = dataset.build_hand_ranks_five(valset=True)
        save_valset(valX,valY,dataset_params['save_dir'],y_dtype='int32')
    elif learning_category == dt.LearningCategories.MULTICLASS_CATEGORIZATION:
        dataset.build_hand_classes(dataset_params)
    elif learning_category == dt.LearningCategories.REGRESSION:
        trainX,trainY,valX,valY = dataset.generate_dataset(dataset_params)
        save_all(trainX,trainY,valX,valY,dataset_params['save_dir'],y_dtype='int8')
    elif learning_category == dt.LearningCategories.BINARY_CATEGORIZATION:
        trainX,trainY = dataset.build_blockers(dataset_params['train_set_size'])
        valX,valY = dataset.build_blockers(dataset_params['val_set_size'])
        save_all(trainX,trainY,valX,valY,dataset_params['save_dir'],y_dtype='uint8')
    else:
        raise ValueError(f'{args.datatype} datatype not understood')
    print(f'dataset took {(time.time() - tic) / 60} Minutes')