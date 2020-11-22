import os
from pathlib import Path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=
        """
        Setup folders and generate data
        """)

    parser.add_argument('--mkfolders','-m',
                        dest='mkfolders',
                        action='store_true'
                        help='Make folders')
    parser.add_argument('--mkdata','-d',
                        dest='mkdata',
                        action='store_true'
                        help='Gen data')
    parser.set_defaults(mkfolders=True)
    parser.set_defaults(mkdata=False)

    args = parser.parse_args()

    if args.mkfolders:
        # Make dirs
        Path('poker/assets').mkdir(parents=True, exist_ok=True)
        Path('poker/checkpoints/frozen_layers').mkdir(parents=True, exist_ok=True)
        Path('poker/checkpoints/baselines').mkdir(parents=True, exist_ok=True)
        Path('poker/checkpoints/training_run/actor').mkdir(parents=True, exist_ok=True)
        Path('poker/checkpoints/training_run/critic').mkdir(parents=True, exist_ok=True)
        Path('poker/checkpoints/production').mkdir(parents=True, exist_ok=True)
        Path('kuhn/assets').mkdir(parents=True, exist_ok=True)
        Path('kuhn/checkpoints/').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/assets/').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/checkpoints/').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/checkpoints/multiclass_categorization').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/checkpoints/regression').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/checkpoints/binary_categorization').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/binary_categorization/blockers/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/binary_categorization/blockers/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/fivecard/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/fivecard/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/ninecard/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/ninecard/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/handranksnine/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/handranksnine/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/handranksfive/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/handranksfive/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/smalldeck/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/smalldeck/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/flush/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/multiclass_categorization/flush/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/regression/tencard/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/regression/tencard/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/regression/thirteencard/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/regression/thirteencard/val').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/regression/partial/train').mkdir(parents=True, exist_ok=True)
        Path('hand_recognition/data/regression/partial/val').mkdir(parents=True, exist_ok=True)
    if args.mkdata:
        # Build data
        os.system('python hand_recognition/build_dataset.py -d fivecard -O data/multiclass_categorization/fivecard')
        os.system('python hand_recognition/build_dataset.py -d ninecard -O data/multiclass_categorization/ninecard')
        os.system('python hand_recognition/build_dataset.py -d tencard -O data/regression/tencard')
        os.system('python hand_recognition/build_dataset.py -d thirteencard -O data/regression/thirteencard')
        os.system('python hand_recognition/build_dataset.py -d blockers -O data/binary_categorization/blockers')
        os.system('python hand_recognition/build_dataset.py -d handranksnine -O data/multiclass_categorization/handranksnine')
        os.system('python hand_recognition/build_dataset.py -d handranksfive -O data/multiclass_categorization/handranksfive')
        os.system('python hand_recognition/build_dataset.py -d partial -O data/regression/partial')