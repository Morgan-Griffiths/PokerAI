import os

# Make dirs
os.makedirs('assets')
os.makedirs('checkpoints/hand_categorization')
os.makedirs('data/hand_types/test')
os.makedirs('data/hand_types/train')
os.makedirs('data/predict_winner')
# Build data
os.system('python card.py --build True --datatype random')
os.system('python card.py --build True --datatype handtype')
os.system('python card.py --build True --datatype handtype -m 5000 -bi 3000000 -O data/hand_types/train')
