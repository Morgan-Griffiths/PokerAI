import os

# Make dirs
os.makedirs('assets')
os.makedirs('checkpoints/hand_categorization')
os.makedirs('data/hand_types/test')
os.makedirs('data/hand_types/train')
os.makedirs('data/predict_winner')
os.makedirs('data/fivecard')
# Build data
os.system('python cards.py -M build -d random -O data/predict_winner')
os.system('python cards.py -M build -d handtype -O data/hand_types/test')
os.system('python cards.py -M build -d handtype -m 5000 -bi 3000000 -O data/hand_types/train')
os.system('python cards.py -M build -d fivecard -O data/fivecard')
