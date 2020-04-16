import os

# Make dirs
# os.makedirs('assets')
# os.makedirs('checkpoints/hand_categorization')
# os.makedirs('data/hand_types/test')
# os.makedirs('data/hand_types/train')
# os.makedirs('data/predict_winner')
# Build data
os.system('python cards.py -M build -d random -O data/') #data/predict_winner
os.system('python cards.py -M build -d handtype -O data/hand_types/')
print('final')
os.system('python cards.py -M build -d handtype -m 5000 -bi 3000000 -O data/hand_types/train')
