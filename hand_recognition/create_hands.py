import numpy as np
from itertools import combinations,permutations,combinations_with_replacement,product

# Hardcoded hero and board combos
HERO_COMBINATIONS = np.array([h for h in combinations(range(0,5), 2)])
BOARD_COMBINATIONS = []
for combo in HERO_COMBINATIONS:
    places = set(range(0,5))
    places -= set(combo)
    BOARD_COMBINATIONS.append(list(places))
    
BOARD_COMBINATIONS = np.array(BOARD_COMBINATIONS)

# CORRECT
def straight_flushes():
    flushes = []
    wheel_ranks = np.array([2,3,4,5,14])
    for suit in range(1,5):
        wheel_suits = np.full(5,suit)
        wheel = np.stack((wheel_ranks,wheel_suits))
        flushes.append(wheel)
    for top in range(2+5,14+2):
        straight = np.arange(top-5,top)
        for suit in range(1,5):
            hand = np.stack((straight,np.full(5,suit)))
            flushes.append(hand)
    return flushes

# CORRECT
def quads():
    quad_hands = []
    for top in range(2,14+1): 
        quad_ranks = np.full(4,top)
        quad_suits = np.arange(1,5)
        remaining = set(range(2,15))
        remaining.remove(top)
        for other_rank in remaining:
            for other_suit in range(1,5):
                ranks = np.hstack((quad_ranks,other_rank))
                suits = np.hstack((quad_suits,other_suit))
                hand = np.stack((ranks,suits))
                quad_hands.append(hand)
    return quad_hands

# CORRECT
def full_houses():
    hands = []
    for trip in range(2,14+1):
        trips = np.full(3,trip)
        for trip_suits in combinations(range(1,5),3):
            pair_range = set(range(2,15))
            pair_range.remove(trip)
            for pair in pair_range:
                for pair_suits in combinations(range(1,5),2):
                    pairs = np.full(2,pair)
                    ranks = np.hstack((trips,pairs))
                    suits = np.hstack((trip_suits,pair_suits))
                    hand = np.stack((ranks,suits))
                    hands.append(hand)
    return hands

# CORRECT
def flushes():
    hands = []
    rank_combos = combinations(range(2,15),5)
    for ranks in rank_combos:
        rank_set = set(ranks)
        if rank_set == set([14,2,3,4,5]) or rank_set == set(np.arange(np.min(ranks),np.max(ranks)+1)):
            continue
        else:
            for suit in range(1,5):
                suits = np.full(5,suit)
                hand = np.stack((ranks,suits))
                hands.append(hand)
    return hands

# CORRECT
def straights():
    hands = []
    # Special case wheel
    wheel = np.array([2,3,4,5,14])
    suit_combos = product(np.arange(1,5),repeat=5)
    for suits in suit_combos:
        if len(set(suits)) == 1:
            continue
        else:
            hand = np.stack((wheel,suits))
            hands.append(hand)
    # Reg cases
    for top in range(2+5,14+2):
        straight = np.arange(top-5,top)
        suit_combos = product(np.arange(1,5),repeat=5)
        for suits in suit_combos:
            if len(set(suits)) == 1:
                continue
            else:
                hand = np.stack((straight,suits))
                hands.append(hand)
    return hands

# CORRECT
def trips():
    hands = []
    for trip in range(2,14+1):
        trips = np.full(3,trip)
        remaining_ranks = set(range(2,15))
        remaining_ranks.remove(trip)
        rank_combos = combinations(remaining_ranks,2)
        for other_ranks in rank_combos:
            trip_suit_combos = combinations(range(1,5),3)
            for trip_suits in trip_suit_combos:
                other_suit_combos = product(np.arange(1,5),repeat=2)
                for other_suits in other_suit_combos:
                    ranks = np.hstack((trips,other_ranks))
                    suits = np.hstack((trip_suits,other_suits))
                    hand = np.stack((ranks,suits))
                    hands.append(hand)
    return hands

# CORRECT
def two_pairs():
    hands = []
    for f_pair in range(2,14):
        first_pair = np.full(2,f_pair)
        for s_pair in range(f_pair+1,15):
            second_pair = np.full(2,s_pair)
            remaining_ranks = set(range(2,15))
            remaining_ranks.remove(f_pair)
            remaining_ranks.remove(s_pair)
            for l_rank in remaining_ranks:
                ranks = np.hstack((first_pair,second_pair,l_rank))
                f_suits = combinations(range(1,5),2)
                for first_suits in f_suits:
                    s_suits = combinations(range(1,5),2)
                    for second_suits in s_suits:
                        for l_suit in range(1,5):
                            suits = np.hstack((first_suits,second_suits,l_suit))
                            hand = np.stack((ranks,suits))
                            hands.append(hand)
    return hands

# CORRECT
def one_pairs():
    hands = []
    for f_pair in range(2,15):
        first_pair = np.full(2,f_pair)
        remaining_ranks = set(range(2,15))
        remaining_ranks.remove(f_pair)
        other_rank_combos = combinations(remaining_ranks,3)
        for o_rank in other_rank_combos:
            pair_suit_combos = combinations(range(1,5),2)
            for p_suit in pair_suit_combos:
                remaining_suits = product(range(1,5),repeat=3)
                for r_suits in remaining_suits:
                    suits = np.hstack((p_suit,r_suits))
                    ranks = np.hstack((first_pair,o_rank))
                    hand = np.stack((ranks,suits)) 
                    hands.append(hand)
    return hands

# CORRECT
def high_cards():
    hands = []
    rank_combos = combinations(range(2,15),5)
    for ranks in rank_combos:
        rank_set = set(ranks)
        if rank_set == set([14,2,3,4,5]) or rank_set == set(np.arange(np.min(ranks),np.max(ranks)+1)):
            continue
        else:
            suit_combos = product(range(1,5),repeat=5)
            for suits in suit_combos:
                if len(set(suits)) == 1:
                    continue
                else:
                    hand = np.stack((ranks,suits))
                    hands.append(hand)
    return hands

def sort_hand(cards):
    """
    Takes 5 card hand of shape: (2,5),
    returns sorted by hand+board, shape (2,5)
    """
    hero_ranks = cards[0,:2]
    hero_suits = cards[1,:2]
    board_ranks = cards[0,2:]
    board_suits = cards[1,2:]
    hero_index = np.argsort(hero_suits)
    board_index = np.argsort(board_suits)
    hero_ranks = hero_ranks[hero_index]
    hero_suits = hero_suits[hero_index]
    board_ranks = board_ranks[board_index]
    board_suits = board_suits[board_index]
    hero_index = np.argsort(hero_ranks)
    board_index = np.argsort(board_ranks)
    hero_ranks = hero_ranks[hero_index]
    hero_suits = hero_suits[hero_index]
    board_ranks = board_ranks[board_index]
    board_suits = board_suits[board_index]
    hero = np.vstack([hero_ranks,hero_suits])
    board = np.vstack([board_ranks,board_suits])
    sorted_cards = np.concatenate([hero,board],axis=-1)
    return sorted_cards

def hero_5_cards(cards):
    """
    Takes a 5 card combination, and extrapolates out to all the sorted 2 cards, sorted 3 cards, hand+board combos.
    Effectively multiplies each combo by 10
    """
    hands = []
    hero_rank_combos = cards[0][HERO_COMBINATIONS]
    hero_suit_combos = cards[1][HERO_COMBINATIONS]
    board_rank_combos = cards[0][BOARD_COMBINATIONS]
    board_suit_combos = cards[1][BOARD_COMBINATIONS]
    for i in range(len(board_suit_combos)):
        ranks = np.hstack((hero_rank_combos[i],board_rank_combos[i]))
        suits = np.hstack((hero_suit_combos[i],board_suit_combos[i]))
        hand = np.stack((ranks,suits))
        hands.append(np.transpose(hand))
    return hands