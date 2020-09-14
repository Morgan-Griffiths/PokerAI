import ctypes
from os import path
from sys import platform, argv

release_dir = path.normpath(path.join(
    path.abspath(__file__),
    "../../../rusteval/target/release"
))

if platform == "linux" or platform == "linux2":
    # linux
    lib = ctypes.cdll.LoadLibrary(path.join(release_dir, "librusteval.so"))
elif platform == "darwin":
    # OS X
    lib = ctypes.cdll.LoadLibrary(path.join(release_dir, "librusteval.dylib"))
else:
    raise OSError('Unknown OS')


# takes [2,'s'] or [2,1] and returns the cactus kev encoding for that card
def encode(card):
    """
    Ranks 2-14
    Suits 1-5
    """
    suitnum = card[1]
    if isinstance(card[1],str):
        suitnum = {'s': 1, 'h': 2,'d': 3,'c': 4}.get(card[1], 0)
    return lib.encode(ctypes.c_byte(card[0] - 2), ctypes.c_byte(suitnum - 1))

# takes a cactus kev encoded card and returns a list like [2,'s']
def decode(encoded):
    rank = ((encoded >> 8) & 0xF) + 2
    suit = ['','s','h','','d','','','','c'][(encoded >> 12) & 0xF]
    return [rank, suit]

# takes two 4-card hands and a 5-card board and
# returns 1 for hand1 wins, -1 for hand2 wins, or 0 for tie
def winner(hand1, hand2, board):
    return lib.winner(long_array(hand1), long_array(hand2), long_array(board))

# takes two 2-card hands and a 5-card board and
# returns 1 for hand1 wins, -1 for hand2 wins, or 0 for tie
def holdem_winner(hand1, hand2, board):
    return lib.holdem_winner(long_array(hand1), long_array(hand2), long_array(board))

# takes a 4 card hand and a 5 card board and returns the best rank of the 60 possible combinations
def hand_rank(hand, board):
    return lib.hand_with_board_rank(long_array(hand), long_array(board))

def holdem_hand_rank(hand, board):
    return lib.holdem_hand_with_board_rank(long_array(hand), long_array(board))
# for converting an array to a c array for passing to rust
def long_array(arr):
    return (ctypes.c_long * len(arr))(*arr)

def rank(hand):
    return lib.rank(*hand)

# example usage:
"""
SUITS: Accepts strings [s,c,h,d], or numbers 1-4.
RANKS: Accepts numbers 2-14 inclusive.
>>> hand = [[7, 's'], [5, 'c'], [14, 'h'], [10, 'h']]
>>> hand_en = [encode(c) for c in hand]
>>> hand2 = [[14, 'c'], [2, 's'], [2, 'd'], [11, 's']]
>>> hand2_en = [encode(c) for c in hand]
>>> board = [[10, 'c'], [2, 'h'], [4, 'c'], [13, 'c'], [4, 'h']]
>>> en_board = [encode(c) for c in board]
>>> winner(hand_en,hand2_en,en_board)
>>> rank(hand_en)
>>> hand_rank(hand_en,en_board)
>>> holdem_hand = [[14, 'c'], [2, 's']]
>>> holdem_hand_en = [encode(c) for c in holdem_hand]
>>> holdem_hand_rank(holdem_hand_en,en_board)
"""