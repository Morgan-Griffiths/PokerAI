import ctypes
from os import path
from sys import platform, argv

release_dir = path.normpath(path.join(
    path.abspath(__file__),
    "../../rusteval/target/release"
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

def test():
    # Omaha
    hand = [[14, 'c'], [2, 's'], [2, 'd'], [11, 's'],[5,'c']]
    hand_en = [encode(c) for c in hand]
    h3 = [[7, 's'], [5, 'c'], [14, 'h'], [10, 'h']]
    h4 = [[14, 'c'], [2, 's'], [2, 'd'], [11, 's']]
    board2 = [[10, 'c'], [2, 'h'], [4, 'c'], [13, 'c'], [4, 'h']]
    en_h3 = [encode(c) for c in h3]
    en_h4 = [encode(c) for c in h4]
    en_board2 = [encode(c) for c in board2]
    print(en_h3,en_h4,en_board2)
    print(winner(en_h3,en_h4,en_board2))
    print('Omaha hand+board ranks',hand_rank(en_h3, en_board2), hand_rank(en_h4, en_board2))
    print('Omaha hand rank',rank(hand_en))
    # h3 = [[11, 's'], [10, 'c'], [3, 'h'], [3, 'h']]
    # h4 = [[11, 'c'], [10, 's'], [4, 'd'], [5, 's']]
    # board2 = [[14, 'c'], [13, 'h'], [12, 'c'], [2, 'h'], [4, 's']]
    board2 = [[14,0],[13,1],[12,2],[2,2],[2,3]]
    h3 = [[11,3],[10,3],[3,2],[3,3]]
    h4 = [[11,2],[10,2],[4,0],[4,3]]
    en_h3 = [encode(c) for c in h3]
    en_h4 = [encode(c) for c in h4]
    en_board2 = [encode(c) for c in board2]
    print(en_h3,en_h4,en_board2)
    print(winner(en_h3,en_h4,en_board2))
    print('2 Omaha hand+board ranks',hand_rank(en_h3, en_board2), hand_rank(en_h4, en_board2))
    print('2 Omaha hand rank',rank(hand_en))
    # Holdem
    holdem_hand = [[14, 'c'], [2, 's']]
    holdem_hand2 = [[13, 'c'], [5, 's']]
    holdem_board = [[10, 'c'], [2, 'c'], [4, 'c'], [13, 'c'], [4, 'h']]
    en_hh = [encode(c) for c in holdem_hand]
    en_hh2 = [encode(c) for c in holdem_hand2]
    en_board = [encode(c) for c in holdem_board]
    print(holdem_winner(en_hh,en_hh2,en_board))
    print(holdem_hand_rank(en_hh,en_board))

if __name__ == "__main__":
    test()