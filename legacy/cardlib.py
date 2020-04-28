import ctypes

lib = ctypes.cdll.LoadLibrary("./rusteval/target/release/librusteval.dylib")

# takes [2,'s'] and returns the cactus kev encoding for that card
def encode(card):
    suitnum = {'s': 0, 'h': 1,'d': 2,'c': 3}.get(card[1], 0)
    return lib.encode(ctypes.c_byte(card[0] - 2), ctypes.c_byte(suitnum))

# takes a cactus kev encoded card and returns a list like [2,'s']
def decode(encoded):
    rank = ((encoded >> 8) & 0xF) + 2
    suit = ['','s','h','','d','','','','c'][(encoded >> 12) & 0xF]
    return [rank, suit]

# takes two 4-card hands and a 5-card board and
# returns 1 for hand1 wins, -1 for hand2 wins, or 0 for tie
def winner(hand1, hand2, board):
    return lib.winner(long_array(hand1), long_array(hand2), long_array(board))

# takes a 4 card hand and a 5 card board and returns the best rank of the 60 possible combinations
def hand_rank(hand, board):
    return lib.hand_with_board_rank(long_array(hand), long_array(board))

# for converting an array to a c array for passing to rust
def long_array(arr):
    return (ctypes.c_long * len(arr))(*arr)


# example usage:

h3 = [[7, 's'], [5, 'c'], [14, 'h'], [10, 'h']]
h4 = [[14, 'c'], [2, 's'], [2, 'd'], [11, 's']]
board2 = [[10, 'c'], [2, 'h'], [4, 'c'], [13, 'c'], [4, 'h']]
en_h3 = [encode(c) for c in h3]
en_h4 = [encode(c) for c in h4]
en_board2 = [encode(c) for c in board2]
print(en_h3,en_h4,en_board2)
print(winner(en_h3,en_h4,en_board2))
print(hand_rank(en_h3, en_board2), hand_rank(en_h4, en_board2))