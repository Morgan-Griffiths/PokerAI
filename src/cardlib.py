import ctypes

lib = ctypes.cdll.LoadLibrary("/Users/Shuza/Code/PokerAI/rusteval/target/release/librusteval.dylib")

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

# for converting an array to a c array for passing to rust
def long_array(arr):
    return (ctypes.c_long * len(arr))(*arr)


# example usage:

# a = [encode(c) for c in [[3,'h'],[3,'s'],[14,'s'],[6,'h']]]
# b = [encode(c) for c in [[2,'h'],[3,'c'],[14,'h'],[6,'c']]]
# board = [encode(c) for c in [[4,'c'],[5,'s'],[8,'d'],[11,'d'], [12,'c']]]
#
# print a, [decode(c) for c in a]
# print b, [decode(c) for c in b]
# print board, [decode(c) for c in board]
# print winner(a,b,board)
