
import ctypes
import itertools
import random

lib = ctypes.cdll.LoadLibrary("rusteval/target/release/librusteval.dylib")

def encode(cardstring):
    return lib.encode(ctypes.c_char_p(cardstring.encode('utf-8')))

def decode(encoded):
    decoded = lib.decode(encoded)
    return unichr(decoded >> 8) + unichr(decoded & 0xFF)

def decode_hand(hand):
    result = []
    for card in hand:
        result.append(decode(card))
    return result

def decode_rank(hand):
    return (hand >> 8) & 0xF

def carray(hand):
    return (ctypes.c_long * len(hand))(*hand)

lib.hand_vs_hand.restype = ctypes.c_float

def hand_vs_hand(hand1, hand2, deck, iterations):
    return lib.hand_vs_hand(carray(hand1), carray(hand2), carray(deck), iterations)

cards = []
for s in ['s','h','d','c']:
    for r in reversed(['2','3','4','5','6','7','8','9','T','J','Q','K','A']):
        cardstring = ''.join([r,s]).encode('utf-8')
        encoded = lib.encode(ctypes.c_char_p(cardstring))
        cards.append(encoded)

random.shuffle(cards)
hand1, hand2, deck = cards[:4], cards[4:8], cards[8:]
print decode_hand(hand1), decode_hand(hand2)
print hand_vs_hand(hand1, hand2, deck, 100000);

# rank = lib.rank(*hand)
# print ''.join(decode_hand(hand)), rank

# flushes = [0] * 7937
# straight_flush_rank = 1
# flush_rank = 323
# for hand in itertools.combinations(cards, 5):
#     if decode_rank(hand[0]) - decode_rank(hand[4]) == 4:
#         rank = straight_flush_rank
#         straight_flush_rank += 1
#     elif decode_rank(hand[0]) - decode_rank(hand[1]) == 9:
#         rank = 10
#     else:
#         rank = flush_rank
#         flush_rank += 1
#     lookup = lib.flush_lookup(*hand)
#     flushes[lookup] = rank
#
# print flushes

# unique5 = [0] * 7937
# straight_rank = 1600
# other_rank = 6186
# for hand in itertools.combinations(cards, 5):
#     if decode_rank(hand[0]) - decode_rank(hand[4]) == 4:
#         rank = straight_rank
#         straight_rank += 1
#     elif decode_rank(hand[0]) - decode_rank(hand[1]) == 9:
#         rank = 1609
#     else:
#         rank = other_rank
#         other_rank += 1
#     lookup = lib.flush_lookup(*hand)
#     unique5[lookup] = rank
#
# print unique5

# products = []
# ranks = []
# rank = 11
# # 4 of a kind
# for picks in itertools.permutations(cards, 2):
#     hand = [picks[0], picks[0], picks[0], picks[0], picks[1]]
#     product = lib.prime_lookup(*hand)
#     products.append((product, rank))
#     rank += 1
#
# # full house
# for picks in itertools.permutations(cards, 2):
#     hand = [picks[0], picks[0], picks[0], picks[1], picks[1]]
#     product = lib.prime_lookup(*hand)
#     products.append((product, rank))
#     rank += 1
#
# # trips
# rank = 1610
# for picks in itertools.permutations(cards, 3):
#     if picks[2] > picks[1]:
#         continue
#     hand = [picks[0], picks[0], picks[0], picks[1], picks[2]]
#     product = lib.prime_lookup(*hand)
#     products.append((product, rank))
#     rank += 1
#
# # 2 pair
# for picks in itertools.permutations(cards, 3):
#     if picks[1] > picks[0]:
#         continue
#     hand = [picks[0], picks[0], picks[1], picks[1], picks[2]]
#     product = lib.prime_lookup(*hand)
#     products.append((product, rank))
#     rank += 1
#
# # pair
# for picks in itertools.permutations(cards, 4):
#     if picks[2] > picks[1] or picks[3] > picks[2]:
#         continue
#     hand = [picks[0], picks[0], picks[1], picks[2], picks[3]]
#     product = lib.prime_lookup(*hand)
#     products.append((product, rank))
#     rank += 1
# products.sort(key=lambda tup: tup[0])
#
# product_list = []
# rank_list = []
# for p in products:
#     product_list.append(p[0])
#     rank_list.append(p[1])
# print product_list
# print rank_list

    # if decode_rank(hand[0]) - decode_rank(hand[4]) == 4:
    #     print decode_hand(hand), straight_rank
    #     rank = straight_rank
    #     straight_rank += 1
    # else:
    #     rank = other_rank
    #     other_rank += 1
    #
    # hand_index = lib.flush_lookup(*hand)
    # unique5[hand_index] = rank
# print unique5
