class Player(object):
    def __init__(self, hand, stack, action, position, bettotal, allin):
        self.hand = hand
        self.stack = stack
        self.action = action
        self.position = position
        self.bettotal = bettotal
        self.allin = allin
class Hand(object):
    def __init__(self, cards, machinehand, riverstrength, rivervalues):
        self.cards = cards
        self.machinehand = machinehand
        self.riverstrength = riverstrength
        self.rivervalues = rivervalues
class Board(object):
    def __init__(self, cards, machinehand):
        self.cards = cards
        self.machinehand = machinehand
class Rank(object):
    def __init__(self, cards, suits, connect, highcard, nut, equity):
        self.cards = cards
        self.suits = suits
        self.connect = connect
        self.highcard = highcard
        self.nut = nut
        self.equity = equity
#dictionary
handvalues = {"A" : 14, "K" : 13, "Q" : 12, "J" : 11, "T" : 10,"9" : 9, "8" : 8,
"7" : 7, "6" : 6, "5" :5, "4" : 4, "3" : 3, "2" : 2}

DECK = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],[5,'s'],[4,'s'],[3,'s'],[2,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],[5,'h'],[4,'h'],[3,'h'],[2,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],[5,'c'],[4,'c'],[3,'c'],[2,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d'],[5,'d'],[4,'d'],[3,'d'],[2,'d']]
