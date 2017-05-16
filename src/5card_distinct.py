import numpy as np
import random
import operator
import functools
import Classes as cl
import collections
import pickle
import itertools as it
import os.path
import copy

"""
need to get rid of the duplicates. Need to finish connectedness and equity functions.
Nuts i think can wait.
Then move on to setting up 1 street game. and setting up LSTMs and CFR+
"""

#assign the actual river flush % to suit
# FD chance | -mean/std | -mean/minmax
#DS: FD(+bfd) = 23.5% | 1.695 | 0.546
#AABC: FD = 11.77% | 0.147 | 0.047
#AAAB: FD = 9.89% | -0.101 | -0.032
#AAAA: FD = 8.12% | -0.334 | -0.107
#rainbow = 0 | -1.407 | -0.453
def suit(hand):
    suited = 0
    suits = map(lambda x:x[1],hand)
    #print suits
    #print collections.Counter(suits)
    suitedness = [count for item, count in collections.Counter(suits).items()]
    suitedness.sort(reverse=True)
    #print "suited",suitedness
    if suitedness[0] == 1:
        suited = -0.453
        #print "rainbow"
    elif suitedness[0] == 2:
        if suitedness[1] == 2:
            suited = 0.546
            #print "double suited"
        else:
            #print "2 single suited"
            suited = 0.047
    elif suitedness[0] == 3:
        suited = -0.032
        #print "3"
    else:
        suited = -0.107
        #print "mono"
    return suited

#returns how many unique STs a hand can make
#each straight uses 2 cards. So each ST board can be identified by 3 cards
#perhaps start at AKQJT and work downwards? Each board only counts the highest
#ST. Problem is, it doesn't fully count number of straights.
#Could do the triple for loop and check for STs with evan's code?
#Could do 5 cards choose 3. and solve for straights.
def connect(hand):
    print "connect"
    values = map(lambda x:x[0],hand)
    numstraights = 0
    for i in xrange(14,4,-1):
        for j in xrange(i-3,i-5,-1):
            for k in xrange(j-3,j-5,-1):
                if k == 1:
                    k = 14
                #find combinations of hero hand
                #check if ST is made
                #if ST, then add 1 and move on
        #print "board",board
        #print "hand", values
        print [i for i in board if i in hand]
            #print "yup"
            #numstraights += 1
        #print "i",i
    return numstraights

#returns # of nut river hands or similar
def nut(hand):
    return Nuttedness

#returns high card value. multiply the values together? could use prime numbers
#returns value between 0 and 1
def highcard(hand):
    #print "highcard"
    values = map(lambda x:x[0],hand)
    total = functools.reduce(operator.mul, values,1)
    #14^4 is 38416 - 2^4 = 38400 = max-min
    #mean = 3990 approximately
    total = float(total-16) / 38400
    return total

#returns allin vs random hand. uses evan calculator
def equity(hand):
    return equity

#returns the salient aspects of the hand. Removes the explicit suits
#should = 16432 hands. Without suits = 1820 hands
#Add explicit case for pair which puts the pair on one side and side cards ordered on the other.
def simplify(hand):
    #print 'hand', hand
    #print "simplify"
    simp = copy.deepcopy(hand)
    suits = map(lambda x:x[1],simp)
    values = map(lambda x:x[0],simp)
    #print collections.Counter(suits).items()
    sweet = collections.Counter(suits).items()
    valuesweet = collections.Counter(values).items()
    #print 'sweet1',sweet
    sweet.sort(key=lambda x: x[0], reverse = True)
    sweet.sort(key=lambda x: x[1], reverse = True)
    valuesweet.sort(key=lambda x: x[0], reverse = True)
    valuesweet.sort(key=lambda x: x[1], reverse = True)
    #print 'sweet3',sweet
    suitedness = [count for item, count in collections.Counter(suits).items()]
    duplicates = [count for item, count in collections.Counter(values).items()]
    #print 'suitedness1',suitedness
    #print '1', hand
    #print '2', hand
    suitedness.sort(reverse = True)
    duplicates.sort(reverse = True)
    #print 'suitedness2',suitedness
    #print '3', hand
    #print 'suitedness, suits',suitedness, suits, sweet
    #print sweet[0][0]
    unsuited = 0
    asuit = ''
    #print "sorted d,v",duplicates,valuesweet
    #check for XXYZ
    if duplicates[0] == 2 and duplicates[1] != 2:
        pair = copy.copy(valuesweet[0][0])
        sidecarda = copy.copy(valuesweet[1][0])
        sidecardb = copy.copy(valuesweet[2][0])
        pairs = []
        side = []
        for i in xrange(0,4):
            if simp[i][0] == pair:
                pairs.append(simp[i])
            else:
                side.append(simp[i])
        newhand = side + pairs
        #print "new",newhand
        simp = newhand
    if suitedness[0] == 1:
        #rainbow
        suits = ['a','b','c','d']
        for i in xrange(0,4):
            simp[i][1] = suits[i]
    elif suitedness[0] == 2:
        if suitedness[1] == 2:
            #print "DS"
            for x in xrange(0,4):
                #print 'simp',simp[x][1]
                if x == 0:
                    asuit = copy.copy(simp[x][1])
                    #print 'asuit',asuit
                    simp[x][1] = 'a'
                    #print 'posimp',simp[x][1]
                elif simp[x][1] == asuit:
                    simp[x][1] = 'a'
                    #print 'posimp',simp[x][1]
                else:
                    simp[x][1] = 'b'
        else:
            #SS
            for x in xrange(0,4):
                if simp[x][1] == sweet[0][0]:
                    simp[x][1] = 'a'
                elif unsuited == 0:
                    simp[x][1] = 'b'
                    unsuited += 1
                else:
                    simp[x][1] = 'c'
    elif suitedness[0] == 3:
        for x in xrange(0,4):
            if simp[x][1] == sweet[0][0]:
                simp[x][1] = 'a'
            elif simp[x][1] == sweet[1][0]:
                simp[x][1] = 'b'
    else:
        #mono hand
        for i in xrange(0,4):
            simp[i][1] = 'a'
    #print "simplified", simp
    #print '4', hand
    return simp

"""
def findmean(num,deck):
    values = []
    for i in xrange(0,num):
        random.shuffle(deck)
        newhand = deck[:4]
        values.append(highcardstrength(newhand))
    total = (sum(values))/num
    return total
"""

def combinations5(deck):
    boards = []
    for i in xrange(0,len(deck)):
        for j in xrange(i+1,len(deck)):
            for k in xrange(j+1,len(deck)):
                for l in xrange(k+1,len(deck)):
                    for q in xrange(l+1,len(deck)):
                        board = [deck[i],deck[j],deck[k],deck[l],deck[q]]
                        boards.append(hand)
    return boards

#maybe i should make this a list of an array? or a vector actually would be better
#dictionary + vector?
def makeclasslist(deck):
    hands = combinations4(deck)
    #hands.sort(key = lambda l: (l[0][1],l[1][1], \
    #l[2][1],l[3][1]))
    #hands.sort(key = lambda l: (l[0][0],l[1][0], \
    #l[2][0],l[3][0]),reverse = True)
    hands.sort(key = lambda l: (l[0][1],l[1][1], \
    l[2][1],l[3][1]),reverse = True)
    hands.sort(key = lambda l: (l[0][0],l[1][0], \
    l[2][0],l[3][0]),reverse = True)
    #print "first 100", hands[:100]
    newlist = []
    for i in xrange(0,len(hands)):
        #print 'i',i
        temp = copy.deepcopy(hands[i])
        #print "original",hands[i]
        temp.sort(reverse = True)
        #print "sorted",temp
        #print 'post2', hands[i]
        hand = simplify(temp)
        #print 'post3', hands[i]
        #print 'hand',hand
        newhand = cl.Rank(hand,suit(hands[i]),'connect',highcard(hands[i]),'nuts','equity')
        #print 'post obj', hands[i]
        #print 'hand',hand
        newlist.append(newhand)
        #hand.suit = suit(hands[i])
        #hand.connect = connect(hands[i])
        #hand.highcard = highcard(hands[i])
        #hand.nut = nut(hands[i])
        #hand.equity = equity(hands[i])
    #sort list according to high card values in reversed order
    #print '1',newlist[0].hand
    #print '2',newlist[0].hand[0]
    #print '3',(newlist[0].hand[0][0])
    #print newlist[50000].hand
    alist = sorted(newlist, key = lambda l: (l.suits),reverse = True)
    #alist = sorted(newlist, key = lambda l: (l.hand[0][0],l.hand[1][0], \
    #l.hand[2][0],l.hand[3][0]),reverse = True)
    #alist = sorted(newlist, key = operator.attrgetter(newlist.hand[0][0],newlist.hand[1][0], \
    #newlist.hand[2][0],newlist.hand[3][0]),reverse = True)
    return alist

#list length = 270725
def removeduplicates(hands):
    compare = hands[0].hand
    #print 'compare', compare
    reduced = []
    reduced.append(hands[0])
    i = 0
    for i in xrange(0,len(hands)):
        if compare != hands[i].hand:
            #print "trigger", compare, hands[i].hand
            compare = hands[i].hand
            reduced.append(hands[i])
    return reduced

def makefile(classlist):
    #store data
    with open('handattributes.pickle', 'wb') as handle:
        pickle.dump(classlist, handle, protocol = pickle.HIGHEST_PROTOCOL)


def openpickle():
    #to read file
    with open('handattributes.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b

deck = [[14,'s'],[13,'s'],[12,'s'],[11,'s'],[10,'s'],[9,'s'],[8,'s'],[7,'s'],[6,'s'],[5,'s'],[4,'s'],[3,'s'],[2,'s'],
[14,'h'],[13,'h'],[12,'h'],[11,'h'],[10,'h'],[9,'h'],[8,'h'],[7,'h'],[6,'h'],[5,'h'],[4,'h'],[3,'h'],[2,'h'],
[14,'c'],[13,'c'],[12,'c'],[11,'c'],[10,'c'],[9,'c'],[8,'c'],[7,'c'],[6,'c'],[5,'c'],[4,'c'],[3,'c'],[2,'c'],
[14,'d'],[13,'d'],[12,'d'],[11,'d'],[10,'d'],[9,'d'],[8,'d'],[7,'d'],[6,'d'],[5,'d'],[4,'d'],[3,'d'],[2,'d']]

testhand = [[[14,'s'],[12,'c'],[6,'s'],[2,'c']],[[14,'c'],[12,'s'],[6,'c'],[2,'s']],[[14,'h'],[12,'h'],[6,'s'],[2,'s']],[[14,'s'],[12,'s'],[6,'h'],[2,'h']],[[14,'s'],[12,'c'],[6,'c'],[2,'s']],[[14,'s'],[12,'c'],[6,'s'],[2,'c']]]
anothertest = [[14,'s'],[14,'c'],[14,'d'],[14,'h']]
a = [[[14, 'c'], [13, 's'], [13, 'd'], [11, 'd']],[[14, 'c'], [13, 'd'], [9, 's'], [9, 'd']],[[14, 'c'], [14, 's'], [12, 'a'], [10, 'a']]]
trips = [[[14, 's'], [14, 'h'], [14, 'c'], [14, 'd']], [[14, 's'], [14, 'h'], [14, 'c'], [13, 'c']], [[14, 's'], [14, 'h'], [14, 'c'], [13, 'd']], [[14, 's'], [14, 'h'], [14, 'd'], [13, 'd']], [[14, 's'], [14, 'c'], [14, 'd'], [13, 'd']], [[14, 'h'], [14, 'c'], [14, 'd'], [13, 'd']], [[14, 's'], [14, 'h'], [14, 'c'], [12, 'c']], [[14, 's'], [14, 'h'], [14, 'c'], [12, 'd']], [[14, 's'], [14, 'h'], [14, 'd'], [12, 'd']], [[14, 's'], [14, 'c'], [14, 'd'], [12, 'd']], [[14, 'h'], [14, 'c'], [14, 'd'], [12, 'd']]]

#save_path = openr"~/Users/Shuza/Code/PokerAI/src/handattributes.pickle")
os.chdir(r"/Users/Shuza/Code/PokerAI/src")

#l = makeclasslist(deck)
#reallist = removeduplicates(l)
#print len(reallist)
#makefile(l)
"""
s = copy.deepcopy(testy)
print s
s.sort(key = lambda x: (x[0][1],x[1][1], \
x[2][1],x[3][1]),reverse = True)
print s
s.sort(key = lambda x: (x[0][0],x[1][0], \
x[2][0],x[3][0]),reverse = True)
print s
"""
def compare(x,y):
    handx = x.hand
    handy = y.hand
    valuesx = map(lambda x:x[0],handx)
    valuesy = map(lambda x:x[0],handy)
    #print "x,valuesx",(x,valuesx)
    #print "y,valuesy",(y,valuesy)
    bignessx = [count for item, count in collections.Counter(valuesx).items()]
    bignessy = [count for item, count in collections.Counter(valuesy).items()]
    #print "bigx", bignessx
    #print "bigy", bignessy
    bignessx.sort(reverse = True)
    bignessy.sort(reverse = True)
    #print "bigx sort", bignessx
    #print "bigy sort", bignessy
    sweetx = collections.Counter(valuesx).items()
    sweety = collections.Counter(valuesy).items()
    #print 'sweet1',sweetx,sweety
    sweetx.sort(key=lambda x: x[0], reverse = True)
    sweety.sort(key=lambda x: x[0], reverse = True)
    #print 'sweet2',sweetx,sweety
    sweetx.sort(key=lambda x: x[1], reverse = True)
    sweety.sort(key=lambda x: x[1], reverse = True)
    #print 'sweet3',sweetx,sweety
    length = min(len(bignessx),len(bignessy))
    for i in xrange(0,length):
        #print 'i,swe',i,sweetx[i][0]
        if bignessx[i] > bignessy[i]:
            #print 'x'
            return 1
        elif bignessx[i] < bignessy[i]:
            #print 'y'
            return -1
        elif sweetx[i][0] > sweety[i][0]:
            #print 'x'
            return 1
        elif sweetx[i][0] < sweety[i][0]:
            #print 'y'
            return -1
    #must be tie
    return 0

def counter(x):
    counts = [0,0,0,0,0]
    bad = []
    for i in xrange(0,len(x)):
        #5 shapes. quads,trips,dp,single pair,nopair
        handx = copy.deepcopy(x[i].hand)
        valuesx = map(lambda x:x[0],handx)
        #print "x,valuesx",(x,valuesx)
        bignessx = [count for item, count in collections.Counter(valuesx).items()]
        #print "bigx", bignessx
        bignessx.sort(reverse = True)
        #print 'bignessx', bignessx[0]
        #print bignessx[0] == 4
        #print "count", counts
        if bignessx[0] == 4:
            #print "Quads"
            counts[0] += 1
        elif bignessx[0] == 3:
            #print "Trips"
            counts[1] += 1
        elif bignessx[0] == 2:
            if bignessx[1] == 2:
                #print "DP"
                counts[2] += 1
            else:
                #print "SP"
                #bad.append(x[i])
                counts[3] += 1
        else:
            #print "highcard"
            counts[4] += 1
    return counts

def howmany(hands):
    counter = [[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],[11,0],[12,0],[13,0],[14,0]]
    for i in xrange(0,len(hands)):
        #print 'hands',i,hands[i]
        handx = copy.deepcopy(hands[i].hand)
        valuesx = map(lambda x:x[0],handx)
        #bignessx = [count for item, count in collections.Counter(valuesx).items()]
        #bignessx.sort(reverse = True)
        #print "bigx", bignessx
        sweet = collections.Counter(valuesx).items()
        sweet.sort(key=lambda x: x[0], reverse = True)
        sweet.sort(key=lambda x: x[1], reverse = True)
        #print 'sweet',sweet
        for x in xrange(0,len(counter)):
            #print counter[x][0],sweet[0][1]
            #print sweet[0][0] == counter[x][0]
            if sweet[0][0] == counter[x][0]:
                'ding!'
                counter[x][1] += 1
                break
    return counter
#should = 16432 hands. Without suits = 1820 hands
def shortenfile():
    a = openpickle()
    print "initial",len(a)
    a.sort(key = lambda x: (x.hand[0][0],x.hand[1][0], \
    x.hand[2][0],x.hand[3][0]),reverse = True)
    b = removeduplicates(a)
    b.sort(key = lambda x: (x.hand[0][1],x.hand[1][1], \
    x.hand[2][1],x.hand[3][1]),reverse = True)
    u = removeduplicates(b)
    u.sort(key = lambda x: (x.suits))
    u.sort(cmp=compare,reverse = True)
    for x in xrange(0,len(u)):
        u[x].hand.sort(key = lambda x: x[1],reverse = True)
    q = removeduplicates(u)
    x =  howmany(q)
    z = counter(q)
    print "howmany,count",z,x
    print '3',len(q)
    makefile(m)
    for x in xrange(0,5000):
        print q[x].hand

shortenfile()

#l = makeclasslist(deck)
#print len(l)
#makefile(l)

#print len(l)
#reallist = removeduplicates(l)
#makefile(reallist)


#l = openpickle()
#print len(l)

#for x in xrange(0,len(testhand)):
#    print simplify(testhand[x])


#small = l[:100]
#print [x.hand for x in small]
#reallist = removeduplicates(small)
#print "length",len(reallist)
#print 'reallist',reallist
#makefile(reallist)

#makefile(l)
#l = openpickle()
#print len(l)
#hands = list(it.combinations(deck, 4))
#print highcard(testhand)
#print connect(testy)
#print findmean(100,deck)
