import re
import copy
import Classes as cl
import itertools
import Database_gen as dg
import pickle
import gc
"""
with open("/Users/Shuza/SpiderOak Hive/Poker hands/BovadaHandHistory/BovadaHandHistory_Omaha_PL_2016070722214792111.txt",'r') as f:

All stacks and betsizes tracked in BBs (for cross stakes)
hand converted and ordered. Replace Suits by ABCD (Hero,board,vil)
Keep track of pot size.
Bets and raises in terms of pot size.
calls in terms of pot size.
Track Allins and returned bets

later it will say Hero: posts *blind* *amount* if hero is in blinds.
Can figure out all positions from positions previously

grab digits+commas and periods = ^[0-9]{1,2}([,.][0-9]{1,2})?$
re.search(r"o\s\$([\d]+[,.][\d]+)?",action)
"""

#TODO create pre/flop/turn seperate action groups for infoset
#only use effective stack size. Let it iterate over hands in a folder.
#store hands in rust format for easy eval?
#TODO Fixed overbets in terms of stacksize for HU. To fix for multiplayer
#need to isolate the players and compare the eff_stacksizes. Shouldn't be that hard
#TODO remove Hero and replace with generic position. Add a perspective field?

def get_digits(text):
    return filter(str.isdigit, text)

def import_file(file_name,file_type):
    with open(file_name,'r') as file:
        file_contents = file.readlines()
        length = len(file_contents)
        infoset_list = []
        i = 0
        end = False
        rake_format = False
        #while end == False:
        #for per hand basis
        for x in xrange(0,5):
            print "beginning", i
            players = []
            infoset = cl.Information_set()
            positions = []
            pot = 0
            skip = False
            print file_contents[0]
            print 'length', length
            #get header
            (header,bb,i) = grab_header(file_contents,i,infoset)
            #positions + stacks
            (positions,i,players,skip) = grab_positions_stacks(file_contents,i,bb,positions,players,infoset,skip)
            eff_stack = min(infoset.positions_stacks, key = lambda x: x[1])[1]
            if skip == False:
                #blinds and pot
                (pot,i,players,last_bet) = grab_pot_blinds(file_contents,i,bb,pot,players)
                #get holecards
                (i,players,suit_replace) = grab_hand(file_contents,i,infoset,players)
                #get actions/board cards
                (i,pot,players,suit_replace,position_replace) = grab_actions(file_contents,i,infoset,bb,pot,players,suit_replace,last_bet,eff_stack,position_replace)
                #get outcome
                (i,pot,end,players) = grab_outcome(file_contents,i,infoset,pot,bb,end,length,players,rake_format,suit_replace)
                #move to beginning of next hand
                infoset_list.append(infoset)
            else:
                while not re.search("PokerStars",file_contents[i]):
                    i += 1
                #forward to next hand
            #gc.set_debug(gc.DEBUG_LEAK)
        #return infoset_list
        #makefile(infoset_list)

def grab_header(file_contents,i,infoset):
        #First get stakes and game type
        start = file_contents[i].index('$')
        end = file_contents[i].index(')')
        sliver = file_contents[i][start:end]
        game_stakes = copy.copy(sliver)
        #get bb
        bb_slice = re.search(r"\/\$(\d+)",game_stakes)
        bb = int(copy.copy(bb_slice.group(1)))
        #print game_stakes, bb
        start = file_contents[i].index(':')+2
        end = file_contents[i].index('(')
        sliver = file_contents[i][start:end]
        game_type = copy.copy(sliver)
        #print game_type
        i += 1
        #get ring type
        start = file_contents[i].index('-') - 1
        end = file_contents[i].index('S')
        sliver = file_contents[i][start:end]
        ring_type = copy.copy(sliver)
        #print ring_type
        header = game_stakes + ' ' + game_type + ring_type
        print 'header',header
        infoset.header = header
        i += 1
        return (header, bb, i)

def grab_positions_stacks(file_contents,i,bb,positions,players,infoset,skip):
        while re.search(r"Seat (\d)",file_contents[i]):
            #print "line",file_contents[i]
            values = re.search(r"Seat (\d): (.*)\s\(\$([\d,\.,\,]+)",file_contents[i])
            #print values.group(2)
            position = values.group(2)
            #get rid of commas in large numbers
            stacksize_slice = copy.copy(values.group(3))
            stacksize_slice = stacksize_slice.replace(',','')
            stacksize_bb = (float(stacksize_slice)) / bb
            players.append(cl.Player_load(stacksize_bb,position))
            positions.append([position] + [stacksize_bb])
            i += 1
        print 'positions',positions
        infoset.positions_stacks = positions
        if len(positions) > 2:
            #i += 1
            skip = True
        return (positions,i,players,skip)

#TODO: make this whole system more resilient to more players etc.
#TODO find hero and opponent and replace with generic positions
#create dictionary with names and actual positions
def grab_pot_blinds(file_contents,i,bb,pot,players):
        #print '1', file_contents[i]
        #HU only
        people = []
        while not re.search(r"blind",file_contents[i]):
            i += 1
        last_bet = 0
        blinds = 0
        blinds = re.search(r"\$(\d+)",file_contents[i])
        blind = float(int(blinds.group(1)))/bb
        #print 'SB', blind
        player_small = re.search(r"(.*):",file_contents[i])
        people.append(player_small.group(1))
        #print player_small.group(1)
        #add blinds to bet total for that player
        for d in xrange(0,len(players)):
            if players[d].position == player_small.group(1):
                players[d].street_bet_total = blind
                players[d].bet_total = blind
        #This is for multi-player
        #for d in xrange(0,len(players)):
        #    if players[d].position == "Small Blind":
        #        players[d].street_bet_total = blind
        #        players[d].bet_total = blind
        #        print "triggered"
        #print "SB",blind
        pot += blind
        i +=1
        while not re.search(r"blind",file_contents[i]):
            i += 1
        blinds = re.search(r"\$(\d+)",file_contents[i])
        blind = float(int(blinds.group(1)))/bb
        #print 'BB',blind
        player_big = re.search(r"(.*):",file_contents[i])
        people.append(player_big.group(1))
        #print player_big.group(1)
        for d in xrange(0,len(players)):
            print players[d].position == player_big.group(1)
            if players[d].position == player_big.group(1):
                players[d].street_bet_total = blind
                players[d].bet_total = blind
                last_bet = blind
        #this is for multiplayer
        #for d in xrange(0,len(players)):
        #    if players[d].position == "Big Blind":
        #        players[d].street_bet_total = blind
        #        players[d].bet_total = blind
        #        last_bet = blind
        #        print "last",last_bet
        pot += blind
        while not re.search(r"^\*",file_contents[i]):
            i += 1
            #print 'i', i
        #print "pot",pot
        #print file_contents[i], i
        true_position = ['small blind','big blind']
        position_replace = dict(zip(people,true_position))
        print position_replace
        i += 1
        return (pot,i,players,last_bet,position_replace)

def grab_hand(file_contents,i,infoset,players):
        #add hero's hand. Order hero hand and convert into ints
        holes = re.search(r"\[(.+)\]",file_contents[i])
        #if no hero, skip hand.
        if not holes:
            print 'omg no holes'
        hero_holes = holes.group(1)
        newcards = hand_conversion(hero_holes)
        simplified_hand = dg.simplify(newcards)
        simplified_hand.sort(reverse = True)
        #print '1',simplified_hand,newcards
        suits = map(lambda x:x[1],newcards)
        generic_suits = map(lambda x:x[1],simplified_hand)
        #print '2',suits,generic_suits
        newsuits = unique(suits)
        newgeneric_suits = unique(generic_suits)
        #print '3',newsuits,newgeneric_suits
        suit_replace = dict(zip(newsuits,newgeneric_suits))
        hero_cards = hand_reconversion(newcards)
        generic_hand = hand_reconversion(simplified_hand)
        infoset.hand_sorted = hero_cards
        infoset.hand_generic = generic_hand
        print 'gen hand',infoset.hand_generic
        #print 'suit_replace',suit_replace
        return (i,players,suit_replace)

def hand_conversion(cards):
        cards = ''.join(cards.split())
        #print 'cards',cards
        hero = list(cards)
        #print 'hero',hero
        newcards = [hero[y:y+2] for y in range(0,len(hero),2)]
        for x in xrange(0,len(newcards)):
            newcards[x][0] = cl.handvalues.get(newcards[x][0])
        newcards.sort(reverse = True)
        #print "newhero",newcards
        return newcards

def hand_reconversion(cards):
    #turn 2d list into 1d string
    single = list(itertools.chain.from_iterable(cards))
    #turn 1d list into concatenated str
    hero_cards = ''.join(str(e) for e in single)
    return hero_cards

def unique(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

def grab_actions(file_contents,i,infoset,bb,pot,players,suit_replace,last_bet,eff_stack,position_replace):
        print "grab_actions"
        first_street = True
        street = 0
        infoset.board = []
        infoset.generic_board = []
        infoset.actions_full = []
        #Get all actions and board cards until end of hand
        while not re.search(r"(SHOW DOWN)",file_contents[i]) and not re.search("Uncalled",file_contents[i])\
        and not re.search(r"(collected)",file_contents[i]):
            #resets bettotal for subsequent streets
            if first_street == False:
                for p in xrange(0,len(players)):
                    players[p].street_bet_total = 0
            line = copy.copy(file_contents[i])
            print "start" ,i,file_contents[i]
            place = re.search(r"\* ([A-Z]+) \*",file_contents[i])
            if place:
                print "Street",place.group(1)
                grab_cards = re.findall(r"(\[[a-zA-Z,\s,\d]+\])",file_contents[i])
                segment = grab_cards[-1]
                board = segment.replace("]","")
                #print "board",board
                newboard = board.replace("[","")
                #print "board",newboard
                #have to order and break up the flop, update suit_replace
                board = hand_conversion(newboard)
                print "newboard", board
                (generic_board,suit_replace) = dg.board_simplify(board,suit_replace)
                infoset.board.append(grab_cards[-1])
                infoset.generic_board.append(generic_board)
                infoset.actions_full.append(generic_board)
                print generic_board, board
            i+=1
            #grab actions and positions
            while not re.search(r"^\*",file_contents[i]) and not re.search("Uncalled",file_contents[i]):
            #while re.search("|".join(["raises"]))
                #print "pos1",file_contents[i]
                #print 'pot', pot
                #check for "Table deposit $xxx" and "Seat re-join" and skip
                while re.search(r"Table",file_contents[i]) or re.search(r"Seat re-join",file_contents[i])\
                 or re.search(r"Seat sit down",file_contents[i]) or re.search(r"doesn't show hand",file_contents[i]):
                    print "skip"
                    print i,file_contents[i]
                    i += 1
                if re.search(r"^\*",file_contents[i]) or re.search("Uncalled",file_contents[i])\
                 or re.search(r"(collected)",file_contents[i]):
                    break

                #print 'uncalled boolean',not re.search("Uncalled",file_contents[i])
                print file_contents[i]
                values = re.search(r"(.+): (.+)",file_contents[i])
                #print "values",values.group(1),"\n",values.group(2)
                position = copy.copy(values.group(1))
                temp = copy.copy(values.group(2))
                action = temp[:-1]
                pure_action = re.search(r"([a-z]+)",action)
                print pure_action.group(0)
                #print "if than bets", pure_action.group(0) == "bets"
                #print "if than calls", pure_action.group(0) == "calls"
                firstword = re.search(r"\s\$([\d]+|[,.][\d]+)?",action)
                #print "firstattempt", firstattempt.group(1)
                #nextword = firstattempt = firstattempt.replace(',','')
                #firstword = float(firstattempt)
                if firstword:
                    #print 'firstword', firstword.group(1)
                    for p in xrange(0,len(players)):
                        if players[p].position == position:
                            current_player = players[p]
                    allin = re.search(r"s\s(all-in)",action)
                    if allin:
                        #print 'allin',allin.group(1)
                        current_player.allin = True
                    print 'current_player.bet_total', current_player.bet_total
                    if pure_action.group(0) == "bets":
                        #print "ding bets"
                        true_amnt_slice = re.search(r"s\s\$(\d+[,]?\d+[\.]?[\d+]?)",action)
                        combined = re.sub(r",","",true_amnt_slice.group(1))
                        amnt = float(combined)/bb
                        #HU only, removes the possibility of betting more than eff stack
                        if (amnt + current_player.bet_total) > eff_stack:
                            "effective_bet"
                            amnt = (eff_stack - current_player.bet_total)
                        current_player.bet_total += amnt
                        current_player.street_bet_total += amnt
                        last_bet = copy.copy(current_player.street_bet_total)
                        #print 'pot,addition',pot,amnt
                        ratio = float(amnt)/pot
                        pot += amnt
                    elif pure_action.group(0) == "calls":
                        #print "ding call"
                        true_amnt_slice = re.search(r"s\s\$(\d+[,]?[\d+]?[\.]?[\d+]?)",action)
                        end_value = re.sub(r",","",true_amnt_slice.group(1))
                        #print "true",end_value
                        amnt = float(end_value)/bb
                        current_player.bet_total += amnt
                        current_player.street_bet_total += amnt
                        #TODO: in ring games, need to account for limps pre
                        #last_bet = copy.copy(current_player.street_bet_total)
                        #print "pot,addition",pot,amnt
                        pot += amnt
                        ratio = float(amnt)/pot
                        if current_player.allin == True:
                            #HU only
                            #print "allin trigger"
                            #print players[0].bet_total
                            for q in xrange(0,len(players)):
                                if players[q].position != current_player.position:
                                    villain_player = players[q]
                            print villain_player
                            if current_player.bet_total != villain_player.bet_total:
                                extra = villain_player.bet_total - current_player.bet_total
                                #print 'extra', extra
                                #print villain_player.bet_total, current_player.bet_total
                                pot -= extra
                                villain_player.bet_total -= extra
                    elif pure_action.group(0) == "raises":
                        #print "ding raise"
                        #fix to scrap the cents after period
                        true_amnt_slice = re.search(r"o\s\$(\d+[,]?\d+[\.]?[\d+]?)",action)
                        end_value = re.sub(r",","",true_amnt_slice.group(1))
                        #print "true",end_value
                        amnt = float(end_value)/bb
                        #if true_amnt_slice.group(2):
                        #    print "true2"
                        total_raise = float(end_value)/bb
                        #minus what i have invested
                        true_amnt = total_raise - current_player.street_bet_total
                        #check if allin
                        #print 'sub',(last_bet - current_player.street_bet_total)
                        #print "last", last_bet
                        #print 'pot', pot
                        maxraise = (last_bet) + (last_bet - current_player.street_bet_total) + pot
                        #print 'ratio 2nd',maxraise,amnt
                        #HU only - raise up to eff_stacksize
                        if (true_amnt + current_player.bet_total) > eff_stack:
                            print "effective_raise"
                            true_amnt = (eff_stack - current_player.bet_total)
                        current_player.bet_total += true_amnt
                        current_player.street_bet_total += true_amnt
                        ratio = float(current_player.street_bet_total)/maxraise
                        #print "pot,newpot",pot,true_amnt, ratio,maxraise
                        pot += true_amnt
                        last_bet = copy.copy(current_player.street_bet_total)
                    trim_action = re.search(r"([a-z]+) \$",action)
                    action = (trim_action.group(1) + ' ' + str(ratio))
                    #print "newaction",action
                infoset.actions_full.append(position +' '+ action)
                print 'infoset', infoset.actions_full
                i += 1
            first_street = False
            street += 1
        #end of hand
        return (i,pot,players,suit_replace,position_replace)

def grab_outcome(file_contents,i,infoset,pot,bb,end,length,players,rake_format,suit_replace,position_replace):
        #print "grab_outcome"
        #print file_contents[i]
        tie = False
        infoset.outcome = []
        #Either everyone folded, or someone called. collect winnings, -rake.
        if re.search(r"(SHOW DOWN)",file_contents[i]):
            print "Showdown"
            #grab vilhand and vilhand_generic
            i += 1
            while re.search(r"Hero: shows",file_contents[i]):
                i += 1
                #print "next line",file_contents[i]
            if re.search(r": shows",file_contents[i]):
                print 'capture',file_contents[i]
                vilhand = re.search(r"shows \[(.*)\]",file_contents[i])
                #print 'vilhand',vilhand.group(1)
                sorted_vilhand = hand_conversion(vilhand.group(1))
                #print sorted_vilhand
                (vilhand_generic,suit_replace) = dg.board_simplify(sorted_vilhand,suit_replace)
                #print vilhand_generic
                #print infoset.hand_sorted,infoset.hand_generic
                vilhand = hand_reconversion(sorted_vilhand)
                generic_vilhand = hand_reconversion(vilhand_generic)
                infoset.vilhand_generic = generic_vilhand
                infoset.vilhand = vilhand
                print 'vilhands',infoset.vilhand,infoset.vilhand_generic
            while not re.search(r"\$(.*) from pot",file_contents[i]):
                i += 1
            #print   i,file_contents[i]
            #winner_slice = re.search(r"([a-zA-Z]+|\s[a-zA-Z]+) c",file_contents[i])
            winner_slice = re.search(r"(.*)\sc",file_contents[i])
            winnernew = winner_slice.group(1)
            winner = re.sub(r'\r','',winnernew)
            #print 'winner',winner
            q = i
            q += 1
            #TODO make search resilient to spam "Table deposit" etc
            if re.search(r"(.*)\scollected",file_contents[q]):
                print "tie"
                tie = True
            #print file_contents[i]
            #print winner_slice.group(1)
        elif re.search("Uncalled",file_contents[i]):
            #record the winning player
            #print 'line', file_contents[i]
            winner_slice = re.search(r"o\s(.*)",file_contents[i])
            winnernew = winner_slice.group(1)
            #print 'winner',winnernew
            winner = re.sub(r'\r','',winnernew)
            #print 'first/last char',winner[0],ord(winner[-1])
            #subtract returned bet from pot
            print "uncalled"
            #print file_contents[i]
            true_amnt_slice = re.search(r"\$(\d+[,]?\d+[\.]?[\d+]?)",file_contents[i])
            #print true_amnt_slice.group(1)
            uncalled_bet = re.sub(r",","",true_amnt_slice.group(1))
            #print uncalled_bet
            returned_bet = float(uncalled_bet)/bb
            pot -= returned_bet
            #subtract bet from player's bettotal
            for y in xrange(0,len(players)):
                #print len(players[y].position), len(winner),players[y].position == winner
                if players[y].position == winner:
                    #print "win"
                    players[y].bet_total -= returned_bet
        #pre fold
        else:
            print file_contents[i]
            winner_slice = re.search(r"(.*)\scollected",file_contents[i])
            winnernew = winner_slice.group(1)
            winner = re.sub(r'\r','',winnernew)
            print "prefold",winnernew
        #folded preflop
        #fast forward to summary
        #add pot to winnings of player
        #record loss for other players
        while not re.search("SUMMARY",file_contents[i]):
            i += 1
        i += 1
        losers = []
        if rake_format == True:
            rake_slice = re.search(r"Rake \$(.*)",file_contents[i])
            print rake_slice.group(1)
            rake = float(rake_slice.group(1))/bb
            pot -= rake
            #print 'pot', pot
        if tie == False:
            for y in xrange(0,len(players)):
                if players[y].position == winner:
                    players[y].winnings = pot - players[y].bet_total
                    players[y].bet_total = 0
                    infoset.outcome.append(players[y].position +' '+str(players[y].winnings))
                else:
                    players[y].winnings -= copy.copy(players[y].bet_total)
                    if players[y].winnings < 0:
                        losers.append(players[y].position +' '+str(players[y].winnings))
                    players[y].bet_total = 0
            infoset.outcome.append(losers)
        #HU only
        if tie == True:
            print "tie"
            tie_list = []
            for y in xrange(0,len(players)):
                players[y].bettotal = 0
                #pot / 2
                players[y].winnings = 0
                tie_list.append(players[y].position +' '+str(players[y].winnings))
            infoset.outcome.append(tie_list)

        #for r in xrange(0,len(players)):
        #    print players[r].position,players[r].winnings,players[r].bet_total
        #check if end of file?
        while not re.search("PokerStars",file_contents[i]):
            i += 1
            if i == length:
                end = True
                return (i,infoset,pot,end)
        print 'end',pot,i,infoset.outcome
        return (i,pot,end,players)

    #get end of hand and end hand loop. Start hand while loop with while !== SUMMARY
    #Get pot - rake and search for winner (add in losers as well)
def makefile(classlist):
    #store data
    with open('infoset2.pickle', 'wb') as handle:
        pickle.dump(classlist, handle, protocol = pickle.HIGHEST_PROTOCOL)

#deposit folder
#/Users/Shuza/Code/PokerAI/Pickle
#file_types
bovada_TJ = 0
bovada_revealed = 1


import_file("/Users/Shuza/Code/PokerTestHands/Poker_HU_Hands/Bovada_hands/Export Holdem Manager 2.0 05222017213645.txt",bovada_TJ)

#bovada_hidden(/Users/Shuza/Code/Hands_hidden)

#for line in file:
#    print readline()
