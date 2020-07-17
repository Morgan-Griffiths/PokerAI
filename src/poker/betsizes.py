import poker.datatypes as pdt

class Betsizes():
    LIMIT_MAP = {
        pdt.LimitTypes.NO_LIMIT: return_nolimit_betsize,
        pdt.LimitTypes.POT_LIMIT: return_potlimit_betsize,
        pdt.LimitTypes.LIMIT: return_limit_betsize
    }
    def __init__(self,limit):
        self.return_betsize = self.LIMIT_MAP[limit]

    @staticmethod
    def return_limit_betsize(env,action,betsize_category):
        """TODO Betsizes should be indexed by street"""
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]: # Call
            betsize = min(env.players[env.current_player].stack,env.players[env.last_aggressor.key].street_total - env.players[env.current_player].street_total)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize = min(1,env.players[env.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise can be more than 2 with multiple people
            betsize = min(env.players[env.last_aggressor.key].street_total+1,env.players[env.current_player].stack) - env.players[env.current_player].street_total
        else: # fold or check
            betsize = 0
        return betsize

    ## NO LIMIT ##
    def return_nolimit_betsize(self,action,betsize_category):
        """
        TODO
        Betsize_category would make the most sense if it represented percentages of pot on the first portion.
        And percentages of stack or simply an allin option as the last action.
        """
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]:
            betsize = min(self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total,self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize_value = self.betsizes[betsize_category] * self.pot
            betsize = min(max(1,betsize_value),self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise
            max_raise = (2 * self.players[self.last_aggressor.key].street_total) + (self.pot - self.players[self.current_index.key].street_total)
            betsize_value = self.betsizes[betsize_category] * max_raise
            previous_bet = self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total
            betsize = min(max(previous_bet * 2,betsize_value),self.players[self.current_player].stack)
        else:
            betsize = 0
        return betsize
            
    ## POT LIMIT
    def return_potlimit_betsize(self,action,betsize_category):
        """TODO Betsize_category in POTLIMIT is a float [0,1] representing fraction of pot"""
        if action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.CALL]:
            betsize = min(self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total,self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.BET]: # Bet
            betsize_value = self.betsizes[betsize_category] * self.pot
            betsize = min(max(1,betsize_value),self.players[self.current_player].stack)
        elif action == pdt.Globals.REVERSE_ACTION_ORDER[pdt.Actions.RAISE]: # Raise
            max_raise = (2 * self.players[self.last_aggressor.key].street_total) + (self.pot - self.players[self.current_index.key].street_total)
            betsize_value = (self.betsizes[betsize_category] * max_raise) - self.players[self.current_index.key].street_total
            previous_bet = self.players[self.last_aggressor.key].street_total - self.players[self.current_index.key].street_total
            betsize = min(max(previous_bet * 2,betsize_value),self.players[self.current_player].stack)
        else:
            betsize = 0
        return betsize