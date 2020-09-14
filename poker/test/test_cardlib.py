import unittest
import utils.cardlib as cb

class TestCardlib(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.omaha_hand = [[7, 's'], [5, 'c'], [14, 'h'], [10, 'h']]
        self.omaha_hand2 = [[14, 'c'], [2, 's'], [2, 'd'], [11, 's'],[5,'c']]
        self.board = [[10, 'c'], [2, 'h'], [4, 'c'], [13, 'c'], [4, 'h']]
        self.holdem_hand = [[14, 3], [2, 1]]
        self.holdem_hand2 = [[13, 3], [5, 1]]

    def testEncode(self):
        en_hand = [cb.encode(card) for card in self.omaha_hand]
        assert en_hand == [2102541, 557831, 268446761, 16787479]

    def testDecode(self):
        en_hand = [cb.encode(card) for card in self.omaha_hand]
        hand = [cb.decode(card) for card in en_hand]
        assert hand == [[7, 's'], [5, 'c'], [14, 'h'], [10, 'h']]

    def testWinner(self):
        en_hand = [cb.encode(card) for card in self.omaha_hand]
        en_hand2 = [cb.encode(card) for card in self.omaha_hand2]
        en_board = [cb.encode(card) for card in self.board]
        assert cb.winner(en_hand,en_hand2,en_board) == -1

    def testHandrank(self):
        en_hand = [cb.encode(card) for card in self.omaha_hand]
        en_board = [cb.encode(card) for card in self.board]
        assert cb.hand_rank(en_hand,en_board) == 2985

    def testHoldemHandrank(self):
        en_hand = [cb.encode(card) for card in self.holdem_hand]
        en_board = [cb.encode(card) for card in self.board]
        assert cb.holdem_hand_rank(en_hand,en_board) == 3304

    def testHoldemWinner(self):
        en_hand = [cb.encode(card) for card in self.holdem_hand]
        en_hand2 = [cb.encode(card) for card in self.holdem_hand2]
        en_board = [cb.encode(card) for card in self.board]
        assert cb.holdem_winner(en_hand,en_hand2,en_board) == -1

def cardlibTestSuite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnv('testEncode'))
    suite.addTest(TestEnv('testDecode'))
    suite.addTest(TestEnv('testWinner'))
    suite.addTest(TestEnv('testHandrank'))
    suite.addTest(TestEnv('testHoldemHandrank'))
    suite.addTest(TestEnv('testHoldemWinner'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(cardlibTestSuite())