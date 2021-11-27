# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.10 (default, May 19 2021, 13:12:57) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: utils.py
from .card import Card
CARD_DIGITAL_TABLE = {'SA':270,
 'S2':258,
 'S3':259,
 'S4':260,
 'S5':261,
 'S6':262,
 'S7':263,
 'S8':264,
 'S9':265,
 'ST':266,
 'SJ':267,
 'SQ':268,
 'SK':269,
 'HA':526,
 'H2':514,
 'H3':515,
 'H4':516,
 'H5':517,
 'H6':518,
 'H7':519,
 'H8':520,
 'H9':521,
 'HT':522,
 'HJ':523,
 'HQ':524,
 'HK':525,
 'CA':782,
 'C2':770,
 'C3':771,
 'C4':772,
 'C5':773,
 'C6':774,
 'C7':775,
 'C8':776,
 'C9':777,
 'CT':778,
 'CJ':779,
 'CQ':780,
 'CK':781,
 'DA':1038,
 'D2':1026,
 'D3':1027,
 'D4':1028,
 'D5':1029,
 'D6':1030,
 'D7':1031,
 'D8':1032,
 'D9':1033,
 'DT':1034,
 'DJ':1035,
 'DQ':1036,
 'DK':1037,
 'SB':272,
 'HR':529}


Card2Num = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6,
               '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
Suit2Num = {'S':0, 'H':1, 'C':2, 'D':3}
Rank2Num = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
    '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 
    'JOKER': 13, 'B': 13, 'R': 14
}

SINGLE = 'Single'
PAIR = 'Pair'
TRIPS = 'Trips'
THREE_PAIR = 'ThreePair'
THREE_WITH_TWO = 'ThreeWithTwo'
TWO_TRIPS = 'TwoTrips'
STRAIGHT = 'Straight'
STRAIGHT_FLUSH = 'StraightFlush'
BOMB = 'Bomb'
PASS = 'PASS'

RANK = ('', '', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')

CardType2Num = {
    SINGLE: 0,
    PAIR: 1,
    TRIPS: 2,
    THREE_PAIR: 3,
    THREE_WITH_TWO: 4,
    TWO_TRIPS: 5,
    STRAIGHT: 6,
    STRAIGHT_FLUSH: 7,
    BOMB: 8,
}

ActionType2Num = {
    SINGLE: 0,
    PAIR: 1,
    TRIPS: 2,
    THREE_PAIR: 3,
    THREE_WITH_TWO: 4,
    TWO_TRIPS: 5,
    STRAIGHT: 6,
    STRAIGHT_FLUSH: 7,
    BOMB: 8,
    PASS: 9
}

NUM_ACTION_TYPES = len(ActionType2Num)
NUM_CARD_RANKS = 15


class OverOrder(object):
    __doc__ = '\n    管理完牌次序, 并根据完牌次序, 给出进贡和还贡的关系\n    '

    def __init__(self):
        self.order = []
        self.tri_ship = []
        self.bck_ship = []
        self.tri_cards = []
        self.bck_cards = []
        self.index = 0
        self.first_two_teammate = False
        self.first_play = 0
        self.two_tribute = False

    def add(self, pos):
        """
        将完牌的玩家的座位号添加到列表中
        :param pos: 座位号
        :return: None
        """
        self.order.append(pos)

    def settlement(self):
        """
        根据完牌次序进行结算，并保存进贡还贡关系，返回结算后对应玩家需要升几级
        :return: tuple(int, int, int)
        """
        if (self.order[0] + 2) % 4 == self.order[1]:
            inc = 3
            order = [0, 1, 2, 3]
            self.first_two_teammate = True
            self.tri_ship.append((self.order[(-1)], order[(self.order[(-1)] - 1)]))
            self.tri_ship.append((self.order[(-2)], order[(self.order[(-2)] - 1)]))
            self.bck_ship.append((order[(self.order[(-1)] - 1)], self.order[(-1)]))
            self.bck_ship.append((order[(self.order[(-2)] - 1)], self.order[(-2)]))
        else:
            if (self.order[0] + 2) % 4 == self.order[2]:
                inc = 2
                self.tri_ship.append((self.order[(-1)], self.order[0]))
                self.bck_ship.append((self.order[0], self.order[(-1)]))
            else:
                inc = 1
                self.tri_ship.append((self.order[(-1)], self.order[0]))
                self.bck_ship.append((self.order[0], self.order[(-1)]))
        return (
         self.order[0], (self.order[0] + 2) % 4, inc)

    def find_ship(self, pos, _ship):
        ship = getattr(self, '{}_ship'.format(_ship))
        for s in ship:
            if pos in s:
                return s

    def episode_over(self, pos):
        """
        判断座位号为pos的玩家的队友是否已经在完牌序列中
        :return:
        """
        if (pos + 2) % 4 in self.order:
            return True
        else:
            return False

    def clear(self):
        self.order.clear()
        self.tri_ship.clear()
        self.bck_ship.clear()
        self.tri_cards.clear()
        self.bck_cards.clear()
        self.two_tribute = False

    @property
    def first(self):
        return self.order[0]

    @property
    def second(self):
        return self.order[1]

    @property
    def third(self):
        return self.order[2]

    @property
    def fourth(self):
        return self.order[3]
# okay decompiling utils.pyc

def get_cards(action):
    if action.type == 'PASS' or action.type is None:
        return []
    else:
        return [Card(c[0], c[1]) for c in action.cards]


def get_down_pid(pid):
    return (pid+1) % 4

def get_up_pid(pid):
    return (pid+3) % 4

def get_teammate_pid(pid):
    return (pid+2) % 4
