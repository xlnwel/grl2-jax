import random
from copy import deepcopy
from small_game_env.player import Player
from small_game_env.utils import OverOrder
from small_game_env.action import Action, ActionList
from small_game_env.card import Card
from random import shuffle
from collections import Counter, defaultdict
from json import dumps
import numpy as np

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

RANK = ('', '', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
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


def com_cards(same_cards, cards, n, name):
    """
    使用游标选择指定的数量的相同点数的卡牌（即用于选择两对、三张、炸弹等牌型）
    TEST  如左所示，采用游标法（也可视做二进制法）来生成指定元素个数的组合，1表示所指向的元素
    1101
    :param same_cards: 相同点数的卡牌的列表
    :param cards: 包含相同点数的卡牌列表
    :param n: 指定数量
    :param name: 牌型名称
    :return: List[list[Card]]
    """
    inx_ptr = [0 for _ in range(n)]
    for i in range(n - 1):
        inx_ptr[i] = i

    inx_ptr[n - 1] = len(cards) - 1
    flag = True
    while inx_ptr[0] <= len(cards) - n and flag:
        foo = []
        for inx in inx_ptr:
            foo.append(str(cards[inx]))

        same_cards.append([name, foo[0][1], foo])
        if cards[(inx_ptr[(n - 1)] - 1)] == cards[inx_ptr[(n - 1)]]:
            inx_ptr[(n - 1)] -= 2
        else:
            inx_ptr[(n - 1)] -= 1
        if inx_ptr[(n - 1)] <= inx_ptr[(n - 2)]:
            inx_ptr[n - 1] = len(cards) - 1
            i = n - 2
            while i >= 0:
                inx_ptr[i] += 1
                if inx_ptr[i] == inx_ptr[(i + 1)]:
                    inx_ptr[i] -= 1
                    if i == 0:
                        flag = False
                    i -= 1
                    continue
                else:
                    if cards[inx_ptr[i]] == cards[(inx_ptr[i] - 1)]:
                        inx_ptr[i] += 1
                        if inx_ptr[i] == inx_ptr[(i + 1)]:
                            inx_ptr[i] -= 1
                            if i == 0:
                                flag = False
                            i -= 1
                            continue
                        else:
                            for j in range(i + 1, n - 1):
                                inx_ptr[j] = inx_ptr[(j - 1)] + 1

                            break
                    else:
                        for j in range(i + 1, n - 1):
                            inx_ptr[j] = inx_ptr[(j - 1)] + 1

                        break


class SmallGame(object):
    def __init__(self, players):
        self.over_order = OverOrder() #出完牌的顺序和进贡关系
        self.players = []
        self.deck = []
        self.current_action = Action()
        self.current_pos = -1
        self.greater_action = Action()
        self.greater_pos = -1
        self.action_list = ActionList()

        self.r_order = {'2':2,
         '3':3,  '4':4,  '5':5,  '6':6,  '7':7,  '8':8,  '9':9,  'T':10,  'J':11,  'Q':12,  'K':13,  'A':14,
         'B':16,  'R':17}
        # p_order表示在顺子、三连对等里的rank大小，用于比较
        self.p_order = {'A':1,
         '2':2,  '3':3,  '4':4,  '5':5,  '6':6,  '7':7,  '8':8,  '9':9,  'T':10,  'J':11,  'Q':12,  'K':13,  'B':16,
         'R':17}
        self.end = False
        self.card_play_action_seq = []
        self.all_players_last_move = {0:Action(),1:Action(),2:Action(),3:Action()}
        self.played_cards = {0:[],1:[],2:[],3:[]}
        self.last_move = Action()
        self.last_two_moves = [Action(),Action()]
        self.info_sets = {
            0: InfoSet(0), 1: InfoSet(1), 2: InfoSet(2), 3: InfoSet(3)
        }
        #self.players = [Player('0', 'random'), Player('1', 'random'), Player('2', 'random'), Player('3', 'random')]
        self.players = players
        self.rank = random.choice(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])
        self.r_order[self.rank] = 15
        self.current_pos = random.choice([0, 1, 2, 3])
        self.initialization()
        self.deal()

    def reset(self):
        self.over_order = OverOrder()  # 出完牌的顺序和进贡关系
        self.players = []
        self.deck = []
        self.current_action = Action()
        self.current_pos = -1
        self.greater_action = Action()
        self.greater_pos = -1
        self.action_list = ActionList()

        self.r_order = {'2': 2,
                        '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13,
                        'A': 14,
                        'B': 16, 'R': 17}
        # p_order表示在顺子、三连对等里的rank大小，用于比较
        self.p_order = {'A': 1,
                        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                        'K': 13, 'B': 16,
                        'R': 17}
        self.end = False
        self.card_play_action_seq = []
        self.all_players_last_move = {0: Action(), 1: Action(), 2: Action(), 3: Action()}
        self.played_cards = {0: [], 1: [], 2: [], 3: []}
        self.last_move = Action()
        self.last_two_moves = [Action(), Action()]
        self.info_sets = {
            0: InfoSet(0), 1: InfoSet(1), 2: InfoSet(2), 3: InfoSet(3)
        }
        self.players = [Player('0', 'random'), Player('1', 'random'), Player('2', 'random'), Player('3', 'random')]
        self.rank = random.choice(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])
        self.r_order[self.rank] = 15
        self.current_pos = random.choice([0, 1, 2, 3])
        self.initialization()
        self.deal()

    def __cmp2cards(self, card, other, flag):
        """
        比较两张卡牌，如果前者小于后者，返回True，反之False
        :param card: 卡牌a
        :param other: 卡牌b
        :param flag: 是否忽略花色
        :return: True or False
        """

        card_point = 15 if card.rank == self.rank else card.digital & 255
        other_point = 15 if other.rank == self.rank else other.digital & 255
        if card_point < other_point:
            return True
        else:
            if card_point == other_point:
                if flag:
                    return False
                else:
                    return card.digital & 65280 < other.digital & 65280
            return False

    def __cmp2rank(self, rank_a, rank_b):
        """
        比较两个rank，如果前者<后者返回1，前者>后者返回-1，相等返回0
        """
        if self.r_order[rank_a] < self.r_order[rank_b]:
            return 1
        else:
            if self.r_order[rank_a] > self.r_order[rank_b]:
                return -1
            return 0

    def next_pos(self):
        """
        获得下一位有牌的玩家的座位号
        """
        for i in range(1, 4):
            if len(self.players[((self.current_pos + i) % 4)].hand_cards) > 0:
                return (self.current_pos + i) % 4

    def update_player_message(self, pos, action_list):
        """
        不用网络传输，修改为直接update指定玩家的信息
        :param pos: 玩家座位号
        :param action_list: 动作列表
        :return:
        """
        self.players[pos].public_info = [player.get_public_info() for player in self.players]
        self.players[pos].greater_pos = self.greater_pos
        self.players[pos].greater_action = self.greater_action
        self.players[pos].action_list = action_list
        self.action_list.update(action_list)

    def initialization(self):
        """
        初始化牌堆
        :return: None
        """
        for i in range(2):
            digital_form = 257
            for s in Card.ALL_SUITS:
                for r in Card.ALL_RANKS[:-1]:
                    digital_form += 1
                    self.deck.append(Card(s, r, digital_form))

                digital_form = digital_form & 3841
                digital_form += 257

        for i in range(2):
            self.deck.append(Card('S', 'B', 272))
            self.deck.append(Card('H', 'R', 529))

        shuffle(self.deck)

    def deal(self):
        """
        分发卡牌给四位玩家
        :return: None
        """
        count = 107
        for i in range(4):
            if len(self.players[i].hand_cards) != 0:
                self.players[i].hand_cards = []
            self.players[i].uni_count = 0
            for j in range(27):
                if self.deck[count].rank == self.rank:
                    if self.deck[count].suit == 'H':
                        self.players[i].uni_count += 1
                self.add_card(i, self.deck[count])

                count -= 1
                self.deck.pop()

    def info_init(self):
        self.info_sets[0].player_hand_cards = self.players[0].hand_cards
        self.info_sets[1].player_hand_cards = self.players[1].hand_cards
        self.info_sets[2].player_hand_cards = self.players[2].hand_cards
        self.info_sets[3].player_hand_cards = self.players[3].hand_cards

    def start(self):
        action = self.first_action(self.current_pos)   #可选动作集合
        self.update_player_message(pos=self.current_pos, action_list=action)
        self.game_infoset = self.get_infoset()
        print("游戏开始，本小局打{}, {}号位先手".format(self.rank, self.current_pos))

    def step(self):
        self.game_infoset = self.get_infoset()
        action_index = self.players[self.current_pos].play_choice(self.game_infoset)
        action = self.action_list[action_index]
        self.play(action)

    def play(self, action):
        self.current_action = action
        print("{}号位玩家手牌为{}，打出{}，最大动作为{}号位打出的{}".format(self.current_pos, self.players[self.current_pos].hand_cards, action, self.greater_pos, self.greater_action))
        self.players[self.current_pos].play_area = str(action)
        self.players[self.current_pos].history.append(action)
        self.card_play_action_seq.append(action)
        self.all_players_last_move[self.current_pos] = action
        if action.type == PASS:
            if ((self.current_pos + 1) % 4 == self.greater_pos or (self.current_pos + 2) % 4 == self.greater_pos) and len(self.players[self.greater_pos].hand_cards) == 0:
                current_pos = (self.greater_pos + 2) % 4
                self.reset_all_action()
                act_func = self.first_action
            else:
                current_pos = self.next_pos()
                if current_pos == self.greater_pos:
                    self.reset_all_action()
                    act_func = self.first_action
                    #self.players[current_pos].reward += len(self.greater_action.cards)
                else:
                    act_func = self.second_action
            self.update_player_message(pos=current_pos, action_list=(act_func(current_pos)))
            self.current_pos = current_pos

        else:
            self.players[self.current_pos].play(action.cards, self.rank)
            self.greater_action = action
            self.greater_pos = self.current_pos

            if len(self.players[self.current_pos].hand_cards) == 0:
                self.over_order.add(self.current_pos)
                if self.over_order.episode_over(self.current_pos):
                    rest_cards = []
                    if len(self.over_order.order) != 4:
                        for i in range(4):
                            if i not in self.over_order.order:
                                rest_cards.append(i)
                                self.over_order.add(i)
                    self.end = True
                    print("本小局结束，排名顺序为{}".format(self.over_order.order))
                    return
                else:
                    current_pos = self.next_pos()
                    self.update_player_message(pos=current_pos, action_list=(self.second_action(current_pos)))
                    self.current_pos = current_pos
                    return
            else:
                current_pos = self.next_pos()
                self.update_player_message(pos=current_pos, action_list=(self.second_action(current_pos)))
                self.current_pos = current_pos


    def add_card(self, pos, card):
        """
        将卡牌添加到玩家手中，并保持牌序有序
        :param pos: 玩家座位号
        :param card: 卡牌
        :return: None
        """
        index = 0

        #temp_card = Card((card[0]), (card[1]), digital=(CARD_DIGITAL_TABLE[card]))
        while index != len(self.players[pos].hand_cards):
            if self._SmallGame__cmp2cards(card, self.players[pos].hand_cards[index], False):
                self.players[pos].hand_cards.insert(index, card)
                break
            index += 1

        if index == len(self.players[pos].hand_cards):
            self.players[pos].hand_cards.append(card)

    def reset_all_action(self):
        self.current_pos = -1
        self.current_action.reset()
        self.greater_pos = -1
        self.greater_action.reset()

    def separate_cards(self, index, repeat_flag):
        """
        将位置为index的玩家的有序手牌中的不同点数的单张分别划分到一个列表
        :param index: 玩家座位号
        :param repeat_flag: 列表中是否包含相同点数、相同花色的单张
        :return: List[List[Card]]
        """
        foo = self.players[index].hand_cards[0]
        result = []
        cards = []
        for card in self.players[index].hand_cards:
            if card.rank != foo.rank:
                result.append(cards)
                foo = card
                cards = [card]
            elif repeat_flag:
                foo = card
                cards.append(card)
            elif foo == card and len(cards) != 0:
                continue
            else:
                foo = card
                cards.append(card)

        result.append(cards)
        return result

    def extract_single(self, pos, _rank=None):
        """
        获取位置为pos的玩家的有序手牌中的单张（不包含重复单张），同样利用哨兵法。与separate_cards不同的是，extract_single将所有单张放在一
        个列表中，而不是分点数存放
        :param pos: 玩家座位号
        :param _rank: 是否筛选掉小于rank的单张
        :return: list[["Single", "rank", ['SW']], ...]
        """
        foo = self.players[pos].hand_cards[0]
        if _rank:
            if self.r_order[foo.rank] > self.r_order[_rank]:
                single = [
                 [
                  SINGLE, foo.rank, [str(foo)]]]
            else:
                single = list()
        else:
            single = [
             [
              SINGLE, foo.rank, [str(foo)]]]
        index = 0
        while index != len(self.players[pos].hand_cards):
            if foo == self.players[pos].hand_cards[index]:
                index += 1
                continue
            else:
                foo = self.players[pos].hand_cards[index]
                if _rank:
                    if self.r_order[foo.rank] > self.r_order[_rank]:
                        single.append([SINGLE, foo.rank, [str(foo)]])
                else:
                    single.append([SINGLE, foo.rank, [str(foo)]])

        return single

    def extract_same_cards(self, pos, same_cnt, _rank=None):
        """
        调用separate_cards获取某位玩家手中不同点数卡牌的列表，再调用comCards来获取某位玩家指定数量的相同点数的牌。（对子、炸弹）
        :param pos: 玩家座位号
        :param same_cnt: 相同点数的卡牌的数量
        :param _rank: 是否筛选掉小于rank的组合
        :return: list
        """
        if same_cnt == 2:
            name = PAIR
        else:
            if same_cnt == 3:
                name = TRIPS
            else:
                name = BOMB
        cards = self.separate_cards(pos, repeat_flag=True)
        same_cards = []
        uni_cnt = 0
        for card_list in cards:
            if _rank:
                if self.r_order[card_list[0].rank] <= self.r_order[_rank]:
                    continue
            else:
                if self.players[pos].uni_count == 1:
                    uni_cnt = 1
                elif self.players[pos].uni_count > 1:
                    uni_cnt = 1 if same_cnt == 2 else 2
                elif card_list[(-1)].rank != 'B' and card_list[(-1)].rank != self.rank:
                    for _ in range(uni_cnt):
                        card_list.append(Card('H', self.rank))

                elif card_list[(-1)].rank != 'R':
                    if card_list[(-1)].rank != self.rank:
                        for _ in range(uni_cnt):
                            card_list.append(Card('H', self.rank))

            if len(card_list) >= same_cnt:
                com_cards(same_cards, card_list, same_cnt, name=name)

        if same_cnt == 4:
            if name == BOMB:
                if self.players[pos].hand_cards[-4:] == ['SB', 'SB', 'HR', 'HR']:
                    same_cards.append([BOMB, 'JOKER', ['SB', 'SB', 'HR', 'HR']])
        return same_cards

    def extract_straight(self, pos, flush=False, _rank=None):
        """
        调用separate_cards获取某位玩家手中不同点数的卡牌集合，再用游标法提取该玩家手中的顺子
        :param pos: 玩家座位号
        :param flush: 所提取的是否为同花顺
        :param _rank: 是否筛选掉小于rank的顺子
        :return: list[str]
        """
        if len(self.players[pos].hand_cards) < 5:
            return []
        else:
            result = []
            ranks_inx = [-1 for _ in range(18)]
            cards_set = self.separate_cards(pos, repeat_flag=False)
            for i in range(len(cards_set)):
                foo = cards_set[i][0]
                if foo.digital & 255 == 14:
                    ranks_inx[1] = i
                ranks_inx[cards_set[i][0].digital & 255] = i

            if self.players[pos].uni_count:
                for _cards in cards_set:
                    if not _cards[0].rank == self.rank:
                        if _cards[0].rank == 'B' or _cards[0].rank == 'R':
                            pass
                        else:
                            _cards.append(Card('H', self.rank))

                cards_set.append([Card('H', self.rank)])
            else:
                cards_set.append([Card('H', 'R')])
            for i in range(18):
                if ranks_inx[i] == -1:
                    ranks_inx[i] = len(cards_set) - 1

            key = 1
            r_ptr = [0 for _ in range(5)]
            inx_ptr = [0 for _ in range(5)]
            for i in range(5):
                r_ptr[i] = ranks_inx[(key + i)]

            while key < 11:
                if _rank:
                    if self.p_order[cards_set[r_ptr[0]][inx_ptr[0]].rank] <= self.p_order[_rank]:
                        key += 1
                        for k in range(5):
                            r_ptr[k] = ranks_inx[(key + k)]
                            inx_ptr[k] = 0

                        continue
                else:
                    _guard_cnt = 0
                    _uni_cnt = 0
                    foo = []
                    for i in range(5):
                        temp = cards_set[r_ptr[i]][inx_ptr[i]]
                        if temp.rank == 'R':
                            _guard_cnt += 1
                        else:
                            if temp.rank == self.rank:
                                if temp.suit == 'H':
                                    _uni_cnt += 1
                        foo.append(str(temp))

                    if _guard_cnt > 0:
                        key += 1
                        for i in range(5):
                            inx_ptr[i] = 0
                            r_ptr[i] = ranks_inx[(key + i)]

                        continue
                    else:
                        if _uni_cnt > self.players[pos].uni_count:
                            pass
                        else:
                            rank = cards_set[r_ptr[0]][0].rank
                            if flush:
                                temp = set()
                                for card_str in foo:
                                    temp.add(card_str[0])

                                if len(temp) == 1:
                                    result.append([STRAIGHT_FLUSH, rank, foo])
                            else:
                                result.append([STRAIGHT, rank, foo])
                for i in range(4, -1, -1):
                    if inx_ptr[i] + 1 > len(cards_set[r_ptr[i]]) - 1:
                        inx_ptr[i] = 0
                        if i == 0:
                            key += 1
                            for k in range(5):
                                r_ptr[k] = ranks_inx[(key + k)]

                            break
                    else:
                        inx_ptr[i] += 1
                        break

            return result

    def extract_three_two(self, pos, _rank=None):
        """
        获取某位玩家手中的三带二牌型
        extract_straight负责获取顺子和同花顺
        extract_same_cards通过组合的方式获取三连对、钢板、三带二的牌型
        :param pos: 玩家座位号
        :param _rank: 是否筛选掉小于rank的三带二
        :return: list
        """
        three = self.extract_same_cards(pos, 3)
        two = self.extract_same_cards(pos, 2)
        three_two = []
        for three_cards in three:
            _, three_rank, cards = three_cards
            if _rank:
                if self.r_order[three_rank] <= self.r_order[_rank]:
                    continue
            for two_cards in two:
                _, two_rank, pair = two_cards
                if three_rank == two_rank:
                    continue
                temp_three_two = cards + pair
                counter = Counter(temp_three_two)
                rank = 'H{}'.format(self.rank)
                if rank in counter and counter[rank] > self.players[pos].uni_count:
                    continue
                else:
                    three_two.append([THREE_WITH_TWO, three_rank, temp_three_two])

        return three_two

    def extract_three_pair(self, pos, _rank=None):
        """
        三连对牌型提取
        :param pos: 玩家座位号
        :param _rank: 是否筛选掉小于rank的三连对
        :return: dict[str: list]
        """
        pair = self.extract_same_cards(pos, 2)
        temp_dict = defaultdict(list)
        for p in pair:
            _, rank, cards = p
            temp_dict[rank].append(cards)

        three_pair = []
        ranks = ('A23', '234', '345', '456', '567', '678', '789', '89T', '9TJ', 'TJQ',
                 'JQK', 'QKA')
        for rank in ranks:
            if _rank:
                if self.p_order[rank[0]] <= self.p_order[_rank]:
                    continue
                if rank[0] in temp_dict and rank[1] in temp_dict and rank[2] in temp_dict:
                    for pair_a, pair_b, pair_c in np.product(temp_dict[rank[0]], temp_dict[rank[1]], temp_dict[rank[2]]):
                        uni_rank = 'H' + self.rank
                        result = pair_a + pair_b + pair_c
                        temp_counter = Counter(result)
                        if uni_rank in temp_counter and temp_counter[uni_rank] > self.players[pos].uni_count:
                            continue
                        else:
                            three_pair.append([THREE_PAIR, rank[0], result])

        return three_pair

    def extract_two_trips(self, pos, _rank=None):
        """
        钢板牌型提取
        :param pos: 玩家座位号
        :param _rank: 是否筛选掉小于rank的钢板
        :return: dict[str: list]
        """
        trips = self.extract_same_cards(pos, 3)
        temp_dict = defaultdict(list)
        for t in trips:
            _, rank, cards = t
            temp_dict[rank].append(cards)

        two_trips = []
        ranks = ('A2', '23', '34', '45', '56', '67', '78', '89', '9T', 'TJ', 'JQ',
                 'QK', 'KA')
        for rank in ranks:
            if _rank:
                if self.p_order[rank[0]] <= self.p_order[_rank]:
                    continue
                if rank[0] in trips and rank[1] in trips:
                    for trip_a, trip_b in np.product(temp_dict[rank[0]], temp_dict[rank[1]]):
                        uni_rank = 'H' + self.rank
                        result = trip_a + trip_b
                        temp_counter = Counter(result)
                        if uni_rank in temp_counter and temp_counter[uni_rank] > self.players[pos].uni_count:
                            continue
                        else:
                            two_trips.append([TWO_TRIPS, rank[0], result])

        return two_trips

    def first_action(self, player_index):
        """
        获取某位玩家的先手动作
        :param player_index: 玩家座位号
        :return: list['single', 'r', 'cards']
        """
        action = []
        action.extend(self.extract_single(player_index))
        action.extend(self.extract_same_cards(player_index, 2))
        action.extend(self.extract_same_cards(player_index, 3))
        action.extend(self.extract_three_pair(player_index))
        action.extend(self.extract_three_two(player_index))
        action.extend(self.extract_two_trips(player_index))
        action.extend(self.extract_straight(player_index))
        action.extend(self.extract_straight(player_index, flush=True))
        for i in range(4, 11):
            action.extend(self.extract_same_cards(player_index, i))

        return action

    def second_action(self, player_index):
        action = [
         [
          PASS, PASS, PASS]]
        assert self.greater_action is not None
        add_boom = True
        add_flush = True
        rank = self.greater_action.rank
        if self.greater_action.type == SINGLE:
            action.extend(self.extract_single(player_index, _rank=rank))

        if self.greater_action.type == PAIR:
            action.extend(self.extract_same_cards(player_index, 2, _rank=rank))

        if self.greater_action.type == TRIPS:
            action.extend(self.extract_same_cards(player_index, 3, _rank=rank))

        if self.greater_action.type == THREE_PAIR:
            action.extend(self.extract_three_pair(player_index, _rank=rank))

        if self.greater_action.type == THREE_WITH_TWO:
            action.extend(self.extract_three_two(player_index, _rank=rank))

        if self.greater_action.type == TWO_TRIPS:
            action.extend(self.extract_two_trips(player_index, _rank=rank))

        if self.greater_action.type == STRAIGHT:
            if rank == 'T':
                pass
            else:
                action.extend(self.extract_straight(player_index, _rank=rank))

        if self.greater_action.type == STRAIGHT_FLUSH:
            if rank == 'T':
                pass
            else:
                action.extend(self.extract_straight(player_index, _rank=rank, flush=True))
            for i in range(6, 11):
                action.extend(self.extract_same_cards(player_index, i))

            add_boom, add_flush = (False, False)

        if self.greater_action.type == BOMB:
            add_boom, add_flush = (False, False)
            if rank == 'JOKER':
                pass
            else:
                bomb_len = len(self.greater_action.cards)
                action.extend(self.extract_same_cards(player_index, bomb_len, _rank=rank))
                for i in range(bomb_len + 1, 11):
                    action.extend(self.extract_same_cards(player_index, i))

                if bomb_len == 5:
                    action.extend(self.extract_straight(player_index, flush=True))

        if add_boom:
            for i in range(4, 11):
                action.extend(self.extract_same_cards(player_index, i))

        if add_flush:
            action.extend(self.extract_straight(player_index, flush=True))
        return action

    def get_last_move(self):
        last_move = Action()
        if len(self.card_play_action_seq) != 0:
            last_move = self.card_play_action_seq[-1]

        return last_move

    def get_last_two_moves(self):
        last_two_moves = [Action(), Action()]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_infoset(self):
        i = self.current_pos
        self.info_sets[i].my_pos = i
        self.info_sets[i].last_pid = self.greater_pos
        self.info_sets[i].legal_actions = self.action_list
        self.info_sets[i].last_move = self.get_last_move()
        self.info_sets[i].last_two_moves = self.get_last_two_moves()
        self.info_sets[i].all_players_last_move = self.all_players_last_move
        self.info_sets[i].num_cards_left_dict = {pos : len(self.players[pos].hand_cards) for pos in range(4)}
        self.info_sets[i].player_hand_cards = self.players[i].hand_cards
        self.info_sets[i].other_hand_cards = []
        
        for pos in range(4):
            self.info_sets[i].other_hand_cards += self.players[pos].hand_cards

        self.info_sets[i].played_cards = self.played_cards
        self.info_sets[i].card_play_action_seq = self.card_play_action_seq
        self.info_sets[i].all_handcards = {pos : self.players[pos].hand_cards for pos in range(4)}
        self.info_sets[i].rank = self.rank
        self.info_sets[i].current_action = self.current_action
        self.info_sets[i].greater_action = self.greater_action
        self.info_sets[i].greater_pos = self.greater_pos

        return deepcopy(self.info_sets[i])


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards, the historical moves, etc.
    """
    def __init__(self, player_position):
        self.player_position = player_position
        # The hand cards of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict = None
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other players for the current player
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the postions
        self.all_players_last_move = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict.
        self.all_handcards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None

if __name__ == '__main__':
    from player import Player
    import datetime
    start_time = datetime.datetime.now()
    g = SmallGame([Player('0', 'reyn'), Player('1', 'random'), Player('2', 'reyn'), Player('3', 'random')])
    #g = SmallGame([Player('0', 'rule'),Player('1', 'rule'), Player('2', 'rule'), Player('3', 'rule')])

    #g.reset()
    g.start()
    j = 0
    k = 0
    for i in range(1000):
        while g.end is False:
            g.step()
            j += 1
        if g.over_order.order[0] == 0 or g.over_order.order[0] == 2:
            k += 1
        g = SmallGame([Player('0', 'reyn'), Player('1', 'random'), Player('2', 'reyn'), Player('3', 'random')])

        g.start()
    print("0,2赢了{}".format(k))
    print("平均出牌次数{}".format(j/1000))
    end_time = datetime.datetime.now()
    print("平均每轮对局时间{}".format((end_time-start_time)/10))