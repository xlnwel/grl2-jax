import random
from collections import Counter, defaultdict
from itertools import product
import numpy as np

from .player import Player
from .utils import *
from .action import Action, ActionList
from .card import Card
from .infoset import InfoSet


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


class Game(object):
    def __init__(self,  
                 skip_players=(1, 3), 
                 agent='random', 
                 other='reyn', 
                 max_card=13, 
                 test=False,
                 evaluation=False,
                 **kwargs):
        self.over_order = OverOrder() #出完牌的顺序和进贡关系
        self.skip_players = skip_players
        self.players = [Player(f'{i}', other) if i in skip_players else Player(f'{i}', agent) 
            for i in range(4)]
        self.max_card = max_card
        self.test = test
        self.evaluation = evaluation

        self.r_order_default = {'2':2,
         '3':3,  '4':4,  '5':5,  '6':6,  '7':7,  '8':8,  '9':9,  'T':10,  'J':11,  'Q':12,  'K':13,  'A':14,
         'B':16,  'R':17}
        self.r_order = {'2':2,
         '3':3,  '4':4,  '5':5,  '6':6,  '7':7,  '8':8,  '9':9,  'T':10,  'J':11,  'Q':12,  'K':13,  'A':14,
         'B':16,  'R':17}
        # p_order表示在顺子、三连对等里的rank大小，用于比较
        self.p_order = {'A':1,
         '2':2,  '3':3,  '4':4,  '5':5,  '6':6,  '7':7,  '8':8,  '9':9,  'T':10,  'J':11,  'Q':12,  'K':13,  'B':16,
         'R':17}
        self.end = False
        
        # obs info
        self.numbers_shape = (13, 6 * 4 + 1)    # (6, 13, 5)
        self.jokers_shape = (6 * 4,)
        self.left_cards_shape = (3 * 27,)
        self.is_last_teammate_move_shape = (1,)
        self.is_first_move_shape = ()
        self.last_valid_action_type_shape = (9,)
        self.rank_shape = (13,)
        self.bombs_dealt_shape = (14,)
        self.last_action_numbers_shape = (4, 13, 5)
        self.last_action_jokers_shape = (4, 4)
        self.last_action_types_shape = (4, *self.last_valid_action_type_shape)
        self.last_action_rel_pids_shape = (4, 4)
        self.last_action_filters_shape = (4,)
        self.last_action_first_move_shape = (4,)
        self.action_type_mask_shape = (NUM_ACTION_TYPES,)
        self.card_rank_mask_shape = (NUM_ACTION_TYPES, NUM_CARD_RANKS)
        self.others_numbers_shape = (13, 13)
        self.others_jokers_shape = (12,)
        self.obs_shape = dict(
            numbers=self.numbers_shape,
            jokers=self.jokers_shape,
            left_cards=self.left_cards_shape,
            is_last_teammate_move=self.is_last_teammate_move_shape,
            is_first_move=self.is_first_move_shape,
            last_valid_action_type=self.last_valid_action_type_shape,
            rank=self.rank_shape,
            bombs_dealt=self.bombs_dealt_shape,
            last_action_numbers=self.last_action_numbers_shape,
            last_action_jokers=self.last_action_jokers_shape,
            last_action_types=self.last_action_types_shape,
            last_action_rel_pids=self.last_action_rel_pids_shape,
            last_action_filters=self.last_action_filters_shape,
            last_action_first_move=self.last_action_first_move_shape,
            action_type_mask=self.action_type_mask_shape,
            card_rank_mask=self.card_rank_mask_shape,
            mask=()
        )
        self.obs_dtype = dict(
            numbers=np.float32,
            jokers=np.float32,
            left_cards=np.float32,
            is_last_teammate_move=np.float32,
            is_first_move=np.bool,
            last_valid_action_type=np.float32,
            rank=np.float32,
            bombs_dealt=np.float32,
            last_action_numbers=np.float32,
            last_action_jokers=np.float32,
            last_action_types=np.float32,
            last_action_rel_pids=np.float32,
            last_action_filters=np.bool,
            last_action_first_move=np.float32,
            action_type_mask=np.bool,
            card_rank_mask=np.bool,
            mask=np.float32
        )

        if not self.evaluation:
            self.obs_shape.update(dict(
                others_numbers=self.others_numbers_shape,
                others_jokers=self.others_jokers_shape,
            ))
            self.obs_dtype.update(dict(
                others_numbers=np.float32,
                others_jokers=np.float32,
            ))

        # action info
        self.action_type_shape = ()
        self.card_rank_shape = ()
        self.action_shape = dict(
            action_type=self.action_type_shape,
            card_rank=self.card_rank_shape,
        )
        self.action_dim = dict(
            action_type=NUM_ACTION_TYPES,
            card_rank=NUM_CARD_RANKS,
        )
        self.action_dtype = dict(
            action_type=np.int32,
            card_rank=np.int32,
        )

        self.reward_shape = (4,)

    def game_over(self):
        return self.end

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
            pid = (self.current_pid + i) % 4
            if len(self.players[pid].hand_cards) > 0:
                return pid
            else:
                self._skip_player(pid)

    def update_player_message(self, pos, legal_actions):
        """
        不用网络传输，修改为直接update指定玩家的信息
        :param pos: 玩家座位号
        :param legal_actions: 动作列表
        :return:
        """
        self.players[pos].public_info = [player.get_public_info() for player in self.players]
        self.players[pos].last_pid = self.last_pid
        self.players[pos].last_action = self.last_action
        self.players[pos].last_valid_pid = self.last_valid_pid
        self.players[pos].last_valid_action = self.last_valid_action
        self.players[pos].legal_actions = legal_actions
        self.legal_actions.update(legal_actions)

    def reset(self, deal=True, rank=None):
        self.over_order = OverOrder() #出完牌的顺序和进贡关系
        self.deck = []
        self.last_action = Action()
        self.is_last_action_first_move = False
        self.last_valid_action = Action()
        self.current_pid = -1
        self.last_pid = -1
        self.last_valid_pid = -1
        self.legal_actions = ActionList()
        self.end = False
        self.played_action_seq = [Action() for _ in range(4)]
        if rank is None:
            self.rank = random.choice(['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'][:self.max_card])
        else:
            self.rank = rank
        if self.test:
            print('rank:', self.rank)
        self.current_pid = random.choice([0, 1, 2, 3])
        self.first_pid = self.current_pid
        self.r_order = self.r_order_default.copy()
        self.r_order[self.rank] = 15
        self.bombs_dealt = np.zeros(self.bombs_dealt_shape, dtype=np.float32)
        [p.reset() for p in self.players]
        self.initialize()
        if deal:
            self.deal()

    def initialize(self):
        """
        初始化牌堆
        :return: None
        """
        for i in range(2):
            digital_form = 257
            for s in Card.ALL_SUITS:
                for r in Card.ALL_RANKS[:-1]:
                    digital_form += 1
                    if Card.RANK2DIGIT[r] <= self.max_card:
                        self.deck.append(Card(s, r, digital_form))

                digital_form = digital_form & 3841
                digital_form += 257

        for i in range(2):
            self.deck.append(Card('S', 'B', 272))
            self.deck.append(Card('H', 'R', 529))

        random.shuffle(self.deck)

    def deal(self):
        """
        分发卡牌给四位玩家
        :return: None
        """
        count = len(self.deck) - 1
        for i in range(4):
            if len(self.players[i].hand_cards) != 0:
                self.players[i].hand_cards = []
            self.players[i].uni_count = 0
            for j in range(2*self.max_card+1):
                if self.deck[count].rank == self.rank:
                    if self.deck[count].suit == 'H':
                        self.players[i].uni_count += 1
                self.add_card(i, self.deck[count])

                count -= 1
                self.deck.pop()

    def start(self):
        legal_actions = self.first_action(self.current_pid)   #可选动作集合
        self.update_player_message(pos=self.current_pid, legal_actions=legal_actions)
        self.check_skip_player()
        # print("游戏开始，本小局打{}, {}号位先手".format(self.rank, self.current_pid))

    def random_action(self):
        infoset = self.get_infoset()
        action_index = self.players[self.current_pid].play_choice(infoset)
        action = self.legal_actions[action_index]
        return action

    def random_step(self):
        action = self.random_action()
        self.play(action)

    def play(self, action):
        assert len(self.players[self.current_pid].hand_cards) > 0, len(self.players[self.current_pid].hand_cards)
        if not isinstance(action, Action):
            action = self.legal_actions[action]
        if self.current_pid == self.first_pid:
            assert len(self.players[0].history) == len(self.players[1].history), (self.players[0].history, self.players[1].history)
            assert len(self.players[1].history) == len(self.players[2].history), (self.players[1].history, self.players[2].history)
            assert len(self.players[2].history) == len(self.players[3].history), (self.players[2].history, self.players[3].history)
        assert action.type is not None, action
        if action.type == BOMB or action.type == STRAIGHT_FLUSH:
            self.bombs_dealt[Rank2Num[action.rank]] += 1
        if self.players[self.current_pid].first_round:
            assert self.players[self.current_pid].history == [], 0
        self.players[self.current_pid].first_round = False
        self.last_pid = self.current_pid
        self.last_action = action
        self.is_last_action_first_move = self.last_valid_pid == -1
        if self.is_last_action_first_move:
            assert action.type != PASS, action
        if self.test:
            print("{}号位玩家手牌为{}，打出{}，最大动作为{}号位打出的{}. first action={}".format(
                self.current_pid, 
                self.players[self.current_pid].hand_cards, 
                action, 
                self.last_valid_pid, 
                self.last_valid_action,
                self.is_last_action_first_move))

        self.players[self.current_pid].play_area = str(action)
        self.players[self.current_pid].history.append(action.copy())
        self.players[self.current_pid].is_last_action_first_move = self.is_last_action_first_move
        self.played_action_seq.append(action.copy())
        teammate_id = get_teammate_pid(self.current_pid)
        down_id = get_down_pid(self.current_pid)
        if action.type == PASS:
            if (teammate_id == self.last_valid_pid and len(self.players[teammate_id].hand_cards) == 0 and len(self.players[down_id].hand_cards) == 0) \
                or (down_id == self.last_valid_pid and len(self.players[down_id].hand_cards) == 0):
                current_pid = (self.last_valid_pid + 2) % 4
                pid = down_id
                while pid != current_pid:
                    self._skip_player(pid)
                    pid = (pid + 1) % 4
                self.reset_all_action()
                act_func = self.first_action
            else:
                current_pid = self.next_pos()
                if current_pid == self.last_valid_pid:
                    self.reset_all_action()
                    act_func = self.first_action
                    #self.players[current_pid].reward += len(self.last_valid_action.cards)
                else:
                    act_func = self.second_action
            self.update_player_message(pos=current_pid, legal_actions=act_func(current_pid))
            self.current_pid = current_pid

        else:
            self.players[self.current_pid].play(action.cards, self.rank)
            self.last_valid_action = action
            self.last_valid_pid = self.current_pid

            if len(self.players[self.current_pid].hand_cards) == 0:
                self.over_order.add(self.current_pid)
                if self.over_order.episode_over(self.current_pid):
                    rest_cards = []
                    if len(self.over_order.order) != 4:
                        for i in range(4):
                            if i not in self.over_order.order:
                                rest_cards.append(i)
                                self.over_order.add(i)
                    self.end = True
                    # print("本小局结束，排名顺序为{}".format(self.over_order.order))
                current_pid = self.next_pos()
                self.update_player_message(pos=current_pid, legal_actions=(self.second_action(current_pid)))
                self.current_pid = current_pid
            else:
                current_pid = self.next_pos()
                self.update_player_message(pos=current_pid, legal_actions=(self.second_action(current_pid)))
                self.current_pid = current_pid
        if self.end:
            return
        self.check_skip_player()

    def check_skip_player(self):
        if self.current_pid in self.skip_players:
            infoset = self.get_infoset()
            aid = self.players[self.current_pid].play_choice(infoset)
            self.play(aid)
        
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
            if self._Game__cmp2cards(card, self.players[pos].hand_cards[index], False):
                self.players[pos].hand_cards.insert(index, card)
                break
            index += 1

        if index == len(self.players[pos].hand_cards):
            self.players[pos].hand_cards.append(card)

    def reset_all_action(self):
        self.current_pid = -1
        self.last_pid = -1
        self.last_action.reset()
        self.last_valid_pid = -1
        self.last_valid_action.reset()

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

                if card_list[(-1)].rank != 'B' and card_list[(-1)].rank != 'R' and card_list[(-1)].rank != self.rank:
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
                for pair_a, pair_b, pair_c in product(temp_dict[rank[0]], temp_dict[rank[1]], temp_dict[rank[2]]):
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
            if rank[0] in temp_dict and rank[1] in temp_dict:
                for trip_a, trip_b in product(temp_dict[rank[0]], temp_dict[rank[1]]):
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
        assert self.last_valid_action is not None
        add_boom = True
        add_flush = True
        rank = self.last_valid_action.rank
        if self.last_valid_action.type == SINGLE:
            action.extend(self.extract_single(player_index, _rank=rank))

        if self.last_valid_action.type == PAIR:
            action.extend(self.extract_same_cards(player_index, 2, _rank=rank))

        if self.last_valid_action.type == TRIPS:
            action.extend(self.extract_same_cards(player_index, 3, _rank=rank))

        if self.last_valid_action.type == THREE_PAIR:
            action.extend(self.extract_three_pair(player_index, _rank=rank))

        if self.last_valid_action.type == THREE_WITH_TWO:
            action.extend(self.extract_three_two(player_index, _rank=rank))

        if self.last_valid_action.type == TWO_TRIPS:
            action.extend(self.extract_two_trips(player_index, _rank=rank))

        if self.last_valid_action.type == STRAIGHT:
            if rank == 'T':
                pass
            else:
                action.extend(self.extract_straight(player_index, _rank=rank))

        if self.last_valid_action.type == STRAIGHT_FLUSH:
            if rank == 'T':
                pass
            else:
                action.extend(self.extract_straight(player_index, _rank=rank, flush=True))
            for i in range(6, 11):
                action.extend(self.extract_same_cards(player_index, i))

            add_boom, add_flush = (False, False)

        if self.last_valid_action.type == BOMB:
            add_boom, add_flush = (False, False)
            if rank == 'JOKER':
                pass
            else:
                bomb_len = len(self.last_valid_action.cards)
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

    def get_infoset(self):
        i = self.current_pid
        infoset = InfoSet(i, evaluation=self.evaluation)
        infoset.first_round = self.players[i].first_round
        infoset.last_pid = self.last_pid
        infoset.last_action = self.last_action
        infoset.last_valid_pid = self.last_valid_pid
        infoset.last_valid_action = self.last_valid_action
        infoset.legal_actions = self.legal_actions
        infoset.all_last_actions = [
            p.history[-1] if len(p.history) > 0 else Action() 
            for p in self.players]
        infoset.all_last_action_first_move = [p.is_last_action_first_move for p in self.players]
        infoset.all_num_cards_left = [len(p.hand_cards) for p in self.players]
        infoset.player_hand_cards = self.players[i].hand_cards
        infoset.all_hand_cards = [p.hand_cards for p in self.players]
        played_actions = [p.history for p in self.players]
        infoset.played_cards = [sum([get_cards(a) for a in actions], []) 
            for actions in played_actions]
        infoset.played_action_seq = self.played_action_seq
        infoset.rank = self.rank
        infoset.bombs_dealt = self.bombs_dealt

        return infoset

    def compute_reward(self):
        order = self.over_order.order
        first_id = order[0]
        first_teammate_id = (first_id + 2) % 4
        team_ids = [first_id, first_teammate_id]
        reward = np.zeros(4, dtype=np.float32)
        if first_teammate_id == order[1]:
            reward[team_ids] = 3
        elif first_teammate_id == order[2]:
            reward[team_ids] = 2
        else:
            reward[team_ids] = 1
        return reward

    """ Reset to any point """
    def reset_to_any(self, current_rank, first_pos, n0, n1, n2, n3, list0=None, list1=None, list2=None, list3=None):
        self.reset(deal=False, rank=current_rank)
        self.current_pid = first_pos
        self.deal_to_any(n0, n1, n2, n3, list0, list1, list2, list3)

    def deal_to_any(self, n0, n1, n2, n3, list0=None, list1=None, list2=None, list3=None):
        """
        分发卡牌给四位玩家
        :return: None
        """
        count = 107
        self.deal_to_pos(count, 0, n0, list0)
        count = count - n0
        self.deal_to_pos(count, 1, n1, list1)
        count = count - n1
        self.deal_to_pos(count, 2, n2, list2)
        count = count - n2
        self.deal_to_pos(count, 3, n3, list3)
        count = count - n3

    def deal_to_pos(self, count, pos, n, card_list):
        if len(self.players[pos].hand_cards) != 0:
            self.players[pos].hand_cards = []
        self.players[pos].uni_count = 0
        if card_list is None:
            for j in range(n):
                if self.deck[count].rank == self.rank:
                    if self.deck[count].suit == 'H':
                        self.players[pos].uni_count += 1
                self.add_card(pos, self.deck[count])

                count -= 1
                self.deck.pop()
        else:
            card_list = card_list.split(',')

            for card in card_list:
                card = card.strip()
                card = Card(card[0], card[1], CARD_DIGITAL_TABLE[card])

                if card.rank == self.rank and card.suit == 'H':
                    self.players[pos].uni_count += 1
                self.add_card(pos, card)

                count -= 1
                self.deck.remove(card)

    def _skip_player(self, pid):
        self.players[pid].history.append(Action())
        self.players[pid].is_last_action_first_move = False
        self.played_action_seq.append(Action())

    def close(self):
        pass


if __name__ == '__main__':
    import datetime
    import pickle
    import time
    env = Game([Player('0', 'random'), Player('1', 'random'), Player('2', 'random'), Player('3', 'random')])
    n = 1
    # rewards = []
    # start_time = datetime.datetime.now()
    # infoset_file_name = 'small' + str(int(time.time())) + '.pkl'
    # file = open(infoset_file_name, 'wb')
    # for _ in range(n):
    #     hand_cards_0 = 'S3, H3, H3, S4, H4, H4, C4, D4, D4, H5, S6, S8, ST, HT, DT, SQ, DQ, SK, SA, HA, HA, S2, H2, H2, D2, D2, HR'
    #     hand_cards_1 = 'S3, C3, D3, D3, S4, S5, C5, C5, D5, S6, S7, C7, D7, H8, C8, D8, ST, HT, CT, DT, HQ, DQ, SA, S2, C2, SB, SB'
    #     hand_cards_2 = 'C4, S5, H6, D6, S7, H7, D7, H8, D8, H9, C9, C9, CT, SJ, HJ, CJ, HQ, HK, CK, CK, DK, CA, CA, DA, DA, C2, HR'
    #     hand_cards_3 = 'S3, C3, D4, D4, D5, H6, C6, C6, D6, H7, D7, H8, D8, S9, H9, C9, D9, DT, HJ, SQ, HQ, SK, DK, CA, CA, C2, HR'
    #     env.reset_to_any('2', 2, 27, 27, 27, 27, hand_cards_0, hand_cards_1, hand_cards_2, hand_cards_3)
    #     #env.reset()
    #     env.start()
    #     # step = 0
    #     from cxw_agent import CXWAgent
    #     agent0 = CXWAgent()
    #     agent1 = CXWAgent()
    #     agent2 = CXWAgent()
    #     agent3 = CXWAgent()
    #     agent = [agent0, agent1, agent2, agent3]
    #     while env.game_over() is False:
    #
    #         infoset = env.get_infoset()
    #         pickle.dumps(infoset)
    #         if len(infoset.legal_actions.action_list) == 1:
    #             action_id = 0
    #         else:
    #             from infoset import get_obs
    #             obs = get_obs(infoset)
    #
    #             batch_obs = obs.copy()
    #             batch_obs['eid'] = 0
    #             for k, v in batch_obs.items():
    #                 batch_obs[k] = np.expand_dims(v, 0)
    #             reward = np.expand_dims(0, 0)
    #             discount = np.expand_dims(1, 0)
    #             reset = np.expand_dims(infoset.first_round, 0)
    #             env_output = EnvOutput(batch_obs, reward, discount, reset)
    #
    #             (action_type, card_rank), _ = agent[env.current_pid].agent(env_output, True, False)
    #             action_type = action_type[0]
    #             card_rank = card_rank[0]
    #             if action_type == 0:
    #                 card_rank = 0
    #             action_id = infoset.action2id(action_type, card_rank)
    #             print(infoset.legal_actions[action_id])
    #         env.play(action_id)
    #
    #         #print(env.game_over())
    #         # step += 1
    #         # print(step)
    end_time = datetime.datetime.now()
