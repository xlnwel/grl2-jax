from collections import Counter
import numpy as np
from env.guandan.player import Player
from guandan.small_game import SmallGame
from guandan.action import Action

Card2Column = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6,
               '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
Suit2Row = {'S':0, 'H':1, 'C':2, 'D':3}
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

class Env:
    def __init__(self):
        """
        """
        # Initialize players
        # We use for dummy player for the target position
        # Initialize the internal environment
        #self.players = [Player('0'), Player('1'), Player('2'), Player('3')]
        players = [Player('0'), Player('1'), Player('2'), Player('3')]
        self._env = SmallGame()
        self._env.players = players
        self.infoset = None

    def random_action(self):
        return self._env.random_action()

    def reset(self):
        self._env.reset()
        self._env.info_init()
        self._env.start()
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        #self._env.players[self._acting_player_position].set_action(action)
        self._env.play(action)
        done = False
        reward = 0.0
        info = {}

        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            self.infoset = self._game_infoset
            obs = get_obs(self.infoset)
        return obs, reward, done, info

    def _get_reward(self):
        order = self._env.over_order.order
        reward = {0: 0, 1: 0, 2: 0, 3: 0}
        if (order[0] + 2) % 4 == order[1]:
            reward[order[0]] = 3
        elif (order[0] + 2) % 4 == order[2]:
            reward[order[0]] = 2
        else:
            reward[order[0]] = 1
        # first, first_mate, inc = self._env.over_order.settlement()
        # if first == 0 or first_mate == 0:
        #     reward = inc
        # else:
        #     reward = 0
        return reward

    @property
    def _game_infoset(self):
        """
        Here, infoset is defined as all the information
        in the current situation, including the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perfect information.
        """
        return self._env.get_infoset()

    @property
    def _acting_player_position(self):
        """
        The player that is active.
        """
        return self._env.current_pos

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.end


def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset.

    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """

    return _get_obs(infoset)


def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot


def _cards2array(list_cards):
    if len(list_cards) == 0:
        return np.zeros(108, dtype=np.int8)

    matrix = np.zeros([8, 13], dtype=np.int8)
    jokers = np.zeros(4, dtype=np.int8)

    # TODO: use a 4 x 15 matrix instead of 8 x 15
    for card in list_cards:
        if card.rank != 'R' and card.rank != 'B':
            column = Card2Column[card.rank]
            row = Suit2Row[card.suit]
            if matrix[row, column] == 1:
                matrix[row + 4, column] = 1
            else:
                matrix[row, column] = 1
        elif card.rank == 'B':
            if jokers[0] == 1:
                jokers[1] = 1
            else:
                jokers[0] = 1
        elif card.rank == 'R':
            if jokers[2] == 1:
                jokers[3] = 1
            else:
                jokers[2] = 1

    matrix = np.concatenate((matrix.flatten('F'), jokers))
    return matrix

def _action2array(action):
    if action.type == PASS or action.type is None :
        return np.zeros(108, dtype=np.int8)

    matrix = np.zeros([8, 13], dtype=np.int8)
    jokers = np.zeros(4, dtype=np.int8)

    for card in action.cards:
        if card[1] != 'R' and card[1] != 'B':
            column = Card2Column[card[1]]
            row = Suit2Row[card[0]]
            if matrix[row, column] == 1:
                matrix[row + 4, column] = 1
            else:
                matrix[row, column] = 1
        elif card[1] == 'B':
            if jokers[0] == 1:
                jokers[1] = 1
            else:
                jokers[0] = 1
        elif card[1] == 'R':
            if jokers[2] == 1:
                jokers[3] = 1
            else:
                jokers[2] = 1

    matrix = np.concatenate((matrix.flatten('F'), jokers))
    return matrix

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 12 actions. If there is
    no 12 actions, we pad the features with 0.
    """
    action_seq_array = np.zeros((len(action_seq_list), 108))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _action2array(list_cards)
    action_seq_array = action_seq_array.reshape(3, 432)
    return action_seq_array


def _process_action_seq(sequence, length=12):
    """
    A utility function encoding historical moves. We
    encode 12 moves. If there is no 12 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [Action() for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_obs(infoset):
    """
    """
    num_legal_actions = len(infoset.legal_actions.action_list)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _action2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _action2array(action)

    last_teammate_action = _action2array(
        infoset.last_move_dict[(infoset.player_position+2)% 4])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict[(infoset.player_position+2)% 4], 27)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards[(infoset.player_position+2)% 4])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_down_action = _action2array(
        infoset.last_move_dict[(infoset.player_position+1)% 4])
    last_down_action_batch = np.repeat(
        last_down_action[np.newaxis, :],
        num_legal_actions, axis=0)

    down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict[(infoset.player_position+1)% 4], 27)
    down_num_cards_left_batch = np.repeat(
        down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    down_played_cards = _cards2array(
        infoset.played_cards[(infoset.player_position+1)% 4])
    down_played_cards_batch = np.repeat(
        down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_up_action = _action2array(
        infoset.last_move_dict[(infoset.player_position+3)% 4])
    last_up_action_batch = np.repeat(
        last_up_action[np.newaxis, :],
        num_legal_actions, axis=0)

    up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict[(infoset.player_position+3)% 4], 27)
    up_num_cards_left_batch = np.repeat(
        up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    up_played_cards = _cards2array(
        infoset.played_cards[(infoset.player_position+3)% 4])
    up_played_cards_batch = np.repeat(
        up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    rank = _get_one_hot_array(
        Card2Column[infoset.rank], 13)
    rank_batch = np.repeat(rank[np.newaxis, :], num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         teammate_played_cards_batch,
                         down_played_cards_batch,
                         up_played_cards_batch,
                         last_action_batch,
                         last_teammate_action_batch,
                         last_down_action_batch,
                         last_up_action_batch,
                         teammate_num_cards_left_batch,
                         down_num_cards_left_batch,
                         up_num_cards_left_batch,
                         my_action_batch,
                         rank_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             teammate_played_cards,
                             down_played_cards,
                             up_played_cards,
                             last_action,
                             last_teammate_action,
                             last_down_action,
                             last_up_action,
                             teammate_num_cards_left,
                             down_num_cards_left,
                             up_num_cards_left,
                             rank))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)

    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.float32),
    }
    return obs

