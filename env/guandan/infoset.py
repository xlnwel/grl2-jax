import numpy as np
from env.guandan.action import get_card_type

from env.guandan.utils import *


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards, the historical moves, etc.
    """
    def __init__(self, pid):
        self.pid = pid
        # Last player id
        self.last_pid = None
        # The cards played by self.last_pid
        self.last_action = None
        # Is the last action of each player the first move
        self.all_last_action_first_move = None
        # Last player id that plays a valid move, i.e., not PASS
        self.last_valid_pid = None
        # The cards played by self.last_valid_pid
        self.last_valid_action = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The last actions for all the postions
        self.all_last_actions = None
        # The number of cards left for each player. It is a dict with str-->int
        self.all_num_cards_left = None
        # The hand cards of the current player. A list.
        self.player_hand_cards = None
        # list of the hand cards of all the players
        self.all_hand_cards = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The historical moves. It is a list of list
        self.played_action_seq = None
        # The largest number
        self.rank = None
        # bombs dealt so far
        self.bombs_dealt = None

def _get_one_hot_array(i,n):
    """
    A utility function to obtain one-hot encoding
    """
    one_hot = np.zeros(n, dtype=np.float32)
    one_hot[i - 1] = 1

    return one_hot

def _cards_repr(cards):
    numbers = np.zeros([13, 4], dtype=np.float32)
    jokers = np.zeros(4, dtype=np.float32)
    result = dict(numbers=numbers, jokers=jokers)
    if len(cards) == 0:
        return result

    b_jokers = 0
    r_jokers = 0
    for card in cards:
        if card.rank == 'B':
            b_jokers += 1
        elif card.rank == 'R':
            r_jokers += 1
        else:
            row = Card2Num[card.rank]
            column = Suit2Num[card.suit]
            numbers[row, column] += 1
    numbers /= 2
    if b_jokers:
        jokers[b_jokers-1] = 1
    if r_jokers:
        jokers[r_jokers+1] = 1

    return result

def _action_repr(action, rank):
    numbers = np.zeros([13, 5], dtype=np.float32)
    jokers = np.zeros(4, dtype=np.float32)
    action_type = get_card_type(action)
    result = dict(numbers=numbers, jokers=jokers, action_type=action_type)
    
    if action.type is None or action.type == PASS:
        np.testing.assert_equal(action_type, 0)
        return result

    numbers[:, 4] = rank

    for card in action.cards:
        if card[1] == 'B':
            if jokers[0] == 1:
                jokers[0] = 0
                jokers[1] = 1
            else:
                jokers[0] = 1
        elif card[1] == 'R':
            if jokers[2] == 1:
                jokers[2] = 0
                jokers[3] = 1
            else:
                jokers[2] = 1
        else:
            row = Card2Num[card[1]]
            column = Suit2Num[card[0]]
            numbers[row, column] += 1
    numbers /= 2

    return result

def _get_policy_mask(infoset: InfoSet, is_first_move):
    # TODO: Try more action types
    card_type_mask = np.zeros(4, dtype=np.bool)    # pass, follow, bomb
    follow_mask = np.zeros(15, dtype=np.bool)
    bomb_mask = np.zeros(15, dtype=np.bool)
    for a in infoset.legal_actions:
        assert a.type is not None, a.type
        if a.type == PASS:
            card_type_mask[0] = 1
        else:
            i = Action2Num[a.rank]
            if a.type == BOMB or a.type == STRAIGHT_FLUSH:
                card_type_mask[2] = 1
                bomb_mask[i] = 1
            else:
                card_type_mask[1] = 1
                follow_mask[i] = 1
    assert np.any(card_type_mask), card_type_mask
    assert card_type_mask[0] != is_first_move, (card_type_mask, is_first_move, [a.cards for a in infoset.legal_actions])
    if card_type_mask[1] == False:
        assert np.all(follow_mask == False), follow_mask
    if card_type_mask[2] == False:
        assert np.all(bomb_mask == False), bomb_mask
    return card_type_mask, follow_mask, bomb_mask

def _get_rel_pids():
    # pids = [np.zeros(4, dtype=np.float32) for _ in range(4)]
    # for i, pid in enumerate(pids):
    #     pid[i] = 1
    
    # return np.stack(pids)
    return np.eye(4)

def get_obs(infoset: InfoSet, separate_jokers=True):
    pid = infoset.pid
    down_pid = get_down_pid(infoset.pid)
    teammate_pid = get_teammate_pid(infoset.pid)
    up_pid = get_up_pid(infoset.pid)
    pids = [pid, down_pid, teammate_pid, up_pid]
    others_pids = pids[1:]
    others_hand_cards = [infoset.all_hand_cards[i] for i in others_pids]
    
    """ Observations """
    rank = Card2Num[infoset.rank]
    rank_repr = _get_one_hot_array(rank, 13)
    rank_exp_repr = np.expand_dims(rank_repr, -1)
    cards_reprs = [_cards_repr(cards) for cards in [
        infoset.player_hand_cards,
        sum(others_hand_cards, []),
        *[infoset.played_cards[i] for i in pids]
    ]]
    numbers = np.concatenate([c['numbers'] for c in cards_reprs]+[rank_exp_repr], axis=-1)
    jokers = np.concatenate([c['jokers'] for c in cards_reprs], axis=-1)
    others_num_cards_left = [infoset.all_num_cards_left[i] for i in others_pids]
    others_num_cards_left_repr = [_get_one_hot_array(n, 27) for n in others_num_cards_left]
    left_cards = np.concatenate(others_num_cards_left_repr, axis=-1)
    # left_cards = np.array([
    #     down_num_cards_left,
    #     teammate_num_cards_left,
    #     up_num_cards_left
    # ])
    bombs_dealt = infoset.bombs_dealt
    is_last_teammate_move = infoset.last_valid_pid == teammate_pid
    is_last_teammate_move_repr = is_last_teammate_move * np.ones(1, dtype=np.float32)
    is_first_move = infoset.last_pid == -1
    last_valid_action_type = get_card_type(infoset.last_valid_action)
    if is_first_move:
        assert infoset.last_valid_pid == -1
        assert infoset.last_action.type is None
        assert infoset.last_valid_action.type is None
        np.testing.assert_equal(last_valid_action_type, 0)

    """ History Actions """
    last_actions = [infoset.all_last_actions[i] for i in pids]
    last_action_reprs = [_action_repr(a, rank) for a in last_actions]
    assert np.all([a1 == a2 for a1, a2 in zip(last_actions, infoset.played_action_seq[-4:])])
    last_action_numbers = np.stack([a['numbers'] for a in last_action_reprs])
    last_action_jokers = np.stack([a['jokers'] for a in last_action_reprs])
    last_action_types = np.stack([a['action_type'] for a in last_action_reprs])
    last_action_rel_pids = _get_rel_pids()
    last_action_filter = np.array([a.type is not None for a in last_actions], dtype=np.bool)
    last_action_first_move = np.array([infoset.all_last_action_first_move[i] for i in pids], dtype=np.bool)
    assert np.sum(last_action_first_move) < 2, (last_actions, last_action_first_move)

    """ Unobservable Info: Others' Cards """
    others_handcards = [_cards_repr(c) for c in others_hand_cards]
    others_numbers = np.concatenate([c['numbers'] for c in others_handcards]+[rank_exp_repr], axis=-1)
    others_jokers = np.concatenate([c['jokers'] for c in others_handcards], axis=-1)
    """ Policy Mask """
    card_type_mask, follow_mask, bomb_mask = _get_policy_mask(infoset, is_first_move)
    if is_first_move:
        assert card_type_mask[0] == False, card_type_mask
    else:
        assert card_type_mask[0] == True, card_type_mask

    obs = {
        'pid': pid,
        'numbers': numbers,
        'jokers': jokers,
        'left_cards': left_cards,
        'is_last_teammate_move': is_last_teammate_move,
        'is_first_move': is_first_move,
        'last_valid_action_type': last_valid_action_type,
        'rank': rank_repr,
        'bombs_dealt': bombs_dealt,
        'last_action_numbers': last_action_numbers,
        'last_action_jokers': last_action_jokers,
        'last_action_types': last_action_types,
        'last_action_rel_pids': last_action_rel_pids,
        'last_action_filter': last_action_filter,
        'last_action_first_move': last_action_first_move,
        'card_type_mask': card_type_mask,
        'follow_mask': follow_mask,
        'bomb_mask': bomb_mask,
        'others_numbers': others_numbers,
        'others_jokers': others_jokers,
    }
    return obs
