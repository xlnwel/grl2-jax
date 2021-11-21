import numpy as np
from env.guandan.player import Player
from env.guandan.small_game import InfoSet, SmallGame, get_down_pid, get_teammate_pid, get_up_pid
from env.guandan.action import Action
from env.guandan.utils import Card2Num, Suit2Num, Action2Num, STRAIGHT_FLUSH, BOMB, PASS


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

    def random_action(self):
        return self._env.random_action()

    def reset(self):
        self._env.reset()
        self._env.start()

        return get_obs(self._get_infoset())

    def step(self, action):
        #self._env.players[self._acting_player_position].set_action(action)
        self._env.play(action)
        done = False
        reward = 0.0
        info = {}

        if self._game_over:
            done = True
            reward = self._env.compute_reward()
            obs = None
        else:
            obs = get_obs(self._get_infoset)

        return obs, reward, done, info

    def _get_infoset(self):
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


def _get_one_hot_array(i,n):
    """
    A utility function to obtain one-hot encoding
    """
    one_hot = np.zeros(n, dtype=np.float32)
    one_hot[i - 1] = 1

    return one_hot


def _cards2matrix(cards):
    matrix = np.zeros([15, 4], dtype=np.float32)
    if len(cards) == 0:
        return matrix

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
            matrix[row, column] += 1
    matrix /= 2
    if b_jokers:
        matrix[13, b_jokers-1] = 1
    if r_jokers:
        matrix[14, r_jokers-1] = 1

    return matrix

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
    action_type = np.zeros(3, dtype=np.float32)
    result = dict(numbers=numbers, jokers=jokers, action_type=action_type)

    numbers[:, 4] = rank

    if action.type is None:
        action_type[0] = 1
        return result
    elif action.type == PASS:
        action_type[1] = 1
        return result
    else:
        action_type[2] = 1

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

def _get_rel_pids():
    # pids = [np.zeros(4, dtype=np.float32) for _ in range(4)]
    # for i, pid in enumerate(pids):
    #     pid[i] = 1
    
    # return np.stack(pids)
    return np.eye(4)

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 12 actions. If there is
    no 12 actions, we pad the features with 0.
    """
    action_seq_array = np.zeros((len(action_seq_list), 108))
    for row, cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _action_repr(cards)
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

def _get_last_valid_action_rel_pos(pid, down_pid, teammate_pid, up_pid, last_valid_pid):
    if last_valid_pid == pid:
        return 0
    elif last_valid_pid == up_pid:
        return 1
    elif last_valid_pid == teammate_pid:
        return 2
    elif last_valid_pid == down_pid:
        return 3
    else:
        return -1

def _get_obs(infoset: InfoSet, separate_jokers=True):
    pid = infoset.pid
    down_pid = get_down_pid(infoset.pid)
    teammate_pid = get_teammate_pid(infoset.pid)
    up_pid = get_up_pid(infoset.pid)
    pids = [pid, down_pid, teammate_pid, up_pid]
    others_pids = [down_pid, teammate_pid, up_pid]
    others_hand_cards = [infoset.all_hand_cards[i] for i in others_pids]
    
    # obs
    rank = Card2Num[infoset.rank]
    rank_repr = _get_one_hot_array(rank, 13)
    rank_exp_repr = np.expand_dims(rank_repr, -1)
    # my_handcards_repr = _cards_repr(infoset.player_hand_cards)
    # others_handcards_repr = _cards_repr(sum(others_hand_cards, []))
    # played_cards_repr = _cards_repr(infoset.played_cards[pid])
    # down_played_cards_repr = _cards_repr(infoset.played_cards[down_pid])
    # teammate_played_cards_repr = _cards_repr(infoset.played_cards[teammate_pid])
    # up_played_cards_repr = _cards_repr(infoset.played_cards[up_pid])
    # # NOTE: concatenate or stack? stack is cheaper while concatenate is expensive but more common-seen in literature
    # # # TODO: try recording different types of actions separately
    # numbers2 = np.concatenate([
    #     my_handcards_repr['numbers'],
    #     others_handcards_repr['numbers'],
    #     played_cards_repr['numbers'],
    #     down_played_cards_repr['numbers'],
    #     teammate_played_cards_repr['numbers'],
    #     up_played_cards_repr['numbers'],
    #     rank_exp_repr,
    # ], axis=-1)
    # np.testing.assert_equal(
    #     my_handcards_repr['numbers'] \
    #     + others_handcards_repr['numbers'] \
    #     + down_played_cards_repr['numbers'] \
    #     + teammate_played_cards_repr['numbers'] \
    #     + up_played_cards_repr['numbers'] \
    #     + played_cards_repr['numbers'], 1
    # )
    # jokers2 = np.concatenate([
    #     my_handcards_repr['jokers'],
    #     others_handcards_repr['jokers'],
    #     played_cards_repr['jokers'],
    #     down_played_cards_repr['jokers'],
    #     teammate_played_cards_repr['jokers'],
    #     up_played_cards_repr['jokers'],
    # ], axis=-1)
    cards_reprs = [_cards_repr(cards) for cards in [
        infoset.player_hand_cards,
        sum(others_hand_cards, []),
        *[infoset.played_cards[i] for i in pids]
    ]]
    numbers = np.concatenate([c['numbers'] for c in cards_reprs]+[rank_exp_repr], axis=-1)
    jokers = np.concatenate([c['jokers'] for c in cards_reprs], axis=-1)
    # np.testing.assert_equal(numbers, numbers2)
    # np.testing.assert_equal(jokers, jokers2)
    down_num_cards_left = infoset.all_num_cards_left[down_pid]
    down_num_cards_left_repr = _get_one_hot_array(down_num_cards_left, 27)
    teammate_num_cards_left = infoset.all_num_cards_left[teammate_pid]
    teammate_num_cards_left_repr = _get_one_hot_array(teammate_num_cards_left, 27)
    up_num_cards_left = infoset.all_num_cards_left[up_pid]
    up_num_cards_left_repr = _get_one_hot_array(up_num_cards_left, 27)
    left_cards = np.concatenate([
        down_num_cards_left_repr,
        teammate_num_cards_left_repr,
        up_num_cards_left_repr
    ], axis=-1)
    # left_cards = np.array([
    #     down_num_cards_left,
    #     teammate_num_cards_left,
    #     up_num_cards_left
    # ])
    is_last_teammate_move = infoset.last_valid_pid == teammate_pid
    is_last_teammate_move_repr = is_last_teammate_move * np.ones(1, dtype=np.float32)
    is_first_move = infoset.last_pid == -1
    if is_first_move:
        assert infoset.last_valid_pid == -1
        assert infoset.last_action.type is None
        assert infoset.last_valid_action.type is None

    # history actions
    last_actions = [infoset.all_last_actions[i] for i in pids]
    last_action_reprs = [_action_repr(a, rank) for a in last_actions]
    assert np.all([a1 == a2 for a1, a2 in zip(last_actions, infoset.played_action_seq[-4:])])
    last_action_numbers = np.stack([a['numbers'] for a in last_action_reprs])
    last_action_jokers = np.stack([a['jokers'] for a in last_action_reprs])
    last_action_types = np.stack([a['action_type'] for a in last_action_reprs])
    last_action_rel_pids = _get_rel_pids()
    last_action_filter = np.array([a.type is not None for a in last_actions], dtype=np.bool)
    last_action_first_move = np.array([infoset.all_last_action_first_move[i] for i in pids], dtype=np.bool)
    assert np.all(last_action_filter == (1-last_action_types[:, 0]).astype(np.bool)), (last_action_filter, (1-last_action_types[:, 0]).astype(np.bool))

    # unobservable state: others' cards
    # down_handcards = _cards_repr(others_hand_cards[0])
    # teammate_handcards = _cards_repr(others_hand_cards[1])
    # up_handcards = _cards_repr(others_hand_cards[2])
    # other_numbers = np.concatenate([
    #     down_handcards['numbers'],
    #     teammate_handcards['numbers'],
    #     up_handcards['numbers'],
    #     rank_exp_repr,
    # ], axis=-1)
    # other_jokers = np.concatenate([
    #     down_handcards['jokers'],
    #     teammate_handcards['jokers'],
    #     up_handcards['jokers']
    # ], axis=-1)
    others_handcards = [_cards_repr(c) for c in others_hand_cards]
    others_numbers = np.concatenate([c['numbers'] for c in others_handcards]+[rank_exp_repr], axis=-1)
    others_jokers = np.concatenate([c['jokers'] for c in others_handcards], axis=-1)
    # np.testing.assert_equal(other_numbers, others_numbers)
    # np.testing.assert_equal(other_jokers, others_jokers)
    # TODO: Try more action type
    action_type_mask = np.zeros(3, dtype=np.bool)    # pass, follow, bomb
    follow_mask = np.zeros(15, dtype=np.bool)
    bomb_mask = np.zeros(15, dtype=np.bool)
    for a in infoset.legal_actions:
        if a.type == PASS:
            action_type_mask[0] = 1
        else:
            i = Action2Num[a.rank]
            if a.type == BOMB or a.type == STRAIGHT_FLUSH:
                action_type_mask[2] = 1
                bomb_mask[i] = 1
            else:
                action_type_mask[1] = 1
                follow_mask[i] = 1
    assert action_type_mask[0] != is_first_move, (action_type_mask, is_first_move, [a.cards for a in infoset.legal_actions])
    if action_type_mask[1] == False:
        assert np.all(follow_mask == False), follow_mask
    if action_type_mask[2] == False:
        assert np.all(bomb_mask == False), bomb_mask
    #     print(a.type, a.rank, a.cards, rank)
    # print(action_type)
    # print(follow_mask)
    # print(bomb_mask)
    # assert False
    obs = {
        'pid': pid,
        'numbers': numbers,
        'jokers': jokers,
        'left_cards': left_cards,
        'is_last_teammate_move': is_last_teammate_move_repr,
        'is_first_move': is_first_move,
        'rank': rank_repr,
        'bombs_dealt': infoset.bombs_dealt,
        'last_action_numbers': last_action_numbers,
        'last_action_jokers': last_action_jokers,
        'last_action_types': last_action_types,
        'last_action_rel_pids': last_action_rel_pids,
        'last_action_filter': last_action_filter,
        'last_action_first_move': last_action_first_move,
        'action_type_mask': action_type_mask,
        'follow_mask': follow_mask,
        'bomb_mask': bomb_mask,
        'others_numbers': others_numbers,
        'others_jokers': others_jokers,
    }
    return obs

