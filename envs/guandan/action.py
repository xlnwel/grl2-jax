# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.10 (default, May 19 2021, 13:12:57) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: action.py
import copy
import numpy as np

from .utils import *


def get_action_type(action, action_type=None, one_hot=True):
  if one_hot:
    if action_type is None:
      action_type = np.zeros(NUM_ACTION_TYPES, dtype=np.float32)
    assert action_type.size == 3, action_type
    action_type[ActionType2Num[action.type]] = 1
  else:
    action_type = ActionType2Num[action.type]
  return action_type


def get_action_card(action, action_card=None, one_hot=True):
  if one_hot:
    if action_card is None:
      action_card = np.zeros(15, dtype=np.float32)
    if action.type != PASS:
      action_card[Rank2Num[action.rank]] = 1
  else:
    if action.type == PASS:
      action_card = -1
    else:
      action_card = Rank2Num[action.rank]
  return action_card


def get_card_type(action):
  card_type = np.zeros(9, dtype=np.float32)
  if action.type is not None and action.type != PASS:
    card_type[CardType2Num[action.type]] = 1
  return card_type


class Action(object):

  def __init__(self, _type=None, _rank=None, _cards=None):
    self._type = _type
    self._rank = _rank
    self._cards = _cards

  def update(self, _type, _rank, _cards):
    self._type = _type
    self._rank = _rank
    self._cards = _cards

  @property
  def type(self):
    return self._type

  @property
  def rank(self):
    return self._rank

  @property
  def cards(self):
    return self._cards

  def reset(self):
    self.update(None, None, None)

  def __str__(self):
    return str([self._type, self._rank, self._cards])

  def __repr__(self) -> str:
    return self.__str__()

  def __eq__(self, other) -> bool:
    return self._type == other.type and self._rank == other.rank and self._cards == other.cards

  def copy(self):
    return Action(self.type, self.rank, copy.copy(self.cards))

class ActionList(object):

  def __init__(self):
    self.action_list = None

  def update(self, action_list):
    self.action_list = action_list

  def __getitem__(self, item):
    return Action(*self.action_list[item])

  def __str__(self):
    return str(self.action_list)

  @property
  def valid_range(self):
    if len(self.action_list) == 1:
      return range(0, 1)
    else:
      return range(len(self.action_list))
# okay decompiling action.pyc

