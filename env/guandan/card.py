# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.10 (default, May 19 2021, 13:12:57) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: card.py


class Card(object):
  ALL_SUITS = ('S', 'H', 'C', 'D')
  MAPPING = {'S':'黑桃',  'H':'红桃',  'C':'梅花',  'D':'方片'}
  ALL_RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'W')
  OUTPUT_MODEL = 0
  RANK2DIGIT = {
    'A': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'T': 10,
    'J': 11,
    'Q': 12,
    'K': 13,
  }

  def __init__(self, suit, rank, digital=None):
    self.suit = suit
    self.rank = rank
    self.digital = digital

  def __eq__(self, other):
    if isinstance(other, str):
      return self.suit == other[0] and self.rank == other[1]
    else:
      return self.suit == other.suit and self.rank == other.rank

  def __str__(self):
    if Card.OUTPUT_MODEL:
      return Card.MAPPING[self.suit] + self.rank
    else:
      return self.suit + self.rank

  def __repr__(self):
    return str(self)
# okay decompiling card.pyc
