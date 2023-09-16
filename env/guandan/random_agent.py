import random

class RandomAgent():

  def __init__(self):
    self.name = 'Random'

  def __call__(self, infoset):
    # print('legal actions', infoset.legal_actions)
    # print('hand cards', infoset.player_hand_cards)
    return random.randint(0, len(infoset.legal_actions.action_list)-1)