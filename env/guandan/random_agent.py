import random

class RandomAgent():

    def __init__(self):
        self.name = 'Random'

    def act(self, infoset):
        #print(action_list) 
        #print("可选动作范围为：0至{}".format(len(action_list)-1))
        return random.randint(0, len(infoset.legal_actions.action_list)-1)