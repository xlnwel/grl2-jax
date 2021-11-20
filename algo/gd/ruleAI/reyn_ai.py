import random
from algo.gd.ruleAI.reyn_ai_utils import check_message


class ReynAIAgent():

    def __init__(self):
        self.name = 'reyn_ai'

    def __call__(self, infoset):
        #print(action_list)
        #print("可选动作范围为：0至{}".format(len(action_list)-1))
        self.AI_choice = check_message(infoset, infoset.pid)
        # 由于没有考虑进贡，故而随机，否则bug
        if self.AI_choice == None:
            return random.randint(0, len(infoset.legal_actions.action_list) - 1)
        # print("AI选择的出牌编号为:{}".format(self.AI_choice))
        return self.AI_choice
