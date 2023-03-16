from algo.ma_common.elements.trainer import *
from algo.lka_common.elements.model import LOOKAHEAD


def pop_lookahead(policies):
    policies = [p.copy() for p in policies]
    lka = [p.pop(LOOKAHEAD) for p in policies]
    return policies, lka
