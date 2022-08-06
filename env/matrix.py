from .matrix_env.ipd import IteratedPrisonersDilemma
from .matrix_env.imp import IteratedMatchingPennies

env_map = {
    'ipd': IteratedPrisonersDilemma,
    'imp': IteratedMatchingPennies
}
