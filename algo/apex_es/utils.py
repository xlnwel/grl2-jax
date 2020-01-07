from collections import namedtuple
import numpy as np

from utility.display import pwc

Weights = namedtuple('Weights', 'tag weights')

class Status:
    ACCEPTED='Accepted'
    TOLERATED='Tolerated'
    REJECTED='Rejected'
class Mode:
    LEARNING='Learning'
    EVOLUTION='Evolution'
    REEVALUATION='Reevaluation'
class Tag:
    LEARNED='Learned'
    EVOLVED='Evolved'
    REEVALUATED='Reevaluated'


class BookKeeping:
    def __init__(self, name):
        self.name = name
        self.reset()
        
    def reset(self):
        self.n_evolved_acceptance = 0
        self.n_evolved_refuse = 0
        self.n_learned_acceptance = 0
        self.n_learned_refuse = 0
    
    def add(self, tag, status):
        if tag == Tag.EVOLVED and status == Status.ACCEPTED:
            self.n_evolved_acceptance += 1
        elif tag == Tag.EVOLVED:
            self.n_evolved_refuse += 1
        elif tag == Tag.LEARNED and status == Status.ACCEPTED:
            self.n_learned_acceptance += 1
        elif tag == Tag.LEARNED:
            self.n_learned_refuse += 1
        else:
            raise ValueError(f'Unknown tag({tag}) and status({status})')
    
    def stats(self):
        evolved_accept_rate = self.n_evolved_acceptance and \
            self.n_evolved_acceptance / (self.n_evolved_acceptance + self.n_evolved_refuse)
        learned_accept_rate = self.n_learned_acceptance and \
            self.n_learned_acceptance / (self.n_learned_acceptance + self.n_learned_refuse)
        
        return {
            f'{self.name}_evolved_accept_rate': evolved_accept_rate,
            f'{self.name}_learned_accept_rate': learned_accept_rate,
        }

def get_best_score(store_map):
    sorted_scores = sorted(store_map.keys(), reverse=True)
    best_score = sorted_scores[0]
    return best_score

def remove_worst_weights(store_map):
    del store_map[min(store_map)]

def evolve_weights(store_map, min_evolv_models=2):
    n = np.random.randint(min_evolv_models, len(store_map)+1)
    w = np.asarray([score for score in store_map])
    min_w = np.min(w)
    w -= min_w - 10 # avoid zero probability
    p = w / np.sum(w)
    score_keys = list(store_map)
    selected_keys = np.random.choice(score_keys, size=n, replace=False, p=p)
    selected_weights = [store_map[k].weights for k in selected_keys]

    if isinstance(selected_weights[0], dict):
        net_names = selected_weights[0].keys()
        weights = dict([(name, np.mean([w[name] for w in selected_weights], axis=0)) for name in net_names])
    else:
        weights = np.mean(selected_weights, axis=0)

    return weights, n

def analyze_store(store_map):
    n_learned = len([w.tag == Tag.LEARNED for w in store_map.values()])
    n_evolved = len([w.tag == Tag.EVOLVED for w in store_map.values()])
    n = len(store_map)
    return dict(frac_learned=n_learned / n, frac_evolved=n_evolved / n)

def print_store(store, name):
    store = [(score, weights.tag) for score, weights in store.items()]
    store = sorted(store, key=lambda x: x[0], reverse=True)
    pwc(f"{name}: current stored models", 
        f"{[f'({x[0]:.3g}, {x[1]})' for x in store]}", 
        color='magenta')
