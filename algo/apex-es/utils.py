from collections import namedtuple, defaultdict
import numpy as np

from utility.display import pwc

Weights = namedtuple('Weights', 'tag weights eval_times')

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

def get_best_score(weight_repo):
    sorted_scores = sorted(weight_repo.keys(), reverse=True)
    best_score = sorted_scores[0]
    return best_score

def remove_worst_weights(weight_repo):
    del weight_repo[min(weight_repo)]

def remove_oldest_weights(weight_repo):
    del weight_repo[list(weight_repo)[0]]

def evolve_weights(weight_repo, min_evolv_models=2, max_evolv_models=10, weighted_average=False, slack=10):
    n = np.random.randint(min_evolv_models, len(weight_repo)+1)
    w = np.asarray([score for score in weight_repo])
    min_w = np.min(w)
    w -= min_w - slack # avoid zero probability
    p = w / np.sum(w)
    scores = list(weight_repo)
    selected_keys = np.random.choice(scores, size=n, replace=False, p=p)
    selected_weights = [weight_repo[k].weights for k in selected_keys]

    if weighted_average:
        w = selected_keys - np.min(selected_keys) + slack
        if isinstance(selected_weights[0], dict):
            weights = defaultdict(list)
            net_names = selected_weights[0].keys()
            selected_weights = dict((name, [ws[name] for ws in selected_weights]) for name in net_names)
            weights = dict((name, [np.average(ws, axis=0, weights=w) for ws in zip(*selected_weights[name])]) for name in net_names)
        else:
            weights = [np.average(ws, axis=0, weights=w) for ws in zip(selected_weights)]
    else:
        if isinstance(selected_weights[0], dict):
            net_names = selected_weights[0].keys()
            weights = dict([(name, np.mean([w[name] for w in selected_weights], axis=0)) for name in net_names])
        else:
            weights = np.mean(selected_weights, axis=0)
    return weights, n

def analyze_repo(weight_repo):
    n_learned = sum([w.tag == Tag.LEARNED for w in weight_repo.values()])
    n_evolved = sum([w.tag == Tag.EVOLVED for w in weight_repo.values()])
    n = len(weight_repo)
    return dict(frac_learned=n_learned / n, frac_evolved=n_evolved / n)

def store_weights(weight_repo, score, tag, weights, eval_times, store_cap, fifo=False):
    weight_repo[score] = Weights(tag, weights, eval_times)
    while len(weight_repo) > store_cap:
        if fifo:
            remove_oldest_weights(weight_repo)
        else:
            remove_worst_weights(weight_repo)

def print_repo(repo, name, info=[]):
    for msg, color in info:
        pwc(*msg, color=color)
    repo = [(score, weights.tag, weights.eval_times) for score, weights in repo.items()]
    repo = sorted(repo, key=lambda x: x[0], reverse=True)
    pwc(f"{name}: current stored models", 
        f"{[f'({x[0]:.3g}, {x[1]}, {x[2]})' for x in repo]}", 
        color='magenta')
    return []