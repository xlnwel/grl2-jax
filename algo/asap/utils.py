from collections import namedtuple, defaultdict
import numpy as np

from utility.display import pwc

Weights = namedtuple('Weights', 'tag weights eval_times')
Records = namedtuple('Records', 'mode tag weights eval_times')


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

def ucb(values, ns, c):
    """ 
    Compute the upper confidence bound
    Args:
        values: an array of values
        ns: an array that counts the number of visit times for each value
    """
    N = np.sum(ns)
    return values + c * (np.log(N)/ns)**.5
    
def lcb(values, ns, c):
    """ 
    Compute the lower confidence bound
    Args:
        values: an array of values
        ns: an array that counts the number of visit times for each value
    """
    N = np.sum(ns)
    return values - c * (np.log(N)/ns)**.5

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    scores = (scores - min_score) / (max_score - min_score) * 2 - 1
    return scores

def fitness_from_repo(weight_repo, method, c=1):
    scores = np.array(list(weight_repo))
    norm_scores = normalize_scores(scores)
    if method == 'norm':
        return norm_scores
    else:
        eval_times = np.array([w.eval_times for w in weight_repo.values()])
        if method == 'ucb':
            return ucb(norm_scores, eval_times, c)
        elif method == 'lcb':
            return lcb(norm_scores, eval_times, c)
        else:
            raise NotImplementedError(f'Unknown method({method})')

def get_best_score(weight_repo):
    sorted_scores = sorted(weight_repo.keys(), reverse=True)
    best_score = sorted_scores[0]
    return best_score

def remove_worst_weights(weight_repo, fitness_method, c):
    fitness = fitness_from_repo(weight_repo, fitness_method, c=c)
    scores = list(weight_repo)
    min_score = scores[np.argmin(fitness)]
    weight = weight_repo[min_score]
    del weight_repo[min_score]
    return min_score, weight

def remove_oldest_weights(weight_repo):
    score = list(weight_repo)[0]
    weight = weight_repo[score]
    del weight_repo[score]
    return score, weight

def evolve_weights(weight_repo, min_evolv_models=2, max_evolv_models=5, 
                    wa_selection=False, wa_evolution=False, 
                    fitness_method='lcb', c=1):
    n = np.random.randint(min_evolv_models, min(len(weight_repo), max_evolv_models) + 1)
    scores = np.array(list(weight_repo))
    if wa_selection or wa_evolution:
        fitness = fitness_from_repo(weight_repo, fitness_method, c=c)
        w = np.exp(fitness)
        w = w / np.sum(w)
    else:
        w = None
    idxes = np.random.choice(np.arange(len(scores)), size=n, replace=False, p=w)
    selected_weights = [weight_repo[k].weights for k in scores[idxes]]
    if wa_evolution:
        w = w[idxes]
    else:
        w = None
    weights = average_weights(selected_weights, w)

    return weights, n

def average_weights(model_weights, avg_weights=None):
    if isinstance(model_weights[0], dict):
        net_names = model_weights[0].keys()
        model_weights = dict((name, [ws[name] for ws in model_weights]) for name in net_names)
        weights = dict((name, [np.average(ws, axis=0, weights=avg_weights) 
                    for ws in zip(*model_weights[name])]) for name in net_names)
        # weights = dict((name, np.average([ws[name] for ws in model_weights], axis=0, weights=avg_weights)) for name in net_names)
    else:
        weights = [np.average(ws, axis=0, weights=avg_weights) for ws in zip(model_weights)]
    
    return weights

def analyze_repo(weight_repo):
    n_learned = sum([w.tag == Tag.LEARNED for w in weight_repo.values()])
    n_evolved = sum([w.tag == Tag.EVOLVED for w in weight_repo.values()])
    n = len(weight_repo)
    return dict(
        frac_learned=0 if n == 0 else n_learned / n, 
        frac_evolved=0 if n == 0 else n_evolved / n, 
        repo_score_max=max(weight_repo),
        repo_score_min=min(weight_repo)
    )

def store_weights(weight_repo, mode, score, tag, weights, eval_times, store_cap, 
                fifo=False, fitness_method='lcb', c=1):
    """ store weights to repo, if there is any entry pop out, return the it """
    if mode == Mode.REEVALUATION or len(weight_repo) == 0 or score > min(weight_repo):
        weight_repo[score] = Weights(tag, weights, eval_times)
    score = None
    while len(weight_repo) > store_cap:
        if fifo:
            score, weights = remove_oldest_weights(weight_repo)
        else:
            score, weights = remove_worst_weights(weight_repo, fitness_method, c)

    if score:
        return score, weights
    else:
        return None, None

def print_repo(repo, name, c=1, info=[]):
    for msg, color in info:
        pwc(*msg, color=color)
    scores = list(repo)
    norm_scores = normalize_scores(scores)
    tags, eval_times = list(zip(*[(tag, eval_times) for tag, _, eval_times in repo.values()]))
    eval_times = np.array(eval_times)
    N = np.sum(eval_times)
    interval = c * (np.log(N)/eval_times)**.5
    ucb = norm_scores + interval
    lcb = norm_scores - interval

    repo = [(f'{f"score({s:.3g})":<15s}', f'{t:<10s}', f'{f"eval times({et})":<15s}',
            f'{f"norm score({ns:.2f})":<20s}', f'{f"interval({i:.2f})":<15s}', 
            f'{f"ucb({u:.2f})":<15s}', f'{f"lcb({l:.2f})":<15s}')
             for s, t, et, ns, i, u, l in zip(scores, tags, eval_times, norm_scores, interval, ucb, lcb)]
    idxes = np.argsort(lcb)
    repo = [repo[i] for i in idxes[::-1]]
    pwc(f"{name}: current stored models(score, normalized score, tag, evaluated times, interval)", color='blue')
    pwc(*[''.join(v) for v in repo], color='magenta')
    return []
