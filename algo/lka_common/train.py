import os
import collections
import jax

from core.ckpt.pickle import restore
from core.elements.builder import ElementsBuilder
from core.typing import get_basic_model_name, dict2AttrDict, tree_slice
from tools.display import print_dict_info
from tools.plot import prepare_data_for_plotting, lineplot_dataframe
from tools.store import StateStore
from tools.utils import modify_config, prefix_name
from replay.dual import PRIMAL_REPLAY, DualReplay
from algo.ma_common.train import *


@timeit
def build_dynamics(config, model_config, env_stats):
    root_dir = config.root_dir
    model_name = get_basic_model_name(config.model_name)
    seed = config.seed
    new_model_name = '/'.join([model_name, 'dynamics'])
    model_config = modify_config(
        model_config, 
        max_layer=1, 
        aid=0,
        name=config.algorithm, 
        info=config.info,
        model_info=config.model_info,
        n_runners=config.env.n_runners, 
        n_envs=config.env.n_envs, 
        root_dir=root_dir, 
        model_name=new_model_name, 
        overwrite_existed_only=True, 
        seed=seed+1000
    )

    builder = ElementsBuilder(
        model_config, 
        env_stats, 
        to_save_code=False, 
        max_steps=config.routine.MAX_STEPS
    )
    elements = builder.build_agent_from_scratch(config=model_config)
    dynamics = elements.agent

    return dynamics


@timeit
def dynamics_optimize(dynamics):
    dynamics.train_record()


@timeit
def dynamics_run(agent, dynamics, routine_config, dynamics_routine_config, 
        rng, lka_aids, rollout_fn, name='dynamics'):
    if dynamics_routine_config.model_warm_up and \
        agent.get_env_step() < dynamics_routine_config.model_warm_up_steps:
        return

    def constructor():
        return agent.build_memory()
    
    def enter_set(states):
        states = agent.set_memory(states)
        # we put the data collected from the dynamics into the secondary replay
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(routine_config.lookahead_replay)
        return states
    
    def exit_set(states):
        states = agent.set_memory(states)
        if isinstance(agent.buffer, DualReplay):
            agent.buffer.set_default_replay(PRIMAL_REPLAY)
        return states

    # train lookahead agent
    with StateStore(name, constructor, enter_set, exit_set):
        rollout_fn(agent, dynamics, routine_config, rng, lka_aids)
    return True


@timeit
def lka_optimize(agent):
    agent.lookahead_train()


@timeit
def real_lka_train(agent, runner, routine_config, n_runs, run_fn, opt_fn, **kwargs):
    for _ in range(n_runs):
        run_fn(agent, runner, routine_config, **kwargs)
        opt_fn(agent)


@timeit
def lka_train(agent, dynamics, routine_config, dynamics_routine_config, 
        n_runs, rng, lka_aids, run_fn=dynamics_run, opt_fn=lka_optimize):
    assert n_runs >= 0, n_runs
    for _ in range(n_runs):
        succ = run_fn(agent, dynamics, routine_config, dynamics_routine_config, 
            rng, lka_aids)
        if succ:
            opt_fn(agent)


@timeit
def log_dynamics_errors(errors, outdir, env_step):
    if errors:
        data = collections.defaultdict(dict)
        for k1, errs in errors.items():
            for k2, v in errs.items():
                data[k2][k1] = v
        y = 'abs error'
        outdir = '/'.join([outdir, 'errors'])
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        for k, v in data.items():
            filename = f'{k}-{env_step}'
            filepath = '/'.join([outdir, filename])
            data[k] = prepare_data_for_plotting(
                v, y=y, smooth_radius=0, filepath=filepath)
            # lineplot_dataframe(data[k], filename, y=y, outdir=outdir)


def load_eval_data(filename, filedir='/System/Volumes/Data/mnt/公共区/cxw/data', n=None):
    data = restore(filedir=filedir, filename=filename)
    if n is not None:
        maxlen = data['obs'].shape[0]
        indices = np.random.randint(0, maxlen, n)
        data = tree_slice(data, indices)
    print_dict_info(data)
    return data


@timeit
def eval_policy_distances(agent, data, name=None, n=None, eval_lka=True):
    if not data:
        return
    data = data.copy()
    data.state = dict2AttrDict(data.state, shallow=True)
    if n:
        bs = data.obs.shape[0]
        if n < bs:
            indices = np.random.permutation(bs)[:n]
            data = data.slice(indices)
    stats = agent.model.compute_policy_distances(data, eval_lka=eval_lka)
    stats = prefix_name(stats, name)
    agent.store(**stats)


@timeit
def eval_ego_and_lka(agent, runner, routine_config):
    agent.model.swap_params()
    agent.model.swap_lka_params()
    mu_score, _, _ = evaluate(agent, runner, routine_config)
    lka_score, _, _ = evaluate(agent, runner, routine_config, None)
    agent.model.swap_params()
    agent.model.swap_lka_params()
    agent.model.check_current_params()
    agent.model.check_current_lka_params()
    pi_score, _, _ = evaluate(agent, runner, routine_config)

    mu_score = np.array(mu_score)
    lka_score = np.array(lka_score)
    pi_score = np.array(pi_score)
    lka_mu_score_diff = lka_score - mu_score
    pi_mu_score_diff = pi_score - mu_score
    pi_lka_score_diff = pi_score - lka_score
    agent.store(
        mu_score=mu_score, 
        pi_score=pi_score, 
        lka_score=lka_score, 
        lka_mu_score_diff=lka_mu_score_diff, 
        pi_mu_score_diff=pi_mu_score_diff, 
        pi_lka_score_diff=pi_lka_score_diff, 
    )


@timeit
def eval_and_log(agent, dynamics, runner, routine_config, 
                 train_data, eval_data, errors={}, n=500, eval_lka=True):
    eval_policy_distances(agent, eval_data, name='eval', n=n, eval_lka=eval_lka)
    if train_data:
        seqlen = train_data.obs.shape[1]
        train_data = dict2AttrDict({k: train_data[k] for k in eval_data})
        train_data = jax.tree_util.tree_map(
            lambda x: x[:, :seqlen].reshape(-1, 1, *x.shape[2:]), train_data)
        eval_policy_distances(agent, train_data, n=n, eval_lka=eval_lka)
    if runner is not None:
        eval_ego_and_lka(agent, runner, routine_config)
    save(agent, dynamics)
    log(agent, dynamics, errors)
