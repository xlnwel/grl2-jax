import functools

from core.tf_config import *
from utility.utils import Every
from utility.run import evaluate
from env.gym_env import create_env
from replay.func import create_replay
from replay.data_pipline import Dataset, process_with_env
from run import pkg
from algo.sac.train import get_data_format, train


def main(env_config, model_config, agent_config, replay_config):
    silence_tf_logs()
    configure_gpu()

    env = create_env(env_config)
    assert env.n_envs == 1, \
        f'n_envs({env.n_envs}) > 1 is not supported here as it messes with n-step'
    eval_env_config = env_config.copy()
    eval_env_config['n_envs'] = 1
    eval_env = create_env(eval_env_config)

    replay = create_replay(replay_config)

    data_format = get_data_format(env, replay_config)
    print(data_format)
    process = functools.partial(process_with_env, env=env)
    dataset = Dataset(replay, data_format, process_fn=process)

    create_model, Agent = pkg.import_agent(agent_config)
    models = create_model(
        model_config, 
        action_dim=env.action_dim, 
        is_action_discrete=env.is_action_discrete)
    agent = Agent(name='sac', 
                config=agent_config, 
                models=models, 
                dataset=dataset, 
                env=env)
    
    agent.save_config(dict(
        env=env_config,
        model=model_config,
        agent=agent_config,
        replay=replay_config
    ))
    
    train(agent, env, eval_env, replay)

    # This training process is used for Mujoco tasks, following the same process as OpenAI's spinningup
    # obs = env.reset()
    # epslen = 0
    # to_log = Every(agent.LOG_INTERVAL, start=2*agent.LOG_INTERVAL)
    # for t in range(int(agent.MAX_STEPS)):
    #     if t > 1e4:
    #         action = agent(obs)
    #     else:
    #         action = env.random_action()

    #     nth_obs, reward, done, _ = env.step(action)
    #     epslen += 1
    #     replay.add(obs=obs, action=action, reward=reward, discount=1-done, nth_obs=nth_obs)
    #     obs = nth_obs

    #     if done or epslen == env.max_episode_steps:
    #         agent.store(score=env.score(), epslen=env.epslen())
    #         obs = env.reset()
    #         epslen = 0

    #     if replay.good_to_learn() and t % 50 == 0:
    #         for _ in range(50):
    #             agent.learn_log(t)
    #     if to_log(t):
    #         eval_score, eval_epslen, _ = evaluate(eval_env, agent)

    #         agent.store(eval_score=eval_score, eval_epslen=eval_epslen)
    #         agent.log(step=t)
    #         agent.save(steps=t)
