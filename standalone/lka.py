import copy

from standalone.happo import *


def build_lka_name(args):
    if args.name is None:
        name = __file__.split('.')[0]
    else:
        name = args.name
    name = '-'.join([
        name, 
        f'horizon={args.horizon}', 
        f'epochs={args.epochs}', 
        f'lr={args.lr}', 
        f'state_size={args.state_size}', 
        f'action_dims={args.action_dims}', 
        f'n_lka_steps={args.n_lka_steps}', 
    ])

    return name


""" Training """
def look_ahead(
    rng, 
    n, 
    x_logits, 
    reward, 
    transition, 
    gamma,
    opts, 
    states, 
    horizon, 
    epochs, 
):
    # NOTE: x does not converge
    for _ in range(n):
        mu = prob_from_logits(x_logits)
        x_logits, states = happo_train(
            rng=rng, 
            pi_logits=x_logits, 
            mu=mu, 
            reward=reward, 
            transition=transition, 
            gamma=gamma, 
            opts=opts, 
            states=states, 
            horizon=horizon, 
            epochs=epochs
        )

    return x_logits


def lka_train(
    args, 
    pi_logits, 
    reward, 
    rho, 
    transition, 
    opts, 
    states, 
    points=100, 
):
    interval = args.iteration // points
    print('Interval:', interval)
    rng = random.PRNGKey(args.seed)
    # print(f'Initial PRNG for seed={seed}:', rng)
    mu = prob_from_logits(pi_logits)
    steps = [0]
    scores = [evaluate(mu, reward, rho, transition)]
    print(f'{args.name} Iteration {0}:\t{scores[0]}')
    
    x_logits = copy.deepcopy(pi_logits)
    for i in range(1, args.iteration+1):
        x_logits = look_ahead(
            rng, 
            args.n_lka_steps, 
            x_logits=x_logits, 
            reward=reward, 
            transition=transition, 
            gamma=args.gamma,
            opts=opts, 
            states=states, 
            horizon=args.horizon, 
            epochs=args.epochs
        )
        mu = prob_from_logits(x_logits)
        pi_logits, states = happo_train(
            rng=rng, 
            pi_logits=pi_logits, 
            mu=mu, 
            reward=reward, 
            transition=transition, 
            gamma=args.gamma, 
            opts=opts, 
            states=states, 
            horizon=args.horizon, 
            epochs=args.epochs
        )

        pi = prob_from_logits(pi_logits)
        if i % interval == 0:
            score = evaluate(pi, reward, rho, transition)
            steps.append(i)
            scores.append(score)
            print(f'{args.name} Iteration {i}:\t{score}')
    steps = np.array(steps)
    scores = np.array(scores)

    return steps, scores


# reimplementation of happo.main since
# ray does not acccept func.partial function as a remote
def lka_main(
    args, 
    build_initial_policy=build_initial_policy, 
    build_optimizers=build_optimizers, 
    build_dynamics=build_dynamics, 
    train=lka_train
):
    horizon = args.horizon
    epochs = args.epochs
    state_size = args.state_size
    action_dims = args.action_dims
    n_agents = len(action_dims)
    print('Horizon:', horizon)
    print('Epochs:', epochs)
    print('#Agents:', n_agents)
    print('State size:', state_size)
    print('Action dimensions:', action_dims)

    data = {}
    points = 100
    max_seed = args.seed
    if max_seed == 0:
        steps, scores = build_and_train(
            args, 
            points=points, 
            build_initial_policy=build_initial_policy, 
            build_optimizers=build_optimizers, 
            build_dynamics=build_dynamics, 
            train=train
        )
    else:
        processes = []
        ray_bt = ray.remote(build_and_train)
        for seed in range(max_seed):
            args.seed = seed
            p = ray_bt.remote(
                args, 
                points=points, 
                build_initial_policy=build_initial_policy, 
                build_optimizers=build_optimizers, 
                build_dynamics=build_dynamics, 
                train=train
            )
            processes.append(p)

        results = ray.get(processes)
        steps, scores = list(zip(*results))
        steps = np.concatenate(steps)
        scores = np.concatenate(scores)
    data['steps'] = steps
    data['score'] = scores
    data['legend'] = [args.name] * data['steps'].shape[0]

    return data


if __name__ == '__main__':
    ray.init()

    args = parse_args()
    args.algo = 'happo'
    args.name = 'happo'
    args.name = build_name(args)
    ray_happo_main = ray.remote(main)
    happo_p = ray_happo_main.remote(args)

    args.algo = 'lka'
    args.name = 'lka'
    args.name = build_lka_name(args)
    ray_lka_main = ray.remote(lka_main)
    lka_p = ray_lka_main.remote(args)
    happo_data, lka_data = ray.get([happo_p, lka_p])

    data = {}
    for k in happo_data.keys():
        data[k] = np.concatenate([happo_data[k], lka_data[k]])

    dir_path = args.dir_path
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    data = process_data(data, dir_path, args.name)
    plot(data, dir_path, args.name)
    
    ray.shutdown()
