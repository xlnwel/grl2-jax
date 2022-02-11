import numpy as np
import tensorflow as tf

from utility.timer import timeit


def get_data(n_trajs, min_start, max_start, random_action=False, noise=0):
    states = []
    actions = []
    next_states = []
    for _ in range(n_trajs):
        trajlen = np.random.randint(50, 100)
        s = []
        a = []
        start = np.random.uniform(min_start, max_start)
        s.append(start)
        for _ in range(trajlen):
            if random_action:
                a.append(np.random.uniform(-1, 1) * s[-1])
            else:
                a.append(-.1 * np.log(np.abs(s[-1])) - .1 * s[-1])
            s.append(s[-1] + a[-1])

        s_next = s[1:]
        s = s[:-1]
        transform = lambda x: np.stack(x).reshape(-1, 1).astype(np.float32)
        s = transform(s)
        a = transform(a)
        s_next = transform(s_next)
        states.append(s)
        actions.append(a)
        next_states.append(s_next)
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    next_states = np.concatenate(next_states)

    return states, actions, next_states

def pred(s, a, w_layers, b_layers):
    assert len(w_layers) == len(b_layers), (w_layers, b_layers)
    x = s
    for w_layer, b_layer in zip(w_layers, b_layers):
        w = w_layer(a)
        b = b_layer(a)
        w = tf.reshape(w, (-1, 1, 1))
        x = tf.einsum('bh,bho->bo', x, w) + b
    return x

def train(n_trajs, min_start, max_start, n_epochs, mb_size, n_layers, noise=0, lr=0.01):
    s, a, s_next = get_data(n_trajs, min_start, max_start, noise=noise)
    w_layers = []
    b_layers = []
    for _ in range(n_layers):
        w_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='sigmoid'), 
            tf.keras.layers.Dense(1)
        ])
        # b_var = tf.Variable(tf.ones(1), name='b')
        # b_layer = lambda x: b_var
        b_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='sigmoid'), 
            tf.keras.layers.Dense(1)
        ])
        w_layers.append(w_layer)
        b_layers.append(b_layer)
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def optimize(s, a, s_next):
        with tf.GradientTape() as t:
            x = pred(s, a, w_layers, b_layers)
            loss = tf.reduce_mean((s_next - x)**2)
            # if isinstance(b, tf.Variable):
            #     variables = w_layer.variables + [b_var]
            # else:
            variables = sum([l.variables for l in (w_layers + b_layers)], [])
#             l2 = tf.reduce_sum([l2_w * tf.nn.l2_loss(var) for var in variables])
        grads = t.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, .5)
        opt.apply_gradients(zip(grads, variables))
        return variables, grads, loss

    idx = np.arange(len(s))
    for i in range(n_epochs):
        np.random.shuffle(idx)
        s = s[idx]
        a = a[idx]
        s_next = s_next[idx]
        for start in range(0, len(s), mb_size):
            variables, grads, loss = optimize(
                s[start:start+mb_size], 
                a[start:start+mb_size], 
                s_next[start:start+mb_size]
            )
#         print([np.squeeze(v.numpy()) for v in variables], [np.squeeze(v.numpy()) for v in grads], sep='\n')
    return w_layers, b_layers, variables, grads, loss

def pred_mlp(s, a, layers):
    x = tf.concat([s, a], 1)
    x = layers(x)
    return x

def train_mlp(n_trajs, min_start, max_start, n_epochs, mb_size, n_layers, noise=0, lr=0.01):
    s, a, s_next = get_data(n_trajs, min_start, max_start, noise=noise)
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1 if i == n_layers-1 else 10) 
        for i in range(n_layers)]
    )
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def optimize(s, a, s_next):
        with tf.GradientTape() as t:
            x = pred_mlp(s, a, model)
            loss = tf.reduce_mean((s_next - x)**2)
            variables = sum([l.variables for l in model.layers], [])
#             l2 = tf.reduce_sum([l2_w * tf.nn.l2_loss(var) for var in variables])
        grads = t.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, .5)
        opt.apply_gradients(zip(grads, variables))
        return variables, grads, loss

    idx = np.arange(len(s))
    for i in range(n_epochs):
        np.random.shuffle(idx)
        s = s[idx]
        a = a[idx]
        s_next = s_next[idx]
        for start in range(0, len(s), mb_size):
            variables, grads, loss = optimize(
                s[start:start+mb_size], 
                a[start:start+mb_size], 
                s_next[start:start+mb_size]
            )
#         print([np.squeeze(v.numpy()) for v in variables], [np.squeeze(v.numpy()) for v in grads], sep='\n')
    return model, variables, grads, loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hn', action='store_true')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    is_hn = args.hn
    print('is_hn', is_hn)
    for n_trajs in np.logspace(3, 5, 3).astype(np.int32):
        min_start = 10
        max_start = 100
        n_epochs = 10
        mb_size = 100

        for n_layers in range(1, 4):
            for noise in [0]:
                for lr in np.logspace(-3, -1, 3):
                    vars, losses, ws, bs, random_errors, deter_errors = [], [], [], [], [], []
                    for _ in range(1):
                        if is_hn:
                            w_layers, b_layers, variables, grads, loss = timeit(train,
                                n_trajs=n_trajs, 
                                min_start=min_start, 
                                max_start=max_start, 
                                n_epochs=n_epochs, 
                                mb_size=mb_size, 
                                n_layers=n_layers,
                                noise=noise,
                                lr=lr,
                            )
                            w_layers = np.squeeze(w_layers.numpy())
                            b_layers = np.squeeze(b_layers.numpy())
                        else:
                            model, variables, grads, loss = timeit(train_mlp,
                                n_trajs=n_trajs, 
                                min_start=min_start, 
                                max_start=max_start, 
                                n_epochs=n_epochs, 
                                mb_size=mb_size, 
                                n_layers=n_layers,
                                noise=noise,
                                lr=lr,
                            )
                        #         if i % 10000 == 0:
                        variables = [tf.squeeze(v) for v in variables]
                        loss = loss.numpy()
                        
                        for random_action in [True, False]:
                            s, a, s_next = get_data(
                                n_trajs, min_start, max_start, 
                                random_action=random_action
                            )
                            if is_hn:
                                s_next_pred = pred(s, a, w_layers, b_layers)
                            else:
                                s_next_pred = pred_mlp(s, a, model)
                            error = np.squeeze(np.abs(s_next - s_next_pred))
                            if random_action:
                                random_errors.append(error)
                            else:
                                deter_errors.append(error)
                        vars.append(variables)
                        losses.append(loss)
                        if is_hn:
                            ws.append(w_layers)
                            bs.append(b_layers)

                    random_errors = np.stack(random_errors)
                    deter_errors = np.stack(deter_errors)
                    print('n_trajs', n_trajs)
                    print('noise', noise)
                    print('n_layers', n_layers)
                    print('lr', lr)
                    print('variables', *vars, sep='\n\t')
                    print('losses', losses)
                    if is_hn:
                        ws = np.stack(ws)
                        bs = np.stack(bs)
                        if len(ws.shape) == 2:
                            print('w', ws[:, :5], ws[:, -5:], sep='\n\t')
                        else:
                            print('w', ws[:5], ws[-5:], sep='\n\t')
                        if len(bs.shape) == 2:
                            print('b', bs[:, :5], bs[:, -5:], sep='\n\t')
                        else:
                            print('b', bs[:5], bs[-5:], sep='\n\t')
                    print('prediction start', s[0])
                    print('next states', np.squeeze(s_next)[:5], np.squeeze(s_next)[-5:], sep='\n\t')
                    print('next predicted states', np.squeeze(s_next_pred)[:5], np.squeeze(s_next_pred)[-5:], sep='\n\t')
                    print('prediction random error', np.mean(random_errors), 
                        random_errors[:, :5], random_errors[:, -5:], sep='\n\t')
                    print('prediction deterministic error', np.mean(deter_errors), 
                        deter_errors[:, :5], deter_errors[:, -5:], sep='\n\t')
    # w_layer, b_layer, variables, grads, loss, w, b = train(3, 10, 100000, .01)
    # s, a, s_next = get_data(300, start=30)
    # s_next_pred = pred(s, a, w_layer, b_layer)
    # error = np.squeeze(np.abs(s_next - s_next_pred))
    # print('variables', *variables, sep='\n\t')
    # print('losses', loss)
    # print('w', w, sep='\n\t')
    # print('b', b, sep='\n\t')
    # # print('prediction error', error, sep='\n\t')