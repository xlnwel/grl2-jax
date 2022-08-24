import tensorflow as tf

from utility.tf_utils import gather


def filter_null_grads(grads, out_grads):
    new_grads, new_out_grads = [], []
    for g, og in zip(grads, out_grads):
        if og is not None:
            new_grads.append(g)
            new_out_grads.append(og)
    return new_grads, new_out_grads


def compute_meta_grads_at_single_step(
    meta_tape, 
    eta, 
    grads, 
    theta, 
    out_grads, 
    i=None
):
    grads, out_grads = filter_null_grads(grads, out_grads)
    if i == 0:
        eta_grads = meta_tape.gradient(grads, eta, out_grads)
        return eta_grads, out_grads
    else:
        with meta_tape:
            d = tf.reduce_sum([tf.reduce_sum(v * g) for v, g in zip(grads, out_grads)])
        out_grads = [g1 + g2 for g1, g2 in zip(out_grads, meta_tape.gradient(d, theta))]
        eta_grads = meta_tape.gradient(d, eta)
        return eta_grads, out_grads


def compute_meta_gradients(
    meta_tape, 
    meta_loss, 
    grads_list, 
    theta, 
    eta, 
    return_out_grads=False, 
):
    inner_steps = len(grads_list)
    out_grads = meta_tape.gradient(meta_loss, theta)
    for g in out_grads:
        if g is not None:
            tf.debugging.assert_all_finite(g, f'Bad {g.name}')
    eta_grads_list = [meta_tape.gradient(meta_loss, eta)]
    out_grads_list = [out_grads]
    for i in reversed(range(inner_steps)):
        eta_grads, out_grads = compute_meta_grads_at_single_step(
            meta_tape, 
            eta, 
            grads_list[i], 
            theta, 
            out_grads, 
            i
        )
        for g in out_grads:
            if g is not None:
                tf.debugging.assert_all_finite(g, f'Bad {g.name}')
        eta_grads_list.append(eta_grads)
        out_grads_list.append(out_grads)
    if isinstance(eta_grads_list[0], (tuple, list)):
        eta_grads_list = [[g for g in mg if g is not None] for mg in zip(*eta_grads_list)]
        for gs in eta_grads_list:
            for g in gs:
                tf.debugging.assert_all_finite(g, f'Bad {g.name}')
    if return_out_grads:
        return eta_grads_list, out_grads_list
    else:
        return eta_grads_list


def inner_epoch(
    *, 
    config, 
    opt, 
    loss_fn,  
    use_meta=False, 
    debug=True, 
    use_dice=None, 
    return_grads=False, 
    **data,
):
    n_mbs = config.get('n_mbs', 1)
    grads_list = []
    if n_mbs > 1:
        indices = tf.range(next(iter(data.values())).shape[0], dtype=tf.int32)
        indices = tf.random.shuffle(indices)
        indices = tf.reshape(indices, (n_mbs, -1))
        for i in range(n_mbs):
            k = indices[i]
            data_k = gather(data, k)
            with tf.GradientTape() as tape:
                loss, terms = loss_fn(
                    tape=tape, 
                    **data_k, 
                    use_meta=use_meta, 
                    use_dice=use_dice, 
                    debug=debug
                )
            terms['grads_norm'], var_norms = opt(
                tape, loss, return_var_norms=True
            )
            terms['var_norm'] = list(var_norms.values())
            if return_grads:
                grads = opt.get_transformed_grads()
                grads = list(grads.values())
                for g in grads:
                    tf.debugging.assert_all_finite(g, f'Bad {g.name}')
                terms[f'trans_grads_norm'] = tf.linalg.global_norm(grads)
                grads_list.append(grads)
    else:
        with tf.GradientTape() as tape:
            loss, terms = loss_fn(
                tape=tape, 
                **data, 
                use_meta=use_meta, 
                use_dice=use_dice, 
                debug=debug
            )
        terms['grads_norm'], var_norms = opt(
            tape, loss, return_var_norms=True
        )
        terms['var_norm'] = list(var_norms.values())
        if return_grads:
            grads = opt.get_transformed_grads()
            grads = list(grads.values())
            for g in grads:
                tf.debugging.assert_all_finite(g, f'Bad {g.name}')
            terms['trans_grads_norm'] = tf.linalg.global_norm(grads)
            grads_list.append(grads)

    if return_grads:
        return terms, grads_list
    else:
        return terms
