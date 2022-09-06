import tensorflow as tf

from core.utils import get_vars_for_modules
from tools.tf_utils import assert_finite, gather


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
    theta, 
    grads, 
    out_grads, 
    i=None
):
    grads, out_grads = filter_null_grads(grads, out_grads)
    if i == 0:
        eta_grads = meta_tape.gradient(grads, eta, out_grads)
    else:
        eta_grads = meta_tape.gradient(grads, eta, out_grads)
        out_grads = [g1 + g2 for g1, g2 in zip(
            out_grads, meta_tape.gradient(grads, theta, out_grads))]
    return eta_grads, out_grads


def compute_meta_gradients(
    meta_tape, 
    meta_loss, 
    grads_list, 
    theta_list, 
    eta, 
    return_out_grads=False, 
):
    assert len(grads_list) == len(theta_list) - 1, (len(grads_list), len(theta_list))
    inner_steps = len(grads_list)
    out_grads = meta_tape.gradient(meta_loss, theta_list[-1])
    eta_grads = meta_tape.gradient(meta_loss, eta)
    assert_finite(out_grads, 'out grads')
    assert_finite(eta_grads, 'eta grads')
    eta_grads_list = [eta_grads]
    out_grads_list = [out_grads]
    for i in reversed(range(inner_steps)):
        eta_grads, out_grads = compute_meta_grads_at_single_step(
            meta_tape, 
            eta=eta, 
            theta=theta_list[i], 
            grads=grads_list[i], 
            out_grads=out_grads, 
            i=i
        )
        assert_finite(eta_grads, f"Iteration {i} out grads")
        assert_finite(eta_grads, f"Iteration {i} eta grads")
        eta_grads_list.append(eta_grads)
        out_grads_list.append(out_grads)
    if isinstance(eta_grads_list[0], (tuple, list)):
        eta_grads_list = [[g for g in mg if g is not None] for mg in zip(*eta_grads_list)]
    eta_grads = [sum(mg) for mg in eta_grads_list]
    if return_out_grads:
        return eta_grads, out_grads_list
    else:
        return eta_grads


def compute_meta_grads_for_modules(
    meta_tape, 
    meta_modules, 
    meta_loss, 
    grads_list, 
    theta, 
):
    eta = get_vars_for_modules(meta_modules)
    meta_grads_list = compute_meta_gradients(
        meta_tape=meta_tape, 
        meta_loss=meta_loss, 
        grads_list=grads_list, 
        theta=theta, 
        eta=eta, 
    )
    meta_grads = [sum(mg) for mg in meta_grads_list]
    assert len(meta_grads) == len(eta), (len(meta_grads), len(eta))
    
    return meta_grads


def inner_epoch(
    *, 
    config={}, 
    opt, 
    loss_fn,  
    use_meta=False, 
    debug=True, 
    use_dice=None, 
    return_stats_for_meta=False, 
    **data,
):
    n_mbs = config.get('n_mbs', 1)
    vars_list = opt.variables
    theta_list = []
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
            if return_stats_for_meta:
                grads = opt.get_transformed_grads(vars_list)
                grads = list(grads.values())
                assert_finite(grads)
                grads = list(grads.values())
                terms[f'trans_grads_norm'] = tf.linalg.global_norm(grads)
                grads_list.append(grads)
                
                theta = opt.get_new_vars(vars_list)
                theta = list(theta.values())
                theta_list.append(theta)
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
        if return_stats_for_meta:
            grads = opt.get_transformed_grads()
            grads = list(grads.values())
            assert_finite(grads)
            terms['trans_grads_norm'] = tf.linalg.global_norm(grads)
            grads_list.append(grads)
            
            theta = opt.get_new_vars(vars_list)
            theta = list(theta.values())
            theta_list.append(theta)

    if return_stats_for_meta:
        return terms, theta_list, grads_list
    else:
        return terms
