import tensorflow as tf


def compute_meta_grads_at_single_step(
    meta_tape, 
    eta, 
    grads, 
    theta, 
    out_grads, 
    i=None
):
    new_grads, new_out_grads = [], []
    for g, og in zip(grads, out_grads):
        if og is not None:
            new_grads.append(g)
            new_out_grads.append(og)
    grads, out_grads = new_grads, new_out_grads
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
    for gs in out_grads:
        for g in gs:
            tf.debugging.assert_all_finite(g, f'Bad {g.name}')
    grads = [meta_tape.gradient(meta_loss, eta)]
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
        for gs in out_grads:
            for g in gs:
                tf.debugging.assert_all_finite(g, f'Bad {g.name}')
        grads.append(eta_grads)
        out_grads_list.append(out_grads)
    if isinstance(grads[0], (tuple, list)):
        grads = [[g for g in mg if g is not None] for mg in zip(*grads)]
        for gs in grads:
            for g in gs:
                tf.debugging.assert_all_finite(g, f'Bad {g.name}')
    if return_out_grads:
        return grads, out_grads
    else:
        return grads
