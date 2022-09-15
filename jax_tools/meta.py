import jax


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