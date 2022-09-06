import chex


def assert_rank_compatibility(tensors, rank=None):
    tensors = [t for t in tensors if t is not None]
    if rank:
        chex.assert_equal_rank(tensors, rank)
    else:
        chex.assert_rank(tensors)

def assert_shape_compatibility(tensors, shape=None):
    tensors = [t for t in tensors if t is not None]
    if shape:
        chex.assert_shape(tensors, shape)
    else:
        chex.assert_equal_shape(tensors)

def assert_all_finite(tensors):
    chex.assert_tree_all_finite(tensors)
