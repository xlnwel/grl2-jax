from tools.tree_ops import tree_map


def update_params(source, target, polyak):
  return tree_map(
    lambda x, y: polyak * x + (1.-polyak) * y, target, source)
