def tree_flatten(tree):
  """Flatten a nested structure into a list of leaves and a structure descriptor."""
  if isinstance(tree, (list, tuple)):
    flat_leaves = []
    structure = []
    for subtree in tree:
      sub_leaves, sub_structure = tree_flatten(subtree)
      flat_leaves.extend(sub_leaves)
      structure.append(sub_structure)
    return flat_leaves, (type(tree), structure)
  elif isinstance(tree, dict):
    flat_leaves = []
    structure = {}
    for key, subtree in tree.items():
      sub_leaves, sub_structure = tree_flatten(subtree)
      flat_leaves.extend(sub_leaves)
      structure[key] = sub_structure
    return flat_leaves, (type(tree), structure)
  else:
    return [tree], None

def tree_unflatten(structure, leaves):
  """Reconstruct a nested structure from a list of leaves and a structure descriptor."""
  def _tree_unflatten(structure, leaves):
    if structure is None:
      return leaves[0], leaves[1:]
    elif isinstance(structure, tuple):
      type_, sub_structures = structure
      if isinstance(type_(), (list, tuple)):
        unflattened = []
        for sub_structure in sub_structures:
          subtree, leaves = _tree_unflatten(sub_structure, leaves)
          unflattened.append(subtree)
        if hasattr(type_, '_fields'):
          return type_(*unflattened), leaves
        else:
          return type_(unflattened), leaves
      elif isinstance(type_(), dict):
        unflattened = {}
        for key, sub_structure in sub_structures.items():
          subtree, leaves = _tree_unflatten(sub_structure, leaves)
          unflattened[key] = subtree
        return type_(unflattened), leaves
    else:
      raise TypeError("Unsupported structure type")

  return _tree_unflatten(structure, leaves)[0]

def tree_map(func, tree):
  if tree is None:
    return tree
  elif isinstance(tree, (list, tuple)):
    if hasattr(tree, '_fields'):
      return type(tree)(*[tree_map(func, x) for x in tree])
    else:
      return type(tree)(tree_map(func, x) for x in tree)
  elif isinstance(tree, dict):
    return type(tree)({k: tree_map(func, v) for k, v in tree.items()})
  else:
    return func(tree)

def tree_slice(d, loc=None, indices=None, axis=None):
  if loc is None:
    return tree_map(lambda x: x.take(indices=indices, axis=axis), d)
  else:
    return tree_map(lambda x: x[loc], d)
