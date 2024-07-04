def tree_flatten(tree):
    """Flatten a nested structure into a list of leaves and a structure descriptor."""
    if isinstance(tree, (list, tuple)):
        flat_leaves = []
        structure = []
        for i, subtree in enumerate(tree):
            sub_leaves, sub_structure = tree_flatten(subtree)
            flat_leaves.extend(sub_leaves)
            structure.append((type(tree), i, sub_structure))
        return flat_leaves, structure
    elif isinstance(tree, dict):
        flat_leaves = []
        structure = {}
        for key, subtree in tree.items():
            sub_leaves, sub_structure = tree_flatten(subtree)
            flat_leaves.extend(sub_leaves)
            structure[key] = sub_structure
        return flat_leaves, structure
    else:
        return [tree], None

def tree_unflatten(structure, leaves):
    """Reconstruct a nested structure from a list of leaves and a structure descriptor."""
    if isinstance(structure, list):
        unflattened = []
        for item in structure:
            type_, idx, sub_structure = item
            subtree, leaves = tree_unflatten(sub_structure, leaves)
            unflattened.append(subtree)
        return type_(unflattened), leaves
    elif isinstance(structure, dict):
        unflattened = {}
        for key, sub_structure in structure.items():
            subtree, leaves = tree_unflatten(sub_structure, leaves)
            unflattened[key] = subtree
        return unflattened, leaves
    else:
        return leaves[0], leaves[1:]

# Example usage
tree = {'a': [1, 2, {'b': 3}], 'c': (4, 5)}

# Flatten the tree
leaves, structure = tree_flatten(tree)
print("Leaves:", leaves)
print("Structure:", structure)

# Unflatten the tree
reconstructed_tree, _ = tree_unflatten(structure, leaves)
print("Reconstructed tree:", reconstructed_tree)
