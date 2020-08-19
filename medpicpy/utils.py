"""
Utility functions for the package that the user shouldn't need.
"""
def remove_sub_paths(paths):
    """Since glob.glob with recursive doesn't
    only take the longest path we need to remove
    paths that are a part of other paths.

    Args:
        paths (array): array of paths

    Returns:
        array: array of paths that aren't a subset of other paths
    """
    for path in paths:
        for other in paths:
            if path in other and path != other:
                paths.remove(path)
                break
    
    return paths