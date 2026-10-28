def get_nested_key(data, path, default_val=None):
    keys = path.split('.')
    curr = data
    for k in keys:
        if not isinstance(curr, dict):
            return default_val
        curr = curr.get(k)
        if curr is None:
            return default_val
    return curr
