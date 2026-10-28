def sliding_averages(data, window_size):
    if not data or window_size <= 0 or len(data) < window_size:
        return []
    res = []
    for i in range(len(data) - window_size + 1):
        chunk = data[i : i + window_size]
        res.append(sum(chunk) / window_size)
    return res
