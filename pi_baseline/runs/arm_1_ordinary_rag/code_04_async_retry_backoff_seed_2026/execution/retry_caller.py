def compute_delay(attempt, base_delay=1.0, max_delay=30.0):
    delay = base_delay * (2 ** (attempt - 1))
    return min(delay, max_delay)
