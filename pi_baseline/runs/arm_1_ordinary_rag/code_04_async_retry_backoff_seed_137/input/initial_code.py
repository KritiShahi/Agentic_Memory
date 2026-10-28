def compute_delay(attempt, base_delay=1.0, max_delay=30.0):
    return base_delay * attempt # Bug: linear delay instead of exponential
