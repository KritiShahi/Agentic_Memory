def build_query(template, params):
    q = template
    for k, v in params.items():
        q = q.replace(f':{k}', f'"{v}"') # Bug: quotes all values indiscriminately
    return q
