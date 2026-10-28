def build_query(template, params):
    q = template
    for k, v in params.items():
        if isinstance(v, (int, float)):
            q = q.replace(f':{k}', str(v))
        elif isinstance(v, bool):
            q = q.replace(f':{k}', 'TRUE' if v else 'FALSE')
        else:
            q = q.replace(f':{k}', f'"{v}"')
    return q
