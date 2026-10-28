import re
def parse_semver(v_str):
    m = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9_.-]+))?$', v_str)
    if not m:
        return None
    return {'major': int(m.group(1)), 'minor': int(m.group(2)), 'patch': int(m.group(3)), 'tag': m.group(4)}
