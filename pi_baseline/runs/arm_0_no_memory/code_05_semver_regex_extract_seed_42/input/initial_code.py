import re
def parse_semver(v_str):
    m = re.match(r'^(\d+)\.(\d+)\.(\d+)$', v_str) # Bug: fails on pre-release tags
    if not m:
        return None
    return {'major': int(m.group(1)), 'minor': int(m.group(2)), 'patch': int(m.group(3)), 'tag': None}
