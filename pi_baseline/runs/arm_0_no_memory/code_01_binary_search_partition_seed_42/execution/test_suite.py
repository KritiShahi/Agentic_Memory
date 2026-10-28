import pytest
from partitioner import partition_range

def test_partition_even():
    arr = [0, 1, 2, 3, 4, 5]
    left, right = partition_range(arr, 0, 5)
    assert left == [0, 1] and right == [2, 3, 4, 5]

def test_partition_boundary():
    arr = [10, 20, 30]
    left, right = partition_range(arr, 0, 2)
    assert len(left) + len(right) == 3
    assert right[-1] == 30

if __name__ == '__main__':
    import sys
    failed = False
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS: {name}")
            except Exception as e:
                print(f"FAIL: {name}: {e}", file=sys.stderr)
                failed = True
    if failed:
        sys.exit(1)
