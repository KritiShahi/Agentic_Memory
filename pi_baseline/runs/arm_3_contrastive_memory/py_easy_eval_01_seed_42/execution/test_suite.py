from window_stat import sliding_averages

def test_sliding_window_complete():
    nums = [1.0, 2.0, 3.0, 4.0, 5.0]
    avgs = sliding_averages(nums, 3)
    # Windows: [1,2,3]->2.0, [2,3,4]->3.0, [3,4,5]->4.0. Total 3 windows.
    assert len(avgs) == 3, f"Expected 3 windows, got {len(avgs)}"
    assert avgs == [2.0, 3.0, 4.0]

def test_sliding_window_exact_size():
    nums = [10.0, 20.0]
    assert sliding_averages(nums, 2) == [15.0]

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
