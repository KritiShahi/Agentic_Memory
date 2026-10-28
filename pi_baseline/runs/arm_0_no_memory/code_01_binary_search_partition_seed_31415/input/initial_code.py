def partition_range(arr, start, end):
    if start > end:
        return []
    mid = (start + end) // 2
    # Bug: end + 1 slices past the intended partition
    return arr[start:mid], arr[mid:end + 1]
