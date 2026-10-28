def partition_range(arr, start, end):
    if start > end:
        return []
    mid = (start + end) // 2
    return arr[start:mid], arr[mid:end + 1 if end + 1 <= len(arr) else len(arr)]
