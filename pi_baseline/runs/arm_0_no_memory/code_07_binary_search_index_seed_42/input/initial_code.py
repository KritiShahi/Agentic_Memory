def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left < right: # Bug: misses single element or right boundary
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return -1
