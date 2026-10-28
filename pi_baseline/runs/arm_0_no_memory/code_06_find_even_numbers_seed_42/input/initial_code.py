def find_even_numbers(numbers):
    # Bug: filters odd numbers instead of even
    return [x for x in numbers if x % 2 != 0]
