import math
import csv
import sys

EPS = 0.01
DELTA = EPS / 2

SQRT_OF_5 = math.sqrt(5)

# Golden ratio coefficients
GR_COEFF_1 = (3 - SQRT_OF_5) / 2
GR_COEFF_2 = (SQRT_OF_5 - 1) / 2

# Fibonacci coefficients
FIB_COEFF_1 = (1 + SQRT_OF_5) / 2
FIB_COEFF_2 = (1 - SQRT_OF_5) / 2


def f(x):
    return 1 - (math.exp(-math.log(x) ** 2 / 0.5)) / (0.5 * x * math.sqrt(2 * math.pi))


def dichotomy(a, b):
    return (a + b - DELTA) / 2, (a + b + DELTA) / 2


def golden_ratio(a, b):
    return a + GR_COEFF_1 * (b - a), a + GR_COEFF_2 * (b - a)


def fibonacci_number(n):
    return (FIB_COEFF_1 ** n - FIB_COEFF_2 ** n) / SQRT_OF_5


def fibonacci(a, b):
    n = 1
    while fibonacci_number(n + 2) <= (b - a) / EPS:
        n += 1

    return a + fibonacci_number(n) / fibonacci_number(n + 2) * (b - a), \
           a + fibonacci_number(n + 1) / fibonacci_number(n + 2) * (b - a)


def write_to_file(data, path='output.txt', mode='w'):
    try:
        with open(path, mode, newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(data)
    except IOError:
        print('Error')


def search_of_min(a, b, func):
    if (b - a) < EPS:
        return (a + b) / 2

    x1, x2 = func(a, b)

    f1 = f(x1)
    f2 = f(x2)

    write_to_file([a, x1, x2, b], func.__name__ + '.csv', 'a')

    if f1 == f2:
        return search_of_min(x1, x2, func)
    elif f1 < f2:
        return search_of_min(a, x2, func)
    else:
        return search_of_min(x1, b, func)


def min_interval_search(x0):
    x = [x0]
    f0 = f(x[0])
    h = DELTA

    if f0 > f(x[0] + DELTA):
        x.append(x0 + DELTA)
        h *= 1
    elif f0 > f(x[0] - DELTA):
        x.append(x0 - DELTA)
        h *= -1

    k = 1
    while True:
        h *= 2
        x.append(x[k] + h)

        if f(x[k]) > f(x[k + 1]):
            k += 1
        else:
            break

    return [x[k - 1], x[k + 1]]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: linear_search.py [a] [b]')
        sys.exit()

    interval = [float(sys.argv[1]), float(sys.argv[2])]

    for func in [dichotomy, golden_ratio, fibonacci]:
        print(func.__name__)
        write_to_file(['a', 'x1', 'x2', 'b'], func.__name__ + '.csv')
        res = search_of_min(interval[0], interval[1], func)
        print('Min of f(x) is located at %d\n' % res)

    print('Minimum interval search')
    res = min_interval_search(interval[0])
    print('min of f(x) is located at', res)
