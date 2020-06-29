import math
import numpy as np
import scipy.optimize as op
import sys
import csv
import warnings

EPS = 1e-03


def f(x):
    return (x[0] - x[1]) ** 2 + 10 * (x[0] + 5) ** 2


def g(x):
    return - x[0] - x[1]


def h(x):
    return x[0] + x[1] - 1


def h_penalty(value, power=1):
    return abs(value) ** power


def g_penalty(value, power=1):
    return ((value + abs(value)) / 2) ** power


def g_barrier_inv(value):
    return -1 / value if value < 0 else np.finfo(np.float64).max


def g_barrier_log(value):
    return -math.log(-value) if value < 0 else np.finfo(np.float64).max


def write_to_file(data, path='output.txt', mode='w'):
    try:
        with open(path, mode, newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(data)
    except IOError:
        print('Error')


def penalty_method(x, r, funcs, delta, filename='output.csv'):
    """
    Penalty/barrier method realization
    :param x: initial value vector
    :param r: vector of coefficients
    :param funcs: vector of penalty/barrier functions
    :param delta: r change coefficient
    :return: min f(x) with conditions from funcs (g(x) <= 0, h(x) = 0)

    Q(x, r) = f(x) + sum(rj * H(h(x))) + sum(rj * G(g(x)))
    Some functions of H and G may be missing
    """
    
    data_array = [['x', 'f(x)', 'r'], [x, f(x), r]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        while True:
            x_old = x.copy()
            x = op.fmin_powell(lambda arg: f(arg) + sum(r * funcs(arg)), x, disp=False, maxiter=100)
            if np.linalg.norm(x - x_old) > EPS:
                r = np.multiply(r, delta)
                data_array.append([x, f(x), r])
            else:
                break

    write_to_file(data_array, filename)

    return x


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: penalty_functions_method.py [x begin vector]')
        sys.exit()

    x_init = np.array([float(x) for x in sys.argv[1:]])

    # Penalty method, inequality limitation
    print('Penalty method:\nG(x) = ((x + |x|) / 2)^a\n')
    for a in [1, 2, 4]:
        res = penalty_method(x_init, np.array([1]), lambda arg: np.array([g_penalty(g(arg), a)]), 1.5, 'pmg%d.csv' % a)
        print('a = {}\nx = {}\nf(x) = {}\n'.format(a, res, f(res)))

    # Penalty method, equality limitation
    print('H(x) = |x|^a\n')
    for a in [1, 2, 4]:
        res = penalty_method(x_init, np.array([1]), lambda arg: np.array([h_penalty(h(arg), a)]), 1.5, 'pmh%d.csv' % a)
        print("a = {}\nx = {}\nf(x) = {}\n".format(a, res, f(res)))

    # Barrier method, inequality limitation
    res = penalty_method(x_init, np.array([1]), lambda arg: np.array([g_barrier_inv(g(arg))]), 0.5, 'bm_inv.csv')
    print('Barrier method:\n')
    print('G(x) = -1 / x:\nx = {}\nf(x) = {}\n'.format(res, f(res)))

    res = penalty_method(x_init, np.array([1]), lambda arg: np.array([g_barrier_log(g(arg))]), 0.5, 'bm_log.csv')
    print('G(x) = -ln(-x):\nx = {}\nf(x) = {}\n'.format(res, f(res)))
