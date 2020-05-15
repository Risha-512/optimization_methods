import math
import numpy as np
import scipy.optimize as op
import sys

EPS = 1e-07


def f(x):
    return (x[0] - x[1]) ** 2 + 10 * (x[0] + 5) ** 2


def h(x):
    return x[0] + x[1] - 1


def g(x):
    return - x[0] - x[1]


def h_penalty(value, power=1):
    return abs(value) ** power


def g_penalty(value, power=1):
    return ((value + abs(value)) / 2) ** power


def g_barrier_inv(value):
    return -1 / value if value else 0


def g_barrier_log(value):
    return -math.log(-value) if value < 0 else 0


def continue_searching(r, funcs, delta):
    search = False
    for index in range(len(r)):
        if abs(r[index] * funcs[index]) > EPS:
            r[index] *= delta
            search = True

    return search


def penalty_method(x, r, funcs, delta):
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
    while continue_searching(r, funcs(x), delta):
        curr_min = op.fmin_powell(lambda arg: f(arg) + sum(r * funcs(arg)), x, disp=False, maxiter=100)
        if g(curr_min) <= 0:
            x = curr_min

    return x


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: penalty_functions_method.py [x begin] [y begin]")
        sys.exit()

    x_init = np.array([float(sys.argv[1]), float(sys.argv[2])])

    print("f(x) = (x - y)^2 + 10 * (x + 5)^2\nh(x) = x + y - 1\ng(x) = - x - y\nx initial = {}\n".format(x_init))

    result = penalty_method(x_init, np.ones(2), lambda arg: np.array([h_penalty(h(arg)), g_penalty(g(arg))]), 1.1)
    print("Penalty method results:\nH(x) = |x|^a\nG(x) = ((x + |x|) / 2)^a\n")
    print("a = 1\nx = {}\nf(x) = {}\n".format(result, f(result)))

    result = penalty_method(x_init, np.ones(2), lambda arg: np.array([h_penalty(h(arg), 2), g_penalty(g(arg), 2)]), 1.1)
    print("a = 2\nx = {}\nf(x) = {}\n".format(result, f(result)))

    result = penalty_method(x_init, np.ones(2), lambda arg: np.array([h_penalty(h(arg), 4), g_penalty(g(arg), 4)]), 1.1)
    print("a = 2\nx = {}\nf(x) = {}\n".format(result, f(result)))

    result = penalty_method(x_init, np.array([550]), lambda arg: np.array([g_barrier_inv(g(arg))]), 0.9)
    print("Barrier method results:\n")
    print("G(x) = -1 / x:\nx = {}\nf(x) = {}\n".format(result, f(result)))

    result = penalty_method(x_init, np.array([550]), lambda arg: np.array([g_barrier_log(g(arg))]), 0.9)
    print("G(x) = -ln(-x):\nx = {}\nf(x) = {}\n".format(result, f(result)))
