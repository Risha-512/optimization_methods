import sys
import math
import numpy as np

EPS = 1e-6
DELTA = EPS / 2
MAX_ITER = 50
DIRECTION = 0


def f(vector):
    x = vector[0]
    y = vector[1]

    # this function doesn't have global minimum
    # return math.exp(- (x - 3) ** 2 - (y - 1) ** 2 / 9) + 2 * math.exp(- (x - 2) ** 2 / 4 - (y - 2) ** 2)

    # these functions doesn't have global maximum
    return 100 * (y - x) ** 2 + (1 - x) ** 2
    # return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def grad_f(vector):
    x = vector[0]
    y = vector[1]

    # der_x = -2 * (x - 3) * math.exp(- (x - 3) ** 2 - 1 / 9 * (y - 1) ** 2) - (x - 2) * \
    #         math.exp(- 1 / 4 * (x - 2) ** 2 - (y - 2) ** 2)
    # der_y = -4 * (y - 2) * math.exp(- 1 / 4 * (x - 2) ** 2 - (y - 2) ** 2) - 2 / 9 * (y - 1) * \
    #         math.exp(- (x - 3) ** 2 - 1 / 9 * (y - 1) ** 2)

    der_x = 202 * x - 200 * y - 2
    der_y = 200 * (y - x)

    # der_x = 400 * x ** 3 - 400 * x * y + 2 * x - 2
    # der_y = 200 * (y - x ** 2)

    return np.array([der_x, der_y])


def dichotomy(a, b):
    return (a + b - DELTA) / 2, (a + b + DELTA) / 2


# calculate arg min(x + old_lambda * s)
def linear_search(x, s, a=-50, b=50):
    lamb1, lamb2 = dichotomy(a, b)

    if (b - a) < EPS:
        return (lamb1 + lamb2) / 2

    f1 = f(x + lamb1 * s)
    f2 = f(x + lamb2 * s)

    if f1 == f2:
        return linear_search(x, s, lamb1, lamb2)
    elif f1 < f2 and DIRECTION == -1 or f1 > f2 and DIRECTION == 1:
        return linear_search(x, s, a, lamb2)
    else:
        return linear_search(x, s, lamb1, b)


# calculate approximation to the Hesse matrix
def count_matrix(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    delta_x = np.array([x_new - x]).transpose()

    # this value can ruin program if initial values will be too distant from answer
    coef = np.dot(delta_x.transpose(), grad)
    if coef == 0:
        print("Can't calculate, please set another initial values")
        sys.exit()
    else:
        # sum of the A-matrices will be equal to the invertible Hesse matrix
        a_matrix = np.dot(delta_x, delta_x.transpose()) / coef

        # sum of the B-matrices is reduced with the initial value
        b_matrix = np.dot(matrix, grad)
        b_matrix = np.dot(b_matrix, grad.transpose())
        b_matrix = np.dot(b_matrix, matrix)
        b_matrix /= np.dot(np.dot(grad.transpose(), matrix), grad)

        return matrix + a_matrix - b_matrix


def davidon_fletcher_powell_method(x):
    matrix = np.identity(2)
    iteration = 0

    while np.linalg.norm(grad_f(x)) > EPS and iteration < MAX_ITER:
        iteration += 1
        s = DIRECTION * np.dot(matrix, grad_f(x))  # vector of direction
        lamb = linear_search(x, s)   # arg min(x + old_lambda * s)

        x_new = x + lamb * s
        matrix = count_matrix(x, x_new, matrix)   # calculate approximation to the Hesse matrix

        x = x_new

    return x, iteration


def conjugate_gradient_method(x):
    s = DIRECTION * grad_f(x)
    iteration = 0

    while np.linalg.norm(grad_f(x)) > EPS and iteration < MAX_ITER:
        iteration += 1
        lamb = linear_search(x, s)   # arg min(x + old_lambda * s)
        x_new = x + lamb * s

        w = np.linalg.norm(grad_f(x_new)) ** 2 / np.linalg.norm(grad_f(x)) ** 2   # weight coefficient
        s = DIRECTION * grad_f(x_new) + w * s  # vector of direction

        x = x_new

    return x, iteration


if __name__ == "__main__":
    if len(sys.argv) != 4 and sys.argv[3] != "min" and sys.argv[3] != "max":
        print("Usage: penalty_functions_method.py [x begin] [y begin] [type - min or max]")
        sys.exit()

    x_init = np.array([float(sys.argv[1]), float(sys.argv[2])])
    DIRECTION = 1 if sys.argv[3] == "max" else -1

    print("Searching %simum of function\n" % sys.argv[3])

    res, it = davidon_fletcher_powell_method(x_init)
    if it == MAX_ITER:
        print("This method can't calculate or function doesn't have global %simum. "
              "Value on %d iter:\n" % (sys.argv[3], MAX_ITER))

    print("Davidon-Fletcher-Powell method:\nx = %f\ny = %f\nf(x, y) = %f\niter:%d\n" % (res[0], res[1], f(res), it))

    res, it = conjugate_gradient_method(x_init)
    if it == MAX_ITER:
        print("This method can't calculate or function doesn't have global %simum. "
              "Value on %d iter:\n" % (sys.argv[3], MAX_ITER))
    print("Conjugate gradient method:\nx = %f\ny = %f\nf(x, y) = %f\niter:%d\n" % (res[0], res[1], f(res), it))
