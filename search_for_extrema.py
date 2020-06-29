import sys
import csv
import math
import numpy as np

EPS = 1e-03
DELTA = EPS / 2
MAX_ITER = 50
DIRECTION = 0


def f(vector):
    x = vector[0]
    y = vector[1]

    return 5 * (2 * x + y - 10) ** 2 + (x - 2 * y + 4) ** 2


def grad_f(vector):
    x = vector[0]
    y = vector[1]

    der_x = 42 * x + 16 * y - 192
    der_y = 16 * x + 18 * y - 116

    return np.array([der_x, der_y])


def write_to_file(data, path='output.txt', mode='w'):
    try:
        with open(path, mode, newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(data)
    except IOError:
        print('Error')


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
def davidon_fletcher_powell(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    delta_x = np.array([x_new - x]).transpose()

    # this value can ruin program if initial values will be too distant from answer
    coef = np.dot(delta_x.transpose(), grad)
    if coef == 0:
        print('Can\'t calculate, please set another initial values')
        sys.exit()
    # sum of the A-matrices will be equal to the invertible Hesse matrix
    a_matrix = np.dot(delta_x, delta_x.transpose()) / coef

    # sum of the B-matrices is reduced with the initial value
    b_matrix = np.dot(matrix, grad)
    b_matrix = np.dot(b_matrix, grad.transpose())
    b_matrix = np.dot(b_matrix, matrix)
    b_matrix /= np.dot(np.dot(grad.transpose(), matrix), grad)

    return matrix + a_matrix - b_matrix


def newton_raphson(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    numerator = np.dot(matrix, grad)
    numerator = np.dot(numerator, numerator.transpose())
    denominator = np.dot(grad.transpose(), matrix)
    denominator = np.dot(denominator, grad)

    if denominator == 0:
        print('Can\'t calculate, please set another initial values')
        sys.exit()

    return matrix - numerator / denominator


def broyden(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    delta_x = np.array([x_new - x]).transpose()

    numerator = delta_x - np.dot(matrix, grad)
    numerator = np.dot(numerator, numerator.transpose())
    denominator = (delta_x - np.dot(matrix, grad)).transpose()
    denominator = np.dot(denominator, grad)

    return numerator / denominator


def pearson3(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    delta_x = np.array([x_new - x]).transpose()

    numerator = delta_x - np.dot(matrix, grad)
    numerator = np.dot(numerator, np.dot(matrix, grad).transpose())
    denominator = np.dot(grad.transpose(), matrix)
    denominator = np.dot(denominator, grad)

    return numerator / denominator


def fletcher(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    delta_x = np.array([x_new - x]).transpose()

    numerator = np.dot(delta_x, grad.transpose())
    denominator = np.dot(delta_x.transpose(), grad)
    matrix_a = np.dot(np.identity(2) - numerator / denominator, matrix)

    numerator = np.dot(grad, delta_x.transpose())
    denominator = np.dot(delta_x.transpose(), grad)
    matrix_a = np.dot(matrix_a, np.identity(2) - numerator / denominator)

    numerator = np.dot(delta_x, delta_x.transpose())
    denominator = np.dot(delta_x.transpose(), grad)

    return matrix_a + numerator / denominator


def greenstadt(x, x_new, matrix):
    grad = np.array([grad_f(x_new) - grad_f(x)]).transpose()
    delta_x = np.array([x_new - x]).transpose()

    denominator = np.dot(grad.transpose(), matrix)
    denominator = np.dot(denominator, grad)
    coef = 1 / denominator

    matrix_a = np.dot(delta_x, grad.transpose())
    matrix_a = np.dot(matrix_a, matrix)
    matrix_b = np.dot(matrix, grad)
    matrix_b = np.dot(matrix_b, delta_x.transpose())

    numerator1 = np.dot(grad.transpose(), delta_x)
    numerator2 = np.dot(grad.transpose(), matrix)
    numerator2 = np.dot(numerator2, grad)

    numerator = numerator1 + numerator2
    numerator = numerator * matrix
    numerator = np.dot(numerator, grad)
    numerator = np.dot(numerator, grad.transpose())
    numerator = np.dot(numerator, matrix)

    denominator = np.dot(grad.transpose(), matrix)
    denominator = np.dot(denominator, grad)

    return coef * (matrix_a + matrix_b - numerator / denominator)


def variable_metric_method(x, count_matrix, filename='output.csv'):
    matrix = np.identity(2)
    iteration = 0

    data_array = [['x', 'f(x)', 'grad', 'matrix', 's']]

    while np.linalg.norm(grad_f(x)) > EPS and iteration < MAX_ITER:
        iteration += 1
        s = DIRECTION * np.dot(matrix, grad_f(x))  # vector of direction
        lamb = linear_search(x, s)   # arg min(x + old_lambda * s)

        data_array.append([x, f(x), grad_f(x), matrix, s])
        x_new = x + lamb * s
        if np.linalg.norm(x_new - x) <= EPS:
            break

        matrix = count_matrix(x, x_new, matrix)   # calculate approximation to the Hesse matrix
        x = x_new

    data_array.append([x, f(x), grad_f(x), matrix, s])
    write_to_file(data_array, filename)

    return x, iteration


def conjugate_gradient_method(x, filename='output.csv'):
    s = DIRECTION * grad_f(x)
    iteration = 0

    data_array = [['x', 'f(x)', 'grad', 'w', 's']]

    while np.linalg.norm(grad_f(x)) > EPS and iteration < MAX_ITER:
        iteration += 1
        lamb = linear_search(x, s)   # arg min(x + old_lambda * s)
        x_new = x + lamb * s

        w = np.linalg.norm(grad_f(x_new)) ** 2 / np.linalg.norm(grad_f(x)) ** 2   # weight coefficient
        s = DIRECTION * grad_f(x_new) + w * s  # vector of direction

        data_array.append([x, f(x), grad_f(x), w, s])
        x = x_new

    data_array.append([x, f(x), grad_f(x), w, s])
    write_to_file(data_array, filename)

    return x, iteration


if __name__ == '__main__':
    if len(sys.argv) != 4 and sys.argv[3] != 'min' and sys.argv[3] != 'max':
        print('Usage: search_for_extrema.py [x begin] [y begin] [type - min or max]')
        sys.exit()

    x_init = np.array([float(sys.argv[1]), float(sys.argv[2])])
    DIRECTION = 1 if sys.argv[3] == 'max' else -1

    print('Searching %simum of function\n' % sys.argv[3])

    # Variable metric methods
    for method in [davidon_fletcher_powell, newton_raphson, broyden, pearson3, fletcher, greenstadt]:
        res, it = variable_metric_method(x_init, method, method.__name__ + '.csv')
        if it == MAX_ITER:
            print('This method can\'t calculate or function doesn\'t have global %simum. '
                  'Value on %d iter:\n' % (sys.argv[3], MAX_ITER))
        print('%s:\nx = %f\ny = %f\nf(x, y) = %f\niter:%d\n' % (method.__name__, res[0], res[1], f(res), it))

    # Conjugate gradient
    res, it = conjugate_gradient_method(x_init, 'conjugate_gradient.csv')
    if it == MAX_ITER:
        print('This method can\'t calculate or function doesn\'t have global %simum. '
              'Value on %d iter:\n' % (sys.argv[3], MAX_ITER))
    print('conjugate_gradient:\nx = %f\ny = %f\nf(x, y) = %f\niter:%d\n' % (res[0], res[1], f(res), it))
