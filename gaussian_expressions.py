import numpy
def invert_sign_matrix(matrix):
    for i in range(len(matrix)):
        invert_sign_line(matrix, i)

def invert_sign_line(matrix, line):
    matrix[line] = -matrix[line]

def scalar_multiplication(matrix, scalar):
    for i in range(len(matrix)):
        scalar_multiplication_in_line(matrix, i, scalar)

def scalar_multiplication_in_line(matrix, line, scalar):
    matrix[line] *= scalar

def transpose_matrix(matrix):
    transposed = numpy.transpose(matrix)
    transposed = transposed.copy()

    matrix.reshape(numpy.shape(transposed))

    for i in range(len(matrix)):
        matrix[i] = transposed[i]

def sum_matrices(matrix_result, matrix_1, matrix_2):
    for i in range(len(matrix_result)):
        matrix_result[i] = matrix_1[i] + matrix_2[i]

def sum_matrices_lines(matrix_result, matrix_1, line_1, matrix_2, line_2):
    matrix_result[line_1] = matrix_1[line_1] + matrix_2[line_2]

def subtract_matrices(matrix_result, matrix_1, matrix_2):
    for i in range(len(matrix_result)):
        matrix_result[i] = matrix_1[i] - matrix_2[i]

def subtract_matrices_lines(matrix_result, matrix_1, line_1, matrix_2, line_2):
    matrix_result[line_1] = matrix_1[line_1] - matrix_2[line_2]

def multiply_matrices(matrix_result, matrix_1, matrix_2):
    for i in range(len(matrix_result)):
        matrix_result[i] = matrix_1[i] * matrix_2[i]