import numpy
from test_units import *

def do_gaussian_tests():
    test_gaussian = TestGausssian()
    test_gaussian.test_invert_sign_matrix()
    test_gaussian.test_invert_sign_line()
    test_gaussian.test_scalar_multiplication()
    test_gaussian.test_scalar_multiplication_in_line()
    test_gaussian.test_transpose_matrix()
    test_gaussian.test_sum_matrices()
    test_gaussian.test_sum_matrices_lines()
    test_gaussian.test_subtract_matrices()
    test_gaussian.test_subtract_matrices_lines()
    test_gaussian.test_multiply_matrices()


if __name__=="__main__":
    do_gaussian_tests()
