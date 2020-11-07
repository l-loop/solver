import numpy
import gaussian_expressions as gaussian

class TestGausssian():
    def test_invert_sign_matrix(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])
        result_matrix = numpy.array([
            [-1,-2,-3],
            [4,-5,6],
            [7,8,9]
        ])

        gaussian.invert_sign_matrix(matrix)
        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_invert_sign_line(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])
        result_matrix = numpy.array([
            [-1,-2,-3],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        line = 0

        gaussian.invert_sign_line(matrix, 0)


        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_scalar_multiplication(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])
        result_matrix = numpy.array([
            [-2,-4,-6],
            [8,-10,12],
            [14,16,18]
        ])

        gaussian.scalar_multiplication(matrix, -2)


        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_scalar_multiplication_in_line(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])
        result_matrix = numpy.array([
            [-2,-4,-6],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        gaussian.scalar_multiplication_in_line(matrix, 0,  -2)


        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_transpose_matrix(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])
        result_matrix = numpy.array([
            [1,-4,-7],
            [2,5,-8],
            [3,-6,-9]
        ])

        gaussian.transpose_matrix(matrix)

        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_sum_matrices(self):
        matrix_1 = numpy.array([
            [4,0,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        matrix_2 = numpy.array([
            [-7,-8,-9],
            [3,2,3],
            [-4,5,-6]
        ])

        result_matrix = numpy.array([
            [-3,-8,-6],
            [-1,7,-3],
            [-11,-3,-15]
        ])

        gaussian.sum_matrices(matrix_1, matrix_1, matrix_2)

        numpy.testing.assert_almost_equal(
            matrix_1, result_matrix, decimal=12
        )

    def test_sum_matrices_lines(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        result_matrix = numpy.array([
            [1,2,3],
            [-11,-3,-15],
            [-7,-8,-9]
        ])

        gaussian.sum_matrices_lines(matrix, matrix, 1, matrix, 2)

        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_subtract_matrices(self):
        matrix_1 = numpy.array([
            [4,0,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        matrix_2 = numpy.array([
            [-7,-8,-9],
            [3,2,3],
            [-4,5,-6]
        ])

        result_matrix = numpy.array([
            [11,8,12],
            [-7,3,-9],
            [-3,-13,-3]
        ])

        gaussian.subtract_matrices(matrix_1, matrix_1, matrix_2)

        numpy.testing.assert_almost_equal(
            matrix_1, result_matrix, decimal=12
        )

    def test_subtract_matrices_lines(self):
        matrix = numpy.array([
            [1,2,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        result_matrix = numpy.array([
            [1,2,3],
            [3,13,3],
            [-7,-8,-9]
        ])

        gaussian.subtract_matrices_lines(matrix, matrix, 1, matrix, 2)

        numpy.testing.assert_almost_equal(
            matrix, result_matrix, decimal=12
        )

    def test_multiply_matrices(self):
        matrix_1 = numpy.array([
            [4,0,3],
            [-4,5,-6],
            [-7,-8,-9]
        ])

        matrix_2 = numpy.array([
            [-7,-8,-9],
            [3,2,3],
            [-4,5,-6]
        ])

        result_matrix = numpy.array([
            [-28,0,-27],
            [-12,10,-18],
            [ 28,-40,54]
        ])

        gaussian.multiply_matrices(matrix_1, matrix_1, matrix_2)

        numpy.testing.assert_almost_equal(
            matrix_1, result_matrix, decimal=12
        )