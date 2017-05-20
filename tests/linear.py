import unittest


class TestLinearRegression(unittest.TestCase):
    def test_basic_linear_regression(self):
        import numpy as np
        from mlp.regression.linear import BasicLinearRegression

        X = np.array([[x] for x in range(6)])
        y = np.array([x for x in range(6)])

        model = BasicLinearRegression()

        model.fit(X, y)

        self.assertEqual(2, len(model.coefficients))
        self.assertEqual(0, model.coefficients[0])
        self.assertEqual(1, model.coefficients[1])

        y_hat = model.transform(X)

        for i in range(len(y)):
            self.assertEqual(y[i], y_hat[i])


if __name__ == '__main__':
    unittest.main()
