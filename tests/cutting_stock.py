import unittest
import os
import numpy as np
import pandas as pd
from src.cutting_stock import CuttingStockProblem


class TestCuttingStockProblem(unittest.TestCase):
    def setUp(self):
        self.csp = CuttingStockProblem(
            data_url="https://raw.githubusercontent.com/EKU-Summer-2021/" +
                    "intelligent_system_data/main/Intelligent%20System%20Data/CSP/CSP_360.csv",
            stock_size=60)

    def test_init(self):
        self.assertEqual(60, self.csp.stock_size)
        self.assertEqual(True, self.csp.count > 0)
        self.assertEqual(True, self.csp.initial_position.all())

    def test_solutions_from_position_mx(self):
        test_mx = np.array([[1, 2], [51, 52]])
        self.assertEqual("[[1, 2]]\n[[51], [52]]", self.csp.solutions_from_position_mx(test_mx))

    def test_solution_from_position(self):
        test_mx = np.array([1, 2, 51, 52])
        self.assertEqual("[[1, 2, 51], [52]]", self.csp.solution_from_position(test_mx))

    def test_save_solutions_to_file(self):
        test_mx = np.array([[1, 2, 51, 52]])
        cost = np.array([2])
        self.csp.save_solutions_to_file(test_mx, cost, 'test.csv')
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../solutions', 'test.csv'))
        self.assertEqual(False, df.empty)


if __name__ == '__main__':
    unittest.main()
