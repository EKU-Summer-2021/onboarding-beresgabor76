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
        #given
        EXPECTED_STOCK_SIZE = 60
        EXPECTED_PIECE_COUNT = 13
        EXPECTED_INITIAL_POSITION = np.array([22., 22., 36., 36., 34., 34., 19., 19., 19., 19., 32., 32., 32.])
        #when
        ACTUAL_STOCK_SIZE = self.csp.stock_size
        ACTUAL_PIECE_COUNT = self.csp.count
        ACTUAL_INITIAL_POSITION = self.csp.initial_position
        #then
        self.assertEqual(EXPECTED_STOCK_SIZE, ACTUAL_STOCK_SIZE)
        self.assertEqual(EXPECTED_PIECE_COUNT, ACTUAL_PIECE_COUNT)
        np.allclose(EXPECTED_INITIAL_POSITION, ACTUAL_INITIAL_POSITION)

    def test_solutions_from_position_mx(self):
        #given
        POSITION_MX = np.array([[1, 2], [51, 52]])
        EXPECTED_SOLUTION = "[[1, 2]]\n[[51], [52]]"
        #when
        ACTUAL_SOLUTION = self.csp.solutions_from_position_mx(POSITION_MX)
        #then
        self.assertEqual(EXPECTED_SOLUTION, ACTUAL_SOLUTION)

    def test_solution_from_position(self):
        #given
        POSITION = np.array([1, 2, 51, 52])
        EXPECTED_SOLUTION = "[[1, 2, 51], [52]]"
        #when
        ACTUAL_SOLUTION = self.csp.solution_from_position(POSITION)
        #then
        self.assertEqual(EXPECTED_SOLUTION, ACTUAL_SOLUTION)

    def test_save_solutions_to_file(self):
        #given
        TEST_POSITION_MX = np.array([[1, 2, 51, 52]])
        TEST_COST_MX = np.array([2])
        #when
        self.csp.save_solutions_to_file(TEST_POSITION_MX, TEST_COST_MX, 'test.csv')
        #then
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../solutions', 'test.csv'))
        self.assertEqual(2, df.iloc[0, 0])
        self.assertEqual(str([1, 2, 51]), df.iloc[0, 1])
        self.assertEqual(str([52]), df.iloc[0, 2])



if __name__ == '__main__':
    unittest.main()
