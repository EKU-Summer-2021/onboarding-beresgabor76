import unittest
import numpy as np
from src.cutting_stock import CuttingStock


class TestCuttingStock(unittest.TestCase):
    def setUp(self):
        self.csp = CuttingStock(
            data_url="https://raw.githubusercontent.com/EKU-Summer-2021/" +
                    "intelligent_system_data/main/Intelligent%20System%20Data/CSP/CSP_360.csv",
            stock_size=60)

    def test_config_data(self):
        self.assertEqual(True, self.csp._CuttingStock__swarm_size > 0)
        self.assertEqual(True, self.csp._CuttingStock__iterations > 0)
        self.assertEqual(True, self.csp._CuttingStock__inertia != 0)
        self.assertEqual(True, self.csp._CuttingStock__accel_best != 0)
        self.assertEqual(True, self.csp._CuttingStock__accel_global_best != 0)

    def test_initialize(self):
        self.csp.initialize()
        self.assertEqual(True, self.csp._CuttingStock__position_mx.all())
        self.assertEqual(True, np.array_equal(self.csp._CuttingStock__velocity_mx,
                                              np.around(self.csp._CuttingStock__velocity_mx)))
        self.assertEqual(True, self.csp._CuttingStock__best_position_mx.all())
        self.assertEqual(True, self.csp._CuttingStock__best_cost_mx.all())
        self.assertEqual(False, self.csp._CuttingStock__global_best_position.any())
        self.assertEqual(self.csp._CuttingStock__count, self.csp._CuttingStock__global_best_cost)

    def test_calculate_cost_and_update_best_positions(self):
        self.csp.initialize()
        self.csp._CuttingStock__calculate_cost_amd_update_best_positions()
        self.assertEqual(True, self.csp._CuttingStock__best_cost_mx.all())
        self.assertEqual(True, self.csp._CuttingStock__best_position_mx.all())

    def test__select_global_best(self):
        self.csp.initialize()
        self.csp._CuttingStock__calculate_cost_amd_update_best_positions()
        self.csp._CuttingStock__select_global_best()
        self.assertEqual(True, self.csp._CuttingStock__global_best_cost > 0)
        self.assertEqual(True, self.csp._CuttingStock__global_best_position.all())

    def test_calculate_new_velocities(self):
        self.csp.initialize()
        self.csp._CuttingStock__calculate_cost_amd_update_best_positions()
        self.csp._CuttingStock__select_global_best()
        old_velocity_mx = self.csp._CuttingStock__velocity_mx
        self.csp._CuttingStock__calculate_new_velocities()
        self.assertEqual(False,  np.array_equal(self.csp._CuttingStock__velocity_mx, old_velocity_mx))

    def test_calculate_new_positions(self):
        self.csp.initialize()
        self.csp._CuttingStock__calculate_cost_amd_update_best_positions()
        self.csp._CuttingStock__select_global_best()
        self.csp._CuttingStock__calculate_new_velocities()
        old_position_mx = np.copy(self.csp._CuttingStock__position_mx)
        self.csp._CuttingStock__calculate_new_positions()
        self.assertEqual(False,  np.array_equal(self.csp._CuttingStock__position_mx, old_position_mx))

    def test_find_best_solution(self):
        self.csp.initialize()
        initial_best_cost_mx = np.copy(self.csp._CuttingStock__best_cost_mx)
        initial_best_position_mx = np.copy(self.csp._CuttingStock__best_position_mx)
        self.csp.find_best_solution()
        self.assertEqual(False, np.array_equal(initial_best_cost_mx,
                                               self.csp._CuttingStock__best_cost_mx))
        self.assertEqual(False, np.array_equal(initial_best_position_mx,
                                               self.csp._CuttingStock__best_position_mx))


if __name__ == '__main__':
    unittest.main()
