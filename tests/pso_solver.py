import unittest
import numpy as np
from src.cutting_stock import CuttingStockProblem
from src.pso_solver import PsoSolver


class TestPsoSolver(unittest.TestCase):

    def setUp(self):
        self.csp = CuttingStockProblem(
            data_url="https://raw.githubusercontent.com/EKU-Summer-2021/" +
                    "intelligent_system_data/main/Intelligent%20System%20Data/CSP/CSP_360.csv",
            stock_size=60)
        self.solver = PsoSolver(self.csp)

    def test_config_data(self):
        self.assertEqual(True, self.solver._PsoSolver__swarm_size > 0)
        self.assertEqual(True, self.solver._PsoSolver__iterations > 0)
        self.assertEqual(True, self.solver._PsoSolver__inertia != 0)
        self.assertEqual(True, self.solver._PsoSolver__accel_best != 0)
        self.assertEqual(True, self.solver._PsoSolver__accel_global_best != 0)

    def test_init(self):
        self.assertEqual(False, self.solver._PsoSolver__position_mx.any())
        self.assertEqual(False, self.solver._PsoSolver__velocity_mx.any())
        self.assertEqual(False, self.solver._PsoSolver__best_position_mx.any())
        self.assertEqual(True, self.solver._PsoSolver__best_cost_mx.all())
        self.assertEqual(False, self.solver._PsoSolver__global_best_position.any())
        self.assertEqual(self.solver._PsoSolver__count, self.solver._PsoSolver__global_best_cost)

    def test_generate_initial_position_and_velocity(self):
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.assertEqual(True, self.solver._PsoSolver__position_mx.all())
        self.assertEqual(True, self.solver._PsoSolver__best_position_mx.all())
        self.assertEqual(True, np.array_equal(self.solver._PsoSolver__velocity_mx,
                                              np.around(self.solver._PsoSolver__velocity_mx)))

    def test_calculate_cost_and_update_best_positions(self):
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.assertEqual(True, self.solver._PsoSolver__best_cost_mx.all())
        self.assertEqual(True, self.solver._PsoSolver__best_position_mx.all())

    def test_select_global_best(self):
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        self.assertEqual(True, self.solver._PsoSolver__global_best_cost > 0)
        self.assertEqual(True, self.solver._PsoSolver__global_best_position.all())

    def test_calculate_new_velocities(self):
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        old_velocity_mx = self.solver._PsoSolver__velocity_mx
        self.solver._PsoSolver__calculate_new_velocities()
        self.assertEqual(False,  np.array_equal(self.solver._PsoSolver__velocity_mx, old_velocity_mx))

    def test_calculate_new_positions(self):
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        self.solver._PsoSolver__calculate_new_velocities()
        old_position_mx = np.copy(self.solver._PsoSolver__position_mx)
        self.solver._PsoSolver__calculate_new_positions()
        self.assertEqual(False,  np.array_equal(self.solver._PsoSolver__position_mx, old_position_mx))

    def test_solve(self):
        initial_best_cost_mx = np.copy(self.solver._PsoSolver__best_cost_mx)
        initial_best_position_mx = np.copy(self.solver._PsoSolver__best_position_mx)
        self.solver.solve()
        self.assertEqual(False, np.array_equal(initial_best_cost_mx,
                                               self.solver._PsoSolver__best_cost_mx))
        self.assertEqual(False, np.array_equal(initial_best_position_mx,
                                               self.solver._PsoSolver__best_position_mx))


if __name__ == '__main__':
    unittest.main()
