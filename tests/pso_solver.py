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
        self.solver = PsoSolver()
        np.random.seed(42)

    def test_config_data(self):
        self.assertEqual(True, self.solver._PsoSolver__swarm_size == 5)
        self.assertEqual(True, self.solver._PsoSolver__iterations == 10)
        self.assertEqual(True, self.solver._PsoSolver__inertia == 0.5)
        self.assertEqual(True, self.solver._PsoSolver__accel_best == 0.5)
        self.assertEqual(True, self.solver._PsoSolver__accel_global_best == 0.5)

    def test_initialize_solver(self):
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.assertEqual(False, self.solver._PsoSolver__position_mx.any())
        self.assertEqual(False, self.solver._PsoSolver__velocity_mx.any())
        self.assertEqual(False, self.solver._PsoSolver__best_position_mx.any())
        self.assertEqual(self.solver._PsoSolver__count * self.solver._PsoSolver__swarm_size,
                         self.solver._PsoSolver__best_cost_mx.sum())
        self.assertEqual(False, self.solver._PsoSolver__global_best_position.any())
        self.assertEqual(self.solver._PsoSolver__count, self.solver._PsoSolver__global_best_cost)

    def test_generate_initial_position_and_velocity(self):
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        expected_pos_mx = np.array([[32., 19., 19., 19., 36., 32., 22., 36., 32., 34., 34., 19., 22.],
                                    [32., 19., 19., 19., 34., 32.,36., 32., 34., 22., 22., 36., 19.],
                                    [34., 19., 34., 32., 19., 32., 22., 22., 19., 19., 36., 36., 32.],
                                    [19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.],
                                    [34., 19., 22., 34., 19., 19., 19., 36., 32., 32., 36., 22., 32.]])
        np.allclose(expected_pos_mx, self.solver._PsoSolver__position_mx)
        np.allclose(expected_pos_mx, self.solver._PsoSolver__best_position_mx)
        expected_vel_mx = np.array([[11., 6., 3., 4., 2., 7., - 3., 5., - 8., - 7., 0., - 6., - 2.],
                                    [8., 2., - 2., 8., - 2., 0., - 3., - 4., - 6., 2., 0., - 9., - 5.],
                                    [7., - 1.,  0.,  9.,  5., 5., 2., - 7., - 1., - 1., - 8., - 4., - 10.],
                                    [0., 8., 4., 1., - 4., 6., 0., 3., - 4., - 3., - 8., 0., - 2.],
                                    [9., 0., 7., - 3., - 4.,  1., - 5., 1., 3., - 4., - 5., - 4., - 10.]])
        np.allclose(expected_vel_mx, self.solver._PsoSolver__velocity_mx)

    def test_calculate_cost_and_update_best_positions(self):
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        expected_cost_mx = np.array([9., 9., 8., 7., 8.])
        np.allclose(expected_cost_mx, self.solver._PsoSolver__best_cost_mx)
        expected_pos_mx = np.array([[32., 19., 19., 19., 36., 32., 22., 36., 32., 34., 34., 19., 22.],
                                    [32., 19., 19., 19., 34., 32., 36., 32., 34., 22., 22., 36., 19.],
                                    [34., 19., 34., 32., 19., 32., 22., 22., 19., 19., 36., 36., 32.],
                                    [19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.],
                                    [34., 19., 22., 34., 19., 19., 19., 36., 32., 32., 36., 22., 32.]])
        np.allclose(expected_pos_mx, self.solver._PsoSolver__best_position_mx)

    def test_select_global_best(self):
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        self.assertEqual(7., self.solver._PsoSolver__global_best_cost)
        expected_global_best_pos = np.array([19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.])
        np.allclose(expected_global_best_pos, self.solver._PsoSolver__global_best_position)

    def test_calculate_new_velocities(self):
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        expected_vel_mx = [[  1., 8., 6., 3., -0., 4., -2., 1., -8., -9., 1., -3., 4.],
                            [ -0., 6., 3., 5., -2., 1., -6., -2., -8., -0., 5., -10., 3.],
                            [ -2., 5., -1., 1., 7., 3., 1., -0., -0., -0., -4., -8., -4.],
                            [  0., 4., 2., 0., -2., 3., 0., 2., -2., -2., -4., 0., -1.],
                            [ -1., 5., 7., -6., 2., 6., -1., -1., -3., -6., -2., -3., -4.]]
        self.solver._PsoSolver__calculate_new_velocities()
        np.allclose(expected_vel_mx, self.solver._PsoSolver__velocity_mx)

    def test_calculate_new_positions(self):
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        self.solver._PsoSolver__calculate_new_velocities()
        self.solver._PsoSolver__calculate_new_positions()
        expected_pos_mx = np.array([[34.,  32.,  32.,  19.,  36.,  22.,  19.,  36.,  19.,  32.,  19.,  34.,  22., ],
                                    [34.,  22.,  32.,  32.,  19.,  36.,  34.,  32.,  19.,  19.,  22.,  36.,  19., ],
                                    [34.,  32.,  34.,  36.,  19.,  19.,  22.,  32.,  32.,  19.,  22.,  19.,   36., ],
                                    [19.,  32.,  34.,  32.,  22.,  22.,  36.,  34.,  19.,  32.,  19.,  36.,  19., ],
                                    [19.,  34.,  22.,  36.,  19.,  19.,  32.,  32.,  32.,  22.,  19.,  34.,  36., ]])
        np.allclose(expected_pos_mx,  self.solver._PsoSolver__position_mx)

    def test_solve(self):
        self.solver.solve(self.csp)
        expected_best_cost_mx = np.array([8., 7., 8., 7., 7.])        
        np.allclose(expected_best_cost_mx, self.solver._PsoSolver__best_cost_mx)
        expected_best_pos_mx = np.array([[34., 32., 32., 19., 36., 22., 19., 36., 19., 32., 19., 34., 22.],
                                        [22., 36., 22., 34., 32., 19., 36., 19., 32., 19., 34., 32., 19.],
                                        [34., 19., 34., 32., 19., 32., 22., 22., 19., 19., 36., 36., 32.],
                                        [19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.],
                                        [19., 32., 22., 36., 32., 19., 19., 36., 19., 34., 34., 22., 32.]])
        np.allclose(expected_best_pos_mx, self.solver._PsoSolver__best_position_mx)

if __name__ == '__main__':
    unittest.main()
