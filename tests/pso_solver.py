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
        #given
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        EXPECTED_POSITION_MX = np.array([[32., 19., 19., 19., 36., 32., 22., 36., 32., 34., 34., 19., 22.],
                                        [32., 19., 19., 19., 34., 32.,36., 32., 34., 22., 22., 36., 19.],
                                        [34., 19., 34., 32., 19., 32., 22., 22., 19., 19., 36., 36., 32.],
                                        [19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.],
                                        [34., 19., 22., 34., 19., 19., 19., 36., 32., 32., 36., 22., 32.]])
        EXPECTED_VELOCITY_MX = np.array([[11., 6., 3., 4., 2., 7., - 3., 5., - 8., - 7., 0., - 6., - 2.],
                                        [8., 2., - 2., 8., - 2., 0., - 3., - 4., - 6., 2., 0., - 9., - 5.],
                                        [7., - 1.,  0.,  9.,  5., 5., 2., - 7., - 1., - 1., - 8., - 4., - 10.],
                                        [0., 8., 4., 1., - 4., 6., 0., 3., - 4., - 3., - 8., 0., - 2.],
                                        [9., 0., 7., - 3., - 4.,  1., - 5., 1., 3., - 4., - 5., - 4., - 10.]])
        #when
        ACTUAL_POSITION_MX = self.solver._PsoSolver__position_mx
        ACTUAL_BEST_POSITION_MX = self.solver._PsoSolver__best_position_mx
        ACTUAL_VELOCITY_MX = self.solver._PsoSolver__velocity_mx
        #then
        np.allclose(EXPECTED_POSITION_MX, ACTUAL_POSITION_MX)
        np.allclose(EXPECTED_POSITION_MX, ACTUAL_BEST_POSITION_MX)
        np.allclose(EXPECTED_VELOCITY_MX, ACTUAL_VELOCITY_MX)

    def test_calculate_cost_and_update_best_positions(self):
        #given
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        EXPECTED_COST_MX = np.array([9., 9., 8., 7., 8.])
        EXPECTED_POSITION_MX = np.array([[32., 19., 19., 19., 36., 32., 22., 36., 32., 34., 34., 19., 22.],
                                        [32., 19., 19., 19., 34., 32., 36., 32., 34., 22., 22., 36., 19.],
                                        [34., 19., 34., 32., 19., 32., 22., 22., 19., 19., 36., 36., 32.],
                                        [19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.],
                                        [34., 19., 22., 34., 19., 19., 19., 36., 32., 32., 36., 22., 32.]])
        #when
        ACTUAL_COST_MX = self.solver._PsoSolver__best_cost_mx
        ACTUAL_POSITION_MX = self.solver._PsoSolver__best_position_mx
        #then
        np.allclose(EXPECTED_COST_MX, ACTUAL_COST_MX)
        np.allclose(EXPECTED_POSITION_MX, ACTUAL_POSITION_MX)

    def test_select_global_best(self):
        #given
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        EXPECTED_GLOBAL_BEST_COST = 7.0
        EXPECTED_GLOBAL_BEST_POSITION = np.array([19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.])
        #when
        ACTUAL_GLOBAL_BEST_COST = self.solver._PsoSolver__global_best_cost
        ACTUAL_GLOBAL_BEST_POSITION = self.solver._PsoSolver__global_best_position
        #then
        self.assertEqual(EXPECTED_GLOBAL_BEST_COST, ACTUAL_GLOBAL_BEST_COST)
        np.allclose(EXPECTED_GLOBAL_BEST_POSITION, ACTUAL_GLOBAL_BEST_POSITION)

    def test_calculate_new_velocities(self):
        #given
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        self.solver._PsoSolver__calculate_new_velocities()
        EXPECTED_VELOCITY_MX = [[  1., 8., 6., 3., -0., 4., -2., 1., -8., -9., 1., -3., 4.],
                            [ -0., 6., 3., 5., -2., 1., -6., -2., -8., -0., 5., -10., 3.],
                            [ -2., 5., -1., 1., 7., 3., 1., -0., -0., -0., -4., -8., -4.],
                            [  0., 4., 2., 0., -2., 3., 0., 2., -2., -2., -4., 0., -1.],
                            [ -1., 5., 7., -6., 2., 6., -1., -1., -3., -6., -2., -3., -4.]]
        #when
        ACTUAL_VELOCITY_MX = self.solver._PsoSolver__velocity_mx
        #then
        np.allclose(EXPECTED_VELOCITY_MX, ACTUAL_VELOCITY_MX)

    def test_calculate_new_positions(self):
        #given
        self.solver._PsoSolver__initialize_solver(self.csp)
        self.solver._PsoSolver__generate_initial_position_and_velocity()
        self.solver._PsoSolver__calculate_cost_amd_update_best_positions()
        self.solver._PsoSolver__select_global_best()
        self.solver._PsoSolver__calculate_new_velocities()
        self.solver._PsoSolver__calculate_new_positions()
        EXPECTED_POSITION_MX = np.array([[34.,  32.,  32.,  19.,  36.,  22.,  19.,  36.,  19.,  32.,  19.,  34.,  22., ],
                                        [34.,  22.,  32.,  32.,  19.,  36.,  34.,  32.,  19.,  19.,  22.,  36.,  19., ],
                                        [34.,  32.,  34.,  36.,  19.,  19.,  22.,  32.,  32.,  19.,  22.,  19.,   36., ],
                                        [19.,  32.,  34.,  32.,  22.,  22.,  36.,  34.,  19.,  32.,  19.,  36.,  19., ],
                                        [19.,  34.,  22.,  36.,  19.,  19.,  32.,  32.,  32.,  22.,  19.,  34.,  36., ]])
        #when
        ACTUAL_POSITION_MX = self.solver._PsoSolver__position_mx
        #then
        np.allclose(EXPECTED_POSITION_MX, ACTUAL_POSITION_MX)

    def test_solve(self):
        #given
        self.solver.solve(self.csp)
        expected_best_cost_mx = np.array([8., 7., 8., 7., 7.])        
        np.allclose(expected_best_cost_mx, self.solver._PsoSolver__best_cost_mx)
        EXPECTED_BEST_POSITION_MX = \
                                np.array([[34., 32., 32., 19., 36., 22., 19., 36., 19., 32., 19., 34., 22.],
                                        [22., 36., 22., 34., 32., 19., 36., 19., 32., 19., 34., 32., 19.],
                                        [34., 19., 34., 32., 19., 32., 22., 22., 19., 19., 36., 36., 32.],
                                        [19., 34., 32., 22., 32., 34., 22., 32., 19., 19., 36., 19., 36.],
                                        [19., 32., 22., 36., 32., 19., 19., 36., 19., 34., 34., 22., 32.]])
        #when
        ACTUAL_BEST_POSITION_MX = self.solver._PsoSolver__best_position_mx
        #then
        np.allclose(EXPECTED_BEST_POSITION_MX, ACTUAL_BEST_POSITION_MX)

if __name__ == '__main__':
    unittest.main()
