"""
Module for solving problems with Particle Swarm Optimization
"""
import logging
import os
import configparser
import numpy as np


class PsoSolver:
    """
    Class for solving problems with Particle Swarm Optimization
    """

    def __init__(self):
        """
        Constructor for PSO class, input parameter: problem to solve, logger odject
        """
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), '../config', 'cutting_stock.conf'))
        self.__swarm_size = int(config['Parameters']['SwarmSize'])
        self.__iterations = int(config['Parameters']['Iterations'])
        self.__inertia = float(config['Parameters']['Inertia'])
        self.__accel_best = float(config['Parameters']['AccelerationBest'])
        self.__accel_global_best = float(config['Parameters']['AccelerationGlobalBest'])
        self.__problem = None
        self.__count = 0
        self.__position_mx = None
        self.__velocity_mx = None
        self.__best_position_mx = None
        self.__best_cost_mx = None
        self.__global_best_position = None
        self.__global_best_cost = 0

    def solve(self, problem, random_seed=None):
        """
        Runs Particle Swarm Optimization algorithm
        """
        self.__problem = problem
        self.__initialize_solver()
        if random_seed is not None:
            np.random.seed(random_seed)
        logging.basicConfig(level=logging.INFO,
                            filename=self.__problem.logfile,
                            format='%(asctime)s %(levelname)s {%(module)s} %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')        
        open(self.__problem.logfile, 'w').close()
        self.__generate_initial_position_and_velocity()
        logging.info("\nSolutions in 0. iteration: \n%s",
                     self.__problem.solutions_from_position_mx(self.__position_mx))
        for i in range(self.__iterations):
            self.__calculate_cost_amd_update_best_positions()
            self.__select_global_best()
            logging.info("\nBest solutions in %s. iteration:\n%s", str(i),
                         self.__problem.solutions_from_position_mx(self.__best_position_mx))
            logging.info("\nBest costs in %s. iteration:\n%s", str(i),
                         str(self.__best_cost_mx))
            self.__calculate_new_velocities()
            self.__calculate_new_positions()
            logging.info("\nSolutions in %s. iteration:\n%s", str(i + 1),
                         self.__problem.solutions_from_position_mx(self.__position_mx))
        self.__calculate_cost_amd_update_best_positions()
        self.__select_global_best()
        logging.info("\nBest solutions in last iteration:\n%s",
                     self.__problem.solutions_from_position_mx(self.__best_position_mx))
        logging.info("\nBest costs in last iteration:\n%s",
                     str(self.__best_cost_mx))

    def __initialize_solver(self, problem=None):
        if problem is not None:
            self.__problem = problem
        self.__count = self.__problem.count
        self.__position_mx = np.zeros((self.__swarm_size, self.__count))
        self.__velocity_mx = np.zeros((self.__swarm_size, self.__count))
        self.__best_position_mx = np.zeros((self.__swarm_size, self.__count))
        self.__best_cost_mx = np.fromiter(
            (self.__count for _ in range(self.__swarm_size)), float)
        self.__global_best_position = np.zeros((self.__count,))
        self.__global_best_cost = self.__count

    def __generate_initial_position_and_velocity(self):
        """
        Generates the initial random positions and velocities
        """
        for particle_ix in range(self.__swarm_size):
            for size in self.__problem.initial_position:
                piece_ix = np.random.randint(self.__count)
                while self.__position_mx[particle_ix, piece_ix] != 0:
                    piece_ix = np.random.randint(self.__count)
                self.__position_mx[particle_ix, piece_ix] = size
                new_piece_ix = np.random.randint(self.__count)
                velocity = new_piece_ix - piece_ix
                self.__velocity_mx[particle_ix, piece_ix] = velocity
        self.__best_position_mx = np.copy(self.__position_mx)

    def __calculate_cost_amd_update_best_positions(self):
        """
        Calculates cost values for each swarm and stores the position vector if improves
        """
        i = 0
        for particle in self.__position_mx:
            stock_number = 1
            sum_length = 0
            for piece_length in particle:
                if sum_length + piece_length <= self.__problem.stock_size:
                    sum_length += piece_length
                else:
                    stock_number += 1
                    sum_length = piece_length
            if self.__best_cost_mx[i] > stock_number:
                self.__best_cost_mx[i] = stock_number
                self.__best_position_mx[i] = np.copy(self.__position_mx[i])
            i += 1

    def __select_global_best(self):
        """
        Selects the position vector with the lowest cost value and stores it
        """
        particle_ix = 0
        best_particle_ix = -1
        for cost in self.__best_cost_mx:
            if self.__global_best_cost > cost:
                self.__global_best_cost = cost
                best_particle_ix = particle_ix
            particle_ix += 1
        if best_particle_ix >= 0:
            self.__global_best_position = np.copy(self.__best_position_mx[best_particle_ix])

    def __calculate_new_velocities(self):
        """
        Calculates new velocity matrix for PSO algorithm
        """
        rand1 = np.random.random()
        rand2 = np.random.random()
        self.__velocity_mx = self.__inertia * self.__velocity_mx + \
                             self.__accel_best * rand1 * \
                             (self.__best_position_mx - self.__position_mx) + \
                             self.__accel_global_best * rand2 * \
                             (self.__global_best_position - self.__position_mx)
        self.__velocity_mx = np.around(self.__velocity_mx)

    def __calculate_new_positions(self):
        """
        Calculates new position matrix from previous position and current velocity matrix
        """
        for i in range(self.__swarm_size):
            for j in range(self.__count):
                move = self.__velocity_mx[i, j]
                if move != 0:
                    piece_length = self.__position_mx[i, j]
                    piece_old_position = j
                    piece_new_position = int(j + move)
                    if piece_new_position < 0:
                        piece_new_position = 0
                    if piece_new_position > self.__count - 1:
                        piece_new_position = self.__count - 1
                    new_position_row = np.insert(
                        self.__position_mx[i], piece_new_position, piece_length)
                    if piece_new_position > piece_old_position:
                        self.__position_mx[i] = np.delete(new_position_row, piece_old_position)
                    else:
                        self.__position_mx[i] = np.delete(new_position_row, piece_old_position + 1)

    def print_solutions(self):
        """
        Prints out the 10 best solutions for the given problem
        """
        ix_order = np.argsort(self.__best_cost_mx)
        if len(ix_order) > 10:
            ix_order = ix_order[:10]
        ordered_best_position_mx = self.__best_position_mx[ix_order]
        print("Best solutions in ascending cost order:")
        print(self.__problem.solutions_from_position_mx(ordered_best_position_mx))
        print("Best costs:")
        print(self.__best_cost_mx[ix_order])
        print("Global best solution:")
        print(self.__problem.solution_from_position(self.__global_best_position))
        print("Global best cost:")
        print(self.__global_best_cost)

    def save_solutions(self, filename):
        """
        Saves best solutions to a csv file
        """
        self.__problem.save_solutions_to_file(self.__best_position_mx,
                                              self.__best_cost_mx,
                                              filename)
