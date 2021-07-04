"""
Module for solving Cutting Stock Problem with Swarm Particle Optimization
"""
import logging
import os
import configparser
import numpy as np
import pandas as pd


class CuttingStock:
    """
    Class for solving Cutting Stock Problem with Swarm Particle Optimization
    """

    def __init__(self, data_url, stock_size):
        """
        Constructor for Cutting Stock Problem class
        """
        self.__data_url = data_url
        self.__stock_size = stock_size
        self.__initial_data = None
        self.__count = 0
        self.__position_mx = None
        self.__velocity_mx = None
        self.__best_position_mx = None
        self.__best_cost_mx = None
        self.__global_best_position = None
        self.__global_best_cost = 0
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), '../config', 'cutting_stock.conf'))
        self.__swarm_size = int(config['Parameters']['SwarmSize'])
        self.__iterations = int(config['Parameters']['Iterations'])
        self.__inertia = float(config['Parameters']['Inertia'])
        self.__accel_best = float(config['Parameters']['AccelerationBest'])
        self.__accel_global_best = float(config['Parameters']['AccelerationGlobalBest'])

    def initialize(self):
        """
        Initializes arrays and variables for Swarm Particle Optimization algorithm
        """
        logfile = os.path.join(os.path.dirname(__file__), '../log', 'cutting_stock.log')
        logging.basicConfig(level=logging.INFO,
                            filename=logfile,
                            format='%(asctime)s %(levelname)s {%(module)s} %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        open(logfile, 'w').close()
        self.__initial_data = pd.read_csv(self.__data_url, names=['pieces', 'size'])
        for i in self.__initial_data.index:
            self.__count += self.__initial_data.loc[i, 'pieces']
        self.__position_mx = np.zeros((self.__swarm_size, self.__count))
        self.__velocity_mx = np.zeros((self.__swarm_size, self.__count))
        self.__best_position_mx = np.zeros((self.__swarm_size, self.__count))
        self.__best_cost_mx = np.fromiter(
                                (self.__count for _ in range(self.__swarm_size)), float)
        self.__global_best_position = np.zeros((self.__count,))
        self.__global_best_cost = self.__count
        self.__generate_initial_position_and_velocity()

    def find_best_solution(self):
        """
        Runs Swarm Particle Optimization algorithm
        """
        logging.info("\nSolutions in 0. iteration:\n" +
                     self.__solutions_from_position_mx(self.__position_mx))
        for i in range(self.__iterations):
            self.__calculate_cost_amd_update_best_positions()
            self.__select_global_best()
            logging.info("\nBest solutions in " + str(i) + ". iteration:\n" +
                         self.__solutions_from_position_mx(self.__best_position_mx))
            logging.info("\nBest costs in " + str(i) + ". iteration:\n" +
                         str(self.__best_cost_mx))
            self.__calculate_new_velocities()
            self.__calculate_new_positions()
            logging.info("\nSolutions in " + str(i + 1) + ". iteration:\n" +
                         self.__solutions_from_position_mx(self.__position_mx))
        self.__calculate_cost_amd_update_best_positions()
        self.__select_global_best()
        logging.info("\nBest solutions in last iteration:\n" +
                     self.__solutions_from_position_mx(self.__best_position_mx))
        logging.info("\nBest costs in last iteration:\n" +
                     str(self.__best_cost_mx))

    def __generate_initial_position_and_velocity(self):
        """
        Generates the initial positions of pieces to cut and their velocity
        """
        for particle_ix in range(self.__swarm_size):
            for i in self.__initial_data.index:
                for _ in range(self.__initial_data.loc[i, 'pieces']):
                    k = np.random.randint(self.__count)
                    while self.__position_mx[particle_ix, k] != 0:
                        k = np.random.randint(self.__count)
                    self.__position_mx[particle_ix, k] = self.__initial_data.loc[i, 'size']
                    self.__velocity_mx[particle_ix, k] = np.random.randint(self.__count)
        self.__best_position_mx = np.copy(self.__position_mx)

    def __calculate_cost_amd_update_best_positions(self):
        """
        Calculates cost values (the necessary stock) for each swarm
        and stores the position vector if improves
        """
        i = 0
        for particle in self.__position_mx:
            stock_number = 1
            sum_length = 0
            for piece_length in particle:
                if sum_length + piece_length <= self.__stock_size:
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
        Selects the position vector with the lowest stock number
        and stores it into member variables
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
                    while piece_new_position < 0:
                        piece_new_position += self.__count
                    while piece_new_position >= self.__count:
                        piece_new_position -= self.__count
                    new_position_row = np.insert(
                        self.__position_mx[i], piece_new_position, piece_length)
                    if piece_new_position > piece_old_position:
                        self.__position_mx[i] = np.delete(new_position_row, piece_old_position)
                    else:
                        self.__position_mx[i] = np.delete(new_position_row, piece_old_position + 1)

    def __solutions_from_position_mx(self, position_mx):
        solution_mx = ""
        for position in position_mx:
            solution_mx += (self.__solution_from_position(position) + "\n")
        return solution_mx[0:len(solution_mx)-1]

    def __solution_from_position(self, position):
        sum_length = 0
        stock = []
        solution = []
        for piece_length in position:
            if sum_length + piece_length <= self.__stock_size:
                sum_length += piece_length
                stock.append(piece_length)
            else:
                solution.append(stock)
                sum_length = piece_length
                stock = [piece_length]
        solution.append(stock)
        return str(solution)

    def print_solutions(self):
        """
        Prints out the 10 best solutions for cutting stock problem
        """
        ix = np.argsort(self.__best_cost_mx)
        if len(ix) > 10:
            ix = ix[:10]
        ordered_best_position_mx = self.__best_position_mx[ix]
        print("Best solutions in ascending cost order:")
        print(self.__solutions_from_position_mx(ordered_best_position_mx))
        print("Best costs:")
        print(self.__best_cost_mx[ix])
        print("Global best solution:")
        print(self.__solution_from_position(self.__global_best_position))
        print("Global best cost:")
        print(self.__global_best_cost)

    def save_solutions_to_file(self):
        """
        Saves cutting stock best solutions to csv file
        """
        results_df = pd.DataFrame({'cost': self.__best_cost_mx})
        for i in range(int(np.max(self.__best_cost_mx))):
            results_df[str(i + 1)] = np.zeros(len(self.__best_cost_mx))
        for solution_ix in range(len(self.__best_position_mx)):
            sum_length = 0
            stock = []
            stock_ix = 1
            for piece_length in self.__best_position_mx[solution_ix]:
                if sum_length + piece_length <= self.__stock_size:
                    sum_length += piece_length
                    stock.append(piece_length)
                else:
                    results_df.iloc[solution_ix, stock_ix] = str(stock)
                    stock_ix += 1
                    sum_length = piece_length
                    stock = [piece_length]
            results_df.iloc[solution_ix, stock_ix] = str(stock)
            solution_ix += 1
        csv_file = os.path.join(os.path.dirname(__file__), '../solutions', 'cutting_stock.csv')
        results_df.to_csv(path_or_buf=csv_file)

