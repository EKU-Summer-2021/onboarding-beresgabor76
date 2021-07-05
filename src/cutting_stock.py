"""
Module for solving Cutting Stock Problem class
"""
import os
from urllib.error import HTTPError

import numpy as np
import pandas as pd


class CuttingStockProblem:
    """
    Class for Cutting Stock Problem representation
    """

    def __init__(self, data_url, stock_size):
        """
        Constructor for Cutting Stock Problem class
        """
        self.stock_size = stock_size
        try:
            self.__initial_data = pd.read_csv(data_url, names=['pieces', 'size'])
        except HTTPError:
            print('The given data URL is not valid!')
            exit(1)
        self.count = 0
        for i in self.__initial_data.index:
            self.count += self.__initial_data.loc[i, 'pieces']
        self.initial_position = np.zeros(self.count)
        ctr = 0
        for i in self.__initial_data.index:
            for j in range(self.__initial_data.loc[i, 'pieces']):
                self.initial_position[ctr + j] = self.__initial_data.loc[i, 'size']
            ctr += self.__initial_data.loc[i, 'pieces']
        if stock_size < np.max(self.initial_position):
            print('There are longer pieces than the given stock size!')
            exit(1)
        self.logfile = os.path.join(os.path.dirname(__file__), '../log', 'cutting_stock.log')

    def solutions_from_position_mx(self, position_mx):
        """
        Creates cutting stock solutions' matrix from Solver data representation
        """
        solution_mx = ""
        for position in position_mx:
            solution_mx += (self.solution_from_position(position) + "\n")
        return solution_mx[0:len(solution_mx)-1]

    def solution_from_position(self, position):
        """
        Creates a cutting stock solution vector from Solver data representation
        """
        sum_length = 0
        stock = []
        solution = []
        for piece_length in position:
            if sum_length + piece_length <= self.stock_size:
                sum_length += piece_length
                stock.append(piece_length)
            else:
                solution.append(stock)
                sum_length = piece_length
                stock = [piece_length]
        solution.append(stock)
        return str(solution)

    def save_solutions_to_file(self, best_position_mx, best_cost_mx, filename):
        """
        Saves best solutions to a csv file
        """
        results_df = pd.DataFrame({'cost': best_cost_mx})
        for i in range(int(np.max(best_cost_mx))):
            results_df[str(i + 1)] = np.zeros(len(best_cost_mx))
        for solution_ix in range(len(best_position_mx)):
            sum_length = 0
            stock = []
            stock_ix = 1
            for piece_length in best_position_mx[solution_ix]:
                if sum_length + piece_length <= self.stock_size:
                    sum_length += piece_length
                    stock.append(piece_length)
                else:
                    results_df.iloc[solution_ix, stock_ix] = str(stock)
                    stock_ix += 1
                    sum_length = piece_length
                    stock = [piece_length]
            results_df.iloc[solution_ix, stock_ix] = str(stock)
            solution_ix += 1
        csv_file = os.path.join(os.path.dirname(__file__), '../solutions', filename)
        results_df.to_csv(path_or_buf=csv_file)
