from src.file_reader import read_file, print_summary
from src import CuttingStockProblem
from src import PsoSolver

if __name__ == '__main__':
    df = read_file("https://raw.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" +
                   "main/Intelligent%20System%20Data/CSP/CSP_360.csv")
    print_summary(df)

    print('Cutting Stock Problem solution with Particle Swarm Optimization')
    cutting_stock = CuttingStockProblem(data_url="https://raw.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" +
                                          "main/Intelligent%20System%20Data/CSP/CSP_360.csv",
                                        stock_size=60)
    if cutting_stock.is_solvable():
        solver = PsoSolver(cutting_stock)
        solver.solve()
        solver.print_solutions()
        solver.save_solutions('cutting_stock.csv')
