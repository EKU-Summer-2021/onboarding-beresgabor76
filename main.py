from src.file_reader import read_file, print_summary
from src import CuttingStock

if __name__ == '__main__':
    df = read_file("https://raw.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" +
                   "main/Intelligent%20System%20Data/CSP/CSP_360.csv")
    print_summary(df)

    cutting_stock = CuttingStock(data_url="https://raw.githubusercontent.com/EKU-Summer-2021/intelligent_system_data/" +
                                          "main/Intelligent%20System%20Data/CSP/CSP_360.csv",
                                 stock_size=60)
    cutting_stock.initialize()
    cutting_stock.find_best_solution()
    cutting_stock.print_solution()
