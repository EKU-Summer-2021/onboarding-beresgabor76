'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.polynomial import Polynomial
from src.cutting_stock import CuttingStock

__all__ = [
    'Polynomial',
    'CuttingStock'
]
