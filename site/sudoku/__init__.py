from .digit_process import get_matrix
from .sudoku import solve

def image_to_matrix(src):
    return get_matrix(src)

def solve_sudoku(string):
    return solve(string)
