from distutils.core import setup, Extension

module1 = Extension('sudoku',
                    sources = ['sudoku.cpp'])

setup (name = 'SudokuSolver',
       version = '1.0',
       description = 'This is a sudoku package',
       ext_modules = [module1])
