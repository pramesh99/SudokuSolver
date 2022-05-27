# SudokuSolver
Solves sudoku puzzles using a Boolean satisfiability solver

Let us write a Sudoku solver. We want to get as input a Sudoku with some cells filled with values, and we want to get as output a solution, if one exists, and otherwise a notice that the input Sudoku puzzle has no solutions.

The way we go about solving Sudoku is prototypical of a very large number of problems in computer science. In these problems, the solution is attained through a mix of search (we attempt to fill a square with a number and see if it works out), and constraint propagation (if we fill a square with, say, a 1, then there can be no 1's in the same row, column, and 3x3 square).
