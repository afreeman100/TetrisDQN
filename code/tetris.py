import random
import numpy as np
from tetriminos import tetrimino_dict
from numba import jit


def print_game(grid):
    """
    Print grid with some nice formatting to aid human interpretation. Board is displayed with (0,0) in bottom
    left corner, and top 4 rows sliced off, as these are not meant to be visible to player
    """
    for row in grid[:(grid.shape[0] - 4):][::-1]:
        st = ''.join(u"\u2B1B" if r else u"\u2B1C" for r in row)
        print(st)
    print()


@jit(nopython=True)
def drop(grid, tetrimino, column):
    """
    Determine how far down a column a tetromino will fall.
    Slice out a window from the grid, then check if the tetromino will fit into it, i.e no 1s overlap. If there
    is an overlap, return the previous row, since this is where tetromino will stop falling.
    """
    row = grid.shape[0] - 4
    while row >= 0:
        grid_window = grid[row:row + tetrimino.shape[0], column:column + tetrimino.shape[1]]
        blocked = np.any(np.logical_and(grid_window, tetrimino))
        if blocked:
            return row + 1
        row -= 1
    return 0


@jit(nopython=True)
def insert(grid, tetrimino, row, column):
    """
    Insert a tetromino into the grid at the given coordinates. [row][col] will be the bottom left of the
    tetromino.
    """
    window = grid[row:row + tetrimino.shape[0], column:column + tetrimino.shape[1]]
    new_window = np.logical_or(tetrimino, window)
    grid[row:row + tetrimino.shape[0], column:column + tetrimino.shape[1]] = new_window
    return grid



def clear_rows(grid):
    """
    See if any rows are full. If they are, delete them and increment reward.
    """
    is_row_full = np.all(grid, axis=1)
    # No rows to clear
    if not np.any(is_row_full):
        return grid, 0
    # Get number of rows to clear and their indices
    rows_to_clear = np.nonzero(is_row_full)[0]
    rows_cleared = len(rows_to_clear)
    grid = np.delete(grid, rows_to_clear, axis=0)
    # Add empty rows to top of grid to maintain size
    for i in range(rows_cleared):
        grid = np.vstack((grid, np.zeros((1, grid.shape[1]), dtype=int)))
    return grid, rows_cleared


class Game:
    def __init__(self, rows=20, columns=10, tetriminos=(0, 1, 2, 3, 4, 5, 6), end_at=1000000):
        self.game_over = False
        self.grid = np.zeros((rows + 4, columns), dtype=int)
        self.available_tetriminos = tetriminos
        self.next_tetrimino = random.choice(self.available_tetriminos)
        self.lookahead = random.choice(self.available_tetriminos)
        self.end_at = end_at
        self.score_counter = 0


    def state(self):
        grid = self.grid[:(self.grid.shape[0] - 4), :].flatten()
        tetrimino = np.eye(7)[self.next_tetrimino]
        state = np.append(grid, tetrimino)
        return state


    def place_tetrimino(self, rotation, column):
        tetrimino = tetrimino_dict[self.next_tetrimino][rotation]

        # Determine what row the tetrimino will drop to, insert it there, then clear full rows and calculate reward
        row = drop(self.grid, tetrimino, column)
        self.grid = insert(self.grid, tetrimino, row, column)
        self.grid, reward = clear_rows(self.grid)

        self.game_over = np.any(self.grid[self.grid.shape[0] - 4])
        if self.game_over:
            return 0

        # Next tetriminos
        self.next_tetrimino = self.lookahead
        self.lookahead = random.choice(self.available_tetriminos)

        # Prevent games going on too long (or forever in the case of simpler games!)
        self.score_counter += reward
        if self.score_counter >= self.end_at:
            self.game_over = True

        return reward
