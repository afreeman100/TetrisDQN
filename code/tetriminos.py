import numpy as np


# Dictionary containing all 7 tetrominoes. Each entry stores each possible rotation for that tetromino
tetrimino_dict = {
    0: [np.array([[1, 1], [1, 1]])],  # O
    1: [np.array([[1, 1, 1, 1]]),  # I
        np.array([[1], [1], [1], [1]])],
    2: [np.array([[1, 1, 0], [0, 1, 1]]),  # Z
        np.array([[0, 1], [1, 1], [1, 0]])],
    3: [np.array([[0, 1, 1], [1, 1, 0]]),  # S
        np.array([[1, 0], [1, 1], [0, 1]])],
    4: [np.array([[1, 0, 0], [1, 1, 1]]),  # J
        np.array([[1, 1, 1], [0, 0, 1]]),
        np.array([[1, 1], [1, 0], [1, 0]]),
        np.array([[0, 1], [0, 1], [1, 1]])],
    5: [np.array([[0, 0, 1], [1, 1, 1]]),  # L
        np.array([[1, 1, 1], [1, 0, 0]]),
        np.array([[1, 0], [1, 0], [1, 1]]),
        np.array([[1, 1], [0, 1], [0, 1]])],
    6: [np.array([[0, 1, 0], [1, 1, 1]]),  # T
        np.array([[1, 1, 1], [0, 1, 0]]),
        np.array([[1, 0], [1, 1], [1, 0]]),
        np.array([[0, 1], [1, 1], [0, 1]])]
}


def validity_mask(tetrimino, columns):
    """
    Not all combinations of rotation and column are valid for each tetrimino. Given a
    tetrimino and the width of the grid (columns), this returns a vector that can be
    used to mask the output of a NN such that only valid moves are considered.
    """
    validity = np.zeros((4, columns), dtype=bool)
    for i, t in enumerate(tetrimino_dict[tetrimino]):
        l = columns - t.shape[1] + 1
        validity[i, :l] = 1
    return validity.flatten()


def action_to_rc(action, columns):
    """
    NNs select actions a single integer in the range (0, 4 * columns).
    This needs to be converted to a rotation and column, which can then be
    used to perform the desired action in the game.
    """
    r = action // columns
    c = action % columns
    return r, c
