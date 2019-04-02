from tetriminos import tetrimino_dict
from tqdm import tqdm
import tetris
import random
import numpy as np
import pickle


class Agent:
    def __init__(self, training_steps, a=0.1, y=0.9, e_start=1, e_min=0.05, q_file="q_table.pickle"):
        self.a = a
        self.y = y

        # Decay epsilon linearly from e_start to e_min over first n% of interactions
        self.e = e_start
        self.e_min = e_min
        n = 20
        self.e_step = (e_start - e_min) / (training_steps * n / 100)

        self.q_file = q_file
        self.q_table = {}

        self.rows = 5
        self.columns = 5
        self.tetriminos = (2, 6)
        self.end_at = 1000


    def train(self, interactions):
        current_interaction = 0

        while current_interaction < interactions:
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over and current_interaction < interactions:
                current_interaction += 1

                # State: s
                board = game.grid[:(game.grid.shape[0] - 4):, :].tobytes()
                tetrimino = game.next_tetrimino

                # Choose action: a
                rotation, column, _ = self.best_action(board, tetrimino)

                # Chance of random action: a
                if np.random.rand() < self.e:
                    rotation = random.randint(0, len(tetrimino_dict[tetrimino]) - 1)
                    column = random.randint(0, self.columns - tetrimino_dict[tetrimino][rotation].shape[1])
                    key = (board, tetrimino, rotation, column)
                    # If Q(s,a) has not been initialized for this action
                    if key not in self.q_table:
                        self.q_table[key] = 0
                # Decay e
                self.e = max(self.e_min, self.e - self.e_step)

                # Q(s, a)
                q = self.q_table[board, tetrimino, rotation, column]
                reward = game.place_tetrimino(rotation, column)

                # State: s'
                board_new = game.grid[:(game.grid.shape[0] - 4):][:].tobytes()
                tetrimino_new = game.next_tetrimino

                # Max: Q(s', a')
                _, _, q_new = self.best_action(board_new, tetrimino_new)

                # Update Q table
                self.q_table[board, tetrimino, rotation, column] = q + self.a * (reward + self.y * q_new - q)


    def best_action(self, board, tetrimino):
        """ Given a board state and a tetrimino, return the rotation and column that has the best Q value """
        best_q = -np.inf
        best_rotation = 0
        best_column = 0

        # Initialize Q table for current state if not already done
        for rotation in range(len(tetrimino_dict[tetrimino])):
            for column in range(self.columns - tetrimino_dict[tetrimino][rotation].shape[1] + 1):
                key = (board, tetrimino, rotation, column)
                if key not in self.q_table:
                    self.q_table[key] = 0
                if self.q_table[key] > best_q:
                    best_q = self.q_table[key]
                    best_rotation = rotation
                    best_column = column

        return best_rotation, best_column, best_q


    def learning_curve(self, training_intervals, interval_interactions):
        """
        Train agent in intervals, testing performance after each. Returns mean score and standard deviation after each
        interval, which can be used to plot a learning curve.
        """
        interactions = np.zeros(training_intervals+1)
        scores = np.zeros(training_intervals+1)
        devs = np.zeros(training_intervals+1)

        for interval in tqdm(range(training_intervals+1)):
            score, dev = self.test(100)
            self.train(interval_interactions)
            interactions[interval] = interval * interval_interactions
            scores[interval] = score
            devs[interval] = dev
        return interactions, scores, devs


    def test(self, n):
        """ Play using current policy for given number of games. Return mean score and standard deviation """
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = self.play()

        return np.mean(scores), np.std(scores)


    def play(self, verbose=False):
        """ Play a single game, returning total score """
        score = 0
        game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
        while not game.game_over:
            board = game.grid[:(game.grid.shape[0] - 4):, :].tobytes()
            tetrimino = game.next_tetrimino

            rotation, column, _ = self.best_action(board, tetrimino)
            score += game.place_tetrimino(rotation, column)

            if verbose:
                tetris.print_game(game.grid)

        return score


    def load_q_table(self):
        with open(self.q_file, "rb") as file:
            self.q_table = pickle.load(file)


    def save_q_table(self):
        with open(self.q_file, "wb") as file:
            pickle.dump(self.q_table, file)
