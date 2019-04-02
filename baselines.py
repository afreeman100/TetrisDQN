from tetriminos import tetrimino_dict
from tqdm import tqdm
import tetris
import random


class RandomAgent:
    def __init__(self, rows, columns, tetriminos, end_at):
        self.rows = rows
        self.columns = columns
        self.tetriminos = tetriminos
        self.end_at = end_at


    def select_move(self, board, tetrimino):
        r = random.choice(range(len(tetrimino_dict[tetrimino])))
        c = random.randint(0, board.shape[1] - tetrimino_dict[tetrimino][r].shape[1])
        return r, c


    def test(self, n):
        total_score = 0
        for _ in tqdm(range(n)):
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                rotation, column = self.select_move(game.grid, game.next_tetrimino)
                total_score += game.place_tetrimino(rotation, column)
        return total_score / n


class Human:
    def __init__(self, rows, columns, tetriminos, end_at):
        self.rows = rows
        self.columns = columns
        self.tetriminos = tetriminos
        self.end_at = end_at


    def play(self):
        score = 0
        game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
        while not game.game_over:
            for i in tetrimino_dict[game.next_tetrimino]:
                print(i[:][::-1], '\n--------')
            tetris.print_game(game.grid)

            rotation = int(input("Enter a rotation: "))
            column = int(input("Enter a column: "))
            score += game.place_tetrimino(rotation, column)

        tetris.print_game(game.grid)
        print(score)


# Average score from random agent, over 10000 games.
r = RandomAgent(rows=20, columns=10, tetriminos=(0, 1, 2, 3, 4, 5, 6), end_at=1000000)
print(r.test(10000))

# Interactive game, played on command line.
h = Human(20, 10, (0, 1, 2, 3, 4, 5, 6), 10000)
h.play()
