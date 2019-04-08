import matplotlib.pyplot as plt
import numpy as np
import DQN
import os
from networks import *

using_GPU = False

if not using_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def smooth(scores, window_size=5):
    # Applies moving average to scores - smooths learning curves for increased clarity
    length = len(scores)
    smoothed = np.zeros_like(scores)
    for i in range(length):
        window = range(max(i - window_size + 1, 0), min(i + window_size + 1, length))
        smoothed[i] = np.sum(scores[window]) / len(window)
    return smoothed


class GameSettings:
    def __init__(self, rows=20, columns=10, tetriminos=(0, 1, 2, 3, 4, 5, 6), end_at=10000000):
        self.rows = rows
        self.columns = columns
        self.tetriminos = tetriminos
        self.end_at = end_at


def run():
    training_intervals = 100
    interval_interactions = 100
    training_steps = training_intervals * interval_interactions

    game_settings = GameSettings()

    title = 'Convolutional DDQN + PER'
    print(title)
    plt.title(title)

    agent = DQN.Agent(game_settings, training_steps)
    interactions, scores, devs = agent.learning_curve(training_intervals, interval_interactions)
    plt.plot(interactions, smooth(scores), label='Conv. DDQN+PER')

    # plt.fill_between(interactions, scores - devs, scores + devs, alpha=0.3)
    plt.xlabel('Interactions')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


run()
