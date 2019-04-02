import numpy as np


class ReplayMemory:
    def __init__(self, size, state_size, action_size):
        self.count = 0
        self.max_size = size

        self.state = np.zeros([size, state_size])
        self.action = np.zeros([size, 1], dtype=int)
        self.reward = np.zeros([size, 1])
        self.game_over = np.zeros([size, 1], dtype=int)
        self.state_new = np.zeros([size, state_size])
        self.mask = np.zeros([size, action_size], dtype=int)
        self.mask_new = np.zeros([size, action_size], dtype=int)


    def save(self, state, action, reward, game_over, state_new, mask, mask_new):
        index = self.count % self.max_size
        self.count += 1

        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.game_over[index] = game_over
        self.state_new[index] = state_new
        self.mask[index] = mask
        self.mask_new[index] = mask_new


    def retrieve(self, num_samples):
        indices = np.random.randint(min(self.count, self.max_size), size=num_samples)

        state = self.state[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        game_over = self.game_over[indices]
        state_new = self.state_new[indices]
        mask = self.mask[indices]
        mask_new = self.mask_new[indices]

        return state, action, reward, game_over, state_new, mask, mask_new


class PriorityMemory:
    def __init__(self, size, state_size, action_size):
        self.count = 0
        self.max_size = size

        self.state = np.zeros([size, state_size])
        self.action = np.zeros([size, 1], dtype=int)
        self.reward = np.zeros([size, 1])
        self.game_over = np.zeros([size, 1], dtype=int)
        self.state_new = np.zeros([size, state_size])
        self.mask = np.zeros([size, action_size], dtype=int)
        self.mask_new = np.zeros([size, action_size], dtype=int)

        # Prioritisation parameters
        self.priorities = np.zeros(size)
        self.e = 0.01
        self.n = 0.6
        self.b = 0.4
        # Increase b to 1 as training continues
        b_num_steps = 100000
        self.b_step_size = (1 - 0.4) / b_num_steps


    def save(self, state, action, reward, game_over, state_new, mask, mask_new, td_error):

        index = self.count % self.max_size
        self.count += 1

        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.game_over[index] = game_over
        self.state_new[index] = state_new
        self.mask[index] = mask
        self.mask_new[index] = mask_new
        self.priorities[index] = td_error + self.e


    def retrieve(self, num_samples):
        # Number of experiences in memory
        size = min(self.count, self.max_size)

        # Normalised probabilities ([:size] ensures not including unfilled memory)
        p = np.array(self.priorities[:size]) ** self.n
        p = p / np.linalg.norm(p, ord=1)

        # Importance weightings - normalised
        weights = ((1 / size) * (1 / p)) ** self.b
        weights = weights * (1 / np.max(weights))

        # Indices of experiences to use - sampled according to distribution p
        indices = np.arange(size)
        indices = np.random.choice(indices, num_samples, p=p[:size])

        state = self.state[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        game_over = self.game_over[indices]
        state_new = self.state_new[indices]
        mask = self.mask[indices]
        mask_new = self.mask_new[indices]

        return state, action, reward, game_over, state_new, mask, mask_new, indices, weights[indices]
