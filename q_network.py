import tetris
import tensorflow as tf
from tqdm import tqdm
from tetriminos import *


class Convolutional:
    def __init__(self, input_size, output_size, n_hidden, a):
        self.output_size = output_size

        rows = 20
        cols = 10
        hidden_nodes = [250]

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(shape=[None, rows * cols + 7], dtype=tf.float32, name='Input')

            state, tetrimino = tf.split(self.inputs, [rows * cols, 7], axis=1)
            state = tf.reshape(state, [-1, rows, cols, 1])

            conv = tf.layers.conv2d(inputs=state, filters=64, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
            conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
            conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

            conv_out = tf.concat((tf.layers.flatten(conv), tetrimino), 1)

            hidden = tf.layers.dense(inputs=conv_out,
                                     units=hidden_nodes[0],
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                     name='Hidden')
            self.q_out = tf.layers.dense(inputs=hidden,
                                         units=output_size,
                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                         name='Q')

            self.next_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')
            self.loss = tf.losses.huber_loss(self.next_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            with tf.name_scope('Adam_Optimiser'):
                trainer = tf.train.AdamOptimizer(learning_rate=a)
                self.update_model = trainer.minimize(self.loss)

            self.init = tf.global_variables_initializer()


class Dueling:
    def __init__(self, input_size, output_size, n_hidden, a):
        self.input_size = input_size
        self.output_size = output_size

        hidden_nodes = [100, 100, 100]

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='Input')

            # Value stream
            value_hidden = tf.layers.dense(inputs=self.inputs,
                                           units=hidden_nodes[0],
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.keras.initializers.he_uniform(),
                                           name='Value_H1')
            for i in range(len(hidden_nodes) - 1):
                value_hidden = tf.layers.dense(inputs=value_hidden,
                                               units=hidden_nodes[i + 1],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               name='Value_H' + str(i + 2))

            # Advantage stream
            advantage_hidden = tf.layers.dense(inputs=self.inputs,
                                               units=hidden_nodes[0],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               name='Adv_H1')
            for i in range(len(hidden_nodes) - 1):
                advantage_hidden = tf.layers.dense(inputs=advantage_hidden,
                                                   units=hidden_nodes[i + 1],
                                                   activation=tf.nn.relu,
                                                   kernel_initializer=tf.keras.initializers.he_uniform(),
                                                   name='Adv_H' + str(i + 2))

            value_out = tf.layers.dense(inputs=value_hidden,
                                        units=1,
                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                        name='Value_Out')

            advantage_out = tf.layers.dense(inputs=advantage_hidden,
                                            units=output_size,
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            name='Adv_Out')

            with tf.name_scope('Q'):
                self.q_out = value_out + tf.subtract(advantage_out, tf.reduce_mean(advantage_out, axis=1, keepdims=True))

            self.next_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')
            self.loss = tf.losses.huber_loss(self.next_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            with tf.name_scope('Adam_Optimiser'):
                trainer = tf.train.AdamOptimizer(learning_rate=a)
                self.update_model = trainer.minimize(self.loss)

            self.init = tf.global_variables_initializer()


class NN:
    def __init__(self, input_size, output_size, n_hidden, a):
        self.input_size = input_size
        self.output_size = output_size

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='Input')

            hidden = tf.layers.dense(inputs=self.inputs,
                                     units=100,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                     name='Hidden_1')

            hidden = tf.layers.dense(inputs=hidden,
                                     units=100,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                     name='Hidden_2')

            hidden = tf.layers.dense(inputs=hidden,
                                     units=100,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                     name='Hidden_3')

            self.q_out = tf.layers.dense(inputs=hidden,
                                         units=output_size,
                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                         name='Q')

            self.next_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')
            self.loss = tf.losses.huber_loss(self.next_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            with tf.name_scope('Adam_Optimiser'):
                trainer = tf.train.AdamOptimizer(learning_rate=a)
                self.update_model = trainer.minimize(self.loss)

            self.init = tf.global_variables_initializer()


class Agent:
    def __init__(self, game_settings, a=0.01, y=0.9, e=0.05, n_hidden=100):

        # Learning hyperparameters
        self.a = a
        self.y = y
        self.e = e

        # Game parameters
        self.rows = game_settings.rows
        self.columns = game_settings.columns
        self.tetriminos = game_settings.tetriminos
        self.end_at = game_settings.end_at

        # Create network for game this size
        input_size = self.rows * self.columns + 7
        output_size = self.columns * 4
        self.net = Convolutional(input_size=input_size, output_size=output_size, n_hidden=n_hidden, a=a)

        self.session = tf.Session(graph=self.net.graph)


    def learning_curve(self, training_intervals=100, interval_interactions=100):

        # Extra element for performance after 0 training
        scores_list = np.zeros(training_intervals + 1)
        devs_list = np.zeros(training_intervals + 1)
        ep_list = np.zeros(training_intervals + 1)

        with self.session as sess:
            sess.run(self.net.init)

            tf.summary.FileWriter('./graphs', sess.graph)

            for episode in tqdm(range(training_intervals)):
                current_interaction = 0

                while current_interaction < interval_interactions:
                    game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
                    while not game.game_over and current_interaction < interval_interactions:
                        current_interaction += 1

                        # Get current state and perform feed forward pass
                        state = [game.state()]
                        q_values = sess.run(self.net.q_out, feed_dict={self.net.inputs: state})

                        # Validity mask for the current tetrimino, then select best action
                        mask = validity_mask(game.next_tetrimino, self.columns)
                        q_masked = np.where(mask, q_values, -np.inf)
                        action = np.argmax(q_masked, 1)[0]

                        # Epsilon greedy action - ensuring that action is valid
                        if np.random.rand() < self.e:
                            action = np.random.choice(
                                [i for i in range(self.net.output_size) if q_masked[0, i] > -np.inf])

                        # convert action to a rotation and column which can be used to update the game
                        rotation, column = action_to_rc(action, self.columns)
                        reward = game.place_tetrimino(rotation, column)

                        # State: s'
                        state_new = [game.state()]
                        mask_new = [validity_mask(game.next_tetrimino, self.columns)]

                        # Max Q(s', a')
                        q_new = sess.run(self.net.q_out, feed_dict={self.net.inputs: state_new})
                        q_new_masked = np.where(mask_new, q_new, -1000)
                        q_max = np.max(q_new_masked)

                        # Update Q values
                        q_target = q_values
                        q_target[0, action] = reward + self.y * q_max

                        # No more reward is possible if game is over
                        if game.game_over:
                            q_target[0, action] = reward

                            # Train network using target and predicted Q values
                        _ = sess.run([self.net.update_model],
                                     feed_dict={self.net.inputs: state, self.net.next_q: q_target})

                # Test current performance of agent and store so that learning curve can be plotted
                av_score, dev = self.test(100)
                scores_list[episode + 1] = av_score
                devs_list[episode + 1] = dev
                ep_list[episode + 1] = episode * interval_interactions

        return ep_list, scores_list, devs_list


    def test(self, num_tests):
        """ Test performance of agent, returning mean score and standard deviation from given number of test games. """
        score_list = np.zeros(num_tests)
        for n in range(num_tests):
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                # Pass state to network - perform best action and add reward to log
                state = [game.state()]
                q_values = self.session.run(self.net.q_out, feed_dict={self.net.inputs: state})
                mask = validity_mask(game.next_tetrimino, self.columns)
                q_masked = np.where(mask, q_values, -np.inf)
                action = np.argmax(q_masked, 1)[0]
                rotation, column = action_to_rc(action, self.columns)
                score_list[n] += game.place_tetrimino(rotation, column)
        return np.mean(score_list), np.std(score_list)


class GameSettings:
    def __init__(self, rows=20, columns=10, tetriminos=(0, 1, 2, 3, 4, 5, 6), end_at=10000000):
        self.rows = rows
        self.columns = columns
        self.tetriminos = tetriminos
        self.end_at = end_at


agent = Agent(game_settings=GameSettings())
agent.learning_curve()
