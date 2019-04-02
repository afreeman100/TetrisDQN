import tetris
from tqdm import tqdm
from tetriminos import *
from networks import *
from replay import ReplayMemory


def update_targets(tau):
    values = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_Net")
    targets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Target")

    assignments = []
    with tf.variable_scope('Assignment'):
        for target, value in zip(targets, values):
            assignments.append(target.assign((value.value() * tau) + ((1 - tau) * target.value())))

    return assignments


def perform_updates(assignments, sess):
    for assignment in assignments:
        sess.run(assignment)


class Agent:
    def __init__(self,
                 game_settings,
                 training_steps,
                 network_architecture=FullyConnected,
                 hidden_nodes=np.array([250, 100]),
                 a=0.00001,
                 y=0.99,
                 e_start=1,
                 e_min=0.05,
                 e_rate=20,
                 tau=0.001,
                 replay_samples=32,
                 replay_size=100000,
                 use_replay=True,
                 use_double=True,
                 use_summaries=False,
                 n_rows=2
                 ):

        self.training_steps = training_steps
        self.a = a
        self.y = y

        # Decay epsilon linearly from e_start to e_min over first n% of interactions
        # Decay epsilon linearly from e_start to e_min over first e_rate % of interactions
        self.e = e_start
        self.e_min = e_min
        self.e_step = (e_start - e_min) / (training_steps * e_rate / 100)

        # Game parameters
        self.rows = game_settings.rows
        self.columns = game_settings.columns
        self.tetriminos = game_settings.tetriminos
        self.end_at = game_settings.end_at

        # Only look at top n non-empty rows
        self.n = n_rows
        self.input_size = self.n * self.columns + 7
        self.output_size = self.columns * 4

        # Set up experience replay
        self.use_replay = use_replay
        self.replay_samples = replay_samples
        self.replay_memory = ReplayMemory(size=replay_size, state_size=self.input_size, action_size=self.output_size)
        if use_replay:
            self.populate_memory(10000)

        tf.reset_default_graph()
        self.net = network_architecture(n_rows, self.columns, self.output_size, hidden_nodes, a, 'Q_Net')
        self.target = network_architecture(n_rows, self.columns, self.output_size, hidden_nodes, a, 'Target')

        self.init = tf.global_variables_initializer()
        self.initial_targets = update_targets(1)

        self.tau = tau
        self.use_double = use_double
        self.updater = update_targets(tau)

        self.writer = None
        self.use_summaries = use_summaries


    def process_state(self, game):
        board = game.board[:self.rows, :]
        non_empty_rows = np.any(board, axis=1)

        arg_non_empty = np.argwhere(non_empty_rows)
        if len(arg_non_empty) == 0:
            top = self.n
        else:
            top = max(np.max(np.argwhere(non_empty_rows)) + 1, self.n)

        bottom = top - self.n
        board_clipped = board[bottom:top, :]

        tetrimino = np.eye(7)[game.next_tetrimino]
        state = np.append(board_clipped.flatten(), tetrimino)
        return state


    def populate_memory(self, n):
        i = 0
        while i < n:
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                # state = game.state()
                state = self.process_state(game)
                mask = validity_mask(game.next_tetrimino, self.columns)
                action = np.random.choice([i for i in range(self.output_size) if mask[i] == 1])
                rotation, column = action_to_rc(action, self.columns)
                reward = game.place_tetrimino(rotation, column)
                # state_new = game.state()
                state_new = self.process_state(game)
                mask_new = validity_mask(game.next_tetrimino, self.columns)
                self.replay_memory.save(state, action, reward, game.game_over, state_new, mask, mask_new)
                i += 1


    def learning_curve(self, training_intervals, interval_interactions):

        # Extra element for performance after 0 training
        scores_list = np.zeros(training_intervals + 1)
        devs_list = np.zeros(training_intervals + 1)
        ep_list = np.zeros(training_intervals + 1)

        with tf.Session() as sess:
            sess.run(self.init)
            perform_updates(self.initial_targets, sess)

            if self.use_summaries:
                self.writer = tf.summary.FileWriter('./graphs', sess.graph)

            for episode in tqdm(range(training_intervals)):
                current_interaction = 0

                while current_interaction < interval_interactions:
                    game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
                    while not game.game_over and current_interaction < interval_interactions:
                        current_interaction += 1

                        # --------- Interact ---------

                        # Get current state and perform feed forward pass
                        # state = game.state()
                        state = self.process_state(game)
                        q_values = sess.run(self.net.q_out, feed_dict={self.net.inputs: [state]})

                        # Validity mask for the current tetrimino, then select best action
                        mask = validity_mask(game.next_tetrimino, self.columns)
                        q_masked = np.where(mask, q_values, -np.inf)
                        action = np.argmax(q_masked, 1)[0]

                        # Epsilon greedy action - ensuring that action is valid
                        if np.random.rand() < self.e:
                            valid_act_args = [i for i in range(self.output_size) if mask[i] == 1]
                            action = np.random.choice(valid_act_args)
                        # Decay e
                        self.e = max(self.e_min, self.e - self.e_step)

                        # convert action to a rotation and column which can be used to update the game
                        rotation, column = action_to_rc(action, self.columns)
                        reward = game.place_tetrimino(rotation, column)

                        # State: s'
                        # state_new = game.state()
                        state_new = self.process_state(game)
                        mask_new = validity_mask(game.next_tetrimino, self.columns)

                        # Save this experience
                        self.replay_memory.save(state, action, reward, game.game_over, state_new, mask, mask_new)

                        # --------- Train ---------
                        self.replay_train(sess)

                # Test current performance of agent and store so that learning curve can be plotted
                av_score, dev = self.test(sess, 100)
                scores_list[episode + 1] = av_score
                devs_list[episode + 1] = dev
                ep_list[episode + 1] = episode * interval_interactions

        return ep_list, scores_list, devs_list


    def replay_train(self, session):

        # Retrieve batch of experiences
        states, actions, rewards, game_overs, new_states, masks, new_masks = self.replay_memory.retrieve(
            self.replay_samples)

        # Predicted Q values for s and s'
        q_values, summary = session.run([self.net.q_out, self.net.merged], feed_dict={self.net.inputs: states})
        q_new, summary2 = session.run([self.target.q_out, self.target.merged],
                                      feed_dict={self.target.inputs: new_states})
        q_new_masked = np.where(new_masks, q_new, -1000)

        if self.use_summaries:
            self.writer.add_summary(summary)
            self.writer.add_summary(summary2)

        if self.use_double:
            # 1) Q network to select best actions
            q = session.run(self.net.q_out, feed_dict={self.net.inputs: new_states})
            q = np.where(new_masks, q, -1000)
            a_max = np.argmax(q, 1)
            # 2) Use target network to determine value of actions
            q_new_max = np.array([q_new_masked[i, action] for i, action in enumerate(a_max)])
        else:
            # Use target net to select both actions and values
            q_new_max = np.max(q_new_masked, 1)

        # Set target Q values
        for i in range(self.replay_samples):
            if game_overs[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.y * q_new_max[i]

        # Batch train, then update target network
        _ = session.run([self.net.update_model], feed_dict={self.net.inputs: states, self.net.target_q: q_values})
        perform_updates(self.updater, session)


    def test(self, session, num_tests):
        """ Test performance of agent, returning mean score and standard deviation from given number of test games. """
        score_list = np.zeros(num_tests)
        for n in range(num_tests):
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                # Pass state to network - perform best action and add reward to log
                # state = [game.state()]
                state = [self.process_state(game)]
                q_values = session.run(self.net.q_out, feed_dict={self.net.inputs: state})
                mask = validity_mask(game.next_tetrimino, self.columns)
                q_masked = np.where(mask, q_values, -1000)
                action = np.argmax(q_masked, 1)[0]
                rotation, column = action_to_rc(action, self.columns)
                score_list[n] += game.place_tetrimino(rotation, column)
        return np.mean(score_list), np.std(score_list)
