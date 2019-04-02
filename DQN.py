import tetris
from tqdm import tqdm
from tetriminos import *
from networks import *
from replay import ReplayMemory, PriorityMemory
from array2gif import write_gif


def update_targets(tau):
    values = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_Net')
    targets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target')

    assignments = []
    with tf.variable_scope('Update'):
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
                 e_min=0.1,
                 e_rate=20,
                 tau=0.001,
                 replay_samples=32,
                 replay_size=100000,
                 use_replay=True,
                 use_priority=False,
                 use_double=True,
                 use_summaries=False,
                 save_model=False
                 ):

        self.training_steps = training_steps
        self.a = a
        self.y = y

        # Decay epsilon linearly from e_start to e_min over first e_rate % of interactions
        self.e = e_start
        self.e_min = e_min
        self.e_step = (e_start - e_min) / (training_steps * e_rate / 100)

        # Game parameters
        self.rows = game_settings.rows
        self.columns = game_settings.columns
        self.tetriminos = game_settings.tetriminos
        self.end_at = game_settings.end_at

        self.input_size = self.rows * self.columns + 7
        self.output_size = self.columns * 4

        # Experience replay
        self.use_replay = use_replay
        self.use_priority = use_priority
        self.replay_samples = replay_samples

        memory = PriorityMemory if use_priority else ReplayMemory
        self.replay_memory = memory(size=replay_size, state_size=self.input_size, action_size=self.output_size)

        if use_replay:
            self.populate_memory(10000)

        tf.reset_default_graph()
        self.net = network_architecture(self.rows, self.columns, self.output_size, hidden_nodes, a, 'Q_Net', use_priority)
        self.target = network_architecture(self.rows, self.columns, self.output_size, hidden_nodes, a, 'Target', use_priority)

        self.init = tf.global_variables_initializer()
        self.initial_targets = update_targets(1)

        self.save_model = save_model
        self.saver = tf.train.Saver()
        self.save_directory = './model/tetris.ckpt'

        self.tau = tau
        self.use_double = use_double
        self.updater = update_targets(tau)

        self.writer = None
        self.use_summaries = use_summaries


    def populate_memory(self, n):
        # Ensure replay memory has sufficient samples in it before training begins
        i = 0
        while i < n:
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                # Interact
                state = game.state()
                mask = validity_mask(game.next_tetrimino, self.columns)
                action = np.random.choice([i for i in range(self.output_size) if mask[i] == 1])
                rotation, column = action_to_rc(action, self.columns)
                reward = game.place_tetrimino(rotation, column)
                state_new = game.state()
                mask_new = validity_mask(game.next_tetrimino, self.columns)

                # Save
                if self.use_priority:
                    self.replay_memory.save(state, action, reward, game.game_over, state_new, mask, mask_new, np.max(self.replay_memory.priorities))
                else:
                    self.replay_memory.save(state, action, reward, game.game_over, state_new, mask, mask_new)

                i += 1


    def learning_curve(self, training_intervals, interval_interactions):

        # Extra element for performance after 0 training
        scores_list = np.zeros(training_intervals + 1)
        devs_list = np.zeros(training_intervals + 1)
        ep_list = np.arange(training_intervals + 1) * interval_interactions

        with tf.Session() as sess:
            sess.run(self.init)
            # Ensure target network is initialised to same values as Q network
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
                        state = game.state()
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
                        state_new = game.state()
                        mask_new = validity_mask(game.next_tetrimino, self.columns)

                        # Save this experience
                        if self.use_priority:
                            self.replay_memory.save(state, action, reward, game.game_over, state_new, mask, mask_new, np.max(self.replay_memory.priorities))
                        else:
                            self.replay_memory.save(state, action, reward, game.game_over, state_new, mask, mask_new)

                        # --------- Train ---------

                        # If not using experience replay, perform single update for this experience
                        if not self.use_replay:

                            # Q values for s' according to target network
                            q_new = sess.run(self.target.q_out, feed_dict={self.target.inputs: [state_new]})
                            q_new_masked = np.where(mask_new, q_new, -1000)

                            if self.use_double:
                                # 1) Q network to select best action
                                q = sess.run(self.net.q_out, feed_dict={self.net.inputs: [state_new]})
                                q = np.where(mask_new, q, -1000)
                                a_max = np.argmax(q, 1)
                                # 2) Use target network to determine value of action
                                q_max = q_new[0, a_max]
                            else:
                                # Use target net to select the action and its value
                                q_max = np.max(q_new_masked)

                            # Set target values
                            q_target = q_values
                            if game.game_over:
                                q_target[0, action] = reward
                            else:
                                q_target[0, action] = reward + self.y * q_max

                            # Train network
                            _ = sess.run(self.net.update_model, feed_dict={self.net.inputs: [state], self.net.target_q: q_target})
                            perform_updates(self.updater, sess)

                        else:
                            if self.use_priority:
                                self.priority_replay_train(sess)
                            else:
                                self.replay_train(sess)

                # Test current performance of agent and store so that learning curve can be plotted
                av_score, dev = self.test(sess, 100)
                scores_list[episode + 1] = av_score
                devs_list[episode + 1] = dev

            if self.save_model:
                self.saver.save(sess, self.save_directory)
                print('Saved!')

        return ep_list, scores_list, devs_list


    def replay_train(self, session):
        # Retrieve batch of experiences
        states, actions, rewards, game_overs, new_states, masks, new_masks = self.replay_memory.retrieve(self.replay_samples)

        # Predicted Q values for s and s'
        q_values, summary = session.run([self.net.q_out, self.net.merged], feed_dict={self.net.inputs: states})
        q_new, summary2 = session.run([self.target.q_out, self.target.merged], feed_dict={self.target.inputs: new_states})
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


    def priority_replay_train(self, session):

        # Retrieve indices of experiences to use, and their current importance weights
        states, actions, rewards, game_overs, new_states, masks, new_masks, indices, weights = self.replay_memory.retrieve(self.replay_samples)

        # Predicted Q values for s and s'
        q_values, summary = session.run([self.net.q_out, self.net.merged], feed_dict={self.net.inputs: states})
        q_new, summary2 = session.run([self.target.q_out, self.target.merged], feed_dict={self.target.inputs: new_states})
        q_new_masked = np.where(new_masks, q_new, -1000)

        # Q(s,a) for each experience - used with max(Q(s',a')) to calculate TD error for PER
        td_errs = np.array([q_values[i, action] for i, action in enumerate(actions)])

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

            # Use target values to calculate TD error for REP
            td_errs[i] = np.abs(q_values[i, actions[i]] - td_errs[i])

        # Update replay priorities
        for i in range(self.replay_samples):
            self.replay_memory.priorities[indices[i]] = td_errs[i] + self.replay_memory.e

        weights = np.reshape(weights, [self.replay_samples, 1])

        # Batch train, then update target network
        _ = session.run([self.net.update_model],
                        feed_dict={self.net.inputs: states, self.net.target_q: q_values, self.net.weights: weights})
        perform_updates(self.updater, session)


    def load_and_play(self):
        with tf.Session() as sess:
            sess.run(self.init)
            self.saver.restore(sess, self.save_directory)

            score, dev = self.test(sess, 100)
            print('Average score: ', score)


    def test(self, session, num_tests):
        """ Test performance of agent, returning mean score and standard deviation from given number of test games. """
        score_list = np.zeros(num_tests)
        for n in range(num_tests):
            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                # Pass state to network - perform best action and add reward to log
                state = [game.state()]
                q_values = session.run(self.net.q_out, feed_dict={self.net.inputs: state})
                mask = validity_mask(game.next_tetrimino, self.columns)
                q_masked = np.where(mask, q_values, -1000)
                action = np.argmax(q_masked, 1)[0]
                rotation, column = action_to_rc(action, self.columns)
                score_list[n] += game.place_tetrimino(rotation, column)
        return np.mean(score_list), np.std(score_list)


    def load_and_gif(self):
        with tf.Session() as sess:
            sess.run(self.init)
            self.saver.restore(sess, self.save_directory)

            frames = []


            def add_frame(s):
                frame = np.reshape(s[:self.rows * self.columns], (self.rows, self.columns))
                # Upscale so GIF is larger than 20x10 pixels
                frame = np.kron(frame, np.ones((20, 20)))[::-1]
                # Convert to RGB array
                frame = np.stack((frame, frame, frame)) * 100
                frames.append(frame)


            game = tetris.Game(self.rows, self.columns, self.tetriminos, self.end_at)
            while not game.game_over:
                state = game.state()
                add_frame(state)

                q_values = sess.run(self.net.q_out, feed_dict={self.net.inputs: [state]})
                mask = validity_mask(game.next_tetrimino, self.columns)
                q_masked = np.where(mask, q_values, -1000)
                action = np.argmax(q_masked, 1)[0]
                rotation, column = action_to_rc(action, self.columns)
                game.place_tetrimino(rotation, column)

        add_frame(game.state())
        write_gif(frames, 'gif/game.gif', fps=2)
        print('GIF saved')
