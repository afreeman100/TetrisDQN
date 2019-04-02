import tensorflow as tf


class Convolutional:
    def __init__(self, rows, cols, output_size, hidden_nodes, a, name, use_weights=False):
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(shape=[None, rows * cols + 7], dtype=tf.float32, name='Input')

            state, tetrimino = tf.split(self.inputs, [rows * cols, 7], axis=1)
            state = tf.reshape(state, [-1, rows, cols, 1])

            conv = tf.layers.conv2d(inputs=state, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

            # Flatten the output of the conv layer and append the tetrimino to give a 'convolutional representation' of the state
            conv_state = tf.concat((tf.layers.flatten(conv), tetrimino), 1)

            hidden = tf.layers.dense(inputs=conv_state,
                                     units=hidden_nodes[0],
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                     name='Hidden_1')
            for i in range(len(hidden_nodes) - 1):
                hidden = tf.layers.dense(inputs=hidden,
                                         units=hidden_nodes[i + 1],
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                         name='Hidden_' + str(i + 2))

            self.q_out = tf.layers.dense(inputs=hidden,
                                         units=output_size,
                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                         name='Q')

            self.target_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')

            with tf.name_scope('Loss'):
                if use_weights:
                    self.weights = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='IW')
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, weights=self.weights,
                                                     reduction=tf.losses.Reduction.NONE)
                else:
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            trainer = tf.train.AdamOptimizer(learning_rate=a)
            self.update_model = trainer.minimize(self.loss)

            summaries = [
                tf.summary.histogram(name + '_Hidden', hidden),
                tf.summary.histogram(name + '_Q', self.q_out)
            ]
            self.merged = tf.summary.merge(summaries)


class FullyConnected:
    def __init__(self, rows, cols, output_size, hidden_nodes, a, name, use_weights=False):
        input_size = rows * cols + 7

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='Input')

            hidden = tf.layers.dense(inputs=self.inputs,
                                     units=hidden_nodes[0],
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                     name='Hidden_1')
            for i in range(len(hidden_nodes) - 1):
                hidden = tf.layers.dense(inputs=hidden,
                                         units=hidden_nodes[i + 1],
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                         name='Hidden_' + str(i + 2))

            self.q_out = tf.layers.dense(inputs=hidden,
                                         units=output_size,
                                         kernel_initializer=tf.keras.initializers.he_uniform(),
                                         name='Q')

            self.target_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')

            # If using prioritised replay, importance-sampling weights used to correct bias
            with tf.name_scope('Loss'):
                if use_weights:
                    self.weights = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='IW')
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, weights=self.weights,
                                                     reduction=tf.losses.Reduction.NONE)
                else:
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            trainer = tf.train.AdamOptimizer(learning_rate=a)
            self.update_model = trainer.minimize(self.loss)

            # For TensorBoard
            summaries = [
                tf.summary.histogram(name + '_Hidden', hidden),
                tf.summary.histogram(name + '_Q', self.q_out)
            ]
            self.merged = tf.summary.merge(summaries)


class Dueling:
    def __init__(self, rows, cols, output_size, hidden_nodes, a, name, use_weights=False):
        input_size = rows * cols + 7

        with tf.variable_scope(name):
            self.input_size = input_size
            self.output_size = output_size

            self.inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='Input')

            # Value stream
            value_hidden = tf.layers.dense(inputs=self.inputs,
                                           units=hidden_nodes[0],
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.keras.initializers.he_uniform(),
                                           name='Value_Hidden_1')
            for i in range(len(hidden_nodes) - 1):
                value_hidden = tf.layers.dense(inputs=value_hidden,
                                               units=hidden_nodes[i + 1],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               name='Value_Hidden_' + str(i + 2))

            value_out = tf.layers.dense(inputs=value_hidden,
                                        units=1,
                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                        name='Value_Out')

            # Advantage stream
            advantage_hidden = tf.layers.dense(inputs=self.inputs,
                                               units=hidden_nodes[0],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               name='Adv_Hidden_1')
            for i in range(len(hidden_nodes) - 1):
                advantage_hidden = tf.layers.dense(inputs=advantage_hidden,
                                                   units=hidden_nodes[i + 1],
                                                   activation=tf.nn.relu,
                                                   kernel_initializer=tf.keras.initializers.he_uniform(),
                                                   name='Adv_Hidden_' + str(i + 2))
            advantage_out = tf.layers.dense(inputs=advantage_hidden,
                                            units=output_size,
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            name='Adv_Out')

            with tf.name_scope('Q_Values'):
                self.q_out = value_out + tf.subtract(advantage_out, tf.reduce_mean(advantage_out, axis=1, keepdims=True))

            self.target_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')

            with tf.name_scope('Loss'):
                if use_weights:
                    self.weights = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='IW')
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, weights=self.weights, reduction=tf.losses.Reduction.NONE)
                else:
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            trainer = tf.train.AdamOptimizer(learning_rate=a)
            self.update_model = trainer.minimize(self.loss)

            summaries = [
                tf.summary.histogram(name + '_Value', value_out),
                tf.summary.histogram(name + '_Advantage', advantage_out),
                tf.summary.histogram(name + '_Q', self.q_out)
            ]
            self.merged = tf.summary.merge(summaries)


class ConvolutionalDueling:
    def __init__(self, rows, cols, output_size, hidden_nodes, a, name, use_weights=False):
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(shape=[None, rows * cols + 7], dtype=tf.float32, name='Input')

            state, tetrimino = tf.split(self.inputs, [rows * cols, 7], axis=1)
            state = tf.reshape(state, [-1, rows, cols, 1])

            conv = tf.layers.conv2d(inputs=state, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

            conv_state = tf.concat((tf.layers.flatten(conv), tetrimino), 1)

            # Value stream
            value_hidden = tf.layers.dense(inputs=conv_state,
                                           units=hidden_nodes[0],
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.keras.initializers.he_uniform(),
                                           name='Value_Hidden_1')
            for i in range(len(hidden_nodes) - 1):
                value_hidden = tf.layers.dense(inputs=value_hidden,
                                               units=hidden_nodes[i + 1],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               name='Value_Hidden_' + str(i + 2))

            value_out = tf.layers.dense(inputs=value_hidden,
                                        units=1,
                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                        name='Value_Out')

            # Advantage stream
            advantage_hidden = tf.layers.dense(inputs=conv_state,
                                               units=hidden_nodes[0],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.keras.initializers.he_uniform(),
                                               name='Adv_Hidden_1')
            for i in range(len(hidden_nodes) - 1):
                advantage_hidden = tf.layers.dense(inputs=advantage_hidden,
                                                   units=hidden_nodes[i + 1],
                                                   activation=tf.nn.relu,
                                                   kernel_initializer=tf.keras.initializers.he_uniform(),
                                                   name='Adv_Hidden_' + str(i + 2))
            advantage_out = tf.layers.dense(inputs=advantage_hidden,
                                            units=output_size,
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            name='Adv_Out')

            with tf.name_scope('Q_Values'):
                self.q_out = value_out + tf.subtract(advantage_out, tf.reduce_mean(advantage_out, axis=1, keepdims=True))

            self.target_q = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='Q_Targets')

            with tf.name_scope('Loss'):
                if use_weights:
                    self.weights = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='IW')
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, weights=self.weights, reduction=tf.losses.Reduction.NONE)
                else:
                    self.loss = tf.losses.huber_loss(self.target_q, self.q_out, reduction=tf.losses.Reduction.NONE)

            trainer = tf.train.AdamOptimizer(learning_rate=a)
            self.update_model = trainer.minimize(self.loss)

            summaries = [
                tf.summary.histogram(name + '_Value', value_out),
                tf.summary.histogram(name + '_Advantage', advantage_out),
                tf.summary.histogram(name + '_Q', self.q_out)
            ]
            self.merged = tf.summary.merge(summaries)
