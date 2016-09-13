from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers


RLModel = namedtuple("RLModel",
                     ["inputs", "scores",
                      "rewards", "masks",
                      "all_losses", "loss"])


def score_beam(state, candidates):
    embedding_dim = state.get_shape().as_list()[-1]
    num_candidates = candidates.get_shape().as_list()[1]

    # Calculate score for each candidate (batched dot product)
    # batch_size * beam_size
    scores = tf.reshape(tf.batch_matmul(candidates,
                                        tf.expand_dims(state, 2)),
                        (-1, num_candidates))

    return scores


def make_cell_state_placeholder(cell, name):
    if isinstance(cell.state_size, int):
        return tf.placeholder(tf.float32, shape=(None, cell.state_size),
                              name=name)
    elif isinstance(cell.state_size, tuple):
        return tuple([tf.placeholder(tf.float32, shape=(None, size_i),
                                     name="%s/cell_%i" % (name, i))
                      for i, size_i in enumerate(cell.state_size)])
    else:
        raise ValueError("Unknown cell state size declaration %s"
                         % cell.state_size)


def make_cell_zero_state(cell, batch_size):
    if isinstance(cell.state_size, int):
        return np.zeros((batch_size, cell.state_size), dtype=np.float32)
    elif isinstance(cell.state_size, tuple):
        return tuple([np.zeros((batch_size, state_i))
                      for state_i in cell.state_size])
    else:
        raise ValueError("Unknown cell state size declaration %s"
                         % cell.state_size)


def rnn_model(beam_size, num_timesteps, embedding_dim, inputs=None, cells=None,
              name="model"):
    with tf.variable_scope(name):
        # Embedding of current articles (pre-computed)
        current_nodes = [tf.placeholder(tf.float32,
                                        shape=(None, embedding_dim),
                                        name="current_node_%i" % t)
                        for t in range(num_timesteps)]
        # Embedding of the query (pre-computed)
        query = tf.placeholder(tf.float32, shape=(None, embedding_dim),
                name="query")
        # Embedding of all candidates on the beam (pre-computed)
        candidates = [tf.placeholder(tf.float32,
                                    shape=(None, beam_size, embedding_dim),
                                    name="candidates_%i" % t)
                    for t in range(num_timesteps)]

        batch_size = tf.shape(current_nodes[0])[0]

        if cells is None:
            cells = [tf.nn.rnn_cell.BasicLSTMCell(1024, state_is_tuple=True)]

        # Run stacked RNN.
        inputs = [tf.concat(1, [current_nodes_t, query])
                  for current_nodes_t in current_nodes]
        hid_vals = [cell.zero_state(batch_size, tf.float32)
                    for cell in cells]
        scores = []

        def step(hid_prev_t, input_t):
            """
            Build a single vertical-unrolled step in the RNN graph.
            """
            inp = input_t
            hid_t = []
            for i, (cell, hid_prev) in enumerate(zip(cells, hid_prev_t)):
                inp, hid_t_i = cell(inp, hid_prev, scope="layer%i" % i)
                hid_t.append(hid_t_i)

            hid_vals.append(hid_t)

            # Use top hidden layer to calculate scores.
            last_out = inp
            if cells[-1].output_size != embedding_dim:
                last_out = layers.fully_connected(last_out,
                        embedding_dim, activation_fn=tf.tanh,
                        scope="state_projection")

            scores_t = score_beam(last_out, candidates[t])

            return scores_t, hid_t

        for t in range(num_timesteps):
            if t > 0: tf.get_variable_scope().reuse_variables()

            scores_t, hid_vals = step(hid_vals, inputs[t])
            scores.append(scores_t)

        inputs = (current_nodes, query, candidates)
        outputs = (scores,)
        return inputs, outputs


def comm_scores(scores, state, agent, name="communication"):
    """
    Predict scores for communcation actions given wrapped environment actions
    and agent hidden state.
    """
    with tf.op_scope([scores, state], name):
        # HACK: Don't use wrapped action scores for now.
        comm_state = state
        comm_actions = agent.vocab_size + 1
        comm_scores_t = layers.fully_connected(comm_state, comm_actions,
                                                activation_fn=None,
                                                scope="comm_scores")
        # HACK: Disable utterances with the numeric tokens
        batch_size = tf.shape(scores)[0]
        comm_scores_t = tf.concat(1, [comm_scores_t[:, :1],
                                      tf.fill(tf.pack((batch_size, agent.vocab_size - 1)), -100.0),
                                      comm_scores_t[:, agent.vocab_size:]])

        return comm_scores_t


def rnn_comm_model(beam_size, agent, num_timesteps, embedding_dim, inputs=None,
                   cells=None, name="model"):
    with tf.variable_scope(name):
        # Embedding of current articles (pre-computed)
        current_nodes = [tf.placeholder(tf.float32,
                                        shape=(None, embedding_dim),
                                        name="current_node_%i" % t)
                        for t in range(num_timesteps)]
        # Embedding of the query (pre-computed)
        query = tf.placeholder(tf.float32, shape=(None, embedding_dim),
                name="query")
        # Embedding of all candidates on the beam (pre-computed)
        candidates = [tf.placeholder(tf.float32,
                                    shape=(None, beam_size, embedding_dim),
                                    name="candidates_%i" % t)
                    for t in range(num_timesteps)]

        # Bag-of-words messages sent/received
        messages_sent = [tf.placeholder(tf.float32, shape=(None, agent.vocab_size),
                                        name="message_sent_%i" % t)
                         for t in range(num_timesteps)]
        messages_recv = [tf.placeholder(tf.float32, shape=(None, agent.vocab_size),
                                        name="message_recv_%i" % t)
                         for t in range(num_timesteps)]

        rnn_inputs = (current_nodes, query, candidates,
                      messages_sent, messages_recv)

        #####################

        batch_size = tf.shape(current_nodes[0])[0]

        if cells is None:
            cells = [tf.nn.rnn_cell.BasicLSTMCell(1024, state_is_tuple=True)]

        # Project messages into embedding space.
        # NB: Will fail for large vocabulary sizes.
        with tf.op_scope(messages_sent + messages_recv, "message_projection"):
            message_embeddings = tf.get_variable(
                    "embeddings", shape=(agent.vocab_size, embedding_dim))
            messages_sent_proj = [tf.matmul(messages_t, message_embeddings)
                                  for messages_t in messages_sent]
            messages_recv_proj = [tf.matmul(messages_t, message_embeddings)
                                  for messages_t in messages_recv]

        # Run stacked RNN.
        inputs = [tf.concat(1, [current_nodes_t, query,
                                message_sent_t, message_recv_t])
                  for current_nodes_t, message_sent_t, message_recv_t
                  in zip(current_nodes, messages_sent_proj, messages_recv_proj)]
        hid_vals = [cell.zero_state(batch_size, tf.float32)
                    for cell in cells]
        scores = []

        def step(t, hid_prev_t, input_t):
            """
            Build a single vertical-unrolled step in the RNN graph.
            """
            inp = input_t
            hid_t = []
            for i, (cell, hid_prev) in enumerate(zip(cells, hid_prev_t)):
                inp, hid_t_i = cell(inp, hid_prev, scope="layer%i" % i)
                hid_t.append(hid_t_i)

            hid_vals.append(hid_t)

            # Use top hidden layer to calculate scores.
            last_out = inp
            if cells[-1].output_size != embedding_dim:
                scoring_state = layers.fully_connected(last_out,
                        embedding_dim, activation_fn=tf.tanh,
                        scope="state_projection")

            scores_t = score_beam(scoring_state, candidates[t])

            # HACK: add position-aware delta based on message
            scores_t += layers.fully_connected(messages_recv_proj[t],
                    beam_size, activation_fn=None,
                    scope="position_aware_score_delta")

            comm_scores_t = comm_scores(scores_t, last_out, agent)

            scores_t = tf.concat(1, [scores_t, comm_scores_t])

            return scores_t, hid_t

        for t in range(num_timesteps):
            if t > 0: tf.get_variable_scope().reuse_variables()

            scores_t, hid_vals = step(t, hid_vals, inputs[t])
            scores.append(scores_t)

        outputs = (scores,)
        return rnn_inputs, outputs

def q_model(inputs, scores, num_timesteps, embedding_dim, gamma=0.99,
            name="model"):
    # Per-timestep non-discounted rewards
    rewards = [tf.placeholder(tf.float32, (None,), name="rewards_%i" % t)
               for t in range(num_timesteps)]
    # Mask learning for those examples which stop early
    masks = [tf.placeholder(tf.float32, (None,), name="mask_%i" % t)
             for t in range(num_timesteps)]

    # Calculate Q targets.
    q_targets = []
    q_a_pred = [tf.reduce_max(scores_t, 1) for scores_t in scores]
    for t in range(num_timesteps):
        target = rewards[t]
        if t < num_timesteps - 1:
            # Bootstrap with max_a Q_{t+1}
            target += gamma * q_a_pred[t + 1]

        q_targets.append(target)

        # Summary: mean and variance of Q targets at this timestep
        targets_mean, targets_var = tf.nn.moments(target, [0])
        tf.scalar_summary("q/%i/target/mean" % t, targets_mean)
        tf.scalar_summary("q/%i/target/var" % t, targets_var)

        # Summary: mean and variance of predicted Q values for optimal
        # actions at this timestep
        q_a_pred_mean, q_a_pred_var = tf.nn.moments(q_a_pred[t], [0])
        tf.scalar_summary("q/%i/pred_max/mean" % t, q_a_pred_mean)
        tf.scalar_summary("q/%i/pred_max/var" % t, q_a_pred_var)

        # Summary: mean and variance of all Q values in batch
        q_pred_mean, q_pred_var = tf.nn.moments(scores[t], [0, 1])
        tf.scalar_summary("q/%i/pred/mean" % t, q_pred_mean)
        tf.scalar_summary("q/%i/pred/var" % t, q_pred_var)

    losses = [tf.reduce_mean(tf.square(mask_t * (q_target_t - q_a_pred_t)))
              for q_target_t, q_a_pred_t, mask_t
              in zip(q_targets, q_a_pred, masks)]
    loss = tf.add_n(losses) / float(len(losses))

    tf.scalar_summary("loss", loss)

    return RLModel(inputs, scores,
                   rewards, masks,
                   losses, loss)


class QCommModel(object):

    def __init__(self, beam_size, environment, path_length,
                 embedding_dim, gamma=0.99):
        self.beam_size = beam_size
        self.path_length = path_length
        self.embedding_dim = embedding_dim

        self.env = environment
        agent = self.env.b_agent

        rnn_inputs, rnn_outputs = rnn_comm_model(beam_size, agent, path_length,
                                                 embedding_dim)

        self.current_node, self.query, self.candidates, \
                self.message_sent, self.message_recv = rnn_inputs
        self.scores, = rnn_outputs

        all_inputs = self.current_node + self.candidates + \
                self.message_sent + self.message_recv + [self.query]
        q_tuple = q_model(all_inputs, self.scores, path_length, gamma)

        self.rewards, self.masks, self.all_losses, self.loss = \
                q_tuple[-4:]

        self.sm = None

    @property
    def all_feeds(self):
        return self.current_node + self.candidates + self.message_sent + \
                self.message_recv + [self.query] + self.rewards + self.masks

    @property
    def all_fetches(self):
        return self.scores + self.all_losses + [self.loss]

    @classmethod
    def build(cls, args, env):
        """
        Build a communicative Q-learning model.

        Args:
            args: CLI args
            env: Representative instance of communication environment
        """
        webnav_env = env._env
        return cls(args.beam_size, env, args.path_length,
                   webnav_env.embedding_dim, args.gamma)

    def _reset_batch(self, batch_size):
        """
        Allocate reusable
        """
        if hasattr(self, "_d_query") and len(self._d_query) == batch_size:
            self._d_message_sent.fill(0.0)
            return

        self._d_query = np.empty((batch_size, self.embedding_dim))
        self._d_current_nodes = np.empty((batch_size, self.embedding_dim))
        self._d_candidates = np.empty((batch_size, self.beam_size,
                                       self.embedding_dim))

        self._d_message_sent = np.zeros((batch_size, self.env.vocab_size))
        self._d_message_recv = np.empty((batch_size, self.env.vocab_size))

    def __call__(self, t, observations, masks):
        """
        Compute Q(s, *) for a batch of states.

        Args:
            t: timestep integer
            observations: List of env observations
            masks: Training cost masks for the current timestep
        """
        batch_size = len(observations)
        if t == 0:
            self._reset_batch(batch_size)

        for i, obs_i in enumerate(observations):
            # TODO integrate message_sent / message_recv
            nav_obs, message = obs_i
            query_i, current_node_i, beam_i = nav_obs

            self._d_query[i] = query_i
            self._d_current_nodes[i] = current_node_i
            self._d_candidates[i] = beam_i
            self._d_message_recv[i] = message

        feed = {
            self.current_node[t]: self._d_current_nodes,
            self.candidates[t]: self._d_candidates,
            self.message_sent[t]: self._d_message_sent,
            self.message_recv[t]: self._d_message_recv,
            self.masks[t]: masks,
        }
        if t == 0:
            feed[self.query] = self._d_query

        # Calculate action scores.
        scores_t = self.sm.run(self.scores[t], feed)
        return scores_t
