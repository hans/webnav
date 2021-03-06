"""
Shared model graph definitions and concrete model class definitions.
In that order.
"""

from collections import namedtuple

import numpy as np
from rllab.misc.overrides import overrides
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tensorflow.contrib.layers import utils as layers_utils

from webnav import util
from webnav.agents.oracle import OracleAgent, WebNavMaxOverlapAgent


class DropoutWrapper(tf.nn.rnn_cell.RNNCell):

    """
    A customized RNNCell dropout wrapper that supports tensor `is_training`
    flag.
    """

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 is_training=True, seed=None):
        """Create a cell with added input and/or output dropout.
        Dropout is never used on the state.
        Args:
        cell: an RNNCell, a projection to output_size is added to it.
        input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
        output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
        seed: (optional) integer, the randomness seed.
        Raises:
        TypeError: if cell is not an RNNCell.
        ValueError: if keep_prob is not between 0 and 1.
        """
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
            not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
            not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError("Parameter output_keep_prob must be between 0 and 1: %d"
                              % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed
        self._is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        dropped_inputs = inputs
        if (not isinstance(self._input_keep_prob, float) or
            self._input_keep_prob < 1):
            dropped_inputs = tf.nn.dropout(inputs, self._input_keep_prob,
                                           seed=self._seed)

        inputs = layers_utils.smart_cond(self._is_training,
                lambda: dropped_inputs, lambda: inputs)
        output, new_state = self._cell(inputs, state, scope)

        dropped_output = output
        if (not isinstance(self._output_keep_prob, float) or
            self._output_keep_prob < 1):
            dropped_output = tf.nn.dropout(output, self._output_keep_prob,
                                           seed=self._seed)

        output = layers_utils.smart_cond(self._is_training,
                lambda: dropped_output, lambda: output)
        return output, new_state


def score_beam(state, candidates):
    embedding_dim = state.get_shape().as_list()[-1]
    num_candidates = candidates.get_shape().as_list()[1]

    # Calculate score for each candidate (batched dot product)
    # batch_size * beam_size
    scores = tf.reshape(tf.batch_matmul(candidates,
                                        tf.expand_dims(state, 2)),
                        (-1, num_candidates))

    return scores


def rnn_model(beam_size, num_timesteps, embedding_dim, inputs=None, cells=None,
              keep_prob=1.0, is_training=True, name="model"):
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
        if keep_prob < 1.0:
            # Dropout on RNN cell outputs
            cells = [DropoutWrapper(cell, is_training=is_training,
                                    output_keep_prob=keep_prob)
                     for cell in cells]

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

            # Use dropout-masked top hidden layer to compute scores.
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


def comm_scores(scores, state, agent, is_training=True, name="communication"):
    """
    Predict scores for communcation actions given wrapped environment actions
    and agent hidden state.
    """
    with tf.op_scope([scores, state], name):
        # Batch norm the Q-scores coming from the navigation model.
        scores = layers.batch_norm(scores, scope="scores_bn",
                                   is_training=is_training,
                                   reuse=tf.get_variable_scope()._reuse)

        comm_state = tf.concat(1, (scores, state))
        comm_actions = agent.vocab_size + 1
        comm_scores_t = layers.fully_connected(comm_state, comm_actions,
                                                activation_fn=None,
                                                scope="comm_scores")

        return comm_scores_t


def rnn_comm_model(beam_size, agent, num_timesteps, embedding_dim, inputs=None,
                   cells=None, keep_prob=1.0, is_training=True, name="model"):
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
        if keep_prob < 1.0:
            cells = [DropoutWrapper(cell, is_training=is_training,
                                    output_keep_prob=keep_prob)
                     for cell in cells]

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

            comm_scores_t = comm_scores(scores_t, last_out, agent,
                                        is_training=is_training)

            scores_t = tf.concat(1, [scores_t, comm_scores_t])

            return scores_t, hid_t

        for t in range(num_timesteps):
            if t > 0: tf.get_variable_scope().reuse_variables()

            scores_t, hid_vals = step(t, hid_vals, inputs[t])
            scores.append(scores_t)

        outputs = (scores,)
        return rnn_inputs, outputs


def q_learn(scores, num_timesteps, sarsa=False, gamma=0.99):
    """
    Build a graph to train the given scoring function by Q-learning.

    Supports variable-length rollouts.

    Args:
        scores: `batch_size * n_actions` tensor of scores
        num_timesteps: Maximum length of rollout
        sarsa: If `True`, use actual actions taken rather than max-scoring
            actions to compute backups
        gamma: Scalar discount factor

    Returns:
        inputs: `(actions, rewards, masks)` placeholders
        outputs: `(all_losses, loss)`
    """

    # Per-timestep non-discounted rewards
    rewards = [tf.placeholder(tf.float32, (None,), name="rewards_%i" % t)
               for t in range(num_timesteps)]
    # Mask learning for those examples which stop early
    masks = [tf.placeholder(tf.float32, (None,), name="mask_%i" % t)
             for t in range(num_timesteps)]

    # By default, actions are just argmax; this can/should be overridden
    actions = [tf.to_int32(tf.argmax(scores_t, 1), name="actions_%i" % t)
               for t, scores_t in enumerate(scores)]

    # metadata
    batch_size = tf.shape(scores[0])[0]
    n_actions = tf.shape(scores[0])[1]

    # Q(s, a) for states visited, actions taken
    # easiest to do this lookup as Gather on a flattened scores array
    q_a_pred = [tf.gather(tf.reshape(scores_t, (-1,)),
                          actions_t + tf.range(batch_size) * n_actions)
                for scores_t, actions_t in zip(scores, actions)]
    # max_a' Q(s, a') for states visited
    q_max_pred = [tf.reduce_max(scores_t, 1) for scores_t in scores]

    q_targets = []
    for t in range(num_timesteps):
        target = rewards[t]
        if t < num_timesteps - 1:
            # Bootstrap with Q-learning or SARSA update
            backup = q_a_pred[t + 1] if sarsa else q_max_pred[t + 1]
            target += gamma * backup

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

    inputs = (actions, rewards, masks)
    outputs = (losses, loss)
    return inputs, outputs


class Model(object):
    def __init__(self, beam_size, environment, path_length, embedding_dim,
                 keep_prob=1.0, gamma=0.99):
        self.beam_size = beam_size
        self.path_length = path_length
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.gamma = gamma
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

        self.env = environment

    @property
    def all_feeds(self):
        return []

    @property
    def all_fetches(self):
        return []

    @classmethod
    def build(cls, args, env):
        """
        Build a navigation model instance.

        Args:
            args: CLI args
            env: Representative environment instance
        """
        return cls(args.beam_size, env, args.path_length,
                   env.embedding_dim, gamma=args.gamma,
                   keep_prob=args.rnn_keep_prob,
                   sarsa=args.algorithm == "sarsa")

    def step(self, t, observations, masks_t, is_training):
        """
        Compute Q(s, *) for a batch of states.

        Args:
            t: timestep integer
            observations: List of env observations
            masks_t: Training cost masks for the current timestep
        """
        raise NotImplementedError

    def get_losses(self, actions, rewards, masks):
        """
        Compute Q-learning loss after a rollout has completed.

        Args:
            actions: Sequence of `batch-size` length action indices
            rewards: Sequence of `batch_size`-length per-timestep reward arrays
            masks: Sequence of `batch_size` loss masks

        Returns:
            losses: Sequence of masked loss scalars
        """
        raise NotImplementedError


class QNavigatorModel(Model):

    """
    A Q-learning web navigator (no communication).
    """

    def __init__(self, *args, **kwargs):
        sarsa = kwargs.pop("sarsa", False)
        super(QNavigatorModel, self).__init__(*args, **kwargs)

        rnn_inputs, rnn_outputs = rnn_model(self.beam_size,
                                            self.path_length,
                                            self.embedding_dim,
                                            is_training=self.is_training,
                                            keep_prob=self.keep_prob)

        self.current_node, self.query, self.candidates = rnn_inputs
        self.scores, = rnn_outputs

        q_inputs, q_outputs = q_learn(self.scores, self.path_length,
                                      gamma=self.gamma, sarsa=sarsa)
        self.actions, self.rewards, self.masks = q_inputs
        self.all_losses, self.loss = q_outputs

        self.sm = None

    @property
    @overrides
    def all_feeds(self):
        return self.current_node + self.candidates + [self.query] + \
                self.actions + self.rewards + self.masks + [self.is_training]

    @property
    @overrides
    def all_fetches(self):
        return self.scores + self.all_losses + [self.loss]

    def _reset_batch(self, batch_size, is_training):
        """
        Allocate reusable
        """
        if hasattr(self, "_d_query") and len(self._d_query) == batch_size:
            return

        self._d_query = np.empty((batch_size, self.embedding_dim))
        self._d_current_nodes = np.empty((batch_size, self.embedding_dim))
        self._d_candidates = np.empty((batch_size, self.beam_size,
                                       self.embedding_dim))
        self._d_is_training = is_training

    @overrides
    def step(self, t, observations, masks, is_training):
        batch_size = len(observations)
        if t == 0:
            self._reset_batch(batch_size, is_training)

        for i, obs_i in enumerate(observations):
            query_i, current_node_i, beam_i = obs_i

            self._d_query[i] = query_i
            self._d_current_nodes[i] = current_node_i
            self._d_candidates[i] = beam_i

        feed = {
            self.current_node[t]: self._d_current_nodes,
            self.candidates[t]: self._d_candidates,
            self.masks[t]: masks,
        }
        if t == 0:
            feed[self.query] = self._d_query
            feed[self.is_training] = self._d_is_training

        # Calculate action scores.
        scores_t = self.sm.run(self.scores[t], feed)
        return scores_t

    def get_losses(self, actions, rewards, masks):
        fetches = self.all_losses
        feeds = {self.rewards[t]: rewards_t
                 for t, rewards_t in enumerate(rewards)}
        feeds.update({self.actions[t]: actions_t
                      for t, actions_t in enumerate(actions)})
        losses = self.sm.run(fetches, feeds)
        return losses


class CommModel(Model):

    def __init__(self, *args, **kwargs):
        super(CommModel, self).__init__(*args, **kwargs)

        self.agent = self.env.b_agent

    @classmethod
    @overrides
    def build(cls, args, env):
        """
        Build a communicative Q-learning model.

        Args:
            args: CLI args
            env: Representative instance of communication environment
        """
        webnav_env = env._env
        return cls(args.beam_size, env, args.path_length,
                   webnav_env.embedding_dim, gamma=args.gamma,
                   keep_prob=args.rnn_keep_prob,
                   sarsa=args.algorithm == "sarsa")


class QCommModel(CommModel):

    def __init__(self, *args, **kwargs):
        sarsa = kwargs.pop("sarsa", False)
        super(QCommModel, self).__init__(*args, **kwargs)

        rnn_inputs, rnn_outputs = rnn_comm_model(self.beam_size, self.agent,
                                                 self.path_length,
                                                 self.embedding_dim,
                                                 is_training=self.is_training,
                                                 keep_prob=self.keep_prob)

        self.current_node, self.query, self.candidates, \
                self.message_sent, self.message_recv = rnn_inputs
        self.scores, = rnn_outputs

        q_inputs, q_outputs = q_learn(self.scores, self.path_length,
                                      gamma=self.gamma, sarsa=sarsa)
        self.actions, self.rewards, self.masks = q_inputs
        self.all_losses, self.loss = q_outputs

        self.sm = None

    @property
    @overrides
    def all_feeds(self):
        return self.current_node + self.candidates + self.message_sent + \
                self.message_recv + [self.query] + self.actions + self.rewards + \
                self.masks + [self.is_training]

    @property
    @overrides
    def all_fetches(self):
        return self.scores + self.all_losses + [self.loss]

    def _reset_batch(self, batch_size, is_training):
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

        self._d_is_training = is_training

    @overrides
    def step(self, t, observations, masks, is_training):
        batch_size = len(observations)
        if t == 0:
            self._reset_batch(batch_size, is_training)

        for i, obs_i in enumerate(observations):
            nav_obs, message_recv, message_sent = obs_i
            query_i, current_node_i, beam_i = nav_obs

            self._d_query[i] = query_i
            self._d_current_nodes[i] = current_node_i
            self._d_candidates[i] = beam_i
            self._d_message_recv[i] = message_recv
            self._d_message_sent[i] = message_sent

        feed = {
            self.current_node[t]: self._d_current_nodes,
            self.candidates[t]: self._d_candidates,
            self.message_sent[t]: self._d_message_sent,
            self.message_recv[t]: self._d_message_recv,
            self.masks[t]: masks,
        }
        if t == 0:
            feed[self.query] = self._d_query
            feed[self.is_training] = self._d_is_training

        # Calculate action scores.
        scores_t = self.sm.run(self.scores[t], feed)
        return scores_t

    def get_losses(self, actions, rewards, masks):
        fetches = self.all_losses
        feeds = {self.rewards[t]: rewards_t
                 for t, rewards_t in enumerate(rewards)}
        feeds.update({self.actions[t]: actions_t
                      for t, actions_t in enumerate(actions)})
        losses = self.sm.run(fetches, feeds)
        return losses


class OracleCommModel(CommModel):

    """
    A greedy model for operating with an oracle agent.
    """

    # FSM states
    QUERY = 0
    SEND = 1
    RECEIVE = 2

    def __init__(self, *args, **kwargs):
        kwargs.pop("sarsa", None)
        super(OracleCommModel, self).__init__(*args, **kwargs)

        assert isinstance(self.agent, OracleAgent)
        self.sm = None

        self.cycle_probability = 0.25 if self.agent.allow_cycle else 0

    def _reset_batch(self, batch_size):
        self._state = self.QUERY
        self._batch_size = batch_size

        # Prepare score return for sending query
        self._send_query = np.zeros((batch_size, self.env.action_space.n))
        self._send_query[:, -1] = 1

        # Prepare score return when we want to utter "which"
        self._query_which = np.zeros((batch_size, self.env.action_space.n))
        which_idx = self.agent._token2idx["which"]
        which_action = self.env._env.action_space.n + which_idx
        self._query_which[:, which_action] = 1

        # Prepare score return when we want to utter "cycle"
        if self.agent.allow_cycle:
            self._query_cycle = np.zeros((batch_size, self.env.action_space.n))
            cycle_idx = self.agent._token2idx["cycle"]
            cycle_action = self.env._env.action_space.n + cycle_idx
            self._query_cycle[:, cycle_action] = 1

    def step(self, t, observations, masks, is_training):
        if t == 0:
            self._reset_batch(len(observations))

        if isinstance(self._state, tuple) and self._state[0] == self.SEND:
            self._state = self._state[1]
            return self._send_query
        elif self._state == self.RECEIVE:
            # Read the response from agent.
            messages = [obs_i[1].nonzero()[0][0] for obs_i in observations]
            action_idxs = np.asarray([int(self.agent.vocab[idx])
                                      for idx in messages])

            scores = np.zeros((self._batch_size, self.env.action_space.n))
            scores[np.arange(self._batch_size), action_idxs] = 1.0

            self._state = self.QUERY
            return scores
        elif self._state == self.QUERY:
            # With some probability, cycle the elements on the beam instead
            do_cycle = np.random.random() < self.cycle_probability

            if do_cycle:
                self._state = (self.SEND, self.QUERY)
                return self._query_cycle
            else:
                self._state = (self.SEND, self.RECEIVE)
                return self._query_which

    def get_losses(self, actions, rewards, masks):
        return 0.0
