from functools import partial

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc.tensor_utils import compile_function
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product

from webnav.environment import EmbeddingWebNavEnvironment
from webnav.rnn_model import rnn_model, make_cell_zero_state
from webnav.session import PartialRunSessionManager


class RankingRecurrentPolicy(StochasticPolicy, Serializable):

    def __init__(self, name, env):
        Serializable.quick_init(self, locals())
        super(RankingRecurrentPolicy, self).__init__(env)

        self._beam_size = env.beam_size
        assert isinstance(env.observation_space, Box)
        box = env.observation_space
        assert box.shape[0] == self._beam_size + 2
        self.embedding_dim = box.shape[1]

        # Build a template for the RNN graph.
        self._scope = "model"
        self._cells = [tf.nn.rnn_cell.BasicLSTMCell(1024, state_is_tuple=True)]
        model_fn = partial(rnn_model, env.beam_size, env.path_length,
                           env.embedding_dim, cells=self._cells, name=name)
        self._recurrence_template = tf.make_template(self._scope, model_fn)

        # Prepare the single-step graph.
        single_inputs, single_outputs = \
                self._recurrence_template(single_step_graph=True)

        # Prepare single-step function.
        current_nodes, query, candidates, hids_prev = single_inputs
        scores_single, hid_single = single_outputs
        probs_single = tf.nn.softmax(scores_single)
        self.f_step = compile_function(
                [current_nodes, query, candidates] + hids_prev,
                [probs_single] + hid_single)

        self._prev_hidden = None
        self._distribution = RecurrentCategorical(env.action_space.n)

    def get_params(self, **tags):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=self._scope)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        """
        Args:
            obs_var: `batch_size * n_timesteps * obs_dim`
            state_info_vars:
        """
        batch_size = tf.shape(obs_var)[0]
        n_timesteps = tf.shape(obs_var)[1]
        shape = (batch_size, n_timesteps, self._beam_size + 2, self.embedding_dim)
        obs_var = tf.reshape(obs_var, tf.pack(shape))

        # Transpose from (batch_size, n_timesteps, n_embeddings, embedding_dim)
        # to (n_embeddings, n_timesteps, batch_size, embedding_dim)
        obs_var = tf.transpose(obs_var, (2, 1, 0, 3))
        query = obs_var[0, 0, :, :]
        query.set_shape((None, self.embedding_dim))
        current_nodes = obs_var[1, :, :, :]
        current_nodes.set_shape((None, None, self.embedding_dim))

        # Transpose candidates to
        # (n_timesteps, batch_size, beam_size, embedding_dim)
        candidates = tf.transpose(obs_var[2:, :, :, :], (1, 2, 0, 3))
        candidates.set_shape((None, None, self._beam_size, self.embedding_dim))

        inputs = (query, current_nodes, candidates)
        _, outputs = self._recurrence_template(inputs=inputs,
                                               single_step_graph=False)
        scores, = outputs
        probs = tf.pack([tf.nn.softmax(scores_t) for scores_t in scores])
        return dict(prob=probs)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]

        dones = np.asarray(dones)
        batch_size = len(dones)
        self._prev_hiddens = [make_cell_zero_state(cell)
                              for cell in self._cells]

    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.iteritems()}

    @overrides
    def get_actions(self, observations):
        current_node, query, candidates = observations

        # Execute a single step.
        ret = self.f_step(current_node, query, candidates, *self._prev_hiddens)
        probs, self._prev_hiddens = ret[0], ret[1:]

        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        agent_info = {"prob": probs}

        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._distribution
