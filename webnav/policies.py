import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import special
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc.tensor_utils import compile_function
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product

from webnav.environment import EmbeddingWebNavEnvironment
from webnav.rnn_model import rnn_model, make_cell_zero_state
from webnav.session import PartialRunSessionManager


class RankingRecurrentPolicy(StochasticPolicy, Serializable):

    def __init__(self, name, env):
        assert isinstance(env, EmbeddingWebNavEnvironment)
        assert isinstance(env.observation_space, Product)
        Serializable.quick_init(self, locals())
        super(RankingRecurrentPolicy, self).__init__(env)

        self._cells = [tf.nn.rnn_cell.BasicLSTMCell(1024, state_is_tuple=True)]

        rollout_graph, single_graph = \
                rnn_model(env.beam_size, env.path_length, env.embedding_dim,
                          cells=self._cells, build_single_step_graph=True,
                          name=name)

        # Prepare unrolled RNN function.
        rollout_inputs, rollout_outputs = rollout_graph
        current_node, query, candidates = rollout_inputs
        scores, = rollout_outputs
        probs = [tf.nn.softmax(scores_t) for scores_t in scores]
        self.f_recurrence = compile_function(
                current_node + [query] + candidates,
                probs)

        # Prepare single-step function.
        single_inputs, single_outputs = single_graph
        current_node, query, candidates, hids_prev = single_inputs
        scores_single, hid_single = single_outputs
        probs_single = tf.nn.softmax(scores_single)
        self.f_step = compile_function(
                [current_node, query, candidate] + hids_prev,
                [probs_single] + hid_single)

        self._prev_hidden = None
        self._distribution = RecurrentCategorical(env.action_space.n)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        """
        Args:
            obs_var: `batch_size * n_timesteps * obs_dim`
            state_info_vars:
        """
        current_node, query, candidates = obs_var
        # TODO: probably won't be structured like this.. hack rllab?
        probs = self.f_recurrence(current_node, query, candidates)
        return dict(probs=probs)

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
        agent_info = {"probs": probs}

        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._distribution
