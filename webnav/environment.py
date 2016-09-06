import random

import numpy as np
from rllab.envs.base import Env, Step
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete

from webnav import web_graph


class WebNavEnvironment(Env):

    """
    Abstract web-navigation environment with beam-search (discrete
    classification) action space. Leaves observation space and reward structure
    abstract.
    """

    def __init__(self, beam_size, graph, is_training=True, oracle=True,
                 *args, **kwargs):
        """
        Args:
            beam_size:
            graph:
            is_training:
            oracle: If True, always follow the gold path regardless of
                provided agent actions.
        """
        super(WebNavEnvironment, self).__init__(*args, **kwargs)

        self._graph = graph

        self.beam_size = beam_size
        self.path_length = self._graph.path_length
        self.is_training = is_training

        navigator_cls = web_graph.OracleBatchNavigator if oracle \
                else web_graph.BatchNavigator
        self._navigator = navigator_cls(self._graph, self.beam_size,
                                        self.path_length)

        self._action_space = Discrete(self.beam_size + 1)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        raise NotImplementedError

    def reset(self):
        return self.reset_batch(self, 1)[0]

    def reset_batch(self, batch_size):
        self._navigator.reset(batch_size, self.is_training)
        return self._observe_batch()

    @property
    def cur_article_ids(self):
        return self._navigator.cur_article_ids

    @property
    def gold_actions(self):
        return self._navigator.gold_actions

    @property
    def gold_path_lengths(self):
        return self._navigator.gold_path_lengths

    def get_article_for_action(self, example_idx, action):
        return self._navigator.get_article_for_action(example_idx, action)

    def step(self, action):
        observations, dones, rewards = self.step_batch([action])[0]
        return Step(observation=observations[0],
                    done=dones[0],
                    reward=rewards[0])

    def step_batch(self, actions):
        rewards = self._reward_batch(actions)

        # Take the step in graph-space!
        self._navigator.step(actions)

        return self._observe_batch(), self._navigator.dones, rewards

    def _observe(self):
        return self._observe_batch()[0]

    def _observe_batch(self):
        # abstract
        raise NotImplementedError

    def _reward(self, action):
        return self._reward_batch([action])[0]

    def _reward_batch(self, actions):
        """
        Compute reward after having taken the actions specified by `actions`.
        (i.e. class state already reflects the given actions)
        """
        # abstract
        raise NotImplementedError


class EmbeddingWebNavEnvironment(WebNavEnvironment):

    """
    WebNavEnvironment which uses word embeddings all over the place.
    """

    def __init__(self, beam_size, graph, *args, **kwargs):
        super(EmbeddingWebNavEnvironment, self).__init__(
                beam_size, graph, *args, **kwargs)

        self.embedding_dim = self._graph.embedding_dim

        self._just_reset = False
        self._query_embeddings = False

    @property
    def observation_space(self):
        # 2 embeddings (query and current page) plus the embeddings of articles
        # on the beam
        return Box(low=-5, high=5,
                   shape=(2 + self.beam_size, self.embedding_dim))

    def reset_batch(self, batch_size):
        self._just_reset = True
        return super(EmbeddingWebNavEnvironment, self).reset_batch(batch_size)

    def _observe_batch(self):
        if self._just_reset:
            self._query_embeddings = self._graph.get_query_embeddings(
                    self._navigator._paths)
            self._just_reset = False

        current_page_embeddings = self._graph.get_article_embeddings(
                self._navigator.cur_article_ids)
        beam_embeddings = self._graph.get_article_embeddings(
                self._navigator._beams)

        return self._query_embeddings, current_page_embeddings, \
                beam_embeddings

    def _reward_batch(self, actions):
        # TODO make smart
        return [0.0] * len(actions)
