from functools import partial
import random

import numpy as np
from rllab.envs.base import Env, Step
from rllab.misc import logger
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product

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

        navigator_cls = web_graph.OracleNavigator if oracle \
                else web_graph.Navigator
        self._navigator = navigator_cls(self._graph, self.beam_size,
                                        self.path_length)

        self._action_space = Discrete(self.beam_size)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def cur_article_id(self):
        return self._navigator.cur_article_id

    @property
    def gold_action(self):
        return self._navigator.gold_action

    @property
    def gold_path_length(self):
        return self._navigator.gold_path_length

    def get_article_for_action(self, action):
        return self._navigator.get_article_for_action(action)

    def reset(self):
        self._navigator.reset(self.is_training)
        return self._observe()

    def step(self, action):
        self._navigator.step(action)
        return Step(observation=self._observe(),
                    done=self._navigator.done,
                    reward=self._reward(action))

    def step_wrapped(self, action, timestep):
        # TODO make use of timestep?
        return self.step(action)

    def _observe(self):
        # abstract
        raise NotImplementedError

    def _reward(self, action):
        """
        Compute reward after having taken the action specified by `action`.
        (i.e. class state already reflects the given action)
        """
        # abstract
        raise NotImplementedError


def get_default_wikispeedia_graph():
    if not hasattr(get_default_wikispeedia_graph, "graph"):
        path_length = 10
        graph = web_graph.EmbeddedWikispeediaGraph(
                "./data/wikispeedia/wikispeedia.pkl",
                "./data/wikispeedia/wikispeedia_embeddings.npz",
                path_length)
        get_default_wikispeedia_graph.graph = graph
    return get_default_wikispeedia_graph.graph


class EmbeddingWebNavEnvironment(WebNavEnvironment):

    """
    WebNavEnvironment which uses word embeddings all over the place.
    """

    def __init__(self, beam_size=10, graph=None, goal_reward=10.0,
                 oracle=False, *args, **kwargs):
        if graph is None:
            # Use a default graph so that we can function with rllab.
            graph = get_default_wikispeedia_graph()

        super(EmbeddingWebNavEnvironment, self).__init__(
                beam_size, graph, oracle=oracle, *args, **kwargs)

        self.embedding_dim = self._graph.embedding_dim
        self.goal_reward = goal_reward

        self._just_reset = False
        self._query_embedding = None

    @property
    def observation_space(self):
        # 2 embeddings (query and current page) plus the embeddings of articles
        # on the beam
        return Box(low=-5, high=5,
                   shape=(2 + self.beam_size, self.embedding_dim))

    def reset(self):
        self._just_reset = True
        return super(EmbeddingWebNavEnvironment, self).reset()

    def _observe(self):
        if self._just_reset:
            self._query_embedding = \
                    self._graph.get_query_embeddings([self._navigator._path])[0]
            self._just_reset = False

        current_page_embedding = self._graph.get_article_embeddings(
                [self._navigator.cur_article_id])[0]
        beam_embeddings = self._graph.get_article_embeddings(
                self._navigator._beam)

        return self._query_embedding, current_page_embedding, \
                beam_embeddings

    def reward_for_hop(self, source, target):
        overlap = self._graph.get_relative_word_overlap(source, target)
        return overlap * self.goal_reward

    def _reward(self, idx):
        if self._navigator.success:
            return self.goal_reward
        elif self._navigator.done:
            return 0.0

        return self.reward_for_hop(
                self._navigator.cur_article_id,
                self._navigator.target_id)
