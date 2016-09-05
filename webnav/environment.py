import random

import numpy as np
from rllab.envs.base import Env, Step
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete


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

        # Hack: use another random page as a dummy page which will fill up
        # beams which are too small.
        # Again, works in expectation.
        self._dummy_page = np.random.choice(len(self._graph.articles))

        assert self._dummy_page != self._graph.stop_sentinel, \
                "A very improbable event has occurred. Please restart."

        self.beam_size = beam_size
        self.path_length = self._graph.path_length
        self.is_training = is_training
        self.oracle = oracle

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
        self._ids, self._paths, self._lengths = \
                self._graph.sample_paths(batch_size, self.is_training)
        self._cursors = np.zeros_like(self._lengths, dtype=np.int32)

        self._prepare_actions()
        return self._observe_batch()

    @property
    def cur_article_ids(self):
        if self.oracle:
            return [path[idx] if idx < len(path) else None
                    for path, idx in zip(self._paths, self._cursors)]
        else:
            return self._cur_article_ids

    def _get_candidates(self):
        """
        For each example, build a beam of candidate next-page IDs consisting of
        available links on the corresponding current article.

        NB: The candidate lists returned may have a regular pattern, e.g. the
        stop sentinel / filler candidates (for candidate lists which are smaller
        than the beam size) may always be in the same position in the list.
        Make sure to not build models (e.g. ones with output biases) that might
        capitalize on this pattern.

        Returns:
            candidates: List of lists of article IDs, each sublist of length
                `self.beam_size`.
        """
        candidates = []
        for article_id in self.cur_article_ids:
            all_links = self._graph.get_article_links(article_id)

            # Sample `beam_size - 1`; add the STOP sentinel
            links = random.sample(all_links, min(self.beam_size - 1,
                                                 len(all_links)))
            links.append(self.stop_sentinel)

            if len(links) < self.beam_size:
                links.extend([self._dummy_page] * (self.beam_size - len(links)))

            candidates.append(links)

        return candidates

    def _get_oracle_candidates(self):
        """
        For each example, build a beam of candidate next-page IDs consisting of
        the valid solution and other negatively-sampled candidate links on the
        page.

        NB: The candidate lists returned may have a regular pattern, e.g. the
        stop sentinel / filler candidates (for candidate lists which are smaller
        than the beam size) may always be in the same position in the list.
        Make sure to not build models (e.g. ones with output biases) that might
        capitalize on this pattern.

        Returns:
            candidates: List of lists of article IDs, each sublist of length
                `self.beam_size`. Each list is guaranteed to contain 1) the
                gold next page according to the oracle trajectory and 2) the
                stop sentinel. (Note that these two will make up just one
                candidate if the valid next action is to stop.)
            ys: List of integers, one for each sublist of `candidates`.
                Indicates the position of the gold action in the corresponding
                sublist.
        """
        candidates, ys = [], []
        for cursor, path in zip(self._cursors, self._paths):
            cur_id = path[cursor]
            # Retrieve gold next-page choice for this example
            try:
                gold_next_id = path[cursor + 1]
            except IndexError:
                # We are at the end of this path and ready to quit. Prepare a
                # dummy beam that won't have any effect.
                candidates.append([self._dummy_page] * self.beam_size)
                ys.append(0)
                continue

            ids = self._graph.get_article_links(cur_id)
            ids = [int(x) for x in ids if x != gold_next_id]

            # Beam must be large enough to hold gold + STOP + a distractor
            assert self.beam_size >= 3
            gold_is_stop = gold_next_id == self._graph.stop_sentinel

            # Number of distractors to sample
            sample_size = self.beam_size - 1 if gold_is_stop \
                    else self.beam_size - 2

            if len(ids) > sample_size:
                ids = random.sample(ids, sample_size)
            if len(ids) < sample_size:
                ids += [self._dummy_page] * (sample_size - len(ids))

            # Add the gold page.
            ids = [gold_next_id] + ids
            if not gold_is_stop:
                ids += [self._graph.stop_sentinel]
            random.shuffle(ids)

            assert len(ids) == self.beam_size

            candidates.append(ids)
            ys.append(ids.index(gold_next_id))

        return candidates, ys

    def _prepare_actions(self):
        """
        Compute the available actions for each example in the current batch.
        """
        if self.oracle:
            beams, self.gold_actions = self._get_oracle_candidates()
        else:
            beams = self._get_candidates()

        self._beams = np.array(beams)

    def get_page_for_action(self, example_idx, action):
        return self._beams[example_idx, action] \
                if action < self.beam_size else self._graph.stop_sentinel

    def step(self, action):
        observations, dones, rewards = self.step_batch([action])[0]
        return Step(observation=observations[0],
                    done=dones[0],
                    reward=rewards[0])

    def step_batch(self, actions):
        rewards = self._reward_batch(actions)

        if self.oracle:
            self._cursors += 1
        else:
            self._cur_page_ids = self._beams[np.arange(self.beams.shape[0]),
                                             actions]

        # Prepare action beam for the following timestep.
        self._prepare_actions()

        dones = self._cursors >= self._lengths
        observations = self._observe_batch()

        return observations, dones, rewards

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
                    self._paths)
            self._just_reset = False

        current_page_embeddings = self._graph.get_article_embeddings(self.cur_article_ids)
        beam_embeddings = self._graph.get_article_embeddings(self._beams)

        return self._query_embeddings, current_page_embeddings, \
                beam_embeddings

    def _reward_batch(self, actions):
        # Ignore in supervised implementation for now.
        pass
