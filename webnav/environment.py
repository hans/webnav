import random

import numpy as np
from rllab.envs.base import Env, Step
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete
from stanza.text import vocab

from webnav.ext import qp, wiki, wiki_emb


class WebNavEnvironment(Env):

    """
    Abstract web-navigation environment with beam-search (discrete
    classification) action space. Leaves observation space and reward structure
    abstract.
    """

    # ID of a "dummy page" which represents an invalid choice present on the
    # beam.
    DUMMY_PAGE = -1

    def __init__(self, beam_size, wiki_path, qp_path, *args, **kwargs):
        super(WebNavEnvironment, self).__init__(*args, **kwargs)

        self._load_dataset(wiki_path, qp_path)

        self.beam_size = beam_size
        self._action_space = Discrete(self.beam_size + 1)

    def _load_dataset(self, wiki_path, qp_path):
        self._wiki = wiki.Wiki(wiki_path)

        data = qp.QP(qp_path)
        self._all_queries = data.get_queries(["train"])
        self._all_paths = data.get_queries(["train"])

        assert len(self._all_queries) == len(self._all_paths)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        raise NotImplementedError

    def reset(self):
        return self.reset_batch(self, 1)[0]

    def reset_batch(self, batch_size):
        self._qp_ids = np.random.choice(len(self._queries), size=batch_size)
        self._queries = self._all_queries[self._qp_ids]
        self._paths = [self._all_paths[idx] for idx in self._qp_ids]
        # TODO why is this data nested?
        self._paths = [xs[0] for xs in self._paths]
        self._cursors = np.zeros((batch_size,), dtype=np.int32)

        self._prepare_actions()
        return self._observe_batch()

    @property
    def _cur_article_ids(self):
        return [path[idx] if idx < len(path) else None
                for path, idx in zip(self._paths, self._cursors)]

    def _get_candidate_beams(self):
        """
        For each example, build a beam of candidate next-page IDs consisting of
        the valid solution and other negatively-sampled candidate links on the
        page.
        """
        # Gold next-page choices for each example
        gold_next_ids = [path[idx + 1] if idx < len(path) - 1 else None
                         for path, idx in zip(self._paths, self._cursors)]

        # Get IDs of candidate next pages for each example.
        candidate_strs = self._wiki.f["links"][self._cur_article_ids]

        for candidate_str, gold_next_id in zip(candidate_strs, gold_next_ids):
            ids = candidate_str.strip().split(" ")
            if ids[0] == "":
                assert False, "Shouldn't reach this spot in supervised mode"
            else:
                ids = [int(x) for id in ids if x != gold_next_id]

            sample_size = self.beam_size - 1
            if len(ids) > sample_size:
                ids = random.sample(ids, self.beam_size)
            # Pad to reach the sample size.
            ids = ids + [self.DUMMY_PAGE] * max(0, self.beam_size - len(ids))

            # Include the gold page of course.
            ids = [gold_next_id] + ids

            yield ids

    def _prepare_actions(self):
        """
        Compute the available actions for each example in the current batch.
        """
        # Only supports supervised / oracle case right now, where trajectory
        # history always follows the gold path
        self._beams = np.array(self._get_candidate_beams())

    def step(self, action):
        observations, dones, rewards = self.step_batch([action])[0]
        return Step(observation=observations[0],
                    done=dones[0],
                    reward=rewards[0])

    def step_batch(self, actions):
        # Only supports oracle case. Just follow the gold path.
        self._cursors += 1

        observations = self._observe_batch()
        dones = self._cursors >= self._lengths
        rewards = self._reward_batch(actions)

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

    def __init__(self, beam_size, wiki_path, qp_path, wiki_emb_path,
                 vocab_source=vocab.GloveVocab, *args, **kwargs):
        super(WebNavEnvironment, self).__init__(beam_size, wiki_path, qp_path,
                                                *args, **kwargs)

        self._vocab = vocab_source()
        self.embedding_dim = self._vocab.n_dim

        self._page_embeddings = wiki_emb.WikiEmb(wiki_emb_path)

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
            query_page_ids = [path[-1] for path in self._paths]
            self._query_embeddings = self._page_embeddings[query_page_ids]

        current_page_embeddings = self._page_embeddings[self._cur_article_ids]

        # DEV
        print self._beams.shape, self._page_embeddings.shape
        beam_embeddings = self._page_embeddings[self._beams]
        print beam_embeddings.shape

        return self._query_embeddings, current_page_embeddings, \
                beam_embeddings
