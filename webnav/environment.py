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

    def __init__(self, beam_size, wiki_path, qp_path, path_length,
                 is_training=True, *args, **kwargs):
        super(WebNavEnvironment, self).__init__(*args, **kwargs)

        self._load_dataset(wiki_path, qp_path, is_training)

        # Hack: use a random page as the "STOP" sentinel.
        # Works in expectation. :)
        self._stop_sentinel = np.random.choice(len(self._wiki.f["title"]))

        self.beam_size = beam_size
        self.path_length = path_length
        self.is_training = is_training

        if not is_training:
            self._eval_cursor = 0

        self._action_space = Discrete(self.beam_size + 1)

    def _load_dataset(self, wiki_path, qp_path, is_training):
        self._wiki = wiki.Wiki(wiki_path)

        dataset_name = "train" if is_training else "valid"
        data = qp.QP(qp_path)
        self._all_queries = data.get_queries([dataset_name])[0]
        self._all_paths = data.get_paths([dataset_name])[0]

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
        if self.is_training:
            # Sample random minibatch
            self._qp_ids = np.random.choice(len(self._all_queries), size=batch_size)
        else:
            if self._eval_cursor >= len(self._all_queries):
                self._eval_cursor = 0
            self._qp_ids = np.arange(self._eval_cursor,
                                     min(len(self._all_queries) - 1,
                                         self._eval_cursor + batch_size))
            self._eval_cursor += batch_size

        self._queries, self._paths, self._num_hops = [], [], []
        for idx in self._qp_ids:
            path = self._all_paths[idx][0]
            # Pad short paths with STOP targets.
            pad_length = max(0, self.path_length + 1 - len(path))
            path = path + [self._stop_sentinel] * pad_length

            self._queries.append(self._all_queries[idx])
            self._paths.append(path)
            self._num_hops.append(len(path) - 1)

        self._num_hops = np.array(self._num_hops)
        self._cursors = np.zeros_like(self._num_hops, dtype=np.int32)

        self._prepare_actions()
        return self._observe_batch()

    @property
    def cur_article_ids(self):
        return [path[idx] if idx < len(path) else None
                for path, idx in zip(self._paths, self._cursors)]

    def _get_candidate_beams(self):
        """
        For each example, build a beam of candidate next-page IDs consisting of
        the valid solution and other negatively-sampled candidate links on the
        page.
        """
        candidates, ys = [], []
        for cursor, path in zip(self._cursors, self._paths):
            cur_id = path[cursor]
            # Retrieve gold next-page choice for this example
            try:
                gold_next_id = path[cursor + 1]
            except IndexError:
                # We are at the end of this path and ready to quit. No need to
                # prepare a beam.
                candidates.append(None)
                ys.append(None)
                continue

            ids = self._wiki.get_article_links(cur_id)
            ids = [int(x) for x in ids if x != gold_next_id]

            # Beam must be large enough to hold gold + STOP + a distractor
            assert self.beam_size >= 3
            gold_is_stop = gold_next_id == self._stop_sentinel

            # Number of distractors to sample
            sample_size = self.beam_size - 1 if gold_is_stop \
                    else self.beam_size - 2

            if len(ids) > sample_size:
                ids = random.sample(ids, sample_size)
            if len(ids) < sample_size:
                while True:
                    distractors = set(np.random.choice(len(self._wiki.f["title"]),
                                                       size=sample_size - len(ids),
                                                       replace=False))
                    if not distractors & set(ids):
                        break
                ids += distractors

            # Add the gold page.
            ids = [gold_next_id] + ids
            if not gold_is_stop:
                ids += [self._stop_sentinel]
            random.shuffle(ids)

            assert len(ids) == self.beam_size

            candidates.append(ids)
            ys.append(ids.index(gold_next_id))

        return candidates, ys

    def _prepare_actions(self):
        """
        Compute the available actions for each example in the current batch.
        """
        # Only supports supervised / oracle case right now, where trajectory
        # history always follows the gold path
        beams, gold_actions = self._get_candidate_beams()

        self._beams = np.array(beams)
        self.gold_actions = gold_actions

    def get_page_for_action(self, example_idx, action):
        return self._beams[example_idx, action] \
                if action < self.beam_size else self._stop_sentinel

    def get_page_title(self, page_id):
        if page_id == self._stop_sentinel:
            return "<STOP>"
        return self._wiki.f["title"][page_id]

    def step(self, action):
        observations, dones, rewards = self.step_batch([action])[0]
        return Step(observation=observations[0],
                    done=dones[0],
                    reward=rewards[0])

    def step_batch(self, actions):
        # Only supports oracle case. Just follow the gold path.
        self._cursors += 1

        observations = self._observe_batch()
        dones = self._cursors >= self._num_hops
        rewards = self._reward_batch(actions)

        # Prepare action beam for the following timestep.
        self._prepare_actions()

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
                 path_length, vocab_source=vocab.GloveVocab, *args, **kwargs):
        super(EmbeddingWebNavEnvironment, self).__init__(
                beam_size, wiki_path, qp_path, path_length, *args, **kwargs)

        # self._vocab = vocab_source()
        # self.embedding_dim = self._vocab.n_dim

        self._page_embeddings = wiki_emb.WikiEmb(wiki_emb_path).f["emb"]
        self.embedding_dim = self._page_embeddings[0].size

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
            self._just_reset = False

        current_page_embeddings = self._page_embeddings[self.cur_article_ids]
        beam_embeddings = self._page_embeddings[self._beams]

        return self._query_embeddings, current_page_embeddings, \
                beam_embeddings

    def _reward_batch(self, actions):
        # Ignore in supervised implementation for now.
        pass
