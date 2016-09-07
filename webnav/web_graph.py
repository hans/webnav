"""
Defines a common web graph navigation interface to WikiNav, Wikispeedia, etc.
"""

from collections import namedtuple
import random

import numpy as np


EmbeddedArticle = namedtuple("EmbeddedArticle", ["title", "embedding", "text"])


class EmbeddedWebGraph(object):

    embedding_dim = 128

    def __init__(self, articles, datasets, path_length, stop_sentinel=None):
        self.articles = articles
        self.datasets = {name: (all_paths, np.array(lengths))
                         for name, (all_paths, lengths) in datasets.items()}
        self.path_length = path_length

        assert "train" in self.datasets
        assert "valid" in self.datasets

        # Hack: use a random page as the "STOP" sentinel.
        # Works in expectation. :)
        self.stop_sentinel = stop_sentinel or np.random.choice(len(self.articles))

        self._eval_cursor = 0

    def sample_paths(self, batch_size, is_training=True):
        all_paths, lengths = self.datasets["train" if is_training else "valid"]

        if is_training:
            ids = np.random.choice(len(all_paths), size=batch_size)
        else:
            if self._eval_cursor >= len(all_paths):
                self._eval_cursor = 0
            ids = np.arange(self._eval_cursor,
                            min(len(all_paths) - 1,
                                self._eval_cursor + batch_size))
            self._eval_cursor += batch_size

        paths = [self._prepare_path(all_paths[idx]) for idx in ids]
        return ids, paths, lengths[ids]

    def get_num_paths(self, is_training=True):
        return len(self.datasets["train" if is_training else "valid"][0])

    def get_article_links(self, article_idx):
        raise NotImplementedError

    def get_article_title(self, article_idx):
        if article_idx == self.stop_sentinel:
            return "<STOP>"
        return self.articles[article_idx].title

    def get_relative_word_overlap(self, article1_idx, article2_idx):
        """
        Get the proportion of words in `article1` that are also in `article2`.
        """
        article1 = self.articles[article1_idx]
        article2 = self.articles[article2_idx]

        article1_types = set(article1.text)
        article2_types = set(article2.text)
        return len(article1_types & article2_types) / float(len(article1_types))

    def get_query_embeddings(self, path_ids):
        raise NotImplementedError

    def get_article_embeddings(self, article_ids):
        raise NotImplementedError

    def _prepare_path(self, path):
        raise NotImplementedError


class EmbeddedWikiNavGraph(EmbeddedWebGraph):

    embedding_dim = 500

    class ArticlesDict(object):
        def __init__(self, wiki, wiki_emb):
            self.wiki = wiki
            self.wiki_emb = wiki_emb

        def __getitem__(self, idx):
            embedding = self.wiki_emb.get_article_embedding(idx) \
                    if self.wiki_emb else None
            return EmbeddedArticle(
                    self.wiki.get_article_title(idx), embedding,
                    self.wiki.get_article_text(idx))

        def __len__(self):
            return len(self.wiki.f["title"])

    def __init__(self, wiki_path, qp_path, wiki_emb_path, path_length):
        from webnav.ext import qp, wiki, wiki_emb

        wiki = wiki.Wiki(wiki_path)
#        wiki_emb = wiki_emb.WikiEmb(wiki_emb_path)

        self._wiki = wiki
        self._article_embeddings = np.random.random((len(self._wiki.f["title"]), 20)) * 2 - 1 # wiki_emb.f["emb"]
        articles = self.ArticlesDict(wiki, None)#wiki_emb)

        data = qp.QP(qp_path)
        paths_train, paths_val = data.get_paths(["train", "valid"])

        datasets = {
            "train": (paths_train, [len(path[0]) for path in paths_train]),
            "valid": (paths_valid, [len(path[0]) for path in paths_valid]),
        }

        super(EmbeddedWikiNavGraph, self).__init__(articles, datasets,
                                                   path_length)

    def get_article_links(self, article_idx):
        return self._wiki.get_article_links(article_idx)

    def get_query_embeddings(self, paths):
        """
        Fetch representations for a batch of paths.
        """

        # Get the last non-STOP page in each corresponding path.
        last_pages = [[idx for idx in path if idx != self.stop_sentinel][-1]
                      for path in paths]
        return self._article_embeddings[last_pages]

    def get_article_embeddings(self, article_ids):
        return self._article_embeddings[article_ids]

    def _prepare_path(self, path):
        path = path[0]
        # Pad short paths with STOP targets.
        pad_length = max(0, self.path_length + 1 - len(path))
        path = path + [self.stop_sentinel] * pad_length
        return path


class EmbeddedWikispeediaGraph(EmbeddedWebGraph):

    def __init__(self, data_path, emb_path, path_length):
        try:
            import cPickle as pickle
        except: import pickle

        with open(data_path, "rb") as data_f:
            data = pickle.load(data_f)
        self._data = data

        self.embeddings = embeddings = np.load(emb_path)["arr_0"]
        self.embedding_dim = embeddings.shape[1]

        articles = [EmbeddedArticle(
                        article["name"], embeddings[i],
                        set(token.lower() for token in article["lead_tokens"]))
                    for i, article in enumerate(data["articles"])]

        # Use a random article as a stop sentinel.
        # TODO: Probably better to make a dedicated sentinel here, since this
        # graph is relatively small
        stop_sentinel = np.random.choice(len(articles))

        datasets = {}
        for dataset_name, dataset in data["paths"].iteritems():
            paths, original_lengths, n_skipped = [], [], 0
            for path in dataset:
                if len(path["articles"]) > path_length - 1:
                    n_skipped += 1
                    continue

                # Pad with STOP sentinel (every path gets at least one)
                pad_length = max(0, path_length + 1 - len(path["articles"]))
                original_length = len(path["articles"]) + 1
                path = path["articles"] + [stop_sentinel] * pad_length

                paths.append(path)
                original_lengths.append(original_length)

            print "%s set: skipped %i of %i paths due to length limit" \
                    % (dataset_name, n_skipped, len(dataset))
            datasets[dataset_name] = (paths, np.array(original_lengths))

        super(EmbeddedWikispeediaGraph, self).__init__(articles, datasets,
                                                       path_length,
                                                       stop_sentinel=stop_sentinel)

    def get_article_links(self, article_idx):
        return self._data["links"].get(article_idx, [self.stop_sentinel])

    def get_query_embeddings(self, paths):
        # Get the last non-STOP page in each corresponding path.
        last_pages = [[idx for idx in path if idx != self.stop_sentinel][-1]
                      for path in paths]
        return self.get_article_embeddings(last_pages)

    def get_article_embeddings(self, article_ids):
        return self.embeddings[article_ids]

    def _prepare_path(self, path):
        return path


class BatchNavigator(object):

    def __init__(self, graph, beam_size, path_length):
        self.graph = graph
        self.beam_size = beam_size
        self.path_length = path_length

        # Hack: use a random page as a dummy page which will fill up beams
        # which are too small.
        # Works in expectation. :)
        self._dummy_page = np.random.choice(len(self.graph.articles))

        assert self._dummy_page != self.graph.stop_sentinel, \
                "A very improbable event has occurred. Please restart."

        self._ids, self._paths, self._lengths = None, None, None
        self._beams = None

    def reset(self, batch_size, is_training):
        """
        Prepare a new navigation batch with the given parameters.
        """
        # TODO: Sample outside of the training set.
        self._ids, self._paths, self._lengths = \
                self.graph.sample_paths(batch_size, is_training)
        self._cur_article_ids = [path[0] for path in self._paths]

        self._targets = np.array([path[length - 2] for path, length
                                  in zip(self._paths, self._lengths)])
        self._on_target = np.array([False] * len(self._ids))
        self._successes = np.array([False] * len(self._ids))

        self._num_steps = 0
        self._reset(batch_size, is_training)
        self._prepare()

    def _reset(self, batch_size, is_training):
        # For subclasses.
        pass

    def step(self, actions):
        """
        Make a navigation step with the given actions.
        """
        self._step(actions)

        # Did we just stop at the target page?
        stopped_at_target = np.logical_and(self._on_target,
                                           actions == self.graph.stop_sentinel)
        self._successes = np.logical_or(self._successes, stopped_at_target)
        self._on_target = self.cur_article_ids == self._targets

        self._num_steps += 1
        self._prepare()

    def _step(self, actions):
        """
        For subclasses. Modify state using `actions`. Metadata handled by this
        superclass.
        """
        self._cur_article_ids = self._beams[np.arange(self._beams.shape[0]),
                                            actions]

    @property
    def cur_article_ids(self):
        return self._cur_article_ids

    @property
    def gold_actions(self):
        """
        Return the gold navigation actions for the current state.
        """
        raise RuntimeError("Gold actions not defined for this navigator!")

    @property
    def targets(self):
        """
        Return list of target article IDs for this batch.
        """
        return self._targets

    @property
    def gold_path_lengths(self):
        """
        Return length of un-padded version of gold path (including stop
        sentinel).
        """
        raise RuntimeError("Gold paths not defined for this navigator!")

    @property
    def dones(self):
        """
        Return a boolean for each example indicating whether the traversal has
        finished.
        """
        done = self._num_steps < self.path_length
        return [done] * len(self._ids)

    @property
    def successes(self):
        """
        Return a boolean for each example indicating whether it has
        successfully reached the target.
        """
        return self._successes

    def get_article_for_action(self, example_idx, action):
        """
        Get the article ID corresponding to an action ID in a particular
        example.
        """
        return self._beams[example_idx, action]

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
        for article_id in self._cur_article_ids:
            all_links = self.graph.get_article_links(article_id)

            # Sample `beam_size - 1`; add the STOP sentinel
            links = random.sample(all_links, min(self.beam_size - 1,
                                                 len(all_links)))
            links.append(self.graph.stop_sentinel)

            if len(links) < self.beam_size:
                links.extend([self._dummy_page] * (self.beam_size - len(links)))

            candidates.append(links)

        return candidates

    def _prepare(self):
        """
        Prepare/update information about the current navigator state.
        Should be called after reset / steps are taken.
        """
        self._beams = np.array(self._get_candidates())


class OracleBatchNavigator(BatchNavigator):

    def _reset(self, batch_size, is_training):
        self._cursors = np.zeros_like(self._lengths, dtype=np.int32)

    def _step(self, actions):
        # Ignore the actions; we are following gold paths.
        self._cursors += 1

    @property
    def cur_article_ids(self):
        return np.array([path[idx] if idx < length
                         else self.graph.stop_sentinel
                         for idx, (path, length)
                         in enumerate(zip(self._paths, self._lengths))])

    @property
    def gold_actions(self):
        return self._gold_actions

    @property
    def gold_path_lengths(self):
        return self._lengths

    @property
    def dones(self):
        return self._cursors >= self._lengths

    def _get_candidates(self):
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

            ids = self.graph.get_article_links(cur_id)
            ids = [int(x) for x in ids if x != gold_next_id]

            # Beam must be large enough to hold gold + STOP + a distractor
            assert self.beam_size >= 3
            gold_is_stop = gold_next_id == self.graph.stop_sentinel

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
                ids += [self.graph.stop_sentinel]
            random.shuffle(ids)

            assert len(ids) == self.beam_size

            candidates.append(ids)
            ys.append(ids.index(gold_next_id))

        self._gold_actions = ys
        return candidates

