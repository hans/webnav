"""
Defines a common web graph navigation interface to WikiNav, Wikispeedia, etc.
"""

from collections import namedtuple

import numpy as np


Dataset = namedtuple("Dataset", ["queries", "paths"])
EmbeddedArticle = namedtuple("EmbeddedArticle", ["name", "embedding", "text"])


class EmbeddedWebGraph(object):

    embedding_dim = 128

    def __init__(self, articles, datasets, path_length):
        self.articles = articles
        self.datasets = datasets
        self.path_length = path_length

        assert "train" in self.datasets
        assert "valid" in self.datasets

        # Hack: use a random page as the "STOP" sentinel.
        # Works in expectation. :)
        self.stop_sentinel = np.random.choice(len(self.articles))

        self._eval_cursor = 0

    def sample_queries_paths(self, batch_size, is_training=True):
        dataset = self.datasets["train" if is_training else "valid"]

        if is_training:
            ids = np.random.choice(len(dataset.queries), size=batch_size)
        else:
            if self._eval_cursor > len(dataset.queries):
                self._eval_cursor = 0
            ids = np.arange(self._eval_cursor,
                            min(len(dataset.queries) - 1,
                                self._eval_cursor + batch_size))
            self._eval_cursor += batch_size

        queries, paths = [], []
        for idx in ids:
            query, path = self._prepare_query_path(dataset.queries[idx],
                                                   dataset.paths[idx])
            queries.append(query)
            paths.append(path)

        return ids, queries, paths

    def get_article_links(self, article_idx):
        raise NotImplementedError

    def get_article_title(self, article_idx):
        if article_idx == self.stop_sentinel:
            return "<STOP>"
        return self.articles[article_idx].title

    def get_query_embeddings(self, query_ids):
        raise NotImplementedError

    def get_article_embeddings(self, article_ids):
        raise NotImplementedError

    def _prepare_query_path(self, query, path):
        raise NotImplementedError


class EmbeddedWikiNavGraph(EmbeddedWebGraph):

    embedding_dim = 500

    class ArticlesDict(object):
        def __init__(self, wiki, wiki_emb):
            self.wiki = wiki
            self.wiki_emb = wiki_emb

        def __getitem__(self, idx):
            return EmbeddedArticle(
                    self.wiki.get_article_title(idx),
                    self.wiki_emb.get_article_embedding(idx) if self.wiki_emb else None,
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
        queries_train, queries_val = data.get_queries(["train", "valid"])
        paths_train, paths_val = data.get_paths(["train", "valid"])

        datasets = {
            "train": Dataset(queries_train, paths_train),
            "valid": Dataset(queries_val, paths_val),
        }

        super(EmbeddedWikiNavGraph, self).__init__(articles, datasets,
                                                   path_length)

    def get_article_links(self, article_idx):
        return self._wiki.get_article_links(article_idx)

    def get_query_embeddings(self, queries, paths):
        """
        Fetch representations for a batch of query IDs.
        """

        # Get the last non-STOP page in each corresponding path.
        last_pages = [[idx for idx in path if idx != self.stop_sentinel][-1]
                      for path in paths]
        return self._article_embeddings[last_pages]

    def get_article_embeddings(self, article_ids):
        return self._article_embeddings[article_ids]

    def _prepare_query_path(self, query, path):
        path = path[0]
        # Pad short paths with STOP targets.
        pad_length = max(0, self.path_length + 1 - len(path))
        path = path + [self.stop_sentinel] * pad_length
        return query, path
