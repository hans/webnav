"""
Defines a common web graph navigation interface to WikiNav, Wikispeedia, etc.
"""

from collections import namedtuple

import numpy as np


EmbeddedArticle = namedtuple("EmbeddedArticle", ["title", "embedding", "text"])


class EmbeddedWebGraph(object):

    embedding_dim = 128

    def __init__(self, articles, datasets, path_length, stop_sentinel=None):
        self.articles = articles
        self.datasets = datasets
        self.path_length = path_length

        assert "train" in self.datasets
        assert "valid" in self.datasets

        # Hack: use a random page as the "STOP" sentinel.
        # Works in expectation. :)
        self.stop_sentinel = stop_sentinel or np.random.choice(len(self.articles))

        self._eval_cursor = 0

    def sample_paths(self, batch_size, is_training=True):
        dataset = self.datasets["train" if is_training else "valid"]

        if is_training:
            ids = np.random.choice(len(dataset), size=batch_size)
        else:
            if self._eval_cursor > len(dataset):
                self._eval_cursor = 0
            ids = np.arange(self._eval_cursor,
                            min(len(dataset) - 1,
                                self._eval_cursor + batch_size))
            self._eval_cursor += batch_size

        paths = [self._prepare_path(dataset[idx]) for idx in ids]
        return ids, paths

    def get_article_links(self, article_idx):
        raise NotImplementedError

    def get_article_title(self, article_idx):
        if article_idx == self.stop_sentinel:
            return "<STOP>"
        return self.articles[article_idx].title

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
            "train": paths_train,
            "valid": paths_val,
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

        articles = [EmbeddedArticle(article["name"], embeddings[i],
                                    article["lead_tokens"])
                    for i, article in enumerate(data["articles"])]

        # Use a random article as a stop sentinel.
        # TODO: Probably better to make a dedicated sentinel here, since this
        # graph is relatively small
        stop_sentinel = np.random.choice(len(articles))

        datasets = {}
        for dataset_name, dataset in data["paths"].iteritems():
            paths, n_skipped = [], 0
            for path in dataset:
                if len(path["articles"]) > path_length - 1:
                    n_skipped += 1
                    continue

                # Pad with STOP sentinel (every path gets at least one)
                pad_length = max(0, path_length + 1 - len(path["articles"]))
                path = path["articles"] + [stop_sentinel] * pad_length
                paths.append(path)

            print "%s set: skipped %i of %i paths due to length limit" \
                    % (dataset_name, n_skipped, len(dataset))
            datasets[dataset_name] = paths

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
