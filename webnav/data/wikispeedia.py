import codecs
from collections import defaultdict, namedtuple
import os
import re
import random
import sys
import urllib

try:
    import cPickle as pickle
except:
    import pickle

import nltk
from nltk.corpus import stopwords
import numpy as np
from stanza.text import GloveVocab


def decode_name(name):
    return urllib.unquote(name).decode("utf-8")


Wikispeedia = namedtuple("Wikispeedia", ["articles", "categories",
                                         "category_articles", "links",
                                         "paths"])
Article = namedtuple("Article", ["name", "lead_tokens", "cleaned_name",
                                 "categories", "is_dummy"])
Path = namedtuple("Path", ["duration", "articles", "has_backtrack"])


DUMMY_PAGES = ["_Stop", "_Dummy"]


def load_raw_data(data_dir, lead_text_num_tokens=300):
    titles, original_titles = [], []
    with open(os.path.join(data_dir, "articles.tsv"), "r") as titles_f:
        for line in titles_f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            original_titles.append(line)
            line = decode_name(line)
            titles.append(line)

    original_title2id = {original_title: idx for idx, original_title
                         in enumerate(original_titles)}
    title2id = {title: idx for idx, title in enumerate(titles)}

    for i, dummy_page_name in enumerate(DUMMY_PAGES):
        assert title2id[dummy_page_name] == i

    categories = []
    category2id = {}
    category_articles = defaultdict(list)
    article_categories = defaultdict(list)
    with open(os.path.join(data_dir, "categories.tsv"), "r") as categories_f:
        for line in categories_f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            title, category = line.split("\t")
            if category not in category2id:
                category2id[category] = len(categories)
                categories.append(category)

            article_id = original_title2id[title]
            category_id = category2id[category]
            category_articles[category_id].append(article_id)
            article_categories[article_id].append(category_id)


    articles = []
    for idx, title in enumerate(original_titles):
        articles.append(load_article(data_dir, title, article_categories[idx],
                                     lead_text_num_tokens))

    links = defaultdict(list)
    with open(os.path.join(data_dir, "links.tsv"), "r") as links_f:
        for line in links_f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            source_name, tgt_name = line.split("\t")
            source_id = title2id[decode_name(source_name)]
            tgt_id = title2id[decode_name(tgt_name)]
            if tgt_id == source_id:
                continue
            links[source_id].append(tgt_id)


    paths = []
    with open(os.path.join(data_dir, "paths_finished.tsv"), "r") as paths_f:
        for line in paths_f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            titles = fields[3].split(";")
            cursor, ids = 0, []
            has_backtrack = False
            for title in titles:
                if title == "<":
                    has_backtrack = True
                    ids.append(ids[cursor - 2])
                    cursor -= 1
                else:
                    ids.append(title2id[decode_name(title)])
                    cursor += 1

            paths.append(Path(int(fields[2]), ids, has_backtrack))

    # Split paths into train/val/test.
    n_train = int(0.9 * len(paths))
    ids = range(len(paths))
    random.shuffle(ids)

    train_ids = ids[:n_train]
    dev_ids = ids[n_train:n_train + int(0.05 * len(paths))]
    test_ids = ids[n_train + int(0.05 * len(paths)):]

    train_paths = [dict(paths[idx]._asdict()) for idx in train_ids]
    dev_paths = [dict(paths[idx]._asdict()) for idx in dev_ids]
    test_paths = [dict(paths[idx]._asdict()) for idx in test_ids]
    paths = dict(train=train_paths, valid=dev_paths, test=test_paths)

    # Convert to serialization-friendly dicts
    articles = [dict(a._asdict()) for a in articles]

    return dict(articles=articles, categories=categories,
                category_articles=dict(category_articles),
                links=dict(links), paths=paths)


main_text_line = re.compile(r"""^   [^ ].*$""", re.MULTILINE)
number_re = re.compile(r"^[0-9]{0,3}$")
stopwords = set(stopwords.words("english"))
stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
stopwords.update(["'s", "''", "``"])
stopwords.update(["also", "much", "very"])

def clean_tokens(tokens):
    tokens = [token.lower() for token in tokens]
    return [token for token in tokens
            if token not in stopwords and not number_re.match(token)]

def load_article(data_dir, title, category_ids, lead_text_num_tokens):
    if title in DUMMY_PAGES:
        return Article(title, [], clean_tokens(title.split("_")), [],
                       is_dummy=True)

    path = os.path.join(data_dir, "plaintext_articles", title + ".txt")
    with codecs.open(os.path.join(path), "r", encoding="utf-8") as article_f:
        match_iter = re.finditer(main_text_line, article_f.read())
        tokens = []
        while len(tokens) < lead_text_num_tokens:
            try:
                match = next(match_iter).group().strip()
            except StopIteration: break

            if "#copyright" in match:
                continue

            match_tokens = nltk.word_tokenize(match)
            tokens.extend(clean_tokens(match_tokens))

    tokens = tokens[:lead_text_num_tokens]

    cleaned_name = clean_tokens(decode_name(title).split("_"))
    return Article(title, tokens, cleaned_name, category_ids, False)


def make_article_embeddings(wikispeedia_data, vocab_cls=GloveVocab,
                            use_title=False):
    vocab = vocab_cls()
    articles = wikispeedia_data["articles"]
    for article in articles:
        tokens = article["cleaned_name"] if use_title else article["lead_tokens"]
        vocab.update(tokens)
    print "loading embeddings"
    E = vocab.get_embeddings()
    print "done"

    article_embeddings = np.empty((len(articles), E.shape[1]))
    for i, article in enumerate(articles):
        n, embedding = 0, np.zeros((E.shape[1]))
        tokens = article["cleaned_name"] if use_title else article["lead_tokens"]

        if article["name"] in DUMMY_PAGES:
            # Use a random embedding.
            article_embeddings[i] = np.random.normal(scale=0.15,
                                                     size=E.shape[1])
        else:
            for token in tokens:
                if token in vocab:
                    embedding += E[vocab[token]]
                    n += 1
            article_embeddings[i] = embedding / float(n)

    return article_embeddings


if __name__ == "__main__":
    data_dir, out_file, emb_out_file = sys.argv[1:]

    print "processing core wikispeedia data"
    if os.path.exists(out_file):
        with open(out_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = load_raw_data(data_dir)
        with open(sys.argv[2], "wb") as out_f:
            pickle.dump(data, out_f, pickle.HIGHEST_PROTOCOL)
    print "done\n"

    embeddings = make_article_embeddings(data)
    np.savez(sys.argv[3], embeddings)
