from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.layers import layers


AgentModel = namedtuple("AgentModel",
        ["num_candidates", "ys",
         "current_node", "query", "candidates",
         "loss"])


def build_model(beam_size, embedding_dim, hidden_dims=(256,),
                hidden_activation=tf.nn.tanh, name="model"):
    """
    Build a supervised model which operates with a beam at training time.

    (The beam can be made arbitrarily large in order to replicate the
    original model.)
    """

    with tf.variable_scope(name):
        # TODO: These are all pre-computed, but maybe they should live on the
        # GPU instead? + we can look them up with provided integer indices

        # Number of valid options for each example. <= beam_size
        num_candidates = tf.placeholder(tf.int32, shape=(None,),
                name="num_candidates")
        # Correct action for each example.
        ys = tf.placeholder(tf.int32, shape=(None,), name="ys")

        # Embedding of the current node (pre-computed).
        current_node = tf.placeholder(tf.float32, shape=(None, embedding_dim),
                name="current_node")
        # Embedding of the query (pre-computed).
        query = tf.placeholder(tf.float32, shape=(None, embedding_dim),
                name="query")
        # Embedding of all candidates on the beam (pre-computed).
        candidates = tf.placeholder(
                tf.float32, shape=(None, beam_size, embedding_dim),
                name="candidate_embeddings")

        input_dim = embedding_dim * 2
        hidden_val = [current_node, query]
        assert hidden_dims[-1] == embedding_dim
        for out_dim in hidden_dims:
            hidden_val = layers.fully_connected(hidden_val, out_dim,
                    activation_fn=hidden_activation)

        # Calculate score of "stop" action.
        stop_embedding = tf.get_variable("stop_embedding", (embedding_dim,))
        stop_scores = tf.matmul(hidden_val, stop_embedding)

        # Calculate score for each candidate.
        # batch_size * beam_size
        scores = tf.squeeze(tf.matmul(tf.expand_dims(hidden_val, 1), candidates))
        scores = tf.concat(1, [scores, tf.expand_dims(stop_scores, 1)])

        # TODO: weight loss based on `num_candidates`?
        # Or maybe just feed in constant embedding for smaller beams, and let
        # the model learn to always assign low score there.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                scores, ys)

    return AgentModel(num_candidates, ys,
                      current_node, query, candidates,
                      loss)
