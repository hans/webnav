import argparse
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import tqdm, trange

from webnav.environment import EmbeddingWebNavEnvironment


AgentModel = namedtuple("AgentModel",
        ["num_candidates", "ys",
         "current_node", "query", "candidates",
         "loss"])


def score_beam(state, candidates):
    embedding_dim = state.get_shape().tolist()[-1]

    # Calculate score of "stop" action.
    stop_embedding = tf.get_variable("stop_embedding", (embedding_dim, 1))
    stop_scores = tf.matmul(state, stop_embedding)

    # Calculate score for each candidate.
    # batch_size * beam_size
    scores = tf.squeeze(tf.batch_matmul(candidates,
                                        tf.expand_dims(state, 2)))

    # Add "stop" action
    scores = tf.concat(1, [scores, stop_scores])
    return scores


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

        hidden_val = tf.concat(1, [current_node, query])
        assert hidden_dims[-1] == embedding_dim
        for out_dim in hidden_dims:
            hidden_val = layers.fully_connected(hidden_val, out_dim,
                    activation_fn=hidden_activation)

        scores = score_beam(hidden_val, candidates)

        # TODO: weight loss based on `num_candidates`?
        # Or maybe just feed in constant embedding for smaller beams, and let
        # the model learn to always assign low score there.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                scores, ys)
        loss = tf.reduce_mean(loss)

    return AgentModel(num_candidates, ys,
                      current_node, query, candidates,
                      loss)


def build_recurrent_model(beam_size, num_timesteps, embedding_dim,
                          cells=None, name="model"):
    with tf.variable_scope(name):
        if cells is None:
            cells = [tf.nn.rnn_cell.BasicLSTMCell(256),
                     tf.nn.rnn_cell.BasicLSTMCell(embedding_dim)]

        # Recurrent versions of the inputs in `build_model`
        ys = [tf.placeholder(tf.int32, shape=(None,), name="ys_%i" % t)
                             for t in range(num_timesteps)]
        current_nodes = [tf.placeholder(tf.float32,
                                        shape=(None, embedding_dim),
                                        name="current_node_%i" % t)
                         for t in range(num_timesteps)]
        # Embedding of the query (pre-computed).
        query = tf.placeholder(tf.float32, shape=(None, embedding_dim),
                name="query")
        # Embedding of all candidates on the beam (pre-computed).
        candidates = [tf.placeholder(tf.float32,
                                     shape=(None, beam_size, embedding_dim),
                                     name="candidates_%i" % t)
                      for t in range(num_timesteps)]

        assert cells[-1].output_size == embedding_dim

        # Run stacked RNN.
        hid_vals = [[cell.zero_state() for cell in cells]]
        scores = []
        for t in range(num_timesteps):
            if t > 0: tf.get_variable_scope.reuse_variables()

            inp = tf.concat(1, [current_nodes[t], query])
            hid_prev_t = hid_vals[-1]
            hids = []
            for cell, hid_prev in zip(cells, hid_prev_t):
                out, state = cell(inp, hid_prev)

                inp = out
                hids.append((out, state))

            hid_vals.append(hids)

            # Use top hidden layer to calculate scores.
            last_hidden = hid_vals[-1][0]
            scores_t = score_beam(last_hidden, candidates[t])
            scores.append(scores_t)

        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                        scores_t, ys_t)
                  for scores_t, ys_t in zip(scores, ys)]
        loss = tf.add_n(losses) / float(num_timesteps)
        loss = tf.reduce_mean(loss)

    return AgentModel(None, ys,
                      current_node, query, candidates,
                      loss)


def train(args):
    env = EmbeddingWebNavEnvironment(args.beam_size, args.wiki_path,
                                     args.qp_path, args.emb_path,
                                     args.path_length)

    model = build_model(args.beam_size, env.embedding_dim,
                        hidden_dims=(256, env.embedding_dim))

    global_step = tf.Variable(0, trainable=False, name="global_step")
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(model.loss, global_step=global_step)

    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step)

    with sv.managed_session() as sess:
        for e in range(args.num_epochs):
            if sv.should_stop():
                break

            for i in trange(len(env._all_queries), desc="epoch %i" % e):
                if sv.should_stop():
                    break

                query, cur_page, beam = env.reset_batch(args.batch_size)
                t, done = 0, False

                while not done:
                    feed = {
                        model.current_node: cur_page,
                        model.query: query,
                        model.candidates: beam,
                        model.ys: env.gold_actions,
                    }

                    loss, _ = sess.run([model.loss, train_op], feed)
                    tqdm.write(str(loss))

                    # Take the gold step.
                    observations, dones, _ = env.step_batch(None)
                    query, cur_page, beam = observations

                    t += 1
                    done = dones.all()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--path_length", default=3, type=int)
    p.add_argument("--beam_size", default=10, type=int)
    p.add_argument("--batch_size", default=64, type=int)

    p.add_argument("--num_epochs", default=3, type=int)

    p.add_argument("--logdir", default="/tmp/webnav_supervised")
    p.add_argument("--summary_step_interval", default=100)

    p.add_argument("--wiki_path", required=True)
    p.add_argument("--qp_path", required=True)
    p.add_argument("--emb_path", required=True)

    args = p.parse_args()
    train(args)
