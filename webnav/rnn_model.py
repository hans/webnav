from collections import namedtuple
import tensorflow as tf

from tensorflow.contrib.layers import layers


def score_beam(state, candidates):
    embedding_dim = state.get_shape().as_list()[-1]
    num_candidates = candidates.get_shape().as_list()[1]

    # Calculate score for each candidate (batched dot product)
    # batch_size * beam_size
    scores = tf.reshape(tf.batch_matmul(candidates,
                                        tf.expand_dims(state, 2)),
                        (-1, num_candidates))

    return scores


def rnn_model(beam_size, num_timesteps, embedding_dim, cells=None,
              name="model"):
    with tf.variable_scope(name):
        if cells is None:
            cells = [tf.nn.rnn_cell.BasicLSTMCell(1024, state_is_tuple=True)]

        # Embedding of current articles (pre-computed)
        current_nodes = [tf.placeholder(tf.float32,
                                        shape=(None, embedding_dim),
                                        name="current_node_%i" % t)
                         for t in range(num_timesteps)]
        # Embedding of the query (pre-computed)
        query = tf.placeholder(tf.float32, shape=(None, embedding_dim),
                name="query")
        # Embedding of all candidates on the beam (pre-computed)
        candidates = [tf.placeholder(tf.float32,
                                     shape=(None, beam_size, embedding_dim),
                                     name="candidates_%i" % t)
                      for t in range(num_timesteps)]

        # Run stacked RNN.
        batch_size = tf.shape(current_nodes[0])[0]
        hid_vals = [[cell.zero_state(batch_size, tf.float32)
                     for cell in cells]]
        scores = []
        for t in range(num_timesteps):
            if t > 0: tf.get_variable_scope().reuse_variables()

            inp = tf.concat(1, [current_nodes[t], query])
            hid_prev_t = hid_vals[-1]
            hid_t = []
            for i, (cell, hid_prev) in enumerate(zip(cells, hid_prev_t)):
                inp, hid_t_i = cell(inp, hid_prev, scope="layer%i" % i)
                hid_t.append(hid_t_i)

            hid_vals.append(hid_t)

            # Use top hidden layer to calculate scores.
            last_out = inp
            if cells[-1].output_size != embedding_dim:
                last_out = layers.fully_connected(last_out,
                        embedding_dim, activation_fn=tf.tanh,
                        scope="state_projection")

            scores_t = score_beam(last_out, candidates[t])
            scores.append(scores_t)

    inputs = (current_nodes, query, candidates)
    outputs = (scores,)
    return inputs, outputs

