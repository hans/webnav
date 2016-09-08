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


def make_cell_state_placeholder(cell, name):
    if isinstance(cell.state_size, int):
        return tf.placeholder(tf.float32, shape=(None, cell.state_size),
                              name=name)
    elif isinstance(cell.state_size, tuple):
        return tuple([tf.placeholder(tf.float32, shape=(None, size_i),
                                     name="%s/cell_%i" % (name, i))
                      for i, size_i in enumerate(cell.state_size)])
    else:
        raise ValueError("Unknown cell state size declaration %s"
                         % cell.state_size)


def make_cell_zero_state(cell, batch_size):
    if isinstance(cell.state_size, int):
        return np.zeros((batch_size, cell.state_size), dtype=np.float32)
    elif isinstance(cell.state_size, tuple):
        return tuple([np.zeros((batch_size, state_i))
                      for state_i in cell.state_size])
    else:
        raise ValueError("Unknown cell state size declaration %s"
                         % cell.state_size)


def rnn_model(beam_size, num_timesteps, embedding_dim, cells=None,
              build_single_step_graph=False, name="model"):
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
        inputs = [tf.concat(1, [current_nodes_t, query])
                  for current_nodes_t in current_nodes]
        hid_vals = [cell.zero_state(batch_size, tf.float32)
                    for cell in cells]
        scores = []

        def step(hid_prev_t, input_t):
            """
            Build a single vertical-unrolled step in the RNN graph.
            """
            inp = input_t
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

            return scores_t, hid_t

        for t in range(num_timesteps):
            if t > 0: tf.get_variable_scope().reuse_variables()

            scores_t, hid_vals = step(hid_vals, inputs[t])
            scores.append(scores_t)

        # Also build a single-step graph.
        tf.get_variable_scope().reuse_variables()
        hid_prev_single = [
                make_cell_state_placeholder(cell_i, "hid_prev_%i" % i)
                for i, cell_i in enumerate(cells)]
        input_single = inputs[0]
        scores_single, hid_single = step(hid_prev_single, input_single)

    inputs = (current_nodes, query, candidates)
    outputs = (scores,)

    single_inputs = (current_nodes[0], query, candidates[0], hid_prev_single)
    single_outputs = (scores_single, hid_single)

    if build_single_step_graph:
        return (inputs, outputs), (single_inputs, single_outputs)
    else:
        return inputs, outputs

