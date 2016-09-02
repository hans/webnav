import argparse
from collections import namedtuple
from functools import partial
import pprint
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import tqdm, trange

from webnav.environment import EmbeddingWebNavEnvironment
from webnav.session import PartialRunSessionManager
from webnav.web_graph import EmbeddedWikiNavGraph
from webnav.web_graph import EmbeddedWikispeediaGraph


AgentModel = namedtuple("AgentModel",
        ["num_candidates", "ys",
         "current_node", "query", "candidates",
         "scores", "all_losses", "loss"])


def score_beam(state, candidates):
    embedding_dim = state.get_shape().as_list()[-1]
    num_candidates = candidates.get_shape().as_list()[1]

    # Calculate score for each candidate (batched dot product)
    # batch_size * beam_size
    scores = tf.reshape(tf.batch_matmul(candidates,
                                        tf.expand_dims(state, 2)),
                        (-1, num_candidates))

    return scores


def build_recurrent_model(beam_size, num_timesteps, embedding_dim,
                          cells=None, name="model"):
    with tf.variable_scope(name):
        if cells is None:
            cells = [tf.nn.rnn_cell.BasicLSTMCell(1024, state_is_tuple=True)]

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
            last_hidden = hid_t[-1][0]
            if cells[-1].output_size != embedding_dim:
                last_hidden = layers.fully_connected(last_hidden,
                        embedding_dim, scope="state_projection")

            scores_t = score_beam(last_hidden, candidates[t])
            scores.append(scores_t)

        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                        scores_t, ys_t)
                  for scores_t, ys_t in zip(scores, ys)]
        losses = [tf.reduce_mean(loss_t) for loss_t in losses]
        loss = tf.add_n(losses) / float(len(losses))

        tf.scalar_summary("loss", loss)

    return AgentModel(None, ys,
                      current_nodes, query, candidates,
                      scores, losses, loss)


def eval(model, env, sv, sm, sess, args):
    """
    Evaluate the given model on a test environment and log detailed results.

    Args:
        model:
        env:
        sv: Supervisor
        sm: SessionManager (for partial runs)
        sess: Session
        args:
    """

    assert not env.is_training
    num_iters = len(env._graph.datasets["valid"]) / args.batch_size + 1
    gold_trajectories, trajectories = [], []
    losses = []

    for i in trange(num_iters, desc="evaluating", leave=True):
        query, cur_page, beam = env.reset_batch(args.batch_size)
        t, done = 0, False
        losses_i = []

        # Sample a trajectory from a random batch element.
        sample_idx = np.random.choice(query.shape[0])
        # (gold, sampled)
        start_page = env.cur_article_ids[sample_idx]
        gold_traj, sampled_traj = [start_page], [start_page]

        while not done:
            feed = {
                model.current_node[t]: cur_page,
                model.candidates[t]: beam,
                model.ys[t]: env.gold_actions,
            }
            if t == 0:
                feed[model.query] = query

            fetch = [model.all_losses[t], model.scores[t]]
            loss, scores = sess.partial_run(sm.partial_handle, fetch, feed)
            losses_i.append(loss)

            # Just sample one batch element
            a_pred = scores[sample_idx].argmax()
            gold_traj.append(env._paths[sample_idx][env._cursors[sample_idx] + 1])
            sampled_traj.append(env.get_page_for_action(sample_idx, a_pred))

            # Take the gold step.
            observations, dones, _ = env.step_batch(None)
            query, cur_page, beam = observations

            t += 1
            done = dones.all()

        losses.append(losses_i)
        gold_trajectories.append(gold_traj)
        trajectories.append(sampled_traj)

        sm.reset_partial_handle(sess)

    losses = np.array(losses)

    loss = losses.mean()
    tqdm.write("Validation loss: %.10f" % loss)

    per_timestep_losses = losses.mean(axis=0)
    tqdm.write("Per-timestep validation losses:\n%s\n"
               % "\n".join("\t% 2i: %.10f" % (t, loss_t)
                           for t, loss_t in enumerate(per_timestep_losses)))

    # Log random trajectories
    traj_pairs = zip(gold_trajectories, trajectories)
    random.shuffle(traj_pairs)
    for gold_traj, traj in traj_pairs[:args.n_eval_trajectories]:
        tqdm.write("Trajectory:")
        for start_id, pred_id in zip(gold_traj, traj[1:]):
            # NB: Assumes traj with oracle
            tqdm.write("\t%-40s\t%-40s" % (env._graph.get_article_title(start_id),
                                           env._graph.get_article_title(pred_id)))
        tqdm.write("\t%-40s" % env._graph.get_article_title(gold_traj[-1]))

    # Write summaries using supervisor.
    summary = tf.Summary()
    summary.value.add(tag="eval/loss", simple_value=np.asscalar(loss))
    for t, loss_t in enumerate(per_timestep_losses):
        summary.value.add(tag="eval/loss_t%i" % t,
                          simple_value=np.asscalar(loss_t))
    sv.summary_computed(sess, summary)


def train(args):
    if args.data_type == "wikinav":
        if not args.qp_path:
            raise ValueError("--qp_path required for wikinav data")
        graph = EmbeddedWikiNavGraph(args.wiki_path, args.qp_path, args.emb_path,
                                     args.path_length)
    elif args.data_type == "wikispeedia":
        graph = EmbeddedWikispeediaGraph(args.wiki_path, args.emb_path,
                                         args.path_length)
    else:
        raise ValueError("Invalid data_type %s" % args.data_type)

    env = EmbeddingWebNavEnvironment(args.beam_size, graph, is_training=True)
    eval_env = EmbeddingWebNavEnvironment(args.beam_size, graph,
                                          is_training=False)

    model = build_recurrent_model(args.beam_size, args.path_length,
                                  env.embedding_dim)

    global_step = tf.Variable(0, trainable=False, name="global_step")
    opt = tf.train.MomentumOptimizer(args.learning_rate, 0.9)
    train_op_ = opt.minimize(model.loss, global_step=global_step)
    # Build a `train_op` Tensor which depends on the actual train op target.
    # This is a hack to get around the current design of partial_run, which
    # does not support targets as fetches.
    # https://github.com/tensorflow/tensorflow/issues/1899
    with tf.control_dependencies([train_op_]):
        train_op = tf.constant(0., name="train_op")

    summary_op = tf.merge_all_summaries()

    # Build a Supervisor session that supports partial runs.
    sm = PartialRunSessionManager(
            partial_fetches=model.scores + model.all_losses + \
                    [train_op, summary_op, model.loss],
            partial_feeds=model.current_node + model.candidates + \
                    model.ys + [model.query])
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             session_manager=sm, summary_op=None)

    batches_per_epoch = len(env._graph.datasets["train"]) / args.batch_size + 1
    with sv.managed_session() as sess:
        for e in range(args.num_epochs):
            if sv.should_stop():
                break

            for i in trange(batches_per_epoch, desc="epoch %i" % e):
                if sv.should_stop():
                    break

                batch_num = i + e * batches_per_epoch
                if batch_num % args.eval_interval == 0:
                    tqdm.write("============================\n"
                               "Evaluating at batch %i, epoch %i"
                               % (i, e))
                    eval(model, eval_env, sv, sm, sess, args)

                query, cur_page, beam = env.reset_batch(args.batch_size)
                t, done = 0, False

                while not done:
                    feed = {
                        model.current_node[t]: cur_page,
                        model.candidates[t]: beam,
                        model.ys[t]: env.gold_actions,
                    }
                    if t == 0:
                        feed[model.query] = query

                    do_summary = batch_num % args.summary_interval == 0
                    summary_fetch = summary_op if do_summary else train_op

                    sess.partial_run(sm.partial_handle, model.all_losses[t], feed)

                    # Take the gold step.
                    observations, dones, _ = env.step_batch(None)
                    query, cur_page, beam = observations

                    t += 1
                    done = dones.all()

                do_summary = i % args.summary_interval == 0
                summary_fetch = summary_op if do_summary else train_op

                _, summary, loss = sess.partial_run(
                        sm.partial_handle, [train_op, summary_fetch, model.loss])
                sm.reset_partial_handle(sess)

                if do_summary:
                    sv.summary_computed(sess, summary)
                    tqdm.write("Batch % 5i training loss: %f" %
                               (batch_num, loss))


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--path_length", default=3, type=int)
    p.add_argument("--beam_size", default=10, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--learning_rate", default=0.001, type=float)

    p.add_argument("--num_epochs", default=3, type=int)

    p.add_argument("--logdir", default="/tmp/webnav_supervised")
    p.add_argument("--eval_interval", default=100, type=int)
    p.add_argument("--summary_interval", default=100, type=int)
    p.add_argument("--n_eval_trajectories", default=5, type=int)

    p.add_argument("--data_type", choices=["wikinav", "wikispeedia"],
                   default="wikinav")
    p.add_argument("--wiki_path", required=True)
    p.add_argument("--qp_path")
    p.add_argument("--emb_path", required=True)

    args = p.parse_args()
    pprint.pprint(vars(args))

    train(args)
