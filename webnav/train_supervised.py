import argparse
from collections import namedtuple
from functools import partial
import pprint
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import tqdm, trange

from webnav import web_graph
from webnav.environment import EmbeddingWebNavEnvironment
from webnav.rnn_model import rnn_model
from webnav.session import PartialRunSessionManager


SupervisedAgent = namedtuple("SupervisedAgent",
                             ["current_node", "query", "candidates",
                              "scores",
                              "ys", "lengths",
                              "all_losses", "loss"])


def supervised_model(beam_size, num_timesteps, embedding_dim, name="model"):
    rnn_inputs, rnn_outputs = rnn_model(beam_size, num_timesteps,
                                        embedding_dim, name=name)
    current_node, query, candidates = rnn_inputs
    scores, = rnn_outputs

    # Real length of each path (used to weight sequence cost).
    rollout_lengths = tf.placeholder(tf.int32, (None,), name="lengths")
    # Gold actions at each timestep
    ys = [tf.placeholder(tf.int32, (None,), name="y%i" % t)
          for t in range(num_timesteps)]

    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(scores_t, ys_t)
              for scores_t, ys_t in zip(scores, ys)]
    losses = [tf.to_float(i < rollout_lengths) * loss_t
              for i, loss_t in enumerate(losses)]
    losses = [tf.reduce_mean(loss_t) for loss_t in losses]
    loss = tf.add_n(losses) / float(len(losses))

    tf.scalar_summary("loss", loss)

    return SupervisedAgent(current_node, query, candidates,
                           scores,
                           ys, rollout_lengths,
                           losses, loss)


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
    num_iters = env._graph.get_num_paths(False) / args.batch_size + 1
    gold_trajectories, trajectories = [], []
    losses = []

    navigator = env._navigator
    assert isinstance(navigator, web_graph.OracleBatchNavigator)

    for i in trange(num_iters, desc="evaluating", leave=True):
        observations = env.reset_batch(args.batch_size)
        t, done = 0, False
        losses_i = []

        # Sample a trajectory from a random batch element.
        sample_idx = np.random.choice(observations[0].shape[0])
        # (gold, sampled)
        start_page = env.cur_article_ids[sample_idx]
        gold_traj, sampled_traj = [start_page], [start_page]

        for t in range(args.path_length):
            query, cur_page, beam = observations

            feed = {
                model.current_node[t]: cur_page,
                model.candidates[t]: beam,
                model.ys[t]: env.gold_actions,
            }
            if t == 0:
                feed[model.query] = query
                feed[model.lengths] = env.gold_path_lengths

            fetch = [model.all_losses[t], model.scores[t]]
            loss, scores = sess.partial_run(sm.partial_handle, fetch, feed)
            losses_i.append(loss)

            # Just sample one batch element
            a_pred = scores[sample_idx].argmax()
            gold_traj.append(
                    navigator._paths[sample_idx][navigator._cursors[sample_idx] + 1])
            sampled_traj.append(
                    navigator.get_article_for_action(sample_idx, a_pred))

            # Take the gold step.
            observations, dones, _ = env.step_batch(None)

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
            if start_id == env._graph.stop_sentinel:
                break

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
        graph = web_graph.EmbeddedWikiNavGraph(args.wiki_path, args.qp_path,
                                               args.emb_path, args.path_length)
    elif args.data_type == "wikispeedia":
        graph = web_graph.EmbeddedWikispeediaGraph(args.wiki_path,
                                                   args.emb_path,
                                                   args.path_length)
    else:
        raise ValueError("Invalid data_type %s" % args.data_type)

    env = EmbeddingWebNavEnvironment(args.beam_size, graph, is_training=True)
    eval_env = EmbeddingWebNavEnvironment(args.beam_size, graph,
                                          is_training=False)

    model = supervised_model(args.beam_size, args.path_length,
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
                    model.ys + [model.query, model.lengths])
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             session_manager=sm, summary_op=None)

    batches_per_epoch = env._graph.get_num_paths(True) / args.batch_size + 1
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

                observations = env.reset_batch(args.batch_size)

                for t in range(args.path_length):
                    query, cur_page, beam = observations

                    feed = {
                        model.current_node[t]: cur_page,
                        model.candidates[t]: beam,
                        model.ys[t]: env.gold_actions,
                    }
                    if t == 0:
                        feed[model.query] = query
                        feed[model.lengths] = env.gold_path_lengths

                    sess.partial_run(sm.partial_handle, model.all_losses[t], feed)

                    # Take the gold step.
                    observations, dones, _ = env.step_batch(None)

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
