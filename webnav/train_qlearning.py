"""
Train a web navigation agent purely via Q-learning RL.
"""


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
from webnav.util import discount_cumsum


RLModel = namedtuple("RLModel",
                     ["current_node", "query", "candidates",
                      "scores",
                      "rewards", "masks",
                      "all_losses", "loss"])


def q_model(beam_size, num_timesteps, embedding_dim, gamma=0.99,
            name="model"):
    # The Q-learning model uses the RNN scoring function as the
    # Q-function.
    rnn_inputs, rnn_outputs = rnn_model(beam_size, num_timesteps,
                                        embedding_dim, name=name)
    current_node, query, candidates = rnn_inputs
    scores, = rnn_outputs

    # Per-timestep non-discounted rewards
    rewards = [tf.placeholder(tf.float32, (None,), name="rewards_%i" % t)
               for t in range(num_timesteps)]
    # Mask learning for those examples which stop early
    masks = [tf.placeholder(tf.float32, (None,), name="mask_%i" % t)
             for t in range(num_timesteps)]

    # Calculate Q targets.
    q_targets = []
    q_a_pred = [tf.reduce_max(scores_t, 1) for scores_t in scores]
    for t in range(num_timesteps):
        target = rewards[t]
        if t < num_timesteps - 1:
            # Bootstrap with max_a Q_{t+1}
            target += gamma * q_a_pred[t + 1]

        q_targets.append(target)

        # Summary: mean and variance of Q targets at this timestep
        targets_mean, targets_var = tf.nn.moments(target, [0])
        tf.scalar_summary("q/%i/target/mean" % t, targets_mean)
        tf.scalar_summary("q/%i/target/var" % t, targets_var)

        # Summary: mean and variance of predicted Q values for optimal
        # actions at this timestep
        q_a_pred_mean, q_a_pred_var = tf.nn.moments(q_a_pred[t], [0])
        tf.scalar_summary("q/%i/pred_max/mean" % t, q_a_pred_mean)
        tf.scalar_summary("q/%i/pred_max/var" % t, q_a_pred_var)

        # Summary: mean and variance of all Q values in batch
        q_pred_mean, q_pred_var = tf.nn.moments(scores[t], [0, 1])
        tf.scalar_summary("q/%i/pred/mean" % t, q_pred_mean)
        tf.scalar_summary("q/%i/pred/var" % t, q_pred_var)

    losses = [tf.reduce_mean(tf.square(mask_t * (q_target_t - q_a_pred_t)))
              for q_target_t, q_a_pred_t, mask_t
              in zip(q_targets, q_a_pred, masks)]
    loss = tf.add_n(losses) / float(len(losses))

    tf.scalar_summary("loss", loss)

    return RLModel(current_node, query, candidates,
                   scores,
                   rewards, masks,
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
    trajectories, targets = [], []
    losses = []
    navigator = env._navigator

    # Per-timestep loss accumulator.
    per_timestep_losses = np.zeros((args.path_length,))
    total_returns = 0.0

    for i in trange(args.n_eval_iters, desc="evaluating", leave=True):
        observations = env.reset_batch(args.batch_size)
        masks_t = [1.0] * len(observations[0])
        rewards, masks = [], []

        # Sample a trajectory from a random batch element.
        sample_idx = np.random.choice(observations[0].shape[0])
        traj = [(env.cur_article_ids[sample_idx], 0.0)]
        targets.append(navigator._targets[sample_idx])

        for t in range(args.path_length):
            query, cur_page, beam = observations

            feed = {
                model.current_node[t]: cur_page,
                model.candidates[t]: beam,
                model.masks[t]: masks_t,
            }
            if t == 0:
                feed[model.query] = query

            fetch = [model.all_losses[t], model.scores[t]]
            scores_t = sess.partial_run(sm.partial_handle, model.scores[t],
                                        feed)
            actions = scores_t.argmax(axis=1)

            # Track our single batch element
            if masks_t[sample_idx] > 0.0:
                traj_article = navigator.get_article_for_action(
                        sample_idx, actions[sample_idx])

            observations, dones, rewards_t = env.step_batch(actions)

            if masks_t[sample_idx] > 0.0:
                traj.append((traj_article, rewards_t[sample_idx]))

            masks_t = 1.0 - np.array(dones).astype(np.float32)
            rewards.append(rewards_t)
            masks.append(masks_t)


        # Compute Q-learning loss.
        fetches = model.all_losses
        feeds = {model.rewards[t]: rewards_t
                 for t, rewards_t in enumerate(rewards)}
        losses_i = sess.partial_run(sm.partial_handle, fetches, feeds)

        # Accumulate.
        per_timestep_losses += losses_i
        total_returns += np.array(rewards).sum(axis=0).mean()
        trajectories.append(traj)

        sm.reset_partial_handle(sess)

    per_timestep_losses /= float(args.n_eval_iters)
    total_returns /= float(args.n_eval_iters)

    loss = per_timestep_losses.mean()
    tqdm.write("Validation loss: %.10f" % loss)

    tqdm.write("Per-timestep validation losses:\n%s\n"
               % "\n".join("\t% 2i: %.10f" % (t, loss_t)
                           for t, loss_t in enumerate(per_timestep_losses)))

    # Log random trajectories
    for traj, target in zip(trajectories, targets):
        tqdm.write("Trajectory: (target %s)"
                   % env._graph.get_article_title(target))
        for article_id, reward in traj:
            # NB: Assumes traj with oracle
            tqdm.write("\t%-40s\t%.5f"
                       % (env._graph.get_article_title(article_id), reward))
            if article_id == env._graph.stop_sentinel:
                break

    # Write summaries using supervisor.
    summary = tf.Summary()
    summary.value.add(tag="eval/loss", simple_value=np.asscalar(loss))
    summary.value.add(tag="eval/mean_reward",
                      simple_value=np.asscalar(total_returns))
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

    env = EmbeddingWebNavEnvironment(args.beam_size, graph, is_training=True,
                                     oracle=False)
    eval_env = EmbeddingWebNavEnvironment(args.beam_size, graph,
                                          is_training=False, oracle=False)

    model = q_model(args.beam_size, args.path_length, env.embedding_dim,
                    args.gamma)

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
                    model.rewards + model.masks + [model.query])
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             session_manager=sm, summary_op=None)

    batches_per_epoch = env._graph.get_num_paths(True) / args.batch_size + 1
    with sv.managed_session() as sess:
        for e in range(args.n_epochs):
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
                mask = [1.0] * args.batch_size
                rewards = []

                for t in range(args.path_length):
                    query, cur_page, beam = observations

                    feed = {
                        model.current_node[t]: cur_page,
                        model.candidates[t]: beam,
                        model.masks[t]: mask,
                    }
                    if t == 0:
                        feed[model.query] = query

                    scores_t = sess.partial_run(sm.partial_handle,
                                                model.scores[t], feed)
                    actions = np.argmax(scores_t, axis=1)

                    observations, dones, rewards_t = env.step_batch(actions)
                    rewards.append(rewards_t)

                    # Compute mask for next prediction timestep.
                    mask = 1.0 - np.array(dones).astype(np.float32)

                do_summary = i % args.summary_interval == 0
                summary_fetch = summary_op if do_summary else train_op

                fetches = [train_op, summary_fetch, model.loss]
                feeds = {model.rewards[t]: rewards_t
                         for t, rewards_t in enumerate(rewards)}
                _, summary, loss = sess.partial_run(sm.partial_handle,
                                                    fetches, feeds)

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
    p.add_argument("--gamma", default=0.99, type=float)

    p.add_argument("--n_epochs", default=3, type=int)
    p.add_argument("--n_eval_iters", default=2, type=int)

    p.add_argument("--logdir", default="/tmp/webnav_qlearning")
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
