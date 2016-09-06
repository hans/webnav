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
    q_pred = [tf.reduce_max(scores_t, 1) for scores_t in scores]
    for t in range(num_timesteps):
        target = rewards[t]
        if t < num_timesteps - 1:
            # Bootstrap with max_a Q_{t+1}
            target += gamma * q_pred[t + 1]

        q_targets.append(target)

    losses = [tf.reduce_mean(tf.square(mask_t * (q_target_t - q_pred_t)))
              for q_target_t, q_pred_t, mask_t
              in zip(q_targets, q_pred, masks)]
    loss = tf.add_n(losses) / float(len(losses))

    tf.scalar_summary("loss", loss)

    return RLModel(current_node, query, candidates,
                   scores,
                   rewards, masks,
                   losses, loss)


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
        for e in range(args.num_epochs):
            if sv.should_stop():
                break

            for i in trange(batches_per_epoch, desc="epoch %i" % e):
                if sv.should_stop():
                    break

                batch_num = i + e * batches_per_epoch
                # if batch_num % args.eval_interval == 0:
                #     tqdm.write("============================\n"
                #                "Evaluating at batch %i, epoch %i"
                #                % (i, e))
                #     eval(model, eval_env, sv, sm, sess, args)

                observations = env.reset_batch(args.batch_size)
                mask_t = [1.0] * args.batch_size
                rewards, masks = [], []

                for t in range(args.path_length):
                    query, cur_page, beam = observations

                    feed = {
                        model.current_node[t]: cur_page,
                        model.candidates[t]: beam,
                        model.masks[t]: mask_t,
                    }
                    if t == 0:
                        feed[model.query] = query

                    scores_t = sess.partial_run(sm.partial_handle,
                                                model.scores[t], feed)
                    actions = np.argmax(scores_t, axis=1)

                    # TODO does dones == mask? I think mask should be True at
                    # last timestep whereas done is false
                    observations, mask_t, rewards_t = env.step_batch(actions)
                    rewards.append(rewards_t)
                    masks.append(mask_t)

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
