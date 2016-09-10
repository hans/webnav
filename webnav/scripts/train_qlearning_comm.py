"""
Train a web navigation agent purely via Q-learning RL.
"""


import argparse
from collections import namedtuple
from functools import partial
import os
import pprint
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layers
from tqdm import tqdm, trange

from webnav import web_graph
from webnav.agents.oracle import WebNavMaxOverlapAgent
from webnav.environments import EmbeddingWebNavEnvironment, SituatedConversationEnvironment
from webnav.rnn_model import q_model
from webnav.session import PartialRunSessionManager
from webnav.util import discount_cumsum, transpose_list


def rollout(model, envs, sm, sess, args):
    observations = [env.reset() for env in envs]
    batch_size = len(envs)
    embedding_dim = observations[0][0][0].size
    beam_size = observations[0][0][2].shape[0]

    masks_t = [1.0] * batch_size

    query = np.empty((batch_size, embedding_dim))
    current_nodes = np.empty((batch_size, embedding_dim))
    beams = np.empty((batch_size, beam_size, embedding_dim))

    messages = np.empty((batch_size, env.vocab_size))

    for t in range(args.path_length):
        for i, obs_i in enumerate(observations):
            nav_obs, message = obs_i
            query_i, current_node_i, beam_i = nav_obs

            query[i] = query_i
            current_nodes[i] = current_node_i
            beams[i] = beam_i
            messages[i] = message

        feed = {
            model.current_node[t]: current_nodes,
            model.candidates[t]: beams,
            model.masks[t]: masks_t,
        }
        if t == 0:
            feed[model.query] = query

        scores_t = sess.partial_run(sm.partial_handle, model.scores[t],
                                    feed)
        actions = scores_t.argmax(axis=1)

        next_steps = [env.step(action)
                      for env, action in zip(envs, actions)]
        obs_next, rewards_t, dones_t, _ = map(list, zip(*next_steps))

        yield t, observations, scores_t, actions, rewards_t, dones_t

        observations = [next_step.observation for next_step in next_steps]
        dones = [next_step.done for next_step in next_steps]
        masks_t = 1.0 - np.asarray(dones).astype(np.float32)


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

    webnav_envs = [EmbeddingWebNavEnvironment(args.beam_size, graph,
                                              is_training=True, oracle=False)
                   for _ in range(args.batch_size)]
    # Wrap core environment in conversation environment.
    # TODO maybe don't need to replicate agents?
    envs = [SituatedConversationEnvironment(env, WebNavMaxOverlapAgent(env))
            for env in webnav_envs]

    model = q_model(args.beam_size, args.path_length,
                    webnav_envs[0].embedding_dim, args.gamma)

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

    # Open a file for detailed progress logging.
    log_f = open(os.path.join(args.logdir, "debug.log"), "w")

    batches_per_epoch = graph.get_num_paths(True) / args.batch_size + 1
    with sv.managed_session() as sess:
        for e in range(args.n_epochs):
            if sv.should_stop():
                break

            for i in trange(batches_per_epoch, desc="epoch %i" % e):
                if sv.should_stop():
                    break

                batch_num = i + e * batches_per_epoch
                # if batch_num % args.eval_interval == 0:
                #     tqdm.write("============================\n"
                #                "Evaluating at batch %i, epoch %i"
                #                % (i, e), log_f)
                #     eval(model, eval_envs, sv, sm, sess, log_f, args)
                #     log_f.flush()

                rewards = []
                for iter_info in rollout(model, envs, sm, sess, args):
                    t, _, _, _, _, rewards_t = iter_info
                    rewards.append(rewards_t)

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

    log_f.close()


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
