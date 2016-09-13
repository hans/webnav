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

from webnav import util
from webnav import web_graph
from webnav.agents.oracle import WebNavMaxOverlapAgent
from webnav.environments import EmbeddingWebNavEnvironment, SituatedConversationEnvironment
from webnav.environments.conversation import UTTER, WRAPPED, SEND, RECEIVE
from webnav.rnn_model import QCommModel, OracleCommModel
from webnav.session import PartialRunSessionManager
from webnav.util import rollout



def eval(model, envs, sv, sm, log_f, args):
    """
    Evaluate the given model on a test environment and log detailed results.

    Args:
        model:
        env:
        sv: Supervisor
        sm: SessionManager (for partial runs)
        log_f:
        args:
    """

    assert not envs[0]._env.is_training
    graph = envs[0]._env._graph

    trajectories, targets = [], []
    losses = []

    # Per-timestep loss accumulator.
    per_timestep_losses = np.zeros((args.path_length,))
    total_returns = 0.0

    for i in trange(args.n_eval_iters, desc="evaluating", leave=True):
        rewards = []

        # Draw a random batch element to track for this batch.
        sample_idx = np.random.choice(len(envs))
        sample_env = envs[sample_idx]
        sample_navigator = sample_env._env._navigator
        sample_done = False

        for iter_info in rollout(model, envs, args, epsilon=0):
            t, observations, _, actions_t, rewards_t, dones_t = iter_info

            # Set up to track a trajectory of a single batch element.
            if t == 0:
                traj = [(WRAPPED, (0, sample_navigator._path[0]), 0.0)]
                targets.append(sample_navigator.target_id)

            # Track our single batch element.
            if not sample_done:
                sample_done = dones_t[sample_idx]

                action = actions_t[sample_idx]
                action_type, data = sample_env.describe_action(action)
                reward = rewards_t[sample_idx]
                if action_type == WRAPPED:
                    traj.append((WRAPPED,
                                (data, sample_env._env.cur_article_id),
                                 reward))
                elif action_type == SEND:
                    traj.append((action_type, data, reward))

                    recv_event = sample_env._events[-1]
                    traj.append((RECEIVE, recv_event[-1], 0.0))
                else:
                    traj.append((action_type, data, reward))

            rewards.append(rewards_t)

        losses_i = np.asarray(model.loss(rewards))

        # Accumulate.
        per_timestep_losses += losses_i
        total_returns += np.array(rewards).sum(axis=0).mean()
        trajectories.append(traj)

        sm.reset_partial_handle()

    per_timestep_losses /= float(args.n_eval_iters)
    total_returns /= float(args.n_eval_iters)

    ##############

    loss = per_timestep_losses.mean()
    tqdm.write("Validation loss: %.10f" % loss, log_f)

    tqdm.write("Per-timestep validation losses:\n%s\n"
               % "\n".join("\t% 2i: %.10f" % (t, loss_t)
                           for t, loss_t in enumerate(per_timestep_losses)),
               log_f)

    # Log random trajectories
    for traj, target in zip(trajectories, targets):
        util.log_trajectory(traj, target, envs[0], log_f)

    # Write summaries using supervisor.
    summary = tf.Summary()
    summary.value.add(tag="eval/loss", simple_value=np.asscalar(loss))
    summary.value.add(tag="eval/mean_reward",
                      simple_value=np.asscalar(total_returns))
    for t, loss_t in enumerate(per_timestep_losses):
        summary.value.add(tag="eval/loss_t%i" % t,
                          simple_value=np.asscalar(loss_t))
    sv.summary_computed(sm.session, summary)


def train(args):
    graph, envs, eval_envs = util.build_webnav_conversation_envs(args)
    model = QCommModel.build(args, envs[0])

    oracle_model = OracleCommModel.build(args, eval_envs[0])

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
            partial_fetches=model.all_fetches + [train_op, summary_op],
            partial_feeds=model.all_feeds)
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             session_manager=sm, summary_op=None)

    # Prepare model for execution
    model.sm = sm

    # Open a file for detailed progress logging.
    import sys
    log_f = sys.stdout # open(os.path.join(args.logdir, "debug.log"), "w")

    batches_per_epoch = graph.get_num_paths(True) / args.batch_size + 1
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
                               % (i, e), log_f)
                    eval(oracle_model, eval_envs, sv, sm, log_f, args)
                    log_f.flush()

                rewards = []
                for iter_info in rollout(model, envs, args):
                    t, _, _, _, _, rewards_t = iter_info
                    rewards.append(rewards_t)

                do_summary = i % args.summary_interval == 0
                summary_fetch = summary_op if do_summary else train_op

                fetches = [train_op, summary_fetch, model.loss]
                feeds = {model.rewards[t]: rewards_t
                         for t, rewards_t in enumerate(rewards)}
                _, summary, loss = sm.run(fetches, feeds)

                sm.reset_partial_handle()

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
    p.add_argument("--goal_reward", default=10, type=float)

    p.add_argument("--n_epochs", default=3, type=int)
    p.add_argument("--n_eval_iters", default=2, type=int)

    p.add_argument("--logdir", default="/tmp/webnav_q_comm")
    p.add_argument("--eval_interval", default=100, type=int)
    p.add_argument("--summary_interval", default=100, type=int)
    p.add_argument("--n_eval_trajectories", default=5, type=int)

    p.add_argument("--data_type", choices=["wikinav", "wikispeedia"],
                   default="wikinav")
    p.add_argument("--wiki_path", required=True)
    p.add_argument("--qp_path")
    p.add_argument("--emb_path")

    args = p.parse_args()
    pprint.pprint(vars(args))

    train(args)
