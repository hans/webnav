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
from webnav.environments.conversation import UTTER, WRAPPED, SEND, RECEIVE
from webnav.rnn_model import rnn_comm_model, q_model
from webnav.session import PartialRunSessionManager
from webnav.util import discount_cumsum, transpose_list


QCommModel = namedtuple("QCommModel", ["current_node", "query", "candidates",
                                       "message",
                                       "scores", "rewards", "masks",
                                       "all_losses", "loss"])


def build_model(args, env):
    """
    Build a communicative Q-learning model.

    Args:
        args: CLI args
        env: Representative instance of communication environment
    """
    webnav_env = env._env
    rnn_inputs, rnn_outputs = rnn_comm_model(args.beam_size, env.b_agent,
                                             args.path_length,
                                             webnav_env.embedding_dim)
    current_node, query, candidates, message = rnn_inputs
    all_inputs = current_node + candidates + message + [query]
    scores = rnn_outputs[0]

    q_tuple = q_model(all_inputs, scores, args.path_length, args.gamma)
    model = QCommModel(current_node, query, candidates, message,
                       *(q_tuple[1:]))
    return model


def build_envs(args):
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
    webnav_eval_envs = [EmbeddingWebNavEnvironment(args.beam_size, graph,
                                                   is_training=False,
                                                   oracle=False)
                   for _ in range(args.batch_size)]

    # Wrap core environment in conversation environment.
    # TODO maybe don't need to replicate agents?
    envs = [SituatedConversationEnvironment(env, WebNavMaxOverlapAgent(env))
            for env in webnav_envs]
    eval_envs = [SituatedConversationEnvironment(env, WebNavMaxOverlapAgent(env))
                 for env in webnav_eval_envs]

    return graph, envs, eval_envs


def rollout(model, envs, sm, args):
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
            model.message[t]: messages,
        }
        if t == 0:
            feed[model.query] = query

        scores_t = sm.run(model.scores[t], feed)
        actions = scores_t.argmax(axis=1)

        next_steps = [env.step(action)
                      for env, action in zip(envs, actions)]
        obs_next, rewards_t, dones_t, _ = map(list, zip(*next_steps))

        yield t, observations, scores_t, actions, rewards_t, dones_t

        observations = [next_step.observation for next_step in next_steps]
        dones = [next_step.done for next_step in next_steps]
        masks_t = 1.0 - np.asarray(dones).astype(np.float32)


def log_trajectory(trajectory, target, env, log_f):
    graph = env._env._graph

    tqdm.write("Trajectory: (target %s)"
                % graph.get_article_title(target), log_f)
    for action_type, data, reward in trajectory:
        stop = False
        if action_type == WRAPPED:
            article_id = data
            desc = graph.get_article_title(data)
            if data == graph.stop_sentinel:
                stop = True
        elif action_type == UTTER:
            desc = "\"%s\"" % env.vocab[data]
        elif action_type == RECEIVE:
            desc = "\t--> \"%s\"" % " ".join(env.vocab[idx] for idx in data)
        elif action_type == SEND:
            desc = "SEND" % (action_type, data)

        tqdm.write("\t%-40s\t%.5f" % (desc, reward), log_f)

        if stop:
            break


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

        for iter_info in rollout(model, envs, sm, args):
            t, observations, _, actions_t, rewards_t, dones_t = iter_info

            # Set up to track a trajectory of a single batch element.
            if t == 0:
                traj = [(WRAPPED, sample_navigator._path[0], 0.0)]
                targets.append(sample_navigator.target_id)

            # Track our single batch element.
            if not sample_done:
                sample_done = dones_t[sample_idx]

                action = actions_t[sample_idx]
                action_type, data = sample_env.describe_action(action)
                reward = rewards_t[sample_idx]
                if action_type == WRAPPED:
                    traj.append((WRAPPED, sample_env._env.cur_article_id,
                                 reward))
                elif action_type == SEND:
                    traj.append((action_type, data, reward))

                    recv_event = sample_env._events[-1]
                    traj.append((RECEIVE, recv_event[-1], 0.0))
                else:
                    traj.append((action_type, data, reward))

            rewards.append(rewards_t)

        # Compute Q-learning loss.
        fetches = model.all_losses
        feeds = {model.rewards[t]: rewards_t
                 for t, rewards_t in enumerate(rewards)}
        losses_i = sm.run(fetches, feeds)

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
        log_trajectory(traj, target, envs[0], log_f)

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
    graph, envs, eval_envs = build_envs(args)
    model = build_model(args, envs[0])

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
                    [model.query] + model.message + model.rewards + \
                    model.masks)
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             session_manager=sm, summary_op=None)

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
                    eval(model, eval_envs, sv, sm, log_f, args)
                    log_f.flush()

                rewards = []
                for iter_info in rollout(model, envs, sm, args):
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
    p.add_argument("--emb_path", required=True)

    args = p.parse_args()
    pprint.pprint(vars(args))

    train(args)
