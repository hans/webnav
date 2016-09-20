"""
Train a web navigation agent purely via Q-learning RL.
"""


import argparse
from collections import namedtuple, Counter
from functools import partial
import os
import pprint
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from webnav import rnn_model
from webnav import util
from webnav import web_graph
from webnav.environments.conversation import UTTER, WRAPPED, SEND, RECEIVE
from webnav.environments.webnav_env import WebNavEnvironment
from webnav.session import PartialRunSessionManager
from webnav.util import rollout


EvalResults = namedtuple("EvalResults", ["n_rollouts",
                                         "per_timestep_losses", "total_returns",
                                         "success_rate", "reach_rate",
                                         "avg_steps_to_reach",
                                         "action_type_counts"])


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

    # Get webnav envs (they may be wrapped right now)
    if isinstance(envs[0], WebNavEnvironment):
        webnav_envs = envs
    else:
        webnav_envs = [env._env for env in envs]
    assert not webnav_envs[0].is_training
    graph = webnav_envs[0]._graph

    trajectories, targets = [], []
    losses = []

    # Eval info accumulators.
    per_timestep_losses = np.zeros((args.path_length,))
    total_returns, success_rate, reach_rate = 0.0, 0.0, 0.0
    action_type_counter = Counter()
    avg_steps_to_reach = 0.0

    for i in trange(args.n_eval_iters, desc="evaluating", leave=True):
        actions, rewards, masks = [], [], []
        reached_target = np.zeros((len(envs),)) + np.inf

        # Draw a random batch element to track for this batch.
        sample_idx = np.random.choice(len(envs))
        sample_env = envs[sample_idx]
        sample_navigator = webnav_envs[sample_idx]._navigator
        sample_done = False

        for iter_info in rollout(model, envs, args, epsilon=0):
            t, observations, _, actions_t, rewards_t, dones_t, masks_t = \
                    iter_info
            action_descriptions = [env_.describe_action(action)
                                   for env_, action
                                   in zip(envs, actions_t)]

            # Track which examples have reached target.
            for j, env in enumerate(webnav_envs):
                if env._navigator._on_target:
                    reached_target[j] = min(reached_target[j], t)

            # Set up to track a trajectory of a single batch element.
            if t == 0:
                traj = [(WRAPPED, (0, sample_navigator._path[0]), 0.0)]
                targets.append(sample_navigator.target_id)

            # Track our single batch element.
            if not sample_done:
                sample_done = dones_t[sample_idx]
                action = actions_t[sample_idx]
                reward = rewards_t[sample_idx]

                if args.task_type == "communication":
                    action_type, data = action_descriptions[sample_idx]
                    if action_type == WRAPPED:
                        traj.append((WRAPPED,
                                    (data, sample_env._env.cur_article_id),
                                    reward))
                    elif action_type == SEND:
                        traj.append((action_type, data, reward))

                        recv_event = sample_env._events[-1]
                        assert recv_event[0] == RECEIVE
                        traj.append((RECEIVE, recv_event[1], 0.0))
                    else:
                        traj.append((action_type, data, reward))
                else:
                    traj.append((WRAPPED,
                                 (action,
                                  webnav_envs[sample_idx].cur_article_id),
                                 reward))

            # Update general accumulators.
            rewards_t = np.asarray(rewards_t)
            masks_t = np.asarray(masks_t)
            actions.append(actions_t)
            rewards.append(rewards_t * masks_t)
            masks.append(masks_t)
            action_type_counter.update(
                    action_desc[0] for action_desc in action_descriptions)

        losses_i = np.asarray(model.get_losses(actions, rewards, masks))

        successes = [webnav_env._navigator.success
                     for webnav_env in webnav_envs]
        success_rate += np.asarray(successes).mean()

        # # of steps required to reach target, for those rollouts which did
        reached_steps = reached_target[np.isfinite(reached_target)]
        reach_rate += len(reached_steps) / float(len(reached_target))
        if len(reached_steps) > 0:
            avg_steps_to_reach += np.asscalar(reached_steps.mean())

        # Accumulate.
        per_timestep_losses += losses_i
        total_returns += np.asarray(rewards).sum(axis=0).mean()
        trajectories.append(traj)

        sm.reset_partial_handle()

    # NB: assumes same batch size at each eval iter for correctness
    per_timestep_losses /= float(args.n_eval_iters)
    total_returns /= float(args.n_eval_iters)
    success_rate /= float(args.n_eval_iters)
    reach_rate /= float(args.n_eval_iters)
    avg_steps_to_reach /= float(args.n_eval_iters)
    action_type_counter = {
            action: count / float(args.n_eval_iters) / float(len(envs))
            for action, count in action_type_counter.iteritems()
    }

    ##############

    loss = per_timestep_losses.mean()
    tqdm.write("Validation loss: %.10f" % loss, log_f)
    tqdm.write("Success rate: %.5f%%" % (success_rate * 100.0), log_f)
    tqdm.write("Reach rate: %.5f%%" % (reach_rate * 100.0), log_f)
    tqdm.write("Mean steps to reach: %.3f" % avg_steps_to_reach, log_f)
    tqdm.write("Mean undiscounted returns: %f" % total_returns, log_f)

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
    summary.value.add(tag="eval/success_rate",
                      simple_value=np.asscalar(success_rate))
    summary.value.add(tag="eval/reach_rate", simple_value=reach_rate)
    summary.value.add(tag="eval/mean_steps_to_reach",
                      simple_value=avg_steps_to_reach)
    summary.value.add(tag="eval/mean_reward",
                      simple_value=np.asscalar(total_returns))
    for action_type, count in action_type_counter.iteritems():
        summary.value.add(tag="eval/action/%i" % action_type,
                          simple_value=count)
    for t, loss_t in enumerate(per_timestep_losses):
        summary.value.add(tag="eval/loss_t%i" % t,
                          simple_value=np.asscalar(loss_t))
    sv.summary_computed(sm.session, summary)

    n_rollouts = len(envs) * args.n_eval_iters
    return EvalResults(n_rollouts=n_rollouts,
                       per_timestep_losses=per_timestep_losses,
                       total_returns=total_returns,
                       success_rate=success_rate,
                       reach_rate=reach_rate,
                       avg_steps_to_reach=avg_steps_to_reach,
                       action_type_counts=action_type_counter)


def build_core(args):
    if args.task_type == "communication":
        graph, envs, eval_envs = util.build_webnav_conversation_envs(args)
        model = rnn_model.QCommModel.build(args, envs[0])
        oracle_model = rnn_model.OracleCommModel.build(args, eval_envs[0])
    elif args.task_type == "navigation":
        graph, envs, eval_envs = util.build_webnav_envs(args)
        model = rnn_model.QNavigatorModel.build(args, envs[0])
        oracle_model = model

    return graph, envs, eval_envs, model, oracle_model


def run_eval(args):
    """
    Run a standalone evaluation on the given trained model.
    """
    graph, _, eval_envs, model, _ = build_core(args)

    log_f = open(os.path.join(args.logdir, "eval"), "w")

    sm, sv, session_config = util.prepare_session_helpers(args,
            partial_fetches=model.all_fetches,
            partial_feeds=model.all_feeds,
            global_step=None)
    model.sm = sm

    with sv.managed_session(config=session_config) as sess:
        eval_results = eval(model, eval_envs, sv, sm, log_f, args)

    pprint.pprint(eval_results)

    log_f.close()


def train(args):
    graph, envs, eval_envs, model, oracle_model = build_core(args)

    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(
            args.learning_rate, global_step, args.learning_rate_decay_interval,
            args.learning_rate_decay_rate, staircase=True)
    tf.scalar_summary("learning_rate", learning_rate)

    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op_ = opt.minimize(model.loss, global_step=global_step)
    # Also prepare non-training updates (e.g. batch-norm updates).
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_op = tf.group(*update_ops)
    # Build a `train_op` Tensor which depends on the actual train op target.
    # This is a hack to get around the current design of partial_run, which
    # does not support targets as fetches.
    # https://github.com/tensorflow/tensorflow/issues/1899
    with tf.control_dependencies([train_op_, update_op]):
        train_op = tf.constant(0., name="train_op")

    summary_op = tf.merge_all_summaries()

    sm, sv, session_config = util.prepare_session_helpers(args,
            partial_fetches=model.all_fetches + [train_op, summary_op],
            partial_feeds=model.all_feeds,
            global_step=global_step)
    model.sm = sm

    # Open a file for detailed progress logging.
    log_f = open(os.path.join(args.logdir, "debug.log"), "w")

    # Save param info to file
    with open(os.path.join(args.logdir, "params"), "w") as params_f:
        pprint.pprint(vars(args), stream=params_f)

    batches_per_epoch = graph.get_num_paths(True) / args.batch_size + 1
    with sv.managed_session(config=session_config) as sess:
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

                # Sample a model for rollouts.
                is_oracle = np.random.random() < args.oracle_freq
                epsilon = 0.0 if is_oracle else args.epsilon
                active_q_fn = oracle_model if is_oracle else None

                actions, rewards, masks = [], [], []
                rollout_info = rollout(model, envs, args,
                                       epsilon=epsilon,
                                       active_q_fn=active_q_fn)
                for step in rollout_info:
                    t, _, _, actions_t, rewards_t, _, masks_t = step
                    actions.append(actions_t)
                    rewards.append(rewards_t)
                    masks.append(masks_t)

                do_summary = i % args.summary_interval == 0
                summary_fetch = summary_op if do_summary else train_op

                # Do training update (NB: regardless of rollout model used).
                fetches = [train_op, summary_fetch, model.loss]
                feeds = {model.rewards[t]: rewards_t
                         for t, rewards_t in enumerate(rewards)}
                feeds.update({model.actions[t]: actions_t
                              for t, actions_t in enumerate(actions)})
                _, summary, loss = sm.run(fetches, feeds)

                if not np.isfinite(loss):
                    raise RuntimeError("NaN loss -- quitting")

                sm.reset_partial_handle()

                if do_summary:
                    sv.summary_computed(sess, summary)
                    tqdm.write("Batch % 5i training loss: %f" %
                               (batch_num, loss))

    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--task_type", choices=["navigation", "communication"],
                   default="communication")

    p.add_argument("--path_length", default=3, type=int)
    p.add_argument("--beam_size", default=10, type=int)
    p.add_argument("--batch_size", default=64, type=int)

    p.add_argument("--learning_rate", default=0.001, type=float)
    p.add_argument("--learning_rate_decay_interval", default=10000, type=int,
                   help="Number of batches by which we should decay once")
    p.add_argument("--learning_rate_decay_rate", default=0.9, type=float,
                   help="Learning rate exponential decay rate")

    p.add_argument("--gamma", default=0.99, type=float)
    p.add_argument("--epsilon", default=0.1, type=float)
    p.add_argument("--rnn_keep_prob", default=0.9, type=float)
    p.add_argument("--goal_reward", default=10, type=float)
    p.add_argument("--match_reward", default=1, type=float)

    p.add_argument("--n_epochs", default=3, type=int)
    p.add_argument("--n_eval_iters", default=2, type=int)

    p.add_argument("--algorithm", choices=["qlearning", "sarsa"],
                   default="qlearning")
    p.add_argument("--oracle_freq", default=0.0, type=float,
                   help=("Decimal frequency of oracle rollouts vs. all "
                         "rollouts (oracle + learned model)"))

    p.add_argument("--logdir", default="/tmp/webnav_q_comm")
    p.add_argument("--eval_interval", default=100, type=int)
    p.add_argument("--summary_interval", default=100, type=int)
    p.add_argument("--n_eval_trajectories", default=5, type=int)
    p.add_argument("--gpu_memory", default=0.22, type=float)

    p.add_argument("--data_type", choices=["wikinav", "wikispeedia"],
                   default="wikinav")
    p.add_argument("--wiki_path")
    p.add_argument("--qp_path")
    p.add_argument("--emb_path")

    args = util.parse_args_with_file_defaults(p, {"logdir"})

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        run_eval(args)
