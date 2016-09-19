import argparse

import numpy as np
import scipy.signal
import tensorflow as tf
from tqdm import tqdm

from webnav import web_graph
from webnav.agents import oracle
from webnav.environments import EmbeddingWebNavEnvironment
from webnav.environments import SituatedConversationEnvironment
from webnav.environments.conversation import SEND, RECEIVE, UTTER, WRAPPED
from webnav.environments.webnav_env import WebNavEnvironment
from webnav.session import PartialRunSessionManager


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-eq$
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


def build_webnav_envs(args):
    """
    Build webnav environments.
    """
    if args.data_type == "wikinav":
        if not args.qp_path:
            raise ValueError("--qp_path required for wikinav data")
        if not args.emb_path:
            raise ValueError("--emb_path required for wikinav data")
        graph = web_graph.EmbeddedWikiNavGraph(args.wiki_path, args.qp_path,
                                               args.emb_path, args.path_length)
    elif args.data_type == "wikispeedia":
        graph = web_graph.EmbeddedWikispeediaGraph(args.wiki_path,
                                                   args.path_length,
                                                   emb_path=args.emb_path)
    else:
        raise ValueError("Invalid data_type %s" % args.data_type)

    webnav_envs = [EmbeddingWebNavEnvironment(args.beam_size, graph,
                                              goal_reward=args.goal_reward,
                                              is_training=True, oracle=False)
                   for _ in range(args.batch_size)]
    webnav_eval_envs = [EmbeddingWebNavEnvironment(args.beam_size, graph,
                                                   goal_reward=args.goal_reward,
                                                   is_training=False,
                                                   oracle=False)
                        for _ in range(args.batch_size)]

    return graph, webnav_envs, webnav_eval_envs


def build_webnav_conversation_envs(args):
    """
    Build situated conversation environments wrapped around the webnav task.
    """
    graph, webnav_envs, webnav_eval_envs = build_webnav_envs(args)

    # Wrap core environment in conversation environment.
    # TODO maybe don't need to replicate agents?
    envs = [SituatedConversationEnvironment(
                env,
                oracle.WebNavEmbeddingAgent(env, match_reward=args.match_reward),
                include_utterance_in_observation=True)
            for env in webnav_envs]
    eval_envs = [SituatedConversationEnvironment(
                    env,
                    oracle.WebNavEmbeddingAgent(env, match_reward=args.match_reward),
                    include_utterance_in_observation=True)
                 for env in webnav_eval_envs]

    return graph, envs, eval_envs


def log_trajectory(trajectory, target, env, log_f):
    webnav_env = env
    if not isinstance(env, WebNavEnvironment):
        webnav_env = webnav_env._env
    graph = webnav_env._graph

    tqdm.write("Trajectory: (target %s)"
                % graph.get_article_title(target), log_f)
    for action_type, data, reward in trajectory:
        stop = False
        if action_type == WRAPPED:
            action_id, article_id = data
            desc = "%s (%i)" % (graph.get_article_title(article_id), action_id)
            if data == graph.stop_sentinel:
                stop = True
        elif action_type == UTTER:
            desc = "\"%s\"" % env.vocab[data]
        elif action_type == RECEIVE:
            desc = "\t--> \"%s\"" % " ".join(env.vocab[idx] for idx in data)
        elif action_type == SEND:
            desc = "SEND"

        tqdm.write("\t%-40s\t%.5f" % (desc, reward), log_f)

        if stop:
            break


def rollout(q_fn, envs, args, epsilon=0.1, active_q_fn=None):
    """
    Execute a batch of rollouts with the given Q function, on- or off-policy.

    Args:
        q_fn: Q-function Q(s,a) which predicts scores for next action in given
            state. Used to execute rollouts unless `active_q_fn` is given
        envs:
        args:
        epsilon: For eps-greedy action sampling
        active_q_fn: Optional secondary Q-function which both observes states
            and predicts actions to follow. Useful when we want to perform
            off-policy rollouts but also train the model Q-function. (In this
            case our off-policy model is `active_q_fn` and the model would be
            `q_fn`.)
    """
    observations = [env.reset() for env in envs]
    batch_size = len(envs)

    masks_t = [1.0] * batch_size

    for t in range(args.path_length):
        scores_t = q_fn.step(t, observations, masks_t)
        if active_q_fn is not None:
            # Override Q-function scores.
            scores_t = active_q_fn.step(t, observations, masks_t)

        # Epsilon-greedy sampling
        actions = scores_t.argmax(axis=1)
        if epsilon > 0:
            actions_rand = np.random.randint(env.action_space.n,
                                             size=batch_size)
            mask = np.random.random(size=batch_size) < epsilon
            actions = np.choose(mask, (actions_rand, actions))

        # Take the step and collect new observation data
        next_steps = [env.step(action)
                      for env, action in zip(envs, actions)]
        obs_next, rewards_t, dones_t, _ = map(list, zip(*next_steps))

        yield t, observations, scores_t, actions, rewards_t, dones_t, masks_t

        observations = [next_step.observation for next_step in next_steps]
        dones = [next_step.done for next_step in next_steps]
        masks_t = 1.0 - np.asarray(dones).astype(np.float32)


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


def prepare_session_helpers(args, partial_fetches, partial_feeds,
                            global_step=None):
    """
    Prepare SessionManager and Supervisor instances for training/testing.

    Returns:
        session_manager:
        supervisor:
        session_config:
    """

    # Don't hog GPU memory.
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_memory)
    session_config = tf.ConfigProto(gpu_options=gpu_options)

    # Build a Supervisor session that supports partial runs.
    sm = PartialRunSessionManager(partial_fetches=partial_fetches,
                                  partial_feeds=partial_feeds)
    sv = tf.train.Supervisor(logdir=args.logdir, global_step=global_step,
                             session_manager=sm, summary_op=None)

    return sm, sv, session_config


def parse_args_with_file_defaults(parser, ignore_file_args=None):
    ignore_file_args = ignore_file_args or []

    # First parse out a possible file directive
    fparser = argparse.ArgumentParser()
    fparser.add_argument("--config", nargs="?", type=argparse.FileType("r"))
    fargs, remaining_args = fparser.parse_known_args()

    ns = argparse.Namespace()
    if fargs.config:
        fconfig = eval(fargs.config.read())
        for key, val in fconfig.iteritems():
            if key not in ignore_file_args:
                setattr(ns, key, val)

    return parser.parse_args(remaining_args, ns)
