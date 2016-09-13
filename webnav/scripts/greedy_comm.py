"""
A greedy rule-based conversational web navigation agent.

Serves as a baseline for RL-learned conversational web navigators.
"""

import argparse
import sys

import numpy as np

from webnav.environments.conversation import WRAPPED, SEND, RECEIVE, UTTER
from webnav.util import build_webnav_conversation_envs
from webnav.util import log_trajectory


def rollout(env, args, beta=1.0):
    observation = env.reset()
    wrapped_env = env._env

    trajectory = [(WRAPPED, (0, wrapped_env.cur_article_id), 0.0)]
    target = env._env._navigator.target_id

    for t in range(args.path_length):
        (query, current_node, beam), message = observation

        scores = np.dot(beam, query)
        scores /= np.linalg.norm(beam, axis=1)
        scores /= np.linalg.norm(query)
        action = scores.argmax()

        next_step = env.step(action)
        observation = next_step.observation

        traj_data = (action, wrapped_env.get_article_for_action(action))
        trajectory.append((WRAPPED, traj_data, next_step.reward))

        if next_step.done:
            break

    return trajectory, target


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--path_length", default=3, type=int)
    p.add_argument("--beam_size", default=10, type=int)
    p.add_argument("--batch_size", default=1, type=int)
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

    graph, envs, eval_envs = build_webnav_conversation_envs(args)

    log_f = sys.stdout
    traj, target = rollout(envs[0], args)
    log_trajectory(traj, target, envs[0], log_f)
