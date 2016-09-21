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
    navigator = wrapped_env._navigator
    graph = wrapped_env._graph

    trajectory = [(WRAPPED, (0, wrapped_env.cur_article_id), 0.0)]
    target = env._env._navigator.target_id
    steps_to_target = None

    for t in range(args.path_length):
        (query, current_node, beam), message_sent, message_recv = observation

        if navigator.cur_article_id == target \
                or navigator.cur_article_id == graph.stop_sentinel:
            action = [action for action in range(args.beam_size)
                      if navigator._beam[action] == graph.stop_sentinel][0]
            steps_to_target = t
        else:
            scores = np.dot(beam, query)
            scores /= np.linalg.norm(beam, axis=1)
            scores /= np.linalg.norm(query)
            action = scores.argmax()

        next_step = env.step(action)
        observation = next_step.observation

        traj_data = (action, wrapped_env.cur_article_id)
        trajectory.append((WRAPPED, traj_data, next_step.reward))

        if next_step.done:
            break

    return trajectory, target, steps_to_target


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--path_length", default=3, type=int)
    p.add_argument("--beam_size", default=10, type=int)
    p.add_argument("--n_rollouts", default=1000, type=int)

    p.add_argument("--goal_reward", default=10, type=float)
    p.add_argument("--match_reward", default=2.0, type=float)

    p.add_argument("--logdir", default="/tmp/webnav_q_comm")

    p.add_argument("--data_type", choices=["wikinav", "wikispeedia"],
                   default="wikinav")
    p.add_argument("--wiki_path", required=True)
    p.add_argument("--qp_path")
    p.add_argument("--emb_path", required=True)

    args = p.parse_args()
    args.batch_size = 1
    args.task_type = "navigation"

    graph, envs, eval_envs = build_webnav_conversation_envs(args)
    log_f = sys.stdout

    n, successes = 0, []
    for _ in xrange(args.n_rollouts):
        traj, target, steps_to_target = rollout(envs[0], args)
        n += 1
        if steps_to_target is not None:
            successes.append(steps_to_target)

        log_trajectory(traj, target, envs[0], log_f)

    print "Success: %f%%" % (len(successes) / float(n) * 100)
    print "Mean # steps to success: %f" % np.mean(successes)
