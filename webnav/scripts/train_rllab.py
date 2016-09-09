import argparse
import copy
from pprint import pprint

import numpy as np
import tensorflow as tf

import rllab.config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from webnav.policies import RankingRecurrentPolicy
from webnav.environment import EmbeddingWebNavEnvironment
from webnav.sampler import VectorizedSampler
from webnav.web_graph import EmbeddedWikispeediaGraph


stub(globals())


DEFAULTS = {
    "batch_size": 50000,
    "max_path_length": 10,
    "n_itr": 500,
    "step_size": 0.01,

    "goal_reward": 10.0,
}


def run_experiment(cli_args, **params):
    base_params = copy.copy(DEFAULTS)
    base_params.update(params)
    params = base_params
    pprint(params)

    assert params["max_path_length"] == cli_args.path_length
    graph = EmbeddedWikispeediaGraph(cli_args.wiki_path, cli_args.emb_path,
                                     cli_args.path_length)

    env = EmbeddingWebNavEnvironment(cli_args.beam_size, graph,
                                     is_training=True, oracle=False)
    baseline = ZeroBaseline(env)

    policy = RankingRecurrentPolicy("policy", env)

    optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=params["batch_size"],
            max_path_length=cli_args.path_length,
            n_itr=params["n_itr"],
            discount=0.99,
            step_size=params["step_size"],
            optimizer=optimizer,
            sampler_cls=VectorizedSampler,
    )

    run_experiment_lite(
            algo.train(),
            n_parallel=1,
            snapshot_mode="last",
            exp_prefix="webnav",
            variant=params,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--path_length", default=3, type=int)
    p.add_argument("--beam_size", default=10, type=int)

    p.add_argument("--logdir", default="/tmp/webnav_qlearning")
    p.add_argument("--eval_interval", default=100, type=int)
    p.add_argument("--summary_interval", default=100, type=int)
    p.add_argument("--n_eval_trajectories", default=5, type=int)

    p.add_argument("--wiki_path", required=True)
    p.add_argument("--emb_path", required=True)

    args = p.parse_args()
    pprint(vars(args))

    rllab.config.LOG_DIR = args.logdir

    run_experiment(args)
