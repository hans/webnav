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
    for t in range(num_timesteps):
        target = rewards[t]
        if t < num_timesteps - 1:
            # Bootstrap with max_a Q_{t+1}
            target += gamma * tf.maximum(scores[t + 1], axis=1)

        q_targets.append(target)

    losses = [tf.reduce_mean(tf.square(mask_t * (q_target_t - scores_t)))
              for q_target_t, scores_t, mask_t
              in zip(q_targets, scores, masks)]
    loss = tf.add_n(losses) / float(len(losses))

    tf.scalar_summary("loss", loss)

    return RLModel(current_node, query, candidates,
                   rewards, masks,
                   all_losses, loss)
