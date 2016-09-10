from functools import partial
import random

import numpy as np

from rllab.envs.base import Env, Step
from rllab.misc import logger
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product

from webnav.spaces import DiscreteBinaryBag


# Event identifiers. (Not the same thing as actions.)
#
# `WRAPPED` = action in wrapped environment
# `UTTER` = utter a single token
# `SEND` = send all tokens uttered and end turn
WRAPPED = 0
UTTER = 1
SEND = 2
RECEIVE = 3


class SituatedConversationEnvironment(Env):

    """
    An environment which simulates conversation between two agents A
    and B in some situated environment.

    A represents the policy being learned. It can emit token sequences
    to B and process the responses from B.
    B is some fixed external resource (i.e., not a policy being learned)
    which accepts A's utterances and returns its own utterances. These may
    be a stochastic function of A's input.
    B can also directly (through Python code) modify the state of the
    environment.

    A can also take actions in the situated environment being wrapped by this
    class. A's action space thus consists of the product of the action
    space of the wrapped environment and the utterance space.
    """

    def __init__(self, env, b_agent, *args, **kwargs):
        """
        Args:
            env: Environment which is being wrapped by this conversational
                environment. This environment must have a discrete action
                space. The RL agent will be able to take any of the actions
                in this environment, or select a new action `utterance`,
                managed by this class.
            b_agent: `Agent` instance with which the learned agent will
                interact.
        """
        super(SituatedConversationEnvironment, self).__init__(*args, **kwargs)

        assert isinstance(env.action_space, Discrete)
        self._env = env

        self.num_tokens = b_agent.num_tokens
        self.vocab = b_agent.vocab
        self.vocab_size = len(self.vocab)

        # Observations are a combination of observations from the wrapped
        # environment and a representation of any utterance received from the
        # agent.
        self._received_message_space = DiscreteBinaryBag(self.vocab_size)
        self._obs_space = Product(env.observation_space,
                                  self._received_message_space)

        # The agent can choose to take any action in the wrapped env, to add a
        # single token to its message, or to send a message to the agent.
        #
        # First `N` actions correspond to taking an action in the wrapped env.
        # Next `V` actions correspond to uttering a word from the vocabulary.
        # Final action corresponds to sending the message.
        action_space = Discrete(env.action_space.n + b_agent.vocab_size + 1)
        self._action_space = action_space

        self._b_agent = b_agent

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        # Reset agent.
        self._b_agent.reset()

        # Reset tracking state.
        self._timestep = 0
        self._message = []
        self._sent, self._received = [], []

        self._events = []

        # Reset wrapped env and pull an initial observation.
        self._last_wrapped_obs = self._env.reset()

        # Add a dummy message to the history.
        self._received.append(np.zeros((self.vocab_size,), dtype=np.uint8))

        return (self._last_wrapped_obs, self._received[-1])

    def step(self, action):
        """
        Args:
            action: A token sequence emitted by `A`.
        """

        # Base environment reward + per-turn penalty
        reward = -1
        done = False

        if action < self._env.action_space.n:
            # Agent took an action in wrapped env.
            self._events.append((WRAPPED, action))

            wrapped_step = self._env.step_wrapped(action, self._timestep)
            self._last_wrapped_obs = wrapped_step.observation

            # Override reward and done states from this env.
            reward += wrapped_step.reward
            done = wrapped_step.done
        elif action < self._env.action_space.n + self.vocab_size:
            # Agent chose to output a token.
            token_id = action - self._env.action_space.n
            self._events.append((UTTER, token_id))

            self._message.append(token_id)

            reward += 0.0
        else:
            # Agent chose to send the message.
            self._sent.append(self._message)
            self._events.append((SEND, self._message))

            # Send the message and get a response.
            response, reward_delta = self._b_agent(self._env, self._message)
            reward += reward_delta

            self._received.append(self._received_message_space.from_idx_sequence(response))
            self._events.append((RECEIVE, response))

            self._message = []

        observation = (self._last_wrapped_obs, self._received[-1])
        self._timestep += 1
        return Step(observation=observation, reward=reward, done=done)

    def render(self):
        if len(self._events) == 0:
            print "\n\n====================\n"
            return

        last_event, data = self._events[-1]
        if last_event == WRAPPED:
            print "A took action %i" % data
        elif last_event == UTTER:
            print "A: ", self.vocab[data]
        elif last_event == RECEIVE:
            send_event, send_data = self._events[-2]
            print "A sends message \"%s\"" % "".join(self.vocab[idx] for idx in send_data)
            print "B sends message \"%s\"" % "".join(self.vocab[idx] for idx in data)

    log_fn = partial(logger.log, with_prefix=False, with_timestamp=False)

    def log_diagnostics(self, paths):
        l = self.log_fn

        for _ in range(5):
            random_path = random.choice(paths)
            l("============== Random path (returns %f):" % sum(random_path["rewards"]))
            self.log_path(random_path, l)
            l("==============")

        best_path = max(paths, key=lambda path: sum(path["rewards"]))
        l("\n============== Best path (returns %f):" % sum(best_path["rewards"]))
        self.log_path(best_path, l)
        l("==============")

    def log_path(self, path, log_fn):
        actions, observations = path["actions"], path["observations"]
        rewards = path["rewards"]
        just_sent = False
        a_message = []

        for i, (observation, action, reward) in enumerate(zip(observations, actions, rewards)):
            self.log_timestep(i, observation, action, reward, log_fn)
            # TODO
            wrapped_obs, received_message = self.observation_space.unflatten(observation)
            action = self.action_space.unflatten(action)

    # def log_timestep(self, path, log_fn):
    #     l = log_fn
    #         out_str = None
    #         if just_sent:
    #             message_idxs = received_message.nonzero()[0]
    #             l("B sent message \"%s\""
    #               % "".join(self.vocab[idx] for idx in message_idxs))
    #             just_sent = False

    #         if action < self._env.action_space.n:
    #             if hasattr(self._env, "action_names"):
    #                 action_name = self._env.action_names[action]
    #             else:
    #                 action_name = str(action)
    #             out_str = "A took action \"%s\"" % action_name
    #         elif action < self._env.action_space.n + self.vocab_size:
    #             char = self.vocab[action - self._env.action_space.n]
    #             out_str = "A: %s" % char
    #             a_message.append(char)
    #         else:
    #             out_str = "A sent message \"%s\"" % "".join(a_message)
    #             just_sent = True
    #             a_message = []

    #         l("%-25s\t% 5f" % (out_str, reward))
