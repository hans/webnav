import numpy as np
from rllab.misc.overrides import overrides

from webnav.agents import Agent


class OracleAgent(Agent):
    """
    An agent which responds to queries with explicit action indices.
    """
    def __init__(self, env, match_reward=2.0):
        self.env = env
        self.beam_size = env.beam_size
        self.path_length = env.path_length
        self.match_reward = match_reward

        # For now, vocab = set of valid messages
        # (all messages are 1 token long)
        self._valid_receive = ["which"]
        self._valid_send = [str(x) for x in range(self.beam_size)]
        self._vocab = self._valid_receive + self._valid_send

        self._token2idx = {token: idx for idx, token in enumerate(self.vocab)}

    @property
    @overrides
    def vocab(self):
        return self._vocab

    @property
    @overrides
    def num_tokens(self):
        return 1

    @overrides
    def reset(self):
        pass

    @overrides
    def __call__(self, env, message):
        assert env == self.env

        message_str = " ".join(self.vocab[idx] for idx in message)

        response, reward = self.respond(env, message_str)
        response = response.split(" ")
        response = [self._token2idx[token] for token in response if token]
        return response, reward

    def respond(self, env, message_str):
        raise NotImplementedError


class WebNavEmbeddingAgent(OracleAgent):
    """
    An oracle agent which, when prompted, returns the index of the candidate
    next article which has the highest embedding dot-product with the target
    article's embedding.
    """

    def __init__(self, *args, **kwargs):
        super(WebNavEmbeddingAgent, self).__init__(*args, **kwargs)

        embedding_set = 0
        if len(self.env._graph.embeddings) > 1:
            # Use secondary embedding set if available.
            embedding_set = 1
        self.embeddings = self.env._graph.embeddings[embedding_set]

    @overrides
    def respond(self, env, message_str):
        response = ""
        matched, reward = False, 0.0

        if message_str.startswith("which"):
            # TODO don't reward multiple consecutive matching queries
            matched = True

            graph = env._graph
            beam = self.embeddings[env._navigator._beam]
            query = self.embeddings[env._navigator.target_id]

            scores = np.dot(beam, query)
            scores /= np.linalg.norm(beam, axis=1)
            scores /= np.linalg.norm(query)
            action = scores.argmax()

            response = str(action)

        if matched:
            reward += self.match_reward

        return response, reward


class WebNavMaxOverlapAgent(OracleAgent):

    """
    An Agent which, when prompted, returns the index of the candidate next
    article which has the highest n-gram overlap with the goal.
    """

    @overrides
    def respond(self, env, message_str):
        response = ""
        matched, reward = False, 0.0

        if message_str.startswith("which"):
            # TODO don't reward multiple queries on same node without moving in
            # between
            matched = True

            # Evaluate overlap for each candidate.
            source = env.cur_article_id
            best_target = max(enumerate(env._navigator._beam),
                    key=lambda (_, target): env.reward_for_hop(source, target))
            action = best_target[0]
            response = str(action)

        if matched:
            reward += self.match_reward

        return response, reward
