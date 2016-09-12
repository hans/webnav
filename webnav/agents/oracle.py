from rllab.misc.overrides import overrides

from webnav.agents import Agent


class WebNavMaxOverlapAgent(Agent):

    """
    An Agent which, when prompted, returns the index of the candidate next
    article which has the highest n-gram overlap with the goal.
    """

    def __init__(self, env):
        self.env = env
        self.beam_size = env.beam_size
        self.path_length = env.path_length

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
        response = ""
        matched, reward = False, 0.0

        if message_str.startswith("which"):
            # TODO don't reward multiple queries on same node without moving in
            # between
            matched = True

        response = response.split(" ")
        response = [self._token2idx[token] for token in response if token]
        return response, reward
