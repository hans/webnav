"""
This module defines fixed conversational agents for various environments.
"""


class Agent(object):

    @property
    def vocab(self):
        """
        A list of tokens that this agent uses in its utterances and expects in
        messages directed to it.
        """
        raise NotImplementedError

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def num_tokens(self):
        """
        Maximum number of tokens in an utterance sent/received by this agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the agent for a new environment instance.
        """
        pass

    def __call__(self, environment, message):
        """
        Respond to a message, which is a sequence of tokens from this agent's
        vocabulary.

        Args:
            environment: Current stateful environment object.
            message: Token sequence from another agent. A sequence of integer
                indexes into `self.vocabulary`.

        Returns:
            response: Token sequence directed at the agent who sent the
                original message. A sequence of integer indexes into
                `self.vocabulary`.
            reward: Reward that the sending agent should receive for sending
                this message. 0.0 is a good default!
        """
        raise NotImplementedError
