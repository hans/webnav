import numpy as np
import scipy.signal

from webnav import web_graph
from webnav.agents.oracle import WebNavMaxOverlapAgent
from webnav.environments import EmbeddingWebNavEnvironment
from webnav.environments import SituatedConversationEnvironment


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-eq$
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


def transpose_list(xs):
    """
    Transpose an `M * N` list of lists into an `N * M` list of lists.
    """
    print np.asarray(xs).shape
    print np.asarray(xs).T.shape
    return np.asarray(xs).T.tolist()


def build_webnav_conversation_envs(args):
    """
    Build situated conversation environments wrapped around the webnav task.
    """
    if args.data_type == "wikinav":
        if not args.qp_path:
            raise ValueError("--qp_path required for wikinav data")
        graph = web_graph.EmbeddedWikiNavGraph(args.wiki_path, args.qp_path,
                                               args.emb_path, args.path_length)
    elif args.data_type == "wikispeedia":
        graph = web_graph.EmbeddedWikispeediaGraph(args.wiki_path,
                                                   args.emb_path,
                                                   args.path_length)
    else:
        raise ValueError("Invalid data_type %s" % args.data_type)

    webnav_envs = [EmbeddingWebNavEnvironment(args.beam_size, graph,
                                              is_training=True, oracle=False)
                   for _ in range(args.batch_size)]
    webnav_eval_envs = [EmbeddingWebNavEnvironment(args.beam_size, graph,
                                                   is_training=False,
                                                   oracle=False)
                   for _ in range(args.batch_size)]

    # Wrap core environment in conversation environment.
    # TODO maybe don't need to replicate agents?
    envs = [SituatedConversationEnvironment(env, WebNavMaxOverlapAgent(env))
            for env in webnav_envs]
    eval_envs = [SituatedConversationEnvironment(env, WebNavMaxOverlapAgent(env))
                 for env in webnav_eval_envs]

    return graph, envs, eval_envs

