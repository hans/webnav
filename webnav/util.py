import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-eq$
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]


def transpose_list(xs):
    """
    Transpose an `M * N` list of lists into an `N * M` list of lists.
    """
    return np.asarray(xs).T.tolist()
