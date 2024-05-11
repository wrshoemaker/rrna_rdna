#from __future__ import divsion
import numpy




def fd_weights_all(x, x0=0, n=1):
    """
    Return finite difference weights for derivatives of all orders up to n.
    Parameters
    ----------
    x : vector, length m
        x-coordinates for grid points
    x0 : scalar
        location where approximations are to be accurate
    n : scalar integer
        highest derivative that we want to find weights for
    Returns
    -------
    weights :  array, shape n+1 x m
        contains coefficients for the j'th derivative in row j (0 <= j <= n)
    Notes
    -----
    The x values can be arbitrarily spaced but must be distinct and len(x) > n.
    The Fornberg algorithm is much more stable numerically than regular
    vandermonde systems for large values of n.
    See also
    --------
    fd_weights
    References
    ----------
    B. Fornberg (1998)
    "Calculation of weights_and_points in finite difference formulas",
    SIAM Review 40, pp. 685-691.
    http://www.scholarpedia.org/article/Finite_difference_method
    """
    m = len(x)
    _assert(n < m, 'len(x) must be larger than n')

    weights = numpy.zeros((m, n + 1))
    _fd_weights_all(weights, x, x0, n)
    return weights.T


# from numba import jit, float64, int64, int32, int8, void
# @jit(void(float64[:,:], float64[:], float64, int64))
def _fd_weights_all(weights, x, x0, n):
    m = len(x)
    c1, c4 = 1, x[0] - x0
    weights[0, 0] = 1
    for i in range(1, m):
        j = numpy.arange(0, min(i, n) + 1)
        c2, c5, c4 = 1, c4, x[i] - x0
        for v in range(i):
            c3 = x[i] - x[v]
            c2, c6, c7 = c2 * c3, j * weights[v, j - 1], weights[v, j]
            weights[v, j] = (c4 * c7 - c6) / c3
        weights[i, j] = c1 * (c6 - c5 * c7) / c2
        c1 = c2


def fd_weights(x, x0=0, n=1):
    """
    Return finite difference weights for the n'th derivative.
    Parameters
    ----------
    x : vector
        abscissas used for the evaluation for the derivative at x0.
    x0 : scalar
        location where approximations are to be accurate
    n : scalar integer
        order of derivative. Note for n=0 this can be used to evaluate the
        interpolating polynomial itself.
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.fornberg as ndf
    >>> x = numpy.linspace(-1, 1, 5) * 1e-3
    >>> w = ndf.fd_weights(x, x0=0, n=1)
    >>> df = numpy.dot(w, numpy.exp(x))
    >>> numpy.allclose(df, 1)
    True
    See also
    --------
    fd_weights_all
    """
    return fd_weights_all(x, x0, n)[-1]



def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def fd_derivative(fx, x, n=1, m=2):
    """
    Return the n'th derivative for all points using Finite Difference method.
    Parameters
    ----------
    fx : vector
        function values which are evaluated on x i.e. fx[i] = f(x[i])
    x : vector
        abscissas on which fx is evaluated.  The x values can be arbitrarily
        spaced but must be distinct and len(x) > n.
    n : scalar integer
        order of derivative.
    m : scalar integer
        defines the stencil size. The stencil size is of 2 * mm + 1
        points in the interior, and 2 * mm + 2 points for each of the 2 * mm
        boundary points where mm = n // 2 + m.
    fd_derivative evaluates an approximation for the n'th derivative of the
    vector function f(x) using the Fornberg finite difference method.
    Restrictions: 0 < n < len(x) and 2*mm+2 <= len(x)
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.fornberg as ndf
    >>> x = numpy.linspace(-1, 1, 25)
    >>> fx = numpy.exp(x)
    >>> df = ndf.fd_derivative(fx, x, n=1)
    >>> numpy.allclose(df, fx)
    True
    See also
    --------
    fd_weights
    """
    num_x = len(x)
    _assert(n < num_x, 'len(x) must be larger than n')
    _assert(num_x == len(fx), 'len(x) must be equal len(fx)')

    du = numpy.zeros_like(fx)

    mm = n // 2 + m
    size = 2 * mm + 2  # stencil size at boundary
    # 2 * mm boundary points
    for i in range(mm):
        du[i] = numpy.dot(fd_weights(x[:size], x0=x[i], n=n), fx[:size])
        du[-i - 1] = numpy.dot(fd_weights(x[-size:], x0=x[-i - 1], n=n), fx[-size:])

    # interior points
    for i in range(mm, num_x - mm):
        du[i] = numpy.dot(fd_weights(x[i - mm:i + mm + 1], x0=x[i], n=n),
                       fx[i - mm:i + mm + 1])

    return du
