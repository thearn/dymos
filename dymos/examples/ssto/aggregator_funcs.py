import numpy as np

from openmdao.api import Group, Problem, ExplicitComponent

"""
Constraint types:
    inequality:
        feasible infinite, infeasible infinite (half-plane) e.g. g: real. g < 0
        feasible finite, infeasible infinite e.g g positive: g < 1
        *feasible infinite, infeasible finite e.g. g positiive: , g > 1
    equality e.g. g = 0

constraint aggregates are positive when infeasible, 0 when feasible

STRATEGY: sum all constraints subject to transformation:
            where feasible: influence of these points is zero on aggregate
            where infeasible: influence of these points is linear or nonlinear on aggregate


test prob:
    need mix of g < 0 and 0 < g > m constraints

    ii-ff:
        max distance constraint (stay in circle)
        max state /computed output variable
    fi-if:
        min distance constraints (keep out zones)
        SOC/fuel/resource limit


object placement to maximize coverage inside of object?

    max coverage of n points
    w.r.t placement of points, extent of individual 'zone'
    s.t. all within a space (sphere, etc)
         fixed keepout regions
         no overlapping zones


"""

def KS(g, rho=50.0):
    """
    Kreisselmeier-Steinhauser constraint aggregation function.
    """
    g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
    g_diff = g - g_max
    exponents = np.exp(rho * g_diff)
    summation = np.sum(exponents, axis=-1)[:, np.newaxis]

    KS = g_max + 1.0 / rho * np.log(summation)

    dsum_dg = rho * exponents
    dKS_dsum = 1.0 / (rho * summation)
    dKS_dg = dKS_dsum * dsum_dg

    dsum_drho = np.sum(g_diff * exponents, axis=-1)[:, np.newaxis]
    dKS_drho = dKS_dsum * dsum_drho

    return KS, dKS_dg.flatten()

def RePU(g, p = 1):
    """
    RePU (Rectified Linear Unit) aggregator.
    """
    y = np.zeros(g.shape)
    dy = np.zeros(g.shape)

    # select infeasible
    idx = np.where(g > 0)

    # apply RePU
    y[idx] = g[idx]**p
    dy[idx] = p * g[idx] ** (p - 1)

    # to aggregate: use sum(y)
    return y, dy

def PRePU(g, p=2.0):

    y = np.zeros(g.shape)
    dy = np.zeros(g.shape)

    # select infeasible
    idx1 = np.where((g > 0)&(g <=1.0))
    idx2 = np.where(g > 1.0)

    p1 = 1.1
    y[idx1] = g[idx1]**p1
    dy[idx1] = p1 * g[idx1] ** (p1 - 1)

    p2 = 2.0
    y[idx2] = g[idx2]**p2/p2 + p1*g[idx2] - g[idx2]
    dy[idx2] = g[idx2] ** (p2 - 1) + p1 - 1

    # to aggregate: use sum(y)
    return y, dy

def ReEU(g, a = 0.5):
    """
    RePU (Rectified Linear Unit) aggregator.
    """
    y = np.zeros(g.shape)
    dy = np.zeros(g.shape)

    # select infeasible
    idx = np.where(g > 0)

    # apply RePU
    y[idx] = np.exp(a * g[idx]) - 1.0
    dy[idx] = a*np.exp(a*g[idx])

    # to aggregate: use sum(y)
    return y, dy

def sigmoid(x, a = 10.0):
    """
    Smooth sigmoidal aggregator.
    """
    y = (np.tanh(a*x) + 1) / 2.0 * x

    #y[np.where(y < 0)] = 0.0

    dy = 0.5*a*x*(-np.tanh(a*x)**2 + 1) + 0.5*np.tanh(a*x) + 0.5
    return y, dy

def sigmoid_sq(x, a = 10.0):
    """
    Smooth sigmoidal aggregator.
    """
    y = (np.tanh(a*x) + 1) / 2.0 * x**2

    #y[np.where(y < 0)] = 0.0

    dy = 0.5*a*x**2*(-np.tanh(a*x)**2 + 1) + 2*x*(0.5*np.tanh(a*x) + 0.5)
    return y, dy

def erf(x, a = 25.0):
    """
    Smooth sigmoidal aggregator.
    """
    y = (np.erf(a*x) + 1)/2 * x**2
    dy = a*x**2*np.exp(-a**2*x**2)/np.sqrt(np.pi) + 2*x*(np.erf(a*x)/2 + 1/2)
    return y, dy

import matplotlib.pyplot as plt

def non_agg(x, rho=1):
    return x, np.ones(x.shape)

# g = np.linspace(-1,10, 100)

# y, dy = RePU(g, 2)

# plt.plot(g, y)

# y, dy = ReEU(g, 0.72)

# plt.plot(g, y)


# plt.show()
# quit()

def transform(g, kf, sign=1, rho=50.0):
    k, dk = kf(sign*g, rho)
    return sign*k, sign*dk


aggf = {'RePU' : RePU,
        'PRePU' : PRePU,
        'ReEU' : ReEU,
        'KS' : KS,
        'Sigmoid' : sigmoid,
        'SigmoidSq' : sigmoid_sq,
        'Erf' : erf,
        'None' : non_agg}


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    g = np.linspace(-2, 2, 1000)

    k1, dk1 = transform(g, RePU, 1, 2.0)
    #k2, dk2 = transform(g, RePU, -1, 2)

    k1, dk1 = PRePU(g)
    plt.plot(g, k1)
    plt.plot(g, dk1)
    plt.show()




