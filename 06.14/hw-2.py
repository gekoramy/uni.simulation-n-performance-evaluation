# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
# ---

# %% [markdown]
# |        |         |                                @ |
# |:-------|:--------|---------------------------------:|
# | Luca   | Mosetti | luca.mosetti-1@studenti.unitn.it |
# | Shandy | Darma   |   shandy.darma@studenti.unitn.it |

# %%
from typing import Iterable, Iterator, Callable
from matplotlib_inline.backend_inline import set_matplotlib_formats
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from numpy.typing import NDArray

import doctest
import math
import numpy as np
import more_itertools as mit
import itertools as it
import scipy as sp
import matplotlib.pyplot as plt


# %% [markdown]
# Compare different Monte-Carlo methods for the estimation of $\pi$.
# In particular compare the following:
#
# 1. The naive estimator that computes $\frac \pi 4$ as the ratio of points within a circle of radius equal to $1$ centered on $(0, 0)$ to the total number of points drawn at random within a square of side length $2$, also centered on $(0, 0)$;
# 2. The improved estimator seen in class that exploits conditioning and antithetic random numbers;
# 3. The improved estimator seen in class that exploits conditioning, antithetic random numbers, and stratification.
#
# Set your preferred stopping rules in terms of
# - the number of correct digits of π estimated by your method;
# - the size of the 95% confidence interval for the estimated value of $\pi$.
#
# Show how many iterations the above methods need in each case.
#
# For each of the three above Monte-Carlo methods, it may also be curious to check the histogram of the values you compute the sample average of.

# %%
def uniform_coordinates(*seeds: int) -> Iterator[NDArray[np.dtype((int, 2))]]:
    rngs: list[np.random.Generator] = [
        np.random.default_rng(seed)
        for seed in seeds
    ]

    while True:
        batch: NDArray[np.dtype((int, 2))] = np.hstack([
            rng.uniform(0, 1, (1_000_000, 1))
            for rng in rngs
        ])

        for point in batch:
            yield point


# %%
gamma: float = .95

def cnt(acc: NDArray[np.dtype((int, 2))], x: bool) -> NDArray[np.dtype((int, 2))]:
    acc[int(x)] += 1
    return acc


def precision(decimals: int) -> Callable[[NDArray[np.dtype((int, 2))]], bool]:
    def predicate(oi: NDArray[np.dtype((int, 2))]) -> bool:
        [o, i] = oi
        n: int = o + i
        pi_hat: float = 4 * (i / n)
        return int(pi_hat * 10 ** decimals) != int(math.pi * 10 ** decimals)

    return predicate


def CI(limit: float) -> Callable[[NDArray[np.dtype((int, 2))]], bool]:
    def predicate(oi: NDArray[np.dtype((int, 2))]) -> bool:
        [o, i] = oi
        n: int = o + i
        mu: float = (i / n)
        var: float = (o * (0 - mu) ** 2 + i * (1 - mu) ** 2) / n
        delta: float = sp.stats.norm.ppf(q=(1 + gamma) / 2) * math.sqrt(var / n)
        return delta > limit

    return predicate


ois: Iterator[NDArray[np.dtype((int, 2))]] = it.accumulate(
    (np.linalg.norm(point) <= 1 for point in uniform_coordinates(3, 5)),
    cnt,
    initial=np.zeros(2)
)

mit.consume(ois, 10)

[oi] = mit.take(1, it.dropwhile(CI(0.003), ois))
sum(oi)

# %% [markdown]
# $$
# I =
# \begin{cases}
# 1 & \text{if } V_1^2 + V_2^2 \leq 1
# \\
# 0 & \text{otherwise}
# \end{cases}
# $$
#
# We estimate
# $$
# \mathbf E \big[ I\ |\ V_1 \big] = \sqrt{1 - V_1^2}
# $$

# %%
xs = (math.sqrt(1 - u ** 2) + math.sqrt(1 - (1 - u) ** 2) for [u] in uniform_coordinates(3))

np.mean(mit.take(1_000, xs)) * .5 * 4

# %%

