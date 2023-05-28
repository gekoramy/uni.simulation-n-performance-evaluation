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
from typing import Iterable
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

# %%
set_matplotlib_formats('svg')


# %% [markdown]
# # Exercise 1
#
# Consider the network scenario of Fig. 1.
#
# A source $S$ wants to transmit a packet to destination $D$.
# A multi-hop network separates $S$ from $D$.
#
# Specifically, there are $r$ stages, each of which contains $N$ relays.
# The source is connected to all nodes of stage 1.
# Each node of stage 1 is connected to all nodes of stage 2; each node of stage 2 is connected to all nodes of stage 3, and so on.
# Finally, all nodes of stage $r$ are connected to $D$.
#
# The probability of error over every link in the whole network is equal to $p$.
#
# $S$ employs a flooding policy to send its packet through the network.
# This means that every node that receives the packet correctly will re-forward it exactly once.
#
# For example, at relay stage 1, the probability that any node will fail to receive the packet from $S$ is $p$.
# However, say that $k$ nodes at stage $i$ receive the packet correctly: because of the flooding policy, all $k$ nodes will retransmit the packet.
# Therefore, the probability that a node at stage $i + 1$ fails to receive the packet is not $p$, but rather $p^k$
# (i.e., the probability that *no* transmissions from any of the $k$ relays at stage $i$ is received by the node at stage $i + 1$).
#
# 1. Use Monte-Carlo simulation to estimate the probability that a packet transmitted by the source $S$ *fails to reach* the destination $D$.
# Consider two different cases: $r = 2, N = 2$, and $r = 5, N = 5$.
# For each monte-carlo trial, simulate the transmission of the packet by $S$, the correct or incorrect reception by the relays at stage 1, the retransmission of the packet towards the next stages, and so forth until the packet reaches $D$ or is lost in the process.
# (*Hint*: remember that the probability to fail the reception of a packet is $p^k$, where $k$ is the number of nodes that hold a copy of the packet at the previous stage.)
#
# 2. Repeat the above process for different values of the link error probability $p$.
# Plot the probability of error at $D$ against $p$ for the two cases $\{r = 2, N = 2\}$, and $\{r = 5, N = 5\}$.
# Plot also the 95%-confidence intervals (e.g., as error bars) for each simulation point.
#
# 3. Compare your results against the theoretical error curves provided in the file `theory_ex_flooding.csv`
# ( column 1: values of $p$
# ; column 2: probability of error at $D$ for $\{r = 2, N = 2\}$
# ; column 3: probability of error at $D$ for $\{r = 5, N = 5\}$ ).
#
# 4. Draw conclusions on the behavior of the network for the chosen values of $r$ and $N$.
#
# 5. Plot the average number of successful nodes at each stage, and the corresponding confidence intervals.
# What can you say about the relationship between the number of successful nodes and the probability of error at $D$?
#
# 6. *Facultative*: Repeat point 1 by applying post-stratification on your computed average probability of error.
# You can choose the number of relays that get the packet at Stage 1 as the stratum variable
# (i.e., you have $N + 1$ strata, as the number of relays that get the packet correctly from the source can be $0, 1, \ldots, N$ ).
# How does your precision improve?

# %%
def simulate_flooding(seed: int, p: float, r: int, n: int) -> NDArray[int]:
    """
    Reproducible simulation of multi-hop network with flooding policy
    For each stage, it returns the total number of succesful nodes

    >>> simulate_flooding(3, .8, 10, 10) == simulate_flooding(3, .8, 10, 10)
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True])

    >>> simulate_flooding(3, .8, 10, 10)[0] in [0, 1]
    True

    >>> simulate_flooding(3, .8, 10, 10)[-1] in [0, 1]
    True
    """

    rng: np.random.Generator = np.random.default_rng(seed)

    def next_stage(relays: int, nodes: int) -> int:
        node2packets: NDArray[np.dtype((float, relays))] = rng.random((nodes, relays))
        # if p = 0 then it will return always false
        node2failure: NDArray[bool] = np.all(node2packets < p, axis=1)
        return nodes - np.count_nonzero(node2failure)

    network: Iterable[int] = it.chain([n] * r, [1])
    return np.fromiter(it.accumulate(network, next_stage, initial=1), int)


# %%
doctest.testmod()

# %% [markdown]
# ## Compare networks
# $$
# \{ r = 2, N = 2 \} \quad \text{vs} \quad \{ r = 5, N = 5 \}
# $$
# with
# $$
# p = \frac 1 2
# $$
#
# Finite-horizon simulation
#
# We use independent replications technique
#
# $$
# Y_1\, Y_2\, \ldots\, Y_b
# $$
#
# $$
# \hat V = \frac 1 {b - 1} \sum_i^b (Y_i - \overline Y_b)^2
# \qquad
# \overline Z_b = \frac 1 b \sum_i^b Y_i
# $$
#
# $$
# \left[ \overline Z_b \pm t_{b - 1, \frac {1 + \gamma} 2} \sqrt{\frac {\hat V} b} \right]_\gamma
# $$

# %%
gamma: float = .95
nets: list[tuple[int, int]] = [(2, 2), (5, 5)]

# %%
seeds: NDArray[int] = np.fromiter(mit.sieve(500_000), int, 5_001)[1:]

# %%
net2irs: dict[tuple[int, int], NDArray[bool]] = {
    (r, n): np.asarray(
        [
            0 == simulate_flooding(seed, .5, r, n)[-1]
            for seed in seeds
        ],
        bool
    )
    for r, n in nets
}

# %%
b: int = len(seeds)
net2grand_mean_v_delta: dict[tuple[int, int], tuple[float, float, float]] = {
    net: (grand_mean, v, delta)
    for net, irs in net2irs.items()
    for grand_mean in [np.mean(irs).item()]
    for v in [sum((irs - grand_mean) ** 2) / (b - 1)]
    for delta in [sp.stats.t.ppf((1 + gamma) / 2, df=b - 1) * math.sqrt(v / b)]
}

# %%
a: plt.Axes
_, a = plt.subplots(1, 1)

for i, net in enumerate(nets):
    (r, n) = net
    grand_mean, v, delta = net2grand_mean_v_delta[net]
    a.axhspan(
        grand_mean - delta,
        grand_mean + delta,
        alpha=.5,
        color=f'C{i}',
        label=f'CI {gamma}',
    )
    a.axhline(
        grand_mean,
        color=f'C{i}',
        label=f'$r = {r}, n = {n}$'
    )
    a.set_xticks([])

a.legend(loc='center')
a.set_ylabel(r'$\mathbf{P}\left[{\rm lost}\right]$')
a.set_title(f'${seeds[0]} \\ldots {seeds[-1]} \\vdash {len(seeds)}$ samples')
plt.show()

# %% [markdown]
# ## Compare networks
# $$
# \{ r = 2, N = 2 \} \quad \text{vs} \quad \{ r = 5, N = 5 \}
# $$
# with
# $$
# p \in \{ 0, \frac 1 {20}, \frac 2 {20}, \cdots, \frac {18} {20}, \frac {19} {20} , 1 \}
# $$
# against theoretical values

# %%
file: NDArray[np.dtype((float, 3))] = np.genfromtxt('theory_ex_flooding.csv', delimiter=',')

# %%
seeds: NDArray[int] = np.fromiter(mit.sieve(500_000), int, 3_001)[1:]
ps: NDArray[float] = np.linspace(0, 1, 20)

# %%
net2p2irs: dict[tuple[int, int], dict[float, NDArray[bool]]] = {
    (r, n): {
        p: np.asarray(
            [
                0 == simulate_flooding(seed, p, r, n)[-1]
                for seed in seeds
            ],
            bool
        )
        for p in ps
    }
    for r, n in nets
}

# %%
b: int = len(seeds)
net2p2grand_mean_v_delta: dict[tuple[int, int], dict[float, tuple[float, float, float]]] = {
    net: {
        p: (grand_mean, v, delta)
        for p, irs in p2irs.items()
        for grand_mean in [np.mean(irs).item()]
        for v in [sum((irs - grand_mean) ** 2) / (b - 1)]
        for delta in [sp.stats.t.ppf((1 + gamma) / 2, df=b - 1) * math.sqrt(v / b)]
    }
    for net, p2irs in net2p2irs.items()
}

# %%
id2axs: dict[str, plt.Axes]
_, id2axs = plt.subplot_mosaic([['A', 'A', 'A'], ['A', 'A', 'A'], ['B', 'C', 'D']], figsize=(12, 5 * 2))

for a in id2axs.values():
    for i, net in enumerate(nets):
        (r, n) = net
        p2grand_mean_v_delta: dict[float, tuple[float, float, float]] = net2p2grand_mean_v_delta[net]
        ps, grand_means, deltas = zip(*[
            (p, grand_mean, delta) for p, (grand_mean, _, delta) in p2grand_mean_v_delta.items()
        ])
        a.errorbar(
            ps,
            grand_means,
            deltas,
            fmt='.',
            alpha=.5,
            color=f'C{i}',
            label=f'$[r = {r}, n = {n} \\pm \\delta]_{{{gamma}}}$',
        )

    a.plot(
        file[:, 0],
        file[:, 1],
        alpha=.5,
        label=f'Theoretical $r = 2, n = 2$',
    )

    a.plot(
        file[:, 0],
        file[:, 2],
        alpha=.5,
        label=f'Theoretical $r = 5, n = 5$',
    )

mark_inset(id2axs['A'], id2axs['B'], 1, 2)
mark_inset(id2axs['A'], id2axs['C'], 1, 2)
mark_inset(id2axs['A'], id2axs['D'], 1, 2)

id2axs['B'].set_xlim(-0.01, 0.22)
id2axs['B'].set_ylim(-0.01, 0.22)

id2axs['C'].set_xlim(0.29, 0.51)
id2axs['C'].set_ylim(-0.01, 0.1)

id2axs['D'].set_xlim(0.78, 1.01)
id2axs['D'].set_ylim(0.78, 1.01)

id2axs['A'].legend(loc='upper left')
id2axs['A'].set_ylabel(r'$\mathbf{P}\left[{\rm lost}\right]$')
id2axs['A'].set_xlabel(r'$p$')
id2axs['A'].set_title(f'${seeds[0]} \\ldots {seeds[-1]} \\vdash {len(seeds)} \\cdot {len(ps)}$ samples')
plt.show()

# %% [markdown]
# As $r \to \infty$, the probability of failing to deliver to $D$ increases.
# As $N \to \infty$, the network's reliability improves significantly.
# Between these two factors, the influence of $N$ on the final probability is much more pronounced.
#
# Comparing the 2 configuration: namely $(r = 2, N = 2), (r = 5, N = 5)$.
# Despite the former having a smaller $r$ than the latter, the probability of failing to deliver to $D$ is always smaller in the latter.
# The reason has to be attributed to the larger value of $N$ in the latter configuration.
