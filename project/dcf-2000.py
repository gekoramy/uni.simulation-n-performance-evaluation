# %%
from dataclasses import dataclass
from numpy.random import SeedSequence
from numpy.typing import NDArray
from scipy.optimize import fsolve
from matplotlib.backends.backend_pdf import PdfPages

import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import typing as t

# %%
matplotlib.use('macosx')


# %% [markdown]
# ```
# SIFS < DIFS
# SIFS < slot_time
# DIFS = 2 * slot_time + SIFS < 3 * slot_time
# ACK timeout > slot_time
# ACK timeout > SIFS + ACK propagation
# ACK timeout > DIFS

# in this case, there is no need to care about DIFS, because everyone will record the very same DIFS!
# i.e., if a node registers a DIFS, every node registers the same DIFS
# that is because every node keeps sending messages, w/out stopping

# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# DIFS
# collision
# the longest prop time + DIFS
# FAKE - FAKE - FAKE - FAKE - FAKE - FAKE
# 1 1 1 1 1 1 1 1 1
# 2 4 1 5 6 7 8 4 5
#     ^
# 1 3 0 4 5 6 7 3 4
# success
# (longest) prop time + ACK + DIFS
# 1 1 1 1 1 1 1 1 1
# 1 3 9 4 5 6 7 3 4
# ^
# 0 2 8 3 4 5 6 2 3
# success
# (longest) prop time + ACK + DIFS
# 1 1 1 1 1 1 1 1 1
# 7 2 8 3 4 5 6 2 3
#   ^           ^
# 5 0 6 1 2 3 4 0 1
# collision
# the longest prop time + DIFS
# 1 2 1 1 1 1 1 2 1
# 5 0 6 1 2 3 4 0 1
# ...
# ```

# %% [markdown]
# $$
# \begin{cases}
# p = 1 - (1 - \tau)^{n - 1} &\\
# \tau = \frac 2 {1 + W + p W \cdot \sum_{k = 0}^{m - 1} (2p)^k} &
# \end{cases}
# $$
#
# $$
# P_\text{tr} = 1 - (1 - \tau)^n
# $$
#
# $$
# P_\text{s} = \frac{n \tau (1 - \tau)^{n - 1}} {P_\text{tr}}
# $$
#
# $$
# S = \frac
# {P_\text{s} P_\text{tr} E\big[P\big]}
# {(1 - P_\text{tr}) \sigma + P_\text{tr} P_\text{s} T_\text{s} + P_\text{tr} (1 - P_\text{s}) T_\text{c}}
# $$

# %%
def tau_p(n: int, cw_min: int, m: int) -> tuple[float, float]:
    def system(x: tuple[float, float]) -> tuple[float, float]:
        tau, p = x
        return (
            - p + 1 - (1 - tau) ** (n - 1),
            - tau + (2 / (1 + cw_min + p * cw_min * np.sum((2 * p) ** np.arange(m))))
        )

    return fsolve(system, x0=np.full(2, .5))


def S(
        n: int,
        cw_min: int,
        m: int,
        sifs: int,
        difs: int,
        phy_h: int,
        mac_h: int,
        payload: int,
        ack: int,
        channel_bit_rate: int,
        propagation_delay: int,
        slot_time: int,
) -> float:
    tau, p = tau_p(n, cw_min, m)
    Ptr: float = 1 - (1 - tau) ** n
    Ps: float = (n * tau * (1 - tau) ** (n - 1)) / Ptr
    Ts: float = sifs + difs + (phy_h + mac_h + payload + ack) / channel_bit_rate + 2 * propagation_delay
    Tc: float = difs + (phy_h + mac_h + payload) / channel_bit_rate + propagation_delay
    return (Ps * Ptr * payload) / ((1 - Ptr) * slot_time + Ptr * Ps * Ts + Ptr * (1 - Ps) * Tc)  # bit / mus -> Mbit / s


# %%
@dataclass
class Did:
    span: int
    who: int
    attempt: int


@dataclass
class Didnt:
    span: int
    who: NDArray[int]
    attempt: NDArray[int]


def simulation(
        seeds: t.Iterator[NDArray[np.uint32]],
        n: int,
        W: int,
        m: int
) -> t.Iterator[Did | Didnt]:
    rng4backoff: np.random.Generator = np.random.default_rng(next(seeds))

    retries: NDArray[int] = np.full(n, m, dtype=int)
    waiting: NDArray[int] = np.zeros(n, dtype=int)

    while True:

        span: int = np.amin(waiting)
        contenders: NDArray[bool] = np.flatnonzero(waiting == span)

        waiting = waiting - span - 1

        match contenders.size:

            case 1:
                [who] = contenders
                yield Did(
                    span=span,
                    who=who,
                    attempt=retries[who]
                )

                retries[who] = 0

            case _:
                yield Didnt(
                    span=span,
                    who=contenders,
                    attempt=retries[contenders]
                )

                retries[contenders] += 1
                retries[retries > m] = 0

        waiting[contenders] = rng4backoff.integers(
            low=0,
            high=W * 2 ** retries[contenders],
            endpoint=False,
        )


# %%
payload: int = 8184  # 1500 * 8  # 8184
mac_h: int = 272
phy_h: int = 128
ack: int = 112 + phy_h
rts: int = 160 + phy_h
cts: int = 112 + phy_h

channel_bit_rate: int = 1  # 54  # Mbit/s -> bit/mus DO NOT CHANGE OTHERWISE TIMESPANS MUST RETURN FLOAT
propagation_delay: int = 1
slot_time: int = 50  # 9  # 50
sifs: int = 28  # 16  # 28
difs: int = 128  # 34  # 128
ack_to: int = 300
cts_to: int = 300

n: int = 30  # nodes
cw_min: int = 2 ** 5  # 2 ** 4  # 2 ** 5
m: int = 5  # 6  # 5  # max # of attempts

# %%
ax: plt.Axes
_, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.arange(5, 51, .05)

for cw_min in [32, 128]:
    for m in [3, 5]:
        ax.plot(
            ns,
            [
                S(n, cw_min, m, sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, slot_time)
                for n
                in ns
            ],
            label=f'$W = {cw_min}, m = {m}$'
        )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('BAS')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('# of stations')
ax.legend()

# %%
ax: plt.Axes
_, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.arange(5, 51, .05)

for cw_min in [32, 128]:  # [16, 32]:  # [32, 128]:
    for m in [3, 5]:  # [6, 7]:  # [3, 5]:
        ax.plot(ns, [
            p
            for n in ns
            for _, p in [tau_p(n, cw_min, m)]
        ], label=f'$W = {cw_min}, m = {m}$')

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('BAS')
ax.set_ylabel('$p$')
ax.set_xlabel('# of stations')
ax.legend()


# %%
def timespans(logs: t.Iterator[Did | Didnt]) -> t.Iterator[tuple[int, int, bool]]:
    for l in logs:
        match l:
            case Did():
                yield (
                    l.span * slot_time,
                    (phy_h + mac_h + payload + ack) // channel_bit_rate + sifs + 2 * propagation_delay + difs,
                    True,
                )  # MSG p. + SIFS + ACK p. + DIFS

            case Didnt():
                yield (
                    l.span * slot_time,
                    (phy_h + mac_h + payload) // channel_bit_rate + propagation_delay + difs,
                    False,
                )  # MSG p. + DIFS


def who(logs: t.Iterator[Did | Didnt], n: int) -> t.Iterator[NDArray[int]]:
    for l in logs:
        mask: NDArray[int] = np.zeros(n)
        mask[l.who] = 1
        yield mask


# %%
shop: t.Iterator[NDArray[np.uint32]] = iter(np.random.SeedSequence(17).generate_state(10_000))

# %%
sim1: t.Iterator[Did | Didnt]
sim2: t.Iterator[Did | Didnt]
sim1, sim2 = it.tee(simulation(shop, n=n, W=cw_min, m=m), 2)

samples: int = 1_000

# %%
data: NDArray[...] = np.fromiter(
    timespans(sim1),
    np.dtype((int, 3)),
    samples
)

merge: NDArray[int] = np.empty(data.shape[0] * 2, dtype=int)
merge[0::2] = data[:, 0]
merge[1::2] = data[:, 1]

mmm: NDArray[int] = np.zeros(data.shape[0] * 2, dtype=int)
mmm[0::2] = np.where(data[:, 2], 1, -1)

success: NDArray[int] = data[:, 2] * payload  # bit

span_end: NDArray[int] = data[:, 0] + data[:, 1]  # mus

throughput: NDArray[float] = np.cumsum(success) / np.cumsum(span_end)  # bit/mus -> Mbit/s

f: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[1, 3])

ax1.fill_between(np.cumsum(merge) / 1e6, np.where(mmm < 0, 0, mmm), step='post', color='#6CCDAF')
ax1.fill_between(np.cumsum(merge) / 1e6, np.where(mmm > 0, 0, mmm), step='post', color='#ED706B')

ax2.plot(np.cumsum(span_end) / 1e6, throughput)

ax1.set_ylabel('channel usage')
ax1.set_yticks([])
ax2.set_ylabel('throughput [Mbit/s]')
ax2.set_xlabel('time [s]')
ax2.grid(True, linestyle='--')

ax1.set_title(f'BAS $n = {n}, W = {cw_min}, m = {m}$')
f.subplots_adjust(hspace=0)

# %%
whos: NDArray[int] = np.fromiter(
    who(sim2, n),
    np.dtype((int, n)),
    samples
)

collisions = np.count_nonzero(whos[:], 1) != 1
station2collisions = np.where(collisions[:, np.newaxis], whos, 0)
station2successes = np.where(collisions[:, np.newaxis], 0, whos)

collision_p: NDArray[float] = np.mean(
    np.cumsum(station2collisions, axis=0) / np.cumsum(station2collisions + station2successes, axis=0),
    axis=1
)

ax: plt.Axes
_, ax = plt.subplots(1, 1)

ax.plot(range(samples), collision_p)

ax.set_ylabel('$p$')
ax.set_xlabel('contentions')
ax.grid(True, linestyle='--')

ax.set_title(f'BAS $n = {n}, W = {cw_min}, m = {m}$')


# %%
def non_overlapping_batches(xs: t.Iterator, dtype: np.dtype, b: int, size: int) -> NDArray:
    """
    From sequence of tuple to non-overlapping batches NDArray

    #>>> non_overlapping_batches(zip(it.count(), it.count(10)), np.dtype((int, 2)), 2, 2)
    array([[[ 0, 10],
            [ 1, 11]],
    <BLANKLINE>
           [[ 2, 12],
            [ 3, 13]]])
    """
    vs: NDArray = np.fromiter(xs, dtype, size * b)
    return np.stack([vs[i * size:(i + 1) * size] for i in range(b)])


# %%
b: int = 100
batch_size: int = 5_000

# %%
batches: NDArray[int] = non_overlapping_batches(
    timespans(simulation(shop, n=n, W=cw_min, m=m)),
    np.dtype((int, 3)),
    b,
    batch_size
)

successes: NDArray[int] = batches[:, :, 2] * payload  # bit

span_ends: NDArray[int] = batches[:, :, 0] + batches[:, :, 1]  # mus

throughputs: NDArray[float] = np.sum(successes, 1) / np.sum(span_ends, 1)  # bit/mus -> Mbit/s

grand_mean: float = np.mean(throughputs)
ci: tuple[float, float] = sp.stats.t.interval(
    confidence=.95,
    loc=grand_mean,
    scale=sp.stats.sem(throughputs),
    df=b - 1
)

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[3, 1])

ax1.hlines(
    y=throughputs,
    xmin=np.arange(0, b) * batch_size,
    xmax=(np.arange(0, b) + 1) * batch_size,
    colors=[f'C{i}' for i in range(b)],
    alpha=.2,
)

for ax in [ax1, ax2]:
    ax.axhspan(
        ci[0],
        ci[1],
        alpha=.5,
        facecolor='none',
        edgecolor='C0',
        hatch='\\' * 5,
        label=f'CI {.95}',
        linewidth=0,
    )
    ax.axhline(
        grand_mean,
        label=r'$\hat\theta$',
        color='C0',
    )
    ax.axhline(
        S(n, cw_min, m, sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, slot_time),
        alpha=.5,
        label=r'$S$',
        color='red',
        linestyle='--',
    )

    ax.set_ylabel('throughput [Mbit/s]')

ax2.set_xlabel('samples')
# ax2.set_yticks(np.around(np.append(grand_mean, ci), decimals=3))
f.subplots_adjust(hspace=0)
ax1.set_title(f'BAS $n = {n}, W = {cw_min}, m = {m}$')

# %%
batches: NDArray[int] = non_overlapping_batches(
    who(simulation(shop, n=n, W=cw_min, m=m), n=n),
    np.dtype((int, n)),
    b,
    batch_size
)

collisionss = np.count_nonzero(batches, 2) != 1
station2collisionss = np.where(collisionss[:, :, np.newaxis], batches, 0)
station2successess = np.where(collisionss[:, :, np.newaxis], 0, batches)

collision_ps: NDArray[float] = np.mean(
    np.sum(station2collisionss, axis=1) / np.sum(station2collisionss + station2successess, axis=1),
    axis=1
)

grand_mean: float = np.mean(collision_ps)
ci: tuple[float, float] = sp.stats.t.interval(
    confidence=.95,
    loc=grand_mean,
    scale=sp.stats.sem(collision_ps),
    df=b - 1
)

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[3, 1])

ax1.hlines(
    y=collision_ps,
    xmin=np.arange(0, b) * batch_size,
    xmax=(np.arange(0, b) + 1) * batch_size,
    colors=[f'C{i}' for i in range(b)],
    alpha=.2,
)

for ax in [ax1, ax2]:
    ax.axhspan(
        ci[0],
        ci[1],
        alpha=.5,
        facecolor='none',
        edgecolor='C0',
        hatch='\\' * 5,
        label=f'CI {.95}',
        linewidth=0,
    )
    ax.axhline(
        grand_mean,
        label=r'$\hat\theta$',
        color='C0',
    )
    ax.axhline(
        tau_p(n, cw_min, m)[1],
        alpha=.5,
        label=r'$S$',
        color='red',
        linestyle='--',
    )

    ax.set_ylabel('$p$')

ax2.set_xlabel('samples')
# ax2.set_yticks(np.around(np.append(grand_mean, ci), decimals=3))
f.subplots_adjust(hspace=0)
ax1.set_title(f'BAS $n = {n}, W = {cw_min}, m = {m}$')

# %%
ax: plt.Axes
_, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 10_000)

for cw_min in [32, 128, 256]:
    for m in [3, 5]:
        ax.plot(
            ns,
            [
                S(n, cw_min, m, sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, slot_time)
                for n
                in ns
            ],
            label=f'$W = {cw_min}, m = {m}$'
        )

        simulated: NDArray[...] = np.asarray([
            (n, grand_mean, grand_mean - ci[0])
            for n in [5, 10, 15, 20, 30, 50]
            for batches in [non_overlapping_batches(timespans(simulation(shop, n=n, W=cw_min, m=m)), np.dtype((int, 3)), b, batch_size)]
            for successes in [batches[:, :, 2] * payload]
            for span_ends in [batches[:, :, 0] + batches[:, :, 1]]
            for throughputs in [np.sum(successes, 1) / np.sum(span_ends, 1)]
            for grand_mean in [np.mean(throughputs)]
            for ci in [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(throughputs), df=b - 1)]
        ])

        ax.errorbar(
            simulated[:, 0],
            simulated[:, 1],
            simulated[:, 2],
            label=f'simulated $W = {cw_min}, m = {m}$',
            marker=4,
            capsize=4,
            linestyle='',
        )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('BAS')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('# of stations')
ax.legend()


# %%
with PdfPages('multipage_2000.pdf') as pdf:
    for f in map(plt.figure, plt.get_fignums()):
        f.set_size_inches(11.69,8.27)
        pdf.savefig(f)
