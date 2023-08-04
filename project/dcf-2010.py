# %%
from dataclasses import dataclass
from numpy.random import SeedSequence
from numpy.typing import NDArray
from scipy.optimize import fsolve
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import typing as t

# %%
matplotlib.use('macosx')


# %%
def tau_p(n: int, W: int, R: int) -> tuple[float, float]:
    def system(x: tuple[float, float]) -> tuple[float, float]:
        tau, p = x
        return (
            - p + 1 - (1 - tau) ** (n - 1),
            - tau + 1 / (1 + (1 - p) / (2 * (1 - p ** (R + 1))) * sum([p ** j * (W * 2 ** j - 1) - (1 - p ** (R + 1)) for j in range(R + 1)])
            )
        )

    return fsolve(system, x0=np.full(2, .5))


def S(
        n: int,
        W: int,
        R: int,
        sifs: int,
        difs: int,
        payload: int,
        mpdu: int,
        ack: int,
        channel_bit_rate: int,
        slot_time: int,
) -> float:
    tau, p = tau_p(n, W, R)
    Pb: float = 1 - (1 - tau) ** n
    Ps: float = n * tau * (1 - tau) ** (n - 1)
    Ts: float = (mpdu + ack) / channel_bit_rate + sifs + difs
    Tc: float = (mpdu + ack) / channel_bit_rate + sifs + difs + slot_time
    sEP: float = payload * W / (W - 1)
    sTs: float = Ts * W / (W - 1) + slot_time
    sTc: float = Tc + slot_time
    return (Ps * sEP) / ((1 - Pb) * slot_time + Ps * sTs + (Pb - Ps) * sTc)


# %%
@dataclass
class Did:
    span: int
    who: int
    retry: int


@dataclass
class Didnt:
    span: int
    who: NDArray[int]
    retry: NDArray[int]


def simulation(
        seeds: t.Iterator[NDArray[np.uint32]],
        n: int,
        W: int,
        m: int,
        R: int,
) -> t.Iterator[Did | Didnt]:
    rng4backoff: np.random.Generator = np.random.default_rng(next(seeds))

    retries: NDArray[int] = np.full(n, R, dtype=int)
    waiting: NDArray[int] = np.zeros(n, dtype=int)

    while True:

        span: int = np.amin(waiting)
        contenders: NDArray[bool] = np.flatnonzero(waiting == span)

        waiting -= span

        match contenders.size:

            case 1:
                [who] = contenders
                yield Did(
                    span=span,
                    who=who,
                    retry=retries[who]
                )

                retries[who] = 0

            case _:
                yield Didnt(
                    span=span,
                    who=contenders,
                    retry=retries[contenders]
                )

                retries[contenders] += 1
                retries[retries > R] = 0

                waiting -= 1

        waiting[contenders] = rng4backoff.integers(
            low=0,
            high=W * 2 ** np.minimum(m, retries[contenders]),
            endpoint=False,
        )


# %%
payload: int = 8184
mac_h: int = 272
phy_h: int = 128
mpdu: int = mac_h + phy_h + payload
ack: int = 112 + phy_h

channel_bit_rate: int = 1  # Mbit/s -> bit/mus DO NOT CHANGE OTHERWISE TIMESPANS MUST RETURN FLOAT

slot_time: int = 50
sifs: int = 28
difs: int = sifs + 2 * slot_time

n: int = 30  # # of STAs
W: int = 2 ** 4  # W = W min
m: int = 6  # W max = W * 2 ** m
R: int = m  # max # of retries


# %%
def timespans(logs: t.Iterator[Did | Didnt]) -> t.Iterator[tuple[int, float, bool]]:
    for l in logs:
        match l:
            case Did():
                yield (
                    l.span * slot_time,
                    (mpdu + ack) / channel_bit_rate + sifs + difs,
                    True,
                )  # MSG p. + SIFS + ACK p. + DIFS

            case Didnt():
                yield (
                    l.span * slot_time,
                    (mpdu + ack) / channel_bit_rate + sifs + difs + slot_time,
                    False,
                )  # MSG p. + DIFS


# %%
def non_overlapping_batches(xs: t.Iterator, dtype: np.dtype, b: int, size: int) -> NDArray:
    vs: NDArray = np.fromiter(xs, dtype, size * b)
    return np.stack([vs[i * size:(i + 1) * size] for i in range(b)])


# %%
shop: t.Iterator[NDArray[np.uint32]] = iter(np.random.SeedSequence(17).generate_state(10_000))
b: int = 200
batch_size: int = 1_000

# %%
batches: NDArray[int] = non_overlapping_batches(
    timespans(simulation(shop, n=n, W=W, m=m, R=R)),
    np.dtype((float, 3)),
    b,
    batch_size
)

successes: NDArray[int] = batches[:, :, 2] * payload  # bit

span_ends: NDArray[float] = batches[:, :, 0] + batches[:, :, 1]  # mus

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
        S(n, W, R, sifs, difs, payload, mpdu, ack, channel_bit_rate, slot_time),
        alpha=.5,
        label=r'$S$',
        color='red',
        linestyle='--',
    )

    ax.set_ylabel('throughput [Mbit/s]')

ax2.set_xlabel('samples')
f.subplots_adjust(hspace=0)
ax1.set_title(f'BAS $n = {n}, W = {W}, m = {m}, R = {R}$')


# %%
ax: plt.Axes
_, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

for W in [32, 128, 256]:
    for m in [3, 5]:
        for R in [m, m + 2]:
            ax.plot(
                ns,
                [S(n, W, R, sifs, difs, payload, mpdu, ack, channel_bit_rate, slot_time) for n in ns],
                label=f'$W = {W}, m = {m}, R = {R}$'
            )

            simulated: NDArray[...] = np.asarray([
                (n, grand_mean, grand_mean - ci[0])
                for n in [5, 10, 15, 20, 30, 50]
                for batches in [non_overlapping_batches(timespans(simulation(shop, n=n, W=W, m=m, R=R)), np.dtype((int, 3)), b, batch_size)]
                for successes in [batches[:, :, 2] * payload]
                for span_ends in [batches[:, :, 0] + batches[:, :, 1]]
                for throughputs in [np.sum(successes, 1) / np.sum(span_ends, 1)]
                for grand_mean in [np.mean(throughputs)]
                for ci in
                [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(throughputs), df=b - 1)]
            ])

            ax.errorbar(
                simulated[:, 0],
                simulated[:, 1],
                simulated[:, 2],
                label=f'simulated $W = {W}, m = {m}, R = {R}$',
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
with PdfPages('multipage_2010.pdf') as pdf:
    for f in map(plt.figure, plt.get_fignums()):
        f.set_size_inches(11.69,8.27)
        pdf.savefig(f)
