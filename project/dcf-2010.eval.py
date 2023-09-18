# %%
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from pathlib import Path
from scipy.optimize import fsolve

import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

# %%
out: Path = Path('assets/graph')
out.mkdir(parents=True, exist_ok=True)

matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

cmap: Colormap = matplotlib.colormaps['Paired']


# %%
def tau_p(n: int, W: int, R: int) -> tuple[float, float]:
    def system(x: tuple[float, float]) -> tuple[float, float]:
        tau, p = x
        return (
            - p + 1 - (1 - tau) ** (n - 1),
            - tau + 1 / (1 + (1 - p) / (2 * (1 - p ** (R + 1))) * sum([
                p ** j * (W * 2 ** j - 1) - (1 - p ** (R + 1))
                for j in range(R + 1)
            ]))
        )

    return fsolve(system, x0=np.full(2, .5))


def S(
        n: int,
        W: int,
        R: int,
        payload: int,
        slot_time: int,
        Ts: float,
        Tc: float,
) -> float:
    tau, p = tau_p(n, W, R)
    Pb: float = 1 - (1 - tau) ** n
    Ps: float = n * tau * (1 - tau) ** (n - 1)
    sEP: float = payload * W / (W - 1)
    sTs: float = Ts * W / (W - 1) + slot_time
    sTc: float = Tc + slot_time
    return (Ps * sEP) / ((1 - Pb) * slot_time + Ps * sTs + (Pb - Ps) * sTc)


# %%
def BAS_time_success(
        mpdu: int,
        sifs: int,
        difs: int,
        ack: int,
        channel_bit_rate: int,
) -> float:
    return (mpdu + ack) / channel_bit_rate + sifs + difs


def BAS_time_collision(
        mpdu: int,
        sifs: int,
        difs: int,
        ack: int,
        channel_bit_rate: int,
) -> float:
    return BAS_time_success(mpdu, sifs, difs, ack, channel_bit_rate)


def RTSCTS_time_success(
        mpdu: int,
        sifs: int,
        difs: int,
        ack: int,
        channel_bit_rate: int,
        rts: int,
        cts: int,
) -> float:
    return 3 * sifs + difs + (rts + cts + mpdu + ack) / channel_bit_rate


def RTSCTS_time_collision(
        sifs: int,
        difs: int,
        ack: int,
        channel_bit_rate: int,
        rts: int,
) -> float:
    return sifs + difs + (rts + ack) / channel_bit_rate


# %%
payload: int = 8 * 1500
mac_h: int = 36
phy_h: int = 120
mpdu: int = mac_h + phy_h + payload
ack: int = 112 + phy_h
rts: int = 160 + phy_h
cts: int = 112 + phy_h

channel_bit_rate: int = 6  # Mbit/s -> bit/mus

slot_time: int = 9
sifs: int = 16
difs: int = sifs + 2 * slot_time

n: int = 30  # # of STAs
W: int = 2 ** 5  # W = W min
m: int = 5  # W max = W * 2 ** m
R: int = m  # max # of retries

# %%
b: int = 500
batch_size: int = 1_000

# %%
logs: pd.DataFrame = pd.read_csv(f'assets/2010.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)

contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

spans: NDArray[int] = logs.iloc[:, 0] * slot_time
ts: NDArray[int] = np.where(
    successes,
    BAS_time_success(mpdu, sifs, difs, ack, channel_bit_rate),
    BAS_time_collision(mpdu, sifs, difs, ack, channel_bit_rate) + slot_time,
)

success: NDArray[int] = successes * payload  # bit

span_end: NDArray[int] = spans + ts  # mus

success_s: NDArray[int] = success.reshape(b, batch_size)  # bit

span_end_s: NDArray[int] = span_end.to_numpy().reshape(b, batch_size)  # mus

throughput_s: NDArray[float] = np.sum(success_s, 1) / np.sum(span_end_s, 1)  # bit/mus -> Mbit/s

grand_mean: float = np.mean(throughput_s)
ci: tuple[float, float] = sp.stats.t.interval(
    confidence=.95,
    loc=grand_mean,
    scale=sp.stats.sem(throughput_s),
    df=b - 1
)

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[3, 1])

ax1.hlines(
    y=throughput_s,
    xmin=np.arange(0, b) * batch_size,
    xmax=(np.arange(0, b) + 1) * batch_size,
    colors=[cmap(i) for i in range(b)],
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
        S(
            n=n, W=W, R=R, payload=payload, slot_time=slot_time,
            Ts=BAS_time_success(mpdu, sifs, difs, ack, channel_bit_rate),
            Tc=BAS_time_collision(mpdu, sifs, difs, ack, channel_bit_rate)
        ),
        alpha=.5,
        label=r'$S$',
        color='red',
        linestyle='--',
    )

    ax.set_ylabel('throughput [Mbit/s]')

ax2.set_xlabel('samples')
# ax2.set_yticks(np.around(np.append(grand_mean, ci), decimals=3))
f.subplots_adjust(hspace=0)
ax1.set_title(f'2010 BAS $n = {n}, W = {W}, m = {m}, R = {R}$')
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2010.BAS.throughputs.n = {n}, W = {W}, m = {m}, R = {R}.pgf')

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

for i, (W, (m, R)) in enumerate(it.product([32, 128, 256], [(m, R) for m in [3, 5] for R in [m, m + 2]])):
    ax.plot(
        ns,
        [
            S(
                n=n, W=W, R=R, payload=payload, slot_time=slot_time,
                Ts=BAS_time_success(mpdu, sifs, difs, ack, channel_bit_rate),
                Tc=BAS_time_collision(mpdu, sifs, difs, ack, channel_bit_rate),
            )
            for n
            in ns
        ],
        color=cmap(i),
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2010.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            BAS_time_success(mpdu, sifs, difs, ack, channel_bit_rate),
            BAS_time_collision(mpdu, sifs, difs, ack, channel_bit_rate) + slot_time,
        )]
        for success in [successes * payload]  # bit
        for span_end in [spans + ts]  # mus
        for success_s in [success.reshape(b, batch_size)]  # bit
        for span_end_s in [span_end.to_numpy().reshape(b, batch_size)]  # mus
        for throughput_s in [np.sum(success_s, 1) / np.sum(span_end_s, 1)]  # bit/mus -> Mbit/s
        for grand_mean in [np.mean(throughput_s)]
        for ci in [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(throughput_s), df=b - 1)]
    ])

    ax.errorbar(
        simulated[:, 0],
        simulated[:, 1],
        simulated[:, 2],
        marker=4,
        capsize=4,
        linestyle='',
        color=cmap(i),
    )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('2010 BAS')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
        matplotlib.patches.Patch(
            color=cmap(i),
            label=f'$W = {W}, m = {m}, R={R}$'
        )
        for i, (W, (m, R)) in enumerate(it.product([32, 128, 256], [(m, R) for m in [3, 5] for R in [m, m + 2]]))
    ]
)
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2010.BAS.multi-throughput.pgf')

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

for i, (W, (m, R)) in enumerate(it.product([32, 128, 256], [(m, R) for m in [3, 5] for R in [m, m + 2]])):
    ax.plot(
        ns,
        [
            S(
                n=n, W=W, R=R, payload=payload, slot_time=slot_time,
                Ts=RTSCTS_time_success(mpdu, sifs, difs, ack, channel_bit_rate, rts, cts),
                Tc=RTSCTS_time_collision(mpdu, difs, ack, channel_bit_rate, rts),
            )
            for n
            in ns
        ],
        color=cmap(i),
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2010.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            RTSCTS_time_success(mpdu, sifs, difs, ack, channel_bit_rate, rts, cts),
            RTSCTS_time_collision(mpdu, difs, ack, channel_bit_rate, rts) + slot_time,
        )]
        for success in [successes * payload]  # bit
        for span_end in [spans + ts]  # mus
        for success_s in [success.reshape(b, batch_size)]  # bit
        for span_end_s in [span_end.to_numpy().reshape(b, batch_size)]  # mus
        for throughput_s in [np.sum(success_s, 1) / np.sum(span_end_s, 1)]  # bit/mus -> Mbit/s
        for grand_mean in [np.mean(throughput_s)]
        for ci in [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(throughput_s), df=b - 1)]
    ])

    ax.errorbar(
        simulated[:, 0],
        simulated[:, 1],
        simulated[:, 2],
        marker=4,
        capsize=4,
        linestyle='',
        color=cmap(i),
    )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('2010 RTS/CTS')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
        matplotlib.patches.Patch(
            color=cmap(i),
            label=f'$W = {W}, m = {m}, R = {R}$'
        )
        for i, (W, (m, R)) in enumerate(it.product([32, 128, 256], [(m, R) for m in [3, 5] for R in [m, m + 2]]))
    ]
)
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2010.RTSCTS.multi-throughput.pgf')

# %%
b: int = 100
batch_size: int = 5_000

# %%
logs: pd.DataFrame = pd.read_csv(f'assets/2010.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)

contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

station2successes_s = np.where(successes[:, np.newaxis], contenders, 0).reshape(b, batch_size, n)
station2collisions_s = np.where(successes[:, np.newaxis], 0, contenders).reshape(b, batch_size, n)
collision_p_s = np.mean(
    np.sum(station2collisions_s, axis=1) / np.sum(station2collisions_s + station2successes_s, axis=1),
    axis=1
)

grand_mean: float = np.mean(collision_p_s)
ci: tuple[float, float] = sp.stats.t.interval(
    confidence=.95,
    loc=grand_mean,
    scale=sp.stats.sem(collision_p_s),
    df=b - 1,
)

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[3, 1])

ax1.hlines(
    y=collision_p_s,
    xmin=np.arange(0, b) * batch_size,
    xmax=(np.arange(0, b) + 1) * batch_size,
    colors=[cmap(i) for i in range(b)],
    alpha=.2,
)

_, p = tau_p(n, W, R)

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
        p,
        alpha=.5,
        label=r'$p$',
        color='red',
        linestyle='--',
    )

    ax.set_ylabel('$p$')

ax2.set_xlabel('samples')
# ax2.set_yticks(np.around(np.append(grand_mean, ci), decimals=3))
f.subplots_adjust(hspace=0)
ax1.set_title(f'2010 $n = {n}, W = {W}, m = {m}, R = {R}$')
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2010.p.n = {n}, W = {W}, m = {m}, R = {R}.pgf')

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

for i, (W, (m, R)) in enumerate(it.product([32, 128, 256], [(m, R) for m in [3, 5] for R in [m, m + 2]])):
    plt.plot(
        ns,
        [
            p
            for n in ns
            for _, p in [tau_p(n, W, R)]
        ],
        color=cmap(i),
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2010.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for station2successes_s in [np.where(successes[:, np.newaxis], contenders, 0).reshape(b, batch_size, n)]
        for station2collisions_s in [np.where(successes[:, np.newaxis], 0, contenders).reshape(b, batch_size, n)]
        for collision_p_s in [np.mean(
            np.sum(station2collisions_s, axis=1) / np.sum(station2collisions_s + station2successes_s, axis=1),
            axis=1
        )]
        for grand_mean in [np.mean(collision_p_s)]
        for ci in [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(collision_p_s), df=b - 1)]
    ])

    ax.errorbar(
        simulated[:, 0],
        simulated[:, 1],
        simulated[:, 2],
        marker=4,
        capsize=4,
        linestyle='',
        color=cmap(i),
    )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('2010')
ax.set_ylabel('$p$')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
        matplotlib.patches.Patch(
            color=cmap(i),
            label=f'$W = {W}, m = {m}, R = {R}$',
        )
        for i, (W, (m, R)) in enumerate(it.product([32, 128, 256], [(m, R) for m in [3, 5] for R in [m, m + 2]]))
    ]
)
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2010.multi-p.pgf')

# %%
f: plt.Figure
ax: plt.Axes
f, ax = plt.subplots(1, 1)

W: int = 128
m: int = 5
R: int = 5
samples: int = 10_000

for i, n in enumerate([5, 10, 15, 20, 30, 50]):
    logs: pd.DataFrame = pd.read_csv(f'assets/2010.n={n} W={W} m={m} R={R}.csv', nrows=samples)

    contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
    successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

    station2successes = np.where(successes[:, np.newaxis], contenders, 0)
    station2collisions = np.where(successes[:, np.newaxis], 0, contenders)

    collision_p: NDArray[float] = np.mean(
        np.cumsum(station2collisions, axis=0) / np.cumsum(station2collisions + station2successes, axis=0),
        axis=1,
    )

    ax.plot(range(samples), collision_p, label=f'$n = {n}$', color=cmap(i))

ax.set_ylabel('$p$')
ax.set_xlabel('contentions')
ax.grid(True, linestyle='--')
ax.legend()
ax.set_title(f'2010 $W = {W}, m = {m}$')

f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2010.p-init-bias.W = {W}, m = {m}, R = {R}.pgf')
