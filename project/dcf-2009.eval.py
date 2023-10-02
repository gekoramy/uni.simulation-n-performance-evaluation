# %%
from matplotlib.colors import Colormap, ListedColormap
from numpy.typing import NDArray
from pathlib import Path
from scipy.optimize import fsolve

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

width: float = 8.75
height: float = 6.25

cmap: Colormap = matplotlib.colormaps['Set1']


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


def tau_p_revised(n: int, W: int, m: int, R: int) -> tuple[float, float]:
    def system(x: tuple[float, float]) -> tuple[float, float]:
        tau, p = x
        return (
            - p + 1 - (1 - tau) ** (n - 1),
            - tau + 1 / (1 + (1 - p) / (2 * (1 - p ** (R + 1))) * sum([
                p ** j * (W * 2 ** min(m, j) - 1) - (1 - p ** (R + 1))
                for j in range(R + 1)
            ]))
        )

    return fsolve(system, x0=np.full(2, .5))


def S(
        n: int,
        W: int,
        payload: int,
        slot_time: int,
        tau: float,
        Ts: float,
        Tc: float,
) -> float:
    Pb: float = 1 - (1 - tau) ** n
    Ps: float = n * tau * (1 - tau) ** (n - 1)
    sEP: float = payload * W / (W - 1)
    sTs: float = Ts * W / (W - 1) + slot_time
    sTc: float = Tc + slot_time
    return (Ps * sEP) / ((1 - Pb) * slot_time + Ps * sTs + (Pb - Ps) * sTc)

# %%
payload: int = 8 * 1023
mac_h: int = 8 * 34
phy_h: int = 8 * 16
mpdu: int = mac_h + phy_h + payload
ack: int = 8 * 14 + phy_h
rts: int = 8 * 20 + phy_h
cts: int = 8 * 14 + phy_h

channel_bit_rate: int = 1  # Mbit/s -> bit/mus

slot_time: int = 50
sifs: int = 28
difs: int = sifs + 2 * slot_time

n: int = 5  # # of STAs
W: int = 2 ** 4  # W = W min
m: int = 5  # W max = W * 2 ** m
R: int = m  # max # of retries

# %%
BAS_time_success: float = (mpdu + ack) / channel_bit_rate + sifs + difs
BAS_time_collision: float = BAS_time_success

RTSCTS_time_success: float = 3 * sifs + difs + (rts + cts + mpdu + ack) / channel_bit_rate
RTSCTS_time_collision: float = sifs + difs + (rts + ack) / channel_bit_rate

# %%
samples: int = 200
logs: pd.DataFrame = pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=samples)

contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

spans: NDArray[int] = logs.iloc[:, 0] * slot_time
ts: NDArray[int] = np.where(
    successes,
    BAS_time_success,
    BAS_time_collision + slot_time,
)

# %%
merge: NDArray[int] = np.empty(ts.shape[0] * 2, dtype=int)
merge[0::2] = spans
merge[1::2] = ts

mmm: NDArray[int] = np.repeat(np.where(successes, 1, -1), 2)

success: NDArray[int] = successes * payload  # bit

span_end: NDArray[int] = spans + ts  # mus

throughput: NDArray[float] = np.cumsum(success) / np.cumsum(span_end)  # bit/mus -> Mbit/s

f: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[1, 5])

ax1.fill_between(np.cumsum(merge) / 1e6, 1, where=mmm > 0, color='#6CCDAF')
ax1.fill_between(np.cumsum(merge) / 1e6, 1, where=mmm < 0, color='#ED706B')

ax2.plot(np.cumsum(span_end) / 1e6, throughput)

ax1.set_ylabel('channel\nusage')
ax1.set_ylim(-.5, 1.5)
ax1.set_yticks([])
ax2.set_ylabel('throughput [Mbit/s]')
ax2.set_xlabel('time [s]')
ax2.grid(True, linestyle='--')

ax1.set_title(f'2009 BAS $n = {n}, W = {W}, m = {m}, R = {R}$')
f.subplots_adjust(hspace=0)
f.set_size_inches(w=width * 2, h=height * 2 / 3)
f.savefig(out / f'2009.BAS.throughput.n = {n}, W = {W}, m = {m}, R = {R}.pgf', bbox_inches='tight')

# %%
contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]

station2successes = np.where(successes[:, np.newaxis], contenders, 0)
station2collisions = np.where(successes[:, np.newaxis], 0, contenders)

station2collision_p: NDArray[float] = np.cumsum(station2collisions, axis=0) / np.cumsum(station2collisions + station2successes, axis=0)

f: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[1, 3])

ax1.pcolor(
    station2successes.T - station2collisions.T,
    cmap=ListedColormap(['#ED706B', 'white', '#6CCDAF']),
    vmin=-1,
    vmax=1
)

for collision_p in station2collision_p.T:
    ax2.plot(
        range(samples),
        collision_p,
        color='grey',
        alpha=.2,
    )

ax2.plot(range(samples), np.mean(station2collision_p, axis=1))

ax2.set_ylabel('$p$')
ax2.set_xlabel('contentions')
ax2.grid(True, axis='y', linestyle='--')

ax1.set_title(f'2009 $n = {n}, W = {W}, m = {m}, R = {R}$')
ax1.set_yticks([.5, 1.5, 2.5, 3.5, 4.5], [f'STA \#{i + 1}' for i in range(n)])
f.subplots_adjust(hspace=0)
f.set_size_inches(w=width * 2, h=height * 2 / 3)
f.savefig(out / f'2009.p.n = {n}, W = {W}, m = {m}, R = {R}.pgf', bbox_inches='tight')

# %%
n: int = 30
W: int = 2 ** 7
b: int = 50 // 2
batch_size: int = 10_000 * 2

# %%
logs: pd.DataFrame = pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)

contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

spans: NDArray[int] = logs.iloc[:, 0] * slot_time
ts: NDArray[int] = np.where(
    successes,
    BAS_time_success,
    BAS_time_collision + slot_time,
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
    colors=['grey'] * b,
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
            n=n, W=W, payload=payload, slot_time=slot_time, tau=tau_p(n=n, W=W, R=R)[0],
            Ts=BAS_time_success,
            Tc=BAS_time_collision
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
ax1.set_title(f'2009 BAS $n = {n}, W = {W}, m = {m}, R = {R}$')
f.set_size_inches(w=width, h=height)
f.savefig(out / f'2009.BAS.throughputs.n = {n}, W = {W}, m = {m}, R = {R}.pgf', bbox_inches='tight')

# %%
b: int = 500
batch_size: int = 1_000

# %%
for W in [16, 32, 128]:

    ax: plt.Axes
    f, ax = plt.subplots(1, 1)

    ns: NDArray[float] = np.linspace(5, 50, 1_000)

    for i, (m, R) in enumerate([(m, R) for m in [3, 5] for R in [m, m + 4]]):
        ax.plot(
            ns,
            [
                S(
                    n=n, W=W, payload=payload, slot_time=slot_time,
                    tau=tau_p(n, W, R)[0],
                    Ts=BAS_time_success,
                    Tc=BAS_time_collision
                )
                for n
                in ns
            ],
            color=cmap(i),
            alpha=.5,
        )

        ax.plot(
            ns,
            [
                S(
                    n=n, W=W, payload=payload, slot_time=slot_time,
                    tau=tau_p_revised(n, W, m, R)[0],
                    Ts=BAS_time_success,
                    Tc=BAS_time_collision
                )
                for n
                in ns
            ],
            color=cmap(i),
            linestyle='--',
            alpha=.5,
        )

        simulated: NDArray[...] = np.asarray([
            (n, grand_mean, grand_mean - ci[0])
            for n in [5, 10, 15, 20, 30, 50]
            for logs in [pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
            for contenders in [logs.iloc[:, 1:1 + n]]
            for successes in [np.count_nonzero(contenders, 1) == 1]
            for spans in [logs.iloc[:, 0] * slot_time]
            for ts in [np.where(
                successes,
                BAS_time_success,
                BAS_time_collision + slot_time,
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
    ax.set_title(f'2009 BAS $W = {W}$')
    ax.set_ylabel('saturation throughput [Mbit/s]')
    ax.set_xlabel('STAs')
    ax.legend(
        handles=[
            matplotlib.patches.Patch(
                color=cmap(i),
                label=f'$m = {m}, R={R}$'
            )
            for i, (m, R) in enumerate([(m, R) for m in [3, 5] for R in [m, m + 4]])
        ]
    )
    f.set_size_inches(w=width, h=height)
    f.savefig(out / f'2009.BAS.multi-throughput.W = {W}.pgf', bbox_inches='tight')

# %%
for W in [16, 32, 128]:

    ax: plt.Axes
    f, ax = plt.subplots(1, 1)

    ns: NDArray[float] = np.linspace(5, 50, 1_000)

    for i, (m, R) in enumerate([(m, R) for m in [3, 5] for R in [m, m + 4]]):
        ax.plot(
            ns,
            [
                S(
                    n=n, W=W, payload=payload, slot_time=slot_time,
                    tau=tau_p(n, W, R)[0],
                    Ts=RTSCTS_time_success,
                    Tc=RTSCTS_time_collision,
                )
                for n
                in ns
            ],
            color=cmap(i),
            alpha=.5,
        )

        ax.plot(
            ns,
            [
                S(
                    n=n, W=W, payload=payload, slot_time=slot_time,
                    tau=tau_p_revised(n, W, m, R)[0],
                    Ts=RTSCTS_time_success,
                    Tc=RTSCTS_time_collision,
                )
                for n
                in ns
            ],
            color=cmap(i),
            linestyle='--',
            alpha=.5,
        )

        simulated: NDArray[...] = np.asarray([
            (n, grand_mean, grand_mean - ci[0])
            for n in [5, 10, 15, 20, 30, 50]
            for logs in [pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
            for contenders in [logs.iloc[:, 1:1 + n]]
            for successes in [np.count_nonzero(contenders, 1) == 1]
            for spans in [logs.iloc[:, 0] * slot_time]
            for ts in [np.where(
                successes,
                RTSCTS_time_success,
                RTSCTS_time_collision + slot_time,
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
    ax.set_title(f'2009 RTS/CTS $W = {W}$')
    ax.set_ylabel('saturation throughput [Mbit/s]')
    ax.set_xlabel('STAs')
    ax.legend(
        handles=[
            matplotlib.patches.Patch(
                color=cmap(i),
                label=f'$m = {m}, R = {R}$'
            )
            for i, (m, R) in enumerate([(m, R) for m in [3, 5] for R in [m, m + 4]])
        ]
    )
    f.set_size_inches(w=width, h=height)
    f.savefig(out / f'2009.RTSCTS.multi-throughput.W = {W}.pgf', bbox_inches='tight')

# %%
b: int = 100
batch_size: int = 5_000

# %%
logs: pd.DataFrame = pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)

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
_, p_revised = tau_p_revised(n, W, m, R)

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
    ax.axhline(
        p_revised,
        alpha=.5,
        label=r'$p\'$',
        color='purple',
        linestyle='--',
    )

    ax.set_ylabel('$p$')

ax2.set_xlabel('samples')
# ax2.set_yticks(np.around(np.append(grand_mean, ci), decimals=3))
f.subplots_adjust(hspace=0)
ax1.set_title(f'2009 $n = {n}, W = {W}, m = {m}, R = {R}$')
f.set_size_inches(w=width, h=height)
f.savefig(out / f'2009.p.n = {n}, W = {W}, m = {m}, R = {R}.pgf', bbox_inches='tight')

# %%
for W in [16, 32, 128]:

    ax: plt.Axes
    f, ax = plt.subplots(1, 1)

    ns: NDArray[float] = np.linspace(5, 50, 1_000)

    for i, (m, R) in enumerate([(m, R) for m in [3, 5] for R in [m, m + 4]]):
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

        plt.plot(
            ns,
            [
                p
                for n in ns
                for _, p in [tau_p_revised(n, W, m, R)]
            ],
            color=cmap(i),
            linestyle='--',
            alpha=.5,
        )

        simulated: NDArray[...] = np.asarray([
            (n, grand_mean, grand_mean - ci[0])
            for n in [5, 10, 15, 20, 30, 50]
            for logs in [pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
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
    ax.set_title(f'2009 $W = {W}$')
    ax.set_ylabel('$p$')
    ax.set_xlabel('STAs')
    ax.legend(
        handles=[
            matplotlib.patches.Patch(
                color=cmap(i),
                label=f'$m = {m}, R = {R}$',
            )
            for i, (m, R) in enumerate([(m, R) for m in [3, 5] for R in [m, m + 4]])
        ]
    )
    f.set_size_inches(w=width, h=height)
    f.savefig(out / f'2009.multi-p.W = {W}.pgf', bbox_inches='tight')

# %%
f: plt.Figure
ax: plt.Axes
f, ax = plt.subplots(1, 1)

W: int = 128
m: int = 5
R: int = 5
samples: int = 10_000

for i, n in enumerate([5, 10, 15, 20, 30, 50]):
    logs: pd.DataFrame = pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=samples)

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
ax.set_title(f'2009 $W = {W}, m = {m}$')

f.set_size_inches(w=width, h=height)
f.savefig(out / f'2009.p-init-bias.W = {W}, m = {m}, R = {R}.pgf', bbox_inches='tight')

# %%
b: int = 500
batch_size: int = 10_000
R: int = 9
W: int = 2 ** 5

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

ax.plot(
    ns,
    [
        S(
            n=n, W=W, payload=payload, slot_time=slot_time,
            tau=tau_p(n, W, R)[0],
            Ts=BAS_time_success,
            Tc=BAS_time_collision
        )
        for n
        in ns
    ],
    color='black',
    alpha=.5,
)

for i, m in enumerate([3, 5, 7, 9]):
    ax.plot(
        ns,
        [
            S(
                n=n, W=W, payload=payload, slot_time=slot_time,
                tau=tau_p_revised(n, W, m, R)[0],
                Ts=BAS_time_success,
                Tc=BAS_time_collision
            )
            for n
            in ns
        ],
        color=cmap(i),
        linestyle='--',
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            BAS_time_success,
            BAS_time_collision + slot_time,
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
ax.set_title(f'2009 BAS $W = {W}, R = {R}$')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
                matplotlib.patches.Patch(
                    color='black',
                    label='Official',
                )
            ] + [
                matplotlib.patches.Patch(
                    color=cmap(i),
                    label=f'$m = {m}$'
                )
                for i, m in enumerate([3, 5, 7, 9])
            ]
)
f.set_size_inches(w=width, h=height)
f.savefig(out / f'2009.BAS.focus-throughput.W = {W}, R = {R}.pgf', bbox_inches='tight')

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

ax.plot(
    ns,
    [
        S(
            n=n, W=W, payload=payload, slot_time=slot_time,
            tau=tau_p(n, W, R)[0],
            Ts=RTSCTS_time_success,
            Tc=RTSCTS_time_collision,
        )
        for n
        in ns
    ],
    color='black',
    alpha=.5,
)

for i, m in enumerate([3, 5, 7, 9]):
    ax.plot(
        ns,
        [
            S(
                n=n, W=W, payload=payload, slot_time=slot_time,
                tau=tau_p_revised(n, W, m, R)[0],
                Ts=RTSCTS_time_success,
                Tc=RTSCTS_time_collision,
            )
            for n
            in ns
        ],
        color=cmap(i),
        linestyle='--',
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            RTSCTS_time_success,
            RTSCTS_time_collision + slot_time,
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
ax.set_title(f'2009 RTS/CTS $W = {W}, R = {R}$')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
                matplotlib.patches.Patch(
                    color='black',
                    label='Official',
                )
            ] + [
                matplotlib.patches.Patch(
                    color=cmap(i),
                    label=f'$m = {m}$'
                )
                for i, m in enumerate([3, 5, 7, 9])
            ]
)
f.set_size_inches(w=width, h=height)
f.savefig(out / f'2009.RTSCTS.focus-throughput.W = {W}, R = {R}.pgf', bbox_inches='tight')

# %%
b: int = 500
batch_size: int = 10_000

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 1_000)

plt.plot(
    ns,
    [
        p
        for n in ns
        for _, p in [tau_p(n, W, R)]
    ],
    color='black',
    alpha=.5,
)

for i, m in enumerate([3, 5, 7, 9]):
    plt.plot(
        ns,
        [
            p
            for n in ns
            for _, p in [tau_p_revised(n, W, m, R)]
        ],
        color=cmap(i),
        linestyle='--',
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2009.n={n} W={W} m={m} R={R}.csv', nrows=b * batch_size)]
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
ax.set_title(f'2009 $W = {W}, R = {R}$')
ax.set_ylabel('$p$')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
                matplotlib.patches.Patch(
                    color='black',
                    label='Official',
                )
            ] + [
                matplotlib.patches.Patch(
                    color=cmap(i),
                    label=f'$m = {m}$',
                )
                for i, m in enumerate([3, 5, 7, 9])
            ]
)
f.set_size_inches(w=width, h=height)
f.savefig(out / f'2009.focus-p.W = {W}, R = {R}.pgf', bbox_inches='tight')
