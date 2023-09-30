# %%
from matplotlib import colors
from numpy.typing import NDArray
from pathlib import Path
from scipy.optimize import fsolve, minimize_scalar, OptimizeResult

import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import typing as t

# %%
thrghpt: Path = Path('assets/throughput')
thrghpt.mkdir(parents=True, exist_ok=True)

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
        payload: int,
        slot_time: int,
        tau: float,
        Ts: float,
        Tc: float,
) -> float:
    Ptr: float = 1 - (1 - tau) ** n
    Ps: float = (n * tau * (1 - tau) ** (n - 1)) / Ptr
    return (Ps * Ptr * payload) / ((1 - Ptr) * slot_time + Ptr * Ps * Ts + Ptr * (1 - Ps) * Tc)  # bit / mus -> Mbit / s


# %%
def BAS_time_success(
        sifs: int,
        difs: int,
        phy_h: int,
        mac_h: int,
        payload: int,
        ack: int,
        channel_bit_rate: int,
        propagation_delay: int,
) -> float:
    return sifs + difs + (phy_h + mac_h + payload + ack) / channel_bit_rate + 2 * propagation_delay


def BAS_time_collision(
        difs: int,
        phy_h: int,
        mac_h: int,
        payload: int,
        channel_bit_rate: int,
        propagation_delay: int,
) -> float:
    return difs + (phy_h + mac_h + payload) / channel_bit_rate + propagation_delay


def RTSCTS_time_success(
        sifs: int,
        difs: int,
        phy_h: int,
        mac_h: int,
        payload: int,
        ack: int,
        channel_bit_rate: int,
        propagation_delay: int,
        rts: int,
        cts: int,
) -> float:
    return 3 * sifs + difs + (rts + cts + phy_h + mac_h + payload + ack) / channel_bit_rate + 4 * propagation_delay


def RTSCTS_time_collision(
        difs: int,
        channel_bit_rate: int,
        propagation_delay: int,
        rts: int,
) -> float:
    return difs + rts / channel_bit_rate + propagation_delay


# %%
payload: int = 8184
mac_h: int = 272
phy_h: int = 128
ack: int = 112 + phy_h
rts: int = 160 + phy_h
cts: int = 112 + phy_h

channel_bit_rate: int = 1  # Mbit/s -> bit/mus DO NOT CHANGE OTHERWISE TIMESPANS MUST RETURN FLOAT
propagation_delay: int = 1
slot_time: int = 50
sifs: int = 28
difs: int = 128
ack_to: int = 300
cts_to: int = 300

n: int = 5  # nodes
W: int = 2 ** 5
m: int = 5  # max # of attempts

# %%
samples: int = 200
logs: pd.DataFrame = pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=samples)

contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

spans: NDArray[int] = logs.iloc[:, 0] * slot_time
ts: NDArray[int] = np.where(
    successes,
    BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
    BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay),
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

ax1.set_ylabel('channel usage')
ax1.set_ylim(-.5, 1.5)
ax1.set_yticks([])
ax2.set_ylabel('throughput [Mbit/s]')
ax2.set_xlabel('time [s]')
ax2.grid(True, linestyle='--')

ax1.set_title(f'2000 BAS $n = {n}, W = {W}, m = {m}$')
f.subplots_adjust(hspace=0)
f.set_size_inches(w=3.5 * 2.5 * 2, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.BAS.throughput.n = {n}, W = {W}, m = {m}.pgf')

# %%
contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]

station2successes = np.where(successes[:, np.newaxis], contenders, 0)
station2collisions = np.where(successes[:, np.newaxis], 0, contenders)

station2collision_p: NDArray[float] = np.cumsum(station2collisions, axis=0) / np.cumsum(station2collisions + station2successes, axis=0)

f: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', height_ratios=[1, 3])

cmap: colors.ListedColormap = colors.ListedColormap(['#ED706B', 'white', '#6CCDAF'])

ax1.pcolor(station2successes.T - station2collisions.T, cmap=cmap, vmin=-1, vmax=1)

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

ax1.set_title(f'2000 $n = {n}, W = {W}, m = {m}$')
ax1.set_yticks([.5, 1.5, 2.5, 3.5, 4.5], [f'STA \#{i + 1}' for i in range(n)])
f.subplots_adjust(hspace=0)
f.set_size_inches(w=3.5 * 2.5 * 2, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.p.n = {n}, W = {W}, m = {m}.pgf')

# %%
b: int = 50 // 2
batch_size: int = 10_000 * 2

# %%
logs: pd.DataFrame = pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)

contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

spans: NDArray[int] = logs.iloc[:, 0] * slot_time
ts: NDArray[int] = np.where(
    successes,
    BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
    BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay),
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
    colors=['grey' for _ in range(b)],
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
            n=n, payload=payload, slot_time=slot_time, tau=tau_p(n=n, cw_min=W, m=m)[0],
            Ts=BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
            Tc=BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay)
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
ax1.set_title(f'2000 BAS $n = {n}, W = {W}, m = {m}$')
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.BAS.throughputs.n = {n}, W = {W}, m = {m}.pgf')

# %%
n: int = 30

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 10_000)

for i, (W, m) in enumerate(it.product([32, 128, 256], [3, 5])):
    ax.plot(
        ns,
        [
            S(
                n=n, payload=payload, slot_time=slot_time, tau=tau_p(n=n, cw_min=W, m=m)[0],
                Ts=BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
                Tc=BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay),
            )
            for n
            in ns
        ],
        color=f'C{i}',
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
            BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay),
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
        color=f'C{i}',
    )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('2000 BAS')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
        matplotlib.patches.Patch(
            color=f'C{i}',
            label=f'$W = {W}, m = {m}$'
        )
        for i, (W, m) in enumerate(it.product([32, 128, 256], [3, 5]))
    ]
)
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.BAS.multi-throughput.pgf')

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 10_000)

for i, (W, m) in enumerate(it.product([32, 128, 256], [3, 5])):
    ax.plot(
        ns,
        [
            S(
                n=n, payload=payload, slot_time=slot_time, tau=tau_p(n=n, cw_min=W, m=m)[0],
                Ts=RTSCTS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, rts, cts),
                Tc=RTSCTS_time_collision(difs, channel_bit_rate, propagation_delay, rts),
            )
            for n
            in ns
        ],
        color=f'C{i}',
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            RTSCTS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, rts, cts),
            RTSCTS_time_collision(difs, channel_bit_rate, propagation_delay, rts),
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
        color=f'C{i}',
    )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('2000 RTS/CTS')
ax.set_ylabel('saturation throughput [Mbit/s]')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
        matplotlib.patches.Patch(
            color=f'C{i}',
            label=f'$W = {W}, m = {m}$'
        )
        for i, (W, m) in enumerate(it.product([32, 128, 256], [3, 5]))
    ]
)
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.RTSCTS.multi-throughput.pgf')

# %%
b: int = 100
batch_size: int = 5_000

# %%
logs: pd.DataFrame = pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)

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
    colors=[f'C{i}' for i in range(b)],
    alpha=.2,
)

_, p = tau_p(n, W, m)

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
        label=r'$S$',
        color='red',
        linestyle='--',
    )

    ax.set_ylabel('$p$')

ax2.set_xlabel('samples')
# ax2.set_yticks(np.around(np.append(grand_mean, ci), decimals=3))
f.subplots_adjust(hspace=0)
ax1.set_title(f'2000 $n = {n}, W = {W}, m = {m}$')
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.p.n = {n}, W = {W}, m = {m}.pgf')

# %%
ax: plt.Axes
f, ax = plt.subplots(1, 1)

ns: NDArray[float] = np.linspace(5, 50, 10_000)

for i, (W, m) in enumerate(it.product([32, 128, 256], [3, 5])):
    plt.plot(
        ns,
        [
            p
            for n in ns
            for _, p in [tau_p(n, W, m)]
        ],
        color=f'C{i}',
        alpha=.5,
    )

    simulated: NDArray[...] = np.asarray([
        (n, grand_mean, grand_mean - ci[0])
        for n in [5, 10, 15, 20, 30, 50]
        for logs in [pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)]
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
        color=f'C{i}',
    )

ax.set_xticks(range(0, 51, 5))
ax.grid(True, linestyle='--')
ax.set_title('2000')
ax.set_ylabel('$p$')
ax.set_xlabel('STAs')
ax.legend(
    handles=[
        matplotlib.patches.Patch(
            color=f'C{i}',
            label=f'$W = {W}, m = {m}$'
        )
        for i, (W, m) in enumerate(it.product([32, 128, 256], [3, 5]))
    ]
)
f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.multi-p.pgf')

# %%
f: plt.Figure
ax: plt.Axes
f, ax = plt.subplots(1, 1)

W: int = 128
m: int = 5
samples: int = 10_000

for n in [5, 10, 15, 20, 30, 50]:
    logs: pd.DataFrame = pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=samples)

    contenders: pd.DataFrame = logs.iloc[:, 1:1 + n]
    successes: NDArray[bool] = np.count_nonzero(contenders, 1) == 1

    station2successes = np.where(successes[:, np.newaxis], contenders, 0)
    station2collisions = np.where(successes[:, np.newaxis], 0, contenders)

    collision_p: NDArray[float] = np.mean(
        np.cumsum(station2collisions, axis=0) / np.cumsum(station2collisions + station2successes, axis=0),
        axis=1,
    )

    ax.plot(range(samples), collision_p, label=f'$n = {n}$')

ax.set_ylabel('$p$')
ax.set_xlabel('contentions')
ax.grid(True, linestyle='--')
ax.legend()
ax.set_title(f'2000 $W = {W}, m = {m}$')

f.set_size_inches(w=3.5 * 2.5, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.p-init-bias.W = {W}, m = {m}.pgf')

# %%
m: int = 6

# %%
f: plt.Figure
ax: plt.Axes
f, (ax1, ax2) = plt.subplots(1, 2, sharey='all')

for i, n in enumerate([5, 10, 20, 50]):
    simulated: NDArray[...] = np.asarray([
        (x, grand_mean, grand_mean - ci[0])
        for x, W in enumerate(2 ** np.arange(3, 3 + 8))
        for logs in [pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
            BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay),
        )]
        for success in [successes * payload]  # bit
        for span_end in [spans + ts]  # mus
        for success_s in [success.reshape(b, batch_size)]  # bit
        for span_end_s in [span_end.to_numpy().reshape(b, batch_size)]  # mus
        for throughput_s in [np.sum(success_s, 1) / np.sum(span_end_s, 1)]  # bit/mus -> Mbit/s
        for grand_mean in [np.mean(throughput_s)]
        for ci in [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(throughput_s), df=b - 1)]
    ])

    ax1.errorbar(
        simulated[:, 0],
        simulated[:, 1],
        simulated[:, 2],
        marker=4,
        capsize=4,
        color=f'C{i}',
        label=f'$n = {n}$',
    )

    pd.DataFrame(simulated[:, 1:], columns=['S', 'CI']).to_csv(thrghpt / f'BAS.maximum.n={n} m={m}.csv')

    simulated: NDArray[...] = np.asarray([
        (x, grand_mean, grand_mean - ci[0])
        for x, W in enumerate(2 ** np.arange(3, 3 + 8))
        for logs in [pd.read_csv(f'assets/2000.n={n} W={W} m={m}.csv', nrows=b * batch_size)]
        for contenders in [logs.iloc[:, 1:1 + n]]
        for successes in [np.count_nonzero(contenders, 1) == 1]
        for spans in [logs.iloc[:, 0] * slot_time]
        for ts in [np.where(
            successes,
            RTSCTS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, rts, cts),
            RTSCTS_time_collision(difs, channel_bit_rate, propagation_delay, rts),
        )]
        for success in [successes * payload]  # bit
        for span_end in [spans + ts]  # mus
        for success_s in [success.reshape(b, batch_size)]  # bit
        for span_end_s in [span_end.to_numpy().reshape(b, batch_size)]  # mus
        for throughput_s in [np.sum(success_s, 1) / np.sum(span_end_s, 1)]  # bit/mus -> Mbit/s
        for grand_mean in [np.mean(throughput_s)]
        for ci in [sp.stats.t.interval(confidence=.95, loc=grand_mean, scale=sp.stats.sem(throughput_s), df=b - 1)]
    ])

    ax2.errorbar(
        simulated[:, 0],
        simulated[:, 1],
        simulated[:, 2],
        marker=4,
        capsize=4,
        color=f'C{i}',
        label=f'$n = {n}$',
    )

    pd.DataFrame(simulated[:, 1:], columns=['S', 'CI']).to_csv(thrghpt / f'RTSCTS.maximum.n={n} m={m}.csv')

for ax in [ax1, ax2]:
    ax.set_xticks(range(8), 2 ** np.arange(3, 3 + 8))
    ax.set_xlabel('$W$')
    ax.grid(True, linestyle='--')

ax1.set_title(f'BAS')
ax1.set_ylabel('saturation throughput [Mbit/s]')

ax2.set_title(f'RTS/CTS')
ax2.legend()

f.suptitle(f'2000 $m = {m}')
f.subplots_adjust(wspace=.05)
f.set_size_inches(w=3.5 * 2.5 * 2, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.throughput-vs-W.m = {m}.pgf')

# %%
f: plt.Figure
ax: plt.Axes
f, (ax1, ax2) = plt.subplots(1, 2, sharey='all')

fn1: t.Callable[[float], float] = lambda tau: S(
    n=n, payload=payload, slot_time=slot_time, tau=tau,
    Ts=BAS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay),
    Tc=BAS_time_collision(difs, phy_h, mac_h, payload, channel_bit_rate, propagation_delay),
)
fn2: t.Callable[[float], float] = lambda tau: S(
    n=n, payload=payload, slot_time=slot_time, tau=tau,
    Ts=RTSCTS_time_success(sifs, difs, phy_h, mac_h, payload, ack, channel_bit_rate, propagation_delay, rts, cts),
    Tc=RTSCTS_time_collision(difs, channel_bit_rate, propagation_delay, rts),
)

for i, n in enumerate([5, 10, 20, 50]):
    xs: NDArray[float] = np.linspace(-.2, .3, 30_000)

    for ax, fn in zip([ax1, ax2], [fn1, fn2]):
        ax.plot(
            xs,
            fn(xs),
            color=f'C{i}',
            alpha=.5,
            label=f'$n = {n}$',
        )

        xopt: OptimizeResult = minimize_scalar(
            fun=lambda x: -fn(x),
            bounds=(0, 1),
        )
        ax.scatter(
            xopt.x,
            -xopt.fun,
            marker='2',
            color=f'C{i}',
        )

    for ax, mode in zip([ax1, ax2], ['BAS', 'RTSCTS']):
        simulated: pd.DataFrame = pd.read_csv(thrghpt / f'{mode}.maximum.n={n} m={m}.csv')

        idxmax: int = simulated['S'].idxmax()
        maximum: pd.DataFrame = simulated.loc[idxmax]

        ax.errorbar(
            tau_p(n, 2 ** (3 + idxmax), m)[0],
            maximum['S'],
            maximum['CI'],
            marker=4,
            capsize=4,
            linestyle='',
            color=f'C{i}',
        )

for ax in [ax1, ax2]:
    ax.set_ylim(.4, .9)
    ax.set_xlabel(r'$\tau$')
    ax.grid(True, linestyle='--')

ax1.set_title(f'BAS')
ax1.set_xlim(0, .1)
ax1.set_ylabel('saturation throughput [Mbit/s]')

ax2.set_title(f'RTS/CTS')
ax2.set_xlim(0, .25)

ax2.scatter(-1, 0, marker='2', color='black', label='model', alpha=.5),
ax2.errorbar(-1, 0, 1, marker=4, capsize=4, linestyle='', color='black', label='simulated', alpha=.5)
ax2.legend()

f.suptitle(f'2000 $m = {m}')
f.subplots_adjust(wspace=.05)
f.set_size_inches(w=3.5 * 2.5 * 2, h=4.8 * 3.5 * 2.5 / 6.4)
f.savefig(out / f'2000.model.throughput-vs-W.m = {m}.pgf')
