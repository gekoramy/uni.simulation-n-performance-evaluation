# %%
from dataclasses import dataclass
from numpy.random import SeedSequence
from numpy.typing import NDArray
from pathlib import Path
from tqdm.auto import tqdm

import more_itertools as mit
import numpy as np
import pandas as pd
import typing as t

# %%
out: Path = Path('assets')
out.mkdir(parents=True, exist_ok=True)


# %%
@dataclass
class Log:
    span: int
    contenders: NDArray[int]
    attempt: NDArray[int]


def simulation(
        seeds: t.Iterator[NDArray[np.uint32]],
        n: int,
        W: int,
        m: int,
        ack_timeout: int = 6,
) -> t.Iterator[Log]:
    rng4backoff: np.random.Generator = np.random.default_rng(next(seeds))

    timeouts: NDArray[bool] = np.full(n, fill_value=False)
    retries: NDArray[int] = np.zeros(n, dtype=int)
    waiting: NDArray[int] = np.zeros(n, dtype=int)

    while True:

        span: int = np.amin(waiting + timeouts * ack_timeout)
        mask: NDArray[bool] = (waiting + timeouts * ack_timeout) == span
        contenders: NDArray[int] = np.flatnonzero(mask)

        yield Log(
            span=span,
            contenders=mask,
            attempt=retries.copy(),
        )

        waiting = waiting + np.minimum(timeouts * ack_timeout, span) - span - 1

        match contenders.size:

            case 1:
                timeouts = np.full(n, fill_value=False)
                retries[contenders] = 0

            case _:
                timeouts = mask
                retries[contenders] += 1
                retries.clip(max=m, out=retries)

        waiting[contenders] = rng4backoff.integers(
            low=0,
            high=W * 2 ** retries[contenders],
            endpoint=False,
        )


# %%
shop: t.Iterator[NDArray[np.uint32]] = iter(SeedSequence(17).generate_state(10_000))

# %%
for n, W, m in tqdm([(n, W, m) for n in [5, 10, 15, 20, 30, 50] for W in [32, 128, 256] for m in [3, 5]]):
    ls: list[Log] = mit.take(
        100 * 5_000,
        simulation(shop, n=n, W=W, m=m)
    )

    spans, contenders, attempts = zip(*[(x.span, x.contenders, x.attempt) for x in ls])

    pd.concat(
        [
            pd.DataFrame(spans, columns=['span']),
            pd.concat([
                pd.DataFrame(contenders),
                pd.DataFrame(attempts),
            ], axis=1, keys=['contenders', 'attempt'])
        ]
        , axis=1
    ).to_csv(out / f'2000.n={n} W={W} m={m}.csv', index=False)

# %%
for n, W, m in tqdm([(n, W, m) for n in [5, 10, 20, 50] for W in [ 2 ** (3 + i) for i in range(8)] for m in [6]]):
    ls: list[Log] = mit.take(
        100 * 5_000,
        simulation(shop, n=n, W=W, m=m)
    )

    spans, contenders, attempts = zip(*[(x.span, x.contenders, x.attempt) for x in ls])

    pd.concat(
        [
            pd.DataFrame(spans, columns=['span']),
            pd.concat([
                pd.DataFrame(contenders),
                pd.DataFrame(attempts),
            ], axis=1, keys=['contenders', 'attempt'])
        ]
        , axis=1
    ).to_csv(out / f'2000.n={n} W={W} m={m}.csv', index=False)
