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
        R: int,
) -> t.Iterator[Log]:
    rng4backoff: np.random.Generator = np.random.default_rng(next(seeds))

    retries: NDArray[int] = np.full(n, R, dtype=int)
    waiting: NDArray[int] = np.zeros(n, dtype=int)

    while True:

        span: int = np.amin(waiting)
        mask: NDArray[bool] = waiting == span
        contenders: NDArray[int] = np.flatnonzero(mask)

        yield Log(
            span=span,
            contenders=mask,
            attempt=retries.copy(),
        )

        waiting = waiting - span

        match contenders.size:

            case 1:
                retries[contenders] = 0

            case _:
                waiting -= 1
                retries[contenders] += 1
                retries[retries > R] = 0

        waiting[contenders] = rng4backoff.integers(
            low=0,
            high=W * 2 ** np.minimum(m, retries[contenders]),
            endpoint=False,
        )


# %%
shop: t.Iterator[NDArray[np.uint32]] = iter(SeedSequence(17).generate_state(10_000))

# %%
for n, W, m, R in tqdm([(n, W, m, R) for n in [5, 10, 15, 20, 30, 50] for W in [32, 128, 256] for m in [3, 5] for R in [m, m + 2]]):
    ls: list[Log] = mit.take(
        100 * 5_000,
        simulation(shop, n=n, W=W, m=m, R=R)
    )

    pd.concat(
        [
            pd.DataFrame([x.span for x in ls], columns=['span']),
            pd.concat([
                pd.DataFrame([x.contenders for x in ls]),
                pd.DataFrame([x.attempt for x in ls]),
            ], axis=1, keys=['contenders', 'attempt'])
        ]
        , axis=1
    ).to_csv(out / f'2009.n={n} W={W} m={m} R={R}.csv', index=False)

# %%
for n, W, m, R in tqdm([(n, W, m, R) for n in [5, 10, 15, 20, 30, 50] for W in [32] for m in [3, 5, 7, 9] for R in [9]]):
    ls: list[Log] = mit.take(
        1_000 * 5_000,
        simulation(shop, n=n, W=W, m=m, R=R)
    )

    pd.concat(
        [
            pd.DataFrame([x.span for x in ls], columns=['span']),
            pd.concat([
                pd.DataFrame([x.contenders for x in ls]),
                pd.DataFrame([x.attempt for x in ls]),
            ], axis=1, keys=['contenders', 'attempt'])
        ]
        , axis=1
    ).to_csv(out / f'2009.n={n} W={W} m={m} R={R}.csv', index=False)
