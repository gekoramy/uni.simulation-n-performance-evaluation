# IEEE 802.11 DCF - DES against SOTA analytical model

[Report](report.pdf)

|              |                                @ |
|--------------|---------------------------------:|
| Luca Mosetti | luca.mosetti-1@studenti.unitn.it |
| Shandy Darma |   shandy.darma@studenti.unitn.it |

## DES

```sh
python dcf-2000.simulator.py
python dcf-2009.simulator.py
```

This command runs several simulations with different configurations.
These simulations are then stored in csv format, in `./assets`.

`dcf-2000.simulator.py` requires 5 parameters:

1. `seed`
2. `n`, number of stations
3. `W`, minimum contention window
4. `m`, maximum backoff stage
5. `ack_timeout`, ack timeout in slot time unit

`dcf-2009.simulator.py` requires 5 parameters:

1. `seed`
2. `n`, number of stations
3. `W`, minimum contention window
4. `m`, maximum backoff stage
5. `R`, maximum number of retries

> Note: the simulation configurations are hard-coded

## Metric analysis

```sh
python dcf-2000.eval.py
python dcf-2009.eval.py
```

This command computes several comparison analyses based on the csv files in `./assets`.
These analyses are then stored in pgf format, in `./assets/graph`.

> Note: the simulation configurations expected from the analysis scripts are hard-coded
