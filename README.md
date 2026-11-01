# Controlled Islanding and Reconnection with Stability Guarantees

Python reference implementation of the algorithms in

> **Controlled Islanding and Reconnection with Stability Guarantees via Submodular Optimization**
> Shiyu Cheng and Andrew Clark — [arXiv:2609.12335](https://arxiv.org/abs/2609.12335)

The repository contains:

- the proposed spanning-tree local search;
- a counterexample-guided stability refinement loop;
- a Gurobi-based MILP baseline;
- the benchmark instances;
- the scripts that run the experiments and render their figures and tables.

## Overview

When a local fault makes a networked dynamical system unstable, the fault can spread through
the interconnections. **Controlled islanding** contains it by shedding faulty nodes and cutting
coupling links, so that each remaining island is stable on its own. **Reconnection** then
restores some of the cut links to merge islands back together, as long as the merged system
stays stable.

The goal is to choose the shed nodes, cut edges, and reconnected edges together, minimizing
economic cost while guaranteeing stability of every island and every reconnected component.
Two things make this hard:

- the number of possible partitions grows exponentially with network size;
- stability is a property of the system's dynamics, not of the graph alone.

## Method

```
 network + faults
        │
        ▼
 augmented graph ──► local search over spanning trees ──► candidate partition
                              ▲                                   │
                              │  add counterexample states        ▼
                              └────────────────────── exact stability check
                                                                  │ no counterexample
                                                                  ▼
                                              certified islanding + reconnection
```

1. **Graphic-matroid encoding.** The network is extended into an augmented graph:
   - a source supernode (stable side) and a sink supernode (shed side) are added;
   - each original edge is replaced by a three-edge relay path.

   The spanning trees of the augmented graph, which are the bases of its graphic matroid,
   induce the admissible islanding-and-reconnection pairs. Each spanning tree that contains the
   source–sink edge decodes into stable islands, shed nodes, cut edges, and reconnected edges.
2. **Sampled objective.** The objective adds economic cost to two stability penalties. The
   penalties measure violations of Lyapunov-based sufficient conditions, one for islands and
   one for reconnection, evaluated on finite sets of sampled states.
3. **Local search (inner loop).** The search starts from a seeded spanning tree and repeatedly
   applies the best single-edge swap that keeps the tree a spanning tree. A swap is accepted
   only if it improves the objective by more than a factor `(1 + ε)`.
4. **Counterexample-guided refinement (outer loop).** Each partition the inner loop returns is
   checked exactly for stability:
   - every unstable island or reconnected component yields counterexample states;
   - those states are added to the sample sets, and the inner problem is solved again;
   - the loop ends when no counterexample remains.
5. **MILP baseline.** The same sampled inner problem is formulated as a mixed-integer linear
   program and solved with Gurobi. It runs inside the same outer loop, for a like-for-like
   comparison with the local search.

### Theoretical guarantees and their scope

- **Supermodular structure.** With the sample sets fixed, the paper shows that the inner problem
  admits an equivalent **nonincreasing supermodular** formulation over the graphic-matroid
  encoding.
- **Local-search guarantee.** On that basis, the paper establishes a **performance guarantee for
  the local search on the inner problem with fixed sample sets**, under the conditions stated
  in the paper. The guarantee does not extend to the full stability-constrained problem.
- **Zero objective.** The sampled objective is nonnegative. If the local search reaches zero,
  the partition is globally optimal **for that sampled inner problem**, but not necessarily for
  the complete stability-constrained problem.
- **Stability certification.** Stability of the returned partition comes from the outer loop's
  exact check: a partition is certified only when the loop terminates with no counterexample.

### General framework vs. this implementation

The paper's framework applies to networked dynamical systems with Lyapunov-based stability
conditions. The code mirrors this with a generic `SystemModel` interface that takes dynamics,
Lyapunov terms, a sampler, and a counterexample oracle.

The only model implemented here is **grounded linear consensus**. For this model the exact
stability check is an eigenvalue test on each unit's grounded Laplacian: an island or component
is stable when the smallest eigenvalue of `M = D + L` is positive. Eigenvectors with
non-positive eigenvalues serve as the counterexamples.

### Paper ↔ code notation

| Paper | Code (`islanding/cost.py`) | Meaning |
|---|---|---|
| F1 | `f1 + f3 − f4` (printed as *economic cost*) | shed-node cost + cut-edge cost − reconnection credit |
| F2 | `f5` (`f5_stability_penalty`) | sampled island-stability penalty |
| F3 | `f6` (`f6_reconnection_stability`) | sampled reconnection-stability penalty |

## Quick Start

**Requirements:** Python ≥ 3.11. A Gurobi license is needed only for the MILP baseline.

```bash
git clone https://github.com/shinyuc/controlled_islanding.git
cd controlled_islanding
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the 10-node example with the proposed local search (no Gurobi license needed):

```bash
python -m examples.linear_10node.run_linear_10node
```

Key lines of the output (abridged):

```
reason:                     converged
Stable islands (2):
  [1, 4, 5, 6, 8, 9]  lambda_min = 1.0000
  [2, 3, 7]  lambda_min = 1.0000
Unstable islands (1):
  [10]
Reconnected edges: [(2, 5), (2, 6), (6, 7)]
Reconnected components (1):
  [1, 2, 3, 4, 5, 6, 7, 8, 9]  lambda_min = 1.0000
Economic cost (F1 + F3 - F4):    6.9
```

- The faulty node 10 is shed.
- The remaining nodes form two stable islands.
- Reconnecting three cut edges merges those islands into one component, and that component is
  still certified stable.

The cost label uses code notation, so `F1 + F3 - F4` is the paper's F1. The run takes a few
seconds and writes a timestamped JSON record to `results/`.

Run the tests that do not require Gurobi:

```bash
pytest --ignore=tests/test_baseline.py
```

Always launch scripts from the repository root with `python -m`.

## Implementation and Engineering

- **Pluggable inner solver.**
  - `islanding.outer_loop.solve(problem, max_outer_iterations, inner_solver=...)` takes the
    inner solver as a parameter.
  - The local search and the MILP both return a solver-agnostic `InnerResult` (a partition,
    not a spanning tree), so they run under the identical refinement loop.
  - A test asserts that the MILP objective equals the reference cost functions on the same
    partition.
- **Layered architecture.**
  - `islanding/` holds the core algorithm and performs no I/O.
  - `baseline/` (MILP) and `utils/` (persistence) depend on it, never the reverse.
  - Experiment drivers and plotting live in separate packages.
- **Numerical computation.**
  - For quadratic models, the island-stability penalty is evaluated in closed form with one
    `einsum` over all samples. It is checked at runtime against the reference per-sample loop.
  - Induced-edge sets are cached per topology.
  - Eigenvalue tests use an explicit tolerance, so structurally zero eigenvalues are not
    misread as stable.
- **Defensive validation.**
  - Parameters and costs are checked for finiteness, sign, and required orderings.
  - The checks raise `ValueError` rather than using `assert`, so they still run under
    `python -O`.
- **Reproducibility.**
  - Sample draws and tree initialization are seeded.
  - Benchmark instances come from committed, seeded generators, and tests check that the
    generators reproduce them exactly.
  - Each run saves its configuration, seeds, per-pass history, runtime, and convergence status
    as JSON.
- **Solver integration.**
  - Gurobi models are built through `gurobipy`, with a MIP gap and a time limit set per case.
  - Solver status, incumbent, best bound, and gap are recorded.
  - A time-limited pass is reported as such rather than discarded.

## Experiments

### Instances

All instances are synthetic linear-consensus networks with negative-weight faults. The 10-node
case is hand-built; the other four are seeded random geometric graphs at matched node density.

| Package | Nodes | Edges | Faults |
|---|---|---|---|
| `linear_10node` | 10 | 17 | 1 node, 2 edges (hand-built) |
| `linear_10node_rgg` | 10 | 19 | 1 node, 2 edges |
| `linear_20node` | 20 | 42 | 3 nodes |
| `linear_30node` | 30 | 82 | 2 nodes, 4 edges |
| `linear_40node` | 40 | 109 | 2 nodes, 2 edges |

### Single cases

Proposed local search (no Gurobi needed):

```bash
python -m examples.linear_10node.run_linear_10node
python -m examples.linear_10node_rgg.run_linear_10node_rgg
python -m examples.linear_20node.run_linear_20node
python -m examples.linear_30node.run_linear_30node
python -m examples.linear_40node.run_linear_40node
```

MILP baseline (needs a Gurobi license):

```bash
python -m examples.linear_10node.run_baseline_10node
python -m examples.linear_10node_rgg.run_baseline_10node_rgg
python -m examples.linear_20node.run_baseline_20node
python -m examples.linear_30node.run_baseline_30node
python -m examples.linear_40node.run_baseline_40node
```

### Comparison studies

```bash
# Proposed vs. MILP on N = 10, 20, 30, 40 (needs Gurobi; the script estimates ~40 min)
python -m examples.run_milp_benchmark

# 30-node fault-position sweep: 10 seeded fault placements (no Gurobi needed)
python -m examples.linear_30node.run_fault_sweep_30node
python -m examples.linear_30node.run_fault_sweep_30node --only 3                     # rerun one configuration (0-based index)
python -m examples.linear_30node.run_fault_sweep_30node --max-outer-iterations 30    # raise the outer-loop cap
```

**Benchmark setup.**

- Both methods run sequentially on the same instance, configuration, sample seed, and stability
  oracle; only the inner solver differs.
- Gurobi uses its default threading, and neither side is pinned to specific threads, so
  wall-clock times are not a resource-matched comparison.
- The comparison table reports the sample-independent economic cost (the paper's F1).

**Interpreting MILP results.**

- Each MILP pass stops when it reaches the configured relative MIP gap (`1e-6`) or the per-case
  time limit (300–1800 s per pass).
- A pass is optimal only when Gurobi reports status `OPTIMAL`, and then only within that gap
  and for that pass's sampled inner problem.
- Time-limited passes are saved with their incumbent, best bound, and gap, and should not be
  read as optimal.

### Outputs, figures, and tables

All runs write timestamped JSON files to `results/`; existing files are never overwritten. No
precomputed results are included. The presentation scripts read the most recent matching
result and render it without re-running anything. The figure script needs `matplotlib`
(`python -m pip install matplotlib`).

```bash
python -m presentation.plot_linear_30node      # three 30-node network figures (PDF, in figures/)
python -m presentation.table_milp_benchmark    # LaTeX table for the MILP benchmark
python -m presentation.table_fault_sweep       # LaTeX table for the fault sweep
python -m presentation.stats_fault_sweep       # plain-text summary statistics for the sweep
```

## Testing

| Command | Gurobi license | Tests |
|---|---|---|
| `pytest --ignore=tests/test_baseline.py` | not required | 401 |
| `pytest` | **required** for 7 tests | 422 |

The suite covers:

- cost terms and input validation;
- augmented-graph decoding;
- local search and the outer refinement loop;
- the counterexample oracle;
- agreement between the vectorized/cached paths and their reference implementations;
- result serialization;
- reproducibility of the benchmark instances;
- run-script configuration parity;
- the MILP baseline.

**Running the full MILP tests.**

- Seven tests in `tests/test_baseline.py` build and solve Gurobi models.
- `gurobipy` is installed with the package, but these tests also need a **valid full Gurobi
  license**, for example a free academic license. The size-limited license bundled with the pip
  package is not sufficient for these models.
- Gurobi finds the license file in its default location or through the `GRB_LICENSE_FILE`
  environment variable.
- Without a valid license these seven tests **fail rather than skip**.
- The other 14 tests in that file do not need a license; the `--ignore` command above leaves
  them out together with the seven.

## Repository Structure

```
islanding/       Core algorithm (no I/O)
  augmented.py       augmented-graph construction
  search.py          spanning-tree initialization and decoding
  cost.py            objective terms, SystemModel interface, linear-consensus model
  local_search.py    proposed inner solver
  outer_loop.py      counterexample-guided refinement loop
  counterexample.py  exact stability check and counterexample generation
  inner.py           solver-agnostic result and failure types
baseline/        MILP baseline (Gurobi)
examples/        Benchmark instances, per-case configs, run scripts
presentation/    Figures and LaTeX tables from saved results
utils/           Result persistence (JSON)
tests/           pytest suite
```

## Scope and Limitations

- **Implemented model.**
  - The theoretical framework covers networked dynamical systems with Lyapunov-based
    stability conditions.
  - This repository implements only grounded linear consensus, for which the stability check
    is exact.
  - Other model classes would need their own `SystemModel` and counterexample oracle.
- **Synthetic instances.** All experiments use synthetic linear-consensus networks with 10 to 40
  nodes. The repository contains no power-system models (power flow or swing dynamics).
- **Penalty weights.** The stability-penalty weights were tuned for the hand-built 10-node case.
  The generated cases reuse them, and their config files mark them as untuned.
- **Optimality.**
  - The local search uses a single seeded start and stops at an approximate (`1 + ε`) local
    optimum.
  - Its theoretical guarantee applies to the sampled inner problem, under the paper's
    conditions.
  - The MILP is optimal only for passes that Gurobi certifies `OPTIMAL`, within the configured
    gap.
- **Certification.** A partition is certified stable only when the outer loop terminates with no
  counterexample. Runs that hit the iteration cap are recorded as not converged.
- **Design notes.** Some code comments refer to internal design documents (`docs/`) that are not
  part of this repository.

## Citation

```bibtex
@misc{cheng2026controlled,
  title         = {Controlled Islanding and Reconnection with Stability Guarantees via Submodular Optimization},
  author        = {Cheng, Shiyu and Clark, Andrew},
  year          = {2026},
  eprint        = {2609.12335},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.12335}
}
```
