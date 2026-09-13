"""Aggregate statistics over the 30-node fault-position sweep.

Reads a sweep summary and reports the spread of runtime, inner-solve
count, per-solve time and economic cost across configurations, plus the
extremes and the convergence/stability checks. Nothing is recomputed --
every figure comes from the summary the sweep wrote.

For analysis, not for the paper: this prints plain text, not LaTeX. The
paper table is `python -m presentation.table_fault_sweep --style compact`.

How to run
----------
    python -m presentation.stats_fault_sweep
    python -m presentation.stats_fault_sweep --summary results/....json
"""

import argparse
import pathlib
import statistics
import sys

from presentation.table_fault_sweep import latest_summary
import json


def _spread(values, with_stdev):
    """min / max / mean / median, and stdev only where it was asked for.

    stdev is reported for runtime and cost but not for solve counts: a
    standard deviation over ten small integers invites a normality
    reading the sample does not support, and min/max/median already say
    what the spread is.
    """
    out = {
        "min": min(values), "max": max(values),
        "mean": statistics.fmean(values), "median": statistics.median(values),
    }
    if with_stdev:
        # Sample stdev (n-1): these ten configurations are a draw from the
        # space of fault placements, not the whole of it.
        out["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def _fmt(stats, places=2):
    parts = [f"min {stats['min']:.{places}f}", f"max {stats['max']:.{places}f}",
             f"mean {stats['mean']:.{places}f}",
             f"median {stats['median']:.{places}f}"]
    if "stdev" in stats:
        parts.append(f"stdev {stats['stdev']:.{places}f}")
    return "  ".join(parts)


def report(summary, out=sys.stdout):
    rows = summary["configurations"]
    label = {row["index"]: row["index"] + 1 for row in rows}

    def w(text=""):
        print(text, file=out)

    # total_runtime on summaries written since per-solve timing was added;
    # runtime_seconds on the earlier ones. Both name the same measurement
    # -- the wall clock around solve() -- so falling back lets this script
    # read a pre-instrumentation summary rather than crash on it.
    def runtime_of(row):
        value = row.get("total_runtime")
        return row["runtime_seconds"] if value is None else value

    runtimes = [runtime_of(r) for r in rows]
    solves = [r["inner_solve_count"] for r in rows]
    costs = [r["economic_cost"] for r in rows]
    means = [r["mean_inner_solve_time"] for r in rows
             if r.get("mean_inner_solve_time") is not None]

    w(f"Configurations: {len(rows)}")
    caps = sorted({r.get("max_outer_iterations") for r in rows})
    w(f"Outer cap: {caps[0] if len(caps) == 1 else caps} "
      f"({'uniform' if len(caps) == 1 else 'NOT uniform'})")
    w()
    w("SPREAD ACROSS CONFIGURATIONS")
    w(f"  total runtime (s)      {_fmt(_spread(runtimes, True))}")
    w(f"  inner solves           {_fmt(_spread(solves, False), places=1)}")
    if means:
        w(f"  mean inner-solve (s)   {_fmt(_spread(means, False))}")
    else:
        w("  mean inner-solve (s)   -- not recorded in this summary")
    w(f"  economic cost          {_fmt(_spread(costs, True), places=3)}")
    w()

    def extreme(key, pick, unit=""):
        getter = runtime_of if key == "total_runtime" else (lambda r: r[key])
        row = pick(rows, key=getter)
        return f"config {label[row['index']]} ({getter(row):.2f}{unit})"

    w("EXTREMES")
    w(f"  fastest                {extreme('total_runtime', min, ' s')}")
    w(f"  slowest                {extreme('total_runtime', max, ' s')}")
    w(f"  fewest inner solves    config "
      f"{label[min(rows, key=lambda r: r['inner_solve_count'])['index']]} "
      f"({min(solves)})")
    w(f"  most inner solves      config "
      f"{label[max(rows, key=lambda r: r['inner_solve_count'])['index']]} "
      f"({max(solves)})")
    if means:
        w(f"  largest mean per solve {extreme('mean_inner_solve_time', max, ' s')}")
    w()

    w("CHECKS")
    converged = [r for r in rows if r["converged"]]
    w(f"  converged              {len(converged)}/{len(rows)}"
      + ("" if len(converged) == len(rows) else "   <-- NOT ALL"))
    capped = [label[r["index"]] for r in rows
              if r["reason"] != "converged"]
    if capped:
        w(f"  stopped at outer cap   configs {capped}")
    inner_capped = [label[r["index"]] for r in rows if r.get("hit_inner_cap")]
    w(f"  hit inner iteration cap {inner_capped or 'none'}")

    have_lambda = [r for r in rows if r.get("island_lambda_min") is not None]
    if have_lambda:
        failed = [label[r["index"]] for r in have_lambda
                  if not r.get("all_stable_islands_positive_definite")]
        worst = min(r["min_stable_lambda_min"] for r in have_lambda
                    if r.get("min_stable_lambda_min") is not None)
        w(f"  stable islands PD      "
          f"{len(have_lambda) - len(failed)}/{len(have_lambda)}"
          + (f"   <-- FAILED: configs {failed}" if failed else ""))
        w(f"  smallest lambda_min    {worst:.6f}")
    else:
        w("  stable islands PD      -- eigenvalues not recorded in this summary")
    w()

    if any(r.get("inner_solve_times") for r in rows):
        w("TIMING BREAKDOWN")
        total_inner = sum(r["total_inner_solve_time"] for r in rows)
        total_all = sum(runtimes)
        w(f"  inner-solve time       {total_inner:.1f} s of {total_all:.1f} s "
          f"({100 * total_inner / total_all:.1f}%)")
        w(f"  non-inner overhead     {total_all - total_inner:.1f} s "
          f"({100 * (total_all - total_inner) / total_all:.1f}%)")
        w()
        w("  per-configuration inner-solve times (s), in pass order:")
        for row in rows:
            times = row.get("inner_solve_times") or []
            w(f"    config {label[row['index']]:>2}: "
              + ", ".join(f"{t:.1f}" for t in times))
        w()
        w("  first vs last inner solve, per configuration:")
        for row in rows:
            times = row.get("inner_solve_times") or []
            if len(times) >= 2:
                w(f"    config {label[row['index']]:>2}: "
                  f"first {times[0]:.1f} s, last {times[-1]:.1f} s, "
                  f"ratio {times[-1] / times[0]:.2f}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    path = args.summary or latest_summary()
    print(f"# {path}")
    return report(json.loads(path.read_text()))


if __name__ == "__main__":
    sys.exit(main())
