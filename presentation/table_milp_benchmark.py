"""Paper table and diagnostics for the proposed-vs-MILP benchmark.

Reads a milp_benchmark_summary_*.json and emits the compact comparison as
LaTeX. Nothing is recomputed -- every number comes from the summary the
benchmark wrote, so the table cannot disagree with the run.

WHAT THE PAPER TABLE SHOWS, and why it is only five columns:

    N, M, Method, Inner solves, Time (s), F1

F1 here is the PAPER's F1, the code's economic cost
(code_F1 + code_F3 - code_F4). It is a function of the partition alone --
no sample set enters it -- so it is comparable between the two rows of a
pair without qualification.

The sampled stability penalties are deliberately ABSENT, but NOT because
they are incomparable -- an earlier version of this docstring said that
and was wrong. The harness evaluates code F5 and F6 (the paper's F2 and
F3) on problem.samples_1 and problem.samples_2, the COMMON INITIAL sample
arrays, which are identical within a pair because both runs are built
from the same setup at the same sample seed. So the recorded F5, F6 and
total_with_penalties ARE directly comparable within a pair, as diagnostic
scores on a shared sample basis.

They are omitted because this table's job is the sample-INDEPENDENT
outcome. F1 is a function of the partition alone; the penalty terms are
scores against one particular draw of 20+20 samples, which is a
diagnostic about the returned partition rather than a property of it.
Those belong in the result files, and are there.

WHICH BASIS EACH RECORDED QUANTITY USES, since the distinction is easy to
lose:

  paper_F1 (economic cost)  no sample set enters it -- comparable
                            unconditionally, across pairs and within one
  code_F5, code_F6,         the COMMON INITIAL arrays, K_Phi = K_Psi = 20,
  total_with_penalties      identical within a pair -- comparable within
                            a pair; across different N they score
                            different problems
  objective                 each method's OWN FINAL augmented sample sets,
                            whose K differs between the two runs of a pair
                            -- NOT guaranteed comparable, and the reason
                            it is absent from every table here

How to run
----------
    python -m presentation.table_milp_benchmark              # paper table
    python -m presentation.table_milp_benchmark --style checks
    python -m presentation.table_milp_benchmark --style milp
    python -m presentation.table_milp_benchmark --out tables/benchmark.tex

Requires \\usepackage{booktabs}, nothing else.
"""

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

PROPOSED_LABEL = "Proposed"
MILP_LABEL = "MILP"


def latest_summary(results_dir=RESULTS_DIR):
    candidates = sorted(results_dir.glob("milp_benchmark_summary_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no benchmark summary in {results_dir}. Run "
            "`python -m examples.run_milp_benchmark` first."
        )
    return candidates[-1]


def _pairs(summary):
    """(N, proposed row, MILP row), in ascending N."""
    by_size = {}
    for row in summary["runs"]:
        by_size.setdefault(row["N"], {})[row["method"]] = row
    out = []
    for size in sorted(by_size):
        entry = by_size[size]
        out.append((size, entry.get("proposed"), entry.get("MILP")))
    return out


def build_paper_table(summary):
    lines = [
        r"\begin{tabular}{@{}rrlrrr@{}}",
        r"\toprule",
        r"$N$ & $M$ & Method & Inner solves & Time (s) & $F_1$ \\",
        r"\midrule",
    ]
    for index, (size, proposed, milp) in enumerate(_pairs(summary)):
        if index:
            # A rule between pairs: the table's unit is the PAIR, and
            # without a separator ten rows read as ten independent runs.
            lines.append(r"\addlinespace")
        for label, row in ((PROPOSED_LABEL, proposed), (MILP_LABEL, milp)):
            if row is None:
                continue
            # N and M printed once per pair, on the first row: repeating
            # them would suggest the two runs might differ in them, and
            # the whole point is that they cannot.
            head = (f"{size} & {row['M']} & " if label == PROPOSED_LABEL
                    else r"& & ")
            lines.append(
                head + f"{label} & {row['inner_solve_count']} & "
                f"{row['total_runtime']:.1f} & {row['paper_F1']:.3f}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_milp_table(summary):
    """Per-inner-solve MILP diagnostics. Internal, not for the paper."""
    lines = [
        r"\begin{tabular}{@{}rrrrrrrr@{}}",
        r"\toprule",
        r"$N$ & Pass & $K_\Phi$ & $K_\Psi$ & Status & Gap "
        r"& Vars (bin.) & Constrs \\",
        r"\midrule",
    ]
    for size, _, milp in _pairs(summary):
        if milp is None:
            continue
        for entry in milp.get("milp_passes", []):
            gap = entry.get("attained_mip_gap")
            gap_text = "--" if gap is None else f"{gap:.1e}"
            lines.append(
                f"{size} & {entry['pass']} & {entry['K_phi']} & "
                f"{entry['K_psi']} & {entry['status']} & {gap_text} & "
                f"{entry['n_vars']:,} ({entry['n_binary_vars']:,}) & "
                f"{entry['n_constrs']:,}" + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def validity_report(summary, out=sys.stdout):
    """The gate from section 6, evaluated rather than assumed.

    A run may enter the paper table only if the outer loop converged, and
    every stable island and every reconnected component passes the
    eigenvalue test. For the MILP, every inner solve must additionally
    have terminated within the requested tolerance. Printed as a pass/fail
    per run so a failure is visible rather than absorbed.
    """
    def w(text=""):
        print(text, file=out)

    failures = []
    w(f"{'N':>4} {'method':>9} {'converged':>10} {'islands':>8} {'components':>11} "
      f"{'MILP optimal':>13} {'verdict':>8}")
    for row in summary["runs"]:
        checks = {
            "converged": row["converged"],
            "islands": row["all_stable_islands_pass"],
            "components": row["all_components_pass"],
        }
        if row["method"] == "MILP":
            checks["milp_optimal"] = row["all_solves_optimal"]
        ok = all(checks.values())
        if not ok:
            failures.append((row["N"], row["method"],
                             [k for k, v in checks.items() if not v]))
        w(f"{row['N']:>4} {row['method']:>9} {str(row['converged']):>10} "
          f"{str(row['all_stable_islands_pass']):>8} "
          f"{str(row['all_components_pass']):>11} "
          f"{str(row.get('all_solves_optimal', '--')):>13} "
          f"{'PASS' if ok else 'FAIL':>8}")
    w()
    if failures:
        w("VALIDITY FAILURES -- do not build a paper table from this run:")
        for size, method, which in failures:
            w(f"  N={size} {method}: failed {which}")
    else:
        w("All runs pass the section-6 validity gate.")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, default=None)
    parser.add_argument("--style", default="paper",
                        choices=("paper", "milp", "checks"))
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    path = args.summary or latest_summary()
    summary = json.loads(path.read_text())

    if args.style == "checks":
        return 1 if validity_report(summary) else 0

    builder = {"paper": build_paper_table, "milp": build_milp_table}[args.style]
    text = builder(summary) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Read   {path}\nWrote  {args.out}")
    else:
        print(f"% generated from {path.name} by "
              f"presentation/table_milp_benchmark.py")
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
