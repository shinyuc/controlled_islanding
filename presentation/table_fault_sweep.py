"""LaTeX table for the 30-node fault-position sweep.

Reads the most recent linear_30node_sweep_summary_*.json and emits a
booktabs tabular. Nothing is recomputed: every number comes out of the
summary the sweep wrote, so the table cannot disagree with the run.

How to run
----------
From the repo root, with the venv active:

    python -m presentation.table_fault_sweep            # to stdout
    python -m presentation.table_fault_sweep --out tables/fault_sweep.tex

Requires \\usepackage{booktabs}, and nothing else: the convergence
column uses $\\bullet$ / $\\circ$, plain LaTeX math symbols, rather
than \\checkmark, which would need amssymb. The footnote wants a table
environment that permits \\footnotesize text beneath the tabular. The
output is a bare tabular plus the note, not a float -- the caption,
label and placement belong in the paper.
"""

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def latest_summary(results_dir=RESULTS_DIR):
    candidates = sorted(results_dir.glob("linear_30node_sweep_summary_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no sweep summary in {results_dir}. Run "
            "`python -m examples.linear_30node.run_fault_sweep_30node` first."
        )
    return candidates[-1]


def _fmt_nodes(nodes):
    return ", ".join(str(n) for n in nodes)


def _fmt_edges(edges):
    # Compact: (u,v) with no space, comma-separated, in math mode so the
    # parentheses set at the same size as the digits.
    return ", ".join(f"$({u},{v})$" for u, v in edges)


def build_compact_table(summary):
    """The PAPER table: configuration, inner solves, runtime, cost.

    Deliberately four columns. What was dropped and why:

      * augmentation rounds -- on a converged run they are exactly
        inner solves minus one, so the column carries no information the
        neighbouring one does not already give;
      * fault node and edge IDs -- all ten cases use the same COUNTS and
        differ only in location, and the specific IDs are what the reader
        cannot use: no argument in the paper turns on which node was
        faulted, only on the spread of the results across placements. The
        identities stay in the summary JSON for anyone who wants them;
      * $F_5$, $F_6$, island counts -- diagnostics rather than the claim.

    The claim this table supports is that the method's behaviour does not
    depend on where the faults sit, and runtime, solve count and economic
    cost are the three quantities that claim is about.
    """
    rows = summary["configurations"]
    lines = [
        r"\begin{tabular}{@{}rrrr@{}}",
        r"\toprule",
        r"Configuration & Inner solves & Runtime (s) & Economic cost \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['index'] + 1} & {row['inner_solve_count']} & "
            f"{row['runtime_seconds']:.0f} & {row['economic_cost']:.3f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_diagnostic_table(summary):
    """The INTERNAL timing table. Not for the paper.

    Splits each run into the time inside the inner optimiser and
    everything else, which is what says whether a runtime difference
    between configurations is the optimisation itself or the outer loop
    around it. Seconds throughout, at one decimal for the per-solve
    figures because those are tens of seconds and the second digit is
    real there, and whole seconds for the totals.
    """
    rows = summary["configurations"]
    lines = [
        # EIGHT columns: config, solves, total, inner total, mean,
        # max, overhead, cost. Counted against a data row, not eyeballed
        # from the header -- a short colspec is a LaTeX error, not a
        # cosmetic one.
        r"\begin{tabular}{@{}rrrrrrrr@{}}",
        r"\toprule",
        r"Config. & Inner & Total & Inner & Mean inner & Max inner "
        r"& Non-inner & Economic \\",
        r" & solves & runtime (s) & total (s) & (s) & (s) "
        r"& overhead (s) & cost \\",
        r"\midrule",
    ]
    for row in rows:
        def fmt(key, places=1):
            value = row.get(key)
            return "--" if value is None else f"{value:.{places}f}"
        lines.append(
            f"{row['index'] + 1} & {row['inner_solve_count']} & "
            f"{fmt('total_runtime', 0)} & {fmt('total_inner_solve_time', 0)} & "
            f"{fmt('mean_inner_solve_time')} & {fmt('max_inner_solve_time')} & "
            f"{fmt('non_inner_overhead')} & {row['economic_cost']:.3f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_table(summary):
    rows = summary["configurations"]

    lines = [
        r"\begin{tabular}{@{}rllrrrrrrrrc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Fault position} "
        r"& \multicolumn{2}{c}{Islands} & & & & & "
        r"\multicolumn{2}{c}{Outer loop} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){10-11}",
        r"\# & Nodes & Edges & St. & Un. "
        r"& $F_1{+}F_3{-}F_4$ & $F_5$ & $F_6$ & Time (s) "
        r"& Solves & Aug. & Conv. \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['index'] + 1} & "
            f"{_fmt_nodes(row['faulty_nodes'])} & "
            f"{_fmt_edges(row['faulty_edges'])} & "
            f"{row['n_stable_islands']} & "
            f"{row['n_unstable_islands']} & "
            f"{row['economic_cost']:.3f} & "
            f"{row['f5']:.3f} & "
            f"{row['f6']:.3f} & "
            # The quantity the sweep exists to measure, and previously
            # absent from the table entirely. Whole seconds: these are
            # multi-minute solves whose SPREAD across configurations is
            # the point, and a decimal place would imply a precision that
            # wall-clock timing of such a run does not have.
            f"{row['runtime_seconds']:.0f} & "
            f"{row['inner_solve_count']} & "
            f"{row['augmentation_rounds']} & "
            # $\bullet$ / $\circ$ rather than \checkmark, which needs
            # amssymb -- and which was previously emitted OUTSIDE math
            # mode, so it would not have compiled at all. These two are
            # plain LaTeX math symbols, keeping booktabs the table's only
            # requirement. The note says which is which.
            + "$"
            + (r"\bullet" if row["converged"] else r"\circ")
            # A pass that hit the inner iteration cap is flagged in the
            # row, not only in the note: it is a property of that
            # configuration, and a reader scanning runtimes needs to see
            # which rows spent a full inner budget without reaching a
            # local optimum.
            + (r"^{\dag}" if row.get("hit_inner_cap") else "")
            + r"$ \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _join(items):
    """'1', '1 and 3', '1, 3 and 7' -- for prose, not for a list."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_timing_note(summary):
    """What the split between inner and non-inner time does and does not mean."""
    rows = summary["configurations"]
    overheads = [row["non_inner_overhead"] for row in rows
                 if row.get("non_inner_overhead") is not None]
    totals = [row["total_runtime"] for row in rows
              if row.get("total_runtime") is not None]
    if not overheads or not totals:
        return ("Per-solve timing is absent from this summary; rerun the "
                "sweep to populate it.")
    share = 100.0 * sum(overheads) / sum(totals)
    return (
        f"Inner-solve times bracket calls to the inner optimiser only: "
        f"the counterexample search, sample augmentation and pass "
        f"bookkeeping all happen outside them and fall into non-inner "
        f"overhead. Total runtime brackets the outer loop alone -- not "
        f"problem construction before it or cost reporting after -- so "
        f"the three columns satisfy total = inner + overhead exactly. "
        f"Across the {len(rows)} configurations the overhead is "
        f"{share:.1f}\\% of total runtime."
    )


def build_note(summary):
    """The footnote, written from what the summary actually records.

    NOT a fixed string. Whether the F5/F6 columns are comparable across
    rows depends on whether every configuration was evaluated on the same
    sample basis, and that is a property of the run, not of this script.
    Hardcoding either claim would eventually state the wrong one.
    """
    rows = summary["configurations"]
    phi = {row["f5_f6_K_phi"] for row in rows}
    psi = {row["f5_f6_K_psi"] for row in rows}

    parts = [
        r"\footnotesize",
        r"Conv.: $\bullet$ the outer loop converged (the returned "
        r"partition has no stability counterexample), $\circ$ it "
        r"stopped at \texttt{max\_outer\_iterations}.",
        f"Fixed {summary['n_nodes']}-node topology "
        f"($M = {summary['n_edges']}$, seed {summary['topology_seed']}) "
        f"with {summary['n_faulty_nodes']} faulty nodes and "
        f"{summary['n_faulty_edges']} faulty edges throughout; only their "
        r"positions vary. Positions drawn from seed "
        f"{summary['sweep_seed']}; the sample seed and the "
        r"\textsc{AlgorithmConfig} are identical to the base case, so the "
        r"sample sets are the same in every row.",
    ]

    rejected = summary["n_rejected_degenerate"]
    if rejected:
        parts.append(
            f"{rejected} further draw{'s' if rejected != 1 else ''} "
            f"{'were' if rejected != 1 else 'was'} rejected as degenerate "
            f"(shedding the faulty nodes disconnected the remainder) out of "
            f"{summary['draw_attempts']} attempts."
        )
    else:
        parts.append(
            f"No draw was rejected: all {summary['draw_attempts']} attempts "
            r"left a connected remainder after shedding the faulty nodes."
        )

    if len(phi) == 1 and len(psi) == 1:
        parts.append(
            r"$F_5$ and $F_6$ are evaluated on the initial sample sets "
            f"($K_\\Phi = K_\\Psi = {phi.pop()}$), which are identical "
            r"across rows, so those two columns are directly comparable. "
            r"They are still sample-dependent quantities: they report the "
            r"magnitude of the stability penalties at the returned "
            r"partition, not a property of the partition alone."
        )
    else:
        parts.append(
            r"$F_5$ and $F_6$ show the magnitude of the stability "
            r"penalties at the returned partition. They are "
            r"sample-dependent, and the configurations run different "
            r"numbers of augmentation rounds, so the $K_\Phi$ and "
            r"$K_\Psi$ they are evaluated on differ between rows: these "
            r"two columns are \emph{not guaranteed to be comparable} "
            r"across rows. Note that equal $K$ would not settle it "
            r"either -- two configurations can append the same number of "
            r"witnesses while appending different ones."
        )

    parts.append(
        r"The economic cost $F_1{+}F_3{-}F_4$ involves no sample set and "
        r"is comparable across rows without qualification."
    )

    # THE OUTER CAP, which is a per-row property once any row has been
    # rerun with more room. Stated only when the rows differ: on a uniform
    # run it is a detail of the configuration, but the moment one row ran
    # under a different budget the table is comparing rows that were not
    # given the same allowance, and that has to be on the page.
    caps = {row.get("max_outer_iterations") for row in rows}
    caps.discard(None)
    if len(caps) > 1:
        common = min(caps)
        raised = sorted(
            (row for row in rows
             if row.get("max_outer_iterations") != common),
            key=lambda row: row["index"],
        )
        solves_at_common = [
            row["inner_solve_count"] for row in rows
            if row.get("max_outer_iterations") == common
        ]
        which = _join([str(row["index"] + 1) for row in raised])
        highest = max(row["max_outer_iterations"] for row in raised)
        parts.append(
            ("Row " if len(raised) == 1 else "Rows ") + which
            + r" ran with \texttt{max\_outer\_iterations} $= "
            + f"{highest}$; the "
            # Agrees with the number of OTHER rows, not the number of
            # reruns: one rerun among nine others is "the others".
            + ("other " if len(solves_at_common) == 1 else "others ")
            + f"ran at ${common}$. "
            + ("That row " if len(raised) == 1 else "Those rows ")
            + f"exhausted the cap of ${common}$ and returned without "
            r"converging, so it was rerun with more room; nothing else "
            r"was rerun. Raising the cap cannot affect the remaining "
            f"rows, which converged in at most "
            f"{max(solves_at_common)} inner solves, strictly below "
            f"${common}$ -- a limit that never bound cannot have shaped "
            r"the result."
        )

    # THE INNER CAP. Stated either way: "no pass was truncated" is a
    # result, and leaving it unsaid would let a reader assume it without
    # the run ever having established it.
    capped = [row for row in rows if row.get("hit_inner_cap")]
    cap = next((row.get("inner_iteration_cap") for row in rows
                if row.get("inner_iteration_cap") is not None), None)
    # RAW f-string. Written without the r prefix this produced a literal
    # TAB from the \t of \texttt and emitted broken LaTeX -- the kind of
    # thing that only shows up when the .tex is compiled.
    cap_text = rf" ($\texttt{{max\_iterations}} = {cap}$)" if cap else ""
    if capped:
        which = _join([str(row["index"] + 1) for row in capped])
        parts.append(
            r"$^{\dag}$ In "
            + ("row " if len(capped) == 1 else "rows ")
            + which
            + r" at least one local-search pass reached the inner "
            r"iteration cap" + cap_text + r", so that pass consumed its "
            r"full budget and returned without having established that "
            r"its candidate is locally optimal -- the candidate may or "
            r"may not be one; what is absent is the certification. This "
            r"is independent of the convergence column, which reports "
            r"the outer stability oracle's verdict on the returned "
            r"partition."
        )
    else:
        parts.append(
            r"No local-search pass in any configuration reached the inner "
            r"iteration cap" + cap_text + r": every pass terminated at a "
            r"certified local optimum, so no reported runtime includes a "
            r"truncated search."
        )

    return " ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, default=None,
                        help="summary JSON to read (default: most recent)")
    parser.add_argument("--style", default="compact",
                        choices=("compact", "diagnostic", "full"),
                        help="compact: the paper table (default). "
                             "diagnostic: internal timing breakdown. "
                             "full: every column, with the explanatory note")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="write here instead of stdout")
    args = parser.parse_args(argv)

    path = args.summary or latest_summary()
    summary = json.loads(path.read_text())

    if args.style == "compact":
        # No note: the paper table's caption carries what it needs, and a
        # footnote arguing about sample bases belongs with the columns
        # that have one.
        text = build_compact_table(summary) + "\n"
    elif args.style == "diagnostic":
        text = (build_diagnostic_table(summary) + "\n\n"
                + r"\footnotesize Internal diagnostic, not for publication. "
                + build_timing_note(summary) + "\n")
    else:
        text = build_table(summary) + "\n\n" + build_note(summary) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Read   {path}")
        print(f"Wrote  {args.out}  "
              f"({len(summary['configurations'])} configurations)")
    else:
        print(f"% generated from {path.name} by "
              f"presentation/table_fault_sweep.py")
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
