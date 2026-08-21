"""MILP baseline for the inner islanding problem.

Replaces `islanding.local_search.local_search` only; the outer
constraint-generation loop in islanding/outer_loop.py is unchanged and
takes the inner solver as a parameter. See docs/milp_baseline_spec.md.
"""
