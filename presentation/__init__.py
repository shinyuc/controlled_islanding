"""Presentation of saved results: figures, comparison tables, scaling plots.

Distinct from utils/, which WRITES results. Nothing here computes or
solves anything -- every module in this package reads a saved artefact
from results/ and renders it. That split is deliberate: a figure that
re-ran the experiment to draw itself could disagree with the numbers in
the text, and a figure that hardcoded a partition would silently stop
tracking the experiment it claims to show.
"""
