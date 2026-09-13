"""Three paper figures for the 30-node case, drawn from a saved result.

Renders the same 30-node network three times -- original, post-islanding,
reconnected -- from the most recent local-search result in results/. The
three share one node layout so a reader can hold the eye on a node and
follow it across the sequence; that is the whole reason they work as a
row.

READS THE PARTITION, never hardcodes it. stable_islands, the unstable
island, the cut set and the reconnected edge all come out of the JSON, so
the figures track the experiment: re-run the case with different faults
or a different seed and the figures follow. Only the node POSITIONS come
from the topology module, because they are the fixture the result is an
answer about, not part of the answer.

Nothing here solves anything. See presentation/__init__.py.

How to run
----------
From the repo root, with the venv active:

    python -m presentation.plot_linear_30node

Writes three PDFs to figures/ and prints their paths and dimensions. Requires matplotlib, which is not in the
project's runtime dependencies: this is presentation code, not part of
the solver, so it is installed alongside rather than required by it.

SIZING. Targets a two-column IEEE template with three figures side by
side in one row, so roughly 5.7 cm of column width each. FIGURE_CM below
is that width; the figures are square, and every font size and line width
is chosen for legibility at that size rather than scaled down from a
screen-sized default. Vector PDF, no title -- the caption belongs in
LaTeX, where it can be numbered and referenced.

NO NODE LABELS. Thirty numeric labels on a 5.7 cm square crowded the
dense lower-right cluster, and the figures do not need them: colour
carries the whole story, and the three share one layout so a node can be
followed across the row by position alone. Islands are coloured on their
EDGES as well as their nodes, which is what lets each one read as a
single connected object rather than as dots joined by neutral lines.
The edge reconnection restores is purple-dashed in figure 2 and plain
blue in figure 3, so the change between the two is stated rather than
left to be found by comparison.

"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.linear_30node.linear_30node_topology import POSITIONS, EDGES

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# One column of a two-column IEEE page is about 8.8 cm; three figures in a
# row across the full 18.2 cm text width leaves roughly 5.7 cm each after
# inter-figure spacing. Square, so the row has a consistent height and the
# layout does not depend on the graph's aspect ratio.
FIGURE_CM = 5.7
CM_PER_INCH = 2.54

# Sized for 5.7 cm, not for a screen, and scaled so the ORDINARY lines
# carry weight on the printed page rather than only being present. The
# first pass here was too fine: 0.45 pt interior edges reproduced as
# hairlines. Everything moved up together, with the emphasised elements
# scaled to hold roughly their old ratio against the ordinary ones --
# raising the baseline without raising the accents would have flattened
# exactly the contrast the figures depend on.
#
#   element                 was    now    ratio to ordinary (was -> now)
#   interior edge          0.45    1.0    --
#   cut edge                0.4    0.9    --
#   node marker             4.0   12.0    --
#   purple dashed (fig 2)   1.0    2.0    2.22x -> 2.00x
#   faulty edge (fig 1)     1.1    2.2    2.44x -> 2.20x
#   faulty node (fig 1)    11.0   32.0    2.75x -> 2.67x  (marker AREA)
NODE_SIZE = 12.0           # marker area in pt^2 -> 3.91 pt across
# Thickened with the edges. At 1.0 pt interior lines a 0.25 pt ring no
# longer separated a node from the edges meeting it -- the marker is only
# about four line-widths across, so the ring has to be a real gap rather
# than an outline. The closest node pair in this layout sits 6.3 pt apart
# against a 3.91 pt marker, so there is room for it.
NODE_EDGE_WIDTH = 0.5
EDGE_WIDTH = 1.0
CUT_EDGE_WIDTH = 0.9

# The reconnection candidate in figure 2, which carries that figure's key
# information and so is given weight rather than matched to the ordinary
# cut edges. It differs from them on all three visual channels at once --
# width, colour and dash shape -- because any one alone was too weak at
# 5.7 cm: at 0.4 pt a dark purple simply read as a dark grey dash.
#
# Dash patterns here are multiplied by the line width (matplotlib's
# lines.scale_dashes, on by default), so the tuples are not directly
# comparable between the two and BOTH had to be revisited when the widths
# went up: the same tuple at twice the width draws dashes twice as long.
# As rendered, grey is 1.98/1.44 pt on a 0.9 pt line and purple 5.2/3.4 pt
# on a 2.0 pt line -- a period of 3.4 pt against 8.6 pt. That 2.5x gap in
# dash length is what keeps the two distinguishable in SHAPE and not only
# in colour, and therefore still distinguishable in greyscale.
#
# Both were checked against the shortest edge each has to draw: 10.6 pt
# for the shortest cut edge (about three grey periods, so it still reads
# as dashed rather than as one segment) and 34.0 pt for the reconnected
# edge (about four purple periods).
RECONNECT_WIDTH = 2.0            # comparable to figure 1's faulty edges
CUT_DASH = (0, (2.2, 1.6))       # short dashes, tight gaps
RECONNECT_DASH = (0, (2.6, 1.7))  # longer dashes, wider gaps once scaled

# Figure 1 only: the fault should be the first thing the eye lands on, so
# faulty nodes and edges are drawn heavier than the healthy ones rather
# than only differently coloured. Red on black already separates them;
# the weight is what makes them the FIGURE and the rest the ground.
FAULTY_NODE_SIZE = 32.0    # ~2.7x the healthy marker area, 6.4 pt across
FAULTY_EDGE_WIDTH = 2.2    # ~2.2x the healthy edge width

COLOUR_NORMAL = "black"
COLOUR_FAULTY = "#d62728"     # red -- faulty nodes and edges, unstable island
COLOUR_LARGE = "#1f77b4"      # blue -- larger stable island, merged component
COLOUR_SMALL = "#2ca02c"      # green -- smaller stable island
COLOUR_CUT = "#9e9e9e"        # grey -- cut edges, dashed
# Fully saturated violet, not the muted #7b3fa0 this started as: a
# desaturated purple at a thin width is indistinguishable from dark grey,
# which defeated the point of colouring it at all.
COLOUR_RECONNECT = "#9400d3"  # purple -- the edge figure 3 brings back

def latest_result(results_dir=RESULTS_DIR):
    """The most recent LOCAL-SEARCH 30-node result.

    The _milp_ files are excluded by name: they are the same case solved
    by the other method and would silently swap which run the figures
    describe, since both write to the same directory with the same
    timestamp format.
    """
    candidates = [
        path for path in results_dir.glob("linear_30node_2*.json")
        if "_milp_" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no local-search 30-node results in {results_dir}. Run "
            "`python -m examples.linear_30node.run_linear_30node` first."
        )
    return max(candidates, key=lambda path: path.name)


def read_partition(path):
    """The partition, as the result recorded it.

    stable_islands entries are dicts of {'nodes', 'lambda_min'}, not bare
    lists -- the lambda_min travels with the island so a reader of the
    JSON can see why it was called stable. Only the node sets are needed
    here, but they have to be reached through the key.
    """
    data = json.loads(path.read_text())

    islands = [set(entry["nodes"]) for entry in data["stable_islands"]]
    # Largest first, so "blue is the larger island" is a property of the
    # data rather than of the order the solver happened to emit.
    islands.sort(key=len, reverse=True)

    return {
        "stable_islands": islands,
        "unstable": {n for island in data["unstable_islands"] for n in island},
        "cut_edges": {_canon(e) for e in data["cut_edges"]},
        "reconnected": {_canon(e) for e in data["reconnected_edges"]},
        "faulty_nodes": set(data["setup"]["faulty_nodes"]),
        "faulty_edges": {_canon(e) for e in data["setup"]["faulty_edges"]},
        "source": path,
    }


def _canon(edge):
    u, v = edge
    return (u, v) if u < v else (v, u)


def merged_components(islands, reconnected):
    """Stable islands grouped by which ones reconnection joins.

    Union-find over the islands, not a hardcoded "the two become one".
    With one reconnected edge joining two islands this returns a single
    group, which is the current case -- but a run with three islands and
    one reconnection would leave one of them separate, and the figure
    should show that rather than colouring everything blue.
    """
    parent = list(range(len(islands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {node: i for i, island in enumerate(islands) for node in island}
    for u, v in reconnected:
        if u in owner and v in owner:
            a, b = find(owner[u]), find(owner[v])
            if a != b:
                parent[b] = a

    groups = {}
    for index in range(len(islands)):
        groups.setdefault(find(index), set()).update(islands[index])
    return sorted(groups.values(), key=len, reverse=True)


def _group_colours(groups, unstable):
    """node -> colour, and a lookup for which group a node belongs to.

    `groups` is stable node sets in the order blue, green, ... -- largest
    first. Returns (colour_of_node, group_of_node); an edge is coloured
    only when its two endpoints share a group, which is exactly the
    condition for it to lie INSIDE an island.
    """
    stable_colours = [COLOUR_LARGE, COLOUR_SMALL]
    colour_of = {}
    group_of = {}
    for index, group in enumerate(groups):
        colour = stable_colours[min(index, len(stable_colours) - 1)]
        for node in group:
            colour_of[node] = colour
            group_of[node] = index
    for node in unstable:
        colour_of[node] = COLOUR_FAULTY
        group_of[node] = "unstable"
    return colour_of, group_of


def _draw_island_edges(axes, cut_edges, colour_of, group_of, pending=()):
    """Every edge: coloured by its island, or dashed if cut.

    Colouring the edges, not just the nodes, is what makes an island read
    as one connected coloured object at 5.7 cm. An edge whose endpoints
    are in different groups cannot be retained -- it would join two
    islands -- so the fallback is only reached if a result ever recorded
    a cut set inconsistent with its own partition, and black is the right
    way to show that rather than to hide it.

    `pending` is the set of edges that are cut HERE but reconnected in the
    next figure. They are drawn dashed like any other cut edge, but
    purple, which does two things:

      * it puts the information in figure 2 instead of requiring the
        reader to diff figure 2 against figure 3 to discover which of
        nineteen grey dashes is the one that matters. The figure says
        "this one comes back" at the point where the reader is looking at
        the cut, not one figure later;
      * it is drawn at RECONNECT_WIDTH with its own dash pattern, not at
        cut-edge width, so it differs from the grey dashes in weight and
        shape as well as hue. Colour alone was not enough at this size,
        and it would have vanished entirely in greyscale;
      * purple rather than blue because in figure 2 this edge SPANS the
        blue and green islands. A blue dash would read as "inside blue",
        which it is not -- it belongs to neither island, which is exactly
        why it was cut. A colour outside both islands' palettes is the
        honest way to say "not part of either, yet".

    Figure 3 passes nothing here: by then the edge is genuinely interior
    to the merged component, so it takes that component's colour like any
    other interior edge. The switch from purple dash to plain blue solid
    is the statement that it has JOINED the component rather than
    remaining a special case that needs its own mark.
    """
    pending = set(pending)
    for edge in EDGES:
        edge = _canon(edge)
        if edge in cut_edges:
            if edge in pending:
                _draw_edge(axes, edge, COLOUR_RECONNECT, RECONNECT_WIDTH,
                           RECONNECT_DASH, zorder=4)
            else:
                _draw_edge(axes, edge, COLOUR_CUT, CUT_EDGE_WIDTH,
                           CUT_DASH, zorder=1)
            continue
        u, v = edge
        same = group_of.get(u) is not None and group_of.get(u) == group_of.get(v)
        _draw_edge(axes, edge,
                   colour_of.get(u, COLOUR_NORMAL) if same else COLOUR_NORMAL,
                   EDGE_WIDTH, "-", zorder=2)


def _new_axes():
    """A figure whose PDF page is EXACTLY FIGURE_CM square.

    The axes fill the whole canvas and the figure is saved without a
    tight bounding box, deliberately. bbox_inches="tight" crops the page
    to its content, which sounds harmless but is not: it produced a
    4.48 cm page, so placing it at 5.7 cm in LaTeX scaled everything up by
    1.27x and a 5 pt label arrived on the page at 6.4 pt. Fixing the page
    size is what makes the font and line widths below mean what they say.
    Air around the drawing therefore comes from the data limits, not from
    the crop.
    """
    inches = FIGURE_CM / CM_PER_INCH
    figure = plt.figure(figsize=(inches, inches))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_aspect("equal")
    axes.axis("off")

    xs = [p[0] for p in POSITIONS.values()]
    ys = [p[1] for p in POSITIONS.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    # Square data window centred on the layout, so the aspect never
    # distorts and the same window is used by all three figures.
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    half = span / 2 + 0.075 * span     # room for labels at the boundary
    axes.set_xlim(cx - half, cx + half)
    axes.set_ylim(cy - half, cy + half)
    return figure, axes


def _draw_edge(axes, edge, colour, width, style="-", zorder=1):
    (x1, y1), (x2, y2) = POSITIONS[edge[0]], POSITIONS[edge[1]]
    axes.plot([x1, x2], [y1, y2], color=colour, linewidth=width,
              linestyle=style, zorder=zorder, solid_capstyle="round")


def _draw_nodes(axes, colour_of, size_of=None):
    for node, (x, y) in POSITIONS.items():
        axes.scatter([x], [y],
                     s=NODE_SIZE if size_of is None else size_of(node),
                     c=colour_of(node), edgecolors="white",
                     linewidths=NODE_EDGE_WIDTH, zorder=3)


def _save(figure, path):
    FIGURES_DIR.mkdir(exist_ok=True)
    # No bbox_inches="tight" -- see _new_axes for why the page size is
    # fixed rather than cropped.
    figure.savefig(path, format="pdf")
    plt.close(figure)
    return path


def figure_original(partition, path):
    """Figure 1 -- the system before islanding.

    Red marks what is broken, black what is not, and the red is drawn
    HEAVIER so the fault is what the eye lands on first. This is the
    figure that establishes which edges are faulty, which is why figures
    2 and 3 need not distinguish a faulty cut from a chosen one: the
    reader already knows, and repeating it there would compete with the
    cut/retained distinction those figures exist to make.
    """
    figure, axes = _new_axes()
    for edge in EDGES:
        edge = _canon(edge)
        faulty = edge in partition["faulty_edges"]
        _draw_edge(axes, edge,
                   COLOUR_FAULTY if faulty else COLOUR_NORMAL,
                   FAULTY_EDGE_WIDTH if faulty else EDGE_WIDTH,
                   "-", zorder=2 if faulty else 1)

    def colour_of(node):
        return COLOUR_FAULTY if node in partition["faulty_nodes"] else COLOUR_NORMAL

    def size_of(node):
        return FAULTY_NODE_SIZE if node in partition["faulty_nodes"] else NODE_SIZE

    _draw_nodes(axes, colour_of, size_of)
    return _save(figure, path)


def figure_islanded(partition, path):
    """Figure 2 -- after islanding, before reconnection.

    Blue is the larger stable island, green the smaller, red the unstable
    one -- on the EDGES as well as the nodes, so each island is one
    coloured object. Cut edges are dashed grey whether they were faulty
    or chosen; figure 1 already said which.

    The one exception is the edge reconnection will bring back, drawn
    purple dashed -- see _draw_island_edges for why it is marked here
    rather than left for the reader to find by comparison.
    """
    figure, axes = _new_axes()
    colour_of, group_of = _group_colours(
        partition["stable_islands"], partition["unstable"]
    )
    _draw_island_edges(axes, partition["cut_edges"], colour_of, group_of,
                       pending=partition["reconnected"])
    _draw_nodes(axes, lambda n: colour_of.get(n, COLOUR_NORMAL))
    return _save(figure, path)


def figure_reconnected(partition, path):
    """Figure 3 -- after reconnection.

    Green is gone: the two stable islands are one blue component. The
    reconnected edge is drawn blue and solid at the ordinary interior
    width, with no special mark -- it is interior to the merged component
    now, and treating it as an exception here would say the opposite of
    what the figure means. Figure 2's purple dash is where its identity
    is established; this figure's job is to show it absorbed.
    """
    figure, axes = _new_axes()
    components = merged_components(
        partition["stable_islands"], partition["reconnected"]
    )
    # Any stable island reconnection did NOT absorb keeps its own colour,
    # so a future run with a partial merge is drawn honestly.
    colour_of, group_of = _group_colours(components, partition["unstable"])
    still_cut = partition["cut_edges"] - partition["reconnected"]

    # No `pending` and no redraw: the reconnected edge is not in
    # still_cut, so _draw_island_edges paints it as an ordinary interior
    # edge of the merged component, which is precisely the point.
    _draw_island_edges(axes, still_cut, colour_of, group_of)
    _draw_nodes(axes, lambda n: colour_of.get(n, COLOUR_NORMAL))
    return _save(figure, path)


def main():
    source = latest_result()
    partition = read_partition(source)

    print(f"Source result: {source.relative_to(REPO_ROOT)}")
    islands = partition["stable_islands"]
    print(f"  stable islands:    {[len(i) for i in islands]} nodes")
    print(f"  unstable island:   {sorted(partition['unstable'])}")
    print(f"  cut edges:         {len(partition['cut_edges'])}")
    print(f"  reconnected edges: {sorted(partition['reconnected'])}")
    print(f"  faulty nodes:      {sorted(partition['faulty_nodes'])}")
    print(f"  faulty edges:      {sorted(partition['faulty_edges'])}")
    print()

    inches = FIGURE_CM / CM_PER_INCH
    print(f"Figure size: {FIGURE_CM:.2f} x {FIGURE_CM:.2f} cm "
          f"({inches:.3f} x {inches:.3f} in), square, vector PDF, no labels")
    print(f"  node marker        {NODE_SIZE} pt^2 "
          f"(faulty, fig 1 only: {FAULTY_NODE_SIZE} pt^2)")
    print(f"  edge               {EDGE_WIDTH} pt "
          f"(faulty, fig 1 only: {FAULTY_EDGE_WIDTH} pt)")
    print(f"  cut edge           {CUT_EDGE_WIDTH} pt dashed {COLOUR_CUT}, "
          f"dashes {CUT_DASH[1]} x lw = "
          f"({CUT_DASH[1][0] * CUT_EDGE_WIDTH:.2f}, "
          f"{CUT_DASH[1][1] * CUT_EDGE_WIDTH:.2f}) pt")
    print(f"  reconnected edge   fig 2: {RECONNECT_WIDTH} pt dashed "
          f"{COLOUR_RECONNECT}, dashes {RECONNECT_DASH[1]} x lw = "
          f"({RECONNECT_DASH[1][0] * RECONNECT_WIDTH:.2f}, "
          f"{RECONNECT_DASH[1][1] * RECONNECT_WIDTH:.2f}) pt")
    print(f"                     fig 3: {EDGE_WIDTH} pt solid "
          f"{COLOUR_LARGE} (ordinary interior edge)")
    print()

    written = []
    for index, (name, draw) in enumerate((
        ("original", figure_original),
        ("islanded", figure_islanded),
        ("reconnected", figure_reconnected),
    ), start=1):
        path = FIGURES_DIR / f"linear_30node_fig{index}_{name}.pdf"
        draw(partition, path)
        print(f"  {path.relative_to(REPO_ROOT)}  "
              f"({path.stat().st_size / 1024:.1f} KB)")
        written.append(path)
    print()

    return written


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
