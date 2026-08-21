"""MILP baseline: certificate precompute, model construction, solver.

See docs/milp_baseline_spec.md for the implementation decisions and
docs/milp_algorithm.tex for the formulation itself.
"""

from dataclasses import dataclass

import numpy as np

from islanding.cost import canon


@dataclass(frozen=True)
class PhiCertificates:
    """The Phi family of spec section 5, over samples_1 (K = P_1, uses k_1).

    Entries are numpy arrays of shape (K,) -- one scalar per sample --
    keyed as the LaTeX indexes them:

        Phi_n[i]        node terms
        Phi_e[(i,j)]    canonical (min,max) edge key; symmetric in i,j
        Phi_t[(i,j,k)]  ORDERED triple, j,k in N_i, k == j included

    Phi_t[(i,j,k)] != Phi_t[(i,k,j)] in general: i is the distinguished
    "self" node, j indexes the edge whose partial is taken and k the edge
    whose dynamics are taken.
    """

    K: int
    node_order: list
    Phi_n: dict
    Phi_e: dict
    Phi_t: dict


@dataclass(frozen=True)
class PsiCertificates:
    """The Psi family of spec section 5, over samples_2 (K = P_2).

    Uses BOTH decay rates: k_1 - k_2 in the node/edge slack terms and
    k_2 / 2 in Psi_r. Keyed as the LaTeX indexes them:

        Psi_n[i]         node slack, always <= 0
        Psi_e[(i,j)]     canonical edge key, edge slack
        Psi_r[(i,k)]     ORDERED, k in N_i
        Psi_sr[(i,j,k)]  ORDERED, j,k in N_i
        Psi_rr[(i,j,k)]  ORDERED, j,k in N_i

    Kept separate from PhiCertificates because the two families are
    evaluated over DIFFERENT sample sets: the outer loop grows samples_1
    with island witnesses and samples_2 with component witnesses
    independently, so K_Phi != K_Psi in general (spec section 6).
    """

    K: int
    node_order: list
    Psi_n: dict
    Psi_e: dict
    Psi_r: dict
    Psi_sr: dict
    Psi_rr: dict


def _index_sets(topology):
    """Shared node/edge/neighbour bookkeeping, in one deterministic order."""
    nodes = sorted(topology.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    edges = sorted({canon((u, v)) for u, v in topology.edges()})

    neighbors = {i: set() for i in nodes}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    neighbors = {i: sorted(js) for i, js in neighbors.items()}

    return nodes, node_index, edges, neighbors


def _sample_cache(nodes, neighbors, states, model):
    """Per-sample evaluations of the model callables, computed once.

    `grad_ij[(i, j)]` is the i-SIDE partial of edge (i,j)'s Lyapunov term,
    i.e. dW_ij/dx_i -- grad_W_ij returns BOTH partials as a pair and the
    i-side one is element 0 (spec section 5, trap 1).

    `grad_ji_of[(i, j)]` is the OTHER partial from that SAME call, i.e.
    element 1 of grad_W_ij(i, j, x_i, x_j) = dW_ij/dx_j. It is deliberately
    not obtained as grad_ij[(j, i)], which would be a reversed invocation:
    the two agree only for a model with W_ji(x_j,x_i) == W_ij(x_i,x_j) and
    matching gradient order-equivariance. islanding/cost.py takes both
    outputs of one canonical call, so taking them the same way here keeps
    the MILP and evaluate() identical even for a model without that
    symmetry.

    `f_pair[(i, j)]` is f_ij(i, j, x_i, x_j). Note f_ji is
    f_ij(j, i, x_j, x_i) -- a DIFFERENT vector, looked up as
    f_pair[(j, i)] (spec section 5, trap 2). Both orderings are stored;
    f_ij has no single-call two-output form, so the reversed call is the
    only way to get it.
    """
    grad_i = {i: model.grad_W_i(i, states[i]) for i in nodes}
    f_self = {i: model.f_i(i, states[i]) for i in nodes}
    W_self = {i: model.W_i(i, states[i]) for i in nodes}

    grad_ij = {}
    grad_ji_of = {}
    f_pair = {}
    W_pair = {}
    for i in nodes:
        for j in neighbors[i]:
            # (i, j) ordered: i is "self", j is "neighbour". One call, both
            # partials -- see the docstring on why the j-side is kept here
            # rather than looked up under the reversed key.
            partial_i, partial_j = model.grad_W_ij(i, j, states[i], states[j])
            grad_ij[(i, j)] = partial_i
            grad_ji_of[(i, j)] = partial_j
            f_pair[(i, j)] = model.f_ij(i, j, states[i], states[j])
            W_pair[(i, j)] = model.W_ij(i, j, states[i], states[j])

    return grad_i, f_self, W_self, grad_ij, grad_ji_of, f_pair, W_pair


def phi_certificates(topology, model, config, samples):
    """Evaluate Phi^n, Phi^e, Phi^t over `samples` (spec section 5).

    Pass samples_1: the Phi family is the F5 side and uses k_1.
    """
    nodes, node_index, edges, neighbors = _index_sets(topology)
    K = len(samples)
    k_1 = config.k_1

    triples = [(i, j, k) for i in nodes for j in neighbors[i] for k in neighbors[i]]

    Phi_n = {i: np.zeros(K) for i in nodes}
    Phi_e = {edge: np.zeros(K) for edge in edges}
    Phi_t = {key: np.zeros(K) for key in triples}

    for ell, sample in enumerate(samples):
        states = {node: sample[node_index[node]] for node in nodes}
        grad_i, f_self, W_self, grad_ij, grad_ji_of, f_pair, W_pair = _sample_cache(
            nodes, neighbors, states, model
        )

        for i in nodes:
            # Phi^n_i = grad W_i . f_i + k_1 W_i
            Phi_n[i][ell] = float(np.dot(grad_i[i], f_self[i])) + k_1 * W_self[i]

        for (i, j) in edges:
            # Phi^e_ij = dW_ij/dx_i . f_i + dW_ij/dx_j . f_j
            #            + grad W_i . f_ij + grad W_j . f_ji + k_1 W_ij
            # f_ji is f_ij(j, i, x_j, x_i) -- f_pair[(j, i)], NOT f_pair[(i, j)].
            Phi_e[(i, j)][ell] = (
                float(np.dot(grad_ij[(i, j)], f_self[i]))
                + float(np.dot(grad_ji_of[(i, j)], f_self[j]))
                + float(np.dot(grad_i[i], f_pair[(i, j)]))
                + float(np.dot(grad_i[j], f_pair[(j, i)]))
                + k_1 * W_pair[(i, j)]
            )

        for (i, j, k) in triples:
            # Phi^t_ijk = dW_ij/dx_i . f_ik   (k == j included)
            Phi_t[(i, j, k)][ell] = float(
                np.dot(grad_ij[(i, j)], f_pair[(i, k)])
            )

    return PhiCertificates(
        K=K, node_order=nodes, Phi_n=Phi_n, Phi_e=Phi_e, Phi_t=Phi_t
    )


def psi_certificates(topology, model, config, samples):
    """Evaluate Psi^n, Psi^e, Psi^r, Psi^sr, Psi^rr over `samples`.

    Pass samples_2: the Psi family is the F6 side and uses k_1 - k_2 for
    the slack terms and k_2 / 2 inside Psi^r.
    """
    nodes, node_index, edges, neighbors = _index_sets(topology)
    K = len(samples)
    k_2 = config.k_2
    decay_gap = config.k_1 - config.k_2

    triples = [(i, j, k) for i in nodes for j in neighbors[i] for k in neighbors[i]]
    pairs = [(i, k) for i in nodes for k in neighbors[i]]

    Psi_n = {i: np.zeros(K) for i in nodes}
    Psi_e = {edge: np.zeros(K) for edge in edges}
    Psi_r = {key: np.zeros(K) for key in pairs}
    Psi_sr = {key: np.zeros(K) for key in triples}
    Psi_rr = {key: np.zeros(K) for key in triples}

    for ell, sample in enumerate(samples):
        states = {node: sample[node_index[node]] for node in nodes}
        grad_i, f_self, W_self, grad_ij, grad_ji_of, f_pair, W_pair = _sample_cache(
            nodes, neighbors, states, model
        )

        for i in nodes:
            # Psi^n_i = -(k_1 - k_2) W_i
            Psi_n[i][ell] = -decay_gap * W_self[i]

        for (i, j) in edges:
            # Psi^e_ij = -(k_1 - k_2) W_ij
            Psi_e[(i, j)][ell] = -decay_gap * W_pair[(i, j)]

        for (i, k) in pairs:
            # Psi^r_ik = grad W_i . f_ik + dW_ik/dx_i . f_i + (k_2 / 2) W_ik
            Psi_r[(i, k)][ell] = (
                float(np.dot(grad_i[i], f_pair[(i, k)]))
                + float(np.dot(grad_ij[(i, k)], f_self[i]))
                + 0.5 * k_2 * W_pair[(i, k)]
            )

        for (i, j, k) in triples:
            # Psi^rr_ijk = dW_ij/dx_i . f_ik; Psi^sr adds the j<->k swap.
            value = float(np.dot(grad_ij[(i, j)], f_pair[(i, k)]))
            Psi_rr[(i, j, k)][ell] = value
            Psi_sr[(i, j, k)][ell] = value + float(
                np.dot(grad_ij[(i, k)], f_pair[(i, j)])
            )

    return PsiCertificates(
        K=K, node_order=nodes, Psi_n=Psi_n, Psi_e=Psi_e,
        Psi_r=Psi_r, Psi_sr=Psi_sr, Psi_rr=Psi_rr,
    )


# ---------------------------------------------------------------------------
# STEP 2 -- model construction (docs/milp_algorithm.tex, eq:milp_baseline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MILPModel:
    """A built Gurobi model plus the variable handles needed to decode it."""

    model: object
    order: list          # sorted nodes; slot r is identified with node r
    pos: dict            # node -> 1-based rank, the "r <= i" convention
    edges: list          # canonical (min,max) edge keys
    a: dict              # a[(i, r)]  binary, r <= i
    b: dict              # b[r]       binary
    y: dict              # y[(i, j)]  binary, canonical edge key
    g: dict              # g[(i, r)]
    w: dict              # w[(r, edge)]
    wb: dict             # wb[(r, edge)]
    tau: dict            # tau[(r, (i, j, k))]  -- diagonal k == j aliases wb
    qr: dict             # qr[(r, (i, k))]
    qsr: dict            # qsr[(r, (i, j, k))]  -- no alias; diagonal is real
    qrr: dict            # qrr[(r, (i, j, k))]  -- diagonal k == j aliases qr
    sigma_phi: dict      # sigma_phi[(r, ell)]
    sigma_psi: dict      # sigma_psi[(r, ell)]


def _slots_for(pos, limit_rank):
    """Slots r admissible when every BOUNDING index has rank >= limit_rank.

    The representative convention (a_ir = 0 for i < r) means slot r holds
    only nodes >= r, so a family is restricted by whichever of its indices
    the product forces into slot r. That bound is per-family, not a
    uniform min over all indices -- callers pass the right limit_rank
    (spec section 4, "The bound on r is per-family, not uniform"). Spec
    section 4 is also emphatic that generating outside these ranges is the
    difference between a tractable and an intractable model.
    """
    return [r for r, rank in pos.items() if rank <= limit_rank]


def build_model(problem, milp_config, phi_cert, psi_cert):
    """Build the MILP of docs/milp_algorithm.tex for `problem`.

    `phi_cert` must come from problem.samples_1 and `psi_cert` from
    problem.samples_2; their K values are K_Phi and K_Psi and need not be
    equal.
    """
    import gurobipy as gp
    from gurobipy import GRB

    topology = problem.topology
    config = problem.config
    cost_data = problem.cost_data

    order, _, edges, neighbors = _index_sets(topology)
    N = len(order)
    pos = {node: idx + 1 for idx, node in enumerate(order)}

    if phi_cert.K != config.P_1:
        raise ValueError(f"phi_cert.K ({phi_cert.K}) != config.P_1 ({config.P_1})")
    if psi_cert.K != config.P_2:
        raise ValueError(f"psi_cert.K ({psi_cert.K}) != config.P_2 ({config.P_2})")

    model = gp.Model("islanding_milp")
    model.setParam("OutputFlag", 1 if milp_config.verbose else 0)
    model.setParam("TimeLimit", milp_config.time_limit)
    model.setParam("MIPGap", milp_config.mip_gap)

    # -- variables, each only over the range where it can be nonzero -------
    a, g = {}, {}
    for i in order:
        for r in _slots_for(pos, pos[i]):
            a[(i, r)] = model.addVar(vtype=GRB.BINARY, name=f"a[{i},{r}]")
            g[(i, r)] = model.addVar(lb=0.0, ub=1.0, name=f"g[{i},{r}]")

    b = {r: model.addVar(vtype=GRB.BINARY, name=f"b[{r}]") for r in order}
    y = {e: model.addVar(vtype=GRB.BINARY, name=f"y[{e[0]},{e[1]}]") for e in edges}

    w, wb, xi = {}, {}, {}
    for (i, j) in edges:
        for r in _slots_for(pos, min(pos[i], pos[j])):
            w[(r, (i, j))] = model.addVar(lb=0.0, ub=1.0, name=f"w[{r},{i},{j}]")
            wb[(r, (i, j))] = model.addVar(lb=0.0, ub=1.0, name=f"wb[{r},{i},{j}]")
            # xi is DIRECTED: one variable per orientation of the edge.
            xi[(r, (i, j))] = model.addVar(lb=0.0, name=f"xi[{r},{i},{j}]")
            xi[(r, (j, i))] = model.addVar(lb=0.0, name=f"xi[{r},{j},{i}]")

    # Index ranges: only indices that must BE MEMBERS of slot r bound r.
    # A reconnection partner lies in a DIFFERENT island by construction, so
    # its position says nothing about r. Per family:
    #   tau_{r,ijk} = wb_{r,ij} * a_{kr}  -- i, j, k all in slot r
    #   qr_{r,ik}   = g_{ir}    * y_{ik}  -- only i in slot r
    #   qsr_{r,ijk} = wb_{r,ij} * y_{ik}  -- i, j in slot r; k is not
    #   qrr_{r,ijk} = qr_{r,ij} * y_{ik}  -- only i in slot r; j, k are not
    # This matches the per-family table in docs/milp_baseline_spec.md
    # section 4. Using min(i,j,k) uniformly would delete needed q^sr/q^rr
    # variables (and min(i,k) likewise for q^r), silently dropping an
    # island's reconnection contribution from its Psi row.
    qr = {}
    for i in order:
        for k in neighbors[i]:
            for r in _slots_for(pos, pos[i]):
                qr[(r, (i, k))] = model.addVar(lb=0.0, ub=1.0, name=f"qr[{r},{i},{k}]")

    # tau and qrr are ALIASED on the diagonal (k == j), so no variable is
    # created there: tau_{r,ijj} = wb_{r,ij} and qrr_{r,ijj} = qr_{r,ij}.
    # qsr has no alias -- its diagonal is a real variable that eligibility
    # forces to zero (w_{r,ij}=1 => z_ij=1 => y_ij=0).
    tau, qsr, qrr = {}, {}, {}
    for i in order:
        for j in neighbors[i]:
            for k in neighbors[i]:
                for r in _slots_for(pos, min(pos[i], pos[j], pos[k])):
                    if k != j:
                        tau[(r, (i, j, k))] = model.addVar(
                            lb=0.0, ub=1.0, name=f"tau[{r},{i},{j},{k}]")
                for r in _slots_for(pos, min(pos[i], pos[j])):
                    qsr[(r, (i, j, k))] = model.addVar(
                        lb=0.0, ub=1.0, name=f"qsr[{r},{i},{j},{k}]")
                for r in _slots_for(pos, pos[i]):
                    if k != j:
                        qrr[(r, (i, j, k))] = model.addVar(
                            lb=0.0, ub=1.0, name=f"qrr[{r},{i},{j},{k}]")

    sigma_phi = {
        (r, ell): model.addVar(lb=0.0, name=f"sPhi[{r},{ell}]")
        for r in order for ell in range(phi_cert.K)
    }
    sigma_psi = {
        (r, ell): model.addVar(lb=0.0, name=f"sPsi[{r},{ell}]")
        for r in order for ell in range(psi_cert.K)
    }

    # Alias accessors -- resolve the diagonal to the aliased variable.
    def tau_var(r, i, j, k):
        return wb[(r, canon((i, j)))] if k == j else tau[(r, (i, j, k))]

    def qrr_var(r, i, j, k):
        return qr[(r, (i, j))] if k == j else qrr[(r, (i, j, k))]

    model.update()

    # -- (eq:milp:assign) every node in exactly one slot -------------------
    for i in order:
        model.addConstr(
            gp.quicksum(a[(i, r)] for r in _slots_for(pos, pos[i])) == 1,
            name=f"assign[{i}]",
        )

    # -- (eq:milp:rep) representative convention ---------------------------
    for i in order:
        for r in _slots_for(pos, pos[i]):
            if r != i:
                model.addConstr(a[(i, r)] <= a[(r, r)], name=f"rep[{i},{r}]")
    for r in order:
        model.addConstr(b[r] <= a[(r, r)], name=f"repb[{r}]")

    # -- (eq:milp:g) g_ir = a_ir * b_r -------------------------------------
    for (i, r), var in g.items():
        model.addConstr(var <= a[(i, r)], name=f"g_ub_a[{i},{r}]")
        model.addConstr(var <= b[r], name=f"g_ub_b[{i},{r}]")
        model.addConstr(var >= a[(i, r)] + b[r] - 1, name=f"g_lb[{i},{r}]")

    # -- (eq:milp:w) w_r,ij = a_ir * a_jr ----------------------------------
    for (r, (i, j)), var in w.items():
        model.addConstr(var <= a[(i, r)], name=f"w_ub_i[{r},{i},{j}]")
        model.addConstr(var <= a[(j, r)], name=f"w_ub_j[{r},{i},{j}]")
        model.addConstr(var >= a[(i, r)] + a[(j, r)] - 1, name=f"w_lb[{r},{i},{j}]")

    # -- (eq:milp:wb) wb_r,ij = w_r,ij * b_r -------------------------------
    for (r, (i, j)), var in wb.items():
        model.addConstr(var <= w[(r, (i, j))], name=f"wb_ub_w[{r},{i},{j}]")
        model.addConstr(var <= b[r], name=f"wb_ub_b[{r},{i},{j}]")
        model.addConstr(var >= w[(r, (i, j))] + b[r] - 1, name=f"wb_lb[{r},{i},{j}]")

    # -- (eq:milp:tau) tau_r,ijk = wb_r,ij * a_kr --------------------------
    for (r, (i, j, k)), var in tau.items():
        base = wb[(r, canon((i, j)))]
        model.addConstr(var <= base, name=f"tau_ub_w[{r},{i},{j},{k}]")
        model.addConstr(var <= a[(k, r)], name=f"tau_ub_a[{r},{i},{j},{k}]")
        model.addConstr(var >= base + a[(k, r)] - 1, name=f"tau_lb[{r},{i},{j},{k}]")

    # -- (eq:milp:qr) qr_r,ik = g_ir * y_ik --------------------------------
    for (r, (i, k)), var in qr.items():
        edge = canon((i, k))
        model.addConstr(var <= g[(i, r)], name=f"qr_ub_g[{r},{i},{k}]")
        model.addConstr(var <= y[edge], name=f"qr_ub_y[{r},{i},{k}]")
        model.addConstr(var >= g[(i, r)] + y[edge] - 1, name=f"qr_lb[{r},{i},{k}]")

    # -- (eq:milp:qsr) qsr_r,ijk = wb_r,ij * y_ik --------------------------
    for (r, (i, j, k)), var in qsr.items():
        base = wb[(r, canon((i, j)))]
        edge = canon((i, k))
        model.addConstr(var <= base, name=f"qsr_ub_w[{r},{i},{j},{k}]")
        model.addConstr(var <= y[edge], name=f"qsr_ub_y[{r},{i},{j},{k}]")
        model.addConstr(var >= base + y[edge] - 1, name=f"qsr_lb[{r},{i},{j},{k}]")

    # -- (eq:milp:qrr) qrr_r,ijk = qr_r,ij * y_ik --------------------------
    for (r, (i, j, k)), var in qrr.items():
        base = qr[(r, (i, j))]
        edge = canon((i, k))
        model.addConstr(var <= base, name=f"qrr_ub_q[{r},{i},{j},{k}]")
        model.addConstr(var <= y[edge], name=f"qrr_ub_y[{r},{i},{j},{k}]")
        model.addConstr(var >= base + y[edge] - 1, name=f"qrr_lb[{r},{i},{j},{k}]")

    # -- (eq:milp:elig) reconnection eligibility ---------------------------
    # z_ij = sum_r w_r,ij is an EXPRESSION, not a variable.
    z = {
        edge: gp.quicksum(
            w[(r, edge)] for r in _slots_for(pos, min(pos[edge[0]], pos[edge[1]]))
        )
        for edge in edges
    }
    for (i, j) in edges:
        model.addConstr(y[(i, j)] <= 1 - z[(i, j)], name=f"elig_cut[{i},{j}]")
        model.addConstr(
            y[(i, j)] <= gp.quicksum(g[(i, r)] for r in _slots_for(pos, pos[i])),
            name=f"elig_i[{i},{j}]",
        )
        model.addConstr(
            y[(i, j)] <= gp.quicksum(g[(j, r)] for r in _slots_for(pos, pos[j])),
            name=f"elig_j[{i},{j}]",
        )

    # -- (eq:milp:flow) single-commodity connectivity ----------------------
    # Node r itself has NO row: its supply is implied by summing the others.
    # Arcs missing from xi (because r > min(pos i, pos j)) are identically
    # zero, so they simply drop out of the sum.
    for r in order:
        for i in order:
            if pos[i] <= pos[r]:
                continue
            inflow = gp.quicksum(
                xi[(r, (j, i))] for j in neighbors[i] if (r, (j, i)) in xi
            )
            outflow = gp.quicksum(
                xi[(r, (i, j))] for j in neighbors[i] if (r, (i, j)) in xi
            )
            model.addConstr(inflow - outflow == a[(i, r)], name=f"flow[{r},{i}]")

    # -- (eq:milp:cap) flow only on edges internal to slot r ---------------
    for (r, (i, j)), w_var in w.items():
        capacity = (N - pos[r]) * w_var
        model.addConstr(xi[(r, (i, j))] <= capacity, name=f"cap_f[{r},{i},{j}]")
        model.addConstr(xi[(r, (j, i))] <= capacity, name=f"cap_b[{r},{i},{j}]")

    # -- (eq:milp:phirow) ---------------------------------------------------
    for r in order:
        r_rank = pos[r]
        node_terms = [(i, phi_cert.Phi_n[i]) for i in order if pos[i] >= r_rank]
        edge_terms = [
            (e, phi_cert.Phi_e[e]) for e in edges
            if min(pos[e[0]], pos[e[1]]) >= r_rank
        ]
        triple_terms = [
            ((i, j, k), phi_cert.Phi_t[(i, j, k)])
            for i in order for j in neighbors[i] for k in neighbors[i]
            if min(pos[i], pos[j], pos[k]) >= r_rank
        ]
        for ell in range(phi_cert.K):
            rhs = gp.quicksum(g[(i, r)] * arr[ell] for i, arr in node_terms)
            rhs += gp.quicksum(wb[(r, e)] * arr[ell] for e, arr in edge_terms)
            rhs += gp.quicksum(
                tau_var(r, i, j, k) * arr[ell] for (i, j, k), arr in triple_terms
            )
            model.addConstr(sigma_phi[(r, ell)] >= rhs, name=f"phirow[{r},{ell}]")

    # -- (eq:milp:psirow) ---------------------------------------------------
    for r in order:
        r_rank = pos[r]
        node_terms = [(i, psi_cert.Psi_n[i]) for i in order if pos[i] >= r_rank]
        edge_terms = [
            (e, psi_cert.Psi_e[e]) for e in edges
            if min(pos[e[0]], pos[e[1]]) >= r_rank
        ]
        # Ranges mirror the variable families above: qr and qrr need only
        # i in slot r; qsr needs i and j.
        pair_terms = [
            ((i, k), psi_cert.Psi_r[(i, k)])
            for i in order for k in neighbors[i]
            if pos[i] >= r_rank
        ]
        sr_terms = [
            ((i, j, k), psi_cert.Psi_sr[(i, j, k)])
            for i in order for j in neighbors[i] for k in neighbors[i]
            if min(pos[i], pos[j]) >= r_rank
        ]
        rr_terms = [
            ((i, j, k), psi_cert.Psi_rr[(i, j, k)])
            for i in order for j in neighbors[i] for k in neighbors[i]
            if pos[i] >= r_rank
        ]
        for ell in range(psi_cert.K):
            rhs = gp.quicksum(g[(i, r)] * arr[ell] for i, arr in node_terms)
            rhs += gp.quicksum(wb[(r, e)] * arr[ell] for e, arr in edge_terms)
            rhs += gp.quicksum(qr[(r, (i, k))] * arr[ell] for (i, k), arr in pair_terms)
            rhs += gp.quicksum(
                qsr[(r, (i, j, k))] * arr[ell] for (i, j, k), arr in sr_terms
            )
            rhs += gp.quicksum(
                qrr_var(r, i, j, k) * arr[ell] for (i, j, k), arr in rr_terms
            )
            model.addConstr(sigma_psi[(r, ell)] >= rhs, name=f"psirow[{r},{ell}]")

    # -- (eq:milp:obj) ------------------------------------------------------
    # alpha mapping (spec section 6): LaTeX alpha_0/1/2/3 are
    # kappa_op / kappa_rec / config.alpha_1 / config.alpha_2.
    # mu is DERIVED from K, never stored. There is no F2 term.
    alpha_0 = config.kappa_op
    alpha_1 = config.kappa_rec
    alpha_2 = config.alpha_1
    alpha_3 = config.alpha_2
    mu_phi = 1.0 / phi_cert.K
    mu_psi = 1.0 / psi_cert.K

    node_cost = gp.quicksum(
        cost_data.node_costs[i]
        * (1 - gp.quicksum(g[(i, r)] for r in _slots_for(pos, pos[i])))
        for i in order
    )
    cut_cost = alpha_0 * gp.quicksum(
        cost_data.edge_costs[e] * (1 - z[e]) for e in edges
    )
    rec_reward = alpha_1 * gp.quicksum(cost_data.edge_costs[e] * y[e] for e in edges)
    phi_pen = alpha_2 * mu_phi * gp.quicksum(sigma_phi.values())
    psi_pen = alpha_3 * mu_psi * gp.quicksum(sigma_psi.values())

    model.setObjective(
        node_cost + cut_cost - rec_reward + phi_pen + psi_pen, GRB.MINIMIZE
    )
    model.update()

    return MILPModel(
        model=model, order=order, pos=pos, edges=edges,
        a=a, b=b, y=y, g=g, w=w, wb=wb,
        tau=tau, qr=qr, qsr=qsr, qrr=qrr,
        sigma_phi=sigma_phi, sigma_psi=sigma_psi,
    )


# ---------------------------------------------------------------------------
# STEP 3 -- solver interface (spec section 7)
# ---------------------------------------------------------------------------

_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED",
    12: "NUMERIC", 13: "SUBOPTIMAL",
}


def _decode(built):
    """Read the partition off a solved model (spec section 7).

    Islands are the groups of nodes sharing a slot via a[i,r]; a slot is
    stable iff b[r] is on. Reconnected edges come from y[i,j] as canonical
    tuples. Islands are returned sorted by their minimum member, matching
    what decode_islanding produces, so the two inner solvers' results are
    directly comparable.
    """
    slots = {}
    for (i, r), var in built.a.items():
        if var.X > 0.5:
            slots.setdefault(r, set()).add(i)

    stable, unstable = [], []
    for r, members in slots.items():
        island = frozenset(members)
        if built.b[r].X > 0.5:
            stable.append(island)
        else:
            unstable.append(island)

    stable.sort(key=min)
    unstable.sort(key=min)

    reconnected = {edge for edge, var in built.y.items() if var.X > 0.5}
    return stable, unstable, reconnected


def milp_solver(problem, milp_config, initial_tree=None):
    """Solve the inner problem exactly (to the gap/time limit) as a MILP.

    Drop-in for islanding.local_search.local_search: same InnerResult, so
    islanding.outer_loop.solve takes it via its inner_solver parameter.

    `initial_tree` is accepted and IGNORED -- the MILP has no tree to warm
    start from. It exists only so the signature matches what solve()
    passes on its first pass.
    """
    import time

    from islanding.inner import InnerResult

    topology = problem.topology
    config = problem.config

    phi_cert = phi_certificates(topology, problem.model, config, problem.samples_1)
    psi_cert = psi_certificates(topology, problem.model, config, problem.samples_2)

    built = build_model(problem, milp_config, phi_cert, psi_cert)
    model = built.model

    start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start

    status = _STATUS_NAMES.get(model.Status, str(model.Status))
    if model.SolCount == 0:
        raise RuntimeError(
            f"MILP produced no feasible solution (status {status}). "
            "A time limit that expires before the first incumbent is a "
            "reportable outcome, but there is no partition to return."
        )

    stable, unstable, reconnected = _decode(built)

    return InnerResult(
        stable_islands=stable,
        unstable_islands=unstable,
        reconnected_edges=reconnected,
        objective=model.ObjVal,
        info={
            "status": status,
            "mip_gap": model.MIPGap,
            "solve_time": solve_time,
            "n_vars": model.NumVars,
            "n_constrs": model.NumConstrs,
        },
    )
