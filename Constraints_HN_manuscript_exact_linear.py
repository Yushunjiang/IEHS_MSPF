from __future__ import annotations

import gurobipy as gp
import os

from Constraints_HN_manuscript import Constraints_HN


_VERBOSE_PRODUCT_NAMES = os.environ.get("HN_VERBOSE_PRODUCT_NAMES", "0") == "1"
_PRODUCT_NAME_COUNTER = 0


def _binary_temperature_product(
    model, product, binary, temperature, active, lower, upper, name=None
):
    """Add the exact four-constraint binary-temperature product formulation.

    These constraints are generated millions of times.  Detailed names are
    disabled by default to reduce Python string allocation and Gurobi metadata
    overhead; set HN_VERBOSE_PRODUCT_NAMES=1 when detailed diagnostics are
    required.  The mathematical formulation is identical in both modes.
    """
    if _VERBOSE_PRODUCT_NAMES:
        global _PRODUCT_NAME_COUNTER
        if name is None:
            name = f"HN_ExactProduct_{_PRODUCT_NAME_COUNTER}"
            _PRODUCT_NAME_COUNTER += 1
        model.addConstr(product >= lower * binary, name=f"{name}_lb")
        model.addConstr(product <= upper * binary, name=f"{name}_ub")
        model.addConstr(
            product >= temperature - upper * (active - binary),
            name=f"{name}_link_lb",
        )
        model.addConstr(
            product <= temperature - lower * (active - binary),
            name=f"{name}_link_ub",
        )
    else:
        # A short fixed name avoids allocating a long index-based string for
        # every product while retaining the faster named-constraint path on
        # Gurobi versions where unnamed addConstr calls are slower.
        model.addConstr(product >= lower * binary, name="HNProd_lb")
        model.addConstr(product <= upper * binary, name="HNProd_ub")
        model.addConstr(
            product >= temperature - upper * (active - binary),
            name="HNProd_link_lb",
        )
        model.addConstr(
            product <= temperature - lower * (active - binary),
            name="HNProd_link_ub",
        )


def Constraints_HN_manuscript_exact_linear(model, data, var, scheme=None, include_regulation_temperature=True):
    """Build the physical-structure-driven exact linear heat-network model."""
    if getattr(data.HN, "linearization_mode", None) != "physical_exact":
        raise ValueError("physical_exact mode is required")

    Constraints_HN(
        model, data, var, scheme=scheme, include_relaxation=False,
        include_regulation_temperature=include_regulation_temperature,
    )
    hn = data.HN
    v = var.HN
    paths = frozenset(tuple(item) for item in hn.path_index)
    load_nodes = list(hn.load_node_idx)
    station_nodes = list(hn.station_node_idx)
    load_nodes_set = set(load_nodes)
    station_nodes_set = set(station_nodes)
    fixed_flow = hn.fixed_load_flow
    fixed_flow_float = fixed_flow.astype(float, copy=False)
    node_to_idx = hn.node_to_idx
    load_node_to_idx = {int(node): idx for idx, node in enumerate(load_nodes)}
    t_low = float(hn.T_R_min)
    t_high = float(hn.T_R_max)
    ts_low = float(hn.T_S_min)
    ts_high = float(hn.T_S_max)

    # Sparse path incidence is reused by the exact temperature products.
    valid_loads_by_pipe_station_year = {}
    for p, l, g, y in paths:
        valid_loads_by_pipe_station_year.setdefault((p, g, y), []).append(l)

    def present(p, l, g, y):
        return (p, l, g, y) in paths

    def x(name, p, l, g, y):
        return getattr(v, name)[p, l, g, y]

    # EL-7: node flow-temperature products.
    for n in range(hn.N_node):
        for y in range(data.year):
            for t in range(data.period):
                for s in range(data.scene.N):
                    if n in load_nodes_set:
                        l = load_node_to_idx[n]
                        fixed = fixed_flow_float[l, y]
                        model.addConstr(v.F_TSex[n, t, s, y] == -fixed * v.T_S_ex[n, t, s, y])
                        model.addConstr(v.F_TRex[n, t, s, y] == -fixed * v.T_R_ex[n, t, s, y])
                        model.addConstr(v.F_TRn[n, t, s, y] == -fixed * v.T_R_node[n, t, s, y])
                    elif n not in station_nodes_set:
                        model.addConstr(v.F_TSex[n, t, s, y] == 0)
                        model.addConstr(v.F_TRex[n, t, s, y] == 0)
                        model.addConstr(v.F_TRn[n, t, s, y] == 0)

    # Source-node affiliation-temperature products and source products.
    for g, n in enumerate(station_nodes):
        for y in range(data.year):
            active = v.y_station[g, y]
            for t in range(data.period):
                for s in range(data.scene.N):
                    for l in range(hn.N_load):
                        alpha = v.source_zone_affiliation[load_nodes[l], g, y]
                        _binary_temperature_product(model, v.z_aff_TSex[l, g, t, s, y], alpha, v.T_S_ex[n, t, s, y], active, ts_low, ts_high)
                        _binary_temperature_product(model, v.z_aff_TRex[l, g, t, s, y], alpha, v.T_R_ex[n, t, s, y], active, t_low, t_high)
                        _binary_temperature_product(model, v.z_aff_TRn[l, g, t, s, y], alpha, v.T_R_node[n, t, s, y], active, t_low, t_high)
                    model.addConstr(v.F_TSex[n, t, s, y] == gp.quicksum(float(fixed_flow[l, y]) * v.z_aff_TSex[l, g, t, s, y] for l in range(hn.N_load)), name=f"HN_ExactSourceTSex_{g}_{t}_{s}_{y}")
                    model.addConstr(v.F_TRex[n, t, s, y] == gp.quicksum(float(fixed_flow[l, y]) * v.z_aff_TRex[l, g, t, s, y] for l in range(hn.N_load)), name=f"HN_ExactSourceTRex_{g}_{t}_{s}_{y}")
                    model.addConstr(v.F_TRn[n, t, s, y] == gp.quicksum(float(fixed_flow[l, y]) * v.z_aff_TRn[l, g, t, s, y] for l in range(hn.N_load)), name=f"HN_ExactSourceTRn_{g}_{t}_{s}_{y}")

    # EL-8: return pipe flow-temperature products.
    for p in range(hn.N_pipe):
        head = node_to_idx[int(hn.head[p])]
        tail = node_to_idx[int(hn.tail[p])]
        for y in range(data.year):
            for t in range(data.period):
                for s in range(data.scene.N):
                    ij_terms = []
                    ji_terms = []
                    for g in range(hn.N_station):
                        for l in valid_loads_by_pipe_station_year.get((p, g, y), ()):
                            xij = x("x_path_ij", p, l, g, y)
                            xji = x("x_path_ji", p, l, g, y)
                            _binary_temperature_product(model, v.z_path_ij_TRi[p, l, g, y, t, s], xij, v.T_R_i[p, t, s, y], v.y_pipe[p, y], t_low, t_high)
                            _binary_temperature_product(model, v.z_path_ji_TRj[p, l, g, y, t, s], xji, v.T_R_j[p, t, s, y], v.y_pipe[p, y], t_low, t_high)
                            _binary_temperature_product(model, v.z_rin_ij_TRn[p, l, g, y, t, s], xij, v.T_R_node[head, t, s, y], v.y_node[head, y], t_low, t_high)
                            _binary_temperature_product(model, v.z_rin_ji_TRn[p, l, g, y, t, s], xji, v.T_R_node[tail, t, s, y], v.y_node[tail, y], t_low, t_high)
                            flow = fixed_flow_float[l, y]
                            ij_terms.append(flow * v.z_path_ij_TRi[p, l, g, y, t, s])
                            ji_terms.append(flow * v.z_path_ji_TRj[p, l, g, y, t, s])
                    model.addConstr(v.FR_bij_TRi[p, t, s, y] == gp.quicksum(ij_terms), name=f"HN_ExactFRijTRi_{p}_{t}_{s}_{y}")
                    model.addConstr(v.FR_bji_TRj[p, t, s, y] == -gp.quicksum(ji_terms), name=f"HN_ExactFRjiTRj_{p}_{t}_{s}_{y}")

    # EL-9: return inlet products at every node.
    for n in range(hn.N_node):
        for y in range(data.year):
            for t in range(data.period):
                for s in range(data.scene.N):
                    terms = []
                    for p in hn.set_head[n]:
                        for l in range(hn.N_load):
                            for g in range(hn.N_station):
                                if present(p, l, g, y):
                                    terms.append(float(fixed_flow[l, y]) * v.z_rin_ij_TRn[p, l, g, y, t, s])
                    for p in hn.set_tail[n]:
                        for l in range(hn.N_load):
                            for g in range(hn.N_station):
                                if present(p, l, g, y):
                                    terms.append(float(fixed_flow[l, y]) * v.z_rin_ji_TRn[p, l, g, y, t, s])
                    model.addConstr(v.FRin_TRn[n, t, s, y] == gp.quicksum(terms), name=f"HN_ExactFRinTRn_{n}_{t}_{s}_{y}")

    if model.NumQConstrs > 0:
        raise RuntimeError(f"physical_exact HN model contains {model.NumQConstrs} quadratic constraints")
    return model
