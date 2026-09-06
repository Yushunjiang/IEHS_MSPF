import gurobipy as gp
import os
import numpy as np
from gurobipy import GRB





def _validate_gamma_call(data):
    expected_shape = (data.HN.N_station, data.period, data.scene.N, data.year)
    gamma_call = getattr(data.HN, "gamma_call", None)
    if gamma_call is None:
        gamma_call = np.zeros(expected_shape, dtype=float)
    gamma_call = np.asarray(gamma_call, dtype=float)
    if gamma_call.shape != expected_shape:
        raise ValueError(
            f"HN.gamma_call shape must be {expected_shape}, got {gamma_call.shape}"
        )
    if not np.all(np.isfinite(gamma_call)):
        raise ValueError("HN.gamma_call must contain only finite values")
    if np.any(gamma_call < -1.0) or np.any(gamma_call > 1.0):
        raise ValueError("HN.gamma_call must be within [-1, 1]")
    data.HN.gamma_call = gamma_call
    data.HN.gamma_call_all_zero = bool(np.all(np.abs(gamma_call) <= 1e-9))


def _gamma_call(data, station, time, scene, year):
    gamma_ru = getattr(data.HN, "gamma_call_ru", None)
    gamma_rd = getattr(data.HN, "gamma_call_rd", None)
    if gamma_ru is not None or gamma_rd is not None:
        ru = 0.0 if gamma_ru is None else float(gamma_ru[station, time, scene, year])
        rd = 0.0 if gamma_rd is None else float(gamma_rd[station, time, scene, year])
        if not 0.0 <= ru <= 1.0 or not 0.0 <= rd <= 1.0:
            raise ValueError("HN.gamma_call_ru and gamma_call_rd must be within [0, 1]")
        return ru - rd, ru, rd, max(ru, rd)
    gamma = float(data.HN.gamma_call[station, time, scene, year])
    if gamma < -1.0 - 1e-9 or gamma > 1.0 + 1e-9:
        raise ValueError("HN.gamma_call must be within [-1, 1]")
    gamma = min(1.0, max(-1.0, gamma))
    return gamma, max(gamma, 0.0), max(-gamma, 0.0), abs(gamma)


def get_gamma_components(data, station, time, scene, year):
    return _gamma_call(data, station, time, scene, year)


def _node_heat_upper(data, node, year):
    """Return the year-specific safe heat-injection upper bound."""
    annual = getattr(data.HN, "H_node_max_by_year", None)
    if annual is not None:
        annual = np.asarray(annual, dtype=float)
        expected_shape = (int(data.HN.N_node), int(data.year))
        if annual.shape != expected_shape:
            raise ValueError(
                f"HN.H_node_max_by_year must have shape {expected_shape}, "
                f"got {annual.shape}"
            )
        return float(annual[node, year])
    return float(data.HN.H_node_max[node])


def _node_heat_interval(data, node, time, scene, year):
    """Return the local physical heat interval without using an incumbent."""
    lower_all = getattr(data.HN, "H_node_min_all", None)
    upper_all = getattr(data.HN, "H_node_max_all", None)
    if lower_all is not None and upper_all is not None:
        lower = float(lower_all[node, time, scene, year])
        upper = float(upper_all[node, time, scene, year])
    else:
        lower = 0.0
        upper = _node_heat_upper(data, node, year)
    if lower > upper + 1e-9:
        raise ValueError(
            f"empty node heat interval at {(node, time, scene, year)}: "
            f"[{lower}, {upper}]"
        )
    return min(lower, upper), max(lower, upper)


def _divide_interval_by_fixed_sign_interval(value_lower, value_upper,
                                             divisor_lower, divisor_upper):
    """Divide two intervals when the divisor excludes zero."""
    divisor_lower = float(divisor_lower)
    divisor_upper = float(divisor_upper)
    if divisor_lower <= 0.0 <= divisor_upper:
        return None
    values = (
        float(value_lower) / divisor_lower,
        float(value_lower) / divisor_upper,
        float(value_upper) / divisor_lower,
        float(value_upper) / divisor_upper,
    )
    return min(values), max(values)


def _positive_flow_lower(data, year):
    configured = getattr(data.HN, "positive_flow_lower_by_year", None)
    if configured is not None:
        return max(0.0, float(np.asarray(configured, dtype=float)[year]))
    flows = np.asarray(data.HN.f_load, dtype=float).reshape(-1)
    states = np.asarray(data.HN.load_state, dtype=float)
    active = flows[(states[:, year] > 0.5) & (flows > 1e-10)]
    return float(np.min(active)) if active.size else 0.0


def _annual_flow_upper(data, year):
    configured = getattr(data.HN, "active_load_flow_upper_by_year", None)
    if configured is not None:
        return max(0.0, float(np.asarray(configured, dtype=float)[year]))
    flows = np.asarray(data.HN.f_load, dtype=float).reshape(-1)
    states = np.asarray(data.HN.load_state, dtype=float)
    return float(np.sum(np.maximum(flows, 0.0) * states[:, year]))


def _source_flow_upper(data, station, year):
    configured = getattr(data.HN, "source_flow_upper_by_year", None)
    if configured is not None:
        return max(0.0, float(np.asarray(configured, dtype=float)[station, year]))
    return min(max(0.0, float(data.HN.f_station[station])), _annual_flow_upper(data, year))


def _source_flow_lower(data, station, year):
    """Return the topology-proven source-flow lower bound."""
    configured = getattr(data.HN, "source_flow_lower_by_year", None)
    if configured is not None:
        configured = np.asarray(configured, dtype=float)
        expected_shape = (int(data.HN.N_station), int(data.year))
        if configured.shape != expected_shape:
            raise ValueError(
                f"HN.source_flow_lower_by_year must have shape {expected_shape}, "
                f"got {configured.shape}"
            )
        return max(0.0, float(configured[station, year]))
    return _positive_flow_lower(data, year)


def _conditional_segment_temperature_bounds(
    data,
    group,
    index,
    time,
    scene,
    year,
    family,
    local_temperature_bounds,
    partition,
):
    """Return safe temperature intervals for each nonzero flow segment.

    The source heat balance uses the lower magnitude of the selected flow to
    bound the supply/return temperature difference.  Pipe return temperatures
    are intersected with the two endpoint node-return intervals implied by
    the pipe temperature equalities.  These are outer-interval operations and
    therefore cannot remove a feasible original solution.
    """
    local_lower, local_upper = map(float, local_temperature_bounds)
    points = [float(value) for value in partition["breakpoints"]]
    default = [(local_lower, local_upper) for _ in range(len(points) - 1)]
    bounds = []
    for segment in range(len(points) - 1):
        flow_lower = points[segment]
        flow_upper = points[segment + 1]
        candidate_lower, candidate_upper = local_lower, local_upper

        if group == "F_NODE" and family in {"F_TSex", "F_TRex", "F_TRn"}:
            node_id = data.HN.node[index]
            if node_id in data.HN.station:
                heat_capacity = float(data.HN.c_w)
                if heat_capacity <= 0.0:
                    raise ValueError("HN.c_w must be positive")
                if family == "F_TSex":
                    heat_time = time
                    reference_time = time - 1 if time > 0 else data.period - 1
                    heat_lower, heat_upper = _node_heat_interval(
                        data, index, heat_time, scene, year
                    )
                    delta = _divide_interval_by_fixed_sign_interval(
                        heat_lower / heat_capacity,
                        heat_upper / heat_capacity,
                        flow_lower,
                        flow_upper,
                    )
                    if delta is not None:
                        reference_lower = float(
                            data.HN.TRex_min_all[index, reference_time, scene, year]
                        )
                        reference_upper = float(
                            data.HN.TRex_max_all[index, reference_time, scene, year]
                        )
                        candidate_lower = max(
                            candidate_lower, reference_lower + delta[0]
                        )
                        candidate_upper = min(
                            candidate_upper, reference_upper + delta[1]
                        )
                else:
                    # T_R_ex(t), and therefore source T_R_node(t), enters the
                    # source heat balance at t+1 because of the one-period
                    # return delay used by the manuscript formulation.
                    heat_time = time + 1 if time + 1 < data.period else 0
                    heat_lower, heat_upper = _node_heat_interval(
                        data, index, heat_time, scene, year
                    )
                    delta = _divide_interval_by_fixed_sign_interval(
                        heat_lower / heat_capacity,
                        heat_upper / heat_capacity,
                        flow_lower,
                        flow_upper,
                    )
                    if delta is not None:
                        reference_lower = float(
                            data.HN.TSex_min_all[index, heat_time, scene, year]
                        )
                        reference_upper = float(
                            data.HN.TSex_max_all[index, heat_time, scene, year]
                        )
                        candidate_lower = max(
                            candidate_lower, reference_lower - delta[1]
                        )
                        candidate_upper = min(
                            candidate_upper, reference_upper - delta[0]
                        )

        if group in {"FR_BIJ", "FR_BJI"}:
            # T_R_i and T_R_j are equal along a built pipe.  The manuscript
            # direction constraints link the active ij return flow to the tail
            # node and the active ji return flow to the head node.  Use only
            # that proven endpoint relation; the opposite endpoint may be a
            # mixing node and must not be used for contraction.
            node_lookup = {int(node_id): node_index for node_index, node_id in enumerate(data.HN.node)}
            pipe_index = int(index)
            head = node_lookup[int(data.HN.head[pipe_index])]
            tail = node_lookup[int(data.HN.tail[pipe_index])]
            endpoint = tail if group == "FR_BIJ" else head
            candidate_lower = max(
                candidate_lower,
                float(data.HN.TRn_min_all[endpoint, time, scene, year]),
            )
            candidate_upper = min(
                candidate_upper,
                float(data.HN.TRn_max_all[endpoint, time, scene, year]),
            )

        if group == "FR_IN":
            candidate_lower = max(candidate_lower, float(data.HN.TRn_min_all[index, time, scene, year]))
            candidate_upper = min(candidate_upper, float(data.HN.TRn_max_all[index, time, scene, year]))
        lower = max(local_lower, candidate_lower)
        upper = min(local_upper, candidate_upper)
        if lower > upper + 1e-9:
            raise ValueError(
                "empty flow-segment-conditional temperature interval: "
                f"family={family}, group={group}, index={index}, "
                f"time={time}, scene={scene}, year={year}, segment={segment}, "
                f"flow=[{flow_lower},{flow_upper}], "
                f"local_temperature=[{local_lower},{local_upper}], "
                f"candidate_temperature=[{candidate_lower},{candidate_upper}]"
            )
        else:
            bounds.append((min(lower, upper), max(lower, upper)))
    return bounds


def _ensure_variables(model, data, hn):
    shape = (data.HN.N_station, data.period, data.scene.N, data.year)
    if not hasattr(hn, "Lambda_H"):
        hn.Lambda_H = model.addVars(
            data.HN.N_station, data.period, data.scene.N, data.year,
            lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_Lambda_H"
        )


def _ensure_flow_activity_variables(model, data, hn):
    """Create binary states for actual nonzero aggregate flows.

    The states are deliberately endogenous: an installed station/pipe may be
    unused, so installation or connectivity alone must not force a positive
    flow. Each state is gated by the corresponding planning/direction binary
    and is forced to one whenever the associated continuous flow is nonzero by
    the bounds added below.
    """
    if not hasattr(hn, "source_flow_active"):
        hn.source_flow_active = model.addVars(
            data.HN.N_station, data.year, vtype=GRB.BINARY,
            name="HN_source_flow_active"
        )
    if not hasattr(hn, "inlet_flow_active"):
        hn.inlet_flow_active = model.addVars(
            data.HN.N_node, data.year, vtype=GRB.BINARY,
            name="HN_inlet_flow_active"
        )
    if not hasattr(hn, "pipe_flow_active_ij"):
        hn.pipe_flow_active_ij = model.addVars(
            data.HN.N_pipe, data.year, vtype=GRB.BINARY,
            name="HN_pipe_flow_active_ij"
        )
    if not hasattr(hn, "pipe_flow_active_ji"):
        hn.pipe_flow_active_ji = model.addVars(
            data.HN.N_pipe, data.year, vtype=GRB.BINARY,
            name="HN_pipe_flow_active_ji"
        )


def add_manuscript_regulation_capacity(model, data, var, include_regulation_temperature=True):
    """Add load-side regulation margins and bidirectional residual heat limits."""
    hn = var.HN
    _ensure_variables(model, data, hn)
    capacity_based_totals = bool(
        getattr(data.HN, "ntc_capacity_regulation_totals", False)
    )
    c_w = float(data.HN.c_w)
    temperature_range = max(
        float(data.HN.T_S_max) - float(data.HN.T_S_min),
        float(data.HN.T_R_max) - float(data.HN.T_R_min),
        1.0,
    )
    maximum_flow = max((max(0.0, float(flow)) for flow in data.HN.f_load), default=0.0)
    max_regulation_heat = max(
        abs(float(data.CHP.k_HE)) * max(float(value) for value in data.HN.S_CHP_max),
        abs(float(data.EB.k_HE)) * max(float(value) for value in data.HN.S_EB_max),
        0.0,
    )
    required_big_m = c_w * maximum_flow * temperature_range + max_regulation_heat
    big_m = float(getattr(data.HN, "M_H", required_big_m))
    if big_m + 1e-9 < required_big_m:
        raise ValueError(f"HN.M_H must be at least {required_big_m}, got {big_m}")
    data.HN.M_H_used = big_m
    data.HN.regulation_D_max_used = 0
    data.HN.manuscript_regulation_external_gamma = True

    for year in range(data.year):
        for scene in range(data.scene.N):
            for time in range(data.period):
                for station in range(data.HN.N_station):
                    chp_power = var.CHP.P[station, time, scene, year]
                    eb_power = var.EB.P[station, time, scene, year]
                    # Load-side convention: RU increases electric consumption;
                    # RD decreases electric consumption.
                    model.addConstr(
                        var.CHP.P_RU[station, time, scene, year]
                        <= hn.S_CHP[station, year] - chp_power,
                        name=f"HN_Manuscript_CHPUp_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.CHP.P_RU1[station, time, scene, year]
                        <= hn.S_CHP[station, year] - chp_power,
                        name=f"HN_Manuscript_CHPUp1_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.CHP.P_RD[station, time, scene, year] <= chp_power,
                        name=f"HN_Manuscript_CHPDown_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.CHP.P_RD1[station, time, scene, year] <= chp_power,
                        name=f"HN_Manuscript_CHPDown1_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.EB.P_RU[station, time, scene, year]
                        <= hn.S_EB[station, year] - eb_power,
                        name=f"HN_Manuscript_EBUp_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.EB.P_RU1[station, time, scene, year]
                        <= hn.S_EB[station, year] - eb_power,
                        name=f"HN_Manuscript_EBUp1_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.EB.P_RD[station, time, scene, year] <= eb_power,
                        name=f"HN_Manuscript_EBDown_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.EB.P_RD1[station, time, scene, year] <= eb_power,
                        name=f"HN_Manuscript_EBDown1_{station}_{time}_{scene}_{year}",
                    )
                    # H_RU is the signed heat change associated with increasing
                    # electric load: CHP-down decreases heat and EB-up increases it.
                    model.addConstr(
                        hn.H_RU[station, time, scene, year]
                        == -float(data.CHP.k_HE) * var.CHP.P_RD[station, time, scene, year]
                        + float(data.EB.k_HE) * var.EB.P_RU[station, time, scene, year],
                        name=f"HN_Manuscript_HeatUp_{station}_{time}_{scene}_{year}",
                    )
                    # H_RD is the signed heat change associated with decreasing
                    # electric load: CHP-up increases heat and EB-down decreases it.
                    model.addConstr(
                        hn.H_RD[station, time, scene, year]
                        == float(data.CHP.k_HE) * var.CHP.P_RU[station, time, scene, year]
                        - float(data.EB.k_HE) * var.EB.P_RD[station, time, scene, year],
                        name=f"HN_Manuscript_HeatDown_{station}_{time}_{scene}_{year}",
                    )
                    if include_regulation_temperature:
                        model.addConstr(
                            hn.Lambda_H[station, time, scene, year] == 0,
                            name=f"HN_LegacyResidualThermalMargin_{station}_{time}_{scene}_{year}",
                        )
    
                        for load in range(data.HN.N_load):
                            node = data.HN.load_node_idx[load]
                            affiliation = hn.source_zone_affiliation[node, station, year]
                            flow = float(data.HN.f_load[load]) * float(data.HN.load_state[load, year])
                            inactive = (1 - affiliation) * big_m
                            heat_up = hn.H_RU[station, time, scene, year]
                            heat_down = hn.H_RD[station, time, scene, year]
    
                            supply_lower = (
                                -c_w * flow
                                * (
                                    hn.T_S_node[node, time, scene, year]
                                    - float(data.HN.T_S_min)
                                )
                                - inactive
                            )
                            supply_upper = (
                                c_w * flow
                                * (
                                    float(data.HN.T_S_max)
                                    - hn.T_S_node[node, time, scene, year]
                                )
                                + inactive
                            )
                            return_lower = (
                                -c_w * flow
                                * (
                                    hn.T_R_node[node, time, scene, year]
                                    - float(data.HN.T_R_min)
                                )
                                - inactive
                            )
                            return_upper = (
                                c_w * flow
                                * (
                                    float(data.HN.T_R_max)
                                    - hn.T_R_node[node, time, scene, year]
                                )
                                + inactive
                            )
    
                            model.addConstr(
                                supply_lower <= heat_up,
                                name=f"HN_Manuscript_SupplyLowerUp_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                heat_up <= supply_upper,
                                name=f"HN_Manuscript_SupplyUpperUp_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                return_lower <= heat_up,
                                name=f"HN_Manuscript_ReturnLowerUp_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                heat_up <= return_upper,
                                name=f"HN_Manuscript_ReturnUpperUp_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                supply_lower <= heat_down,
                                name=f"HN_Manuscript_SupplyLowerDown_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                heat_down <= supply_upper,
                                name=f"HN_Manuscript_SupplyUpperDown_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                return_lower <= heat_down,
                                name=f"HN_Manuscript_ReturnLowerDown_{load}_{station}_{time}_{scene}_{year}",
                            )
                            model.addConstr(
                                heat_down <= return_upper,
                                name=f"HN_Manuscript_ReturnUpperDown_{load}_{station}_{time}_{scene}_{year}",
                            )
                if capacity_based_totals:
                    ru_terms = (
                        var.CHP.P_RD1[station, time, scene, year]
                        + var.EB.P_RU1[station, time, scene, year]
                        for station in range(data.HN.N_station)
                    )
                else:
                    ru_terms = (
                        var.CHP.P_RD[station, time, scene, year]
                        + var.EB.P_RU[station, time, scene, year]
                        for station in range(data.HN.N_station)
                    )
                model.addConstr(
                    hn.P_RU_all[time, scene, year] == gp.quicksum(ru_terms),
                    name=f"HN_Manuscript_TotalUp_{time}_{scene}_{year}",
                )
                model.addConstr(
                    hn.P_RU_all1[time, scene, year] == gp.quicksum(
                        var.CHP.P_RD1[station, time, scene, year]
                        + var.EB.P_RU1[station, time, scene, year]
                        for station in range(data.HN.N_station)
                    ),
                    name=f"HN_Manuscript_TotalUp1_{time}_{scene}_{year}",
                )
                if capacity_based_totals:
                    rd_terms = (
                        var.CHP.P_RU1[station, time, scene, year]
                        + var.EB.P_RD1[station, time, scene, year]
                        for station in range(data.HN.N_station)
                    )
                else:
                    rd_terms = (
                        var.CHP.P_RU[station, time, scene, year]
                        + var.EB.P_RD[station, time, scene, year]
                        for station in range(data.HN.N_station)
                    )
                model.addConstr(
                    hn.P_RD_all[time, scene, year] == gp.quicksum(rd_terms),
                    name=f"HN_Manuscript_TotalDown_{time}_{scene}_{year}",
                )
                model.addConstr(
                    hn.P_RD_all1[time, scene, year] == gp.quicksum(
                        var.CHP.P_RU1[station, time, scene, year]
                        + var.EB.P_RD1[station, time, scene, year]
                        for station in range(data.HN.N_station)
                    ),
                    name=f"HN_Manuscript_TotalDown1_{time}_{scene}_{year}",
                )

def Constraints_HN(model, data, var, scheme=None, include_relaxation=True, include_regulation_temperature=True):
    data.HN.reverse_flow_sign_convention = "signed_negative_bji"
    _ensure_flow_activity_variables(model, data, var.HN)
    # =========================================================================
    # 0. 预设方案约束 (固定二进制变量)
    # =========================================================================
    if scheme is not None and hasattr(scheme, 'HN'):
        model.addConstrs(
            (var.HN.y_pipe[i, y] == scheme.HN.y_pipe[i, y]
             for y in range(data.year) for i in range(data.HN.N_pipe)),
            name="Scheme_HN_y_pipe",
        )
        model.addConstrs(
            (var.HN.b_ij[i, y] == scheme.HN.b_ij[i, y]
             for y in range(data.year) for i in range(data.HN.N_pipe)),
            name="Scheme_HN_b_ij",
        )
        model.addConstrs(
            (var.HN.b_ji[i, y] == scheme.HN.b_ji[i, y]
             for y in range(data.year) for i in range(data.HN.N_pipe)),
            name="Scheme_HN_b_ji",
        )

    # =========================================================================
    # 1. 供热站规划 (CHP, EB)
    # =========================================================================
    for i in range(data.HN.N_station):
        for y in range(data.year):
            # 供热站状态保持
            model.addConstr(var.HN.y_station[i, y] >= data.HN.y_station[i], name=f"HN_y_station_{i}_{y}")

            # CHP规划与总容量限制
            model.addConstr(var.HN.S_CHP_plan[i, y] == var.HN.N_CHP[i, y] * data.CHP.S_ref, name=f"HN_S_CHP_plan_{i}_{y}")
            model.addConstr(var.HN.S_CHP[i, y] <= data.HN.S_CHP_max[i] * var.HN.y_station[i, y], name=f"HN_S_CHP_max_{i}_{y}")
            model.addConstr(var.HN.S_CHP[i, y] >= data.HN.S_CHP[i] * var.HN.y_station[i, y], name=f"HN_S_CHP_min_{i}_{y}")
            model.addConstr(var.HN.S_CHP_plan[i, y] <= 500, name=f"HN_S_CHP_plan_{i}_{y}")

            # EB规划与总容量限制
            model.addConstr(var.HN.S_EB_plan[i, y] == var.HN.N_EB[i, y] * data.EB.S_ref, name=f"HN_S_EB_plan_{i}_{y}")
            model.addConstr(var.HN.S_EB[i, y] <= data.HN.S_EB_max[i] * var.HN.y_station[i, y], name=f"HN_S_EB_max_{i}_{y}")
            model.addConstr(var.HN.S_EB[i, y] >= data.HN.S_EB[i] * var.HN.y_station[i, y], name=f"HN_S_EB_min_{i}_{y}")

            # 建设完成与退役 (CHP)
            if y >= data.CHP.T_build:
                model.addConstr(var.HN.S_CHP_plan_e[i, y] == var.HN.S_CHP_plan[i, y - int(data.CHP.T_build)])
            else:
                model.addConstr(var.HN.S_CHP_plan_e[i, y] == 0)

            if y >= data.CHP.T_life:
                model.addConstr(var.HN.S_CHP_plan_r[i, y] == var.HN.S_CHP_plan_e[i, y - int(data.CHP.T_life)])
            else:
                model.addConstr(var.HN.S_CHP_plan_r[i, y] == 0)

            # 建设完成与退役 (EB)
            if y >= data.EB.T_build:
                model.addConstr(var.HN.S_EB_plan_e[i, y] == var.HN.S_EB_plan[i, y - int(data.EB.T_build)])
            else:
                model.addConstr(var.HN.S_EB_plan_e[i, y] == 0)

            if y >= data.EB.T_life:
                model.addConstr(var.HN.S_EB_plan_r[i, y] == var.HN.S_EB_plan_e[i, y - int(data.EB.T_life)])
            else:
                model.addConstr(var.HN.S_EB_plan_r[i, y] == 0)

            # 可用容量聚合
            exp_e_chp = gp.quicksum(var.HN.S_CHP_plan_e[i, yy] for yy in range(y + 1))
            exp_r_chp = gp.quicksum(var.HN.S_CHP_plan_r[i, yy] for yy in range(y + 1))
            model.addConstr(var.HN.S_CHP_plan_u[i, y] == exp_e_chp - exp_r_chp)
            model.addConstr(var.HN.S_CHP[i, y] == var.HN.S_CHP_plan_u[i, y] + data.HN.S_CHP[i])

        # CHP总供热容量上限：S_CHP为电功率容量，需乘以k_HE转换为
        # CHP供热容量，再按年内最大逐时总热负荷的一定比例进行限制。
            exp_e_eb = gp.quicksum(var.HN.S_EB_plan_e[i, yy] for yy in range(y + 1))
            exp_r_eb = gp.quicksum(var.HN.S_EB_plan_r[i, yy] for yy in range(y + 1))
            model.addConstr(var.HN.S_EB_plan_u[i, y] == exp_e_eb - exp_r_eb)
            model.addConstr(var.HN.S_EB[i, y] == var.HN.S_EB_plan_u[i, y] + data.HN.S_EB[i])

    # for y in range(data.year):
    #     chp_heat_capacity_ratio = float(data.HN.CHP_heat_capacity_ratio)
    #     chp_heat_capacity_limit = chp_heat_capacity_ratio * float(data.HN.H_peak_by_year[y])
    #     model.addConstr(
    #         float(data.CHP.k_HE) * gp.quicksum(
    #             var.HN.S_CHP[i, y] for i in range(data.HN.N_station)
    #         ) <= chp_heat_capacity_limit,
    #         name=f"HN_CHP_TotalHeatCapacityRatio_{y}",
    #     )

    # =========================================================================
    # 2. 热网管道扩容规划
    # =========================================================================
    for i in range(data.HN.N_pipe):
        for y in range(data.year):
            if y + 1 >= data.year - data.pipe.T_build:
                model.addConstr(var.HN.N_pipe_exp[i, y] == 0)

            if y >= data.pipe.T_build:
                model.addConstr(var.HN.N_pipe_exp_e[i, y] == var.HN.N_pipe_exp[i, y - int(data.pipe.T_build)])
            else:
                model.addConstr(var.HN.N_pipe_exp_e[i, y] == 0)

            if y >= data.pipe.T_life:
                model.addConstr(var.HN.N_pipe_exp_r[i, y] == var.HN.N_pipe_exp_e[i, y - int(data.pipe.T_life)])
            else:
                model.addConstr(var.HN.N_pipe_exp_r[i, y] == 0)

            exp_e_pipe = gp.quicksum(var.HN.N_pipe_exp_e[i, yy] for yy in range(y + 1))
            exp_r_pipe = gp.quicksum(var.HN.N_pipe_exp_r[i, yy] for yy in range(y + 1))
            model.addConstr(var.HN.N_pipe_exp_u[i, y] == exp_e_pipe - exp_r_pipe)
            model.addConstr(var.HN.N_pipe[i, y] == data.HN.N_pipe_initial[i] + var.HN.N_pipe_exp_u[i, y])

    # =========================================================================
    # 3. 投资成本与残值计算
    # =========================================================================
    for y in range(data.year):
        # 管道投资
        pipe_inv = gp.quicksum(data.pipe.c_inv * var.HN.N_pipe_exp[i, y] * data.HN.L_pipe[i] for i in range(data.HN.N_pipe))
        model.addConstr(var.HN.C_pipe_inv[y] == pipe_inv / ((1 + data.r) ** (y + 1)))
        if y + 1 >= data.year - data.pipe.T_life - data.pipe.T_build:
            res_val = (var.HN.C_pipe_inv[y] - (data.year - (y + 1)) * var.HN.C_pipe_inv[y] / data.pipe.T_life) / ((1 + data.r) ** data.year)
            model.addConstr(var.HN.C_pipe_res[y] == res_val)
        else:
            model.addConstr(var.HN.C_pipe_res[y] == 0)

        # CHP投资
        chp_inv = data.CHP.c_inv * gp.quicksum(var.HN.S_CHP_plan[i, y] for i in range(data.HN.N_station))
        model.addConstr(var.HN.C_CHP_inv[y] == chp_inv / ((1 + data.r) ** (y + 1)))
        if y + 1 >= data.year - data.CHP.T_life - data.CHP.T_build:
            res_val = (var.HN.C_CHP_inv[y] - (data.year - (y + 1)) * var.HN.C_CHP_inv[y] / data.CHP.T_life) / ((1 + data.r) ** data.year)
            model.addConstr(var.HN.C_CHP_res[y] == res_val)
        else:
            model.addConstr(var.HN.C_CHP_res[y] == 0)

        # EB投资
        eb_inv = data.EB.c_inv * gp.quicksum(var.HN.S_EB_plan[i, y] for i in range(data.HN.N_station))
        model.addConstr(var.HN.C_EB_inv[y] == eb_inv / ((1 + data.r) ** (y + 1)))
        if y + 1 >= data.year - data.EB.T_life - data.EB.T_build:
            res_val = (var.HN.C_EB_inv[y] - (data.year - (y + 1)) * var.HN.C_EB_inv[y] / data.EB.T_life) / ((1 + data.r) ** data.year)
            model.addConstr(var.HN.C_EB_res[y] == res_val)
        else:
            model.addConstr(var.HN.C_EB_res[y] == 0)

        model.addConstr(var.HN.C_inv[y] == var.HN.C_pipe_inv[y] + var.HN.C_CHP_inv[y] + var.HN.C_EB_inv[y])
        model.addConstr(var.HN.C_res[y] == var.HN.C_pipe_res[y] + var.HN.C_CHP_res[y] + var.HN.C_EB_res[y])

    # =========================================================================
    # 4. 热网拓扑与生成树约束
    # =========================================================================
    for y in range(data.year):
        for i in range(data.HN.N_pipe):
            model.addConstr(var.HN.b_ij[i, y] + var.HN.b_ji[i, y] == var.HN.y_pipe[i, y], name=f"HN_PipeDirSum_{i}_{y}")
            model.addConstr(var.HN.y_pipe[i, y] <= var.HN.N_pipe[i, y], name=f"HN_PipeBuiltLink_{i}_{y}")

        for i in range(data.HN.N_node):
            node_id = data.HN.node[i]
            tail_sum = gp.quicksum(var.HN.b_ij[l, y] for l in data.HN.set_tail[i])
            head_sum = gp.quicksum(var.HN.b_ji[l, y] for l in data.HN.set_head[i])
            model.addConstr(var.HN.c[i, y] == tail_sum + head_sum, name=f"HN_NodeConnectivity_{i}_{y}")

            if node_id in data.HN.station:
                j = list(data.HN.station).index(node_id)
                model.addConstr(var.HN.c[i, y] <= 1 - var.HN.y_station[j, y], name=f"HN_StationConnectivity_{i}_{y}")
                pipe_sum = gp.quicksum(var.HN.y_pipe[l, y] for l in data.HN.set_tail[i]) + gp.quicksum(var.HN.y_pipe[l, y] for l in data.HN.set_head[i])
                model.addConstr(pipe_sum <= (var.HN.c[i, y] + var.HN.y_station[j, y]) * 6, name=f"HN_StationPipeLink_{i}_{y}")
                model.addConstr(var.HN.y_node[i, y] == var.HN.y_station[j, y], name=f"HN_StationNodeActive_{i}_{y}")
            elif node_id in data.HN.load:
                j = list(data.HN.load).index(node_id)
                model.addConstr(var.HN.c[i, y] >= data.HN.load_state[j, y], name=f"HN_LoadConnected_{i}_{y}")
                model.addConstr(var.HN.y_node[i, y] == var.HN.c[i, y], name=f"HN_LoadNodeActive_{i}_{y}")
            else:
                model.addConstr(var.HN.c[i, y] <= 1, name=f"HN_OtherNodeDegree_{i}_{y}")
                model.addConstr(var.HN.y_node[i, y] == var.HN.c[i, y], name=f"HN_OtherNodeActive_{i}_{y}")

    # =========================================================================
    # 5. 设备运行约束 (CHP, EB) 与配电网耦合
    # =========================================================================
    for i in range(data.HN.N_station):
        for y in range(data.year):
            for s in range(data.scene.N):
                for t in range(data.period):
                    # CHP 转换关系
                    # Baseline CHP/EB schedules define available regulation margins.
                    model.addConstr(var.CHP.G[i, t, s, y] * float(data.CHP.k_EG) == var.CHP.P[i, t, s, y])
                    model.addConstr(var.CHP.H[i, t, s, y] == var.CHP.P[i, t, s, y] * float(data.CHP.k_HE))
                    model.addConstr(var.CHP.P[i, t, s, y] <= var.HN.S_CHP[i, y])
                    model.addConstr(var.CHP.P[i, t, s, y] >= 0)
                    model.addConstr(var.EB.H[i, t, s, y] == var.EB.P[i, t, s, y] * float(data.EB.k_HE))
                    model.addConstr(var.EB.P[i, t, s, y] <= var.HN.S_EB[i, y])
                    model.addConstr(var.EB.P[i, t, s, y] >= 0)

                    # Actual CHP/EB operating states after the scheduled call.
                    _, gamma_plus, gamma_minus, _ = _gamma_call(data, i, t, s, y)
                    model.addConstr(
                        var.CHP.P_call[i, t, s, y]
                        == var.CHP.P[i, t, s, y]
                        - gamma_plus * var.CHP.P_RD[i, t, s, y]
                        + gamma_minus * var.CHP.P_RU[i, t, s, y],
                        name=f"CHP_CalledPower_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.EB.P_call[i, t, s, y]
                        == var.EB.P[i, t, s, y]
                        + gamma_plus * var.EB.P_RU[i, t, s, y]
                        - gamma_minus * var.EB.P_RD[i, t, s, y],
                        name=f"EB_CalledPower_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.CHP.P_call[i, t, s, y] <= var.HN.S_CHP[i, y],
                        name=f"CHP_CalledCapacity_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.EB.P_call[i, t, s, y] <= var.HN.S_EB[i, y],
                        name=f"EB_CalledCapacity_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.CHP.H_call[i, t, s, y]
                        == var.CHP.P_call[i, t, s, y] * float(data.CHP.k_HE),
                        name=f"CHP_CalledHeat_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.CHP.G_call[i, t, s, y] * float(data.CHP.k_EG)
                        == var.CHP.P_call[i, t, s, y],
                        name=f"CHP_CalledGas_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.EB.H_call[i, t, s, y]
                        == var.EB.P_call[i, t, s, y] * float(data.EB.k_HE),
                        name=f"EB_CalledHeat_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.emission.CO[i, t, s, y]
                        == var.CHP.P_call[i, t, s, y] * float(data.CHP.k_CO),
                        name=f"CHP_CalledCO_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.emission.SO[i, t, s, y]
                        == var.CHP.P_call[i, t, s, y] * float(data.CHP.k_SO),
                        name=f"CHP_CalledSO_{i}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.emission.NO[i, t, s, y]
                        == var.CHP.P_call[i, t, s, y] * float(data.CHP.k_NO),
                        name=f"CHP_CalledNO_{i}_{t}_{s}_{y}",
                    )

    # 热网总体备用能力限制
    add_manuscript_regulation_capacity(model, data, var, include_regulation_temperature=include_regulation_temperature)

    # 供热站与配电网节点功率耦合
    for i in range(data.DN.N_node):
        for y in range(data.year):
            for s in range(data.scene.N):
                for t in range(data.period):
                    idx_list = [j for j, val in enumerate(data.HN.station_DN) if val == data.DN.node[i]]
                    if len(idx_list) > 0:
                        p_sum = gp.quicksum(var.HN.P[j, t, s, y] for j in idx_list)
                        # q_sum = gp.quicksum(var.HN.Q[j, t, s, y] for j in idx_list)
                        model.addConstr(var.HN.P_DN[i, t, s, y] == p_sum)
                        # model.addConstr(var.HN.Q_DN[i, t, s, y] == q_sum)
                    else:
                        model.addConstr(var.HN.P_DN[i, t, s, y] == 0)
                        # model.addConstr(var.HN.Q_DN[i, t, s, y] == 0)

    # =========================================================================
    # 6. 热负荷与节点功率、温度计算
    # =========================================================================
    c_w = float(data.HN.c_w)
    m_t = float(data.HN.M_T)
    m_f = float(data.HN.M_F)
    m_ft = float(data.HN.M_FT)

    for y in range(data.year):
        for s in range(data.scene.N):
            for t in range(data.period):
                t_prev = t - 1 if t > 0 else data.period - 1

                # 计算实际负荷
                for j in range(data.HN.N_load):
                    act_load = float(data.HN.H_load[j]) * float(data.HN.H_year[y]) * float(data.HN.load_state[j, y]) * float(data.scene.k_H[t, s])
                    model.addConstr(var.HN.H_load[j, t, s, y] == act_load)

                # 节点热功率、温度延时混合模型
                for i in range(data.HN.N_node):
                    node_id = data.HN.node[i]
                    model.addConstr(
                        var.HN.H_node[i, t, s, y] <= _node_heat_upper(data, i, y)
                    )

                    if node_id in data.HN.station:
                        j = list(data.HN.station).index(node_id)
                        model.addConstr(
                            var.HN.H_node[i, t, s, y]
                            == var.CHP.H_call[j, t, s, y] + var.EB.H_call[j, t, s, y],
                            name=f"HN_ActualHeatBalance_{j}_{t}_{s}_{y}",
                        )
                        model.addConstr(var.HN.H_node[i, t, s, y] == c_w * (var.HN.F_TSex[i, t, s, y] - var.HN.F_TRex[i, t_prev, s, y]))
                        model.addConstr(
                            var.HN.P[j, t, s, y]
                            == var.EB.P_call[j, t, s, y] - var.CHP.P_call[j, t, s, y],
                            name=f"HN_ActualElectricCoupling_{j}_{t}_{s}_{y}",
                        )
                    elif node_id in data.HN.load:
                        j = list(data.HN.load).index(node_id)
                        model.addConstr(var.HN.H_node[i, t, s, y] == -var.HN.H_load[j, t, s, y])
                        model.addConstr(var.HN.H_node[i, t, s, y] == c_w * (var.HN.F_TSex[i, t, s, y] - var.HN.F_TRex[i, t, s, y]))
                    else:
                        model.addConstr(var.HN.H_node[i, t, s, y] == 0)
                        model.addConstr(c_w * (var.HN.F_TSex[i, t, s, y] - var.HN.F_TRex[i, t, s, y]) == 0)

                    # 温度边界
                    ts_min = float(data.HN.TSex_min_all[i, t, s, y])
                    ts_max = float(data.HN.TSex_max_all[i, t, s, y])
                    trn_min = float(data.HN.TRn_min_all[i, t, s, y])
                    trn_max = float(data.HN.TRn_max_all[i, t, s, y])
                    trex_min = float(data.HN.TRex_min_all[i, t, s, y])
                    trex_max = float(data.HN.TRex_max_all[i, t, s, y])
                    model.addConstr(var.HN.T_S_node[i, t, s, y] <= ts_max * var.HN.y_node[i, y])
                    model.addConstr(var.HN.T_S_node[i, t, s, y] >= ts_min * var.HN.y_node[i, y])
                    model.addConstr(var.HN.T_R_node[i, t, s, y] <= trn_max * var.HN.y_node[i, y])
                    model.addConstr(var.HN.T_R_node[i, t, s, y] >= trn_min * var.HN.y_node[i, y])

                    model.addConstr(var.HN.T_S_ex[i, t, s, y] <= ts_max * var.HN.y_node[i, y])
                    model.addConstr(var.HN.T_S_ex[i, t, s, y] >= ts_min * var.HN.y_node[i, y])
                    model.addConstr(var.HN.T_R_ex[i, t, s, y] <= trex_max * var.HN.y_node[i, y])
                    model.addConstr(var.HN.T_R_ex[i, t, s, y] >= trex_min * var.HN.y_node[i, y])

                    model.addConstr(var.HN.T_S_ex[i, t, s, y] == var.HN.T_S_node[i, t, s, y])
                    if node_id in data.HN.station:
                        model.addConstr(var.HN.T_R_ex[i, t, s, y] == var.HN.T_R_node[i, t, s, y])
                    elif node_id in data.HN.load:
                        # A positive assigned commodity must connect this load
                        # to its source.  With the manuscript no-loss pipe
                        # equalities, supply temperature is constant on that
                        # active source-load path.  Reuse the existing source
                        # affiliation binary; no temperature selector is added.
                        for station in range(data.HN.N_station):
                            source_node = data.HN.station_node_idx[station]
                            affiliation = var.HN.source_zone_affiliation[i, station, y]
                            model.addConstr(
                                var.HN.T_S_node[i, t, s, y]
                                - var.HN.T_S_node[source_node, t, s, y]
                                <= (1 - affiliation) * m_t,
                                name=f"HN_SourceZoneTSUp_{i}_{station}_{t}_{s}_{y}",
                            )
                            model.addConstr(
                                var.HN.T_S_node[i, t, s, y]
                                - var.HN.T_S_node[source_node, t, s, y]
                                >= -(1 - affiliation) * m_t,
                                name=f"HN_SourceZoneTSLo_{i}_{station}_{t}_{s}_{y}",
                            )
                            load_delta = getattr(
                                data.HN, "load_temperature_delta_all", None
                            )
                            if load_delta is not None:
                                delta = float(load_delta[i, t, s, y])
                                # When this existing affiliation is selected,
                                # fixed load flow and heat demand imply
                                # T_R_ex = T_S_source - delta exactly.
                                source_return_residual = (
                                    var.HN.T_R_ex[i, t, s, y]
                                    - var.HN.T_S_node[source_node, t, s, y]
                                    + delta
                                )
                                model.addConstr(
                                    source_return_residual
                                    <= (1 - affiliation) * m_t,
                                    name=f"HN_SourceZoneTRUp_{i}_{station}_{t}_{s}_{y}",
                                )
                                model.addConstr(
                                    source_return_residual
                                    >= -(1 - affiliation) * m_t,
                                    name=f"HN_SourceZoneTRLo_{i}_{station}_{t}_{s}_{y}",
                                )

                # 支路温度约束及节点关联
                for i in range(data.HN.N_pipe):
                    model.addConstr(var.HN.T_S_i[i, t, s, y] <= float(data.HN.T_S_max) * var.HN.y_pipe[i, y])
                    model.addConstr(var.HN.T_S_i[i, t, s, y] >= float(data.HN.T_S_min) * var.HN.y_pipe[i, y])
                    model.addConstr(var.HN.T_S_j[i, t, s, y] <= float(data.HN.T_S_max) * var.HN.y_pipe[i, y])
                    model.addConstr(var.HN.T_S_j[i, t, s, y] >= float(data.HN.T_S_min) * var.HN.y_pipe[i, y])

                    tri_min = float(data.HN.TRi_min_all[i, t, s, y])
                    tri_max = float(data.HN.TRi_max_all[i, t, s, y])
                    trj_min = float(data.HN.TRj_min_all[i, t, s, y])
                    trj_max = float(data.HN.TRj_max_all[i, t, s, y])
                    model.addConstr(var.HN.T_R_i[i, t, s, y] <= tri_max * var.HN.y_pipe[i, y])
                    model.addConstr(var.HN.T_R_i[i, t, s, y] >= tri_min * var.HN.y_pipe[i, y])
                    model.addConstr(var.HN.T_R_j[i, t, s, y] <= trj_max * var.HN.y_pipe[i, y])
                    model.addConstr(var.HN.T_R_j[i, t, s, y] >= trj_min * var.HN.y_pipe[i, y])

                    # 无延时假设
                    model.addConstr(var.HN.T_S_i[i, t, s, y] == var.HN.T_S_j[i, t, s, y])
                    model.addConstr(var.HN.T_R_i[i, t, s, y] == var.HN.T_R_j[i, t, s, y])

                # 温度大 M 关联方程
                for i in range(data.HN.N_node):
                    # 供水管温度
                    for l in data.HN.set_head[i]:
                        model.addConstr(var.HN.T_S_node[i, t, s, y] - var.HN.T_S_i[l, t, s, y] <= (1 - var.HN.y_pipe[l, y]) * m_t)
                        model.addConstr(var.HN.T_S_node[i, t, s, y] - var.HN.T_S_i[l, t, s, y] >= -(1 - var.HN.y_pipe[l, y]) * m_t)
                    for l in data.HN.set_tail[i]:
                        model.addConstr(var.HN.T_S_node[i, t, s, y] - var.HN.T_S_j[l, t, s, y] <= (1 - var.HN.y_pipe[l, y]) * m_t)
                        model.addConstr(var.HN.T_S_node[i, t, s, y] - var.HN.T_S_j[l, t, s, y] >= -(1 - var.HN.y_pipe[l, y]) * m_t)

                    # 回水管温度
                    for l in data.HN.set_tail[i]:
                        model.addConstr(var.HN.T_R_node[i, t, s, y] - var.HN.T_R_j[l, t, s, y] <= (1 - var.HN.b_ij[l, y]) * m_t)
                        model.addConstr(var.HN.T_R_node[i, t, s, y] - var.HN.T_R_j[l, t, s, y] >= -(1 - var.HN.b_ij[l, y]) * m_t)
                    for l in data.HN.set_head[i]:
                        model.addConstr(var.HN.T_R_node[i, t, s, y] - var.HN.T_R_i[l, t, s, y] <= (1 - var.HN.b_ji[l, y]) * m_t)
                        model.addConstr(var.HN.T_R_node[i, t, s, y] - var.HN.T_R_i[l, t, s, y] >= -(1 - var.HN.b_ji[l, y]) * m_t)

                    # 节点温度混合方程
                    sum_head = gp.quicksum(var.HN.FR_bij_TRi[l, t, s, y] for l in data.HN.set_head[i])
                    sum_tail = gp.quicksum(var.HN.FR_bji_TRj[l, t, s, y] for l in data.HN.set_tail[i])
                    model.addConstr(var.HN.FRin_TRn[i, t, s, y] - var.HN.F_TRn[i, t, s, y] == -var.HN.F_TRex[i, t, s, y] + sum_head - sum_tail)

    # =========================================================================
    # 7. 公共有向路径追踪与多商品流重构
    # =========================================================================
    paths = frozenset(tuple(item) for item in data.HN.path_index)
    load_nodes = list(data.HN.load_node_idx)
    station_nodes = list(data.HN.station_node_idx)
    load_node_to_idx = {int(node): idx for idx, node in enumerate(load_nodes)}
    valid_loads_by_pipe_station_year = {}
    for pipe, load, station, year in paths:
        valid_loads_by_pipe_station_year.setdefault((pipe, station, year), []).append(load)

    def path_present(pipe, load, station, year):
        return (pipe, load, station, year) in paths

    # 路径方向必须与管道方向一致，并且只能服务已分配的负荷-热源组合。
    for pipe, load, station, year in paths:
        x_ij = var.HN.x_path_ij[pipe, load, station, year]
        x_ji = var.HN.x_path_ji[pipe, load, station, year]
        alpha = var.HN.source_zone_affiliation[load_nodes[load], station, year]
        model.addConstr(x_ij <= var.HN.b_ij[pipe, year], name=f"HN_PathIJ_{pipe}_{load}_{station}_{year}")
        model.addConstr(x_ji <= var.HN.b_ji[pipe, year], name=f"HN_PathJI_{pipe}_{load}_{station}_{year}")
        model.addConstr(x_ij + x_ji <= alpha, name=f"HN_PathAlpha_{pipe}_{load}_{station}_{year}")

    # 每个负荷-热源组合的路径流守恒：从热源节点到负荷节点。
    for load in range(data.HN.N_load):
        load_node = load_nodes[load]
        for station in range(data.HN.N_station):
            source_node = station_nodes[station]
            for year in range(data.year):
                alpha = var.HN.source_zone_affiliation[load_node, station, year]
                for node in range(data.HN.N_node):
                    outgoing = gp.quicksum(
                        var.HN.x_path_ij[pipe, load, station, year]
                        for pipe in data.HN.set_head[node]
                        if path_present(pipe, load, station, year)
                    ) + gp.quicksum(
                        var.HN.x_path_ji[pipe, load, station, year]
                        for pipe in data.HN.set_tail[node]
                        if path_present(pipe, load, station, year)
                    )
                    incoming = gp.quicksum(
                        var.HN.x_path_ij[pipe, load, station, year]
                        for pipe in data.HN.set_tail[node]
                        if path_present(pipe, load, station, year)
                    ) + gp.quicksum(
                        var.HN.x_path_ji[pipe, load, station, year]
                        for pipe in data.HN.set_head[node]
                        if path_present(pipe, load, station, year)
                    )
                    rhs = alpha * ((node == source_node) - (node == load_node))
                    model.addConstr(outgoing - incoming == rhs, name=f"HN_PathBalance_{node}_{load}_{station}_{year}")

    # 路径流量直接重构热源流量和管道方向流量。
    for station in range(data.HN.N_station):
        for year in range(data.year):
            model.addConstr(
                var.HN.f_station[station, year] == gp.quicksum(
                    float(data.HN.fixed_load_flow[load, year])
                    * var.HN.source_zone_affiliation[load_nodes[load], station, year]
                    for load in range(data.HN.N_load)
                ), name=f"HN_PathStationFlow_{station}_{year}"
            )

    for pipe in range(data.HN.N_pipe):
        for station in range(data.HN.N_station):
            for year in range(data.year):
                ij = gp.quicksum(
                    float(data.HN.fixed_load_flow[load, year])
                    * var.HN.x_path_ij[pipe, load, station, year]
                    for load in valid_loads_by_pipe_station_year.get((pipe, station, year), ())
                )
                ji = gp.quicksum(
                    float(data.HN.fixed_load_flow[load, year])
                    * var.HN.x_path_ji[pipe, load, station, year]
                    for load in valid_loads_by_pipe_station_year.get((pipe, station, year), ())
                )
                model.addConstr(var.HN.f_S[pipe, station, year] == ij + ji, name=f"HN_PathPipeS_{pipe}_{station}_{year}")
                model.addConstr(var.HN.f_R[pipe, station, year] == ij + ji, name=f"HN_PathPipeR_{pipe}_{station}_{year}")
                model.addConstr(var.HN.fS_bij[pipe, station, year] == ij, name=f"HN_PathSij_{pipe}_{station}_{year}")
                model.addConstr(var.HN.fS_bji[pipe, station, year] == -ji, name=f"HN_PathSji_{pipe}_{station}_{year}")
                model.addConstr(var.HN.fR_bij[pipe, station, year] == ij, name=f"HN_PathRij_{pipe}_{station}_{year}")
                model.addConstr(var.HN.fR_bji[pipe, station, year] == -ji, name=f"HN_PathRji_{pipe}_{station}_{year}")

    # =========================================================================
    # 8. 热网多商品流网络 (流量平衡与管道限制)
    # =========================================================================
    for i in range(data.HN.N_pipe):
        for k in range(data.HN.N_station):
            for y in range(data.year):
                model.addConstr(var.HN.f_S[i, k, y] == var.HN.f_R[i, k, y])
                model.addConstr(var.HN.f_S[i, k, y] >= 0)
                model.addConstr(var.HN.f_R[i, k, y] >= 0)
                model.addConstr(var.HN.f_S[i, k, y] <= var.HN.y_pipe[i, y] * m_f)
                model.addConstr(var.HN.f_R[i, k, y] <= var.HN.y_pipe[i, y] * m_f)



    for y in range(data.year):
        load_nodes = set(data.HN.load_node_idx)
        for node in range(data.HN.N_node):
            if node not in load_nodes:
                for station in range(data.HN.N_station):
                    model.addConstr(
                        var.HN.source_zone_affiliation[node, station, y] == 0
                    )

        # 供热站流量约束
        for i in range(data.HN.N_station):
            model.addConstr(var.HN.f_station[i, y] >= 0)
            model.addConstr(var.HN.f_station[i, y] <= var.HN.y_station[i, y] * float(data.HN.f_station[i]))

            # A built station can be unused.  The source-flow state is the OR
            # of its assigned active load affiliations, not y_station itself.
            station_affiliations = [
                var.HN.source_zone_affiliation[data.HN.load_node_idx[j], i, y]
                for j in range(data.HN.N_load)
            ]
            for load, affiliation in enumerate(station_affiliations):
                model.addConstr(
                    var.HN.source_flow_active[i, y] >= affiliation,
                    name=f"HN_SourceFlowActiveLB_{i}_{load}_{y}",
                )
            if station_affiliations:
                model.addConstr(
                    var.HN.source_flow_active[i, y]
                    <= gp.quicksum(station_affiliations),
                    name=f"HN_SourceFlowActiveUB_{i}_{y}",
                )
            else:
                model.addConstr(var.HN.source_flow_active[i, y] == 0)

        # 负荷节点流量匹配
        for j in range(data.HN.N_load):
            node = data.HN.load_node_idx[j]
            model.addConstr(
                gp.quicksum(
                    var.HN.source_zone_affiliation[node, k, y]
                    for k in range(data.HN.N_station)
                )
                == float(data.HN.load_state[j, y])
            )
            for k in range(data.HN.N_station):
                model.addConstr(
                    var.HN.source_zone_affiliation[node, k, y]
                    <= var.HN.y_station[k, y]
                )
                model.addConstr(
                    var.HN.f_load[j, k, y]
                    == float(data.HN.f_load[j])
                    * var.HN.source_zone_affiliation[node, k, y]
                )

        for i in range(data.HN.N_node):
            node_id = data.HN.node[i]
            for k in range(data.HN.N_station):
                if node_id in data.HN.load:
                    j = list(data.HN.load).index(node_id)
                    model.addConstr(var.HN.f_node[i, k, y] == -var.HN.f_load[j, k, y])
                elif node_id in data.HN.station:
                    j = list(data.HN.station).index(node_id)
                    if k == j:
                        model.addConstr(var.HN.f_node[i, k, y] == var.HN.f_station[k, y])
                    else:
                        model.addConstr(var.HN.f_node[i, k, y] == 0)
                else:
                    model.addConstr(var.HN.f_node[i, k, y] == 0)

                f_S_in_val = gp.quicksum(var.HN.fS_bij[l, k, y] for l in data.HN.set_tail[i]) - gp.quicksum(var.HN.fS_bji[l, k, y] for l in data.HN.set_head[i])
                f_S_out_val = gp.quicksum(var.HN.fS_bij[l, k, y] for l in data.HN.set_head[i]) - gp.quicksum(var.HN.fS_bji[l, k, y] for l in data.HN.set_tail[i])
                model.addConstr(var.HN.f_S_in[i, k, y] == f_S_in_val)
                model.addConstr(var.HN.f_S_out[i, k, y] == f_S_out_val)
                model.addConstr(var.HN.f_S_in[i, k, y] + var.HN.f_node[i, k, y] == var.HN.f_S_out[i, k, y])

                f_R_in_val = gp.quicksum(var.HN.fR_bij[l, k, y] for l in data.HN.set_head[i]) - gp.quicksum(var.HN.fR_bji[l, k, y] for l in data.HN.set_tail[i])
                f_R_out_val = gp.quicksum(var.HN.fR_bij[l, k, y] for l in data.HN.set_tail[i]) - gp.quicksum(var.HN.fR_bji[l, k, y] for l in data.HN.set_head[i])
                model.addConstr(var.HN.f_R_in[i, k, y] == f_R_in_val)
                model.addConstr(var.HN.f_R_out[i, k, y] == f_R_out_val)
                model.addConstr(var.HN.f_R_in[i, k, y] - var.HN.f_node[i, k, y] == var.HN.f_R_out[i, k, y])

        # 总流聚合计算
        model.addConstrs((var.HN.F_node[i, y] == gp.quicksum(var.HN.f_node[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_node)), name="HN_F_node")
        model.addConstrs((var.HN.F_S_in[i, y] == gp.quicksum(var.HN.f_S_in[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_node)), name="HN_F_S_in")
        model.addConstrs((var.HN.F_S_out[i, y] == gp.quicksum(var.HN.f_S_out[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_node)), name="HN_F_S_out")
        model.addConstrs((var.HN.F_R_in[i, y] == gp.quicksum(var.HN.f_R_in[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_node)), name="HN_F_R_in")
        model.addConstrs((var.HN.F_R_out[i, y] == gp.quicksum(var.HN.f_R_out[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_node)), name="HN_F_R_out")

        for i in range(data.HN.N_node):

            node_id = data.HN.node[i]
            if node_id in data.HN.station:
                station = list(data.HN.station).index(node_id)
                positive_lower = _source_flow_lower(data, station, y)
                source_upper = _source_flow_upper(data, station, y)
                source_upper = max(source_upper, positive_lower)
                model.addConstr(
                    var.HN.F_node[i, y]
                    >= positive_lower * var.HN.source_flow_active[station, y],
                    name=f"HN_SourceFNodePositiveLower_{i}_{y}",
                )
                model.addConstr(
                    var.HN.F_node[i, y]
                    <= source_upper * var.HN.source_flow_active[station, y],
                    name=f"HN_SourceFNodeActiveUpper_{i}_{y}",
                )

            # Endogenous inlet state: zero return inlet is represented by the
            # state value 0; a positive inlet requires state value 1.  It is
            # additionally gated by the node activation binary.
            model.addConstr(
                var.HN.inlet_flow_active[i, y] <= var.HN.y_node[i, y],
                name=f"HN_InletFlowActiveNode_{i}_{y}",
            )
            positive_lower = _positive_flow_lower(data, y)
            flow_upper = _annual_flow_upper(data, y)
            flow_upper = max(flow_upper, positive_lower)
            model.addConstr(
                var.HN.F_R_in[i, y]
                >= positive_lower * var.HN.inlet_flow_active[i, y],
                name=f"HN_FRInPositiveLower_{i}_{y}",
            )
            model.addConstr(
                var.HN.F_R_in[i, y]
                <= flow_upper * var.HN.inlet_flow_active[i, y],
                name=f"HN_FRInActiveUpper_{i}_{y}",
            )

        model.addConstrs((var.HN.F_S[i, y] == gp.quicksum(var.HN.f_S[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_pipe)), name="HN_F_S")
        model.addConstrs((var.HN.F_R[i, y] == gp.quicksum(var.HN.f_R[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_pipe)), name="HN_F_R")
        model.addConstrs((var.HN.FS_bij[i, y] == gp.quicksum(var.HN.fS_bij[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_pipe)), name="HN_FS_bij")
        model.addConstrs((var.HN.FS_bji[i, y] == gp.quicksum(var.HN.fS_bji[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_pipe)), name="HN_FS_bji")
        model.addConstrs((var.HN.FR_bij[i, y] == gp.quicksum(var.HN.fR_bij[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_pipe)), name="HN_FR_bij")
        model.addConstrs((var.HN.FR_bji[i, y] == gp.quicksum(var.HN.fR_bji[i, k, y] for k in range(data.HN.N_station)) for i in range(data.HN.N_pipe)), name="HN_FR_bji")

        for i in range(data.HN.N_pipe):

            # 管网实际流量上限
            model.addConstr(var.HN.F_pipe[i, y] == var.HN.N_pipe[i, y] * float(data.pipe.f_max))
            model.addConstr(var.HN.F_S[i, y] <= var.HN.F_pipe[i, y])
            model.addConstr(var.HN.F_S[i, y] >= -var.HN.F_pipe[i, y])
            model.addConstr(var.HN.F_R[i, y] <= var.HN.F_pipe[i, y])
            model.addConstr(var.HN.F_R[i, y] >= -var.HN.F_pipe[i, y])

            # Direction/activity-linked signed return-flow bounds.  An
            # installed pipe may be unused, so b_ij/b_ji only gate the
            # activity states and do not themselves force positive flow.
            model.addConstr(
                var.HN.pipe_flow_active_ij[i, y] <= var.HN.b_ij[i, y],
                name=f"HN_PipeFlowActiveDirIJ_{i}_{y}",
            )
            model.addConstr(
                var.HN.pipe_flow_active_ji[i, y] <= var.HN.b_ji[i, y],
                name=f"HN_PipeFlowActiveDirJI_{i}_{y}",
            )
            positive_lower = _positive_flow_lower(data, y)
            model.addConstr(
                var.HN.FR_bij[i, y]
                >= positive_lower * var.HN.pipe_flow_active_ij[i, y],
                name=f"HN_FRBijPositiveLower_{i}_{y}",
            )
            model.addConstr(
                var.HN.FR_bij[i, y]
                <= float(data.HN.f_pipe_max) * var.HN.pipe_flow_active_ij[i, y],
                name=f"HN_FRBijActiveUpper_{i}_{y}",
            )
            model.addConstr(
                var.HN.FR_bji[i, y]
                <= -positive_lower * var.HN.pipe_flow_active_ji[i, y],
                name=f"HN_FRBjiNegativeUpper_{i}_{y}",
            )
            model.addConstr(
                var.HN.FR_bji[i, y]
                >= -float(data.HN.f_pipe_max) * var.HN.pipe_flow_active_ji[i, y],
                name=f"HN_FRBjiActiveLower_{i}_{y}",
            )

    # =========================================================================
    # 8. McCormick Envelopes 双线性包络松弛 (流量 × 温度)
    # =========================================================================
    if not include_relaxation:
        data.HN.mccormick_envelopes_enabled = False
        data.HN.piecewise_partitions_enabled = False
        return

    def _removed_sbt_feature(*args, **kwargs):
        raise RuntimeError(
            "SBT/McCormick piecewise relaxation support was removed; "
            "use main.py with physical_exact linearization."
        )

    def term_key(index, y=None, group=None):
        return (str(group), int(index), None if y is None else int(y))

    def validate_partition(partition):
        return False

    add_shared_flow_bivariate_mccormick = _removed_sbt_feature
    add_shared_flow_mccormick = _removed_sbt_feature
    create_shared_flow_partition = _removed_sbt_feature
    effective_temperature_breakpoints = _removed_sbt_feature

    data.HN.mccormick_envelopes_enabled = True
    data.HN.piecewise_partitions_enabled = bool(
        int(data.HN.T_pw) > 1 or int(data.HN.F_pw) > 1
    )

    M_FT = float(data.HN.M_FT)
    T_pw = int(data.HN.T_pw)
    F_pw = int(data.HN.F_pw)
    total_pw = T_pw * F_pw

    def add_mccormick(w, x, z, x_min, x_max, z_min, z_max, active):
        # Gate both the flow and its relaxed product with the same activity
        # state. An installed but unused component keeps its temperature
        # variable, while the physical product is exactly zero.
        model.addConstr(w >= z_min * x + x_min * z - z_min * x_min - (1 - active) * M_FT)
        model.addConstr(w >= z_max * x + x_max * z - z_max * x_max - (1 - active) * M_FT)
        model.addConstr(w <= z_min * x + x_max * z - z_min * x_max + (1 - active) * M_FT)
        model.addConstr(w <= z_max * x + x_min * z - z_max * x_min + (1 - active) * M_FT)
        model.addConstr(w <= M_FT * active)
        model.addConstr(w >= -M_FT * active)

    if T_pw == 1 and F_pw == 1:
        sparse_partitions = getattr(data.HN, "ra_partitions", {}) or {}
        sparse_binary_count = 0
        sparse_auxiliary_variable_count = 0
        sparse_added_constraint_count = 0
        invalid_sparse_partition_count = 0
        shared_flow_cache = {}
        zero_flow_branch_count = 0
        segment_temperature_stats = {
            "evaluated": 0,
            "contracted": 0,
            "original_width_sum": 0.0,
            "conditional_width_sum": 0.0,
            "conditional_width_min": None,
            "conditional_width_max": None,
        }

        def shared_flow_partition(group, index, year, flow, active, partition):
            nonlocal sparse_binary_count, sparse_added_constraint_count, zero_flow_branch_count
            key = term_key(index, y=year, group=group)
            context = shared_flow_cache.get(key)
            if context is None:
                context = create_shared_flow_partition(
                    model,
                    flow,
                    active,
                    partition,
                    f"{group}_{index}_{year}",
                )
                shared_flow_cache[key] = context
                sparse_binary_count += int(context["binary_count"])
                sparse_added_constraint_count += 2 + 2 * int(context["binary_count"])
                zero_flow_branch_count += int(context.get("allow_zero", False))
            return context

        def add_shared_product(
            product,
            temperature,
            temperature_bounds,
            context,
            partition,
            temperature_kind,
            group,
            index,
            time,
            scene,
            year,
            family,
            name,
            temperature_active,
        ):
            nonlocal sparse_binary_count, sparse_auxiliary_variable_count, sparse_added_constraint_count
            temperature_points = effective_temperature_breakpoints(
                partition,
                temperature_kind,
                temperature_bounds,
            )
            if temperature_points is None:
                conditional_bounds = _conditional_segment_temperature_bounds(
                    data,
                    group,
                    index,
                    time,
                    scene,
                    year,
                    family,
                    temperature_bounds,
                    partition,
                )
                original_width = max(
                    0.0, float(temperature_bounds[1]) - float(temperature_bounds[0])
                )
                for lower, upper in conditional_bounds:
                    width = max(0.0, float(upper) - float(lower))
                    segment_temperature_stats["evaluated"] += 1
                    segment_temperature_stats["contracted"] += int(
                        width < original_width - 1e-10
                    )
                    segment_temperature_stats["original_width_sum"] += original_width
                    segment_temperature_stats["conditional_width_sum"] += width
                    current_min = segment_temperature_stats["conditional_width_min"]
                    current_max = segment_temperature_stats["conditional_width_max"]
                    segment_temperature_stats["conditional_width_min"] = (
                        width if current_min is None else min(current_min, width)
                    )
                    segment_temperature_stats["conditional_width_max"] = (
                        width if current_max is None else max(current_max, width)
                    )
                variable_count, constraint_count = add_shared_flow_mccormick(
                    model,
                    product,
                    temperature,
                    temperature_bounds,
                    context,
                    name,
                    temperature_bounds_by_segment=conditional_bounds,
                    temperature_active=temperature_active,
                )
                binary_count = 0
            else:
                variable_count, constraint_count, binary_count = add_shared_flow_bivariate_mccormick(
                    model,
                    product,
                    temperature,
                    temperature_points,
                    context,
                    name,
                    temperature_active=temperature_active,
                )
            sparse_binary_count += int(binary_count)
            sparse_auxiliary_variable_count += int(variable_count)
            sparse_added_constraint_count += int(constraint_count)
        for y in range(data.year):
            for s in range(data.scene.N):
                for t in range(data.period):
                    for i in range(data.HN.N_pipe):
                        for group, product, flow, temperature, flow_bounds, local_temp_bounds, suffix in (
                            (
                                "FR_BIJ", var.HN.FR_bij_TRi[i, t, s, y], var.HN.FR_bij[i, y],
                                var.HN.T_R_i[i, t, s, y],
                                (data.HN.FR_bij_min[i, y], data.HN.FR_bij_max[i, y]),
                                (data.HN.TRi_min_all[i, t, s, y], data.HN.TRi_max_all[i, t, s, y]),
                                "FR_bij_TRi",
                            ),
                            (
                                "FR_BJI", var.HN.FR_bji_TRj[i, t, s, y], var.HN.FR_bji[i, y],
                                var.HN.T_R_j[i, t, s, y],
                                (data.HN.FR_bji_min[i, y], data.HN.FR_bji_max[i, y]),
                                (data.HN.TRj_min_all[i, t, s, y], data.HN.TRj_max_all[i, t, s, y]),
                                "FR_bji_TRj",
                            ),
                        ):
                            sparse_partition = sparse_partitions.get(term_key(i, y=y, group=group))
                            flow_lower, flow_upper = map(float, flow_bounds)
                            if sparse_partition is not None and (
                                not validate_partition(sparse_partition)
                                or flow_upper - flow_lower <= 1e-10
                            ):
                                sparse_partition = None
                                invalid_sparse_partition_count += 1
                            if sparse_partition is None:
                                pipe_active = (
                                    var.HN.pipe_flow_active_ij[i, y]
                                    if group == "FR_BIJ"
                                    else var.HN.pipe_flow_active_ji[i, y]
                                )
                                add_mccormick(
                                    product, flow, temperature,
                                    flow_lower, flow_upper,
                                    float(local_temp_bounds[0]), float(local_temp_bounds[1]),
                                    pipe_active,
                                )
                            else:
                                pipe_active = (
                                    var.HN.pipe_flow_active_ij[i, y]
                                    if group == "FR_BIJ"
                                    else var.HN.pipe_flow_active_ji[i, y]
                                )
                                context = shared_flow_partition(
                                    group, i, y, flow, pipe_active, sparse_partition
                                )
                                add_shared_product(
                                    product,
                                    temperature,
                                    (float(local_temp_bounds[0]), float(local_temp_bounds[1])),
                                    context,
                                    sparse_partition,
                                    "return",
                                    group, i, t, s, y, suffix,
                                    f"{suffix}_{i}_{t}_{s}_{y}",
                                     var.HN.y_pipe[i, y],
                                )

                        model.addConstr(var.HN.FR_bij_TRi[i, t, s, y] <= var.HN.pipe_flow_active_ij[i, y] * M_FT)
                        model.addConstr(var.HN.FR_bij_TRi[i, t, s, y] >= 0)
                        model.addConstr(var.HN.FR_bji_TRj[i, t, s, y] <= 0)
                        model.addConstr(var.HN.FR_bji_TRj[i, t, s, y] >= -var.HN.pipe_flow_active_ji[i, y] * M_FT)

                    for i in range(data.HN.N_node):
                        active = var.HN.y_node[i, y]
                        node_id = data.HN.node[i]
                        flow_active = active
                        if node_id in data.HN.load:
                            load = list(data.HN.load).index(node_id)
                            fixed_node_flow = (
                                -float(data.HN.f_load[load])
                                * float(data.HN.load_state[load, y])
                            )
                        elif node_id in data.HN.station:
                            fixed_node_flow = None
                            flow_active = var.HN.source_flow_active[
                                list(data.HN.station).index(node_id), y
                            ]
                        else:
                            fixed_node_flow = 0.0
                        sparse_partition = sparse_partitions.get(term_key(i, y=y, group="F_NODE"))
                        if sparse_partition is not None and (
                            not validate_partition(sparse_partition)
                            or float(data.HN.F_node_max[i, y]) - float(data.HN.F_node_min[i, y]) <= 1e-10
                        ):
                            sparse_partition = None
                            invalid_sparse_partition_count += 1
                        if fixed_node_flow is not None:
                            model.addConstr(
                                var.HN.F_TSex[i, t, s, y]
                                == fixed_node_flow * var.HN.T_S_ex[i, t, s, y],
                                name=f"HN_ExactFixed_FTSex_{i}_{t}_{s}_{y}",
                            )
                            model.addConstr(
                                var.HN.F_TRex[i, t, s, y]
                                == fixed_node_flow * var.HN.T_R_ex[i, t, s, y],
                                name=f"HN_ExactFixed_FTRex_{i}_{t}_{s}_{y}",
                            )
                            model.addConstr(
                                var.HN.F_TRn[i, t, s, y]
                                == fixed_node_flow * var.HN.T_R_node[i, t, s, y],
                                name=f"HN_ExactFixed_FTRn_{i}_{t}_{s}_{y}",
                            )
                        elif sparse_partition is None:
                            add_mccormick(
                                var.HN.F_TSex[i, t, s, y],
                                var.HN.F_node[i, y],
                                var.HN.T_S_ex[i, t, s, y],
                                float(data.HN.F_node_min[i, y]),
                                float(data.HN.F_node_max[i, y]),
                                float(data.HN.TSex_min_all[i, t, s, y]),
                                float(data.HN.TSex_max_all[i, t, s, y]),
                                flow_active,
                            )
                        else:
                            context = shared_flow_partition(
                                "F_NODE", i, y, var.HN.F_node[i, y],
                                flow_active, sparse_partition
                            )
                            add_shared_product(
                                var.HN.F_TSex[i, t, s, y],
                                var.HN.T_S_ex[i, t, s, y],
                                (
                                    float(data.HN.TSex_min_all[i, t, s, y]),
                                    float(data.HN.TSex_max_all[i, t, s, y]),
                                ),
                                context,
                                sparse_partition,
                                "supply",
                                "F_NODE", i, t, s, y, "F_TSex",
                                 f"F_TSex_{i}_{t}_{s}_{y}",
                                 active,
                            )
                        if fixed_node_flow is not None:
                            pass
                        elif sparse_partition is None:
                            add_mccormick(
                                var.HN.F_TRex[i, t, s, y],
                                var.HN.F_node[i, y],
                                var.HN.T_R_ex[i, t, s, y],
                                float(data.HN.F_node_min[i, y]),
                                float(data.HN.F_node_max[i, y]),
                                float(data.HN.TRex_min_all[i, t, s, y]),
                                float(data.HN.TRex_max_all[i, t, s, y]),
                                flow_active,
                            )
                            add_mccormick(
                                var.HN.F_TRn[i, t, s, y],
                                var.HN.F_node[i, y],
                                var.HN.T_R_node[i, t, s, y],
                                float(data.HN.F_node_min[i, y]),
                                float(data.HN.F_node_max[i, y]),
                                float(data.HN.TRn_min_all[i, t, s, y]),
                                float(data.HN.TRn_max_all[i, t, s, y]),
                                flow_active,
                            )
                        else:
                            context = shared_flow_partition(
                                "F_NODE", i, y, var.HN.F_node[i, y], flow_active, sparse_partition
                            )
                            add_shared_product(
                                var.HN.F_TRex[i, t, s, y],
                                var.HN.T_R_ex[i, t, s, y],
                                (
                                    float(data.HN.TRex_min_all[i, t, s, y]),
                                    float(data.HN.TRex_max_all[i, t, s, y]),
                                ),
                                context,
                                sparse_partition,
                                "return",
                                "F_NODE", i, t, s, y, "F_TRex",
                                f"F_TRex_{i}_{t}_{s}_{y}",
                                active,
                            )
                            add_shared_product(
                                var.HN.F_TRn[i, t, s, y],
                                var.HN.T_R_node[i, t, s, y],
                                (
                                    float(data.HN.TRn_min_all[i, t, s, y]),
                                    float(data.HN.TRn_max_all[i, t, s, y]),
                                ),
                                context,
                                sparse_partition,
                                "return",
                                "F_NODE", i, t, s, y, "F_TRn",
                                f"F_TRn_{i}_{t}_{s}_{y}",
                                active,
                            )
                        inlet_partition = sparse_partitions.get(term_key(i, y=y, group="FR_IN"))
                        inlet_lower = float(data.HN.FRin_min[i, y])
                        inlet_upper = float(data.HN.FRin_max[i, y])
                        if inlet_partition is not None and (
                            not validate_partition(inlet_partition)
                            or inlet_upper - inlet_lower <= 1e-10
                        ):
                            inlet_partition = None
                            invalid_sparse_partition_count += 1
                        if inlet_partition is None:
                            add_mccormick(
                                var.HN.FRin_TRn[i, t, s, y],
                                var.HN.F_R_in[i, y],
                                var.HN.T_R_node[i, t, s, y],
                                inlet_lower,
                                inlet_upper,
                                float(data.HN.TRn_min_all[i, t, s, y]),
                                float(data.HN.TRn_max_all[i, t, s, y]),
                                var.HN.inlet_flow_active[i, y],
                            )
                        else:
                            context = shared_flow_partition(
                                "FR_IN", i, y, var.HN.F_R_in[i, y],
                                var.HN.inlet_flow_active[i, y], inlet_partition
                            )
                            add_shared_product(
                                var.HN.FRin_TRn[i, t, s, y],
                                var.HN.T_R_node[i, t, s, y],
                                (
                                    float(data.HN.TRn_min_all[i, t, s, y]),
                                    float(data.HN.TRn_max_all[i, t, s, y]),
                                ),
                                context,
                                inlet_partition,
                                "return",
                                "FR_IN", i, t, s, y, "FRin_TRn",
                                f"FRin_TRn_{i}_{t}_{s}_{y}",
                                active,
                            )

                        if node_id in data.HN.load:
                            j = list(data.HN.load).index(node_id)
                            model.addConstr(var.HN.F_TSex[i, t, s, y] <= 0)
                            model.addConstr(var.HN.F_TSex[i, t, s, y] >= -float(data.HN.load_state[j, y]) * M_FT)
                            model.addConstr(var.HN.F_TRex[i, t, s, y] <= 0)
                            model.addConstr(var.HN.F_TRex[i, t, s, y] >= -float(data.HN.load_state[j, y]) * M_FT)
                            model.addConstr(var.HN.F_TRn[i, t, s, y] <= 0)
                            model.addConstr(var.HN.F_TRn[i, t, s, y] >= -float(data.HN.load_state[j, y]) * M_FT)
                            model.addConstr(var.HN.FRin_TRn[i, t, s, y] <= float(data.HN.load_state[j, y]) * M_FT)
                            model.addConstr(var.HN.FRin_TRn[i, t, s, y] >= 0)
                        elif node_id in data.HN.station:
                            j = list(data.HN.station).index(node_id)
                            model.addConstr(var.HN.F_TSex[i, t, s, y] <= var.HN.y_station[j, y] * M_FT)
                            model.addConstr(var.HN.F_TSex[i, t, s, y] >= 0)
                            model.addConstr(var.HN.F_TRex[i, t, s, y] <= var.HN.y_station[j, y] * M_FT)
                            model.addConstr(var.HN.F_TRex[i, t, s, y] >= 0)
                            model.addConstr(var.HN.F_TRn[i, t, s, y] <= var.HN.y_station[j, y] * M_FT)
                            model.addConstr(var.HN.F_TRn[i, t, s, y] >= 0)
                            model.addConstr(var.HN.FRin_TRn[i, t, s, y] <= var.HN.y_station[j, y] * M_FT)
                            model.addConstr(var.HN.FRin_TRn[i, t, s, y] >= 0)
                        else:
                            model.addConstr(var.HN.F_TSex[i, t, s, y] == 0)
                            model.addConstr(var.HN.F_TRex[i, t, s, y] == 0)
                            model.addConstr(var.HN.F_TRn[i, t, s, y] == 0)
        data.HN.ra_added_binary_count = sparse_binary_count
        data.HN.ra_added_auxiliary_variable_count = sparse_auxiliary_variable_count
        data.HN.ra_added_constraint_count = sparse_added_constraint_count
        data.HN.ra_zero_flow_branch_count = zero_flow_branch_count
        data.HN.ra_segment_temperature_evaluated_count = int(
            segment_temperature_stats["evaluated"]
        )

        data.HN.ra_segment_temperature_contracted_count = int(
            segment_temperature_stats["contracted"]
        )
        evaluated = max(1, int(segment_temperature_stats["evaluated"]))
        original_sum = float(segment_temperature_stats["original_width_sum"])
        conditional_sum = float(segment_temperature_stats["conditional_width_sum"])
        data.HN.ra_segment_temperature_original_width_mean = original_sum / evaluated
        data.HN.ra_segment_temperature_conditional_width_mean = conditional_sum / evaluated
        data.HN.ra_segment_temperature_width_reduction = (
            0.0 if original_sum <= 1e-12 else 1.0 - conditional_sum / original_sum
        )
        data.HN.ra_segment_temperature_width_min = (
            segment_temperature_stats["conditional_width_min"]
        )
        data.HN.ra_segment_temperature_width_max = (
            segment_temperature_stats["conditional_width_max"]
        )
        data.HN.ra_shared_flow_partition_count = len(shared_flow_cache)
        data.HN.ra_invalid_partition_count = invalid_sparse_partition_count
        data.HN.source_zone_temperature_link_enabled = True
        data.HN.exact_fixed_node_products_enabled = True
        data.HN.exact_fixed_node_product_count = int(
            3
            * (data.HN.N_node - data.HN.N_station)
            * data.period
            * data.scene.N
            * data.year
        )
        data.HN.temperature_partition_selector_count = 0
        return

    for y in range(data.year):
        for s in range(data.scene.N):
            for t in range(data.period):

                # =============================================================
                # 8.1 管道 (Pipe) 分段松弛约束
                # =============================================================
                for i in range(data.HN.N_pipe):
                    y_pipe_val = var.HN.y_pipe[i, y]
                    bij_val = var.HN.b_ij[i, y]
                    bji_val = var.HN.b_ji[i, y]

                    # (1) 松弛-缩紧约束：非规划/未接通线路的乘积项强制为0
                    model.addConstr(var.HN.FR_bij_TRi[i, t, s, y] <= bij_val * M_FT)
                    model.addConstr(var.HN.FR_bij_TRi[i, t, s, y] >= -bij_val * M_FT)
                    model.addConstr(var.HN.FR_bji_TRj[i, t, s, y] <= 0)
                    model.addConstr(var.HN.FR_bji_TRj[i, t, s, y] >= -bji_val * M_FT)

                    sum_help_ij = gp.quicksum(var.HN.FR_bij_TRi_help[i, t, s, y, k0] for k0 in range(total_pw))
                    sum_help_ji = gp.quicksum(var.HN.FR_bji_TRj_help[i, t, s, y, k0] for k0 in range(total_pw))

                    model.addConstr(var.HN.FR_bij_TRi[i, t, s, y] <= sum_help_ij + (1 - bij_val) * M_FT)
                    model.addConstr(var.HN.FR_bij_TRi[i, t, s, y] >= sum_help_ij - (1 - bij_val) * M_FT)
                    model.addConstr(var.HN.FR_bji_TRj[i, t, s, y] <= sum_help_ji + (1 - bji_val) * M_FT)
                    model.addConstr(var.HN.FR_bji_TRj[i, t, s, y] >= sum_help_ji - (1 - bji_val) * M_FT)

                    # (2) 变量与分段变量加和约束 & 标志位限制
                    model.addConstr(var.HN.FR_bij[i, y] == gp.quicksum(var.HN.FR_bij_pw[i, t, s, y, k] for k in range(F_pw)))
                    model.addConstr(var.HN.FR_bji[i, y] == gp.quicksum(var.HN.FR_bji_pw[i, t, s, y, k] for k in range(F_pw)))
                    model.addConstr(var.HN.T_R_i[i, t, s, y] == gp.quicksum(var.HN.T_R_i_pw[i, t, s, y, k] for k in range(T_pw)))
                    model.addConstr(var.HN.T_R_j[i, t, s, y] == gp.quicksum(var.HN.T_R_j_pw[i, t, s, y, k] for k in range(T_pw)))

                    model.addConstr(gp.quicksum(var.HN.u_TRi_pw[i, t, s, y, k] for k in range(T_pw)) <= y_pipe_val)
                    model.addConstr(gp.quicksum(var.HN.u_TRj_pw[i, t, s, y, k] for k in range(T_pw)) <= y_pipe_val)
                    model.addConstr(gp.quicksum(var.HN.u_FRij_pw[i, t, s, y, k] for k in range(F_pw)) <= y_pipe_val)
                    model.addConstr(gp.quicksum(var.HN.u_FRji_pw[i, t, s, y, k] for k in range(F_pw)) <= y_pipe_val)
                    model.addConstr(gp.quicksum(var.HN.u_FRij_TRi_pw[i, t, s, y, k0] for k0 in range(total_pw)) <= y_pipe_val)
                    model.addConstr(gp.quicksum(var.HN.u_FRji_TRj_pw[i, t, s, y, k0] for k0 in range(total_pw)) <= y_pipe_val)

                    # (3) 温度和流量分段上下限约束
                    for k1 in range(T_pw):
                        TR_min = float(data.HN.T_R_min_pw[k1])
                        TR_max = float(data.HN.T_R_max_pw[k1])
                        model.addConstr(var.HN.T_R_i_pw[i, t, s, y, k1] <= TR_max * var.HN.u_TRi_pw[i, t, s, y, k1])
                        model.addConstr(var.HN.T_R_i_pw[i, t, s, y, k1] >= TR_min * var.HN.u_TRi_pw[i, t, s, y, k1])
                        model.addConstr(var.HN.T_R_j_pw[i, t, s, y, k1] <= TR_max * var.HN.u_TRj_pw[i, t, s, y, k1])
                        model.addConstr(var.HN.T_R_j_pw[i, t, s, y, k1] >= TR_min * var.HN.u_TRj_pw[i, t, s, y, k1])

                    for k2 in range(F_pw):
                        Fij_min = float(data.HN.FR_bij_min_pw[i, k2])
                        Fij_max = float(data.HN.FR_bij_max_pw[i, k2])
                        Fji_min = float(data.HN.FR_bji_min_pw[i, k2])
                        Fji_max = float(data.HN.FR_bji_max_pw[i, k2])
                        model.addConstr(var.HN.FR_bij_pw[i, t, s, y, k2] >= Fij_min * var.HN.u_FRij_pw[i, t, s, y, k2])
                        model.addConstr(var.HN.FR_bij_pw[i, t, s, y, k2] <= Fij_max * var.HN.u_FRij_pw[i, t, s, y, k2])
                        model.addConstr(var.HN.FR_bji_pw[i, t, s, y, k2] >= Fji_min * var.HN.u_FRji_pw[i, t, s, y, k2])
                        model.addConstr(var.HN.FR_bji_pw[i, t, s, y, k2] <= Fji_max * var.HN.u_FRji_pw[i, t, s, y, k2])

                    # (4) 乘积项求和辅助变量约束 (AND逻辑门与大M) 及 McCormick
                    for k1 in range(T_pw):
                        for k2 in range(F_pw):
                            k0 = k1 * F_pw + k2

                            u_Tri = var.HN.u_TRi_pw[i, t, s, y, k1]
                            u_Fij = var.HN.u_FRij_pw[i, t, s, y, k2]
                            u_Trj = var.HN.u_TRj_pw[i, t, s, y, k1]
                            u_Fji = var.HN.u_FRji_pw[i, t, s, y, k2]

                            u_F_TRi = var.HN.u_FRij_TRi_pw[i, t, s, y, k0]
                            u_F_TRj = var.HN.u_FRji_TRj_pw[i, t, s, y, k0]

                            # AND 逻辑门
                            model.addConstr(u_F_TRi <= u_Tri)
                            model.addConstr(u_F_TRi <= u_Fij)
                            model.addConstr(u_F_TRi >= u_Tri + u_Fij - 1)

                            model.addConstr(u_F_TRj <= u_Trj)
                            model.addConstr(u_F_TRj <= u_Fji)
                            model.addConstr(u_F_TRj >= u_Trj + u_Fji - 1)

                            # Helper 大 M 约束
                            model.addConstr(var.HN.FR_bij_TRi_help[i, t, s, y, k0] <= var.HN.FR_bij_TRi_pw[i, t, s, y, k0] + (1 - u_F_TRi) * M_FT)
                            model.addConstr(var.HN.FR_bij_TRi_help[i, t, s, y, k0] >= var.HN.FR_bij_TRi_pw[i, t, s, y, k0] - (1 - u_F_TRi) * M_FT)
                            model.addConstr(var.HN.FR_bij_TRi_help[i, t, s, y, k0] <= u_F_TRi * M_FT)
                            model.addConstr(var.HN.FR_bij_TRi_help[i, t, s, y, k0] >= -u_F_TRi * M_FT)

                            model.addConstr(var.HN.FR_bji_TRj_help[i, t, s, y, k0] <= var.HN.FR_bji_TRj_pw[i, t, s, y, k0] + (1 - u_F_TRj) * M_FT)
                            model.addConstr(var.HN.FR_bji_TRj_help[i, t, s, y, k0] >= var.HN.FR_bji_TRj_pw[i, t, s, y, k0] - (1 - u_F_TRj) * M_FT)
                            model.addConstr(var.HN.FR_bji_TRj_help[i, t, s, y, k0] <= u_F_TRj * M_FT)
                            model.addConstr(var.HN.FR_bji_TRj_help[i, t, s, y, k0] >= -u_F_TRj * M_FT)

                            # McCormick 包络边界
                            TR_min = float(data.HN.T_R_min_pw[k1])
                            TR_max = float(data.HN.T_R_max_pw[k1])
                            Fij_min = float(data.HN.FR_bij_min_pw[i, k2])
                            Fij_max = float(data.HN.FR_bij_max_pw[i, k2])
                            Fji_min = float(data.HN.FR_bji_min_pw[i, k2])
                            Fji_max = float(data.HN.FR_bji_max_pw[i, k2])

                            Fij_pw_val = var.HN.FR_bij_pw[i, t, s, y, k2]
                            Fji_pw_val = var.HN.FR_bji_pw[i, t, s, y, k2]
                            Tri_pw_val = var.HN.T_R_i_pw[i, t, s, y, k1]
                            Trj_pw_val = var.HN.T_R_j_pw[i, t, s, y, k1]

                            val_F_TRi_pw = var.HN.FR_bij_TRi_pw[i, t, s, y, k0]
                            val_F_TRj_pw = var.HN.FR_bji_TRj_pw[i, t, s, y, k0]

                            model.addConstr(val_F_TRi_pw >= TR_min * Fij_pw_val + Fij_min * Tri_pw_val - TR_min * Fij_min - (1 - u_F_TRi) * M_FT)
                            model.addConstr(val_F_TRi_pw >= TR_max * Fij_pw_val + Fij_max * Tri_pw_val - TR_max * Fij_max - (1 - u_F_TRi) * M_FT)
                            model.addConstr(val_F_TRi_pw <= TR_min * Fij_pw_val + Fij_max * Tri_pw_val - TR_min * Fij_max + (1 - u_F_TRi) * M_FT)
                            model.addConstr(val_F_TRi_pw <= TR_max * Fij_pw_val + Fij_min * Tri_pw_val - TR_max * Fij_min + (1 - u_F_TRi) * M_FT)

                            model.addConstr(val_F_TRj_pw >= TR_min * Fji_pw_val + Fji_min * Trj_pw_val - TR_min * Fji_min - (1 - u_F_TRj) * M_FT)
                            model.addConstr(val_F_TRj_pw >= TR_max * Fji_pw_val + Fji_max * Trj_pw_val - TR_max * Fji_max - (1 - u_F_TRj) * M_FT)
                            model.addConstr(val_F_TRj_pw <= TR_min * Fji_pw_val + Fji_max * Trj_pw_val - TR_min * Fji_max + (1 - u_F_TRj) * M_FT)
                            model.addConstr(val_F_TRj_pw <= TR_max * Fji_pw_val + Fji_min * Trj_pw_val - TR_max * Fji_min + (1 - u_F_TRj) * M_FT)

                # =============================================================
                # 8.2 节点 (Node) 分段松弛约束
                # =============================================================
                for i in range(data.HN.N_node):
                    node_id = data.HN.node[i]
                    is_candidate = (node_id in data.HN.station)
                    is_load = (node_id in data.HN.load)

                    if is_candidate or is_load:
                        sum_help_TSex = gp.quicksum(var.HN.F_TSex_help[i, t, s, y, k0] for k0 in range(total_pw))
                        sum_help_TRex = gp.quicksum(var.HN.F_TRex_help[i, t, s, y, k0] for k0 in range(total_pw))
                        sum_help_TRn = gp.quicksum(var.HN.FRin_TRn_help[i, t, s, y, k0] for k0 in range(total_pw))

                        model.addConstr(var.HN.F_TSex[i, t, s, y] == sum_help_TSex)
                        model.addConstr(var.HN.F_TRex[i, t, s, y] == sum_help_TRex)
                        model.addConstr(var.HN.FRin_TRn[i, t, s, y] == sum_help_TRn)

                        for k1 in range(T_pw):
                            TS_min = float(data.HN.T_S_min_pw[k1])
                            TS_max = float(data.HN.T_S_max_pw[k1])
                            TR_min = float(data.HN.T_R_min_pw[k1])
                            TR_max = float(data.HN.T_R_max_pw[k1])

                            model.addConstr(var.HN.T_S_ex_pw[i, t, s, y, k1] <= TS_max * var.HN.u_TSex_pw[i, t, s, y, k1])
                            model.addConstr(var.HN.T_S_ex_pw[i, t, s, y, k1] >= TS_min * var.HN.u_TSex_pw[i, t, s, y, k1])
                            model.addConstr(var.HN.T_R_ex_pw[i, t, s, y, k1] <= TR_max * var.HN.u_TRex_pw[i, t, s, y, k1])
                            model.addConstr(var.HN.T_R_ex_pw[i, t, s, y, k1] >= TR_min * var.HN.u_TRex_pw[i, t, s, y, k1])
                            model.addConstr(var.HN.T_R_node_pw[i, t, s, y, k1] <= TR_max * var.HN.u_TRn_pw[i, t, s, y, k1])
                            model.addConstr(var.HN.T_R_node_pw[i, t, s, y, k1] >= TR_min * var.HN.u_TRn_pw[i, t, s, y, k1])

                        for k2 in range(F_pw):
                            Fn_min = float(data.HN.F_node_min_pw[i, k2])
                            Fn_max = float(data.HN.F_node_max_pw[i, k2])
                            Fin_min = float(data.HN.FRin_min_pw[i, k2])
                            Fin_max = float(data.HN.FRin_max_pw[i, k2])

                            model.addConstr(var.HN.F_node_pw[i, t, s, y, k2] <= Fn_max * var.HN.u_Fn_pw[i, t, s, y, k2])
                            model.addConstr(var.HN.F_node_pw[i, t, s, y, k2] >= Fn_min * var.HN.u_Fn_pw[i, t, s, y, k2])
                            model.addConstr(var.HN.F_R_in_pw[i, t, s, y, k2] <= Fin_max * var.HN.u_FRin_pw[i, t, s, y, k2])
                            model.addConstr(var.HN.F_R_in_pw[i, t, s, y, k2] >= Fin_min * var.HN.u_FRin_pw[i, t, s, y, k2])

                        model.addConstr(var.HN.F_node[i, y] == gp.quicksum(var.HN.F_node_pw[i, t, s, y, k] for k in range(F_pw)))
                        model.addConstr(var.HN.F_R_in[i, y] == gp.quicksum(var.HN.F_R_in_pw[i, t, s, y, k] for k in range(F_pw)))
                        model.addConstr(var.HN.T_S_ex[i, t, s, y] == gp.quicksum(var.HN.T_S_ex_pw[i, t, s, y, k] for k in range(T_pw)))
                        model.addConstr(var.HN.T_R_ex[i, t, s, y] == gp.quicksum(var.HN.T_R_ex_pw[i, t, s, y, k] for k in range(T_pw)))
                        model.addConstr(var.HN.T_R_node[i, t, s, y] == gp.quicksum(var.HN.T_R_node_pw[i, t, s, y, k] for k in range(T_pw)))

                        if is_load:
                            bound = 1
                        else:
                            j = list(data.HN.station).index(node_id)
                            bound = var.HN.y_station[j, y] + var.HN.c[i, y]

                        model.addConstr(gp.quicksum(var.HN.u_TSex_pw[i, t, s, y, k] for k in range(T_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_TRex_pw[i, t, s, y, k] for k in range(T_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_TRn_pw[i, t, s, y, k] for k in range(T_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_Fn_pw[i, t, s, y, k] for k in range(F_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_FRin_pw[i, t, s, y, k] for k in range(F_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_F_TSex_pw[i, t, s, y, k0] for k0 in range(total_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_F_TRex_pw[i, t, s, y, k0] for k0 in range(total_pw)) <= bound)
                        model.addConstr(gp.quicksum(var.HN.u_FRin_TRn_pw[i, t, s, y, k0] for k0 in range(total_pw)) <= bound)

                        for k1 in range(T_pw):
                            for k2 in range(F_pw):
                                k0 = k1 * F_pw + k2

                                u_TSex = var.HN.u_TSex_pw[i, t, s, y, k1]
                                u_TRex = var.HN.u_TRex_pw[i, t, s, y, k1]
                                u_TRn = var.HN.u_TRn_pw[i, t, s, y, k1]
                                u_Fn = var.HN.u_Fn_pw[i, t, s, y, k2]
                                u_Fin = var.HN.u_FRin_pw[i, t, s, y, k2]

                                u_F_TSex = var.HN.u_F_TSex_pw[i, t, s, y, k0]
                                u_F_TRex = var.HN.u_F_TRex_pw[i, t, s, y, k0]
                                u_FRin_TRn = var.HN.u_FRin_TRn_pw[i, t, s, y, k0]

                                model.addConstr(u_F_TSex <= u_TSex)
                                model.addConstr(u_F_TSex <= u_Fn)
                                model.addConstr(u_F_TSex >= u_TSex + u_Fn - 1)
                                model.addConstr(u_F_TRex <= u_TRex)
                                model.addConstr(u_F_TRex <= u_Fn)
                                model.addConstr(u_F_TRex >= u_TRex + u_Fn - 1)
                                model.addConstr(u_FRin_TRn <= u_TRn)
                                model.addConstr(u_FRin_TRn <= u_Fin)
                                model.addConstr(u_FRin_TRn >= u_TRn + u_Fin - 1)

                                model.addConstr(var.HN.F_TSex_help[i, t, s, y, k0] <= var.HN.F_TSex_pw[i, t, s, y, k0] + (1 - u_F_TSex) * M_FT)
                                model.addConstr(var.HN.F_TSex_help[i, t, s, y, k0] >= var.HN.F_TSex_pw[i, t, s, y, k0] - (1 - u_F_TSex) * M_FT)
                                model.addConstr(var.HN.F_TSex_help[i, t, s, y, k0] <= u_F_TSex * M_FT)
                                model.addConstr(var.HN.F_TSex_help[i, t, s, y, k0] >= -u_F_TSex * M_FT)

                                model.addConstr(var.HN.F_TRex_help[i, t, s, y, k0] <= var.HN.F_TRex_pw[i, t, s, y, k0] + (1 - u_F_TRex) * M_FT)
                                model.addConstr(var.HN.F_TRex_help[i, t, s, y, k0] >= var.HN.F_TRex_pw[i, t, s, y, k0] - (1 - u_F_TRex) * M_FT)
                                model.addConstr(var.HN.F_TRex_help[i, t, s, y, k0] <= u_F_TRex * M_FT)
                                model.addConstr(var.HN.F_TRex_help[i, t, s, y, k0] >= -u_F_TRex * M_FT)

                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] <= var.HN.FRin_TRn_pw[i, t, s, y, k0] + (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] >= var.HN.FRin_TRn_pw[i, t, s, y, k0] - (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] <= u_FRin_TRn * M_FT)
                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] >= -u_FRin_TRn * M_FT)

                                TS_min = float(data.HN.T_S_min_pw[k1])
                                TS_max = float(data.HN.T_S_max_pw[k1])
                                TR_min = float(data.HN.T_R_min_pw[k1])
                                TR_max = float(data.HN.T_R_max_pw[k1])
                                Fn_min = float(data.HN.F_node_min_pw[i, k2])
                                Fn_max = float(data.HN.F_node_max_pw[i, k2])
                                Fin_min = float(data.HN.FRin_min_pw[i, k2])
                                Fin_max = float(data.HN.FRin_max_pw[i, k2])

                                Fn_pw_val = var.HN.F_node_pw[i, t, s, y, k2]
                                Fin_pw_val = var.HN.F_R_in_pw[i, t, s, y, k2]
                                TS_pw_val = var.HN.T_S_ex_pw[i, t, s, y, k1]
                                TRex_pw_val = var.HN.T_R_ex_pw[i, t, s, y, k1]
                                TRn_pw_val = var.HN.T_R_node_pw[i, t, s, y, k1]

                                val_F_TSex_pw = var.HN.F_TSex_pw[i, t, s, y, k0]
                                val_F_TRex_pw = var.HN.F_TRex_pw[i, t, s, y, k0]
                                val_FRin_TRn_pw = var.HN.FRin_TRn_pw[i, t, s, y, k0]

                                model.addConstr(val_F_TSex_pw >= TS_min * Fn_pw_val + Fn_min * TS_pw_val - TS_min * Fn_min - (1 - u_F_TSex) * M_FT)
                                model.addConstr(val_F_TSex_pw >= TS_max * Fn_pw_val + Fn_max * TS_pw_val - TS_max * Fn_max - (1 - u_F_TSex) * M_FT)
                                model.addConstr(val_F_TSex_pw <= TS_min * Fn_pw_val + Fn_max * TS_pw_val - TS_min * Fn_max + (1 - u_F_TSex) * M_FT)
                                model.addConstr(val_F_TSex_pw <= TS_max * Fn_pw_val + Fn_min * TS_pw_val - TS_max * Fn_min + (1 - u_F_TSex) * M_FT)

                                model.addConstr(val_F_TRex_pw >= TR_min * Fn_pw_val + Fn_min * TRex_pw_val - TR_min * Fn_min - (1 - u_F_TRex) * M_FT)
                                model.addConstr(val_F_TRex_pw >= TR_max * Fn_pw_val + Fn_max * TRex_pw_val - TR_max * Fn_max - (1 - u_F_TRex) * M_FT)
                                model.addConstr(val_F_TRex_pw <= TR_min * Fn_pw_val + Fn_max * TRex_pw_val - TR_min * Fn_max + (1 - u_F_TRex) * M_FT)
                                model.addConstr(val_F_TRex_pw <= TR_max * Fn_pw_val + Fn_min * TRex_pw_val - TR_max * Fn_min + (1 - u_F_TRex) * M_FT)

                                model.addConstr(val_FRin_TRn_pw >= TR_min * Fin_pw_val + Fin_min * TRn_pw_val - TR_min * Fin_min - (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(val_FRin_TRn_pw >= TR_max * Fin_pw_val + Fin_max * TRn_pw_val - TR_max * Fin_max - (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(val_FRin_TRn_pw <= TR_min * Fin_pw_val + Fin_max * TRn_pw_val - TR_min * Fin_max + (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(val_FRin_TRn_pw <= TR_max * Fin_pw_val + Fin_min * TRn_pw_val - TR_max * Fin_min + (1 - u_FRin_TRn) * M_FT)
                    else:
                        # 其他节点: 仅需松弛一个变量 FRin_TRn
                        model.addConstr(var.HN.F_TSex[i, t, s, y] == 0)
                        model.addConstr(var.HN.F_TRex[i, t, s, y] == 0)

                        c_val = var.HN.c[i, y]
                        model.addConstr(var.HN.FRin_TRn[i, t, s, y] <= c_val * M_FT)
                        model.addConstr(var.HN.FRin_TRn[i, t, s, y] >= -c_val * M_FT)

                        sum_help_in = gp.quicksum(var.HN.FRin_TRn_help[i, t, s, y, k0] for k0 in range(total_pw))
                        model.addConstr(var.HN.FRin_TRn[i, t, s, y] <= sum_help_in + (1 - c_val) * M_FT)
                        model.addConstr(var.HN.FRin_TRn[i, t, s, y] >= sum_help_in - (1 - c_val) * M_FT)

                        model.addConstr(var.HN.F_R_in[i, y] == gp.quicksum(var.HN.F_R_in_pw[i, t, s, y, k] for k in range(F_pw)))
                        model.addConstr(var.HN.T_R_node[i, t, s, y] == gp.quicksum(var.HN.T_R_node_pw[i, t, s, y, k] for k in range(T_pw)))

                        model.addConstr(gp.quicksum(var.HN.u_TRn_pw[i, t, s, y, k] for k in range(T_pw)) <= c_val)
                        model.addConstr(gp.quicksum(var.HN.u_FRin_pw[i, t, s, y, k] for k in range(F_pw)) <= c_val)
                        model.addConstr(gp.quicksum(var.HN.u_FRin_TRn_pw[i, t, s, y, k0] for k0 in range(total_pw)) <= c_val)

                        for k1 in range(T_pw):
                            TR_min = float(data.HN.T_R_min_pw[k1])
                            TR_max = float(data.HN.T_R_max_pw[k1])
                            model.addConstr(var.HN.T_R_node_pw[i, t, s, y, k1] <= TR_max * var.HN.u_TRn_pw[i, t, s, y, k1])
                            model.addConstr(var.HN.T_R_node_pw[i, t, s, y, k1] >= TR_min * var.HN.u_TRn_pw[i, t, s, y, k1])
                        for k2 in range(F_pw):
                            Fin_min = float(data.HN.FRin_min_pw[i, k2])
                            Fin_max = float(data.HN.FRin_max_pw[i, k2])
                            model.addConstr(var.HN.F_R_in_pw[i, t, s, y, k2] <= Fin_max * var.HN.u_FRin_pw[i, t, s, y, k2])
                            model.addConstr(var.HN.F_R_in_pw[i, t, s, y, k2] >= Fin_min * var.HN.u_FRin_pw[i, t, s, y, k2])

                        for k1 in range(T_pw):
                            for k2 in range(F_pw):
                                k0 = k1 * F_pw + k2
                                u_TRn = var.HN.u_TRn_pw[i, t, s, y, k1]
                                u_Fin = var.HN.u_FRin_pw[i, t, s, y, k2]
                                u_FRin_TRn = var.HN.u_FRin_TRn_pw[i, t, s, y, k0]

                                model.addConstr(u_FRin_TRn <= u_TRn)
                                model.addConstr(u_FRin_TRn <= u_Fin)
                                model.addConstr(u_FRin_TRn >= u_TRn + u_Fin - 1)

                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] <= var.HN.FRin_TRn_pw[i, t, s, y, k0] + (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] >= var.HN.FRin_TRn_pw[i, t, s, y, k0] - (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] <= u_FRin_TRn * M_FT)
                                model.addConstr(var.HN.FRin_TRn_help[i, t, s, y, k0] >= -u_FRin_TRn * M_FT)

                                TR_min = float(data.HN.T_R_min_pw[k1])
                                TR_max = float(data.HN.T_R_max_pw[k1])
                                Fin_min = float(data.HN.FRin_min_pw[i, k2])
                                Fin_max = float(data.HN.FRin_max_pw[i, k2])
                                Fin_pw_val = var.HN.F_R_in_pw[i, t, s, y, k2]
                                TRn_pw_val = var.HN.T_R_node_pw[i, t, s, y, k1]
                                val_FRin_TRn_pw = var.HN.FRin_TRn_pw[i, t, s, y, k0]

                                model.addConstr(val_FRin_TRn_pw >= TR_min * Fin_pw_val + Fin_min * TRn_pw_val - TR_min * Fin_min - (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(val_FRin_TRn_pw >= TR_max * Fin_pw_val + Fin_max * TRn_pw_val - TR_max * Fin_max - (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(val_FRin_TRn_pw <= TR_min * Fin_pw_val + Fin_max * TRn_pw_val - TR_min * Fin_max + (1 - u_FRin_TRn) * M_FT)
                                model.addConstr(val_FRin_TRn_pw <= TR_max * Fin_pw_val + Fin_min * TRn_pw_val - TR_max * Fin_min + (1 - u_FRin_TRn) * M_FT)


def Constraints_HN_manuscript_direct_bilinear(model, data, var, scheme=None):
    """Manuscript regulation constraints with exact heat-network products."""
    from solve.Constraints_HN_direct_bilinear import add_direct_bilinear_heat_constraints

    Constraints_HN(model, data, var, scheme, include_relaxation=False)
    data.HN.direct_bilinear_products = True
    data.HN.mccormick_envelopes_enabled = False
    data.HN.piecewise_partitions_enabled = False
    add_direct_bilinear_heat_constraints(model, data, var)
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
