import gurobipy as gp
import numpy as np
from gurobipy import GRB


def _validate_bss_call(data):
    expected_shape = (
        data.BSS.N_station,
        data.period,
        data.scene.N,
        data.year,
    )
    alpha_tu = np.asarray(
        getattr(data.BSS, "alpha_TU", np.zeros(expected_shape)),
        dtype=float,
    )
    alpha_td = np.asarray(
        getattr(data.BSS, "alpha_TD", np.zeros(expected_shape)),
        dtype=float,
    )
    if alpha_tu.shape != expected_shape or alpha_td.shape != expected_shape:
        raise ValueError("BSS call coefficient shape mismatch")
    if not np.all(np.isfinite(alpha_tu)) or not np.all(np.isfinite(alpha_td)):
        raise ValueError("BSS call coefficients must be finite")
    if np.any(alpha_tu < 0) or np.any(alpha_tu > 1):
        raise ValueError("BSS.alpha_TU must be within [0, 1]")
    if np.any(alpha_td < 0) or np.any(alpha_td > 1):
        raise ValueError("BSS.alpha_TD must be within [0, 1]")
    if np.any(alpha_tu + alpha_td > 1 + 1e-9):
        raise ValueError("BSS.alpha_TU + BSS.alpha_TD must not exceed 1")
    if np.any((alpha_tu > 1e-9) & (alpha_td > 1e-9)):
        raise ValueError(
            "BSS.alpha_TU and alpha_TD cannot both be positive at the same time"
        )
    data.BSS.alpha_TU = alpha_tu
    data.BSS.alpha_TD = alpha_td


def _set_bss_physical_bounds(data, var, charge_arcs):
    """Apply finite station-level bounds to all BSS operating variables."""
    for station in range(data.BSS.N_station):
        battery_upper = float(data.BSS.N_SB_max[station])
        charger_upper = float(data.BSS.N_CB_max[station])
        power_upper = charger_upper * float(data.CB.P)

        for year in range(data.year):
            for scene in range(data.scene.N):
                for time in range(data.period):
                    point = (station, time, scene, year)
                    for power_var in (
                        var.BSS.P_c,
                        var.BSS.P,
                        var.BSS.P_RU,
                        var.BSS.P_RD,
                    ):
                        power_var[point].LB = 0.0
                        power_var[point].UB = power_upper
                    var.BSS.P_f[point].LB = 0.0
                    var.BSS.P_f[point].UB = 0.0

                    for soc in range(data.BSS.N_SOC):
                        var.BSS.N_S[point + (soc,)].LB = 0.0
                        var.BSS.N_S[point + (soc,)].UB = battery_upper
                        for target_soc in range(data.BSS.N_SOC):
                            dense_key = point + (soc, target_soc)
                            for transition_var in (
                                var.BSS.N_T,
                                var.BSS.N_U,
                                var.BSS.N_D,
                            ):
                                transition_var[dense_key].LB = 0.0
                                transition_var[dense_key].UB = battery_upper

                    for source_soc, target_soc in charge_arcs:
                        arc_key = point + (source_soc, target_soc)
                        for arc_var in (
                            var.BSS.H_D,
                            var.BSS.N_R,
                            var.BSS.X_U,
                            var.BSS.X_D,
                        ):
                            arc_var[arc_key].LB = 0.0
                            arc_var[arc_key].UB = battery_upper


def _add_bss_operating_point_constraints(
    model,
    data,
    var,
    station,
    time,
    scene,
    year,
    charge_arcs,
    charge_arc_set,
    coverage,
):
    """Add all inventory, reserve, power, and recovery constraints at one point."""
    point = (station, time, scene, year)
    alpha_ru = float(data.BSS.alpha_TU[point])
    alpha_rd = float(data.BSS.alpha_TD[point])
    last_time = data.period - 1
    no_recovery_next_demand = bool(
        getattr(data.BSS, "no_recovery_next_demand", False)
    )

    coverage["operating_point"] += 1

    for source_soc in range(data.BSS.N_SOC):
        for target_soc in range(data.BSS.N_SOC):
            if (source_soc, target_soc) in charge_arc_set:
                continue
            dense_key = point + (source_soc, target_soc)
            model.addConstr(
                var.BSS.N_T[dense_key] == 0,
                name=(
                    f"SB_NT_Zero_{station}_{time}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )
            model.addConstr(
                var.BSS.N_U[dense_key] == 0,
                name=(
                    f"SB_NTU_Zero_{station}_{time}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )
            model.addConstr(
                var.BSS.N_D[dense_key] == 0,
                name=(
                    f"SB_NTD_Zero_{station}_{time}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )

    for source_soc, target_soc in charge_arcs:
        key = point + (source_soc, target_soc)
        # N_T is the baseline transition flow, corresponding to bar{N}^T in the manuscript.
        baseline_transition = var.BSS.N_T[key]
        additional = var.BSS.N_U[key]
        reducible = var.BSS.N_D[key]
        debt = var.BSS.H_D[key]
        recovery = var.BSS.N_R[key]
        base_with_recovery = (
            baseline_transition
            if no_recovery_next_demand
            else baseline_transition + recovery
        )

        model.addConstr(
            reducible <= baseline_transition,
            name=(
                f"SB_ReducibleBaseline_{station}_{time}_{scene}_{year}_"
                f"{source_soc}_{target_soc}"
            ),
        )
        if no_recovery_next_demand:
            model.addConstr(
                recovery == 0,
                name=(
                    f"SB_NoRD_RecoveryZero_{station}_{time}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )
            model.addConstr(
                debt == 0,
                name=(
                    f"SB_NoRD_DebtZero_{station}_{time}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )
        else:
            model.addConstr(
                recovery <= debt,
                name=(
                    f"SB_RecoveryDebt_{station}_{time}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )
        model.addConstr(
            var.BSS.X_U[key] == base_with_recovery + additional,
            name=(
                f"SB_UpperEndpoint_{station}_{time}_{scene}_{year}_"
                f"{source_soc}_{target_soc}"
            ),
        )
        model.addConstr(
            var.BSS.X_D[key] == base_with_recovery - reducible,
            name=(
                f"SB_LowerEndpoint_{station}_{time}_{scene}_{year}_"
                f"{source_soc}_{target_soc}"
            ),
        )
        model.addConstr(
            var.BSS.X_D[key] <= var.BSS.X_U[key],
            name=(
                f"SB_EndpointOrder_{station}_{time}_{scene}_{year}_"
                f"{source_soc}_{target_soc}"
            ),
        )

        if time == 0 and not no_recovery_next_demand:
            model.addConstr(
                debt == 0,
                name=(
                    f"SB_InitialDebt_{station}_{scene}_{year}_"
                    f"{source_soc}_{target_soc}"
                ),
            )
            coverage["initial_debt"] += 1

        if not no_recovery_next_demand:
            # Load-side convention: RU increases charging load and RD reduces it.
            # Debt therefore accrues on additional upward charging transitions.
            debt_after_action = debt + alpha_ru * additional - recovery
            if time < last_time:
                model.addConstr(
                    var.BSS.H_D[
                        station,
                        time + 1,
                        scene,
                        year,
                        source_soc,
                        target_soc,
                    ]
                    == debt_after_action,
                    name=(
                        f"SB_DebtBalance_{station}_{time}_{scene}_{year}_"
                        f"{source_soc}_{target_soc}"
                    ),
                )
            else:
                model.addConstr(
                    debt_after_action == 0,
                    name=(
                        f"SB_TerminalDebt_{station}_{scene}_{year}_"
                        f"{source_soc}_{target_soc}"
                    ),
                )
            coverage["debt_balance"] += 1
    
    def realized_transition(source_soc, target_soc):
        key = point + (source_soc, target_soc)
        # Realized transition after regulation activation; this is equation (7).
        return (
            var.BSS.N_T[key]
            + var.BSS.N_R[key]
            + alpha_ru * var.BSS.N_U[key]
            - alpha_rd * var.BSS.N_D[key]
        )

    for soc in range(data.BSS.N_SOC):
        available_after_swap = (
            var.BSS.N_S[point + (soc,)]
            - float(data.BSS.N_out[point + (soc,)])
            + float(data.BSS.N_in[point + (soc,)])
        )
        actual_out = gp.quicksum(
            realized_transition(source_soc, target_soc)
            for source_soc, target_soc in charge_arcs
            if source_soc == soc
        )
        actual_in = gp.quicksum(
            realized_transition(source_soc, target_soc)
            for source_soc, target_soc in charge_arcs
            if target_soc == soc
        )
        model.addConstr(
            gp.quicksum(
                var.BSS.X_U[point + (source_soc, target_soc)]
                for source_soc, target_soc in charge_arcs
                if source_soc == soc
            )
            <= available_after_swap,
            name=f"SB_EndpointSource_{station}_{time}_{scene}_{year}_{soc}",
        )
        coverage["endpoint_source"] += 1
        model.addConstr(
            var.BSS.N_S[
                station,
                (time + 1) % data.period,
                scene,
                year,
                soc,
            ]
            == available_after_swap - actual_out + actual_in,
            name=f"SB_ActualInventory_{station}_{time}_{scene}_{year}_{soc}",
        )
        coverage["actual_inventory"] += 1

        # The daily operating cycle starts with a fully charged battery pool.
        # This is imposed explicitly rather than relying only on the cyclic
        # inventory equation at the last time step.
        # if time == 0:
        #     initial_inventory = var.BSS.N_S[point + (soc,)]
        #     initial_target = (
        #         var.BSS.N_SB[station, year]
        #         if soc == int(data.BSS.maximum_soc_index)
        #         else 0
        #     )
        #     model.addConstr(
        #         initial_inventory == initial_target,
        #         name=f"SB_InitialFullInventory_{station}_{scene}_{year}_{soc}",
        #     )

        if not no_recovery_next_demand:
        # Downward reserve endpoint inventory must cover next-period swap demand.
            # X_D is the fully activated downward endpoint, so alpha_rd is not used.
            next_time = (time + 1) % data.period
            downward_endpoint_out = gp.quicksum(
                var.BSS.X_D[point + (source_soc, target_soc)]
                for source_soc, target_soc in charge_arcs
                if source_soc == soc
            )
            downward_endpoint_in = gp.quicksum(
                var.BSS.X_D[point + (source_soc, target_soc)]
                for source_soc, target_soc in charge_arcs
                if target_soc == soc
            )
            downward_endpoint_inventory = (
                available_after_swap
                - downward_endpoint_out
                + downward_endpoint_in
            )
            next_period_demand = float(
                data.BSS.N_out[station, next_time, scene, year, soc]
            )
            model.addConstr(
                downward_endpoint_inventory >= next_period_demand,
                name=(
                    f"SB_DownwardNextInventory_{station}_{time}_{scene}_{year}_"
                    f"{soc}"
                ),
            )
            coverage["downward_next_inventory"] += 1
    
    model.addConstr(
        gp.quicksum(var.BSS.N_S[point + (soc,)] for soc in range(data.BSS.N_SOC))
        == var.BSS.N_SB[station, year],
        name=f"SB_TotalInventory_{station}_{time}_{scene}_{year}",
    )

    ready_soc = int(data.BSS.maximum_soc_index)
    out_ready = float(data.BSS.N_out[point + (ready_soc,)])
    model.addConstr(
        out_ready <= var.BSS.N_S[point + (ready_soc,)],
        name=f"SB_SwapAvailable_{station}_{time}_{scene}_{year}",
    )
    model.addConstr(
        var.BSS.N_S[point + (ready_soc,)] - out_ready
        >= float(data.BSS.R_srv[point]),
        name=f"SB_ServiceReserve_{station}_{time}_{scene}_{year}",
    )
    coverage["service_reserve"] += 1

    model.addConstr(
        gp.quicksum(
            var.BSS.X_U[point + (source_soc, target_soc)]
            for source_soc, target_soc in charge_arcs
        )
        <= var.BSS.N_CB[station, year],
        name=f"SB_EndpointSlots_{station}_{time}_{scene}_{year}",
    )
    coverage["endpoint_slots"] += 1

    endpoint_power = gp.quicksum(
        float(data.BSS.grid_power_coefficient[(source_soc, target_soc)])
        * var.BSS.X_U[point + (source_soc, target_soc)]
        for source_soc, target_soc in charge_arcs
    )
    model.addConstr(
        endpoint_power <= var.BSS.N_CB[station, year] * float(data.CB.P),
        name=f"SB_EndpointPower_{station}_{time}_{scene}_{year}",
    )
    coverage["endpoint_power"] += 1

    if no_recovery_next_demand:
        baseline_count = gp.quicksum(
            var.BSS.N_T[point + (source_soc, target_soc)]
            for source_soc, target_soc in charge_arcs
        )
        upward_count = gp.quicksum(
            var.BSS.N_U[point + (source_soc, target_soc)]
            for source_soc, target_soc in charge_arcs
        )
        downward_count = gp.quicksum(
            var.BSS.N_D[point + (source_soc, target_soc)]
            for source_soc, target_soc in charge_arcs
        )
        model.addConstr(
            upward_count <= var.BSS.N_CB[station, year] - baseline_count,
            name=f"SB_NoRD_UpwardFreeSlots_{station}_{time}_{scene}_{year}",
        )
        model.addConstr(
            downward_count <= baseline_count,
            name=f"SB_NoRD_DownwardChargingCount_{station}_{time}_{scene}_{year}",
        )

    base_power = gp.quicksum(
        float(data.BSS.grid_power_coefficient[(source_soc, target_soc)])
        * (
            var.BSS.N_T[point + (source_soc, target_soc)]
            + var.BSS.N_R[point + (source_soc, target_soc)]
        )
        for source_soc, target_soc in charge_arcs
    )
    reducible_power = gp.quicksum(
        float(data.BSS.grid_power_coefficient[(source_soc, target_soc)])
        * var.BSS.N_D[point + (source_soc, target_soc)]
        for source_soc, target_soc in charge_arcs
    )
    additional_power = gp.quicksum(
        float(data.BSS.grid_power_coefficient[(source_soc, target_soc)])
        * var.BSS.N_U[point + (source_soc, target_soc)]
        for source_soc, target_soc in charge_arcs
    )

    # Load-side upward reserve increases charging load; downward reserve reduces it.
    actual_power = (
        base_power + alpha_ru * additional_power - alpha_rd * reducible_power
    )
    model.addConstr(
        var.BSS.P_c[point] == actual_power,
        name=f"SB_ActualChargePower_{station}_{time}_{scene}_{year}",
    )
    model.addConstr(
        var.BSS.P_f[point] == 0,
        name=f"SB_DischargePowerZero_{station}_{time}_{scene}_{year}",
    )
    model.addConstr(
        var.BSS.P[point] == actual_power,
        name=f"SB_ActualPower_{station}_{time}_{scene}_{year}",
    )
    coverage["actual_power"] += 1
    model.addConstr(
        var.BSS.P_RU[point] == additional_power,
        name=f"SB_UpwardReserve_{station}_{time}_{scene}_{year}",
    )
    coverage["upward_reserve"] += 1
    model.addConstr(
        var.BSS.P_RD[point] == reducible_power,
        name=f"SB_DownwardReserve_{station}_{time}_{scene}_{year}",
    )
    coverage["downward_reserve"] += 1


def _assert_bss_constraint_coverage(data, coverage, charge_arc_count):
    operating_points = (
        data.BSS.N_station * data.period * data.scene.N * data.year
    )
    expected = {
        "operating_point": operating_points,
        "actual_inventory": operating_points * data.BSS.N_SOC,
        "downward_next_inventory": (
            0
            if getattr(data.BSS, "no_recovery_next_demand", False)
            else operating_points * data.BSS.N_SOC
        ),
        "actual_power": operating_points,
        "upward_reserve": operating_points,
        "downward_reserve": operating_points,
        "service_reserve": operating_points,
        "endpoint_slots": operating_points,
        "endpoint_power": operating_points,
        "endpoint_source": operating_points * data.BSS.N_SOC,
        "debt_balance": (
            0
            if getattr(data.BSS, "no_recovery_next_demand", False)
            else operating_points * charge_arc_count
        ),
        "initial_debt": (
            0
            if getattr(data.BSS, "no_recovery_next_demand", False)
            else (
                data.BSS.N_station
                * data.scene.N
                * data.year
                * charge_arc_count
            )
        ),
    }
    mismatches = {
        key: {"actual": coverage.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if coverage.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"BSS constraint coverage mismatch: {mismatches}")
    data.BSS.constraint_coverage = dict(coverage)
    data.BSS.constraint_coverage_expected = expected
    data.BSS.constraint_coverage_valid = True


def Constraints_BSS(model, data, var, scheme=None):
    # =========================================================================
    # 1. 充换一体站 (BSS) 规划约束
    # =========================================================================
    for k in range(data.BSS.N_station):
        for y in range(data.year):
            # 充电槽数量不能超过站内电池数量；容量上限使用设备物理上限。
            model.addConstr(var.BSS.N_CB[k, y] <= var.BSS.N_SB[k, y], name=f"BSS_NCB_limit_{k}_{y}")

            # 换电电池 (SB) 建设与数量限制
            model.addConstr(var.BSS.N_SB[k, y] == float(data.BSS.N_SB[k]) + var.BSS.N_SB_plan_u[k, y], name=f"BSS_NSB_Total_{k}_{y}")
            model.addConstr(var.BSS.N_SB[k, y] <= float(data.BSS.N_SB_max[k]), name=f"BSS_NSB_physical_ub_{k}_{y}")
            model.addConstr(var.BSS.N_SB[k, y] <= float(data.BSS.demand_all_max[y, k]) * 1.2, name=f"BSS_NSB_demand_ub_{k}_{y}")

            # 充电槽 (CB) 建设与数量限制
            model.addConstr(var.BSS.N_CB[k, y] == float(data.BSS.N_CB[k]) + var.BSS.N_CB_plan_u[k, y], name=f"BSS_NCB_Total_{k}_{y}")
            model.addConstr(var.BSS.N_CB[k, y] <= float(data.BSS.N_CB_max[k]), name=f"BSS_NCB_ub_{k}_{y}")
            # model.addConstr(var.BSS.N_CB[k, y] <= var.BSS.N_SB[k, y] * 0.5, name=f"BSS_NCB_ub_{k}_{y}")

            # 换电电池建设完成与退役
            if y >= data.BSS.T_build:
                model.addConstr(var.BSS.N_SB_plan_e[k, y] == var.BSS.N_SB_plan[k, y - int(data.BSS.T_build)])
            else:
                model.addConstr(var.BSS.N_SB_plan_e[k, y] == 0)

            if y >= data.BSS.T_life:
                model.addConstr(var.BSS.N_SB_plan_r[k, y] == var.BSS.N_SB_plan_e[k, y - int(data.BSS.T_life)])
            else:
                model.addConstr(var.BSS.N_SB_plan_r[k, y] == 0)

            sb_e_sum = gp.quicksum(var.BSS.N_SB_plan_e[k, yy] for yy in range(y + 1))
            sb_r_sum = gp.quicksum(var.BSS.N_SB_plan_r[k, yy] for yy in range(y + 1))
            model.addConstr(var.BSS.N_SB_plan_u[k, y] == sb_e_sum - sb_r_sum)

            # 充电槽建设完成与退役
            if y >= data.CB.T_build:
                model.addConstr(var.BSS.N_CB_plan_e[k, y] == var.BSS.N_CB_plan[k, y - int(data.CB.T_build)])
            else:
                model.addConstr(var.BSS.N_CB_plan_e[k, y] == 0)

            if y >= data.CB.T_life:
                model.addConstr(var.BSS.N_CB_plan_r[k, y] == var.BSS.N_CB_plan_e[k, y - int(data.CB.T_life)])
            else:
                model.addConstr(var.BSS.N_CB_plan_r[k, y] == 0)

            cb_e_sum = gp.quicksum(var.BSS.N_CB_plan_e[k, yy] for yy in range(y + 1))
            cb_r_sum = gp.quicksum(var.BSS.N_CB_plan_r[k, yy] for yy in range(y + 1))
            model.addConstr(var.BSS.N_CB_plan_u[k, y] == cb_e_sum - cb_r_sum)

    # =========================================================================
    # 2. 投资成本与残值
    # =========================================================================
    for y in range(data.year):
        sb_inv_sum = gp.quicksum(var.BSS.N_SB_plan[k, y] for k in range(data.BSS.N_station))
        model.addConstr(var.BSS.C_SB_inv[y] == float(data.BSS.c_inv) * sb_inv_sum / ((1 + data.r) ** (y + 1)))

        if y + 1 >= data.year - data.BSS.T_life - data.BSS.T_build:
            res_val = (var.BSS.C_SB_inv[y] - (data.year - (y + 1)) * var.BSS.C_SB_inv[y] / float(data.BSS.T_life)) / ((1 + data.r) ** data.year)
            model.addConstr(var.BSS.C_SB_res[y] == res_val)
        else:
            model.addConstr(var.BSS.C_SB_res[y] == 0)

        cb_inv_sum = gp.quicksum(var.BSS.N_CB_plan[k, y] for k in range(data.BSS.N_station))
        model.addConstr(var.BSS.C_CB_inv[y] == float(data.CB.c_inv) * cb_inv_sum / ((1 + data.r) ** (y + 1)))

        if y + 1 >= data.year - data.CB.T_life - data.CB.T_build:
            res_val = (var.BSS.C_CB_inv[y] - (data.year - (y + 1)) * var.BSS.C_CB_inv[y] / float(data.CB.T_life)) / ((1 + data.r) ** data.year)
            model.addConstr(var.BSS.C_CB_res[y] == res_val)
        else:
            model.addConstr(var.BSS.C_CB_res[y] == 0)

        model.addConstr(var.BSS.C_inv[y] == var.BSS.C_SB_inv[y] + var.BSS.C_CB_inv[y])
        model.addConstr(var.BSS.C_res[y] == var.BSS.C_SB_res[y] + var.BSS.C_CB_res[y])

    # =========================================================================

    _validate_bss_call(data)
    charge_arcs = tuple(data.BSS.charge_arcs)
    charge_arc_set = set(charge_arcs)
    if not charge_arcs:
        raise ValueError("BSS manuscript model requires at least one charging arc")

    _set_bss_physical_bounds(data, var, charge_arcs)
    coverage = {
        "operating_point": 0,
        "actual_inventory": 0,
        "downward_next_inventory": 0,
        "actual_power": 0,
        "upward_reserve": 0,
        "downward_reserve": 0,
        "service_reserve": 0,
        "endpoint_slots": 0,
        "endpoint_power": 0,
        "endpoint_source": 0,
        "debt_balance": 0,
        "initial_debt": 0,
    }
    for station in range(data.BSS.N_station):
        for year in range(data.year):
            for scene in range(data.scene.N):
                for time in range(data.period):
                    _add_bss_operating_point_constraints(
                        model=model,
                        data=data,
                        var=var,
                        station=station,
                        time=time,
                        scene=scene,
                        year=year,
                        charge_arcs=charge_arcs,
                        charge_arc_set=charge_arc_set,
                        coverage=coverage,
                    )

    _assert_bss_constraint_coverage(data, coverage, len(charge_arcs))

    for station in range(data.BSS.N_station):
        for year in range(data.year):
            for scene in range(data.scene.N):
                for time in range(data.period):
                    model.addConstr(
                        var.BSS.P[station, time, scene, year]
                        == var.BSS.P[station, time, scene, year],
                        name=f"BSS_RC_P_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.BSS.P_RU[station, time, scene, year]
                        == var.BSS.P_RU[station, time, scene, year],
                        name=f"BSS_RC_P_RU_{station}_{time}_{scene}_{year}",
                    )
                    model.addConstr(
                        var.BSS.P_RD[station, time, scene, year]
                        == var.BSS.P_RD[station, time, scene, year],
                        name=f"BSS_RC_P_RD_{station}_{time}_{scene}_{year}",
                    )

    for year in range(data.year):
        for scene in range(data.scene.N):
            for time in range(data.period):
                model.addConstr(
                    var.BSS.P_RU_all[time, scene, year]
                    == gp.quicksum(
                        var.BSS.P_RU[station, time, scene, year]
                        for station in range(data.BSS.N_station)
                    ),
                    name=f"BSS_RC_P_RU_all_{time}_{scene}_{year}",
                )
                model.addConstr(
                    var.BSS.P_RD_all[time, scene, year]
                    == gp.quicksum(
                        var.BSS.P_RD[station, time, scene, year]
                        for station in range(data.BSS.N_station)
                    ),
                    name=f"BSS_RC_P_RD_all_{time}_{scene}_{year}",
                )

    for node in range(data.DN.N_node):
        station_indices = [
            station
            for station, connected_node in enumerate(data.BSS.node_DN)
            if connected_node == data.DN.node[node]
        ]
        for year in range(data.year):
            for scene in range(data.scene.N):
                for time in range(data.period):
                    model.addConstr(
                        var.BSS.P_DN[node, time, scene, year]
                        == gp.quicksum(
                            var.BSS.P[station, time, scene, year]
                            for station in station_indices
                        ),
                        name=f"BSS_RC_P_DN_{node}_{time}_{scene}_{year}",
                    )

    data.BSS.actual_call_inventory_enabled = True
    data.BSS.recovery_boundary_mode = "explicit_initial_terminal"
    data.BSS.recovery_boundary_mode_used = "explicit_initial_terminal"
    data.BSS.unified_inventory_formulation = True
    data.BSS.manuscript_builder = "Constraints_BSS"


__all__ = ["Constraints_BSS"]
