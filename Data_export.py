import numpy as np
import math


class Struct:
    """用于模拟 MATLAB 结构体的类"""
    pass


def get_val(v):
    """
    通用取值函数：
    - 如果是 Gurobi 变量 (Var)，使用 .X
    - 如果是 Gurobi 线性表达式 (LinExpr)，使用 .getValue()
    - 如果已经是数值，直接返回
    """
    try:
        return v.X
    except AttributeError:
        try:
            return v.getValue()
        except AttributeError:
            return float(v)


def _export_source_zone_audit(data, var, result):
    if not hasattr(var.HN, "source_zone_affiliation"):
        return

    affiliation = np.zeros(
        (data.HN.N_node, data.HN.N_station, data.year), dtype=np.int8
    )
    for node in range(data.HN.N_node):
        for station in range(data.HN.N_station):
            for year in range(data.year):
                affiliation[node, station, year] = int(
                    round(
                        get_val(
                            var.HN.source_zone_affiliation[node, station, year]
                        )
                    )
                )

    result.HN.source_zone_affiliation = affiliation
    result.HN.k_f_load = affiliation[
        np.asarray(data.HN.load_node_idx, dtype=int), :, :
    ].copy()

    reachable_nodes = np.zeros_like(affiliation)
    reachability = np.zeros(
        (data.HN.N_load, data.HN.N_station, data.year), dtype=np.int8
    )
    service_zone_node_ids = np.empty(
        (data.year, data.HN.N_station), dtype=object
    )
    service_zone_load_ids = np.empty(
        (data.year, data.HN.N_station), dtype=object
    )
    max_path_delay = np.zeros((data.HN.N_station, data.year), dtype=int)
    max_delay_path_node_ids = np.empty(
        (data.year, data.HN.N_station), dtype=object
    )
    max_delay_path_pipe_ids = np.empty(
        (data.year, data.HN.N_station), dtype=object
    )
    pipe_delays = list(
        getattr(data.HN, "contract_pipe_delay_used", [0] * data.HN.N_pipe)
    )

    for year in range(data.year):
        adjacency = [[] for _ in range(data.HN.N_node)]
        for pipe in range(data.HN.N_pipe):
            head = data.HN.node_to_idx[int(data.HN.head[pipe])]
            tail = data.HN.node_to_idx[int(data.HN.tail[pipe])]
            if int(round(result.HN.b_ij[pipe, year])) == 1:
                adjacency[head].append((tail, pipe))
            elif int(round(result.HN.b_ji[pipe, year])) == 1:
                adjacency[tail].append((head, pipe))

        for station, root in enumerate(data.HN.station_node_idx):
            visited = set()
            stack = [(root, [root], [], 0)]
            best_delay = 0
            best_nodes = [root]
            best_pipes = []
            while stack:
                node, path_nodes, path_pipes, path_delay = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                reachable_nodes[node, station, year] = 1
                if (
                    node in data.HN.load_node_idx
                    and affiliation[node, station, year] == 1
                    and path_delay >= best_delay
                ):
                    best_delay = path_delay
                    best_nodes = path_nodes
                    best_pipes = path_pipes
                for next_node, pipe in adjacency[node]:
                    if next_node not in visited:
                        stack.append(
                            (
                                next_node,
                                path_nodes + [next_node],
                                path_pipes + [pipe],
                                path_delay + int(pipe_delays[pipe]),
                            )
                        )

            max_path_delay[station, year] = best_delay
            max_delay_path_node_ids[year, station] = np.asarray(
                [int(data.HN.node[node]) for node in best_nodes], dtype=int
            )
            max_delay_path_pipe_ids[year, station] = np.asarray(
                [int(data.HN.pipe[pipe]) for pipe in best_pipes], dtype=int
            )
            service_zone_node_ids[year, station] = np.asarray(
                [
                    int(data.HN.node[node])
                    for node in range(data.HN.N_node)
                    if affiliation[node, station, year] == 1
                ],
                dtype=int,
            )
            service_zone_load_ids[year, station] = np.asarray(
                [
                    int(data.HN.load[load])
                    for load, node in enumerate(data.HN.load_node_idx)
                    if affiliation[node, station, year] == 1
                ],
                dtype=int,
            )

        for load, node in enumerate(data.HN.load_node_idx):
            reachability[load, :, year] = reachable_nodes[node, :, year]

    mismatch = np.maximum(affiliation - reachable_nodes, 0).astype(np.int8)
    assigned_station = np.full((data.HN.N_node, data.year), -1, dtype=int)
    for node in range(data.HN.N_node):
        for year in range(data.year):
            assigned = np.flatnonzero(affiliation[node, :, year])
            if assigned.size:
                assigned_station[node, year] = int(assigned[0])

    if data.year > 1:
        cross_stage_change = (
            assigned_station[:, 1:] != assigned_station[:, :-1]
        ).astype(np.int8)
    else:
        cross_stage_change = np.zeros((data.HN.N_node, 0), dtype=np.int8)

    result.HN.station_load_reachability = reachability
    result.HN.source_zone_reachable_nodes = reachable_nodes
    result.HN.source_zone_reachability_mismatch = mismatch
    result.HN.source_zone_reachability_consistent = bool(not np.any(mismatch))
    result.HN.service_zone_node_ids = service_zone_node_ids
    result.HN.service_zone_load_ids = service_zone_load_ids
    result.HN.assigned_station_index = assigned_station
    result.HN.cross_stage_affiliation_change = cross_stage_change
    result.HN.max_active_path_delay = max_path_delay
    result.HN.max_delay_path_node_ids = max_delay_path_node_ids
    result.HN.max_delay_path_pipe_ids = max_delay_path_pipe_ids


def _export_contract_tube_results(data, var, result):
    suffixes = ("S_upper", "S_lower", "R_upper", "R_lower")
    if not all(hasattr(var.HN, f"tube_cap_{suffix}") for suffix in suffixes):
        return

    result.HN.contract_tube = Struct()
    result.HN.contract_tube.enabled = True
    result.HN.contract_tube.delay_window = int(
        getattr(data.HN, "contract_tube_D_max_used", 0)
    )
    result.HN.contract_tube.exact_binary_product = bool(
        getattr(data.HN, "contract_tube_exact_binary_product", False)
    )
    result.HN.contract_tube.product_formulation = (
        "exact_binary_linearization"
        if result.HN.contract_tube.exact_binary_product
        else "single_box_mccormick"
    )
    result.HN.contract_tube.supply_margin_upper = float(data.HN.T_S_max) - float(data.HN.T_S_min)
    result.HN.contract_tube.return_margin_upper = float(data.HN.T_R_max) - float(data.HN.T_R_min)
    result.HN.contract_tube.flow_upper = np.asarray(data.HN.f_station, dtype=float).copy()
    result.HN.contract_tube.water_heat_capacity = float(data.HN.c_w)

    result.HN.f_station = np.zeros((data.HN.N_station, data.year))
    for station in range(data.HN.N_station):
        for year in range(data.year):
            result.HN.f_station[station, year] = get_val(var.HN.f_station[station, year])

    for suffix in suffixes:
        setattr(result.HN, f"margin_{suffix}", {})
        setattr(result.HN, f"tube_cap_{suffix}", {})
        setattr(result.HN, f"physical_margin_{suffix}", {})
        setattr(result.HN, f"controlling_node_{suffix}", {})
    for name in ("dH_SA", "dH_DC", "rH"):
        setattr(result.HN, name, {})
    result.HN.P_HN_del_SA = {}
    result.HN.P_HN_del_DC = {}

    for year in range(data.year):
        for scene in range(data.scene.N):
            key = (year, scene)
            for suffix in suffixes:
                margin = np.zeros((data.period, data.HN.N_station))
                capacity = np.zeros((data.period, data.HN.N_station))
                physical_margin = np.zeros((data.period, data.HN.N_station))
                controlling_node = np.full(
                    (data.period, data.HN.N_station), -1, dtype=int
                )
                for time in range(data.period):
                    for station in range(data.HN.N_station):
                        margin[time, station] = get_val(
                            getattr(var.HN, f"margin_{suffix}")[station, time, scene, year]
                        )
                        capacity[time, station] = get_val(
                            getattr(var.HN, f"tube_cap_{suffix}")[station, time, scene, year]
                        )
                        assigned_nodes = np.flatnonzero(
                            result.HN.source_zone_affiliation[:, station, year]
                        )
                        if assigned_nodes.size:
                            node_margins = []
                            for node in assigned_nodes:
                                if suffix == "S_upper":
                                    value = float(data.HN.T_S_max) - get_val(
                                        var.HN.T_S_node[node, time, scene, year]
                                    )
                                elif suffix == "S_lower":
                                    value = get_val(
                                        var.HN.T_S_node[node, time, scene, year]
                                    ) - float(data.HN.T_S_min)
                                elif suffix == "R_upper":
                                    value = float(data.HN.T_R_max) - get_val(
                                        var.HN.T_R_node[node, time, scene, year]
                                    )
                                else:
                                    value = get_val(
                                        var.HN.T_R_node[node, time, scene, year]
                                    ) - float(data.HN.T_R_min)
                                node_margins.append(float(value))
                            controlling_index = int(np.argmin(node_margins))
                            controlling_node[time, station] = int(
                                data.HN.node[assigned_nodes[controlling_index]]
                            )
                            physical_margin[time, station] = node_margins[
                                controlling_index
                            ]
                getattr(result.HN, f"margin_{suffix}")[key] = margin
                getattr(result.HN, f"tube_cap_{suffix}")[key] = capacity
                getattr(result.HN, f"physical_margin_{suffix}")[key] = physical_margin
                getattr(result.HN, f"controlling_node_{suffix}")[key] = controlling_node

            for name in ("dH_SA", "dH_DC", "rH"):
                values = np.zeros((data.period, data.HN.N_station))
                for time in range(data.period):
                    for station in range(data.HN.N_station):
                        values[time, station] = get_val(
                            getattr(var.HN, name)[station, time, scene, year]
                        )
                getattr(result.HN, name)[key] = values

            result.HN.P_HN_del_SA[key] = np.array([
                get_val(var.HN.P_HN_del_SA[time, scene, year])
                for time in range(data.period)
            ])
            result.HN.P_HN_del_DC[key] = np.array([
                get_val(var.HN.P_HN_del_DC[time, scene, year])
                for time in range(data.period)
            ])


def Data_export(data, var, obj):
    """
    结果导出模块 (Python + 纯 for 循环实现)
    """
    result = Struct()
    scheme = Struct()

    # 初始化子结构
    result.cost = Struct()
    result.DN = Struct()
    result.PV = Struct()
    result.ES = Struct()
    result.BSS = Struct()
    result.BSS = Struct()
    result.HN = Struct()
    result.CHP = Struct()
    result.EB = Struct()
    result.emission = Struct()
    result.SBT = Struct()

    scheme.DN = Struct()
    scheme.ES = Struct()
    scheme.BSS = Struct()
    scheme.HN = Struct()

    # =========================================================================
    # 1. 优化目标与成本结果
    # =========================================================================
    result.cost.C_inv = get_val(obj.C_inv) - get_val(obj.C_res)
    result.cost.C_line_inv = get_val(obj.C_line_inv) - get_val(obj.C_line_res)
    result.cost.C_node_inv = get_val(obj.C_node_inv) - get_val(obj.C_node_res)
    result.cost.C_ES_inv = get_val(obj.C_ES_inv) - get_val(obj.C_ES_res)
    result.cost.C_PV_inv = get_val(obj.C_PV_inv) - get_val(obj.C_PV_res)
    result.cost.C_BSS_inv = get_val(obj.C_BSS_inv) - get_val(obj.C_BSS_res)
    result.cost.C_pipe_inv = get_val(obj.C_pipe_inv) - get_val(obj.C_pipe_res)
    result.cost.C_CHP_inv = get_val(obj.C_CHP_inv) - get_val(obj.C_CHP_res)
    result.cost.C_EB_inv = get_val(obj.C_EB_inv) - get_val(obj.C_EB_res)
    result.cost.C_DN_inv = get_val(obj.C_DN_inv) - get_val(obj.C_DN_res)
    result.cost.C_HN_inv = get_val(obj.C_HN_inv) - get_val(obj.C_HN_res)

    # 多主体成本：IEHS 与 BSS 的规划、运行及购售电结算。
    result.cost.C_IEHS_plan_inv = get_val(obj.C_IEHS_plan_inv)
    result.cost.C_IEHS_plan_res = get_val(obj.C_IEHS_plan_res)
    result.cost.C_IEHS_plan = get_val(obj.C_IEHS_plan)
    for name in (
        "C_IEHS_line_inv", "C_IEHS_line_res",
        "C_IEHS_node_inv", "C_IEHS_node_res",
        "C_IEHS_pipe_inv", "C_IEHS_pipe_res",
        "C_IEHS_CHP_inv", "C_IEHS_CHP_res",
        "C_IEHS_EB_inv", "C_IEHS_EB_res",
        "C_IEHS_ES_inv", "C_IEHS_ES_res",
    ):
        setattr(result.cost, name, get_val(getattr(obj, name)))
    result.cost.C_BSS_plan_inv = get_val(obj.C_BSS_plan_inv)
    result.cost.C_BSS_plan_res = get_val(obj.C_BSS_plan_res)
    result.cost.C_BSS_plan = get_val(obj.C_BSS_plan)
    result.cost.C_SB_inv = get_val(obj.C_SB_inv)
    result.cost.C_SB_res = get_val(obj.C_SB_res)
    result.cost.C_CB_inv = get_val(obj.C_CB_inv)
    result.cost.C_CB_res = get_val(obj.C_CB_res)

    result.cost.C_ope = get_val(obj.C_ope)
    result.cost.C_ele = get_val(obj.C_ele)
    result.cost.C_gas = get_val(obj.C_gas)
    result.cost.C_emi = get_val(obj.C_emi)
    result.cost.C_IEHS_sale = get_val(obj.C_IEHS_sale)
    result.cost.C_BSS_buy = get_val(obj.C_BSS_buy)
    result.cost.C_IEHS_ope = get_val(obj.C_IEHS_ope)
    result.cost.C_BSS_ope = get_val(obj.C_BSS_ope)

    result.cost.C_multi_agent_array = np.asarray(
        [
            result.cost.C_IEHS_plan,
            result.cost.C_IEHS_ope,
            result.cost.C_BSS_plan,
            result.cost.C_BSS_ope,
            result.cost.C_IEHS_sale,
            result.cost.C_BSS_buy,
        ],
        dtype=float,
    )
    result.cost.C_multi_agent_array_labels = np.asarray(
        [
            "IEHS规划成本",
            "IEHS运行成本",
            "BSS规划成本",
            "BSS运行成本",
            "IEHS向BSS售电收益",
            "BSS向IEHS购电成本",
        ],
        dtype=object,
    )
    result.C_IEHS_plan = result.cost.C_IEHS_plan
    result.C_IEHS_ope = result.cost.C_IEHS_ope
    result.C_BSS_plan = result.cost.C_BSS_plan
    result.C_BSS_ope = result.cost.C_BSS_ope
    result.C_IEHS_sale = result.cost.C_IEHS_sale
    result.C_BSS_buy = result.cost.C_BSS_buy
    result.C_multi_agent_array = result.cost.C_multi_agent_array
    result.C_multi_agent_array_labels = result.cost.C_multi_agent_array_labels

    result.cost.C_risk = get_val(obj.C_risk)
    result.cost.C_PV = get_val(obj.C_PV)
    result.cost.C_lack = get_val(obj.C_lack)
    result.cost.C_all = get_val(obj.F)

    result.cost.C_detailed_array = np.asarray(
        [
            result.cost.C_inv,
            result.cost.C_ope,
            result.cost.C_risk,
            result.cost.C_line_inv,
            result.cost.C_node_inv,
            result.cost.C_ES_inv,
            result.cost.C_pipe_inv,
            result.cost.C_CHP_inv,
            result.cost.C_EB_inv,
            result.cost.C_BSS_inv,
            result.cost.C_ele + result.cost.C_gas,
            result.cost.C_emi,
            result.cost.C_PV,
            result.cost.C_lack,
        ],
        dtype=float,
    )
    result.cost.C_detailed_array_labels = np.asarray(
        [
            "\u603b\u6295\u8d44\u6210\u672c",
            "\u603b\u8fd0\u884c\u6210\u672c",
            "\u603b\u60e9\u7f5a\u6210\u672c",
            "\u603b\u7ebf\u8def\u6295\u8d44\u6210\u672c",
            "\u603b\u5206\u53d8\u7535\u7ad9\u6295\u8d44\u6210\u672c",
            "\u603b\u50a8\u80fd\u6295\u8d44\u6210\u672c",
            "\u603b\u70ed\u7f51\u7ba1\u9053\u6295\u8d44\u6210\u672c",
            "\u603bCHP\u6295\u8d44\u6210\u672c",
            "\u603bEB\u6295\u8d44\u6210\u672c",
            "\u603bBSS\u6295\u8d44\u6210\u672c",
            "\u603b\u8d2d\u7535\u8d2d\u6c14\u6210\u672c",
            "\u603b\u73af\u5883\u6cbb\u7406\u6210\u672c",
            "\u603b\u5f03\u5149\u60e9\u7f5a\u6210\u672c",
            "\u603b\u7075\u6d3b\u6027\u77ed\u7f3a\u60e9\u7f5a\u6210\u672c",
        ],
        dtype=object,
    )
    result.C_detailed_array = result.cost.C_detailed_array
    result.C_detailed_array_labels = result.cost.C_detailed_array_labels

    result.cost.C_all_detail = np.zeros(3)
    result.cost.C_all_detail[0] = result.cost.C_inv
    result.cost.C_all_detail[1] = result.cost.C_ope
    result.cost.C_all_detail[2] = result.cost.C_risk

    result.cost.C_inv_detail = np.zeros(8)
    result.cost.C_inv_detail[0] = result.cost.C_line_inv
    result.cost.C_inv_detail[1] = result.cost.C_node_inv
    result.cost.C_inv_detail[2] = result.cost.C_PV_inv
    result.cost.C_inv_detail[3] = result.cost.C_ES_inv
    result.cost.C_inv_detail[4] = result.cost.C_pipe_inv
    result.cost.C_inv_detail[5] = result.cost.C_CHP_inv
    result.cost.C_inv_detail[6] = result.cost.C_EB_inv
    result.cost.C_inv_detail[7] = result.cost.C_BSS_inv

    result.cost.C_year = np.zeros((data.year, 3))
    result.cost.C_inv_year = np.zeros((data.year, 7))

    for y in range(data.year):
        result.cost.C_year[y, 0] = get_val(var.cost.C_inv[y])
        result.cost.C_year[y, 1] = get_val(var.cost.C_ope[y])
        result.cost.C_year[y, 2] = get_val(var.cost.C_risk[y])

        result.cost.C_inv_year[y, 0] = get_val(var.DN.C_line_inv[y])
        result.cost.C_inv_year[y, 1] = get_val(var.DN.C_node_inv[y])
        result.cost.C_inv_year[y, 2] = get_val(var.HN.C_pipe_inv[y])
        result.cost.C_inv_year[y, 3] = get_val(var.HN.C_CHP_inv[y])
        result.cost.C_inv_year[y, 4] = get_val(var.HN.C_EB_inv[y])
        result.cost.C_inv_year[y, 5] = get_val(var.BSS.C_inv[y])
        result.cost.C_inv_year[y, 6] = get_val(var.ES.C_inv[y])

    # =========================================================================
    # 2. 配电网 (DN) 结果
    # =========================================================================
    # 提取 2D 变量
    dn_node_vars = {'c', 'N_node_exp', 'N_node_exp_e', 'N_node_exp_u', 'S_node_exp', 'S_node'}
    dn_line_vars = {'y_line', 'b_ij', 'b_ji', 'N_line_exp', 'N_line_exp_e', 'N_line_exp_u', 'S_line'}
    for var_name in ['y_line', 'c', 'b_ij', 'b_ji', 'N_line_exp', 'N_line_exp_e', 'N_line_exp_u', 'S_line', 'N_node_exp', 'N_node_exp_e', 'N_node_exp_u', 'S_node_exp', 'S_node']:
        mat_rows = data.DN.N_node if var_name in dn_node_vars else data.DN.N_line
        mat = np.zeros((mat_rows, data.year))
        src_var = getattr(var.DN, var_name)
        for i in range(mat.shape[0]):
            for y in range(data.year):
                val = get_val(src_var[i, y])
                mat[i, y] = round(val) if ('y_line' in var_name or 'c' in var_name or 'b_' in var_name) else val

        # 【修改点】：只对 S_line 和 S_node 精确加 _plan 后缀
        if var_name in ['S_line', 'S_node']:
            attr_name = var_name + '_plan'
        else:
            attr_name = var_name

        setattr(result.DN, attr_name, mat)

    # 提取多维时序变量 (存入 dict 模拟 MATLAB 的 cell array)
    result.DN.U = {}
    result.DN.P_node = {}
    result.DN.Q_node = {}
    result.DN.P_load = {}
    result.DN.Q_load = {}
    result.DN.P_sub = {}
    result.DN.Q_sub = {}
    result.DN.P_line = {}
    result.DN.Q_line = {}
    result.DN.P_RU = {}
    result.DN.P_RD = {}
    result.DN.P_RU_lack = {}
    result.DN.P_RD_lack = {}
    result.DN.R_demand = {}
    result.DN.k_RU = {}
    result.DN.k_RD = {}
    result.DN.k_RU_lack = {}
    result.DN.k_RD_lack = {}
    result.DN.S_node = {}
    result.DN.k_node = {}
    result.DN.k_back = {}
    result.DN.S_line = {}
    result.DN.P_RU_detail = {}
    result.DN.P_RD_detail = {}
    result.DN.P_RU_lack_rate_year = np.zeros(data.year)
    result.DN.P_RD_lack_rate_year = np.zeros(data.year)
    result.DN.RU_lack_total_year = np.zeros(data.year)
    result.DN.RD_lack_total_year = np.zeros(data.year)
    result.DN.RU_demand_total_year = np.zeros(data.year)
    result.DN.RD_demand_total_year = np.zeros(data.year)
    # 年度累积加权可调量：[上调量, 下调量]，单位为功率单位·h（通常为 MWh）。
    # 权重包括代表性时段长度、场景天数和全部场景；场景数通过下方 scene 循环体现。
    result.DN.P_RU_RD_total_year = np.zeros((data.year, 2), dtype=float)
    result.DN.P_RU_RD_total_year_labels = np.asarray(["上调量", "下调量"], dtype=object)

    dn_u_min = 1e9
    dn_u_max = -1e9

    for y in range(data.year):
        for s in range(data.scene.N):
            # 初始化存储矩阵 (行: t, 列: i)
            result.DN.U[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.P_node[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.Q_node[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.P_load[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.Q_load[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.P_line[y, s] = np.zeros((data.period, data.DN.N_line))
            result.DN.Q_line[y, s] = np.zeros((data.period, data.DN.N_line))
            result.DN.P_sub[y, s] = np.zeros((data.period, data.DN.N_sub))
            result.DN.Q_sub[y, s] = np.zeros((data.period, data.DN.N_sub))

            result.DN.S_node[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.k_node[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.k_back[y, s] = np.zeros((data.period, data.DN.N_node))
            result.DN.S_line[y, s] = np.zeros((data.period, data.DN.N_line))

            result.DN.P_RU[y, s] = np.zeros(data.period)
            result.DN.P_RD[y, s] = np.zeros(data.period)
            result.DN.P_RU_lack[y, s] = np.zeros(data.period)
            result.DN.P_RD_lack[y, s] = np.zeros(data.period)
            result.DN.R_demand[y, s] = np.zeros(data.period)
            result.DN.k_RU[y, s] = np.zeros(data.period)
            result.DN.k_RD[y, s] = np.zeros(data.period)
            result.DN.k_RU_lack[y, s] = np.zeros(data.period)
            result.DN.k_RD_lack[y, s] = np.zeros(data.period)

            result.DN.P_RU_detail[y, s] = np.zeros((data.period, 4))
            result.DN.P_RD_detail[y, s] = np.zeros((data.period, 4))

            for t in range(data.period):
                # 提取节点变量
                for i in range(data.DN.N_node):
                    u_val = get_val(var.DN.U[i, t, s, y])
                    u_sqrt = math.sqrt(u_val) if u_val > 0 else 0
                    if u_sqrt > dn_u_max: dn_u_max = u_sqrt
                    if u_sqrt < dn_u_min: dn_u_min = u_sqrt

                    result.DN.U[y, s][t, i] = u_sqrt
                    result.DN.P_node[y, s][t, i] = get_val(var.DN.P_node[i, t, s, y])
                    result.DN.Q_node[y, s][t, i] = get_val(var.DN.Q_node[i, t, s, y])
                    result.DN.P_load[y, s][t, i] = get_val(var.DN.P_load[i, t, s, y])
                    result.DN.Q_load[y, s][t, i] = get_val(var.DN.Q_load[i, t, s, y])

                    # 容量计算
                    s_node_val = math.sqrt(result.DN.P_node[y, s][t, i] ** 2 + result.DN.Q_node[y, s][t, i] ** 2)
                    result.DN.S_node[y, s][t, i] = s_node_val
                    s_plan = result.DN.S_node_plan[i, y]
                    if s_plan > 0:
                        result.DN.k_node[y, s][t, i] = s_node_val / s_plan
                        result.DN.k_back[y, s][t, i] = s_node_val / s_plan if s_node_val > 0 else 0

                # 提取线路变量
                for i in range(data.DN.N_line):
                    p_line = get_val(var.DN.P_line[i, t, s, y])
                    q_line = get_val(var.DN.Q_line[i, t, s, y])
                    result.DN.P_line[y, s][t, i] = p_line
                    result.DN.Q_line[y, s][t, i] = q_line
                    result.DN.S_line[y, s][t, i] = math.sqrt(p_line ** 2 + q_line ** 2)

                # 提取变电站变量
                for j in range(data.DN.N_sub):
                    result.DN.P_sub[y, s][t, j] = get_val(var.DN.P_sub[j, t, s, y])
                    result.DN.Q_sub[y, s][t, j] = get_val(var.DN.Q_sub[j, t, s, y])

                # 提取系统备用变量
                p_ru = get_val(var.DN.P_RU[t, s, y])
                p_rd = get_val(var.DN.P_RD[t, s, y])
                r_dem = get_val(var.DN.R_demand[t, s, y])

                result.DN.P_RU[y, s][t] = p_ru
                result.DN.P_RD[y, s][t] = p_rd
                result.DN.P_RU_lack[y, s][t] = get_val(var.DN.P_RU_lack[t, s, y])
                result.DN.P_RD_lack[y, s][t] = get_val(var.DN.P_RD_lack[t, s, y])
                result.DN.R_demand[y, s][t] = r_dem
                result.DN.k_RU[y, s][t] = p_ru / r_dem if r_dem > 0 else 0
                result.DN.k_RD[y, s][t] = p_rd / r_dem if r_dem > 0 else 0
                result.DN.k_RU_lack[y, s][t] = p_ru / r_dem if r_dem > 0 else 0

                result.DN.P_RU_detail[y, s][t, 0] = get_val(var.DN.P_RU[t, s, y])
                result.DN.P_RU_detail[y, s][t, 1] = get_val(var.ES.P_RU_all[t, s, y])
                result.DN.P_RU_detail[y, s][t, 2] = get_val(var.HN.P_RU_all[t, s, y])
                result.DN.P_RU_detail[y, s][t, 3] = get_val(var.BSS.P_RU_all[t, s, y])

                result.DN.P_RD_detail[y, s][t, 0] = get_val(var.DN.P_RD[t, s, y])
                result.DN.P_RD_detail[y, s][t, 1] = get_val(var.ES.P_RD_all[t, s, y])
                result.DN.P_RD_detail[y, s][t, 2] = get_val(var.HN.P_RD_all[t, s, y])
                result.DN.P_RD_detail[y, s][t, 3] = get_val(var.BSS.P_RD_all[t, s, y])

    result.DN.u_fin = dn_u_min
    result.DN.u_fax = dn_u_max

    # Annual regulation shortfall rates, weighted by representative-day counts.
    for y in range(data.year):
        annual_ru_lack = 0.0
        annual_rd_lack = 0.0
        annual_ru_demand = 0.0
        annual_rd_demand = 0.0
        annual_ru_total = 0.0
        annual_rd_total = 0.0
        for s in range(data.scene.N):
            n_day = float(data.scene.N_day[s])
            # 对每个场景、每个时段累加 DN 的上调/下调可调功率，
            # 再乘代表性时段长度和该场景覆盖的天数。
            annual_ru_total += n_day * float(np.sum(result.DN.P_RU[y, s])) * float(data.delta_t_hour)
            annual_rd_total += n_day * float(np.sum(result.DN.P_RD[y, s])) * float(data.delta_t_hour)
            annual_ru_lack += n_day * float(np.sum(result.DN.P_RU_lack[y, s]))
            annual_rd_lack += n_day * float(np.sum(result.DN.P_RD_lack[y, s]))
            annual_demand = n_day * float(np.sum(result.DN.R_demand[y, s]))
            annual_ru_demand += annual_demand
            annual_rd_demand += annual_demand

        result.DN.P_RU_RD_total_year[y, 0] = annual_ru_total
        result.DN.P_RU_RD_total_year[y, 1] = annual_rd_total

        result.DN.RU_lack_total_year[y] = annual_ru_lack
        result.DN.RD_lack_total_year[y] = annual_rd_lack
        result.DN.RU_demand_total_year[y] = annual_ru_demand
        result.DN.RD_demand_total_year[y] = annual_rd_demand
        result.DN.P_RU_lack_rate_year[y] = (
            annual_ru_lack / annual_ru_demand if annual_ru_demand > 0 else 0.0
        )
        result.DN.P_RD_lack_rate_year[y] = (
            annual_rd_lack / annual_rd_demand if annual_rd_demand > 0 else 0.0
        )

    # =========================================================================
    # 3. 光伏 (PV) 结果
    # =========================================================================
    result.PV.P = {}
    result.PV.Q = {}
    result.PV.P_q = {}
    result.PV.P_all = {}
    result.PV.P_q_all = {}
    result.PV.W_scene = np.zeros((data.year, data.scene.N))
    result.PV.W_q_scene = np.zeros((data.year, data.scene.N))

    pv_w_year = np.zeros(data.year)
    pv_wq_year = np.zeros(data.year)
    pv_k_year = np.zeros(data.year)

    for y in range(data.year):
        sum1, sum2 = 0, 0
        for s in range(data.scene.N):
            result.PV.P[y, s] = np.zeros((data.period, data.DN.N_node))
            result.PV.Q[y, s] = np.zeros((data.period, data.DN.N_node))
            result.PV.P_q[y, s] = np.zeros((data.period, data.DN.N_node))
            result.PV.P_all[y, s] = np.zeros(data.period)
            result.PV.P_q_all[y, s] = np.zeros(data.period)

            for t in range(data.period):
                p_sum, pq_sum = 0, 0
                for i in range(data.DN.N_node):
                    p_val = get_val(var.PV.P[i, t, s, y])
                    q_val = get_val(var.PV.Q[i, t, s, y])
                    pq_val = get_val(var.PV.P_q[i, t, s, y])

                    result.PV.P[y, s][t, i] = p_val
                    result.PV.Q[y, s][t, i] = q_val
                    result.PV.P_q[y, s][t, i] = pq_val
                    p_sum += p_val
                    pq_sum += pq_val

                result.PV.P_all[y, s][t] = p_sum
                result.PV.P_q_all[y, s][t] = pq_sum

            w_scene = sum(result.PV.P_all[y, s])
            wq_scene = sum(result.PV.P_q_all[y, s])
            result.PV.W_scene[y, s] = w_scene
            result.PV.W_q_scene[y, s] = wq_scene

            sum1 += w_scene * data.scene.N_day[s]
            sum2 += wq_scene * data.scene.N_day[s]

        pv_w_year[y] = sum1
        pv_wq_year[y] = sum2
        pv_k_year[y] = sum1 / (sum1 + sum2)

    result.PV.W_year = pv_w_year
    result.PV.W_q_year = pv_wq_year
    result.PV.W_all = sum(pv_w_year)
    result.PV.W_q_all = sum(pv_wq_year)
    result.PV.k_year = pv_k_year

    # =========================================================================
    # 4. 储能 (ES) 结果
    # =========================================================================
    result.ES.S_plan = np.zeros((data.ES.N_place, data.year))
    result.ES.S_all = np.zeros((data.DN.N_node, data.year))
    result.ES.E_all = np.zeros((data.DN.N_node, data.year))

    for k in range(data.ES.N_place):
        for y in range(data.year):
            result.ES.S_plan[k, y] = get_val(var.ES.S_plan[k, y])
    for i in range(data.DN.N_node):
        for y in range(data.year):
            result.ES.S_all[i, y] = get_val(var.ES.S_all[i, y])
            result.ES.E_all[i, y] = get_val(var.ES.E_all[i, y])

    result.ES.P = {}
    result.ES.E = {}
    for y in range(data.year):
        for s in range(data.scene.N):
            result.ES.P[y, s] = np.zeros((data.period, data.DN.N_node))
            result.ES.E[y, s] = np.zeros((data.period, data.DN.N_node))
            for t in range(data.period):
                for i in range(data.DN.N_node):
                    result.ES.P[y, s][t, i] = get_val(var.ES.P[i, t, s, y])
                    result.ES.E[y, s][t, i] = get_val(var.ES.E[i, t, s, y])

    # =========================================================================
    # 5. 充换一体站 (BSS) & 电池 (SB)
    # =========================================================================
    for var_name in ['N_FP', 'N_FP_plan', 'N_FP_plan_e', 'N_FP_plan_u', 'N_SB', 'N_SB_plan', 'N_SB_plan_e', 'N_SB_plan_u', 'N_CB', 'N_CB_plan', 'N_CB_plan_e', 'N_CB_plan_u']:
        mat = np.zeros((data.BSS.N_station, data.year))
        src_var = getattr(var.BSS, var_name)
        for k in range(data.BSS.N_station):
            for y in range(data.year):
                mat[k, y] = get_val(src_var[k, y])
        setattr(result.BSS, var_name, mat)

        # 【补充部分】：提取 BSS 各项投资残值成本 (1D: year)
    for var_name in ['C_FP_inv', 'C_FP_res', 'C_SB_inv', 'C_SB_res', 'C_CB_inv', 'C_CB_res']:
        arr = np.zeros(data.year)
        src_var = getattr(var.BSS, var_name)
        for y in range(data.year):
            arr[y] = get_val(src_var[y])
        setattr(result.BSS, var_name, arr)

    result.BSS.N_SB_plan_all = np.sum(result.BSS.N_SB_plan)
    result.BSS.N_CB_plan_all = np.sum(result.BSS.N_CB_plan)

    # Annual total BSS charging energy, weighted by representative-day counts.
    # P_c is in kW; multiplying by delta_t_hour gives kWh per representative day.
    result.BSS.E_charge_year = np.zeros(data.year, dtype=float)
    for y in range(data.year):
        annual_charge = 0.0
        for s in range(data.scene.N):
            annual_charge += float(data.scene.N_day[s]) * sum(
                get_val(var.BSS.P_c[k, t, s, y])
                for k in range(data.BSS.N_station)
                for t in range(data.period)
            )
        result.BSS.E_charge_year[y] = annual_charge * float(data.delta_t_hour)

    # Annual BSS upward/downward flexibility energy, weighted by representative-day counts.
    result.BSS.E_RU_year = np.zeros(data.year, dtype=float)
    result.BSS.E_RD_year = np.zeros(data.year, dtype=float)
    for y in range(data.year):
        for s in range(data.scene.N):
            weight = float(data.scene.N_day[s]) * float(data.delta_t_hour)
            result.BSS.E_RU_year[y] += weight * sum(
                get_val(var.BSS.P_RU_all[t, s, y])
                for t in range(data.period)
            )
            result.BSS.E_RD_year[y] += weight * sum(
                get_val(var.BSS.P_RD_all[t, s, y])
                for t in range(data.period)
            )
    result.BSS.E_RU_total = float(np.sum(result.BSS.E_RU_year))
    result.BSS.E_RD_total = float(np.sum(result.BSS.E_RD_year))

    result.BSS.N_T = {}
    result.BSS.N_U = {}
    result.BSS.N_D = {}
    result.BSS.P = {}

    for y in range(data.year):
        for s in range(data.scene.N):
            result.BSS.N_T[y, s] = {}
            result.BSS.N_U[y, s] = {}
            result.BSS.N_D[y, s] = {}
            result.BSS.P[y, s] = np.zeros((data.period, data.BSS.N_station))

            for k in range(data.BSS.N_station):
                for t in range(data.period):
                    result.BSS.P[y, s][t, k] = get_val(var.BSS.P[k, t, s, y])

                    mat_T = np.zeros((data.BSS.N_SOC, data.BSS.N_SOC))
                    mat_U = np.zeros((data.BSS.N_SOC, data.BSS.N_SOC))
                    mat_D = np.zeros((data.BSS.N_SOC, data.BSS.N_SOC))
                    for r in range(data.BSS.N_SOC):
                        for p in range(data.BSS.N_SOC):
                            mat_T[r, p] = get_val(var.BSS.N_T[k, t, s, y, r, p])
                            mat_U[r, p] = get_val(var.BSS.N_U[k, t, s, y, r, p])
                            mat_D[r, p] = get_val(var.BSS.N_D[k, t, s, y, r, p])

                    result.BSS.N_T[y, s][k, t] = mat_T
                    result.BSS.N_U[y, s][k, t] = mat_U
                    result.BSS.N_D[y, s][k, t] = mat_D

    # =========================================================================
    # 6. 区域热网 (HN) 结果与双线性边界
    # =========================================================================
    hn_pipe_vars = ['y_pipe', 'b_ij', 'b_ji', 'N_pipe_exp', 'N_pipe_exp_e', 'N_pipe_exp_u', 'N_pipe']
    hn_node_vars = ['y_node', 'c']
    hn_station_vars = [
        'y_station', 'S_CHP', 'S_CHP_plan', 'S_CHP_plan_e', 'S_CHP_plan_u',
        'S_EB', 'S_EB_plan', 'S_EB_plan_e', 'S_EB_plan_u'
    ]
    hn_binary_or_integer = {'y_pipe', 'b_ij', 'b_ji', 'y_node', 'c', 'y_station', 'N_pipe_exp', 'N_pipe_exp_e', 'N_pipe_exp_u', 'N_pipe'}

    for var_name in hn_pipe_vars:
        if hasattr(var.HN, var_name):
            mat = np.zeros((data.HN.N_pipe, data.year))
            src_var = getattr(var.HN, var_name)
            for i in range(data.HN.N_pipe):
                for y in range(data.year):
                    val = get_val(src_var[i, y])
                    mat[i, y] = round(val) if var_name in hn_binary_or_integer else val
            setattr(result.HN, var_name, mat)

    for var_name in hn_node_vars:
        if hasattr(var.HN, var_name):
            mat = np.zeros((data.HN.N_node, data.year))
            src_var = getattr(var.HN, var_name)
            for i in range(data.HN.N_node):
                for y in range(data.year):
                    val = get_val(src_var[i, y])
                    mat[i, y] = round(val) if var_name in hn_binary_or_integer else val
            setattr(result.HN, var_name, mat)

    for var_name in hn_station_vars:
        if hasattr(var.HN, var_name):
            mat = np.zeros((data.HN.N_station, data.year))
            src_var = getattr(var.HN, var_name)
            for i in range(data.HN.N_station):
                for y in range(data.year):
                    val = get_val(src_var[i, y])
                    mat[i, y] = round(val) if var_name in hn_binary_or_integer else val
            setattr(result.HN, var_name, mat)

    result.HN.S_CHP_plan_all = np.sum(result.HN.S_CHP_plan) if hasattr(result.HN, 'S_CHP_plan') else 0
    result.HN.S_EB_plan_all = np.sum(result.HN.S_EB_plan) if hasattr(result.HN, 'S_EB_plan') else 0

    result.HN.H_node = {}
    result.HN.T_S_ex = {}
    result.HN.T_R_ex = {}
    result.HN.T_R_node = {}
    result.HN.FRin_TRn = {}
    result.HN.F_TRn = {}
    result.HN.F_TRex = {}
    result.HN.F_TSex = {}
    result.HN.P_RU_all = {}
    result.HN.P_RD_all = {}
    result.HN.P_RU_all1 = {}
    result.HN.P_RD_all1 = {}
    result.HN.E_RU_year = np.zeros(data.year, dtype=float)
    result.HN.E_RD_year = np.zeros(data.year, dtype=float)
    result.HN.E_RU_year1 = np.zeros(data.year, dtype=float)
    result.HN.E_RD_year1 = np.zeros(data.year, dtype=float)

    result.HN.T_S_max = np.zeros(data.year)
    result.HN.T_S_min = np.zeros(data.year)
    result.HN.T_R_max = np.zeros(data.year)
    result.HN.T_R_min = np.zeros(data.year)

    for y in range(data.year):
        max_ts, min_ts = -1e9, 1e9
        max_tr, min_tr = -1e9, 1e9
        has_ts, has_tr = False, False

        for s in range(data.scene.N):
            result.HN.H_node[y, s] = np.zeros((data.period, data.HN.N_node))
            result.HN.T_S_ex[y, s] = np.zeros((data.period, data.HN.N_node))
            result.HN.T_R_ex[y, s] = np.zeros((data.period, data.HN.N_node))
            result.HN.T_R_node[y, s] = np.zeros((data.period, data.HN.N_node))

            result.HN.F_TSex[y, s] = np.zeros((data.period, data.HN.N_node))
            result.HN.F_TRex[y, s] = np.zeros((data.period, data.HN.N_node))
            result.HN.F_TRn[y, s] = np.zeros((data.period, data.HN.N_node))
            result.HN.FRin_TRn[y, s] = np.zeros((data.period, data.HN.N_node))

            result.HN.P_RU_all[y, s] = np.zeros((data.period, 1))
            result.HN.P_RD_all[y, s] = np.zeros((data.period, 1))
            result.HN.P_RU_all1[y, s] = np.zeros((data.period, 1))
            result.HN.P_RD_all1[y, s] = np.zeros((data.period, 1))

            for t in range(data.period):
                ru_power = get_val(var.HN.P_RU_all[t, s, y])
                rd_power = get_val(var.HN.P_RD_all[t, s, y])
                result.HN.P_RU_all[y, s][t, 0] = ru_power
                result.HN.P_RD_all[y, s][t, 0] = rd_power
                result.HN.P_RU_all1[y, s][t, 0] = get_val(var.HN.P_RU_all1[t, s, y])
                result.HN.P_RD_all1[y, s][t, 0] = get_val(var.HN.P_RD_all1[t, s, y])

                weight = float(data.scene.N_day[s]) * float(data.delta_t_hour)
                result.HN.E_RU_year[y] += ru_power * weight
                result.HN.E_RD_year[y] += rd_power * weight
                result.HN.E_RU_year1[y] += result.HN.P_RU_all1[y, s][t, 0] * weight
                result.HN.E_RD_year1[y] += result.HN.P_RD_all1[y, s][t, 0] * weight

                for i in range(data.HN.N_node):
                    # 提取数值
                    result.HN.H_node[y, s][t, i] = get_val(var.HN.H_node[i, t, s, y])
                    v_ts = get_val(var.HN.T_S_ex[i, t, s, y])
                    v_tr = get_val(var.HN.T_R_ex[i, t, s, y])
                    result.HN.T_S_ex[y, s][t, i] = v_ts
                    result.HN.T_R_ex[y, s][t, i] = v_tr
                    result.HN.T_R_node[y, s][t, i] = get_val(var.HN.T_R_node[i, t, s, y])

                    result.HN.F_TSex[y, s][t, i] = get_val(var.HN.F_TSex[i, t, s, y])
                    result.HN.F_TRex[y, s][t, i] = get_val(var.HN.F_TRex[i, t, s, y])
                    result.HN.F_TRn[y, s][t, i] = get_val(var.HN.F_TRn[i, t, s, y])
                    result.HN.FRin_TRn[y, s][t, i] = get_val(var.HN.FRin_TRn[i, t, s, y])

                    # 寻找非零极值
                    if abs(v_ts) > 1e-6:
                        has_ts = True
                        if v_ts > max_ts: max_ts = v_ts
                        if v_ts < min_ts: min_ts = v_ts
                    if abs(v_tr) > 1e-6:
                        has_tr = True
                        if v_tr > max_tr: max_tr = v_tr
                        if v_tr < min_tr: min_tr = v_tr

        result.HN.T_S_max[y] = max_ts if has_ts else 0
        result.HN.T_S_min[y] = min_ts if has_ts else 0
        result.HN.T_R_max[y] = max_tr if has_tr else 0
        result.HN.T_R_min[y] = min_tr if has_tr else 0

    result.HN.E_RU_total = float(np.sum(result.HN.E_RU_year))
    result.HN.E_RD_total = float(np.sum(result.HN.E_RD_year))

    # 提取管道双线性项及边界
    result.HN.FR_bij_TRi = {}
    result.HN.FR_bji_TRj = {}
    result.HN.T_R_i = {}
    result.HN.T_R_j = {}
    for y in range(data.year):
        for s in range(data.scene.N):
            result.HN.FR_bij_TRi[y, s] = np.zeros((data.period, data.HN.N_pipe))
            result.HN.FR_bji_TRj[y, s] = np.zeros((data.period, data.HN.N_pipe))
            result.HN.T_R_i[y, s] = np.zeros((data.period, data.HN.N_pipe))
            result.HN.T_R_j[y, s] = np.zeros((data.period, data.HN.N_pipe))
            for t in range(data.period):
                for i in range(data.HN.N_pipe):
                    result.HN.FR_bij_TRi[y, s][t, i] = get_val(var.HN.FR_bij_TRi[i, t, s, y])
                    result.HN.FR_bji_TRj[y, s][t, i] = get_val(var.HN.FR_bji_TRj[i, t, s, y])
                    result.HN.T_R_i[y, s][t, i] = get_val(var.HN.T_R_i[i, t, s, y])
                    result.HN.T_R_j[y, s][t, i] = get_val(var.HN.T_R_j[i, t, s, y])

    # 提取静态网络流 (F_node, FR_bij等)
    result.HN.F_node = np.zeros((data.HN.N_node, data.year))
    result.HN.F_R_in = np.zeros((data.HN.N_node, data.year))
    result.HN.FR_bij = np.zeros((data.HN.N_pipe, data.year))
    result.HN.FR_bji = np.zeros((data.HN.N_pipe, data.year))

    for y in range(data.year):
        for i in range(data.HN.N_node):
            result.HN.F_node[i, y] = get_val(var.HN.F_node[i, y])
            result.HN.F_R_in[i, y] = get_val(var.HN.F_R_in[i, y])
        for i in range(data.HN.N_pipe):
            result.HN.FR_bij[i, y] = get_val(var.HN.FR_bij[i, y])
            result.HN.FR_bji[i, y] = get_val(var.HN.FR_bji[i, y])

    result.SBT.F_node = result.HN.F_node
    result.SBT.F_R_in = result.HN.F_R_in
    result.SBT.FR_bij = result.HN.FR_bij
    result.SBT.FR_bji = result.HN.FR_bji

    _export_source_zone_audit(data, var, result)
    _export_contract_tube_results(data, var, result)

    # =========================================================================
    # 7. CHP/EB baseline and regulation-call operating states
    # =========================================================================
    result.CHP.P_base = {}
    result.CHP.H_base = {}
    result.CHP.G_base = {}
    result.CHP.P_call = {}
    result.CHP.H_call = {}
    result.CHP.G_call = {}
    result.EB.P_base = {}
    result.EB.H_base = {}
    result.EB.P_call = {}
    result.EB.H_call = {}
    result.emission.CO_by_year = {}
    result.emission.SO_by_year = {}
    result.emission.NO_by_year = {}

    for y in range(data.year):
        for s in range(data.scene.N):
            shape = (data.period, data.HN.N_station)
            result.CHP.P_base[y, s] = np.zeros(shape)
            result.CHP.H_base[y, s] = np.zeros(shape)
            result.CHP.G_base[y, s] = np.zeros(shape)
            result.CHP.P_call[y, s] = np.zeros(shape)
            result.CHP.H_call[y, s] = np.zeros(shape)
            result.CHP.G_call[y, s] = np.zeros(shape)
            result.EB.P_base[y, s] = np.zeros(shape)
            result.EB.H_base[y, s] = np.zeros(shape)
            result.EB.P_call[y, s] = np.zeros(shape)
            result.EB.H_call[y, s] = np.zeros(shape)
            result.emission.CO_by_year[y, s] = np.zeros(shape)
            result.emission.SO_by_year[y, s] = np.zeros(shape)
            result.emission.NO_by_year[y, s] = np.zeros(shape)

            for t in range(data.period):
                for i in range(data.HN.N_station):
                    result.CHP.P_base[y, s][t, i] = get_val(var.CHP.P[i, t, s, y])
                    result.CHP.H_base[y, s][t, i] = get_val(var.CHP.H[i, t, s, y])
                    result.CHP.G_base[y, s][t, i] = get_val(var.CHP.G[i, t, s, y])
                    result.CHP.P_call[y, s][t, i] = get_val(var.CHP.P_call[i, t, s, y])
                    result.CHP.H_call[y, s][t, i] = get_val(var.CHP.H_call[i, t, s, y])
                    result.CHP.G_call[y, s][t, i] = get_val(var.CHP.G_call[i, t, s, y])
                    result.EB.P_base[y, s][t, i] = get_val(var.EB.P[i, t, s, y])
                    result.EB.H_base[y, s][t, i] = get_val(var.EB.H[i, t, s, y])
                    result.EB.P_call[y, s][t, i] = get_val(var.EB.P_call[i, t, s, y])
                    result.EB.H_call[y, s][t, i] = get_val(var.EB.H_call[i, t, s, y])
                    result.emission.CO_by_year[y, s][t, i] = get_val(var.emission.CO[i, t, s, y])
                    result.emission.SO_by_year[y, s][t, i] = get_val(var.emission.SO[i, t, s, y])
                    result.emission.NO_by_year[y, s][t, i] = get_val(var.emission.NO[i, t, s, y])

    # =========================================================================
    # 7. 排放与其他设备 (Emission, CHP, EB)
    # =========================================================================
    result.emission.CO = {}
    result.emission.SO = {}
    result.emission.NO = {}
    co_total, so_total, no_total = 0, 0, 0

    for s in range(data.scene.N):
        result.emission.CO[s] = np.zeros((data.period, data.HN.N_station))
        result.emission.SO[s] = np.zeros((data.period, data.HN.N_station))
        result.emission.NO[s] = np.zeros((data.period, data.HN.N_station))
        for t in range(data.period):
            for i in range(data.HN.N_station):
                # 原 MATLAB 这里的 emission 没有 year 维度
                co_val = get_val(var.emission.CO[i, t, s, 0])
                so_val = get_val(var.emission.SO[i, t, s, 0])
                no_val = get_val(var.emission.NO[i, t, s, 0])
                result.emission.CO[s][t, i] = co_val
                result.emission.SO[s][t, i] = so_val
                result.emission.NO[s][t, i] = no_val

                ndays = data.scene.N_day[s]
                co_total += co_val * ndays
                so_total += so_val * ndays
                no_total += no_val * ndays

    result.emission.CO_all = co_total
    result.emission.SO_all = so_total
    result.emission.NO_all = no_total

    # =========================================================================
    # 8. 方案记录 (Scheme Export)
    # =========================================================================
    scheme.DN.y_line = result.DN.y_line
    scheme.DN.b_ij = result.DN.b_ij
    scheme.DN.b_ji = result.DN.b_ji
    scheme.DN.c = result.DN.c
    scheme.DN.S_node_exp = result.DN.S_node_exp
    scheme.DN.N_line_exp = result.DN.N_line_exp

    scheme.ES.S_plan = result.ES.S_plan
    scheme.BSS.N_SB_plan = result.BSS.N_SB_plan
    scheme.BSS.N_CB_plan = result.BSS.N_CB_plan

    for name in [
        'y_pipe', 'b_ij', 'b_ji', 'c', 'S_CHP_plan', 'S_EB_plan', 'N_pipe_exp',
        'f_S', 'f_R', 'f_S_in', 'f_S_out', 'f_R_in', 'f_R_out',
        'F_S', 'F_R', 'F_S_in', 'F_S_out', 'F_R_in', 'F_R_out',
        'source_zone_affiliation', 'k_f_load', 'f_node', 'F_node',
        'fS_bij', 'fS_bji', 'fR_bij', 'fR_bji',
        'FS_bij', 'FS_bji', 'FR_bij', 'FR_bji'
    ]:
        if hasattr(result.HN, name):
            setattr(scheme.HN, name, getattr(result.HN, name))

    degradation_objective = getattr(obj, "C_BSS_deg", 0.0)
    degradation_variables = getattr(var.cost, "C_BSS_deg", None)
    result.cost.C_BSS_deg = get_val(degradation_objective)
    result.cost.C_BSS_deg_year = np.zeros(data.year)
    if degradation_variables is not None:
        for year in range(data.year):
            result.cost.C_BSS_deg_year[year] = get_val(degradation_variables[year])

    result.BSS.integer_tube = {
        'formulation': 'sparse_integer_charge_discharge',
        'charging_efficiency': float(data.BSS.eta_ch),
        'discharging_efficiency': float(data.BSS.eta_dis),
        'service_reserve_ratio': float(data.BSS.rho_srv),
        'degradation_cost': float(data.BSS.c_deg),
        'equivalent_cycle_life': float(data.BSS.equivalent_cycle_life),
        'reachable_charge_arc_count': len(data.BSS.charge_arcs),
        'reachable_discharge_arc_count': len(data.BSS.discharge_arcs),
    }
    return result, scheme


