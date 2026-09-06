import gurobipy as gp
from gurobipy import GRB


def Constraints_DN(model, data, var, scheme=None, include_bss_reserve=True):
    # =========================================================================
    # 0. 预设方案约束 (如果传入了 scheme，将其变量固定)
    # =========================================================================
    if scheme is not None and hasattr(scheme, 'DN'):
        for y in range(data.year):
            for i in range(data.DN.N_line):
                model.addConstr(var.DN.y_line[i, y] == scheme.DN.y_line[i, y], name=f"Scheme_y_line_{i}_{y}")
                model.addConstr(var.DN.b_ij[i, y] == scheme.DN.b_ij[i, y], name=f"Scheme_b_ij_{i}_{y}")
                model.addConstr(var.DN.b_ji[i, y] == scheme.DN.b_ji[i, y], name=f"Scheme_b_ji_{i}_{y}")

    # =========================================================================
    # 1. 线路扩容规划
    # =========================================================================
    for i in range(data.DN.N_line):
        for y in range(data.year):
            # 线路扩容完成倍数 (考虑建设周期)
            if y >= data.line.T_build:
                model.addConstr(var.DN.N_line_exp_e[i, y] == var.DN.N_line_exp[i, y - int(data.line.T_build)], name=f"DN_Line_Exp_E_{i}_{y}")
            else:
                model.addConstr(var.DN.N_line_exp_e[i, y] == 0, name=f"DN_Line_Exp_E_zero_{i}_{y}")

            # 线路扩容退役倍数 (考虑寿命)
            if y >= data.line.T_life:
                model.addConstr(var.DN.N_line_exp_r[i, y] == var.DN.N_line_exp_e[i, y - int(data.line.T_life)], name=f"DN_Line_Exp_R_{i}_{y}")
            else:
                model.addConstr(var.DN.N_line_exp_r[i, y] == 0, name=f"DN_Line_Exp_R_zero_{i}_{y}")

            # 线路扩容完成可用倍数 (历年完成之和 - 历年退役之和)
            exp_e_sum = gp.quicksum(var.DN.N_line_exp_e[i, yy] for yy in range(y + 1))
            exp_r_sum = gp.quicksum(var.DN.N_line_exp_r[i, yy] for yy in range(y + 1))
            model.addConstr(var.DN.N_line_exp_u[i, y] == exp_e_sum - exp_r_sum, name=f"DN_Line_Exp_U_{i}_{y}")

            # 总线路倍数 = 初始倍数 + 扩容可用倍数
            model.addConstr(var.DN.N_line[i, y] == data.DN.N_line_initial[i] + var.DN.N_line_exp_u[i, y], name=f"DN_N_line_Total_{i}_{y}")

            # 扩容后线路实际容量
            model.addConstr(var.DN.S_line[i, y] == var.DN.N_line[i, y] * data.line.S_ref, name=f"DN_S_line_{i}_{y}")

    # =========================================================================
    # 2. 变电站扩容规划
    # =========================================================================
    for i in range(data.DN.N_node):
        for y in range(data.year):
            # 变电站扩容容量 = 倍数 * 参考步长
            model.addConstr(var.DN.S_node_exp[i, y] == var.DN.N_node_exp[i, y] * data.sub.S_ref, name=f"DN_S_node_exp_{i}_{y}")

            # 扩容完成
            if y >= data.sub.T_build:
                model.addConstr(var.DN.N_node_exp_e[i, y] == var.DN.N_node_exp[i, y - int(data.sub.T_build)], name=f"DN_Node_Exp_E_{i}_{y}")
            else:
                model.addConstr(var.DN.N_node_exp_e[i, y] == 0, name=f"DN_Node_Exp_E_zero_{i}_{y}")

            # 扩容退役
            if y >= data.sub.T_life:
                model.addConstr(var.DN.N_node_exp_r[i, y] == var.DN.N_node_exp_e[i, y - int(data.sub.T_life)], name=f"DN_Node_Exp_R_{i}_{y}")
            else:
                model.addConstr(var.DN.N_node_exp_r[i, y] == 0, name=f"DN_Node_Exp_R_zero_{i}_{y}")

            # 可用时间/倍数
            exp_e_sum = gp.quicksum(var.DN.N_node_exp_e[i, yy] for yy in range(y + 1))
            exp_r_sum = gp.quicksum(var.DN.N_node_exp_r[i, yy] for yy in range(y + 1))
            model.addConstr(var.DN.N_node_exp_u[i, y] == exp_e_sum - exp_r_sum, name=f"DN_Node_Exp_U_{i}_{y}")

            # 扩容后变电站总容量
            model.addConstr(var.DN.S_node[i, y] == data.DN.S_node[i] + var.DN.N_node_exp_u[i, y] * data.sub.S_ref, name=f"DN_S_node_Total_{i}_{y}")

    # =========================================================================
    # 3. 规划成本与残值计算 (考虑时间价值贴现)
    # =========================================================================
    for y in range(data.year):
        # 线路投资成本
        line_inv_cost = gp.quicksum(data.line.c_inv * var.DN.N_line_exp[i, y] * data.DN.L_line[i] for i in range(data.DN.N_line))
        model.addConstr(var.DN.C_line_inv[y] == line_inv_cost / ((1 + data.r) ** (y + 1)), name=f"DN_C_line_inv_{y}")

        # 线路残值
        if y + 1 >= data.year - data.line.T_life - data.line.T_build:
            res_val = (var.DN.C_line_inv[y] - (data.year - (y + 1)) * var.DN.C_line_inv[y] / data.line.T_life) / ((1 + data.r) ** data.year)
            model.addConstr(var.DN.C_line_res[y] == res_val, name=f"DN_C_line_res_{y}")
        else:
            model.addConstr(var.DN.C_line_res[y] == 0, name=f"DN_C_line_res_zero_{y}")

        # 变电站投资成本
        node_inv_cost = data.sub.c_inv * gp.quicksum(var.DN.S_node_exp[i, y] for i in range(data.DN.N_node))
        model.addConstr(var.DN.C_node_inv[y] == node_inv_cost / ((1 + data.r) ** (y + 1)), name=f"DN_C_node_inv_{y}")

        # 变电站残值
        if y + 1 >= data.year - data.sub.T_build - data.sub.T_life:
            res_val = (var.DN.C_node_inv[y] - (data.year - (y + 1)) * var.DN.C_node_inv[y] / data.sub.T_life) / ((1 + data.r) ** data.year)
            model.addConstr(var.DN.C_node_res[y] == res_val, name=f"DN_C_node_res_{y}")
        else:
            model.addConstr(var.DN.C_node_res[y] == 0, name=f"DN_C_node_res_zero_{y}")

        # 总成本汇总
        model.addConstr(var.DN.C_inv[y] == var.DN.C_line_inv[y] + var.DN.C_node_inv[y], name=f"DN_C_inv_total_{y}")
        model.addConstr(var.DN.C_res[y] == var.DN.C_line_res[y] + var.DN.C_node_res[y], name=f"DN_C_res_total_{y}")

    # =========================================================================
    # 4. 配电网拓扑约束 (生成树)
    # =========================================================================
    for y in range(data.year):
        for i in range(data.DN.N_line):
            # 存在线路才能有潮流方向
            model.addConstr(var.DN.y_line[i, y] <= var.DN.N_line[i, y], name=f"DN_y_line_limit_{i}_{y}")
            model.addConstr(var.DN.b_ij[i, y] + var.DN.b_ji[i, y] == var.DN.y_line[i, y], name=f"DN_b_ij_ji_sum_{i}_{y}")

        for i in range(data.DN.N_node):
            # 父节点数量 = 入射树枝之和
            tail_sum = gp.quicksum(var.DN.b_ij[l, y] for l in data.DN.set_tail[i])
            head_sum = gp.quicksum(var.DN.b_ji[l, y] for l in data.DN.set_head[i])
            model.addConstr(var.DN.c[i, y] == tail_sum + head_sum, name=f"DN_c_calc_{i}_{y}")

            # 变电站为根节点(c=0)，其余负荷节点需与网络连通(c >= node_state)
            # data.DN.sub uses the same node numbering as data.DN.node from Excel/Matlab.
            if data.DN.node[i] in data.DN.sub:
                model.addConstr(var.DN.c[i, y] == 0, name=f"DN_c_sub_{i}_{y}")
            else:
                model.addConstr(var.DN.c[i, y] >= data.DN.node_state[i, y], name=f"DN_c_load_{i}_{y}")

    # =========================================================================
    # 5. 运行约束: 负荷分配与节点功率平衡
    # =========================================================================
    for y in range(data.year):
        for s in range(data.scene.N):
            for t in range(data.period):
                # a. 基础负荷定义
                for i in range(data.DN.N_node):
                    model.addConstr(var.DN.P_load[i, t, s, y] == data.DN.P_load[i] * data.DN.P_year[y] * data.DN.node_state[i, y] * data.scene.k_E[t, s], name=f"DN_P_load_calc_{i}_{t}_{s}_{y}")

                # b. 节点功率平衡 (基尔霍夫电流定律)
                for i in range(data.DN.N_node):
                    P_tail_sum = gp.quicksum(var.DN.P_line[l, t, s, y] for l in data.DN.set_tail[i])
                    P_head_sum = gp.quicksum(var.DN.P_line[l, t, s, y] for l in data.DN.set_head[i])
                    model.addConstr(P_tail_sum + var.DN.P_node[i, t, s, y] == P_head_sum, name=f"DN_P_Balance_{i}_{t}_{s}_{y}")

                    # c. 节点交互功率组成
                    if data.DN.node[i] in data.DN.sub:
                        j = list(data.DN.sub).index(data.DN.node[i])  # 找到该编号在 sub 列表中的确切位置
                        model.addConstr(var.DN.P_node[i, t, s, y] == var.DN.P_sub[j, t, s, y], name=f"DN_P_node_sub_{i}_{t}_{s}_{y}")
                        # 变电站容量上限 (不能倒送上级电网时下限为0)
                        model.addConstr(var.DN.P_node[i, t, s, y] >= 0, name=f"DN_P_sub_lb_{i}_{t}_{s}_{y}")
                        model.addConstr(var.DN.P_node[i, t, s, y] <= var.DN.S_node[i, y] * data.DN.k_sub, name=f"DN_P_sub_ub_{i}_{t}_{s}_{y}")
                    else:
                        model.addConstr(var.DN.P_node[i, t, s, y] == -var.DN.P_load[i, t, s, y] + var.PV.P[i, t, s, y] - var.ES.P[i, t, s, y] - var.HN.P_DN[i, t, s, y] - var.BSS.P_DN[i, t, s, y], name=f"DN_P_node_dist_{i}_{t}_{s}_{y}")
                        # 负荷节点变压器容量限制 (考虑允许一定比例的功率倒送 k_back)
                        model.addConstr(var.DN.P_node[i, t, s, y] >= -var.DN.S_node[i, y] * data.DN.k_sub, name=f"DN_P_node_lb_{i}_{t}_{s}_{y}")
                        model.addConstr(var.DN.P_node[i, t, s, y] <= var.DN.S_node[i, y] * data.DN.k_sub, name=f"DN_P_node_ub_{i}_{t}_{s}_{y}")
                        model.addConstr(var.DN.P_node[i, t, s, y] <= var.DN.S_node[i, y] * data.DN.k_back, name=f"DN_P_node_backfeed_{i}_{t}_{s}_{y}")

                # d. 线路容量约束
                for i in range(data.DN.N_line):
                    model.addConstr(var.DN.P_line[i, t, s, y] <= var.DN.S_line[i, y], name=f"DN_P_line_ub_{i}_{t}_{s}_{y}")
                    model.addConstr(var.DN.P_line[i, t, s, y] >= -var.DN.S_line[i, y], name=f"DN_P_line_lb_{i}_{t}_{s}_{y}")

                # =========================================================================
                # 6. 系统可调节能力约束 (上/下备用与大 M 法松弛)
                # =========================================================================
                # 调节需求 = r_load * 总负荷 + r_PV * 总光伏
                P_load_sum = gp.quicksum(var.DN.P_load[i, t, s, y] for i in range(data.DN.N_node))
                P_PV_sum = gp.quicksum(var.PV.P[i, t, s, y] for i in range(data.DN.N_node))

                # 【修改点】强制转换为 float 标量，避免 numpy 数组广播导致 Gurobi 无法识别
                r_load_scalar = float(data.DN.r_load)
                r_PV_scalar = float(data.DN.r_PV)

                model.addConstr(var.DN.R_demand[t, s, y] == r_load_scalar * P_load_sum + r_PV_scalar * P_PV_sum, name=f"DN_R_demand_{t}_{s}_{y}")

                # DN uses nominal reserve capability. Call coefficients are
                # intentionally applied only inside the HN and BSS physical
                # response formulations, not in the grid reserve accounting.
                reserve_ru = var.BSS.P_RU_all[t, s, y] if include_bss_reserve else 0
                model.addConstr(
                    var.DN.P_RU[t, s, y]
                    == var.ES.P_RU_all[t, s, y]
                    + var.HN.P_RU_all[t, s, y]
                    + reserve_ru,
                    name=f"DN_P_RU_{t}_{s}_{y}",
                )

                # 上调能力不足量计算 R_u_lack = max{R_demand - P_RU, 0}
                model.addConstr(var.DN.P_RU_lack[t, s, y] >= var.DN.R_demand[t, s, y] - var.DN.P_RU[t, s, y], name=f"DN_RU_lack_1_{t}_{s}_{y}")
                model.addConstr(var.DN.P_RU_lack[t, s, y] >= 0, name=f"DN_RU_lack_2_{t}_{s}_{y}")
                model.addConstr(var.DN.P_RU_lack[t, s, y] <= var.DN.R_demand[t, s, y] - var.DN.P_RU[t, s, y] + var.DN.u_RU_lack[t, s, y] * data.M, name=f"DN_RU_lack_3_{t}_{s}_{y}")
                model.addConstr(var.DN.P_RU_lack[t, s, y] <= (1 - var.DN.u_RU_lack[t, s, y]) * data.M, name=f"DN_RU_lack_4_{t}_{s}_{y}")

                # DN uses nominal downward reserve capability as well.
                reserve_rd = var.BSS.P_RD_all[t, s, y] if include_bss_reserve else 0
                model.addConstr(
                    var.DN.P_RD[t, s, y]
                    == var.ES.P_RD_all[t, s, y]
                    + var.HN.P_RD_all[t, s, y]
                    + reserve_rd,
                    name=f"DN_P_RD_{t}_{s}_{y}",
                )
                # 下调能力不足量计算 R_d_lack = max{R_demand - P_RD, 0}
                model.addConstr(var.DN.P_RD_lack[t, s, y] >= var.DN.R_demand[t, s, y] - var.DN.P_RD[t, s, y], name=f"DN_RD_lack_1_{t}_{s}_{y}")
                model.addConstr(var.DN.P_RD_lack[t, s, y] >= 0, name=f"DN_RD_lack_2_{t}_{s}_{y}")
                model.addConstr(var.DN.P_RD_lack[t, s, y] <= var.DN.R_demand[t, s, y] - var.DN.P_RD[t, s, y] + var.DN.u_RD_lack[t, s, y] * data.M, name=f"DN_RD_lack_3_{t}_{s}_{y}")
                model.addConstr(var.DN.P_RD_lack[t, s, y] <= (1 - var.DN.u_RD_lack[t, s, y]) * data.M, name=f"DN_RD_lack_4_{t}_{s}_{y}")

    return model
