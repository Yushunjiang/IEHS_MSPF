import gurobipy as gp
from gurobipy import GRB


class Struct:
    """Container for objective components."""
    pass


def object(model, data, var):
    """
    目标函数计算 (Python + Gurobipy 纯 for 循环实现)
    返回: obj (包含各项成本明细的对象)
    """
    obj = Struct()

    # =========================================================================
    # 1. 投资成本汇总
    # =========================================================================
    # 线路与变电站成本
    obj.C_line_inv = gp.quicksum(var.DN.C_line_inv[y] for y in range(data.year))
    obj.C_line_res = gp.quicksum(var.DN.C_line_res[y] for y in range(data.year))
    obj.C_node_inv = gp.quicksum(var.DN.C_node_inv[y] for y in range(data.year))
    obj.C_node_res = gp.quicksum(var.DN.C_node_res[y] for y in range(data.year))

    obj.C_DN_inv = obj.C_line_inv + obj.C_node_inv
    obj.C_DN_res = obj.C_line_res + obj.C_node_res

    # 充换一体站投资成本
    obj.C_BSS_inv = gp.quicksum(var.BSS.C_inv[y] for y in range(data.year))
    obj.C_BSS_res = gp.quicksum(var.BSS.C_res[y] for y in range(data.year))
    obj.C_SB_inv = gp.quicksum(var.BSS.C_SB_inv[y] for y in range(data.year))
    obj.C_SB_res = gp.quicksum(var.BSS.C_SB_res[y] for y in range(data.year))
    obj.C_CB_inv = gp.quicksum(var.BSS.C_CB_inv[y] for y in range(data.year))
    obj.C_CB_res = gp.quicksum(var.BSS.C_CB_res[y] for y in range(data.year))

    # 分布式储能投资成本
    obj.C_ES_inv = gp.quicksum(var.ES.C_inv[y] for y in range(data.year))
    obj.C_ES_res = gp.quicksum(var.ES.C_res[y] for y in range(data.year))

    # 光伏投资成本
    obj.C_PV_inv = gp.quicksum(var.PV.C_inv[y] for y in range(data.year))
    obj.C_PV_res = gp.quicksum(var.PV.C_res[y] for y in range(data.year))

    # 热网投资成本
    obj.C_pipe_inv = gp.quicksum(var.HN.C_pipe_inv[y] for y in range(data.year))
    obj.C_CHP_inv = gp.quicksum(var.HN.C_CHP_inv[y] for y in range(data.year))
    obj.C_EB_inv = gp.quicksum(var.HN.C_EB_inv[y] for y in range(data.year))
    obj.C_pipe_res = gp.quicksum(var.HN.C_pipe_res[y] for y in range(data.year))
    obj.C_CHP_res = gp.quicksum(var.HN.C_CHP_res[y] for y in range(data.year))
    obj.C_EB_res = gp.quicksum(var.HN.C_EB_res[y] for y in range(data.year))

    obj.C_HN_inv = obj.C_pipe_inv + obj.C_CHP_inv + obj.C_EB_inv
    obj.C_HN_res = obj.C_pipe_res + obj.C_CHP_res + obj.C_EB_res

    # 多主体规划成本：IEHS 负责电网、热网和储能，BSS 负责换电站。
    obj.C_IEHS_plan_inv = (
        obj.C_DN_inv + obj.C_HN_inv + obj.C_ES_inv
    )
    obj.C_IEHS_plan_res = (
        obj.C_DN_res + obj.C_HN_res + obj.C_ES_res
    )
    obj.C_IEHS_plan = obj.C_IEHS_plan_inv - obj.C_IEHS_plan_res
    obj.C_IEHS_line_inv = obj.C_line_inv
    obj.C_IEHS_line_res = obj.C_line_res
    obj.C_IEHS_node_inv = obj.C_node_inv
    obj.C_IEHS_node_res = obj.C_node_res
    obj.C_IEHS_pipe_inv = obj.C_pipe_inv
    obj.C_IEHS_pipe_res = obj.C_pipe_res
    obj.C_IEHS_CHP_inv = obj.C_CHP_inv
    obj.C_IEHS_CHP_res = obj.C_CHP_res
    obj.C_IEHS_EB_inv = obj.C_EB_inv
    obj.C_IEHS_EB_res = obj.C_EB_res
    obj.C_IEHS_ES_inv = obj.C_ES_inv
    obj.C_IEHS_ES_res = obj.C_ES_res
    obj.C_BSS_plan_inv = obj.C_BSS_inv
    obj.C_BSS_plan_res = obj.C_BSS_res
    obj.C_BSS_plan = obj.C_BSS_plan_inv - obj.C_BSS_plan_res

    # 总投资与残值
    obj.C_inv = obj.C_DN_inv + obj.C_ES_inv + obj.C_HN_inv + obj.C_BSS_inv
    obj.C_res = obj.C_DN_res + obj.C_ES_res + obj.C_HN_res + obj.C_BSS_res

    # 添加年度投资约束
    for y in range(data.year):
        model.addConstr(var.cost.C_exp[y] == var.DN.C_inv[y] + var.BSS.C_inv[y] + var.ES.C_inv[y] + var.HN.C_inv[y], name=f"Cost_C_exp_{y}")
        model.addConstr(var.cost.C_res[y] == var.DN.C_res[y] + var.BSS.C_res[y] + var.ES.C_res[y] + var.HN.C_res[y], name=f"Cost_C_res_{y}")
        model.addConstr(var.cost.C_inv[y] == var.cost.C_exp[y] - var.cost.C_res[y], name=f"Cost_C_inv_{y}")

    # =========================================================================
    # 2. 运行成本计算
    # =========================================================================
    obj.C_ele = 0  # 购电成本
    obj.C_gas = 0  # 购气成本
    obj.C_emi = 0  # 污染物排放惩罚成本

    # 提取常量
    r_val = float(data.r)

    for y in range(data.year):
        sum1_y = gp.LinExpr()
        sum2_y = gp.LinExpr()
        sum3_y = gp.LinExpr()
        r_discount = (1 + r_val) ** (y + 1)

        for s in range(data.scene.N):
            n_day = float(data.scene.N_day[s])

            for t in range(data.period):
                c_ele = float(data.cost.c_ele[t])
                c_gas = float(data.cost.c_gas[t])

                # 购电成本
                p_sub_sum = gp.quicksum(var.DN.P_sub[i, t, s, y] for i in range(data.DN.N_sub))
                sum1_y += p_sub_sum * c_ele * n_day / r_discount

                # Called CHP fuel consumption is the physical gas use in the
                # regulation-call operating state.
                chp_g_sum = gp.quicksum(var.CHP.G_call[i, t, s, y] for i in range(data.HN.N_station))
                sum2_y += chp_g_sum * c_gas * n_day / r_discount

                # Pollutant-treatment cost uses emissions produced in the called state.
                so_sum = gp.quicksum(var.emission.SO[i, t, s, y] for i in range(data.HN.N_station))
                no_sum = gp.quicksum(var.emission.NO[i, t, s, y] for i in range(data.HN.N_station))
                sum3_y += (so_sum * float(data.cost.c_SO) + no_sum * float(data.cost.c_NO)) * n_day / r_discount

        model.addConstr(var.cost.C_ele[y] == sum1_y, name=f"Cost_ele_def_{y}")
        model.addConstr(var.cost.C_gas[y] == sum2_y, name=f"Cost_gas_def_{y}")
        model.addConstr(var.cost.C_emi[y] == sum3_y, name=f"Cost_emi_def_{y}")
        model.addConstr(var.cost.C_ope[y] == sum1_y + sum2_y + sum3_y, name=f"Cost_ope_def_{y}")

        obj.C_ele += sum1_y
        obj.C_gas += sum2_y
        obj.C_emi += sum3_y

    obj.C_ope = obj.C_ele + obj.C_gas + obj.C_emi

    # BSS 与 IEHS 的购售电结算。BSS.P_DN 为 BSS 从配电网的净购电功率，
    # 与购电电价、典型日权重和规划期折现保持与 C_ele 相同的口径。
    obj.C_BSS_buy = 0
    for y in range(data.year):
        bss_buy_y = gp.LinExpr()
        r_discount = (1 + r_val) ** (y + 1)
        for s in range(data.scene.N):
            n_day = float(data.scene.N_day[s])
            for t in range(data.period):
                bss_power = gp.quicksum(
                    var.BSS.P_DN[i, t, s, y] for i in range(data.DN.N_node)
                )
                bss_buy_y += bss_power * float(data.cost.c_ele[t]) * n_day / r_discount
        obj.C_BSS_buy += bss_buy_y

    obj.C_IEHS_sale = obj.C_BSS_buy
    obj.C_IEHS_ope = obj.C_ele + obj.C_gas + obj.C_emi - obj.C_IEHS_sale
    obj.C_BSS_ope = obj.C_BSS_buy

    # DHN operating cost: gas consumed by CHP, net electricity purchased by
    # EB after CHP generation, and pollutant treatment cost. These are kept
    # as a separate accounting view and are not added to obj.C_ope again.
    obj.C_DHN_gas = 0
    obj.C_DHN_ele = 0
    obj.C_DHN_emi = 0

    for y in range(data.year):
        dhn_gas_y = gp.LinExpr()
        dhn_ele_y = gp.LinExpr()
        dhn_emi_y = gp.LinExpr()
        r_discount = (1 + r_val) ** (y + 1)

        for s in range(data.scene.N):
            n_day = float(data.scene.N_day[s])
            for t in range(data.period):
                c_ele = float(data.cost.c_ele[t])
                c_gas = float(data.cost.c_gas[t])

                chp_g_sum = gp.quicksum(
                    var.CHP.G_call[i, t, s, y] for i in range(data.HN.N_station)
                )
                dhn_net_purchase = gp.quicksum(
                    var.EB.P_call[i, t, s, y] - var.CHP.P_call[i, t, s, y]
                    for i in range(data.HN.N_station)
                )
                so_sum = gp.quicksum(
                    var.emission.SO[i, t, s, y] for i in range(data.HN.N_station)
                )
                no_sum = gp.quicksum(
                    var.emission.NO[i, t, s, y] for i in range(data.HN.N_station)
                )

                dhn_gas_y += chp_g_sum * c_gas * n_day / r_discount
                dhn_ele_y += dhn_net_purchase * c_ele * n_day / r_discount
                dhn_emi_y += (so_sum * float(data.cost.c_SO) + no_sum * float(data.cost.c_NO)) * n_day / r_discount

        obj.C_DHN_gas += dhn_gas_y
        obj.C_DHN_ele += dhn_ele_y
        obj.C_DHN_emi += dhn_emi_y

    obj.C_DHN_ope = obj.C_DHN_gas + obj.C_DHN_ele + obj.C_DHN_emi
    # =========================================================================
    # 3. 风险成本计算 (弃光与调节能力不足)
    # =========================================================================
    obj.C_PV = 0  # 弃光惩罚成本
    obj.C_lack = 0  # 调节能力不足惩罚成本
    obj.C_regulation = 0

    c_pv_q = float(data.cost.c_PV_q)
    c_lack = float(data.cost.c_lack)

    for y in range(data.year):
        sum1_y = gp.LinExpr()
        sum2_y = gp.LinExpr()
        sum3_y = gp.LinExpr()
        r_discount = (1 + r_val) ** (y + 1)

        for s in range(data.scene.N):
            n_day = float(data.scene.N_day[s])
            for t in range(data.period):
                # 弃光惩罚成本
                p_q_sum = gp.quicksum(var.PV.P_q[i, t, s, y] for i in range(data.DN.N_node))
                sum1_y += p_q_sum * c_pv_q * n_day / r_discount

                # 调节能力不足惩罚成本
                sum2_y += (var.DN.P_RU_lack[t, s, y] + var.DN.P_RD_lack[t, s, y]) * c_lack * n_day / r_discount

                # 调节能力不足惩罚成本
                sum3_y += (var.DN.P_RU[t, s, y] + var.DN.P_RD[t, s, y]) * 0.01 * n_day / r_discount

        model.addConstr(var.cost.C_PV[y] == sum1_y, name=f"Cost_PV_risk_def_{y}")
        model.addConstr(var.cost.C_lack[y] == sum2_y, name=f"Cost_lack_def_{y}")
        model.addConstr(var.cost.C_risk[y] == sum1_y + sum2_y, name=f"Cost_risk_def_{y}")

        obj.C_PV += sum1_y
        obj.C_lack += sum2_y
        obj.C_regulation += sum3_y

    obj.C_risk = obj.C_lack + obj.C_PV

    # =========================================================================
    # 4. 设置优化目标 F
    # =========================================================================
    obj.F = obj.C_inv + obj.C_ope + obj.C_lack + obj.C_PV - obj.C_res

    return obj
