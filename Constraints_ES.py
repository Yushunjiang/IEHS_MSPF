import gurobipy as gp
from gurobipy import GRB
import numpy as np


def Constraints_ES(model, data, var, scheme=None):
    """
    储能电站 (ES) 规划与运行约束 (Python + Gurobipy 纯 for 循环实现)
    """

    # 获取需要用到的标量常数
    r = float(data.r)
    big_M = float(data.M)
    ES_k = float(data.ES.k)
    c_inv = float(data.ES.c_inv)
    T_build = int(data.ES.T_build)
    T_life = int(data.ES.T_life)

    # =========================================================================
    # 1. 规划约束 (容量建设、退役与年限)
    # =========================================================================
    for k in range(data.ES.N_place):
        for y in range(data.year):
            # 规划容量 = 规划数量 * 单台步长
            model.addConstr(var.ES.S_plan[k, y] == var.ES.N_plan[k, y] * float(data.ES.S_ref), name=f"ES_S_plan_{k}_{y}")
            # 能量 = 容量 * 2 (充放电倍率设定)
            model.addConstr(var.ES.E_plan[k, y] == var.ES.S_plan[k, y] * 2.0, name=f"ES_E_plan_{k}_{y}")
            model.addConstr(var.ES.E_plan_u[k, y] == var.ES.S_plan_u[k, y] * 2.0, name=f"ES_E_plan_u_{k}_{y}")

            # 储能建设年限约束 (临近寿命末期不再新建)
            if y >= data.year - T_build:
                model.addConstr(var.ES.S_plan[k, y] == 0, name=f"ES_S_plan_end_{k}_{y}")

            # 储能建设完成容量
            if y >= T_build:
                model.addConstr(var.ES.S_plan_e[k, y] == var.ES.S_plan[k, y - T_build], name=f"ES_S_plan_e_{k}_{y}")
            else:
                model.addConstr(var.ES.S_plan_e[k, y] == 0, name=f"ES_S_plan_e_0_{k}_{y}")

            # 储能退役容量
            if y >= T_life:
                model.addConstr(var.ES.S_plan_r[k, y] == var.ES.S_plan_e[k, y - T_life], name=f"ES_S_plan_r_{k}_{y}")
            else:
                model.addConstr(var.ES.S_plan_r[k, y] == 0, name=f"ES_S_plan_r_0_{k}_{y}")

            # 储能建设完成可用容量 (历年投入 - 历年退役)
            exp_e_sum = gp.quicksum(var.ES.S_plan_e[k, yy] for yy in range(y + 1))
            exp_r_sum = gp.quicksum(var.ES.S_plan_r[k, yy] for yy in range(y + 1))
            model.addConstr(var.ES.S_plan_u[k, y] == exp_e_sum - exp_r_sum, name=f"ES_S_plan_u_{k}_{y}")

    # =========================================================================
    # 2. 规划后储能情况与投资成本
    # =========================================================================
    for y in range(data.year):
        # 储能总容量约束 <= 光伏总容量的 10%
        if hasattr(data.PV, 'S'):
            pv_s_sum = float(np.sum(data.PV.S[:, y]))
            es_s_all_sum = gp.quicksum(var.ES.S_all[i, y] for i in range(data.DN.N_node))
            model.addConstr(es_s_all_sum <= pv_s_sum * 0.1, name=f"ES_Total_Cap_Limit_{y}")

        # 将储能规划容量映射到配电网具体节点上
        for i in range(data.DN.N_node):
            if data.DN.node[i] in data.ES.place:
                idx = list(data.ES.place).index(data.DN.node[i])
                model.addConstr(var.ES.S_all[i, y] == float(data.ES.S[i]) + var.ES.S_plan_u[idx, y], name=f"ES_S_all_build_{i}_{y}")
                model.addConstr(var.ES.E_all[i, y] == float(data.ES.E[i]) + var.ES.E_plan_u[idx, y], name=f"ES_E_all_build_{i}_{y}")
            else:
                model.addConstr(var.ES.S_all[i, y] == float(data.ES.S[i]), name=f"ES_S_all_exist_{i}_{y}")
                model.addConstr(var.ES.E_all[i, y] == float(data.ES.E[i]), name=f"ES_E_all_exist_{i}_{y}")

        # 投资成本计算
        e_plan_sum = gp.quicksum(var.ES.E_plan[k, y] for k in range(data.ES.N_place))
        model.addConstr(var.ES.C_inv[y] == c_inv * e_plan_sum / ((1 + r) ** (y + 1)), name=f"ES_C_inv_{y}")

        # 残值计算
        if y + 1 >= data.year - T_life - T_build:
            res_val = (var.ES.C_inv[y] - (data.year - (y + 1)) * var.ES.C_inv[y] / T_life) / ((1 + r) ** data.year)
            model.addConstr(var.ES.C_res[y] == res_val, name=f"ES_C_res_{y}")
        else:
            model.addConstr(var.ES.C_res[y] == 0, name=f"ES_C_res_0_{y}")

    # =========================================================================
    # 3. 运行约束与可调能力约束 (纯 for 循环完全替代 ndgrid 向量化)
    # =========================================================================
    for y in range(data.year):
        for s in range(data.scene.N):
            for t in range(data.period):
                # 汇总当前时刻总的上调/下调能力
                model.addConstr(var.ES.P_RU_all[t, s, y] == gp.quicksum(var.ES.P_RU[k, t, s, y] for k in range(data.DN.N_node)), name=f"ES_PRU_all_{t}_{s}_{y}")
                model.addConstr(var.ES.P_RD_all[t, s, y] == gp.quicksum(var.ES.P_RD[k, t, s, y] for k in range(data.DN.N_node)), name=f"ES_PRD_all_{t}_{s}_{y}")

                for k in range(data.DN.N_node):
                    # a. 充放状态互斥
                    model.addConstr(var.ES.X_c[k, t, s, y] + var.ES.X_f[k, t, s, y] <= 1, name=f"ES_MutEx_{k}_{t}_{s}_{y}")

                    # b. 充放电功率边界约束 (大 M 松弛)
                    model.addConstr(var.ES.P_c[k, t, s, y] <= var.ES.X_c[k, t, s, y] * big_M, name=f"ES_Pc_M_{k}_{t}_{s}_{y}")
                    model.addConstr(var.ES.P_f[k, t, s, y] <= var.ES.X_f[k, t, s, y] * big_M, name=f"ES_Pf_M_{k}_{t}_{s}_{y}")

                    # c. 充放电功率受总容量约束
                    model.addConstr(var.ES.P_c[k, t, s, y] <= var.ES.S_all[k, y], name=f"ES_Pc_S_{k}_{t}_{s}_{y}")
                    model.addConstr(var.ES.P_f[k, t, s, y] <= var.ES.S_all[k, y], name=f"ES_Pf_S_{k}_{t}_{s}_{y}")

                    # d. 净功率
                    model.addConstr(var.ES.P[k, t, s, y] == var.ES.P_c[k, t, s, y] - var.ES.P_f[k, t, s, y], name=f"ES_P_net_{k}_{t}_{s}_{y}")

                    # e. 电量与功率连续性关系 (首尾相连)
                    t_prev = t - 1 if t > 0 else data.period - 1
                    model.addConstr(var.ES.E[k, t, s, y] == var.ES.E[k, t_prev, s, y] + var.ES.P_c[k, t, s, y] * ES_k - var.ES.P_f[k, t, s, y] / ES_k, name=f"ES_E_Balance_{k}_{t}_{s}_{y}")

                    # f. 电量上下限约束
                    model.addConstr(var.ES.E[k, t, s, y] <= var.ES.E_all[k, y], name=f"ES_E_max_{k}_{t}_{s}_{y}")
                    # 下限 E >= 0 已在 addVars 时通过 lb=0 处理

                    # 负荷侧口径：RU 增加耗电功率，RD 减少耗电功率。
                    # RU 对应增加充电或减少放电，终点为 +S_all。
                    model.addConstr(
                        var.ES.P_RU[k, t, s, y]
                        <= var.ES.S_all[k, y] - var.ES.P[k, t, s, y],
                        name=f"ES_PRU_limit1_{k}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.ES.P_RU[k, t, s, y]
                        <= (var.ES.E_all[k, y] - var.ES.E[k, t, s, y]) / ES_k
                        - var.ES.P[k, t, s, y],
                        name=f"ES_PRU_limit2_{k}_{t}_{s}_{y}",
                    )

                    # RD 对应减少充电或增加放电，终点为 -S_all。
                    model.addConstr(
                        var.ES.P_RD[k, t, s, y]
                        <= var.ES.S_all[k, y] + var.ES.P[k, t, s, y],
                        name=f"ES_PRD_limit1_{k}_{t}_{s}_{y}",
                    )
                    model.addConstr(
                        var.ES.P_RD[k, t, s, y]
                        <= var.ES.E[k, t, s, y] * ES_k
                        + var.ES.P[k, t, s, y],
                        name=f"ES_PRD_limit2_{k}_{t}_{s}_{y}",
                    )
