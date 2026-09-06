import gurobipy as gp
from gurobipy import GRB
import math


def Constraints_PV(model, data, var, scheme=None):
    # =========================================================================
    # 1. 规划约束
    # =========================================================================
    for k in range(data.PV.N_place):
        for y in range(data.year):
            # 规划容量 = 规划数量 * 单台步长
            model.addConstr(var.PV.S_plan[k, y] == var.PV.N_plan[k, y] * data.PV.S_ref, name=f"PV_S_plan_calc_{k}_{y}")

            model.addConstr(var.PV.N_plan[k, y] == 0, name=f"PV_N_plan_zero_{k}_{y}")

            # 建设容量上限约束
            model.addConstr(var.PV.S_plan_u[k, y] <= float(data.PV.S_place[k]), name=f"PV_S_plan_u_limit_{k}_{y}")

            # 建设完成容量 (考虑建设周期)
            if y >= data.PV.T_build:
                model.addConstr(var.PV.S_plan_e[k, y] == var.PV.S_plan[k, y - data.PV.T_build], name=f"PV_S_plan_e_{k}_{y}")
            else:
                model.addConstr(var.PV.S_plan_e[k, y] == 0, name=f"PV_S_plan_e_0_{k}_{y}")

            # 建设完成可用容量 (历年投入之和，原代码 PV 没有退役逻辑)
            exp_e_sum = gp.quicksum(var.PV.S_plan_e[k, yy] for yy in range(y + 1))
            model.addConstr(var.PV.S_plan_u[k, y] == exp_e_sum, name=f"PV_S_plan_u_calc_{k}_{y}")

    # 将候选位置的规划容量映射到配电网所有节点
    for i in range(data.DN.N_node):
        for y in range(data.year):
            if data.DN.node[i] in data.PV.place:
                idx = list(data.PV.place).index(data.DN.node[i])
                model.addConstr(var.PV.S_all[i, y] == float(data.PV.S[i, y]) + var.PV.S_plan_u[idx, y], name=f"PV_S_all_build_{i}_{y}")
            else:
                model.addConstr(var.PV.S_all[i, y] == float(data.PV.S[i, y]), name=f"PV_S_all_exist_{i}_{y}")

    # =========================================================================
    # 2. 投资成本与残值计算
    # =========================================================================
    for y in range(data.year):
        # 投资成本计算
        s_plan_sum = gp.quicksum(var.PV.S_plan[k, y] for k in range(data.PV.N_place))
        model.addConstr(var.PV.C_inv[y] == data.PV.c_inv * s_plan_sum / ((1 + data.r) ** (y + 1)), name=f"PV_C_inv_{y}")

        # 残值计算
        res_val = (var.PV.C_inv[y] - (data.year - (y + 1)) * var.PV.C_inv[y] / data.PV.T_life) / ((1 + data.r) ** data.year)
        model.addConstr(var.PV.C_res[y] == res_val, name=f"PV_C_res_{y}")

    # =========================================================================
    # 3. 运行约束 (纯 for 循环完全替代矩阵操作)
    # =========================================================================
    for i in range(data.DN.N_node):
        for y in range(data.year):
            for s in range(data.scene.N):
                for t in range(data.period):
                    # 获取当前时刻的光伏出力系数 (标量)
                    k_pv_val = float(data.scene.k_PV[t, s])

                    # 计算当前时空点的理论出力上限表达式
                    upper_bound = var.PV.S_all[i, y] * k_pv_val

                    # a. 实际有功出力上限约束
                    model.addConstr(var.PV.P[i, t, s, y] <= upper_bound, name=f"PV_P_ub_{i}_{t}_{s}_{y}")

                    # b. 弃光功率 = 理论出力上限 - 实际出力
                    model.addConstr(var.PV.P_q[i, t, s, y] == upper_bound - var.PV.P[i, t, s, y], name=f"PV_P_q_calc_{i}_{t}_{s}_{y}")

                    # c. 无功功率 = 有功出力 * 恒定功率因数的对应系数
                    model.addConstr(var.PV.Q[i, t, s, y] == var.PV.P[i, t, s, y] * math.sin(math.acos(data.PV.k)), name=f"PV_Q_calc_{i}_{t}_{s}_{y}")

                    # d. var.PV.P >= 0 的非负约束已经在 Data_read.py 的 model.addVars(..., lb=0) 中底层保证，无需重复添加
