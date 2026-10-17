import gurobipy as gp
from gurobipy import GRB
import math


def Constraints_WT(model, data, var, scheme=None):
    # 提前提取标量参数，避免放入循环重复计算及 Numpy 类型冲突
    r = float(data.r)
    S_ref = float(data.WT.S_ref)
    c_inv = float(data.WT.c_inv)
    T_build = int(data.WT.T_build)
    T_life = int(data.WT.T_life)
    WT_k = float(data.WT.k)
    sin_acos_k = math.sin(math.acos(WT_k))

    # =========================================================================
    # 1. 规划约束
    # =========================================================================
    for k in range(data.WT.N_place):
        for y in range(data.year):
            # 规划容量 = 规划数量 * 单台步长
            model.addConstr(var.WT.S_plan[k, y] == var.WT.N_plan[k, y] * S_ref, name=f"WT_S_plan_calc_{k}_{y}")

            # 原 MATLAB 代码中有一句 var.WT.N_plan == 0，代表此配置下不新增光伏
            model.addConstr(var.WT.N_plan[k, y] == 0, name=f"WT_N_plan_zero_{k}_{y}")

            # 建设容量上限约束
            model.addConstr(var.WT.S_plan_u[k, y] <= float(data.WT.S_place[k]), name=f"WT_S_plan_u_limit_{k}_{y}")

            # 建设完成容量 (考虑建设周期)
            if y >= T_build:
                model.addConstr(var.WT.S_plan_e[k, y] == var.WT.S_plan[k, y - T_build], name=f"WT_S_plan_e_{k}_{y}")
            else:
                model.addConstr(var.WT.S_plan_e[k, y] == 0, name=f"WT_S_plan_e_0_{k}_{y}")

            # 建设完成可用容量 (历年投入之和，原代码 WT 没有退役逻辑)
            exp_e_sum = gp.quicksum(var.WT.S_plan_e[k, yy] for yy in range(y + 1))
            model.addConstr(var.WT.S_plan_u[k, y] == exp_e_sum, name=f"WT_S_plan_u_calc_{k}_{y}")

    # 将候选位置的规划容量映射到配电网所有节点
    for i in range(data.DN.N_node):
        for y in range(data.year):
            if data.DN.node[i] in data.WT.place:
                idx = list(data.WT.place).index(data.DN.node[i])
                model.addConstr(var.WT.S_all[i, y] == float(data.WT.S[i, y]) + var.WT.S_plan_u[idx, y], name=f"WT_S_all_build_{i}_{y}")
            else:
                model.addConstr(var.WT.S_all[i, y] == float(data.WT.S[i, y]), name=f"WT_S_all_exist_{i}_{y}")

    # =========================================================================
    # 2. 投资成本与残值计算
    # =========================================================================
    for y in range(data.year):
        # 投资成本计算
        s_plan_sum = gp.quicksum(var.WT.S_plan[k, y] for k in range(data.WT.N_place))
        model.addConstr(var.WT.C_inv[y] == c_inv * s_plan_sum / ((1 + r) ** (y + 1)), name=f"WT_C_inv_{y}")

        # 残值计算
        if y >= data.year - T_life - T_build:
            res_val = (var.WT.C_inv[y] - (data.year - (y + 1)) * var.WT.C_inv[y] / T_life) / ((1 + r) ** data.year)
            model.addConstr(var.WT.C_res[y] == res_val, name=f"WT_C_res_{y}")
        else:
            model.addConstr(var.WT.C_res[y] == 0, name=f"WT_C_res_0_{y}")

    # =========================================================================
    # 3. 运行约束 (纯 for 循环完全替代矩阵操作)
    # =========================================================================
    for i in range(data.DN.N_node):
        for y in range(data.year):
            for s in range(data.scene.N):
                for t in range(data.period):
                    # 获取当前时刻的光伏出力系数 (标量)
                    k_WT_val = float(data.scene.k_WT[t, s])

                    # 计算当前时空点的理论出力上限表达式
                    upper_bound = var.WT.S_all[i, y] * k_WT_val

                    # a. 实际有功出力上限约束
                    model.addConstr(var.WT.P[i, t, s, y] <= upper_bound, name=f"WT_P_ub_{i}_{t}_{s}_{y}")

                    # b. 弃光功率 = 理论出力上限 - 实际出力
                    model.addConstr(var.WT.P_q[i, t, s, y] == upper_bound - var.WT.P[i, t, s, y], name=f"WT_P_q_calc_{i}_{t}_{s}_{y}")

                    # c. 无功功率 = 有功出力 * 恒定功率因数的对应系数
                    model.addConstr(var.WT.Q[i, t, s, y] == var.WT.P[i, t, s, y] * sin_acos_k, name=f"WT_Q_calc_{i}_{t}_{s}_{y}")

                    # d. var.WT.P >= 0 的非负约束已经在 Data_read.py 的 model.addVars(..., lb=0) 中底层保证，无需重复添加