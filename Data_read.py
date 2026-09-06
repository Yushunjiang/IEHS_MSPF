import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from openpyxl.utils.cell import range_boundaries
import math
import os
import json
from functools import lru_cache


DIRECT_PLANNING_YEARS = 3
SBT_PLANNING_YEARS = 10
DEFAULT_PLANNING_YEARS = DIRECT_PLANNING_YEARS
DEFAULT_BSS_CHARGING_EFFICIENCY = 0.95
DEFAULT_BSS_DISCHARGING_EFFICIENCY = 0.95
DEFAULT_BSS_SOC_MIN = 0.0
DEFAULT_BSS_SOC_MAX = 1.0
DEFAULT_BSS_SERVICE_RESERVE_RATIO = 0.0
DEFAULT_BSS_EQUIVALENT_CYCLE_LIFE = 3000.0

class Struct:

    pass


@lru_cache(maxsize=32)
def _read_excel_sheet(filepath, sheet_name):
    """Read one workbook sheet once per Python process.

    Data_read requests many disjoint ranges from the same sheets.  The old
    implementation reopened and reparsed the workbook for every range.
    Returning the cached DataFrame is safe because callers only slice it.
    """
    return pd.read_excel(filepath, sheet_name=sheet_name, header=None)


@lru_cache(maxsize=8)
def _read_json_parameter_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=128)
def _excel_range_bounds(cell_range):
    return range_boundaries(cell_range)


def clear_excel_read_cache():
    """Clear cached Excel sheets, useful when a workbook is edited in-process."""
    _read_excel_sheet.cache_clear()
    _read_json_parameter_file.cache_clear()
    _excel_range_bounds.cache_clear()


def read_excel_range(filepath, sheet_name, cell_range):
    """
    优雅地替代 MATLAB 的 xlsread('file', 'sheet', 'range')
    利用 openpyxl 解析范围，例如 "B4:U27" -> 提取对应数据并返回 numpy 数组
    """
    filepath = os.fspath(filepath)
    if str(filepath).lower().endswith(".json"):
        parameter_data = _read_json_parameter_file(filepath)
        try:
            values = parameter_data["sheets"][sheet_name][cell_range]
        except KeyError as error:
            raise KeyError(
                f"Parameter JSON does not contain sheet/range: "
                f"{sheet_name!r}/{cell_range!r}"
            ) from error
        return np.asarray(
            [[np.nan if value is None else value for value in row] for row in values]
        )
    df = _read_excel_sheet(filepath, sheet_name)
    min_col, min_row, max_col, max_row = _excel_range_bounds(cell_range)
    # 转换为 0-based 索引并切片
    # 注意：使用 dropna 剔除读取大范围（如 A3:H1000）时的末尾空行
    data_slice = df.iloc[min_row - 1:max_row, min_col - 1:max_col].dropna(how='all').values
    return data_slice


def configure_bss_integer_data(data):
    sb = data.BSS
    sb.eta_ch = float(
        os.environ.get("BSS_CHARGING_EFFICIENCY", DEFAULT_BSS_CHARGING_EFFICIENCY)
    )
    sb.eta_dis = float(
        os.environ.get("BSS_DISCHARGING_EFFICIENCY", DEFAULT_BSS_DISCHARGING_EFFICIENCY)
    )
    sb.soc_min = float(os.environ.get("BSS_SOC_MIN", DEFAULT_BSS_SOC_MIN))
    sb.soc_max = float(os.environ.get("BSS_SOC_MAX", DEFAULT_BSS_SOC_MAX))
    sb.rho_srv = float(
        os.environ.get("BSS_SERVICE_RESERVE_RATIO", DEFAULT_BSS_SERVICE_RESERVE_RATIO)
    )
    sb.equivalent_cycle_life = float(
        os.environ.get("BSS_EQUIVALENT_CYCLE_LIFE", DEFAULT_BSS_EQUIVALENT_CYCLE_LIFE)
    )

    if not 0 < sb.eta_ch <= 1:
        raise ValueError("BSS charging efficiency must be in (0, 1].")
    if not 0 < sb.eta_dis <= 1:
        raise ValueError("BSS discharging efficiency must be in (0, 1].")
    if not 0 <= sb.soc_min < sb.soc_max <= 1:
        raise ValueError("BSS SOC bounds must satisfy 0 <= soc_min < soc_max <= 1.")
    if sb.rho_srv < 0:
        raise ValueError("BSS service reserve ratio must be nonnegative.")
    if sb.equivalent_cycle_life <= 0:
        raise ValueError("BSS equivalent cycle life must be positive.")

    sb.delta_E = (
        float(sb.E)
        * (sb.soc_max - sb.soc_min)
        / max(1, int(sb.N_SOC) - 1)
    )
    sb.one_period_full_charge_required_power = (
        float(sb.E)
        * (sb.soc_max - sb.soc_min)
        / sb.eta_ch
        / float(data.delta_t_hour)
    )
    sb.configured_slot_power = float(data.CB.P)
    data.CB.P = max(
        float(data.CB.P),
        float(sb.one_period_full_charge_required_power),
    )
    sb.effective_slot_power = float(data.CB.P)
    direct_degradation_cost = os.environ.get("BSS_DEGRADATION_COST")
    if direct_degradation_cost is not None:
        sb.c_deg = float(direct_degradation_cost)
    else:
        usable_battery_energy = float(sb.E) * (sb.soc_max - sb.soc_min)
        sb.c_deg = float(sb.c_inv) / (
            2.0 * sb.equivalent_cycle_life * usable_battery_energy
        )
    if sb.c_deg < 0:
        raise ValueError("BSS degradation cost must be nonnegative.")

    sb.minimum_soc_index = int(
        math.ceil(sb.soc_min * (int(sb.N_SOC) - 1) - 1e-9)
    )
    sb.maximum_soc_index = int(
        math.floor(sb.soc_max * (int(sb.N_SOC) - 1) + 1e-9)
    )
    sb.charge_arcs = []
    sb.discharge_arcs = []
    sb.grid_power_coefficient = {}
    sb.grid_discharge_power_coefficient = {}
    slot_power = float(data.CB.P)
    delta_t = float(data.delta_t_hour)

    for source_soc in range(sb.minimum_soc_index, sb.maximum_soc_index + 1):
        for target_soc in range(source_soc + 1, sb.maximum_soc_index + 1):
            battery_energy = (target_soc - source_soc) * sb.delta_E
            grid_power = battery_energy / sb.eta_ch / delta_t
            if grid_power <= slot_power + 1e-9:
                arc = (source_soc, target_soc)
                sb.charge_arcs.append(arc)
                sb.grid_power_coefficient[arc] = grid_power

        for target_soc in range(sb.minimum_soc_index, source_soc):
            battery_energy = (source_soc - target_soc) * sb.delta_E
            grid_power = battery_energy * sb.eta_dis / delta_t
            if grid_power <= slot_power + 1e-9:
                arc = (source_soc, target_soc)
                sb.discharge_arcs.append(arc)
                sb.grid_discharge_power_coefficient[arc] = grid_power

    if not sb.charge_arcs:
        raise ValueError(
            "No reachable BSS SOC charging arc exists under the configured slot power."
        )
    full_charge_arc = (sb.minimum_soc_index, sb.maximum_soc_index)
    if full_charge_arc not in sb.charge_arcs:
        raise ValueError(
            "BSS one-period full-charge configuration is invalid: "
            f"required arc {full_charge_arc} is unavailable with "
            f"slot power {data.CB.P}."
        )
    if not sb.discharge_arcs:
        raise ValueError(
            "No reachable BSS SOC discharging arc exists under the configured slot power."
        )

    sb.charge_arc_set = set(sb.charge_arcs)
    sb.discharge_arc_set = set(sb.discharge_arcs)
    sb.transition_arcs = sb.charge_arcs + sb.discharge_arcs
    sb.out_arcs = {
        soc: [arc for arc in sb.charge_arcs if arc[0] == soc]
        for soc in range(int(sb.N_SOC))
    }
    sb.in_arcs = {
        soc: [arc for arc in sb.charge_arcs if arc[1] == soc]
        for soc in range(int(sb.N_SOC))
    }
    sb.discharge_out_arcs = {
        soc: [arc for arc in sb.discharge_arcs if arc[0] == soc]
        for soc in range(int(sb.N_SOC))
    }
    sb.discharge_in_arcs = {
        soc: [arc for arc in sb.discharge_arcs if arc[1] == soc]
        for soc in range(int(sb.N_SOC))
    }
    sb.transition_out_arcs = {
        soc: [arc for arc in sb.transition_arcs if arc[0] == soc]
        for soc in range(int(sb.N_SOC))
    }
    sb.transition_in_arcs = {
        soc: [arc for arc in sb.transition_arcs if arc[1] == soc]
        for soc in range(int(sb.N_SOC))
    }

    sb.R_srv = {}
    top_soc = sb.maximum_soc_index
    for station in range(data.BSS.N_station):
        for time in range(data.period):
            next_time = (time + 1) % data.period
            for scene in range(data.scene.N):
                for year in range(data.year):
                    next_demand = int(
                        round(sb.N_out[station, next_time, scene, year, top_soc])
                    )
                    sb.R_srv[station, time, scene, year] = int(
                        math.ceil(sb.rho_srv * next_demand)
                    )
    return sb



def _create_hn_mccormick_variables(model, data, var):
    T_pw = int(data.HN.T_pw)
    F_pw = int(data.HN.F_pw)
    total_pw = T_pw * F_pw
    
    # 管道分段变量
    # 1. 基础分段连续变量
    var.HN.T_R_i_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, T_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_i_pw")
    var.HN.T_R_j_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, T_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_j_pw")
    var.HN.FR_bij_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, F_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bij_pw")
    var.HN.FR_bji_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, F_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bji_pw")
    
    # 2. 分段 0-1 标志变量
    var.HN.u_TRi_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, T_pw, vtype=GRB.BINARY, name="HN_u_TRi_pw")
    var.HN.u_TRj_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, T_pw, vtype=GRB.BINARY, name="HN_u_TRj_pw")
    var.HN.u_FRij_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, F_pw, vtype=GRB.BINARY, name="HN_u_FRij_pw")
    var.HN.u_FRji_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, F_pw, vtype=GRB.BINARY, name="HN_u_FRji_pw")
    
    # 3. 乘积项的分段连续变量与辅助求和变量 (维度为 total_pw)
    var.HN.FR_bij_TRi_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bij_TRi_pw")
    var.HN.FR_bji_TRj_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bji_TRj_pw")
    var.HN.FR_bij_TRi_help = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bij_TRi_help")
    var.HN.FR_bji_TRj_help = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bji_TRj_help")
    
    # 4. 乘积项的 0-1 安德门标志变量
    var.HN.u_FRij_TRi_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, total_pw, vtype=GRB.BINARY, name="HN_u_FRij_TRi_pw")
    var.HN.u_FRji_TRj_pw = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, total_pw, vtype=GRB.BINARY, name="HN_u_FRji_TRj_pw")
    
    # 节点分段变量
    # 1. 基础分段连续变量
    var.HN.T_S_ex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, T_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_S_ex_pw")
    var.HN.T_R_ex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, T_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_ex_pw")
    var.HN.T_R_node_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, T_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_node_pw")
    var.HN.F_node_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, F_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_node_pw")
    var.HN.F_R_in_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, F_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_R_in_pw")
    
    # 2. 分段 0-1 标志变量
    var.HN.u_TSex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, T_pw, vtype=GRB.BINARY, name="HN_u_TSex_pw")
    var.HN.u_TRex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, T_pw, vtype=GRB.BINARY, name="HN_u_TRex_pw")
    var.HN.u_TRn_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, T_pw, vtype=GRB.BINARY, name="HN_u_TRn_pw")
    var.HN.u_Fn_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, F_pw, vtype=GRB.BINARY, name="HN_u_Fn_pw")
    var.HN.u_FRin_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, F_pw, vtype=GRB.BINARY, name="HN_u_FRin_pw")
    
    # 3. 乘积项的分段连续变量与辅助求和变量 (维度为 total_pw)
    var.HN.F_TSex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TSex_pw")
    var.HN.F_TRex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TRex_pw")
    var.HN.FRin_TRn_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FRin_TRn_pw")
    
    var.HN.F_TSex_help = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TSex_help")
    var.HN.F_TRex_help = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TRex_help")
    var.HN.FRin_TRn_help = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FRin_TRn_help")
    
    # 4. 乘积项的 0-1 安德门标志变量
    var.HN.u_F_TSex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, vtype=GRB.BINARY, name="HN_u_F_TSex_pw")
    var.HN.u_F_TRex_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, vtype=GRB.BINARY, name="HN_u_F_TRex_pw")
    var.HN.u_FRin_TRn_pw = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, total_pw, vtype=GRB.BINARY, name="HN_u_FRin_TRn_pw")



def _create_hn_exact_linear_variables(model, data, var):
    path_index = data.HN.path_index
    var.HN.z_aff_TSex = model.addVars(
        data.HN.N_load, data.HN.N_station, data.period, data.scene.N, data.year,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_aff_TSex'
    )
    var.HN.z_aff_TRex = model.addVars(
        data.HN.N_load, data.HN.N_station, data.period, data.scene.N, data.year,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_aff_TRex'
    )
    var.HN.z_aff_TRn = model.addVars(
        data.HN.N_load, data.HN.N_station, data.period, data.scene.N, data.year,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_aff_TRn'
    )
    var.HN.z_path_ij_TRi = model.addVars(
        path_index, data.period, data.scene.N,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_path_ij_TRi'
    )
    var.HN.z_path_ji_TRj = model.addVars(
        path_index, data.period, data.scene.N,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_path_ji_TRj'
    )
    var.HN.z_rin_ij_TRn = model.addVars(
        path_index, data.period, data.scene.N,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_rin_ij_TRn'
    )
    var.HN.z_rin_ji_TRn = model.addVars(
        path_index, data.period, data.scene.N,
        lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='HN_z_rin_ji_TRn'
    )
    
        # =====================================================================================================================================================================
        # 热网设备 (CHP, GB, EB) 参数读取与变量定义

def _configure_hn_mccormick_parameters(data, hn_segment_count):
    segment_count = hn_segment_count
    if segment_count is None:
        segment_count = int(os.environ.get("MCCORMICK_SEGMENTS", "4"))
    segment_count = int(segment_count)
    if segment_count < 1:
        raise ValueError("hn_segment_count must be a positive integer")

    hn = data.HN
    hn.T_pw = segment_count
    hn.F_pw = segment_count
    hn.T_S_delta = (float(hn.T_S_max) - float(hn.T_S_min)) / segment_count
    hn.T_R_delta = (float(hn.T_R_max) - float(hn.T_R_min)) / segment_count
    hn.T_S_min_pw = np.array([hn.T_S_min + k * hn.T_S_delta for k in range(segment_count)], dtype=float)
    hn.T_S_max_pw = np.array([hn.T_S_min + (k + 1) * hn.T_S_delta for k in range(segment_count)], dtype=float)
    hn.T_R_min_pw = np.array([hn.T_R_min + k * hn.T_R_delta for k in range(segment_count)], dtype=float)
    hn.T_R_max_pw = np.array([hn.T_R_min + (k + 1) * hn.T_R_delta for k in range(segment_count)], dtype=float)

    def global_bounds(lower, upper):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        return lower.min(axis=1), upper.max(axis=1)

    pipe_bij_min, pipe_bij_max = global_bounds(hn.FR_bij_min, hn.FR_bij_max)
    pipe_bji_min, pipe_bji_max = global_bounds(hn.FR_bji_min, hn.FR_bji_max)
    node_flow_min, node_flow_max = global_bounds(hn.F_node_min, hn.F_node_max)
    inlet_flow_min, inlet_flow_max = global_bounds(hn.FRin_min, hn.FRin_max)

    hn.FR_bij_delta = (pipe_bij_max - pipe_bij_min) / segment_count
    hn.FR_bji_delta = (pipe_bji_max - pipe_bji_min) / segment_count
    hn.F_node_delta = (node_flow_max - node_flow_min) / segment_count
    hn.FRin_delta = (inlet_flow_max - inlet_flow_min) / segment_count

    hn.FR_bij_min_pw = np.zeros((hn.N_pipe, segment_count), dtype=float)
    hn.FR_bij_max_pw = np.zeros((hn.N_pipe, segment_count), dtype=float)
    hn.FR_bji_min_pw = np.zeros((hn.N_pipe, segment_count), dtype=float)
    hn.FR_bji_max_pw = np.zeros((hn.N_pipe, segment_count), dtype=float)
    hn.F_node_min_pw = np.zeros((hn.N_node, segment_count), dtype=float)
    hn.F_node_max_pw = np.zeros((hn.N_node, segment_count), dtype=float)
    hn.FRin_min_pw = np.zeros((hn.N_node, segment_count), dtype=float)
    hn.FRin_max_pw = np.zeros((hn.N_node, segment_count), dtype=float)
    for k in range(segment_count):
        hn.FR_bij_min_pw[:, k] = pipe_bij_min + k * hn.FR_bij_delta
        hn.FR_bij_max_pw[:, k] = pipe_bij_min + (k + 1) * hn.FR_bij_delta
        hn.FR_bji_min_pw[:, k] = pipe_bji_min + k * hn.FR_bji_delta
        hn.FR_bji_max_pw[:, k] = pipe_bji_min + (k + 1) * hn.FR_bji_delta
        hn.F_node_min_pw[:, k] = node_flow_min + k * hn.F_node_delta
        hn.F_node_max_pw[:, k] = node_flow_min + (k + 1) * hn.F_node_delta
        hn.FRin_min_pw[:, k] = inlet_flow_min + k * hn.FRin_delta
        hn.FRin_max_pw[:, k] = inlet_flow_min + (k + 1) * hn.FRin_delta

    hn.ra_partitions = {}
    hn.mccormick_envelopes_enabled = True
    hn.piecewise_partitions_enabled = segment_count > 1

def Data_read(
    model,
    filepath='parameters.json',
    planning_years=DEFAULT_PLANNING_YEARS,
    create_dense_bss_transitions=True,
    hn_linearization_mode='mccormick',
    hn_segment_count=None,
):
    valid_hn_linearization_modes = {'mccormick', 'physical_exact', 'direct_bilinear'}
    if hn_linearization_mode not in valid_hn_linearization_modes:
        raise ValueError(
            f"hn_linearization_mode must be one of {sorted(valid_hn_linearization_modes)}, "
            f"got {hn_linearization_mode!r}"
        )
    data = Struct()
    var = Struct()

    data.scene = Struct()
    data.cost = Struct()
    data.DN = Struct()
    data.HN = Struct()
    data.HN.linearization_mode = hn_linearization_mode
    data.PV = Struct()
    data.ES = Struct()
    data.EV = Struct()
    data.BSS = Struct()

    var.cost = Struct()
    var.DN = Struct()
    var.HN = Struct()
    var.PV = Struct()
    var.ES = Struct()
    var.EV = Struct()
    var.BSS = Struct()
    # =========================================================================
    # 全局与场景参数
    # =========================================================================
    data.r = 0.05  # 贴现率
    data.M = 5e9  # 大M
    data.year = max(1, int(planning_years))  # 规划周期
    data.period = 24  # 运行周期
    data.period_delta = 24 / data.period
    data.delta_t_hour = data.period_delta
    # 多阶段规划源荷发展情况
    data.DN.P_year = np.array([1 + y * 0.03 for y in range(data.year)])
    data.HN.H_year = np.array([1 + y * 0.05 for y in range(data.year)])
    data.PV.S_year = np.array([1 + y * 0.1 for y in range(data.year)])
    data.EV.N_year = np.array([1 + y * 0.1 for y in range(data.year)])

    # =========================================================================
    # 场景参数
    # =========================================================================
    data.scene.N = 3
    data.scene.N_day = np.array([360 / data.scene.N] * data.scene.N)

    # Explicit 24-hour regulation-call magnitudes. Change these four values
    # here when testing another call coefficient; hourly directions below are
    # intentionally kept unchanged.
    data.scene.HN_RU_call_coefficient = 0
    data.scene.HN_RD_call_coefficient = 0
    data.scene.BSS_TU_call_coefficient = 0
    data.scene.BSS_TD_call_coefficient = 0
    k_E_all = read_excel_range(filepath, '场景', 'B4:U27')
    k_H_all = read_excel_range(filepath, '场景', 'B32:U55')
    k_PV_all = read_excel_range(filepath, '场景', 'B60:U83')

    data.scene.k_E = k_E_all[:, :data.scene.N]
    data.scene.k_H = k_H_all[:, :data.scene.N]
    data.scene.k_PV = k_PV_all[:, :data.scene.N]

    # =========================================================================
    # 价格参数
    # =========================================================================
    data.cost.c_ele = read_excel_range(filepath, '价格', 'B3:B26').flatten()
    data.cost.c_gas = read_excel_range(filepath, '价格', 'C3:C26').flatten()
    data.cost.c_SO = 2
    data.cost.c_NO = 6
    data.cost.c_PV_q = 1
    data.cost.c_lack = 3
    # 价格变量
    var.cost.C_inv = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_inv")
    var.cost.C_exp = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_exp")
    var.cost.C_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_res")
    var.cost.C_ope = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_ope")
    var.cost.C_risk = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_risk")
    var.cost.C_ele = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_ele")
    var.cost.C_gas = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_gas")
    var.cost.C_emi = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_emi")
    var.cost.C_PV = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_PV")
    var.cost.C_lack = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_lack")

    # =========================================================================
    # 电力网参数
    # =========================================================================
    data.DN.U = 10
    data.DN.U_max = 1.1 * data.DN.U
    data.DN.U_min = 0.9 * data.DN.U

    Table1 = read_excel_range(filepath, '电网拓扑', 'A3:H1000')
    # 注意：Python 索引从 0 开始。为方便网络拓扑关联，节点编号建议全部减 1 变为 0-based！
    data.DN.line = (Table1[:, 0]).astype(int)
    data.DN.head = (Table1[:, 1]).astype(int)
    data.DN.tail = (Table1[:, 2]).astype(int)
    data.DN.R = Table1[:, 3]
    data.DN.X = Table1[:, 4]
    data.DN.L_line = Table1[:, 5]
    data.DN.N_line_initial = Table1[:, 6]
    data.DN.S_line = Table1[:, 7]

    Table2 = read_excel_range(filepath, '电网拓扑', 'J3:R200')
    data.DN.node = (Table2[:, 0]).astype(int)
    data.DN.S_node = Table2[:, 1]
    data.DN.P_load = Table2[:, 2]
    data.DN.Q_load = Table2[:, 3]
    data.DN.S_PV = Table2[:, 4]
    data.DN.S_WT = Table2[:, 5]
    data.DN.P_ES = Table2[:, 6]
    data.DN.S_ES = Table2[:, 7]
    data.DN.Y_node = Table2[:, 8].astype(int)

    Table3 = read_excel_range(filepath, '电网拓扑', 'T3:U200')
    data.DN.sub = (Table3[:, 0]).astype(int)
    data.DN.S_sub = Table3[:, 1]

    Table4 = read_excel_range(filepath, '电网拓扑', 'X2:X20')
    data.DN.k_sub = float(Table4[0, 0])
    data.DN.k_PV = float(Table4[1, 0])
    data.DN.k_WT = float(Table4[2, 0])
    data.DN.k_ES = float(Table4[3, 0])
    data.DN.k_load_P = float(Table4[4, 0])
    data.DN.k_load_Q = float(Table4[5, 0])
    data.DN.r_load = float(Table4[6, 0])
    data.DN.r_PV = float(Table4[7, 0])
    data.DN.r_WT = float(Table4[8, 0])
    data.DN.k_back = float(Table4[9, 0])

    data.DN.N_node = len(data.DN.node)
    data.DN.N_sub = len(data.DN.sub)
    data.DN.N_line = len(data.DN.line)
    data.DN.node_to_idx = {int(node): idx for idx, node in enumerate(data.DN.node)}

    # 拓扑集合：首末节点查找 (等效于 MATLAB 的 find)
    data.DN.set_head = {i: np.where(data.DN.head == data.DN.node[i])[0].tolist() for i in range(data.DN.N_node)}
    data.DN.set_tail = {i: np.where(data.DN.tail == data.DN.node[i])[0].tolist() for i in range(data.DN.N_node)}

    # 节点状态矩阵
    data.DN.node_state = np.zeros((data.DN.N_node, data.year))
    for i in range(data.DN.N_node):
        for y in range(data.year):
            # MATLAB Y_node 是从第几年开始（如第1年，第2年），转换为索引需判断
            if y + 1 >= data.DN.Y_node[i]:
                data.DN.node_state[i, y] = 1

    # 线路参数
    # =========================================================================
    # 5. 电网规划参数读取 (线路、变电站、SVC)
    # =========================================================================
    # 线路参数
    data.line = Struct()
    Table_line = read_excel_range(filepath, '电网规划参数', 'B2:B5')
    data.line.S_ref = Table_line[0, 0]  # 线路容量
    data.line.c_inv = Table_line[1, 0]  # 单位长度投资成本
    data.line.T_life = Table_line[2, 0]  # 线路寿命
    data.line.T_build = Table_line[3, 0]  # 建设周期
    data.line.R_ir = (data.r * (1 + data.r) ** data.line.T_life) / ((1 + data.r) ** data.line.T_life - 1)

    # 变电站参数
    data.sub = Struct()
    Table_sub = read_excel_range(filepath, '电网规划参数', 'E2:E5')
    data.sub.S_ref = Table_sub[0, 0]  # 扩容步长
    data.sub.c_inv = Table_sub[1, 0]  # 单位容量投资成本
    data.sub.T_life = Table_sub[2, 0]  # 变电站寿命
    data.sub.T_build = Table_sub[3, 0]  # 建设周期
    data.sub.R_ir = (data.r * (1 + data.r) ** data.sub.T_life) / ((1 + data.r) ** data.sub.T_life - 1)

    # SVC参数
    data.SVC = Struct()
    Table_SVC = read_excel_range(filepath, '电网规划参数', 'H2:H7')
    data.SVC.S_ref = Table_SVC[0, 0]  # 扩容步长
    data.SVC.c_inv = Table_SVC[1, 0]  # 单位容量价格
    data.SVC.T_life = Table_SVC[2, 0]  # 寿命
    data.SVC.Q_max = Table_SVC[3, 0]  # 运行功率上限
    data.SVC.Q_min = Table_SVC[4, 0]  # 运行功率下限
    data.SVC.T_build = Table_SVC[5, 0]  # 建设周期
    data.SVC.R_ir = (data.r * (1 + data.r) ** data.SVC.T_life) / ((1 + data.r) ** data.SVC.T_life - 1)

    # =========================================================================
    # 6. 配电网及 SVC 变量定义
    # =========================================================================
    # 规划变量 - 线路 (Binary)
    var.DN.y_line = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_y_line")
    var.DN.y_line_build = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_y_line_build")
    var.DN.y_line_build_e = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_y_line_build_e")
    var.DN.y_line_build_r = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_y_line_build_r")
    var.DN.y_line_build_u = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_y_line_build_u")

    # 规划变量 - 线路扩容 (Integer & Continuous)
    var.DN.N_line = model.addVars(data.DN.N_line, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_line")
    var.DN.N_line_exp = model.addVars(data.DN.N_line, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_line_exp")
    var.DN.N_line_exp_e = model.addVars(data.DN.N_line, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_line_exp_e")
    var.DN.N_line_exp_r = model.addVars(data.DN.N_line, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_line_exp_r")
    var.DN.N_line_exp_u = model.addVars(data.DN.N_line, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_line_exp_u")
    var.DN.S_line = model.addVars(data.DN.N_line, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_S_line")

    # 规划变量 - 变电站扩容 (Integer & Continuous)
    var.DN.N_node_exp = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_node_exp")
    var.DN.N_node_exp_e = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_node_exp_e")
    var.DN.N_node_exp_r = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_node_exp_r")
    var.DN.N_node_exp_u = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.INTEGER, name="DN_N_node_exp_u")
    var.DN.S_node_exp = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_S_node_exp")
    var.DN.S_node = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_S_node")

    # 投资成本 (Continuous)
    var.DN.C_line_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_C_line_inv")
    var.DN.C_line_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_C_line_res")
    var.DN.C_node_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_C_node_inv")
    var.DN.C_node_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_C_node_res")
    var.DN.C_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_C_inv")
    var.DN.C_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_C_res")

    # SVC规划
    var.SVC = Struct()
    var.SVC.y = model.addVars(data.DN.N_node, data.year, vtype=GRB.BINARY, name="SVC_y")
    var.SVC.S_plan = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.CONTINUOUS, name="SVC_S_plan")

    # 拓扑状态变量
    var.DN.c = model.addVars(data.DN.N_node, data.year, vtype=GRB.BINARY, name="DN_c")
    var.DN.b_ij = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_b_ij")
    var.DN.b_ji = model.addVars(data.DN.N_line, data.year, vtype=GRB.BINARY, name="DN_b_ji")

    # 运行变量 (电压与功率，lb=-GRB.INFINITY 支持无功和反向功率)
    var.DN.U = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_U")
    var.DN.P_line = model.addVars(data.DN.N_line, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_P_line")
    var.DN.Q_line = model.addVars(data.DN.N_line, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_Q_line")
    var.DN.P_node = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_P_node")
    var.DN.Q_node = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_Q_node")
    var.DN.P_load = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_P_load")
    var.DN.Q_load = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_Q_load")
    var.DN.P_sub = model.addVars(data.DN.N_sub, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_P_sub")
    var.DN.Q_sub = model.addVars(data.DN.N_sub, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="DN_Q_sub")
    # 调节能力变量
    var.DN.P_RU = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_P_RU")
    var.DN.P_RD = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_P_RD")
    var.DN.R_demand = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_R_demand")
    var.DN.P_RU_lack = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_P_RU_lack")
    var.DN.P_RD_lack = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="DN_P_RD_lack")
    var.DN.u_RU_lack = model.addVars(data.period, data.scene.N, data.year, vtype=GRB.BINARY, name="DN_u_RU_lack")
    var.DN.u_RD_lack = model.addVars(data.period, data.scene.N, data.year, vtype=GRB.BINARY, name="DN_u_RD_lack")
    # SVC运行变量
    var.SVC.Q = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="SVC_Q")

    # =========================================================================
    # 7. 光伏 (PV) 参数读取与变量定义
    # =========================================================================
    # 容量矩阵读取
    data.PV.S = read_excel_range(filepath, '光伏参数', 'H3:Q200')

    # 候选位置读取 (转换为 0基 索引)
    Table_PV2 = read_excel_range(filepath, '光伏参数', 'A3:B200')
    data.PV.place = (Table_PV2[:, 0]).astype(int).tolist()
    data.PV.S_place = Table_PV2[:, 1]
    data.PV.N_place = len(data.PV.place)

    # 光伏基础参数
    Table_PV3 = read_excel_range(filepath, '光伏参数', 'E2:E7')
    data.PV.S_ref = Table_PV3[0, 0]  # 新建步长
    data.PV.c_inv = Table_PV3[1, 0]  # 单位功率投资成本
    data.PV.T_life = Table_PV3[2, 0]  # 寿命
    data.PV.T_build = Table_PV3[3, 0]  # 建设周期
    data.PV.c_f = Table_PV3[4, 0]  # 运行/维护成本 (对应原代码逻辑)
    data.PV.k = Table_PV3[5, 0]  # 功率因数

    # PV 变量定义
    var.PV = Struct()
    var.PV.N_plan = model.addVars(data.PV.N_place, data.year, lb=0, vtype=GRB.INTEGER, name="PV_N_plan")
    var.PV.S_plan = model.addVars(data.PV.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_S_plan")
    var.PV.S_plan_e = model.addVars(data.PV.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_S_plan_e")
    var.PV.S_plan_u = model.addVars(data.PV.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_S_plan_u")
    var.PV.S_all = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_S_all")

    # 投资成本
    var.PV.C_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_C_inv")
    var.PV.C_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="PV_C_res")

    # 运行变量 (考虑到无功 Q 可能为负，设定 lb=-INFINITY)
    var.PV.P = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_P")
    var.PV.Q = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="PV_Q")
    var.PV.P_q = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="PV_P_q")

    # =========================================================================
    # 8. 储能系统 (ES) 参数与变量定义
    # =========================================================================
    Table_ES1 = read_excel_range(filepath, '电网拓扑', 'P3:Q200')
    data.ES.S = Table_ES1[:, 0]
    data.ES.E = Table_ES1[:, 1]

    Table_ES2 = read_excel_range(filepath, '储能参数', 'A3:C100')
    data.ES.place = (Table_ES2[:, 0]).astype(int).tolist()
    data.ES.S_place = Table_ES2[:, 1]
    data.ES.E_place = Table_ES2[:, 2]
    data.ES.N_place = len(data.ES.place)

    Table_ES3 = read_excel_range(filepath, '储能参数', 'F2:F6')
    data.ES.S_ref = Table_ES3[0, 0]
    data.ES.c_inv = Table_ES3[1, 0]
    data.ES.T_life = Table_ES3[2, 0]
    data.ES.T_build = Table_ES3[3, 0]
    data.ES.k = Table_ES3[4, 0]
    data.ES.R_ir = (data.r * (1 + data.r) ** data.ES.T_life) / ((1 + data.r) ** data.ES.T_life - 1)

    # ES 变量定义
    var.ES = Struct()

    # 规划变量
    var.ES.N_plan = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.INTEGER, name="ES_N_plan")
    var.ES.S_plan = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_S_plan")
    var.ES.E_plan = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_E_plan")
    var.ES.S_plan_e = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_S_plan_e")
    var.ES.S_plan_r = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_S_plan_r")
    var.ES.S_plan_u = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_S_plan_u")
    var.ES.E_plan_u = model.addVars(data.ES.N_place, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_E_plan_u")

    var.ES.S_all = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_S_all")
    var.ES.E_all = model.addVars(data.DN.N_node, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_E_all")

    # 投资成本
    var.ES.C_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_C_inv")
    var.ES.C_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ES_C_res")

    # 运行变量
    var.ES.X_c = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, vtype=GRB.BINARY, name="ES_X_c")
    var.ES.X_f = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, vtype=GRB.BINARY, name="ES_X_f")
    var.ES.P = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ES_P")
    var.ES.P_c = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_P_c")
    var.ES.P_f = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_P_f")
    var.ES.E = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_E")
    var.ES.P_RU = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_P_RU")
    var.ES.P_RD = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_P_RD")
    var.ES.P_RU_all = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_P_RU_all")
    var.ES.P_RD_all = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="ES_P_RD_all")

    # =========================================================================
    # 9. 电动汽车充换一体站 (BSS) 与换电 (SB) 参数及数据预处理
    # =========================================================================
    # BSS 容器已在 Data_read 开始处初始化；此处必须复用它，不能重新赋值，
    # 否则会清除刚读取的 N_station、node_DN 和设备参数。
    Table_BSS1 = read_excel_range(filepath, '充换一体站参数', 'B3:H20')
    data.BSS.node_DN = (Table_BSS1[:, 0]).astype(int)
    data.BSS.N_FP = Table_BSS1[:, 1]
    data.BSS.N_SB = Table_BSS1[:, 2]
    data.BSS.N_CB = Table_BSS1[:, 3]
    data.BSS.N_FP_max = Table_BSS1[:, 4]
    data.BSS.N_SB_max = Table_BSS1[:, 5]
    data.BSS.N_CB_max = Table_BSS1[:, 6]
    data.BSS.N_station = len(data.BSS.node_DN)

    # 换电参数读取
    data.BSS.N_SOC = 10

    data.BSS.N_demand = []
    data.BSS.N_demand.append(read_excel_range(filepath, '换电需求', 'C3:G26'))
    data.BSS.N_demand.append(read_excel_range(filepath, '换电需求', 'J3:N26'))
    data.BSS.N_demand.append(read_excel_range(filepath, '换电需求', 'Q3:U26'))
    data.BSS.N_demand.append(read_excel_range(filepath, '换电需求', 'X3:AB26'))
    data.BSS.N_demand.append(read_excel_range(filepath, '换电需求', 'AE3:AI26'))

    data.BSS.N_demand_all_data = read_excel_range(filepath, '充换一体站参数', 'S3:W26')

    # 数据嵌套容器初始化
    data.BSS.demand = []  # list of lists (按 year 和 station 索引)
    data.BSS.demand_all = np.zeros((data.year, data.period, data.BSS.N_station))
    data.BSS.demand_all_max = np.zeros((data.year, data.BSS.N_station))

    data.BSS.N_in_visual = []  # list of lists (按 year 和 station 索引)
    data.BSS.N_out_visual = []

    # 这里的矩阵维度直接初始化为最大尺寸，便于后续索引 (station, period, scene, year, SOC)
    data.BSS.N_in = np.zeros((data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC))
    data.BSS.N_out = np.zeros((data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC))
    call_shape = (
        data.BSS.N_station,
        data.period,
        data.scene.N,
        data.year,
    )
    data.BSS.alpha_TU = np.zeros(call_shape, dtype=float)
    data.BSS.alpha_TD = np.zeros(call_shape, dtype=float)
    if data.period != 24:
        raise ValueError("BSS hourly call coefficients require exactly 24 periods")

    # BSS call coefficients.  TU increases charging load and TD reduces it.
    # Exactly one direction is enabled in every hour; hence both directions
    # cannot be called simultaneously at the same station and time.
    bss_tu_hourly = np.zeros(data.period, dtype=float)
    bss_td_hourly = np.zeros(data.period, dtype=float)
    bss_tu_hourly[8:22] = data.scene.BSS_TU_call_coefficient
    bss_td_hourly[:8] = data.scene.BSS_TD_call_coefficient
    bss_td_hourly[22:] = data.scene.BSS_TD_call_coefficient
    for station in range(data.BSS.N_station):
        for scene in range(data.scene.N):
            for year in range(data.year):
                data.BSS.alpha_TU[station, :, scene, year] = bss_tu_hourly
                data.BSS.alpha_TD[station, :, scene, year] = bss_td_hourly

    for y in range(data.year):
        demand_year = []
        in_vis_year = []
        out_vis_year = []

        for i in range(data.BSS.N_station):
            # 将原始需求数据按年份缩放并取整
            base_demand = np.array(data.BSS.N_demand[i], dtype=float)
            scaled_demand = np.round(base_demand * data.EV.N_year[y]).astype(int)
            demand_year.append(scaled_demand)

            for t in range(data.period):
                data.BSS.demand_all[y, t, i] = np.sum(scaled_demand[t, :])
            data.BSS.demand_all_max[y, i] = np.max(data.BSS.demand_all[y, :, i])

            # 为当前站点初始化场景列表
            in_vis_scene = []
            out_vis_scene = []

            for s in range(data.scene.N):
                vis_in = np.zeros((data.period, data.BSS.N_SOC))
                vis_out = np.zeros((data.period, data.BSS.N_SOC))

                for t in range(data.period):
                    # 10级电量区间逻辑映射
                    # MATLAB: r 取 1 到 5。对应的列是 2*r-1 和 2*r
                    # Python: r 取 0 到 4。对应的列是 2*r 和 2*r + 1
                    for r in range(5):
                        demand_val = scaled_demand[t, r]
                        if r == 0:
                            val_ceil = math.ceil(demand_val * 0.3)
                            val_floor = math.floor(demand_val * 0.7)
                            vis_in[t, 2 * r] = val_ceil
                            vis_in[t, 2 * r + 1] = val_floor
                            data.BSS.N_in[i, t, s, y, 2 * r] = val_ceil
                            data.BSS.N_in[i, t, s, y, 2 * r + 1] = val_floor
                        else:
                            val_ceil = math.ceil(demand_val * 0.7)
                            val_floor = math.floor(demand_val * 0.3)
                            vis_in[t, 2 * r] = val_ceil
                            vis_in[t, 2 * r + 1] = val_floor
                            data.BSS.N_in[i, t, s, y, 2 * r] = val_ceil
                            data.BSS.N_in[i, t, s, y, 2 * r + 1] = val_floor

                    # 输出逻辑：前 9 级为 0，第 10 级为总和 (Python 索引中 9 对应最后也是第 10 级)
                    data.BSS.N_out[i, t, s, y, :9] = 0
                    vis_out[t, :9] = 0

                    total_in = np.sum(vis_in[t, :])
                    data.BSS.N_out[i, t, s, y, 9] = total_in
                    vis_out[t, 9] = total_in

                in_vis_scene.append(vis_in)
                out_vis_scene.append(vis_out)

        data.BSS.demand.append(demand_year)
        data.BSS.N_in_visual.append(in_vis_scene)
        data.BSS.N_out_visual.append(out_vis_scene)

    # 充换电站基础设备参数
    Table_BSS4 = read_excel_range(filepath, '充换一体站参数', 'B22:B26')
    data.FP = Struct()
    data.FP.P = Table_BSS4[0, 0]
    data.FP.k = Table_BSS4[1, 0]
    data.FP.c_inv = Table_BSS4[2, 0]
    data.FP.T_life = Table_BSS4[3, 0]
    data.FP.T_build = Table_BSS4[4, 0]

    Table_BSS5 = read_excel_range(filepath, '充换一体站参数', 'E22:E25')
    data.BSS.E = Table_BSS5[0, 0]
    data.BSS.c_inv = Table_BSS5[1, 0]
    data.BSS.T_life = Table_BSS5[2, 0]
    data.BSS.T_build = Table_BSS5[3, 0]

    Table_BSS6 = read_excel_range(filepath, '充换一体站参数', 'H22:H25')
    data.CB = Struct()
    data.CB.P = Table_BSS6[0, 0]
    data.CB.c_inv = Table_BSS6[1, 0]
    data.CB.T_life = Table_BSS6[2, 0]
    data.CB.T_build = Table_BSS6[3, 0]

    configure_bss_integer_data(data)

    # 规划变量 - 快充桩 (Integer)
    var.BSS.N_FP = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_FP")
    var.BSS.N_FP_plan = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_FP_plan")
    var.BSS.N_FP_plan_e = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_FP_plan_e")
    var.BSS.N_FP_plan_r = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_FP_plan_r")
    var.BSS.N_FP_plan_u = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_FP_plan_u")

    # 规划变量 - 换电电池 (Integer)
    var.BSS.N_SB = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_SB")
    var.BSS.N_SB_plan = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_SB_plan")
    var.BSS.N_SB_plan_e = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_SB_plan_e")
    var.BSS.N_SB_plan_r = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_SB_plan_r")
    var.BSS.N_SB_plan_u = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_SB_plan_u")

    # 规划变量 - 充电槽 (Integer)
    var.BSS.N_CB = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_CB")
    var.BSS.N_CB_plan = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_CB_plan")
    var.BSS.N_CB_plan_e = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_CB_plan_e")
    var.BSS.N_CB_plan_r = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_CB_plan_r")
    var.BSS.N_CB_plan_u = model.addVars(data.BSS.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="BSS_N_CB_plan_u")

    # 投资成本 (Continuous)
    var.BSS.C_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="BSS_C_inv")
    var.BSS.C_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="BSS_C_res")
    var.BSS.C_FP_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="BSS_C_FP_inv")
    var.BSS.C_FP_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="BSS_C_FP_res")
    var.BSS.C_SB_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="BSS_C_SB_inv")
    var.BSS.C_SB_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="BSS_C_SB_res")
    var.BSS.C_CB_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="BSS_C_CB_inv")
    var.BSS.C_CB_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="BSS_C_CB_res")

    # 运行期库存和各类 SOC 转移均表示电池数量，必须使用整数变量。
    # 即使采用稠密转移变量，也不允许出现分数电池或分数电池转移。
    var.BSS.N_S = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC, lb=0, vtype=GRB.INTEGER, name="SB_N_S")
    if create_dense_bss_transitions:
        var.BSS.N_T = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC, data.BSS.N_SOC, lb=0, vtype=GRB.INTEGER, name="SB_N_T")
        var.BSS.N_U = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC, data.BSS.N_SOC, lb=0, vtype=GRB.INTEGER, name="SB_N_U")
        var.BSS.N_D = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC, data.BSS.N_SOC, lb=0, vtype=GRB.INTEGER, name="SB_N_D")
        var.BSS.N_gap = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, data.BSS.N_SOC, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="SB_N_gap")
    else:
        var.BSS.N_T = {}
        var.BSS.N_U = {}
        var.BSS.N_D = {}
        var.BSS.N_gap = None

    # 换电站功率与调节能力 (Continuous)
    var.BSS.P_c = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="SB_P_c")
    var.BSS.P_f = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="SB_P_f")
    var.BSS.P = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="SB_P")
    var.BSS.P_RU = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="SB_P_RU")
    var.BSS.P_RD = model.addVars(data.BSS.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="SB_P_RD")

    bss_arc_keys = [
        (station, time, scene, year, source_soc, target_soc)
        for station in range(data.BSS.N_station)
        for time in range(data.period)
        for scene in range(data.scene.N)
        for year in range(data.year)
        for source_soc, target_soc in data.BSS.charge_arcs
    ]
    var.BSS.H_D = model.addVars(bss_arc_keys, lb=0, vtype=GRB.INTEGER, name="SB_H_D")
    var.BSS.N_R = model.addVars(bss_arc_keys, lb=0, vtype=GRB.INTEGER, name="SB_N_R")
    var.BSS.X_U = model.addVars(bss_arc_keys, lb=0, vtype=GRB.INTEGER, name="SB_X_U")
    var.BSS.X_D = model.addVars(bss_arc_keys, lb=0, vtype=GRB.INTEGER, name="SB_X_D")

    # 充电站与配电网 (DN) 的耦合参数变量。
    # P/P_RU/P_RD 已在上方创建，这里只创建 DN 耦合和站级聚合变量，
    # 避免重复创建后覆盖 Python 引用并在模型中留下孤立变量。
    var.BSS.P_DN = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="BSS_P_DN")
    var.BSS.P_RU_all = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="BSS_P_RU_all")
    var.BSS.P_RD_all = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="BSS_P_RD_all")

    # =====================================================================================================================================================================
    # 区域热网
    # =====================================================================================================================================================================
    # 基础与拓扑参数读取
    data.HN.c_w = 4.2  # 水比热容，kJ/(kg*℃)

    # 地下管廊信息
    Table_HN1 = read_excel_range(filepath, '热网拓扑', 'A3:G100')
    data.HN.pipe = (Table_HN1[:, 0] - 1).astype(int)
    data.HN.head = (Table_HN1[:, 1] - 1).astype(int)
    data.HN.tail = (Table_HN1[:, 2] - 1).astype(int)
    data.HN.L_pipe = Table_HN1[:, 3]
    data.HN.N_pipe_initial = Table_HN1[:, 6]

    # 节点信息
    Table_HN_node = read_excel_range(filepath, '热网拓扑', 'I3:I100')
    data.HN.node = (Table_HN_node[:, 0] - 1).astype(int)

    # 负荷节点信息
    Table_HN2 = read_excel_range(filepath, '热网拓扑', 'K3:O100')
    data.HN.load = (Table_HN2[:, 0] - 1).astype(int)
    data.HN.f_load = Table_HN2[:, 1]
    data.HN.f_load_min = Table_HN2[:, 2]
    data.HN.H_load = Table_HN2[:, 3]
    data.HN.Y_load = Table_HN2[:, 4].astype(int)

    # 供热站节点信息
    Table_HN3 = read_excel_range(filepath, '热网拓扑', 'Q3:X100')
    data.HN.station = (Table_HN3[:, 0] - 1).astype(int)
    data.HN.station_DN = (Table_HN3[:, 1]).astype(int)
    data.HN.f_station = Table_HN3[:, 2]
    data.HN.S_CHP = Table_HN3[:, 3]
    data.HN.S_EB = Table_HN3[:, 4]
    data.HN.S_CHP_max = Table_HN3[:, 5]
    data.HN.S_EB_max = Table_HN3[:, 6]
    data.HN.y_station = Table_HN3[:, 7]

    # 计算其他节点 (等效 MATLAB: setdiff(setdiff(node, station), load))
    data.HN.other = list(set(data.HN.node) - set(data.HN.station) - set(data.HN.load))

    data.HN.N_node = len(data.HN.node)
    data.HN.N_load = len(data.HN.load)
    data.HN.N_pipe = len(data.HN.pipe)
    data.HN.N_station = len(data.HN.station)
    data.HN.gamma_call = np.zeros(
        (data.HN.N_station, data.period, data.scene.N, data.year), dtype=float
    )
    if data.period != 24:
        raise ValueError("HN hourly call coefficients require exactly 24 periods")

    # HN signed call coefficient: positive values enable RU, negative values
    # enable RD.  Only one sign is nonzero in each hour, so the two directions
    # cannot be called simultaneously.
    hn_gamma_hourly = np.zeros(data.period, dtype=float)
    hn_gamma_hourly[:8] = -data.scene.HN_RD_call_coefficient
    hn_gamma_hourly[8:22] = data.scene.HN_RU_call_coefficient
    hn_gamma_hourly[22:] = -data.scene.HN_RD_call_coefficient
    for station in range(data.HN.N_station):
        for scene in range(data.scene.N):
            for year in range(data.year):
                data.HN.gamma_call[station, :, scene, year] = hn_gamma_hourly

    if np.any(data.HN.gamma_call < -1.0) or np.any(data.HN.gamma_call > 1.0):
        raise ValueError("HN.gamma_call must be within [-1, 1]")
    data.HN.node_to_idx = {int(node): idx for idx, node in enumerate(data.HN.node)}
    data.HN.load_node_idx = [data.HN.node_to_idx[int(node)] for node in data.HN.load]
    data.HN.station_node_idx = [data.HN.node_to_idx[int(node)] for node in data.HN.station]
    data.HN.other_node_idx = [data.HN.node_to_idx[int(node)] for node in data.HN.other]
    # 首末节点集合查找 (生成字典，键为节点索引，值为连接的管道索引列表)
    data.HN.set_head = {i: np.where(data.HN.head == data.HN.node[i])[0].tolist() for i in range(data.HN.N_node)}
    data.HN.set_tail = {i: np.where(data.HN.tail == data.HN.node[i])[0].tolist() for i in range(data.HN.N_node)}

    # 负荷节点状态接入矩阵
    data.HN.load_state = np.zeros((data.HN.N_load, data.year))
    for i in range(data.HN.N_load):
        for y in range(data.year):
            if y + 1 >= data.HN.Y_load[i]:
                data.HN.load_state[i, y] = 1

    data.HN.fixed_load_flow = np.zeros((data.HN.N_load, data.year), dtype=float)
    for load_index in range(data.HN.N_load):
        for year in range(data.year):
            data.HN.fixed_load_flow[load_index, year] = (
                float(data.HN.f_load[load_index])
                * float(data.HN.load_state[load_index, year])
            )
    data.HN.path_index = [
        (pipe, load, station, year)
        for pipe in range(data.HN.N_pipe)
        for load in range(data.HN.N_load)
        for station in range(data.HN.N_station)
        for year in range(data.year)
        if float(data.HN.load_state[load, year]) > 0.5
    ]

    # 温度参数
    data.HN.T_S_max, data.HN.T_S_min = 90, 85
    data.HN.T_R_max, data.HN.T_R_min = 90, 70

    # CHP总供热容量上限比例：
    # CHP可用供热容量不得超过各规划年最大逐时热负荷的该比例。
    # 该参数显式放在热网数据中，便于后续开展敏感性分析。
    data.HN.CHP_heat_capacity_ratio = 0.45
    if not 0.0 < float(data.HN.CHP_heat_capacity_ratio) <= 1.0:
        raise ValueError("HN.CHP_heat_capacity_ratio must be in (0, 1].")
    data.HN.H_peak_by_year = np.zeros(data.year, dtype=float)
    for year in range(data.year):
        year_load = np.asarray(data.HN.H_load, dtype=float)
        year_load = year_load * float(data.HN.H_year[year])
        year_load = year_load * np.asarray(data.HN.load_state[:, year], dtype=float)
        data.HN.H_peak_by_year[year] = max(
            (
                float(np.sum(year_load)) * float(data.scene.k_H[time, scene])
                for time in range(data.period)
                for scene in range(data.scene.N)
            ),
            default=0.0,
        )

    # 流量参数
    f_max = np.sum(data.HN.f_load)
    data.HN.f_max = f_max
    data.HN.f_pipe_max = f_max
    data.HN.f_pipe_min = -f_max

    # 热损失参数
    data.HN.K_loss = 0.002 * data.HN.L_pipe

    # 大 M 法参数
    data.HN.M_F = data.HN.f_max
    data.HN.M_T = data.HN.T_S_max
    data.HN.M_FT = data.HN.M_F * data.HN.M_T

    # 初始化边界矩阵
    data.HN.F_node_min = np.zeros((data.HN.N_node, data.year))
    data.HN.F_node_max = np.zeros((data.HN.N_node, data.year))
    data.HN.FRin_min = np.zeros((data.HN.N_node, data.year))
    data.HN.FRin_max = np.zeros((data.HN.N_node, data.year))

    data.HN.FR_bij_max = np.zeros((data.HN.N_pipe, data.year))
    data.HN.FR_bij_min = np.zeros((data.HN.N_pipe, data.year))
    data.HN.FR_bji_max = np.zeros((data.HN.N_pipe, data.year))
    data.HN.FR_bji_min = np.zeros((data.HN.N_pipe, data.year))

    for y in range(data.year):
        # 节点上下限
        for i in range(data.HN.N_node):
            node_id = data.HN.node[i]
            if node_id in data.HN.load:
                j = list(data.HN.load).index(node_id)
                data.HN.F_node_min[i, y] = -data.HN.f_load[j]
                data.HN.F_node_max[i, y] = -data.HN.f_load[j]
            elif node_id in data.HN.station:
                j = list(data.HN.station).index(node_id)
                data.HN.F_node_min[i, y] = 0
                data.HN.F_node_max[i, y] = data.HN.f_station[j]
            else:
                data.HN.F_node_min[i, y] = 0
                data.HN.F_node_max[i, y] = 0

            data.HN.FRin_min[i, y] = 0
            data.HN.FRin_max[i, y] = data.HN.f_max

        # 管道上下限
        for i in range(data.HN.N_pipe):
            data.HN.FR_bij_max[i, y] = data.HN.f_pipe_max
            data.HN.FR_bij_min[i, y] = 0
            data.HN.FR_bji_max[i, y] = 0
            data.HN.FR_bji_min[i, y] = data.HN.f_pipe_min

    # 高维矩阵初始化 (使用 np.full 广播，避免缓慢的嵌套循环)
    dims_node = (data.HN.N_node, data.period, data.scene.N, data.year)
    dims_pipe = (data.HN.N_pipe, data.period, data.scene.N, data.year)

    data.HN.TSex_max_all = np.full(dims_node, data.HN.T_S_max)
    data.HN.TSex_min_all = np.full(dims_node, data.HN.T_S_min)
    data.HN.TRex_max_all = np.full(dims_node, data.HN.T_R_max)
    data.HN.TRex_min_all = np.full(dims_node, data.HN.T_R_min)
    data.HN.TRn_max_all = np.full(dims_node, data.HN.T_R_max)
    data.HN.TRn_min_all = np.full(dims_node, data.HN.T_R_min)

    data.HN.TRi_max_all = np.full(dims_pipe, data.HN.T_R_max)
    data.HN.TRi_min_all = np.full(dims_pipe, data.HN.T_R_min)
    data.HN.TRj_max_all = np.full(dims_pipe, data.HN.T_R_max)
    data.HN.TRj_min_all = np.full(dims_pipe, data.HN.T_R_min)

    # 分段线性化松弛参数
    if hn_linearization_mode == 'mccormick':
        _configure_hn_mccormick_parameters(data, hn_segment_count)
    else:
        data.HN.T_pw = None
        data.HN.F_pw = None


    data.HN.H_node_max = np.zeros(data.HN.N_node)
    for i in range(data.HN.N_node):
        node_id = data.HN.node[i]
        if node_id in data.HN.station:
            j = list(data.HN.station).index(node_id)
            data.HN.H_node_max[i] = data.HN.c_w * abs(data.HN.f_station[j]) * (data.HN.T_S_max - data.HN.T_R_min)
        elif node_id in data.HN.load:
            j = list(data.HN.load).index(node_id)
            data.HN.H_node_max[i] = data.HN.c_w * abs(data.HN.f_load[j]) * (data.HN.T_S_max - data.HN.T_R_min)
        else:
            data.HN.H_node_max[i] = 0


    # =====================================================================================================================================================================
    # 规划变量
    var.HN.y_pipe = model.addVars(data.HN.N_pipe, data.year, vtype=GRB.BINARY, name="HN_y_pipe")
    var.HN.y_node = model.addVars(data.HN.N_node, data.year, vtype=GRB.BINARY, name="HN_y_node")

    # 管道规划
    var.HN.N_pipe = model.addVars(data.HN.N_pipe, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_pipe")
    var.HN.F_pipe = model.addVars(data.HN.N_pipe, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_F_pipe")
    var.HN.N_pipe_exp = model.addVars(data.HN.N_pipe, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_pipe_exp")
    var.HN.N_pipe_exp_e = model.addVars(data.HN.N_pipe, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_pipe_exp_e")
    var.HN.N_pipe_exp_r = model.addVars(data.HN.N_pipe, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_pipe_exp_r")
    var.HN.N_pipe_exp_u = model.addVars(data.HN.N_pipe, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_pipe_exp_u")

    # 能源站规划
    var.HN.y_station = model.addVars(data.HN.N_station, data.year, vtype=GRB.BINARY, name="HN_y_station")
    var.HN.S_CHP = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_CHP")
    var.HN.S_CHP_plan = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_CHP_plan")
    var.HN.S_CHP_plan_e = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_CHP_plan_e")
    var.HN.S_CHP_plan_r = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_CHP_plan_r")
    var.HN.S_CHP_plan_u = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_CHP_plan_u")
    var.HN.N_CHP = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_CHP")

    var.HN.S_EB = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_EB")
    var.HN.S_EB_plan = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_EB_plan")
    var.HN.S_EB_plan_e = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_EB_plan_e")
    var.HN.S_EB_plan_r = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_EB_plan_r")
    var.HN.S_EB_plan_u = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_S_EB_plan_u")
    var.HN.N_EB = model.addVars(data.HN.N_station, data.year, lb=0, vtype=GRB.INTEGER, name="HN_N_EB")

    # 投资成本
    var.HN.C_pipe_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_C_pipe_inv")
    var.HN.C_pipe_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_C_pipe_res")
    var.HN.C_CHP_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_C_CHP_inv")
    var.HN.C_CHP_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_C_CHP_res")
    var.HN.C_EB_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_C_EB_inv")
    var.HN.C_EB_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_C_EB_res")
    var.HN.C_inv = model.addVars(data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_C_inv")
    var.HN.C_res = model.addVars(data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_C_res")

    # 生成树变量
    var.HN.b_ij = model.addVars(data.HN.N_pipe, data.year, vtype=GRB.BINARY, name="HN_b_ij")
    var.HN.b_ji = model.addVars(data.HN.N_pipe, data.year, vtype=GRB.BINARY, name="HN_b_ji")
    var.HN.c = model.addVars(data.HN.N_node, data.year, vtype=GRB.BINARY, name="HN_c")

    # 三种热网模型共享的有向路径追踪变量
    var.HN.x_path_ij = model.addVars(
        data.HN.path_index, vtype=GRB.BINARY, name="HN_x_path_ij"
    )
    var.HN.x_path_ji = model.addVars(
        data.HN.path_index, vtype=GRB.BINARY, name="HN_x_path_ji"
    )

    # --- 运行变量 ---
    # 功率变量 (可能包含正反向流动，lb=-GRB.INFINITY)
    var.HN.H_node = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_H_node")
    var.HN.H_load = model.addVars(data.HN.N_load, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_H_load")
    var.HN.P = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_P")
    var.HN.P_DN = model.addVars(data.DN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_P_DN")

    # 热网辅助变量

    # 调节能力变量
    var.HN.H_RU = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_H_RU")
    var.HN.H_RD = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_H_RD")
    var.HN.P_RU_all = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_P_RU_all")
    var.HN.P_RD_all = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_P_RD_all")
    var.HN.P_RU_all1 = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_P_RU_all1")
    var.HN.P_RD_all1 = model.addVars(data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="HN_P_RD_all1")

    # 流量变量 (管网多商品流，lb=-GRB.INFINITY 支持回流)
    var.HN.f_node = model.addVars(data.HN.N_node, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_node")
    var.HN.f_station = model.addVars(data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_station")
    var.HN.f_load = model.addVars(data.HN.N_load, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_load")
    var.HN.source_zone_affiliation = model.addVars(
        data.HN.N_node,
        data.HN.N_station,
        data.year,
        vtype=GRB.BINARY,
        name="HN_source_zone_affiliation",
    )
    var.HN.f_S = model.addVars(data.HN.N_pipe, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_S")
    var.HN.f_R = model.addVars(data.HN.N_pipe, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_R")
    var.HN.f_S_in = model.addVars(data.HN.N_node, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_S_in")
    var.HN.f_S_out = model.addVars(data.HN.N_node, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_S_out")
    var.HN.f_R_in = model.addVars(data.HN.N_node, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_R_in")
    var.HN.f_R_out = model.addVars(data.HN.N_node, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_f_R_out")
    var.HN.fS_bij = model.addVars(data.HN.N_pipe, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_fS_bij")
    var.HN.fS_bji = model.addVars(data.HN.N_pipe, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_fS_bji")
    var.HN.fR_bij = model.addVars(data.HN.N_pipe, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_fR_bij")
    var.HN.fR_bji = model.addVars(data.HN.N_pipe, data.HN.N_station, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_fR_bji")

    var.HN.F_node = model.addVars(data.HN.N_node, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_node")
    var.HN.F_S = model.addVars(data.HN.N_pipe, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_S")
    var.HN.F_R = model.addVars(data.HN.N_pipe, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_R")
    var.HN.F_S_in = model.addVars(data.HN.N_node, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_S_in")
    var.HN.F_S_out = model.addVars(data.HN.N_node, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_S_out")
    var.HN.F_R_in = model.addVars(data.HN.N_node, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_R_in")
    var.HN.F_R_out = model.addVars(data.HN.N_node, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_R_out")
    var.HN.FS_bij = model.addVars(data.HN.N_pipe, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FS_bij")
    var.HN.FS_bji = model.addVars(data.HN.N_pipe, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FS_bji")
    var.HN.FR_bij = model.addVars(data.HN.N_pipe, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bij")
    var.HN.FR_bji = model.addVars(data.HN.N_pipe, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bji")

    # 温度变量
    var.HN.T_S_node = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_S_node")
    var.HN.T_R_node = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_node")
    var.HN.T_S_ex = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_S_ex")
    var.HN.T_R_ex = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_ex")
    var.HN.T_S_i = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_S_i")
    var.HN.T_R_i = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_i")
    var.HN.T_S_j = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_S_j")
    var.HN.T_R_j = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_T_R_j")

    # 双线性项及其辅助变量
    var.HN.FR_bij_TRi = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bij_TRi")
    var.HN.FR_bji_TRj = model.addVars(data.HN.N_pipe, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FR_bji_TRj")
    var.HN.F_TSex = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TSex")
    var.HN.F_TRex = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TRex")
    var.HN.F_TRn = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_F_TRn")
    var.HN.FRin_TRn = model.addVars(data.HN.N_node, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="HN_FRin_TRn")



    # 分段 McCormick 松弛专用变量
    if data.HN.linearization_mode == 'mccormick':
        _create_hn_mccormick_variables(model, data, var)
    elif data.HN.linearization_mode == 'physical_exact':
        _create_hn_exact_linear_variables(model, data, var)

    data.pipe = Struct()
    Table_pipe = read_excel_range(filepath, '热网规划参数', 'K2:K8')
    data.pipe.c_inv = Table_pipe[0, 0]
    data.pipe.c_m = Table_pipe[1, 0]
    data.pipe.T_life = Table_pipe[2, 0]
    data.pipe.k_h = Table_pipe[3, 0]
    data.pipe.A = Table_pipe[4, 0]
    data.pipe.f_max = Table_pipe[5, 0]
    data.pipe.T_build = Table_pipe[6, 0]
    data.pipe.R_ir = (data.r * (1 + data.r) ** data.pipe.T_life) / ((1 + data.r) ** data.pipe.T_life - 1)

    data.CHP = Struct()
    Table_CHP = read_excel_range(filepath, '热网规划参数', 'B2:B11')
    data.CHP.S_ref = Table_CHP[0, 0]
    data.CHP.c_inv = Table_CHP[1, 0]
    data.CHP.c_m = Table_CHP[2, 0]
    data.CHP.T_life = Table_CHP[3, 0]
    data.CHP.k_EG = Table_CHP[4, 0]
    data.CHP.k_HE = Table_CHP[5, 0]
    data.CHP.k_CO = Table_CHP[6, 0]
    data.CHP.k_SO = Table_CHP[7, 0]
    data.CHP.k_NO = Table_CHP[8, 0]
    data.CHP.T_build = Table_CHP[9, 0]
    data.CHP.ramp_up_fraction_per_hour = 1.0
    data.CHP.ramp_down_fraction_per_hour = 1.0
    data.CHP.minimum_output_fraction = 0.0
    data.CHP.R_ir = (data.r * (1 + data.r) ** data.CHP.T_life) / ((1 + data.r) ** data.CHP.T_life - 1)

    data.GB = Struct()
    Table_GB = read_excel_range(filepath, '热网规划参数', 'E2:E10')
    data.GB.S_ref = Table_GB[0, 0]
    data.GB.c_inv = Table_GB[1, 0]
    data.GB.c_m = Table_GB[2, 0]
    data.GB.T_life = Table_GB[3, 0]
    data.GB.k_HG = Table_GB[4, 0]
    data.GB.k_CO = Table_GB[5, 0]
    data.GB.k_SO = Table_GB[6, 0]
    data.GB.k_NO = Table_GB[7, 0]
    data.GB.T_build = Table_GB[8, 0]
    data.GB.R_ir = (data.r * (1 + data.r) ** data.GB.T_life) / ((1 + data.r) ** data.GB.T_life - 1)

    data.EB = Struct()
    Table_EB = read_excel_range(filepath, '热网规划参数', 'H2:H7')
    data.EB.S_ref = Table_EB[0, 0]
    data.EB.c_inv = Table_EB[1, 0]
    data.EB.c_m = Table_EB[2, 0]
    data.EB.T_life = Table_EB[3, 0]
    data.EB.k_HE = Table_EB[4, 0]
    data.EB.T_build = Table_EB[5, 0]
    data.EB.ramp_up_fraction_per_hour = 1.0
    data.EB.ramp_down_fraction_per_hour = 1.0
    data.EB.minimum_output_fraction = 0.0
    data.EB.R_ir = (data.r * (1 + data.r) ** data.EB.T_life) / ((1 + data.r) ** data.EB.T_life - 1)

    # --- 设备运行变量 ---
    var.CHP = Struct()
    var.CHP.P = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="CHP_P")
    var.CHP.Q = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="CHP_Q")
    var.CHP.H = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="CHP_H")
    var.CHP.G = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_G")
    var.CHP.P_call = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_P_call")
    var.CHP.H_call = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_H_call")
    var.CHP.G_call = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_G_call")
    var.CHP.P_RU = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_P_RU")
    var.CHP.P_RD = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_P_RD")
    var.CHP.P_RU1 = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_P_RU1")
    var.CHP.P_RD1 = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="CHP_P_RD1")

    var.GB = Struct()
    var.GB.H = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="GB_H")
    var.GB.G = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="GB_G")

    var.EB = Struct()
    var.EB.H = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="EB_H")
    var.EB.P = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="EB_P")
    var.EB.P_call = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="EB_P_call")
    var.EB.H_call = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="EB_H_call")
    var.EB.Q = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="EB_Q")
    var.EB.P_RU = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="EB_P_RU")
    var.EB.P_RD = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="EB_P_RD")
    var.EB.P_RU1 = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="EB_P_RU1")
    var.EB.P_RD1 = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="EB_P_RD1")

    # 碳排放变量
    var.emission = Struct()
    var.emission.CO = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="emission_CO")
    var.emission.SO = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="emission_SO")
    var.emission.NO = model.addVars(data.HN.N_station, data.period, data.scene.N, data.year, lb=0, vtype=GRB.CONTINUOUS, name="emission_NO")

    # 最后记得在整个函数末尾返回 data, var
    return data, var
