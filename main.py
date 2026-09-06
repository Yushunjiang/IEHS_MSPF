"""Physical-structure-driven exact-linear planning entry point."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import sys


_PROJECT_ROOT = Path(__file__).resolve().parent
GENERAL_RESULT_ROOT = _PROJECT_ROOT / "results"
_PROJECT_ROOT_IMPORT = chr(92) * 2 + "?" + chr(92) + str(_PROJECT_ROOT)
for _path_entry in (str(_PROJECT_ROOT), _PROJECT_ROOT_IMPORT):
    if _path_entry not in sys.path:
        sys.path.insert(0, _path_entry)
import tempfile
import time

import gurobipy as gp
import numpy as np
import scipy.io as sio
from gurobipy import GRB

from Constraints_DN import Constraints_DN
from Constraints_ES import Constraints_ES
from Constraints_PV import Constraints_PV
from Constraints_BSS import Constraints_BSS
from Data_export import Data_export
from Data_read import Data_read, Struct
from object import object
from Constraints_HN import Constraints_HN_manuscript_exact_linear


MAIN_PLANNING_YEARS = 10
DEFAULT_RUN_NAME = "main"

# Latest run artifacts exposed to interactive consoles after runfile().
data = None
result = None
scheme = None
output_dir = None
detailed_cost_array = None


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def windows_long_path(path):
    path = Path(path)
    if os.name != "nt":
        return path
    path_text = str(path.resolve())
    if path_text.startswith("\\\\?\\"):
        return Path(path_text)
    return Path("\\\\?\\" + path_text)


def display_path(path):
    text = str(path)
    return text[4:] if text.startswith("\\\\?\\") else text


def _mat_to_struct(value):
    if hasattr(value, "_fieldnames"):
        result = Struct()
        for field in value._fieldnames:
            setattr(result, field, _mat_to_struct(getattr(value, field)))
        return result
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.vectorize(_mat_to_struct, otypes=[object])(value)
        if value.dtype.kind in "uib":
            return value.astype(np.int64, copy=False)
        return value
    if isinstance(value, np.generic) and value.dtype.kind in "uib":
        return int(value)
    return value


def load_scheme(path):
    mat = sio.loadmat(windows_long_path(path), squeeze_me=True, struct_as_record=False)
    return _mat_to_struct(mat["scheme"])


def _convert_structure(value, target="json"):
    if isinstance(value, Struct) or hasattr(value, "__dict__"):
        return {
            key: _convert_structure(item, target)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            converted = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                converted[index] = _convert_structure(value[index], target)
            return converted.tolist() if target == "json" else converted
        if target == "json":
            return _convert_structure(value.tolist(), target)
        return value
    if isinstance(value, dict):
        if target == "matlab":
            if not value:
                return np.empty((0,), dtype=object)
            keys = list(value)
            if all(isinstance(key, (int, np.integer)) and int(key) >= 0 for key in keys):
                shape = (max(int(key) for key in keys) + 1,)
                return _indexed_dict_to_matlab_array(value, shape)
            if all(
                isinstance(key, tuple)
                and key
                and all(isinstance(index, (int, np.integer)) and int(index) >= 0 for index in key)
                for key in keys
            ):
                dimensions = {len(key) for key in keys}
                if len(dimensions) == 1:
                    dimension_count = dimensions.pop()
                    shape = tuple(
                        max(int(key[axis]) for key in keys) + 1
                        for axis in range(dimension_count)
                    )
                    return _indexed_dict_to_matlab_array(value, shape)
            return {
                _matlab_field_name(key): _convert_structure(item, target)
                for key, item in value.items()
            }
        return {str(key): _convert_structure(item, target) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    elif isinstance(value, np.bool_):
        value = bool(value)
    elif isinstance(value, (list, tuple)):
        converted = [_convert_structure(item, target) for item in value]
        return converted if target == "json" else np.asarray(converted, dtype=object)
    if target == "json" and isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _indexed_dict_to_matlab_array(values, shape):
    converted = np.empty(shape, dtype=object)
    converted.fill(None)
    scalar_numeric = True
    for key, item in values.items():
        index = tuple(int(part) for part in key) if isinstance(key, tuple) else int(key)
        converted_item = _convert_structure(item, "matlab")
        converted[index] = converted_item
        scalar_numeric = scalar_numeric and isinstance(
            converted_item, (int, float, complex, np.number, bool)
        )
    if scalar_numeric and all(item is not None for item in converted.flat):
        return np.asarray(converted.tolist())
    return converted


def _matlab_field_name(key):
    text = str(key)
    if text and text[0].isalpha() and all(character.isalnum() or character == "_" for character in text):
        return text
    safe = "".join(character if character.isalnum() or character == "_" else "_" for character in text)
    return f"field_{safe}"


def _matlab_ready(value):
    return _convert_structure(value, "matlab")


def _json_ready(value):
    return _convert_structure(value, "json")


def _safe_model_attr(model, attr):
    try:
        return getattr(model, attr)
    except (gp.GurobiError, AttributeError):
        return None


def _status_name(status):
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


def _run_output_dir(run_name=DEFAULT_RUN_NAME):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = windows_long_path(GENERAL_RESULT_ROOT / run_name / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _detailed_cost_array(result):
    """Return the detailed total-cost array in the documented Excel order."""
    if hasattr(result.cost, "C_detailed_array"):
        return np.asarray(result.cost.C_detailed_array, dtype=float)
    cost = result.cost
    return np.asarray(
        [
            float(cost.C_inv),
            float(cost.C_ope),
            float(cost.C_risk),
            float(cost.C_line_inv),
            float(cost.C_node_inv),
            float(cost.C_ES_inv),
            float(cost.C_pipe_inv),
            float(cost.C_CHP_inv),
            float(cost.C_EB_inv),
            float(cost.C_BSS_inv),
            float(cost.C_ele) + float(cost.C_gas),
            float(cost.C_emi),
            float(cost.C_PV),
            float(cost.C_lack),
        ],
        dtype=float,
    )


def _print_detailed_cost_array(result):
    labels = [
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
    ]
    values = _detailed_cost_array(result)
    print("\n\u8be6\u7ec6\u6210\u672c\u6570\u7ec4\uff08\u53ef\u76f4\u63a5\u590d\u5236\u5230 Excel\uff09\uff1a", flush=True)
    print("\t".join(labels), flush=True)
    print("\t".join(f"{value:.10f}" for value in values), flush=True)
    print("array =", repr(values), flush=True)
    _print_hn_regulation_energy(result)



def _print_hn_regulation_energy(result):
    hn = getattr(result, "HN", None)
    if hn is None or not hasattr(hn, "E_RU_year") or not hasattr(hn, "E_RD_year"):
        return
    print("\n\u70ed\u7f51\u8c03\u8282\u7535\u91cf\uff08\u6309\u573a\u666f\u4ee3\u8868\u5929\u6570\u548c\u65f6\u6bb5\u957f\u5ea6\u52a0\u6743\uff0c\u5355\u4f4d\uff1akWh\uff09\uff1a", flush=True)
    print("\u5e74\u5ea6\t\u4e0a\u8c03\u603b\u7535\u91cf E_RU\t\u4e0b\u8c03\u603b\u7535\u91cf E_RD", flush=True)
    for year, (ru, rd) in enumerate(zip(hn.E_RU_year, hn.E_RD_year), start=1):
        print(f"{year}\t{float(ru):.10f}\t{float(rd):.10f}", flush=True)
    print(
        f"\u89c4\u5212\u671f\u7d2f\u8ba1\t{float(hn.E_RU_total):.10f}\t{float(hn.E_RD_total):.10f}",
        flush=True,
    )

def _cost_summary(result):
    if result is None or not hasattr(result, "cost"):
        return {}
    return _json_ready(result.cost)


def safe_gurobi_write(model, target_path):
    target_path = windows_long_path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.write(str(target_path))
        return None
    except gp.GurobiError as first_error:
        suffix = target_path.suffix or ".gurobi"
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="gurobi_write_", suffix=suffix, delete=False
            ) as temp_file:
                temp_name = temp_file.name
            model.write(temp_name)
            target_path.unlink(missing_ok=True)
            shutil.move(temp_name, target_path)
            return None
        except Exception as second_error:
            if temp_name is not None:
                try:
                    Path(temp_name).unlink(missing_ok=True)
                except OSError:
                    pass
            return f"{first_error}; fallback failed: {second_error}"
    except Exception as error:
        return str(error)




def _timed_call(timings, name, function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    elapsed = time.perf_counter() - started
    timings[name] = timings.get(name, 0.0) + elapsed
    print(f"[timing] {name}: {elapsed:.3f} s", flush=True)
    return value

def save_run_outputs(output_dir, model, result=None, scheme=None, elapsed_seconds=None, timing=None):
    metadata = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "timing": timing or {},
        "status": int(model.status),
        "status_name": _status_name(model.status),
        "sol_count": int(_safe_model_attr(model, "SolCount") or 0),
        "objective": _safe_model_attr(model, "ObjVal"),
        "objective_bound": _safe_model_attr(model, "ObjBound"),
        "mip_gap": _safe_model_attr(model, "MIPGap"),
        "runtime": _safe_model_attr(model, "Runtime"),
        "node_count": _safe_model_attr(model, "NodeCount"),
        "num_vars": model.NumVars,
        "num_binary_vars": model.NumBinVars,
        "num_integer_vars": model.NumIntVars,
        "num_constrs": model.NumConstrs,
        "num_quadratic_constrs": model.NumQConstrs,
        "quadratic_objective_nnz": model.NumQNZs,
        "params": {
            "MIPGap": model.Params.MIPGap,
            "Threads": model.Params.Threads,
            "TimeLimit": model.Params.TimeLimit,
            "NonConvex": model.Params.NonConvex,
            "FeasibilityTol": model.Params.FeasibilityTol,
            "IntFeasTol": model.Params.IntFeasTol,
            "OptimalityTol": model.Params.OptimalityTol,
        },
        "files": {"metadata": "metadata.json"},
    }
    if result is not None:
        result_path = output_dir / "result.pkl"
        scheme_path = output_dir / "scheme.pkl"
        with result_path.open("wb") as result_file:
            pickle.dump(result, result_file, protocol=pickle.HIGHEST_PROTOCOL)
        with scheme_path.open("wb") as scheme_file:
            pickle.dump(scheme, scheme_file, protocol=pickle.HIGHEST_PROTOCOL)
        metadata["files"].update(
            {"result_pickle": result_path.name, "scheme_pickle": scheme_path.name}
        )
        cost_path = output_dir / "cost_summary.json"
        with cost_path.open("w", encoding="utf-8") as cost_file:
            json.dump(_cost_summary(result), cost_file, ensure_ascii=False, indent=2)
        metadata["files"]["cost_summary"] = cost_path.name
        try:
            mat_path = output_dir / "result_scheme.mat"
            sio.savemat(
                mat_path,
                {"result": _matlab_ready(result), "scheme": _matlab_ready(scheme)},
                long_field_names=True,
                do_compression=True,
            )
            metadata["files"]["mat"] = mat_path.name
        except Exception as error:
            metadata["mat_export_error"] = str(error)
        if int(_safe_model_attr(model, "SolCount") or 0) > 0:
            solution_path = output_dir / "solution.sol"
            solution_error = safe_gurobi_write(model, solution_path)
            if solution_error is None:
                metadata["files"]["gurobi_solution"] = solution_path.name
            else:
                metadata["gurobi_solution_export_error"] = solution_error
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(_json_ready(metadata), metadata_file, ensure_ascii=False, indent=2)
    return output_dir


def save_flexibility_snapshot(output_dir, data, var, result=None):
    """Save a solver-indexed snapshot for fixed-point flexibility studies."""
    hn = data.HN
    n_year, n_scene, n_time = int(data.year), int(data.scene.N), int(data.period)
    n_station = int(hn.N_station)
    n_node = int(hn.N_node)
    n_pipe = int(hn.N_pipe)

    def values(variable, shape):
        array = np.zeros(shape, dtype=float)
        for index in np.ndindex(shape):
            array[index] = float(variable[index].X)
        return array

    arrays = {
        "chp_power": values(var.CHP.P, (n_station, n_time, n_scene, n_year)),
        "eb_power": values(var.EB.P, (n_station, n_time, n_scene, n_year)),
        "chp_power_ru": values(var.CHP.P_RU, (n_station, n_time, n_scene, n_year)),
        "chp_power_rd": values(var.CHP.P_RD, (n_station, n_time, n_scene, n_year)),
        "eb_power_ru": values(var.EB.P_RU, (n_station, n_time, n_scene, n_year)),
        "eb_power_rd": values(var.EB.P_RD, (n_station, n_time, n_scene, n_year)),
        "chp_capacity": values(var.HN.S_CHP, (n_station, n_year)),
        "eb_capacity": values(var.HN.S_EB, (n_station, n_year)),
        "t_supply_node": values(var.HN.T_S_node, (n_node, n_time, n_scene, n_year)),
        "t_return_node": values(var.HN.T_R_node, (n_node, n_time, n_scene, n_year)),
        "h_ru": values(var.HN.H_RU, (n_station, n_time, n_scene, n_year)),
        "h_rd": values(var.HN.H_RD, (n_station, n_time, n_scene, n_year)),
        "hn_power_ru_all": values(var.HN.P_RU_all, (n_time, n_scene, n_year)),
        "hn_power_rd_all": values(var.HN.P_RD_all, (n_time, n_scene, n_year)),
        "pipe_built": values(var.HN.y_pipe, (n_pipe, n_year)),
        "pipe_direction_ij": values(var.HN.b_ij, (n_pipe, n_year)),
        "pipe_direction_ji": values(var.HN.b_ji, (n_pipe, n_year)),
        "station_built": values(var.HN.y_station, (n_station, n_year)),
        "source_zone_affiliation": values(
            var.HN.source_zone_affiliation, (n_node, n_station, n_year)
        ),
    }
    arrays["fixed_load_flow"] = np.asarray(
        getattr(hn, "fixed_load_flow", np.zeros((int(hn.N_load), n_year))), dtype=float
    )
    arrays["load_node_idx"] = np.asarray(hn.load_node_idx, dtype=int)
    arrays["station_node_idx"] = np.asarray(hn.station_node_idx, dtype=int)
    arrays["pipe_head"] = np.asarray(hn.head, dtype=int)
    arrays["pipe_tail"] = np.asarray(hn.tail, dtype=int)
    arrays["gamma_call"] = np.asarray(
        getattr(hn, "gamma_call", np.zeros((n_station, n_time, n_scene, n_year))),
        dtype=float,
    )
    arrays["scene_days"] = np.asarray(data.scene.N_day, dtype=float)
    arrays["delta_t_hour"] = np.asarray(float(data.delta_t_hour))
    arrays["k_chp_heat"] = np.asarray(float(data.CHP.k_HE))
    arrays["k_eb_heat"] = np.asarray(float(data.EB.k_HE))
    arrays["c_w"] = np.asarray(float(hn.c_w))
    arrays["t_supply_min"] = np.asarray(float(hn.T_S_min))
    arrays["t_supply_max"] = np.asarray(float(hn.T_S_max))
    arrays["t_return_min"] = np.asarray(float(hn.T_R_min))
    arrays["t_return_max"] = np.asarray(float(hn.T_R_max))
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "flexibility_snapshot.npz"
    temporary_snapshot_path = output_dir / ".flexibility_snapshot.tmp.npz"
    np.savez_compressed(temporary_snapshot_path, **arrays)
    os.replace(temporary_snapshot_path, snapshot_path)
    planning_values = _extract_planning_values(var)
    with (output_dir / "planning_values.pkl").open("wb") as file:
        pickle.dump(planning_values, file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "producer": "main.py",
        "run_output_dir": str(output_dir),
        "planning_years": n_year,
        "period": n_time,
        "scene_count": n_scene,
        "scene_days": [float(value) for value in data.scene.N_day],
        "delta_t_hour": float(data.delta_t_hour),
        "hn_station_count": n_station,
        "hn_node_count": n_node,
        "hn_pipe_count": n_pipe,
        "hn_load_count": int(hn.N_load),
        "power_units": "kW",
        "heat_units": "kW",
        "temperature_units": "degC",
        "flow_units": "model_flow_unit",
        "power_sign_convention": {
            "electricity_consumption": "positive",
            "electricity_generation": "negative",
            "RU": "increase_consumption",
            "RD": "decrease_consumption",
        },
        "hn_heat_change": {
            "H_RU": "-k_CHP*P_CHP_RD + k_EB*P_EB_RU",
            "H_RD": "k_CHP*P_CHP_RU - k_EB*P_EB_RD",
            "temperature_constraints_use": "raw H_RU/H_RD; Lambda_H_up/Lambda_H_dn removed",
        },
        "files": {
            "snapshot": str(snapshot_path),
            "planning_values": str(output_dir / "planning_values.pkl"),
            "result_pickle": "result.pkl" if result is not None else None,
        },
    }
    with (output_dir / "flexibility_snapshot.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def default_scheme_path():
    project_dir = windows_long_path(Path(__file__).resolve().parent)
    search_roots = (project_dir, project_dir.parent, project_dir.parent.parent)
    candidates = []
    for root in search_roots:
        candidates.extend(root.rglob("scheme_*.mat"))
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError("Default scheme MAT file not found; pass scheme_file explicitly")


def default_parameter_path():
    project_dir = Path(__file__).resolve().parent
    candidates = [project_dir / "parameters.json"]
    candidates.extend(
        path for path in project_dir.glob("*.xlsx") if not path.name.startswith("~$")
    )
    candidates = [path for path in candidates if path.is_file()]
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(
        "Default parameter JSON/XLSX file not found; pass parameter_file explicitly"
    )


def apply_gurobi_time_limit(model, time_limit):
    if time_limit is None or float(time_limit) <= 0:
        model.Params.TimeLimit = GRB.INFINITY
        return None
    model.Params.TimeLimit = float(time_limit)
    return float(time_limit)


_LEGACY_PLANNING_VARIABLE_PATHS = (
    ("DN", "y_line"),
    ("DN", "N_line_exp"),
    ("DN", "N_node_exp"),
    ("SVC", "y"),
    ("SVC", "S_plan"),
    ("PV", "N_plan"),
    ("PV", "S_plan"),
    ("ES", "N_plan"),
    ("ES", "S_plan"),
    ("ES", "E_plan"),
    ("BSS", "N_FP_plan"),
    ("BSS", "N_SB_plan"),
    ("BSS", "N_CB_plan"),
    ("HN", "y_pipe"),
    ("HN", "y_node"),
    ("HN", "N_pipe_exp"),
    ("HN", "y_station"),
    ("HN", "N_CHP"),
    ("HN", "N_EB"),
)

CONTINUOUS_PLANNING_FIX_TOLERANCE = 1e-5

# Only these core planning decisions are fixed in stage 2. Capacity and
# operating variables derived from them remain free for the second solve.
_PLANNING_VARIABLE_PATHS = (
    ('DN', 'y_line'),
    ('DN', 'b_ij'),
    ('DN', 'b_ji'),
    ('DN', 'N_line_exp'),
    ('DN', 'N_node_exp'),
    ('HN', 'y_pipe'),
    ('HN', 'b_ij'),
    ('HN', 'b_ji'),
    ('HN', 'y_station'),
    ('HN', 'N_CHP'),
    ('HN', 'N_EB'),
    ('HN', 'N_pipe_exp'),
    ('BSS', 'N_SB_plan'),
    ('BSS', 'N_CB_plan'),
    ('ES', 'N_plan'),
)


def _rounded_planning_value(variable, item):
    value = float(item.X)
    if item.VType == GRB.BINARY:
        return float(int(round(value)))
    if item.VType == GRB.INTEGER:
        return float(int(round(value)))
    lower_bound = float(item.LB)
    upper_bound = float(item.UB)
    bound_tolerance = CONTINUOUS_PLANNING_FIX_TOLERANCE
    if math.isfinite(lower_bound) and value < lower_bound:
        if lower_bound - value <= bound_tolerance:
            value = lower_bound
    if math.isfinite(upper_bound) and value > upper_bound:
        if value - upper_bound <= bound_tolerance:
            value = upper_bound
    return float(value)


def _extract_planning_values(var):
    values = {}
    for group_name, variable_name in _PLANNING_VARIABLE_PATHS:
        group = getattr(var, group_name, None)
        variable = getattr(group, variable_name, None) if group is not None else None
        if variable is None:
            continue
        values[(group_name, variable_name)] = {
            key: _rounded_planning_value(variable, item)
            for key, item in variable.items()
        }
    return values


def _fix_planning_values(model, var, values):
    fixed_count = 0
    for (group_name, variable_name), variable_values in values.items():
        variable = getattr(getattr(var, group_name), variable_name)
        for key, value in variable_values.items():
            item = variable[key]
            model.addConstr(
                item == value,
                name=f"TwoStageFix_{group_name}_{variable_name}_{key}",
            )
            fixed_count += 1
    return fixed_count


def run_two_stage_planning(
    run_name="main_no_flexibility",
    parameter_file=None,
    scheme_file=None,
    planning_years=MAIN_PLANNING_YEARS,
    mip_gap=0.005,
    threads=8,
    time_limit=None,
    seed=1,
    quiet_gurobi=False,
    feasibility_tol=1e-7,
    solver_feasibility_tol=1e-6,
    solver_integer_feasibility_tol=1e-5,
    solver_optimality_tol=1e-6,
    hn_builder=Constraints_HN_manuscript_exact_linear,
    bss_builder=Constraints_BSS,
    flexibility_penalty_lower_bound=None,
):
    """Optimize F1 planning decisions, then F operation costs with them fixed."""
    configure_utf8_stdio()
    start_time = time.time()
    start_time_perf = time.perf_counter()
    timing = {}
    output_dir = _run_output_dir(run_name)
    globals()["result"] = None
    globals()["scheme"] = None
    globals()["output_dir"] = output_dir
    globals()["detailed_cost_array"] = None

    model = gp.Model(f"EnergySystemOptimization_{run_name}_two_stage")
    scheme_path = Path(scheme_file) if scheme_file is not None else default_scheme_path()
    scheme = load_scheme(scheme_path)
    parameter_path = (
        Path(parameter_file).expanduser().resolve()
        if parameter_file is not None
        else default_parameter_path()
    )
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {parameter_path}")

    data, var = _timed_call(timing, "data_read", Data_read,
        model,
        filepath=str(parameter_path),
        planning_years=planning_years,
        create_dense_bss_transitions=True,
        hn_linearization_mode="physical_exact",
    )
    globals()["data"] = data
    data.HN.direct_contract_tube_qconstrs = False
    _timed_call(timing, "constraints_DN", Constraints_DN, model, data, var, scheme)
    _timed_call(timing, "constraints_ES", Constraints_ES, model, data, var, scheme)
    _timed_call(timing, "constraints_PV", Constraints_PV, model, data, var)
    _timed_call(timing, "constraints_BSS", bss_builder, model, data, var, scheme)
    _timed_call(timing, "constraints_HN", hn_builder, model, data, var, scheme)
    objective = _timed_call(timing, "objective_build", object, model, data, var)

    model.Params.LogToConsole = 0 if quiet_gurobi else 1
    model.Params.MIPGap = float(mip_gap)
    model.Params.TuneTimeLimit = 0
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)
    model.Params.NonConvex = 0
    model.Params.FeasibilityTol = float(solver_feasibility_tol)
    model.Params.IntFeasTol = float(solver_integer_feasibility_tol)
    model.Params.OptimalityTol = float(solver_optimality_tol)
    apply_gurobi_time_limit(model, time_limit)
    _timed_call(timing, "model_update", model.update)

    model.setObjective(objective.F1, GRB.MINIMIZE)
    timing["pre_optimize_seconds"] = time.perf_counter() - start_time_perf
    print("Starting stage 1 F1 planning solve...", flush=True)
    _timed_call(timing, "gurobi_optimize", model.optimize)
    if model.status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.reset()
        _timed_call(timing, "gurobi_optimize", model.optimize)
    if model.status != GRB.OPTIMAL and model.SolCount <= 0:
        raise RuntimeError(f"Stage 1 F1 solve failed with status {model.status}")
    stage1_objective = float(model.ObjVal)
    planning_values = _extract_planning_values(var)
    with (output_dir / "stage1_planning_values.pkl").open("wb") as values_file:
        pickle.dump(planning_values, values_file, protocol=pickle.HIGHEST_PROTOCOL)
    safe_gurobi_write(model, output_dir / "stage1_F1_solution.sol")

    fixed_count = _fix_planning_values(model, var, planning_values)
    model.setObjective(objective.F, GRB.MINIMIZE)
    if flexibility_penalty_lower_bound is not None:
        model.addConstr(
            objective.C_lack >= float(flexibility_penalty_lower_bound),
            name="Stage2_FlexibilityPenalty_LowerBound",
        )
    _timed_call(timing, "model_update", model.update)
    print(f"Starting stage 2 F solve with {fixed_count} fixed planning values...", flush=True)
    _timed_call(timing, "gurobi_optimize", model.optimize)
    if model.status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.reset()
        _timed_call(timing, "gurobi_optimize", model.optimize)
    if model.status != GRB.OPTIMAL and model.SolCount <= 0:
        diagnostic = {"status": int(model.status), "fixed_planning_variable_count": fixed_count}
        try:
            model.Params.IISMethod = 1
            model.Params.TimeLimit = 60.0
            model.computeIIS()
            diagnostic["iis_minimal"] = bool(model.IISMinimal)
            diagnostic["write_error"] = safe_gurobi_write(
                model, output_dir / "stage2_conflict_model.ilp"
            )
        except gp.GurobiError as error:
            diagnostic["error"] = str(error)
        with (output_dir / "stage2_iis_diagnostic.json").open("w", encoding="utf-8") as diagnostic_file:
            json.dump(_json_ready(diagnostic), diagnostic_file, ensure_ascii=False, indent=2)
        raise RuntimeError(f"Stage 2 F solve failed with status {model.status}")

    result, export_scheme = Data_export(data, var, objective)
    result.two_stage = Struct()
    result.two_stage.stage1_objective = stage1_objective
    result.two_stage.stage2_objective = float(model.ObjVal)
    result.two_stage.stage1_objective_name = "F1"
    result.two_stage.stage2_objective_name = "F"
    result.two_stage.flexibility_penalty_lower_bound = (
        None
        if flexibility_penalty_lower_bound is None
        else float(flexibility_penalty_lower_bound)
    )
    result.two_stage.fixed_planning_variable_count = fixed_count
    result.two_stage.planning_values_file = "stage1_planning_values.pkl"
    result.BSS.integer_tube["formulation"] = bss_builder.__name__

    globals()["result"] = result
    globals()["scheme"] = export_scheme
    globals()["detailed_cost_array"] = _detailed_cost_array(result)
    _print_detailed_cost_array(result)
    save_run_outputs(
        output_dir,
        model,
        result=result,
        scheme=export_scheme,
        elapsed_seconds=time.time() - start_time,
        timing=timing,
    )
    with (output_dir / "two_stage_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(
            _json_ready(
                {
                    "stage1_objective": stage1_objective,
                    "stage2_objective": float(model.ObjVal),
                    "stage1_objective_name": "F1",
                    "stage2_objective_name": "F",
                    "flexibility_penalty_lower_bound": (
                        None
                        if flexibility_penalty_lower_bound is None
                        else float(flexibility_penalty_lower_bound)
                    ),
                    "timing": timing,
                    "fixed_planning_variable_count": fixed_count,
                    "planning_years": int(data.year),
                }
            ),
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )
    print("Two-stage result saved to:", display_path(output_dir), flush=True)
    return output_dir


def run_planning(
    run_name=DEFAULT_RUN_NAME,
    parameter_file=None,
    scheme_file=None,
    planning_years=MAIN_PLANNING_YEARS,
    mip_gap=0.005,
    threads=8,
    time_limit=None,
    seed=1,
    quiet_gurobi=False,
    feasibility_tol=1e-7,
    solver_feasibility_tol=1e-6,
    solver_integer_feasibility_tol=1e-5,
    solver_optimality_tol=1e-6,
    build_only=False,
    fixed_planning_values=None,
    objective_field="F",
    hn_builder=Constraints_HN_manuscript_exact_linear,
    bss_builder=Constraints_BSS,
    dn_builder=Constraints_DN,
):
    """Build and solve the physical exact-linear heat-network MILP."""
    configure_utf8_stdio()
    start_time = time.time()
    start_time_perf = time.perf_counter()
    timing = {}
    output_dir = _run_output_dir(run_name)
    globals()["result"] = None
    globals()["scheme"] = None
    globals()["output_dir"] = output_dir
    globals()["detailed_cost_array"] = None
    model = gp.Model(f"EnergySystemOptimization_{run_name}")
    scheme_path = Path(scheme_file) if scheme_file is not None else default_scheme_path()
    scheme = load_scheme(scheme_path)
    parameter_path = (
        Path(parameter_file).expanduser().resolve()
        if parameter_file is not None
        else default_parameter_path()
    )
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {parameter_path}")

    data, var = _timed_call(timing, "data_read", Data_read,
        model,
        filepath=str(parameter_path),
        planning_years=planning_years,
        create_dense_bss_transitions=True,
        hn_linearization_mode="physical_exact",
    )
    globals()["data"] = data
    data.HN.direct_contract_tube_qconstrs = False
    _timed_call(timing, "constraints_DN", dn_builder, model, data, var, scheme)
    _timed_call(timing, "constraints_ES", Constraints_ES, model, data, var, scheme)
    _timed_call(timing, "constraints_PV", Constraints_PV, model, data, var)
    _timed_call(timing, "constraints_BSS", bss_builder, model, data, var, scheme)
    _timed_call(timing, "constraints_HN", hn_builder, model, data, var, scheme)

    objective = _timed_call(timing, "objective_build", object, model, data, var)
    if objective_field not in {"F", "F1"}:
        raise ValueError(f"unsupported objective_field: {objective_field}")
    fixed_planning_count = 0
    if fixed_planning_values is not None:
        fixed_planning_count = _fix_planning_values(model, var, fixed_planning_values)
        print(f"Fixed planning solution loaded: {fixed_planning_count} planning variables.", flush=True)
    model.setObjective(getattr(objective, objective_field), GRB.MINIMIZE)
    model.Params.LogToConsole = 0 if quiet_gurobi else 1
    model.Params.MIPGap = float(mip_gap)
    model.Params.TuneTimeLimit = 0
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)
    model.Params.NonConvex = 0
    model.Params.FeasibilityTol = float(solver_feasibility_tol)
    model.Params.IntFeasTol = float(solver_integer_feasibility_tol)
    model.Params.OptimalityTol = float(solver_optimality_tol)
    apply_gurobi_time_limit(model, time_limit)
    _timed_call(timing, "model_update", model.update)

    exact_config = {
        "formulation": hn_builder.__name__,
        "dn_formulation": dn_builder.__name__,
        "include_bss_in_grid_reserve": bool(getattr(dn_builder, "include_bss_reserve", True)),
        "planning_years": int(data.year),
        "linearization_mode": data.HN.linearization_mode,
        "nonconvex_parameter": int(model.Params.NonConvex),
        "num_quadratic_constrs": int(model.NumQConstrs),
        "quadratic_objective_nnz": int(model.NumQNZs),
        "mccormick_envelopes_enabled": bool(
            getattr(data.HN, "mccormick_envelopes_enabled", False)
        ),
        "piecewise_partitions_enabled": bool(
            getattr(data.HN, "piecewise_partitions_enabled", False)
        ),
        "fixed_planning_values": fixed_planning_values is not None,
        "fixed_planning_variable_count": int(fixed_planning_count),
    }
    exact_config["model_structure_valid"] = bool(
        exact_config["linearization_mode"] == "physical_exact"
        and exact_config["nonconvex_parameter"] == 0
        and exact_config["num_quadratic_constrs"] == 0
        and exact_config["quadratic_objective_nnz"] == 0
        and not exact_config["mccormick_envelopes_enabled"]
        and not exact_config["piecewise_partitions_enabled"]
    )
    with (output_dir / "exact_linear_model_config.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(exact_config, config_file, ensure_ascii=False, indent=2)
    if not exact_config["model_structure_valid"]:
        raise RuntimeError(f"Invalid exact-linear model structure: {exact_config}")

    if build_only:
        save_run_outputs(
            output_dir,
            model,
            elapsed_seconds=time.time() - start_time,
            timing=timing,
        )
        print(
            f"Exact-linear build complete: vars={model.NumVars}, "
            f"binaries={model.NumBinVars}, constrs={model.NumConstrs}",
            flush=True,
        )
        return output_dir

    timing["pre_optimize_seconds"] = time.perf_counter() - start_time_perf
    print(
        f"Starting exact-linear Gurobi solve for {planning_years} years... "
        f"threads={model.Params.Threads}, mip_gap={model.Params.MIPGap}",
        flush=True,
    )
    _timed_call(timing, "gurobi_optimize", model.optimize)
    if model.status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.reset()
        _timed_call(timing, "gurobi_optimize", model.optimize)

    result = None
    export_scheme = None
    if model.status == GRB.OPTIMAL or model.SolCount > 0:
        result, export_scheme = Data_export(data, var, objective)
        result.BSS.integer_tube["formulation"] = bss_builder.__name__
        accuracy = {
            "evidence_type": "physical_exact_linear_solution",
            "planning_years": int(data.year),
            "verified_feasible": bool(float(model.MaxVio) <= feasibility_tol),
            "max_constraint_violation": float(model.MaxVio),
            "objective": float(model.ObjVal),
            "objective_bound": float(model.ObjBound),
            "mip_gap": float(model.MIPGap),
            "num_quadratic_constrs": int(model.NumQConstrs),
        }
        with (output_dir / "exact_linear_accuracy.json").open(
            "w", encoding="utf-8"
        ) as accuracy_file:
            json.dump(accuracy, accuracy_file, ensure_ascii=False, indent=2)
        globals()["result"] = result
        globals()["scheme"] = export_scheme
        globals()["detailed_cost_array"] = _detailed_cost_array(result)
        _print_detailed_cost_array(result)
        print(
            "\u63a7\u5236\u53f0\u53d8\u91cf\u5df2\u751f\u6210: result, scheme, output_dir, detailed_cost_array",
            flush=True,
        )
    elif model.status == GRB.INFEASIBLE:
        diagnostic = {"completed": False, "error": None}
        try:
            model.Params.IISMethod = 1
            model.Params.TimeLimit = 60.0
            model.computeIIS()
            diagnostic["completed"] = True
            diagnostic["iis_minimal"] = bool(model.IISMinimal)
            diagnostic["write_error"] = safe_gurobi_write(
                model, output_dir / "conflict_model.ilp"
            )
        except gp.GurobiError as error:
            diagnostic["error"] = str(error)
        with (output_dir / "iis_diagnostic.json").open(
            "w", encoding="utf-8"
        ) as diagnostic_file:
            json.dump(diagnostic, diagnostic_file, ensure_ascii=False, indent=2)

    save_run_outputs(
        output_dir,
        model,
        result=result,
        scheme=export_scheme,
        elapsed_seconds=time.time() - start_time,
        timing=timing,
    )
    if result is not None:
        save_flexibility_snapshot(output_dir, data, var, result=result)
    print("\u8fd0\u884c\u7ed3\u679c\u5df2\u4fdd\u5b58\u81f3:", display_path(output_dir), flush=True)
    return output_dir


def main():
    return run_planning()


if __name__ == "__main__":
    main()
