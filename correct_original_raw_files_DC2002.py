import math
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


# ============================================================
# Utility / indexing
# ============================================================

def SUP_getIdx():
    idxB = {
        "run": 1,
        "hr": 2,
        "min": 3,
        "sec": 4,
        "AoA": 5,
        "AoS": 6,
        "dPb": 7,
        "pBar": 8,
        "temp": 9,
        "B": list(range(10, 16)),
        "B1": 10,
        "B2": 11,
        "B3": 12,
        "B4": 13,
        "B5": 14,
        "B6": 15,
        "rpmWT": 16,
        "rho": 17,
        "q": 18,
        "V": 19,
        "Re": 20,
        "rpsM1": 21,
        "rpsM2": 22,
        "iM1": 23,
        "iM2": 24,
        "dPtQ": 25,
        "tM1": 26,
        "tM2": 27,
        "vM1": 28,
        "vM2": 29,
        "zero": {
            "run": 1,
            "hr": 2,
            "min": 3,
            "sec": 4,
            "AoA": 5,
            "AoS": 6,
            "pBar": 7,
            "temp": 8,
            "B": list(range(9, 15)),
            "B1": 9,
            "B2": 10,
            "B3": 11,
            "B4": 12,
            "B5": 13,
            "B6": 14,
        }
    }
    return idxB


# ============================================================
# Parsing helpers
# ============================================================

def _parse_time_token(time_token: str):
    parts = time_token.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time token: {time_token}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _read_whitespace_table(filepath: Union[str, Path], expected_numeric_after_time: int) -> np.ndarray:
    filepath = Path(filepath)
    rows = []

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    for line in lines[2:]:
        s = line.strip()
        if not s:
            continue

        parts = s.split()
        if len(parts) < 2:
            continue

        run_token = parts[0]
        time_token = parts[1]
        rest = parts[2:]

        if len(rest) != expected_numeric_after_time:
            raise ValueError(
                f"Unexpected number of numeric fields in {filepath.name}. "
                f"Expected {expected_numeric_after_time}, got {len(rest)}.\n"
                f"Line: {line}"
            )

        row = [float(run_token)] + _parse_time_token(time_token) + [float(x) for x in rest]
        rows.append(row)

    return np.asarray(rows, dtype=float)


def _matlab_to_py_col(idx: int) -> int:
    return idx - 1


def _take_col(a: np.ndarray, idx: int) -> np.ndarray:
    return a[:, _matlab_to_py_col(idx)]


def _take_cols(a: np.ndarray, idxs: List[int]) -> np.ndarray:
    return a[:, [i - 1 for i in idxs]]


# ============================================================
# Tunnel operating conditions
# ============================================================

def SUP_LTTq(dPb: np.ndarray, testSec: int) -> np.ndarray:
    dPb = np.asarray(dPb, dtype=float)

    if testSec == 5:
        dPbLim = [100.0, 300.0]
        facQpB = np.array([
            [0.51549, 2.32312, 4.62743e-05, 0.0],
            [0.51549, 2.32312, 4.62743e-05, 0.0],
            [0.51549, 2.32312, 4.62743e-05, 0.0],
        ], dtype=float)
    elif testSec == 7:
        dPbLim = [50.0, 400.0]
        facQpB = np.array([
            [0.909286, 2.44320, -3.79098e-05, 0.0],
            [0.402538, 2.45530, -5.74383e-06, 0.0],
            [3.101380, 2.42426, 5.22943e-05, 0.0],
        ], dtype=float)
    elif testSec == 9:
        dPbLim = [100.0, 5000.0]
        facQpB = np.array([
            [0.0193846, 2.33053, 1.72980e-4, 0.0],
            [0.1903500, 2.33534, 5.07273e-5, 0.0],
            [0.1903500, 2.33534, 5.07273e-5, 0.0],
        ], dtype=float)
    else:
        raise ValueError("No dynamic-pressure calibration data included yet for this test section.")

    qInf = np.zeros_like(dPb, dtype=float)
    for i, val in enumerate(dPb):
        if val < dPbLim[0]:
            idxQ = 0
        elif val < dPbLim[1]:
            idxQ = 1
        else:
            idxQ = 2

        coeffs = facQpB[idxQ]
        qInf[i] = coeffs[0] + coeffs[1] * val + coeffs[2] * val**2 + coeffs[3] * val**3

    return qInf


def SUP_LTTgetOper_BAL(DAT: Dict[str, np.ndarray], testSec: int) -> Dict[str, np.ndarray]:
    oper = {}
    oper["dPb"] = DAT["dPb"]
    oper["tInf"] = DAT["temp"] + 273.15
    oper["pBar"] = DAT["pBar"] * 100.0
    oper["AoA"] = DAT["AoA"]

    oper["qInf"] = SUP_LTTq(oper["dPb"], testSec)
    oper["pInf"] = oper["pBar"]
    oper["rho"] = oper["pInf"] / (oper["tInf"] * 287.05)
    oper["vInf"] = np.sqrt(2.0 * oper["qInf"] / oper["rho"])

    mu = 1.716e-5 * (oper["tInf"] / 273.15) ** 1.5 * (273.15 + 110.4) / (oper["tInf"] + 110.4)
    oper["nu"] = mu / oper["rho"]
    return oper


# ============================================================
# Reading files
# ============================================================

def BAL_readData(filepath: Union[str, Path], idxB: Dict[str, Any]) -> Dict[str, np.ndarray]:
    read_data = _read_whitespace_table(filepath, expected_numeric_after_time=31)

    DATA = {
        "run": _take_col(read_data, idxB["run"]),
        "hr": _take_col(read_data, idxB["hr"]),
        "min": _take_col(read_data, idxB["min"]),
        "sec": _take_col(read_data, idxB["sec"]),
        "AoA": _take_col(read_data, idxB["AoA"]),
        "AoS": _take_col(read_data, idxB["AoS"]),
        "dPb": _take_col(read_data, idxB["dPb"]),
        "pBar": _take_col(read_data, idxB["pBar"]),
        "temp": _take_col(read_data, idxB["temp"]),
        "B": _take_cols(read_data, idxB["B"]),
        "B1": _take_col(read_data, idxB["B1"]),
        "B2": _take_col(read_data, idxB["B2"]),
        "B3": _take_col(read_data, idxB["B3"]),
        "B4": _take_col(read_data, idxB["B4"]),
        "B5": _take_col(read_data, idxB["B5"]),
        "B6": _take_col(read_data, idxB["B6"]),
        "rpmWT": _take_col(read_data, idxB["rpmWT"]),
        "rho": _take_col(read_data, idxB["rho"]),
        "q": _take_col(read_data, idxB["q"]),
        "V": _take_col(read_data, idxB["V"]),
        "Re": _take_col(read_data, idxB["Re"]),
        "rpsM1": _take_col(read_data, idxB["rpsM1"]),
        "rpsM2": _take_col(read_data, idxB["rpsM2"]),
        "iM1": _take_col(read_data, idxB["iM1"]),
        "iM2": _take_col(read_data, idxB["iM2"]),
        "dPtQ": _take_col(read_data, idxB["dPtQ"]),
        "tM1": _take_col(read_data, idxB["tM1"]),
        "tM2": _take_col(read_data, idxB["tM2"]),
        "vM1": _take_col(read_data, idxB["vM1"]),
        "vM2": _take_col(read_data, idxB["vM2"]),
    }
    return DATA


def BAL_read0data(folder: Union[str, Path], fn: Union[str, List[str]], idxB: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if not isinstance(fn, list):
        fn = [fn]

    zero_idx = idxB["zero"]
    aggregated = []

    for one_fn in fn:
        filepath = Path(folder) / one_fn
        raw = _read_whitespace_table(filepath, expected_numeric_after_time=10)

        AoA_round = np.round(_take_col(raw, zero_idx["AoA"]) * 20) / 20
        AoS_round = np.round(_take_col(raw, zero_idx["AoS"]) * 20) / 20
        angles = np.column_stack([AoA_round, AoS_round])

        unique_angles, inverse = np.unique(angles, axis=0, return_inverse=True)

        avg_rows = []
        for j in range(unique_angles.shape[0]):
            avg_rows.append(raw[inverse == j].mean(axis=0))
        avg = np.vstack(avg_rows)

        aggregated.append(avg)

    if len(aggregated) != 1:
        avg_all = np.vstack(aggregated)
    else:
        avg_all = aggregated[0]

    BAL0 = {
        "run": _take_col(avg_all, zero_idx["run"]),
        "hr": _take_col(avg_all, zero_idx["hr"]),
        "min": _take_col(avg_all, zero_idx["min"]),
        "sec": _take_col(avg_all, zero_idx["sec"]),
        "AoA": _take_col(avg_all, zero_idx["AoA"]),
        "AoS": _take_col(avg_all, zero_idx["AoS"]),
        "pBar": _take_col(avg_all, zero_idx["pBar"]),
        "temp": _take_col(avg_all, zero_idx["temp"]),
        "B1": _take_col(avg_all, zero_idx["B1"]),
        "B2": _take_col(avg_all, zero_idx["B2"]),
        "B3": _take_col(avg_all, zero_idx["B3"]),
        "B4": _take_col(avg_all, zero_idx["B4"]),
        "B5": _take_col(avg_all, zero_idx["B5"]),
        "B6": _take_col(avg_all, zero_idx["B6"]),
    }
    return BAL0


# ============================================================
# Calibration data
# ============================================================

def BAL_getCalFactors():
    p = np.array([0.009200961, 0.01594834, 0.06134184, 0.06143589, 0.02461131, 0.01231626], dtype=float)

    pnl = np.array([
        [1.0,          -0.4414104e-03, -0.8049510e-04, -0.1115902e-03, -0.1456587e-03, -0.6872960e-03],
        [0.1102029e-03, 1.0,           -0.1138073e-03,  0.5521437e-04,  0.0,            0.0],
        [-0.2580602e-03,-0.3546250e-03, 1.0,            0.0,            0.0,            0.0],
        [-0.2580602e-03,-0.3546250e-03, 0.0,            1.0,            0.0,            0.0],
        [0.0,            0.0,           0.0,            0.0,            1.0,           -0.1122973e-02],
        [0.0,            0.0,           0.0,            0.0,            0.1084830e-03,  1.0],
    ], dtype=float)

    arm = np.array([
        0.001923, 0.000250, -0.000209, -0.525748, -0.525765, 0.000114,
        -0.262875, 0.263078, -0.000975, -1.050331, -1.050106, -1.049434
    ], dtype=float)

    FX_cor = 0.0
    x_bend = 0.0
    y_bend = 0.0

    eij_flat = np.array([
         0.5101e-05,  0.5101e-05,  0.5101e-05,  0.5101e-05,  0.5101e-05,
        -0.1177e-06, -0.1177e-06, -0.1177e-06, -0.3087e-06, -0.2787e-06,
        -0.2787e-06, -0.6137e-06, -0.6611e-05, -0.6420e-07, -0.1258e-05,
        -0.6611e-05, -0.6611e-05, -0.2430e-07, -0.2430e-07, -0.2430e-07,
        -0.1218e-05, -0.1021e-05, -0.1021e-05,  0.6503e-05,  0.0000e-00,
         0.0000e-00,  0.0000e-00,  0.0000e-00,  0.0000e-00, -0.4002e-07,
        -0.4002e-07, -0.4002e-07, -0.4002e-07,  0.5536e-06,  0.5536e-06,
         0.9792e-06,  0.0000e-00,  0.0000e-00,  0.0000e-00,  0.0000e-00,
         0.0000e-00, -0.4002e-07, -0.4002e-07, -0.4002e-07, -0.4002e-07,
         0.5536e-06,  0.5536e-06,  0.1037e-05,  0.1750e-07,  0.1750e-07,
         0.1750e-07,  0.1750e-07,  0.1750e-07, -0.4417e-07, -0.4417e-07,
        -0.4417e-07, -0.4417e-07, -0.3596e-06, -0.3596e-06, -0.1653e-05,
        -0.4600e-06, -0.1600e-05,  0.6800e-06, -0.4600e-06, -0.4600e-06,
         0.1991e-05,  0.1991e-05,  0.1991e-05,  0.4293e-05, -0.1352e-05,
        -0.1352e-05, -0.1372e-08
    ], dtype=float)

    e = np.reshape(eij_flat, (12, 6), order="F")

    return p, pnl, arm, FX_cor, x_bend, y_bend, e


# ============================================================
# Zero subtraction
# ============================================================

def _round_angle(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x) * 20.0) / 20.0


def _poly22_predict(x_train, y_train, z_train, x_pred, y_pred):
    A = np.column_stack([
        np.ones_like(x_train),
        x_train,
        y_train,
        x_train**2,
        x_train * y_train,
        y_train**2
    ])
    coeffs, *_ = np.linalg.lstsq(A, z_train, rcond=None)

    Ap = np.column_stack([
        np.ones_like(x_pred),
        x_pred,
        y_pred,
        x_pred**2,
        x_pred * y_pred,
        y_pred**2
    ])
    return Ap @ coeffs


def BAL_zero(BAL: Dict[str, Any], BAL0: Dict[str, Any], idxB: Dict[str, Any]) -> Dict[str, Any]:
    AoA0 = _round_angle(BAL0["AoA"])
    AoS0 = _round_angle(BAL0["AoS"])

    AoA0unique = np.unique(AoA0)
    AoS0unique = np.unique(AoS0)

    if len(AoA0unique) >= 1 and len(AoS0unique) == 1:
        attMode = "AoA"
        angle0 = AoA0
        angleOther = AoS0
        angle0meas = _round_angle(BAL["AoA"])
        angleOtherMeas = _round_angle(BAL["AoS"])
    elif len(AoS0unique) > 1 and len(AoA0unique) == 1:
        attMode = "AoS"
        angle0 = AoS0
        angleOther = AoA0
        angle0meas = _round_angle(BAL["AoS"])
        angleOtherMeas = _round_angle(BAL["AoA"])
    else:
        attMode = "both"
        angle0 = AoA0
        angleOther = AoS0
        angle0meas = _round_angle(BAL["AoA"])
        angleOtherMeas = _round_angle(BAL["AoS"])

    bcols = ["B1", "B2", "B3", "B4", "B5", "B6"]
    B0_interp = np.zeros((len(BAL["AoA"]), 6), dtype=float)

    if len(np.unique(angle0)) == 1:
        if not (
            np.all(np.isin(_round_angle(angle0meas), _round_angle(angle0))) and
            np.all(np.isin(_round_angle(angleOtherMeas), _round_angle(angleOther)))
        ):
            raise ValueError("Angle of incidence of zero measurement is different than the one of the measurement files.")

        for j, col in enumerate(bcols):
            B0_interp[:, j] = np.mean(BAL0[col])

    elif len(np.unique(angle0)) > 1:
        if attMode != "both":
            if not np.all(np.isin(_round_angle(angleOtherMeas), _round_angle(angleOther))):
                raise ValueError("Mismatch in angle of incidence between zero file and measurement data.")

            sort_idx = np.argsort(angle0)
            x = angle0[sort_idx]
            for j, col in enumerate(bcols):
                y = np.asarray(BAL0[col])[sort_idx]
                B0_interp[:, j] = np.interp(angle0meas, x, y, left=np.nan, right=np.nan)
        else:
            for i in range(len(angle0meas)):
                idx = (angle0 == angle0meas[i]) & (angleOther == angleOtherMeas[i])
                if np.any(idx):
                    for j, col in enumerate(bcols):
                        B0_interp[i, j] = np.mean(np.asarray(BAL0[col])[idx])
                else:
                    B0_interp[i, :] = np.nan

            if np.isnan(B0_interp).any():
                x_train = np.asarray(angle0, dtype=float)
                y_train = np.asarray(angleOther, dtype=float)
                x_pred = np.asarray(angle0meas, dtype=float)
                y_pred = np.asarray(angleOtherMeas, dtype=float)

                for j, col in enumerate(bcols):
                    z_train = np.asarray(BAL0[col], dtype=float)
                    pred = _poly22_predict(x_train, y_train, z_train, x_pred, y_pred)
                    nanmask = np.isnan(B0_interp[:, j])
                    B0_interp[nanmask, j] = pred[nanmask]

    BAL["BAL0_intp"] = B0_interp
    BAL["B16zeroed"] = BAL["B"] - B0_interp
    return BAL


# ============================================================
# Balance calibration
# ============================================================

def BAL_cal(bal, p, pnl, arm, FX_cor, x_bend, y_bend, e):
    bal = np.asarray(bal, dtype=float)

    f = pnl @ (p * bal)

    F = np.zeros(3, dtype=float)
    M = np.zeros(3, dtype=float)

    F[0] = f[0]
    F[1] = f[1] + f[5]
    F[2] = f[2] + f[3] + f[4]

    F[0] = F[0] + FX_cor * F[2]

    arm_new = arm + (e @ f)

    M[0] = -f[1] * arm_new[10] - f[5] * arm_new[11] + f[2] * arm_new[6] + f[3] * arm_new[7] + f[4] * arm_new[8]
    M[1] = +f[0] * arm_new[9]  - f[2] * arm_new[1]  - f[3] * arm_new[2] - f[4] * arm_new[3]
    M[2] = -f[0] * arm_new[5]  + f[1] * arm_new[0]  + f[5] * arm_new[4]

    M[0] = M[0] - F[2] * F[1] * y_bend
    M[1] = M[1] + F[2] * F[0] * x_bend
    M[2] = M[2] + F[1] * F[0] * y_bend - F[0] * F[1] * x_bend

    return F, M


# ============================================================
# Force / coefficient computation
# ============================================================

def BAL_forces(
    BAL: Dict[str, Any],
    BAL0: Dict[str, Any],
    idxB: Dict[str, Any],
    D: float,
    S: float,
    b: float,
    c: float,
    XmRefB: List[float],
    XmRefM: List[float],
    dAoA: float,
    dAoS: float,
    modelType: str,
    modelPos: str,
    testSec: int,
):
    oper = SUP_LTTgetOper_BAL(BAL, testSec)

    BAL["q"] = oper["qInf"]
    BAL["rho"] = oper["rho"]
    BAL["V"] = oper["vInf"]
    BAL["temp"] = oper["tInf"]
    BAL["pInf"] = oper["pInf"]
    BAL["pBar"] = oper["pBar"]
    BAL["nu"] = oper["nu"]

    with np.errstate(divide="ignore", invalid="ignore"):
        BAL["J_M1"] = BAL["V"] / (BAL["rpsM1"] * D)
        BAL["J_M2"] = BAL["V"] / (BAL["rpsM2"] * D)
    BAL["Re"] = BAL["V"] * c / BAL["nu"]

    p, pnl, arm, FX_cor, x_bend, y_bend, e = BAL_getCalFactors()

    BAL = BAL_zero(BAL, BAL0, idxB)

    n = BAL["B16zeroed"].shape[0]
    F = np.zeros((n, 3), dtype=float)
    M = np.zeros((n, 3), dtype=float)

    for i in range(n):
        F[i], M[i] = BAL_cal(BAL["B16zeroed"][i], p, pnl, arm, FX_cor, x_bend, y_bend, e)

    CF = np.zeros_like(F)
    CM = np.zeros_like(M)

    CF[:, 0] = F[:, 0] / (oper["qInf"] * S)
    CF[:, 1] = F[:, 1] / (oper["qInf"] * S)
    CF[:, 2] = F[:, 2] / (oper["qInf"] * S)

    CM[:, 0] = M[:, 0] / (oper["qInf"] * S * b)
    CM[:, 1] = M[:, 1] / (oper["qInf"] * S * c)
    CM[:, 2] = M[:, 2] / (oper["qInf"] * S * b)

    AoA = BAL["AoA"] - dAoA
    AoS = BAL["AoS"] - dAoS
    BAL["AoA"] = AoA
    BAL["AoS"] = AoS

    cosA = np.cos(np.deg2rad(AoA))
    sinA = np.sin(np.deg2rad(AoA))
    cosS = np.cos(np.deg2rad(AoS))
    sinS = np.sin(np.deg2rad(AoS))

    if modelType.lower() in ["aircraft", "3dwing"]:
        CFt = CF[:, 0] * cosA - CF[:, 2] * sinA
        CFn = CF[:, 2] * cosA + CF[:, 0] * sinA
        CFs = -CF[:, 1]

        CMr = CM[:, 0] - XmRefB[1] * CF[:, 2] + XmRefB[2] * CF[:, 1]
        CMp = CM[:, 1] - XmRefB[2] * CF[:, 0] + XmRefB[0] * CF[:, 2]
        CMy = CM[:, 2] + XmRefB[1] * CF[:, 0] - XmRefB[0] * CF[:, 1]

        if True:
            CMr_tmp = CMr.copy()
            CMr = CMr * cosA - CMy * sinA
            CMy = CMy * cosA + CMr_tmp * sinA
        else:
            CMr = CMr * cosA - CMy * sinA
            CMy = CMy * cosA + CMr * sinA
        
        if modelPos.lower() == "normal":
            CFt = +CF[:, 0] * cosA + CF[:, 2] * sinA
            CFn = -CF[:, 2] * cosA + CF[:, 0] * sinA
            CMr = -CMr
            CMp = -CMp
            CMy = -CMy

    elif modelType.lower() == "halfwing":
        CFt = CF[:, 0]
        CFn = CF[:, 1]
        CFs = CF[:, 2]

        CMr = CM[:, 0] + XmRefB[2] * CF[:, 1] * (c / b) - XmRefB[1] * CF[:, 2] * (c / b)
        CMp = -CM[:, 2] * (c / b) - XmRefB[1] * CF[:, 0] + XmRefB[0] * CF[:, 1]
        CMy = CM[:, 1] * (c / b) + XmRefB[0] * CF[:, 2] * (c / b) - XmRefB[2] * CF[:, 0] * (c / b)

        if modelPos.lower() == "inverted":
            CFn = -CFn
            CMr = -CMr
            CMp = -CMp
            CMy = -CMy
    else:
        raise ValueError(f"Unsupported modelType: {modelType}")

    CFl = CFn * cosA - CFt * sinA
    CFd = (CFn * sinA + CFt * cosA) * cosS + CFs * sinS
    CFyaw = -(CFn * sinA + CFt * cosA) * sinS + CFs * cosS
    CMp25c = CMp + CFn * (0.25 - XmRefM[0]) - CFt * (0.0 - XmRefM[2])

    BAL["FX"] = F[:, 0]
    BAL["FY"] = F[:, 1]
    BAL["FZ"] = F[:, 2]
    BAL["MX"] = M[:, 0]
    BAL["MY"] = M[:, 1]
    BAL["MZ"] = M[:, 2]

    BAL["CFX"] = CF[:, 0]
    BAL["CFY"] = CF[:, 1]
    BAL["CFZ"] = CF[:, 2]
    BAL["CMX"] = CM[:, 0]
    BAL["CMY"] = CM[:, 1]
    BAL["CMZ"] = CM[:, 2]

    BAL["CN"] = CFn
    BAL["CT"] = CFt
    BAL["CY"] = CFs

    BAL["CL"] = CFl
    BAL["CD"] = CFd
    BAL["CYaw"] = CFyaw

    BAL["CMroll"] = CMr
    BAL["CMpitch"] = CMp
    BAL["CMpitch25c"] = CMp25c
    BAL["CMyaw"] = CMy

    BAL["b"] = b
    BAL["c"] = c
    BAL["S"] = S

    return BAL


# ============================================================
# Main process wrapper
# ============================================================

def _sanitize_config_name(raw_filename: str) -> str:
    name = Path(raw_filename).stem
    if name.startswith("raw_"):
        name = name[4:]

    if not name:
        raise ValueError("Empty config name after stripping 'raw_'.")

    first = name[0]
    if first.isalpha():
        pass
    elif first == "+":
        name = "p" + name[1:]
    elif first == "-":
        name = "m" + name[1:]
    else:
        raise ValueError(
            "Unexpected character used as first character of filename. "
            "Rename the file or extend the sanitizing logic."
        )

    name = name.replace("+", "p").replace("-", "n")
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return name


def BAL_process(
    diskPath: Union[str, Path],
    fnBAL: List[str],
    fn0: List[Union[str, List[str]]],
    idxB: Dict[str, Any],
    D: float,
    S: float,
    b: float,
    c: float,
    XmRefB: List[float],
    XmRefM: List[float],
    dAoA: float,
    dAoS: float,
    modelType: str,
    modelPos: str,
    testSec: int,
) -> Dict[str, Any]:
    diskPath = Path(diskPath)

    if len(fnBAL) != len(fn0):
        raise ValueError("Enter 1 zero offset file (fn0) for each raw data file (fnBAL).")

    BAL = {"config": [], "windOff": {}, "windOn": {}, "source_files": {}}

    for raw_file, zero_file in zip(fnBAL, fn0):
        config = _sanitize_config_name(raw_file)
        BAL["config"].append(config)
        BAL["source_files"][config] = raw_file

        print(f"Processing balance data; configuration '{config}'; filename '{raw_file}'.")

        BAL["windOff"][config] = BAL_read0data(diskPath, zero_file, idxB)
        BAL["windOn"][config] = BAL_readData(diskPath / raw_file, idxB)
        BAL["windOn"][config] = BAL_forces(
            BAL["windOn"][config],
            BAL["windOff"][config],
            idxB,
            D, S, b, c,
            XmRefB, XmRefM,
            dAoA, dAoS,
            modelType, modelPos, testSec
        )

    return BAL


# ============================================================
# DataFrame helpers
# ============================================================

def bal_dict_to_dataframe(bal_cfg: Dict[str, Any]) -> pd.DataFrame:
    data = {}
    n_rows = None

    for key, value in bal_cfg.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            data[key] = value
            if n_rows is None:
                n_rows = len(value)

    df = pd.DataFrame(data)

    if "B16zeroed" in bal_cfg and isinstance(bal_cfg["B16zeroed"], np.ndarray) and bal_cfg["B16zeroed"].ndim == 2:
        b16 = bal_cfg["B16zeroed"]
        for i in range(b16.shape[1]):
            df[f"B{i+1}_zeroed"] = b16[:, i]

    if "BAL0_intp" in bal_cfg and isinstance(bal_cfg["BAL0_intp"], np.ndarray) and bal_cfg["BAL0_intp"].ndim == 2:
        b0 = bal_cfg["BAL0_intp"]
        for i in range(b0.shape[1]):
            df[f"B{i+1}_zero"] = b0[:, i]

    return df


def parse_rudder_deflection_deg(filename: str) -> float:
    """
    Examples:
    raw_rudder_0_elevator_0.txt   -> 0
    raw_rudder_m5_elevator_0.txt  -> -5
    raw_rudder_m10_elevator_0.txt -> -10
    raw_rudder_p8_elevator_0.txt  -> 8
    """
    stem = Path(filename).stem
    match = re.search(r"rudder_([mp]?\d+)", stem)
    if not match:
        raise ValueError(f"Could not parse rudder deflection from filename: {filename}")

    token = match.group(1)
    if token.startswith("m"):
        return -float(token[1:])
    if token.startswith("p"):
        return float(token[1:])
    return float(token)


def round_to_nearest_half(x: pd.Series) -> pd.Series:
    return np.round(x * 2.0) / 2.0


def build_combined_rudder_dataframe(BAL: Dict[str, Any], D: float = 0.2032) -> pd.DataFrame:
    dfs = []

    for cfg in BAL["config"]:
        df = bal_dict_to_dataframe(BAL["windOn"][cfg]).copy()
        source_file = BAL["source_files"][cfg]

        df["config"] = cfg
        df["source_file"] = source_file
        df["dR"] = parse_rudder_deflection_deg(source_file)
        df["dE"] = 0  # Assuming a default value for dE

        # Requested rounded columns
        df["V_round"] = np.round(df["V"]).astype("Int64")
        df["AoA_round"] = round_to_nearest_half(df["AoA"])
        df["AoS_round"] = np.round(df["AoS"]).astype("Int64")

        # Use processed advance ratio columns
        # Prefer J_M1, fall back to J_M2
        j_m1 = pd.to_numeric(df["J_M1"], errors="coerce")
        j_m2 = pd.to_numeric(df["J_M2"], errors="coerce")

        df["J"] = j_m1.where(j_m1.notna() & np.isfinite(j_m1), j_m2)
        df["J_round"] = np.round(df["J"] * 10.0) / 10.0

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    sort_cols = [c for c in ["dR", "AoS_round", "AoA_round", "V_round", "run"] if c in combined_df.columns]
    combined_df = combined_df.sort_values(sort_cols).reset_index(drop=True)

    return combined_df

# ============================================================
# Output writers
# ============================================================

def save_outputs(BAL: Dict[str, Any], combined_df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save one CSV per configuration
    for cfg in BAL["config"]:
        df_cfg = combined_df.loc[combined_df["config"] == cfg].copy()
        csv_path = output_dir / f"{cfg}.csv"
        df_cfg.to_csv(csv_path, index=False)

    # Save combined CSV
    combined_csv_path = output_dir / "all_rudder_cases_combined.csv"
    combined_df.to_csv(combined_csv_path, index=False)

    # Save one Excel file with one sheet per config + combined sheet
    excel_path = output_dir / "all_rudder_cases.xlsx"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for cfg in BAL["config"]:
            df_cfg = combined_df.loc[combined_df["config"] == cfg].copy()
            df_cfg.to_excel(writer, sheet_name=cfg[:31], index=False)

        combined_df.to_excel(writer, sheet_name="combined_all_cases", index=False)

    print(f"Saved combined CSV   : {combined_csv_path.resolve()}")
    print(f"Saved combined Excel : {excel_path.resolve()}")
    print(f"Saved per-case CSVs  : {output_dir.resolve()}")


# ============================================================
# Main
# ============================================================

def main():
    idxB = SUP_getIdx()

    # Relative path inside your git repo
    diskPath = Path(__file__).resolve().parent / "RAW_TEST_DATA" 
    output_dir = Path(__file__).resolve().parent / "RAW_TEST_DATA" / "processed_data"

    # Raw wind-on measurement files
    fn_BAL = [
        "raw_rudder_0_elevator_0.txt",
        "raw_rudder_m5_elevator_0.txt",
        "raw_rudder_m10_elevator_0.txt",
        "raw_rudder_m20_elevator_0.txt",
    ]

    # Zero file associated with each raw file
    fn0 = [
        "zer_ 20260227-074705.txt",
        "zer_ 20260227-074705.txt",
        "zer_ 20260227-074705.txt",
        "zer_ 20260227-074705.txt",
    ]

    # Geometry
    b = 1.4 * math.cos(math.radians(4.0))
    cR = 0.222
    cT = 0.089
    S = b / 2.0 * (cT + cR)
    taper = cT / cR
    c = 2.0 * cR / 3.0 * (1.0 + taper + taper**2) / (1.0 + taper)

    D = 0.2032
    XmRefB = [0.0, 0.0, 0.0465 / c]
    XmRefM = [0.25, 0.0, 0.0]

    dAoA = 0.0
    dAoS = 0.0
    modelType = "aircraft"
    modelPos = "inverted"
    testSec = 5

    print("Data folder :", diskPath.resolve())
    print("Output folder:", output_dir.resolve())

    BAL = BAL_process(
        diskPath=diskPath,
        fnBAL=fn_BAL,
        fn0=fn0,
        idxB=idxB,
        D=D,
        S=S,
        b=b,
        c=c,
        XmRefB=XmRefB,
        XmRefM=XmRefM,
        dAoA=dAoA,
        dAoS=dAoS,
        modelType=modelType,
        modelPos=modelPos,
        testSec=testSec,
    )

    combined_df = build_combined_rudder_dataframe(BAL, D=D)

    save_outputs(BAL, combined_df, output_dir=output_dir)

    print("\nCombined dataframe preview:")
    preview_cols = [c for c in [
        "config", "dR", "dE", "AoA", "AoA_round", "AoS", "AoS_round",
        "V", "V_round", "rpmWT", "J", "J_round", "CL", "CD", "CY",
        "CMroll", "CMpitch", "CMpitch25c", "CMyaw"
    ] if c in combined_df.columns]
    print(combined_df[preview_cols].head(20))

    return BAL, combined_df


if __name__ == "__main__":
    BAL, combined_df = main()

    