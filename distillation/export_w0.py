"""Export / import BaseController weights in the C++ OLIVE binary layout.

Binary layout (row-major float32), matching OLIVEModel::load_base_weights:

    W1 [D × STATE_DIM] | b1 [D]
    W2 [D × D]         | b2 [D]
    W3 [ACTION_DIM × D]| b3 [ACTION_DIM]
    GateRankNet:
        Wh [GATE_HIDDEN × STATE_DIM] | bh [GATE_HIDDEN]
        wg [GATE_HIDDEN] | bg
        wc [GATE_HIDDEN] | bc

Eigen stores column-major; we write row-major and the C++ loader reads into
Eigen matrices via .data() which is column-major — so we transpose on write
to keep numerical layout consistent with Eigen's memory order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .config import ACTION_DIM, D, GATE_HIDDEN, STATE_DIM
from .student import BaseController


def _to_eigen_colmajor(weight_rowmajor: np.ndarray) -> np.ndarray:
    """nn.Linear.weight is [out, in] row-major; Eigen MatrixXf is col-major
    with the same logical (rows=out, cols=in) shape. Writing the Fortran
    (column-major) bytes matches Eigen::MatrixXf::data()."""
    return np.asfortranarray(weight_rowmajor.astype(np.float32))


def export_w0_binary(
    student: BaseController,
    path: Union[str, Path],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        def write_mat(w: torch.Tensor) -> None:
            arr = _to_eigen_colmajor(w.detach().cpu().numpy())
            f.write(arr.tobytes(order="F"))

        def write_vec(v: torch.Tensor) -> None:
            arr = v.detach().cpu().numpy().astype(np.float32).ravel()
            f.write(arr.tobytes())

        def write_float(x: float) -> None:
            f.write(np.float32(x).tobytes())

        # Policy layers
        write_mat(student.fc1.weight)       # [D, STATE_DIM]
        write_vec(student.fc1.bias)         # [D]
        write_mat(student.fc2.weight)       # [D, D]
        write_vec(student.fc2.bias)         # [D]
        write_mat(student.fc3.weight)       # [ACTION_DIM, D]
        write_vec(student.fc3.bias)         # [ACTION_DIM]

        # GateRankNet — Linear.weight is [out, in]
        write_mat(student.gate_rank.Wh.weight)  # [GATE_HIDDEN, STATE_DIM]
        write_vec(student.gate_rank.Wh.bias)
        write_vec(student.gate_rank.wg.weight.view(-1))
        write_float(float(student.gate_rank.wg.bias.item()))
        write_vec(student.gate_rank.wc.weight.view(-1))
        write_float(float(student.gate_rank.wc.bias.item()))

    return path


def load_w0_binary(path: Union[str, Path], device: str = "cpu") -> BaseController:
    """Inverse of export_w0_binary — useful for round-trip tests."""
    path = Path(path)
    data = path.read_bytes()
    offset = 0

    def read_mat(rows: int, cols: int) -> torch.Tensor:
        nonlocal offset
        n = rows * cols
        raw = np.frombuffer(data, dtype=np.float32, count=n, offset=offset)
        offset += n * 4
        # written as Fortran/col-major → reshape accordingly then to torch
        mat = np.asarray(raw).reshape((rows, cols), order="F")
        return torch.from_numpy(mat.copy())

    def read_vec(n: int) -> torch.Tensor:
        nonlocal offset
        raw = np.frombuffer(data, dtype=np.float32, count=n, offset=offset)
        offset += n * 4
        return torch.from_numpy(np.asarray(raw).copy())

    def read_float() -> float:
        nonlocal offset
        raw = np.frombuffer(data, dtype=np.float32, count=1, offset=offset)
        offset += 4
        return float(raw[0])

    student = BaseController().to(device)
    with torch.no_grad():
        student.fc1.weight.copy_(read_mat(D, STATE_DIM))
        student.fc1.bias.copy_(read_vec(D))
        student.fc2.weight.copy_(read_mat(D, D))
        student.fc2.bias.copy_(read_vec(D))
        student.fc3.weight.copy_(read_mat(ACTION_DIM, D))
        student.fc3.bias.copy_(read_vec(ACTION_DIM))

        student.gate_rank.Wh.weight.copy_(read_mat(GATE_HIDDEN, STATE_DIM))
        student.gate_rank.Wh.bias.copy_(read_vec(GATE_HIDDEN))
        student.gate_rank.wg.weight.view(-1).copy_(read_vec(GATE_HIDDEN))
        student.gate_rank.wg.bias.fill_(read_float())
        student.gate_rank.wc.weight.view(-1).copy_(read_vec(GATE_HIDDEN))
        student.gate_rank.wc.bias.fill_(read_float())

    return student
