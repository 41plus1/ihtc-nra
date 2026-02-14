import shutil
from pathlib import Path

import pandas as pd
import pulp

from core.functions import get_nurses_working
from core.penalties import compute_penalties


def load_instance(instance_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance_path = Path(instance_dir)
    nurse_data = pd.read_csv(instance_path / "nurse_shifts.csv")
    room_data = pd.read_csv(instance_path / "occupied_room_shifts.csv")
    return nurse_data, room_data


def build_model(
    nurse_data: pd.DataFrame,
    room_data: pd.DataFrame,
) -> tuple[pulp.LpProblem, dict[tuple[str, str, int, str], pulp.LpVariable], dict, dict]:
    alphas, betas = compute_penalties(nurse_data, room_data)
    model = pulp.LpProblem("nurse_room_allocation", pulp.LpMinimize)

    x = {}

    room_shifts = room_data[["room_id", "day", "shift"]].drop_duplicates()
    for row in room_shifts.itertuples(index=False):
        r = str(row.room_id)
        d = int(row.day)
        t = str(row.shift)
        for n in get_nurses_working(nurse_data, d, t):
            key = (n, r, d, t)
            x[key] = pulp.LpVariable(f"x_{n}_{r}_{d}_{t}", cat="Binary")

    for row in room_shifts.itertuples(index=False):
        r = str(row.room_id)
        d = int(row.day)
        t = str(row.shift)
        candidates = [x[(n, r, d, t)] for n in get_nurses_working(nurse_data, d, t)]
        model += pulp.lpSum(candidates) == 1, f"cover_{r}_{d}_{t}"

    model += pulp.lpSum(
        (alphas.get(key, 0) + 5 * betas.get(key, 0)) * var for key, var in x.items()
    )

    return model, x


def solver(
    instance_dir: Path,
    output_csv: Path,
) -> float:
    nurse_data, room_data = load_instance(instance_dir)
    model, x = build_model(nurse_data, room_data)

    cbc_path = shutil.which("cbc")
    cbc = pulp.COIN_CMD(path=cbc_path, msg=False) if cbc_path else pulp.PULP_CBC_CMD(msg=False)
    model.solve(cbc)

    rows = []

    for (n, r, d, t), var in x.items():
        value = var.value()
        threshold = 0.5
        if value is None or value < threshold:
            continue
        rows.append({"room_id": r, "day": d, "shift": t, "nurse_id": n, "x": 1})

        solution = pd.DataFrame(rows)
        solution = solution.sort_values(["day", "shift", "room_id"]).reset_index(drop=True)
        solution.to_csv(output_csv, index=False)

    return float(pulp.value(model.objective))
