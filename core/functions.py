import pandas as pd


def get_nurse_skill_level(data: pd.DataFrame, n: str) -> None | int:
    rows = data[data["nurse_id"] == n]["skill_level"]
    if rows.empty:
        return None
    return int(rows.iloc[0])


def get_nurse_max_workload(data: pd.DataFrame, n: str, d: int, t: str) -> None | int:
    rows = data[(data["nurse_id"] == n) & (data["day"] == d) & (data["shift"] == t)]["max_load"]
    if rows.empty:
        return None
    return int(rows.iloc[0])


def get_nurses_working(data: pd.DataFrame, d: int, t: str) -> list[str]:
    return data[(data["day"] == d) & (data["shift"] == t)]["nurse_id"].to_list()


def get_min_skill_level_for_room(data: pd.DataFrame, r: str, d: int, t: str) -> None | int:
    rows = data[(data["room_id"] == r) & (data["day"] == d) & (data["shift"] == t)][
        "max_skill_required"
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0])


def get_workload_for_room(data: pd.DataFrame, r: str, d: int, t: str) -> None | int:
    rows = data[(data["room_id"] == r) & (data["day"] == d) & (data["shift"] == t)][
        "total_room_workload"
    ]
    if rows.empty:
        return None
    return int(rows.iloc[0])
