import pandas as pd

from core.functions import (
    get_min_skill_level_for_room,
    get_nurse_max_workload,
    get_nurse_skill_level,
    get_workload_for_room,
)


def get_nurses(data: pd.DataFrame) -> list[str]:
    return data["nurse_id"].unique().tolist()


def get_rooms(data: pd.DataFrame) -> list[str]:
    return data["room_id"].unique().tolist()


def get_days(data: pd.DataFrame) -> list[int]:
    return data["day"].unique().tolist()


def get_shifts(data: pd.DataFrame) -> list[str]:
    return data["shift"].unique().tolist()


def compute_penalties(
    nurse_data: pd.DataFrame,
    room_data: pd.DataFrame,
) -> tuple[dict[tuple[str, str, int, str], int], dict[tuple[str, str, int, str], int]]:
    alphas = {}
    betas = {}

    for nurse in get_nurses(nurse_data):
        for room in get_rooms(room_data):
            for day in get_days(room_data):
                for shift in get_shifts(nurse_data):
                    alpha = compute_alpha(nurse_data, room_data, nurse, room, day, shift)
                    beta = compute_beta(nurse_data, room_data, nurse, room, day, shift)

                    alphas[(nurse, room, day, shift)] = alpha
                    betas[(nurse, room, day, shift)] = beta

    return alphas, betas


def compute_alpha(
    nurse_data: pd.DataFrame,
    room_data: pd.DataFrame,
    n: str,
    r: str,
    d: int,
    t: str,
) -> int:
    nurse_skill_level = get_nurse_skill_level(nurse_data, n)
    min_skill_level_for_room = get_min_skill_level_for_room(room_data, r, d, t)

    if nurse_skill_level is None or min_skill_level_for_room is None:
        return 0

    return max(0, min_skill_level_for_room - nurse_skill_level)


def compute_beta(
    nurse_data: pd.DataFrame,
    room_data: pd.DataFrame,
    n: str,
    r: str,
    d: int,
    t: str,
) -> int:
    nurse_max_workload = get_nurse_max_workload(nurse_data, n, d, t)
    workload_for_room = get_workload_for_room(room_data, r, d, t)

    if nurse_max_workload is None or workload_for_room is None:
        return 0

    return max(0, workload_for_room - nurse_max_workload)
