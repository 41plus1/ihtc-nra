import random
from pathlib import Path
from typing import Any

import pandas as pd

from core.functions import get_nurses_working
from core.penalties import compute_penalties


def load_instance(instance_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance_path = Path(instance_dir)
    nurse_data = pd.read_csv(instance_path / "nurse_shifts.csv")
    room_data = pd.read_csv(instance_path / "occupied_room_shifts.csv")
    return nurse_data, room_data


def build_room_shift_groups(room_data: pd.DataFrame) -> dict[tuple[int, str], list[str]]:
    groups = {}
    room_shifts = room_data[["room_id", "day", "shift"]].drop_duplicates()
    for row in room_shifts.itertuples(index=False):
        r = str(row.room_id)
        d = int(row.day)
        t = str(row.shift)
        groups.setdefault((d, t), []).append(r)
    return groups


def get_available_nurses(
    nurse_data: pd.DataFrame,
    groups: dict[tuple[int, str], list[str]],
) -> dict[tuple[int, str], list[str]]:
    avaiable_nurses = {}
    for d, t in groups:
        avaiable_nurses[(d, t)] = [str(n) for n in get_nurses_working(nurse_data, d, t)]
    return avaiable_nurses


def build_costs(
    nurse_data: pd.DataFrame,
    room_data: pd.DataFrame,
) -> dict[tuple[str, str, int, str], int]:
    alphas, betas = compute_penalties(nurse_data, room_data)
    costs = {}
    for key in set(alphas) | set(betas):
        costs[key] = alphas.get(key, 0) + 5 * betas.get(key, 0)
    return costs


def create_chromosome(
    groups: dict[tuple[int, str], list[str]],
    availability: dict[tuple[int, str], list[str]],
    rng: random.Random,
) -> dict[tuple[str, int, str], str]:
    chromosome = {}
    for (d, t), rooms in groups.items():
        nurses = availability[(d, t)]
        for room in rooms:
            chromosome[(room, d, t)] = rng.choice(nurses)
    return chromosome


def chromosome_fitness(
    chromosome: dict[tuple[str, int, str], str],
    costs: dict[tuple[str, str, int, str], int],
) -> int:
    total = 0
    for (r, d, t), n in chromosome.items():
        total += costs.get((n, r, d, t), 0)
    return total


def evaluate_population(
    population: list[dict[tuple[str, int, str], str]],
    costs: dict[tuple[str, str, int, str], int],
) -> list[int]:
    return [chromosome_fitness(chromosome, costs) for chromosome in population]


def tournament_select(
    population: list[dict[tuple[str, int, str], str]],
    scores: list[int],
    rng: random.Random,
    size: int,
) -> dict[tuple[str, int, str], str]:
    indices = rng.sample(range(len(population)), k=size)
    best_idx = min(indices, key=lambda idx: scores[idx])
    return population[best_idx]


def crossover(
    parent_a: dict[tuple[str, int, str], str],
    parent_b: dict[tuple[str, int, str], str],
    groups: dict[tuple[int, str], list[str]],
    rng: random.Random,
) -> dict[tuple[str, int, str], str]:
    child = {}
    for (d, t), rooms in groups.items():
        threshold = 0.5
        source = parent_a if rng.random() < threshold else parent_b
        for room in rooms:
            child[(room, d, t)] = source[(room, d, t)]
    return child


def mutate(
    chromosome: dict[tuple[str, int, str], str],
    groups: dict[tuple[int, str], list[str]],
    availability: dict[tuple[int, str], list[str]],
    rng: random.Random,
    mutation_rate: float,
) -> None:
    threshold = 0.5
    for (d, t), rooms in groups.items():
        if rng.random() >= mutation_rate:
            continue
        if len(rooms) <= 1:
            continue
        if rng.random() < threshold:
            i, j = rng.sample(range(len(rooms)), k=2)
            room_i, room_j = rooms[i], rooms[j]
            chromosome[(room_i, d, t)], chromosome[(room_j, d, t)] = (
                chromosome[(room_j, d, t)],
                chromosome[(room_i, d, t)],
            )
        else:
            nurses = availability[(d, t)]
            for room in rooms:
                chromosome[(room, d, t)] = rng.choice(nurses)


def genetic_algorithm(
    instance_dir: Path,
    output_csv: Path,
    population_size: int,
    generations: int,
    tournament_size: int,
    mutation_rate: float,
    elite_fraction: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    nurse_data, room_data = load_instance(instance_dir)
    groups = build_room_shift_groups(room_data)
    availability = get_available_nurses(nurse_data, groups)

    costs = build_costs(nurse_data, room_data)

    population = [create_chromosome(groups, availability, rng) for _ in range(population_size)]
    scores = evaluate_population(population, costs)

    elite_count = max(1, int(population_size * elite_fraction))

    for _ in range(generations):
        ranked = sorted(zip(population, scores), key=lambda item: item[1])
        elites = [chromosome for chromosome, _ in ranked[:elite_count]]

        next_population = list(elites)
        while len(next_population) < population_size:
            parent_a = tournament_select(
                population,
                scores,
                rng,
                tournament_size,
            )
            parent_b = tournament_select(
                population,
                scores,
                rng,
                tournament_size,
            )
            child = crossover(parent_a, parent_b, groups, rng)
            mutate(child, groups, availability, rng, mutation_rate)
            next_population.append(child)

        population = next_population
        scores = evaluate_population(population, costs)

    best_idx = min(range(len(population)), key=lambda idx: scores[idx])
    best_solution = population[best_idx]
    best_score = scores[best_idx]

    rows = []
    for (r, d, t), n in best_solution.items():
        rows.append({"room_id": r, "day": d, "shift": t, "nurse_id": n, "x": 1})

    solution = pd.DataFrame(rows)
    solution.to_csv(output_csv, index=False)

    return float(best_score)
