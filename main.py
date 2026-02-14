from pathlib import Path

from core.genetic_algorithm import genetic_algorithm
from core.solver import solver

instances = ["i04", "i06"]

for instance in instances:
    instance_dir = Path("data") / instance

    solver_output_dir = instance_dir / "solutions" / "solver_solution.csv"
    ga_output_dir = instance_dir / "solutions" / "genetic_algorithm_solution.csv"

    solver_result = solver(instance_dir, solver_output_dir)
    print(f"{instance} - Solver - Best Solution (fitness): {solver_result}")

    genetic_algorithm_result = genetic_algorithm(
        instance_dir,
        ga_output_dir,
        population_size=80,
        generations=1000,
        tournament_size=3,
        mutation_rate=0.2,
        elite_fraction=0.1,
        seed=42,
    )
    print(f"{instance} - GA - Best Solution (fitness): {genetic_algorithm_result}")
