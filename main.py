from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from core.genetic_algorithm import genetic_algorithm
from core.solver import solver

instances = ["i04", "i06"]
all_results = {}
solver_results = {}

for instance in instances:
    instance_dir = Path("data") / instance
    results = []
    
    solver_output_dir = instance_dir / "solutions" / "solver_solution.csv"
    ga_output_dir = instance_dir / "solutions" / "genetic_algorithm_solution.csv"

    solver_result = solver(instance_dir, solver_output_dir)
    print(f"{instance} - Solver - Best Solution (fitness): {solver_result}")

    for _ in range(3):  
        genetic_algorithm_result = genetic_algorithm(
            instance_dir,
            ga_output_dir,
            population_size=80,
            generations=1000,
            tournament_size=3,
            mutation_rate=0.2,
            elite_fraction=0.1,
            seed=datetime.now().timestamp()
        )
        
        print(f"{instance} - GA - Best Solution (fitness): {genetic_algorithm_result}")
        results.append(genetic_algorithm_result)
    all_results[instance] = results    



fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4), sharey=True)
if len(all_results) == 1:
    axes = [axes]

for ax, (instance, vals) in zip(axes, all_results.items()):
    y = np.array(vals, dtype=float)
    x = np.arange(1, len(y) + 1)

    mean = y.mean()
    std = y.std(ddof=1) if len(y) > 1 else 0.0  # ddof=1 é melhor pra amostra pequena

    # pontos por execução
    ax.plot(x, y, marker="o", linewidth=1)

    # média
    ax.axhline(mean, linewidth=1)

    # faixa ±1 desvio padrão (estabilidade)
    ax.fill_between([x.min(), x.max()], mean - std, mean + std, alpha=0.2)

    # referência do solver (se quiser comparar)
    if instance in solver_results:
        ax.axhline(float(solver_results[instance]), linestyle="--", linewidth=1)

    ax.set_title(f"{instance} | média={mean:.4f} | std={std:.4f}")
    ax.set_xlabel("Execução")
    ax.set_ylabel("Best fitness (final)")

plt.tight_layout()
plt.show()