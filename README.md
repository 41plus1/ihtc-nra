# IHTC 2024 - Nurse Room Allocation (NRA)

Este repositório contém os arquivos relacionados à tarefa de alocação de enfermeiros em salas e turnos hospitalares, comparando duas abordagens:

- Solver exato (PuLP/CBC)
- Algoritmo genético

## Estrutura

- `core/`: implementação das regras, penalidades e métodos de solução
	- `solver.py`: modelo exato (PuLP/CBC)
	- `genetic_algorithm.py`: meta-heurística
	- `penalties.py`: cálculo das penalidades
	- `functions.py`: funções auxiliares de consulta dos dados
- `data/`: instâncias de entrada (`i04`, `i06`) e soluções em CSV
- `images/`: gráficos gerados
- `main.ipynb`: fluxo principal de execução

## Requisitos

- Python
- Pandas
- Matplotlib
- PuLP

## Execução

1. Instale as dependências.
2. Abra `main.ipynb`.
3. Execute as células para processar as instâncias e gerar resultados.

## Entradas e Saídas

Para cada instância em `data``:
- Entradas:
	- `nurse_shifts.csv`
	- `occupied_room_shifts.csv`
	- `instance_info.json`
- Saídas (em `solutions/`):
	- `solver_solution.csv`
	- `genetic_algorithm_solution.csv`

## Autores

- Marcos Vinícius Brito de Araújo
- Maurício Aires Pinheiro
- Murillo Martins Figueira