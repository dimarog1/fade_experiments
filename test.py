# import random
# import time
#
# import numpy as np
# import tqdm
#
#
# def evaluate(x):
#     time.sleep(40)
#     return -x ** 4 + 2 * x ** 2 + x + 1
#
#
# def crossover(x1, x2):
#     n = len(x1)
#     child1 = []
#     child2 = []
#     for i in range(n):
#         if random.random() < 0.5:
#             child1.append(x1[i])
#             child2.append(x2[i])
#         else:
#             child1.append(x2[i])
#             child2.append(x1[i])
#     return [child1, child2]
#
#
# def swap_mutation(chromosome):
#     n = len(chromosome)
#     if n < 2:
#         return chromosome
#     i, j = random.sample(range(n), 2)
#     chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
#     return chromosome
#
#
# def genetic_algorithm(bounds, n_generations, n_individuals, crossover_rate, mutation_rate, elitism=True):
#     # Генетический алгоритм
#     population = [[random.uniform(*bound) for bound in bounds]
#                   for _ in range(n_individuals)]
#     print(population)
#
#     values = []
#     for generation in tqdm.tqdm(range(n_generations)):
#         values = [evaluate(*x) for x in population]
#
#         if elitism:
#             best_idx = np.argmax(values)
#             best_params = population[best_idx]
#
#         for i in range(0, len(population), 2):
#             if random.random() < crossover_rate:
#                 parent1, parent2 = population[i], population[i + 1]
#                 child1, child2 = crossover(parent1, parent2)
#                 population[i], population[i + 1] = child1, child2
#
#         for i in range(len(population)):
#             if random.random() < mutation_rate:
#                 population[i] = swap_mutation(population[i])
#
#         if elitism:
#             population[best_idx] = best_params
#
#     best_index = np.argmax(values)
#     best_solution = population[best_index]
#
#     return best_solution, values[best_index]
#
#
# if __name__ == "__main__":
#     bounds = [(a, b) for a, b in zip([-10], [10])]  # границы параметров
#     n_generations = 5  # количество поколений
#     n_individuals = 30  # количество особей
#     crossover_rate = 0.8
#     mutation_rate = 0.1
#     elitism = True
#
#     res = genetic_algorithm(bounds, n_generations, n_individuals, crossover_rate, mutation_rate, elitism)
#     print(res)
#     # for elem in res:
#     #     print(evaluate(*elem))

import lightning_module
import mos

model = lightning_module.BaselineLightningModule.load_from_checkpoint("epoch=3-step=7459.ckpt").eval()

print(mos.calc_mos_dir('data/shuffled', model))
print(mos.calc_mos_dir('data/buckets', model))
