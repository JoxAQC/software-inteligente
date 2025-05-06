import random
import streamlit as st

def random_individual(size=8):
    return [random.randint(1, size) for _ in range(size)]

maxFitness = 28

def fitness(individual):
    horizontal_collisions = sum([individual.count(queen) - 1 for queen in individual]) / 2
    diagonal_collisions = 0
    n = len(individual)
    left_diagonal = [0] * 2 * n
    right_diagonal = [0] * 2 * n
    for i in range(n):
        left_diagonal[i + individual[i] - 1] += 1
        right_diagonal[len(individual) - i + individual[i] - 2] += 1
    for i in range(2 * n - 1):
        if left_diagonal[i] > 1:
            diagonal_collisions += (left_diagonal[i] - 1) / (n - abs(i - n + 1))
        if right_diagonal[i] > 1:
            diagonal_collisions += (right_diagonal[i] - 1) / (n - abs(i - n + 1))
    return int(maxFitness - (horizontal_collisions + diagonal_collisions))

def probability(individual, fitness_func):
    return fitness_func(individual) / maxFitness

def random_pick(population, probabilities):
    total = sum(probabilities)
    r = random.uniform(0, total)
    upto = 0
    for individual, prob in zip(population, probabilities):
        if upto + prob >= r:
            return individual
        upto += prob
    raise RuntimeError("No se pudo seleccionar un individuo")

def reproduce(x, y):
    n = len(x)
    c = random.randint(0, n - 1)
    return x[0:c] + y[c:n]

def mutate(individual):
    n = len(individual)
    c = random.randint(0, n - 1)
    m = random.randint(1, n)
    individual[c] = m
    return individual

def genetic_queen(population, fitness_func):
    mutation_probability = 0.10
    new_population = []
    probabilities = [probability(ind, fitness_func) for ind in population]
    for _ in population:
        x = random_pick(population, probabilities)
        y = random_pick(population, probabilities)
        child = reproduce(x, y)
        if random.random() < mutation_probability:
            child = mutate(child)
        new_population.append(child)
        if fitness_func(child) == maxFitness:
            break
    return new_population

def run():
    st.title("👑 Algoritmo Genético - 8 Reinas")

    population = [random_individual() for _ in range(10)]
    generation = 1
    result_area = st.empty()

    while maxFitness not in [fitness(ind) for ind in population]:
        result_text = f"### 🌀 Generación {generation}\n"
        population = genetic_queen(population, fitness)
        best_fit = max(fitness(ind) for ind in population)
        result_text += f"**Mejor fitness:** {best_fit}\n\n"

        for ind in population:
            result_text += f"{ind}, fitness = {fitness(ind)}, probabilidad = {probability(ind, fitness):.4f}\n"

        result_area.markdown(result_text)
        generation += 1

    st.success(f"✅ ¡Solución encontrada en la generación {generation - 1}!")

    for ind in population:
        if fitness(ind) == maxFitness:
            st.code(f"Solución: {ind}\nFitness: 28")
            break
