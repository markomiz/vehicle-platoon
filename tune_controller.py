import random
import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from controller import *
from vehicle_system import *
from helpers import *
from copy import deepcopy
from world import *
from plotting import *
from scenarios import *

# Set the limits for each gain
parameter_limits = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]

# Define the evaluation function (simulation)
def evaluate_controller(parameters):
    world = World()
    world.set_controller_gains(parameters)
    fitness = run_scenario(world=world, plot=False, num_cars=5)
    return fitness

# Define the genetic algorithm parameters
population_size = 100
generations = 100
mutation_rate = 0.1
tournament_size = 3
elite_size = 5

# Generate initial population
population = []
for _ in range(population_size):
    individual = [random.uniform(limit[0], limit[1]) for limit in parameter_limits]
    population.append(individual)

# Run the genetic algorithm
for generation in range(generations):
    # Evaluate fitness for each individual in the population
    fitness_scores = [evaluate_controller(individual) for individual in population]

    # Select the best individuals for the next generation (elitism)
    elite_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])[:elite_size]
    next_generation = [population[i] for i in elite_indices]
    # Apply tournament selection and crossover to create offspring
    while len(next_generation) < population_size:
        tournament = random.sample(range(population_size), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]

        # Find the indices of the best individuals in the tournament
        selected_indices = [tournament[i] for i in sorted(range(len(tournament)), key=lambda k: tournament_fitness[k])[:tournament_size]]
        parents = [population[i] for i in selected_indices]

        # Perform crossover
        offspring = []
        for i in range(len(parameter_limits)):
            parent1_val = parents[0][i]
            parent2_val = parents[1][i]
            offspring_val = random.uniform(min(parent1_val, parent2_val), max(parent1_val, parent2_val))
            offspring.append(offspring_val)

        # Perform mutation
        for i in range(len(parameter_limits)):
            if random.random() < mutation_rate:
                offspring[i] = random.uniform(parameter_limits[i][0], parameter_limits[i][1])

        next_generation.append(offspring)

    # Replace the current population with the new generation
    population = next_generation

# Evaluate fitness for the final population
fitness_scores = [evaluate_controller(individual) for individual in population]

# Get the best individual and its fitness value
best_index = min(range(len(fitness_scores)), key=lambda k: fitness_scores[k])
best_individual = population[best_index]
best_fitness = fitness_scores[best_index]

# Print the best individual and its fitness value
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)
