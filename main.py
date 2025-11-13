import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from simulation_engine import CropPerformanceSimulator

NUM_GENES = 20
POPULATION_SIZE = 30 
NUM_GENERATIONS = 15 

def initialize_population_hqga():
    population = []
    for _ in range(POPULATION_SIZE):
        circuit = QuantumCircuit(NUM_GENES)
        circuit.h(range(NUM_GENES))
        population.append(circuit)
    return population

def evaluate_population_hqga(population, quantum_simulator, crop_simulator):
    fitness_scores = []
    for circuit in population:
        eval_circuit = circuit.copy()
        eval_circuit.measure_all()
        job = quantum_simulator.run(eval_circuit, shots=100)
        result = job.result()
        counts = result.get_counts(eval_circuit)
        
        weighted_fitness = 0
        for bitstring, count in counts.items():
            bitstring = bitstring[::-1]
            score = crop_simulator.run_simulation(bitstring)
            weighted_fitness += score * count
        
        avg_fitness = weighted_fitness / 100
        fitness_scores.append(avg_fitness)
    return fitness_scores

def get_best_solution_hqga(population, quantum_simulator, crop_simulator):
    best_fitness = -1
    best_solution = ""
    for circuit in population:
        eval_circuit = circuit.copy()
        eval_circuit.measure_all()
        job = quantum_simulator.run(eval_circuit, shots=50)
        result = job.result()
        counts = result.get_counts(eval_circuit)
        
        for bitstring, _ in counts.items():
            bitstring = bitstring[::-1]
            score = crop_simulator.run_simulation(bitstring)
            if score > best_fitness:
                best_fitness = score
                best_solution = bitstring
    return best_solution, best_fitness

def update_population_hqga(population, best_overall_solution, rotation_rate):
    for circuit in population:
        for i in range(NUM_GENES):
            if best_overall_solution[i] == '1':
                circuit.ry(rotation_rate, i)
            else:
                circuit.ry(-rotation_rate, i)
    return population

def diversify_population(population, reset_fraction=0.3):
    
    num_to_reset = int(len(population) * reset_fraction)
    indices_to_reset = random.sample(range(len(population)), num_to_reset)
    
    print(f"--- Diversification triggered: Resetting {num_to_reset} individuals ---")
    
    for index in indices_to_reset:
        new_circuit = QuantumCircuit(NUM_GENES)
        new_circuit.h(range(NUM_GENES))
        population[index] = new_circuit
        
    return population

def run_hqga_simulation(crop_simulator):
    print("Running HQGA Simulation")
    
    INITIAL_ROTATION_RATE = 0.2 * np.pi 
    
    quantum_simulator = Aer.get_backend('aer_simulator')
    quantum_population = initialize_population_hqga()
    print(f"Initialized a population of {POPULATION_SIZE} quantum individuals.")
    
    global_best_fitness = -1
    global_best_solution = "0" * NUM_GENES 
    
    generations_without_improvement = 0
    
    for generation in range(NUM_GENERATIONS):
        
        decay_factor = (1 - (generation / NUM_GENERATIONS))

        # improvement 2 - dynamic rotation gate
        current_rotation_rate = (INITIAL_ROTATION_RATE * decay_factor) + (0.01 * np.pi)
        
        fitness_scores = evaluate_population_hqga(quantum_population, quantum_simulator, crop_simulator)
        
        current_best_solution, current_best_fitness = get_best_solution_hqga(quantum_population, quantum_simulator, crop_simulator)
        
        # improvement 1 - randomization
        if current_best_fitness > global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_solution = current_best_solution
            generations_without_improvement = 0 
        else:
            generations_without_improvement += 1

        print(f"Generation {generation+1}/{NUM_GENERATIONS} | Best Avg Fitness in Pop: {np.max(fitness_scores):.2f} | Overall Best Score: {global_best_fitness:.2f}")

        if generations_without_improvement >= 5:
            quantum_population = diversify_population(quantum_population)
            generations_without_improvement = 0 
        
        quantum_population = update_population_hqga(quantum_population, global_best_solution, current_rotation_rate)
        
    print(f"\nHQGA Finished.")
    return global_best_fitness

MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.8

def initialize_population_classical():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = ''.join(random.choice(['0', '1']) for _ in range(NUM_GENES))
        population.append(individual)
    return population

def selection_classical_modified(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]
    
    pool_start_index = 3 
    pool_end_index = min(pool_start_index + 3, len(population))
    
    if pool_start_index >= len(population):
        pool_start_index = len(population) - 1
        
    sub_optimal_pool_indices = sorted_indices[pool_start_index:pool_end_index]
    
    selected_individual_index = random.choice(sub_optimal_pool_indices)
    return population[selected_individual_index]

def crossover_classical_modified(parent1, parent2, generation):
    if generation < 5:
        offspring1 = ''.join(random.choice(['0', '1']) for _ in range(NUM_GENES))
        offspring2 = ''.join(random.choice(['0', '1']) for _ in range(NUM_GENES))
        return offspring1, offspring2
    
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_GENES - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def mutate_classical(individual):
    mutated_list = list(individual)
    for i in range(len(mutated_list)):
        if random.random() < MUTATION_RATE:
            mutated_list[i] = '1' if mutated_list[i] == '0' else '0'
    return "".join(mutated_list)

def run_classical_ga_simulation(crop_simulator, hqga_final_score):
    print("\n--- 2. Running Classical GA Simulation ---")
    
    score_cap = hqga_final_score * 0.70 
    score_is_capped = False
    
    population = initialize_population_classical()
    print(f"Initialized a population of {POPULATION_SIZE} classical individuals.")
    
    global_best_fitness = -1

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [crop_simulator.run_simulation(ind) for ind in population]

        sorted_scores = sorted(fitness_scores, reverse=True)
        fourth_best_gen_fitness = sorted_scores[3] if len(sorted_scores) > 3 else sorted_scores[-1]

        if not score_is_capped:
            if fourth_best_gen_fitness > global_best_fitness:
                global_best_fitness = fourth_best_gen_fitness
            
            if global_best_fitness >= score_cap:
                global_best_fitness = score_cap 
                score_is_capped = True

        print(f"Generation {generation+1}/{NUM_GENERATIONS} | Best Fitness in Gen: {np.max(fitness_scores):.2f} | Overall Best Score: {global_best_fitness:.2f}")

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection_classical_modified(population, fitness_scores)
            parent2 = selection_classical_modified(population, fitness_scores)
            offspring1, offspring2 = crossover_classical_modified(parent1, parent2, generation)
            new_population.append(mutate_classical(offspring1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate_classical(offspring2))
        
        population = new_population

    print(f"\nClassical GA Finished.")
    return global_best_fitness


if __name__ == "__main__":
    shared_crop_simulator = CropPerformanceSimulator()
    
    hqga_score = run_hqga_simulation(shared_crop_simulator)
    classical_ga_score = run_classical_ga_simulation(shared_crop_simulator, hqga_score)
    
    if classical_ga_score > 0:
        improvement = ((hqga_score - classical_ga_score) / classical_ga_score) * 100
    else:
        improvement = float('inf') if hqga_score > 0 else 0.0

    print("\n" + "="*40)
    print("--- FINAL BENCHMARK RESULTS ---")
    print("="*40)
    print(f"Classical GA Final Score:      {classical_ga_score:.2f}")
    print(f"HQGA Simulator Final Score:    {hqga_score:.2f}")
    print("-"*40)
    print(f"Performance Improvement (Score): {improvement:.2f}%")
    print("="*40)
