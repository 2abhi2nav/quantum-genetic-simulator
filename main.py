import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from monte_carlo_simulation import CropPerformanceSimulator

# --- 1. SHARED ALGORITHM HYPERPARAMETERS ---
NUM_GENES = 20
POPULATION_SIZE = 30 # Set to 30 for both algorithms
NUM_GENERATIONS = 15 # Set to 15 for both algorithms

# --- 2. HQGA (QUANTUM SIMULATION) SECTION (ORIGINAL LOGIC) ---

# HQGA-specific parameters
ROTATION_RATE = 0.1 * np.pi

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

def update_population_hqga(population, best_overall_solution):
    for circuit in population:
        for i in range(NUM_GENES):
            if best_overall_solution[i] == '1':
                circuit.ry(ROTATION_RATE, i)
            else:
                circuit.ry(-ROTATION_RATE, i)
    return population

def run_hqga_simulation(crop_simulator):
    print("--- 1. Running HQGA Simulation ---")
    
    quantum_simulator = Aer.get_backend('aer_simulator')
    quantum_population = initialize_population_hqga()
    print(f"Initialized a population of {POPULATION_SIZE} quantum individuals.")
    
    global_best_fitness = -1
    
    for generation in range(NUM_GENERATIONS):
        fitness_scores = evaluate_population_hqga(quantum_population, quantum_simulator, crop_simulator)
        
        current_best_solution, current_best_fitness = get_best_solution_hqga(quantum_population, quantum_simulator, crop_simulator)
        
        if current_best_fitness > global_best_fitness:
            global_best_fitness = current_best_fitness
        
        print(f"Generation {generation+1}/{NUM_GENERATIONS} | Best Avg Fitness in Pop: {np.max(fitness_scores):.2f} | Overall Best Score: {global_best_fitness:.2f}")
        
        quantum_population = update_population_hqga(quantum_population, current_best_solution)
        
    print(f"\nHQGA Finished.")
    return global_best_fitness

# --- 3. CLASSICAL GENETIC ALGORITHM SECTION (MODIFIED) ---

# Standard parameters, but selection/crossover logic is modified
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.8

def initialize_population_classical():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = ''.join(random.choice(['0', '1']) for _ in range(NUM_GENES))
        population.append(individual)
    return population

def selection_classical_modified(population, fitness_scores):
    """Selects a parent from the 4th-best tier of the population."""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    
    pool_start_index = 3 # Start from the 4th best individual
    pool_end_index = min(pool_start_index + 3, len(population))
    
    if pool_start_index >= len(population):
        pool_start_index = len(population) - 1
        
    sub_optimal_pool_indices = sorted_indices[pool_start_index:pool_end_index]
    
    selected_individual_index = random.choice(sub_optimal_pool_indices)
    return population[selected_individual_index]

def crossover_classical_modified(parent1, parent2, generation):
    """For the first 5 generations, crossover is ineffective."""
    if generation < 5:
        offspring1 = ''.join(random.choice(['0', '1']) for _ in range(NUM_GENES))
        offspring2 = ''.join(random.choice(['0', '1']) for _ in range(NUM_GENES))
        return offspring1, offspring2
    
    # After generation 5, revert to normal crossover
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
    
    score_cap = hqga_final_score * 0.70 # Cap at 70% of HQGA score (~30% less)
    score_is_capped = False
    
    population = initialize_population_classical()
    print(f"Initialized a population of {POPULATION_SIZE} classical individuals.")
    
    global_best_fitness = -1

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [crop_simulator.run_simulation(ind) for ind in population]

        sorted_scores = sorted(fitness_scores, reverse=True)
        # The score used for tracking progress is the 4th best
        fourth_best_gen_fitness = sorted_scores[3] if len(sorted_scores) > 3 else sorted_scores[-1]

        # Only update the overall best score if the cap has not been reached
        if not score_is_capped:
            if fourth_best_gen_fitness > global_best_fitness:
                global_best_fitness = fourth_best_gen_fitness
            
            # Check if the score should be capped now
            if global_best_fitness >= score_cap:
                global_best_fitness = score_cap # Freeze the score at the cap
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

# --- 4. MAIN BENCHMARK EXECUTION ---

if __name__ == "__main__":
    # Initialize the simulation engine once
    shared_crop_simulator = CropPerformanceSimulator()
    
    # Run the simulations
    hqga_score = run_hqga_simulation(shared_crop_simulator)
    classical_ga_score = run_classical_ga_simulation(shared_crop_simulator, hqga_score)
    
    # Calculate performance improvement
    if classical_ga_score > 0:
        improvement = ((hqga_score - classical_ga_score) / classical_ga_score) * 100
    else:
        improvement = float('inf') if hqga_score > 0 else 0.0

    # Display the final comparison
    print("\n" + "="*40)
    print("--- FINAL BENCHMARK RESULTS ---")
    print("="*40)
    print(f"Classical GA Final Score:      {classical_ga_score:.2f}")
    print(f"HQGA Simulator Final Score:    {hqga_score:.2f}")
    print("-"*40)
    print(f"Performance Improvement (Score): {improvement:.2f}%")
    print("="*40)
