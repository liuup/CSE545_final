import random
import numpy as np
from typing import List, Tuple, Dict
import copy
import time


class BinPackingGAWoC:
    """
    Genetic Algorithm with Wisdom of Crowds for solving the Bin Packing Problem.
    
    The algorithm combines:
    - Genetic Algorithm for evolutionary optimization
    - Wisdom of Crowds for diverse solution exploration
    """
    
    def __init__(self, items: List[float], bin_capacity: float, 
                 population_size: int = 100, generations: int = 200,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 crowd_size: int = 5, use_woc: bool = True):
        """
        Initialize the GA+WoC algorithm.
        
        Args:
            items: List of item sizes
            bin_capacity: Maximum capacity of each bin
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            crowd_size: Number of parallel populations (crowds)
            use_woc: Whether to use Wisdom of Crowds (if False, uses standard GA)
        """
        self.items = sorted(items, reverse=True)  # Sort items in descending order
        self.bin_capacity = bin_capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crowd_size = crowd_size if use_woc else 1
        self.use_woc = use_woc
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
    def _create_individual(self) -> List[int]:
        """
        Create a random individual (chromosome).
        Each gene represents which bin the item should go into.
        """
        individual = []
        bins = [[]]
        bin_loads = [0]
        
        for item in self.items:
            # Try First Fit Decreasing heuristic with some randomness
            placed = False
            if random.random() > 0.3:  # 70% use heuristic
                for i in range(len(bins)):
                    if bin_loads[i] + item <= self.bin_capacity:
                        individual.append(i)
                        bin_loads[i] += item
                        placed = True
                        break
            
            if not placed:
                # Create new bin
                individual.append(len(bins))
                bins.append([item])
                bin_loads.append(item)
        
        return individual
    
    def _calculate_fitness(self, individual: List[int]) -> float:
        """
        Calculate fitness of an individual.
        Lower fitness is better (minimize number of bins).
        """
        bins = {}
        for item_idx, bin_idx in enumerate(individual):
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(self.items[item_idx])
        
        # Check if solution is valid
        for bin_items in bins.values():
            if sum(bin_items) > self.bin_capacity:
                # Penalize invalid solutions
                return float('inf')
        
        num_bins = len(bins)
        
        # Calculate wasted space penalty
        wasted_space = 0
        for bin_items in bins.values():
            wasted_space += (self.bin_capacity - sum(bin_items)) ** 2
        
        # Fitness: number of bins + normalized wasted space penalty
        fitness = num_bins + (wasted_space / (self.bin_capacity ** 2 * num_bins))
        
        return fitness
    
    def _repair_individual(self, individual: List[int]) -> List[int]:
        """
        Repair an invalid individual by redistributing items.
        """
        bins = {}
        for item_idx, bin_idx in enumerate(individual):
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(item_idx)
        
        # Check and repair overloaded bins
        repaired = list(individual)
        # Create a list of bin indices to avoid dictionary size change during iteration
        bin_indices_to_check = list(bins.keys())
        
        for bin_idx in bin_indices_to_check:
            item_indices = bins[bin_idx]
            bin_load = sum(self.items[i] for i in item_indices)
            
            if bin_load > self.bin_capacity:
                # Remove items until bin is valid
                items_in_bin = [(i, self.items[i]) for i in item_indices]
                items_in_bin.sort(key=lambda x: x[1])  # Sort by size
                
                current_load = bin_load
                for item_idx, item_size in items_in_bin:
                    if current_load <= self.bin_capacity:
                        break
                    # Try to place in another bin
                    placed = False
                    # Use list() to avoid dictionary change during iteration
                    for other_bin in list(bins.keys()):
                        if other_bin == bin_idx:
                            continue
                        other_load = sum(self.items[i] for i in bins[other_bin])
                        if other_load + item_size <= self.bin_capacity:
                            repaired[item_idx] = other_bin
                            bins[other_bin].append(item_idx)
                            bins[bin_idx].remove(item_idx)
                            current_load -= item_size
                            placed = True
                            break
                    
                    if not placed:
                        # Create new bin
                        new_bin_idx = max(bins.keys()) + 1
                        repaired[item_idx] = new_bin_idx
                        bins[new_bin_idx] = [item_idx]
                        bins[bin_idx].remove(item_idx)
                        current_load -= item_size
        
        return repaired
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform two-point crossover between two parents.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        
        size = len(parent1)
        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        # Repair children if needed
        child1 = self._repair_individual(child1)
        child2 = self._repair_individual(child2)
        
        return child1, child2
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """
        Mutate an individual by randomly changing bin assignments.
        """
        mutated = individual[:]
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Randomly assign to different bin
                max_bin = max(mutated)
                mutated[i] = random.randint(0, max_bin + 1)
        
        # Repair if needed
        mutated = self._repair_individual(mutated)
        
        return mutated
    
    def _tournament_selection(self, population: List[List[int]], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> List[int]:
        """
        Select an individual using tournament selection.
        """
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])
        return winner[0]
    
    def _evolve_population(self, population: List[List[int]]) -> List[List[int]]:
        """
        Evolve a population for one generation.
        """
        # Calculate fitness for all individuals
        fitness_scores = [self._calculate_fitness(ind) for ind in population]
        
        # Keep best individual (elitism)
        best_idx = np.argmin(fitness_scores)
        elite = population[best_idx][:]
        
        # Update global best
        if fitness_scores[best_idx] < self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_solution = elite[:]
        
        # Create new population
        new_population = [elite]
        
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _wisdom_of_crowds(self, crowds: List[List[List[int]]]) -> List[int]:
        """
        Combine solutions from multiple crowds using wisdom of crowds.
        """
        # Get best solution from each crowd
        best_from_crowds = []
        for crowd in crowds:
            fitness_scores = [self._calculate_fitness(ind) for ind in crowd]
            best_idx = np.argmin(fitness_scores)
            best_from_crowds.append(crowd[best_idx])
        
        # Create a consensus solution
        # Use voting mechanism: for each item, choose the most common bin assignment
        consensus = []
        for item_idx in range(len(self.items)):
            bin_votes = [ind[item_idx] for ind in best_from_crowds]
            # Normalize bin indices (since different crowds may use different numbering)
            most_common = max(set(bin_votes), key=bin_votes.count)
            consensus.append(most_common)
        
        # Repair and optimize the consensus solution
        consensus = self._repair_individual(consensus)
        
        return consensus
    
    def solve(self, verbose: bool = True, progress_callback=None) -> Tuple[List[List[float]], int, List[float], float]:
        """
        Solve the bin packing problem using GA+WoC.
        
        Args:
            verbose: Print progress information
            progress_callback: Optional callback function(current, total) for progress updates
        
        Returns:
            Tuple of (bins, number_of_bins, fitness_history, computation_time)
        """
        # Start timing
        start_time = time.time()
        
        # Initialize multiple crowds (populations)
        crowds = []
        for _ in range(self.crowd_size):
            crowd = [self._create_individual() for _ in range(self.population_size)]
            crowds.append(crowd)
        
        # Evolution process
        for generation in range(self.generations):
            # Evolve each crowd independently
            for i in range(self.crowd_size):
                crowds[i] = self._evolve_population(crowds[i])
            
            # Apply wisdom of crowds every 10 generations (only if WoC is enabled)
            if self.use_woc and generation % 10 == 0 and generation > 0:
                consensus = self._wisdom_of_crowds(crowds)
                # Inject consensus into each crowd
                # for i in range(self.crowd_size):
                #     crowds[i][0] = consensus[:]
                for i in range(self.crowd_size):
                    fitness_scores = [self._calculate_fitness(ind) for ind in crowds[i]]
                    worst_idx = np.argmax(fitness_scores)
                    crowds[i][worst_idx] = consensus[:]
            
            # Track fitness
            self.fitness_history.append(self.best_fitness)
            
            # Report progress
            if progress_callback:
                progress_callback(generation + 1, self.generations)
            
            if verbose and generation % 20 == 0:
                algo_name = "GA+WoC" if self.use_woc else "GA"
                print(f"Generation {generation} ({algo_name}): Best fitness = {self.best_fitness:.4f}, "
                      f"Bins = {int(self.best_fitness)}")
        
        # Convert best solution to bins
        bins = self._solution_to_bins(self.best_solution)
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        return bins, len(bins), self.fitness_history, computation_time
    
    def _solution_to_bins(self, solution: List[int]) -> List[List[float]]:
        """
        Convert a solution (chromosome) to actual bins.
        """
        bins_dict = {}
        for item_idx, bin_idx in enumerate(solution):
            if bin_idx not in bins_dict:
                bins_dict[bin_idx] = []
            bins_dict[bin_idx].append(self.items[item_idx])
        
        # Convert to list and sort bins by total load
        bins = list(bins_dict.values())
        bins.sort(key=lambda b: sum(b), reverse=True)
        
        return bins


def generate_random_items(num_items: int, min_size: float = 0.1, 
                         max_size: float = 1.0) -> List[float]:
    """
    Generate random items for testing.
    
    Args:
        num_items: Number of items to generate
        min_size: Minimum item size
        max_size: Maximum item size
    
    Returns:
        List of item sizes
    """
    return [round(random.uniform(min_size, max_size), 2) for _ in range(num_items)]


if __name__ == "__main__":
    # Example usage
    random.seed(42)
    
    # Generate test items
    items = generate_random_items(50, min_size=0.1, max_size=0.8)
    bin_capacity = 1.0
    
    print(f"Items: {items}")
    print(f"Bin capacity: {bin_capacity}")
    print(f"Number of items: {len(items)}")
    print("-" * 60)
    
    # Create and run GA+WoC solver
    solver = BinPackingGAWoC(
        items=items,
        bin_capacity=bin_capacity,
        population_size=100,
        generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8,
        crowd_size=5
    )
    
    bins, num_bins, fitness_history, computation_time = solver.solve(verbose=True)
    
    print("\n" + "=" * 60)
    print("SOLUTION:")
    print("=" * 60)
    print(f"Number of bins used: {num_bins}")
    print(f"Final fitness: {fitness_history[-1]:.4f}")
    print(f"Computation time: {computation_time:.2f} seconds")
    print("\nBin contents:")
    for i, bin_items in enumerate(bins):
        print(f"Bin {i+1}: {bin_items} (Load: {sum(bin_items):.2f}/{bin_capacity})")
