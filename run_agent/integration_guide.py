import logging
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
import networkx as nx
from dataclasses import dataclass, asdict
import copy
from abc import ABC, abstractmethod


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# STEP 2: Add these new classes to your system
@dataclass
class OptimizationSolution:
    """Represents a complete solution for the charging optimization problem"""
    def __init__(self, assignments: Dict[str, 'ChargingSlot']):
        self.assignments = assignments
        self.fitness = 0.0
        self.total_waiting_time = 0.0
        self.total_cost = 0.0
        self.total_travel_time = 0.0
        self.port_utilization = 0.0
        self.average_remaining_soc = 0.0
        self.average_satisfied_facilities = 0.0
        
    def calculate_fitness(self, weights: Dict[str, float] = None):
        """Calculate fitness score for this solution"""
        if weights is None:
            weights = {
                'waiting_time': 0.9,
                'travel_time': 0.005,
                'utilization': 0.01,
                'remaining_soc':0.0001,
                'satisfied_facilities' : 0.12
            }
        # Calculate metrics
        self.total_travel_time = sum(slot.travel_time for slot in self.assignments.values())
        self.average_remaining_soc = sum(slot.remaining_soc for slot in self.assignments.values()) / len(self.assignments)
        self.average_satisfied_facilities = sum(slot.satisfied_facilities for slot in self.assignments.values())/ len(self.assignments)
        # Calculate port utilization
        used_ports = set((slot.cs_id, slot.port_id) for slot in self.assignments.values())
        self.port_utilization = len(used_ports) / max(1, len(self.assignments))
        # Normalize and calculate fitness
        max_waiting = max(1, self.total_waiting_time)
        max_travel = max(1, self.total_travel_time)
        
        cs_counts = defaultdict(int)
        for slot in self.assignments.values():
            cs_counts[slot.cs_id] += 1

        # Calculate probability distribution (P(cs))
        total_assignments = sum(cs_counts.values())
        cs_probs = [count / total_assignments for count in cs_counts.values()]

        # Calculate entropy (higher is better: more spread)
        entropy = -sum(p * math.log(p + 1e-9) for p in cs_probs)

        # Normalize entropy
        max_entropy = math.log(len(cs_counts) + 1e-9)  # log(number of CS used)
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0
        # logger.info(f"param is {self.total_waiting_time} | {self.total_travel_time} | {self.port_utilization} | {self.average_satisfied_facilities} | {entropy_score}")
        # Final fitness (with entropy reward)
        self.fitness = (
            weights['waiting_time'] * (1 - self.total_waiting_time) +
            weights['travel_time'] * (1 - self.total_travel_time) +
            weights['utilization'] * self.port_utilization +
            weights['remaining_soc'] * self.average_remaining_soc + 
            weights['satisfied_facilities'] * self.average_satisfied_facilities
            + weights.get('cs_entropy', 0.5) * entropy_score  # reward for spread!
        )
    
    def update_fitness(self, requests: List):
        port_schedules = {}  # key: (cs_id, port_id), value: list of (start_time, end_time)
        total_waiting_time = 0

        for request in requests:
            if request.car_id in self.assignments:
                slot = self.assignments[request.car_id]
                port_key = (slot.cs_id, slot.port_id)

                # Initialize schedule list if not already
                if port_key not in port_schedules:
                    port_schedules[port_key] = []

                port_schedule = port_schedules[port_key]
                arrival_time = slot.start_time - slot.waiting_time
                # Start with requested start/end
                actual_start_time = slot.start_time
                duration = slot.end_time - slot.start_time

                # Adjust start time if overlaps
                for busy_start, busy_end in sorted(port_schedule):
                    if actual_start_time < busy_end and (actual_start_time + duration) > busy_start:
                        # Move start to end of this busy slot
                        actual_start_time = busy_end  

                # Final start/end
                actual_end_time = actual_start_time + duration

                # Compute waiting time
                waiting_time = max(0, actual_start_time - arrival_time)
                total_waiting_time += waiting_time

                # Update port schedule
                port_schedule.append((actual_start_time, actual_end_time))
                port_schedule.sort()
                port_schedules[port_key] = port_schedule

        self.total_waiting_time = total_waiting_time
        self.fitness = 1.0 / (1.0 + total_waiting_time)

@dataclass
class ChargingSlot:
    cs_id: int
    port_id: int
    node_id: int
    start_time: float
    end_time: float
    energy: float
    cost: float
    waiting_time: float = 0.0
    travel_time: float = 0.0
    car_id: str = ""
    remaining_soc:float = 0.0
    satisfied_facilities:int = 0.0

class MetaheuristicOptimizer(ABC):
    """Abstract base class for metaheuristic optimization algorithms"""
    
    @abstractmethod
    def optimize(self, requests: List, available_slots: Dict[str, List[ChargingSlot]]) -> OptimizationSolution:
        pass

class GeneticAlgorithmOptimizer(MetaheuristicOptimizer):
    """Genetic Algorithm implementation for charging optimization"""
    
    def __init__(self, population_size=50, generations=50, mutation_rate=0.5, crossover_rate=0.8, weights=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.weights = weights
        
    def optimize(self, requests: List, available_slots: Dict[str, List[ChargingSlot]]) -> OptimizationSolution:
        """Main optimization method using Genetic Algorithm"""
        if not requests or not available_slots:
            return OptimizationSolution({})
            
        # logger.info(f"Starting GA optimization for {len(requests)} cars")
        # Initialize population
        population = self._initialize_population(requests, available_slots)
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness for all solutions
            for solution in population:
                solution.calculate_fitness(self.weights)
                if solution.fitness > best_fitness:
                    best_fitness = solution.fitness
                    best_solution = copy.deepcopy(solution)
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep best solutions
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite_size = max(1, self.population_size // 10)
            new_population.extend(population[:elite_size])
            
            # Generate new offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, available_slots)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1, parent2])
            
            # Mutation
            for solution in new_population[elite_size:]:
                if random.random() < self.mutation_rate:
                    self._mutate(solution, available_slots)
            
            population = population[:self.population_size]
            
            if generation % 10 == 0:
                # logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
                pass
        
        # logger.info(f"GA optimization completed. Best fitness: {best_fitness:.4f}")
        return best_solution or OptimizationSolution({})
    
    
    def _initialize_population(self, requests: List, available_slots: Dict[str, List[ChargingSlot]]) -> List[OptimizationSolution]:
        """Initialize random population"""
        population = []
        
        # for _ in range(self.population_size):
        #     assignments = {}
        #     for request in requests:
        #         if request.car_id in available_slots and available_slots[request.car_id] is not None and len(available_slots[request.car_id])>0:
        #             slot = random.choice(available_slots[request.car_id])
        #             slot_copy = copy.deepcopy(slot)
        #             slot_copy.car_id = request.car_id
        #             assignments[request.car_id] = slot_copy
        #     solution = OptimizationSolution(assignments)
        #     solution.update_fitness(requests)
        #     population.append(solution)
        for _ in range(self.population_size):
            assignments = {}
            used_slots = {}  # key: (cs_id, port_id), value: list of (start_time, end_time)

            for request in requests:
                if request.car_id in available_slots and available_slots[request.car_id]:
                    valid_slots = []

                    for slot in available_slots[request.car_id]:
                        key = (slot.cs_id, slot.port_id)
                        overlaps = False
                        if key in used_slots:
                            for (used_start, used_end) in used_slots[key]:
                                # Check overlap: [a,b) overlaps with [c,d) if a < d and c < b
                                if slot.start_time < used_end and used_start < slot.end_time:
                                    overlaps = True
                                    break
                        if not overlaps:
                            valid_slots.append(slot)

                    if valid_slots:
                        slot = random.choice(valid_slots)
                        slot_copy = copy.deepcopy(slot)
                        slot_copy.car_id = request.car_id
                        assignments[request.car_id] = slot_copy

                        # Record this slot as used
                        key = (slot_copy.cs_id, slot_copy.port_id)
                        used_slots.setdefault(key, []).append((slot_copy.start_time, slot_copy.end_time))
            solution = OptimizationSolution(assignments)
            solution.update_fitness(requests)
            population.append(solution)

        return population
    
    def _tournament_selection(self, population: List[OptimizationSolution], tournament_size=30) -> OptimizationSolution:
        """Tournament selection for parent selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: OptimizationSolution, parent2: OptimizationSolution, 
                  available_slots: Dict[str, List[ChargingSlot]]) -> Tuple[OptimizationSolution, OptimizationSolution]:
        """Order crossover for creating offspring"""
        car_ids = list(parent1.assignments.keys())
        if len(car_ids) <= 1:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        crossover_point = random.randint(1, len(car_ids) - 1)
        
        child1_assignments = {}
        child2_assignments = {}
        
        # Copy first part from parents
        for i in range(crossover_point):
            car_id = car_ids[i]
            if car_id in parent2.assignments and car_id in parent1.assignments:
                child1_assignments[car_id] = copy.deepcopy(parent1.assignments[car_id])
                child2_assignments[car_id] = copy.deepcopy(parent2.assignments[car_id])
        
        # Copy second part from opposite parents
        for i in range(crossover_point, len(car_ids)):
            car_id = car_ids[i]
            if car_id in parent2.assignments and car_id in parent1.assignments:
                child1_assignments[car_id] = copy.deepcopy(parent2.assignments[car_id])
                child2_assignments[car_id] = copy.deepcopy(parent1.assignments[car_id])
        
        return OptimizationSolution(child1_assignments), OptimizationSolution(child2_assignments)
    
    def _mutate(self, solution: OptimizationSolution, available_slots: Dict[str, List[ChargingSlot]]):
        """Mutate a solution by changing random assignments"""
        car_ids = list(solution.assignments.keys())
        if not car_ids:
            return
        
        car_id = random.choice(car_ids)
        
        if car_id in available_slots and available_slots[car_id]:
            new_slot = random.choice(available_slots[car_id])
            new_slot_copy = copy.deepcopy(new_slot)
            new_slot_copy.car_id = car_id
            solution.assignments[car_id] = new_slot_copy

print("Step 1: Enhanced optimization classes defined")
