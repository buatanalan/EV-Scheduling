import numpy as np
import random
import copy
from typing import List, Dict, Tuple
import logging
from run_agent.charging_optimize import OptimizationSolution, ChargingSlot


logger = logging.getLogger(__name__)

class Particle:
    
    def __init__(self, car_ids: List[str], available_slots: Dict[str, List]):
        self.car_ids = car_ids
        self.available_slots = available_slots
        self.position = self._initialize_position()
        self.velocity = self._initialize_velocity()
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = float('-inf')
        self.fitness = float('-inf')
    
    def _initialize_position(self) -> Dict[str, int]:

        position = {}
        for car_id in self.car_ids:
            if car_id in self.available_slots and self.available_slots[car_id]:
                position[car_id] = random.randint(0, len(self.available_slots[car_id]) - 1)
            else:
                position[car_id] = 0
        return position
    
    def _initialize_velocity(self) -> Dict[str, float]:

        return {car_id: random.uniform(-1, 1) for car_id in self.car_ids}
    
    def update_velocity(self, global_best_position: Dict[str, int], w=0.7, c1=1.5, c2=1.5):

        for car_id in self.car_ids:
            r1, r2 = random.random(), random.random()
            
            cognitive = c1 * r1 * (self.best_position[car_id] - self.position[car_id])
            social = c2 * r2 * (global_best_position[car_id] - self.position[car_id])
            
            self.velocity[car_id] = w * self.velocity[car_id] + cognitive + social
    
    def update_position(self):

        for car_id in self.car_ids:
            if car_id in self.available_slots and self.available_slots[car_id]:
                max_index = len(self.available_slots[car_id]) - 1
                
                new_pos = self.position[car_id] + self.velocity[car_id]
                
                self.position[car_id] = max(0, min(int(round(new_pos)), max_index))
    
    def get_solution(self):
        
        assignments = {}
        for car_id in self.car_ids:
            if (car_id in self.available_slots and 
                self.available_slots[car_id] and 
                0 <= self.position[car_id] < len(self.available_slots[car_id])):
                
                slot = self.available_slots[car_id][self.position[car_id]]
                slot_copy = copy.deepcopy(slot)
                slot_copy.car_id = car_id
                assignments[car_id] = slot_copy
        
        return OptimizationSolution(assignments)

class ParticleSwarmOptimizer:
    
    def __init__(self, swarm_size=30, max_iterations=100, w=0.7, c1=1.5, c2=1.5, weights=None):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w  
        self.c1 = c1 
        self.c2 = c2 
        self.weights = weights
    
    def optimize(self, requests, available_slots):

        
        logger.info(f"Starting PSO optimization for {len(requests)} cars")
        
        car_ids = [req.car_id for req in requests]

        swarm = []
        for _ in range(self.swarm_size):
            particle = Particle(car_ids, available_slots)
            swarm.append(particle)
        
        global_best_fitness = float('-inf')
        global_best_position = None
        global_best_solution = None
        # PSO main loop
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all particles
            for particle in swarm:
                solution = particle.get_solution()
                fitness = solution.calculate_fitness(self.weights)
                particle.fitness = fitness
                
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = copy.deepcopy(particle.position)
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = copy.deepcopy(particle.position)
                    global_best_solution = copy.deepcopy(solution)
            
            # Update velocities and positions
            for particle in swarm:
                particle.update_velocity(global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
            
            # Adaptive parameters
            self.w = max(0.4, self.w * 0.99)  # Decrease inertia over time
            
            if iteration % 20 == 0:
                # logger.info(f"PSO Iteration {iteration}: Best fitness = {global_best_fitness:.4f}")
                pass
        
        # logger.info(f"PSO optimization completed. Best fitness: {global_best_fitness:.4f}")
        return global_best_solution


