import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class ChargingSession:
    cs_id: int
    port_id: int
    energy: float
    cost: float
    time: float
    start_time: float
    stop_time: float
    node_id: int
    car_id: str

@dataclass
class ChargingRequest:
    car_id: str
    soc: float
    capacity: float
    current_node: int
    route: List[int]
    arrival_time: float = 0.0
    priority: float = 0.0
    
    def __lt__(self, other):
        return self.priority > other.priority

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

class OptimizationSolution:

    def __init__(self, assignments: Dict[str, ChargingSlot]):
        self.assignments = assignments  # car_id -> ChargingSlot
        self.fitness = 0.0
        self.total_waiting_time = 0.0
        self.total_cost = 0.0
        self.total_travel_time = 0.0
        self.port_utilization = 0.0
        
    def calculate_fitness(self, weights: Dict[str, float] = None):

        if weights is None:
            weights = {
                'waiting_time': 0.4,
                'travel_time': 0.3,
                'utilization': 0.1,
                'remaining_soc':0.5,
                'satisfied_facilities' : 0.2
            }
        
        # Calculate metrics
        self.total_waiting_time = sum(slot.waiting_time for slot in self.assignments.values())
        self.total_cost = sum(slot.cost for slot in self.assignments.values())
        self.total_travel_time = sum(slot.travel_time for slot in self.assignments.values())
        if len(self.assignments) <= 0:
            divider = 1
        else:
            divider = len(self.assignments)
        self.average_remaining_soc = sum(slot.remaining_soc for slot in self.assignments.values()) / divider
        self.average_satisfied_facilities = sum(slot.satisfied_facilities for slot in self.assignments.values())/ divider
        
        # Calculate port utilization (higher is better)
        used_ports = set((slot.cs_id, slot.port_id) for slot in self.assignments.values())
        self.port_utilization = len(used_ports) / max(1, len(self.assignments))
        
        # Normalize metrics (lower is better for waiting time, cost, travel time)
        max_waiting = max(1, self.total_waiting_time)
        max_cost = max(1, self.total_cost)
        max_travel = max(1, self.total_travel_time)
        
        # Fitness calculation (higher is better)
        self.fitness = (
            weights['waiting_time'] * (1 - self.total_waiting_time / max_waiting) +
            weights['travel_time'] * (1 - self.total_travel_time / max_travel) +
            weights['utilization'] * self.port_utilization +
            weights['remaining_soc'] * self.average_remaining_soc + 
            weights['satisfied_facilities'] * self.average_satisfied_facilities
        )
        
        return self.fitness



