import json
import logging
import threading
import time
import random
import math
from typing import Dict, List, Any, Optional, DefaultDict
from collections import deque
import networkx as nx
from dataclasses import dataclass, asdict
import copy
import datetime
from haversine import haversine, Unit
from pyproj import Transformer
import paho.mqtt.client as mqtt
import redis
from charging_stations.refactored_charging_station import ChargingStation

from run_agent.integration_guide import OptimizationSolution, ChargingSlot, GeneticAlgorithmOptimizer
from run_agent.particle_swarm_optimizer import ParticleSwarmOptimizer

transformer_to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


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
    car_id: str = ""

@dataclass
class ChargingRequest:
    car_id: str
    soc: float
    capacity: float
    current_node: int
    route: List[int]
    facilities:List[str]
    arrival_time: float = 0.0
    priority: float = 0.0
    dest: int = 0
    time: int = 0
    alternate_radius: float = 1.0 
    energy_charged_at_current_cs: float = 0.0 
    
    def __lt__(self, other):
        return self.priority > other.priority

@dataclass
class Schedule:
    car_id: str
    chargingSessions: List[ChargingSession]

@dataclass
class PortSession:
    car_id: str
    start_time: float
    stop_time: float
    session_type: str = "charging"
    
    def __repr__(self):
        return f"PortSession(car_id={self.car_id}, start={self.start_time:.2f}, stop={self.stop_time:.2f}, type={self.session_type})"


class EnhancedAgent:
    def __init__(self, G, optimizer_type='genetic', weights=None, optimizer_params=None, time_factor=0.1):
        self.G = G
        self.schedules = []
        self.requestQueue:List[ChargingRequest] = []
        self.running = True
        self.port_reservations:Dict[tuple, List[PortSession]] = DefaultDict(list)
        self.cs_info = {}
        self.cs_objects : Dict[int, ChargingStation] = {}
        self.time_factor = time_factor
        self.mqtt_client = mqtt.Client(client_id="03-id-agent")
        self.waiting_queues: Dict[tuple, deque[ChargingRequest]] = DefaultDict(deque)

        #SETUP
        self.redisClient = redis.Redis(host='localhost', port=6379, db=0)
        
        self.mqtt_client.connect('localhost', 1883)
        self.mqtt_client.loop_start()

        # Default optimizer params
        if optimizer_params is None:
            optimizer_params = {}

        # Initialize metaheuristic optimizer
        if optimizer_type == 'genetic':
            self.optimizer = GeneticAlgorithmOptimizer(
                population_size=optimizer_params.get("population_size", 130),
                generations=optimizer_params.get("generations", 490),
                mutation_rate=optimizer_params.get("mutation_rate", 0.0894),
                crossover_rate=optimizer_params.get("crossover_rate", 0.6043),
                weights=weights
            )
        else:
            self.optimizer = ParticleSwarmOptimizer(
                swarm_size=optimizer_params.get("swarm_size", 30),
                max_iterations=optimizer_params.get("max_iterations", 50),
                w=optimizer_params.get("w", 0.7),
                c1=optimizer_params.get("c1", 1.5),
                c2=optimizer_params.get("c2", 1.5),
                weights=weights
            )
        
        self._initialize_cs_info()

        self.mqtt_client.subscribe(topic="agent/request/route")

        self.mqtt_client.message_callback_add(sub="agent/request/route", callback=self._on_request_route)
        
        logger.info(f"Enhanced Agent initialized with {optimizer_type} optimizer")
    
    def _initialize_cs_info(self):

        for node_id, node_data in self.G.nodes(data=True):
            if 'charging_station' in node_data:
                cs = node_data['charging_station']
                self.cs_info[cs.id] = {
                    'node_id': cs.node_id,
                    'name': getattr(cs, 'name', f"CS-{cs.id}"),
                    'lat': getattr(cs, 'lat', 0),
                    'lon': getattr(cs, 'lon', 0),
                    'ports': [],
                    'facilities' : getattr(cs, 'facilities', 0)
                }

                self.cs_objects[cs.id] = cs
                
                for port in cs.ports:
                    self.cs_info[cs.id]['ports'].append({
                        'id': port.id,
                        'power': port.power,
                        'price': port.price
                    })
                    
                    key = (cs.id, port.id)
                    self.port_reservations[key] = []
    

    def _process_requests(self):

        while self.running:
            try:                    
                self._optimize_charging_schedules()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing requests: {e}")
                time.sleep(1)

    def _on_request_route(self, client, userdata, message):

        payload = json.loads(message.payload.decode())
        car_id = payload.get("car_id")
        origin = payload.get("origin")
        destination = payload.get("destination")
        route = nx.shortest_path(self.G, origin, destination, weight='length')
        self.mqtt_client.publish(
            topic=f"car/route/{car_id}",
            payload=json.dumps({"route": route}),
        )

    def update_request(self):
        try:
            new_request = [asdict(req) for req in self.requestQueue]
            self.mqtt_client.publish("agent/request/updates", json.dumps({"update" : new_request}))
        except Exception as e:
            logger.error(f"Error handling update charging request: {e}")

    def get_request(self):
        try:
            req_strs = self.redisClient.lrange("queue:request", 0, -1)

            requests = []
            for req_str in req_strs:
                data = json.loads(req_str.decode())
                req_obj = ChargingRequest(**data)
                requests.append(req_obj)
            self.requestQueue = requests
        except Exception as e:
            logger.error(f"Error handling update charging request: {e}")

    def update_waiting(self):
        try:
            for key, values in self.waiting_queues.items():
                new_waiting = [asdict(req) for req in values]
                self.mqtt_client.publish("agent/waiting/charging", json.dumps({"key" : key,"update" : new_waiting}))
        except Exception as e:
            logger.error(f"Error handling update charging request: {e}")

    def get_waiting(self):
        try:
            keys = self.redisClient.keys("queue:waiting")

            for key in keys:
                val = self.redisClient.get(key)
                if val:
                    decoded_list = json.loads(val.decode())
                    cast_list = [ChargingRequest(**req) for req in decoded_list]
                    self.waiting_queues[key.decode()] = cast_list

        except Exception as e:
            logger.error(f"Error handling update charging request: {e}")


    def _optimize_charging_schedules(self):

        try:
            self.cleaned_old_session()

            self.get_request()
            self.get_waiting()
            all_current_requests_for_optimization: Dict[str, ChargingRequest] = {}

            for req in self.requestQueue:
                all_current_requests_for_optimization[req.car_id] = req

            for q_key in self.waiting_queues:
                for waiting_req in self.waiting_queues[q_key]:

                    if waiting_req.car_id not in all_current_requests_for_optimization:
                        all_current_requests_for_optimization[waiting_req.car_id] = waiting_req

            active_requests_for_optimization = []
            for req_id, req in all_current_requests_for_optimization.items():
                car_twin = self.get_car_update(req.car_id)
                if car_twin and not car_twin.get('isFinish', False):

                    updated_req = ChargingRequest(
                        car_id=req.car_id,
                        soc=car_twin.get("soc", req.soc),
                        capacity=car_twin.get("capacity", req.capacity),
                        current_node=car_twin.get("current_node", req.current_node),
                        route=car_twin.get("route", req.route),
                        facilities=car_twin.get("facilities", req.facilities),
                        arrival_time=req.arrival_time, 
                        priority=req.priority, 
                        dest=car_twin.get("dest", req.dest),
                        time=car_twin.get("time", req.time), 
                        alternate_radius=car_twin.get("alternate_radius", req.alternate_radius)
                    )

                    updated_req.priority = 1.0 - (updated_req.soc / updated_req.capacity) if updated_req.capacity > 0 else 0.0
                    active_requests_for_optimization.append(updated_req)

            if not active_requests_for_optimization:
                self.requestQueue = []
                self.update_request()
                return

            # Sort all active requests by priority
            active_requests_for_optimization.sort(key=lambda r: r.priority, reverse=True)
            self.requestQueue = active_requests_for_optimization 
            self.update_request()

            start_optimization_time = datetime.datetime.now()
            logger.info(f"Optimizing schedules for {len(self.requestQueue)} cars using metaheuristics at {start_optimization_time.isoformat()}")

            available_slots = {}
            for request in self.requestQueue:
                car_twin_state = self.get_car_update(request.car_id)
                
                slots = []
                if car_twin_state.get('current_node') is not None and 'charging_station' in self.G.nodes.get(car_twin_state.get('current_node'), {}):
                    slots = self._find_all_charging_slots_new_scheme(request)
                elif car_twin_state.get('driving') == True: 
                    slots = self._find_all_charging_slots_enhanced(request)
                
                if slots:
                    available_slots[request.car_id] = slots
                else:

                    cs_at_node = self._get_cs_at_node(request.current_node)
                    if cs_at_node and cs_at_node.ports:
                        port_key = (cs_at_node.id, cs_at_node.ports[0].id) 
                        if request.car_id not in [r.car_id for r in self.waiting_queues[port_key]]:
                            self.waiting_queues[port_key].append(request)
                            logger.info(f"[Agent] Car {request.car_id} added to waiting queue for CS {cs_at_node.id}.")

            self.update_waiting()
            best_solution = self.optimizer.optimize(self.requestQueue, available_slots)

            if best_solution and best_solution.assignments:
                self._apply_optimized_solution(best_solution)
                # logger.info(f"  Optimization completed at {self.env.now:.2f}:")
                # logger.info(f"  Total waiting time: {best_solution.total_waiting_time:.2f}")
                # logger.info(f"  Total cost: {best_solution.total_cost:.2f}")
                # logger.info(f"  Total travel time: {best_solution.total_travel_time:.2f}")
                # logger.info(f"  Port utilization: {best_solution.port_utilization:.2f}")
                # logger.info(f"  Fitness score: {best_solution.fitness:.4f}")
            else:
                logger.info(f"No optimal solution found or assignments empty at {datetime.datetime.now()}.")

            now = datetime.datetime.now()
            duration = (now-start_optimization_time).total_seconds()
            self.mqtt_client.publish(
                topic="agent/searching",
                payload=json.dumps({
                    "timestamp": now.isoformat(),
                    "duration": duration,
                    "cars_involved" : len(self.requestQueue), "success_rate" : len(best_solution.assignments)/len(self.requestQueue) if len(self.requestQueue)>0 else 0
                })
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced optimization: {e}", exc_info=True)

    def _get_next_available_time(self, port_sessions: List[PortSession], current_sim_time: float) -> float:

        if not port_sessions:
            return current_sim_time

        sorted_sessions = sorted(port_sessions, key=lambda s: s.start_time) 
        
        available_time = current_sim_time
        for session in sorted_sessions:
            if session.start_time >= available_time:
                return available_time
            else:
                available_time = max(available_time, session.stop_time)
        return available_time

    def _find_all_charging_slots_enhanced(self, request: ChargingRequest) -> List[ChargingSlot]:

        possible_slots = []

        if len(request.route)==0 :
            return
        current_node = request.current_node
        current_index = request.route.index(current_node)
        next_route = request.route[current_index+6:]
        cs_along_route = self._get_cs_along_route(next_route, request.alternate_radius)
        current_time = request.time
        current_soc = request.soc
        found_cs = False
        try:
            distance = nx.shortest_path_length(self.G, current_node, request.dest, weight='length')
        except (nx.NetworkXNoPath, KeyError):
            return 
        energy_required = self._calculate_energy_consumption(distance) + 2
        if request.soc > energy_required:
            return None
        for cs in cs_along_route:
            try:
                if current_node is None or cs.node_id is None:
                    continue
                try:
                    distance = nx.shortest_path_length(self.G, current_node, cs.node_id, weight='length')
                except (nx.NetworkXNoPath, KeyError):
                    continue
                
                energy_required = self._calculate_energy_consumption(distance)
                travel_time = self._calculate_travel_time(distance) / self.time_factor

                if energy_required >= current_soc:
                    continue
                found_cs = True
                arrival_time = current_time + travel_time
                remaining_soc = current_soc - energy_required
                energy_to_charge = min(request.capacity * 0.8 - remaining_soc, request.capacity * 0.6)
                # logger.info(f"[Car {request.car_id}] predicted arrival time at {cs.node_id} is {arrival_time}")
                distance_cs_to_dest = nx.shortest_path_length(self.G, cs.node_id, request.dest, weight='length')
                energy_needed_cs_to_dest = self._calculate_energy_consumption(distance_cs_to_dest)
                if energy_to_charge <= 0:
                    energy_to_charge = min(request.capacity, energy_needed_cs_to_dest)
                satisfied_facilities = sum(1 for f in (request.facilities or []) if f in (cs.facilities or []))
                # self.cleaned_old_session()
                # print('===============')
                # print(len(cs_along_route))
                # print(arrival_time)
                # print(cs.id)
                # Check each port
                cs_rest_route = self._find_cs_rest_of_route(request=request, cs_node=cs.node_id)
                if cs_rest_route:
                    min_energy = min(self._get_minimum_amount_charge(cs.node_id, cs_rest_route, energy_to_charge), energy_needed_cs_to_dest)
                else:
                    min_energy = energy_to_charge - 1

                for port in cs.ports:

                    # Loop dari min_energy (dibulatkan) + 1 sampai energy_to_charge (dibulatkan)
                    for energy in range(round(min_energy) + 2, max(round(min_energy)+2,round(energy_to_charge) + 2)):
                        # Hitung charging time untuk energy ini
                        charging_time = energy / port.power * 3600 / self.time_factor

                        # Cari slot yang memungkinkan
                        key = (cs.id, port.id)
                        port_sessions = self.port_reservations.get(key, [])
                        possible_start_times = self._find_possible_start_times(port_sessions, arrival_time, charging_time, current_time)
                        # logger.info(f"possible start time is {possible_start_times[0]}")
                        for start_time in possible_start_times:
                            end_time = start_time + charging_time
                            waiting_time = (start_time - arrival_time) / 60 * self.time_factor # dalam menit
                            cost = energy * port.price

                            slot = ChargingSlot(
                                cs_id=cs.id,
                                port_id=port.id,
                                node_id=cs.node_id,
                                start_time=start_time,
                                end_time=end_time,
                                energy=energy,
                                cost=cost,
                                waiting_time=max(0, waiting_time),
                                travel_time=travel_time / 60,  # dalam menit
                                car_id=request.car_id,
                                remaining_soc=request.soc - energy_required,
                                satisfied_facilities=satisfied_facilities
                            )

                            possible_slots.append(slot)

            except Exception as e:
                logger.warning(f"Error finding slots at CS {cs.id}: {e}")
                continue
        
        if not found_cs:
            logger.warning(f"No suitable CS found along route for car {request.car_id}, trying alternate CS...")

            alternate_slots = self._find_alternate_cs(
                current_node=current_node,
                current_soc=current_soc,
                current_time=current_time,
                request=request
            )

            possible_slots.extend(alternate_slots)
            request.alternate_radius *= 1.5
        # logger.info(f"possible slot for {request.car_id} is {set([slot.cs_id for slot in possible_slots])} slots")
        return possible_slots

    def _find_all_charging_slots_new_scheme(self, request: ChargingRequest) -> List[ChargingSlot]:

        possible_slots = []
        if not request.route or request.current_node is None:
            return []
        logger.info(f"[Car {request.car_id}] is request on new schema")
        current_node = request.current_node
        current_time = request.time
        current_soc = request.soc

        cs_current = self._get_cs_at_node(current_node)
        if cs_current is None:
            logger.warning(f"No CS at node {current_node} for car {request.car_id}")
            return []

        cs_rest_route = self._find_cs_rest_of_route(request=request, cs_node=cs_current.node_id)
        energy_to_charge = min(request.capacity * 0.8 - current_soc, request.capacity * 0.6)
        energy_to_charge = max(0, energy_to_charge) + 2
        if cs_rest_route:
            min_energy = self._get_minimum_amount_charge(cs_current.node_id, cs_rest_route, energy_to_charge)
        else:
            min_energy = energy_to_charge - 2
        target_energy = max(energy_to_charge, min_energy)

        satisfied_facilities = sum(1 for f in (request.facilities or []) if f in (cs_current.facilities or []))

        for port in cs_current.ports:
            key = (cs_current.id, port.id)
            port_sessions = self.port_reservations.get(key, [])
            waiting_queue = self.waiting_queues.get(key, deque())

            earliest_available_port_time = self._get_next_available_time(port_sessions, current_time)

            charging_time_needed_sec_for_request = (target_energy / port.power) * 3600 / self.time_factor
            
            charge_start_time_current_car = earliest_available_port_time
            charge_end_time_current_car = charge_start_time_current_car + charging_time_needed_sec_for_request
            
            waiting_duration_for_request = max(0, charge_start_time_current_car - request.arrival_time) * self.time_factor
            
            full_slot_current_car = ChargingSlot(
                cs_id=cs_current.id,
                port_id=port.id,
                node_id=cs_current.node_id,
                start_time=charge_start_time_current_car,
                end_time=charge_end_time_current_car,
                energy=target_energy,
                cost=target_energy * port.price,
                waiting_time=waiting_duration_for_request,
                travel_time=0, 
                car_id=request.car_id,
                remaining_soc=current_soc + target_energy, 
                satisfied_facilities=satisfied_facilities
            )
            possible_slots.append(full_slot_current_car)

            initial_charge_duration_sec = min(charging_time_needed_sec_for_request, 300 / self.time_factor)
            
            if initial_charge_duration_sec > 0 and charging_time_needed_sec_for_request > initial_charge_duration_sec:

                split1_end_time_car1 = charge_start_time_current_car + initial_charge_duration_sec
                energy1 = port.power * initial_charge_duration_sec / 3600 * self.time_factor
                cost1 = energy1 * port.price

                slot1_car1 = ChargingSlot(
                    cs_id=cs_current.id,
                    port_id=port.id,
                    node_id=cs_current.node_id,
                    start_time=charge_start_time_current_car,
                    end_time=split1_end_time_car1,
                    energy=energy1,
                    cost=cost1,
                    waiting_time=waiting_duration_for_request,
                    travel_time=0,
                    car_id=request.car_id,
                    remaining_soc=current_soc + energy1,
                    satisfied_facilities=satisfied_facilities
                )
                possible_slots.append(slot1_car1)

                if waiting_queue:
                    next_waiting_request = waiting_queue[0] 
                    
                    waiting_car_target_energy = min(next_waiting_request.capacity * 0.8 - next_waiting_request.soc, next_waiting_request.capacity * 0.6)
                    waiting_car_target_energy = max(0, waiting_car_target_energy) 

                    if waiting_car_target_energy > 0:
                        waiting_car_charging_time_sec = (waiting_car_target_energy / port.power) * 3600
                        
                        waiting_car_start_charge_time = split1_end_time_car1
                        waiting_car_end_charge_time = waiting_car_start_charge_time + waiting_car_charging_time_sec

                        slot_waiting_car_charge = ChargingSlot(
                            cs_id=cs_current.id,
                            port_id=port.id,
                            node_id=cs_current.node_id,
                            start_time=waiting_car_start_charge_time,
                            end_time=waiting_car_end_charge_time,
                            energy=waiting_car_target_energy,
                            cost=waiting_car_target_energy * port.price,
                            waiting_time=max(0, waiting_car_start_charge_time - next_waiting_request.arrival_time),
                            travel_time=0,
                            car_id=next_waiting_request.car_id,
                            remaining_soc=next_waiting_request.soc + waiting_car_target_energy,
                            satisfied_facilities=sum(1 for f in (next_waiting_request.facilities or []) if f in (cs_current.facilities or []))
                        )
                        possible_slots.append(slot_waiting_car_charge)

                        remaining_energy_car1 = target_energy - energy1
                        if remaining_energy_car1 > 0:
                            slot2_car1_start_time = waiting_car_end_charge_time
                            slot2_car1_charging_duration = (remaining_energy_car1 / port.power) * 3600 * self.time_factor
                            slot2_car1_end_time = slot2_car1_start_time + slot2_car1_charging_duration
                            cost2 = remaining_energy_car1 * port.price

                            slot2_car1 = ChargingSlot(
                                cs_id=cs_current.id,
                                port_id=port.id,
                                node_id=cs_current.node_id,
                                start_time=slot2_car1_start_time,
                                end_time=slot2_car1_end_time,
                                energy=remaining_energy_car1,
                                cost=cost2,
                                waiting_time=max(0, slot2_car1_start_time - request.arrival_time),
                                car_id=request.car_id,
                                remaining_soc=current_soc + energy1 + remaining_energy_car1,
                                satisfied_facilities=satisfied_facilities
                            )
                            possible_slots.append(slot2_car1)
                else:

                    remaining_energy_car1 = target_energy - energy1
                    if remaining_energy_car1 > 0:
                        slot2_car1_start_time = split1_end_time_car1
                        slot2_car1_charging_duration = (remaining_energy_car1 / port.power) * 3600 * self.time_factor
                        slot2_car1_end_time = slot2_car1_start_time + slot2_car1_charging_duration
                        cost2 = remaining_energy_car1 * port.price
                        
                        slot2_car1 = ChargingSlot(
                            cs_id=cs_current.id,
                            port_id=port.id,
                            node_id=cs_current.node_id,
                            start_time=slot2_car1_start_time,
                            end_time=slot2_car1_end_time,
                            energy=remaining_energy_car1,
                            cost=cost2,
                            waiting_time=waiting_duration_for_request,
                            travel_time=0,
                            car_id=request.car_id,
                            remaining_soc=current_soc + energy1 + remaining_energy_car1,
                            satisfied_facilities=satisfied_facilities
                        )
                        possible_slots.append(slot2_car1)
            

            if waiting_queue:

                current_consideration_time_for_queue = earliest_available_port_time

                for waiting_req in waiting_queue:

                    if waiting_req.car_id == request.car_id:
                        continue 
                    
                    wc_energy_to_charge = min(waiting_req.capacity * 0.8 - waiting_req.soc, waiting_req.capacity * 0.6)
                    wc_energy_to_charge = max(0, wc_energy_to_charge)
                    
                    if wc_energy_to_charge <= 0:
                        continue 
                    
                    wc_charging_time = (wc_energy_to_charge / port.power) * 3600 * self.time_factor
                    
                    potential_start_time_for_wc = current_consideration_time_for_queue
                    potential_end_time_for_wc = potential_start_time_for_wc + wc_charging_time
                    
                    wc_waiting_time = max(0, potential_start_time_for_wc - waiting_req.arrival_time)
                    
                    wc_slot = ChargingSlot(
                        cs_id=cs_current.id,
                        port_id=port.id,
                        node_id=cs_current.node_id,
                        start_time=potential_start_time_for_wc,
                        end_time=potential_end_time_for_wc,
                        energy=wc_energy_to_charge,
                        cost=wc_energy_to_charge * port.price,
                        waiting_time=wc_waiting_time,
                        travel_time=0,
                        car_id=waiting_req.car_id,
                        remaining_soc=waiting_req.soc + wc_energy_to_charge,
                        satisfied_facilities=sum(1 for f in (waiting_req.facilities or []) if f in (cs_current.facilities or []))
                    )
                    possible_slots.append(wc_slot)
                    
                    current_consideration_time_for_queue = potential_end_time_for_wc

        return possible_slots


    def _get_cs_at_node(self, current_node: int) -> Optional[ChargingStation]:
        return self.G.nodes.get(current_node, {}).get('charging_station', None)

    def _find_alternate_cs(self, current_node: int, current_soc: float, current_time: float, request: ChargingRequest) -> List[ChargingSlot]:

        alternate_slots = []
        
        # Convert current_node coordinates to lat/lon for distance calculation
        try:
            current_node_data = self.G.nodes[current_node]
            current_lon_utm, current_lat_utm = current_node_data['x'], current_node_data['y']
            current_lon_geo, current_lat_geo = transformer_to_latlon.transform(current_lon_utm, current_lat_utm)
        except KeyError:
            logger.warning(f"Current node {current_node} data not found for alternate CS search.")
            return []

        cs_with_distance = []
        for cs_id, cs_obj in self.cs_objects.items(): # Iterate through actual CS objects
            try:
                cs_node_id = cs_obj.node_id
                cs_node_data = self.G.nodes[cs_node_id]
                cs_lon_utm, cs_lat_utm = cs_node_data['x'], cs_node_data['y']
                cs_lon_geo, cs_lat_geo = transformer_to_latlon.transform(cs_lon_utm, cs_lat_utm)
                
                distance_km = haversine(
                    (current_lat_geo, current_lon_geo),
                    (cs_lat_geo, cs_lon_geo),
                    unit=Unit.KILOMETERS
                )
                
                if distance_km <= request.alternate_radius: 
                    cs_with_distance.append((cs_obj, distance_km))
            except KeyError:
                logger.debug(f"CS node {cs_obj.node_id} data not found, skipping for alternate search.")
            except Exception as e:
                logger.warning(f"Error calculating distance to CS {cs_id} for alternate search: {e}")
                continue

        cs_with_distance.sort(key=lambda x: x[1])

        for cs, distance_km in cs_with_distance:
            try:
                energy_required_to_reach_cs = self._calculate_energy_consumption(distance_km) 
                travel_time_to_cs = self._calculate_travel_time(distance_km) * self.time_factor
                
                if energy_required_to_reach_cs >= current_soc:
                    continue # Cannot reach this alternate CS

                arrival_time_at_cs = current_time + travel_time_to_cs
                remaining_soc_at_cs = current_soc - energy_required_to_reach_cs
                
                energy_to_charge = min(request.capacity * 0.8 - remaining_soc_at_cs, request.capacity * 0.6)
                energy_to_charge = max(0, energy_to_charge)

                try:
                    distance_cs_to_dest = nx.shortest_path_length(self.G, cs.node_id, request.dest, weight='length')
                    energy_needed_cs_to_dest = self._calculate_energy_consumption(distance_cs_to_dest)
                    energy_to_charge = max(energy_to_charge, energy_needed_cs_to_dest * 1.2 - remaining_soc_at_cs)
                    energy_to_charge = min(energy_to_charge, request.capacity - remaining_soc_at_cs) 
                    energy_to_charge = max(0, energy_to_charge)

                except (nx.NetworkXNoPath, KeyError):
                    logger.warning(f"No path from alternate CS {cs.node_id} to destination {request.dest} for car {request.car_id}. Using default charge amount.")

                if energy_to_charge <= 0:
                    continue

                satisfied_facilities = sum(1 for f in (request.facilities or []) if f in (cs.facilities or []))

                for port in cs.ports:
                    charging_time_sec = (energy_to_charge / port.power) * 3600 * self.time_factor
                    
                    key = (cs.id, port.id)
                    port_sessions = self.port_reservations.get(key, [])
                    
                    start_time = self._get_next_available_time(port_sessions, arrival_time_at_cs)
                    end_time = start_time + charging_time_sec
                    
                    
                    waiting_time_sec = max(0, start_time - arrival_time_at_cs)

                    slot = ChargingSlot(
                        cs_id=cs.id,
                        port_id=port.id,
                        node_id=cs.node_id,
                        start_time=start_time,
                        end_time=end_time,
                        energy=energy_to_charge,
                        cost=energy_to_charge * port.price,
                        waiting_time=waiting_time_sec,
                        travel_time=travel_time_to_cs,
                        car_id=request.car_id,
                        remaining_soc=remaining_soc_at_cs + energy_to_charge,
                        satisfied_facilities=satisfied_facilities
                    )
                    alternate_slots.append(slot)
                    
                    break
            except Exception as e:
                logger.warning(f"Error generating alternate slot for CS {cs.id} for car {request.car_id}: {e}")
                continue
        
        if not alternate_slots and request.alternate_radius < 50:
            request.alternate_radius += 1.5
            logger.info(f"Increasing alternate search radius for car {request.car_id} to {request.alternate_radius:.2f} km.")

        return alternate_slots

    def _apply_optimized_solution(self, solution: OptimizationSolution):
        
        scheduled_car_ids_this_run = set()
        
        # Process assignments from the solution
        for car_id, slot in solution.assignments.items():
            scheduled_car_ids_this_run.add(car_id)

            session_duration = slot.end_time - slot.start_time
            session = ChargingSession(
                cs_id=slot.cs_id,
                port_id=slot.port_id,
                energy=slot.energy,
                cost=slot.cost,
                time=session_duration,
                start_time=slot.start_time,
                stop_time=slot.end_time,
                node_id=slot.node_id,
                car_id=car_id
            )
            
            existing_schedule = next((s for s in self.schedules if s.car_id == car_id), None)
            if existing_schedule:

                existing_schedule.chargingSessions = [session] 
            else:
                schedule = Schedule(car_id=car_id, chargingSessions=[session])
                self.schedules.append(schedule)
            
            # Update port reservations using the helper
            key = (slot.cs_id, slot.port_id)
            
            new_port_reservations_for_key = []
            
            for existing_session in self.port_reservations.get(key, []):

                if existing_session.car_id == car_id and \
                   max(existing_session.start_time, slot.start_time) < min(existing_session.stop_time, slot.end_time):
                    continue

                new_port_reservations_for_key.append(existing_session)
            
            new_port_reservations_for_key.append(
                PortSession(
                    car_id=car_id,
                    start_time=slot.start_time,
                    stop_time=slot.end_time,
                    session_type="charging" 
                )
            )
            
            self.port_reservations[key] = sorted(new_port_reservations_for_key, key=lambda s: s.start_time)

            # Send schedule to car
            schedule =  next((s for s in self.schedules if s.car_id == car_id), None)
            self._send_schedule_to_car(car_id, schedule)
            self.build_new_route(car_id, slot.node_id)
        self.update_schedule()
        
        for req in self.requestQueue:
            if req.car_id not in solution.assignments.keys():
                self._send_schedule_to_car(car_id, Schedule(car_id=car_id, chargingSessions=[]))

        for key in list(self.waiting_queues.keys()):
            self.waiting_queues[key] = deque([
                req for req in self.waiting_queues[key] 
                if req.car_id not in scheduled_car_ids_this_run
            ])
            if not self.waiting_queues[key]:
                del self.waiting_queues[key]
        
        self.update_waiting()

    def update_schedule(self):
        try:
            new_schedules = [asdict(schedule) for schedule in self.schedules]
            self.mqtt_client.publish(topic='schedule/global', payload=json.dumps({"new_schedule" : new_schedules}))

        except Exception as e:
            pass

    def build_new_route(self, car_id, cs_node_id):

        car_twin = self.get_car_update(car_id)
        prev_route = car_twin.get("route")
        current_node = car_twin.get("current_node")
        dest = car_twin.get("dest")
        try:
            current_index = prev_route.index(current_node) + 2
        except ValueError:
            current_index = 0

        partial_route = prev_route[0 : current_index]

        if current_index < 0 or len(partial_route) == 0:
            logger.warning(f"[Agent] Invalid current_index {current_index} or empty partial_route")
            return

        path_to_cs = nx.shortest_path(self.G, source=partial_route[-1], target=cs_node_id, weight='length')

        path_cs_to_dest = nx.shortest_path(self.G, source=cs_node_id, target=dest, weight='length')

        new_route = partial_route + path_to_cs[1:] + path_cs_to_dest[1:]
        # if len(new_route) != len(set(new_route)):
        #     logging.info(f"ada duplicate route pada {car_id}")
        # logger.info(f"[Agent] New route composed: partial {len(partial_route)} + to_cs {len(path_to_cs)} + cs_to_dest {len(path_cs_to_dest)} = total {len(new_route)}")

        self.mqtt_client.publish(
            topic=f"car/route/{car_id}",
            payload=json.dumps({"route": new_route}),
        )
        # logger.info(f"new route for {car_id}")
        # print(new_route)

    def _get_cs_along_route(self, route: List[int], tolerance_distance_km: float) -> List[ChargingStation]:
        visited_cs_nodes = set()
        cs_list = []

        all_cs_nodes = {node_id for node_id, data in self.G.nodes(data=True) if 'charging_station' in data}
        
        cs_geo_coords = {}
        for cs_node_id in all_cs_nodes:
            try:
                x, y = self.G.nodes[cs_node_id]['x'], self.G.nodes[cs_node_id]['y']
                lon, lat = transformer_to_latlon.transform(x, y)
                cs_geo_coords[cs_node_id] = (lat, lon)
            except KeyError:
                logger.debug(f"CS node {cs_node_id} coordinates not found.")
                continue

        for node_id_on_route in route:
            if len(cs_list) > 10: 
                break
            
            try:
                x1, y1 = self.G.nodes[node_id_on_route]['x'], self.G.nodes[node_id_on_route]['y']
                lon1, lat1 = transformer_to_latlon.transform(x1, y1)
            except KeyError:
                logger.debug(f"Route node {node_id_on_route} coordinates not found.")
                continue

            for cs_node_id in all_cs_nodes:
                if cs_node_id in visited_cs_nodes:
                    continue

                if cs_node_id not in cs_geo_coords: 
                    continue

                lat2, lon2 = cs_geo_coords[cs_node_id]
                distance_km = haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)
                
                if distance_km < tolerance_distance_km:
                    cs_obj = self.G.nodes[cs_node_id]["charging_station"]
                    cs_list.append(cs_obj)
                    visited_cs_nodes.add(cs_node_id)
        
        return cs_list
    
    def _send_schedule_to_car(self, car_id: str, schedule: Schedule):

        try:
            sessions_data = []
            for session in schedule.chargingSessions:
                cs_info = self.cs_info.get(session.cs_id, {})
                
                sessions_data.append({
                    "stationId": session.cs_id,
                    "stationName": cs_info.get('name', f"CS-{session.cs_id}"),
                    "location": {
                        "latitude": cs_info.get('lat', 0),
                        "longitude": cs_info.get('lon', 0)
                    },
                    "portNumber": session.port_id,
                    "startTime": session.start_time,
                    "endTime": session.stop_time,
                    "energyAmount": session.energy,
                    "cost": session.cost,
                    "node_id": session.node_id,
                    "car_id": session.car_id 
                })
                logger.info(f"Sent charging schedule with {session.node_id} sessions to car {car_id}")
            
            self.mqtt_client.publish(
                topic=f"car/charging/{car_id}",
                payload=json.dumps({"charge_sessions": sessions_data})
            )
            
        except Exception as e:
            logger.error(f"Error sending schedule to car {car_id}: {e}", exc_info=True)

    def _find_cs_rest_of_route(self, request: ChargingRequest, cs_node):
        try:
            current_node = request.current_node
            route = request.route

            if not route or current_node not in route:
                logger.warning(f"[Agent] Route not valid or current node {current_node} not in route.")
                return []

            idx = self.find_nearest_node_idx_euclid(route, cs_node) + 10

            rest_route = route[idx + 1:] if idx + 1 < len(route) else []

            cs_rest_route = self._get_cs_along_route(rest_route, tolerance_distance_km=1)

            # logger.info(f"[Agent] Found {len(cs_rest_route)} charging stations along rest of route after node {current_node}.")
            return [cs.node_id for cs in cs_rest_route]

        except Exception as e:
            logger.error(f"[Agent] Failed to find CS along rest of route: {e}")
            return []

    def get_optimization_stats(self) -> Dict:
        if not self.schedules:
            return {}
        
        total_cost = sum(
            session.cost
            for schedule in self.schedules 
            for session in schedule.chargingSessions
        )
        
        return {
            'total_cars_scheduled': len(self.schedules),
            'total_sessions': sum(len(s.chargingSessions) for s in self.schedules),
            'total_cost': total_cost,
            'average_cost_per_car': total_cost / len(self.schedules) if self.schedules else 0,
            'optimizer_type': type(self.optimizer).__name__
        }

    def stop(self):
        """Keep your existing implementation"""
        self.running = False
        # The processing_thread is now a SimPy process, no need for .join() here directly.
        # SimPy will handle process termination when env.run() stops.
        logger.info("Enhanced Agent stopped")

    def cleaned_old_session(self):
        cleaned_session = {}

        for key in list(self.port_reservations.keys()):
            new_port_session = []
            sessions = self.port_reservations[key]

            for session in sessions:
                car_state = self.get_car_update(session.car_id)
                if car_state.get("driving") == True:
                    continue  
                
                new_port_session.append(session)

            self.port_reservations[key] = []

            cleaned_session[key] = len(new_port_session)

        return cleaned_session

    def update_charging_request_from_twin(self, car_id: str, alternate_radius:float):
        try:
            car_twin = self.get_car_update(car_id)
            if not car_twin:
                logger.warning(f"[update] No digital twin found for car {car_id}")
                return
            
            state = car_twin
            
            if state.get('isFinish', False) or state.get("soc", 0) <= 0:
                self.requestQueue = [req for req in self.requestQueue if req.car_id != car_id]
                self.update_request()
                for q_key in list(self.waiting_queues.keys()):
                    self.waiting_queues[q_key] = deque([req for req in self.waiting_queues[q_key] if req.car_id != car_id])
                    if not self.waiting_queues[q_key]: del self.waiting_queues[q_key]
                self.update_waiting()
                return
            
            soc = state.get("soc", 50)
            capacity = state.get("capacity", 100)
            current_node = state.get("current_node")
            route = state.get("route", [])
            dest=state.get("dest")
            facilities=state.get("facilities", [])
            time=state.get("time")

            if current_node is None:
                logger.warning(f"[update] Incomplete twin state for car {car_id} (missing current_node).")
                return
            
            priority = 1.0 - (soc / capacity) if capacity > 0 else 0.0

            existing_request = next((req for req in self.requestQueue if req.car_id == car_id), None)

            if existing_request:
                existing_request.soc = soc
                existing_request.capacity = capacity
                existing_request.current_node = current_node
                existing_request.route = route
                existing_request.dest = dest
                existing_request.facilities = facilities
                existing_request.priority = priority
                existing_request.time = time 
                existing_request.alternate_radius = alternate_radius 
            else:
                new_request = ChargingRequest(
                    car_id=car_id,
                    soc=soc,
                    capacity=capacity,
                    current_node=current_node,
                    route=route,
                    arrival_time=self.env.now, 
                    priority=priority,
                    dest=dest,
                    facilities=facilities,
                    time=time,
                    alternate_radius=alternate_radius
                )
                self.requestQueue.append(new_request)
            self.update_request()
        except Exception as e:
            logger.error(f"[update] Failed to create/update ChargingRequest for {car_id}: {e}", exc_info=True)

    def _find_possible_start_times(self, port_sessions: List[PortSession], arrival_time: float, charging_time: float, max_slots: int = 3, current_time: float = 0.0) -> List[float]:
        
        sorted_sessions = sorted(port_sessions, key=lambda x: x.start_time)
        possible_times = []
        

        if not sorted_sessions or (sorted_sessions[0].start_time >= arrival_time and (sorted_sessions[0].start_time - arrival_time) >= charging_time):
            possible_times.append(arrival_time)
        
        for i in range(len(sorted_sessions)):
            current_session = sorted_sessions[i]
            
            gap_start = max(arrival_time, current_session.stop_time) 
            
            if i == len(sorted_sessions) - 1:
                if gap_start >= current_time or current_session.stop_time >= current_time: 
                    possible_times.append(gap_start)

            else:
                next_session = sorted_sessions[i + 1]
                gap_end = next_session.start_time
                
                if gap_end - gap_start >= charging_time:
                    if gap_start >= current_time:
                        possible_times.append(gap_start)
        
        if not possible_times:

            if not sorted_sessions or sorted_sessions[0].start_time > current_time:
                possible_times.append(current_time)

        possible_times = sorted(list(set(possible_times)))
        
        
        return possible_times[:max_slots]

    def find_nearest_node_idx_euclid(self, route, cs_node):
        cs_x, cs_y = self.G.nodes[cs_node]['x'], self.G.nodes[cs_node]['y']
        lon1, lat1 = transformer_to_latlon.transform(cs_x, cs_y)
        min_dist = float('inf')
        nearest_node = None

        for node in route:
            x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
            lon2, lat2 = transformer_to_latlon.transform(x, y)
                
            dist= haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        idx = len(route) - 1 - route[::-1].index(nearest_node)
        return idx

    def _get_minimum_amount_charge(self, current_cs_node, cs_rest_route, min_energy):

        if not cs_rest_route:
            logger.warning("[Agent] No CS found in rest of route.")
            return 0  

        for cs_node in cs_rest_route:
            try:
                distance_cs_to_cs = nx.shortest_path_length(
                    self.G, current_cs_node, cs_node, weight='length'
                )
                energy_needed = self._calculate_energy_consumption(distance_cs_to_cs)

                if energy_needed < min_energy:
                    min_energy = energy_needed

            except nx.NetworkXNoPath:
                # logger.warning(f"[Agent] No path between {current_cs_node} and {cs_node}. Skipping.")
                pass
            except Exception as e:
                logger.error(f"[Agent] Error calculating energy to {cs_node}: {e}")

        return min_energy

    def get_car_update(self, car_id):
        data = self.redisClient.hgetall(f"car:status:{car_id}")
        decoded_data = {}
        for k, v in data.items():
            key = k.decode()
            value = v.decode()
            
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                if value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
            decoded_data[key] = value

        return decoded_data
    
    def _calculate_energy_consumption(self, distance: float) -> float:

        return (0.081 * 20 ** 2 + 2.61 * 20 + 50) * (distance / 1000000)
    
    def _calculate_travel_time(self, distance: float) -> float:

        return distance / ( 20 )

    def get_optimization_stats(self) -> Dict:

        if not self.schedules:
            return {}
        
        total_cost = sum(
            session.cost
            for schedule in self.schedules 
            for session in schedule.chargingSessions
        )
        
        return {
            'total_cars_scheduled': len(self.schedules),
            'total_sessions': sum(len(s.chargingSessions) for s in self.schedules),
            'total_cost': total_cost,
            'average_cost_per_car': total_cost / len(self.schedules) if self.schedules else 0,
            'optimizer_type': type(self.optimizer).__name__
        }

    def stop(self):

        self.running = False
        logger.info("Enhanced Agent stopped")

print(" Step 2: Enhanced Agent class ready for integration")
