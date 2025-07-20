import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import json
import threading
import time
import datetime
import os
import paho.mqtt.client as mqtt

@dataclass
class CarEvaluationMetrics:
    car_id: str
    total_of_charge_energy: List[float] = field(default_factory=list)
    total_of_charge_cost: List[float] = field(default_factory=list)
    total_of_charge_waiting_time: List[float] = field(default_factory=list)
    total_of_charge_time: List[float] = field(default_factory=list)
    total_travel_time: float = 0
    jumlah_node: int = 0
    total_distance: float = 0
    isFinish: bool = False
    number_of_charge: int = 0
    initial_soc: float = 0.0 
    final_soc: float = 0.0 
    energy_consumed_driving: float = 0.0
    battery_depleted: bool = False
    satisfied_facilities: float = 0.0 
    assigned_stations: List[str] = field(default_factory=list)
    estimated_energy: float = 0.0
    estimated_distance : float = 0.0

@dataclass
class StationEvaluationMetrics:
    station_id: str
    total_ports: int
    port_utilization_history: Dict[int, List[Tuple[float, bool]]] = field(default_factory=dict)  # port_id -> [(time, in_use)]
    cars_served: int = 0
    total_energy_delivered: float = 0.0
    total_revenue: float = 0.0
    average_session_duration: float = 0.0

@dataclass
class SearchingIteration:
    timestamp: float
    duration: float
    cars_involved: int
    success_rate: float

class SearchingEvaluator:
    def __init__(self):
        self.search_log: List[SearchingIteration] = []

    def record_search(self, duration: float, cars: int, success: float):
        self.search_log.append(
            SearchingIteration(time.time(), duration, cars, success)
        )

    def get_summary(self):
        durations = [s.duration for s in self.search_log]
        return {
            "avg_search_time": np.mean(durations) if durations else 0.0,
            "max_search_time": max(durations) if durations else 0.0,
            "count": len(durations)
        }

class EnhancedChargingEvaluator:
    
    def __init__(self):
        self.car_metrics: Dict[str, CarEvaluationMetrics] = {}
        self.station_metrics: Dict[str, StationEvaluationMetrics] = {}
        self.simulation_start_time: float = 0.0
        self.simulation_end_time: float = 0.0
        self.total_cars: int = 0 
        self.running = False
        self.monitor_thread = None
        self.searching_iteration : List[SearchingIteration] = [] 
        
        self.mqtt_client = mqtt.Client(client_id="01-id-evaluator")
        
        self.mqtt_client.connect('localhost', 1883)
        self.mqtt_client.loop_start()
        
        self._setup_network_monitoring()
        
    def _setup_network_monitoring(self):

        self.mqtt_client.subscribe(topic="car/status/#")
        self.mqtt_client.subscribe(topic="car/finish/#")
        self.mqtt_client.subscribe(topic="sim/create_vehicle")
        self.mqtt_client.subscribe(topic="cs/status/#")
        self.mqtt_client.subscribe(topic="sim/create_station")
        self.mqtt_client.subscribe(topic="agent/searching")

        self.mqtt_client.message_callback_add(sub="car/status/#", callback=self._on_car_status)
        self.mqtt_client.message_callback_add(sub="car/finish/#", callback=self._on_car_finish)
        self.mqtt_client.message_callback_add(sub="sim/create_vehicle", callback=self._on_car_created)
        self.mqtt_client.message_callback_add(sub="cs/status/#", callback=self._on_station_status)
        self.mqtt_client.message_callback_add(sub="sim/create_station", callback=self._on_station_created)
        self.mqtt_client.message_callback_add(sub="agent/searching", callback=self._on_searching_result)

    def _on_searching_result(self, client, user_data, message):
        try:
            payload = json.loads(message.payload.decode())
            timestamp = payload.get("timestamp")
            duration = payload.get("duration")
            cars_involved = payload.get("cars_involved")
            success_rate = payload.get("success_rate")

            self.searching_iteration.append(SearchingIteration(timestamp, duration, cars_involved, success_rate))
        except Exception as e:
            print(f"Error in searching result: {e}")

    def _on_car_created(self, client, user_data, message):
        try:
            payload = json.loads(message.payload.decode())
            car_id = payload.get('user_id')
            if car_id not in self.car_metrics: # Prevent re-creation if message is sent multiple times
                self.car_metrics[car_id] = CarEvaluationMetrics(
                    car_id=car_id,
                    initial_soc=payload.get('initial_soc', 0.0) # Assume initial_soc is part of creation
                )
                self.total_cars += 1
        except Exception as e:
            print(f"Error in _on_car_created: {e}")

    def _on_station_created(self, client, user_data, message):
        try:
            payload = json.loads(message.payload.decode())
            station_id = payload.get('stationId')
            ports = payload.get("ports")
            if station_id not in self.station_metrics: # Prevent re-creation if message is sent multiple times
                self.station_metrics[station_id] = StationEvaluationMetrics(
                    station_id=station_id,
                    total_ports=len(ports)
                )
        except Exception as e:
            print(f"Error in _on_car_created: {e}")
            
    def _on_car_status(self, client, user_data,  message):
       
        try:
            payload = json.loads(message.payload.decode())
            topic = message.topic
            car_id = payload.get("car_id", None)
            if not car_id:
                print(f"Warning: Car status message missing 'car_id': {message.payload}")
                return

            if car_id not in self.car_metrics:
                print(f"Warning: Status for car {car_id} received before creation message. Initializing metrics.")
                self.car_metrics[car_id] = CarEvaluationMetrics(car_id=car_id)

            car = self.car_metrics[car_id]
            
            if "soc" in payload and payload["soc"] is not None:
                car.final_soc = payload["soc"]

            car.total_travel_time = payload.get("total_travel_time", car.total_travel_time)
            car.jumlah_node = payload.get("jumlah_node", car.jumlah_node)
            car.total_distance = payload.get("total_distance", car.total_distance)

        except Exception as e:
            print(f"Error in _on_car_status for car {car_id}: {e}")

    def _on_station_status(self, client, user_data, message):
        """
        Updates station metrics based on ongoing status messages.
        """
        try:
            payload = json.loads(message.payload.decode())
            topic = message.topic
            split_topic = topic.split('/')
            station_id = split_topic[-2]
            port_id = split_topic[-1]
            if not station_id:
                print(f"Warning: Station status message missing 'station_id': {message.payload}")
                return

            if station_id not in self.station_metrics:
                self.station_metrics[station_id] = StationEvaluationMetrics(
                    station_id=station_id,
                    total_ports=0
                )
            
            station = self.station_metrics[station_id]
            station.total_ports = len(station.port_utilization_history)
            current_time = payload.get("time")
            isCharging = payload.get("isCharging")
            
            # Update port utilization history
            if port_id not in station.port_utilization_history:
                station.port_utilization_history[port_id] = []
            
            station.port_utilization_history[port_id].append((current_time, isCharging))

            # Update other station metrics if available in the payload
            station.cars_served = payload.get("cars_served", station.cars_served)
            station.total_energy_delivered += payload.get("total_energy_delivered", 0)
            station.total_revenue += payload.get("total_revenue", 0)

        except Exception as e:
            print(f"Error in _on_station_status for station {station_id}: {e}")

    def _on_car_finish(self, client, user_data,  message):

        try:
            topic = message.topic
            car_id = topic.split('/')[-1]
            payload = json.loads(message.payload.decode())

            if car_id not in self.car_metrics:
                print(f"Warning: Car {car_id} finished but was not created in metrics. Initializing metrics.")
                self.car_metrics[car_id] = CarEvaluationMetrics(car_id=car_id)

            car = self.car_metrics[car_id]
            
            car.total_travel_time = payload.get("total_travel_time", car.total_travel_time)
            car.jumlah_node = payload.get("jumlah_node", car.jumlah_node)
            car.total_distance = payload.get("total_distance", car.total_distance)
            car.isFinish = payload.get("isFinish", True) 
            car.number_of_charge = payload.get("number_of_charge", car.number_of_charge)
            
            car.total_of_charge_energy = payload.get("total_of_charge_energy", car.total_of_charge_energy)
            car.total_of_charge_cost = payload.get("total_of_charge_cost", car.total_of_charge_cost)
            car.total_of_charge_time = payload.get("total_of_charge_time", car.total_of_charge_time)
            car.total_of_charge_waiting_time = payload.get("total_of_waiting_time", car.total_of_charge_waiting_time)
            car.estimated_distance = payload.get('estimated_distance', 0)
            
            car.initial_soc = payload.get("initial_soc", car.initial_soc) 
            car.final_soc = payload.get("final_soc", car.final_soc)
            car.energy_consumed_driving = car.initial_soc + sum(car.total_of_charge_energy) - car.final_soc
            car.battery_depleted = payload.get("battery_depleted", car.battery_depleted)
            car.satisfied_facilities = np.mean(payload.get("satisfied_facilities", car.satisfied_facilities))
            car.assigned_stations = payload.get("assigned_stations", car.assigned_stations)
            if car.isFinish:
                car.estimated_energy = payload.get("estimated_energy", 0)
            print(car.car_id, car.initial_soc, car.final_soc, car.final_soc)

        except Exception as e:
            print(f"Error in _on_car_finish for car {car_id}: {e}")

    def start_monitoring(self):
        """Starts the simulation monitoring by recording the start time."""
        self.simulation_start_time = time.time()
        self.running = True
        print(f"Monitoring started at {datetime.datetime.fromtimestamp(self.simulation_start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_monitoring(self):
        """Stops the simulation monitoring by recording the end time."""
        self.simulation_end_time = time.time()
        self.running = False
        print(f"Monitoring stopped at {datetime.datetime.fromtimestamp(self.simulation_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate all evaluation metrics adapted to your system"""
        results = {}
        
        # 1. Car Scheduling Metrics
        print(f"Calculating metrics for {len(self.car_metrics)} cars...")

        # A car is "scheduled" if it has finished its journey (isFinish=True)
        # or if it has received any travel/charge related updates.
        # For simplicity, we'll assume `isFinish=True` implies a successfully scheduled journey.
        # Adjust this logic if 'scheduled' means something else in your simulation.
        scheduled_cars = sum(1 for car in self.car_metrics.values() if car.number_of_charge > 0)
        
        # 'average_charging_stops_per_car' directly uses 'number_of_charge'
        avg_charging_stops = np.mean([car.number_of_charge for car in self.car_metrics.values()]) if self.car_metrics else 0

        results['scheduling'] = {
            'total_cars': self.total_cars,
            'cars_scheduled': scheduled_cars,
            'scheduling_success_rate': scheduled_cars / self.total_cars if self.total_cars > 0 else 0,
            'average_charging_stops_per_car': avg_charging_stops,
            'searching_time_average' : np.mean([rate.duration for rate in self.searching_iteration]) if self.searching_iteration else 0,
            'searching_cars_involved_avg' : np.mean([rate.cars_involved for rate in self.searching_iteration]) if self.searching_iteration else 0,
            'battery_depleted_car': sum(1 for car in self.car_metrics.values() if car.battery_depleted)
        }
        
        # 2. Waiting Time Analysis
        # Derive total waiting time by summing the list `total_of_charge_waiting_time`
        waiting_times_summed = [sum(car.total_of_charge_waiting_time)/60 for car in self.car_metrics.values() if car.total_of_charge_waiting_time]
        if waiting_times_summed:
            results['waiting_time'] = {
                'average_waiting_time': np.mean(waiting_times_summed),
                'min_waiting_time': np.min(waiting_times_summed),
                'max_waiting_time': np.max(waiting_times_summed),
                'std_waiting_time': np.std(waiting_times_summed),
                'median_waiting_time': np.median(waiting_times_summed),
                'percentile_95': np.percentile(waiting_times_summed, 95)
            }
        else:
            results['waiting_time'] = {k: 0.0 for k in ['average_waiting_time', 'min_waiting_time', 'max_waiting_time', 'std_waiting_time', 'median_waiting_time', 'percentile_95']}
        
        # 3. Resource Utilization (adapted to your Port system)
        station_utilizations = []
        station_utilizations_dict = {}

        # Ensure simulation_end_time is set for a meaningful overall utilization calculation
        sim_duration_for_utilization = self.simulation_end_time - self.simulation_start_time
        if sim_duration_for_utilization <= 0: # Handle cases where simulation didn't run or was too short
            sim_duration_for_utilization = 1 # Avoid division by zero, but note this would mean 0 utilization
        port_visited = 0
        cars_served = []
        revenue = []
        energy_delivered = []
        for station in self.station_metrics.values():
            port_utilizations = []
            for port_id, history in station.port_utilization_history.items():
                if len(history) > 1:
                    occupied_time = 0
                    # Iterate through the history to calculate occupied time
                    for i in range(len(history) - 1):
                        start_time_segment, is_occupied = history[i]
                        end_time_segment = history[i+1][0]
                        print(is_occupied)
                        print(start_time_segment, end_time_segment)
                        # Clamp times to simulation start/end for accurate overall utilization
                        segment_start = max(start_time_segment, self.simulation_start_time)
                        segment_end = min(end_time_segment, self.simulation_end_time)
                        if is_occupied and segment_end > segment_start:
                            occupied_time += (segment_end - segment_start)
                    if occupied_time > 0:
                        port_visited+=1
                    if sim_duration_for_utilization > 0:
                        utilization = occupied_time / sim_duration_for_utilization
                        port_utilizations.append(utilization)
                    else:
                        port_utilizations.append(0.0)
            if port_utilizations:
                station_avg_utilization = np.mean(port_utilizations)
            else:
                station_avg_utilization = 0.0
            if station.cars_served > 0:
                cars_served.append(station.cars_served)
            if station.total_revenue > 0:
                revenue.append(station.total_revenue)
            if station.total_energy_delivered > 0:
                energy_delivered.append(station.total_energy_delivered)
            station_utilizations.append(station_avg_utilization)
            station_utilizations_dict[station.station_id] = station_avg_utilization

        results['resource_utilization'] = {
            'average_port_utilization': float(np.mean(station_utilizations)) if station_utilizations else 0.0,
            'station_visited_percentage': (sum(1 for i in station_utilizations if i > 0) / len(station_utilizations) if station_utilizations else 0.0),
            'port_visited_number' : port_visited,
            'min_port_utilization': float(np.min(station_utilizations)) if station_utilizations else 0.0,
            'max_port_utilization': float(np.max(station_utilizations)) if station_utilizations else 0.0,
            'average_satisfied_facilities': float(np.mean([car.satisfied_facilities for car in self.car_metrics.values()])) if self.car_metrics else 0.0
        }
        print(f" energy delivered is {energy_delivered}")

        results['station_analysis'] = {
            'cars_served_average' : np.mean(cars_served) if len(cars_served)>0 else 0.0,
            'total_session_served' : sum(cars_served),
            'total_revenue' : sum(revenue),
            'average_energy_delivered' : np.mean(energy_delivered) if len(energy_delivered)>0 else 0.0,
            'total_energy_delivered' : sum(energy_delivered),
        }
        
        # 4. Energy Efficiency Analysis
        # 'total_of_charge_energy' is a list of charge energies per session. Sum them up for total.
        total_energy_charged = sum(sum(car.total_of_charge_energy) for car in self.car_metrics.values())
        total_energy_consumed = sum(car.energy_consumed_driving for car in self.car_metrics.values())
        total_energy_required = sum(car.estimated_energy for car in self.car_metrics.values())
        # SOC improvement analysis
        soc_improvements = []
        for car in self.car_metrics.values():
            # A common way to calculate 'SOC improvement' is total energy charged divided by
            # the total energy consumed for driving. This shows how well charging covered consumption.
            # If initial/final SOC are direct measures, a simple `final_soc - initial_soc` is also a 'net change'.
            
            # Using the ratio of charged energy to consumed driving energy as 'SOC improvement' proxy
            if car.energy_consumed_driving > 0:
                # If total_of_charge_energy is empty, sum() returns 0
                soc_improvement_ratio = sum(car.total_of_charge_energy) / car.energy_consumed_driving
                soc_improvements.append(soc_improvement_ratio)
            elif sum(car.total_of_charge_energy) > 0: # Car charged but didn't drive (e.g., just moved to station)
                soc_improvements.append(1.0) # Represents significant "improvement" if no consumption
            else:
                soc_improvements.append(0.0) # No charging, no driving consumption, no improvement


        results['energy_efficiency'] = {
            'total_energy_consumed_driving': total_energy_consumed,
            'total_energy_received_charging': total_energy_charged,
            'net_energy_balance': total_energy_charged - total_energy_consumed,
            'average_soc_improvement': np.mean(soc_improvements) if soc_improvements else 0.0, # This is a ratio, not a percentage
            'energy_efficiency_ratio': total_energy_charged / max(total_energy_consumed, 1) if total_energy_consumed > 0 else (total_energy_charged if total_energy_charged > 0 else 0.0),
            'total_estimated_energy_required' : total_energy_required,
            'total_estimated_detour_energy' : total_energy_consumed - total_energy_required,
            'average_estimated_detour_energy' : (total_energy_consumed - total_energy_required) / len(self.car_metrics)
        }
        
        # 5. Route and Journey Analysis
        # 'jumlah_node' is used as route length in your CarEvaluationMetrics
        route_lengths = [car.jumlah_node for car in self.car_metrics.values() if car.jumlah_node > 0] 
        route_distance = [car.total_distance for car in self.car_metrics.values() if car.total_distance > 0]
        estimated_distance = [car.estimated_distance for car in self.car_metrics.values() if car.estimated_distance > 0]
        # 'number_of_charge' is the number of charging stops
        charging_stops = [car.number_of_charge for car in self.car_metrics.values()]
    
        # 'total_travel_time' from CarEvaluationMetrics
        travel_times = [(car.total_travel_time/60) for car in self.car_metrics.values() if car.total_travel_time > 0]
        
        results['journey_analysis'] = {
            'average_route_length': np.mean(route_lengths) if route_lengths else 0.0,
            'average_distance' : np.mean(route_distance) if route_distance else 0.0,
            'average_estimated_distance' : np.mean(estimated_distance) if estimated_distance else 0.0,
            'average_charging_stops': np.mean(charging_stops) if charging_stops else 0.0,
            'max_charging_stops': max(charging_stops) if charging_stops else 0,
            'cars_needing_charging': sum(1 for stops in charging_stops if stops > 0),
            'charging_dependency_rate': sum(1 for stops in charging_stops if stops > 0) / len(charging_stops) if charging_stops else 0.0,
            'average_travel_time' : np.mean(travel_times) if travel_times else 0.0,
            'min_travel_time' : min(travel_times) if travel_times else 0.0,
            'max_travel_time' : max(travel_times) if travel_times else 0.0
        }

        
        
        return results
    
    def generate_detailed_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        metrics = self.calculate_comprehensive_metrics()
        
        duration = self.simulation_end_time - self.simulation_start_time
        minutes = duration / 60
        
        report = f"""
=== ENHANCED EV CHARGING SIMULATION EVALUATION REPORT ===
Simulation Duration: {minutes:.1f} minutes

1. SCHEDULING PERFORMANCE:
    - Total Cars: {metrics['scheduling']['total_cars']}
    - Avg Charging Stops per Car: {metrics['scheduling']['average_charging_stops_per_car']:.1f}
    - Searching Time Average: {metrics['scheduling']['searching_time_average']:.1f} seconds
    - Battery Depleted Car: {metrics['scheduling']['battery_depleted_car']}

2. WAITING TIME ANALYSIS:
    - Average Waiting Time: {metrics['waiting_time']['average_waiting_time']:.2f} minutes
    - Min/Max Waiting Time: {metrics['waiting_time']['min_waiting_time']:.2f} / {metrics['waiting_time']['max_waiting_time']:.2f} minutes
    - Standard Deviation: {metrics['waiting_time']['std_waiting_time']:.2f} minutes
    - 95th Percentile: {metrics['waiting_time']['percentile_95']:.2f} minutes

3. RESOURCE UTILIZATION:
    - Average Port Utilization: {metrics['resource_utilization']['average_port_utilization']:.2%}
    - Station Visited Percentage: {metrics['resource_utilization']['station_visited_percentage']:.2%}
    - Port Visited Number: {metrics['resource_utilization']['port_visited_number']}
    - Station Utilization Range: {metrics['resource_utilization']['min_port_utilization']:.2%} - {metrics['resource_utilization']['max_port_utilization']:.2%}
    - Average Satisfied Facilities: {metrics['resource_utilization']['average_satisfied_facilities']:.2%}

4. ENERGY EFFICIENCY:
    - Total Energy Consumed (Driving): {metrics['energy_efficiency']['total_energy_consumed_driving']:.2f} kWh
    - Total Energy Received (Charging): {metrics['energy_efficiency']['total_energy_received_charging']:.2f} kWh
    - Net Energy Balance: {metrics['energy_efficiency']['net_energy_balance']:.2f} kWh
    - Energy Efficiency Ratio: {metrics['energy_efficiency']['energy_efficiency_ratio']:.2f}
    - Average SOC Improvement (Charged/Consumed Ratio): {metrics['energy_efficiency']['average_soc_improvement']:.2f}
    - Total Estimated Energy Required: {metrics['energy_efficiency']['total_estimated_energy_required']}
    - Total Estimated Detour Energy: {metrics['energy_efficiency']['total_estimated_detour_energy']}
    - Average Estimated Detour Energy: {metrics['energy_efficiency']['average_estimated_detour_energy']}

5. JOURNEY ANALYSIS:
    - Average Route Length: {metrics['journey_analysis']['average_route_length']:.1f} nodes
    - Average Charging Stops: {metrics['journey_analysis']['average_charging_stops']:.1f}
    - Max Charging Stops: {metrics['journey_analysis']['max_charging_stops']}
    - Cars Needing Charging: {metrics['journey_analysis']['cars_needing_charging']}
    - Charging Dependency Rate: {metrics['journey_analysis']['charging_dependency_rate']:.2%}
    - Average Travel Time: {metrics['journey_analysis']['average_travel_time']:.2f} minutes
    
6. STATION ANALYSIS:
    - Average Car Served: {metrics['station_analysis']['cars_served_average']}  
    - Total Session Served: {metrics['station_analysis']['total_session_served']}
    - Total Revenue: {metrics['station_analysis']['total_revenue']}
    - Average Energy Delivered: {metrics['station_analysis']['average_energy_delivered']}
    - Total Energy Delivered: {metrics['station_analysis']['total_energy_delivered']}
    
# 7. INDIVIDUAL CAR PERFORMANCE:
# """
#         # Add individual car details
#         for car_id, car_metric in self.car_metrics.items():
#             report += f"""
#     Car {car_id}:
#       - Is Finished: {'✓' if car_metric.isFinish else '✗'}
#       - Initial SOC: {car_metric.initial_soc:.1f}%
#       - Final SOC: {car_metric.final_soc:.1f}%
#       - Charging Stops: {car_metric.number_of_charge}
#       - Total Waiting Time: {sum(car_metric.total_of_charge_waiting_time)/60:.1f} min
#       - Total Charge Energy: {sum(car_metric.total_of_charge_energy):.2f} kWh
#       - Total Charge Cost: {sum(car_metric.total_of_charge_cost):.2f}
#       - Total Charge Time: {sum([tup[1]-tup[0] for tup in car_metric.total_of_charge_time])/60:.1f} min
#       - Total Travel Time: {car_metric.total_travel_time/60:.1f} min
#       - Total Distance: {car_metric.total_distance:.1f} units
#       - Number of Nodes (Route Length): {car_metric.jumlah_node}
#       - Energy Consumed Driving: {car_metric.energy_consumed_driving:.2f} kWh
#       - Battery Depleted: {'✓' if car_metric.battery_depleted else '✗'}
#       - Satisfied Facilities: {car_metric.satisfied_facilities:.2f}
#       - Assigned Stations: {', '.join(car_metric.assigned_stations) if car_metric.assigned_stations else 'None'}
# """
        
        return report
    
    def export_metrics_to_csv(self, filename: str = "ev_simulation_metrics.csv"):
        """Export detailed metrics to CSV for further analysis"""
        car_data = []
        for car_id, car_metric in self.car_metrics.items():
            car_data.append({
                'car_id': car_id,
                'is_finished': car_metric.isFinish,
                'total_waiting_time_minutes': sum(car_metric.total_of_charge_waiting_time)/60,
                'initial_soc': car_metric.initial_soc,
                'final_soc': car_metric.final_soc,
                'charging_stops': car_metric.number_of_charge,
                'route_length_nodes': car_metric.jumlah_node, # Using jumlah_node as route length
                'total_energy_received_charging_kwh': sum(car_metric.total_of_charge_energy),
                'total_energy_consumed_driving_kwh': car_metric.energy_consumed_driving,
                'total_charging_cost': sum(car_metric.total_of_charge_cost),
                'total_charging_time_minutes': sum([tup[1]-tup[0] for tup in car_metric.total_of_charge_time])/60,
                'total_travel_time_minutes': car_metric.total_travel_time/60,
                'total_distance_units': car_metric.total_distance,
                'battery_depleted': car_metric.battery_depleted,
                'satisfied_facilities': car_metric.satisfied_facilities,
                'assigned_stations': ';'.join(car_metric.assigned_stations)
            })
        
        df = pd.DataFrame(car_data)
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Metrics exported to {filename}")
        else:
            print("No car data to export to CSV.")
        
    def plot_enhanced_metrics(self, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        
        car_ids = list(map(int, self.car_metrics.keys()))
        if not car_ids:
            print("No car data to plot.")
            return

        # Plot 1: Total Waiting Time per Car
        fig, ax = plt.subplots(figsize=(10, 5))
        wait_times = [sum(m.total_of_charge_waiting_time)/60 for m in self.car_metrics.values()]
        ax.bar(car_ids, wait_times)
        ax.set_title("Total Waiting Time per Car")
        ax.set_xlabel("Car ID")
        ax.set_ylabel("Minutes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"waiting_time_per_car_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()

        # Plot 2: Final SoC per Car
        fig, ax = plt.subplots(figsize=(10, 5))
        final_socs = [m.final_soc for m in self.car_metrics.values()]
        ax.bar(car_ids, final_socs)
        ax.set_title("Final SoC per Car")
        ax.set_xlabel("Car ID")
        ax.set_ylabel("SoC (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Final_SoC_per_car_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()

        # # Plot 3: Average Port Utilization per Station
        # if self.station_metrics:
        #     station_ids = list(self.station_metrics.keys())
        #     # Ensure calculate_comprehensive_metrics has been called to populate station_utilizations_dict
        #     station_avg_utilizations = [
        #         station.average_session_duration for station in self.station_metrics.values()
        #     ]
        #     fig, ax = plt.subplots(figsize=(10, 5))
        #     ax.bar(station_ids, station_avg_utilizations, color='orange')
        #     ax.set_title("Average Port Utilization per Station")
        #     ax.set_xlabel("Station ID")
        #     ax.set_ylabel("Average Utilization (%)")
        #     plt.xticks(rotation=45)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(save_dir, f"avg_port_utilization_per_station_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        #     plt.close()
        # else:
        #     print("No station data to plot average port utilization.")

        # Plot 4: Battery Depleted Cars
        depleted_cars = [m.car_id for m in self.car_metrics.values() if m.battery_depleted]
        if depleted_cars:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(depleted_cars, [1]*len(depleted_cars), color='red')
            ax.set_title("Battery Depleted Cars")
            ax.set_xlabel("Car ID")
            ax.set_ylabel("Depleted (1=Yes)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"battery_depleted_cars_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            plt.close()
        else:
            print("No cars reported battery depletion.")

        # Plot 5: Number of Charging Sessions per Car (using number_of_charge)
        fig, ax = plt.subplots(figsize=(10, 5))
        sessions = [m.number_of_charge for m in self.car_metrics.values()]
        ax.bar(car_ids, sessions, color='green')
        ax.set_title("Charging Sessions per Car")
        ax.set_xlabel("Car ID")
        ax.set_ylabel("Sessions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"charging_sessions_per_car_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()

        # Plot 6: Total Energy Received from Charging per Car
        fig, ax = plt.subplots(figsize=(10, 5))
        energy_received = [sum(m.total_of_charge_energy) for m in self.car_metrics.values()]
        ax.bar(car_ids, energy_received, color='purple')
        ax.set_title("Total Energy Received per Car")
        ax.set_xlabel("Car ID")
        ax.set_ylabel("Energy (kWh)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"energy_received_per_car_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()

        print(f"All evaluator plots saved to '{os.path.abspath(save_dir)}'")


# Integration helper functions
def create_evaluator_for_simulation() -> EnhancedChargingEvaluator:
    """Create and configure an evaluator for your simulation"""
    evaluator = EnhancedChargingEvaluator()
    return evaluator

def run_evaluation_with_simulation(simulation_duration: int = 300):
    """Run a complete evaluation session"""
    print("🚀 Starting Enhanced EV Charging Evaluation")
    
    # Create evaluator
    evaluator = create_evaluator_for_simulation()
    
    # Start monitoring
    evaluator.start_monitoring()
    
    try:
        # Let the simulation run.
        # In a real scenario, your simulation engine would be running here
        # and publishing messages to the 'network' module.
        print(f"⏱️  Running simulation for {simulation_duration} seconds (simulated time)...")
        time.sleep(simulation_duration) # This simulates the passage of real time
        
    finally:
        # Stop monitoring
        evaluator.stop_monitoring()
        
        # Generate report
        print("\n" + "="*60)
        print("FINAL EVALUATION REPORT")
        print("="*60)
        report = evaluator.generate_detailed_report()
        print(report)
        
        # Export data
        evaluator.export_metrics_to_csv()
        
        # Generate plots
        evaluator.plot_enhanced_metrics()
        
    return evaluator
