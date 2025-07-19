from typing import List, Optional, Dict, Any
import json
import logging
import time
import threading
import simpy
from shapely.geometry import LineString, Point
from shapely.wkt import loads
from pyproj import Geod, Transformer
import networkx as nx
import random


import paho.mqtt.client as mqtt # MQTT import restored

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


transformer_to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

geod = Geod(ellps="WGS84")

class Car:

    def __init__(self, id: str, env=None, G=None, soc=50.0, current_lat=None, current_lon=None, 
                 origin=None, dest=None, vel=50, capacity=100.0, time_factor=1.0, facilities_preference=[]):
        self.id = id
        self.env = env
        self.G = G  
        self.soc = soc
        self.capacity = capacity
        self.current_lat = current_lat
        self.current_lon = current_lon
        self.current_node = origin
        self.origin = origin
        self.dest = dest
        self.vel = vel
        self.route = []
        self.isCharging = False
        self.chargeSession = [] 
        self.time_factor = time_factor
        self.position = None
        self.elapsed_time = 0
        self.action = None 
        self.running = True
        self.driving = False
        self.facilities_preference = facilities_preference
        self.last_charging_node = None
        self.arrival_time = 0
        self.total_travel_time = 0
        self.jumlah_node = 0
        self.total_distance = 0
        self.is_finish = False
        self.number_of_charge = 0
        self.total_of_charge_energy = []
        self.total_of_charge_cost = []
        self.total_of_charge_time = []
        self.total_of_waiting_time = []
        
        # MQTT client initialization and setup restored
        self.mqtt_client = mqtt.Client(client_id=id)
        self.mqtt_client.on_connect = self._on_mqtt_connect # Only connect handler needed if not subscribing
        self.mqtt_client.connect('localhost', 1883)
        self.mqtt_client.loop_start() # Start MQTT loop
        
        # Status reporter process restored
        self.env.process(self._status_reporter()) 

        # Start driving process if environment is provided
        if self.env and self.origin and self.dest and self.G:
            delay = 300 * int(self.id) / self.time_factor          
            self.env.process(self._delayed_start(delay))

    def _delayed_start(self, delay: float):
        yield self.env.timeout(delay)
        logger.info(f"[Car {self.id}] Starting drive after {delay:.2f} seconds delay.")
        
        # Directly calculate route at start
        if self.origin and self.dest and self.G:
            try:
                self.route = nx.shortest_path(self.G, source=self.origin, target=self.dest, weight='length')
                logger.info(f"[Car {self.id}] Initial route calculated with {len(self.route)} nodes.")
            except nx.NetworkXNoPath:
                logger.error(f"[Car {self.id}] No path found from {self.origin} to {self.dest}.")
                self.is_finish = True
                self.driving = False
                self.finish_report() # MQTT call restored
                return
            except Exception as e:
                logger.error(f"[Car {self.id}] Error calculating initial route: {e}")
                self.is_finish = True
                self.driving = False
                self.finish_report() # MQTT call restored
                return
        else:
            logger.warning(f"[Car {self.id}] Origin, Destination or Graph not set. Cannot start driving.")
            self.is_finish = True
            self.driving = False
            self.finish_report() # MQTT call restored
            return

        self.driving = True
        self.arrival_time = self.env.now
        self.action = self.env.process(self.drive())

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"[Car {self.id}] Connected to MQTT broker")
            # No subscriptions needed as car doesn't receive commands via MQTT
        else:
            logger.error(f"[Car {self.id}] MQTT connection failed with code {rc}")

    # Removed _on_mqtt_message, _on_route_update, _on_nearest_node_response, _on_charging_update
    # as car doesn't subscribe to these topics anymore.

    def _status_reporter(self):
        while self.running:
            try:                
                # Publish status to MQTT client (restored)
                self.mqtt_client.publish(
                    topic=f"car/status/{self.id}",
                    payload=json.dumps({
                        "time": self.env.now,
                        "car_id": self.id,
                        "soc": round(self.soc, 2),
                        "location": {
                            "lat": self.current_lat,
                            "lon": self.current_lon
                        },
                        "isCharging": self.isCharging,
                        "charging_session": self.chargeSession, # This will likely be empty now
                        "vel": self.vel,
                        "current_node": self.current_node,
                        "isFinish" : self.is_finish,
                        "route" : self.route,
                        "driving" : self.driving,
                        "dest" : self.dest,
                        "capacity" : self.capacity
                    }),
                )
                yield self.env.timeout(60)
            except Exception as e:
                logger.error(f"[Car {self.id}] Error in status reporter: {e}")
                time.sleep(1) 
    
    def getSegmentVel(self, u, v):
        if not self.G:
            return None, 50 
        try:
            edges = self.G[u][v]
            key = next(iter(edges))
            edge_data = edges[key]
            geom = edge_data.get('geometry', None)
            if geom is None:
                start_x, start_y = self.G.nodes[u].get('x', 0), self.G.nodes[u].get('y', 0)
                end_x, end_y = self.G.nodes[v].get('x', 0), self.G.nodes[v].get('y', 0)
                geom = LineString([(start_x, start_y), (end_x, end_y)])
            vel = edge_data.get('speed', 0)
            if vel == 0:
                vel = max(50, min(70, 50 + 20 * (0.5 - (0.5 - random.random()))))
            return geom, vel/(2.5+random.random())
        except Exception as e:
            logger.error(f"[Car {self.id}] Error getting segment velocity: {e}")
            return None, 50 
    
    def getSegmentTime(self, u, v):
        if not self.G:
            return 30
        try:
            geom, vel = self.getSegmentVel(u, v)
            if geom is None:
                start_x, start_y = self.G.nodes[u].get('x', 0), self.G.nodes[u].get('y', 0)
                end_x, end_y = self.G.nodes[v].get('x', 0), self.G.nodes[v].get('y', 0)
                length = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
            else:
                length = geom.length
            if vel > 30: 
                vel = vel / 3.6  
            if vel == 0:
                return float('inf')  # avoid division by zero
            return length / vel
        except Exception as e:
            logger.error(f"[Car {self.id}] Error calculating segment time: {e}")
            return 60  # Default time on error
    
    def consume_energy(self, velocity, distance):
        try:
            energy_used = (0.081 * velocity ** 2 + 2.61 * velocity + 50) * (distance / 1000000)
            return energy_used
        except Exception as e:
            logger.error(f"[Car {self.id}] Error calculating energy consumption: {e}")
            return max(0.001, 0.001 * distance)  
    
    def drive(self):
        try:
            if not self.route:
                logger.warning(f"[Car {self.id}] No route found. Stopping drive.")
                self.is_finish = True
                self.driving = False
                self.finish_report() # MQTT call restored
                return

            logger.info(f"[Car {self.id}] Starting drive from {self.origin} to {self.dest} with {len(self.route)} nodes and {self.soc:.2f} kWh SOC.")
            
            i = 0
            while i < len(self.route)-1 and self.running and not self.is_finish:
                u = self.route[i]
                v = self.route[i + 1]
                self.current_node = v

                try:
                    edges = self.G[u][v]
                    key = next(iter(edges))  
                    edge_data = edges[key]
                except (KeyError, StopIteration):
                    logger.warning(f"[Car {self.id}] No edge found between {u} and {v}. Skipping segment.")
                    i += 1
                    continue

                geom, vel = self.getSegmentVel(u,v)
                total_length_meters = edge_data.get('length')
                
                if total_length_meters is None:
                    logger.warning(f"[Car {self.id}] Length not found for edge {u}->{v}. Skipping segment.")
                    i += 1
                    continue
                
                energy_used_kwh = self.consume_energy(vel, total_length_meters)
                self.soc -= energy_used_kwh

                if self.soc < 0:
                    self.soc = 0 
                    logger.info(f"[Car {self.id}] Battery depleted! Stopping at node {self.current_node}.")
                    self.is_finish = True 
                    self.driving = False
                    self.finish_report() # MQTT call restored
                    return

                if (vel / 3.6) == 0: 
                    travel_time_sec = float('inf')
                else:
                    travel_time_sec = total_length_meters / (vel)

                try:
                    if geom:
                        line = geom if isinstance(geom, LineString) else loads(geom)
                        point = line.interpolate(0.5, normalized=True) 
                        self.position = point
                        self.current_lon, self.current_lat = transformer_to_latlon.transform(point.x, point.y)
                except Exception as e:
                    logger.debug(f"[Car {self.id}] Position update failed for {u}->{v}: {e}")

                # Simulate travel time
                yield self.env.timeout(travel_time_sec / self.time_factor)
                
                self.total_distance += total_length_meters
                self.total_travel_time += travel_time_sec
                
                # Check if current node has a charging station and SOC is low
                if self.current_node in self.G.nodes and 'charging_station' in self.G.nodes[self.current_node]:
                    cs = self.G.nodes[self.current_node]['charging_station']
                    # Define a threshold for charging, e.g., if SOC is below 30%
                    if self.soc / self.capacity < 0.2 and not self.isCharging: 
                        logger.info(f"[Car {self.id}] Arrived at node {self.current_node} with CS. SOC is {self.soc:.2f} kWh. Attempting to recharge.")
                        self.driving = False # Stop driving to recharge
                        yield from self.recharge(self.current_node)
                        self.driving = True # Resume driving after recharge
                        yield self.env.timeout(60) # Small timeout after charging

                i += 1 

            if not self.is_finish:
                logger.info(f"[Car {self.id}] Reached destination {self.dest} at {self.env.now:.2f}.")
                self.is_finish = True
                self.jumlah_node = len(self.route)
                self.finish_report() # MQTT call restored
            
            self.driving = False
            self.isCharging = False
        except simpy.Interrupt:
            logger.info(f"[Car {self.id}] Drive process interrupted.")
            self.driving = False
        except Exception as e:
            logger.error(f"[Car {self.id}] Error in drive process: {e}", exc_info=True)
            self.driving = False 

    
    def recharge(self, node_id: int):
        """Simulate recharging at a charging station based on finding one."""
        try:
            available_port = None
            arrive_time = self.env.now
            while not available_port:
                cs = self.G.nodes[node_id].get('charging_station')
                if cs is None:
                    logger.error(f"[Car {self.id}] Charging station not found at node {node_id}.")
                    return
                
                # Find an available port
                for port in cs.ports:
                    if port.isAvailable:
                        available_port = port
                        break
                
                if available_port is None:
                    logger.info(f"[Car {self.id}] No available ports at CS {cs.id} at node {node_id}. Cannot charge.")
                    yield self.env.timeout(60)
                    continue 

                port_id = available_port.id
                logger.info(f"[Car {self.id}] Arrived at CS {cs.id}, Port {port_id}. Current time: {self.env.now:.2f}")
            
            wait_seconds_to_start = self.env.now - arrive_time
            
            self.total_of_waiting_time.append(wait_seconds_to_start)             

            time_to_full_charge = (self.capacity - self.soc) * 3600 / available_port.power if available_port.power > 0 else float('inf')
            
            target_soc = self.capacity * 0.8 
            time_to_reach_target_soc = (target_soc - self.soc) * 3600 / available_port.power if available_port.power > 0 else float('inf')
            
            minimum_soc = self.consume_energy(25, nx.shortest_path_length(self.G, self.current_node, self.dest, 'length'))
            percentage = max(0.1,random.random()*0.3)
            max_charging_session_time_sec = (minimum_soc + self.capacity * percentage) * 3600 / available_port.power if available_port.power > 0 else float('inf') 
            actual_charging_duration_sec = min(time_to_full_charge, time_to_reach_target_soc, max_charging_session_time_sec)
            actual_charging_duration_sec = max(0.0, actual_charging_duration_sec) 

            if actual_charging_duration_sec <= 0:
                logger.info(f"[Car {self.id}] Skipping recharge.")
                return

            self.isCharging = True
            self.number_of_charge += 1

            logger.info(f'[Car {self.id}] Actual Charging duration: {actual_charging_duration_sec:.2f} seconds ({actual_charging_duration_sec/60:.2f} minutes). Power: {available_port.power} kW.')
            self.start_charging_report(cs.id, port_id) # MQTT call restored

            charged_energy = 0.0
            charge_start_sim_time = self.env.now 
            
            available_port.isAvailable = False 

            time_step_for_charge = 10 
            while self.env.now < (charge_start_sim_time + actual_charging_duration_sec) and self.soc < self.capacity:
                
                time_remaining_in_step = (charge_start_sim_time + actual_charging_duration_sec) - self.env.now
                if time_remaining_in_step <= 0: break 

                time_to_yield = min(time_step_for_charge, time_remaining_in_step)
                
                energy_gain_this_step = available_port.power * time_to_yield / 3600 

                if self.soc + energy_gain_this_step > self.capacity:
                    energy_gain_this_step = self.capacity - self.soc
                    if available_port.power > 0:
                        time_to_yield = energy_gain_this_step * 3600 / available_port.power
                    else:
                        time_to_yield = 0 
                    if time_to_yield <= 0: 
                        break

                self.soc += energy_gain_this_step
                charged_energy += energy_gain_this_step

                yield self.env.timeout(time_to_yield)
            
            available_port.isAvailable = True 

            self.last_charging_node = node_id
            self.total_of_charge_energy.append(charged_energy)
            self.total_of_charge_cost.append(charged_energy * available_port.price)
            self.total_of_charge_time.append((charge_start_sim_time, self.env.now)) 

            cs.usage_time += (self.env.now - charge_start_sim_time)
            energy_delivered = charged_energy
            revenue = charged_energy * available_port.price
            cs.number_of_charging_session += 1
            available_port.usage_time += (self.env.now - charge_start_sim_time)
            available_port.number_of_charging_session += 1
            self.chargeSession = [] 
            self.stop_charging_report(cs.id, port_id, cs.number_of_charging_session, energy_delivered, revenue)
            
            self.isCharging = False 
            self.driving = True     
            
            logger.info(f"[Car {self.id}] Finished recharge at node {node_id} with SOC: {self.soc:.2f} kWh, Charged: {charged_energy:.2f} kWh.")
        
        except simpy.Interrupt:
            logger.info(f"[Car {self.id}] Recharge process interrupted by SimPy. Car state: isCharging={self.isCharging}")
            self.isCharging = False 
        except Exception as e:
            logger.error(f"[Car {self.id}] Error in recharge process: {e}", exc_info=True)
            self.isCharging = False 
            self.driving = True 
    
    def stop(self):
        logger.info(f"[Car {self.id}] Stopping car simulation")
        logger.info(f"Dest is {self.dest} and current is {self.current_node}")
        if self.soc == 0:
            logger.info(f"route covered for {self.id} is {self.route}")
        self.running = False
        
        if self.action:
            try:
                self.action.interrupt()
            except Exception as e:
                logger.debug(f"[Car {self.id}] Error interrupting action: {e}")
        
        # MQTT client disconnect restored
        if hasattr(self, 'mqtt_client') and self.mqtt_client:
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()
        
        logger.info(f"[Car {self.id}] Car stopped successfully")
    
    # MQTT reporting methods restored
    def start_charging_report(self, cs_id, port_id: int):
        self.mqtt_client.publish(topic=f"car/status/{self.id}", 
                                 payload=json.dumps({
                                    "time": self.env.now,
                                    "car_id": self.id,
                                    "soc": round(self.soc, 2),
                                    "location": {
                                        "lat": self.current_lat,
                                        "lon": self.current_lon
                                    },
                                    "isCharging": True, 
                                    "charging_session": self.chargeSession, 
                                    "vel": self.vel,
                                    "current_node": self.current_node,
                                    "isFinish": self.is_finish
                                }),
                                 )
        
        self.mqtt_client.publish(topic=f"cs/status/{cs_id}/{port_id}",
                                 payload=json.dumps({
                                     "time" : self.env.now,
                                    "isCharging" : True, # Set to True when charging starts
                                 }))
        
    def stop_charging_report(self,cs_id, port_id, cars_served:int, total_energy_delivered:float, total_revenue:int):
        self.mqtt_client.publish(topic=f"car/status/{self.id}", 
                                 payload=json.dumps({
                                    "time": self.env.now,
                                    "car_id": self.id,
                                    "soc": round(self.soc, 2),
                                    "location": {
                                        "lat": self.current_lat,
                                        "lon": self.current_lon
                                    },
                                    "isCharging": False, 
                                    "charging_session": self.chargeSession, 
                                    "vel": self.vel,
                                    "current_node": self.current_node,
                                    "isFinish": self.is_finish
                                                    }),
                                 )
        
        self.mqtt_client.publish(topic=f"cs/status/{cs_id}/{port_id}",
                                 payload=json.dumps({
                                    "time" : self.env.now,
                                    "isCharging" : False, # Set to False when charging stops
                                    "cars_served" : cars_served,
                                    "total_energy_delivered" : total_energy_delivered,
                                    "total_revenue" : total_revenue
                                 })
                                 )
        
    def finish_report(self):
        self.mqtt_client.publish(
                    topic=f"car/status/{self.id}",
                    payload=json.dumps({
                        "time": self.env.now,
                        "car_id": self.id,
                        "soc": round(self.soc, 2),
                        "location": {
                            "lat": self.current_lat,
                            "lon": self.current_lon
                        },
                        "isCharging": self.isCharging,
                        "charging_session": self.chargeSession,
                        "vel": self.vel,
                        "current_node": self.current_node,
                        "isFinish" : True, 
                        "route" : self.route,
                        "battery_depleted" : self.current_node != self.dest
                    }),
                )
        
        self.mqtt_client.publish(topic=f"car/finish/{self.id}", 
                                 payload=json.dumps({
                                       "total_travel_time" : self.total_travel_time,
                                        "jumlah_node" : self.jumlah_node,
                                        "total_distance" : self.total_distance,
                                        "estimated_distance" : nx.shortest_path_length(self.G, self.origin, self.dest, 'length'),
                                        "isFinish" : True,
                                        "number_of_charge" : self.number_of_charge,
                                        "total_of_charge_energy" : self.total_of_charge_energy,
                                        "total_of_charge_cost" : self.total_of_charge_cost,
                                        "total_of_charge_time" : self.total_of_charge_time,
                                        "total_of_waiting_time" : self.total_of_waiting_time,
                                        "estimated_energy" : self.consume_energy(20, nx.shortest_path_length(self.G, self.origin, self.dest, 'length'))
                                 })
                                 )
        
    def extract_linestring_from_route(self, route):
        coords = []
        for u, v in zip(route[:-1], route[1:]):
            edge_data = self.G.get_edge_data(u, v)
            if not edge_data:
                continue  

            edge = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]

            if "geometry" in edge:
                coords.extend(list(edge["geometry"].coords))
            else:
                coords.append((self.G.nodes[u]["x"], self.G.nodes[u]["y"]))
                coords.append((self.G.nodes[v]["x"], self.G.nodes[v]["y"]))

        return LineString(coords)