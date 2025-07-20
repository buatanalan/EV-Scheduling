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
import datetime
import sys

import paho.mqtt.client as mqtt

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
                 origin=None, dest=None, vel=50, capacity=100.0, time_factor=0.1, facilities_preference=[]):
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
        self.satisfied_facilities = []
        self.mqtt_client = mqtt.Client(client_id=id)
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_connect = self._on_mqtt_connect
        
        self.mqtt_client.connect('localhost', 1883)
        self.mqtt_client.loop_start()
        self.env.process(self._status_reporter())
        # Start driving process if environment is provided
        if self.env and self.origin and self.dest and self.G:
            delay = 600 * int(self.id) / self.time_factor          
            self.env.process(self._delayed_start(delay))

    def _delayed_start(self, delay: float):

        yield self.env.timeout(delay)
        logger.info(f"[Car {self.id}] Starting drive after {delay:.2f} seconds delay.")
        self.driving = True
        self.arrival_time = self.env.now
        self.action = self.env.process(self.drive())

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"[Car {self.id}] Connected to MQTT broker")
            # Subscribe topics
            self.mqtt_client.subscribe(topic=f"car/route/{self.id}")
            self.mqtt_client.subscribe(topic=f"car/charging/{self.id}")
            self.mqtt_client.subscribe(topic=f"car/nearest_node/{self.id}")
        else:
            logger.error(f"[Car {self.id}] MQTT connection failed with code {rc}")

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            # logger.info(f"[Car {self.id}] Received MQTT message on {topic}:")

            if topic.startswith(f"car/route/{self.id}"):
                self._on_route_update(topic, payload)
            elif topic.startswith(f"car/charging/{self.id}"):
                self._on_charging_update(topic, payload)
            elif topic.startswith(f"car/nearest_node/{self.id}"):
                self._on_nearest_node_response(topic, payload)
            else:
                logger.warning(f"[Car {self.id}] Unknown topic: {topic}")

        except Exception as e:
            logger.error(f"[Car {self.id}] Error handling MQTT message: {e}")

    def _status_reporter(self):

        while self.running:
            try:                
                # Publish status to the self.mqtt_client
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
                        "isFinish" : self.is_finish,
                        "route" : self.route,
                        "driving" : self.driving,
                        "dest" : self.dest,
                        "capacity" : self.capacity
                    }),
                )
                
                yield self.env.timeout(60)
                # self.publish_route_response(self.id, self.route, self.chargeSession)

            except Exception as e:
                logger.error(f"[Car {self.id}] Error in status reporter: {e}")
                time.sleep(1) 
    
    def _on_route_update(self, topic,  payload):
        try:
            new_route = payload.get('route', [])
            if new_route and new_route != self.route:
                self.route = new_route
                # logger.info(f"[Car {self.id}] Route updated with {len(self.route)} nodes.")
        
        except Exception as e:
            logger.error(f"[Car {self.id}] Failed to process route message: {e}")
    
    def _on_nearest_node_response(self, topic, payload):

        try:
            node_id = payload.get("node_id")
            is_origin = payload.get("is_origin", True)
            
            if node_id is not None:
                if is_origin:
                    self.origin = node_id
                    self.current_node = node_id
                    # logger.info(f"[Car {self.id}] Set origin node to {node_id}")
                else:
                    self.dest = node_id
                    # logger.info(f"[Car {self.id}] Set destination node to {node_id}")
                
                if self.origin is not None and self.dest is not None:
                    self._request_route()
        
        except Exception as e:
            logger.error(f"[Car {self.id}] Failed to process nearest node response: {e}")
    
    def _on_charging_update(self, topic, payload):
        
        try:
            if not self.isCharging:
                new_charge_sessions = payload.get('charge_sessions', [])
                self.chargeSession = new_charge_sessions
                # logger.info(f"[Car {self.id}] Charging sessions updated with {len(self.chargeSession)} sessions")
        
        except Exception as e:
            logger.error(f"[Car {self.id}] Failed to process charging message: {e}")
    
    def _request_route(self):

        try:
            if self.origin is None or self.dest is None:
                logger.warning(f"[Car {self.id}] Cannot request route: origin ({self.origin}) or dest ({self.dest}) is None.")
                return
            self.mqtt_client.publish(
                topic="agent/request/route",
                payload=json.dumps({
                    "car_id": self.id,
                    "origin": self.origin,
                    "destination": self.dest
                }),
                qos=1
            )
        except Exception as e:
            logger.error(f"[Car {self.id}] Failed to request route: {e}")
    
    def _request_charging_schedule(self):
        
        try:
            if self.is_finish:
                return
            logger.info(f'[Car {self.id}] Requesting charging schedule with {len(self.route)} route')
            self.mqtt_client.publish(
                topic="agent/request/charging",
                payload=json.dumps({
                    "car_id": self.id,
                    "soc": self.soc,
                    "capacity": self.capacity,
                    "current_node": self.current_node,
                    "route": self.route,
                    "dest" : self.dest,
                    "facilities" : self.facilities_preference,
                }),
            )
        except Exception as e:
            logger.error(f"[Car {self.id}] Failed to request charging schedule: {e}")
    
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
            
            # Return travel time in seconds
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
                logger.info(f"[Car {self.id}] No route. Requesting route from {self.origin} to {self.dest}")
                self._request_route()

                timeout_time = self.env.now + 900 
                while not self.route and self.env.now < timeout_time:
                    yield self.env.timeout(10) 
                if not self.route:
                    logger.warning(f"[Car {self.id}] No route available after timeout. Car {self.id} stopped driving.")
                    self.is_finish = True 
                    self.driving = False
                    self.finish_report()
                    return

            logger.info(f"[Car {self.id}] Starting drive from {self.origin} to {self.dest} with {len(self.route)} nodes and {self.soc:.2f} kWh SOC.")
            
            self._request_charging_schedule()
            yield self.env.timeout(10) 

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
                    self.finish_report()
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

                # logger.info(f"[Car {self.id}] Driving {u}->{v} | Length: {total_length_meters:.1f} m | Time: {travel_time_sec:.1f} s | SOC: {self.soc:.2f} kWh.")
                
                # Simulate travel time
                yield self.env.timeout(travel_time_sec / self.time_factor)
                
                self.total_distance += total_length_meters
                self.total_travel_time += travel_time_sec
                i += 1 # Move next

                current_node_sessions = [cs for cs in self.chargeSession if cs.get("node_id") == v]
                if current_node_sessions:
                    logger.info(f"[Car {self.id}] Found scheduled charging at node {v}. Stopping drive for recharge.")
                    self.driving = False # Stop driving to recharge
                    yield from self.recharge(v)

                    self.driving = True # Resume driving
                    yield self.env.timeout(60)
                    
            if not self.is_finish:
                logger.info(f"[Car {self.id}] Reached destination {self.dest} at {self.env.now:.2f}.")
                self.is_finish = True
                self.jumlah_node = len(self.route)
                self.finish_report()
            
            self.driving = False
            self.isCharging = False
        except simpy.Interrupt:
            logger.info(f"[Car {self.id}] Drive process interrupted.")
            self.driving = False
        except Exception as e:
            logger.error(f"[Car {self.id}] Error in drive process: {e}", exc_info=True)
            self.driving = False 

    
    def recharge(self, node_id: int):
        """Simulate recharging at a charging station based on provided schedule."""
        try:
            # Find the specific scheduled charging session for this car at this node
            current_charging_session_for_car = None
            if self.chargeSession:
                for session_info in self.chargeSession:
                    # Check if the session is for this car, at this node, and is still in the future or current
                    if session_info.get("node_id") == node_id and \
                       session_info.get("car_id") == self.id and \
                       session_info.get("endTime") > self.env.now: # Ensure session is not in the past
                        current_charging_session_for_car = session_info
                        break

            if current_charging_session_for_car is None:
                logger.info(f"[Car {self.id}] No valid, future charging session found in schedule for node {node_id}. Requesting new schedule.")
                self._request_charging_schedule()
                return # Exit recharge, car will resume driving (if self.driving is True)

            scheduled_start_time = current_charging_session_for_car.get("startTime")
            scheduled_end_time = current_charging_session_for_car.get("endTime")
            port_id = current_charging_session_for_car.get("portNumber")
            
            cs = self.G.nodes[node_id].get('charging_station')
            if cs is None:
                logger.error(f"[Car {self.id}] Charging station not found at node {node_id}.")
                return
            
            thisPort = next((p for p in cs.ports if p.id == port_id), None)
            if thisPort is None:
                logger.error(f"[Car {self.id}] Port {port_id} not found at CS {cs.id}.")
                return

            logger.info(f"[Car {self.id}] Arrived at CS {cs.id}, Port {port_id}. Scheduled start: {scheduled_start_time:.2f}, end: {scheduled_end_time:.2f}. Current time: {self.env.now:.2f}")

            # Calculate waiting time until scheduled start
            wait_seconds_to_start = max(0.0, scheduled_start_time - self.env.now)
            self.total_of_waiting_time.append(wait_seconds_to_start * self.time_factor) 
            arrive_time = self.env.now
            if wait_seconds_to_start > 0:
                logger.info(f"[Car {self.id}] Waiting for {wait_seconds_to_start* self.time_factor:.2f} seconds until scheduled charging starts.")
                yield self.env.timeout(wait_seconds_to_start)
            logger.info(f"[Car {self.id}] Waiting for {wait_seconds_to_start * self.time_factor:.2f} seconds until scheduled charging starts.")

            # --- Actual Charging Logic ---
            # Calculate actual possible charging duration
            # Ensure it doesn't charge past scheduled_end_time or full capacity
            while not thisPort.isAvailable:
                yield self.env.timeout(20)
            
            wait_seconds_to_start = max(0.0, self.env.now - arrive_time)
            logger.info(f"[Car {self.id}] Waiting again for {wait_seconds_to_start * self.time_factor:.2f} seconds until scheduled charging starts.")

            self.total_of_waiting_time.append(wait_seconds_to_start * self.time_factor) 
            # Time remaining in scheduled slot, from current simulation time
            remaining_slot_duration = max(0.0, scheduled_end_time - self.env.now)
            
            # Time needed to fully charge
            time_to_full_charge = (self.capacity - self.soc) * 3600 / thisPort.power / self.time_factor if thisPort.power > 0 else float('inf')

            # Actual charging duration is limited by remaining slot and time to full charge
            actual_charging_duration_sec = min(remaining_slot_duration, time_to_full_charge)
            actual_charging_duration_sec = max(0.0, actual_charging_duration_sec) # Ensure positive

            if actual_charging_duration_sec <= 0:
                logger.info(f"[Car {self.id}] Actual charging duration is 0 or less (already full or slot expired). Skipping recharge.")
                self._request_charging_schedule() # Request update for next steps
                return

            self.isCharging = True
            self.number_of_charge += 1

            logger.info(f'[Car {self.id}] Actual Charging duration: {actual_charging_duration_sec:.2f} seconds ({actual_charging_duration_sec/60:.2f} minutes). Power: {thisPort.power} kW.')
            self.start_charging_report(cs.id, port_id)

            charged_energy = 0.0
            charge_start_sim_time = self.env.now
            if actual_charging_duration_sec <=0:
                return
            thisPort.isAvailable = False

            time_step_for_charge = 10 
            while self.env.now < (charge_start_sim_time + actual_charging_duration_sec) and self.soc < self.capacity:
                
                time_remaining_in_step = (charge_start_sim_time + actual_charging_duration_sec) - self.env.now
                if time_remaining_in_step <= 0: break 

                time_to_yield = min(time_step_for_charge, time_remaining_in_step)
                
                energy_gain_this_step = thisPort.power * time_to_yield / 3600 * self.time_factor# kWh

                if self.soc + energy_gain_this_step > self.capacity:
                    energy_gain_this_step = self.capacity - self.soc
                    # Recalculate time_to_yield for the adjusted energy gain
                    if thisPort.power > 0:
                        time_to_yield = energy_gain_this_step * 3600 / thisPort.power
                    else:
                        time_to_yield = 0 
                    if time_to_yield <= 0: 
                        break

                self.soc += energy_gain_this_step
                charged_energy += energy_gain_this_step

                yield self.env.timeout(time_to_yield)
            sat = sum([ 1 for fac in self.facilities_preference if fac in cs.facilities])/2 if self.facilities_preference else 1

            self.satisfied_facilities = sat
            thisPort.isAvailable = True
            # Update final stats after charging completes
            self.last_charging_node = node_id
            self.total_of_charge_energy.append(charged_energy)
            self.total_of_charge_cost.append(charged_energy * thisPort.price)
            self.total_of_charge_time.append((charge_start_sim_time, self.env.now)) # Actual time spent charging

            # Update CS and Port usage stats (these are attributes of CS/Port objects)
            # Assuming these attributes are updated *globally* in the CS/Port objects
            cs.usage_time += (self.env.now - charge_start_sim_time)
            energy_delivered = charged_energy
            revenue = charged_energy * thisPort.price
            cs.number_of_charging_session += 1
            thisPort.usage_time += (self.env.now - charge_start_sim_time)
            thisPort.number_of_charging_session += 1
            self.chargeSession = []
            self.stop_charging_report(cs.id, port_id, cs.number_of_charging_session, energy_delivered, revenue)
            
            # Request new schedule after charging is done, in case route or needs changed
            # This is crucial for the agent to re-evaluate the car's next steps.
            self._request_charging_schedule()
            self.isCharging = False # Car is no longer charging
            self.driving = True     # Car can resume driving
            
            logger.info(f"[Car {self.id}] Finished recharge at node {node_id} with SOC: {self.soc:.2f} kWh, Charged: {charged_energy:.2f} kWh.")
        
        except simpy.Interrupt:
            logger.info(f"[Car {self.id}] Recharge process interrupted by SimPy. Car state: isCharging={self.isCharging}")
            self.isCharging = False # Ensure state is updated on interrupt
            # If interrupted, car might need a new schedule immediately
            self._request_charging_schedule() 
        except Exception as e:
            logger.error(f"[Car {self.id}] Error in recharge process: {e}", exc_info=True)
            self.isCharging = False # Ensure state is updated on error
            self.driving = True # Allow it to resume driving if possible
            self._request_charging_schedule() # Try to get new schedule
    
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
        
        # Wait for status thread to finish (compatible with older Python versions)
        # if self.status_thread and self.status_thread.is_alive():
        #     try:
        #         # Use join without timeout parameter for compatibility
        #         self.status_thread.join()
        #         # Give it a moment to finish
        #         time.sleep(0.5)
        #     except Exception as e:
        #         logger.debug(f"[Car {self.id}] Error joining status thread: {e}")
        
        logger.info(f"[Car {self.id}] Car stopped successfully")

    def get_charging_access(self, cs_id, port_number):
        try:
            payload = {
                "cs_id" : cs_id,
                "port_number" : port_number,
                "time" : self.env.now
            }
            self.mqtt_client.publish(topic=f'request/confirmation/{self.id}', payload=json.dumps(payload))
        except Exception as e:
            logger.error(f"[Car {self.id}] Error in dget charging access process: {e}", exc_info=True)
            self.driving = False # Ensure driving is set to False on error
    
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
                                    "isCharging": True, # Explicitly true
                                    "charging_session": self.chargeSession,
                                    "vel": self.vel,
                                    "current_node": self.current_node,
                                    "isFinish": self.is_finish
                                }),
                                 )
        
        self.mqtt_client.publish(topic=f"cs/status/{cs_id}/{port_id}",
                                 payload=json.dumps({
                                     "time" : self.env.now,
                                    "isCharging" : True,
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
                                    "isCharging": False, # Explicitly false
                                    "charging_session": self.chargeSession,
                                    "vel": self.vel,
                                    "current_node": self.current_node,
                                    "isFinish": self.is_finish
                                                    }),
                                 )
        
        self.mqtt_client.publish(topic=f"cs/status/{cs_id}/{port_id}",
                                 payload=json.dumps({
                                    "time" : self.env.now,
                                    "isCharging" : False,
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
                                        "estimated_energy" : self.consume_energy(20, nx.shortest_path_length(self.G, self.origin, self.dest, 'length')),
                                        "satisfied_facilities" : self.satisfied_facilities
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

    # def publish_route_response(self, car_id: str, new_route_nodes: list, charging_schedules: list):
    #     """
    #     Converts new_route from nodes to Location[], and publish the full route response via MQTT.
    #     """
    #     # Convert list of nodes to Location[]
    #     linestring = self.extract_linestring_from_route(self.route)
    #     converted_route = [
    #         {"latitude": float(lat), "longitude": float(lon)}
    #         for lon, lat in linestring.coords  # shapely coords are (x, y) = (lon, lat)
    #     ]
    #     total_distance = nx.shortest_path_length(self.G, self.origin, self.dest)
    #     cost = 0
    #     if self.chargeSession:
    #         cost = self.chargeSession[0].get('cost')

    #     message = {
    #         "route": converted_route,
    #         "chargingSchedules": charging_schedules,
    #         "totalDistance": total_distance,
    #         "estimatedTravelTime": total_distance/20,
    #         "totalCost": cost
    #     }

    #     topic = f"car/real/route/{car_id}"
    #     payload = json.dumps(message)
    #     self.mqtt_client.publish(topic=topic, payload=payload)
    #     print(f" Published route to {topic}")

        