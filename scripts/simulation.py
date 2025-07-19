import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd
import random
from charging_stations.refactored_charging_station import Port
import datetime
import paho.mqtt.client as mqtt
from dataclasses import asdict

# Import from our modules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Simulation:
    """
    Refactored Simulation class that communicates via the network layer
    """
    def __init__(self, G):
        self.G = G
        self.running = True
        
        # Register charging stations from the graph
        self.mqtt_client = mqtt.Client(client_id="02-id-simulation")

        
        self.mqtt_client.connect('localhost', 1883)
        self.mqtt_client.loop_start()
        # Start the simulation update thread
        # self.update_thread = threading.Thread(target=self._simulation_updater)
        # self.update_thread.daemon = True
        # self.update_thread.start()
        
        self._register_charging_stations()
        logger.info("Simulation initialized and ready")
    
    def _register_charging_stations(self):
        """Register charging stations from the graph as digital twins"""
        for node_id, node_data in self.G.nodes(data=True):
            if 'charging_station' in node_data:
                cs = node_data['charging_station']
                if hasattr(cs, 'id') and hasattr(cs, 'name') and hasattr(cs, 'lat') and hasattr(cs, 'lon'):
                    # Register the charging station
                    # cs_twin = twin_registry.register_charging_station(
                    #     str(cs.id), 
                    #     cs.name, 
                    #     cs.lat, 
                    #     cs.lon, 
                    #     cs.node_id
                    # )
                    
                    # Register ports
                    if hasattr(cs, 'ports'):
                        if cs.ports is None:
                            # Create a new default port (adjust as needed)
                            default_port = Port(env=cs.env, id=1, power=22, price=2247, isAvailable=True, portType='standard', portSession=[])  # You can generate unique ID here
                            cs.ports = [default_port]

                        for port in cs.ports:
                            # cs_twin.update_port(
                            #     port.id,
                            #     True,   # Assume initially available
                            #     []      # No sessions initially
                            # )
                            self.mqtt_client.publish(topic=f"cs/status/{cs.id}/{port.id}",
                                 payload=json.dumps({
                                    "time" : 0,
                                    "isCharging" : False,
                                    "cars_served" : 0,
                                    "total_energy_delivered" : 0,
                                    "total_revenue" : 0
                                 }))
                    
                    # logger.info(f"Registered charging station {cs.id} at node {node_id}")
                self.mqtt_client.publish(topic="sim/create_station", payload=json.dumps({"stationId" : cs.id, 'name' : cs.name, 'facilities' : cs.facilities, 'latitude' : cs.lat, 'longitude' : cs.lon ,"ports" : [i.__dict__ for i in cs.ports]}))
                