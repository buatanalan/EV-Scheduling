from dataclasses import dataclass


@dataclass
class PortSession:
    car_id : int
    start_time : int
    stop_time : int

class ChargingStation:
    def __init__(self, env, id, ports, lat, lon, simulation, node_id, name, facilities):
        self.env = env
        self.id = id
        self.ports = ports
        self.lat = lat
        self.lon = lon
        self.simulation = simulation
        self.node_id = node_id
        self.name=name
        self.facilities = facilities
        self.usage_time = 0
        self.number_of_charging_session = 0

    def registerTosimulation(self, simulation):
        simulation.add_cs(self)

    def getPort(self, port_id):
        for port in self.ports:
            if port.id == port_id:
                return port

        return None
    
    def setEnv(self, env, simulation):
        self.env = env
        self.registerTosimulation(simulation)

class Port:
    def __init__(self, env, id, power, price, portType, isAvailable, portSession):
        self.env = env
        self.id = id
        self.power = power
        self.price = price
        self.portType = portType
        self.isAvailable = isAvailable
        self.portSession = portSession
        self.usage_time = 0
        self.number_of_charging_session = 0

    def get_earliest_available_time(self, current_time):
        earliest_time = current_time
        if self.portSession == None:
            return earliest_time
        sorted_port_session = sorted(self.portSession, key=lambda x: x.start_time)
        if sorted_port_session[0].start_time > earliest_time:
            return earliest_time
        for i in range(len(sorted_port_session)):
            if sorted_port_session[i].stop_time < current_time:
                continue
            if sorted_port_session[i].start_time > earliest_time:
                return earliest_time
            else:
                earliest_time = sorted_port_session[i].stop_time
                continue 

        return earliest_time
    
    def add_reservation(self, car_id, start_time, stop_time):
        if self.portSession == None:
            self.portSession = [PortSession(car_id, start_time, stop_time)]
        else:
            self.portSession.append(PortSession(car_id, start_time, stop_time))