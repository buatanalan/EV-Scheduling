import simpy
import time
import threading
from scripts.evaluation import create_evaluator_for_simulation
import pickle
import osmnx as ox
from scripts.simulation import Simulation
import paho.mqtt.client as mqtt
import json
import yaml
import argparse
import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379

#SETUP
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

class EvaluatedSimulationRunner:
    
    def __init__(self, G, evaluation_duration=30000, weights=None, optimizer_params=None):
        self.G = G
        self.evaluation_duration = evaluation_duration

        # Initialize empty component placeholders
        self.env = None
        self.evaluator = None
        self.simulation = None
        self.cars = {}
        
        # Simulation control
        self.running = False
        self.sim_thread = None
        self.weights = weights
        self.optimizer_params = optimizer_params
        self.mqtt_client = mqtt.Client(client_id="00-id-simulation")
        
        self.mqtt_client.connect('localhost', 1883)
        self.mqtt_client.loop_start()
        self.mqtt_client.subscribe('create/real')
        self.mqtt_client.message_callback_add(sub='create/real', callback=self.create_new_car)

    def create_new_car(self, client, userdata, message):
        print('create real is called')
        from scripts.car import Car

        car = Car(
                id="car_0",
                env=self.env,
                G=self.G,
                soc=17,
                origin=74381,
                dest=73970,
                capacity=42,
                facilities_preference=["Solaria, Alfamart"]
                )
        
        return car

    def _setup_components(self):

        print("🔄 Resetting simulation components...")

        self.env = simpy.Environment()

        # Create evaluator
        self.evaluator = create_evaluator_for_simulation()

        # Create simulation components
        self.simulation = Simulation(self.G)
        
        # Clear cars
        self.cars = {}
        
        print("Components ready for new simulation")
    
    def create_cars_from_config(self, users_config, schedule):
        if schedule == 'y' or schedule == 'Y':
            from scripts.car import Car
        else:
            from scripts.car_2 import Car
        if users_config:

            for user_config in users_config:
                try:
                    for key in r.scan_iter("car:*:info"):
                        car_raw_data = r.hgetall(key)
                        car_data = {k.decode(): v.decode() for k, v in car_raw_data.items()}
                        if car_data.get("type") == "bmw i3":
                            break

                    user_raw_data = r.hgetall(f"user:{user_config.get('user_id')}:info")
                    user = {k.decode(): v.decode() for k, v in user_raw_data.items()}
                    car = Car(
                        id=str(user_config["user_id"]),
                        env=self.env,
                        G=self.G,
                        soc=user_config["soc"],
                        origin=user_config["origin"],
                        dest=user_config["destination"],
                        capacity=float(car_data.get('capacity')),
                        facilities_preference=user.get("facilities_preferences")
                    )
                    self.cars[user_config["user_id"]] = car
                    self.mqtt_client.publish(topic="sim/create_vehicle", payload=json.dumps({"user_id": car.id, "start_time": 0, 'initial_soc' : user_config['soc']}))            

                except Exception as e:
                    print(f" Failed to create car {user_config['user_id']}: {e}")
                
    def run_simulation_with_evaluation(self, cars_config_path, schedule):
        print(1)
        try:
            print("Starting Evaluated Simulation")

            print("Network layer started")

            self._setup_components()
            try : 
                with open(cars_config_path, "r") as f:
                    cars_config = yaml.safe_load(f)
            except Exception as e:
                cars_config = None
            self.evaluator.start_monitoring()
            print("Evaluation monitoring started")

            self.create_cars_from_config(cars_config, schedule)

            self.running = True
            self.sim_thread = threading.Thread(target=self._simulation_loop)
            self.sim_thread.daemon = True
            self.sim_thread.start()
            print(" Simulation loop started")

            while self.running:
                time.sleep(10)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        finally:

            self._stop_simulation()
            if self.sim_thread and self.sim_thread.is_alive():
                self.sim_thread.join()

            self._generate_final_report()

    def _simulation_loop(self):

        try:
            i = 0
            while self.running:
                self.env.run(until=self.env.now + 1)
                if i%100==0:
                    self.cek_is_all_finish()
                i+=1
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in simulation loop: {e}")

    def cek_is_all_finish(self):
        num_finish = 0
        for car in self.cars.values():
            if car.dest == car.current_node or car.is_finish:
                num_finish+=1
        # print(f"{num_finish} cars finish from {len(self.cars)}")
        if num_finish == len(self.cars):
            self._stop_simulation()

    def _stop_simulation(self):

        print("\nStopping simulation...")
        self.running = False

        for car in self.cars.values():
            try:
                car.stop()
            except Exception as e:
                print(f"Error stopping car {car.id}: {e}")
        self._stop_all_cs()
        self.evaluator.stop_monitoring()

        try:
            self.agent.stop()
        except:
            pass

        try:
            self.simulation.stop()
        except:
            pass

        print("All components stopped")

    def _stop_all_cs(self):
        all_cs = [
            data.get('charging_station') 
            for node, data in self.G.nodes(data=True) 
            if 'charging_station' in data
        ]

        for cs in all_cs:
            # Buat report port detail
            ports_report = []
            for port in cs.ports:
                
                ports_report.append({
                    "port_id": port.id,
                    "usage_time": port.usage_time,
                    "number_of_charging_session": port.number_of_charging_session,
                    "number_of_power_delivered" : port.usage_time * port.power
                })

            payload = json.dumps({
                "usage_time": cs.usage_time,
                "number_of_charging_session": cs.number_of_charging_session,
                "number_of_energy_delivered": sum([port.get('number_of_power_delivered') for port in ports_report])
            })

            self.mqtt_client.publish(
                topic=f"cs/report/{cs.id}",
                payload=payload
            )
            # print(f"Published CS report for {cs.id}")


    def _generate_final_report(self):

        print("\n" + "="*80)
        print("FINAL EVALUATION REPORT")
        print("="*80)

        report = self.evaluator.generate_detailed_report()
        print(report)

        self.evaluator.export_metrics_to_csv("simulation_results.csv")

        metrics = self.evaluator.calculate_comprehensive_metrics()

        print("KEY PERFORMANCE INDICATORS:")
        print(f"   • Scheduling Success Rate: {metrics['scheduling']['scheduling_success_rate']:.1%}")
        print(f"   • Average Waiting Time: {metrics['waiting_time']['average_waiting_time']:.1f} minutes")
        print(f"   • Average Port Utilization: {metrics['resource_utilization']['average_port_utilization']:.1%}")
        print(f"   • Energy Efficiency Ratio: {metrics['energy_efficiency']['energy_efficiency_ratio']:.2f}")
        print(f"   • Charging Dependency Rate: {metrics['journey_analysis']['charging_dependency_rate']:.1%}")

        self.evaluator.plot_enhanced_metrics()

def create_test_graph():

    with open("./resources/graph_with_cs.pkl", "rb") as f:
        G = pickle.load(f)
        if "crs" not in G.graph:
            G.graph["crs"] = "EPSG:4326"
        G = ox.project_graph(G, to_crs='EPSG:3857')
        G = G.to_undirected()
    
    return G

def main():

    try:
        parser = argparse.ArgumentParser(description="Integration script")
        parser.add_argument("test_scenario")
        parser.add_argument("schedule")
        args = parser.parse_args()

        print("Loading graph...")
    
        G = create_test_graph()
        
        # Create and run evaluated simulation
        runner = EvaluatedSimulationRunner(G, evaluation_duration=7200)
        runner.run_simulation_with_evaluation(cars_config_path=f"./scripts/test_case_{args.test_scenario}.yaml", schedule=args.schedule)
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
