import pickle
import osmnx as ox
import simpy
import time
import threading
from run_agent.agent import EnhancedAgent
import signal
import sys
import argparse

class AgentRunner:
    def __init__(self, graph_path, optimizer_type='genetic'):
        # Load and prepare graph
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)

        if "crs" not in self.G.graph:
            self.G.graph["crs"] = "EPSG:3857"

        self.G = self.G.to_undirected()
        
        self.agent = EnhancedAgent(
            self.G,
            optimizer_type=optimizer_type,
            weights=None,
            optimizer_params=None
        )
        self.agent_thread = None

    def run_agent(self):
        print("Running simulation before Flask starts...")
        self.agent._process_requests()
        print("Simulation finished.")

    def start_agent(self):
        self.agent_thread = threading.Thread(target=self.run_agent)
        self.agent_thread.start()

    def shutdown(self, signum, frame):
        print("Stopping agent...")
        self.agent.stop()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Integration script")
    parser.add_argument("optimizer_type")
    args = parser.parse_args()
    # Initialize runner
    sim_runner = AgentRunner("./resources/graph_with_cs.pkl", optimizer_type=args.optimizer_type)

    # Setup signal handlers
    signal.signal(signal.SIGINT, sim_runner.shutdown)
    signal.signal(signal.SIGTERM, sim_runner.shutdown)
    sim_runner.start_agent()


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sim_runner.shutdown(signal.SIGINT, None)

if __name__ == "__main__":
    main()
