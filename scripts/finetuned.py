import optuna
import pickle
import osmnx as ox
from integration import EvaluatedSimulationRunner
import networkx as nx

def create_test_graph():
    with open("graph_with_cs.pkl", "rb") as f:
        G = pickle.load(f)
        G = ox.project_graph(G, to_crs='EPSG:4326')
        G = G.to_undirected()
    return G

def objective(trial, G):

    weights = {
        'waiting_time': 0.4,
        'travel_time': 0.3,
        'utilization': 0.1,
        'remaining_soc': 0.5,  
        'satisfied_facilities': 0.2
    }

    param_population_size = trial.suggest_int("population_size", 10, 50)
    param_generations = trial.suggest_int("generations", 10, 100)
    param_mutation_rate = trial.suggest_float("mutation_rate", 0.05, 0.5)
    param_crossover_rate = trial.suggest_float("crossover_rate", 0.5, 0.9)

    # Run simulation
    runner = EvaluatedSimulationRunner(
        G,
        evaluation_duration=60,
        weights=weights,
        optimizer_params={
            "population_size": param_population_size,
            "generations": param_generations,
            "mutation_rate": param_mutation_rate,
            "crossover_rate": param_crossover_rate
        }
    )
    
    runner.run_simulation_with_evaluation()

    # Get metrics
    metrics = runner.evaluator.calculate_comprehensive_metrics()

    avg_waiting_time = metrics["waiting_time"]["average_waiting_time"]
    avg_travel_time = metrics["journey_analysis"]["average_travel_time"]
    avg_utilization = metrics["resource_utilization"]["average_port_utilization"]
    avg_soc_improvement = metrics["energy_efficiency"]["average_soc_improvement"]  # pakai soc_improvement
    avg_satisfied_facilities = metrics["resource_utilization"]["average_satisfied_facilities"]

    # Calculate weighted objective
    objective_value = (
        weights["waiting_time"] * avg_waiting_time +
        weights["travel_time"] * avg_travel_time +
        weights["utilization"] * (1.0 - avg_utilization) +
        weights["remaining_soc"] * (1.0 - avg_soc_improvement) +
        weights["satisfied_facilities"] * (1.0 - avg_satisfied_facilities)
    )

    print(f"Trial {trial.number} → Objective: {objective_value:.4f} "
          f"| population_size: {param_population_size}, generations: {param_generations}, "
          f"mutation_rate: {param_mutation_rate:.3f}, crossover_rate: {param_crossover_rate:.3f}")
    
    return objective_value

# Main tuning loop
def main():
    G = create_test_graph()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, G), n_trials=10)

    print("Best trial:")
    print(study.best_trial)

    # Save study
    import joblib
    joblib.dump(study, "study_results_genetic.pkl")

    # Save best params
    best_params = study.best_trial.params
    with open("best_params.json", "w") as f:
        import json
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    main()
