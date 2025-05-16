import numpy as np
import random
import sys
from datetime import datetime
import time
import argparse
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle

# Redirect output to a file
output_filename = f"results.txt"
sys.stdout = open(output_filename, "w")


# --- 1. Distance Generation ---
def generate_distance_matrix(num_cities, min_dist=3, max_dist=50):
  """Generates a symmetric distance matrix for a given number of cities."""
  if num_cities <= 0:
    return np.array([])
  distances = np.zeros((num_cities, num_cities))
  for i in range(num_cities):  # to loop through first row
    # to loop through columns
    # we start from i + 1 to make sure all cities are connected but not same city
    # the diagonal is 0
    for j in range(i + 1, num_cities):
      dist = random.randint(min_dist, max_dist)
      distances[i, j] = dist
      distances[j, i] = dist
  return distances


# --- 2. ACO Algorithm Implementation ---
class AntColonyOptimizer:
  def __init__(
      self,
      distances,
      n_ants,
      n_iterations,
      rho,
      q_val,
      initial_pheromone=1.0,
      alpha=1.0,     # Pheromone influence
      beta=2.0,      # Distance influence
      enable_local_search=False,
      track_performance=True,
      visualization=True,
  ):
    """
    distances: 2D numpy array of distances between cities
    n_ants: Number of ants
    n_iterations: Number of iterations
    rho: Pheromone evaporation rate
    q_val: Pheromone deposit constant (Q in literature, often 1 or related to tour length)
    initial_pheromone: Initial pheromone level on all paths
    alpha: Controls importance of pheromone trail (higher = more influence)
    beta: Controls importance of distance (higher = more influence)
    enable_local_search: Whether to apply 2-opt local search after each iteration
    track_performance: Whether to track performance metrics
    visualization: Whether to generate visualizations
    """
    self.distances = distances
    self.n_cities = distances.shape[0]
    self.n_ants = n_ants
    self.n_iterations = n_iterations
    self.rho = rho
    self.q_val = q_val
    self.initial_pheromone = initial_pheromone
    self.alpha = alpha
    self.beta = beta
    self.enable_local_search = enable_local_search
    self.track_performance = track_performance
    self.visualization = visualization

    # Heuristic information (eta = 1/distance)
    # Add a small epsilon to avoid division by zero if any distance is 0 (though our generator prevents this)
    epsilon = 1e-10
    self.eta = 1.0 / (distances + epsilon)
    # No heuristic value for staying in the same city
    np.fill_diagonal(self.eta, 0)

    # Pheromone matrix, initialized
    self.pheromones = np.full(
        (self.n_cities, self.n_cities), self.initial_pheromone
    )
    np.fill_diagonal(
        self.pheromones, 0
    )  # No pheromone for staying in the same city

    self.best_tour = None
    self.best_tour_length = float("inf")

    # Performance tracking
    if self.track_performance:
      self.iteration_best_lengths = []
      self.iteration_avg_lengths = []
      self.iteration_times = []
      self.pheromone_history = []
      self.best_tour_history = []

  def _select_next_city(self, current_city, visited_cities, ant_pheromones):
    """Probabilistically selects the next city for an ant to visit."""
    probabilities = []
    unvisited_cities = []

    # Calculate numerators for probability calculation
    # P(i,j) = (τ[i][j]^α * η[i][j]^β)
    for city_idx in range(self.n_cities):
      if not visited_cities[city_idx]:
        unvisited_cities.append(city_idx)
        tau_ij = ant_pheromones[current_city, city_idx]
        eta_ij = self.eta[current_city, city_idx]
        # Apply alpha and beta influence parameters
        prob_numerator = (tau_ij ** self.alpha) * (eta_ij ** self.beta)
        probabilities.append(prob_numerator)

    if not unvisited_cities:  # Should not happen if tour is not complete
      return None

    # Normalize probabilities
    sum_probs = sum(probabilities)
    if (
        sum_probs == 0
    ):  # Fallback: if all probabilities are zero (e.g., no pheromone)
      # then choose randomly among unvisited cities
      if unvisited_cities:
        return random.choice(unvisited_cities)
      else:
        return None  # Should not be reached

    probabilities = [p / sum_probs for p in probabilities]

    # Select next city using roulette wheel selection
    next_city = random.choices(unvisited_cities, weights=probabilities, k=1)[0]
    return next_city

  def _construct_tour(self, start_city, ant_pheromones):
    """Constructs a tour for a single ant starting from start_city."""
    tour = [start_city]
    current_city = start_city
    visited_cities = np.zeros(self.n_cities, dtype=bool)
    visited_cities[start_city] = True
    tour_length = 0.0

    for _ in range(self.n_cities - 1):
      next_city = self._select_next_city(
          current_city, visited_cities, ant_pheromones
      )
      if next_city is None:  # Should not happen in a valid TSP setup
        break
      tour.append(next_city)
      visited_cities[next_city] = True
      tour_length += self.distances[current_city, next_city]
      current_city = next_city

    # Complete the loop: back to start city
    if tour:  # Check if tour was actually built
      tour_length += self.distances[current_city, start_city]

    return tour, tour_length

  def _apply_2opt_local_search(self, tour, tour_length):
    """
    Applies 2-opt local search to improve a tour.
    2-opt repeatedly replaces two edges with two other edges to reduce tour length.
    """
    improved = True
    best_tour = tour.copy()
    best_length = tour_length

    # Continue until no more improvements are found
    while improved:
      improved = False

      # Try all possible 2-opt swaps
      for i in range(1, len(best_tour) - 1):
        for j in range(i + 1, len(best_tour)):
          # Skip adjacent edges
          if j - i == 1:
            continue

          # Create a new tour by reversing the segment between i and j
          new_tour = best_tour.copy()
          new_tour[i:j] = best_tour[i:j][::-1]

          # Calculate the new tour length
          new_length = 0
          for k in range(len(new_tour) - 1):
            new_length += self.distances[new_tour[k], new_tour[k + 1]]
          new_length += self.distances[new_tour[-1],
                                       new_tour[0]]  # Close the loop

          # If the new tour is better, keep it
          if new_length < best_length:
            best_tour = new_tour
            best_length = new_length
            improved = True
            break  # Early termination: accept the first improvement

        if improved:
          break

    return best_tour, best_length

  def _update_pheromones(self, ant_tours):
    """Updates pheromone trails based on ant tours and evaporation."""
    # 1. Evaporation
    self.pheromones *= 1.0 - self.rho

    # 2. Deposition
    for tour, tour_length in ant_tours:
      if tour_length == 0:
        continue  # Avoid division by zero for invalid tours
      pheromone_deposit = self.q_val / tour_length
      for i in range(self.n_cities):
        city1_idx = tour[i]
        city2_idx = tour[(i + 1) % self.n_cities]  # Loop back to start

        self.pheromones[city1_idx, city2_idx] += pheromone_deposit
        self.pheromones[city2_idx, city1_idx] += pheromone_deposit  # Symmetric

    # Save pheromone state if tracking is enabled
    if self.track_performance:
      self.pheromone_history.append(self.pheromones.copy())

  def _save_performance_snapshot(self, iteration, iteration_start_time, ant_tours):
    """
    Saves performance metrics for current iteration.
    Used for analysis and visualization.
    """
    # Calculate average tour length for this iteration
    tour_lengths = [length for _, length in ant_tours if length > 0]
    avg_length = sum(tour_lengths) / len(tour_lengths) if tour_lengths else 0

    # Track iteration metrics
    self.iteration_best_lengths.append(self.best_tour_length)
    self.iteration_avg_lengths.append(avg_length)
    self.iteration_times.append(time.time() - iteration_start_time)

    # Track best tour history (for visualization)
    self.best_tour_history.append(
        self.best_tour.copy() if self.best_tour else None)

    # Print detailed progress every 10 iterations
    if (iteration + 1) % 10 == 0 or iteration == 0:
      print(f"Iteration {iteration + 1}/{self.n_iterations}:")
      print(f"  Best tour length: {self.best_tour_length:.2f}")
      print(f"  Average tour length: {avg_length:.2f}")
      print(f"  Iteration time: {self.iteration_times[-1]:.4f} seconds")

      # Print pheromone matrix summary (diagonal values are always 0)
      if (iteration + 1) % 10 == 0:
        print("\nPheromone Matrix (values from selected edges):")

        # Print a sample of pheromone values (for clarity)
        for i in range(min(5, self.n_cities)):
          for j in range(min(5, self.n_cities)):
            if i != j:  # Skip diagonal (self-loops)
              print(f"  Edge ({i},{j}): {self.pheromones[i, j]:.4f}")
        print("  ...")  # Indicate truncation

        # Print information about the current best tour
        if self.best_tour:
          print(f"\nCurrent Best Tour: {self.best_tour}")
          # Calculate pheromone strength on best tour
          best_tour_pheromone = 0
          for i in range(len(self.best_tour)):
            city1 = self.best_tour[i]
            city2 = self.best_tour[(i + 1) % len(self.best_tour)]
            best_tour_pheromone += self.pheromones[city1, city2]
          print(f"Total pheromone on best tour: {best_tour_pheromone:.4f}")
        print("-" * 40)

  def visualize_results(self, city_count, ant_count):
    """
    Creates visualizations of the optimization process:
    1. Tour length convergence over iterations
    2. Best tour visualization (if coordinates are available)
    3. Pheromone concentration heatmap
    """
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)

    # 1. Plot convergence graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, self.n_iterations + 1),
             self.iteration_best_lengths, 'b-', label='Best Tour Length')
    plt.plot(range(1, self.n_iterations + 1),
             self.iteration_avg_lengths, 'r--', label='Average Tour Length')
    plt.xlabel('Iteration')
    plt.ylabel('Tour Length')
    plt.title(f'ACO Convergence - {city_count} Cities, {ant_count} Ants')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(
        f"visualizations/convergence_{city_count}cities_{ant_count}ants.png")
    plt.close()

    # 2. Visualize pheromone concentration as a heatmap
    plt.figure(figsize=(8, 6))
    # Use the final pheromone matrix
    masked_pheromones = np.ma.masked_where(
        np.eye(self.n_cities) == 1, self.pheromones)
    plt.imshow(masked_pheromones, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Pheromone Strength')
    plt.title(
        f'Final Pheromone Distribution - {city_count} Cities, {ant_count} Ants')
    plt.xlabel('Destination City')
    plt.ylabel('Origin City')
    plt.tight_layout()
    plt.savefig(
        f"visualizations/pheromone_{city_count}cities_{ant_count}ants.png")
    plt.close()

    # 3. Create an animation of pheromone development (saves snapshots at regular intervals)
    snapshot_interval = max(1, self.n_iterations // 5)  # Save 5 snapshots
    for i in range(0, self.n_iterations, snapshot_interval):
      if i < len(self.pheromone_history):
        plt.figure(figsize=(8, 6))
        masked_pheromones = np.ma.masked_where(np.eye(self.n_cities) == 1,
                                               self.pheromone_history[i])
        plt.imshow(masked_pheromones, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Pheromone Strength')
        plt.title(
            f'Pheromone at Iteration {i+1} - {city_count} Cities, {ant_count} Ants')
        plt.xlabel('Destination City')
        plt.ylabel('Origin City')
        plt.tight_layout()
        plt.savefig(
            f"visualizations/pheromone_{city_count}cities_{ant_count}ants_iter{i+1}.png")
        plt.close()

    print(f"\nVisualizations saved to ./visualizations/ directory")

  def save_solution(self, city_count, ant_count):
    """
    Saves the best solution found to file for later analysis.
    """
    # Create output directory if it doesn't exist
    os.makedirs("solutions", exist_ok=True)

    solution_data = {
        "city_count": city_count,
        "ant_count": ant_count,
        "best_tour": self.best_tour,
        "best_tour_length": self.best_tour_length,
        "parameters": {
            "n_iterations": self.n_iterations,
            "rho": self.rho,
            "q_val": self.q_val,
            "alpha": self.alpha,
            "beta": self.beta,
            "initial_pheromone": self.initial_pheromone,
        },
        "performance": {
            "iteration_best_lengths": self.iteration_best_lengths,
            "iteration_avg_lengths": self.iteration_avg_lengths,
            "iteration_times": self.iteration_times,
        }
    }

    # Save as JSON for human readability
    with open(f"solutions/solution_{city_count}cities_{ant_count}ants.json", 'w') as f:
      json.dump(solution_data, f, indent=2)

    # Save full data (including pheromone history) as pickle for potential reuse
    with open(f"solutions/full_data_{city_count}cities_{ant_count}ants.pkl", 'wb') as f:
      pickle.dump({
          "solution": solution_data,
          "pheromone_history": self.pheromone_history,
          "best_tour_history": self.best_tour_history,
          "distances": self.distances
      }, f)

    print(f"Solution saved to ./solutions/ directory")

  def run(self, city_count=None, ant_count=None):
    """Runs the ACO algorithm."""
    if self.n_cities == 0:
      print("No cities to run ACO on.")
      return [], float("inf")

    print(
        f"\nRunning ACO: {self.n_ants} ants, {self.n_iterations} iterations, "
        f"rho={self.rho}, Q={self.q_val}, alpha={self.alpha}, beta={self.beta}"
    )

    # Use tqdm for a progress bar if available
    total_start_time = time.time()

    # Initialize performance tracking if enabled
    if self.track_performance:
      self.iteration_best_lengths = []
      self.iteration_avg_lengths = []
      self.iteration_times = []
      self.pheromone_history = []
      self.best_tour_history = []

    # Main optimization loop
    for iteration in tqdm(range(self.n_iterations), desc="ACO Progress"):
      iteration_start_time = time.time()

      # Store (tour, tour_length) for all ants in this iteration
      ant_tours = []

      # Each ant constructs a tour
      for ant_id in range(self.n_ants):
        # Start city can be random for each ant or fixed (e.g., ant_id % n_cities)
        # For simplicity here, let each ant pick a random start
        # Or, to ensure diversity if n_ants <= n_cities:
        start_city = ant_id % self.n_cities
        # start_city = random.randint(0, self.n_cities - 1)

        # Ants should use the global pheromone trails for decisions
        tour, tour_length = self._construct_tour(start_city, self.pheromones)

        # Apply local search if enabled
        if self.enable_local_search and tour:
          tour, tour_length = self._apply_2opt_local_search(tour, tour_length)

        ant_tours.append((tour, tour_length))

        if tour_length < self.best_tour_length:
          self.best_tour_length = tour_length
          self.best_tour = tour.copy()  # Make a deep copy to avoid reference issues

      # Update pheromones based on all tours from this iteration
      self._update_pheromones(ant_tours)

      # Track performance metrics
      if self.track_performance:
        self._save_performance_snapshot(
            iteration, iteration_start_time, ant_tours)

    total_time = time.time() - total_start_time
    print(f"\nFinished ACO. Best tour length: {self.best_tour_length:.2f}")
    print(f"Best tour: {self.best_tour}")
    print(f"Total optimization time: {total_time:.2f} seconds")

    # Generate visualizations if enabled
    if self.visualization and city_count and ant_count:
      self.visualize_results(city_count, ant_count)

    # Save solution data if tracking is enabled
    if self.track_performance and city_count and ant_count:
      self.save_solution(city_count, ant_count)

    return self.best_tour, self.best_tour_length


# --- 3. Main Execution Block ---
def parse_command_line_arguments():
  """
  Parse command line arguments to allow easy parameter tuning and experimentation.
  """
  parser = argparse.ArgumentParser(
      description='Ant Colony Optimization for TSP')

  # City configuration
  parser.add_argument('--cities10', type=int, default=10,
                      help='Number of cities for the first problem (default: 10)')
  parser.add_argument('--cities20', type=int, default=20,
                      help='Number of cities for the second problem (default: 20)')

  # Algorithm parameters
  parser.add_argument('--ants', type=str, default='1,5,10,20',
                      help='Comma-separated list of ant counts to try (default: 1,5,10,20)')
  parser.add_argument('--iterations', type=int, default=50,
                      help='Number of iterations (default: 50)')
  parser.add_argument('--rho', type=float, default=0.3,
                      help='Pheromone evaporation rate (default: 0.3)')
  parser.add_argument('--q', type=float, default=100.0,
                      help='Pheromone deposit factor (default: 100.0)')
  parser.add_argument('--alpha', type=float, default=1.0,
                      help='Pheromone influence factor (default: 1.0)')
  parser.add_argument('--beta', type=float, default=2.0,
                      help='Distance influence factor (default: 2.0)')
  parser.add_argument('--initial-pheromone', type=float, default=1.0,
                      help='Initial pheromone level (default: 1.0)')

  # Optimization options
  parser.add_argument('--local-search', action='store_true',
                      help='Enable 2-opt local search improvement')
  parser.add_argument('--no-visualization', action='store_true',
                      help='Disable visualization generation')
  parser.add_argument('--memory-efficient', action='store_true',
                      help='Run in memory-efficient mode (reduces tracking)')
  parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')

  # Parse arguments
  args = parser.parse_args()

  # Convert ant counts from string to list of integers
  try:
    args.ant_counts = [int(x.strip()) for x in args.ants.split(',')]
  except ValueError:
    print("Error: Ant counts must be comma-separated integers")
    args.ant_counts = [1, 5, 10, 20]  # Fallback to defaults

  # Set random seed if provided
  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed set to {args.seed}")

  return args


# --- 4. Experiment and Visualization ---
def run_experiments():
  """
  Run a series of ACO experiments with different configurations.
  Supports command line argument overrides for easy experimentation.
  """
  # Parse command line arguments
  args = parse_command_line_arguments()

  # Use argument values for parameters
  NUM_CITIES_10 = args.cities10
  NUM_CITIES_20 = args.cities20
  RHO = args.rho
  Q_VAL = args.q
  N_ITERATIONS = args.iterations
  INITIAL_PHEROMONE = args.initial_pheromone
  ALPHA = args.alpha
  BETA = args.beta
  ENABLE_LOCAL_SEARCH = args.local_search
  VISUALIZATION = not args.no_visualization
  MEMORY_EFFICIENT = args.memory_efficient

  # Handle distance matrices
  global distances_10_cities, distances_20_cities
  if 'NUM_CITIES_10' not in globals() or 'distances_10_cities' not in globals() or 'distances_20_cities' not in globals():
    print("Generating new distance matrices...")
    _, _, distances_10_cities, distances_20_cities = generate_and_print_distances()

  # Setup city configurations
  city_configs = [
      {
          "name": f"{NUM_CITIES_10} Cities",
          "num_cities": NUM_CITIES_10,
          "distances": distances_10_cities,
      },
      {
          "name": f"{NUM_CITIES_20} Cities",
          "num_cities": NUM_CITIES_20,
          "distances": distances_20_cities,
      },
  ]

  # Use parsed ant agent counts
  ant_agent_counts = args.ant_counts

  # Store results for comparison
  results = {}
  execution_times = {}

  # Header for the output
  print(f"\n{'='*20} EXPERIMENT CONFIGURATION {'='*20}")
  print(f"City sizes: {NUM_CITIES_10} and {NUM_CITIES_20}")
  print(f"Ant counts: {ant_agent_counts}")
  print(f"Iterations: {N_ITERATIONS}")
  print(f"Parameters: rho={RHO}, Q={Q_VAL}, alpha={ALPHA}, beta={BETA}")
  print(f"Local search: {'Enabled' if ENABLE_LOCAL_SEARCH else 'Disabled'}")
  print(f"Visualization: {'Enabled' if VISUALIZATION else 'Disabled'}")
  print(
      f"Memory efficient mode: {'Enabled' if MEMORY_EFFICIENT else 'Disabled'}")
  print('='*60)

  # Run experiments for each city configuration
  for config in city_configs:
    print(f"\n{'='*10} EXPERIMENTS FOR {config['name']} {'='*10}")
    current_distances = config["distances"]
    results[config["name"]] = {}
    execution_times[config["name"]] = {}

    # For each ant count
    for n_ants in ant_agent_counts:
      # Adjust ant count if necessary to avoid exceeding city count
      if n_ants > config["num_cities"] and config["num_cities"] > 0:
        print(
            f"\nAdjusting number of ants from {n_ants} to {config['num_cities']} "
            f"as n_ants cannot exceed n_cities for the chosen start_city assignment logic."
        )
        current_n_ants = config["num_cities"]
      else:
        current_n_ants = n_ants

      # Skip if no cities (e.g. bad generation)
      if config["num_cities"] == 0:
        print(
            f"\nSkipping {n_ants} ants for {config['name']} due to 0 cities.")
        continue

      print(
          f"\n--- Running with {current_n_ants} Ant(s) for {config['name']} ---")

      # Record start time
      run_start_time = time.time()

      # Initialize the ACO optimizer with enhanced parameters
      aco = AntColonyOptimizer(
          distances=current_distances,
          n_ants=current_n_ants,
          n_iterations=N_ITERATIONS,
          rho=RHO,
          q_val=Q_VAL,
          initial_pheromone=INITIAL_PHEROMONE,
          alpha=ALPHA,
          beta=BETA,
          enable_local_search=ENABLE_LOCAL_SEARCH,
          track_performance=not MEMORY_EFFICIENT,
          visualization=VISUALIZATION,
      )

      # Extract city count from configuration name
      city_count = config["num_cities"]

      # Run the optimization
      best_tour, best_length = aco.run(
          city_count=city_count, ant_count=current_n_ants)

      # Record end time and calculate total run time
      run_time = time.time() - run_start_time

      # Store results
      results[config["name"]][f"{current_n_ants} Ants"] = {
          "tour": best_tour,
          "length": best_length,
          "runtime": run_time,
      }
      execution_times[config["name"]][f"{current_n_ants} Ants"] = run_time

  # Print final summary of results
  print("\n\n" + "=" * 20 + " FINAL RESULTS SUMMARY " + "=" * 20)
  for city_set_name, ant_results in results.items():
    print(f"\nResults for {city_set_name}:")

    # Sort results by tour length (best first)
    sorted_results = sorted(ant_results.items(), key=lambda x: x[1]["length"])

    for ant_config_name, result_data in sorted_results:
      print(f"  {ant_config_name}:")
      print(f"    Best Tour Length: {result_data['length']:.2f}")
      print(f"    Runtime: {result_data['runtime']:.2f} seconds")
      print(f"    Best Tour Path: {result_data['tour']}")
    print("-" * 20)

    # Find the best configuration for this city set
    best_config = min(ant_results.items(), key=lambda x: x[1]["length"])
    print(
        f"Best configuration for {city_set_name}: {best_config[0]} with length {best_config[1]['length']:.2f}")
    print(
        f"Improvement over worst: {(max(ant_results.values(), key=lambda x: x['length'])['length'] / best_config[1]['length'] - 1) * 100:.2f}%")

  # Compare city sizes
  print("\n" + "=" * 20 + " CITY SIZE COMPARISON " + "=" * 20)
  # Compare best results between city sizes
  city_10_best = min(
      results[f"{NUM_CITIES_10} Cities"].values(), key=lambda x: x["length"])
  city_20_best = min(
      results[f"{NUM_CITIES_20} Cities"].values(), key=lambda x: x["length"])

  print(
      f"Best result for {NUM_CITIES_10} cities: {city_10_best['length']:.2f}")
  print(
      f"Best result for {NUM_CITIES_20} cities: {city_20_best['length']:.2f}")
  print(
      f"Scaling factor: {city_20_best['length']/city_10_best['length']:.2f}x")

  # Compare runtimes
  print("\n" + "=" * 20 + " RUNTIME ANALYSIS " + "=" * 20)
  for city_set_name, ant_results in execution_times.items():
    avg_runtime = sum(ant_results.values()) / len(ant_results)
    print(f"Average runtime for {city_set_name}: {avg_runtime:.2f} seconds")

    # Runtime by ant count
    print(f"Runtime breakdown by ant count:")
    for ant_config, runtime in sorted(ant_results.items(), key=lambda x: int(x[0].split()[0])):
      print(f"  {ant_config}: {runtime:.2f} seconds")

  # Create a directory for experiment summaries
  os.makedirs("experiment_summaries", exist_ok=True)

  # Save experiment summary
  summary_filename = f"experiment_summaries/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
  with open(summary_filename, 'w') as f:
    summary = {
        "parameters": {
            "rho": RHO,
            "q_val": Q_VAL,
            "n_iterations": N_ITERATIONS,
            "alpha": ALPHA,
            "beta": BETA,
            "initial_pheromone": INITIAL_PHEROMONE,
            "local_search": ENABLE_LOCAL_SEARCH,
        },
        "city_configs": [
            {"name": config["name"], "num_cities": config["num_cities"]} for config in city_configs
        ],
        "ant_counts": ant_agent_counts,
        "results": {city: {ant: {"length": data["length"], "runtime": data["runtime"]}
                           for ant, data in ant_results.items()}
                    for city, ant_results in results.items()},
    }
    json.dump(summary, f, indent=2)

  print(f"\nExperiment summary saved to {summary_filename}")


# --- 5. Main function and Printing in Output File ---
def generate_and_print_distances():
  print("--- Generating Distances ---")
  NUM_CITIES_10 = 10
  distances_10_cities = generate_distance_matrix(NUM_CITIES_10)
  print(f"\nDistance Matrix for {NUM_CITIES_10} cities:")
  print(distances_10_cities)

  NUM_CITIES_20 = 20
  distances_20_cities = generate_distance_matrix(NUM_CITIES_20)
  print(f"\nDistance Matrix for {NUM_CITIES_20} cities:")
  print(distances_20_cities)
  print("-" * 30)
  return NUM_CITIES_10, NUM_CITIES_20, distances_10_cities, distances_20_cities
# Allow the script to be run directly or imported
if __name__ == "__main__":
  # If the script was imported, just define functions
  # If it was run directly, execute experiments
  # Generate distances before running experiments
  NUM_CITIES_10, NUM_CITIES_20, distances_10_cities, distances_20_cities = generate_and_print_distances()

  # Run the experiments with all the enhancements
  run_experiments()


# Close the output file
sys.stdout.close()
# Reset stdout to console
sys.stdout = sys.__stdout__
print(f"Results have been saved to {output_filename}")

# you can run it from the command line with different parameters
# eg. python3 TSP_Ant_Colony_Optimization_Agent.py --iterations 10 --alpha 1.5 --beta 3.0 --local-search