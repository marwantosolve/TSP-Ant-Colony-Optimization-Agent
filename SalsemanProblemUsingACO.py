import numpy as np
import random


# Distance Generation
def generate_distance_matrix(num_cities, min_dist=3, max_dist=50):
    if num_cities <= 0:
        return np.array([])
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities): # to loop through first row
        # to loop through columns 
        # we start from i + 1 to make sure all cities are connected but not same city
        # the diagonal is 0
        for j in range(i + 1, num_cities):
            dist = random.randint(min_dist, max_dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


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


# ACO Algorithm Implementation
class AntColonyOptimizer:
    def __init__(
        self,
        distances,
        n_ants,
        n_iterations,
        rho,
        q_val,
        initial_pheromone=1.0,
    ):
        """
        distances: 2D numpy array of distances between cities
        n_ants: Number of ants
        n_iterations: Number of iterations
        rho: Pheromone evaporation rate
        q_val: Pheromone deposit constant (Q in literature, often 1 or related to tour length)
        initial_pheromone: Initial pheromone level on all paths
        """
        self.distances = distances
        self.n_cities = distances.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.rho = rho
        self.q_val = q_val
        self.initial_pheromone = initial_pheromone

        # Heuristic information (eta = 1/distance)
        # Add a small epsilon to avoid division by zero 
        epsilon = 1e-10
        self.eta = 1.0 / (distances + epsilon)
        np.fill_diagonal(self.eta, 0)  # No heuristic value for staying in the same city

        # Pheromone matrix, initialized
        # make all cities phermone with initial phermone value
        self.pheromones = np.full(
            (self.n_cities, self.n_cities), self.initial_pheromone
        )
        # the diagonal in city matrix is 0 so it has no phermone too as it's same city
        np.fill_diagonal(
            self.pheromones, 0
        )

        self.best_tour = None
        self.best_tour_length = float("inf")

    def _select_next_city(self, current_city, visited_cities, ant_pheromones):
        probabilities = []
        unvisited_cities = []

        # Calculate numerators for probability calculation
        # P(i,j) = (τ[i][j] * η[i][j])
        for city_idx in range(self.n_cities):
            if not visited_cities[city_idx]:
                unvisited_cities.append(city_idx)
                tau_ij = ant_pheromones[current_city, city_idx]
                eta_ij = self.eta[current_city, city_idx]
                prob_numerator = (tau_ij) * (eta_ij)
                probabilities.append(prob_numerator)

        # If all cities are visited, return None
        if not unvisited_cities:
            return None

        # Normalize probabilities
        sum_probs = sum(probabilities)
        if sum_probs == 0:
            # If all probabilities are zero, select a random unvisited city
            if unvisited_cities:
                return random.choice(unvisited_cities)
            else:
                return None

        probabilities = [p / sum_probs for p in probabilities]

        # Select next city using roulette wheel selection
        next_city = random.choices(unvisited_cities, weights=probabilities, k=1)[0]
        return next_city

    def _construct_tour(self, start_city, ant_pheromones):
        tour = [start_city]
        current_city = start_city
        visited_cities = np.zeros(self.n_cities, dtype=bool)
        visited_cities[start_city] = True
        tour_length = 0.0

        for _ in range(self.n_cities - 1):
            next_city = self._select_next_city(
                current_city, visited_cities, ant_pheromones
            )
            if next_city is None:
                break
            tour.append(next_city)
            visited_cities[next_city] = True
            tour_length += self.distances[current_city, next_city]
            current_city = next_city

        # Complete the loop: back to start city
        if tour:  # Check if tour was actually built
            tour_length += self.distances[current_city, start_city]

        return tour, tour_length

    def _update_pheromones(self, ant_tours):
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

    def run(self):
        if self.n_cities == 0:
            print("No cities to run ACO on.")
            return [], float("inf")

        print(
            f"\nRunning ACO: {self.n_ants} ants, {self.n_iterations} iterations, "
            f"rho={self.rho}, Q={self.q_val}"
        )

        for iteration in range(self.n_iterations):
            ant_tours = []  # Store (tour, tour_length) for all ants in this iteration

            # Each ant constructs a tour
            for ant_id in range(self.n_ants):
                # Randomly select a starting city for each ant
                start_city = ant_id % self.n_cities

                # Ants should use the global pheromone trails for decisions
                tour, tour_length = self._construct_tour(start_city, self.pheromones)
                ant_tours.append((tour, tour_length))

                if tour_length < self.best_tour_length:
                    self.best_tour_length = tour_length
                    self.best_tour = tour

            # Update pheromones based on all tours from this iteration
            self._update_pheromones(ant_tours)

            if (iteration + 1) % 10 == 0 or iteration == 0:  # Print progress
                print(
                    f"Iteration {iteration + 1}/{self.n_iterations}: "
                    f"Best tour length so far: {self.best_tour_length:.2f}"
                )

        print(f"Finished ACO. Best tour length: {self.best_tour_length:.2f}")
        print(f"Best tour: {self.best_tour}")
        return self.best_tour, self.best_tour_length



# ACO Parameters
RHO = 0.3  # Pheromone evaporation rate
Q_VAL = 100.0  # Pheromone deposit factor
N_ITERATIONS = 50
INITIAL_PHEROMONE = 1.0

city_configs = [
    {
        "name": "10 Cities",
        "num_cities": NUM_CITIES_10,
        "distances": distances_10_cities,
    },
    {
        "name": "20 Cities",
        "num_cities": NUM_CITIES_20,
        "distances": distances_20_cities,
    },
]
ant_agent_counts = [1, 5, 10, 20]

results = {}

for config in city_configs:
    print(f"\n{'='*10} EXPERIMENTS FOR {config['name']} {'='*10}")
    current_distances = config["distances"]
    results[config["name"]] = {}

    for n_ants in ant_agent_counts:
        if n_ants > config["num_cities"] and config["num_cities"] > 0:
            print(
                f"\nAdjusting number of ants from {n_ants} to {config['num_cities']} "
                f"as n_ants cannot exceed n_cities for the chosen start_city assignment logic."
            )
            current_n_ants = config["num_cities"]
        else:
            current_n_ants = n_ants
        if config["num_cities"] == 0:  # Skip if no cities (e.g. bad generation)
            print(f"\nSkipping {n_ants} ants for {config['name']} due to 0 cities.")
            continue

        print(
            f"\n--- Running with {current_n_ants} Ant(s) for {config['name']} ---"
        )
        aco = AntColonyOptimizer(
            distances=current_distances,
            n_ants=current_n_ants,
            n_iterations=N_ITERATIONS,
            rho=RHO,
            q_val=Q_VAL,
            initial_pheromone=INITIAL_PHEROMONE,
        )
        best_tour, best_length = aco.run()
        results[config["name"]][f"{current_n_ants} Ants"] = {
            "tour": best_tour,
            "length": best_length,
        }

print("\n\n" + "=" * 20 + " FINAL RESULTS SUMMARY " + "=" * 20)
for city_set_name, ant_results in results.items():
    print(f"\nResults for {city_set_name}:")
    for ant_config_name, result_data in ant_results.items():
        print(f"  {ant_config_name}:")
        print(f"    Best Tour Length: {result_data['length']:.2f}")
        print(f"    Best Tour Path: {result_data['tour']}")
    print("-" * 20)
