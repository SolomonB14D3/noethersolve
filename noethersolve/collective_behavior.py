"""
Collective Behavior Module — Maps biological swarm intelligence to optimization algorithms.

This module provides computational models for collective biological behaviors
and their algorithmic analogues:
- Slime mold (Physarum) network optimization
- Ant colony pheromone-based routing
- Swarm aggregation and consensus

Core insight: Distributed biological systems solve optimization problems
(shortest paths, resource allocation, consensus) using only local rules
and stigmergic communication. These solutions often match or exceed
centralized algorithms.

Tools provided:
- slime_mold_optimization() — Physarum-inspired network solver
- ant_pheromone_routing() — ACO routing algorithm
- swarm_consensus() — Distributed agreement protocol
- flock_formation() — Reynolds boids-style flocking
- bacterial_quorum_sensing() — Threshold-based collective decision
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set
from enum import Enum
import math
import random


class OptimizationType(Enum):
    """Types of optimization problems."""
    SHORTEST_PATH = "shortest_path"
    MINIMUM_SPANNING_TREE = "mst"
    TRAVELING_SALESMAN = "tsp"
    RESOURCE_ALLOCATION = "resource"
    CONSENSUS = "consensus"


@dataclass
class NetworkEdge:
    """Edge in a network graph."""
    source: int
    target: int
    distance: float
    conductance: float = 1.0  # Physarum tube thickness
    pheromone: float = 1.0  # Ant pheromone level
    flow: float = 0.0

    def __hash__(self):
        return hash((self.source, self.target))


@dataclass
class SlimeMoldResult:
    """Result of slime mold network optimization."""
    optimal_edges: List[Tuple[int, int]]
    total_length: float
    efficiency: float  # Length / theoretical minimum
    iterations: int
    tube_conductances: Dict[Tuple[int, int], float]
    converged: bool
    biological_properties: Dict[str, any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Slime Mold Optimization Result:",
            f"  Optimal edges: {len(self.optimal_edges)}",
            f"  Total length: {self.total_length:.2f}",
            f"  Efficiency: {self.efficiency:.2%}",
            f"  Iterations: {self.iterations}",
            f"  Converged: {self.converged}",
        ]
        if self.biological_properties:
            lines.append("  Biological properties:")
            for k, v in self.biological_properties.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


@dataclass
class AntColonyResult:
    """Result of ant colony optimization."""
    best_path: List[int]
    path_length: float
    iterations: int
    pheromone_matrix: Optional[List[List[float]]] = None
    convergence_history: List[float] = field(default_factory=list)
    biological_properties: Dict[str, any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Ant Colony Optimization Result:",
            f"  Best path: {' -> '.join(map(str, self.best_path))}",
            f"  Path length: {self.path_length:.2f}",
            f"  Iterations: {self.iterations}",
        ]
        if self.convergence_history:
            lines.append(f"  Convergence: {self.convergence_history[0]:.2f} -> {self.convergence_history[-1]:.2f}")
        return "\n".join(lines)


@dataclass
class SwarmState:
    """State of a swarm agent."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    heading: float  # radians
    id: int


def slime_mold_optimization(
    nodes: List[Tuple[float, float]],
    food_sources: List[int],
    iterations: int = 100,
    mu: float = 1.0,  # Tube growth rate
    gamma: float = 1.8,  # Flow-conductance exponent (Physarum uses ~1.8)
    decay_rate: float = 0.1,
    threshold: float = 0.01
) -> SlimeMoldResult:
    """
    Solve network optimization using Physarum polycephalum (slime mold) algorithm.

    Physarum naturally finds shortest paths and efficient networks by:
    1. Growing tubes between food sources
    2. Reinforcing tubes with high flow
    3. Shrinking tubes with low flow

    The dynamics follow: dD/dt = f(|Q|) - gamma * D
    Where D = conductance, Q = flow, f(|Q|) = |Q|^mu

    This is mathematically equivalent to finding the Steiner tree
    (minimum length network connecting specified nodes).

    Args:
        nodes: List of (x, y) positions
        food_sources: Indices of nodes that must be connected
        iterations: Maximum iterations
        mu: Flow-conductance coupling exponent
        gamma: Conductance growth exponent
        decay_rate: Rate of tube decay
        threshold: Convergence threshold

    Returns:
        SlimeMoldResult with optimal network
    """
    n = len(nodes)
    if n < 2:
        return SlimeMoldResult(
            optimal_edges=[],
            total_length=0.0,
            efficiency=1.0,
            iterations=0,
            tube_conductances={},
            converged=True
        )

    # Initialize complete graph with distances
    edges: Dict[Tuple[int, int], NetworkEdge] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = nodes[i][0] - nodes[j][0]
            dy = nodes[i][1] - nodes[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            edges[(i, j)] = NetworkEdge(i, j, dist, conductance=1.0)
            edges[(j, i)] = NetworkEdge(j, i, dist, conductance=1.0)

    # Food source pressures (boundary conditions)
    pressures = [0.0] * n
    if len(food_sources) >= 2:
        pressures[food_sources[0]] = 1.0
        pressures[food_sources[-1]] = 0.0

    converged = False
    for iteration in range(iterations):
        # Solve for flows using Kirchhoff's laws
        # Simplified: flow proportional to pressure difference * conductance
        for (i, j), edge in edges.items():
            dp = pressures[i] - pressures[j]
            edge.flow = edge.conductance * dp / edge.distance

        # Update pressures (Gauss-Seidel iteration)
        for _ in range(10):
            for i in range(n):
                if i in food_sources[:1] or i in food_sources[-1:]:
                    continue
                total_conductance = 0.0
                weighted_pressure = 0.0
                for j in range(n):
                    if i != j and (i, j) in edges:
                        D = edges[(i, j)].conductance / edges[(i, j)].distance
                        total_conductance += D
                        weighted_pressure += D * pressures[j]
                if total_conductance > 0:
                    pressures[i] = weighted_pressure / total_conductance

        # Update conductances: dD/dt = |Q|^mu - decay * D
        max_change = 0.0
        for edge in edges.values():
            flow_term = abs(edge.flow) ** mu
            decay_term = decay_rate * edge.conductance
            dD = flow_term - decay_term
            edge.conductance = max(0.001, edge.conductance + 0.1 * dD)
            max_change = max(max_change, abs(dD))

        if max_change < threshold:
            converged = True
            break

    # Extract optimal edges (high conductance)
    conductance_threshold = max(e.conductance for e in edges.values()) * 0.1
    optimal_edges = []
    seen = set()
    for (i, j), edge in edges.items():
        if edge.conductance > conductance_threshold and (j, i) not in seen:
            optimal_edges.append((i, j))
            seen.add((i, j))

    # Calculate total length
    total_length = sum(edges[(i, j)].distance for i, j in optimal_edges)

    # Calculate theoretical minimum (MST)
    mst_edges = _minimum_spanning_tree(nodes, food_sources)
    mst_length = sum(
        math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
        for i, j in mst_edges
    )
    efficiency = mst_length / total_length if total_length > 0 else 1.0

    # Extract final conductances
    tube_conductances = {
        (i, j): edges[(i, j)].conductance
        for i, j in optimal_edges
    }

    return SlimeMoldResult(
        optimal_edges=optimal_edges,
        total_length=total_length,
        efficiency=efficiency,
        iterations=iteration + 1,
        tube_conductances=tube_conductances,
        converged=converged,
        biological_properties={
            "mu": mu,
            "gamma": gamma,
            "decay_rate": decay_rate,
            "matches_tokyo_rail": efficiency > 0.9  # Physarum famously matched Tokyo rail
        }
    )


def _minimum_spanning_tree(nodes: List[Tuple[float, float]], required: List[int]) -> List[Tuple[int, int]]:
    """Compute MST using Prim's algorithm."""
    if len(required) < 2:
        return []

    n = len(nodes)
    in_tree = {required[0]}
    edges = []

    while len(in_tree) < len(required):
        best_edge = None
        best_dist = float('inf')

        for i in in_tree:
            for j in required:
                if j not in in_tree:
                    dx = nodes[i][0] - nodes[j][0]
                    dy = nodes[i][1] - nodes[j][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_edge = (i, j)

        if best_edge:
            edges.append(best_edge)
            in_tree.add(best_edge[1])
        else:
            break

    return edges


def ant_pheromone_routing(
    distances: List[List[float]],
    n_ants: int = 20,
    iterations: int = 100,
    alpha: float = 1.0,  # Pheromone importance
    beta: float = 2.0,   # Distance importance
    rho: float = 0.5,    # Evaporation rate
    Q: float = 100.0,    # Pheromone deposit amount
    seed: Optional[int] = None
) -> AntColonyResult:
    """
    Solve TSP using Ant Colony Optimization (ACO).

    Inspired by how ants find shortest paths to food:
    1. Ants leave pheromone on paths
    2. Other ants prefer paths with more pheromone
    3. Pheromone evaporates over time
    4. Shorter paths get more pheromone (more round trips)

    This creates positive feedback that converges to good solutions.

    Args:
        distances: Distance matrix between cities
        n_ants: Number of ants per iteration
        iterations: Maximum iterations
        alpha: Pheromone influence exponent
        beta: Heuristic (1/distance) influence exponent
        rho: Pheromone evaporation rate (0-1)
        Q: Pheromone deposit constant
        seed: Random seed for reproducibility

    Returns:
        AntColonyResult with best path found
    """
    if seed is not None:
        random.seed(seed)

    n = len(distances)
    if n < 2:
        return AntColonyResult(
            best_path=[0] if n == 1 else [],
            path_length=0.0,
            iterations=0
        )

    # Initialize pheromone matrix
    pheromone = [[1.0 / n for _ in range(n)] for _ in range(n)]

    # Precompute heuristic (1/distance)
    heuristic = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and distances[i][j] > 0:
                heuristic[i][j] = 1.0 / distances[i][j]

    best_path = list(range(n))
    best_length = _path_length(best_path, distances)
    convergence_history = []

    for iteration in range(iterations):
        all_paths = []
        all_lengths = []

        for ant in range(n_ants):
            # Construct solution
            path = _construct_ant_path(n, pheromone, heuristic, alpha, beta)
            length = _path_length(path, distances)
            all_paths.append(path)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_path = path.copy()

        # Evaporate pheromone
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - rho)

        # Deposit pheromone
        for path, length in zip(all_paths, all_lengths):
            deposit = Q / length if length > 0 else 0
            for i in range(len(path)):
                j = (i + 1) % len(path)
                pheromone[path[i]][path[j]] += deposit
                pheromone[path[j]][path[i]] += deposit

        convergence_history.append(best_length)

    return AntColonyResult(
        best_path=best_path,
        path_length=best_length,
        iterations=iterations,
        pheromone_matrix=pheromone,
        convergence_history=convergence_history,
        biological_properties={
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "exploitation_vs_exploration": alpha / (alpha + beta),
            "stigmergy": "Indirect coordination through environment modification"
        }
    )


def _construct_ant_path(
    n: int,
    pheromone: List[List[float]],
    heuristic: List[List[float]],
    alpha: float,
    beta: float
) -> List[int]:
    """Construct a path for one ant using probabilistic selection."""
    path = [random.randint(0, n - 1)]
    unvisited = set(range(n)) - {path[0]}

    while unvisited:
        current = path[-1]

        # Calculate probabilities for each unvisited city
        probs = []
        for j in unvisited:
            tau = pheromone[current][j] ** alpha
            eta = heuristic[current][j] ** beta
            probs.append((j, tau * eta))

        total = sum(p for _, p in probs)
        if total == 0:
            # Random selection if all probabilities are 0
            next_city = random.choice(list(unvisited))
        else:
            # Roulette wheel selection
            r = random.random() * total
            cumsum = 0
            next_city = probs[0][0]
            for j, p in probs:
                cumsum += p
                if cumsum >= r:
                    next_city = j
                    break

        path.append(next_city)
        unvisited.remove(next_city)

    return path


def _path_length(path: List[int], distances: List[List[float]]) -> float:
    """Calculate total path length including return to start."""
    total = 0.0
    for i in range(len(path)):
        j = (i + 1) % len(path)
        total += distances[path[i]][path[j]]
    return total


def swarm_consensus(
    initial_opinions: List[float],
    adjacency: List[List[float]],
    iterations: int = 100,
    convergence_threshold: float = 0.001
) -> Dict[str, any]:
    """
    Simulate distributed consensus like bee swarms selecting nest sites.

    Agents update opinions based on neighbors:
    x_i(t+1) = sum(w_ij * x_j(t)) / sum(w_ij)

    This is DeGroot learning / consensus dynamics.

    Args:
        initial_opinions: Starting opinions for each agent
        adjacency: Weighted adjacency matrix (influence weights)
        iterations: Maximum iterations
        convergence_threshold: Stop when max change < threshold

    Returns:
        Dictionary with consensus results
    """
    n = len(initial_opinions)
    opinions = initial_opinions.copy()
    history = [opinions.copy()]

    for iteration in range(iterations):
        new_opinions = []
        for i in range(n):
            total_weight = sum(adjacency[i])
            if total_weight > 0:
                new_opinion = sum(adjacency[i][j] * opinions[j] for j in range(n)) / total_weight
            else:
                new_opinion = opinions[i]
            new_opinions.append(new_opinion)

        max_change = max(abs(new_opinions[i] - opinions[i]) for i in range(n))
        opinions = new_opinions
        history.append(opinions.copy())

        if max_change < convergence_threshold:
            break

    # Check if consensus reached
    opinion_variance = sum((o - sum(opinions)/n)**2 for o in opinions) / n
    consensus_reached = opinion_variance < convergence_threshold

    return {
        "final_opinions": opinions,
        "consensus_value": sum(opinions) / n,
        "consensus_reached": consensus_reached,
        "iterations": iteration + 1,
        "final_variance": opinion_variance,
        "history": history,
        "biological_analogue": "Bee swarm nest selection, fish schooling direction"
    }


def flock_formation(
    positions: List[Tuple[float, float]],
    velocities: List[Tuple[float, float]],
    n_steps: int = 100,
    separation_weight: float = 1.5,
    alignment_weight: float = 1.0,
    cohesion_weight: float = 1.0,
    perception_radius: float = 50.0,
    max_speed: float = 5.0,
    max_force: float = 0.5
) -> Dict[str, any]:
    """
    Simulate flocking behavior using Reynolds' boids rules.

    Three simple rules create complex emergent flocking:
    1. Separation: Avoid crowding neighbors
    2. Alignment: Steer towards average heading of neighbors
    3. Cohesion: Steer towards average position of neighbors

    Args:
        positions: Initial (x, y) positions
        velocities: Initial (vx, vy) velocities
        n_steps: Simulation steps
        separation_weight: Weight for separation rule
        alignment_weight: Weight for alignment rule
        cohesion_weight: Weight for cohesion rule
        perception_radius: How far agents can see
        max_speed: Maximum agent speed
        max_force: Maximum steering force

    Returns:
        Dictionary with flock dynamics
    """
    n = len(positions)
    if n == 0:
        return {"valid": False, "reason": "No agents"}

    agents = [
        SwarmState(
            position=positions[i],
            velocity=velocities[i],
            heading=math.atan2(velocities[i][1], velocities[i][0]),
            id=i
        )
        for i in range(n)
    ]

    position_history = [[(a.position[0], a.position[1]) for a in agents]]
    order_parameter_history = []

    for step in range(n_steps):
        new_agents = []

        for agent in agents:
            # Find neighbors
            neighbors = []
            for other in agents:
                if other.id != agent.id:
                    dx = other.position[0] - agent.position[0]
                    dy = other.position[1] - agent.position[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < perception_radius:
                        neighbors.append((other, dist))

            # Calculate steering forces
            sep_force = (0.0, 0.0)
            ali_force = (0.0, 0.0)
            coh_force = (0.0, 0.0)

            if neighbors:
                # Separation
                for other, dist in neighbors:
                    if dist > 0:
                        dx = agent.position[0] - other.position[0]
                        dy = agent.position[1] - other.position[1]
                        sep_force = (sep_force[0] + dx/dist**2, sep_force[1] + dy/dist**2)

                # Alignment
                avg_vx = sum(o.velocity[0] for o, _ in neighbors) / len(neighbors)
                avg_vy = sum(o.velocity[1] for o, _ in neighbors) / len(neighbors)
                ali_force = (avg_vx - agent.velocity[0], avg_vy - agent.velocity[1])

                # Cohesion
                avg_x = sum(o.position[0] for o, _ in neighbors) / len(neighbors)
                avg_y = sum(o.position[1] for o, _ in neighbors) / len(neighbors)
                coh_force = (avg_x - agent.position[0], avg_y - agent.position[1])

            # Combine forces
            force_x = (separation_weight * sep_force[0] +
                      alignment_weight * ali_force[0] +
                      cohesion_weight * coh_force[0])
            force_y = (separation_weight * sep_force[1] +
                      alignment_weight * ali_force[1] +
                      cohesion_weight * coh_force[1])

            # Limit force
            force_mag = math.sqrt(force_x**2 + force_y**2)
            if force_mag > max_force:
                force_x *= max_force / force_mag
                force_y *= max_force / force_mag

            # Update velocity
            new_vx = agent.velocity[0] + force_x
            new_vy = agent.velocity[1] + force_y

            # Limit speed
            speed = math.sqrt(new_vx**2 + new_vy**2)
            if speed > max_speed:
                new_vx *= max_speed / speed
                new_vy *= max_speed / speed

            # Update position
            new_x = agent.position[0] + new_vx
            new_y = agent.position[1] + new_vy

            new_agents.append(SwarmState(
                position=(new_x, new_y),
                velocity=(new_vx, new_vy),
                heading=math.atan2(new_vy, new_vx),
                id=agent.id
            ))

        agents = new_agents
        position_history.append([(a.position[0], a.position[1]) for a in agents])

        # Compute order parameter (alignment measure)
        if n > 0:
            avg_heading_x = sum(math.cos(a.heading) for a in agents) / n
            avg_heading_y = sum(math.sin(a.heading) for a in agents) / n
            order = math.sqrt(avg_heading_x**2 + avg_heading_y**2)
            order_parameter_history.append(order)

    return {
        "valid": True,
        "final_positions": [(a.position[0], a.position[1]) for a in agents],
        "final_velocities": [(a.velocity[0], a.velocity[1]) for a in agents],
        "order_parameter": order_parameter_history[-1] if order_parameter_history else 0.0,
        "order_history": order_parameter_history,
        "position_history": position_history,
        "steps": n_steps,
        "biological_analogue": "Bird flocking, fish schooling, locust swarms"
    }


def bacterial_quorum_sensing(
    n_bacteria: int,
    signal_production_rate: float = 0.1,
    signal_threshold: float = 1.0,
    degradation_rate: float = 0.05,
    diffusion_rate: float = 0.8,
    iterations: int = 100,
    seed: Optional[int] = None
) -> Dict[str, any]:
    """
    Simulate bacterial quorum sensing — density-dependent gene activation.

    Bacteria produce signaling molecules (autoinducers). When local
    concentration exceeds a threshold, genes are activated. This allows
    bacteria to coordinate behavior based on population density.

    This is analogous to:
    - Threshold-based consensus in distributed systems
    - Byzantine fault tolerance (need >2/3 agreement)
    - Neural population coding (threshold firing)

    Args:
        n_bacteria: Number of bacteria
        signal_production_rate: Autoinducer production per bacterium
        signal_threshold: Concentration threshold for gene activation
        degradation_rate: Signal degradation rate
        diffusion_rate: Signal mixing rate (0-1)
        iterations: Simulation steps
        seed: Random seed

    Returns:
        Dictionary with quorum sensing dynamics
    """
    if seed is not None:
        random.seed(seed)

    # Initialize bacteria with random positions in unit square
    positions = [(random.random(), random.random()) for _ in range(n_bacteria)]
    activated = [False] * n_bacteria
    local_signals = [0.0] * n_bacteria

    global_signal = 0.0
    activation_history = []
    signal_history = []

    for iteration in range(iterations):
        # Each bacterium produces signal
        global_signal += n_bacteria * signal_production_rate

        # Signal degrades
        global_signal *= (1 - degradation_rate)

        # Update local signals (mix of local production + diffusion from global)
        for i in range(n_bacteria):
            local_signals[i] = (1 - diffusion_rate) * local_signals[i] + \
                              diffusion_rate * (global_signal / n_bacteria) + \
                              signal_production_rate

            # Check threshold activation
            if local_signals[i] >= signal_threshold and not activated[i]:
                activated[i] = True

        n_activated = sum(activated)
        activation_history.append(n_activated / n_bacteria)
        signal_history.append(global_signal / n_bacteria)

    # Determine if quorum was reached
    quorum_reached = sum(activated) / n_bacteria > 0.5
    activation_time = None
    for i, frac in enumerate(activation_history):
        if frac > 0.5:
            activation_time = i
            break

    return {
        "quorum_reached": quorum_reached,
        "final_activation_fraction": sum(activated) / n_bacteria,
        "activation_time": activation_time,
        "final_signal_concentration": global_signal / n_bacteria,
        "activation_history": activation_history,
        "signal_history": signal_history,
        "iterations": iterations,
        "biological_properties": {
            "threshold": signal_threshold,
            "cooperativity": "All-or-none activation above threshold",
            "real_world_examples": [
                "Vibrio fischeri bioluminescence",
                "Pseudomonas aeruginosa biofilm formation",
                "Staphylococcus aureus virulence"
            ]
        },
        "ai_analogue": "Byzantine fault tolerance, threshold cryptography, voting systems"
    }


# MCP Tool wrappers

def slime_mold_network_solver(
    node_positions: List[Tuple[float, float]],
    food_source_indices: List[int]
) -> str:
    """
    Find optimal network connecting food sources using slime mold algorithm.

    Physarum polycephalum naturally computes Steiner trees and has been shown
    to match or exceed human-designed networks (e.g., Tokyo rail system).
    """
    result = slime_mold_optimization(node_positions, food_source_indices)
    return str(result)


def ant_colony_tsp_solver(
    distance_matrix: List[List[float]],
    n_ants: int = 20,
    iterations: int = 100
) -> str:
    """
    Solve traveling salesman problem using ant colony optimization.

    Based on how ants find shortest paths through pheromone trail laying
    and following. Demonstrates stigmergic coordination.
    """
    result = ant_pheromone_routing(distance_matrix, n_ants, iterations)
    return str(result)
