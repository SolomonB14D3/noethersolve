"""Quantitative sociology calculator — social networks, inequality, demographics.

Covers Gini coefficients, segregation indices, social mobility matrices,
demographic transitions, dependency ratios, network centrality, Granovetter
threshold cascades, homophily, Dunbar layers, and polarization measurement.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GiniResult:
    """Result of Gini coefficient calculation."""
    gini: float
    n: int
    mean_income: float
    median_income: float
    interpretation: str
    income_share_bottom_20: float  # fraction of total held by bottom 20%
    income_share_top_20: float    # fraction of total held by top 20%

    def __str__(self) -> str:
        lines = [
            "Gini Coefficient:",
            f"  Gini = {self.gini:.4f}",
            f"  Interpretation: {self.interpretation}",
            f"  N = {self.n}, Mean = {self.mean_income:.2f}, Median = {self.median_income:.2f}",
            f"  Bottom 20% share: {self.income_share_bottom_20:.1%}",
            f"  Top 20% share:    {self.income_share_top_20:.1%}",
        ]
        return "\n".join(lines)


@dataclass
class SegregationResult:
    """Result of dissimilarity index calculation."""
    dissimilarity_index: float
    n_areas: int
    total_group_a: float
    total_group_b: float
    interpretation: str
    most_segregated_area: int  # area index with highest contribution
    pct_would_move: float     # fraction of minority that would need to move

    def __str__(self) -> str:
        lines = [
            "Segregation (Dissimilarity Index):",
            f"  D = {self.dissimilarity_index:.4f}",
            f"  Interpretation: {self.interpretation}",
            f"  Areas: {self.n_areas}",
            f"  Group A total: {self.total_group_a:.0f}, Group B total: {self.total_group_b:.0f}",
            f"  Most segregated area: {self.most_segregated_area}",
            f"  % that would need to move for integration: {self.pct_would_move:.1%}",
        ]
        return "\n".join(lines)


@dataclass
class MobilityResult:
    """Result of social mobility calculation."""
    parent_quintile: int
    child_quintile: int
    transition_probability: float
    upward_probability: float    # P(child > parent quintile)
    downward_probability: float  # P(child < parent quintile)
    stay_probability: float      # P(child == parent quintile)
    ige: float                   # intergenerational elasticity
    absolute_mobility: float     # P(child income > parent income)
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Social Mobility:",
            f"  Parent quintile: {self.parent_quintile}, Child quintile: {self.child_quintile}",
            f"  P(this transition) = {self.transition_probability:.3f}",
            f"  P(upward) = {self.upward_probability:.3f}",
            f"  P(downward) = {self.downward_probability:.3f}",
            f"  P(stay) = {self.stay_probability:.3f}",
            f"  IGE (intergenerational elasticity) = {self.ige:.3f}",
            f"  Absolute mobility = {self.absolute_mobility:.3f}",
            f"  Interpretation: {self.interpretation}",
        ]
        return "\n".join(lines)


@dataclass
class DemographicResult:
    """Result of demographic transition calculation."""
    birth_rate: float       # per 1000
    death_rate: float       # per 1000
    migration_rate: float   # per 1000
    natural_increase: float # per 1000
    growth_rate: float      # per 1000
    growth_rate_pct: float  # percentage
    doubling_time: Optional[float]  # years, None if declining
    stage: int              # 1-5 demographic transition stage
    stage_name: str
    interpretation: str

    def __str__(self) -> str:
        dt_str = f"{self.doubling_time:.1f} years" if self.doubling_time else "N/A (declining)"
        lines = [
            "Demographic Transition:",
            f"  Birth rate:  {self.birth_rate:.1f}/1000",
            f"  Death rate:  {self.death_rate:.1f}/1000",
            f"  Migration:   {self.migration_rate:+.1f}/1000",
            f"  Natural increase: {self.natural_increase:.1f}/1000",
            f"  Growth rate: {self.growth_rate:.1f}/1000 ({self.growth_rate_pct:.2f}%)",
            f"  Doubling time: {dt_str}",
            f"  Stage: {self.stage} — {self.stage_name}",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


@dataclass
class DependencyResult:
    """Result of dependency ratio calculation."""
    pop_under_15: int
    pop_15_64: int
    pop_over_64: int
    total_population: int
    youth_ratio: float      # per 100 working-age
    old_age_ratio: float    # per 100 working-age
    total_ratio: float      # per 100 working-age
    working_age_share: float  # fraction of total pop
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Dependency Ratios:",
            f"  Population: {self.total_population:,} total",
            f"    Under 15: {self.pop_under_15:,} ({self.pop_under_15/self.total_population:.1%})",
            f"    15-64:    {self.pop_15_64:,} ({self.working_age_share:.1%})",
            f"    Over 64:  {self.pop_over_64:,} ({self.pop_over_64/self.total_population:.1%})",
            f"  Youth dependency:   {self.youth_ratio:.1f} per 100 working-age",
            f"  Old-age dependency: {self.old_age_ratio:.1f} per 100 working-age",
            f"  Total dependency:   {self.total_ratio:.1f} per 100 working-age",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


@dataclass
class CentralityResult:
    """Result of network centrality calculation."""
    n_nodes: int
    n_edges: int
    degree_centrality: Dict[str, float]
    betweenness_centrality: Dict[str, float]
    closeness_centrality: Dict[str, float]
    most_central_node: str
    most_peripheral_node: str
    bridges: List[Tuple[str, str]]  # edges whose removal disconnects
    density: float

    def __str__(self) -> str:
        lines = [
            "Network Centrality:",
            f"  Nodes: {self.n_nodes}, Edges: {self.n_edges}, Density: {self.density:.3f}",
            f"  Most central: {self.most_central_node}",
            f"  Most peripheral: {self.most_peripheral_node}",
            f"  Bridges: {self.bridges if self.bridges else 'None'}",
            "  Degree centrality:",
        ]
        for node in sorted(self.degree_centrality, key=self.degree_centrality.get, reverse=True):
            lines.append(
                f"    {node}: deg={self.degree_centrality[node]:.3f}"
                f"  btw={self.betweenness_centrality[node]:.3f}"
                f"  cls={self.closeness_centrality[node]:.3f}"
            )
        return "\n".join(lines)


@dataclass
class CollectiveActionResult:
    """Result of Granovetter's threshold model simulation."""
    n_agents: int
    thresholds: List[float]
    cascade_steps: List[int]    # number of new adopters per step
    final_adopters: int
    final_fraction: float
    tipping_point: Optional[float]  # threshold that initiates cascade
    cascade_occurred: bool
    n_steps: int
    interpretation: str

    def __str__(self) -> str:
        tp_str = f"{self.tipping_point:.3f}" if self.tipping_point is not None else "None"
        lines = [
            "Granovetter Threshold Model:",
            f"  Agents: {self.n_agents}",
            f"  Final adopters: {self.final_adopters}/{self.n_agents} ({self.final_fraction:.1%})",
            f"  Cascade occurred: {self.cascade_occurred}",
            f"  Steps to equilibrium: {self.n_steps}",
            f"  Tipping point: {tp_str}",
            f"  Cascade sequence: {self.cascade_steps}",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


@dataclass
class HomophilyResult:
    """Result of homophily index calculation."""
    homophily_index: float  # Coleman's H
    within_group_fraction: float  # observed w
    expected_within_fraction: float  # expected under random mixing
    n_edges: int
    n_within: int
    n_between: int
    group_counts: Dict[str, int]
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Homophily Index (Coleman):",
            f"  H = {self.homophily_index:.4f}",
            f"  Observed within-group ties: {self.within_group_fraction:.3f} ({self.n_within}/{self.n_edges})",
            f"  Expected under random mixing: {self.expected_within_fraction:.3f}",
            f"  Between-group ties: {self.n_between}",
            f"  Group sizes: {self.group_counts}",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


@dataclass
class DunbarResult:
    """Result of Dunbar layer classification."""
    group_size: int
    layer_name: str
    layer_number: int      # 1-6
    layer_max: int         # upper bound of this layer
    emotional_closeness: str
    contact_frequency: str
    layers: List[Tuple[str, int]]  # all (name, size) pairs
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Dunbar's Social Brain Layers:",
            f"  Group size: {self.group_size}",
            f"  Falls in layer {self.layer_number}: {self.layer_name} (up to {self.layer_max})",
            f"  Emotional closeness: {self.emotional_closeness}",
            f"  Contact frequency: {self.contact_frequency}",
            "  All layers:",
        ]
        for name, size in self.layers:
            marker = " <--" if name == self.layer_name else ""
            lines.append(f"    {name}: ~{size}{marker}")
        lines.append(f"  {self.interpretation}")
        return "\n".join(lines)


@dataclass
class PolarizationResult:
    """Result of opinion polarization measurement."""
    bimodality_coefficient: float
    is_polarized: bool
    variance: float
    skewness: float
    kurtosis: float
    esteban_ray_index: float
    n_opinions: int
    mean_opinion: float
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Polarization Index:",
            f"  Bimodality coefficient: {self.bimodality_coefficient:.4f}",
            f"  Polarized (BC > 0.555): {self.is_polarized}",
            f"  Esteban-Ray index: {self.esteban_ray_index:.4f}",
            f"  Mean opinion: {self.mean_opinion:.3f}",
            f"  Variance: {self.variance:.4f}",
            f"  Skewness: {self.skewness:.4f}, Kurtosis: {self.kurtosis:.4f}",
            f"  N = {self.n_opinions}",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calculator functions
# ---------------------------------------------------------------------------

def calc_gini_coefficient(incomes: List[float]) -> GiniResult:
    """Calculate the Gini coefficient for an income distribution.

    Uses the formula: G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n+1)/n
    where y_i are sorted incomes (1-indexed).

    Args:
        incomes: List of non-negative income values.

    Returns:
        GiniResult with coefficient, interpretation, and distributional stats.

    Example:
        >>> r = calc_gini_coefficient([10, 10, 10, 10])
        >>> r.gini  # perfect equality
        0.0
    """
    if not incomes:
        raise ValueError("Income list must be non-empty")
    if any(x < 0 for x in incomes):
        raise ValueError("Incomes must be non-negative")

    n = len(incomes)
    sorted_y = sorted(incomes)
    total = sum(sorted_y)

    if total == 0:
        return GiniResult(
            gini=0.0, n=n, mean_income=0.0, median_income=0.0,
            interpretation="All incomes zero — perfect equality by default",
            income_share_bottom_20=0.0, income_share_top_20=0.0,
        )

    # G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n+1)/n
    # i is 1-indexed
    weighted_sum = sum((i + 1) * y for i, y in enumerate(sorted_y))
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1) / n

    # Clamp to [0, 1] for floating-point edge cases
    gini = max(0.0, min(1.0, gini))

    mean_income = total / n
    if n % 2 == 1:
        median_income = sorted_y[n // 2]
    else:
        median_income = (sorted_y[n // 2 - 1] + sorted_y[n // 2]) / 2.0

    # Income shares for bottom/top 20%
    cutoff_low = max(1, n // 5)
    cutoff_high = n - cutoff_low
    bottom_20_share = sum(sorted_y[:cutoff_low]) / total
    top_20_share = sum(sorted_y[cutoff_high:]) / total

    if gini < 0.25:
        interp = "Low inequality"
    elif gini < 0.35:
        interp = "Moderate inequality"
    elif gini < 0.45:
        interp = "High inequality"
    else:
        interp = "Extreme inequality"

    return GiniResult(
        gini=gini, n=n, mean_income=mean_income, median_income=median_income,
        interpretation=interp,
        income_share_bottom_20=bottom_20_share,
        income_share_top_20=top_20_share,
    )


def calc_segregation_index(
    group_a: List[float], group_b: List[float]
) -> SegregationResult:
    """Calculate the dissimilarity index (index of segregation).

    D = 0.5 * sum(|a_i/A - b_i/B|) where a_i, b_i are group counts
    in area i and A, B are totals.

    Args:
        group_a: Count of group A in each area.
        group_b: Count of group B in each area.

    Returns:
        SegregationResult with D index and interpretation.

    Example:
        >>> r = calc_segregation_index([100, 0], [0, 100])
        >>> r.dissimilarity_index  # complete segregation
        1.0
    """
    if len(group_a) != len(group_b):
        raise ValueError("group_a and group_b must have the same length")
    if not group_a:
        raise ValueError("Must have at least one area")

    n_areas = len(group_a)
    A = sum(group_a)
    B = sum(group_b)

    if A == 0 or B == 0:
        raise ValueError("Both groups must have non-zero total population")

    contributions = []
    for i in range(n_areas):
        c = abs(group_a[i] / A - group_b[i] / B)
        contributions.append(c)

    D = 0.5 * sum(contributions)
    D = max(0.0, min(1.0, D))

    most_seg = max(range(n_areas), key=lambda i: contributions[i])

    if D < 0.30:
        interp = "Low segregation"
    elif D < 0.60:
        interp = "Moderate segregation"
    else:
        interp = "High segregation"

    return SegregationResult(
        dissimilarity_index=D, n_areas=n_areas,
        total_group_a=A, total_group_b=B,
        interpretation=interp, most_segregated_area=most_seg,
        pct_would_move=D,  # D = fraction of minority needing to relocate
    )


# Default US intergenerational mobility transition matrix (quintile to quintile).
# Rows = parent quintile (0=bottom, 4=top), cols = child quintile.
# Based on Chetty et al. (2014) estimates.
_US_MOBILITY_MATRIX = [
    [0.335, 0.241, 0.181, 0.139, 0.104],  # bottom quintile parents
    [0.244, 0.233, 0.207, 0.178, 0.138],
    [0.180, 0.200, 0.215, 0.214, 0.191],
    [0.137, 0.178, 0.210, 0.228, 0.247],
    [0.104, 0.148, 0.187, 0.241, 0.320],  # top quintile parents
]


def calc_social_mobility(
    parent_quintile: int,
    child_quintile: int,
    transition_matrix: Optional[List[List[float]]] = None,
) -> MobilityResult:
    """Calculate social mobility statistics from a transition matrix.

    Args:
        parent_quintile: Parent's income quintile (1-5, where 1=bottom).
        child_quintile: Child's income quintile (1-5).
        transition_matrix: 5x5 matrix of P(child_q | parent_q). Rows sum to 1.
            Defaults to US intergenerational mobility (Chetty et al. 2014).

    Returns:
        MobilityResult with transition probability, upward/downward odds, IGE.

    Example:
        >>> r = calc_social_mobility(1, 5)  # bottom to top
        >>> r.transition_probability  # ~10.4% in US
        0.104
    """
    if parent_quintile < 1 or parent_quintile > 5:
        raise ValueError("parent_quintile must be 1-5")
    if child_quintile < 1 or child_quintile > 5:
        raise ValueError("child_quintile must be 1-5")

    matrix = transition_matrix if transition_matrix is not None else _US_MOBILITY_MATRIX

    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise ValueError("Transition matrix must be 5x5")

    pi = parent_quintile - 1  # 0-indexed
    ci = child_quintile - 1

    row = matrix[pi]
    trans_prob = row[ci]

    upward = sum(row[j] for j in range(pi + 1, 5))
    downward = sum(row[j] for j in range(0, pi))
    stay = row[pi]

    # Absolute mobility: weighted average P(child > parent) across all quintiles
    # Simplified: sum of above-diagonal elements weighted by parent distribution
    # Assuming uniform parent distribution (each quintile = 20%)
    abs_mob = 0.0
    for p in range(5):
        for c in range(p + 1, 5):
            abs_mob += 0.2 * matrix[p][c]

    # Intergenerational elasticity (IGE): estimated from expected child quintile
    # given parent quintile. IGE ~ slope of log(child income) on log(parent income).
    # Approximation using quintile midpoints.
    quintile_midpoints = [0.1, 0.3, 0.5, 0.7, 0.9]  # normalized
    expected_child = []
    for p in range(5):
        ec = sum(matrix[p][c] * quintile_midpoints[c] for c in range(5))
        expected_child.append(ec)

    # IGE from regression of log(expected child) on log(parent midpoint)
    log_parent = [math.log(m) for m in quintile_midpoints]
    log_child = [math.log(ec) for ec in expected_child]
    mean_lp = sum(log_parent) / 5
    mean_lc = sum(log_child) / 5
    num = sum((log_parent[i] - mean_lp) * (log_child[i] - mean_lc) for i in range(5))
    den = sum((log_parent[i] - mean_lp) ** 2 for i in range(5))
    ige = num / den if den > 0 else 0.0

    if ige < 0.2:
        interp = "High mobility (Scandinavian-level)"
    elif ige < 0.35:
        interp = "Moderate mobility (Western European-level)"
    elif ige < 0.5:
        interp = "Low mobility (US-level)"
    else:
        interp = "Very low mobility (high persistence of inequality)"

    return MobilityResult(
        parent_quintile=parent_quintile, child_quintile=child_quintile,
        transition_probability=trans_prob,
        upward_probability=upward, downward_probability=downward,
        stay_probability=stay, ige=ige, absolute_mobility=abs_mob,
        interpretation=interp,
    )


def calc_demographic_transition(
    birth_rate: float, death_rate: float, migration_rate: float = 0.0,
) -> DemographicResult:
    """Classify a population's demographic transition stage.

    Rates are per 1000 population per year.

    Args:
        birth_rate: Crude birth rate per 1000.
        death_rate: Crude death rate per 1000.
        migration_rate: Net migration rate per 1000 (positive = net immigration).

    Returns:
        DemographicResult with growth rate, doubling time, stage classification.

    Example:
        >>> r = calc_demographic_transition(birth_rate=40, death_rate=38)
        >>> r.stage
        1
    """
    if birth_rate < 0 or death_rate < 0:
        raise ValueError("Birth and death rates must be non-negative")

    natural_increase = birth_rate - death_rate
    growth_rate = natural_increase + migration_rate
    growth_pct = growth_rate / 10.0  # per 1000 -> percentage

    if growth_pct > 0:
        doubling_time = 70.0 / growth_pct
    else:
        doubling_time = None

    # Stage classification based on birth/death rate thresholds
    if birth_rate >= 35 and death_rate >= 30:
        stage, name = 1, "Pre-transition"
        interp = "High birth and death rates; slow growth; pre-industrial pattern"
    elif birth_rate >= 30 and death_rate < 30:
        stage, name = 2, "Early transition"
        interp = "Death rate declining (sanitation, medicine); rapid population growth"
    elif birth_rate < 30 and birth_rate >= 15 and death_rate < 20:
        stage, name = 3, "Late transition"
        interp = "Birth rate declining (urbanization, education); growth slowing"
    elif birth_rate < 15 and death_rate < 15 and birth_rate >= death_rate:
        stage, name = 4, "Post-transition"
        interp = "Low birth and death rates; near-zero growth; industrialized pattern"
    elif birth_rate < death_rate:
        stage, name = 5, "Sub-replacement"
        interp = "Birth rate below death rate; population decline without immigration"
    else:
        # Fallback for ambiguous cases
        if birth_rate >= 25:
            stage, name = 2, "Early transition"
            interp = "Transitioning; death rate declining faster than birth rate"
        else:
            stage, name = 4, "Post-transition"
            interp = "Low rates; near demographic equilibrium"

    return DemographicResult(
        birth_rate=birth_rate, death_rate=death_rate,
        migration_rate=migration_rate, natural_increase=natural_increase,
        growth_rate=growth_rate, growth_rate_pct=growth_pct,
        doubling_time=doubling_time, stage=stage, stage_name=name,
        interpretation=interp,
    )


def calc_dependency_ratio(
    pop_under_15: int, pop_15_64: int, pop_over_64: int,
) -> DependencyResult:
    """Calculate age dependency ratios.

    Args:
        pop_under_15: Population aged 0-14.
        pop_15_64: Population aged 15-64 (working age).
        pop_over_64: Population aged 65+.

    Returns:
        DependencyResult with youth, old-age, and total ratios per 100 working-age.

    Example:
        >>> r = calc_dependency_ratio(200, 600, 100)
        >>> r.total_ratio
        50.0
    """
    if pop_15_64 <= 0:
        raise ValueError("Working-age population must be positive")
    if pop_under_15 < 0 or pop_over_64 < 0:
        raise ValueError("Population counts must be non-negative")

    total = pop_under_15 + pop_15_64 + pop_over_64
    youth = pop_under_15 / pop_15_64 * 100
    old_age = pop_over_64 / pop_15_64 * 100
    total_dep = youth + old_age
    working_share = pop_15_64 / total

    if total_dep < 40:
        interp = "Low dependency — demographic dividend window"
    elif total_dep < 60:
        interp = "Moderate dependency — typical for middle-income countries"
    elif total_dep < 80:
        interp = "High dependency — significant fiscal pressure on working-age population"
    else:
        interp = "Very high dependency — severe strain on productive population"

    if old_age > youth:
        interp += ". Aging society (old-age > youth dependency)"
    elif youth > old_age * 2:
        interp += ". Young society (youth-heavy dependency)"

    return DependencyResult(
        pop_under_15=pop_under_15, pop_15_64=pop_15_64,
        pop_over_64=pop_over_64, total_population=total,
        youth_ratio=youth, old_age_ratio=old_age,
        total_ratio=total_dep, working_age_share=working_share,
        interpretation=interp,
    )


def _bfs_shortest_paths(
    adjacency: Dict[str, List[str]], source: str,
) -> Dict[str, int]:
    """BFS from source, returning shortest distances to all reachable nodes."""
    dist: Dict[str, int] = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def _all_shortest_paths_count(
    adjacency: Dict[str, List[str]], source: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """BFS from source returning (distances, num_shortest_paths)."""
    dist: Dict[str, int] = {source: 0}
    count: Dict[str, int] = {source: 1}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                count[neighbor] = count[node]
                queue.append(neighbor)
            elif dist[neighbor] == dist[node] + 1:
                count[neighbor] += count[node]
    return dist, count


def calc_network_centrality(
    adjacency: Dict[str, List[str]],
) -> CentralityResult:
    """Calculate degree, betweenness, and closeness centrality for a network.

    Args:
        adjacency: Undirected graph as {node: [neighbors]}. Every edge should
            appear in both directions.

    Returns:
        CentralityResult with per-node centrality measures and structural info.

    Example:
        >>> g = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
        >>> r = calc_network_centrality(g)
        >>> r.density  # complete graph
        1.0
    """
    nodes = list(adjacency.keys())
    n = len(nodes)
    if n < 2:
        raise ValueError("Network must have at least 2 nodes")

    # Build symmetric adjacency (in case user forgot both directions)
    adj: Dict[str, set] = {node: set() for node in nodes}
    for node, neighbors in adjacency.items():
        for nb in neighbors:
            if nb not in adj:
                adj[nb] = set()
                nodes = list(adj.keys())
                n = len(nodes)
            adj[node].add(nb)
            adj[nb].add(node)

    # Convert back for BFS
    adj_list: Dict[str, List[str]] = {k: list(v) for k, v in adj.items()}

    n_edges = sum(len(v) for v in adj.values()) // 2
    max_edges = n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Degree centrality
    degree_cent: Dict[str, float] = {}
    for node in nodes:
        degree_cent[node] = len(adj[node]) / (n - 1)

    # Closeness centrality
    closeness_cent: Dict[str, float] = {}
    all_dists: Dict[str, Dict[str, int]] = {}
    for node in nodes:
        dists = _bfs_shortest_paths(adj_list, node)
        all_dists[node] = dists
        reachable = [d for nd, d in dists.items() if nd != node and d > 0]
        if reachable:
            avg_dist = sum(reachable) / len(reachable)
            closeness_cent[node] = 1.0 / avg_dist if avg_dist > 0 else 0.0
        else:
            closeness_cent[node] = 0.0

    # Betweenness centrality (Brandes-like using BFS)
    betweenness: Dict[str, float] = {node: 0.0 for node in nodes}

    for source in nodes:
        # BFS
        dist: Dict[str, int] = {source: 0}
        num_paths: Dict[str, int] = {source: 1}
        stack: List[str] = []
        queue = deque([source])
        predecessors: Dict[str, List[str]] = {node: [] for node in nodes}

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj_list.get(v, []):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    num_paths[w] = num_paths[v]
                    queue.append(w)
                    predecessors[w].append(v)
                elif dist[w] == dist[v] + 1:
                    num_paths[w] += num_paths[v]
                    predecessors[w].append(v)

        # Back-propagation of dependencies
        dependency: Dict[str, float] = {node: 0.0 for node in nodes}
        while stack:
            w = stack.pop()
            if w == source:
                continue
            for v in predecessors[w]:
                if num_paths.get(w, 0) > 0:
                    dependency[v] += (num_paths.get(v, 0) / num_paths[w]) * (1.0 + dependency[w])
            betweenness[w] += dependency[w]

    # Normalize (undirected: divide by 2)
    norm = (n - 1) * (n - 2) if n > 2 else 1
    betweenness_cent: Dict[str, float] = {}
    for node in nodes:
        betweenness_cent[node] = betweenness[node] / norm if norm > 0 else 0.0

    # Most central (by degree) and most peripheral
    most_central = max(nodes, key=lambda nd: degree_cent[nd])
    most_peripheral = min(nodes, key=lambda nd: degree_cent[nd])

    # Find bridges: edge (u, v) where removing it increases components
    bridges: List[Tuple[str, str]] = []
    seen_edges: set = set()
    for u in nodes:
        for v in adj[u]:
            edge = tuple(sorted([u, v]))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            # Remove edge and check connectivity
            adj[u].discard(v)
            adj[v].discard(u)
            test_adj = {k: list(vs) for k, vs in adj.items()}
            reachable = _bfs_shortest_paths(test_adj, u)
            if v not in reachable:
                bridges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)

    return CentralityResult(
        n_nodes=n, n_edges=n_edges,
        degree_centrality=degree_cent,
        betweenness_centrality=betweenness_cent,
        closeness_centrality=closeness_cent,
        most_central_node=most_central,
        most_peripheral_node=most_peripheral,
        bridges=bridges, density=density,
    )


def calc_granovetters_threshold(thresholds: List[float]) -> CollectiveActionResult:
    """Simulate Granovetter's threshold model of collective behavior.

    Each agent has a threshold: the fraction of others who must already be
    acting before they join. Agents with threshold 0 act unconditionally.

    Args:
        thresholds: List of threshold values in [0, 1] for each agent.

    Returns:
        CollectiveActionResult with cascade dynamics and tipping point.

    Example:
        >>> r = calc_granovetters_threshold([0.0, 0.1, 0.2, 0.3, 0.4])
        >>> r.final_fraction  # full cascade
        1.0
    """
    if not thresholds:
        raise ValueError("Must have at least one agent")
    if any(t < 0 or t > 1 for t in thresholds):
        raise ValueError("Thresholds must be in [0, 1]")

    n = len(thresholds)
    sorted_thresholds = sorted(thresholds)

    # Simulation
    active = set()
    cascade_steps: List[int] = []

    # Step 0: unconditional actors (threshold = 0)
    for i, t in enumerate(thresholds):
        if t == 0.0:
            active.add(i)
    if active:
        cascade_steps.append(len(active))

    # Subsequent steps
    changed = True
    while changed:
        changed = False
        fraction_active = len(active) / n
        new_active = set()
        for i, t in enumerate(thresholds):
            if i not in active and fraction_active >= t:
                new_active.add(i)
                changed = True
        if new_active:
            active.update(new_active)
            cascade_steps.append(len(new_active))

    final_adopters = len(active)
    final_fraction = final_adopters / n

    # Tipping point: find the critical threshold
    # The cascade stalls when sorted_thresholds[k] > k/n
    tipping_point = None
    for k, t in enumerate(sorted_thresholds):
        if t > k / n:
            # Cascade would stall here
            if k > 0:
                tipping_point = sorted_thresholds[k - 1]
            break
    else:
        # Full cascade — tipping point is the max threshold
        tipping_point = sorted_thresholds[-1] if sorted_thresholds else None

    cascade_occurred = final_fraction > 0.5

    if final_fraction >= 0.9:
        interp = "Full cascade — nearly all agents adopted"
    elif final_fraction >= 0.5:
        interp = "Partial cascade — majority adopted"
    elif final_fraction > 0:
        interp = "Stalled cascade — only early adopters joined"
    else:
        interp = "No cascade — no unconditional actors to seed adoption"

    return CollectiveActionResult(
        n_agents=n, thresholds=sorted_thresholds,
        cascade_steps=cascade_steps,
        final_adopters=final_adopters, final_fraction=final_fraction,
        tipping_point=tipping_point, cascade_occurred=cascade_occurred,
        n_steps=len(cascade_steps),
        interpretation=interp,
    )


def calc_homophily_index(
    network: Dict[str, List[str]], attributes: Dict[str, str],
) -> HomophilyResult:
    """Calculate Coleman's homophily index for a network.

    H = (w - w_expected) / (1 - w_expected) where w is the observed fraction
    of within-group ties and w_expected is the fraction expected under random
    mixing.

    Args:
        network: Adjacency list {node: [neighbors]}.
        attributes: {node: group_label} for each node.

    Returns:
        HomophilyResult with Coleman's H and group statistics.

    Example:
        >>> net = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
        >>> attr = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        >>> r = calc_homophily_index(net, attr)
        >>> r.homophily_index > 0  # perfect homophily
        True
    """
    if not network:
        raise ValueError("Network must be non-empty")

    # Count edges (undirected — count each edge once)
    n_within = 0
    n_between = 0
    seen: set = set()

    for node, neighbors in network.items():
        for nb in neighbors:
            edge = tuple(sorted([node, nb]))
            if edge in seen:
                continue
            seen.add(edge)
            if node not in attributes or nb not in attributes:
                continue
            if attributes[node] == attributes[nb]:
                n_within += 1
            else:
                n_between += 1

    n_edges = n_within + n_between
    if n_edges == 0:
        raise ValueError("Network has no edges with attributed nodes")

    w = n_within / n_edges

    # Expected within-group fraction under random mixing
    # w_expected = sum(n_g * (n_g - 1)) / (N * (N - 1)) for undirected
    group_counts: Dict[str, int] = {}
    for node in network:
        if node in attributes:
            g = attributes[node]
            group_counts[g] = group_counts.get(g, 0) + 1

    N = sum(group_counts.values())
    if N < 2:
        raise ValueError("Need at least 2 attributed nodes")

    w_expected = sum(
        ng * (ng - 1) for ng in group_counts.values()
    ) / (N * (N - 1))

    # Coleman's H
    if abs(1.0 - w_expected) < 1e-12:
        # Only one group — homophily undefined
        H = 0.0
    elif w >= w_expected:
        H = (w - w_expected) / (1.0 - w_expected)
    else:
        # Heterophily case: H = (w - w_expected) / w_expected
        H = (w - w_expected) / w_expected if w_expected > 0 else 0.0

    if H > 0.3:
        interp = "Strong homophily — ties concentrated within groups"
    elif H > 0.05:
        interp = "Moderate homophily — some within-group preference"
    elif H > -0.05:
        interp = "Near-random mixing — no significant group preference"
    elif H > -0.3:
        interp = "Moderate heterophily — preference for between-group ties"
    else:
        interp = "Strong heterophily — ties concentrated between groups"

    return HomophilyResult(
        homophily_index=H, within_group_fraction=w,
        expected_within_fraction=w_expected,
        n_edges=n_edges, n_within=n_within, n_between=n_between,
        group_counts=group_counts, interpretation=interp,
    )


# Dunbar's concentric layers
_DUNBAR_LAYERS = [
    (1, "Support clique", 5, "Highest", "Daily or near-daily"),
    (2, "Sympathy group", 15, "High", "Weekly"),
    (3, "Close friends", 50, "Moderate", "Monthly"),
    (4, "Acquaintances (Dunbar's number)", 150, "Low", "Yearly"),
    (5, "Known names", 500, "Minimal", "Occasionally"),
    (6, "Recognized faces", 1500, "None", "Rarely"),
]


def calc_dunbar_layers(group_size: int) -> DunbarResult:
    """Classify a group size within Dunbar's concentric social layers.

    Dunbar's number (~150) is the cognitive limit on stable social
    relationships. Layers scale by roughly 3x: 5, 15, 50, 150, 500, 1500.

    Args:
        group_size: Number of people in the group.

    Returns:
        DunbarResult with layer classification and social characteristics.

    Example:
        >>> r = calc_dunbar_layers(100)
        >>> r.layer_name
        'Acquaintances (Dunbar\\'s number)'
    """
    if group_size < 1:
        raise ValueError("Group size must be at least 1")

    layers = [(name, size) for _, name, size, _, _ in _DUNBAR_LAYERS]

    # Find the layer this group size falls into
    matched = _DUNBAR_LAYERS[-1]  # default to largest
    for layer_info in _DUNBAR_LAYERS:
        if group_size <= layer_info[2]:
            matched = layer_info
            break

    layer_num, layer_name, layer_max, closeness, frequency = matched

    if group_size <= 5:
        interp = (
            "Within the support clique — these are your closest confidants. "
            "Maximum emotional intensity and mutual aid."
        )
    elif group_size <= 15:
        interp = (
            "Within the sympathy group — close enough to feel genuine grief "
            "at loss. Core social support network."
        )
    elif group_size <= 50:
        interp = (
            "Within close friends — regular social contact, trusted for "
            "favors. Key collaborators."
        )
    elif group_size <= 150:
        interp = (
            "Within Dunbar's number — stable social relationships maintained "
            "through occasional contact. Typical community/company size."
        )
    elif group_size <= 500:
        interp = (
            "Beyond Dunbar's number — you know names but relationships are "
            "shallow. Typical school or extended organization."
        )
    elif group_size <= 1500:
        interp = (
            "Recognized faces only — you can put names to faces but have no "
            "real relationship. Typical large organization."
        )
    else:
        interp = (
            f"Beyond all Dunbar layers ({group_size} > 1500). No individual "
            "can maintain this many relationships — requires institutional "
            "structure (hierarchy, roles, norms)."
        )

    return DunbarResult(
        group_size=group_size, layer_name=layer_name,
        layer_number=layer_num, layer_max=layer_max,
        emotional_closeness=closeness, contact_frequency=frequency,
        layers=layers, interpretation=interp,
    )


def calc_polarization_index(opinions: List[float]) -> PolarizationResult:
    """Measure opinion polarization using bimodality and Esteban-Ray index.

    Args:
        opinions: List of opinion values (any numeric scale).

    Returns:
        PolarizationResult with bimodality coefficient, ER index, and stats.

    Example:
        >>> # Two extreme camps
        >>> r = calc_polarization_index([0.0]*50 + [1.0]*50)
        >>> r.is_polarized
        True
    """
    if len(opinions) < 3:
        raise ValueError("Need at least 3 opinions to measure polarization")

    n = len(opinions)
    mean_op = sum(opinions) / n

    # Variance
    variance = sum((x - mean_op) ** 2 for x in opinions) / n

    # Skewness (Fisher's)
    if variance == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        sd = math.sqrt(variance)
        m3 = sum((x - mean_op) ** 3 for x in opinions) / n
        m4 = sum((x - mean_op) ** 4 for x in opinions) / n
        skewness = m3 / (sd ** 3)
        kurtosis = m4 / (sd ** 4)  # raw kurtosis (not excess)

    # Bimodality coefficient: BC = (skewness^2 + 1) / kurtosis
    # BC > 5/9 ≈ 0.555 suggests bimodal distribution
    if kurtosis > 0:
        bc = (skewness ** 2 + 1) / kurtosis
    else:
        bc = 0.0  # degenerate case

    is_polarized = bc > 5.0 / 9.0

    # Simplified Esteban-Ray polarization index
    # ER = sum_i sum_j |x_i - x_j| * p_i^(1+alpha) * p_j
    # Using alpha=1.6 (standard), treating each opinion as its own group (p=1/n)
    # Simplification: use pairwise absolute differences weighted by identification
    # For continuous opinions, bin into groups
    if variance > 0:
        # Bin into 2 groups around the mean for ER
        below = [x for x in opinions if x <= mean_op]
        above = [x for x in opinions if x > mean_op]
        p1 = len(below) / n
        p2 = len(above) / n
        m1 = sum(below) / len(below) if below else mean_op
        m2 = sum(above) / len(above) if above else mean_op

        alpha = 1.6  # standard ER parameter
        # ER = p1^(1+alpha) * p2 * |m1-m2| + p2^(1+alpha) * p1 * |m1-m2|
        dist = abs(m1 - m2)
        er = dist * (p1 ** (1 + alpha) * p2 + p2 ** (1 + alpha) * p1)
    else:
        er = 0.0

    if is_polarized and er > 0.1:
        interp = "Highly polarized — bimodal opinion distribution with distinct camps"
    elif is_polarized:
        interp = "Moderately polarized — bimodal distribution but camps overlap"
    elif variance > 0.1:
        interp = "Dispersed but not polarized — opinions vary without clustering into camps"
    else:
        interp = "Consensus — opinions are concentrated, low variance"

    return PolarizationResult(
        bimodality_coefficient=bc, is_polarized=is_polarized,
        variance=variance, skewness=skewness, kurtosis=kurtosis,
        esteban_ray_index=er, n_opinions=n, mean_opinion=mean_op,
        interpretation=interp,
    )
