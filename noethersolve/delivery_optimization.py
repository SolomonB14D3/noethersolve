"""Delivery and logistics optimization calculator — derives answers from first principles.

Covers Economic Order Quantity (EOQ), vehicle routing bounds, newsvendor model,
safety stock calculation, and bin packing bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── EOQ (Economic Order Quantity) ─────────────────────────────────────


@dataclass
class EOQReport:
    """Result of Economic Order Quantity calculation."""
    demand_rate: float          # D: annual demand
    setup_cost: float           # S: cost per order
    holding_cost: float         # H: annual holding cost per unit
    optimal_quantity: float     # Q* = sqrt(2DS/H)
    total_cost: float           # TC = D/Q*·S + Q*/2·H
    ordering_cost: float        # D/Q*·S
    holding_cost_total: float   # Q*/2·H
    num_orders_per_year: float  # D/Q*
    reorder_point: Optional[float]  # if lead time given
    cycle_time: float           # Q*/D (fraction of year)

    def __str__(self) -> str:
        lines = [
            "Economic Order Quantity (EOQ):",
            f"  Demand (D): {self.demand_rate:.1f} units/year",
            f"  Setup cost (S): ${self.setup_cost:.2f}/order",
            f"  Holding cost (H): ${self.holding_cost:.2f}/unit/year",
            f"  Optimal order quantity (Q*): {self.optimal_quantity:.2f} units",
            f"  Total annual cost: ${self.total_cost:.2f}",
            f"    Ordering cost: ${self.ordering_cost:.2f}",
            f"    Holding cost:  ${self.holding_cost_total:.2f}",
            f"  Orders per year: {self.num_orders_per_year:.2f}",
            f"  Cycle time: {self.cycle_time:.4f} years ({self.cycle_time * 365:.1f} days)",
        ]
        if self.reorder_point is not None:
            lines.append(f"  Reorder point: {self.reorder_point:.2f} units")
        return "\n".join(lines)


def calc_eoq(
    demand: float,
    setup_cost: float,
    holding_cost: float,
    lead_time_days: Optional[float] = None,
) -> EOQReport:
    """Compute Economic Order Quantity and total cost.

    Q* = sqrt(2DS/H)
    Total cost = D/Q* * S + Q*/2 * H

    Args:
        demand: Annual demand in units (D).
        setup_cost: Fixed cost per order in dollars (S).
        holding_cost: Annual holding cost per unit in dollars (H).
        lead_time_days: Lead time in days (for reorder point calculation).

    Returns:
        EOQReport with optimal quantity, costs, and reorder point.

    Example:
        >>> r = calc_eoq(10000, 50, 2.5)
        >>> round(r.optimal_quantity, 1)
        632.5
    """
    if demand <= 0:
        raise ValueError(f"Demand must be positive, got {demand}")
    if setup_cost <= 0:
        raise ValueError(f"Setup cost must be positive, got {setup_cost}")
    if holding_cost <= 0:
        raise ValueError(f"Holding cost must be positive, got {holding_cost}")

    q_star = math.sqrt(2 * demand * setup_cost / holding_cost)
    ordering = demand / q_star * setup_cost
    holding_total = q_star / 2 * holding_cost
    total = ordering + holding_total
    num_orders = demand / q_star
    cycle_time = q_star / demand

    reorder_point = None
    if lead_time_days is not None:
        if lead_time_days < 0:
            raise ValueError(f"Lead time must be non-negative, got {lead_time_days}")
        daily_demand = demand / 365.0
        reorder_point = daily_demand * lead_time_days

    return EOQReport(
        demand_rate=demand,
        setup_cost=setup_cost,
        holding_cost=holding_cost,
        optimal_quantity=q_star,
        total_cost=total,
        ordering_cost=ordering,
        holding_cost_total=holding_total,
        num_orders_per_year=num_orders,
        reorder_point=reorder_point,
        cycle_time=cycle_time,
    )


# ── Vehicle Routing Bounds ────────────────────────────────────────────


@dataclass
class VehicleRoutingReport:
    """Result of vehicle routing bound computation."""
    n_stops: int
    mst_lower_bound: float          # MST weight (lower bound on TSP)
    nearest_neighbor_upper: float   # NN heuristic tour length
    nearest_neighbor_tour: List[int]
    savings_upper: float            # Clarke-Wright savings tour length
    savings_tour: List[int]
    gap_percent: float              # (upper - lower) / lower * 100

    def __str__(self) -> str:
        best_upper = min(self.nearest_neighbor_upper, self.savings_upper)
        lines = [
            f"Vehicle Routing Bounds ({self.n_stops} stops):",
            f"  MST lower bound: {self.mst_lower_bound:.2f}",
            f"  Nearest-neighbor tour: {self.nearest_neighbor_upper:.2f}",
            f"    Tour: {' → '.join(str(s) for s in self.nearest_neighbor_tour)}",
            f"  Clarke-Wright savings tour: {self.savings_upper:.2f}",
            f"    Tour: {' → '.join(str(s) for s in self.savings_tour)}",
            f"  Best upper bound: {best_upper:.2f}",
            f"  Optimality gap: {self.gap_percent:.1f}%",
        ]
        return "\n".join(lines)


def _distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Build Euclidean distance matrix from 2D coordinates."""
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _prim_mst(dist: List[List[float]]) -> float:
    """Compute MST weight using Prim's algorithm. O(n^2)."""
    n = len(dist)
    if n <= 1:
        return 0.0
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        # Pick minimum edge not in MST
        u = -1
        for v in range(n):
            if not in_mst[v] and (u == -1 or min_edge[v] < min_edge[u]):
                u = v
        in_mst[u] = True
        total += min_edge[u]
        for v in range(n):
            if not in_mst[v] and dist[u][v] < min_edge[v]:
                min_edge[v] = dist[u][v]
    return total


def _nearest_neighbor(dist: List[List[float]], start: int = 0) -> Tuple[float, List[int]]:
    """Nearest-neighbor heuristic for TSP. Returns (tour_length, tour)."""
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    total = 0.0
    current = start

    for _ in range(n - 1):
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True
        total += best_dist
        current = best_next

    # Return to start
    total += dist[current][start]
    tour.append(start)
    return total, tour


def _clarke_wright_savings(dist: List[List[float]], depot: int = 0) -> Tuple[float, List[int]]:
    """Clarke-Wright savings algorithm for single-vehicle TSP.

    Starts with star routes depot→i→depot for each stop, then merges
    using savings s(i,j) = d(depot,i) + d(depot,j) - d(i,j).
    """
    n = len(dist)
    if n <= 2:
        tour = list(range(n)) + [0] if n > 0 else []
        length = sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) if len(tour) > 1 else 0.0
        return length, tour

    # Non-depot nodes
    stops = [i for i in range(n) if i != depot]

    # Compute savings
    savings = []
    for i in stops:
        for j in stops:
            if i < j:
                s = dist[depot][i] + dist[depot][j] - dist[i][j]
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # Each stop starts in its own route (as a list)
    # route_of[i] = index of the route containing node i
    routes: List[List[int]] = [[s] for s in stops]
    route_of = {}
    for idx, s in enumerate(stops):
        route_of[s] = idx

    for s_val, i, j in savings:
        if s_val <= 0:
            break
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue  # same route, skip

        # Can only merge if i and j are at the ends of their routes
        route_i = routes[ri]
        route_j = routes[rj]

        # Check if i is at an end of route_i and j is at an end of route_j
        i_at_start = route_i[0] == i
        i_at_end = route_i[-1] == i
        j_at_start = route_j[0] == j
        j_at_end = route_j[-1] == j

        if not (i_at_start or i_at_end):
            continue
        if not (j_at_start or j_at_end):
            continue

        # Merge: arrange so i is at end of route_i, j at start of route_j
        if i_at_start:
            route_i.reverse()
        if j_at_end:
            route_j.reverse()

        merged = route_i + route_j
        routes[ri] = merged
        routes[rj] = []  # empty
        for node in merged:
            route_of[node] = ri

    # Collect the single non-empty route
    final_route = []
    for r in routes:
        if r:
            final_route.extend(r)

    # Build tour: depot → route → depot
    tour = [depot] + final_route + [depot]
    length = sum(dist[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
    return length, tour


def calc_vehicle_routing(
    coordinates: List[Tuple[float, float]],
    depot: int = 0,
) -> VehicleRoutingReport:
    """Compute vehicle routing bounds for a set of 2D stops.

    Lower bound: Minimum Spanning Tree weight.
    Upper bounds: Nearest-neighbor heuristic and Clarke-Wright savings.

    Args:
        coordinates: List of (x, y) positions for each stop.
        depot: Index of the depot node (default 0).

    Returns:
        VehicleRoutingReport with bounds and heuristic tours.

    Example:
        >>> coords = [(0,0), (1,0), (1,1), (0,1)]
        >>> r = calc_vehicle_routing(coords)
        >>> r.mst_lower_bound <= r.nearest_neighbor_upper
        True
    """
    n = len(coordinates)
    if n < 2:
        raise ValueError(f"Need at least 2 stops, got {n}")
    if depot < 0 or depot >= n:
        raise ValueError(f"Depot index {depot} out of range [0, {n - 1}]")

    dist = _distance_matrix(coordinates)
    mst_weight = _prim_mst(dist)
    nn_length, nn_tour = _nearest_neighbor(dist, start=depot)
    cw_length, cw_tour = _clarke_wright_savings(dist, depot=depot)

    best_upper = min(nn_length, cw_length)
    gap = (best_upper - mst_weight) / mst_weight * 100 if mst_weight > 0 else 0.0

    return VehicleRoutingReport(
        n_stops=n,
        mst_lower_bound=mst_weight,
        nearest_neighbor_upper=nn_length,
        nearest_neighbor_tour=nn_tour,
        savings_upper=cw_length,
        savings_tour=cw_tour,
        gap_percent=gap,
    )


# ── Newsvendor Model ─────────────────────────────────────────────────


@dataclass
class NewsvendorReport:
    """Result of newsvendor (perishable goods) optimization."""
    price: float            # p: selling price
    cost: float             # c: purchase cost
    salvage: float          # s: salvage value
    critical_ratio: float   # CR = (p - c) / (p - s)
    demand_mean: float      # mu
    demand_std: float       # sigma
    optimal_quantity: float  # Q* where F(Q*) = CR
    expected_profit: float
    overage_cost: float     # c - s (cost of ordering too much)
    underage_cost: float    # p - c (cost of ordering too little)
    z_score: float          # number of std devs above mean

    def __str__(self) -> str:
        lines = [
            "Newsvendor Model (Perishable Goods):",
            f"  Price: ${self.price:.2f}  Cost: ${self.cost:.2f}  Salvage: ${self.salvage:.2f}",
            f"  Critical ratio: {self.critical_ratio:.4f}",
            f"  Underage cost (Cu): ${self.underage_cost:.2f}",
            f"  Overage cost (Co): ${self.overage_cost:.2f}",
            f"  Demand: N({self.demand_mean:.1f}, {self.demand_std:.1f})",
            f"  Optimal order quantity (Q*): {self.optimal_quantity:.2f} units",
            f"  Z-score: {self.z_score:.4f}",
            f"  Expected profit: ${self.expected_profit:.2f}",
        ]
        return "\n".join(lines)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function (math.erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function).

    Uses rational approximation (Abramowitz & Stegun 26.2.23 / Peter Acklam).
    Accurate to ~1.15e-9 for 0 < p < 1.
    """
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    # Acklam's algorithm
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


def _norm_loss(z: float) -> float:
    """Standard normal loss function L(z) = E[max(Z - z, 0)] = phi(z) - z*(1-Phi(z))."""
    return _norm_pdf(z) - z * (1.0 - _norm_cdf(z))


def calc_newsvendor(
    price: float,
    cost: float,
    salvage: float,
    demand_mean: float,
    demand_std: float,
) -> NewsvendorReport:
    """Solve the newsvendor problem for perishable goods.

    Finds Q* such that F(Q*) = (p - c) / (p - s) under normal demand.

    Args:
        price: Selling price per unit (p).
        cost: Purchase cost per unit (c).
        salvage: Salvage value per unsold unit (s).
        demand_mean: Mean of normal demand distribution (mu).
        demand_std: Standard deviation of normal demand (sigma).

    Returns:
        NewsvendorReport with optimal quantity and expected profit.

    Example:
        >>> r = calc_newsvendor(price=10, cost=6, salvage=2, demand_mean=100, demand_std=20)
        >>> r.critical_ratio
        0.5
        >>> round(r.optimal_quantity, 1)
        100.0
    """
    if price <= cost:
        raise ValueError(f"Price ({price}) must exceed cost ({cost})")
    if salvage > cost:
        raise ValueError(f"Salvage ({salvage}) must not exceed cost ({cost})")
    if price <= salvage:
        raise ValueError(f"Price ({price}) must exceed salvage ({salvage})")
    if demand_std <= 0:
        raise ValueError(f"Demand std must be positive, got {demand_std}")

    cu = price - cost      # underage cost
    co = cost - salvage    # overage cost
    cr = cu / (cu + co)    # critical ratio = (p - c) / (p - s)

    z = _norm_ppf(cr)
    q_star = demand_mean + z * demand_std

    # Expected profit = cu * E[min(Q, D)] - co * E[max(Q - D, 0)]
    # For normal: E[min(Q, D)] = mu - sigma * L(-z) where z = (Q - mu)/sigma
    # Simpler: E[profit] = (p - s) * [mu * Phi(z) + sigma * phi(z)] - (c - s) * Q
    # Or equivalently:
    # E[sold] = Q - sigma * L(z) = Q - sigma * [phi(z) - z*(1-Phi(z))]
    # But we use: E[profit] = cu * mu - cu * sigma * L(z_star) where z_star is already computed
    # Actually: E[profit] = cu * E[min(Q,D)] - co * E[max(Q-D,0)]
    # E[max(Q-D,0)] = sigma * L(-z) = sigma * [phi(z) + z * Phi(z) - z]
    # Cleaner formulation:
    loss_z = _norm_loss(z)
    expected_sales = q_star - demand_std * loss_z  # E[min(Q, D)]
    expected_leftover = q_star - expected_sales    # E[max(Q - D, 0)]
    expected_profit = cu * expected_sales - co * expected_leftover

    return NewsvendorReport(
        price=price,
        cost=cost,
        salvage=salvage,
        critical_ratio=cr,
        demand_mean=demand_mean,
        demand_std=demand_std,
        optimal_quantity=q_star,
        expected_profit=expected_profit,
        overage_cost=co,
        underage_cost=cu,
        z_score=z,
    )


# ── Safety Stock Calculator ──────────────────────────────────────────


@dataclass
class SafetyStockReport:
    """Result of safety stock calculation."""
    demand_rate: float          # average demand per period
    demand_std: float           # demand standard deviation per period
    lead_time: float            # lead time in periods
    service_level: float        # target service level (e.g. 0.95)
    z_score: float              # z for target service level
    safety_stock: float         # z * sigma_d * sqrt(L)
    reorder_point: float        # mean demand during LT + safety stock
    lead_time_demand: float     # average demand during lead time
    lead_time_demand_std: float # std dev of demand during lead time

    def __str__(self) -> str:
        lines = [
            "Safety Stock Analysis:",
            f"  Demand rate: {self.demand_rate:.1f} units/period",
            f"  Demand std: {self.demand_std:.1f} units/period",
            f"  Lead time: {self.lead_time:.1f} periods",
            f"  Service level: {self.service_level * 100:.1f}%",
            f"  Z-score: {self.z_score:.4f}",
            f"  Lead time demand (avg): {self.lead_time_demand:.2f} units",
            f"  Lead time demand (std): {self.lead_time_demand_std:.2f} units",
            f"  Safety stock: {self.safety_stock:.2f} units",
            f"  Reorder point (ROP): {self.reorder_point:.2f} units",
        ]
        return "\n".join(lines)


def calc_safety_stock(
    demand_rate: float,
    demand_std: float,
    lead_time: float,
    service_level: float = 0.95,
) -> SafetyStockReport:
    """Compute safety stock and reorder point.

    Safety stock = z * sigma_d * sqrt(L)
    Reorder point = d_avg * L + safety_stock

    Args:
        demand_rate: Average demand per period.
        demand_std: Standard deviation of demand per period.
        lead_time: Lead time in the same period units.
        service_level: Target service level (probability, 0 < sl < 1).

    Returns:
        SafetyStockReport with safety stock, reorder point.

    Example:
        >>> r = calc_safety_stock(200, 40, 4, 0.95)
        >>> round(r.safety_stock, 1)
        131.5
    """
    if demand_rate < 0:
        raise ValueError(f"Demand rate must be non-negative, got {demand_rate}")
    if demand_std < 0:
        raise ValueError(f"Demand std must be non-negative, got {demand_std}")
    if lead_time <= 0:
        raise ValueError(f"Lead time must be positive, got {lead_time}")
    if not (0 < service_level < 1):
        raise ValueError(f"Service level must be in (0, 1), got {service_level}")

    z = _norm_ppf(service_level)
    lt_demand = demand_rate * lead_time
    lt_std = demand_std * math.sqrt(lead_time)
    ss = z * lt_std
    rop = lt_demand + ss

    return SafetyStockReport(
        demand_rate=demand_rate,
        demand_std=demand_std,
        lead_time=lead_time,
        service_level=service_level,
        z_score=z,
        safety_stock=ss,
        reorder_point=rop,
        lead_time_demand=lt_demand,
        lead_time_demand_std=lt_std,
    )


# ── Bin Packing Bounds ───────────────────────────────────────────────


@dataclass
class BinPackingReport:
    """Result of bin packing analysis."""
    n_items: int
    bin_capacity: float
    total_size: float
    l1_lower_bound: int         # ceil(sum / capacity)
    l2_lower_bound: int         # improved bound
    ffd_bins_used: int          # First Fit Decreasing heuristic
    ffd_assignment: List[List[int]]  # bin contents (item indices)
    gap: str                    # "optimal is between L2 and FFD"

    def __str__(self) -> str:
        lines = [
            f"Bin Packing Analysis ({self.n_items} items, capacity={self.bin_capacity}):",
            f"  Total item size: {self.total_size:.2f}",
            f"  L1 lower bound (sum/capacity): {self.l1_lower_bound} bins",
            f"  L2 lower bound: {self.l2_lower_bound} bins",
            f"  FFD heuristic: {self.ffd_bins_used} bins",
            f"  {self.gap}",
            f"  FFD assignment:",
        ]
        for i, b in enumerate(self.ffd_assignment):
            items_str = ", ".join(str(idx) for idx in b)
            bin_total = sum(self._sizes[idx] for idx in b) if hasattr(self, '_sizes') else 0
            lines.append(f"    Bin {i}: items [{items_str}]")
        return "\n".join(lines)


def _l2_bound(sizes: List[float], capacity: float) -> int:
    """Compute L2 lower bound for bin packing.

    For each threshold t in (0, capacity/2], count items > capacity - t,
    and items in (capacity/2 - epsilon, capacity - t]. The L2 bound considers
    pairwise packing infeasibility.

    Uses the formulation: L2 = max over alpha in (0, C/2] of
        n1(alpha) + max(0, n2(alpha) - residual_space(alpha))
    where n1 = items > C - alpha, n2 = items in (alpha, C - alpha].
    """
    n = len(sizes)
    if n == 0:
        return 0

    # Simple but effective L2: try several thresholds
    best = math.ceil(sum(sizes) / capacity)  # L1 as starting point

    # Sort descending for analysis
    sorted_sizes = sorted(sizes, reverse=True)

    # Try thresholds at each unique "boundary"
    for alpha_idx in range(n):
        alpha = capacity - sorted_sizes[alpha_idx]
        if alpha <= 0 or alpha > capacity / 2:
            continue

        # n1: items strictly greater than C - alpha (i.e., > sorted_sizes[alpha_idx])
        n1 = 0
        residual = 0.0
        for s in sorted_sizes:
            if s > capacity - alpha:
                n1 += 1
                residual += capacity - s
            else:
                break

        # n2: items in (alpha, C - alpha]
        n2 = 0
        sum_n2 = 0.0
        for s in sorted_sizes:
            if alpha < s <= capacity - alpha:
                n2 += 1
                sum_n2 += s

        extra = max(0, math.ceil((sum_n2 - residual) / capacity))
        candidate = n1 + extra
        best = max(best, candidate)

    return best


def _first_fit_decreasing(sizes: List[float], capacity: float) -> Tuple[int, List[List[int]]]:
    """First Fit Decreasing heuristic for bin packing.

    Sort items by decreasing size, then place each into the first bin that fits.
    Returns (num_bins, assignment) where assignment[bin_idx] = [item_indices].
    """
    n = len(sizes)
    # Sort by size descending, keep original indices
    indexed = sorted(enumerate(sizes), key=lambda x: -x[1])

    bins: List[Tuple[float, List[int]]] = []  # (remaining_capacity, [item_indices])

    for orig_idx, size in indexed:
        if size > capacity:
            raise ValueError(
                f"Item {orig_idx} (size={size}) exceeds bin capacity ({capacity})"
            )
        placed = False
        for i, (remaining, items) in enumerate(bins):
            if size <= remaining:
                bins[i] = (remaining - size, items + [orig_idx])
                placed = True
                break
        if not placed:
            bins.append((capacity - size, [orig_idx]))

    assignment = [b[1] for b in bins]
    return len(bins), assignment


def calc_bin_packing(
    sizes: List[float],
    capacity: float,
) -> BinPackingReport:
    """Compute bin packing bounds and FFD heuristic solution.

    Lower bounds: L1 = ceil(sum/capacity), L2 (improved).
    Upper bound: First Fit Decreasing heuristic.

    Args:
        sizes: List of item sizes.
        capacity: Bin capacity (all items must be <= capacity).

    Returns:
        BinPackingReport with bounds and heuristic assignment.

    Example:
        >>> r = calc_bin_packing([0.7, 0.5, 0.3, 0.2, 0.8, 0.4], 1.0)
        >>> r.l1_lower_bound
        3
    """
    if capacity <= 0:
        raise ValueError(f"Capacity must be positive, got {capacity}")
    if not sizes:
        raise ValueError("Need at least one item")

    for i, s in enumerate(sizes):
        if s <= 0:
            raise ValueError(f"Item {i} size must be positive, got {s}")
        if s > capacity:
            raise ValueError(f"Item {i} (size={s}) exceeds capacity ({capacity})")

    total = sum(sizes)
    l1 = math.ceil(total / capacity)
    l2 = _l2_bound(sizes, capacity)
    ffd_count, ffd_assign = _first_fit_decreasing(sizes, capacity)

    gap = f"Optimal is between {l2} and {ffd_count} bins"
    if l2 == ffd_count:
        gap = f"FFD is optimal: {ffd_count} bins"

    report = BinPackingReport(
        n_items=len(sizes),
        bin_capacity=capacity,
        total_size=total,
        l1_lower_bound=l1,
        l2_lower_bound=l2,
        ffd_bins_used=ffd_count,
        ffd_assignment=ffd_assign,
        gap=gap,
    )
    # Store sizes for __str__ display
    report._sizes = sizes  # type: ignore[attr-defined]
    return report
