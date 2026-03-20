#!/usr/bin/env python3
"""supply_chain_lab.py -- Supply chain optimization analysis prototype.

Chains NoetherSolve operations research tools (EOQ, vehicle routing, newsvendor,
safety stock, bin packing) to analyze supply chain scenarios and optimize
inventory, logistics, and warehouse operations.

Usage:
    python labs/supply_chain_lab.py
    python labs/supply_chain_lab.py --verbose
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.delivery_optimization import (
    calc_eoq,
    calc_vehicle_routing,
    calc_newsvendor,
    calc_safety_stock,
    calc_bin_packing,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "supply_chain"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class InventoryScenario:
    """An inventory management scenario to analyze."""
    name: str
    description: str
    # EOQ parameters
    annual_demand: float
    order_cost: float  # $ per order
    holding_cost: float  # $ per unit per year
    lead_time_days: float
    # Safety stock parameters
    demand_std_daily: float  # standard deviation of daily demand
    service_level: float  # target service level


@dataclass
class PerishableScenario:
    """A perishable goods (newsvendor) scenario."""
    name: str
    description: str
    price: float  # selling price
    cost: float  # purchase cost
    salvage: float  # salvage value
    demand_mean: float
    demand_std: float


@dataclass
class LogisticsScenario:
    """A vehicle routing scenario."""
    name: str
    description: str
    depot: int
    stop_coords: List[Tuple[float, float]]  # (x, y) coordinates


@dataclass
class WarehouseScenario:
    """A bin packing / warehouse scenario."""
    name: str
    description: str
    bin_capacity: float
    item_sizes: List[float]


# Sample scenarios
INVENTORY_SCENARIOS: List[InventoryScenario] = [
    InventoryScenario(
        name="retail_electronics",
        description="Consumer electronics retailer - moderate demand, high holding cost",
        annual_demand=5000,
        order_cost=150.0,
        holding_cost=25.0,
        lead_time_days=7,
        demand_std_daily=5.0,
        service_level=0.95,
    ),
    InventoryScenario(
        name="auto_parts",
        description="Auto parts distributor - high demand, low holding cost",
        annual_demand=50000,
        order_cost=75.0,
        holding_cost=3.0,
        lead_time_days=14,
        demand_std_daily=30.0,
        service_level=0.98,
    ),
    InventoryScenario(
        name="pharma_warehouse",
        description="Pharmaceutical warehouse - critical supply, high service level",
        annual_demand=2000,
        order_cost=200.0,
        holding_cost=40.0,
        lead_time_days=5,
        demand_std_daily=3.0,
        service_level=0.995,
    ),
]

PERISHABLE_SCENARIOS: List[PerishableScenario] = [
    PerishableScenario(
        name="bakery_bread",
        description="Daily bread production - high margin, short shelf life",
        price=5.0,
        cost=1.5,
        salvage=0.2,
        demand_mean=200,
        demand_std=40,
    ),
    PerishableScenario(
        name="flower_shop",
        description="Fresh flower inventory - seasonal variation",
        price=25.0,
        cost=10.0,
        salvage=2.0,
        demand_mean=50,
        demand_std=15,
    ),
    PerishableScenario(
        name="seafood_market",
        description="Fresh seafood - premium pricing, high loss on unsold",
        price=40.0,
        cost=20.0,
        salvage=5.0,
        demand_mean=30,
        demand_std=10,
    ),
]

LOGISTICS_SCENARIOS: List[LogisticsScenario] = [
    LogisticsScenario(
        name="urban_delivery",
        description="Urban last-mile delivery - 8 stops in 5km radius",
        depot=0,
        stop_coords=[
            (0, 0),  # depot
            (1, 2), (2, 1), (3, 3), (4, 1),
            (2, 4), (1, 3), (3, 2), (4, 3),
        ],
    ),
    LogisticsScenario(
        name="regional_distribution",
        description="Regional distribution - 12 stops spread over 50km",
        depot=0,
        stop_coords=[
            (0, 0),  # depot
            (10, 5), (5, 15), (20, 10), (15, 25),
            (30, 5), (25, 20), (35, 15), (40, 30),
            (10, 35), (5, 25), (45, 10), (50, 25),
        ],
    ),
]

WAREHOUSE_SCENARIOS: List[WarehouseScenario] = [
    WarehouseScenario(
        name="shipping_containers",
        description="Packing items into shipping containers - mixed sizes",
        bin_capacity=100.0,
        item_sizes=[45, 35, 30, 25, 20, 18, 15, 12, 10, 8, 25, 40, 55],
    ),
    WarehouseScenario(
        name="pallet_loading",
        description="Loading pallets into truck - weight-based",
        bin_capacity=1000.0,
        item_sizes=[350, 280, 220, 180, 150, 420, 310, 95, 75, 450, 200, 160],
    ),
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

@dataclass
class InventoryResult:
    """Result from inventory scenario analysis."""
    name: str
    optimal_order_qty: float
    total_annual_cost: float
    orders_per_year: float
    reorder_point: float
    safety_stock: float
    cycle_stock: float
    total_avg_inventory: float
    inventory_turnover: float


def analyze_inventory(scenario: InventoryScenario, verbose: bool = False) -> InventoryResult:
    """Analyze an inventory management scenario."""
    # EOQ calculation
    eoq = calc_eoq(
        demand=scenario.annual_demand,
        setup_cost=scenario.order_cost,
        holding_cost=scenario.holding_cost,
        lead_time_days=scenario.lead_time_days,
    )
    if verbose:
        print(eoq)

    # Safety stock calculation (convert daily to period units matching lead time)
    daily_demand = scenario.annual_demand / 365
    ss = calc_safety_stock(
        demand_rate=daily_demand,
        demand_std=scenario.demand_std_daily,
        lead_time=scenario.lead_time_days,
        service_level=scenario.service_level,
    )
    if verbose:
        print(ss)

    cycle_stock = eoq.optimal_quantity / 2
    total_avg_inv = cycle_stock + ss.safety_stock
    turnover = scenario.annual_demand / total_avg_inv if total_avg_inv > 0 else float('inf')

    return InventoryResult(
        name=scenario.name,
        optimal_order_qty=eoq.optimal_quantity,
        total_annual_cost=eoq.total_cost,
        orders_per_year=eoq.num_orders_per_year,
        reorder_point=ss.reorder_point,
        safety_stock=ss.safety_stock,
        cycle_stock=cycle_stock,
        total_avg_inventory=total_avg_inv,
        inventory_turnover=turnover,
    )


@dataclass
class PerishableResult:
    """Result from perishable goods analysis."""
    name: str
    optimal_quantity: float
    critical_ratio: float
    expected_profit: float
    profit_margin_pct: float
    stockout_probability: float


def analyze_perishable(scenario: PerishableScenario, verbose: bool = False) -> PerishableResult:
    """Analyze a perishable goods (newsvendor) scenario."""
    nv = calc_newsvendor(
        price=scenario.price,
        cost=scenario.cost,
        salvage=scenario.salvage,
        demand_mean=scenario.demand_mean,
        demand_std=scenario.demand_std,
    )
    if verbose:
        print(nv)

    profit_margin = (scenario.price - scenario.cost) / scenario.price * 100
    stockout_prob = 1 - nv.critical_ratio

    return PerishableResult(
        name=scenario.name,
        optimal_quantity=nv.optimal_quantity,
        critical_ratio=nv.critical_ratio,
        expected_profit=nv.expected_profit,
        profit_margin_pct=profit_margin,
        stockout_probability=stockout_prob,
    )


@dataclass
class LogisticsResult:
    """Result from vehicle routing analysis."""
    name: str
    n_stops: int
    mst_lower_bound: float
    best_tour_length: float
    optimality_gap_pct: float
    best_algorithm: str


def analyze_logistics(scenario: LogisticsScenario, verbose: bool = False) -> LogisticsResult:
    """Analyze a vehicle routing scenario."""
    vr = calc_vehicle_routing(
        coordinates=scenario.stop_coords,
        depot=scenario.depot,
    )
    if verbose:
        print(vr)

    best_length = min(vr.nearest_neighbor_upper, vr.savings_upper)
    best_alg = "nearest_neighbor" if vr.nearest_neighbor_upper < vr.savings_upper else "clarke_wright"

    return LogisticsResult(
        name=scenario.name,
        n_stops=vr.n_stops,
        mst_lower_bound=vr.mst_lower_bound,
        best_tour_length=best_length,
        optimality_gap_pct=vr.gap_percent,
        best_algorithm=best_alg,
    )


@dataclass
class WarehouseResult:
    """Result from bin packing analysis."""
    name: str
    n_items: int
    bins_lower_bound: int
    bins_used: int
    is_optimal: bool
    utilization_pct: float


def analyze_warehouse(scenario: WarehouseScenario, verbose: bool = False) -> WarehouseResult:
    """Analyze a warehouse / bin packing scenario."""
    bp = calc_bin_packing(
        sizes=scenario.item_sizes,
        capacity=scenario.bin_capacity,
    )
    if verbose:
        print(bp)

    utilization = bp.total_size / (bp.ffd_bins_used * scenario.bin_capacity) * 100

    return WarehouseResult(
        name=scenario.name,
        n_items=bp.n_items,
        bins_lower_bound=bp.l2_lower_bound,
        bins_used=bp.ffd_bins_used,
        is_optimal=bp.l2_lower_bound == bp.ffd_bins_used,
        utilization_pct=utilization,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(
    inv_results: List[InventoryResult],
    per_results: List[PerishableResult],
    log_results: List[LogisticsResult],
    wh_results: List[WarehouseResult],
):
    """Print comprehensive supply chain analysis report."""
    print("\n" + "=" * 78)
    print("  SUPPLY CHAIN OPTIMIZATION LAB -- Analysis Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # Inventory Management
    print("\n  1. INVENTORY MANAGEMENT (EOQ + Safety Stock)")
    print(f"  {'Scenario':20s} {'EOQ':>8s} {'$/year':>10s} {'ROP':>8s} {'SS':>8s} {'Turns':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for r in inv_results:
        print(f"  {r.name:20s} {r.optimal_order_qty:8.1f} "
              f"${r.total_annual_cost:9.0f} {r.reorder_point:8.1f} "
              f"{r.safety_stock:8.1f} {r.inventory_turnover:6.1f}")

    # Perishable Goods
    print("\n  2. PERISHABLE GOODS (Newsvendor Model)")
    print(f"  {'Scenario':16s} {'Q*':>8s} {'CR':>6s} {'E[Profit]':>10s} {'Margin':>8s} {'P(SO)':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    for r in per_results:
        print(f"  {r.name:16s} {r.optimal_quantity:8.1f} {r.critical_ratio:6.3f} "
              f"${r.expected_profit:9.0f} {r.profit_margin_pct:7.1f}% "
              f"{r.stockout_probability:7.1%}")

    # Vehicle Routing
    print("\n  3. VEHICLE ROUTING (TSP Bounds)")
    print(f"  {'Scenario':20s} {'Stops':>6s} {'LB':>10s} {'Best':>10s} {'Gap':>8s} {'Method':>15s}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*15}")
    for r in log_results:
        print(f"  {r.name:20s} {r.n_stops:6d} {r.mst_lower_bound:10.2f} "
              f"{r.best_tour_length:10.2f} {r.optimality_gap_pct:7.1f}% {r.best_algorithm:>15s}")

    # Warehouse / Bin Packing
    print("\n  4. WAREHOUSE / BIN PACKING")
    print(f"  {'Scenario':20s} {'Items':>6s} {'LB':>6s} {'Used':>6s} {'Optimal':>8s} {'Util':>8s}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for r in wh_results:
        opt_str = "YES" if r.is_optimal else "NO"
        print(f"  {r.name:20s} {r.n_items:6d} {r.bins_lower_bound:6d} "
              f"{r.bins_used:6d} {opt_str:>8s} {r.utilization_pct:7.1f}%")

    # Summary
    print("\n" + "=" * 78)
    print("  Summary:")
    print(f"    Inventory scenarios: {len(inv_results)} analyzed")
    print(f"    Perishable scenarios: {len(per_results)} analyzed")
    print(f"    Logistics scenarios: {len(log_results)} analyzed")
    print(f"    Warehouse scenarios: {len(wh_results)} analyzed")

    total_cost_savings = sum(r.total_annual_cost for r in inv_results)
    total_expected_profit = sum(r.expected_profit for r in per_results)
    print(f"    Total inventory cost optimized: ${total_cost_savings:,.0f}/year")
    print(f"    Total expected profit (perishables): ${total_expected_profit:,.0f}")
    print("=" * 78 + "\n")


def save_results(
    inv_results: List[InventoryResult],
    per_results: List[PerishableResult],
    log_results: List[LogisticsResult],
    wh_results: List[WarehouseResult],
    outpath: Path,
):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "supply_chain_lab v0.1",
        "inventory": [asdict(r) for r in inv_results],
        "perishable": [asdict(r) for r in per_results],
        "logistics": [asdict(r) for r in log_results],
        "warehouse": [asdict(r) for r in wh_results],
        "summary": {
            "n_inventory": len(inv_results),
            "n_perishable": len(per_results),
            "n_logistics": len(log_results),
            "n_warehouse": len(wh_results),
        },
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Supply Chain Optimization Lab")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool output")
    args = parser.parse_args()

    print(f"\n  Analyzing {len(INVENTORY_SCENARIOS)} inventory + "
          f"{len(PERISHABLE_SCENARIOS)} perishable + "
          f"{len(LOGISTICS_SCENARIOS)} logistics + "
          f"{len(WAREHOUSE_SCENARIOS)} warehouse scenarios...")

    inv_results = []
    for scenario in INVENTORY_SCENARIOS:
        try:
            result = analyze_inventory(scenario, verbose=args.verbose)
            inv_results.append(result)
        except Exception as e:
            print(f"  ERROR (inventory): {scenario.name}: {e}")

    per_results = []
    for scenario in PERISHABLE_SCENARIOS:
        try:
            result = analyze_perishable(scenario, verbose=args.verbose)
            per_results.append(result)
        except Exception as e:
            print(f"  ERROR (perishable): {scenario.name}: {e}")

    log_results = []
    for scenario in LOGISTICS_SCENARIOS:
        try:
            result = analyze_logistics(scenario, verbose=args.verbose)
            log_results.append(result)
        except Exception as e:
            print(f"  ERROR (logistics): {scenario.name}: {e}")

    wh_results = []
    for scenario in WAREHOUSE_SCENARIOS:
        try:
            result = analyze_warehouse(scenario, verbose=args.verbose)
            wh_results.append(result)
        except Exception as e:
            print(f"  ERROR (warehouse): {scenario.name}: {e}")

    print_report(inv_results, per_results, log_results, wh_results)

    outpath = RESULTS_DIR / "optimization_results.json"
    save_results(inv_results, per_results, log_results, wh_results, outpath)


if __name__ == "__main__":
    main()
