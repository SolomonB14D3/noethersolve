"""Tests for delivery_optimization module — EOQ, vehicle routing, newsvendor,
safety stock, and bin packing calculators."""

import math
import pytest

from noethersolve.delivery_optimization import (
    EOQReport,
    calc_eoq,
    VehicleRoutingReport,
    calc_vehicle_routing,
    NewsvendorReport,
    calc_newsvendor,
    SafetyStockReport,
    calc_safety_stock,
    BinPackingReport,
    calc_bin_packing,
    _norm_cdf,
    _norm_ppf,
)


# ── EOQ Tests ────────────────────────────────────────────────────────


class TestEOQ:
    def test_basic_eoq(self):
        """Classic textbook: D=10000, S=50, H=2.5 → Q*=632.46."""
        r = calc_eoq(10000, 50, 2.5)
        assert isinstance(r, EOQReport)
        expected_q = math.sqrt(2 * 10000 * 50 / 2.5)
        assert abs(r.optimal_quantity - expected_q) < 0.01

    def test_ordering_equals_holding_at_eoq(self):
        """At EOQ, ordering cost = holding cost (fundamental EOQ property)."""
        r = calc_eoq(5000, 100, 4.0)
        assert abs(r.ordering_cost - r.holding_cost_total) < 0.01

    def test_total_cost(self):
        r = calc_eoq(10000, 50, 2.5)
        assert abs(r.total_cost - (r.ordering_cost + r.holding_cost_total)) < 0.01

    def test_reorder_point_with_lead_time(self):
        r = calc_eoq(3650, 25, 5.0, lead_time_days=10)
        # Daily demand = 3650/365 = 10
        assert r.reorder_point is not None
        assert abs(r.reorder_point - 100.0) < 0.01

    def test_reorder_point_none_without_lead_time(self):
        r = calc_eoq(1000, 10, 1.0)
        assert r.reorder_point is None

    def test_cycle_time(self):
        r = calc_eoq(1000, 50, 5.0)
        assert abs(r.cycle_time - r.optimal_quantity / 1000) < 1e-6

    def test_num_orders(self):
        r = calc_eoq(1200, 30, 3.0)
        assert abs(r.num_orders_per_year - 1200 / r.optimal_quantity) < 1e-6

    def test_invalid_demand(self):
        with pytest.raises(ValueError, match="Demand must be positive"):
            calc_eoq(0, 50, 2.5)

    def test_invalid_setup_cost(self):
        with pytest.raises(ValueError, match="Setup cost must be positive"):
            calc_eoq(1000, -10, 2.0)

    def test_invalid_holding_cost(self):
        with pytest.raises(ValueError, match="Holding cost must be positive"):
            calc_eoq(1000, 50, 0)

    def test_negative_lead_time(self):
        with pytest.raises(ValueError, match="Lead time must be non-negative"):
            calc_eoq(1000, 50, 2.5, lead_time_days=-5)

    def test_str_output(self):
        r = calc_eoq(10000, 50, 2.5)
        s = str(r)
        assert "Economic Order Quantity" in s
        assert "Optimal order quantity" in s


# ── Vehicle Routing Tests ────────────────────────────────────────────


class TestVehicleRouting:
    def test_square_tour(self):
        """4 corners of unit square: optimal tour = 4.0."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        r = calc_vehicle_routing(coords)
        assert isinstance(r, VehicleRoutingReport)
        assert r.n_stops == 4
        # MST of unit square = 3.0 (three edges)
        assert abs(r.mst_lower_bound - 3.0) < 0.01
        # Both heuristics should find the 4.0 tour (or close)
        assert r.nearest_neighbor_upper <= 5.0  # NN can be suboptimal
        assert r.savings_upper <= 5.0

    def test_mst_is_lower_bound(self):
        coords = [(0, 0), (3, 0), (3, 4), (0, 4), (1.5, 2)]
        r = calc_vehicle_routing(coords)
        assert r.mst_lower_bound <= r.nearest_neighbor_upper
        assert r.mst_lower_bound <= r.savings_upper

    def test_collinear_points(self):
        """Points on a line: optimal = 2 * max_distance."""
        coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
        r = calc_vehicle_routing(coords)
        # Optimal tour for collinear points = 2 * 3 = 6
        assert r.mst_lower_bound <= 6.01
        assert r.nearest_neighbor_upper >= r.mst_lower_bound

    def test_two_stops(self):
        coords = [(0, 0), (3, 4)]
        r = calc_vehicle_routing(coords)
        assert abs(r.mst_lower_bound - 5.0) < 0.01
        # Tour: 0→1→0 = 10.0
        assert abs(r.nearest_neighbor_upper - 10.0) < 0.01

    def test_tour_starts_and_ends_at_depot(self):
        coords = [(0, 0), (1, 0), (1, 1)]
        r = calc_vehicle_routing(coords)
        assert r.nearest_neighbor_tour[0] == 0
        assert r.nearest_neighbor_tour[-1] == 0
        assert r.savings_tour[0] == 0
        assert r.savings_tour[-1] == 0

    def test_too_few_stops(self):
        with pytest.raises(ValueError, match="at least 2"):
            calc_vehicle_routing([(0, 0)])

    def test_invalid_depot(self):
        with pytest.raises(ValueError, match="out of range"):
            calc_vehicle_routing([(0, 0), (1, 1)], depot=5)

    def test_gap_is_nonnegative(self):
        coords = [(0, 0), (2, 0), (4, 0), (6, 0), (3, 3)]
        r = calc_vehicle_routing(coords)
        assert r.gap_percent >= 0

    def test_str_output(self):
        coords = [(0, 0), (1, 0), (1, 1)]
        r = calc_vehicle_routing(coords)
        s = str(r)
        assert "Vehicle Routing Bounds" in s
        assert "MST lower bound" in s


# ── Newsvendor Tests ─────────────────────────────────────────────────


class TestNewsvendor:
    def test_critical_ratio_half(self):
        """When CR = 0.5, Q* = mean (z=0)."""
        r = calc_newsvendor(price=10, cost=6, salvage=2, demand_mean=100, demand_std=20)
        assert abs(r.critical_ratio - 0.5) < 1e-6
        assert abs(r.optimal_quantity - 100.0) < 0.1

    def test_high_margin_orders_more(self):
        """Higher profit margin → order more than mean."""
        r = calc_newsvendor(price=20, cost=6, salvage=2, demand_mean=100, demand_std=20)
        assert r.optimal_quantity > 100  # CR > 0.5 → z > 0

    def test_low_margin_orders_less(self):
        """Low margin → order less than mean."""
        r = calc_newsvendor(price=8, cost=6, salvage=2, demand_mean=100, demand_std=20)
        assert r.optimal_quantity < 100  # CR < 0.5 → z < 0

    def test_underage_overage_costs(self):
        r = calc_newsvendor(price=15, cost=8, salvage=3, demand_mean=200, demand_std=50)
        assert abs(r.underage_cost - 7.0) < 1e-6  # p - c
        assert abs(r.overage_cost - 5.0) < 1e-6   # c - s

    def test_critical_ratio_formula(self):
        r = calc_newsvendor(price=15, cost=8, salvage=3, demand_mean=200, demand_std=50)
        expected_cr = (15 - 8) / (15 - 3)
        assert abs(r.critical_ratio - expected_cr) < 1e-6

    def test_expected_profit_positive(self):
        r = calc_newsvendor(price=20, cost=5, salvage=1, demand_mean=500, demand_std=100)
        assert r.expected_profit > 0

    def test_price_below_cost_error(self):
        with pytest.raises(ValueError, match="Price.*must exceed cost"):
            calc_newsvendor(price=5, cost=10, salvage=2, demand_mean=100, demand_std=20)

    def test_salvage_above_cost_error(self):
        with pytest.raises(ValueError, match="Salvage.*must not exceed cost"):
            calc_newsvendor(price=15, cost=5, salvage=8, demand_mean=100, demand_std=20)

    def test_negative_demand_std_error(self):
        with pytest.raises(ValueError, match="Demand std must be positive"):
            calc_newsvendor(price=10, cost=5, salvage=2, demand_mean=100, demand_std=0)

    def test_str_output(self):
        r = calc_newsvendor(price=10, cost=6, salvage=2, demand_mean=100, demand_std=20)
        s = str(r)
        assert "Newsvendor Model" in s
        assert "Critical ratio" in s


# ── Safety Stock Tests ───────────────────────────────────────────────


class TestSafetyStock:
    def test_basic_safety_stock(self):
        """SS = z * sigma * sqrt(L). For 95% SL, z ≈ 1.645."""
        r = calc_safety_stock(200, 40, 4, 0.95)
        z_95 = _norm_ppf(0.95)
        expected_ss = z_95 * 40 * math.sqrt(4)
        assert abs(r.safety_stock - expected_ss) < 0.1
        # z ≈ 1.645, SS ≈ 1.645 * 40 * 2 = 131.6
        assert abs(r.safety_stock - 131.6) < 1.0

    def test_reorder_point(self):
        r = calc_safety_stock(100, 20, 9, 0.95)
        expected_rop = 100 * 9 + r.safety_stock
        assert abs(r.reorder_point - expected_rop) < 0.01

    def test_higher_service_more_stock(self):
        r1 = calc_safety_stock(100, 20, 4, 0.90)
        r2 = calc_safety_stock(100, 20, 4, 0.99)
        assert r2.safety_stock > r1.safety_stock

    def test_longer_lead_time_more_stock(self):
        r1 = calc_safety_stock(100, 20, 1, 0.95)
        r2 = calc_safety_stock(100, 20, 16, 0.95)
        assert r2.safety_stock > r1.safety_stock

    def test_lead_time_demand_std(self):
        r = calc_safety_stock(100, 30, 4, 0.95)
        assert abs(r.lead_time_demand_std - 30 * math.sqrt(4)) < 0.01

    def test_zero_variability(self):
        r = calc_safety_stock(100, 0, 5, 0.95)
        assert abs(r.safety_stock) < 0.01  # no variability → no safety stock

    def test_invalid_service_level(self):
        with pytest.raises(ValueError, match="Service level"):
            calc_safety_stock(100, 20, 4, 1.0)
        with pytest.raises(ValueError, match="Service level"):
            calc_safety_stock(100, 20, 4, 0.0)

    def test_invalid_lead_time(self):
        with pytest.raises(ValueError, match="Lead time must be positive"):
            calc_safety_stock(100, 20, 0, 0.95)

    def test_str_output(self):
        r = calc_safety_stock(200, 40, 4, 0.95)
        s = str(r)
        assert "Safety Stock" in s
        assert "Reorder point" in s


# ── Bin Packing Tests ────────────────────────────────────────────────


class TestBinPacking:
    def test_exact_fit(self):
        """Items exactly fill bins."""
        r = calc_bin_packing([1.0, 1.0, 1.0], 1.0)
        assert r.l1_lower_bound == 3
        assert r.ffd_bins_used == 3

    def test_basic_packing(self):
        sizes = [0.7, 0.5, 0.3, 0.2, 0.8, 0.4]
        r = calc_bin_packing(sizes, 1.0)
        assert isinstance(r, BinPackingReport)
        # sum = 2.9, L1 = 3
        assert r.l1_lower_bound == 3
        assert r.ffd_bins_used >= r.l1_lower_bound
        assert r.l2_lower_bound >= r.l1_lower_bound

    def test_ffd_assignment_covers_all_items(self):
        sizes = [0.4, 0.3, 0.5, 0.2, 0.6, 0.1]
        r = calc_bin_packing(sizes, 1.0)
        all_items = set()
        for b in r.ffd_assignment:
            all_items.update(b)
        assert all_items == set(range(len(sizes)))

    def test_no_bin_exceeds_capacity(self):
        sizes = [0.3, 0.7, 0.4, 0.6, 0.5, 0.2, 0.8]
        r = calc_bin_packing(sizes, 1.0)
        for b in r.ffd_assignment:
            bin_total = sum(sizes[i] for i in b)
            assert bin_total <= 1.0 + 1e-9

    def test_single_item(self):
        r = calc_bin_packing([0.5], 1.0)
        assert r.ffd_bins_used == 1
        assert r.l1_lower_bound == 1

    def test_all_half_capacity(self):
        """10 items of size 0.5 in capacity-1 bins → need 5 bins."""
        r = calc_bin_packing([0.5] * 10, 1.0)
        assert r.l1_lower_bound == 5
        assert r.ffd_bins_used == 5

    def test_l2_can_exceed_l1(self):
        """L2 can be tighter than L1 for carefully constructed instances."""
        # 4 items of 0.4 and 2 items of 0.7 in capacity 1.0
        # L1 = ceil(4*0.4 + 2*0.7) / 1.0 = ceil(3.0) = 3
        # But each 0.7 needs its own bin with at most one 0.3 left
        # So we need at least 2 bins for 0.7s + bins for remaining
        sizes = [0.7, 0.7, 0.4, 0.4, 0.4, 0.4]
        r = calc_bin_packing(sizes, 1.0)
        assert r.l2_lower_bound >= r.l1_lower_bound

    def test_item_exceeds_capacity_error(self):
        with pytest.raises(ValueError, match="exceeds capacity"):
            calc_bin_packing([0.5, 1.5], 1.0)

    def test_empty_items_error(self):
        with pytest.raises(ValueError, match="at least one item"):
            calc_bin_packing([], 1.0)

    def test_zero_capacity_error(self):
        with pytest.raises(ValueError, match="Capacity must be positive"):
            calc_bin_packing([0.5], 0)

    def test_ffd_optimal_example(self):
        """When FFD matches lower bound, gap string says optimal."""
        r = calc_bin_packing([1.0, 1.0], 1.0)
        assert "optimal" in r.gap.lower() or r.ffd_bins_used == r.l2_lower_bound

    def test_str_output(self):
        r = calc_bin_packing([0.7, 0.3, 0.5], 1.0)
        s = str(r)
        assert "Bin Packing" in s
        assert "FFD heuristic" in s


# ── Normal Distribution Utilities ────────────────────────────────────


class TestNormalUtils:
    def test_norm_cdf_symmetry(self):
        assert abs(_norm_cdf(0) - 0.5) < 1e-10

    def test_norm_cdf_extremes(self):
        assert _norm_cdf(10) > 0.999
        assert _norm_cdf(-10) < 0.001

    def test_norm_ppf_inverse(self):
        """ppf and cdf should be inverses."""
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            z = _norm_ppf(p)
            p_back = _norm_cdf(z)
            assert abs(p_back - p) < 1e-6, f"Failed for p={p}: got {p_back}"

    def test_norm_ppf_known_values(self):
        assert abs(_norm_ppf(0.5)) < 1e-6
        assert abs(_norm_ppf(0.975) - 1.96) < 0.01
        assert abs(_norm_ppf(0.95) - 1.645) < 0.01
        assert abs(_norm_ppf(0.99) - 2.326) < 0.01
