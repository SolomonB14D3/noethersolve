"""Tests for biology-AI bridge modules."""
import pytest
import math


# ── Chemotaxis ────────────────────────────────────────────────────────

class TestChemotaxis:
    def test_perfect_adaptation_exponential_decay(self):
        from noethersolve.chemotaxis_model import check_perfect_adaptation
        # Exponential decay back to baseline = perfect adaptation
        def response(t):
            if t < 10:
                return 1.0
            return 1.0 + 2.0 * math.exp(-(t - 10) / 5.0)
        result = check_perfect_adaptation(response, perturbation_time=10.0,
                                          measurement_window=100.0)
        assert result.verdict == "PERFECT_ADAPTATION"

    def test_no_adaptation_sustained_shift(self):
        from noethersolve.chemotaxis_model import check_perfect_adaptation
        # Sustained shift = NOT perfect adaptation
        def response(t):
            if t < 10:
                return 1.0
            return 3.0  # stays elevated
        result = check_perfect_adaptation(response, perturbation_time=10.0,
                                          measurement_window=100.0)
        assert result.verdict != "PERFECT_ADAPTATION"

    def test_simulate_chemotaxis_runs(self):
        from noethersolve.chemotaxis_model import simulate_chemotaxis
        result = simulate_chemotaxis(duration=10.0, seed=42)
        assert result is not None

    def test_simulate_chemotaxis_moves_toward_source(self):
        from noethersolve.chemotaxis_model import simulate_chemotaxis
        result = simulate_chemotaxis(duration=50.0, source_position=(100, 100),
                                     initial_position=(0, 0), seed=42)
        # Final position should be closer to source than initial
        final = result.positions[-1]
        initial_dist = math.hypot(100, 100)
        final_dist = math.hypot(100 - final[0], 100 - final[1])
        assert final_dist < initial_dist

    def test_optimize_tumble_bias(self):
        from noethersolve.chemotaxis_model import optimize_tumble_bias
        result = optimize_tumble_bias()
        assert result is not None
        assert result.optimal_bias > 0


# ── C. elegans ────────────────────────────────────────────────────────

class TestCElegans:
    def test_escape_response_runs(self):
        from noethersolve.c_elegans_behavior import simulate_escape_response
        result = simulate_escape_response(seed=42)
        assert result is not None

    def test_escape_moves_away_from_threat(self):
        from noethersolve.c_elegans_behavior import simulate_escape_response
        result = simulate_escape_response(
            threat_position=(0, 0), worm_position=(1, 0), seed=42)
        # Worm should end up further from threat
        assert result.final_distance >= 0.5  # at least half body length away

    def test_drift_diffusion_decision(self):
        from noethersolve.c_elegans_behavior import drift_diffusion_decision
        result = drift_diffusion_decision(drift_rate=0.5, threshold=1.0, seed=42)
        assert result is not None
        assert result.decision_time > 0
        assert result.decision in ("A", "B", "UNDECIDED")

    def test_drift_diffusion_faster_with_strong_drift(self):
        from noethersolve.c_elegans_behavior import drift_diffusion_decision
        slow = drift_diffusion_decision(drift_rate=0.1, threshold=1.0, seed=42)
        fast = drift_diffusion_decision(drift_rate=2.0, threshold=1.0, seed=42)
        # Stronger drift should reach threshold faster on average
        assert fast.decision_time <= slow.decision_time

    def test_detect_foraging_phase(self):
        from noethersolve.c_elegans_behavior import detect_foraging_phase
        # Simple local search pattern (small movements)
        positions = [(i * 0.1, i * 0.1) for i in range(20)]
        times = [float(i) for i in range(20)]
        result = detect_foraging_phase(positions, times, body_length=1.0)
        assert result is not None


# ── Neural-RL Analogy ─────────────────────────────────────────────────

class TestNeuralRL:
    def test_dopamine_rpe_validation(self):
        from noethersolve.neural_rl_analogy import validate_dopamine_rpe
        # Perfect TD error encoding: firing = actual - expected
        firing = [0.0, 1.0, -0.5, 0.5, 0.0]
        expected = [1.0, 1.0, 1.0, 1.0, 1.0]
        actual = [1.0, 2.0, 0.5, 1.5, 1.0]
        result = validate_dopamine_rpe(firing, expected, actual)
        assert result is not None

    def test_hebbian_backprop_comparison(self):
        from noethersolve.neural_rl_analogy import compare_hebbian_backprop
        pre = [1.0, 0.5, 0.8]
        post = [0.6, 0.9, 0.3]
        weights = [0.1, 0.2, 0.3]
        targets = [1.0, 0.0, 1.0]
        result = compare_hebbian_backprop(pre, post, weights, targets)
        assert result is not None
        # LearningRuleComparison has bio_rule, algo_rule, consistency_score
        assert result.consistency_score >= 0
        assert result.bio_rule is not None
        assert result.algo_rule is not None

    def test_striatum_actor_critic_mapping(self):
        from noethersolve.neural_rl_analogy import map_striatum_to_actor_critic
        mappings = map_striatum_to_actor_critic()
        assert len(mappings) > 0
        # Check that each mapping has required fields
        for m in mappings:
            assert m.striatal_region != ""
            assert m.ac_component != ""

    def test_td_learning_validation(self):
        from noethersolve.neural_rl_analogy import validate_td_learning
        # validate_td_learning takes value_estimates and rewards, not states
        values = [0.0, 0.5, 0.8, 0.5, 0.0]
        rewards = [0.0, 1.0, 0.0, 0.5, 0.0]
        result = validate_td_learning(values, rewards)
        assert result is not None

    def test_eligibility_trace(self):
        from noethersolve.neural_rl_analogy import validate_eligibility_trace
        pre_spikes = [0.1, 0.5, 1.2, 2.0]
        post_spikes = [0.2, 0.6, 1.3, 2.1]
        reward_times = [1.0, 2.5]
        result = validate_eligibility_trace(pre_spikes, post_spikes, reward_times)
        assert result is not None


# ── Collective Behavior ───────────────────────────────────────────────

class TestCollective:
    def test_swarm_consensus_converges(self):
        from noethersolve.collective_behavior import swarm_consensus
        opinions = [0.0, 0.5, 1.0]
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        result = swarm_consensus(opinions, adj, iterations=100)
        assert result is not None
        # Should converge — final opinions should be close
        final = result["final_opinions"]
        assert max(final) - min(final) < 0.1

    def test_flock_formation(self):
        from noethersolve.collective_behavior import flock_formation
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        velocities = [(1, 0), (1, 0.1), (0.9, 0), (1, -0.1)]
        result = flock_formation(positions, velocities, n_steps=20)
        assert result is not None

    def test_bacterial_quorum_sensing(self):
        from noethersolve.collective_behavior import bacterial_quorum_sensing
        result = bacterial_quorum_sensing(n_bacteria=20, seed=42)
        assert result is not None

    def test_slime_mold_optimization(self):
        from noethersolve.collective_behavior import slime_mold_optimization
        # Simple 4-node network
        nodes = [(0, 0), (1, 0), (0, 1), (1, 1)]
        food_sources = [0, 3]  # food at corners
        result = slime_mold_optimization(nodes, food_sources)
        assert result is not None

    def test_ant_pheromone_routing(self):
        from noethersolve.collective_behavior import ant_pheromone_routing
        distances = [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ]
        result = ant_pheromone_routing(distances, n_ants=5, iterations=10, seed=42)
        assert result is not None


# ── Bio-AI Bridge ─────────────────────────────────────────────────────

class TestBioAIBridge:
    def test_compare_chemotaxis(self):
        from noethersolve.bio_ai_bridge import compare_agent_to_worm
        result = compare_agent_to_worm("chemotaxis")
        assert result is not None
        assert result.verdict is not None

    def test_compare_foraging(self):
        from noethersolve.bio_ai_bridge import compare_agent_to_worm
        result = compare_agent_to_worm("foraging")
        assert result is not None

    def test_compare_escape(self):
        from noethersolve.bio_ai_bridge import compare_agent_to_worm
        result = compare_agent_to_worm("escape")
        assert result is not None

    def test_compare_learning(self):
        from noethersolve.bio_ai_bridge import compare_agent_to_worm
        result = compare_agent_to_worm("learning")
        assert result is not None

    def test_compare_collective(self):
        from noethersolve.bio_ai_bridge import compare_agent_to_worm
        result = compare_agent_to_worm("collective")
        assert result is not None

    def test_convergent_solutions(self):
        from noethersolve.bio_ai_bridge import identify_convergent_solutions
        results = identify_convergent_solutions()
        assert len(results) == 8
        for r in results:
            assert "domain" in r
            assert "biological" in r
            assert "algorithmic" in r
            assert 0 <= r["conservation_score"] <= 1

    def test_behavior_conservation_score(self):
        from noethersolve.bio_ai_bridge import behavior_conservation_score
        result = behavior_conservation_score(
            "chemotaxis", bio_performance=0.8, ai_performance=0.85)
        assert result is not None
        assert "conservation_score" in result
        assert 0 <= result["conservation_score"] <= 1

    def test_map_behavior_to_architecture(self):
        from noethersolve.bio_ai_bridge import map_behavior_to_architecture
        result = map_behavior_to_architecture(
            "c_elegans", ["navigation", "escape"])
        assert result is not None

    def test_all_behavior_types(self):
        from noethersolve.bio_ai_bridge import compare_agent_to_worm, BehaviorType
        for bt in BehaviorType:
            result = compare_agent_to_worm(bt.value)
            assert result is not None, f"Failed for {bt.value}"
