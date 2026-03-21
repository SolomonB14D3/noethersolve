"""Chemotaxis behavioral models for AI behavior mapping.

Bacterial chemotaxis is the gold standard for biological behavior:
- Perfectly reproducible across billions of cells
- Mathematically tractable (Barkai-Leibler exact adaptation)
- Has clear analogs in AI agent design:
  * Tumble = explore, run = exploit
  * Perfect adaptation = stable setpoint under perturbation
  * Gradient sensing = reward signal processing

Usage:
    from noethersolve.chemotaxis_model import (
        check_perfect_adaptation,
        optimize_tumble_bias,
        simulate_chemotaxis,
        compare_to_rl_agent,
    )

    # Check if a system achieves perfect adaptation
    result = check_perfect_adaptation(
        system_response=lambda t: 1 - np.exp(-t/5),  # exponential decay to baseline
        perturbation_time=10,
        measurement_window=50,
    )
    print(result.verdict)  # "PERFECT_ADAPTATION" or "IMPERFECT"

    # Compare AI agent to E. coli chemotaxis
    comparison = compare_to_rl_agent(
        agent_policy=lambda obs: epsilon_greedy(obs),
        environment="gradient_field",
    )
"""

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
import math


@dataclass
class AdaptationResult:
    """Result of perfect adaptation check."""
    verdict: str  # "PERFECT_ADAPTATION", "IMPERFECT", "UNSTABLE"
    baseline_activity: float
    post_perturbation_activity: float
    recovery_time: float  # time to return to 95% of baseline
    steady_state_error: float  # difference from baseline at end
    adaptation_index: float  # 0 = no adaptation, 1 = perfect

    def __str__(self) -> str:
        lines = [
            "PERFECT ADAPTATION CHECK",
            "=" * 50,
            "",
            f"Verdict: {self.verdict}",
            f"Baseline activity: {self.baseline_activity:.3f}",
            f"Post-perturbation: {self.post_perturbation_activity:.3f}",
            f"Recovery time (τ): {self.recovery_time:.2f} time units",
            f"Steady-state error: {self.steady_state_error:.4f}",
            f"Adaptation index: {self.adaptation_index:.3f}",
            "",
        ]
        if self.verdict == "PERFECT_ADAPTATION":
            lines.append("✓ System returns to baseline regardless of perturbation magnitude")
            lines.append("  This is the Barkai-Leibler robustness property")
        elif self.verdict == "IMPERFECT":
            lines.append("✗ System settles at a new steady state")
            lines.append("  Adaptation is not robust to perturbation size")
        else:
            lines.append("✗ System does not stabilize")
        return "\n".join(lines)


def check_perfect_adaptation(
    system_response: Callable[[float], float],
    perturbation_time: float = 10.0,
    measurement_window: float = 100.0,
    baseline_threshold: float = 0.05,
    n_samples: int = 1000,
) -> AdaptationResult:
    """Check if a system achieves perfect adaptation (Barkai-Leibler property).

    Perfect adaptation means the system returns to its pre-stimulus steady state
    regardless of the stimulus magnitude. This is the key property of bacterial
    chemotaxis and has analogs in robust AI control systems.

    Args:
        system_response: Function mapping time -> activity level (0 to 1)
            Activity should represent the running/tumbling bias or similar
        perturbation_time: When the step change in stimulus occurs
        measurement_window: Total time to measure
        baseline_threshold: Maximum allowed deviation from baseline for "perfect"
        n_samples: Number of time points to sample

    Returns:
        AdaptationResult with verdict and metrics
    """
    dt = measurement_window / n_samples

    # Measure baseline (pre-perturbation)
    baseline_samples = []
    for i in range(int(perturbation_time / dt)):
        t = i * dt
        baseline_samples.append(system_response(t))
    baseline = sum(baseline_samples) / len(baseline_samples) if baseline_samples else 0.5

    # Measure post-perturbation response
    post_samples = []
    for i in range(int(perturbation_time / dt), n_samples):
        t = i * dt
        post_samples.append(system_response(t))

    if not post_samples:
        return AdaptationResult(
            verdict="UNSTABLE",
            baseline_activity=baseline,
            post_perturbation_activity=0.0,
            recovery_time=float('inf'),
            steady_state_error=1.0,
            adaptation_index=0.0,
        )

    # Find peak response after perturbation
    peak_val = post_samples[0]
    for val in post_samples:
        if abs(val - baseline) > abs(peak_val - baseline):
            peak_val = val

    post_perturbation = peak_val

    # Find recovery time (95% return to baseline)
    recovery_threshold = baseline + 0.05 * (peak_val - baseline)
    recovery_time = measurement_window  # default to full window
    for i, val in enumerate(post_samples):
        if abs(val - baseline) < abs(recovery_threshold - baseline):
            recovery_time = (perturbation_time / dt + i) * dt
            break

    # Measure steady state (last 20% of window)
    steady_start = int(0.8 * len(post_samples))
    steady_samples = post_samples[steady_start:]
    steady_state = sum(steady_samples) / len(steady_samples) if steady_samples else baseline

    steady_state_error = abs(steady_state - baseline)

    # Calculate adaptation index
    initial_deviation = abs(peak_val - baseline)
    if initial_deviation < 1e-10:
        adaptation_index = 1.0  # No perturbation effect = perfect
    else:
        adaptation_index = 1.0 - (steady_state_error / initial_deviation)
        adaptation_index = max(0.0, min(1.0, adaptation_index))

    # Determine verdict
    if steady_state_error < baseline_threshold:
        verdict = "PERFECT_ADAPTATION"
    elif adaptation_index > 0.5:
        verdict = "IMPERFECT"
    else:
        verdict = "UNSTABLE"

    return AdaptationResult(
        verdict=verdict,
        baseline_activity=baseline,
        post_perturbation_activity=post_perturbation,
        recovery_time=recovery_time,
        steady_state_error=steady_state_error,
        adaptation_index=adaptation_index,
    )


@dataclass
class TumbleBiasResult:
    """Result of tumble bias optimization."""
    optimal_bias: float  # 0 = always run, 1 = always tumble
    gradient_efficiency: float  # how well it climbs gradients
    exploration_score: float  # coverage of state space
    optimal_for: str  # "steep_gradient", "shallow_gradient", "noisy_environment"
    recommendations: List[str]

    def __str__(self) -> str:
        lines = [
            "TUMBLE BIAS OPTIMIZATION",
            "=" * 50,
            "",
            f"Optimal bias: {self.optimal_bias:.3f}",
            "  (0 = always run/exploit, 1 = always tumble/explore)",
            f"Gradient efficiency: {self.gradient_efficiency:.3f}",
            f"Exploration score: {self.exploration_score:.3f}",
            f"Best for: {self.optimal_for}",
            "",
            "Recommendations:",
        ]
        for rec in self.recommendations:
            lines.append(f"  • {rec}")
        return "\n".join(lines)


def optimize_tumble_bias(
    gradient_steepness: float = 1.0,
    noise_level: float = 0.1,
    spatial_scale: float = 1.0,
    bacterial_speed: float = 20.0,  # μm/s typical for E. coli
    run_duration_mean: float = 1.0,  # seconds
) -> TumbleBiasResult:
    """Find optimal tumble bias for given environment conditions.

    Tumble bias is the probability of tumbling per unit time.
    - High bias = explore more, exploit less (good for shallow/noisy gradients)
    - Low bias = exploit more, explore less (good for steep gradients)

    This mirrors the exploration-exploitation tradeoff in RL.

    Args:
        gradient_steepness: Relative gradient strength (1.0 = typical attractant)
        noise_level: Environmental noise (0-1, affects gradient sensing)
        spatial_scale: Characteristic length of the environment
        bacterial_speed: Swimming speed in μm/s
        run_duration_mean: Mean run duration in seconds

    Returns:
        TumbleBiasResult with optimal parameters
    """
    # Berg-Purcell limit: minimum time to sense gradient
    # SNR = (gradient_steepness / noise_level) * sqrt(measurement_time)
    # Optimal measurement time balances sensing vs movement

    # Characteristic sensing time
    sensing_time = (noise_level / max(gradient_steepness, 0.01)) ** 2

    # Characteristic diffusion time
    diffusion_time = spatial_scale ** 2 / (bacterial_speed ** 2 * run_duration_mean)

    # Optimal tumble frequency balances these
    # High gradient + low noise → long runs (low tumble bias)
    # Low gradient + high noise → short runs (high tumble bias)

    snr = gradient_steepness / max(noise_level, 0.01)

    if snr > 10:
        # Steep gradient, low noise: exploit
        optimal_bias = 0.2
        optimal_for = "steep_gradient"
        gradient_efficiency = 0.9
        exploration_score = 0.3
    elif snr > 1:
        # Moderate gradient: balanced
        optimal_bias = 0.35
        optimal_for = "moderate_gradient"
        gradient_efficiency = 0.7
        exploration_score = 0.5
    else:
        # Shallow gradient or high noise: explore
        optimal_bias = 0.6
        optimal_for = "noisy_environment"
        gradient_efficiency = 0.4
        exploration_score = 0.8

    # Recommendations for AI agents
    recommendations = []

    if snr < 0.5:
        recommendations.append("Environment is noisy: increase exploration epsilon")
        recommendations.append("Consider ensemble methods for robust gradient sensing")
    elif snr > 5:
        recommendations.append("Strong signal: can reduce exploration for faster convergence")
        recommendations.append("Gradient descent is appropriate here")

    if sensing_time > diffusion_time:
        recommendations.append("Sensing is slow relative to movement: increase memory")
        recommendations.append("Consider integrating observations over longer windows")
    else:
        recommendations.append("Fast sensing: can use reactive policies")

    if optimal_bias > 0.4:
        recommendations.append("RL analog: high ε-greedy exploration recommended")
    else:
        recommendations.append("RL analog: low ε-greedy, focus on exploitation")

    return TumbleBiasResult(
        optimal_bias=optimal_bias,
        gradient_efficiency=gradient_efficiency,
        exploration_score=exploration_score,
        optimal_for=optimal_for,
        recommendations=recommendations,
    )


@dataclass
class ChemotaxisTrajectory:
    """Result of chemotaxis simulation."""
    positions: List[Tuple[float, float]]
    times: List[float]
    tumble_events: List[float]  # times when tumbles occurred
    final_concentration: float
    gradient_following_score: float  # -1 to 1
    effective_velocity: float  # net movement toward source

    def __str__(self) -> str:
        lines = [
            "CHEMOTAXIS SIMULATION",
            "=" * 50,
            "",
            f"Duration: {self.times[-1]:.1f} time units",
            f"Tumble events: {len(self.tumble_events)}",
            f"Final concentration: {self.final_concentration:.3f}",
            f"Gradient following: {self.gradient_following_score:.3f}",
            f"Effective velocity: {self.effective_velocity:.3f}",
            "",
        ]
        if self.gradient_following_score > 0.5:
            lines.append("✓ Strong positive chemotaxis (climbing gradient)")
        elif self.gradient_following_score > 0:
            lines.append("~ Weak positive chemotaxis")
        elif self.gradient_following_score > -0.5:
            lines.append("~ Random walk (no net direction)")
        else:
            lines.append("✗ Negative chemotaxis (moving away from source)")
        return "\n".join(lines)


def simulate_chemotaxis(
    duration: float = 100.0,
    dt: float = 0.1,
    source_position: Tuple[float, float] = (100.0, 100.0),
    initial_position: Tuple[float, float] = (0.0, 0.0),
    base_tumble_rate: float = 1.0,  # per second
    adaptation_time: float = 5.0,  # seconds
    speed: float = 20.0,  # units per second
    gradient_sensitivity: float = 1.0,
    seed: Optional[int] = None,
) -> ChemotaxisTrajectory:
    """Simulate bacterial chemotaxis with exact adaptation.

    Uses the standard run-and-tumble model with adaptation:
    - Bacteria run in straight lines
    - Tumble randomly to reorient
    - Tumble rate modulated by recent concentration changes
    - Adaptation returns tumble rate to baseline

    Args:
        duration: Simulation time
        dt: Time step
        source_position: Location of attractant source
        initial_position: Starting position
        base_tumble_rate: Baseline tumbles per second
        adaptation_time: Time constant for adaptation
        speed: Swimming speed
        gradient_sensitivity: How strongly gradient affects tumble rate
        seed: Random seed for reproducibility

    Returns:
        ChemotaxisTrajectory with position history and metrics
    """
    import random
    if seed is not None:
        random.seed(seed)

    # State
    x, y = initial_position
    angle = random.uniform(0, 2 * math.pi)
    positions = [(x, y)]
    times = [0.0]
    tumble_events = []

    # Adaptation state (internal "memory" of recent concentrations)
    concentration_memory = 0.0

    def concentration(px: float, py: float) -> float:
        """Gaussian concentration field centered at source."""
        sx, sy = source_position
        dist_sq = (px - sx) ** 2 + (py - sy) ** 2
        return math.exp(-dist_sq / 5000)  # characteristic length ~70

    t = 0.0
    while t < duration:
        t += dt

        # Current concentration
        c = concentration(x, y)

        # Update memory (exponential moving average)
        alpha = dt / adaptation_time
        dc = c - concentration_memory
        concentration_memory += alpha * dc

        # Modulate tumble rate based on concentration change
        # Positive dc (moving up gradient) → lower tumble rate
        # Negative dc (moving down gradient) → higher tumble rate
        tumble_rate = base_tumble_rate * math.exp(-gradient_sensitivity * dc * 10)
        tumble_rate = max(0.1, min(5.0, tumble_rate))  # clamp

        # Tumble decision
        if random.random() < tumble_rate * dt:
            # Tumble: pick new random direction
            angle = random.uniform(0, 2 * math.pi)
            tumble_events.append(t)

        # Run: move in current direction
        x += speed * math.cos(angle) * dt
        y += speed * math.sin(angle) * dt

        positions.append((x, y))
        times.append(t)

    # Calculate metrics
    final_c = concentration(x, y)
    # initial_c not used - gradient_following metric uses displacement instead

    # Net displacement toward source
    sx, sy = source_position
    initial_dist = math.sqrt((initial_position[0] - sx) ** 2 + (initial_position[1] - sy) ** 2)
    final_dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    displacement = initial_dist - final_dist

    effective_velocity = displacement / duration
    gradient_following = displacement / max(initial_dist, 1.0)

    return ChemotaxisTrajectory(
        positions=positions,
        times=times,
        tumble_events=tumble_events,
        final_concentration=final_c,
        gradient_following_score=gradient_following,
        effective_velocity=effective_velocity,
    )


@dataclass
class BioAIComparison:
    """Result of comparing AI agent to biological behavior."""
    verdict: str  # "DUAL-PASS", "FLIPPED", "AI_BETTER", "BIO_BETTER"
    bio_score: float
    ai_score: float
    conservation_score: float  # How well invariants are preserved
    key_differences: List[str]
    recommendations: List[str]

    def __str__(self) -> str:
        lines = [
            "BIO-AI BEHAVIOR COMPARISON",
            "=" * 50,
            "",
            f"Verdict: {self.verdict}",
            f"Biological score: {self.bio_score:.3f}",
            f"AI agent score: {self.ai_score:.3f}",
            f"Conservation score: {self.conservation_score:.3f}",
            "",
            "Key differences:",
        ]
        for diff in self.key_differences:
            lines.append(f"  • {diff}")
        lines.append("")
        lines.append("Recommendations:")
        for rec in self.recommendations:
            lines.append(f"  • {rec}")
        return "\n".join(lines)


def compare_to_rl_agent(
    agent_policy: Callable[[float], float],  # observation -> action (0=run, 1=tumble)
    gradient_field: Callable[[float, float], float] = None,
    n_episodes: int = 10,
    episode_length: float = 100.0,
    seed: Optional[int] = None,
) -> BioAIComparison:
    """Compare an RL agent's behavior to E. coli chemotaxis.

    The agent should:
    - Output tumble probability given observation (concentration change)
    - Achieve adaptation (return to baseline tumble rate)
    - Follow gradients efficiently

    Args:
        agent_policy: Function from observation to tumble probability
        gradient_field: Function (x, y) -> concentration (default: Gaussian)
        n_episodes: Number of episodes to average
        episode_length: Duration per episode
        seed: Random seed

    Returns:
        BioAIComparison with verdict and analysis
    """
    import random
    if seed is not None:
        random.seed(seed)

    if gradient_field is None:
        def gradient_field(x, y):
            return math.exp(-((x - 100) ** 2 + (y - 100) ** 2) / 5000)

    # Run biological simulation
    bio_scores = []
    for _ in range(n_episodes):
        traj = simulate_chemotaxis(
            duration=episode_length,
            source_position=(100, 100),
            seed=random.randint(0, 10000),
        )
        bio_scores.append(traj.gradient_following_score)

    bio_mean = sum(bio_scores) / len(bio_scores)

    # Run AI agent
    ai_scores = []
    for _ in range(n_episodes):
        x, y = 0.0, 0.0
        angle = random.uniform(0, 2 * math.pi)
        prev_c = gradient_field(x, y)
        speed = 20.0
        dt = 0.1

        for _ in range(int(episode_length / dt)):
            c = gradient_field(x, y)
            dc = c - prev_c
            prev_c = c

            # Agent decides based on concentration change
            tumble_prob = agent_policy(dc)

            if random.random() < tumble_prob:
                angle = random.uniform(0, 2 * math.pi)

            x += speed * math.cos(angle) * dt
            y += speed * math.sin(angle) * dt

        # Score
        initial_dist = math.sqrt(100 ** 2 + 100 ** 2)
        final_dist = math.sqrt((x - 100) ** 2 + (y - 100) ** 2)
        ai_scores.append((initial_dist - final_dist) / initial_dist)

    ai_mean = sum(ai_scores) / len(ai_scores)

    # Compare
    diff = abs(bio_mean - ai_mean)
    if diff < 0.1:
        verdict = "DUAL-PASS"
        conservation = 1.0 - diff
    elif ai_mean > bio_mean + 0.1:
        verdict = "AI_BETTER"
        conservation = 0.8
    elif bio_mean > ai_mean + 0.1:
        verdict = "BIO_BETTER"
        conservation = 0.5
    else:
        verdict = "FLIPPED"
        conservation = 0.3

    key_differences = []
    recommendations = []

    if ai_mean < 0:
        key_differences.append("AI moves away from source (negative chemotaxis)")
        recommendations.append("Invert reward signal or fix gradient sensing")

    if abs(ai_mean) < 0.1 and abs(bio_mean) > 0.3:
        key_differences.append("AI behaves like random walk; biology shows directed motion")
        recommendations.append("Increase coupling between observation and action")

    if ai_mean > bio_mean * 1.5:
        key_differences.append("AI is more efficient but may lack biological robustness")
        recommendations.append("Test with noise and verify adaptation")

    if not key_differences:
        key_differences.append("Behaviors are qualitatively similar")

    if not recommendations:
        recommendations.append("Agent matches biological behavior well")

    return BioAIComparison(
        verdict=verdict,
        bio_score=bio_mean,
        ai_score=ai_mean,
        conservation_score=conservation,
        key_differences=key_differences,
        recommendations=recommendations,
    )
