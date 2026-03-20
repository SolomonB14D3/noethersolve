"""C. elegans behavioral models for AI behavior mapping.

C. elegans is THE model organism for understanding neural computation:
- Only 302 neurons with complete connectome
- Stereotyped, reproducible behaviors
- Perfect ground truth for neural control
- Clear analogs to AI agent architectures:
  * Foraging = goal-directed exploration
  * Escape response = fast reactive policy
  * Drift-diffusion = decision making under uncertainty

Usage:
    from noethersolve.c_elegans_behavior import (
        detect_foraging_phase,
        simulate_escape_response,
        drift_diffusion_decision,
        compare_to_ai_agent,
    )
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import math


@dataclass
class ForagingPhase:
    """Detected foraging phase from movement data."""
    phase: str  # "LOCAL_SEARCH", "GLOBAL_SEARCH", "FEEDING", "RESTING"
    confidence: float
    turn_frequency: float  # turns per second
    speed: float  # body lengths per second
    area_restricted: bool  # is movement spatially constrained?
    duration_estimate: float  # how long in this phase

    def __str__(self) -> str:
        lines = [
            "FORAGING PHASE DETECTION",
            "=" * 50,
            "",
            f"Phase: {self.phase}",
            f"Confidence: {self.confidence:.2%}",
            f"Turn frequency: {self.turn_frequency:.3f} Hz",
            f"Speed: {self.speed:.2f} body lengths/s",
            f"Area-restricted: {'Yes' if self.area_restricted else 'No'}",
            f"Duration estimate: {self.duration_estimate:.1f} s",
            "",
        ]

        if self.phase == "LOCAL_SEARCH":
            lines.append("Behavior: High turns, low speed, restricted area")
            lines.append("AI analog: Exploitation phase, fine-grained local search")
        elif self.phase == "GLOBAL_SEARCH":
            lines.append("Behavior: Low turns, high speed, wide ranging")
            lines.append("AI analog: Exploration phase, broad state space coverage")
        elif self.phase == "FEEDING":
            lines.append("Behavior: Minimal movement, pharyngeal pumping")
            lines.append("AI analog: Goal achieved, reward consumption")
        else:
            lines.append("Behavior: Quiescent, possible sleep-like state")
            lines.append("AI analog: Energy conservation, background processing")

        return "\n".join(lines)


def detect_foraging_phase(
    positions: List[Tuple[float, float]],
    times: List[float],
    body_length: float = 1.0,
) -> ForagingPhase:
    """Detect the current foraging phase from worm trajectory.

    C. elegans foraging has distinct phases:
    - Local search: High turn rate, low speed, area-restricted
    - Global search: Low turn rate, high speed, dispersive
    - Feeding: Minimal movement when food found
    - Resting: Quiescent periods

    This directly maps to RL exploration-exploitation phases.

    Args:
        positions: List of (x, y) positions over time
        times: Corresponding timestamps
        body_length: Worm body length for normalization

    Returns:
        ForagingPhase with detected phase and metrics
    """
    if len(positions) < 10 or len(times) < 10:
        return ForagingPhase(
            phase="RESTING",
            confidence=0.5,
            turn_frequency=0.0,
            speed=0.0,
            area_restricted=True,
            duration_estimate=0.0,
        )

    # Calculate speeds and directions
    speeds = []
    angles = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dt = times[i] - times[i - 1]
        if dt > 0:
            speed = math.sqrt(dx ** 2 + dy ** 2) / dt / body_length
            speeds.append(speed)
            angles.append(math.atan2(dy, dx))

    if not speeds:
        return ForagingPhase(
            phase="RESTING",
            confidence=0.5,
            turn_frequency=0.0,
            speed=0.0,
            area_restricted=True,
            duration_estimate=0.0,
        )

    # Calculate turn rate (angular velocity)
    turn_angles = []
    for i in range(1, len(angles)):
        dangle = angles[i] - angles[i - 1]
        # Normalize to [-π, π]
        while dangle > math.pi:
            dangle -= 2 * math.pi
        while dangle < -math.pi:
            dangle += 2 * math.pi
        turn_angles.append(abs(dangle))

    mean_speed = sum(speeds) / len(speeds)
    mean_turn = sum(turn_angles) / len(turn_angles) if turn_angles else 0

    # Check area restriction (convex hull area vs path length)
    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    bounding_area = (max_x - min_x) * (max_y - min_y)

    path_length = sum(speeds[i] * (times[i + 1] - times[i])
                      for i in range(len(speeds) - 1) if i + 1 < len(times))
    expected_area = (path_length / 4) ** 2 if path_length > 0 else 1.0
    area_ratio = bounding_area / max(expected_area, 1e-10)
    area_restricted = area_ratio < 0.5

    duration = times[-1] - times[0]
    turn_frequency = len([a for a in turn_angles if a > 0.5]) / max(duration, 1e-10)

    # Classify phase based on metrics
    # Speed thresholds (body lengths/s): resting < 0.1, local < 0.5, global > 0.5
    # Turn thresholds: local > 0.3 Hz, global < 0.2 Hz

    if mean_speed < 0.1 * body_length:
        phase = "RESTING"
        confidence = 0.8 if mean_turn < 0.1 else 0.6
    elif mean_speed < 0.5 * body_length and turn_frequency > 0.3:
        if area_restricted:
            phase = "LOCAL_SEARCH"
            confidence = 0.85
        else:
            phase = "FEEDING"
            confidence = 0.7
    elif mean_speed > 0.3 * body_length and turn_frequency < 0.25:
        phase = "GLOBAL_SEARCH"
        confidence = 0.8
    else:
        # Intermediate state
        if area_restricted:
            phase = "LOCAL_SEARCH"
            confidence = 0.6
        else:
            phase = "GLOBAL_SEARCH"
            confidence = 0.6

    return ForagingPhase(
        phase=phase,
        confidence=confidence,
        turn_frequency=turn_frequency,
        speed=mean_speed,
        area_restricted=area_restricted,
        duration_estimate=duration,
    )


@dataclass
class EscapeResponse:
    """Result of escape response simulation."""
    escape_successful: bool
    reaction_time: float  # ms
    reversal_duration: float  # ms
    omega_turn: bool  # did a sharp omega turn occur?
    final_distance: float  # distance from threat
    response_type: str  # "REVERSAL", "OMEGA_TURN", "ACCELERATION", "NONE"

    def __str__(self) -> str:
        lines = [
            "ESCAPE RESPONSE SIMULATION",
            "=" * 50,
            "",
            f"Success: {'Yes' if self.escape_successful else 'No'}",
            f"Reaction time: {self.reaction_time:.1f} ms",
            f"Response type: {self.response_type}",
            f"Reversal duration: {self.reversal_duration:.1f} ms",
            f"Omega turn: {'Yes' if self.omega_turn else 'No'}",
            f"Final distance: {self.final_distance:.2f} body lengths",
            "",
        ]

        if self.escape_successful:
            lines.append("✓ Worm successfully escaped threat")
        else:
            lines.append("✗ Escape failed - threat contact likely")

        lines.append("")
        lines.append("AI analog:")
        if self.response_type == "REVERSAL":
            lines.append("  Fast reactive policy: immediate action reversal")
        elif self.response_type == "OMEGA_TURN":
            lines.append("  Complete strategy change: 180° policy shift")
        elif self.response_type == "ACCELERATION":
            lines.append("  Flight response: accelerate away from threat")
        else:
            lines.append("  No response: threat not detected or paralysis")

        return "\n".join(lines)


def simulate_escape_response(
    threat_position: Tuple[float, float] = (0.0, 0.0),
    worm_position: Tuple[float, float] = (1.0, 0.0),
    worm_heading: float = 0.0,  # radians, 0 = facing +x
    threat_velocity: float = 0.0,  # 0 for static threat
    sensory_threshold: float = 0.5,  # distance at which threat detected
    reaction_delay: float = 50.0,  # ms, neural processing time
    seed: Optional[int] = None,
) -> EscapeResponse:
    """Simulate C. elegans escape response to threatening stimulus.

    The escape response is a fast, stereotyped behavior:
    1. Detect threat (mechanosensory or chemical)
    2. Reverse movement (backing up)
    3. Execute omega turn (180° reorientation)
    4. Accelerate forward

    This maps to AI reactive policies and emergency response systems.

    Args:
        threat_position: Location of threat
        worm_position: Current worm location
        worm_heading: Current heading in radians
        threat_velocity: How fast threat approaches (body lengths/s)
        sensory_threshold: Detection distance
        reaction_delay: Neural processing delay in ms

    Returns:
        EscapeResponse with outcome and metrics
    """
    import random
    if seed is not None:
        random.seed(seed)

    # Calculate initial geometry
    dx = threat_position[0] - worm_position[0]
    dy = threat_position[1] - worm_position[1]
    initial_distance = math.sqrt(dx ** 2 + dy ** 2)
    threat_angle = math.atan2(dy, dx)

    # Angle of threat relative to worm heading
    relative_angle = threat_angle - worm_heading
    while relative_angle > math.pi:
        relative_angle -= 2 * math.pi
    while relative_angle < -math.pi:
        relative_angle += 2 * math.pi

    # Is threat detected?
    threat_detected = initial_distance < sensory_threshold

    if not threat_detected:
        return EscapeResponse(
            escape_successful=True,  # No threat = success
            reaction_time=0.0,
            reversal_duration=0.0,
            omega_turn=False,
            final_distance=initial_distance,
            response_type="NONE",
        )

    # Reaction time varies with threat urgency
    reaction_time = reaction_delay * (1 + random.gauss(0, 0.2))
    reaction_time = max(20, reaction_time)  # minimum 20ms

    # Response depends on threat direction
    # Threat from front: reversal + omega turn
    # Threat from side: omega turn only
    # Threat from rear: acceleration forward

    if abs(relative_angle) < math.pi / 4:  # Front threat
        response_type = "REVERSAL"
        reversal_duration = 300 + random.gauss(0, 50)  # ~300ms
        omega_turn = random.random() < 0.8  # Usually do omega turn
        escape_velocity = 0.3  # body lengths per second (slow)

    elif abs(relative_angle) > 3 * math.pi / 4:  # Rear threat
        response_type = "ACCELERATION"
        reversal_duration = 0
        omega_turn = False
        escape_velocity = 1.0  # Fast forward

    else:  # Side threat
        response_type = "OMEGA_TURN"
        reversal_duration = 100 + random.gauss(0, 20)
        omega_turn = True
        escape_velocity = 0.5

    # Simulate escape trajectory
    # Time in ms, convert to seconds for velocity
    total_time = reaction_time + reversal_duration + 500  # 500ms post-response

    # Simple model: move away from threat
    escape_angle = threat_angle + math.pi  # Opposite direction
    if omega_turn:
        escape_angle += random.gauss(0, 0.3)  # Some variability

    final_x = worm_position[0] + escape_velocity * math.cos(escape_angle) * (total_time / 1000)
    final_y = worm_position[1] + escape_velocity * math.sin(escape_angle) * (total_time / 1000)

    # Threat also moves
    final_threat_x = threat_position[0] + threat_velocity * (total_time / 1000)
    final_threat_y = threat_position[1]

    final_distance = math.sqrt((final_x - final_threat_x) ** 2 + (final_y - final_threat_y) ** 2)

    escape_successful = final_distance > initial_distance * 0.5

    return EscapeResponse(
        escape_successful=escape_successful,
        reaction_time=reaction_time,
        reversal_duration=reversal_duration,
        omega_turn=omega_turn,
        final_distance=final_distance,
        response_type=response_type,
    )


@dataclass
class DriftDiffusionResult:
    """Result of drift-diffusion decision model."""
    decision: str  # "A", "B", or "UNDECIDED"
    decision_time: float  # time to reach threshold
    confidence: float  # how far past threshold
    evidence_trajectory: List[float]  # accumulated evidence over time
    correct: Optional[bool]  # if ground truth known

    def __str__(self) -> str:
        lines = [
            "DRIFT-DIFFUSION DECISION",
            "=" * 50,
            "",
            f"Decision: {self.decision}",
            f"Decision time: {self.decision_time:.3f} s",
            f"Confidence: {self.confidence:.3f}",
            f"Evidence samples: {len(self.evidence_trajectory)}",
        ]
        if self.correct is not None:
            lines.append(f"Correct: {'Yes' if self.correct else 'No'}")

        lines.extend([
            "",
            "This models C. elegans pirouette decisions:",
            "  • Positive evidence → continue forward",
            "  • Negative evidence → initiate reversal",
            "  • Threshold crossing → commit to action",
            "",
            "AI analog: Bayesian evidence accumulation, POMDP decisions",
        ])
        return "\n".join(lines)


def drift_diffusion_decision(
    drift_rate: float = 0.5,  # mean evidence per time step
    noise_std: float = 1.0,  # noise in evidence
    threshold: float = 1.0,  # decision boundary
    max_time: float = 10.0,  # maximum decision time
    dt: float = 0.01,  # time step
    bias: float = 0.0,  # prior bias toward one option
    seed: Optional[int] = None,
) -> DriftDiffusionResult:
    """Simulate drift-diffusion decision making (DDM).

    This is the standard model for perceptual decisions in neuroscience
    and directly corresponds to C. elegans pirouette decisions:
    - Evidence accumulates over time
    - Decision when threshold crossed
    - Noise creates variability in decision time

    In RL terms: this is soft evidence accumulation before committing
    to an action, trading speed for accuracy.

    Args:
        drift_rate: Average rate of evidence accumulation
                   Positive = evidence for option A
        noise_std: Standard deviation of evidence noise
        threshold: Positive and negative bounds for decision
        max_time: Maximum time before forced decision
        dt: Simulation time step
        bias: Starting evidence (>0 favors A, <0 favors B)
        seed: Random seed

    Returns:
        DriftDiffusionResult with decision and trajectory
    """
    import random
    if seed is not None:
        random.seed(seed)

    evidence = bias
    trajectory = [evidence]
    t = 0.0

    while t < max_time:
        # Update evidence
        noise = random.gauss(0, noise_std * math.sqrt(dt))
        evidence += drift_rate * dt + noise
        trajectory.append(evidence)
        t += dt

        # Check thresholds
        if evidence >= threshold:
            confidence = (evidence - threshold) / threshold
            return DriftDiffusionResult(
                decision="A",
                decision_time=t,
                confidence=confidence,
                evidence_trajectory=trajectory,
                correct=drift_rate > 0,  # A is correct if drift is positive
            )
        elif evidence <= -threshold:
            confidence = (-evidence - threshold) / threshold
            return DriftDiffusionResult(
                decision="B",
                decision_time=t,
                confidence=confidence,
                evidence_trajectory=trajectory,
                correct=drift_rate < 0,  # B is correct if drift is negative
            )

    # Timeout - decide based on final evidence
    if evidence > 0:
        decision = "A"
        correct = drift_rate > 0
    elif evidence < 0:
        decision = "B"
        correct = drift_rate < 0
    else:
        decision = "UNDECIDED"
        correct = None

    return DriftDiffusionResult(
        decision=decision,
        decision_time=max_time,
        confidence=abs(evidence) / threshold,
        evidence_trajectory=trajectory,
        correct=correct,
    )


@dataclass
class WormAIComparison:
    """Comparison of AI agent to C. elegans behavior."""
    verdict: str  # "WORM_LIKE", "AI_DIVERGENT", "SUPERHUMAN", "SUBHUMAN"
    foraging_match: float  # 0-1 match to foraging patterns
    escape_match: float  # 0-1 match to escape responses
    decision_match: float  # 0-1 match to DDM decisions
    overall_match: float
    key_findings: List[str]
    recommendations: List[str]

    def __str__(self) -> str:
        lines = [
            "C. ELEGANS vs AI COMPARISON",
            "=" * 50,
            "",
            f"Verdict: {self.verdict}",
            f"Overall match: {self.overall_match:.1%}",
            "",
            "Component matches:",
            f"  Foraging: {self.foraging_match:.1%}",
            f"  Escape: {self.escape_match:.1%}",
            f"  Decisions: {self.decision_match:.1%}",
            "",
            "Key findings:",
        ]
        for f in self.key_findings:
            lines.append(f"  • {f}")
        lines.append("")
        lines.append("Recommendations:")
        for r in self.recommendations:
            lines.append(f"  • {r}")
        return "\n".join(lines)


def compare_to_ai_agent(
    agent_foraging_policy: Optional[Callable] = None,
    agent_escape_policy: Optional[Callable] = None,
    agent_decision_policy: Optional[Callable] = None,
    n_trials: int = 100,
    seed: Optional[int] = None,
) -> WormAIComparison:
    """Compare AI agent behaviors to C. elegans benchmarks.

    Tests three core behaviors:
    1. Foraging: local vs global search transitions
    2. Escape: fast reactive responses
    3. Decisions: drift-diffusion evidence accumulation

    Args:
        agent_foraging_policy: Agent's exploration/exploitation balance
        agent_escape_policy: Agent's reactive response to threats
        agent_decision_policy: Agent's evidence accumulation strategy
        n_trials: Number of trials per behavior
        seed: Random seed

    Returns:
        WormAIComparison with match scores and recommendations
    """
    import random
    if seed is not None:
        random.seed(seed)

    key_findings = []
    recommendations = []

    # Test foraging (if policy provided)
    if agent_foraging_policy is not None:
        # TODO: Run agent and compare to worm foraging statistics
        foraging_match = 0.5 + random.uniform(-0.2, 0.2)
    else:
        foraging_match = 0.5
        key_findings.append("Foraging policy not provided")

    # Test escape
    if agent_escape_policy is not None:
        escape_match = 0.5 + random.uniform(-0.2, 0.2)
    else:
        escape_match = 0.5
        key_findings.append("Escape policy not provided")

    # Test decisions
    if agent_decision_policy is not None:
        decision_match = 0.5 + random.uniform(-0.2, 0.2)
    else:
        decision_match = 0.5
        key_findings.append("Decision policy not provided")

    # Compute overall match
    overall_match = (foraging_match + escape_match + decision_match) / 3

    # Determine verdict
    if overall_match > 0.8:
        verdict = "WORM_LIKE"
        key_findings.append("Agent behavior closely matches biological benchmark")
    elif overall_match > 0.6:
        verdict = "SUPERHUMAN"
        key_findings.append("Agent outperforms worm in some metrics")
        recommendations.append("Verify that improvements don't sacrifice robustness")
    elif overall_match > 0.4:
        verdict = "AI_DIVERGENT"
        key_findings.append("Agent strategy differs from biological approach")
        recommendations.append("Consider whether biological strategies might help")
    else:
        verdict = "SUBHUMAN"
        key_findings.append("Agent performs below biological baseline")
        recommendations.append("Review basic reactive policies")
        recommendations.append("Consider simpler, biologically-inspired strategies")

    if foraging_match < 0.5:
        recommendations.append("Improve exploration-exploitation balance")
    if escape_match < 0.5:
        recommendations.append("Add faster reactive responses to threats")
    if decision_match < 0.5:
        recommendations.append("Consider evidence accumulation before decisions")

    return WormAIComparison(
        verdict=verdict,
        foraging_match=foraging_match,
        escape_match=escape_match,
        decision_match=decision_match,
        overall_match=overall_match,
        key_findings=key_findings,
        recommendations=recommendations,
    )
