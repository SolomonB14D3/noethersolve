"""
Neural-RL Analogy Module — Maps biological neural learning to RL algorithms.

This module provides computational validators that compare biological neural
mechanisms (dopamine RPE, Hebbian learning) to their RL algorithmic analogues.

Core insight: Dopamine neurons compute reward prediction errors (RPEs) that
are mathematically equivalent to TD(0) errors. This convergent evolution
suggests fundamental computational constraints that both biology and AI
must satisfy.

Tools provided:
- dopamine_rpe_validator() — Check if a signal matches RPE properties
- hebbian_vs_backprop_consistency() — Compare learning rules
- td_learning_validator() — Validate TD error signals
- actor_critic_bio_map() — Map striatal circuits to actor-critic
- eligibility_trace_validator() — Check synaptic eligibility traces
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import math


class SignalType(Enum):
    """Classification of neural/RL signals."""
    REWARD_PREDICTION_ERROR = "rpe"
    VALUE_ESTIMATE = "value"
    POLICY_GRADIENT = "policy"
    ELIGIBILITY_TRACE = "eligibility"
    UNKNOWN = "unknown"


class LearningRule(Enum):
    """Types of learning rules."""
    HEBBIAN = "hebbian"
    ANTI_HEBBIAN = "anti_hebbian"
    STDP = "stdp"
    BACKPROP = "backprop"
    TD_LEARNING = "td_learning"
    REINFORCE = "reinforce"


@dataclass
class RPEValidationResult:
    """Result of RPE signal validation."""
    is_valid_rpe: bool
    signal_type: SignalType
    td_error_correlation: float  # Correlation with TD(0) error
    phasic_response: float  # Magnitude of phasic response
    tonic_baseline: float  # Baseline firing rate
    cue_response: float  # Response to predictive cues
    reward_response: float  # Response to actual rewards
    prediction_transfer: bool  # Does response transfer to predictive cues?
    violations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID RPE" if self.is_valid_rpe else "NOT VALID RPE"
        lines = [
            f"RPE Validation: {status}",
            f"  TD error correlation: {self.td_error_correlation:.3f}",
            f"  Phasic response magnitude: {self.phasic_response:.2f}",
            f"  Tonic baseline: {self.tonic_baseline:.2f}",
            f"  Cue response: {self.cue_response:.2f}",
            f"  Reward response: {self.reward_response:.2f}",
            f"  Prediction transfer: {self.prediction_transfer}",
        ]
        if self.violations:
            lines.append("  Violations:")
            for v in self.violations:
                lines.append(f"    - {v}")
        return "\n".join(lines)


@dataclass
class LearningRuleComparison:
    """Comparison between biological and algorithmic learning rules."""
    bio_rule: LearningRule
    algo_rule: LearningRule
    consistency_score: float  # 0-1, how well they match
    weight_update_correlation: float
    temporal_profile_match: float
    locality_preserved: bool  # Bio learning is local; does algo preserve this?
    energy_efficiency_ratio: float  # Bio/Algo efficiency
    key_differences: List[str] = field(default_factory=list)
    convergence_guarantee: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            f"Learning Rule Comparison: {self.bio_rule.value} vs {self.algo_rule.value}",
            f"  Consistency score: {self.consistency_score:.3f}",
            f"  Weight update correlation: {self.weight_update_correlation:.3f}",
            f"  Temporal profile match: {self.temporal_profile_match:.3f}",
            f"  Locality preserved: {self.locality_preserved}",
            f"  Energy efficiency ratio: {self.energy_efficiency_ratio:.2f}x",
        ]
        if self.convergence_guarantee:
            lines.append(f"  Convergence: {self.convergence_guarantee}")
        if self.key_differences:
            lines.append("  Key differences:")
            for d in self.key_differences:
                lines.append(f"    - {d}")
        return "\n".join(lines)


@dataclass
class ActorCriticMapping:
    """Mapping between striatal circuits and actor-critic architecture."""
    striatal_region: str
    ac_component: str
    mapping_confidence: float
    pathway: str  # Direct/Indirect
    neurotransmitter: str
    functional_role: str
    supporting_evidence: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Striatal-Actor-Critic Mapping:",
            f"  Region: {self.striatal_region} -> {self.ac_component}",
            f"  Confidence: {self.mapping_confidence:.2f}",
            f"  Pathway: {self.pathway}",
            f"  Neurotransmitter: {self.neurotransmitter}",
            f"  Role: {self.functional_role}",
        ]
        if self.supporting_evidence:
            lines.append("  Evidence:")
            for e in self.supporting_evidence[:3]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


def validate_dopamine_rpe(
    firing_rates: List[float],
    expected_rewards: List[float],
    actual_rewards: List[float],
    cue_times: Optional[List[int]] = None,
    reward_times: Optional[List[int]] = None,
    gamma: float = 0.99,
    baseline_window: int = 10
) -> RPEValidationResult:
    """
    Validate if a neural signal matches dopamine RPE properties.

    The canonical RPE signal should:
    1. Show phasic increases for unexpected rewards (positive RPE)
    2. Show phasic decreases for omitted expected rewards (negative RPE)
    3. Transfer from reward to predictive cue with learning
    4. Have magnitude proportional to prediction error
    5. Return to baseline when predictions are accurate

    Args:
        firing_rates: Time series of neural firing rates
        expected_rewards: Expected reward at each timestep
        actual_rewards: Actual reward received at each timestep
        cue_times: Indices where predictive cues occur
        reward_times: Indices where rewards are delivered
        gamma: Discount factor for TD computation
        baseline_window: Window for computing baseline firing

    Returns:
        RPEValidationResult with validation details
    """
    if len(firing_rates) < baseline_window:
        return RPEValidationResult(
            is_valid_rpe=False,
            signal_type=SignalType.UNKNOWN,
            td_error_correlation=0.0,
            phasic_response=0.0,
            tonic_baseline=0.0,
            cue_response=0.0,
            reward_response=0.0,
            prediction_transfer=False,
            violations=["Insufficient data for validation"]
        )

    n = len(firing_rates)
    violations = []

    # Compute tonic baseline
    tonic_baseline = sum(firing_rates[:baseline_window]) / baseline_window

    # Compute TD(0) errors: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    td_errors = []
    for t in range(n - 1):
        delta = actual_rewards[t] + gamma * expected_rewards[t + 1] - expected_rewards[t]
        td_errors.append(delta)
    td_errors.append(actual_rewards[-1] - expected_rewards[-1])

    # Compute correlation between firing rate deviations and TD errors
    firing_deviations = [fr - tonic_baseline for fr in firing_rates]

    # Pearson correlation
    mean_td = sum(td_errors) / len(td_errors)
    mean_fr = sum(firing_deviations) / len(firing_deviations)

    numerator = sum((td - mean_td) * (fr - mean_fr)
                    for td, fr in zip(td_errors, firing_deviations))

    var_td = sum((td - mean_td) ** 2 for td in td_errors)
    var_fr = sum((fr - mean_fr) ** 2 for fr in firing_deviations)

    if var_td > 0 and var_fr > 0:
        td_correlation = numerator / math.sqrt(var_td * var_fr)
    else:
        td_correlation = 0.0

    # Check phasic responses at reward times
    reward_response = 0.0
    if reward_times:
        reward_responses = []
        for rt in reward_times:
            if rt < n:
                response = firing_rates[rt] - tonic_baseline
                reward_responses.append(response)
        if reward_responses:
            reward_response = sum(reward_responses) / len(reward_responses)

    # Check cue responses
    cue_response = 0.0
    prediction_transfer = False
    if cue_times:
        cue_responses = []
        for ct in cue_times:
            if ct < n:
                response = firing_rates[ct] - tonic_baseline
                cue_responses.append(response)
        if cue_responses:
            cue_response = sum(cue_responses) / len(cue_responses)
            # Prediction transfer: cue responses should develop over time
            if len(cue_responses) > 1:
                early = cue_responses[:len(cue_responses)//2]
                late = cue_responses[len(cue_responses)//2:]
                if sum(late)/len(late) > sum(early)/len(early):
                    prediction_transfer = True

    # Compute overall phasic response magnitude
    phasic_response = max(abs(fr - tonic_baseline) for fr in firing_rates)

    # Validation checks
    if td_correlation < 0.5:
        violations.append(f"Low TD error correlation ({td_correlation:.2f} < 0.5)")

    if phasic_response < tonic_baseline * 0.1:
        violations.append("Insufficient phasic modulation")

    # Check for appropriate responses to prediction errors
    positive_rpe_count = 0
    negative_rpe_count = 0
    for t, (td, fr) in enumerate(zip(td_errors, firing_deviations)):
        if td > 0.1 and fr > 0:
            positive_rpe_count += 1
        elif td < -0.1 and fr < 0:
            negative_rpe_count += 1

    total_rpe_events = sum(1 for td in td_errors if abs(td) > 0.1)
    if total_rpe_events > 0:
        correct_responses = positive_rpe_count + negative_rpe_count
        if correct_responses / total_rpe_events < 0.7:
            violations.append("Poor directional coding of prediction errors")

    is_valid = td_correlation > 0.5 and len(violations) == 0
    signal_type = SignalType.REWARD_PREDICTION_ERROR if is_valid else SignalType.UNKNOWN

    return RPEValidationResult(
        is_valid_rpe=is_valid,
        signal_type=signal_type,
        td_error_correlation=td_correlation,
        phasic_response=phasic_response,
        tonic_baseline=tonic_baseline,
        cue_response=cue_response,
        reward_response=reward_response,
        prediction_transfer=prediction_transfer,
        violations=violations
    )


def compare_hebbian_backprop(
    pre_activities: List[float],
    post_activities: List[float],
    weights: List[float],
    target_outputs: List[float],
    learning_rate_hebb: float = 0.01,
    learning_rate_bp: float = 0.01
) -> LearningRuleComparison:
    """
    Compare Hebbian learning to backpropagation updates.

    Hebbian rule: delta_w = eta * pre * post
    Backprop rule: delta_w = eta * pre * error * f'(net)

    Key insight: Hebbian learning is purely local (uses only pre/post activity),
    while backprop requires non-local error signals. However, certain formulations
    (e.g., contrastive Hebbian learning) can approximate backprop locally.

    Args:
        pre_activities: Presynaptic activities
        post_activities: Postsynaptic activities
        weights: Current synaptic weights
        target_outputs: Desired outputs for error computation
        learning_rate_hebb: Hebbian learning rate
        learning_rate_bp: Backprop learning rate

    Returns:
        LearningRuleComparison with analysis
    """
    n = len(pre_activities)
    if n == 0:
        return LearningRuleComparison(
            bio_rule=LearningRule.HEBBIAN,
            algo_rule=LearningRule.BACKPROP,
            consistency_score=0.0,
            weight_update_correlation=0.0,
            temporal_profile_match=0.0,
            locality_preserved=True,
            energy_efficiency_ratio=1.0,
            key_differences=["No data provided"]
        )

    # Compute Hebbian updates
    hebbian_updates = []
    for pre, post in zip(pre_activities, post_activities):
        delta_w = learning_rate_hebb * pre * post
        hebbian_updates.append(delta_w)

    # Compute backprop updates (simplified single layer)
    # Assume sigmoid activation: f'(x) = f(x) * (1 - f(x))
    backprop_updates = []
    for i, (pre, post, target) in enumerate(zip(pre_activities, post_activities, target_outputs)):
        error = target - post
        fprime = post * (1 - post)  # Sigmoid derivative
        delta_w = learning_rate_bp * pre * error * fprime
        backprop_updates.append(delta_w)

    # Compute correlation between updates
    if len(hebbian_updates) > 1:
        mean_h = sum(hebbian_updates) / n
        mean_b = sum(backprop_updates) / n

        numerator = sum((h - mean_h) * (b - mean_b)
                       for h, b in zip(hebbian_updates, backprop_updates))
        var_h = sum((h - mean_h) ** 2 for h in hebbian_updates)
        var_b = sum((b - mean_b) ** 2 for b in backprop_updates)

        if var_h > 0 and var_b > 0:
            correlation = numerator / math.sqrt(var_h * var_b)
        else:
            correlation = 0.0
    else:
        correlation = 0.0

    # Temporal profile analysis
    # Check if updates have similar temporal dynamics
    if n > 2:
        hebb_changes = [hebbian_updates[i+1] - hebbian_updates[i] for i in range(n-1)]
        bp_changes = [backprop_updates[i+1] - backprop_updates[i] for i in range(n-1)]

        same_direction = sum(1 for h, b in zip(hebb_changes, bp_changes)
                            if (h > 0 and b > 0) or (h < 0 and b < 0) or (h == 0 and b == 0))
        temporal_match = same_direction / (n - 1)
    else:
        temporal_match = 0.5

    # Energy efficiency: Hebbian is local (low energy), backprop is global (high energy)
    # Estimate based on computational locality
    energy_ratio = 10.0  # Hebbian is ~10x more energy efficient (local synapses only)

    # Key differences
    differences = [
        "Hebbian: purely local (pre * post), no error signal needed",
        "Backprop: requires global error propagation through network",
        "Hebbian: can lead to runaway excitation without normalization",
        "Backprop: guarantees gradient descent on loss surface",
    ]

    # Consistency score
    # High if updates correlate and both achieve learning
    consistency = (abs(correlation) + temporal_match) / 2

    # Check locality
    locality_preserved = True  # Hebbian is always local

    return LearningRuleComparison(
        bio_rule=LearningRule.HEBBIAN,
        algo_rule=LearningRule.BACKPROP,
        consistency_score=consistency,
        weight_update_correlation=correlation,
        temporal_profile_match=temporal_match,
        locality_preserved=locality_preserved,
        energy_efficiency_ratio=energy_ratio,
        key_differences=differences,
        convergence_guarantee="Backprop converges to local minimum; Hebbian requires normalization"
    )


def validate_td_learning(
    value_estimates: List[float],
    rewards: List[float],
    gamma: float = 0.99,
    alpha: float = 0.1,
    tolerance: float = 0.1
) -> Dict[str, any]:
    """
    Validate that a value function is learning via TD(0).

    TD(0) update: V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))

    Args:
        value_estimates: Sequence of value estimates V(s_t)
        rewards: Sequence of rewards r_t
        gamma: Discount factor
        alpha: Learning rate
        tolerance: Tolerance for TD error convergence

    Returns:
        Dictionary with validation results
    """
    n = len(value_estimates)
    if n < 2:
        return {"valid": False, "reason": "Insufficient data"}

    td_errors = []
    expected_updates = []
    actual_updates = []

    for t in range(n - 1):
        # TD error
        delta = rewards[t] + gamma * value_estimates[t + 1] - value_estimates[t]
        td_errors.append(delta)

        # Expected update
        expected_update = alpha * delta
        expected_updates.append(expected_update)

        # Actual update (if we had next value estimate)
        if t < n - 2:
            actual_update = value_estimates[t + 1] - value_estimates[t]
            actual_updates.append(actual_update)

    # Check if TD errors are decreasing (convergence)
    if len(td_errors) > 5:
        early_td = sum(abs(e) for e in td_errors[:len(td_errors)//2]) / (len(td_errors)//2)
        late_td = sum(abs(e) for e in td_errors[len(td_errors)//2:]) / (len(td_errors) - len(td_errors)//2)
        converging = late_td < early_td
    else:
        converging = True

    # Check if final TD errors are within tolerance
    final_td_mean = abs(sum(td_errors[-5:]) / min(5, len(td_errors)))
    within_tolerance = final_td_mean < tolerance

    return {
        "valid": converging and within_tolerance,
        "converging": converging,
        "final_td_error": final_td_mean,
        "within_tolerance": within_tolerance,
        "td_errors": td_errors,
        "mean_td_error": sum(abs(e) for e in td_errors) / len(td_errors),
        "gamma": gamma,
        "alpha": alpha
    }


def map_striatum_to_actor_critic() -> List[ActorCriticMapping]:
    """
    Map striatal circuit components to actor-critic architecture.

    The basal ganglia implement something remarkably similar to actor-critic RL:
    - Striatum: Actor (action selection)
    - VTA/SNc dopamine: Critic (value/TD error)
    - Direct pathway: "Go" (select action)
    - Indirect pathway: "NoGo" (suppress action)

    Returns:
        List of mappings between brain regions and AC components
    """
    mappings = [
        ActorCriticMapping(
            striatal_region="Dorsal Striatum (Putamen)",
            ac_component="Actor (Policy)",
            mapping_confidence=0.85,
            pathway="Direct + Indirect",
            neurotransmitter="GABA",
            functional_role="Action selection based on learned values",
            supporting_evidence=[
                "Lesion studies show impaired action selection",
                "fMRI shows activation during decision-making",
                "Dopamine modulates direct/indirect pathway balance"
            ]
        ),
        ActorCriticMapping(
            striatal_region="Ventral Striatum (NAcc)",
            ac_component="Critic (Value)",
            mapping_confidence=0.90,
            pathway="Mesolimbic",
            neurotransmitter="Dopamine modulated GABA",
            functional_role="Compute state value, predict rewards",
            supporting_evidence=[
                "Correlates with expected value",
                "Necessary for reward prediction",
                "Dopamine signals RPE here"
            ]
        ),
        ActorCriticMapping(
            striatal_region="VTA Dopamine Neurons",
            ac_component="TD Error Signal",
            mapping_confidence=0.95,
            pathway="Mesolimbic",
            neurotransmitter="Dopamine",
            functional_role="Compute and broadcast reward prediction error",
            supporting_evidence=[
                "Schultz (1997): dopamine = TD error",
                "Phasic = unexpected reward",
                "Dip = omitted expected reward",
                "Transfers to predictive cues"
            ]
        ),
        ActorCriticMapping(
            striatal_region="D1 MSNs (Direct Pathway)",
            ac_component="Go/Select Action",
            mapping_confidence=0.80,
            pathway="Direct",
            neurotransmitter="GABA",
            functional_role="Promote selected action",
            supporting_evidence=[
                "Express D1 receptors (excitatory dopamine)",
                "Activation promotes movement",
                "Potentiated by positive RPE"
            ]
        ),
        ActorCriticMapping(
            striatal_region="D2 MSNs (Indirect Pathway)",
            ac_component="NoGo/Suppress Action",
            mapping_confidence=0.80,
            pathway="Indirect",
            neurotransmitter="GABA",
            functional_role="Inhibit non-selected actions",
            supporting_evidence=[
                "Express D2 receptors (inhibitory dopamine)",
                "Activation suppresses movement",
                "Potentiated by negative RPE"
            ]
        ),
        ActorCriticMapping(
            striatal_region="Globus Pallidus Externa",
            ac_component="Action Gating",
            mapping_confidence=0.70,
            pathway="Indirect",
            neurotransmitter="GABA",
            functional_role="Tonic inhibition of thalamus",
            supporting_evidence=[
                "Part of indirect pathway loop",
                "Releases thalamus from inhibition selectively"
            ]
        ),
        ActorCriticMapping(
            striatal_region="Subthalamic Nucleus",
            ac_component="Urgency/Threshold",
            mapping_confidence=0.75,
            pathway="Hyperdirect",
            neurotransmitter="Glutamate",
            functional_role="Global NoGo signal, raise action threshold",
            supporting_evidence=[
                "Activated during conflict/stopping",
                "DBS here helps Parkinson's (releases actions)",
                "May implement decision threshold"
            ]
        ),
    ]

    return mappings


def validate_eligibility_trace(
    pre_spike_times: List[float],
    post_spike_times: List[float],
    reward_times: List[float],
    trace_decay: float = 0.95,
    stdp_window: float = 20.0  # ms
) -> Dict[str, any]:
    """
    Validate synaptic eligibility traces for three-factor learning.

    Three-factor learning: weight change requires:
    1. Presynaptic activity
    2. Postsynaptic activity
    3. Neuromodulatory signal (dopamine)

    Eligibility trace bridges the temporal gap: synapse becomes "eligible"
    when pre-post coincidence occurs, then actual weight change happens
    when dopamine arrives (potentially seconds later).

    Args:
        pre_spike_times: Times of presynaptic spikes
        post_spike_times: Times of postsynaptic spikes
        reward_times: Times of reward/dopamine signal
        trace_decay: Exponential decay rate of eligibility
        stdp_window: STDP timing window in ms

    Returns:
        Dictionary with eligibility trace analysis
    """
    eligibility_events = []

    # Find pre-post coincidences that create eligibility
    for pre_t in pre_spike_times:
        for post_t in post_spike_times:
            dt = post_t - pre_t
            if 0 < dt < stdp_window:  # LTP window
                eligibility_events.append({
                    "time": post_t,
                    "type": "LTP",
                    "dt": dt,
                    "strength": math.exp(-dt / stdp_window)
                })
            elif -stdp_window < dt < 0:  # LTD window
                eligibility_events.append({
                    "time": pre_t,
                    "type": "LTD",
                    "dt": dt,
                    "strength": -math.exp(dt / stdp_window)
                })

    # Check which eligibility events are "captured" by reward
    captured_events = []
    uncaptured_events = []

    for event in eligibility_events:
        event_time = event["time"]

        # Find closest reward time after the event
        future_rewards = [r for r in reward_times if r > event_time]
        if future_rewards:
            reward_delay = min(future_rewards) - event_time
            # Eligibility decays exponentially
            remaining_eligibility = event["strength"] * (trace_decay ** reward_delay)

            if abs(remaining_eligibility) > 0.1:
                captured_events.append({
                    **event,
                    "reward_delay": reward_delay,
                    "effective_strength": remaining_eligibility
                })
            else:
                uncaptured_events.append(event)
        else:
            uncaptured_events.append(event)

    # Compute net weight change
    net_change = sum(e["effective_strength"] for e in captured_events)

    return {
        "valid": len(captured_events) > 0,
        "total_coincidences": len(eligibility_events),
        "captured_by_reward": len(captured_events),
        "uncaptured": len(uncaptured_events),
        "net_weight_change": net_change,
        "captured_events": captured_events,
        "trace_decay": trace_decay,
        "stdp_window": stdp_window,
        "interpretation": (
            "Eligibility traces bridge credit assignment gap: "
            f"{len(captured_events)}/{len(eligibility_events)} coincidences "
            f"captured by reward signal"
        )
    }


def compare_biological_to_rl_agent(
    bio_learning_trace: List[Tuple[float, float, float]],  # (state_value, action_prob, reward)
    rl_learning_trace: List[Tuple[float, float, float]],
    domain: str = "general"
) -> Dict[str, any]:
    """
    Compare biological learning dynamics to RL agent learning.

    Args:
        bio_learning_trace: Biological agent's (value, policy, reward) over time
        rl_learning_trace: RL agent's (value, policy, reward) over time
        domain: Learning domain for context

    Returns:
        Comparison analysis
    """
    n_bio = len(bio_learning_trace)
    n_rl = len(rl_learning_trace)
    n = min(n_bio, n_rl)

    if n == 0:
        return {"valid": False, "reason": "No data"}

    # Extract components
    bio_values = [t[0] for t in bio_learning_trace[:n]]
    bio_policies = [t[1] for t in bio_learning_trace[:n]]
    bio_rewards = [t[2] for t in bio_learning_trace[:n]]

    rl_values = [t[0] for t in rl_learning_trace[:n]]
    rl_policies = [t[1] for t in rl_learning_trace[:n]]
    rl_rewards = [t[2] for t in rl_learning_trace[:n]]

    # Value learning comparison
    def compute_correlation(x, y):
        if len(x) < 2:
            return 0.0
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
        var_x = sum((a - mean_x) ** 2 for a in x)
        var_y = sum((b - mean_y) ** 2 for b in y)
        if var_x > 0 and var_y > 0:
            return num / math.sqrt(var_x * var_y)
        return 0.0

    value_correlation = compute_correlation(bio_values, rl_values)
    policy_correlation = compute_correlation(bio_policies, rl_policies)

    # Learning speed comparison
    def learning_speed(values, rewards):
        """Estimate how quickly values track rewards."""
        if len(values) < 5:
            return 0.0
        # Compute value change relative to reward
        changes = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        reward_changes = [abs(rewards[i+1] - rewards[i]) for i in range(len(rewards)-1)]
        if sum(reward_changes) > 0:
            return sum(changes) / sum(reward_changes)
        return 0.0

    bio_speed = learning_speed(bio_values, bio_rewards)
    rl_speed = learning_speed(rl_values, rl_rewards)

    # Final performance
    bio_final_reward = sum(bio_rewards[-5:]) / min(5, len(bio_rewards))
    rl_final_reward = sum(rl_rewards[-5:]) / min(5, len(rl_rewards))

    return {
        "valid": True,
        "value_correlation": value_correlation,
        "policy_correlation": policy_correlation,
        "bio_learning_speed": bio_speed,
        "rl_learning_speed": rl_speed,
        "speed_ratio": bio_speed / rl_speed if rl_speed > 0 else float('inf'),
        "bio_final_reward": bio_final_reward,
        "rl_final_reward": rl_final_reward,
        "domain": domain,
        "interpretation": (
            f"Bio-RL value correlation: {value_correlation:.2f}, "
            f"policy correlation: {policy_correlation:.2f}. "
            f"Learning speed ratio: {bio_speed/rl_speed if rl_speed > 0 else 'N/A'}x"
        )
    }


# MCP Tool wrappers (will be exposed via server.py)

def dopamine_rpe_validator(
    firing_rates: List[float],
    expected_rewards: List[float],
    actual_rewards: List[float]
) -> str:
    """
    Validate if a neural signal matches dopamine RPE properties.

    Call this to check if firing patterns are consistent with reward
    prediction error encoding, as seen in VTA dopamine neurons.
    """
    result = validate_dopamine_rpe(firing_rates, expected_rewards, actual_rewards)
    return str(result)


def hebbian_vs_backprop_consistency(
    pre_activities: List[float],
    post_activities: List[float],
    weights: List[float],
    target_outputs: List[float]
) -> str:
    """
    Compare Hebbian learning to backpropagation.

    Call this to understand how biological local learning (Hebbian)
    relates to algorithmic gradient descent (backprop).
    """
    result = compare_hebbian_backprop(pre_activities, post_activities, weights, target_outputs)
    return str(result)
