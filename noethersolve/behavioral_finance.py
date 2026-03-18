"""Behavioral finance calculator with prospect theory and cognitive biases.

CRITICAL LLM BLIND SPOTS:
1. Models maximize EXPECTED VALUE, but humans maximize PROSPECT VALUE
2. Loss aversion coefficient λ ≈ 2.25 (losses hurt 2.25x more than gains feel good)
3. Probability weighting: certainty is OVERWEIGHTED (Allais paradox)
4. Temporal discounting: humans use HYPERBOLIC, not exponential
5. Reference dependence: value is RELATIVE to reference point, not absolute

These aren't "irrational quirks" - prospect theory structure IMPROVES alignment
(see KTO paper: Kahneman-Tversky Optimization outperforms DPO).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import math


# ─── Empirical Parameters (Tversky & Kahneman 1992) ─────────────────────────

# Loss aversion coefficient: losses hurt λ times more than equivalent gains
LOSS_AVERSION_LAMBDA = 2.25  # Median from experimental data

# Value function curvature (diminishing sensitivity)
VALUE_CURVATURE_ALPHA = 0.88  # Same for gains and losses

# Probability weighting parameter (certainty effect)
PROB_WEIGHT_GAMMA_GAINS = 0.61  # Overweight small p, underweight large p
PROB_WEIGHT_GAMMA_LOSSES = 0.69  # Less extreme for losses


class DecisionType(Enum):
    """Type of decision outcome."""
    GAIN = "gain"
    LOSS = "loss"
    MIXED = "mixed"


@dataclass
class ProspectTheoryReport:
    """Analysis using Kahneman-Tversky prospect theory."""
    outcomes: List[Tuple[float, float]]  # (outcome, probability) pairs
    reference_point: float
    expected_value: float
    prospect_value: float
    decision_type: DecisionType
    loss_aversion_impact: float  # How much λ affects the decision
    certainty_effect: bool  # Whether certainty effect is triggered
    recommendation: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "PROSPECT THEORY ANALYSIS",
            "=" * 55,
            "",
            "CRITICAL: LLMs use Expected Value. Humans use Prospect Value.",
            f"Loss aversion λ = {LOSS_AVERSION_LAMBDA} (losses hurt {LOSS_AVERSION_LAMBDA}x more)",
            "",
            f"Reference point: ${self.reference_point:,.2f}",
            f"Decision type: {self.decision_type.value.upper()}",
            "",
            "Outcomes:",
        ]
        for outcome, prob in self.outcomes:
            rel = outcome - self.reference_point
            sign = "+" if rel >= 0 else ""
            lines.append(f"  ${outcome:,.2f} ({prob:.1%}) → {sign}${rel:,.2f} from reference")

        lines.extend([
            "",
            f"Expected Value (EV):     ${self.expected_value:,.2f}",
            f"Prospect Value (PV):     {self.prospect_value:+.3f}",
            "",
        ])

        if self.certainty_effect:
            lines.append("⚠️  CERTAINTY EFFECT TRIGGERED")
            lines.append("   Humans overweight certain outcomes vs high probability")
            lines.append("")

        if abs(self.loss_aversion_impact) > 0.1:
            lines.append(f"Loss aversion impact: {self.loss_aversion_impact:+.2f}")
            lines.append("   (Negative = losses dominate the decision)")
            lines.append("")

        lines.append(f"RECOMMENDATION: {self.recommendation}")

        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)


@dataclass
class TemporalDiscountReport:
    """Analysis of time preference (hyperbolic vs exponential)."""
    future_value: float
    time_periods: float
    discount_rate: float
    exponential_pv: float  # Classical finance
    hyperbolic_pv: float   # Behavioral reality
    present_bias: float    # Hyperbolic - Exponential (usually > 0)
    preference_reversal_risk: bool
    k_parameter: float  # Hyperbolic discount rate
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "TEMPORAL DISCOUNTING ANALYSIS",
            "=" * 55,
            "",
            "CRITICAL: Humans discount HYPERBOLICALLY, not exponentially.",
            "This causes present bias and preference reversals.",
            "",
            f"Future value: ${self.future_value:,.2f}",
            f"Time periods: {self.time_periods}",
            f"Discount rate: {self.discount_rate:.1%}",
            "",
            f"Exponential PV (classical): ${self.exponential_pv:,.2f}",
            f"Hyperbolic PV (behavioral): ${self.hyperbolic_pv:,.2f}",
            f"Present bias gap:           ${self.present_bias:,.2f}",
            "",
        ]

        if self.preference_reversal_risk:
            lines.append("⚠️  PREFERENCE REVERSAL RISK")
            lines.append("   Choice may flip as options approach present time")
            lines.append("")

        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)


@dataclass
class AllaisParadoxReport:
    """Detection of Allais paradox / certainty effect violations."""
    lottery_a: List[Tuple[float, float]]  # Certain or near-certain option
    lottery_b: List[Tuple[float, float]]  # Higher EV risky option
    ev_a: float
    ev_b: float
    pv_a: float
    pv_b: float
    ev_choice: str  # "A" or "B" based on EV
    pt_choice: str  # "A" or "B" based on prospect theory
    paradox_detected: bool
    explanation: str

    def __str__(self) -> str:
        lines = [
            "ALLAIS PARADOX ANALYSIS",
            "=" * 55,
            "",
            "The Allais Paradox: humans prefer certainty over higher EV",
            "",
            "Lottery A:",
        ]
        for outcome, prob in self.lottery_a:
            lines.append(f"  ${outcome:,.0f} with {prob:.0%} probability")

        lines.append("")
        lines.append("Lottery B:")
        for outcome, prob in self.lottery_b:
            lines.append(f"  ${outcome:,.0f} with {prob:.0%} probability")

        lines.extend([
            "",
            f"Expected Value A: ${self.ev_a:,.0f}",
            f"Expected Value B: ${self.ev_b:,.0f}",
            f"EV-maximizing choice: {self.ev_choice}",
            "",
            f"Prospect Value A: {self.pv_a:+.3f}",
            f"Prospect Value B: {self.pv_b:+.3f}",
            f"Prospect Theory choice: {self.pt_choice}",
            "",
        ])

        if self.paradox_detected:
            lines.append("⚠️  ALLAIS PARADOX DETECTED")
            lines.append(f"   {self.explanation}")
        else:
            lines.append("No paradox: EV and PT agree")

        return "\n".join(lines)


@dataclass
class LossAversionReport:
    """Analysis of loss aversion in a decision."""
    gain_amount: float
    loss_amount: float
    gain_probability: float
    loss_probability: float
    expected_value: float
    prospect_value: float
    breakeven_gain: float  # Gain needed to offset loss (accounting for λ)
    should_take_gamble_ev: bool
    should_take_gamble_pt: bool
    loss_aversion_coefficient: float

    def __str__(self) -> str:
        lines = [
            "LOSS AVERSION ANALYSIS",
            "=" * 55,
            "",
            f"Loss aversion coefficient λ = {self.loss_aversion_coefficient}",
            f"Interpretation: losses hurt {self.loss_aversion_coefficient}x more than gains",
            "",
            f"Potential gain: ${self.gain_amount:,.2f} ({self.gain_probability:.0%})",
            f"Potential loss: ${self.loss_amount:,.2f} ({self.loss_probability:.0%})",
            "",
            f"Expected Value: ${self.expected_value:+,.2f}",
            f"Prospect Value: {self.prospect_value:+.3f}",
            "",
            f"Breakeven gain (to offset loss): ${self.breakeven_gain:,.2f}",
            f"  (This is {self.breakeven_gain/self.loss_amount:.1f}x the loss amount)",
            "",
            f"EV says: {'TAKE' if self.should_take_gamble_ev else 'REJECT'} the gamble",
            f"PT says: {'TAKE' if self.should_take_gamble_pt else 'REJECT'} the gamble",
        ]

        if self.should_take_gamble_ev != self.should_take_gamble_pt:
            lines.append("")
            lines.append("⚠️  EV AND PT DISAGREE")
            lines.append("   Humans will likely follow PT, not EV")

        return "\n".join(lines)


# ─── Core Functions ─────────────────────────────────────────────────────────

def prospect_value_function(
    x: float,
    reference: float = 0.0,
    alpha: float = VALUE_CURVATURE_ALPHA,
    lambda_: float = LOSS_AVERSION_LAMBDA,
) -> float:
    """Kahneman-Tversky value function.

    v(x) = {
        (x - ref)^α           if x >= ref  [GAINS]
        -λ(ref - x)^α         if x < ref   [LOSSES]
    }

    CRITICAL: This is ASYMMETRIC. Losses are weighted by λ ≈ 2.25.
    """
    relative = x - reference

    if relative >= 0:
        # Gain: concave (diminishing sensitivity)
        return relative ** alpha
    else:
        # Loss: convex, weighted by λ
        return -lambda_ * ((-relative) ** alpha)


def probability_weight(
    p: float,
    gamma: float = PROB_WEIGHT_GAMMA_GAINS,
) -> float:
    """Cumulative prospect theory probability weighting.

    π(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)

    CRITICAL: This OVERWEIGHTS small probabilities and UNDERWEIGHTS large ones.
    At p=1.0, π(1.0) = 1.0 exactly (certainty preserved).
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    numerator = p ** gamma
    denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)

    return numerator / denominator


def calculate_prospect_value(
    outcomes: List[Tuple[float, float]],
    reference: float = 0.0,
    alpha: float = VALUE_CURVATURE_ALPHA,
    lambda_: float = LOSS_AVERSION_LAMBDA,
    gamma_gains: float = PROB_WEIGHT_GAMMA_GAINS,
    gamma_losses: float = PROB_WEIGHT_GAMMA_LOSSES,
) -> float:
    """Calculate prospect theory value of a lottery.

    PV = Σ π(p_i) × v(x_i - reference)
    """
    pv = 0.0

    for outcome, prob in outcomes:
        relative = outcome - reference
        gamma = gamma_gains if relative >= 0 else gamma_losses
        weighted_prob = probability_weight(prob, gamma)
        value = prospect_value_function(outcome, reference, alpha, lambda_)
        pv += weighted_prob * value

    return pv


def calculate_expected_value(outcomes: List[Tuple[float, float]]) -> float:
    """Calculate expected value of a lottery."""
    return sum(outcome * prob for outcome, prob in outcomes)


def analyze_prospect(
    outcomes: List[Tuple[float, float]],
    reference: float = 0.0,
) -> ProspectTheoryReport:
    """Analyze a decision using prospect theory.

    outcomes: List of (outcome_value, probability) tuples
    reference: Reference point for gains/losses (default 0)
    """
    ev = calculate_expected_value(outcomes)
    pv = calculate_prospect_value(outcomes, reference)

    # Determine decision type
    gains = sum(1 for o, _ in outcomes if o > reference)
    losses = sum(1 for o, _ in outcomes if o < reference)

    if gains > 0 and losses > 0:
        decision_type = DecisionType.MIXED
    elif gains > 0:
        decision_type = DecisionType.GAIN
    else:
        decision_type = DecisionType.LOSS

    # Check for certainty effect
    has_certain = any(p >= 0.99 for _, p in outcomes)
    has_high_prob = any(0.8 <= p < 0.99 for _, p in outcomes)
    certainty_effect = has_certain and has_high_prob

    # Calculate loss aversion impact
    # Compare PV with and without loss aversion
    pv_no_la = calculate_prospect_value(outcomes, reference, lambda_=1.0)
    loss_aversion_impact = pv - pv_no_la

    # Generate recommendation
    notes = []
    if pv > 0 and ev > 0:
        recommendation = "TAKE (both EV and PT positive)"
    elif pv < 0 and ev < 0:
        recommendation = "REJECT (both EV and PT negative)"
    elif pv > 0 and ev < 0:
        recommendation = "PT says TAKE, but EV negative - risk-seeking in losses?"
        notes.append("Humans often take negative-EV gambles to avoid certain losses")
    elif pv < 0 and ev > 0:
        recommendation = "PT says REJECT despite positive EV - loss aversion dominates"
        notes.append(f"Loss aversion (λ={LOSS_AVERSION_LAMBDA}) makes this feel like a bad deal")
    else:
        recommendation = "Borderline decision"

    return ProspectTheoryReport(
        outcomes=outcomes,
        reference_point=reference,
        expected_value=ev,
        prospect_value=pv,
        decision_type=decision_type,
        loss_aversion_impact=loss_aversion_impact,
        certainty_effect=certainty_effect,
        recommendation=recommendation,
        notes=notes,
    )


def exponential_discount(
    future_value: float,
    periods: float,
    rate: float,
) -> float:
    """Classical exponential discounting: PV = FV × e^(-rT)"""
    return future_value * math.exp(-rate * periods)


def hyperbolic_discount(
    future_value: float,
    periods: float,
    k: float = 0.1,
    s: float = 1.0,
) -> float:
    """Behavioral hyperbolic discounting: V(t) = A / (1 + kt)^s

    k: discount rate (higher = more impatient)
    s: curvature (s=1 is standard hyperbolic)
    """
    return future_value / ((1 + k * periods) ** s)


def analyze_temporal_discounting(
    future_value: float,
    periods: float,
    annual_rate: float = 0.05,
    k: float = 0.1,
) -> TemporalDiscountReport:
    """Compare exponential vs hyperbolic discounting.

    CRITICAL: Humans use hyperbolic discounting, which causes:
    1. Present bias (overvalue immediate rewards)
    2. Preference reversals (choices flip as time passes)
    """
    exp_pv = exponential_discount(future_value, periods, annual_rate)
    hyp_pv = hyperbolic_discount(future_value, periods, k)
    present_bias = hyp_pv - exp_pv

    notes = []

    # Check for preference reversal risk
    # This happens when hyperbolic > exponential now but reverses later
    preference_reversal = False
    if periods > 1:
        # Check at t=0 vs t=periods/2
        ratio_now = hyp_pv / exp_pv if exp_pv > 0 else 1
        exp_later = exponential_discount(future_value, periods / 2, annual_rate)
        hyp_later = hyperbolic_discount(future_value, periods / 2, k)
        ratio_later = hyp_later / exp_later if exp_later > 0 else 1

        if (ratio_now > 1 and ratio_later < 1) or (ratio_now < 1 and ratio_later > 1):
            preference_reversal = True
            notes.append("Preference may reverse as decision approaches")

    if present_bias > 0:
        notes.append(f"Present bias: hyperbolic values ${present_bias:,.0f} more than exponential")
        notes.append("Human likely to choose smaller-sooner over larger-later")
    elif present_bias < 0:
        notes.append("Unusual: exponential values more than hyperbolic")

    return TemporalDiscountReport(
        future_value=future_value,
        time_periods=periods,
        discount_rate=annual_rate,
        exponential_pv=exp_pv,
        hyperbolic_pv=hyp_pv,
        present_bias=present_bias,
        preference_reversal_risk=preference_reversal,
        k_parameter=k,
        notes=notes,
    )


def analyze_allais_paradox(
    certain_amount: float = 1_000_000,
    risky_high: float = 5_000_000,
    risky_high_prob: float = 0.89,
    risky_mid: float = 1_000_000,
    risky_mid_prob: float = 0.10,
) -> AllaisParadoxReport:
    """Analyze the classic Allais paradox.

    Default: $1M certain vs ($5M at 89%, $1M at 10%, $0 at 1%)
    """
    lottery_a = [(certain_amount, 1.0)]
    lottery_b = [
        (risky_high, risky_high_prob),
        (risky_mid, risky_mid_prob),
        (0, 1 - risky_high_prob - risky_mid_prob),
    ]

    ev_a = calculate_expected_value(lottery_a)
    ev_b = calculate_expected_value(lottery_b)
    pv_a = calculate_prospect_value(lottery_a)
    pv_b = calculate_prospect_value(lottery_b)

    ev_choice = "A" if ev_a >= ev_b else "B"
    pt_choice = "A" if pv_a >= pv_b else "B"

    paradox = (ev_choice != pt_choice)

    if paradox:
        if ev_choice == "B" and pt_choice == "A":
            explanation = "Certainty effect: humans prefer certain $1M over higher-EV gamble"
        else:
            explanation = "Reverse paradox: PT prefers gamble over certainty"
    else:
        explanation = "No paradox detected"

    return AllaisParadoxReport(
        lottery_a=lottery_a,
        lottery_b=lottery_b,
        ev_a=ev_a,
        ev_b=ev_b,
        pv_a=pv_a,
        pv_b=pv_b,
        ev_choice=ev_choice,
        pt_choice=pt_choice,
        paradox_detected=paradox,
        explanation=explanation,
    )


def analyze_loss_aversion(
    gain: float,
    loss: float,
    gain_prob: float = 0.5,
    loss_prob: float = 0.5,
    lambda_: float = LOSS_AVERSION_LAMBDA,
) -> LossAversionReport:
    """Analyze a gamble through the lens of loss aversion.

    CRITICAL: To accept a 50-50 gamble, the gain must be ~2.25x the loss.
    """
    outcomes = [(gain, gain_prob), (-loss, loss_prob)]

    ev = gain * gain_prob - loss * loss_prob
    pv = calculate_prospect_value(outcomes, reference=0.0, lambda_=lambda_)

    # Breakeven gain: what gain makes PV = 0?
    # v(G) × π(p_gain) = λ × v(L) × π(p_loss)
    # For 50-50: G^α = λ × L^α → G = L × λ^(1/α)
    breakeven = loss * (lambda_ ** (1 / VALUE_CURVATURE_ALPHA))

    return LossAversionReport(
        gain_amount=gain,
        loss_amount=loss,
        gain_probability=gain_prob,
        loss_probability=loss_prob,
        expected_value=ev,
        prospect_value=pv,
        breakeven_gain=breakeven,
        should_take_gamble_ev=(ev > 0),
        should_take_gamble_pt=(pv > 0),
        loss_aversion_coefficient=lambda_,
    )


def mental_accounting_violation(
    account_a_gain: float,
    account_b_loss: float,
    segregated_value: float,
    integrated_value: float,
) -> Dict:
    """Detect mental accounting violation (failure of fungibility).

    If segregated_value ≠ integrated_value for the same net cash flow,
    mental accounting is present.
    """
    net_cash_flow = account_a_gain + account_b_loss
    violation_magnitude = abs(segregated_value - integrated_value)

    fungibility_holds = violation_magnitude < 0.01 * abs(net_cash_flow) if net_cash_flow != 0 else violation_magnitude < 0.01

    return {
        "account_a_gain": account_a_gain,
        "account_b_loss": account_b_loss,
        "net_cash_flow": net_cash_flow,
        "segregated_valuation": segregated_value,
        "integrated_valuation": integrated_value,
        "violation_magnitude": violation_magnitude,
        "fungibility_holds": fungibility_holds,
        "mental_accounting_present": not fungibility_holds,
    }


def framing_effect_demo(amount: float = 100) -> str:
    """Demonstrate the framing effect with equivalent choices.

    Shows how the SAME outcome framed differently leads to different preferences.
    """
    lines = [
        "FRAMING EFFECT DEMONSTRATION",
        "=" * 55,
        "",
        f"You have ${amount:,.0f}.",
        "",
        "FRAME 1 (Gain frame):",
        "  A: Keep $50 for sure",
        "  B: 50% chance to keep all $100, 50% chance to lose all",
        "  → Most people choose A (risk-averse in gains)",
        "",
        "FRAME 2 (Loss frame):",
        "  A: Lose $50 for sure",
        "  B: 50% chance to lose nothing, 50% chance to lose all $100",
        "  → Most people choose B (risk-seeking in losses)",
        "",
        "CRITICAL: These are IDENTICAL choices!",
        "  Frame 1A = Frame 2A = end with $50",
        "  Frame 1B = Frame 2B = 50% $100, 50% $0",
        "",
        "Prospect theory explains this:",
        f"  • Reference point shifts between ${amount} and $0",
        "  • Concave utility in gains → risk aversion",
        "  • Convex utility in losses → risk seeking",
        "",
        "LLMs typically miss this because they compute EV,",
        "not prospect value with reference-dependent framing.",
    ]
    return "\n".join(lines)


def herding_cascade_threshold(
    prior_probability: float,
    signal_precision: float,
    n_predecessors: int,
) -> Dict:
    """Calculate when an information cascade triggers.

    In a cascade, agents rationally ignore their private signal and follow
    the crowd. This happens when predecessor consensus is strong enough.

    prior_probability: Prior P(state = good)
    signal_precision: P(signal = state) - accuracy of private signal
    n_predecessors: Number of previous agents who chose same action
    """
    # Likelihood ratio after n concordant predecessors
    # L_n = (q/(1-q))^n where q = signal_precision

    if signal_precision <= 0.5:
        return {
            "error": "Signal precision must be > 0.5 for informative signals"
        }

    q = signal_precision
    lr = (q / (1 - q)) ** n_predecessors

    # Posterior after seeing n same choices (assuming all got good signals)
    prior_odds = prior_probability / (1 - prior_probability)
    posterior_odds = prior_odds * lr
    posterior_prob = posterior_odds / (1 + posterior_odds)

    # Cascade threshold: when does agent ignore own signal?
    # Agent cascades when posterior is so extreme that their signal can't flip it
    # This happens when: posterior > q / (1-q + q×LR) approximately

    cascade_triggered = posterior_prob > signal_precision

    # Minimum n for cascade
    min_n_for_cascade = math.ceil(
        math.log(1 / prior_odds) / math.log(q / (1 - q))
    ) if prior_probability < 0.5 else 0

    return {
        "prior_probability": prior_probability,
        "signal_precision": signal_precision,
        "n_predecessors": n_predecessors,
        "posterior_probability": posterior_prob,
        "cascade_triggered": cascade_triggered,
        "min_predecessors_for_cascade": max(0, min_n_for_cascade),
        "explanation": (
            "Agent will IGNORE their private signal and follow crowd"
            if cascade_triggered else
            "Agent will use their private signal (no cascade yet)"
        ),
    }
