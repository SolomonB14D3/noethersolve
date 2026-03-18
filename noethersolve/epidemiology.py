"""noethersolve.epidemiology — Computational tools for infectious disease modeling.

Provides verified calculations for:
- Basic reproduction number (R0) and effective reproduction number (Rt)
- SIR/SEIR model parameters and dynamics
- Herd immunity threshold
- Doubling time and exponential growth rate
- Attack rate (final epidemic size)
- Generation interval and serial interval
- Vaccine efficacy and impact

Common LLM errors this module corrects:
- Confusing R0 (basic) vs Rt (effective) reproduction numbers
- Wrong formula for herd immunity threshold (it's 1 - 1/R0, not 1/R0)
- Confusing attack rate with incidence rate
- Misunderstanding serial interval vs generation interval
- Wrong relationship between doubling time and growth rate

Conservation law philosophy: Epidemiological models ARE conservation laws.
The SIR model conserves total population (S + I + R = N). The rate of new
infections equals the rate of transition from S to I, which equals the
rate of eventual recovery/death (conservation of infection flux).

Usage:
    from noethersolve.epidemiology import (
        herd_immunity_threshold, reproduction_number, doubling_time,
        attack_rate, sir_model, vaccine_impact,
    )

    # Herd immunity threshold
    r = herd_immunity_threshold(R0=3.0)
    print(r)  # HIT = 66.7% for R0=3

    # Doubling time from growth rate
    r = doubling_time(growth_rate=0.1)
    print(r)  # T_d = 6.93 days
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

LN2 = math.log(2)  # 0.693...


# ─── Report Dataclasses ───────────────────────────────────────────────────────

@dataclass
class HerdImmunityReport:
    """Herd immunity threshold calculation."""
    R0: float
    threshold: float  # fraction (0-1) that must be immune
    threshold_pct: float  # percentage
    susceptible_at_equilibrium: float
    formula: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Herd Immunity Threshold", "=" * 60]
        lines.append(f"  R₀ = {self.R0:.2f}")
        lines.append(f"  Formula: {self.formula}")
        lines.append("-" * 60)
        lines.append(f"  HIT = 1 - 1/R₀ = {self.threshold:.4f} = {self.threshold_pct:.1f}%")
        lines.append(f"  Susceptible at equilibrium: {self.susceptible_at_equilibrium:.4f}")
        lines.append("-" * 60)
        lines.append(f"  Interpretation: {self.threshold_pct:.0f}% of population must be")
        lines.append("  immune to prevent sustained transmission (Rt < 1)")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ReproductionNumberReport:
    """Reproduction number calculation and interpretation."""
    R: float
    R_type: str  # "basic" (R0) or "effective" (Rt)
    susceptible_fraction: Optional[float]
    interpretation: str
    growth_rate: float  # per generation
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, f"  Reproduction Number ({self.R_type.title()})", "=" * 60]
        if self.R_type == "basic":
            lines.append(f"  R₀ = {self.R:.3f}")
            lines.append("  (Average secondary infections from one case in")
            lines.append("   fully susceptible population)")
        else:
            lines.append(f"  Rt = {self.R:.3f}")
            if self.susceptible_fraction is not None:
                lines.append(f"  Susceptible fraction: {self.susceptible_fraction:.3f}")
        lines.append("-" * 60)
        lines.append(f"  Interpretation: {self.interpretation}")
        if self.R > 1:
            lines.append(f"  Per-generation growth: {(self.R - 1) * 100:.1f}%")
        else:
            lines.append(f"  Per-generation decline: {(1 - self.R) * 100:.1f}%")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DoublingTimeReport:
    """Doubling time calculation."""
    doubling_time: float  # time units (days)
    growth_rate: float  # per time unit (per day)
    halving_time: Optional[float]  # if declining
    formula_used: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Doubling Time / Growth Rate", "=" * 60]
        if self.growth_rate > 0:
            lines.append(f"  Growth rate r = {self.growth_rate:.4f} per day")
            lines.append(f"  Doubling time T_d = ln(2)/r = {self.doubling_time:.2f} days")
        elif self.growth_rate < 0:
            lines.append(f"  Decline rate r = {self.growth_rate:.4f} per day")
            if self.halving_time:
                lines.append(f"  Halving time T_h = ln(2)/|r| = {self.halving_time:.2f} days")
        else:
            lines.append("  Growth rate = 0 (stable/endemic)")
        lines.append(f"  Formula: {self.formula_used}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class AttackRateReport:
    """Final attack rate (epidemic size) calculation."""
    R0: float
    attack_rate: float  # fraction infected by end
    attack_rate_pct: float
    final_susceptible: float
    method: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Attack Rate (Final Epidemic Size)", "=" * 60]
        lines.append(f"  R₀ = {self.R0:.2f}")
        lines.append(f"  Method: {self.method}")
        lines.append("-" * 60)
        lines.append(f"  Final attack rate: {self.attack_rate:.4f} = {self.attack_rate_pct:.1f}%")
        lines.append(f"  Final susceptible: {self.final_susceptible:.4f}")
        lines.append("-" * 60)
        lines.append(f"  Interpretation: {self.attack_rate_pct:.0f}% of population will be")
        lines.append("  infected by the end of the epidemic (SIR model)")
        if self.R0 < 1:
            lines.append("  R₀ < 1: Epidemic dies out quickly, low attack rate")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SIRReport:
    """SIR model parameters and steady state."""
    beta: float  # transmission rate
    gamma: float  # recovery rate
    R0: float
    generation_time: float  # 1/gamma
    herd_immunity: float
    peak_infected: Optional[float]
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  SIR Model Analysis", "=" * 60]
        lines.append(f"  Transmission rate β = {self.beta:.4f} per day")
        lines.append(f"  Recovery rate γ = {self.gamma:.4f} per day")
        lines.append(f"  R₀ = β/γ = {self.R0:.3f}")
        lines.append("-" * 60)
        lines.append(f"  Generation time = 1/γ = {self.generation_time:.2f} days")
        lines.append(f"  Herd immunity threshold = {self.herd_immunity * 100:.1f}%")
        if self.peak_infected is not None:
            lines.append(f"  Peak infected fraction ≈ {self.peak_infected * 100:.1f}%")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class VaccineImpactReport:
    """Vaccine impact on epidemic dynamics."""
    R0: float
    vaccine_efficacy: float  # VE (0-1)
    coverage: float  # vaccination coverage (0-1)
    Rt_post_vaccine: float
    herd_immunity_reached: bool
    critical_coverage: float  # coverage needed for herd immunity
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Vaccine Impact Assessment", "=" * 60]
        lines.append(f"  R₀ = {self.R0:.2f}")
        lines.append(f"  Vaccine efficacy VE = {self.vaccine_efficacy * 100:.1f}%")
        lines.append(f"  Coverage = {self.coverage * 100:.1f}%")
        lines.append("-" * 60)
        lines.append(f"  Effective immunity from vaccine: {self.coverage * self.vaccine_efficacy * 100:.1f}%")
        lines.append(f"  Rt after vaccination = {self.Rt_post_vaccine:.3f}")
        if self.herd_immunity_reached:
            lines.append("  ✓ Herd immunity ACHIEVED (Rt < 1)")
        else:
            lines.append("  ✗ Herd immunity NOT achieved (Rt ≥ 1)")
            lines.append(f"  Critical coverage needed: {self.critical_coverage * 100:.1f}%")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class GenerationIntervalReport:
    """Generation interval and serial interval analysis."""
    generation_interval: float  # mean (days)
    serial_interval: Optional[float]  # mean (days)
    difference: Optional[float]
    explanation: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Generation / Serial Interval", "=" * 60]
        lines.append(f"  Generation interval: {self.generation_interval:.2f} days")
        lines.append("  (Time between infection of index and secondary case)")
        if self.serial_interval is not None:
            lines.append(f"  Serial interval: {self.serial_interval:.2f} days")
            lines.append("  (Time between symptom onset of index and secondary case)")
            if self.difference is not None:
                lines.append(f"  Difference: {self.difference:.2f} days")
        lines.append("-" * 60)
        lines.append(f"  {self.explanation}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Public API ───────────────────────────────────────────────────────────────

def herd_immunity_threshold(R0: float) -> HerdImmunityReport:
    """Calculate herd immunity threshold from R0.

    HIT = 1 - 1/R0

    This is the fraction of the population that must be immune (through
    vaccination or natural infection) to prevent sustained transmission.

    Args:
        R0: Basic reproduction number (must be > 0)

    Returns:
        HerdImmunityReport with threshold and interpretation.

    Common LLM error: Some models say HIT = 1/R0 (wrong). The correct
    formula is HIT = 1 - 1/R0.
    """
    if R0 <= 0:
        raise ValueError(f"R0 must be positive, got {R0}")

    if R0 <= 1:
        # No sustained transmission possible
        threshold = 0.0
        notes = ["R0 ≤ 1: No herd immunity needed - epidemic cannot sustain"]
    else:
        threshold = 1.0 - 1.0 / R0
        notes = []

    return HerdImmunityReport(
        R0=R0,
        threshold=threshold,
        threshold_pct=threshold * 100,
        susceptible_at_equilibrium=1.0 / R0 if R0 > 1 else 1.0,
        formula="HIT = 1 - 1/R₀",
        notes=notes,
    )


def reproduction_number(
    R0: Optional[float] = None,
    susceptible_fraction: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> ReproductionNumberReport:
    """Calculate basic or effective reproduction number.

    R0 = β/γ (from SIR parameters) or provided directly.
    Rt = R0 × S (effective, accounting for immunity)

    Args:
        R0: Basic reproduction number (if known)
        susceptible_fraction: Fraction still susceptible (S/N)
        beta: Transmission rate (per day)
        gamma: Recovery rate (per day)

    Returns:
        ReproductionNumberReport with R value and interpretation.
    """
    # Calculate R0 from parameters if not provided
    if R0 is None:
        if beta is not None and gamma is not None:
            if gamma <= 0:
                raise ValueError(f"gamma must be positive, got {gamma}")
            R0 = beta / gamma
        else:
            raise ValueError("Provide either R0 or both beta and gamma")

    if R0 <= 0:
        raise ValueError(f"R0 must be positive, got {R0}")

    # Calculate Rt if susceptible fraction provided
    if susceptible_fraction is not None:
        if not 0 <= susceptible_fraction <= 1:
            raise ValueError(f"susceptible_fraction must be in [0,1], got {susceptible_fraction}")
        Rt = R0 * susceptible_fraction
        R_type = "effective"
        R_val = Rt
    else:
        R_type = "basic"
        R_val = R0

    # Interpretation
    if R_val > 1:
        interpretation = f"Epidemic GROWING - each case produces {R_val:.2f} new cases"
    elif R_val < 1:
        interpretation = f"Epidemic DECLINING - each case produces {R_val:.2f} new cases"
    else:
        interpretation = "Epidemic STABLE - each case produces exactly 1 new case"

    return ReproductionNumberReport(
        R=R_val,
        R_type=R_type,
        susceptible_fraction=susceptible_fraction,
        interpretation=interpretation,
        growth_rate=R_val - 1,
        notes=[],
    )


def doubling_time(
    growth_rate: Optional[float] = None,
    R0: Optional[float] = None,
    generation_time: Optional[float] = None,
) -> DoublingTimeReport:
    """Calculate epidemic doubling time.

    T_d = ln(2) / r

    Growth rate r can be provided directly or computed from R0 and
    generation time: r = (R0 - 1) / T_g (approximate for discrete generations)

    For continuous SIR: r = γ(R0 - 1)

    Args:
        growth_rate: Exponential growth rate per day (r)
        R0: Basic reproduction number (alternative)
        generation_time: Mean generation time in days (with R0)

    Returns:
        DoublingTimeReport with doubling/halving time.
    """
    if growth_rate is not None:
        r = growth_rate
        formula = "T_d = ln(2)/r"
    elif R0 is not None and generation_time is not None:
        if generation_time <= 0:
            raise ValueError(f"generation_time must be positive, got {generation_time}")
        # Euler-Lotka approximation: r ≈ ln(R0) / T_g
        r = math.log(R0) / generation_time if R0 > 0 else 0
        formula = "r = ln(R₀)/T_g, then T_d = ln(2)/r"
    else:
        raise ValueError("Provide either growth_rate or both R0 and generation_time")

    notes = []
    if r > 0:
        T_d = LN2 / r
        T_h = None
    elif r < 0:
        T_d = float('inf')
        T_h = LN2 / abs(r)
        notes.append("Negative growth rate - epidemic declining")
    else:
        T_d = float('inf')
        T_h = None
        notes.append("Zero growth rate - epidemic stable/endemic")

    return DoublingTimeReport(
        doubling_time=T_d,
        growth_rate=r,
        halving_time=T_h,
        formula_used=formula,
        notes=notes,
    )


def attack_rate(R0: float, method: str = "exact") -> AttackRateReport:
    """Calculate final attack rate (epidemic size) for SIR model.

    The final size equation: S_∞ = exp(-R0 × (1 - S_∞))
    Attack rate = 1 - S_∞

    Args:
        R0: Basic reproduction number
        method: "exact" (Newton's method) or "approximate"

    Returns:
        AttackRateReport with final epidemic size.
    """
    if R0 <= 0:
        raise ValueError(f"R0 must be positive, got {R0}")

    notes = []

    if R0 <= 1:
        # No epidemic
        final_S = 1.0
        AR = 0.0
        notes.append("R0 ≤ 1: No sustained epidemic possible")
    else:
        if method == "exact":
            # Solve S_∞ = exp(-R0 × (1 - S_∞)) via Newton's method
            S = 0.5  # initial guess
            for _ in range(50):
                f = S - math.exp(-R0 * (1 - S))
                f_prime = 1 - R0 * math.exp(-R0 * (1 - S))
                if abs(f_prime) < 1e-12:
                    break
                S_new = S - f / f_prime
                if abs(S_new - S) < 1e-12:
                    S = S_new
                    break
                S = max(0, min(1, S_new))
            final_S = S
        else:
            # Approximate formula: AR ≈ 1 - exp(-R0)
            final_S = math.exp(-R0)
            notes.append("Using approximate formula (valid for large R0)")

        AR = 1.0 - final_S

    return AttackRateReport(
        R0=R0,
        attack_rate=AR,
        attack_rate_pct=AR * 100,
        final_susceptible=final_S,
        method=method,
        notes=notes,
    )


def sir_model(
    beta: float,
    gamma: float,
    initial_infected_fraction: float = 0.001,
) -> SIRReport:
    """Analyze SIR model parameters.

    dS/dt = -β × S × I
    dI/dt = β × S × I - γ × I
    dR/dt = γ × I

    Args:
        beta: Transmission rate (per day)
        gamma: Recovery rate (per day)
        initial_infected_fraction: I(0)/N

    Returns:
        SIRReport with model analysis.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if not 0 < initial_infected_fraction < 1:
        raise ValueError(f"initial_infected_fraction must be in (0,1), got {initial_infected_fraction}")

    R0 = beta / gamma
    generation_time = 1.0 / gamma
    HIT = 1.0 - 1.0 / R0 if R0 > 1 else 0.0

    notes = []
    if R0 > 1:
        # Peak infected fraction (approximate formula)
        # I_max ≈ 1 - (1 + ln(R0))/R0
        peak = 1.0 - (1 + math.log(R0)) / R0
        peak = max(0, peak)
    else:
        peak = None
        notes.append("R0 ≤ 1: No epidemic peak (immediate decline)")

    return SIRReport(
        beta=beta,
        gamma=gamma,
        R0=R0,
        generation_time=generation_time,
        herd_immunity=HIT,
        peak_infected=peak,
        notes=notes,
    )


def vaccine_impact(
    R0: float,
    vaccine_efficacy: float,
    coverage: float,
) -> VaccineImpactReport:
    """Calculate vaccine impact on epidemic dynamics.

    Rt = R0 × (1 - VE × coverage)

    Herd immunity achieved when VE × coverage ≥ 1 - 1/R0

    Args:
        R0: Basic reproduction number
        vaccine_efficacy: Vaccine efficacy VE (0-1)
        coverage: Vaccination coverage (0-1)

    Returns:
        VaccineImpactReport with Rt and herd immunity assessment.
    """
    if R0 <= 0:
        raise ValueError(f"R0 must be positive, got {R0}")
    if not 0 <= vaccine_efficacy <= 1:
        raise ValueError(f"vaccine_efficacy must be in [0,1], got {vaccine_efficacy}")
    if not 0 <= coverage <= 1:
        raise ValueError(f"coverage must be in [0,1], got {coverage}")

    # Effective immunity from vaccination
    effective_immunity = vaccine_efficacy * coverage

    # Post-vaccination Rt
    Rt = R0 * (1 - effective_immunity)

    # Critical coverage for herd immunity
    HIT = 1.0 - 1.0 / R0 if R0 > 1 else 0.0
    if vaccine_efficacy > 0:
        critical_coverage = HIT / vaccine_efficacy
        critical_coverage = min(1.0, critical_coverage)  # cap at 100%
    else:
        critical_coverage = 1.0

    herd_reached = Rt < 1

    notes = []
    if critical_coverage > 1.0:
        notes.append(f"VE={vaccine_efficacy*100:.0f}% cannot achieve herd immunity alone")
    if herd_reached and R0 > 1:
        notes.append("Vaccination program sufficient to control epidemic")

    return VaccineImpactReport(
        R0=R0,
        vaccine_efficacy=vaccine_efficacy,
        coverage=coverage,
        Rt_post_vaccine=Rt,
        herd_immunity_reached=herd_reached,
        critical_coverage=critical_coverage,
        notes=notes,
    )


def generation_interval(
    mean_generation: float,
    mean_serial: Optional[float] = None,
    incubation_period: Optional[float] = None,
) -> GenerationIntervalReport:
    """Analyze generation interval and serial interval.

    Generation interval: time from infection of index to infection of secondary
    Serial interval: time from symptom onset of index to symptom onset of secondary

    Serial interval = Generation interval + (incubation_secondary - incubation_index)

    On average, serial ≈ generation interval, but serial can be shorter if
    presymptomatic transmission occurs.

    Args:
        mean_generation: Mean generation interval (days)
        mean_serial: Mean serial interval (days)
        incubation_period: Mean incubation period (days)

    Returns:
        GenerationIntervalReport with analysis.
    """
    if mean_generation <= 0:
        raise ValueError(f"mean_generation must be positive, got {mean_generation}")

    notes = []
    difference = None

    if mean_serial is not None:
        if mean_serial <= 0:
            raise ValueError(f"mean_serial must be positive, got {mean_serial}")
        difference = mean_serial - mean_generation
        if difference < 0:
            explanation = ("Serial interval < generation interval suggests "
                          "significant presymptomatic transmission")
        elif difference > 0:
            explanation = ("Serial interval > generation interval suggests "
                          "most transmission occurs after symptom onset")
        else:
            explanation = "Serial and generation intervals roughly equal"
    else:
        explanation = ("Generation interval is the key parameter for modeling; "
                      "serial interval is easier to measure but may differ")

    return GenerationIntervalReport(
        generation_interval=mean_generation,
        serial_interval=mean_serial,
        difference=difference,
        explanation=explanation,
        notes=notes,
    )


def seir_parameters(
    R0: float,
    latent_period: float,
    infectious_period: float,
) -> Dict[str, float]:
    """Calculate SEIR model rate parameters.

    E: Exposed (infected but not yet infectious)
    σ = 1/latent_period (rate of becoming infectious)
    γ = 1/infectious_period (recovery rate)
    β = R0 × γ (transmission rate)

    Args:
        R0: Basic reproduction number
        latent_period: Mean latent period (days)
        infectious_period: Mean infectious period (days)

    Returns:
        Dict with beta, sigma, gamma parameters.
    """
    if R0 <= 0:
        raise ValueError(f"R0 must be positive, got {R0}")
    if latent_period <= 0:
        raise ValueError(f"latent_period must be positive, got {latent_period}")
    if infectious_period <= 0:
        raise ValueError(f"infectious_period must be positive, got {infectious_period}")

    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    beta = R0 * gamma

    return {
        "beta": beta,
        "sigma": sigma,
        "gamma": gamma,
        "R0": R0,
        "latent_period": latent_period,
        "infectious_period": infectious_period,
        "generation_time": latent_period + infectious_period / 2,  # approximate
    }


# ─── Reference Values ─────────────────────────────────────────────────────────

DISEASE_R0: Dict[str, Tuple[float, float]] = {
    "measles": (12.0, 18.0),
    "pertussis": (12.0, 17.0),
    "chickenpox": (10.0, 12.0),
    "mumps": (10.0, 12.0),
    "rubella": (6.0, 7.0),
    "polio": (5.0, 7.0),
    "smallpox": (5.0, 7.0),
    "covid19_original": (2.5, 3.5),
    "covid19_delta": (5.0, 8.0),
    "covid19_omicron": (8.0, 15.0),
    "influenza_1918": (2.0, 3.0),
    "influenza_seasonal": (1.2, 1.8),
    "ebola": (1.5, 2.5),
    "sars_2003": (2.0, 4.0),
    "mers": (0.4, 0.9),
    "hiv": (2.0, 5.0),
}


def get_disease_R0(disease: str) -> Optional[Tuple[float, float]]:
    """Get reference R0 range for a disease.

    Returns (low, high) tuple or None if unknown.
    """
    return DISEASE_R0.get(disease.lower().replace(" ", "_").replace("-", "_"))


def list_diseases() -> List[str]:
    """List diseases with known R0 values."""
    return sorted(DISEASE_R0.keys())
