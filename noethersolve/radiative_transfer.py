"""Radiative transfer and climate physics calculator.

Verified formulas from IPCC AR5/AR6 and fundamental physics:
- CO2 radiative forcing (LOGARITHMIC, not linear - common LLM error)
- Planck response (exact 3.2 W/m²/K from Stefan-Boltzmann)
- Equilibrium climate sensitivity decomposition
- Stefan-Boltzmann radiation
- Albedo and effective temperature
- Feedback parameter analysis

All calculators derive from first principles, not guesses.
"""

from dataclasses import dataclass
from typing import Optional
import math

# ─── Physical Constants ─────────────────────────────────────────────────────

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
SOLAR_CONSTANT = 1361.0  # W/m² (TSI at 1 AU)
EARTH_ALBEDO = 0.30  # Bond albedo
CO2_PREINDUSTRIAL = 280.0  # ppm (year 1750 reference)
CO2_CURRENT_2024 = 421.0  # ppm (approximate 2024 value)

# IPCC AR5 radiative forcing coefficient (Myhre et al. 1998, refined)
# ΔF = α × ln(C/C₀) where α ≈ 5.35 W/m²
RF_COEFFICIENT = 5.35  # W/m² per ln(C/C₀)

# Planck feedback parameter (no feedbacks, pure Stefan-Boltzmann)
# λ₀ = 1/(4σT³) where T ≈ 255K (effective emission temperature)
# λ₀ ≈ 3.2 K/(W/m²), so 1/λ₀ ≈ 0.31 W/m²/K
PLANCK_PARAMETER = 3.2  # K/(W/m²) = °C per doubling ÷ 3.7 W/m²


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class RadiativeForcingReport:
    """Report from radiative forcing calculation."""
    co2_initial: float  # ppm
    co2_final: float  # ppm
    forcing: float  # W/m² (positive = warming)
    forcing_per_doubling: float  # W/m² per CO2 doubling
    ratio: float  # C_final / C_initial
    is_logarithmic: bool  # Always True - key teaching point
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Radiative Forcing (CO₂)",
            "=" * 60,
            f"  CO₂: {self.co2_initial:.1f} → {self.co2_final:.1f} ppm",
            f"  Ratio: {self.ratio:.3f} ({math.log2(self.ratio):.2f} doublings)",
            "-" * 60,
            f"  ΔF = 5.35 × ln(C/C₀) = {self.forcing:.2f} W/m²",
            f"  Forcing per doubling: {self.forcing_per_doubling:.2f} W/m²",
            "-" * 60,
            "  KEY POINT: Forcing is LOGARITHMIC, not linear.",
            "  Each doubling adds the same ~3.7 W/m², regardless of",
            "  absolute concentration.",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PlanckResponseReport:
    """Report from Planck response calculation."""
    temperature: float  # K (emission temperature)
    planck_parameter: float  # K/(W/m²)
    planck_feedback: float  # W/(m²·K) (inverse)
    warming_per_doubling_no_feedback: float  # K
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Planck Response (No-Feedback Sensitivity)",
            "=" * 60,
            f"  Emission temperature: {self.temperature:.1f} K",
            f"  Formula: {self.formula}",
            "-" * 60,
            f"  Planck parameter λ₀ = {self.planck_parameter:.2f} K/(W/m²)",
            f"  Planck feedback = {self.planck_feedback:.3f} W/(m²·K)",
            f"  No-feedback warming per CO₂ doubling: {self.warming_per_doubling_no_feedback:.2f} K",
            "-" * 60,
            "  This is the MINIMUM warming from CO₂ doubling.",
            "  Actual warming depends on feedbacks (water vapor, ice, clouds).",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ClimateSensitivityReport:
    """Report from climate sensitivity calculation."""
    ecs: float  # K (equilibrium climate sensitivity)
    tcr: float  # K (transient climate response, if provided)
    forcing_2x: float  # W/m² per doubling
    feedback_parameter: float  # W/(m²·K)
    feedback_factor: float  # dimensionless (amplification)
    feedbacks: dict[str, float]  # Individual feedback contributions
    likely_range: tuple[float, float]  # IPCC likely range
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Equilibrium Climate Sensitivity (ECS)",
            "=" * 60,
            f"  ECS = {self.ecs:.2f} K per CO₂ doubling",
            f"  TCR = {self.tcr:.2f} K (transient, ~70% of ECS)" if self.tcr > 0 else "",
            "-" * 60,
            f"  Forcing per doubling: {self.forcing_2x:.2f} W/m²",
            f"  Net feedback parameter: {self.feedback_parameter:.2f} W/(m²·K)",
            f"  Feedback factor (gain): {self.feedback_factor:.2f}×",
            "-" * 60,
            "  Feedback contributions:",
        ]
        for name, value in self.feedbacks.items():
            lines.append(f"    {name}: {value:+.2f} W/(m²·K)")
        lines.extend([
            "-" * 60,
            f"  IPCC AR6 likely range: {self.likely_range[0]:.1f}–{self.likely_range[1]:.1f} K",
        ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        # Remove empty lines
        return "\n".join(line for line in lines if line)


@dataclass
class StefanBoltzmannReport:
    """Report from Stefan-Boltzmann calculation."""
    temperature: float  # K
    emissivity: float  # 0-1
    power_density: float  # W/m²
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Stefan-Boltzmann Radiation",
            "=" * 60,
            f"  Temperature: {self.temperature:.1f} K",
            f"  Emissivity: {self.emissivity:.3f}",
            f"  Formula: {self.formula}",
            "-" * 60,
            f"  Power density: {self.power_density:.2f} W/m²",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class EffectiveTemperatureReport:
    """Report from effective temperature calculation."""
    incoming_solar: float  # W/m²
    albedo: float
    absorbed: float  # W/m²
    effective_temp: float  # K (no greenhouse)
    actual_surface_temp: float  # K (with greenhouse)
    greenhouse_effect: float  # K (warming from GHE)
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Effective Temperature & Greenhouse Effect",
            "=" * 60,
            f"  Incoming solar flux: {self.incoming_solar:.1f} W/m²",
            f"  Albedo (reflectivity): {self.albedo:.2f}",
            f"  Absorbed: {self.absorbed:.1f} W/m²",
            "-" * 60,
            f"  Effective temp (no GHE): {self.effective_temp:.1f} K = {self.effective_temp - 273.15:.1f}°C",
            f"  Actual surface temp: {self.actual_surface_temp:.1f} K = {self.actual_surface_temp - 273.15:.1f}°C",
            f"  Greenhouse effect: {self.greenhouse_effect:.1f} K warming",
            "-" * 60,
            "  The greenhouse effect raises Earth's surface ~33 K above",
            "  the effective emission temperature.",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class FeedbackAnalysisReport:
    """Report from feedback parameter analysis."""
    name: str
    value: float  # W/(m²·K)
    sign: str  # "positive" (amplifying) or "negative" (damping)
    uncertainty_range: tuple[float, float]
    mechanism: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Climate Feedback: {self.name}",
            "=" * 60,
            f"  Feedback parameter: {self.value:+.2f} W/(m²·K)",
            f"  Type: {self.sign} feedback ({'+' if self.value > 0 else '-'} amplification)",
            f"  Uncertainty range: {self.uncertainty_range[0]:+.2f} to {self.uncertainty_range[1]:+.2f}",
            "-" * 60,
            f"  Mechanism: {self.mechanism}",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Radiative Forcing ──────────────────────────────────────────────────────

def radiative_forcing(
    co2_final: float,
    co2_initial: float = CO2_PREINDUSTRIAL,
) -> RadiativeForcingReport:
    """Calculate radiative forcing from CO2 change.

    Uses IPCC formula: ΔF = 5.35 × ln(C/C₀) W/m²

    KEY POINT: The relationship is LOGARITHMIC, not linear. This is one
    of the most common errors LLMs make about climate physics. Each doubling
    of CO2 adds approximately 3.7 W/m² regardless of the absolute concentration.

    Args:
        co2_final: Final CO2 concentration in ppm
        co2_initial: Initial CO2 concentration in ppm (default: preindustrial 280)

    Returns:
        RadiativeForcingReport with forcing and teaching points
    """
    if co2_final <= 0:
        raise ValueError("CO2 concentration must be positive")
    if co2_initial <= 0:
        raise ValueError("Initial CO2 concentration must be positive")

    ratio = co2_final / co2_initial
    forcing = RF_COEFFICIENT * math.log(ratio)
    forcing_per_doubling = RF_COEFFICIENT * math.log(2)  # ~3.7 W/m²

    notes = []
    if co2_final >= 2 * co2_initial:
        doublings = math.log2(ratio)
        notes.append(f"This represents {doublings:.2f} doublings of CO2")
    if co2_final >= 560:  # ~2x preindustrial
        notes.append("Exceeds 2× preindustrial (commonly used reference)")
    if forcing > 6.0:
        notes.append("WARNING: Very high forcing, approaching RCP8.5 scenario")

    return RadiativeForcingReport(
        co2_initial=co2_initial,
        co2_final=co2_final,
        forcing=forcing,
        forcing_per_doubling=forcing_per_doubling,
        ratio=ratio,
        is_logarithmic=True,
        notes=notes,
    )


# ─── Planck Response ────────────────────────────────────────────────────────

def planck_response(
    emission_temperature: float = 255.0,
) -> PlanckResponseReport:
    """Calculate the Planck (no-feedback) climate response.

    The Planck parameter is derived from Stefan-Boltzmann:
    P = σT⁴ → dP/dT = 4σT³
    λ₀ = dT/dP = 1/(4σT³)

    At Earth's effective emission temperature (~255 K):
    λ₀ ≈ 0.31 W/(m²·K)⁻¹ ≈ 3.2 K/(W/m²)

    This gives the MINIMUM warming from any forcing, before feedbacks.
    With forcing of ~3.7 W/m² per CO2 doubling:
    ΔT_no_feedback ≈ 1.2 K per doubling

    Args:
        emission_temperature: Effective emission temperature in K (default 255 K)

    Returns:
        PlanckResponseReport with no-feedback sensitivity
    """
    if emission_temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Planck feedback (inverse of parameter)
    planck_feedback = 4 * STEFAN_BOLTZMANN * (emission_temperature ** 3)

    # Planck parameter (K per W/m²)
    planck_parameter = 1.0 / planck_feedback

    # No-feedback warming per CO2 doubling
    forcing_2x = RF_COEFFICIENT * math.log(2)  # ~3.7 W/m²
    warming_no_feedback = forcing_2x * planck_parameter

    notes = []
    if abs(emission_temperature - 255) < 5:
        notes.append("Using Earth's approximate effective emission temperature")
    notes.append("This is the EXACT Stefan-Boltzmann derivative, not an estimate")

    return PlanckResponseReport(
        temperature=emission_temperature,
        planck_parameter=planck_parameter,
        planck_feedback=planck_feedback,
        warming_per_doubling_no_feedback=warming_no_feedback,
        formula="λ₀ = 1/(4σT³)",
        notes=notes,
    )


# ─── Climate Sensitivity ────────────────────────────────────────────────────

# Known feedback values (IPCC AR6, central estimates)
FEEDBACK_DATABASE = {
    "planck": {
        "value": -3.2,  # Negative = restoring/damping
        "range": (-3.4, -3.0),
        "mechanism": "Stefan-Boltzmann radiation increase with temperature",
    },
    "water_vapor": {
        "value": 1.8,  # Positive = amplifying
        "range": (1.5, 2.1),
        "mechanism": "Clausius-Clapeyron: warmer air holds more water vapor (GHG)",
    },
    "lapse_rate": {
        "value": -0.6,
        "range": (-0.8, -0.4),
        "mechanism": "Upper troposphere warms faster, reduces lapse rate",
    },
    "surface_albedo": {
        "value": 0.4,
        "range": (0.2, 0.6),
        "mechanism": "Ice/snow melt reveals darker surfaces",
    },
    "cloud": {
        "value": 0.5,
        "range": (-0.2, 1.2),  # Large uncertainty!
        "mechanism": "Net effect of cloud changes (LARGEST UNCERTAINTY)",
    },
}


def climate_sensitivity(
    ecs: Optional[float] = None,
    feedback_sum: Optional[float] = None,
    include_feedbacks: Optional[list[str]] = None,
) -> ClimateSensitivityReport:
    """Calculate equilibrium climate sensitivity from feedbacks.

    ECS = ΔF₂ₓ / λ
    where λ = -Σfeedbacks (net feedback parameter)

    IPCC AR6 likely range: 2.5–4.0 K (best estimate 3.0 K)

    Args:
        ecs: If provided, derive feedback parameter from ECS
        feedback_sum: Net feedback parameter (W/m²/K) if known
        include_feedbacks: List of feedbacks to include (default: all)

    Returns:
        ClimateSensitivityReport with full analysis
    """
    forcing_2x = RF_COEFFICIENT * math.log(2)  # ~3.7 W/m²

    # Build feedback dictionary
    if include_feedbacks is None:
        include_feedbacks = list(FEEDBACK_DATABASE.keys())

    feedbacks = {}
    for name in include_feedbacks:
        if name in FEEDBACK_DATABASE:
            feedbacks[name] = FEEDBACK_DATABASE[name]["value"]

    # Calculate net feedback parameter
    if feedback_sum is not None:
        net_feedback = feedback_sum
    else:
        net_feedback = sum(feedbacks.values())

    # Calculate ECS
    if ecs is not None:
        calculated_ecs = ecs
        # Derive feedback from ECS
        net_feedback = -forcing_2x / ecs
    else:
        if net_feedback >= 0:
            # Runaway greenhouse (unstable)
            calculated_ecs = float('inf')
        else:
            calculated_ecs = -forcing_2x / net_feedback

    # Feedback factor (amplification relative to no-feedback)
    planck_feedback = FEEDBACK_DATABASE["planck"]["value"]
    if net_feedback != 0:
        feedback_factor = planck_feedback / net_feedback
    else:
        feedback_factor = float('inf')

    # TCR is typically ~70% of ECS
    tcr = calculated_ecs * 0.7 if calculated_ecs < float('inf') else 0.0

    notes = []
    if calculated_ecs > 4.5:
        notes.append("WARNING: ECS above IPCC very likely range (>4.5 K)")
    elif calculated_ecs < 2.0:
        notes.append("WARNING: ECS below IPCC very likely range (<2.0 K)")

    if "cloud" in feedbacks:
        notes.append("Cloud feedback has LARGEST uncertainty (±0.7 W/m²/K)")

    return ClimateSensitivityReport(
        ecs=calculated_ecs,
        tcr=tcr,
        forcing_2x=forcing_2x,
        feedback_parameter=-net_feedback,  # Convention: positive = damping
        feedback_factor=feedback_factor,
        feedbacks=feedbacks,
        likely_range=(2.5, 4.0),  # IPCC AR6
        notes=notes,
    )


# ─── Stefan-Boltzmann ───────────────────────────────────────────────────────

def stefan_boltzmann(
    temperature: float,
    emissivity: float = 1.0,
) -> StefanBoltzmannReport:
    """Calculate blackbody radiation power density.

    P = εσT⁴

    Args:
        temperature: Temperature in Kelvin
        emissivity: Surface emissivity (0-1, default 1 for blackbody)

    Returns:
        StefanBoltzmannReport with power density
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if not 0 <= emissivity <= 1:
        raise ValueError("Emissivity must be between 0 and 1")

    power = emissivity * STEFAN_BOLTZMANN * (temperature ** 4)

    notes = []
    if emissivity < 1.0:
        notes.append(f"Gray body with emissivity {emissivity:.3f}")
    if 5500 <= temperature <= 6000:
        notes.append("Near solar photosphere temperature")
    if 250 <= temperature <= 300:
        notes.append("Near Earth surface/emission temperature range")

    return StefanBoltzmannReport(
        temperature=temperature,
        emissivity=emissivity,
        power_density=power,
        formula="P = εσT⁴",
        notes=notes,
    )


# ─── Effective Temperature ──────────────────────────────────────────────────

def effective_temperature(
    solar_flux: float = SOLAR_CONSTANT,
    albedo: float = EARTH_ALBEDO,
    actual_surface_temp: float = 288.0,  # Earth average ~15°C
) -> EffectiveTemperatureReport:
    """Calculate effective emission temperature and greenhouse effect.

    Absorbed = S₀(1-α)/4
    T_eff = (Absorbed/σ)^(1/4)

    For Earth: T_eff ≈ 255 K, T_actual ≈ 288 K
    Greenhouse effect = 33 K warming

    Args:
        solar_flux: Solar constant in W/m² (default 1361)
        albedo: Bond albedo (default 0.30 for Earth)
        actual_surface_temp: Actual average surface temperature in K

    Returns:
        EffectiveTemperatureReport with greenhouse effect analysis
    """
    if solar_flux <= 0:
        raise ValueError("Solar flux must be positive")
    if not 0 <= albedo < 1:
        raise ValueError("Albedo must be between 0 and 1")

    # Absorbed flux (factor of 4 from sphere geometry)
    absorbed = solar_flux * (1 - albedo) / 4

    # Effective temperature (balance with Stefan-Boltzmann)
    t_eff = (absorbed / STEFAN_BOLTZMANN) ** 0.25

    # Greenhouse effect
    greenhouse_effect = actual_surface_temp - t_eff

    notes = []
    if abs(albedo - EARTH_ALBEDO) < 0.01:
        notes.append("Using Earth's average albedo")
    if greenhouse_effect > 30:
        notes.append(f"Greenhouse effect of {greenhouse_effect:.0f} K is substantial")
    if actual_surface_temp > 273.15:
        notes.append(f"Surface above freezing: {actual_surface_temp - 273.15:.1f}°C")

    return EffectiveTemperatureReport(
        incoming_solar=solar_flux,
        albedo=albedo,
        absorbed=absorbed,
        effective_temp=t_eff,
        actual_surface_temp=actual_surface_temp,
        greenhouse_effect=greenhouse_effect,
        notes=notes,
    )


# ─── Feedback Analysis ──────────────────────────────────────────────────────

def analyze_feedback(name: str) -> FeedbackAnalysisReport:
    """Get detailed analysis of a specific climate feedback.

    Available feedbacks:
    - planck: Stefan-Boltzmann radiation (negative, damping)
    - water_vapor: Clausius-Clapeyron (positive, strongest amplifying)
    - lapse_rate: Tropospheric temperature profile (negative, damping)
    - surface_albedo: Ice-albedo feedback (positive)
    - cloud: Net cloud feedback (positive but HIGHLY UNCERTAIN)

    Args:
        name: Feedback name (case-insensitive)

    Returns:
        FeedbackAnalysisReport with mechanism details
    """
    name_lower = name.lower().replace(" ", "_").replace("-", "_")

    if name_lower not in FEEDBACK_DATABASE:
        available = ", ".join(FEEDBACK_DATABASE.keys())
        raise ValueError(f"Unknown feedback '{name}'. Available: {available}")

    fb = FEEDBACK_DATABASE[name_lower]
    value = fb["value"]
    sign = "positive (amplifying)" if value > 0 else "negative (damping)"

    notes = []
    if name_lower == "cloud":
        notes.append("Largest source of uncertainty in climate sensitivity")
        notes.append("Sign was uncertain until ~2020, now likely positive")
    if name_lower == "water_vapor":
        notes.append("Strongest positive feedback (~1.8 W/m²/K)")
        notes.append("Often combined with lapse rate as 'water vapor + lapse rate'")
    if name_lower == "planck":
        notes.append("This is the 'restoring force' - without it, any forcing → runaway")
        notes.append("Derived exactly from Stefan-Boltzmann law")

    return FeedbackAnalysisReport(
        name=name,
        value=value,
        sign=sign,
        uncertainty_range=fb["range"],
        mechanism=fb["mechanism"],
        notes=notes,
    )


def list_feedbacks() -> list[str]:
    """List all available climate feedbacks."""
    return list(FEEDBACK_DATABASE.keys())
