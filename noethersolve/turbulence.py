"""Turbulence scaling laws and structure functions.

Verified formulas for turbulent flows:
- Kolmogorov 4/5 law (EXACT for third-order structure function)
- -5/3 energy spectrum (approximate, with intermittency corrections)
- Integral, Taylor, and Kolmogorov microscales
- Reynolds number scaling
- Intermittency corrections (She-Leveque, etc.)

KEY POINT: The 4/5 law is EXACT under isotropy/homogeneity assumptions.
The -5/3 spectrum is an APPROXIMATION. Models often confuse these.
"""

from dataclasses import dataclass
from typing import Optional
import math

# ─── Physical Constants ─────────────────────────────────────────────────────

KOLMOGOROV_CONSTANT_45 = -4/5  # Exact coefficient in 4/5 law


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class Kolmogorov45Report:
    """Report from Kolmogorov 4/5 law calculation."""
    separation: float  # r (separation distance)
    energy_dissipation: float  # ε (dissipation rate)
    third_order_sf: float  # S₃(r) (third-order structure function)
    formula: str
    is_exact: bool  # Always True for 4/5 law
    conditions: list[str]
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Kolmogorov 4/5 Law (Third-Order Structure Function)",
            "=" * 60,
            f"  Separation r = {self.separation:.4g}",
            f"  Dissipation ε = {self.energy_dissipation:.4g}",
            "-" * 60,
            f"  S₃(r) = ⟨(Δu)³⟩ = {self.third_order_sf:.6g}",
            f"  Formula: {self.formula}",
            "-" * 60,
            "  STATUS: This law is EXACT (not approximate!)",
            "  Conditions for exactness:",
        ]
        for cond in self.conditions:
            lines.append(f"    • {cond}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class EnergySpectrumReport:
    """Report from energy spectrum calculation."""
    wavenumber: float  # k
    energy_dissipation: float  # ε
    kolmogorov_constant: float  # C_K (~1.5)
    spectrum: float  # E(k)
    spectral_exponent: float  # -5/3 nominal, with corrections
    intermittency_model: Optional[str]
    intermittency_correction: float
    formula: str
    is_exact: bool  # Always False for spectrum
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Kolmogorov Energy Spectrum",
            "=" * 60,
            f"  Wavenumber k = {self.wavenumber:.4g}",
            f"  Dissipation ε = {self.energy_dissipation:.4g}",
            f"  Kolmogorov constant C_K = {self.kolmogorov_constant:.3f}",
            "-" * 60,
            f"  E(k) = {self.spectrum:.6g}",
            f"  Spectral exponent: {self.spectral_exponent:.4f}",
            f"  Formula: {self.formula}",
        ]
        if self.intermittency_model:
            lines.extend([
                "-" * 60,
                f"  Intermittency model: {self.intermittency_model}",
                f"  Intermittency correction: {self.intermittency_correction:+.4f}",
            ])
        lines.extend([
            "-" * 60,
            "  STATUS: This is APPROXIMATE (dimensional analysis only)",
            "  The -5/3 exponent is NOT exact like the 4/5 law.",
        ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LengthScalesReport:
    """Report from turbulent length scale calculation."""
    integral_scale: float  # L (largest eddies)
    taylor_scale: float  # λ (Taylor microscale)
    kolmogorov_scale: float  # η (smallest eddies)
    reynolds_number: float  # Re_L (large-scale)
    taylor_reynolds: float  # Re_λ
    scale_ratios: dict[str, float]
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Turbulent Length Scales",
            "=" * 60,
            f"  Integral scale L = {self.integral_scale:.4g}",
            f"  Taylor microscale λ = {self.taylor_scale:.4g}",
            f"  Kolmogorov scale η = {self.kolmogorov_scale:.4g}",
            "-" * 60,
            f"  Large-scale Reynolds Re_L = {self.reynolds_number:.2f}",
            f"  Taylor Reynolds Re_λ = {self.taylor_reynolds:.2f}",
            "-" * 60,
            "  Scale ratios:",
            f"    L/η = {self.scale_ratios['L_eta']:.2f} (~ Re^(3/4))",
            f"    L/λ = {self.scale_ratios['L_lambda']:.2f} (~ Re^(1/2))",
            f"    λ/η = {self.scale_ratios['lambda_eta']:.2f} (~ Re^(1/4))",
        ]
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class StructureFunctionReport:
    """Report from structure function scaling."""
    order: int  # p (moment order)
    zeta_p: float  # Scaling exponent
    k41_prediction: float  # Kolmogorov 1941 prediction (p/3)
    intermittency_correction: float
    model: str  # Intermittency model used
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Structure Function Scaling (order p = {self.order})",
            "=" * 60,
            f"  K41 prediction: ζ_p = p/3 = {self.k41_prediction:.4f}",
            f"  Measured/corrected: ζ_p = {self.zeta_p:.4f}",
            f"  Intermittency correction: {self.intermittency_correction:+.4f}",
            "-" * 60,
            f"  Model: {self.model}",
            f"  Formula: {self.formula}",
        ]
        if self.order == 3:
            lines.extend([
                "-" * 60,
                "  NOTE: For p=3, ζ₃ = 1 EXACTLY (4/5 law).",
                "  This is NOT affected by intermittency.",
            ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class IntermittencyReport:
    """Report from intermittency analysis."""
    model: str
    parameters: dict[str, float]
    zeta_2: float  # Second-order exponent
    zeta_4: float  # Fourth-order exponent
    zeta_6: float  # Sixth-order exponent
    flatness_scaling: float  # How flatness grows with Re
    formula: str
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Intermittency Model: {self.model}",
            "=" * 60,
            "  Parameters:",
        ]
        for name, value in self.parameters.items():
            lines.append(f"    {name} = {value:.4f}")
        lines.extend([
            "-" * 60,
            "  Scaling exponents:",
            f"    ζ₂ = {self.zeta_2:.4f} (K41: 0.667)",
            f"    ζ₃ = 1.0000 (exact, 4/5 law)",
            f"    ζ₄ = {self.zeta_4:.4f} (K41: 1.333)",
            f"    ζ₆ = {self.zeta_6:.4f} (K41: 2.000)",
            "-" * 60,
            f"  Flatness scaling: F ~ Re^{{{self.flatness_scaling:.3f}}}",
            f"  Formula: {self.formula}",
        ])
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Kolmogorov 4/5 Law ─────────────────────────────────────────────────────

def kolmogorov_45_law(
    separation: float,
    energy_dissipation: float,
) -> Kolmogorov45Report:
    """Calculate third-order structure function using exact 4/5 law.

    The Kolmogorov 4/5 law states:
    S₃(r) = ⟨(Δu)³⟩ = -(4/5) × ε × r

    This is EXACT under assumptions of:
    - Statistical homogeneity
    - Statistical isotropy
    - High Reynolds number (inertial range)

    Unlike the -5/3 spectrum, this is not dimensional analysis but a
    rigorous derivation from the Navier-Stokes equations.

    Args:
        separation: Scale r in the inertial range
        energy_dissipation: Mean dissipation rate ε

    Returns:
        Kolmogorov45Report with exact result
    """
    if separation <= 0:
        raise ValueError("Separation must be positive")
    if energy_dissipation <= 0:
        raise ValueError("Dissipation rate must be positive")

    # Exact 4/5 law
    third_order_sf = KOLMOGOROV_CONSTANT_45 * energy_dissipation * separation

    conditions = [
        "Statistical homogeneity",
        "Statistical isotropy",
        "Inertial range (η << r << L)",
        "High Reynolds number",
        "Statistically stationary",
    ]

    notes = [
        "This is the ONLY exact scaling law in turbulence",
        "Derived rigorously from Navier-Stokes, not dimensional analysis",
        "Negative sign indicates energy cascade to small scales",
    ]

    return Kolmogorov45Report(
        separation=separation,
        energy_dissipation=energy_dissipation,
        third_order_sf=third_order_sf,
        formula="S₃(r) = -(4/5)εr",
        is_exact=True,
        conditions=conditions,
        notes=notes,
    )


# ─── Energy Spectrum ────────────────────────────────────────────────────────

def energy_spectrum(
    wavenumber: float,
    energy_dissipation: float,
    kolmogorov_constant: float = 1.5,
    intermittency_model: Optional[str] = None,
) -> EnergySpectrumReport:
    """Calculate Kolmogorov energy spectrum.

    The Kolmogorov -5/3 spectrum is:
    E(k) = C_K × ε^(2/3) × k^(-5/3)

    Unlike the 4/5 law, this is APPROXIMATE (dimensional analysis).
    Intermittency corrections modify the exponent slightly.

    Args:
        wavenumber: Wavenumber k in inertial range
        energy_dissipation: Mean dissipation rate ε
        kolmogorov_constant: C_K ≈ 1.5 (experimentally determined)
        intermittency_model: "she_leveque", "k62", or None

    Returns:
        EnergySpectrumReport with approximate result
    """
    if wavenumber <= 0:
        raise ValueError("Wavenumber must be positive")
    if energy_dissipation <= 0:
        raise ValueError("Dissipation rate must be positive")

    # Base K41 exponent
    base_exponent = -5/3

    # Intermittency correction
    correction = 0.0
    if intermittency_model == "she_leveque":
        # She-Leveque model gives small correction to spectrum
        correction = 0.03  # Approximate effect on spectrum exponent
    elif intermittency_model == "k62":
        # Kolmogorov 1962 refined hypothesis
        correction = 0.025
    elif intermittency_model is not None:
        raise ValueError(f"Unknown intermittency model: {intermittency_model}")

    exponent = base_exponent + correction
    spectrum = kolmogorov_constant * (energy_dissipation ** (2/3)) * (wavenumber ** exponent)

    notes = [
        "The -5/3 exponent is from dimensional analysis, NOT exact",
        "The 4/5 law (for S₃) is exact; the spectrum is not",
        f"Kolmogorov constant C_K ≈ {kolmogorov_constant} is empirical",
    ]
    if intermittency_model:
        notes.append(f"Intermittency correction from {intermittency_model} model")

    return EnergySpectrumReport(
        wavenumber=wavenumber,
        energy_dissipation=energy_dissipation,
        kolmogorov_constant=kolmogorov_constant,
        spectrum=spectrum,
        spectral_exponent=exponent,
        intermittency_model=intermittency_model,
        intermittency_correction=correction,
        formula="E(k) = C_K ε^(2/3) k^(-5/3+μ)",
        is_exact=False,
        notes=notes,
    )


# ─── Length Scales ──────────────────────────────────────────────────────────

def length_scales(
    integral_scale: float,
    urms: float,
    kinematic_viscosity: float,
) -> LengthScalesReport:
    """Calculate turbulent length scales from flow parameters.

    Three fundamental scales:
    - Integral scale L: largest eddies, set by geometry
    - Taylor microscale λ: intermediate, λ² = 15νu'²/ε
    - Kolmogorov scale η: smallest eddies, η = (ν³/ε)^(1/4)

    Args:
        integral_scale: Largest eddy size L
        urms: RMS velocity fluctuation u'
        kinematic_viscosity: Kinematic viscosity ν

    Returns:
        LengthScalesReport with all three scales
    """
    if integral_scale <= 0:
        raise ValueError("Integral scale must be positive")
    if urms <= 0:
        raise ValueError("RMS velocity must be positive")
    if kinematic_viscosity <= 0:
        raise ValueError("Viscosity must be positive")

    # Large-scale Reynolds number
    Re_L = urms * integral_scale / kinematic_viscosity

    # Dissipation rate (from energy cascade balance)
    epsilon = urms ** 3 / integral_scale

    # Kolmogorov scale
    eta = (kinematic_viscosity ** 3 / epsilon) ** 0.25

    # Taylor microscale
    lambda_t = (15 * kinematic_viscosity * urms ** 2 / epsilon) ** 0.5

    # Taylor Reynolds number
    Re_lambda = urms * lambda_t / kinematic_viscosity

    # Scale ratios
    ratios = {
        "L_eta": integral_scale / eta,
        "L_lambda": integral_scale / lambda_t,
        "lambda_eta": lambda_t / eta,
    }

    notes = [
        f"Energy dissipation ε ≈ u'³/L = {epsilon:.4g}",
        f"Kolmogorov time scale τ_η = (ν/ε)^(1/2) = {(kinematic_viscosity/epsilon)**0.5:.4g}",
    ]
    if Re_L > 1e4:
        notes.append("High Re: well-developed turbulence")
    elif Re_L < 100:
        notes.append("Low Re: may not have clear inertial range")

    return LengthScalesReport(
        integral_scale=integral_scale,
        taylor_scale=lambda_t,
        kolmogorov_scale=eta,
        reynolds_number=Re_L,
        taylor_reynolds=Re_lambda,
        scale_ratios=ratios,
        formula="η = (ν³/ε)^(1/4), λ = (15νu'²/ε)^(1/2)",
        notes=notes,
    )


# ─── Structure Function Scaling ─────────────────────────────────────────────

def structure_function_exponent(
    order: int,
    model: str = "she_leveque",
) -> StructureFunctionReport:
    """Calculate structure function scaling exponent with intermittency.

    For structure functions S_p(r) = ⟨|Δu|^p⟩ ~ r^ζ_p

    K41 predicts ζ_p = p/3 (linear scaling)
    Intermittency causes deviations, especially for high p.
    For p=3, ζ₃ = 1 EXACTLY (4/5 law, unaffected by intermittency).

    Args:
        order: Moment order p
        model: "k41", "she_leveque", "k62", "beta_model"

    Returns:
        StructureFunctionReport with exponent
    """
    if order < 1:
        raise ValueError("Order must be at least 1")

    k41_prediction = order / 3

    if model == "k41":
        # No intermittency correction
        zeta_p = k41_prediction
        correction = 0.0
        formula = "ζ_p = p/3"
    elif model == "she_leveque":
        # She-Leveque 1994: ζ_p = p/9 + 2(1 - (2/3)^(p/3))
        zeta_p = order/9 + 2 * (1 - (2/3) ** (order/3))
        correction = zeta_p - k41_prediction
        formula = "ζ_p = p/9 + 2(1 - (2/3)^(p/3))"
    elif model == "k62":
        # Kolmogorov 1962: ζ_p = p/3 - μ × p(p-3)/18
        # μ ≈ 0.25 from experiments
        mu = 0.25
        zeta_p = order/3 - mu * order * (order - 3) / 18
        correction = zeta_p - k41_prediction
        formula = "ζ_p = p/3 - μp(p-3)/18, μ≈0.25"
    elif model == "beta_model":
        # β-model: ζ_p = (p + D - 3)/3, D ≈ 2.8
        D = 2.8
        zeta_p = (order + D - 3) / 3
        correction = zeta_p - k41_prediction
        formula = "ζ_p = (p + D - 3)/3, D≈2.8"
    else:
        raise ValueError(f"Unknown model: {model}. Use k41, she_leveque, k62, or beta_model")

    # Special case: p=3 is always exactly 1
    if order == 3:
        zeta_p = 1.0
        correction = 0.0

    notes = []
    if order == 3:
        notes.append("ζ₃ = 1 is EXACT from 4/5 law (no intermittency correction)")
    if order > 6:
        notes.append("High-order moments increasingly affected by rare events")
    if abs(correction) > 0.2:
        notes.append("Large intermittency correction — K41 significantly violated")

    return StructureFunctionReport(
        order=order,
        zeta_p=zeta_p,
        k41_prediction=k41_prediction,
        intermittency_correction=correction,
        model=model,
        formula=formula,
        notes=notes,
    )


# ─── Intermittency Analysis ─────────────────────────────────────────────────

def intermittency_analysis(model: str = "she_leveque") -> IntermittencyReport:
    """Analyze intermittency corrections to turbulence scaling.

    Intermittency refers to the spatial and temporal burstiness of
    turbulent dissipation. It causes deviations from K41 predictions
    for structure function exponents.

    Args:
        model: "she_leveque", "k62", or "beta_model"

    Returns:
        IntermittencyReport with model parameters and predictions
    """
    if model == "she_leveque":
        # She-Leveque 1994 model
        parameters = {
            "β": 2/3,  # Co-dimension of dissipative structures
            "Δ": 2/3,  # Scaling of most intense structures
        }
        # ζ_p = p/9 + 2(1 - (2/3)^(p/3))
        zeta_2 = 2/9 + 2 * (1 - (2/3) ** (2/3))
        zeta_4 = 4/9 + 2 * (1 - (2/3) ** (4/3))
        zeta_6 = 6/9 + 2 * (1 - (2/3) ** (6/3))
        flatness_scaling = 0.12  # Flatness ~ Re^0.12
        formula = "ζ_p = p/9 + 2(1 - (2/3)^(p/3))"
        notes = [
            "Based on log-Poisson model of energy cascade",
            "Assumes filamentary dissipative structures",
            "Best agreement with experiments for moderate p",
        ]
    elif model == "k62":
        # Kolmogorov 1962 refined hypothesis
        mu = 0.25
        parameters = {"μ": mu}
        # ζ_p = p/3 - μ × p(p-3)/18
        zeta_2 = 2/3 - mu * 2 * (-1) / 18  # = 2/3 + μ/9
        zeta_4 = 4/3 - mu * 4 * 1 / 18
        zeta_6 = 6/3 - mu * 6 * 3 / 18
        flatness_scaling = 3 * mu / 2
        formula = "ζ_p = p/3 - μp(p-3)/18"
        notes = [
            "Assumes log-normal distribution of dissipation",
            "μ ≈ 0.25 from experiments",
            "Violates exact ζ₃ = 1 for μ ≠ 0 (physically inconsistent)",
        ]
    elif model == "beta_model":
        D = 2.8
        parameters = {"D": D}
        # ζ_p = (p + D - 3)/3
        zeta_2 = (2 + D - 3) / 3
        zeta_4 = (4 + D - 3) / 3
        zeta_6 = (6 + D - 3) / 3
        flatness_scaling = (3 - D) / 2
        formula = "ζ_p = (p + D - 3)/3"
        notes = [
            "Fractal model with dimension D ≈ 2.8",
            "Simple but violates ζ₃ = 1 exactly",
            "Useful for qualitative understanding",
        ]
    else:
        raise ValueError(f"Unknown model: {model}")

    return IntermittencyReport(
        model=model,
        parameters=parameters,
        zeta_2=zeta_2,
        zeta_4=zeta_4,
        zeta_6=zeta_6,
        flatness_scaling=flatness_scaling,
        formula=formula,
        notes=notes,
    )


# ─── Utility Functions ──────────────────────────────────────────────────────

def is_in_inertial_range(
    scale: float,
    kolmogorov_scale: float,
    integral_scale: float,
) -> bool:
    """Check if a scale is in the inertial range.

    The inertial range is where η << r << L.
    Usually defined as 60η < r < 0.1L.

    Args:
        scale: Scale to check
        kolmogorov_scale: Kolmogorov scale η
        integral_scale: Integral scale L

    Returns:
        True if scale is in inertial range
    """
    return 60 * kolmogorov_scale < scale < 0.1 * integral_scale


def inertial_range_extent(reynolds_number: float) -> float:
    """Calculate the extent of the inertial range in decades.

    The inertial range spans from η to L, with L/η ~ Re^(3/4).
    The usable inertial range is typically ~2 decades for Re ~ 10^4.

    Args:
        reynolds_number: Large-scale Reynolds number

    Returns:
        Extent in decades (log10 scale)
    """
    if reynolds_number <= 0:
        raise ValueError("Reynolds number must be positive")

    # L/η ~ Re^(3/4)
    scale_ratio = reynolds_number ** 0.75
    # Usable range is about 1/60 to 1/10 of full range
    usable_ratio = scale_ratio / 600
    if usable_ratio <= 1:
        return 0.0
    return math.log10(usable_ratio)
