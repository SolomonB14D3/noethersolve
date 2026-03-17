"""
Seismic wave velocity and elastic moduli calculator.

Exact formulas for P-wave and S-wave velocities, Poisson's ratio bounds,
elastic moduli conversions, and wave behavior constraints.

CRITICAL FORMULAS (models often get these wrong):
- Vp = sqrt((K + 4G/3) / rho)  -- P-wave (compressional)
- Vs = sqrt(G / rho)           -- S-wave (shear)
- Vp/Vs = sqrt((K/G + 4/3))    -- ratio depends on K/G, NOT just material
- nu = (Vp^2 - 2*Vs^2) / (2*(Vp^2 - Vs^2))  -- Poisson from velocities

COMMON LLM ERRORS:
1. Forgetting the 4/3 factor in Vp formula
2. Using Vp/Vs = sqrt(3) as universal (only true for nu = 0.25)
3. Confusing K (bulk modulus) with E (Young's modulus)
4. Not enforcing thermodynamic stability bounds on Poisson ratio

Physical bounds on Poisson's ratio:
- Thermodynamic stability: -1 < nu < 0.5
- Most rocks: 0.05 < nu < 0.45
- Liquids/soft sediments: nu -> 0.5 (Vs -> 0)
- Auxetic materials: nu < 0 (rare in geology)
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass
class SeismicVelocityReport:
    """Report for seismic velocity calculations."""

    Vp: float  # P-wave velocity (m/s or km/s depending on input)
    Vs: float  # S-wave velocity
    Vp_Vs_ratio: float
    poisson_ratio: float
    bulk_modulus_K: float  # Pa or GPa depending on input
    shear_modulus_G: float
    youngs_modulus_E: float
    density: float

    # Derived quantities
    acoustic_impedance_P: float  # rho * Vp
    acoustic_impedance_S: float  # rho * Vs

    # Physical validity
    is_thermodynamically_stable: bool
    is_typical_rock: bool
    material_type: str  # "solid", "liquid", "auxetic", "unstable"

    def __str__(self) -> str:
        lines = [
            "Seismic Velocity Report",
            "=" * 50,
            f"P-wave velocity Vp: {self.Vp:.4f}",
            f"S-wave velocity Vs: {self.Vs:.4f}",
            f"Vp/Vs ratio: {self.Vp_Vs_ratio:.4f}",
            "",
            "Elastic Moduli:",
            f"  Bulk modulus K: {self.bulk_modulus_K:.4e}",
            f"  Shear modulus G: {self.shear_modulus_G:.4e}",
            f"  Young's modulus E: {self.youngs_modulus_E:.4e}",
            f"  Poisson's ratio nu: {self.poisson_ratio:.4f}",
            "",
            "Acoustic Impedances:",
            f"  Z_P (rho*Vp): {self.acoustic_impedance_P:.4e}",
            f"  Z_S (rho*Vs): {self.acoustic_impedance_S:.4e}",
            "",
            f"Material type: {self.material_type}",
            f"Thermodynamically stable: {self.is_thermodynamically_stable}",
            f"Typical rock range: {self.is_typical_rock}",
        ]
        return "\n".join(lines)


@dataclass
class PoissonRatioReport:
    """Report for Poisson ratio analysis."""

    poisson_ratio: float
    Vp_Vs_ratio: float
    K_over_G: float

    is_thermodynamically_stable: bool
    is_typical_rock: bool
    material_type: str

    # Limits
    liquid_limit: float = 0.5
    lower_bound: float = -1.0

    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def __str__(self) -> str:
        lines = [
            "Poisson Ratio Analysis",
            "=" * 50,
            f"Poisson's ratio nu: {self.poisson_ratio:.6f}",
            f"Vp/Vs ratio: {self.Vp_Vs_ratio:.4f}",
            f"K/G ratio: {self.K_over_G:.4f}",
            "",
            f"Material type: {self.material_type}",
            f"Thermodynamically stable: {self.is_thermodynamically_stable}",
            f"Typical rock range (0.05-0.45): {self.is_typical_rock}",
        ]
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


@dataclass
class ElasticModuliReport:
    """Report for elastic moduli conversions."""

    bulk_modulus_K: float
    shear_modulus_G: float
    youngs_modulus_E: float
    poisson_ratio_nu: float
    lame_lambda: float
    p_wave_modulus_M: float  # M = K + 4G/3 = lambda + 2*mu

    # Input used
    input_params: str

    def __str__(self) -> str:
        lines = [
            "Elastic Moduli Conversion",
            "=" * 50,
            f"Input: {self.input_params}",
            "",
            "All Moduli:",
            f"  Bulk modulus K: {self.bulk_modulus_K:.6e}",
            f"  Shear modulus G (mu): {self.shear_modulus_G:.6e}",
            f"  Young's modulus E: {self.youngs_modulus_E:.6e}",
            f"  Lame's first parameter lambda: {self.lame_lambda:.6e}",
            f"  P-wave modulus M: {self.p_wave_modulus_M:.6e}",
            f"  Poisson's ratio nu: {self.poisson_ratio_nu:.6f}",
        ]
        return "\n".join(lines)


@dataclass
class ReflectionReport:
    """Report for seismic reflection coefficients."""

    R_P: float  # P-wave reflection coefficient (normal incidence)
    T_P: float  # P-wave transmission coefficient
    R_S: float  # S-wave reflection coefficient
    T_S: float  # S-wave transmission coefficient

    Z1_P: float  # Acoustic impedance layer 1
    Z2_P: float  # Acoustic impedance layer 2

    energy_conserved: bool

    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def __str__(self) -> str:
        lines = [
            "Reflection Coefficient Report (Normal Incidence)",
            "=" * 50,
            f"P-wave reflection R: {self.R_P:.6f}",
            f"P-wave transmission T: {self.T_P:.6f}",
            f"S-wave reflection R: {self.R_S:.6f}",
            f"S-wave transmission T: {self.T_S:.6f}",
            "",
            f"Impedance Z1: {self.Z1_P:.4e}",
            f"Impedance Z2: {self.Z2_P:.4e}",
            f"Impedance contrast: {self.Z2_P/self.Z1_P:.4f}",
            "",
            f"Energy conservation: R^2 + T^2 = {self.R_P**2 + self.T_P**2:.6f}",
        ]
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"Note: {note}")
        return "\n".join(lines)


def calc_seismic_velocity(
    K: float,
    G: float,
    rho: float
) -> SeismicVelocityReport:
    """
    Calculate P-wave and S-wave velocities from elastic moduli.

    CRITICAL FORMULAS:
    - Vp = sqrt((K + 4G/3) / rho)   [P-wave, compressional]
    - Vs = sqrt(G / rho)            [S-wave, shear]

    Args:
        K: Bulk modulus (Pa or GPa - be consistent)
        G: Shear modulus (Pa or GPa - same units as K)
        rho: Density (kg/m^3 or g/cm^3 - consistent with moduli)

    Returns:
        SeismicVelocityReport with velocities and derived quantities

    Example:
        # Granite: K=50 GPa, G=30 GPa, rho=2700 kg/m^3
        # Using GPa and kg/m^3: multiply GPa by 1e9
        result = calc_seismic_velocity(50e9, 30e9, 2700)
        # Vp ≈ 5900 m/s, Vs ≈ 3300 m/s
    """
    if K <= 0:
        raise ValueError(f"Bulk modulus K must be positive, got {K}")
    if G < 0:
        raise ValueError(f"Shear modulus G cannot be negative, got {G}")
    if rho <= 0:
        raise ValueError(f"Density must be positive, got {rho}")

    # P-wave modulus M = K + 4G/3
    M = K + (4.0 * G / 3.0)

    # Velocities
    Vp = math.sqrt(M / rho)
    Vs = math.sqrt(G / rho) if G > 0 else 0.0

    # Vp/Vs ratio
    Vp_Vs = Vp / Vs if Vs > 0 else float('inf')

    # Poisson's ratio from K and G
    # nu = (3K - 2G) / (6K + 2G)
    denominator = 6*K + 2*G
    if abs(denominator) < 1e-30:
        nu = 0.5  # Liquid limit
    else:
        nu = (3*K - 2*G) / denominator

    # Young's modulus E = 9KG / (3K + G)
    denom_E = 3*K + G
    if abs(denom_E) < 1e-30:
        E = 0.0
    else:
        E = 9*K*G / denom_E

    # Acoustic impedances
    Z_P = rho * Vp
    Z_S = rho * Vs

    # Physical validity checks
    is_stable = -1.0 < nu < 0.5
    is_typical = 0.05 < nu < 0.45

    # Material type classification
    if nu >= 0.5:
        material_type = "liquid (nu = 0.5, no shear)"
    elif nu < -1.0:
        material_type = "unstable (nu < -1)"
    elif nu < 0:
        material_type = "auxetic (negative Poisson)"
    elif nu > 0.45:
        material_type = "soft solid / near-liquid"
    else:
        material_type = "solid"

    return SeismicVelocityReport(
        Vp=Vp,
        Vs=Vs,
        Vp_Vs_ratio=Vp_Vs,
        poisson_ratio=nu,
        bulk_modulus_K=K,
        shear_modulus_G=G,
        youngs_modulus_E=E,
        density=rho,
        acoustic_impedance_P=Z_P,
        acoustic_impedance_S=Z_S,
        is_thermodynamically_stable=is_stable,
        is_typical_rock=is_typical,
        material_type=material_type
    )


def calc_velocity_from_poisson(
    E: float,
    nu: float,
    rho: float
) -> SeismicVelocityReport:
    """
    Calculate seismic velocities from Young's modulus and Poisson's ratio.

    Converts (E, nu) to (K, G) then computes velocities.

    Conversion formulas:
    - K = E / (3*(1 - 2*nu))
    - G = E / (2*(1 + nu))

    Args:
        E: Young's modulus (Pa or GPa)
        nu: Poisson's ratio (dimensionless, must be -1 < nu < 0.5)
        rho: Density

    Returns:
        SeismicVelocityReport

    Example:
        # Steel: E=200 GPa, nu=0.3, rho=7800 kg/m^3
        result = calc_velocity_from_poisson(200e9, 0.3, 7800)
    """
    if nu <= -1.0 or nu >= 0.5:
        raise ValueError(f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}")
    if E <= 0:
        raise ValueError(f"Young's modulus must be positive, got {E}")
    if rho <= 0:
        raise ValueError(f"Density must be positive, got {rho}")

    # Convert to K and G
    K = E / (3.0 * (1.0 - 2.0*nu))
    G = E / (2.0 * (1.0 + nu))

    return calc_seismic_velocity(K, G, rho)


def poisson_from_velocities(Vp: float, Vs: float) -> PoissonRatioReport:
    """
    Calculate Poisson's ratio from measured P and S wave velocities.

    CRITICAL FORMULA:
    nu = (Vp^2 - 2*Vs^2) / (2*(Vp^2 - Vs^2))

    This is exact - no approximations.

    Args:
        Vp: P-wave velocity
        Vs: S-wave velocity (must be less than Vp)

    Returns:
        PoissonRatioReport with nu and validity checks

    Common Vp/Vs ratios:
    - sqrt(2) ≈ 1.414: nu = 0.0 (unusual, some single crystals)
    - sqrt(3) ≈ 1.732: nu = 0.25 (Poisson solid, common reference)
    - 2.0: nu = 0.333
    - >>2: nu -> 0.5 (liquids, Vs -> 0)
    """
    if Vp <= 0:
        raise ValueError(f"Vp must be positive, got {Vp}")
    if Vs < 0:
        raise ValueError(f"Vs cannot be negative, got {Vs}")
    if Vs >= Vp:
        raise ValueError(f"Vs must be less than Vp (got Vs={Vs}, Vp={Vp})")

    Vp2 = Vp * Vp
    Vs2 = Vs * Vs

    # Handle Vs = 0 (liquid) case
    if Vs == 0:
        nu = 0.5
        Vp_Vs = float('inf')
        K_over_G = float('inf')
    else:
        Vp_Vs = Vp / Vs

        # nu = (Vp^2 - 2*Vs^2) / (2*(Vp^2 - Vs^2))
        numerator = Vp2 - 2*Vs2
        denominator = 2*(Vp2 - Vs2)
        nu = numerator / denominator

        # K/G from Vp/Vs: Vp^2/Vs^2 = (K/G + 4/3) => K/G = Vp^2/Vs^2 - 4/3
        K_over_G = (Vp2/Vs2) - (4.0/3.0)

    # Physical validity
    is_stable = -1.0 < nu < 0.5
    is_typical = 0.05 < nu < 0.45

    # Material classification
    notes = []
    if nu >= 0.5:
        material_type = "liquid"
        notes.append("Vs = 0 indicates fluid (no shear resistance)")
    elif nu < -1.0:
        material_type = "unstable"
        notes.append("INVALID: Thermodynamically unstable")
    elif nu < 0:
        material_type = "auxetic"
        notes.append("Auxetic material - expands laterally under tension")
    elif nu > 0.45:
        material_type = "soft solid"
        notes.append("Near liquid - very low shear modulus")
    else:
        material_type = "solid"

    # Add common reference points
    if abs(Vp_Vs - math.sqrt(3)) < 0.01:
        notes.append("Vp/Vs ≈ √3: Classic Poisson solid (nu = 0.25)")
    if abs(Vp_Vs - math.sqrt(2)) < 0.01:
        notes.append("Vp/Vs ≈ √2: Zero Poisson ratio material")

    return PoissonRatioReport(
        poisson_ratio=nu,
        Vp_Vs_ratio=Vp_Vs,
        K_over_G=K_over_G if Vs > 0 else float('inf'),
        is_thermodynamically_stable=is_stable,
        is_typical_rock=is_typical,
        material_type=material_type,
        notes=notes
    )


def convert_elastic_moduli(
    K: Optional[float] = None,
    G: Optional[float] = None,
    E: Optional[float] = None,
    nu: Optional[float] = None,
    lam: Optional[float] = None
) -> ElasticModuliReport:
    """
    Convert between any two elastic moduli to get all five.

    Provide exactly two of: K, G, E, nu, lambda (lam)

    The five moduli are related - any two determine the rest:
    - K: Bulk modulus (resistance to uniform compression)
    - G (or mu): Shear modulus (resistance to shape change)
    - E: Young's modulus (tensile stiffness)
    - nu: Poisson's ratio (lateral/axial strain ratio)
    - lambda: Lame's first parameter
    - M: P-wave modulus M = K + 4G/3 = lambda + 2G (computed, not input)

    Key relationships:
    - E = 9KG / (3K + G)
    - nu = (3K - 2G) / (6K + 2G)
    - lambda = K - 2G/3
    - M = K + 4G/3 = lambda + 2G

    Returns:
        ElasticModuliReport with all moduli
    """
    # Count provided parameters
    params = {'K': K, 'G': G, 'E': E, 'nu': nu, 'lambda': lam}
    provided = {k: v for k, v in params.items() if v is not None}

    if len(provided) != 2:
        raise ValueError(f"Provide exactly 2 moduli, got {len(provided)}: {list(provided.keys())}")

    input_str = ", ".join(f"{k}={v}" for k, v in provided.items())

    # Convert based on which two are provided
    if K is not None and G is not None:
        # Direct case - K and G given
        pass

    elif K is not None and E is not None:
        # G = 3KE / (9K - E)
        denom = 9*K - E
        if abs(denom) < 1e-30:
            raise ValueError("Invalid combination: 9K = E")
        G = 3*K*E / denom

    elif K is not None and nu is not None:
        # G = 3K(1-2nu) / (2(1+nu))
        if nu <= -1 or nu >= 0.5:
            raise ValueError(f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}")
        G = 3*K*(1-2*nu) / (2*(1+nu))

    elif K is not None and lam is not None:
        # G = 3(K - lambda) / 2
        G = 1.5 * (K - lam)

    elif G is not None and E is not None:
        # K = EG / (3(3G - E))
        denom = 3*(3*G - E)
        if abs(denom) < 1e-30:
            raise ValueError("Invalid combination: E = 3G")
        K = E*G / denom

    elif G is not None and nu is not None:
        # K = 2G(1+nu) / (3(1-2nu))
        if nu <= -1 or nu >= 0.5:
            raise ValueError(f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}")
        K = 2*G*(1+nu) / (3*(1-2*nu))

    elif G is not None and lam is not None:
        # K = lambda + 2G/3
        K = lam + 2*G/3

    elif E is not None and nu is not None:
        # K = E / (3(1-2nu))
        # G = E / (2(1+nu))
        if nu <= -1 or nu >= 0.5:
            raise ValueError(f"Poisson ratio must satisfy -1 < nu < 0.5, got {nu}")
        K = E / (3*(1-2*nu))
        G = E / (2*(1+nu))

    elif E is not None and lam is not None:
        # More complex - solve quadratic
        # E = mu(3*lambda + 2*mu) / (lambda + mu)
        # This gives: mu^2 + lambda*mu - E*lambda/3 = 0... actually simpler:
        # Use: G = (E - 3*lambda + sqrt((E-3*lambda)^2 + 8*lambda*E)) / 4
        # Simplification: G = (E - 3*lambda + R) / 4 where R = sqrt(E^2 + 9*lambda^2 + 2*E*lambda)
        R = math.sqrt(E*E + 9*lam*lam + 2*E*lam)
        G = (E - 3*lam + R) / 4
        K = lam + 2*G/3

    elif nu is not None and lam is not None:
        # G = lambda(1-2nu) / (2nu)
        if nu <= 0 or nu >= 0.5:
            raise ValueError(f"For lambda+nu input, need 0 < nu < 0.5, got {nu}")
        G = lam*(1-2*nu) / (2*nu)
        K = lam + 2*G/3

    else:
        raise ValueError(f"Unhandled combination: {list(provided.keys())}")

    # Now compute all from K and G
    if K <= 0:
        raise ValueError(f"Computed bulk modulus K={K} is not positive")
    if G < 0:
        raise ValueError(f"Computed shear modulus G={G} is negative")

    # Poisson ratio
    denom_nu = 6*K + 2*G
    nu_calc = (3*K - 2*G) / denom_nu if abs(denom_nu) > 1e-30 else 0.5

    # Young's modulus
    denom_E = 3*K + G
    E_calc = 9*K*G / denom_E if abs(denom_E) > 1e-30 else 0.0

    # Lame lambda
    lam_calc = K - 2*G/3

    # P-wave modulus
    M = K + 4*G/3

    return ElasticModuliReport(
        bulk_modulus_K=K,
        shear_modulus_G=G,
        youngs_modulus_E=E_calc,
        poisson_ratio_nu=nu_calc,
        lame_lambda=lam_calc,
        p_wave_modulus_M=M,
        input_params=input_str
    )


def calc_reflection_coefficient(
    rho1: float, Vp1: float,
    rho2: float, Vp2: float,
    Vs1: Optional[float] = None,
    Vs2: Optional[float] = None
) -> ReflectionReport:
    """
    Calculate reflection and transmission coefficients at normal incidence.

    For P-waves at normal incidence:
    - R = (Z2 - Z1) / (Z2 + Z1)  where Z = rho * Vp
    - T = 2*Z1 / (Z2 + Z1)

    Energy conservation: R^2 + (Z2/Z1)*T^2 = 1
    (Note: T is amplitude coefficient, energy requires impedance ratio)

    Args:
        rho1, Vp1: Density and P-velocity of layer 1 (incident)
        rho2, Vp2: Density and P-velocity of layer 2 (transmitted)
        Vs1, Vs2: Optional S-wave velocities for S-wave reflection

    Returns:
        ReflectionReport with R, T coefficients

    Sign convention:
    - R > 0: Polarity preserved (Z2 > Z1)
    - R < 0: Polarity reversed (Z2 < Z1, "soft" boundary)
    """
    if any(x <= 0 for x in [rho1, Vp1, rho2, Vp2]):
        raise ValueError("All densities and P-velocities must be positive")

    # P-wave impedances
    Z1_P = rho1 * Vp1
    Z2_P = rho2 * Vp2

    # P-wave reflection/transmission
    R_P = (Z2_P - Z1_P) / (Z2_P + Z1_P)
    T_P = 2 * Z1_P / (Z2_P + Z1_P)

    # S-wave coefficients (if velocities provided)
    notes = []
    if Vs1 is not None and Vs2 is not None:
        if Vs1 <= 0 or Vs2 <= 0:
            R_S = 0.0
            T_S = 0.0
            notes.append("Zero S-velocity: fluid layer present")
        else:
            Z1_S = rho1 * Vs1
            Z2_S = rho2 * Vs2
            R_S = (Z2_S - Z1_S) / (Z2_S + Z1_S)
            T_S = 2 * Z1_S / (Z2_S + Z1_S)
    else:
        R_S = 0.0
        T_S = 0.0
        notes.append("S-wave velocities not provided - S coefficients set to 0")

    # Check energy conservation for P-wave
    # Amplitude: R + T = 1 + R (always true by construction)
    # Energy: R^2 + (Z2/Z1)*T^2 should = 1
    energy_check = R_P**2 + (Z2_P/Z1_P) * T_P**2
    energy_conserved = abs(energy_check - 1.0) < 1e-10

    if R_P < 0:
        notes.append("Negative R: polarity reversal (soft boundary)")
    if abs(R_P) > 0.5:
        notes.append("Strong reflector: |R| > 0.5")

    return ReflectionReport(
        R_P=R_P,
        T_P=T_P,
        R_S=R_S,
        T_S=T_S,
        Z1_P=Z1_P,
        Z2_P=Z2_P,
        energy_conserved=energy_conserved,
        notes=notes
    )


def critical_angle(Vp1: float, Vp2: float) -> float:
    """
    Calculate critical angle for total internal reflection.

    sin(theta_c) = V1 / V2 (V2 > V1 required)

    At angles >= critical angle, no transmitted wave - all energy reflected.

    Args:
        Vp1: Velocity in layer 1 (incident medium)
        Vp2: Velocity in layer 2 (must be > Vp1)

    Returns:
        Critical angle in degrees

    Raises:
        ValueError: If V1 >= V2 (no critical angle exists)
    """
    if Vp1 <= 0 or Vp2 <= 0:
        raise ValueError("Velocities must be positive")
    if Vp1 >= Vp2:
        raise ValueError(f"V1 must be < V2 for critical angle to exist. Got V1={Vp1}, V2={Vp2}")

    sin_theta = Vp1 / Vp2
    theta_rad = math.asin(sin_theta)
    theta_deg = math.degrees(theta_rad)

    return theta_deg


def snells_law(
    theta1_deg: float,
    V1: float,
    V2: float
) -> Tuple[float, bool]:
    """
    Apply Snell's law to find refracted angle.

    sin(theta1)/V1 = sin(theta2)/V2

    Args:
        theta1_deg: Incident angle in degrees
        V1: Velocity in medium 1
        V2: Velocity in medium 2

    Returns:
        Tuple of (theta2_deg, is_transmitted)
        If angle exceeds critical, returns (90.0, False)
    """
    if V1 <= 0 or V2 <= 0:
        raise ValueError("Velocities must be positive")
    if not 0 <= theta1_deg <= 90:
        raise ValueError(f"Incident angle must be 0-90 degrees, got {theta1_deg}")

    theta1_rad = math.radians(theta1_deg)
    sin_theta2 = (V2 / V1) * math.sin(theta1_rad)

    if sin_theta2 > 1.0:
        # Beyond critical angle
        return 90.0, False

    theta2_rad = math.asin(sin_theta2)
    theta2_deg = math.degrees(theta2_rad)

    return theta2_deg, True


def vp_vs_ratio_bounds() -> dict:
    """
    Return theoretical and practical bounds on Vp/Vs ratio.

    CRITICAL RELATIONSHIP:
    Vp/Vs = sqrt((K/G) + 4/3) = sqrt((2*(1-nu))/(1-2*nu))

    Bounds from Poisson ratio constraints:
    - nu = 0: Vp/Vs = sqrt(2) ≈ 1.414
    - nu = 0.25: Vp/Vs = sqrt(3) ≈ 1.732 (Poisson solid)
    - nu -> 0.5: Vp/Vs -> infinity (liquid)
    - nu = -1 (unstable limit): Vp/Vs = 0 (impossible)

    Returns:
        Dictionary with theoretical and practical bounds
    """
    return {
        "theoretical_minimum": math.sqrt(2),
        "theoretical_minimum_nu": 0.0,
        "poisson_solid": math.sqrt(3),
        "poisson_solid_nu": 0.25,
        "typical_rock_low": 1.5,  # ~nu = 0.1
        "typical_rock_high": 2.5,  # ~nu = 0.4
        "water_saturated_sediment": 2.5,  # nu ~ 0.4
        "shale_high": 3.0,  # nu ~ 0.44
        "liquid_limit": float('inf'),
        "liquid_limit_nu": 0.5,
        "common_values": {
            "granite": (1.70, 1.76),
            "sandstone_dry": (1.50, 1.70),
            "sandstone_saturated": (1.80, 2.20),
            "shale": (2.00, 3.00),
            "limestone": (1.80, 2.00),
            "dolomite": (1.75, 1.90),
            "salt": (1.77, 1.77),  # Very consistent
            "water": (float('inf'), float('inf')),
        },
        "notes": [
            "Vp/Vs < sqrt(2) requires nu < 0 (auxetic materials)",
            "Vp/Vs = sqrt(3) is the classic 'Poisson solid' reference",
            "Fluid saturation increases Vp/Vs (increases K, not G)",
            "Gas saturation decreases Vp/Vs (decreases K dramatically)",
        ]
    }
