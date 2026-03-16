"""noethersolve.pk_model — Pharmacokinetic modeling computational engine.

Computes pharmacokinetic parameters from first principles: one-compartment
IV bolus and oral dosing, half-life, clearance, steady-state concentrations,
AUC, and dose adjustments for CYP inhibition/induction.

Conservation law philosophy: pharmacokinetics IS mass balance. The
one-compartment model is a conservation law: rate of drug in = rate of
drug out + rate of accumulation. At steady state, input rate = elimination
rate (the PK steady-state is a conservation constraint).

Replaces the static pharmacokinetics lookup table with actual computation,
while keeping the CYP interaction database for drug interaction checking.

Usage:
    from noethersolve.pk_model import (
        one_compartment_iv, one_compartment_oral, half_life,
        steady_state, auc_single_dose, dose_adjustment,
    )

    # IV bolus
    r = one_compartment_iv(dose=500, Vd=50, ke=0.1, t=6)
    print(r)  # C(6h) = 5.49 mg/L

    # Oral dosing
    r = one_compartment_oral(dose=500, F=0.8, Vd=50, ka=1.5, ke=0.1, t=4)
    print(r)  # C(4h) = ...

    # Half-life from clearance
    r = half_life(CL=10, Vd=50)
    print(r)  # t½ = 3.47 h

    # Steady state
    r = steady_state(dose=500, F=0.8, CL=10, tau=8)
    print(r)  # Css_avg = 5.0 mg/L
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

LN2 = math.log(2)  # 0.693...


# ─── Report Dataclasses ──────────────────────────────────────────────────────

@dataclass
class IVBolusReport:
    """One-compartment IV bolus pharmacokinetics."""
    dose: float          # mg
    Vd: float            # L (volume of distribution)
    ke: float            # h⁻¹ (elimination rate constant)
    t: float             # h (time point)
    C0: float            # mg/L (initial concentration = dose/Vd)
    Ct: float            # mg/L (concentration at time t)
    half_life: float     # h
    CL: float            # L/h (clearance = ke × Vd)
    AUC_inf: float       # mg·h/L (area under curve, 0→∞)
    fraction_remaining: float
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  One-Compartment IV Bolus Model", "=" * 60]
        lines.append(f"  Dose = {self.dose:.4g} mg    Vd = {self.Vd:.4g} L    ke = {self.ke:.4g} h⁻¹")
        lines.append(f"  C₀ = Dose/Vd = {self.C0:.4g} mg/L")
        lines.append(f"  C(t={self.t:.2g}h) = C₀ × e^(-ke×t) = {self.Ct:.4g} mg/L")
        lines.append(f"  Fraction remaining: {self.fraction_remaining:.1%}")
        lines.append("-" * 60)
        lines.append(f"  Half-life = ln(2)/ke = {self.half_life:.4g} h")
        lines.append(f"  Clearance = ke × Vd = {self.CL:.4g} L/h")
        lines.append(f"  AUC(0→∞) = Dose/CL = {self.AUC_inf:.4g} mg·h/L")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class OralDosingReport:
    """One-compartment oral dosing pharmacokinetics."""
    dose: float          # mg
    F: float             # bioavailability (0-1)
    Vd: float            # L
    ka: float            # h⁻¹ (absorption rate constant)
    ke: float            # h⁻¹ (elimination rate constant)
    t: float             # h
    Ct: float            # mg/L (concentration at time t)
    Cmax: float          # mg/L (peak concentration)
    Tmax: float          # h (time to peak)
    half_life: float     # h
    CL: float            # L/h
    AUC_inf: float       # mg·h/L
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  One-Compartment Oral Dosing Model", "=" * 60]
        lines.append(f"  Dose = {self.dose:.4g} mg    F = {self.F:.2f}    Vd = {self.Vd:.4g} L")
        lines.append(f"  ka = {self.ka:.4g} h⁻¹    ke = {self.ke:.4g} h⁻¹")
        lines.append(f"  C(t={self.t:.2g}h) = {self.Ct:.4g} mg/L")
        lines.append("-" * 60)
        lines.append(f"  Cmax = {self.Cmax:.4g} mg/L at Tmax = {self.Tmax:.4g} h")
        lines.append(f"  Half-life = {self.half_life:.4g} h")
        lines.append(f"  Clearance = ke × Vd = {self.CL:.4g} L/h")
        lines.append(f"  AUC(0→∞) = F×Dose/CL = {self.AUC_inf:.4g} mg·h/L")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class HalfLifeReport:
    """Half-life calculation from PK parameters."""
    half_life: float     # h
    ke: float            # h⁻¹
    CL: Optional[float]  # L/h (if provided)
    Vd: Optional[float]  # L (if provided)
    method: str          # how it was calculated
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Half-Life Calculation", "=" * 60]
        lines.append(f"  t½ = {self.half_life:.4g} h")
        lines.append(f"  ke = ln(2)/t½ = {self.ke:.4g} h⁻¹")
        if self.CL is not None:
            lines.append(f"  CL = {self.CL:.4g} L/h")
        if self.Vd is not None:
            lines.append(f"  Vd = {self.Vd:.4g} L")
        lines.append(f"  Method: {self.method}")
        lines.append("-" * 60)
        lines.append(f"  After 1 t½: 50% remaining")
        lines.append(f"  After 3 t½: 12.5% remaining")
        lines.append(f"  After 5 t½: 3.1% remaining (effectively eliminated)")
        lines.append(f"  Time to 97% elimination: {5 * self.half_life:.4g} h")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SteadyStateReport:
    """Steady-state pharmacokinetics for repeated dosing."""
    dose: float          # mg
    F: float             # bioavailability
    CL: float            # L/h
    tau: float           # h (dosing interval)
    Css_avg: float       # mg/L (average steady-state concentration)
    Css_peak: float      # mg/L (peak at steady state, IV approx)
    Css_trough: float    # mg/L (trough at steady state)
    accumulation_factor: float  # ratio of Css to single-dose Cmax
    time_to_ss: float    # h (time to ~97% of steady state = 5 × t½)
    half_life: float     # h
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Steady-State Pharmacokinetics", "=" * 60]
        lines.append(f"  Dose = {self.dose:.4g} mg    F = {self.F:.2f}    τ = {self.tau:.4g} h")
        lines.append(f"  CL = {self.CL:.4g} L/h    t½ = {self.half_life:.4g} h")
        lines.append("-" * 60)
        lines.append(f"  Css_avg = F×Dose/(CL×τ) = {self.Css_avg:.4g} mg/L")
        lines.append(f"  Css_peak ≈ {self.Css_peak:.4g} mg/L")
        lines.append(f"  Css_trough ≈ {self.Css_trough:.4g} mg/L")
        lines.append(f"  Accumulation factor = {self.accumulation_factor:.4g}")
        lines.append(f"  Time to steady state (~97%) = 5 × t½ = {self.time_to_ss:.4g} h")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class AUCReport:
    """Area under the curve for a single dose."""
    dose: float          # mg
    F: float             # bioavailability
    CL: float            # L/h
    AUC_inf: float       # mg·h/L
    AUC_units: str       # "mg·h/L"
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Area Under the Curve (AUC)", "=" * 60]
        lines.append(f"  Dose = {self.dose:.4g} mg    F = {self.F:.2f}    CL = {self.CL:.4g} L/h")
        lines.append(f"  AUC(0→∞) = F×Dose/CL = {self.AUC_inf:.4g} {self.AUC_units}")
        lines.append("-" * 60)
        lines.append("  AUC is the primary measure of total drug exposure.")
        lines.append("  Bioequivalence: two formulations are equivalent if AUC ratio")
        lines.append("  is within 80-125% (FDA guideline).")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DoseAdjustmentReport:
    """Dose adjustment for CYP inhibition/induction or organ impairment."""
    original_dose: float    # mg
    adjusted_dose: float    # mg
    fold_change: float      # AUC fold-change from interaction
    adjustment_factor: float  # dose multiplier
    reason: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Dose Adjustment", "=" * 60]
        lines.append(f"  Original dose: {self.original_dose:.4g} mg")
        lines.append(f"  Adjusted dose: {self.adjusted_dose:.4g} mg")
        lines.append(f"  Adjustment factor: {self.adjustment_factor:.4g}×")
        lines.append(f"  AUC fold-change: {self.fold_change:.4g}×")
        lines.append(f"  Reason: {self.reason}")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Public API ──────────────────────────────────────────────────────────────

def one_compartment_iv(
    dose: float,
    Vd: float,
    ke: float,
    t: float,
) -> IVBolusReport:
    """Compute drug concentration after IV bolus using one-compartment model.

    C(t) = (Dose/Vd) × e^(-ke×t)

    Args:
        dose: dose in mg
        Vd: volume of distribution in L
        ke: elimination rate constant in h⁻¹
        t: time in hours

    Returns:
        IVBolusReport with concentration, half-life, clearance, and AUC.
    """
    if dose <= 0:
        raise ValueError(f"Dose must be positive, got {dose}")
    if Vd <= 0:
        raise ValueError(f"Vd must be positive, got {Vd}")
    if ke <= 0:
        raise ValueError(f"ke must be positive, got {ke}")
    if t < 0:
        raise ValueError(f"Time must be non-negative, got {t}")

    C0 = dose / Vd
    Ct = C0 * math.exp(-ke * t)
    t_half = LN2 / ke
    CL = ke * Vd
    AUC = dose / CL
    frac = Ct / C0 if C0 > 0 else 0.0

    notes = []
    n_halflives = t / t_half if t_half > 0 else 0
    if n_halflives >= 5:
        notes.append(f"≥5 half-lives elapsed: drug is effectively eliminated ({frac:.1%} remaining)")
    if Vd > 100:
        notes.append(f"Large Vd ({Vd:.0f} L) suggests extensive tissue distribution")

    return IVBolusReport(
        dose=dose, Vd=Vd, ke=ke, t=t, C0=C0, Ct=Ct,
        half_life=t_half, CL=CL, AUC_inf=AUC,
        fraction_remaining=frac, notes=notes,
    )


def one_compartment_oral(
    dose: float,
    F: float,
    Vd: float,
    ka: float,
    ke: float,
    t: float,
) -> OralDosingReport:
    """Compute drug concentration after oral dosing (Bateman function).

    C(t) = (F×Dose×ka) / (Vd×(ka-ke)) × (e^(-ke×t) - e^(-ka×t))

    Args:
        dose: dose in mg
        F: bioavailability (0 to 1)
        Vd: volume of distribution in L
        ka: absorption rate constant in h⁻¹
        ke: elimination rate constant in h⁻¹
        t: time in hours

    Returns:
        OralDosingReport with concentration, Cmax, Tmax, and PK parameters.
    """
    if dose <= 0:
        raise ValueError(f"Dose must be positive, got {dose}")
    if not 0 < F <= 1:
        raise ValueError(f"Bioavailability F must be in (0, 1], got {F}")
    if Vd <= 0:
        raise ValueError(f"Vd must be positive, got {Vd}")
    if ka <= 0:
        raise ValueError(f"ka must be positive, got {ka}")
    if ke <= 0:
        raise ValueError(f"ke must be positive, got {ke}")
    if t < 0:
        raise ValueError(f"Time must be non-negative, got {t}")
    if abs(ka - ke) < 1e-10:
        raise ValueError("ka and ke must differ (ka ≈ ke causes numerical singularity)")

    # Bateman function
    coeff = (F * dose * ka) / (Vd * (ka - ke))
    Ct = coeff * (math.exp(-ke * t) - math.exp(-ka * t))
    Ct = max(Ct, 0.0)  # numerical safety

    # Time to peak (Tmax)
    Tmax = math.log(ka / ke) / (ka - ke)
    Cmax = coeff * (math.exp(-ke * Tmax) - math.exp(-ka * Tmax))
    Cmax = max(Cmax, 0.0)

    t_half = LN2 / ke
    CL = ke * Vd
    AUC = F * dose / CL

    notes = []
    if ka < 2 * ke:
        notes.append("ka/ke ratio < 2: absorption and elimination overlap significantly (flip-flop kinetics possible)")
    if F < 0.3:
        notes.append(f"Low bioavailability (F={F:.2f}): significant first-pass metabolism")

    return OralDosingReport(
        dose=dose, F=F, Vd=Vd, ka=ka, ke=ke, t=t,
        Ct=Ct, Cmax=Cmax, Tmax=Tmax, half_life=t_half,
        CL=CL, AUC_inf=AUC, notes=notes,
    )


def half_life(
    CL: Optional[float] = None,
    Vd: Optional[float] = None,
    ke: Optional[float] = None,
) -> HalfLifeReport:
    """Compute elimination half-life from PK parameters.

    t½ = ln(2) / ke = 0.693 × Vd / CL

    Provide either ke alone, or CL and Vd.

    Args:
        CL: clearance in L/h
        Vd: volume of distribution in L
        ke: elimination rate constant in h⁻¹

    Returns:
        HalfLifeReport with half-life and derived parameters.
    """
    if ke is not None:
        if ke <= 0:
            raise ValueError(f"ke must be positive, got {ke}")
        t_half = LN2 / ke
        method = "t½ = ln(2)/ke"
        # Derive CL if Vd provided
        if Vd is not None and Vd > 0:
            CL = ke * Vd
    elif CL is not None and Vd is not None:
        if CL <= 0:
            raise ValueError(f"CL must be positive, got {CL}")
        if Vd <= 0:
            raise ValueError(f"Vd must be positive, got {Vd}")
        ke = CL / Vd
        t_half = LN2 / ke
        method = "t½ = 0.693 × Vd / CL"
    else:
        raise ValueError("Provide either ke, or both CL and Vd")

    notes = []
    if t_half < 1:
        notes.append(f"Very short half-life ({t_half:.2g}h): may need frequent dosing or continuous infusion")
    elif t_half > 24:
        notes.append(f"Long half-life ({t_half:.1f}h): once-daily or less frequent dosing possible")
        notes.append(f"Time to steady state: ~{5*t_half:.0f}h ({5*t_half/24:.1f} days)")

    return HalfLifeReport(
        half_life=t_half, ke=ke, CL=CL, Vd=Vd,
        method=method, notes=notes,
    )


def steady_state(
    dose: float,
    F: float,
    CL: float,
    tau: float,
    Vd: Optional[float] = None,
) -> SteadyStateReport:
    """Compute steady-state concentrations for repeated dosing.

    Css_avg = F × Dose / (CL × τ)

    Args:
        dose: dose per administration in mg
        F: bioavailability (0 to 1)
        CL: clearance in L/h
        tau: dosing interval in hours
        Vd: volume of distribution in L (for peak/trough estimates)

    Returns:
        SteadyStateReport with average, peak, and trough concentrations.
    """
    if dose <= 0:
        raise ValueError(f"Dose must be positive, got {dose}")
    if not 0 < F <= 1:
        raise ValueError(f"F must be in (0, 1], got {F}")
    if CL <= 0:
        raise ValueError(f"CL must be positive, got {CL}")
    if tau <= 0:
        raise ValueError(f"Dosing interval τ must be positive, got {tau}")

    Css_avg = F * dose / (CL * tau)

    # Derive ke for peak/trough if Vd provided
    if Vd is not None and Vd > 0:
        ke = CL / Vd
        t_half = LN2 / ke
        # IV bolus approximation for peak/trough
        C0 = F * dose / Vd
        accum = 1 / (1 - math.exp(-ke * tau))
        Css_peak = C0 * accum
        Css_trough = C0 * accum * math.exp(-ke * tau)
    else:
        ke = CL / 50  # rough estimate if Vd not provided
        t_half = LN2 / ke
        Vd_est = CL / ke
        C0 = F * dose / Vd_est
        accum = 1 / (1 - math.exp(-ke * tau))
        Css_peak = C0 * accum
        Css_trough = C0 * accum * math.exp(-ke * tau)

    time_to_ss = 5 * t_half
    accum_factor = 1 / (1 - math.exp(-ke * tau))

    notes = []
    if tau > 2 * t_half:
        notes.append(f"Dosing interval ({tau:.1f}h) > 2 half-lives ({t_half:.1f}h): "
                     "significant peak-trough fluctuation")
    if tau < 0.5 * t_half:
        notes.append(f"Dosing interval ({tau:.1f}h) < ½ half-life ({t_half:.1f}h): "
                     "near-constant drug levels (like continuous infusion)")
    if accum_factor > 3:
        notes.append(f"High accumulation (factor {accum_factor:.1f}×): "
                     "monitor for toxicity during loading period")

    return SteadyStateReport(
        dose=dose, F=F, CL=CL, tau=tau,
        Css_avg=Css_avg, Css_peak=Css_peak, Css_trough=Css_trough,
        accumulation_factor=accum_factor, time_to_ss=time_to_ss,
        half_life=t_half, notes=notes,
    )


def auc_single_dose(
    dose: float,
    F: float,
    CL: float,
) -> AUCReport:
    """Compute area under the curve for a single dose.

    AUC(0→∞) = F × Dose / CL

    Args:
        dose: dose in mg
        F: bioavailability (0 to 1)
        CL: clearance in L/h

    Returns:
        AUCReport with total drug exposure.
    """
    if dose <= 0:
        raise ValueError(f"Dose must be positive, got {dose}")
    if not 0 < F <= 1:
        raise ValueError(f"F must be in (0, 1], got {F}")
    if CL <= 0:
        raise ValueError(f"CL must be positive, got {CL}")

    AUC = F * dose / CL

    notes = []
    return AUCReport(
        dose=dose, F=F, CL=CL, AUC_inf=AUC,
        AUC_units="mg·h/L", notes=notes,
    )


def dose_adjustment(
    original_dose: float,
    fold_change_auc: float,
    reason: str = "",
) -> DoseAdjustmentReport:
    """Compute adjusted dose given a fold-change in AUC from drug interaction
    or organ impairment.

    If a CYP inhibitor causes a 5× increase in AUC, the dose should be
    reduced by 1/5 to maintain the same exposure.

    Args:
        original_dose: current dose in mg
        fold_change_auc: fold-change in AUC (e.g., 5.0 for strong CYP3A4 inhibitor)
        reason: description of why adjustment is needed

    Returns:
        DoseAdjustmentReport with adjusted dose and explanation.
    """
    if original_dose <= 0:
        raise ValueError(f"Dose must be positive, got {original_dose}")
    if fold_change_auc <= 0:
        raise ValueError(f"AUC fold-change must be positive, got {fold_change_auc}")

    adjustment_factor = 1.0 / fold_change_auc
    adjusted = original_dose * adjustment_factor

    notes = []
    if fold_change_auc > 5:
        notes.append(f"Strong interaction ({fold_change_auc:.1f}× AUC increase): "
                     "consider avoiding combination or use alternative drug")
    elif fold_change_auc > 2:
        notes.append(f"Moderate interaction ({fold_change_auc:.1f}× AUC increase): "
                     "dose reduction recommended")
    elif fold_change_auc < 0.5:
        notes.append(f"Strong induction ({fold_change_auc:.2f}× AUC, i.e., "
                     f"{(1-fold_change_auc)*100:.0f}% decrease): "
                     "consider dose increase or alternative drug")

    if not reason:
        if fold_change_auc > 1:
            reason = f"CYP inhibition causing {fold_change_auc:.1f}× AUC increase"
        else:
            reason = f"CYP induction causing {(1-fold_change_auc)*100:.0f}% AUC decrease"

    return DoseAdjustmentReport(
        original_dose=original_dose, adjusted_dose=adjusted,
        fold_change=fold_change_auc, adjustment_factor=adjustment_factor,
        reason=reason, notes=notes,
    )
