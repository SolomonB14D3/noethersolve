"""Human physiology calculator — cardiovascular, renal, and pulmonary systems.

Pure-math implementations of standard clinical equations. No external dependencies.

Usage:
    from noethersolve.physiology import calc_cardiac_output, calc_map, calc_gfr_ckd_epi

    co = calc_cardiac_output(heart_rate=72, stroke_volume_ml=70)
    print(co)  # CO = 5.04 L/min, CI = 2.91 L/min/m² — Normal

    gfr = calc_gfr_ckd_epi(creatinine_mgdl=1.2, age=55, is_female=False)
    print(gfr)  # eGFR = 72.3 mL/min/1.73m² — Stage G2
"""

from dataclasses import dataclass
from typing import Optional
import math


# ---------------------------------------------------------------------------
# 1. Cardiac Output
# ---------------------------------------------------------------------------

@dataclass
class CardiacOutputResult:
    heart_rate: float
    stroke_volume_ml: float
    cardiac_output_lpm: float
    cardiac_index: float
    bsa: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Cardiac Output\n"
            f"  HR: {self.heart_rate:.0f} bpm\n"
            f"  SV: {self.stroke_volume_ml:.1f} mL\n"
            f"  CO: {self.cardiac_output_lpm:.2f} L/min\n"
            f"  CI: {self.cardiac_index:.2f} L/min/m² (BSA={self.bsa:.2f})\n"
            f"  {self.interpretation}"
        )


def calc_cardiac_output(
    heart_rate: float,
    stroke_volume_ml: float,
    bsa: float = 1.73,
) -> CardiacOutputResult:
    """Calculate cardiac output and cardiac index.

    CO = HR x SV. Normal CO 4-8 L/min, CI 2.5-4.0 L/min/m².

    Args:
        heart_rate: beats per minute
        stroke_volume_ml: stroke volume in mL
        bsa: body surface area in m² (default 1.73)
    """
    co = heart_rate * stroke_volume_ml / 1000.0  # mL -> L
    ci = co / bsa

    parts = []
    if co < 4.0:
        parts.append("Low CO (<4 L/min) — consider cardiogenic shock or hypovolemia")
    elif co > 8.0:
        parts.append("High CO (>8 L/min) — consider sepsis, thyrotoxicosis, or AV fistula")
    else:
        parts.append("CO within normal range (4-8 L/min)")

    if ci < 2.5:
        parts.append("Low CI (<2.5) — inadequate tissue perfusion")
    elif ci > 4.0:
        parts.append("Elevated CI (>4.0) — hyperdynamic state")
    else:
        parts.append("CI within normal range (2.5-4.0)")

    return CardiacOutputResult(
        heart_rate=heart_rate,
        stroke_volume_ml=stroke_volume_ml,
        cardiac_output_lpm=co,
        cardiac_index=ci,
        bsa=bsa,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 2. Mean Arterial Pressure
# ---------------------------------------------------------------------------

@dataclass
class MAPResult:
    systolic: float
    diastolic: float
    map_mmhg: float
    pulse_pressure: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Mean Arterial Pressure\n"
            f"  SBP/DBP: {self.systolic:.0f}/{self.diastolic:.0f} mmHg\n"
            f"  MAP: {self.map_mmhg:.1f} mmHg\n"
            f"  Pulse pressure: {self.pulse_pressure:.0f} mmHg\n"
            f"  {self.interpretation}"
        )


def calc_map(systolic: float, diastolic: float) -> MAPResult:
    """Calculate mean arterial pressure and pulse pressure.

    MAP = DBP + 1/3(SBP - DBP). Normal MAP 70-105 mmHg.
    Normal pulse pressure 30-50 mmHg.

    Args:
        systolic: systolic blood pressure in mmHg
        diastolic: diastolic blood pressure in mmHg
    """
    pp = systolic - diastolic
    map_val = diastolic + pp / 3.0

    parts = []
    if map_val < 60:
        parts.append("MAP critically low (<60 mmHg) — risk of organ hypoperfusion")
    elif map_val < 70:
        parts.append("MAP low (60-70 mmHg) — borderline perfusion")
    elif map_val > 105:
        parts.append("MAP elevated (>105 mmHg) — hypertensive")
    else:
        parts.append("MAP within normal range (70-105 mmHg)")

    if pp > 60:
        parts.append("Wide pulse pressure — consider aortic regurgitation, atherosclerosis, or hyperthyroidism")
    elif pp < 25:
        parts.append("Narrow pulse pressure — consider heart failure, aortic stenosis, or tamponade")
    else:
        parts.append("Pulse pressure normal (30-50 mmHg)")

    return MAPResult(
        systolic=systolic,
        diastolic=diastolic,
        map_mmhg=map_val,
        pulse_pressure=pp,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 3. Systemic Vascular Resistance
# ---------------------------------------------------------------------------

@dataclass
class VascularResistanceResult:
    map_mmhg: float
    cvp_mmhg: float
    cardiac_output_lpm: float
    svr: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Systemic Vascular Resistance\n"
            f"  MAP: {self.map_mmhg:.1f} mmHg, CVP: {self.cvp_mmhg:.1f} mmHg\n"
            f"  CO: {self.cardiac_output_lpm:.2f} L/min\n"
            f"  SVR: {self.svr:.0f} dynes*s/cm^5\n"
            f"  {self.interpretation}"
        )


def calc_vascular_resistance(
    map_mmhg: float,
    cvp_mmhg: float,
    cardiac_output_lpm: float,
) -> VascularResistanceResult:
    """Calculate systemic vascular resistance.

    SVR = (MAP - CVP) x 80 / CO. Normal SVR 800-1200 dynes*s/cm^5.

    Args:
        map_mmhg: mean arterial pressure in mmHg
        cvp_mmhg: central venous pressure in mmHg
        cardiac_output_lpm: cardiac output in L/min
    """
    if cardiac_output_lpm <= 0:
        raise ValueError("Cardiac output must be positive")

    svr = (map_mmhg - cvp_mmhg) * 80.0 / cardiac_output_lpm

    if svr < 800:
        interp = "Low SVR (<800) — vasodilatory state, consider sepsis or anaphylaxis"
    elif svr > 1200:
        interp = "High SVR (>1200) — vasoconstricted state, consider cardiogenic shock or hypovolemia"
    else:
        interp = "SVR within normal range (800-1200 dynes*s/cm^5)"

    return VascularResistanceResult(
        map_mmhg=map_mmhg,
        cvp_mmhg=cvp_mmhg,
        cardiac_output_lpm=cardiac_output_lpm,
        svr=svr,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 4. Law of Laplace — Wall Stress
# ---------------------------------------------------------------------------

@dataclass
class WallStressResult:
    pressure_mmhg: float
    radius_cm: float
    wall_thickness_cm: float
    wall_stress_dynes_per_cm2: float
    wall_stress_kpa: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Law of Laplace — Wall Stress (sphere)\n"
            f"  Pressure: {self.pressure_mmhg:.1f} mmHg\n"
            f"  Radius: {self.radius_cm:.2f} cm, Wall thickness: {self.wall_thickness_cm:.2f} cm\n"
            f"  Wall stress: {self.wall_stress_dynes_per_cm2:.0f} dynes/cm^2 ({self.wall_stress_kpa:.1f} kPa)\n"
            f"  {self.interpretation}"
        )


_MMHG_TO_DYNES_CM2 = 1333.22
_DYNES_CM2_TO_KPA = 0.0001


def calc_laplace_wall_stress(
    pressure_mmhg: float,
    radius_cm: float,
    wall_thickness_cm: float,
) -> WallStressResult:
    """Calculate wall stress using the Law of Laplace for a spherical chamber.

    sigma = (P x r) / (2 x h). Converts mmHg to dynes/cm^2 internally.
    Applicable to cardiac ventricles (modeled as spheres).

    Args:
        pressure_mmhg: intracavitary pressure in mmHg
        radius_cm: chamber radius in cm
        wall_thickness_cm: wall thickness in cm
    """
    if wall_thickness_cm <= 0:
        raise ValueError("Wall thickness must be positive")

    pressure_dynes = pressure_mmhg * _MMHG_TO_DYNES_CM2
    stress = (pressure_dynes * radius_cm) / (2.0 * wall_thickness_cm)
    stress_kpa = stress * _DYNES_CM2_TO_KPA

    # Interpretation based on LV context
    parts = []
    r_h_ratio = radius_cm / wall_thickness_cm
    if r_h_ratio > 4.0:
        parts.append(f"High r/h ratio ({r_h_ratio:.1f}) — thin wall relative to cavity, increased stress (dilated cardiomyopathy pattern)")
    elif r_h_ratio < 2.0:
        parts.append(f"Low r/h ratio ({r_h_ratio:.1f}) — thick wall relative to cavity, reduced stress (hypertrophic pattern)")
    else:
        parts.append(f"r/h ratio {r_h_ratio:.1f} — within typical range")

    if stress_kpa > 20:
        parts.append("Elevated wall stress — increased myocardial oxygen demand")
    else:
        parts.append("Wall stress within typical physiological range")

    return WallStressResult(
        pressure_mmhg=pressure_mmhg,
        radius_cm=radius_cm,
        wall_thickness_cm=wall_thickness_cm,
        wall_stress_dynes_per_cm2=stress,
        wall_stress_kpa=stress_kpa,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 5. GFR — CKD-EPI 2021 (Race-Free)
# ---------------------------------------------------------------------------

@dataclass
class GFRResult:
    creatinine_mgdl: float
    age: int
    is_female: bool
    egfr: float
    ckd_stage: str
    interpretation: str

    def __str__(self) -> str:
        sex = "Female" if self.is_female else "Male"
        return (
            f"Estimated GFR (CKD-EPI 2021)\n"
            f"  Creatinine: {self.creatinine_mgdl:.2f} mg/dL, Age: {self.age}, Sex: {sex}\n"
            f"  eGFR: {self.egfr:.1f} mL/min/1.73m^2\n"
            f"  {self.ckd_stage}\n"
            f"  {self.interpretation}"
        )


def _ckd_stage(egfr: float) -> str:
    """Return CKD stage string from eGFR value."""
    if egfr >= 90:
        return "Stage G1 (normal or high, >=90)"
    elif egfr >= 60:
        return "Stage G2 (mildly decreased, 60-89)"
    elif egfr >= 45:
        return "Stage G3a (mildly to moderately decreased, 45-59)"
    elif egfr >= 30:
        return "Stage G3b (moderately to severely decreased, 30-44)"
    elif egfr >= 15:
        return "Stage G4 (severely decreased, 15-29)"
    else:
        return "Stage G5 (kidney failure, <15)"


def calc_gfr_ckd_epi(
    creatinine_mgdl: float,
    age: int,
    is_female: bool = False,
) -> GFRResult:
    """Estimate GFR using the CKD-EPI 2021 race-free equation.

    Female: 142 x min(Scr/0.7, 1)^(-0.241) x max(Scr/0.7, 1)^(-1.200) x 0.9938^age x 1.012
    Male:   142 x min(Scr/0.9, 1)^(-0.302) x max(Scr/0.9, 1)^(-1.200) x 0.9938^age

    Args:
        creatinine_mgdl: serum creatinine in mg/dL
        age: age in years
        is_female: True for female, False for male
    """
    if creatinine_mgdl <= 0:
        raise ValueError("Creatinine must be positive")

    if is_female:
        kappa = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        sex_factor = 1.0

    scr_ratio = creatinine_mgdl / kappa
    term1 = min(scr_ratio, 1.0) ** alpha
    term2 = max(scr_ratio, 1.0) ** (-1.200)
    egfr = 142.0 * term1 * term2 * (0.9938 ** age) * sex_factor

    stage = _ckd_stage(egfr)

    # Clinical interpretation
    if egfr >= 90:
        interp = "Normal kidney function — monitor if risk factors present"
    elif egfr >= 60:
        interp = "Mildly reduced — evaluate for CKD causes, monitor annually"
    elif egfr >= 30:
        interp = "Moderately reduced — nephrology referral recommended, adjust renally-cleared drugs"
    elif egfr >= 15:
        interp = "Severely reduced — prepare for renal replacement therapy"
    else:
        interp = "Kidney failure — dialysis or transplant indicated"

    return GFRResult(
        creatinine_mgdl=creatinine_mgdl,
        age=age,
        is_female=is_female,
        egfr=egfr,
        ckd_stage=stage,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 6. Creatinine Clearance — Cockcroft-Gault
# ---------------------------------------------------------------------------

@dataclass
class CrClResult:
    creatinine_mgdl: float
    age: int
    weight_kg: float
    is_female: bool
    crcl: float
    interpretation: str

    def __str__(self) -> str:
        sex = "Female" if self.is_female else "Male"
        return (
            f"Creatinine Clearance (Cockcroft-Gault)\n"
            f"  Creatinine: {self.creatinine_mgdl:.2f} mg/dL, Age: {self.age}, Weight: {self.weight_kg:.1f} kg, Sex: {sex}\n"
            f"  CrCl: {self.crcl:.1f} mL/min\n"
            f"  {self.interpretation}"
        )


def calc_creatinine_clearance(
    creatinine_mgdl: float,
    age: int,
    weight_kg: float,
    is_female: bool = False,
) -> CrClResult:
    """Estimate creatinine clearance using the Cockcroft-Gault equation.

    CrCl = [(140-age) x weight] / (72 x Scr) x (0.85 if female).
    Used for drug dosing adjustments. Not adjusted for BSA.

    Args:
        creatinine_mgdl: serum creatinine in mg/dL
        age: age in years
        weight_kg: actual body weight in kg
        is_female: True for female
    """
    if creatinine_mgdl <= 0:
        raise ValueError("Creatinine must be positive")

    crcl = ((140.0 - age) * weight_kg) / (72.0 * creatinine_mgdl)
    if is_female:
        crcl *= 0.85

    if crcl >= 90:
        interp = "Normal clearance — standard drug dosing"
    elif crcl >= 60:
        interp = "Mildly reduced — check for dose adjustments on renally-cleared drugs"
    elif crcl >= 30:
        interp = "Moderately reduced — dose reduction likely needed for many drugs"
    elif crcl >= 15:
        interp = "Severely reduced — significant dose reduction or avoidance of nephrotoxic drugs"
    else:
        interp = "Minimal clearance — most renally-cleared drugs contraindicated without dialysis"

    return CrClResult(
        creatinine_mgdl=creatinine_mgdl,
        age=age,
        weight_kg=weight_kg,
        is_female=is_female,
        crcl=crcl,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 7. Alveolar Gas Equation
# ---------------------------------------------------------------------------

@dataclass
class AlveolarGasResult:
    fio2: float
    paco2: float
    patm: float
    rq: float
    pao2_alveolar: float
    pao2_arterial: float
    aa_gradient: float
    expected_aa_gradient: Optional[float]
    interpretation: str

    def __str__(self) -> str:
        lines = [
            f"Alveolar Gas Equation",
            f"  FiO2: {self.fio2:.2f}, PaCO2: {self.paco2:.1f} mmHg",
            f"  Patm: {self.patm:.0f} mmHg, RQ: {self.rq:.2f}",
            f"  PAO2 (alveolar): {self.pao2_alveolar:.1f} mmHg",
            f"  PaO2 (arterial): {self.pao2_arterial:.1f} mmHg",
            f"  A-a gradient: {self.aa_gradient:.1f} mmHg",
        ]
        if self.expected_aa_gradient is not None:
            lines.append(f"  Expected A-a gradient for age: {self.expected_aa_gradient:.1f} mmHg")
        lines.append(f"  {self.interpretation}")
        return "\n".join(lines)


def calc_alveolar_gas(
    fio2: float,
    paco2: float,
    pao2: float = 95.0,
    patm: float = 760.0,
    rq: float = 0.8,
    age: Optional[int] = None,
) -> AlveolarGasResult:
    """Calculate alveolar oxygen tension and A-a gradient.

    PAO2 = FiO2 x (Patm - 47) - PaCO2/RQ.
    A-a gradient = PAO2 - PaO2.
    Normal A-a gradient = 2.5 + 0.21 x age.

    Args:
        fio2: fraction of inspired oxygen (0.21-1.0)
        paco2: arterial CO2 in mmHg
        pao2: arterial O2 in mmHg (for A-a gradient calculation)
        patm: atmospheric pressure in mmHg (default 760 at sea level)
        rq: respiratory quotient (default 0.8)
        age: patient age for expected A-a gradient (optional)
    """
    # 47 mmHg = water vapor pressure at body temperature (37C)
    pao2_alv = fio2 * (patm - 47.0) - paco2 / rq
    aa_grad = pao2_alv - pao2

    expected_aa = None
    if age is not None:
        expected_aa = 2.5 + 0.21 * age

    parts = []
    if aa_grad < 0:
        parts.append("Negative A-a gradient — check PaO2/FiO2 values for error")
    elif expected_aa is not None and aa_grad > expected_aa + 5:
        parts.append(
            f"Elevated A-a gradient ({aa_grad:.1f} > expected {expected_aa:.1f}) — "
            "suggests V/Q mismatch, shunt, or diffusion impairment"
        )
    elif expected_aa is not None:
        parts.append(f"A-a gradient within expected range for age — if hypoxemic, consider hypoventilation")
    else:
        if aa_grad <= 15:
            parts.append("A-a gradient normal (<15 mmHg on room air)")
        else:
            parts.append("A-a gradient elevated — suggests V/Q mismatch, shunt, or diffusion impairment")

    if paco2 > 45:
        parts.append("Hypercapnia present — hypoventilation contributing to hypoxemia")

    return AlveolarGasResult(
        fio2=fio2,
        paco2=paco2,
        patm=patm,
        rq=rq,
        pao2_alveolar=pao2_alv,
        pao2_arterial=pao2,
        aa_gradient=aa_grad,
        expected_aa_gradient=expected_aa,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 8. Oxygen Delivery
# ---------------------------------------------------------------------------

@dataclass
class OxygenDeliveryResult:
    hemoglobin: float
    sao2: float
    pao2: float
    cardiac_output: float
    cao2: float
    do2: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Oxygen Delivery\n"
            f"  Hgb: {self.hemoglobin:.1f} g/dL, SaO2: {self.sao2:.1f}%, PaO2: {self.pao2:.1f} mmHg\n"
            f"  CO: {self.cardiac_output:.2f} L/min\n"
            f"  CaO2: {self.cao2:.2f} mL O2/dL\n"
            f"  DO2: {self.do2:.0f} mL O2/min\n"
            f"  {self.interpretation}"
        )


def calc_oxygen_delivery(
    hgb: float,
    sao2: float,
    pao2: float,
    cardiac_output: float,
) -> OxygenDeliveryResult:
    """Calculate arterial oxygen content and oxygen delivery.

    CaO2 = (Hgb x 1.34 x SaO2/100) + (0.003 x PaO2) in mL O2/dL.
    DO2 = CO x CaO2 x 10 in mL O2/min.
    Normal DO2 ~950-1150 mL O2/min.

    Args:
        hgb: hemoglobin in g/dL
        sao2: arterial oxygen saturation as percentage (0-100)
        pao2: arterial partial pressure of oxygen in mmHg
        cardiac_output: cardiac output in L/min
    """
    # 1.34 mL O2 per gram Hgb (Huffner's constant)
    # 0.003 mL O2 per mmHg PaO2 (dissolved O2)
    cao2 = (hgb * 1.34 * sao2 / 100.0) + (0.003 * pao2)
    do2 = cardiac_output * cao2 * 10.0  # dL->L conversion

    parts = []
    if do2 < 600:
        parts.append("Critically low DO2 (<600 mL/min) — tissue hypoxia likely, urgent intervention needed")
    elif do2 < 800:
        parts.append("Low DO2 — approaching critical threshold, optimize CO, Hgb, and SaO2")
    elif do2 > 1200:
        parts.append("Supranormal DO2 — may indicate hyperdynamic state")
    else:
        parts.append("DO2 within normal range (~950-1150 mL O2/min)")

    if hgb < 7.0:
        parts.append("Severe anemia — transfusion threshold for most patients")
    if sao2 < 90:
        parts.append("Hypoxemia (SaO2 < 90%) — steep portion of oxyhemoglobin dissociation curve")

    return OxygenDeliveryResult(
        hemoglobin=hgb,
        sao2=sao2,
        pao2=pao2,
        cardiac_output=cardiac_output,
        cao2=cao2,
        do2=do2,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 9. Anion Gap
# ---------------------------------------------------------------------------

@dataclass
class AnionGapResult:
    na: float
    cl: float
    hco3: float
    albumin: float
    anion_gap: float
    corrected_ag: float
    delta_delta: Optional[float]
    interpretation: str

    def __str__(self) -> str:
        lines = [
            f"Anion Gap Analysis",
            f"  Na: {self.na:.0f}, Cl: {self.cl:.0f}, HCO3: {self.hco3:.0f} mEq/L",
            f"  Albumin: {self.albumin:.1f} g/dL",
            f"  Anion Gap: {self.anion_gap:.1f} mEq/L",
            f"  Corrected AG (albumin-adjusted): {self.corrected_ag:.1f} mEq/L",
        ]
        if self.delta_delta is not None:
            lines.append(f"  Delta-delta ratio: {self.delta_delta:.2f}")
        lines.append(f"  {self.interpretation}")
        return "\n".join(lines)


def calc_anion_gap(
    na: float,
    cl: float,
    hco3: float,
    albumin: float = 4.0,
) -> AnionGapResult:
    """Calculate anion gap with albumin correction and delta-delta ratio.

    AG = Na - (Cl + HCO3). Normal 8-12 mEq/L.
    Corrected AG = AG + 2.5 x (4.0 - albumin).
    Delta-delta = (AG - 12) / (24 - HCO3).

    Args:
        na: sodium in mEq/L
        cl: chloride in mEq/L
        hco3: bicarbonate in mEq/L
        albumin: serum albumin in g/dL (default 4.0)
    """
    ag = na - (cl + hco3)
    corrected = ag + 2.5 * (4.0 - albumin)

    # Delta-delta only meaningful when HCO3 < 24
    delta_delta = None
    if hco3 < 24 and (24 - hco3) > 0:
        delta_delta = (corrected - 12.0) / (24.0 - hco3)

    parts = []
    if corrected > 12:
        parts.append(f"Elevated corrected AG ({corrected:.1f}) — anion gap metabolic acidosis (AGMA)")
        parts.append("Consider MUDPILES: Methanol, Uremia, DKA, Propylene glycol, INH/Iron, Lactic acidosis, Ethylene glycol, Salicylates")
    elif corrected < 8:
        parts.append("Low AG — consider hypoalbuminemia (already corrected), lithium, or lab error")
    else:
        parts.append("AG within normal range (8-12 mEq/L)")

    if delta_delta is not None:
        if delta_delta < 1.0:
            parts.append("Delta-delta <1 — concurrent non-AG metabolic acidosis (hyperchloremic)")
        elif delta_delta > 2.0:
            parts.append("Delta-delta >2 — concurrent metabolic alkalosis or pre-existing elevated HCO3")
        else:
            parts.append("Delta-delta 1-2 — pure anion gap metabolic acidosis")

    if albumin < 4.0:
        parts.append(f"Albumin {albumin:.1f} g/dL — AG corrected upward by {2.5 * (4.0 - albumin):.1f}")

    return AnionGapResult(
        na=na,
        cl=cl,
        hco3=hco3,
        albumin=albumin,
        anion_gap=ag,
        corrected_ag=corrected,
        delta_delta=delta_delta,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 10. Serum Osmolality
# ---------------------------------------------------------------------------

@dataclass
class OsmolalityResult:
    na: float
    glucose: float
    bun: float
    ethanol: float
    calculated_osm: float
    measured_osm: Optional[float]
    osmolal_gap: Optional[float]
    interpretation: str

    def __str__(self) -> str:
        lines = [
            f"Serum Osmolality",
            f"  Na: {self.na:.0f} mEq/L, Glucose: {self.glucose:.0f} mg/dL, BUN: {self.bun:.0f} mg/dL",
        ]
        if self.ethanol > 0:
            lines.append(f"  Ethanol: {self.ethanol:.0f} mg/dL")
        lines.append(f"  Calculated Osm: {self.calculated_osm:.0f} mOsm/kg")
        if self.measured_osm is not None:
            lines.append(f"  Measured Osm: {self.measured_osm:.0f} mOsm/kg")
            lines.append(f"  Osmolal Gap: {self.osmolal_gap:.0f} mOsm/kg")
        lines.append(f"  {self.interpretation}")
        return "\n".join(lines)


def calc_osmolality(
    na: float,
    glucose: float,
    bun: float,
    ethanol: float = 0.0,
    measured: Optional[float] = None,
) -> OsmolalityResult:
    """Calculate serum osmolality and osmolal gap.

    Calculated = 2*Na + glucose/18 + BUN/2.8 + ethanol/4.6.
    Normal 275-295 mOsm/kg. Osmolal gap = measured - calculated (normal <10).

    Args:
        na: sodium in mEq/L
        glucose: glucose in mg/dL
        bun: blood urea nitrogen in mg/dL
        ethanol: ethanol level in mg/dL (default 0)
        measured: measured osmolality in mOsm/kg (optional, for gap calculation)
    """
    calc_osm = 2.0 * na + glucose / 18.0 + bun / 2.8
    if ethanol > 0:
        calc_osm += ethanol / 4.6

    gap = None
    if measured is not None:
        gap = measured - calc_osm

    parts = []
    if calc_osm < 275:
        parts.append("Hypo-osmolar — consider SIADH, water intoxication, or hyponatremia")
    elif calc_osm > 295:
        parts.append("Hyperosmolar — consider hyperglycemia, uremia, dehydration, or toxic ingestion")
    else:
        parts.append("Osmolality within normal range (275-295 mOsm/kg)")

    if gap is not None:
        if gap > 10:
            parts.append(f"Elevated osmolal gap ({gap:.0f}) — consider toxic alcohols (methanol, ethylene glycol, isopropanol)")
        elif gap < -10:
            parts.append(f"Negative osmolal gap ({gap:.0f}) — possible lab error or artifact")
        else:
            parts.append("Osmolal gap within normal range (<10)")

    return OsmolalityResult(
        na=na,
        glucose=glucose,
        bun=bun,
        ethanol=ethanol,
        calculated_osm=calc_osm,
        measured_osm=measured,
        osmolal_gap=gap,
        interpretation="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# 11. Winter's Formula
# ---------------------------------------------------------------------------

@dataclass
class WintersResult:
    hco3: float
    expected_paco2_low: float
    expected_paco2_high: float
    expected_paco2_mid: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Winter's Formula — Respiratory Compensation\n"
            f"  HCO3: {self.hco3:.0f} mEq/L\n"
            f"  Expected PaCO2: {self.expected_paco2_mid:.0f} mmHg "
            f"(range {self.expected_paco2_low:.0f}-{self.expected_paco2_high:.0f})\n"
            f"  {self.interpretation}"
        )


def calc_winters_formula(
    hco3: float,
    actual_paco2: Optional[float] = None,
) -> WintersResult:
    """Calculate expected PaCO2 for metabolic acidosis using Winter's formula.

    Expected PaCO2 = 1.5 x HCO3 + 8 +/- 2.
    Used to assess adequacy of respiratory compensation in metabolic acidosis.

    Args:
        hco3: serum bicarbonate in mEq/L
        actual_paco2: actual measured PaCO2 for comparison (optional)
    """
    mid = 1.5 * hco3 + 8.0
    low = mid - 2.0
    high = mid + 2.0

    if actual_paco2 is not None:
        if actual_paco2 < low:
            interp = (
                f"Actual PaCO2 ({actual_paco2:.0f}) below expected range — "
                "concurrent respiratory alkalosis (superimposed primary process)"
            )
        elif actual_paco2 > high:
            interp = (
                f"Actual PaCO2 ({actual_paco2:.0f}) above expected range — "
                "inadequate respiratory compensation or concurrent respiratory acidosis"
            )
        else:
            interp = (
                f"Actual PaCO2 ({actual_paco2:.0f}) within expected range — "
                "appropriate respiratory compensation for metabolic acidosis"
            )
    else:
        interp = (
            f"Compare measured PaCO2 to range {low:.0f}-{high:.0f}: "
            "below = concomitant respiratory alkalosis, above = respiratory acidosis, "
            "within = appropriate compensation"
        )

    return WintersResult(
        hco3=hco3,
        expected_paco2_low=low,
        expected_paco2_high=high,
        expected_paco2_mid=mid,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 12. Electrolyte-Free Water Clearance
# ---------------------------------------------------------------------------

@dataclass
class FreeWaterResult:
    urine_osm: float
    serum_osm: float
    urine_volume_ml_per_hr: float
    free_water_clearance_ml_per_hr: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"Electrolyte-Free Water Clearance\n"
            f"  Urine Osm: {self.urine_osm:.0f} mOsm/kg, Serum Osm: {self.serum_osm:.0f} mOsm/kg\n"
            f"  Urine volume: {self.urine_volume_ml_per_hr:.1f} mL/hr\n"
            f"  CH2O: {self.free_water_clearance_ml_per_hr:.1f} mL/hr\n"
            f"  {self.interpretation}"
        )


def calc_free_water_clearance(
    urine_osm: float,
    serum_osm: float,
    urine_volume_ml_per_hr: float,
) -> FreeWaterResult:
    """Calculate electrolyte-free water clearance.

    CH2O = V x (1 - Uosm/Sosm).
    Positive = net free water excretion (diluting).
    Negative = net free water reabsorption (concentrating).

    Args:
        urine_osm: urine osmolality in mOsm/kg
        serum_osm: serum osmolality in mOsm/kg
        urine_volume_ml_per_hr: urine output in mL/hr
    """
    if serum_osm <= 0:
        raise ValueError("Serum osmolality must be positive")

    ch2o = urine_volume_ml_per_hr * (1.0 - urine_osm / serum_osm)

    if ch2o > 0:
        interp = (
            f"Positive CH2O ({ch2o:.1f} mL/hr) — kidney excreting free water "
            "(dilute urine, appropriate in hyponatremia or water excess)"
        )
    elif ch2o < -50:
        interp = (
            f"Strongly negative CH2O ({ch2o:.1f} mL/hr) — kidney retaining free water "
            "(concentrated urine, consider SIADH, volume depletion, or high ADH state)"
        )
    elif ch2o < 0:
        interp = (
            f"Negative CH2O ({ch2o:.1f} mL/hr) — kidney retaining free water "
            "(concentrated urine relative to serum)"
        )
    else:
        interp = "CH2O = 0 — urine isotonic to serum"

    return FreeWaterResult(
        urine_osm=urine_osm,
        serum_osm=serum_osm,
        urine_volume_ml_per_hr=urine_volume_ml_per_hr,
        free_water_clearance_ml_per_hr=ch2o,
        interpretation=interp,
    )
