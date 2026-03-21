"""Hematology and pathophysiology calculations.

Covers oxygen-hemoglobin dissociation, oxygen content, anemia classification,
reticulocyte correction, coagulation pattern analysis, DIC scoring,
transfusion triggers, and shock classification.

Pure math — no external dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional
import math


# ---------------------------------------------------------------------------
# 1. Oxygen-hemoglobin dissociation
# ---------------------------------------------------------------------------

@dataclass
class OxyHbResult:
    """Result of oxygen-hemoglobin dissociation calculation."""
    po2: float
    so2_pct: float
    p50: float
    p50_normal: float
    shift_direction: str  # "left", "right", or "normal"
    shift_factors: List[str]
    clinical_interpretation: str

    def __str__(self) -> str:
        lines = [
            "=== Oxygen-Hemoglobin Dissociation ===",
            f"PO2:              {self.po2:.1f} mmHg",
            f"SO2:              {self.so2_pct:.1f}%",
            f"P50:              {self.p50:.1f} mmHg (normal {self.p50_normal:.1f})",
            f"Curve shift:      {self.shift_direction}",
        ]
        if self.shift_factors:
            lines.append(f"Shift factors:    {', '.join(self.shift_factors)}")
        lines.append(f"Interpretation:   {self.clinical_interpretation}")
        return "\n".join(lines)


def calc_oxygen_hemoglobin(
    po2: float,
    ph: float = 7.4,
    pco2: float = 40.0,
    temp_c: float = 37.0,
    dpg: float = 1.0,
) -> OxyHbResult:
    """Calculate oxygen-hemoglobin saturation via the Hill equation.

    Uses Bohr effect (pH, pCO2), temperature, and 2,3-DPG corrections
    to shift P50 from its normal value of 26.6 mmHg.

    Args:
        po2: Partial pressure of oxygen (mmHg).
        ph: Arterial pH (default 7.4).
        pco2: Partial pressure of CO2 (mmHg, default 40).
        temp_c: Temperature in Celsius (default 37).
        dpg: 2,3-DPG factor (1.0 = normal; >1 = elevated).

    Returns:
        OxyHbResult with saturation, shifted P50, and clinical interpretation.
    """
    P50_NORMAL = 26.6
    HILL_N = 2.7

    # Bohr / temperature shift exponent
    exponent = 0.48 * (7.4 - ph) + 0.024 * (pco2 - 40.0) + 0.013 * (temp_c - 37.0)
    p50_shifted = P50_NORMAL * (10.0 ** exponent) * dpg

    # Hill equation
    so2 = (po2 ** HILL_N) / (p50_shifted ** HILL_N + po2 ** HILL_N)
    so2_pct = so2 * 100.0

    # Determine shift direction and contributing factors
    shift_factors: List[str] = []
    if ph < 7.35:
        shift_factors.append("acidosis (right shift)")
    elif ph > 7.45:
        shift_factors.append("alkalosis (left shift)")
    if pco2 > 45:
        shift_factors.append("hypercapnia (right shift)")
    elif pco2 < 35:
        shift_factors.append("hypocapnia (left shift)")
    if temp_c > 38.0:
        shift_factors.append("fever (right shift)")
    elif temp_c < 36.0:
        shift_factors.append("hypothermia (left shift)")
    if dpg > 1.1:
        shift_factors.append("elevated 2,3-DPG (right shift)")
    elif dpg < 0.9:
        shift_factors.append("low 2,3-DPG (left shift)")

    delta = p50_shifted - P50_NORMAL
    if abs(delta) < 0.5:
        shift_direction = "normal"
    elif delta > 0:
        shift_direction = "right"
    else:
        shift_direction = "left"

    # Clinical interpretation
    if so2_pct >= 95:
        interp = "Normal oxygenation"
    elif so2_pct >= 90:
        interp = "Mild hypoxemia"
    elif so2_pct >= 80:
        interp = "Moderate hypoxemia — supplemental O2 indicated"
    elif so2_pct >= 60:
        interp = "Severe hypoxemia — urgent intervention required"
    else:
        interp = "Critical hypoxemia — life-threatening"

    if shift_direction == "right":
        interp += "; right-shifted curve favors O2 unloading to tissues"
    elif shift_direction == "left":
        interp += "; left-shifted curve impairs O2 unloading to tissues"

    return OxyHbResult(
        po2=po2,
        so2_pct=so2_pct,
        p50=p50_shifted,
        p50_normal=P50_NORMAL,
        shift_direction=shift_direction,
        shift_factors=shift_factors,
        clinical_interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 2. Oxygen content
# ---------------------------------------------------------------------------

@dataclass
class O2ContentResult:
    """Result of arterial oxygen content calculation."""
    cao2: float  # mL O2/dL
    bound_o2: float  # mL O2/dL (Hgb-bound)
    dissolved_o2: float  # mL O2/dL
    hgb: float
    sao2_pct: float
    pao2: float
    clinical_interpretation: str

    def __str__(self) -> str:
        return "\n".join([
            "=== Arterial Oxygen Content ===",
            f"CaO2 (total):     {self.cao2:.1f} mL O2/dL",
            f"  Bound (Hgb):    {self.bound_o2:.1f} mL O2/dL ({self.bound_o2 / self.cao2 * 100:.1f}%)" if self.cao2 > 0 else f"  Bound (Hgb):    {self.bound_o2:.1f} mL O2/dL",
            f"  Dissolved:      {self.dissolved_o2:.2f} mL O2/dL",
            f"Hgb:              {self.hgb:.1f} g/dL",
            f"SaO2:             {self.sao2_pct:.1f}%",
            f"PaO2:             {self.pao2:.1f} mmHg",
            f"Interpretation:   {self.clinical_interpretation}",
        ])


def calc_oxygen_content(
    hgb: float,
    sao2: float,
    pao2: float,
) -> O2ContentResult:
    """Calculate arterial oxygen content (CaO2).

    CaO2 = (Hgb x 1.34 x SaO2/100) + (0.003 x PaO2)

    Args:
        hgb: Hemoglobin in g/dL.
        sao2: Arterial O2 saturation in percent (0-100).
        pao2: Partial pressure of O2 in mmHg.

    Returns:
        O2ContentResult with total, bound, and dissolved oxygen.
    """
    bound_o2 = hgb * 1.34 * (sao2 / 100.0)
    dissolved_o2 = 0.003 * pao2
    cao2 = bound_o2 + dissolved_o2

    if cao2 >= 18:
        interp = "Normal oxygen content"
    elif cao2 >= 15:
        interp = "Mildly reduced oxygen content"
    elif cao2 >= 12:
        interp = "Moderately reduced — consider transfusion if symptomatic"
    else:
        interp = "Severely reduced oxygen content — transfusion likely indicated"

    if hgb < 7.0:
        interp += "; severe anemia contributing to low CaO2"
    elif hgb < 10.0:
        interp += "; anemia contributing to reduced CaO2"

    return O2ContentResult(
        cao2=cao2,
        bound_o2=bound_o2,
        dissolved_o2=dissolved_o2,
        hgb=hgb,
        sao2_pct=sao2,
        pao2=pao2,
        clinical_interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 3. Anemia classification
# ---------------------------------------------------------------------------

@dataclass
class AnemiaClassResult:
    """Result of anemia classification."""
    is_anemic: bool
    hgb: float
    mcv: float
    mcv_class: str  # "microcytic", "normocytic", "macrocytic"
    reticulocyte_pct: float
    retic_index: float
    marrow_response: str  # "adequate" or "inadequate"
    differential: List[str]
    clinical_interpretation: str

    def __str__(self) -> str:
        lines = [
            "=== Anemia Classification ===",
            f"Anemic:           {'Yes' if self.is_anemic else 'No'}",
            f"Hgb:              {self.hgb:.1f} g/dL",
            f"MCV:              {self.mcv:.0f} fL ({self.mcv_class})",
            f"Reticulocyte:     {self.reticulocyte_pct:.1f}%",
            f"Retic index (RI): {self.retic_index:.2f}%",
            f"Marrow response:  {self.marrow_response}",
            f"Differential:     {', '.join(self.differential)}",
            f"Interpretation:   {self.clinical_interpretation}",
        ]
        return "\n".join(lines)


def _maturation_factor(hct: float) -> float:
    """Return maturation correction factor for reticulocyte index.

    Hct 45% -> 1.0, 35% -> 1.5, 25% -> 2.0, 15% -> 2.5.
    Linear interpolation between these points.
    """
    # Piecewise linear: factor = 1.0 + (45 - hct) / 20
    # Clamp to reasonable range
    factor = 1.0 + (45.0 - hct) / 20.0
    return max(1.0, min(3.0, factor))


def classify_anemia(
    hgb: float,
    mcv: float,
    reticulocyte_pct: float,
    retic_index: float = None,
    is_female: bool = False,
) -> AnemiaClassResult:
    """Classify anemia by MCV and reticulocyte response.

    Args:
        hgb: Hemoglobin in g/dL.
        mcv: Mean corpuscular volume in fL.
        reticulocyte_pct: Reticulocyte percentage.
        retic_index: Pre-calculated reticulocyte production index.
            If None, estimated from Hgb assuming normal Hct relationship.
        is_female: Use female reference range (Hgb <12 vs <13).

    Returns:
        AnemiaClassResult with MCV class, RI, and differential diagnosis.
    """
    anemia_cutoff = 12.0 if is_female else 13.0
    is_anemic = hgb < anemia_cutoff

    # MCV classification
    if mcv < 80:
        mcv_class = "microcytic"
    elif mcv <= 100:
        mcv_class = "normocytic"
    else:
        mcv_class = "macrocytic"

    # Reticulocyte index estimation if not provided
    if retic_index is None:
        # Estimate Hct from Hgb (Hct ~ 3 x Hgb)
        est_hct = hgb * 3.0
        corrected = reticulocyte_pct * (est_hct / 45.0)
        mat_factor = _maturation_factor(est_hct)
        retic_index = corrected / mat_factor
    ri = retic_index

    if ri > 2.0:
        marrow_response = "adequate"
    else:
        marrow_response = "inadequate"

    # Build differential
    ddx: List[str] = []
    if mcv_class == "microcytic":
        if marrow_response == "inadequate":
            ddx = [
                "Iron deficiency anemia (most common)",
                "Anemia of chronic disease",
                "Thalassemia trait",
                "Sideroblastic anemia",
                "Lead poisoning",
            ]
        else:
            ddx = [
                "Acute blood loss with iron depletion",
                "Thalassemia with hemolysis",
            ]
    elif mcv_class == "normocytic":
        if marrow_response == "inadequate":
            ddx = [
                "Anemia of chronic disease (most common)",
                "Chronic kidney disease (low EPO)",
                "Early iron deficiency",
                "Bone marrow infiltration/failure",
                "Mixed deficiency (iron + B12/folate)",
            ]
        else:
            ddx = [
                "Acute blood loss",
                "Hemolytic anemia (autoimmune, sickle cell, G6PD, etc.)",
                "Splenic sequestration",
            ]
    else:  # macrocytic
        if marrow_response == "inadequate":
            ddx = [
                "Vitamin B12 deficiency",
                "Folate deficiency",
                "Myelodysplastic syndrome",
                "Hypothyroidism",
                "Drug-induced (methotrexate, hydroxyurea, AZT)",
                "Liver disease / alcohol use",
            ]
        else:
            ddx = [
                "Hemolytic anemia with brisk reticulocytosis",
                "Acute blood loss with reticulocyte response",
            ]

    # Interpretation
    if not is_anemic:
        interp = "No anemia by Hgb criteria"
    else:
        severity = "mild" if hgb >= 10 else ("moderate" if hgb >= 8 else "severe")
        interp = (
            f"{severity.capitalize()} {mcv_class} anemia with "
            f"{marrow_response} marrow response (RI={ri:.2f}%)"
        )

    return AnemiaClassResult(
        is_anemic=is_anemic,
        hgb=hgb,
        mcv=mcv,
        mcv_class=mcv_class,
        reticulocyte_pct=reticulocyte_pct,
        retic_index=ri,
        marrow_response=marrow_response,
        differential=ddx,
        clinical_interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 4. Corrected reticulocyte count
# ---------------------------------------------------------------------------

@dataclass
class ReticulocyteResult:
    """Result of corrected reticulocyte calculation."""
    raw_retic_pct: float
    hematocrit: float
    corrected_retic: float
    maturation_factor: float
    rpi: float  # reticulocyte production index
    interpretation: str

    def __str__(self) -> str:
        return "\n".join([
            "=== Corrected Reticulocyte Count ===",
            f"Raw retic:        {self.raw_retic_pct:.1f}%",
            f"Hematocrit:       {self.hematocrit:.1f}%",
            f"Corrected retic:  {self.corrected_retic:.2f}%",
            f"Maturation factor:{self.maturation_factor:.2f}",
            f"RPI:              {self.rpi:.2f}%",
            f"Interpretation:   {self.interpretation}",
        ])


def calc_corrected_reticulocyte(
    reticulocyte_pct: float,
    hematocrit: float,
) -> ReticulocyteResult:
    """Calculate corrected reticulocyte count and reticulocyte production index.

    Corrected retic = retic% x (Hct / 45).
    RPI = corrected_retic / maturation_factor.

    Maturation factors: Hct 45%=1.0, 35%=1.5, 25%=2.0, 15%=2.5.

    Args:
        reticulocyte_pct: Raw reticulocyte percentage.
        hematocrit: Hematocrit in percent.

    Returns:
        ReticulocyteResult with corrected count, RPI, and interpretation.
    """
    corrected = reticulocyte_pct * (hematocrit / 45.0)
    mat_factor = _maturation_factor(hematocrit)
    rpi = corrected / mat_factor

    if rpi > 3.0:
        interp = "Brisk marrow response — hemolysis or acute blood loss likely"
    elif rpi > 2.0:
        interp = "Adequate marrow response — suggests peripheral destruction or loss"
    elif rpi > 1.0:
        interp = "Borderline response — may be early recovery or partial marrow suppression"
    else:
        interp = "Inadequate marrow response — production problem (nutritional, marrow failure, etc.)"

    return ReticulocyteResult(
        raw_retic_pct=reticulocyte_pct,
        hematocrit=hematocrit,
        corrected_retic=corrected,
        maturation_factor=mat_factor,
        rpi=rpi,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 5. Coagulation pattern analysis
# ---------------------------------------------------------------------------

@dataclass
class CoagulationResult:
    """Result of coagulation pattern analysis."""
    pt_sec: float
    inr: float
    aptt_sec: float
    fibrinogen: Optional[float]
    d_dimer: Optional[float]
    pt_status: str  # "normal" or "prolonged"
    aptt_status: str  # "normal" or "prolonged"
    pattern: str
    likely_causes: List[str]
    dic_concern: bool
    clinical_interpretation: str

    def __str__(self) -> str:
        lines = [
            "=== Coagulation Analysis ===",
            f"PT:               {self.pt_sec:.1f} s ({self.pt_status})",
            f"INR:              {self.inr:.2f}",
            f"aPTT:             {self.aptt_sec:.1f} s ({self.aptt_status})",
        ]
        if self.fibrinogen is not None:
            lines.append(f"Fibrinogen:       {self.fibrinogen:.0f} mg/dL")
        if self.d_dimer is not None:
            lines.append(f"D-dimer:          {self.d_dimer:.2f} ug/mL")
        lines.extend([
            f"Pattern:          {self.pattern}",
            f"Likely causes:    {', '.join(self.likely_causes)}",
        ])
        if self.dic_concern:
            lines.append("DIC concern:      YES")
        lines.append(f"Interpretation:   {self.clinical_interpretation}")
        return "\n".join(lines)


def check_coagulation(
    pt_sec: float,
    inr: float,
    aptt_sec: float,
    fibrinogen: float = None,
    d_dimer: float = None,
) -> CoagulationResult:
    """Analyze coagulation pattern from PT, INR, aPTT, and optional labs.

    Normal ranges: PT 11-13.5 s, INR ~1.0, aPTT 25-35 s.

    Args:
        pt_sec: Prothrombin time in seconds.
        inr: International normalized ratio.
        aptt_sec: Activated partial thromboplastin time in seconds.
        fibrinogen: Fibrinogen in mg/dL (optional).
        d_dimer: D-dimer in ug/mL (optional).

    Returns:
        CoagulationResult with pattern analysis and differential.
    """
    pt_prolonged = pt_sec > 13.5
    aptt_prolonged = aptt_sec > 35.0
    pt_status = "prolonged" if pt_prolonged else "normal"
    aptt_status = "prolonged" if aptt_prolonged else "normal"

    causes: List[str] = []
    dic_concern = False

    if pt_prolonged and not aptt_prolonged:
        pattern = "Isolated PT prolongation (extrinsic pathway)"
        causes = [
            "Factor VII deficiency",
            "Early warfarin therapy",
            "Early vitamin K deficiency",
            "Mild liver disease",
        ]
    elif not pt_prolonged and aptt_prolonged:
        pattern = "Isolated aPTT prolongation (intrinsic pathway)"
        causes = [
            "Factor VIII deficiency (Hemophilia A)",
            "Factor IX deficiency (Hemophilia B)",
            "Factor XI deficiency",
            "Factor XII deficiency (no bleeding risk)",
            "Heparin therapy",
            "Lupus anticoagulant (paradoxical thrombosis risk)",
            "von Willebrand disease",
        ]
    elif pt_prolonged and aptt_prolonged:
        pattern = "Both PT and aPTT prolonged (common pathway or global)"
        causes = [
            "Common pathway factor deficiency (X, V, II, fibrinogen)",
            "DIC (disseminated intravascular coagulation)",
            "Severe liver disease",
            "Warfarin supratherapeutic",
            "Combined heparin + warfarin",
            "Massive transfusion (dilutional coagulopathy)",
            "Vitamin K deficiency (advanced)",
        ]
        # Check for DIC
        if fibrinogen is not None and d_dimer is not None:
            if fibrinogen < 200 and d_dimer > 0.5:
                dic_concern = True
                causes.insert(0, "DIC highly likely (low fibrinogen + elevated D-dimer)")
    else:
        pattern = "Normal coagulation screen"
        causes = ["No coagulation abnormality detected"]

    # Build interpretation
    interp_parts = [pattern]
    if inr > 3.0:
        interp_parts.append(f"INR {inr:.1f} — significantly supratherapeutic / high bleeding risk")
    elif inr > 2.0:
        interp_parts.append(f"INR {inr:.1f} — therapeutic for most indications")
    if dic_concern:
        interp_parts.append("DIC pattern present — check platelets, smear for schistocytes")

    return CoagulationResult(
        pt_sec=pt_sec,
        inr=inr,
        aptt_sec=aptt_sec,
        fibrinogen=fibrinogen,
        d_dimer=d_dimer,
        pt_status=pt_status,
        aptt_status=aptt_status,
        pattern=pattern,
        likely_causes=causes,
        dic_concern=dic_concern,
        clinical_interpretation="; ".join(interp_parts),
    )


# ---------------------------------------------------------------------------
# 6. DIC score (ISTH)
# ---------------------------------------------------------------------------

@dataclass
class DICScoreResult:
    """Result of ISTH DIC scoring."""
    total_score: int
    platelet_score: int
    d_dimer_score: int
    pt_score: int
    fibrinogen_score: int
    is_overt_dic: bool
    clinical_interpretation: str

    def __str__(self) -> str:
        return "\n".join([
            "=== ISTH DIC Score ===",
            f"Platelet score:   {self.platelet_score} (0/1/2)",
            f"D-dimer score:    {self.d_dimer_score} (0/2/3)",
            f"PT prolong score: {self.pt_score} (0/1/2)",
            f"Fibrinogen score: {self.fibrinogen_score} (0/1)",
            f"Total score:      {self.total_score} / 8",
            f"Overt DIC:        {'YES (>=5)' if self.is_overt_dic else 'No (<5)'}",
            f"Interpretation:   {self.clinical_interpretation}",
        ])


def calc_dic_score(
    platelets: int,
    d_dimer: str,
    pt_prolongation: float,
    fibrinogen: float,
) -> DICScoreResult:
    """Calculate ISTH overt DIC score.

    Args:
        platelets: Platelet count (x10^3/uL).
        d_dimer: D-dimer level as "no_increase", "moderate", or "strong".
        pt_prolongation: PT prolongation above normal in seconds.
        fibrinogen: Fibrinogen in g/L.

    Returns:
        DICScoreResult. Score >=5 is compatible with overt DIC.
    """
    # Platelets
    if platelets >= 100:
        plt_score = 0
    elif platelets >= 50:
        plt_score = 1
    else:
        plt_score = 2

    # D-dimer
    d_dimer_lower = d_dimer.lower().replace("-", "_").replace(" ", "_")
    if d_dimer_lower in ("no_increase", "none", "normal"):
        dd_score = 0
    elif d_dimer_lower in ("moderate", "mod"):
        dd_score = 2
    elif d_dimer_lower in ("strong", "high", "markedly_elevated"):
        dd_score = 3
    else:
        dd_score = 0  # default to no increase if unrecognized

    # PT prolongation
    if pt_prolongation < 3.0:
        pt_score = 0
    elif pt_prolongation <= 6.0:
        pt_score = 1
    else:
        pt_score = 2

    # Fibrinogen (in g/L)
    if fibrinogen >= 1.0:
        fib_score = 0
    else:
        fib_score = 1

    total = plt_score + dd_score + pt_score + fib_score
    is_overt = total >= 5

    if is_overt:
        interp = (
            f"Score {total}/8: Compatible with overt DIC. "
            "Repeat daily. Consider heparin if thrombosis predominant, "
            "or blood products if bleeding predominant."
        )
    elif total >= 3:
        interp = (
            f"Score {total}/8: Not overt DIC but suggestive. "
            "Repeat in 1-2 days. Monitor for progression."
        )
    else:
        interp = f"Score {total}/8: Not suggestive of DIC."

    return DICScoreResult(
        total_score=total,
        platelet_score=plt_score,
        d_dimer_score=dd_score,
        pt_score=pt_score,
        fibrinogen_score=fib_score,
        is_overt_dic=is_overt,
        clinical_interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 7. Transfusion trigger
# ---------------------------------------------------------------------------

@dataclass
class TransfusionResult:
    """Result of transfusion trigger assessment."""
    hgb: float
    threshold_used: float
    strategy: str  # "restrictive" or "liberal"
    should_transfuse: bool
    units_suggested: int
    expected_hgb_post: float
    rationale: str
    clinical_interpretation: str

    def __str__(self) -> str:
        return "\n".join([
            "=== Transfusion Assessment ===",
            f"Current Hgb:      {self.hgb:.1f} g/dL",
            f"Strategy:         {self.strategy}",
            f"Threshold:        {self.threshold_used:.0f} g/dL",
            f"Transfuse:        {'YES' if self.should_transfuse else 'No'}",
            f"Units suggested:  {self.units_suggested} pRBC",
            f"Expected post-Hgb:{self.expected_hgb_post:.1f} g/dL",
            f"Rationale:        {self.rationale}",
            f"Interpretation:   {self.clinical_interpretation}",
        ])


def calc_transfusion_trigger(
    hgb: float,
    symptoms: bool = False,
    cardiac_disease: bool = False,
    active_bleeding: bool = False,
) -> TransfusionResult:
    """Assess transfusion need based on hemoglobin and clinical context.

    Restrictive strategy (Hgb <7 or <8 for cardiac) is standard per TRICC,
    TRACS, and TITRe2 trials. Liberal strategy (Hgb <9-10) for active
    bleeding or symptomatic anemia.

    Each unit of pRBC raises Hgb by approximately 1 g/dL.

    Args:
        hgb: Current hemoglobin in g/dL.
        symptoms: Symptomatic anemia (dyspnea, tachycardia, syncope).
        cardiac_disease: Known cardiac disease or ACS.
        active_bleeding: Active hemorrhage.

    Returns:
        TransfusionResult with trigger assessment and dosing.
    """
    # Determine threshold and strategy
    if active_bleeding:
        threshold = 9.0
        strategy = "liberal"
        rationale = "Active bleeding — liberal threshold to maintain oxygen delivery"
    elif symptoms:
        threshold = 9.0
        strategy = "liberal"
        rationale = "Symptomatic anemia — liberal threshold for symptom relief"
    elif cardiac_disease:
        threshold = 8.0
        strategy = "restrictive"
        rationale = "Cardiac disease — restrictive threshold per FOCUS/TITRe2 trials"
    else:
        threshold = 7.0
        strategy = "restrictive"
        rationale = "Hemodynamically stable — restrictive threshold per TRICC trial"

    should_transfuse = hgb < threshold

    # Calculate units needed (target ~1 g/dL above threshold, 1 unit per g/dL)
    if should_transfuse:
        deficit = threshold - hgb + 1.0  # aim 1 above threshold
        units = max(1, min(4, math.ceil(deficit)))
    else:
        units = 0

    expected_post = hgb + units * 1.0

    if should_transfuse:
        if hgb < 5.0:
            interp = "Critical anemia — urgent transfusion required"
        elif hgb < 7.0:
            interp = "Below universal threshold — transfusion indicated"
        else:
            interp = f"Below {strategy} threshold ({threshold:.0f} g/dL) — transfusion indicated"
    else:
        interp = (
            f"Above {strategy} threshold ({threshold:.0f} g/dL) — "
            "transfusion not indicated; reassess if clinical status changes"
        )

    return TransfusionResult(
        hgb=hgb,
        threshold_used=threshold,
        strategy=strategy,
        should_transfuse=should_transfuse,
        units_suggested=units,
        expected_hgb_post=expected_post,
        rationale=rationale,
        clinical_interpretation=interp,
    )


# ---------------------------------------------------------------------------
# 8. Shock classification
# ---------------------------------------------------------------------------

@dataclass
class ShockClassResult:
    """Result of shock classification."""
    map_mmhg: float
    heart_rate: float
    cvp: Optional[float]
    cardiac_output: Optional[float]
    svr: Optional[float]
    lactate: Optional[float]
    is_shock: bool
    shock_type: str  # "distributive", "cardiogenic", "hypovolemic", "obstructive", "undifferentiated", "none"
    confidence: str  # "high", "moderate", "low"
    distinguishing_features: List[str]
    lactate_interpretation: str
    clinical_interpretation: str

    def __str__(self) -> str:
        lines = [
            "=== Shock Classification ===",
            f"MAP:              {self.map_mmhg:.0f} mmHg",
            f"Heart rate:       {self.heart_rate:.0f} bpm",
        ]
        if self.cvp is not None:
            lines.append(f"CVP:              {self.cvp:.0f} mmHg")
        if self.cardiac_output is not None:
            lines.append(f"Cardiac output:   {self.cardiac_output:.1f} L/min")
        if self.svr is not None:
            lines.append(f"SVR:              {self.svr:.0f} dynes*s/cm5")
        if self.lactate is not None:
            lines.append(f"Lactate:          {self.lactate:.1f} mmol/L")
        lines.extend([
            f"Shock:            {'YES' if self.is_shock else 'No'}",
            f"Type:             {self.shock_type}",
            f"Confidence:       {self.confidence}",
        ])
        if self.distinguishing_features:
            lines.append(f"Features:         {', '.join(self.distinguishing_features)}")
        if self.lactate is not None:
            lines.append(f"Lactate status:   {self.lactate_interpretation}")
        lines.append(f"Interpretation:   {self.clinical_interpretation}")
        return "\n".join(lines)


def classify_shock(
    map_mmhg: float,
    heart_rate: float,
    cvp: float = None,
    cardiac_output: float = None,
    svr: float = None,
    lactate: float = None,
) -> ShockClassResult:
    """Classify shock type from hemodynamic parameters.

    Normal ranges: MAP 70-105, HR 60-100, CVP 2-8, CO 4-8 L/min,
    SVR 800-1200 dynes*s/cm5, Lactate <2 mmol/L.

    Types:
    - Distributive: low SVR, high/normal CO (sepsis, anaphylaxis, neurogenic)
    - Cardiogenic: high SVR, low CO, high CVP
    - Hypovolemic: high SVR, low CO, low CVP
    - Obstructive: high SVR, low CO, high CVP (PE, tamponade, tension pneumo)

    Args:
        map_mmhg: Mean arterial pressure in mmHg.
        heart_rate: Heart rate in bpm.
        cvp: Central venous pressure in mmHg (optional).
        cardiac_output: Cardiac output in L/min (optional).
        svr: Systemic vascular resistance in dynes*s/cm5 (optional).
        lactate: Serum lactate in mmol/L (optional).

    Returns:
        ShockClassResult with type classification and interpretation.
    """
    features: List[str] = []

    # Determine if shock is present
    is_shock = map_mmhg < 65 or (
        map_mmhg < 70 and heart_rate > 100
    )
    if lactate is not None and lactate > 4.0:
        is_shock = True

    # Lactate interpretation
    if lactate is None:
        lactate_interp = "Not measured"
    elif lactate < 2.0:
        lactate_interp = "Normal — no tissue hypoperfusion"
    elif lactate < 4.0:
        lactate_interp = "Elevated — tissue hypoperfusion present"
    else:
        lactate_interp = "Severely elevated (>4) — severe shock / end-organ hypoperfusion"

    # Classify based on available hemodynamic data
    low_co = cardiac_output is not None and cardiac_output < 4.0
    high_co = cardiac_output is not None and cardiac_output >= 4.0
    low_svr = svr is not None and svr < 800
    high_svr = svr is not None and svr > 1200
    low_cvp = cvp is not None and cvp < 4
    high_cvp = cvp is not None and cvp > 10

    shock_type = "undifferentiated"
    confidence = "low"

    if not is_shock:
        shock_type = "none"
        confidence = "high"
    elif svr is not None and cardiac_output is not None:
        # Full hemodynamic profile available
        confidence = "high"
        if low_svr and (high_co or cardiac_output >= 3.5):
            shock_type = "distributive"
            features.append("Low SVR with preserved/elevated CO")
            features.append("Consider: sepsis, anaphylaxis, neurogenic, adrenal crisis")
        elif high_svr and low_co and high_cvp:
            # Could be cardiogenic or obstructive
            shock_type = "cardiogenic"
            features.append("High SVR, low CO, elevated CVP")
            features.append("Consider: MI, myocarditis, valvular emergency, arrhythmia")
            features.append("Rule out obstructive causes (PE, tamponade) with echo/CT")
            confidence = "moderate"  # can't distinguish from obstructive without imaging
        elif high_svr and low_co and low_cvp:
            shock_type = "hypovolemic"
            features.append("High SVR, low CO, low CVP")
            features.append("Consider: hemorrhage, dehydration, third-spacing, burns")
        elif high_svr and low_co:
            if cvp is None:
                shock_type = "cardiogenic or hypovolemic"
                features.append("High SVR, low CO — need CVP to differentiate")
                confidence = "moderate"
            else:
                shock_type = "obstructive"
                features.append("High SVR, low CO")
                features.append("Consider: PE, cardiac tamponade, tension pneumothorax")
        else:
            shock_type = "undifferentiated"
            features.append("Mixed hemodynamic profile")
            confidence = "low"
    elif cardiac_output is not None:
        confidence = "moderate"
        if low_co:
            if low_cvp:
                shock_type = "hypovolemic"
                features.append("Low CO with low CVP")
            elif high_cvp:
                shock_type = "cardiogenic"
                features.append("Low CO with elevated CVP")
            else:
                shock_type = "cardiogenic or hypovolemic"
                features.append("Low CO — need CVP/SVR to differentiate")
        elif high_co:
            shock_type = "distributive"
            features.append("High CO pattern — likely distributive")
    elif svr is not None:
        confidence = "moderate"
        if low_svr:
            shock_type = "distributive"
            features.append("Low SVR pattern")
        elif high_svr:
            shock_type = "non-distributive (cardiogenic/hypovolemic/obstructive)"
            features.append("High SVR — need CO and CVP to subclassify")
    else:
        # Only MAP, HR, +/- lactate available
        confidence = "low"
        if heart_rate > 120 and map_mmhg < 60:
            shock_type = "undifferentiated"
            features.append("Severe hemodynamic compromise — obtain full profile (CO, SVR, CVP)")
        else:
            shock_type = "undifferentiated"
            features.append("Insufficient data for classification — need CO, SVR, CVP")

    # Add tachycardia flag
    if heart_rate > 100:
        features.append(f"Tachycardia (HR {heart_rate:.0f})")

    # Build interpretation
    if not is_shock:
        interp = "Hemodynamic parameters within acceptable range — no shock"
        if lactate is not None and lactate > 2.0:
            interp += "; however, elevated lactate warrants monitoring"
    else:
        interp = f"{shock_type.capitalize()} shock"
        if lactate is not None:
            if lactate > 4.0:
                interp += " with severe tissue hypoperfusion (lactate >4)"
            elif lactate > 2.0:
                interp += " with tissue hypoperfusion"
        interp += f" (MAP {map_mmhg:.0f} mmHg)"

    return ShockClassResult(
        map_mmhg=map_mmhg,
        heart_rate=heart_rate,
        cvp=cvp,
        cardiac_output=cardiac_output,
        svr=svr,
        lactate=lactate,
        is_shock=is_shock,
        shock_type=shock_type,
        confidence=confidence,
        distinguishing_features=features,
        lactate_interpretation=lactate_interp,
        clinical_interpretation=interp,
    )
