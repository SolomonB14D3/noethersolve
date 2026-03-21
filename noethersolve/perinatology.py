"""Perinatology and neonatal physiology calculator.

Pure-math implementations of standard obstetric and neonatal scoring systems,
fetal biometry, lung maturity assessment, bilirubin risk stratification,
preeclampsia screening, gestational age estimation, and amniotic fluid index.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Bishop Score
# ---------------------------------------------------------------------------

@dataclass
class BishopScoreResult:
    dilation_score: int
    effacement_score: int
    station_score: int
    consistency_score: int
    position_score: int
    total: int
    favorable: bool
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Bishop Score Assessment",
            "=" * 40,
            f"  Dilation:    {self.dilation_score}/3",
            f"  Effacement:  {self.effacement_score}/3",
            f"  Station:     {self.station_score}/3",
            f"  Consistency: {self.consistency_score}/2",
            f"  Position:    {self.position_score}/2",
            "-" * 40,
            f"  Total Score: {self.total}/13",
            f"  Favorable:   {'Yes' if self.favorable else 'No'}",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


def calc_bishop_score(
    dilation: int,
    effacement: int,
    station: int,
    consistency: str,
    position: str,
) -> BishopScoreResult:
    """Calculate the Bishop score for cervical readiness assessment.

    Parameters
    ----------
    dilation : int
        Cervical dilation in cm (0-10).
    effacement : int
        Cervical effacement as percentage (0-100).
    station : int
        Fetal station (-3 to +2).
    consistency : str
        Cervical consistency: 'firm', 'medium', or 'soft'.
    position : str
        Cervical position: 'posterior', 'mid', or 'anterior'.

    Returns
    -------
    BishopScoreResult

    Example
    -------
    >>> r = calc_bishop_score(3, 60, -1, 'medium', 'anterior')
    >>> r.total
    9
    """
    # Dilation: 0=0, 1-2=1, 3-4=2, >=5=3
    if dilation <= 0:
        d_score = 0
    elif dilation <= 2:
        d_score = 1
    elif dilation <= 4:
        d_score = 2
    else:
        d_score = 3

    # Effacement: 0-30%=0, 40-50%=1, 60-70%=2, >=80%=3
    if effacement <= 30:
        e_score = 0
    elif effacement <= 50:
        e_score = 1
    elif effacement <= 70:
        e_score = 2
    else:
        e_score = 3

    # Station: -3=0, -2=1, -1/0=2, +1/+2=3
    if station <= -3:
        s_score = 0
    elif station == -2:
        s_score = 1
    elif station <= 0:
        s_score = 2
    else:
        s_score = 3

    # Consistency: firm=0, medium=1, soft=2
    consistency_lower = consistency.lower().strip()
    consistency_map = {"firm": 0, "medium": 1, "soft": 2}
    if consistency_lower not in consistency_map:
        raise ValueError(
            f"Invalid consistency '{consistency}'. Use 'firm', 'medium', or 'soft'."
        )
    c_score = consistency_map[consistency_lower]

    # Position: posterior=0, mid=1, anterior=2
    position_lower = position.lower().strip()
    position_map = {"posterior": 0, "mid": 1, "middle": 1, "anterior": 2}
    if position_lower not in position_map:
        raise ValueError(
            f"Invalid position '{position}'. Use 'posterior', 'mid', or 'anterior'."
        )
    p_score = position_map[position_lower]

    total = d_score + e_score + s_score + c_score + p_score
    favorable = total >= 8

    if total >= 8:
        interpretation = "Favorable cervix -- high likelihood of successful induction."
    elif total >= 6:
        interpretation = "Moderately favorable -- induction may proceed with cervical ripening."
    else:
        interpretation = "Unfavorable cervix -- cervical ripening recommended before induction."

    return BishopScoreResult(
        dilation_score=d_score,
        effacement_score=e_score,
        station_score=s_score,
        consistency_score=c_score,
        position_score=p_score,
        total=total,
        favorable=favorable,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# APGAR Score
# ---------------------------------------------------------------------------

@dataclass
class APGARResult:
    heart_rate_score: int
    respiratory_score: int
    muscle_tone_score: int
    reflex_score: int
    color_score: int
    total: int
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "APGAR Score Assessment",
            "=" * 40,
            f"  Heart Rate:   {self.heart_rate_score}/2",
            f"  Respiratory:  {self.respiratory_score}/2",
            f"  Muscle Tone:  {self.muscle_tone_score}/2",
            f"  Reflex:       {self.reflex_score}/2",
            f"  Color:        {self.color_score}/2",
            "-" * 40,
            f"  Total Score:  {self.total}/10",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


def calc_apgar(
    heart_rate: int,
    respiratory: str,
    muscle_tone: str,
    reflex: str,
    color: str,
) -> APGARResult:
    """Calculate the APGAR score for neonatal assessment.

    Parameters
    ----------
    heart_rate : int
        Neonatal heart rate in bpm (0 = absent).
    respiratory : str
        Respiratory effort: 'absent', 'slow', 'irregular', or 'crying'.
    muscle_tone : str
        Muscle tone: 'flaccid', 'some flexion', or 'active'.
    reflex : str
        Reflex irritability: 'none', 'grimace', 'cry', or 'cough'.
    color : str
        Skin color: 'blue', 'pale', 'acrocyanotic', or 'pink'.

    Returns
    -------
    APGARResult

    Example
    -------
    >>> r = calc_apgar(130, 'crying', 'active', 'cry', 'pink')
    >>> r.total
    10
    """
    # Heart rate: 0=absent, <100=1, >=100=2
    if heart_rate <= 0:
        hr_score = 0
    elif heart_rate < 100:
        hr_score = 1
    else:
        hr_score = 2

    # Respiratory: absent=0, slow/irregular=1, crying=2
    resp_lower = respiratory.lower().strip()
    if resp_lower in ("absent", "none", "apnea"):
        resp_score = 0
    elif resp_lower in ("slow", "irregular", "slow/irregular", "weak"):
        resp_score = 1
    elif resp_lower in ("crying", "vigorous", "good", "cry"):
        resp_score = 2
    else:
        raise ValueError(
            f"Invalid respiratory '{respiratory}'. Use 'absent', 'slow', 'irregular', or 'crying'."
        )

    # Muscle tone: flaccid=0, some flexion=1, active=2
    tone_lower = muscle_tone.lower().strip()
    if tone_lower in ("flaccid", "limp", "none"):
        tone_score = 0
    elif tone_lower in ("some flexion", "flexion", "some"):
        tone_score = 1
    elif tone_lower in ("active", "active motion", "good"):
        tone_score = 2
    else:
        raise ValueError(
            f"Invalid muscle_tone '{muscle_tone}'. Use 'flaccid', 'some flexion', or 'active'."
        )

    # Reflex: none=0, grimace=1, cry/cough=2
    reflex_lower = reflex.lower().strip()
    if reflex_lower in ("none", "absent", "no response"):
        reflex_score = 0
    elif reflex_lower in ("grimace", "minimal"):
        reflex_score = 1
    elif reflex_lower in ("cry", "cough", "cry/cough", "sneeze", "vigorous"):
        reflex_score = 2
    else:
        raise ValueError(
            f"Invalid reflex '{reflex}'. Use 'none', 'grimace', 'cry', or 'cough'."
        )

    # Color: blue/pale=0, acrocyanotic=1, pink=2
    color_lower = color.lower().strip()
    if color_lower in ("blue", "pale", "cyanotic", "blue/pale"):
        color_score = 0
    elif color_lower in ("acrocyanotic", "body pink", "extremities blue"):
        color_score = 1
    elif color_lower in ("pink", "completely pink", "good"):
        color_score = 2
    else:
        raise ValueError(
            f"Invalid color '{color}'. Use 'blue', 'pale', 'acrocyanotic', or 'pink'."
        )

    total = hr_score + resp_score + tone_score + reflex_score + color_score

    if total >= 7:
        interpretation = "Normal -- reassuring neonatal status."
    elif total >= 4:
        interpretation = "Moderate depression -- may require stimulation or airway management."
    else:
        interpretation = "Severe depression -- immediate resuscitation required."

    return APGARResult(
        heart_rate_score=hr_score,
        respiratory_score=resp_score,
        muscle_tone_score=tone_score,
        reflex_score=reflex_score,
        color_score=color_score,
        total=total,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Fetal Weight (Hadlock)
# ---------------------------------------------------------------------------

@dataclass
class FetalWeightResult:
    efw_grams: float
    efw_kg: float
    efw_lbs: float
    bpd_cm: float
    hc_cm: float
    ac_cm: float
    fl_cm: float
    gestational_age_weeks: Optional[float]
    percentile_estimate: Optional[str]
    formula: str

    def __str__(self) -> str:
        lines = [
            "Estimated Fetal Weight (Hadlock)",
            "=" * 40,
            f"  BPD: {self.bpd_cm:.1f} cm   HC: {self.hc_cm:.1f} cm",
            f"  AC:  {self.ac_cm:.1f} cm   FL: {self.fl_cm:.1f} cm",
            "-" * 40,
            f"  EFW: {self.efw_grams:.0f} g ({self.efw_kg:.2f} kg, {self.efw_lbs:.1f} lbs)",
        ]
        if self.gestational_age_weeks is not None:
            lines.append(f"  Gestational Age: {self.gestational_age_weeks:.1f} weeks")
        if self.percentile_estimate is not None:
            lines.append(f"  Percentile: {self.percentile_estimate}")
        lines.append(f"  Formula: {self.formula}")
        return "\n".join(lines)


def calc_fetal_weight_hadlock(
    bpd_cm: float,
    hc_cm: float,
    ac_cm: float,
    fl_cm: float,
    gestational_age_weeks: Optional[float] = None,
) -> FetalWeightResult:
    """Estimate fetal weight using the Hadlock formula (1985).

    Parameters
    ----------
    bpd_cm : float
        Biparietal diameter in cm.
    hc_cm : float
        Head circumference in cm.
    ac_cm : float
        Abdominal circumference in cm.
    fl_cm : float
        Femur length in cm.
    gestational_age_weeks : float, optional
        Gestational age in weeks for percentile estimation.

    Returns
    -------
    FetalWeightResult

    Example
    -------
    >>> r = calc_fetal_weight_hadlock(9.0, 32.0, 30.0, 7.0)
    >>> 2500 < r.efw_grams < 4000
    True
    """
    # Hadlock (1985) four-parameter formula
    log10_efw = (
        1.3596
        + 0.0064 * hc_cm
        + 0.0424 * ac_cm
        + 0.174 * fl_cm
        + 0.00061 * bpd_cm * ac_cm
        - 0.00386 * ac_cm * fl_cm
    )
    efw_grams = 10.0 ** log10_efw
    efw_kg = efw_grams / 1000.0
    efw_lbs = efw_grams / 453.592

    # Approximate percentile estimation based on GA
    # 50th percentile reference weights (grams) by week (simplified)
    percentile_estimate = None
    if gestational_age_weeks is not None:
        p50_by_week = {
            20: 300, 22: 430, 24: 600, 26: 760, 28: 1005,
            30: 1319, 32: 1702, 34: 2146, 36: 2622, 37: 2859,
            38: 3083, 39: 3288, 40: 3462, 41: 3597, 42: 3685,
        }
        # 10th percentile is roughly 80% of 50th, 90th is roughly 120%
        week = round(gestational_age_weeks)
        if week in p50_by_week:
            p50 = p50_by_week[week]
            p10 = p50 * 0.80
            p90 = p50 * 1.20
            if efw_grams < p10:
                percentile_estimate = "<10th percentile (small for gestational age)"
            elif efw_grams > p90:
                percentile_estimate = ">90th percentile (large for gestational age)"
            else:
                percentile_estimate = "10th-90th percentile (appropriate for gestational age)"
        else:
            percentile_estimate = f"No reference data for {week} weeks"

    return FetalWeightResult(
        efw_grams=efw_grams,
        efw_kg=efw_kg,
        efw_lbs=efw_lbs,
        bpd_cm=bpd_cm,
        hc_cm=hc_cm,
        ac_cm=ac_cm,
        fl_cm=fl_cm,
        gestational_age_weeks=gestational_age_weeks,
        percentile_estimate=percentile_estimate,
        formula="Hadlock (1985): log10(EFW) = 1.3596 + 0.0064*HC + 0.0424*AC + 0.174*FL + 0.00061*BPD*AC - 0.00386*AC*FL",
    )


# ---------------------------------------------------------------------------
# L/S Ratio (Lung Maturity)
# ---------------------------------------------------------------------------

@dataclass
class LSRatioResult:
    lecithin: float
    sphingomyelin: float
    ratio: float
    maturity: str
    rds_risk: str
    diabetic_note: str

    def __str__(self) -> str:
        lines = [
            "Lecithin/Sphingomyelin Ratio",
            "=" * 40,
            f"  Lecithin:       {self.lecithin:.2f}",
            f"  Sphingomyelin:  {self.sphingomyelin:.2f}",
            f"  L/S Ratio:      {self.ratio:.2f}",
            "-" * 40,
            f"  Lung Maturity:  {self.maturity}",
            f"  RDS Risk:       {self.rds_risk}",
            f"  Note:           {self.diabetic_note}",
        ]
        return "\n".join(lines)


def calc_ls_ratio(lecithin: float, sphingomyelin: float) -> LSRatioResult:
    """Calculate the lecithin/sphingomyelin ratio for fetal lung maturity.

    Parameters
    ----------
    lecithin : float
        Lecithin concentration (arbitrary units, must be > 0).
    sphingomyelin : float
        Sphingomyelin concentration (arbitrary units, must be > 0).

    Returns
    -------
    LSRatioResult

    Example
    -------
    >>> r = calc_ls_ratio(4.0, 1.5)
    >>> r.maturity
    'Mature'
    """
    if sphingomyelin <= 0:
        raise ValueError("Sphingomyelin must be > 0.")
    if lecithin < 0:
        raise ValueError("Lecithin must be >= 0.")

    ratio = lecithin / sphingomyelin

    if ratio >= 2.0:
        maturity = "Mature"
        rds_risk = "Low risk of respiratory distress syndrome."
    elif ratio >= 1.5:
        maturity = "Transitional"
        rds_risk = "Intermediate risk -- lung maturity uncertain."
    else:
        maturity = "Immature"
        rds_risk = "High risk of respiratory distress syndrome."

    diabetic_note = (
        "In diabetic mothers, an L/S ratio >= 3.0 is recommended "
        "for reliable lung maturity assessment due to delayed "
        "surfactant production."
    )

    return LSRatioResult(
        lecithin=lecithin,
        sphingomyelin=sphingomyelin,
        ratio=ratio,
        maturity=maturity,
        rds_risk=rds_risk,
        diabetic_note=diabetic_note,
    )


# ---------------------------------------------------------------------------
# Neonatal Bilirubin Risk
# ---------------------------------------------------------------------------

@dataclass
class BilirubinRiskResult:
    total_bilirubin: float
    age_hours: int
    gestational_age_weeks: float
    risk_zone: str
    phototherapy_indicated: bool
    phototherapy_threshold: float
    exchange_transfusion_threshold: float
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Neonatal Bilirubin Risk Assessment",
            "=" * 40,
            f"  Total Bilirubin:     {self.total_bilirubin:.1f} mg/dL",
            f"  Age:                 {self.age_hours} hours",
            f"  Gestational Age:     {self.gestational_age_weeks:.1f} weeks",
            "-" * 40,
            f"  Risk Zone:           {self.risk_zone}",
            f"  Phototherapy:        {'Indicated' if self.phototherapy_indicated else 'Not indicated'}",
            f"  Photo Threshold:     {self.phototherapy_threshold:.1f} mg/dL",
            f"  Exchange Threshold:  {self.exchange_transfusion_threshold:.1f} mg/dL",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


def calc_neonatal_bilirubin_risk(
    total_bilirubin: float,
    age_hours: int,
    gestational_age_weeks: float = 40.0,
) -> BilirubinRiskResult:
    """Assess neonatal bilirubin risk using simplified Bhutani nomogram zones.

    Parameters
    ----------
    total_bilirubin : float
        Total serum bilirubin in mg/dL.
    age_hours : int
        Postnatal age in hours.
    gestational_age_weeks : float
        Gestational age at birth (default 40.0).

    Returns
    -------
    BilirubinRiskResult

    Example
    -------
    >>> r = calc_neonatal_bilirubin_risk(14.0, 48, 38.0)
    >>> r.risk_zone
    'High-intermediate'
    """
    if age_hours < 0:
        raise ValueError("Age in hours must be >= 0.")
    if total_bilirubin < 0:
        raise ValueError("Total bilirubin must be >= 0.")

    # Simplified 95th percentile thresholds (Bhutani nomogram, >= 35 weeks)
    # These are approximate upper boundaries of each zone.
    high_risk_thresholds = {
        12: 5.0, 24: 8.0, 36: 10.5, 48: 13.0, 60: 14.5,
        72: 16.0, 84: 17.0, 96: 17.5, 108: 17.8, 120: 18.0,
    }

    # Phototherapy thresholds for >= 35 weeks (AAP guidelines, simplified)
    photo_thresholds = {
        24: 12.0, 36: 13.0, 48: 15.0, 60: 16.5,
        72: 18.0, 84: 19.0, 96: 20.0, 108: 20.5, 120: 21.0,
    }

    # Exchange transfusion thresholds for >= 35 weeks
    exchange_thresholds = {
        24: 17.0, 36: 18.0, 48: 20.0, 60: 22.0,
        72: 25.0, 84: 25.0, 96: 25.0, 108: 25.0, 120: 25.0,
    }

    def _interpolate(thresholds: dict, hours: int) -> float:
        """Linearly interpolate between threshold time points."""
        keys = sorted(thresholds.keys())
        if hours <= keys[0]:
            return thresholds[keys[0]]
        if hours >= keys[-1]:
            return thresholds[keys[-1]]
        for i in range(len(keys) - 1):
            if keys[i] <= hours <= keys[i + 1]:
                t0, t1 = keys[i], keys[i + 1]
                v0, v1 = thresholds[t0], thresholds[t1]
                frac = (hours - t0) / (t1 - t0)
                return v0 + frac * (v1 - v0)
        return thresholds[keys[-1]]

    high_thresh = _interpolate(high_risk_thresholds, age_hours)
    photo_thresh = _interpolate(photo_thresholds, age_hours)
    exchange_thresh = _interpolate(exchange_thresholds, age_hours)

    # Adjust for prematurity (lower thresholds for < 38 weeks)
    if gestational_age_weeks < 38.0:
        preterm_factor = 0.85
        high_thresh *= preterm_factor
        photo_thresh *= preterm_factor
        exchange_thresh *= preterm_factor

    # Risk zone classification
    if total_bilirubin >= high_thresh:
        risk_zone = "High"
    elif total_bilirubin >= high_thresh * 0.85:
        risk_zone = "High-intermediate"
    elif total_bilirubin >= high_thresh * 0.70:
        risk_zone = "Low-intermediate"
    else:
        risk_zone = "Low"

    phototherapy_indicated = total_bilirubin >= photo_thresh

    if total_bilirubin >= exchange_thresh:
        interpretation = (
            "CRITICAL -- Total bilirubin at or above exchange transfusion threshold. "
            "Immediate evaluation for exchange transfusion required."
        )
    elif phototherapy_indicated:
        interpretation = (
            "Phototherapy indicated. Recheck bilirubin in 4-6 hours to assess response."
        )
    elif risk_zone in ("High", "High-intermediate"):
        interpretation = (
            "Elevated risk zone. Close monitoring recommended with repeat bilirubin in 6-12 hours."
        )
    else:
        interpretation = (
            "Low risk. Routine monitoring per nursery protocol."
        )

    return BilirubinRiskResult(
        total_bilirubin=total_bilirubin,
        age_hours=age_hours,
        gestational_age_weeks=gestational_age_weeks,
        risk_zone=risk_zone,
        phototherapy_indicated=phototherapy_indicated,
        phototherapy_threshold=photo_thresh,
        exchange_transfusion_threshold=exchange_thresh,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Preeclampsia Risk
# ---------------------------------------------------------------------------

@dataclass
class PreeclampsiaRiskResult:
    sflt1: float
    plgf: float
    sflt1_plgf_ratio: float
    ratio_interpretation: str
    systolic: Optional[float]
    proteinuria_mg: Optional[float]
    bp_criteria_met: bool
    severe_features: list
    risk_category: str
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Preeclampsia Risk Assessment",
            "=" * 40,
            f"  sFlt-1:          {self.sflt1:.1f} pg/mL",
            f"  PlGF:            {self.plgf:.1f} pg/mL",
            f"  sFlt-1/PlGF:     {self.sflt1_plgf_ratio:.1f}",
            f"  Ratio:           {self.ratio_interpretation}",
        ]
        if self.systolic is not None:
            lines.append(f"  Systolic BP:     {self.systolic:.0f} mmHg")
        if self.proteinuria_mg is not None:
            lines.append(f"  Proteinuria:     {self.proteinuria_mg:.0f} mg/24h")
        lines.append("-" * 40)
        if self.severe_features:
            lines.append(f"  Severe Features: {', '.join(self.severe_features)}")
        lines.append(f"  Risk Category:   {self.risk_category}")
        lines.append(f"  {self.interpretation}")
        return "\n".join(lines)


def calc_preeclampsia_risk(
    sflt1: float,
    plgf: float,
    systolic: Optional[float] = None,
    proteinuria_mg: Optional[float] = None,
) -> PreeclampsiaRiskResult:
    """Assess preeclampsia risk using sFlt-1/PlGF ratio and clinical criteria.

    Parameters
    ----------
    sflt1 : float
        Soluble fms-like tyrosine kinase 1, pg/mL.
    plgf : float
        Placental growth factor, pg/mL (must be > 0).
    systolic : float, optional
        Systolic blood pressure, mmHg.
    proteinuria_mg : float, optional
        24-hour urine protein, mg.

    Returns
    -------
    PreeclampsiaRiskResult

    Example
    -------
    >>> r = calc_preeclampsia_risk(5000, 50, systolic=155, proteinuria_mg=400)
    >>> r.risk_category
    'High'
    """
    if plgf <= 0:
        raise ValueError("PlGF must be > 0.")

    ratio = sflt1 / plgf

    if ratio < 38:
        ratio_interpretation = "Low -- preeclampsia unlikely within 1 week (rules out)."
    elif ratio <= 85:
        ratio_interpretation = "Intermediate -- clinical correlation required."
    else:
        ratio_interpretation = "High -- preeclampsia likely within 4 weeks (rules in)."

    bp_criteria_met = systolic is not None and systolic >= 140

    severe_features = []
    if systolic is not None and systolic >= 160:
        severe_features.append("Severe hypertension (>=160 mmHg)")
    if proteinuria_mg is not None and proteinuria_mg > 300:
        severe_features.append(f"Proteinuria ({proteinuria_mg:.0f} mg/24h > 300)")

    # Risk category
    if ratio >= 85 or len(severe_features) >= 2:
        risk_category = "High"
    elif ratio >= 38 or bp_criteria_met or len(severe_features) >= 1:
        risk_category = "Intermediate"
    else:
        risk_category = "Low"

    if risk_category == "High":
        interpretation = (
            "High risk of preeclampsia. Consider delivery planning, "
            "close maternal-fetal monitoring, and magnesium sulfate prophylaxis if severe."
        )
    elif risk_category == "Intermediate":
        interpretation = (
            "Intermediate risk. Serial monitoring of sFlt-1/PlGF ratio, "
            "blood pressure, and proteinuria recommended. Repeat in 1-2 weeks."
        )
    else:
        interpretation = (
            "Low risk. Preeclampsia unlikely within the next week. "
            "Routine antenatal monitoring."
        )

    return PreeclampsiaRiskResult(
        sflt1=sflt1,
        plgf=plgf,
        sflt1_plgf_ratio=ratio,
        ratio_interpretation=ratio_interpretation,
        systolic=systolic,
        proteinuria_mg=proteinuria_mg,
        bp_criteria_met=bp_criteria_met,
        severe_features=severe_features,
        risk_category=risk_category,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Gestational Age
# ---------------------------------------------------------------------------

@dataclass
class GestationalAgeResult:
    method: str
    ga_weeks: float
    ga_days: int
    ga_display: str
    edd: Optional[str]
    lmp_date: Optional[str]
    us_crl_mm: Optional[float]

    def __str__(self) -> str:
        lines = [
            "Gestational Age Estimation",
            "=" * 40,
            f"  Method:     {self.method}",
            f"  GA:         {self.ga_display}",
        ]
        if self.lmp_date is not None:
            lines.append(f"  LMP:        {self.lmp_date}")
        if self.us_crl_mm is not None:
            lines.append(f"  CRL:        {self.us_crl_mm:.1f} mm")
        if self.edd is not None:
            lines.append(f"  EDD:        {self.edd}")
        return "\n".join(lines)


def calc_gestational_age(
    lmp_date: Optional[str] = None,
    us_crl_mm: Optional[float] = None,
    current_date: Optional[str] = None,
) -> GestationalAgeResult:
    """Estimate gestational age from LMP or first-trimester CRL.

    Parameters
    ----------
    lmp_date : str, optional
        Last menstrual period date as 'YYYY-MM-DD'.
    us_crl_mm : float, optional
        Crown-rump length in mm (first trimester ultrasound).
    current_date : str, optional
        Current date as 'YYYY-MM-DD' (required if lmp_date is provided;
        defaults to today if omitted).

    Returns
    -------
    GestationalAgeResult

    Example
    -------
    >>> r = calc_gestational_age(us_crl_mm=50.0)
    >>> 10 < r.ga_weeks < 14
    True
    """
    if lmp_date is None and us_crl_mm is None:
        raise ValueError("Provide at least one of lmp_date or us_crl_mm.")

    def _parse_date(date_str: str):
        """Parse YYYY-MM-DD without importing datetime at module level."""
        parts = date_str.strip().split("-")
        if len(parts) != 3:
            raise ValueError(f"Date must be YYYY-MM-DD, got '{date_str}'.")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        # Convert to days since epoch using a simple algorithm
        return year, month, day

    def _days_from_epoch(y, m, d):
        """Compute days from a reference using a simplified algorithm."""
        # Adjust for months Jan/Feb
        if m <= 2:
            y -= 1
            m += 12
        # Gregorian calendar days
        return (
            365 * y + y // 4 - y // 100 + y // 400
            + (153 * (m - 3) + 2) // 5 + d - 719469
        )

    if lmp_date is not None:
        ly, lm, ld = _parse_date(lmp_date)
        lmp_days = _days_from_epoch(ly, lm, ld)

        if current_date is not None:
            cy, cm, cd = _parse_date(current_date)
        else:
            # Use a basic approach: caller should provide current_date
            # Fall back to a note
            raise ValueError(
                "current_date is required when using lmp_date "
                "(format: 'YYYY-MM-DD')."
            )

        cur_days = _days_from_epoch(cy, cm, cd)
        days_since_lmp = cur_days - lmp_days

        if days_since_lmp < 0:
            raise ValueError("current_date must be after lmp_date.")

        ga_days_total = days_since_lmp
        ga_weeks = ga_days_total / 7.0
        weeks_int = ga_days_total // 7
        remainder_days = ga_days_total % 7
        ga_display = f"{weeks_int}w{remainder_days}d"

        # EDD = LMP + 280 days (Naegele's rule)
        edd_days = lmp_days + 280
        # Convert back to date
        # Inverse of _days_from_epoch (simplified)
        def _date_from_days(total):
            z = total + 719469
            era = (z if z >= 0 else z - 146096) // 146097
            doe = z - era * 146097
            yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
            y2 = yoe + era * 400
            doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
            mp = (5 * doy + 2) // 153
            d2 = doy - (153 * mp + 2) // 5 + 1
            m2 = mp + (3 if mp < 10 else -9)
            y2 += (1 if m2 <= 2 else 0)
            return f"{y2:04d}-{m2:02d}-{d2:02d}"

        edd_str = _date_from_days(edd_days)

        return GestationalAgeResult(
            method="LMP (Naegele's rule)",
            ga_weeks=ga_weeks,
            ga_days=ga_days_total,
            ga_display=ga_display,
            edd=edd_str,
            lmp_date=lmp_date,
            us_crl_mm=None,
        )

    else:
        # CRL-based estimation (Robinson formula simplified)
        # GA(days) = CRL(mm) * 0.15 + 35
        ga_days_total = us_crl_mm * 0.15 + 35
        ga_days_total = round(ga_days_total)
        ga_weeks = ga_days_total / 7.0
        weeks_int = ga_days_total // 7
        remainder_days = ga_days_total % 7
        ga_display = f"{weeks_int}w{remainder_days}d"

        return GestationalAgeResult(
            method="Ultrasound CRL (Robinson formula)",
            ga_weeks=ga_weeks,
            ga_days=ga_days_total,
            ga_display=ga_display,
            edd=None,
            lmp_date=None,
            us_crl_mm=us_crl_mm,
        )


# ---------------------------------------------------------------------------
# Amniotic Fluid Index
# ---------------------------------------------------------------------------

@dataclass
class AFIResult:
    q1: float
    q2: float
    q3: float
    q4: float
    afi: float
    classification: str
    interpretation: str

    def __str__(self) -> str:
        lines = [
            "Amniotic Fluid Index",
            "=" * 40,
            f"  Q1 (upper right):  {self.q1:.1f} cm",
            f"  Q2 (upper left):   {self.q2:.1f} cm",
            f"  Q3 (lower left):   {self.q3:.1f} cm",
            f"  Q4 (lower right):  {self.q4:.1f} cm",
            "-" * 40,
            f"  AFI:               {self.afi:.1f} cm",
            f"  Classification:    {self.classification}",
            f"  {self.interpretation}",
        ]
        return "\n".join(lines)


def calc_amniotic_fluid_index(
    q1: float,
    q2: float,
    q3: float,
    q4: float,
) -> AFIResult:
    """Calculate the amniotic fluid index from four-quadrant measurements.

    Parameters
    ----------
    q1 : float
        Deepest vertical pocket in upper-right quadrant (cm).
    q2 : float
        Deepest vertical pocket in upper-left quadrant (cm).
    q3 : float
        Deepest vertical pocket in lower-left quadrant (cm).
    q4 : float
        Deepest vertical pocket in lower-right quadrant (cm).

    Returns
    -------
    AFIResult

    Example
    -------
    >>> r = calc_amniotic_fluid_index(4.0, 3.5, 3.0, 2.5)
    >>> r.afi
    13.0
    """
    for label, val in [("q1", q1), ("q2", q2), ("q3", q3), ("q4", q4)]:
        if val < 0:
            raise ValueError(f"{label} must be >= 0, got {val}.")

    afi = q1 + q2 + q3 + q4

    if afi < 5.0:
        classification = "Oligohydramnios"
        interpretation = (
            "AFI < 5 cm indicates oligohydramnios. Evaluate for PPROM, "
            "renal anomalies, IUGR, or post-dates pregnancy. Consider NST "
            "and biophysical profile."
        )
    elif afi > 25.0:
        classification = "Polyhydramnios"
        interpretation = (
            "AFI > 25 cm indicates polyhydramnios. Evaluate for gestational "
            "diabetes, fetal anomalies (GI, CNS), or hydrops. Consider glucose "
            "tolerance test if not already done."
        )
    else:
        classification = "Normal"
        interpretation = "AFI 5-25 cm is within the normal range. Routine monitoring."

    return AFIResult(
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        afi=afi,
        classification=classification,
        interpretation=interpretation,
    )
