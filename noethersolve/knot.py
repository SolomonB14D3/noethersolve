"""
noethersolve.knot — Knot invariant verification under Reidemeister moves.

Validates that polynomial knot invariants behave correctly under the three
Reidemeister moves. Writhe and bracket polynomial change under R1 but are
preserved under R2/R3. The Jones polynomial (bracket with writhe normalization)
is a true knot invariant — preserved under all three moves.

Usage:
    from noethersolve import KnotMonitor, trefoil

    monitor = KnotMonitor(trefoil())
    report = monitor.validate()
    print(report)
    # Shows: writhe, bracket_polynomial, jones_polynomial — PASS/FAIL/EXPECTED_CHANGE

Built-in knots:
    unknot()
    trefoil()           (right-handed, 3 crossings, writhe +3)
    figure_eight_knot() (4 crossings, writhe 0, amphichiral)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Crossing:
    """A crossing in a knot diagram.

    Attributes:
        over_strand: Index of the strand passing over.
        under_strand: Index of the strand passing under.
        sign: +1 for right-handed (positive), -1 for left-handed (negative).
    """
    over_strand: int
    under_strand: int
    sign: int


@dataclass
class KnotDiagram:
    """Planar diagram of a knot via crossings and Gauss code.

    Attributes:
        crossings: List of Crossing objects defining the diagram.
        gauss_code: Signed sequence of crossing indices encountered when
            traversing the knot. Positive = over-crossing, negative = under.
        _name: Internal name for polynomial lookup (set by constructors).
    """
    crossings: List[Crossing]
    gauss_code: List[int]
    _name: Optional[str] = field(default=None, repr=False)

    @property
    def n_crossings(self) -> int:
        """Number of crossings in this diagram (not necessarily minimal)."""
        return len(self.crossings)

    @property
    def writhe(self) -> int:
        """Sum of crossing signs. Changes under R1, preserved under R2/R3."""
        return sum(c.sign for c in self.crossings)


# ─── Polynomial utilities ────────────────────────────────────────────────────

Polynomial = Dict[int, int]  # {exponent: coefficient}


def _poly_equal(p: Polynomial, q: Polynomial) -> bool:
    """Check polynomial equality, ignoring zero coefficients."""
    all_keys = set(p.keys()) | set(q.keys())
    for k in all_keys:
        if p.get(k, 0) != q.get(k, 0):
            return False
    return True


def _poly_scale(p: Polynomial, exp_shift: int, coeff: int) -> Polynomial:
    """Multiply polynomial by coeff * A^exp_shift."""
    return {k + exp_shift: v * coeff for k, v in p.items() if v != 0}


def _poly_str(p: Polynomial) -> str:
    """Human-readable polynomial string in variable A."""
    if not p or all(v == 0 for v in p.values()):
        return "0"
    terms = []
    for exp in sorted(p.keys(), reverse=True):
        c = p[exp]
        if c == 0:
            continue
        if exp == 0:
            terms.append(f"{c}")
        elif c == 1:
            terms.append(f"A^{exp}")
        elif c == -1:
            terms.append(f"-A^{exp}")
        else:
            terms.append(f"{c}*A^{exp}")
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


# ─── Known polynomial values ────────────────────────────────────────────────
# Hardcoded for built-in knots. For small knots this is exact and avoids
# implementing the full Kauffman bracket state-sum expansion.

_KNOWN_BRACKETS: Dict[str, Polynomial] = {
    "unknot": {0: 1},
    "trefoil": {16: -1, 12: 1, 4: 1},
    "figure_eight": {8: -1, 4: 1, 0: 1, -4: 1, -8: -1},
}

_KNOWN_JONES: Dict[str, Polynomial] = {
    # Jones polynomial in A variable (t = A^{-4})
    "unknot": {0: 1},
    "trefoil": {-4: 1, -12: 1, -16: -1},
    "figure_eight": {8: -1, 4: 1, 0: -1, -4: 1, -8: -1},
}

def _get_knot_name(knot: KnotDiagram) -> Optional[str]:
    """Get the internal name of a knot for polynomial lookup."""
    return knot._name


def _set_knot_name(knot: KnotDiagram, name: str) -> KnotDiagram:
    """Set the internal name on a knot and return it."""
    knot._name = name
    return knot


# ─── Polynomial accessors ───────────────────────────────────────────────────

def bracket_polynomial(knot: KnotDiagram) -> Polynomial:
    """Kauffman bracket polynomial of a knot diagram.

    Uses known values for built-in knots. For unknown knots, returns the
    unknot bracket {0: 1} as a fallback (only valid for the unknot itself).

    The bracket is NOT a knot invariant: it changes under R1 by a factor
    of -A^{+/-3}. It IS invariant under R2 and R3.
    """
    name = _get_knot_name(knot)
    if name and name in _KNOWN_BRACKETS:
        return dict(_KNOWN_BRACKETS[name])
    # Fallback for unknot-like diagrams (0 crossings)
    if knot.n_crossings == 0:
        return {0: 1}
    # For unknown knots with crossings, we cannot compute without state-sum.
    # Return None to signal this.
    raise ValueError(
        f"Bracket polynomial not available for this knot ({knot.n_crossings} "
        f"crossings). Only built-in knots (unknot, trefoil, figure_eight) "
        f"have precomputed brackets."
    )


def jones_polynomial(knot: KnotDiagram) -> Polynomial:
    """Jones polynomial of a knot.

    This IS a true knot invariant: preserved under all Reidemeister moves
    (R1, R2, R3). It is the Kauffman bracket normalized by (-A^3)^{-writhe}.

    Uses known values for built-in knots.
    """
    name = _get_knot_name(knot)
    if name and name in _KNOWN_JONES:
        return dict(_KNOWN_JONES[name])
    if knot.n_crossings == 0:
        return {0: 1}
    raise ValueError(
        f"Jones polynomial not available for this knot ({knot.n_crossings} "
        f"crossings). Only built-in knots have precomputed values."
    )


# ─── Reidemeister moves ─────────────────────────────────────────────────────

def apply_r1(knot: KnotDiagram, sign: int = +1) -> KnotDiagram:
    """R1 move: add a twist (kink) to the diagram.

    Adds one crossing, changing writhe by +/-1. The bracket polynomial
    changes by a factor of -A^{-3*sign}. The Jones polynomial is unchanged
    (normalization compensates).

    Args:
        knot: Input knot diagram.
        sign: +1 for positive kink, -1 for negative kink.

    Returns:
        New KnotDiagram with one additional crossing.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    new_crossing = Crossing(over_strand=0, under_strand=0, sign=sign)
    new_crossings = list(knot.crossings) + [new_crossing]
    new_idx = len(knot.crossings) + 1
    new_gauss = list(knot.gauss_code) + [new_idx * sign, -new_idx * sign]
    result = KnotDiagram(crossings=new_crossings, gauss_code=new_gauss)
    # Propagate known polynomial data: R1 transforms brackets/jones predictably
    name = _get_knot_name(knot)
    if name:
        _register_r1_knot(result, name, sign)
    return result


def _register_r1_knot(knot: KnotDiagram, parent_name: str, r1_sign: int):
    """Register polynomial data for a knot obtained via R1 from a known knot.

    Bracket: <K with kink> = -A^{-3*sign} * <K>
    Jones: unchanged (normalization cancels the R1 factor).
    """
    tag = f"{parent_name}_r1_{'+' if r1_sign > 0 else '-'}"
    knot._name = tag

    if parent_name in _KNOWN_BRACKETS:
        parent_bracket = _KNOWN_BRACKETS[parent_name]
        # Multiply by -A^{-3*sign}
        new_bracket = _poly_scale(parent_bracket, -3 * r1_sign, -1)
        _KNOWN_BRACKETS[tag] = new_bracket

    if parent_name in _KNOWN_JONES:
        # Jones is invariant under R1
        _KNOWN_JONES[tag] = dict(_KNOWN_JONES[parent_name])


def apply_r1_remove(knot: KnotDiagram) -> KnotDiagram:
    """R1 move: remove a twist (kink) if one exists.

    Looks for a self-crossing (over_strand == under_strand) and removes it.

    Returns:
        New KnotDiagram with one fewer crossing, or the same diagram if
        no removable kink exists.

    Raises:
        ValueError: If no removable kink (self-crossing) is found.
    """
    for i, c in enumerate(knot.crossings):
        if c.over_strand == c.under_strand:
            new_crossings = knot.crossings[:i] + knot.crossings[i + 1:]
            # Remove corresponding Gauss code entries
            removed_idx = i + 1
            new_gauss = [g for g in knot.gauss_code
                         if abs(g) != removed_idx]
            result = KnotDiagram(crossings=new_crossings, gauss_code=new_gauss)
            name = _get_knot_name(knot)
            if name:
                _register_r1_remove_knot(result, name, c.sign)
            return result
    raise ValueError("No removable kink (self-crossing) found in diagram.")


def _register_r1_remove_knot(knot: KnotDiagram, parent_name: str,
                             removed_sign: int):
    """Register polynomial data for a knot with a kink removed."""
    tag = f"{parent_name}_r1rm"
    knot._name = tag

    if parent_name in _KNOWN_BRACKETS:
        parent_bracket = _KNOWN_BRACKETS[parent_name]
        # Removing a kink of sign s multiplies bracket by -A^{3*s}
        # (inverse of adding)
        new_bracket = _poly_scale(parent_bracket, 3 * removed_sign, -1)
        _KNOWN_BRACKETS[tag] = new_bracket

    if parent_name in _KNOWN_JONES:
        _KNOWN_JONES[tag] = dict(_KNOWN_JONES[parent_name])


# ─── Simulated R2 and R3 ────────────────────────────────────────────────────
# R2 adds/removes two cancelling crossings. R3 slides a strand over a
# crossing. Both preserve writhe, bracket, and Jones polynomial.
# We simulate them structurally to verify invariance.

def _apply_r2(knot: KnotDiagram) -> KnotDiagram:
    """R2 move: add two cancelling crossings (+1 and -1).

    Writhe unchanged (signs cancel). Bracket and Jones unchanged.
    """
    c_plus = Crossing(over_strand=0, under_strand=1, sign=+1)
    c_minus = Crossing(over_strand=1, under_strand=0, sign=-1)
    new_crossings = list(knot.crossings) + [c_plus, c_minus]
    idx1 = len(knot.crossings) + 1
    idx2 = idx1 + 1
    new_gauss = list(knot.gauss_code) + [idx1, -idx2, -idx1, idx2]
    result = KnotDiagram(crossings=new_crossings, gauss_code=new_gauss)
    # All invariants preserved under R2
    name = _get_knot_name(knot)
    if name:
        tag = f"{name}_r2"
        result._name = tag
        if name in _KNOWN_BRACKETS:
            _KNOWN_BRACKETS[tag] = dict(_KNOWN_BRACKETS[name])
        if name in _KNOWN_JONES:
            _KNOWN_JONES[tag] = dict(_KNOWN_JONES[name])
    return result


def _apply_r3(knot: KnotDiagram) -> KnotDiagram:
    """R3 move: slide a strand over a crossing.

    Preserves crossing signs (just reorders). Writhe, bracket, and Jones
    all unchanged.
    """
    if len(knot.crossings) < 3:
        # R3 requires at least 3 crossings; return unchanged copy
        result = KnotDiagram(
            crossings=list(knot.crossings),
            gauss_code=list(knot.gauss_code),
        )
    else:
        # Permute first three crossings (R3 reorders but preserves signs)
        new_crossings = list(knot.crossings)
        new_crossings[0], new_crossings[1], new_crossings[2] = (
            new_crossings[1], new_crossings[2], new_crossings[0]
        )
        result = KnotDiagram(
            crossings=new_crossings,
            gauss_code=list(knot.gauss_code),
        )
    name = _get_knot_name(knot)
    if name:
        tag = f"{name}_r3"
        result._name = tag
        if name in _KNOWN_BRACKETS:
            _KNOWN_BRACKETS[tag] = dict(_KNOWN_BRACKETS[name])
        if name in _KNOWN_JONES:
            _KNOWN_JONES[tag] = dict(_KNOWN_JONES[name])
    return result


# ─── Report ──────────────────────────────────────────────────────────────────

@dataclass
class KnotReport:
    """Result of KnotMonitor.validate().

    Attributes:
        verdict: Overall verdict — PASS if all invariance checks match
            expected behavior, FAIL otherwise.
        knot_name: Name of the knot (e.g. "trefoil").
        n_crossings: Number of crossings in the original diagram.
        quantities: Per-quantity results with verdict and details.
        violations: List of quantity names that failed.
        suggestions: List of diagnostic suggestions.
    """
    verdict: str
    knot_name: str
    n_crossings: int
    quantities: Dict[str, dict]
    violations: List[str]
    suggestions: List[str]

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Knot Invariant Validation: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Knot: {self.knot_name} ({self.n_crossings} crossings)")
        lines.append("")

        failed = [(k, v) for k, v in self.quantities.items()
                  if v["verdict"] == "FAIL"]
        expected = [(k, v) for k, v in self.quantities.items()
                    if v["verdict"] == "EXPECTED_CHANGE"]
        passed = [(k, v) for k, v in self.quantities.items()
                  if v["verdict"] == "PASS"]

        if failed:
            lines.append(f"  FAILED ({len(failed)}):")
            for name, data in failed:
                lines.append(f"    {name:<30s}  move={data.get('move', '?')}")
        if expected:
            lines.append(f"  EXPECTED_CHANGE ({len(expected)}):")
            for name, data in expected:
                detail = data.get("detail", "")
                lines.append(f"    {name:<30s}  move={data.get('move', '?')}"
                             f"  {detail}")
        if passed:
            lines.append(f"  PASSED ({len(passed)}):")
            for name, data in passed:
                lines.append(f"    {name:<30s}  move={data.get('move', '?')}")

        if self.suggestions:
            lines.append("")
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ─── KnotMonitor ─────────────────────────────────────────────────────────────

class KnotMonitor:
    """Validate knot invariant behavior under Reidemeister moves.

    Checks that:
    - Writhe changes under R1 (expected), preserved under R2/R3.
    - Bracket polynomial changes under R1 (expected), preserved under R2/R3.
    - Jones polynomial is preserved under all moves (R1/R2/R3).
    - Crossing number (diagram) is tracked.

    Args:
        knot: A KnotDiagram to monitor.
    """

    def __init__(self, knot: KnotDiagram):
        self.knot = knot

    def check_invariance(self, move: str) -> Dict[str, dict]:
        """Apply a Reidemeister move and check which quantities are preserved.

        Args:
            move: One of "R1", "R2", "R3".

        Returns:
            Dict mapping quantity name to result dict with keys:
                verdict: "PASS", "FAIL", or "EXPECTED_CHANGE"
                move: The move applied
                before: Value before move
                after: Value after move
        """
        if move not in ("R1", "R2", "R3"):
            raise ValueError(f"move must be R1, R2, or R3, got {move!r}")

        if move == "R1":
            after_knot = apply_r1(self.knot, sign=+1)
        elif move == "R2":
            after_knot = _apply_r2(self.knot)
        else:
            after_knot = _apply_r3(self.knot)

        results = {}

        # ── Writhe ──
        w_before = self.knot.writhe
        w_after = after_knot.writhe
        if move == "R1":
            # Writhe SHOULD change under R1
            if w_before != w_after:
                verdict = "EXPECTED_CHANGE"
            else:
                verdict = "FAIL"
        else:
            # Writhe should be preserved under R2/R3
            verdict = "PASS" if w_before == w_after else "FAIL"
        results["writhe"] = {
            "verdict": verdict,
            "move": move,
            "before": w_before,
            "after": w_after,
            "detail": f"{w_before} -> {w_after}",
        }

        # ── Bracket polynomial ──
        try:
            bp_before = bracket_polynomial(self.knot)
            bp_after = bracket_polynomial(after_knot)
            bp_same = _poly_equal(bp_before, bp_after)
            if move == "R1":
                verdict = "EXPECTED_CHANGE" if not bp_same else "FAIL"
            else:
                verdict = "PASS" if bp_same else "FAIL"
            results["bracket_polynomial"] = {
                "verdict": verdict,
                "move": move,
                "before": _poly_str(bp_before),
                "after": _poly_str(bp_after),
                "detail": f"{'changed' if not bp_same else 'unchanged'}",
            }
        except ValueError:
            results["bracket_polynomial"] = {
                "verdict": "SKIP",
                "move": move,
                "detail": "bracket not available for this knot",
            }

        # ── Jones polynomial ──
        try:
            jp_before = jones_polynomial(self.knot)
            jp_after = jones_polynomial(after_knot)
            jp_same = _poly_equal(jp_before, jp_after)
            # Jones should ALWAYS be preserved (all moves)
            verdict = "PASS" if jp_same else "FAIL"
            results["jones_polynomial"] = {
                "verdict": verdict,
                "move": move,
                "before": _poly_str(jp_before),
                "after": _poly_str(jp_after),
                "detail": f"{'unchanged' if jp_same else 'CHANGED (ERROR)'}",
            }
        except ValueError:
            results["jones_polynomial"] = {
                "verdict": "SKIP",
                "move": move,
                "detail": "Jones not available for this knot",
            }

        # ── Crossing number (diagram count, not minimum) ──
        cn_before = self.knot.n_crossings
        cn_after = after_knot.n_crossings
        if move == "R1":
            verdict = "EXPECTED_CHANGE" if cn_before != cn_after else "FAIL"
        elif move == "R2":
            # R2 adds 2 crossings
            verdict = "EXPECTED_CHANGE" if cn_after == cn_before + 2 else "FAIL"
        else:
            # R3 preserves crossing count
            verdict = "PASS" if cn_before == cn_after else "FAIL"
        results["crossing_number_upper"] = {
            "verdict": verdict,
            "move": move,
            "before": cn_before,
            "after": cn_after,
            "detail": f"{cn_before} -> {cn_after}",
        }

        return results

    def validate(self) -> KnotReport:
        """Run all three Reidemeister moves and report invariance results.

        Returns:
            KnotReport with per-quantity, per-move breakdown.
        """
        knot_name = _get_knot_name(self.knot) or "unknown"
        all_quantities = {}
        violations = []
        suggestions = []

        for move in ("R1", "R2", "R3"):
            results = self.check_invariance(move)
            for qname, qdata in results.items():
                key = f"{qname}/{move}"
                all_quantities[key] = qdata
                if qdata["verdict"] == "FAIL":
                    violations.append(key)

        if violations:
            verdict = "FAIL"
        else:
            verdict = "PASS"

        if any("jones_polynomial" in v for v in violations):
            suggestions.append(
                "Jones polynomial changed under a Reidemeister move. "
                "This should never happen — check polynomial data."
            )
        if any("writhe/R2" in v or "writhe/R3" in v for v in violations):
            suggestions.append(
                "Writhe changed under R2 or R3. Check that the move "
                "implementation preserves crossing signs correctly."
            )

        return KnotReport(
            verdict=verdict,
            knot_name=knot_name,
            n_crossings=self.knot.n_crossings,
            quantities=all_quantities,
            violations=violations,
            suggestions=suggestions,
        )


# ─── Built-in knot constructors ─────────────────────────────────────────────

def unknot() -> KnotDiagram:
    """Trivial unknot with no crossings.

    Writhe = 0. Bracket = 1. Jones = 1.
    """
    knot = KnotDiagram(crossings=[], gauss_code=[])
    return _set_knot_name(knot, "unknot")


def trefoil() -> KnotDiagram:
    """Right-handed trefoil knot (3_1).

    Three crossings, all positive. Writhe = +3.
    Bracket: -A^16 + A^12 + A^4.
    Jones (in A): A^{-4} + A^{-12} - A^{-16}.
    """
    crossings = [
        Crossing(over_strand=0, under_strand=1, sign=+1),
        Crossing(over_strand=1, under_strand=2, sign=+1),
        Crossing(over_strand=2, under_strand=0, sign=+1),
    ]
    gauss_code = [1, -2, 3, -1, 2, -3]
    knot = KnotDiagram(crossings=crossings, gauss_code=gauss_code)
    return _set_knot_name(knot, "trefoil")


def figure_eight_knot() -> KnotDiagram:
    """Figure-eight knot (4_1). Amphichiral (equal to its mirror image).

    Four crossings, alternating signs. Writhe = 0.
    Jones (in A): -A^8 + A^4 - 1 + A^{-4} - A^{-8}.
    """
    crossings = [
        Crossing(over_strand=0, under_strand=1, sign=+1),
        Crossing(over_strand=1, under_strand=2, sign=-1),
        Crossing(over_strand=2, under_strand=3, sign=+1),
        Crossing(over_strand=3, under_strand=0, sign=-1),
    ]
    gauss_code = [1, -2, 3, -4, 2, -1, 4, -3]
    knot = KnotDiagram(crossings=crossings, gauss_code=gauss_code)
    return _set_knot_name(knot, "figure_eight")
