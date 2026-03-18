"""
noethersolve.conjecture_status — Mathematical conjecture status checker.

Validates claims about the status of mathematical conjectures and open problems.
Catches: claiming open problems are solved, confusing partial results with full
proofs, missing key implications, getting the status of controversial proofs wrong.

Covers ~70 conjectures across 6 domains: Millennium Problems, Number Theory,
Algebra/Topology, Proof Techniques/Logic, Analysis/PDE, and Computational.

Usage:
    from noethersolve.conjecture_status import (
        check_conjecture, check_claim, list_conjectures, get_conjecture,
        ConjectureReport, ConjectureIssue, ConjectureInfo,
    )

    # Look up a conjecture and validate a claimed status
    report = check_conjecture("Riemann Hypothesis", claimed_status="SOLVED")
    print(report)
    # FAIL — 1 issue: STATUS_CHECK [HIGH] ...

    # Parse a natural-language claim
    report = check_claim("the Riemann Hypothesis was proved in 2018")
    print(report)
    # FAIL — claimed SOLVED but actual status is OPEN

    # List all open conjectures
    for name in list_conjectures(status="OPEN"):
        print(name)

    # Get full info
    info = get_conjecture("Poincare conjecture")
    print(info.status, info.solver, info.year)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Conjecture database ─────────────────────────────────────────────────────

@dataclass
class ConjectureInfo:
    """Full information about a mathematical conjecture."""
    name: str
    status: str               # OPEN, SOLVED, DISPUTED, INDEPENDENT, MOSTLY_SOLVED, CONJECTURED_TRUE
    year: int                 # year stated or first formulated
    domain: str               # Millennium, Number Theory, Algebra/Topology, etc.
    solver: Optional[str]     # who solved it, if solved
    solver_year: Optional[int]  # year solved
    prize: Optional[str]      # prize amount if applicable
    key_facts: List[str]      # important facts about this conjecture
    implications: List[str]   # what it implies if true/solved
    common_errors: List[str]  # what people get wrong about it

    def __str__(self):
        lines = [f"{self.name} ({self.status})"]
        lines.append(f"  Domain: {self.domain}, Year: {self.year}")
        if self.solver:
            lines.append(f"  Solver: {self.solver} ({self.solver_year})")
        if self.prize:
            lines.append(f"  Prize: {self.prize}")
        if self.key_facts:
            lines.append("  Key facts:")
            for f in self.key_facts:
                lines.append(f"    - {f}")
        if self.implications:
            lines.append("  Implications:")
            for imp in self.implications:
                lines.append(f"    - {imp}")
        if self.common_errors:
            lines.append("  Common errors:")
            for err in self.common_errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


def _build_database() -> Dict[str, ConjectureInfo]:
    """Build the conjecture database. Called once at module load."""
    db = {}

    def _add(name, **kwargs):
        kwargs.setdefault("solver", None)
        kwargs.setdefault("solver_year", None)
        kwargs.setdefault("prize", None)
        kwargs.setdefault("key_facts", [])
        kwargs.setdefault("implications", [])
        kwargs.setdefault("common_errors", [])
        db[name.lower()] = ConjectureInfo(name=name, **kwargs)

    # ── Millennium Problems ──────────────────────────────────────────────

    _add("Riemann Hypothesis",
         status="OPEN", year=1859, domain="Millennium",
         prize="$1M",
         key_facts=[
             "All non-trivial zeros of the Riemann zeta function have real part 1/2",
             "Over 10 trillion zeros verified numerically, all on the critical line",
             "Equivalent to tight bounds on the error term in the prime counting function",
         ],
         implications=[
             "Sharp bounds on prime distribution (pi(x) - Li(x))",
             "Implies the Lindelof hypothesis",
             "Would prove the Miller-Rabin primality test is deterministic in polynomial time",
         ],
         common_errors=[
             "Claiming RH is proved (it is not as of 2025)",
             "Claiming RH implies P != NP (it does not — they are unrelated)",
             "Confusing numerical verification of zeros with a proof",
             "Claiming de Branges proved it (his claimed proofs have not been accepted)",
         ])

    _add("P vs NP",
         status="OPEN", year=1971, domain="Millennium",
         prize="$1M",
         key_facts=[
             "Asks whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time",
             "Formally: does P = NP?",
             "Most experts believe P != NP but no proof exists",
             "Cook-Levin theorem (1971) established NP-completeness",
         ],
         implications=[
             "If P = NP: most cryptographic systems would be breakable in polynomial time",
             "If P != NP: fundamental limits on efficient computation are real",
             "Resolution either way would transform complexity theory",
         ],
         common_errors=[
             "Claiming P vs NP is solved (it is not)",
             "Claiming quantum computers solve NP-complete problems efficiently (they do not — BQP is not known to contain NP)",
             "Conflating average-case and worst-case hardness",
         ])

    _add("Navier-Stokes existence and smoothness",
         status="OPEN", year=2000, domain="Millennium",
         prize="$1M",
         key_facts=[
             "Asks whether smooth solutions always exist for 3D incompressible Navier-Stokes",
             "Or whether finite-time blowup (singularities) can occur",
             "2D case is solved (smooth solutions always exist)",
             "Partial regularity results: Caffarelli-Kohn-Nirenberg (1982) — singular set has zero 1D Hausdorff measure",
         ],
         implications=[
             "Fundamental to understanding turbulence",
             "Would clarify the mathematical foundations of fluid dynamics",
         ],
         common_errors=[
             "Claiming Navier-Stokes is solved (it is not)",
             "Confusing 2D results (solved) with 3D (open)",
             "Confusing Euler equations (inviscid) with Navier-Stokes (viscous)",
             "Claiming Tao's 2016 result proves blowup (it proves blowup for an averaged version, not the actual equations)",
         ])

    _add("Yang-Mills existence and mass gap",
         status="OPEN", year=2000, domain="Millennium",
         prize="$1M",
         key_facts=[
             "Prove that Yang-Mills theory exists as a quantum field theory and has a mass gap",
             "The mass gap is the difference in energy between the vacuum and the next lowest state",
             "Lattice QCD provides strong numerical evidence but not a proof",
         ],
         implications=[
             "Would provide rigorous mathematical foundation for the Standard Model",
             "Explains why the strong force is short-range despite massless gluons",
         ],
         common_errors=[
             "Claiming Yang-Mills is solved (it is not)",
             "Confusing lattice QCD numerical evidence with a mathematical proof",
         ])

    _add("Hodge conjecture",
         status="OPEN", year=1950, domain="Millennium",
         prize="$1M",
         key_facts=[
             "Certain cohomology classes on projective algebraic varieties are algebraic",
             "Specifically: every Hodge class on a smooth projective variety is a rational linear combination of classes of algebraic cycles",
             "Known for divisors (codimension 1) by Lefschetz theorem",
         ],
         implications=[
             "Deep connection between topology and algebraic geometry",
             "Would unify several areas of algebraic geometry",
         ],
         common_errors=[
             "Claiming the Hodge conjecture is solved (it is not)",
             "Confusing it with the Hodge theorem (which IS proved)",
         ])

    _add("Birch and Swinnerton-Dyer conjecture",
         status="OPEN", year=1965, domain="Millennium",
         prize="$1M",
         key_facts=[
             "Relates the rank of an elliptic curve to the order of vanishing of its L-function at s=1",
             "Proved for rank 0 and rank 1 cases (Gross-Zagier, Kolyvagin, 1987-1990)",
             "Full conjecture remains open for rank >= 2",
         ],
         implications=[
             "Would give an algorithm to determine the rank of any elliptic curve",
             "Connects arithmetic geometry and analytic number theory",
         ],
         common_errors=[
             "Claiming BSD is solved (only rank 0 and 1 cases are proved)",
             "Confusing partial results (rank 0/1) with the full conjecture",
         ])

    _add("Poincare conjecture",
         status="SOLVED", year=1904, domain="Millennium",
         solver="Grigori Perelman", solver_year=2003,
         prize="$1M (declined by Perelman)",
         key_facts=[
             "Every simply connected, closed 3-manifold is homeomorphic to the 3-sphere",
             "Proved by Perelman using Ricci flow with surgery (Hamilton's program)",
             "Perelman declined the Fields Medal (2006) and the $1M Clay prize",
             "Higher-dimensional generalizations were proved earlier: dim >= 5 (Smale 1961), dim 4 (Freedman 1982)",
         ],
         implications=[
             "Completes the classification of compact 3-manifolds (via geometrization)",
             "Validates Thurston's geometrization conjecture (also proved by Perelman)",
         ],
         common_errors=[
             "Claiming the Poincare conjecture is still open (it was solved by Perelman in 2003)",
             "Claiming Perelman accepted the Clay prize (he declined it)",
             "Confusing the Poincare conjecture with the smooth Poincare conjecture in dimension 4 (still open)",
         ])

    # ── Number Theory ────────────────────────────────────────────────────

    _add("Goldbach conjecture",
         status="OPEN", year=1742, domain="Number Theory",
         key_facts=[
             "Every even integer greater than 2 is the sum of two primes",
             "Verified computationally up to 4 x 10^18",
             "Weak/ternary Goldbach (every odd > 5 is the sum of three primes) proved by Helfgott in 2013",
             "Vinogradov (1937) proved it for sufficiently large odd numbers",
         ],
         implications=[
             "Fundamental statement about the additive structure of primes",
         ],
         common_errors=[
             "Claiming Goldbach conjecture is solved (only the WEAK/ternary version is proved)",
             "Confusing weak Goldbach (3 primes, odd numbers, SOLVED) with strong Goldbach (2 primes, even numbers, OPEN)",
             "Treating Helfgott's 2013 result as proving the full conjecture",
         ])

    _add("Twin prime conjecture",
         status="OPEN", year=1846, domain="Number Theory",
         key_facts=[
             "There are infinitely many pairs of primes differing by 2",
             "Zhang (2013) proved bounded gaps between primes (gap <= 70,000,000)",
             "Maynard-Tao (2013-2014) improved the bound to 246",
             "Polymath8 project refined Zhang's bound",
         ],
         implications=[
             "Would confirm a fundamental pattern in the distribution of primes",
         ],
         common_errors=[
             "Claiming twin primes conjecture is solved (bounded gaps are proved, but not specifically gap = 2)",
             "Confusing Zhang's bounded gaps result with proving twin primes",
             "Stating the current bound is 70 million (it has been reduced to 246)",
         ])

    _add("Collatz conjecture",
         status="OPEN", year=1937, domain="Number Theory",
         key_facts=[
             "For any positive integer: if even divide by 2, if odd multiply by 3 and add 1; repeat — conjecture says all sequences reach 1",
             "Verified for all starting values up to approximately 2.95 x 10^20",
             "Tao (2019) proved almost all Collatz orbits attain almost bounded values",
             "Erdos: 'Mathematics is not yet ready for such problems'",
         ],
         implications=[
             "Would reveal deep structure in the iteration of simple arithmetic operations",
         ],
         common_errors=[
             "Claiming Collatz is solved (it is not)",
             "Interpreting Tao's 2019 result as a full proof (it proves 'almost all', not all)",
             "Confusing computational verification with proof",
         ])

    _add("ABC conjecture",
         status="DISPUTED", year=1985, domain="Number Theory",
         key_facts=[
             "For coprime positive integers a+b=c, the radical rad(abc) is usually not much smaller than c",
             "Mochizuki claimed a proof in 2012 using Inter-universal Teichmuller theory (IUT)",
             "Published in PRIMS (2021), a journal where Mochizuki is editor-in-chief",
             "Scholze and Stix identified a gap in Corollary 3.12 of IUT that Mochizuki has not resolved to their satisfaction",
             "The mathematical community does NOT widely accept the proof as valid",
         ],
         implications=[
             "Would imply Fermat's Last Theorem for sufficiently large exponents",
             "Has numerous consequences in Diophantine equations",
             "Would imply Szpiro's conjecture and the modified Szpiro conjecture",
         ],
         common_errors=[
             "Claiming ABC conjecture is proved (the proof is disputed — Mochizuki's IUT is not widely accepted)",
             "Claiming ABC is definitively disproved (it is disputed, not refuted)",
             "Presenting the PRIMS publication as validating the proof (conflict of interest concerns)",
             "Ignoring the Scholze-Stix objection to Corollary 3.12",
         ])

    _add("Legendre conjecture",
         status="OPEN", year=1798, domain="Number Theory",
         key_facts=[
             "There is a prime between n^2 and (n+1)^2 for every positive integer n",
             "Ingham (1937) proved there is a prime between n^3 and (n+1)^3 for large n",
             "Implied by the Riemann Hypothesis",
             "Bertrand's postulate (proved) gives a prime between n and 2n, which is weaker",
         ],
         implications=[
             "Would refine our understanding of prime gaps",
         ],
         common_errors=[
             "Confusing Legendre conjecture with Bertrand's postulate (which IS proved)",
             "Claiming Legendre conjecture is solved (it is not)",
         ])

    _add("Catalan conjecture",
         status="SOLVED", year=1844, domain="Number Theory",
         solver="Preda Mihailescu", solver_year=2002,
         key_facts=[
             "The only solution in natural numbers to x^a - y^b = 1 with a,b > 1 is 3^2 - 2^3 = 1",
             "Now known as Mihailescu's theorem",
             "Proved using cyclotomic field theory",
         ],
         implications=[
             "Settles the question of consecutive perfect powers",
         ],
         common_errors=[
             "Claiming Catalan conjecture is still open (it was proved by Mihailescu in 2002)",
             "Not knowing the new name: Mihailescu's theorem",
         ])

    _add("Fermat's Last Theorem",
         status="SOLVED", year=1637, domain="Number Theory",
         solver="Andrew Wiles", solver_year=1995,
         key_facts=[
             "No three positive integers a, b, c satisfy a^n + b^n = c^n for n > 2",
             "Proved by Wiles (with Taylor) via modularity of semistable elliptic curves",
             "The proof goes through the Taniyama-Shimura-Weil conjecture (now modularity theorem)",
             "Wiles' original 1993 announcement had a gap, fixed with Taylor by 1995",
         ],
         implications=[
             "Proved the modularity theorem for semistable elliptic curves",
             "Opened the door to the full modularity theorem (Breuil-Conrad-Diamond-Taylor, 2001)",
         ],
         common_errors=[
             "Claiming FLT is still open (it was proved by Wiles in 1995)",
             "Claiming Fermat had a proof (he almost certainly did not — his 'margin' proof is lost and likely flawed)",
             "Not knowing that the proof uses elliptic curves and modular forms, not elementary methods",
         ])

    _add("Goldbach weak conjecture",
         status="SOLVED", year=1742, domain="Number Theory",
         solver="Harald Helfgott", solver_year=2013,
         key_facts=[
             "Every odd integer greater than 5 is the sum of three primes",
             "Also called the ternary Goldbach conjecture or odd Goldbach conjecture",
             "Vinogradov (1937) proved it for sufficiently large odd numbers",
             "Helfgott completed the proof for ALL odd numbers > 5",
         ],
         implications=[
             "Partial progress toward the full (binary/strong) Goldbach conjecture",
         ],
         common_errors=[
             "Confusing weak Goldbach (SOLVED) with strong Goldbach (OPEN)",
             "Not knowing Helfgott proved it (crediting only Vinogradov)",
         ])

    _add("Landau's fourth problem",
         status="OPEN", year=1912, domain="Number Theory",
         key_facts=[
             "Are there infinitely many primes of the form n^2 + 1?",
             "One of Landau's four problems presented at the 1912 ICM",
             "Friedlander-Iwaniec (1998) proved infinitely many primes of form a^2 + b^4",
             "Iwaniec (1978) proved infinitely many n such that n^2 + 1 has at most two prime factors",
         ],
         implications=[
             "Would illuminate the relationship between polynomial values and primes",
         ],
         common_errors=[
             "Claiming this is solved (it is not)",
             "Confusing with Bunyakovsky conjecture (more general, also open)",
         ])

    _add("Brocard's problem",
         status="OPEN", year=1876, domain="Number Theory",
         key_facts=[
             "Are there integer solutions to n! + 1 = m^2 other than n = 4, 5, 7?",
             "No other solutions found up to n = 10^9",
             "Related to the ABC conjecture — ABC would imply finitely many solutions",
         ],
         implications=[
             "Connects factorials and perfect squares",
         ],
         common_errors=[
             "Claiming Brocard's problem is solved (it is not)",
         ])

    _add("Bunyakovsky conjecture",
         status="OPEN", year=1857, domain="Number Theory",
         key_facts=[
             "An irreducible polynomial with positive leading coefficient and no fixed prime divisor represents infinitely many primes",
             "Generalizes Dirichlet's theorem on primes in arithmetic progressions",
             "Open even for f(x) = x^2 + 1",
         ],
         implications=[
             "Would imply Landau's fourth problem and many other conjectures about primes from polynomials",
         ],
         common_errors=[
             "Claiming this is solved (it is not)",
         ])

    _add("Erdos-Straus conjecture",
         status="OPEN", year=1948, domain="Number Theory",
         key_facts=[
             "For every integer n >= 2, the fraction 4/n can be written as a sum of three unit fractions",
             "Verified computationally for n up to 10^17",
         ],
         implications=[
             "Would settle a basic question about Egyptian fraction representations",
         ],
         common_errors=[
             "Claiming this is solved (it is not)",
             "Confusing computational verification with proof",
         ])

    # ── Algebra/Topology ─────────────────────────────────────────────────

    _add("Jacobian conjecture",
         status="OPEN", year=1939, domain="Algebra/Topology",
         key_facts=[
             "If a polynomial map C^n -> C^n has a constant nonzero Jacobian determinant, is it invertible?",
             "Open even for n = 2",
             "Many false proofs have been published — it is notoriously tricky",
             "Smale included it in his list of 18 great problems for the 21st century",
         ],
         implications=[
             "Fundamental question about polynomial automorphisms",
         ],
         common_errors=[
             "Claiming the Jacobian conjecture is solved (it is not — there are many false proofs)",
             "Confusing the inverse function theorem (local) with the Jacobian conjecture (global)",
         ])

    _add("Hadamard conjecture",
         status="OPEN", year=1893, domain="Algebra/Topology",
         key_facts=[
             "A Hadamard matrix of order n exists for every n divisible by 4",
             "Hadamard matrices are n x n matrices with entries +/-1 and maximal determinant",
             "Known to exist for many values but not all multiples of 4",
             "Smallest unresolved order is 668",
         ],
         implications=[
             "Relevant to error-correcting codes, signal processing, and combinatorial design theory",
         ],
         common_errors=[
             "Claiming all Hadamard matrices have been constructed (they have not)",
             "Confusing Hadamard matrices with Hadamard's inequality (which IS proved)",
         ])

    _add("Invariant subspace problem",
         status="OPEN", year=1949, domain="Algebra/Topology",
         key_facts=[
             "Does every bounded linear operator on an infinite-dimensional separable Hilbert space have a non-trivial closed invariant subspace?",
             "Solved for Banach spaces: Enflo (1976/1987) and Read (1985) constructed counterexamples",
             "For Hilbert spaces, still open",
             "Lomonosov (1973) proved it for operators commuting with a compact operator",
         ],
         implications=[
             "Fundamental to the structure theory of operators on Hilbert spaces",
         ],
         common_errors=[
             "Claiming the invariant subspace problem is solved for Hilbert spaces (only solved for general Banach spaces)",
             "Confusing the Banach space counterexamples with the Hilbert space case",
         ])

    _add("Kervaire invariant one problem",
         status="MOSTLY_SOLVED", year=1960, domain="Algebra/Topology",
         solver="Hill-Hopkins-Ravenel", solver_year=2009,
         key_facts=[
             "Asks in which dimensions n there exist framed manifolds with Kervaire invariant one",
             "Hill-Hopkins-Ravenel (2009) proved no such manifolds exist in dimensions n > 126",
             "Known to exist in dimensions 2, 6, 14, 30, 62",
             "Dimension 126 remains open — the sole unsettled case",
         ],
         implications=[
             "Resolved a major problem in stable homotopy theory",
             "Uses equivariant stable homotopy theory and the slice spectral sequence",
         ],
         common_errors=[
             "Claiming it is fully solved (dimension 126 is still open)",
             "Claiming it is open (it is resolved in all dimensions except 126)",
         ])

    _add("Baum-Connes conjecture",
         status="OPEN", year=1982, domain="Algebra/Topology",
         key_facts=[
             "Relates K-theory of the reduced C*-algebra of a group to equivariant K-homology",
             "Proved for many classes of groups (a-T-menable, hyperbolic, etc.)",
             "With coefficients, counterexamples exist (Higson-Lafforgue-Skandalis, 2002)",
             "Without coefficients, still open in general",
         ],
         implications=[
             "Implies the Novikov conjecture and the Kadison-Kaplansky conjecture",
         ],
         common_errors=[
             "Claiming Baum-Connes is fully proved (only for specific group classes)",
             "Confusing Baum-Connes (no coefficients) with Baum-Connes with coefficients (counterexamples exist)",
         ])

    _add("Borel conjecture",
         status="OPEN", year=1953, domain="Algebra/Topology",
         key_facts=[
             "Two closed aspherical manifolds that are homotopy equivalent are homeomorphic",
             "Proved in dimensions 1, 2, and for many higher-dimensional cases",
             "Closely related to the Farrell-Jones conjecture",
         ],
         implications=[
             "Would show that topology of aspherical manifolds is determined by their fundamental group",
         ],
         common_errors=[
             "Claiming Borel is solved in general (only for specific classes of manifolds)",
         ])

    _add("Smooth Poincare conjecture in dimension 4",
         status="OPEN", year=1904, domain="Algebra/Topology",
         key_facts=[
             "Is every smooth homotopy 4-sphere diffeomorphic to the standard S^4?",
             "The topological version (Freedman 1982) is proved — every topological homotopy 4-sphere is homeomorphic to S^4",
             "The smooth version is uniquely open in dimension 4 — resolved in all other dimensions",
             "Related to the existence of exotic 4-spheres",
         ],
         implications=[
             "Would determine whether exotic 4-spheres exist",
         ],
         common_errors=[
             "Confusing with the (now proved) Poincare conjecture in dimension 3",
             "Confusing topological and smooth categories — topological is proved, smooth is open",
         ])

    _add("Andrews-Curtis conjecture",
         status="OPEN", year=1965, domain="Algebra/Topology",
         key_facts=[
             "Every balanced presentation of the trivial group can be reduced to the trivial presentation by Andrews-Curtis moves",
             "Potential counterexamples exist but none are proven",
             "Related to the smooth Poincare conjecture in dimension 4",
         ],
         implications=[
             "Relevant to 4-manifold topology",
         ],
         common_errors=[
             "Claiming it is solved (it is not)",
         ])

    _add("Novikov conjecture",
         status="OPEN", year=1965, domain="Algebra/Topology",
         key_facts=[
             "Higher signatures of a closed oriented manifold are homotopy invariants",
             "Proved for many classes of groups (hyperbolic, amenable, linear, etc.)",
             "Would follow from the Baum-Connes conjecture",
         ],
         implications=[
             "Fundamental to surgery theory and manifold classification",
         ],
         common_errors=[
             "Claiming Novikov conjecture is fully proved (only for many group classes, not all)",
         ])

    _add("Zeeman conjecture",
         status="OPEN", year=1963, domain="Algebra/Topology",
         key_facts=[
             "If K is a contractible 2-complex then K x I is collapsible",
             "Implies the Poincare conjecture (now proved by other means) and the Andrews-Curtis conjecture",
         ],
         implications=[
             "Would imply Andrews-Curtis conjecture",
         ],
         common_errors=[
             "Claiming it is solved (it is not, even though the Poincare conjecture it implies IS solved)",
         ])

    # ── Proof Techniques / Logic ─────────────────────────────────────────

    _add("Continuum hypothesis",
         status="INDEPENDENT", year=1878, domain="Proof Techniques",
         key_facts=[
             "There is no set whose cardinality is strictly between that of the integers and the reals",
             "Godel (1940) proved it is consistent with ZFC (cannot be disproved)",
             "Cohen (1963) proved it is independent of ZFC (cannot be proved either)",
             "Neither true nor false in standard set theory — it is independent",
         ],
         implications=[
             "Demonstrated the limits of ZFC set theory",
             "Sparked development of forcing and large cardinal axioms",
         ],
         common_errors=[
             "Claiming CH is true or false (it is independent of ZFC — neither provable nor disprovable)",
             "Claiming CH is 'unsolvable' (it is resolved — its independence IS the resolution)",
             "Treating independence as meaning we don't know (we DO know — it's formally independent)",
         ])

    _add("P = BPP?",
         status="CONJECTURED_TRUE", year=1975, domain="Proof Techniques",
         key_facts=[
             "Widely believed that P = BPP (randomness does not help for decision problems)",
             "Impagliazzo-Wigderson (1997): if E requires exponential circuits, then P = BPP",
             "Nisan-Wigderson generators provide conditional derandomization",
             "NOT yet proved unconditionally",
         ],
         implications=[
             "Would mean every randomized polynomial-time algorithm can be derandomized",
         ],
         common_errors=[
             "Claiming P = BPP is proved (it is conjectured but not proved)",
             "Claiming P != BPP (most experts believe they are equal)",
         ])

    _add("Axiom of constructibility (V=L)",
         status="INDEPENDENT", year=1938, domain="Proof Techniques",
         key_facts=[
             "Godel's constructible universe L: V=L states every set is constructible",
             "Consistent with ZFC (Godel 1938) but not provable from ZFC",
             "Implies CH and GCH",
             "Most set theorists consider V=L too restrictive",
         ],
         implications=[
             "If adopted, would resolve CH (true) and many other set-theoretic questions",
         ],
         common_errors=[
             "Claiming V=L is the standard axiom (most set theorists reject it as too restrictive)",
         ])

    _add("Woodin's Ultimate-L",
         status="OPEN", year=2010, domain="Proof Techniques",
         key_facts=[
             "Woodin's program to find a canonical inner model compatible with large cardinals",
             "Aims to resolve CH (would imply CH is false in Ultimate-L framework)",
             "Still a research program, not a completed theory",
         ],
         implications=[
             "Could provide a principled resolution of CH beyond mere independence",
         ],
         common_errors=[
             "Claiming Ultimate-L resolves CH (the program is not complete)",
         ])

    _add("Erdos discrepancy problem",
         status="SOLVED", year=1932, domain="Proof Techniques",
         solver="Terence Tao", solver_year=2015,
         key_facts=[
             "For any sequence of +1 and -1, the partial sums along any arithmetic progression are unbounded",
             "Tao's proof uses the Elliott conjecture on correlations of multiplicative functions",
             "Konev-Lisitsa (2014) gave a computer-assisted proof for discrepancy 2",
         ],
         implications=[
             "Settled a long-standing combinatorial problem",
         ],
         common_errors=[
             "Claiming it is still open (proved by Tao in 2015)",
         ])

    _add("Boolean Pythagorean triples problem",
         status="SOLVED", year=1980, domain="Proof Techniques",
         solver="Heule-Kullmann-Marek", solver_year=2016,
         key_facts=[
             "The set {1, ..., 7824} can be 2-colored with no monochromatic Pythagorean triple, but {1, ..., 7825} cannot",
             "Proved by SAT solver — the proof is 200 terabytes (largest mathematical proof at the time)",
             "The number 7825 is the exact threshold",
         ],
         implications=[
             "Demonstrated the power of SAT solving for combinatorial problems",
         ],
         common_errors=[
             "Claiming the proof is human-readable (it is a computer proof, 200 TB)",
         ])

    _add("Large cardinal hierarchy consistency",
         status="OPEN", year=1930, domain="Proof Techniques",
         key_facts=[
             "Whether ZFC + various large cardinal axioms is consistent",
             "Cannot be proved within ZFC itself (Godel's second incompleteness theorem)",
             "Empirical consistency: no contradictions found despite decades of work",
             "Forms a well-ordered hierarchy of consistency strength",
         ],
         implications=[
             "Underpins much of modern set theory and proof theory",
         ],
         common_errors=[
             "Claiming large cardinal consistency is proved (it cannot be, by Godel's theorem)",
             "Confusing 'no contradiction found' with 'proved consistent'",
         ])

    _add("NP = co-NP?",
         status="OPEN", year=1971, domain="Proof Techniques",
         key_facts=[
             "Whether every problem whose NO instances can be verified efficiently also has efficiently verifiable YES instances",
             "If P = NP then NP = co-NP (but not conversely)",
             "Most experts believe NP != co-NP",
         ],
         implications=[
             "If NP = co-NP, tautologies would have short proofs",
         ],
         common_errors=[
             "Claiming NP = co-NP is resolved (it is not)",
             "Confusing with P vs NP",
         ])

    _add("Unique games conjecture",
         status="OPEN", year=2002, domain="Proof Techniques",
         key_facts=[
             "For every epsilon > 0, there exists a label set size such that approximating Unique Games within some constant is NP-hard",
             "Proposed by Subhash Khot in 2002",
             "If true, implies optimal inapproximability results for many problems (Max-Cut, Vertex Cover, etc.)",
             "Not universally believed — some experts think it might be false",
         ],
         implications=[
             "Would establish tight inapproximability bounds for many optimization problems",
             "Would show the Goemans-Williamson algorithm is optimal for Max-Cut",
         ],
         common_errors=[
             "Claiming UGC is proved (it is a conjecture, not a theorem)",
             "Claiming UGC is refuted (it is open)",
         ])

    _add("Inverse Galois problem",
         status="OPEN", year=1832, domain="Proof Techniques",
         key_facts=[
             "Is every finite group the Galois group of some extension of Q?",
             "Known for: all solvable groups, symmetric groups S_n, alternating groups A_n, most sporadic simple groups",
             "Open for some families of finite simple groups of Lie type",
         ],
         implications=[
             "Would complete the connection between field extensions and group theory",
         ],
         common_errors=[
             "Claiming it is solved (it is not, for all finite groups)",
         ])

    _add("Vaught's conjecture",
         status="OPEN", year=1961, domain="Proof Techniques",
         key_facts=[
             "A countable first-order theory has either countably many or 2^aleph_0 countable models",
             "Morley (1970) proved it for omega-stable theories",
             "No counterexample known; some set-theoretic approaches exist",
         ],
         implications=[
             "Would clarify the structure of countable models in model theory",
         ],
         common_errors=[
             "Claiming Vaught's conjecture is proved (it is not in full generality)",
         ])

    # ── Analysis / PDE ───────────────────────────────────────────────────

    _add("Kakeya conjecture",
         status="OPEN", year=1917, domain="Analysis/PDE",
         key_facts=[
             "A Besicovitch set (containing a unit line segment in every direction) in R^n has Hausdorff dimension n",
             "Proved in dimension 2 (Davies, 1971)",
             "Open in dimensions 3 and higher",
             "Hong Wang and Joshua Zahl (2025) proved the 3D case",
             "Related to restriction conjectures in harmonic analysis",
         ],
         implications=[
             "Would resolve several open problems in harmonic analysis and PDE",
             "Connected to the restriction conjecture for the Fourier transform",
         ],
         common_errors=[
             "Claiming Kakeya is fully solved in all dimensions (Wang-Zahl 2025 proved dimension 3; higher dimensions open)",
             "Confusing the Kakeya needle problem (measure zero sets exist) with the Kakeya conjecture (about Hausdorff dimension)",
             "Not knowing the 2D case is solved",
         ])

    _add("Carleson's theorem",
         status="SOLVED", year=1913, domain="Analysis/PDE",
         solver="Lennart Carleson", solver_year=1966,
         key_facts=[
             "The Fourier series of an L^2 function converges pointwise almost everywhere",
             "Proved by Carleson in 1966, extending to L^p for p > 1 by Hunt (1968)",
             "The question (Lusin's conjecture) was open since 1913",
             "For L^1 functions, convergence can fail (Kolmogorov 1926 counterexample)",
         ],
         implications=[
             "Settled a fundamental question about Fourier analysis",
         ],
         common_errors=[
             "Claiming Carleson's theorem is still a conjecture (it is proved)",
             "Claiming Fourier series converge pointwise for all L^1 functions (false — only L^p for p > 1)",
         ])

    _add("Navier-Stokes regularity",
         status="OPEN", year=2000, domain="Analysis/PDE",
         prize="$1M",
         key_facts=[
             "Same as Navier-Stokes existence and smoothness (Millennium Problem)",
             "Asks whether 3D Navier-Stokes has global smooth solutions or admits finite-time blowup",
             "2D case is completely resolved (global regularity proved)",
         ],
         implications=[
             "Same as Navier-Stokes existence and smoothness",
         ],
         common_errors=[
             "Claiming NS regularity is solved (it is not)",
             "Confusing 2D (solved) with 3D (open)",
         ])

    _add("Schanuel's conjecture",
         status="OPEN", year=1962, domain="Analysis/PDE",
         key_facts=[
             "If z_1, ..., z_n are Q-linearly independent complex numbers, then the transcendence degree of {z_1,...,z_n, e^z_1,...,e^z_n} over Q is >= n",
             "Implies the Lindemann-Weierstrass theorem and the Gelfond-Schneider theorem",
             "Would resolve many open problems about transcendental numbers",
         ],
         implications=[
             "Would prove e + pi is transcendental",
             "Would prove e * pi is transcendental",
             "Implies the four exponentials conjecture",
         ],
         common_errors=[
             "Claiming Schanuel's conjecture is proved (it is not)",
         ])

    _add("Sendov's conjecture",
         status="OPEN", year=1958, domain="Analysis/PDE",
         key_facts=[
             "For a polynomial with all roots in the unit disk, each root has a critical point within distance 1",
             "Proved for degrees up to 8 (Brown-Xiang 1999) and for roots on the unit circle",
             "Tao (2022) proved it for sufficiently high degree polynomials",
         ],
         implications=[
             "Would complete our understanding of the geometry of polynomial roots and critical points",
         ],
         common_errors=[
             "Claiming it is fully solved (Tao proved it only for sufficiently large degree; small degrees still open in full generality)",
         ])

    _add("Brennan conjecture",
         status="OPEN", year=1978, domain="Analysis/PDE",
         key_facts=[
             "For a conformal map f on a simply connected domain, integral of |f'|^p is finite for -2 < p < 2/3",
             "Known for -2 < p < 1.752 (Hedenmalm-Shimorin 2005)",
         ],
         implications=[
             "Fundamental to geometric function theory",
         ],
         common_errors=[
             "Claiming it is solved (it is not for the full range)",
         ])

    _add("Hilbert-Polya conjecture",
         status="OPEN", year=1914, domain="Analysis/PDE",
         key_facts=[
             "The nontrivial zeros of the Riemann zeta function correspond to eigenvalues of a self-adjoint operator",
             "Would provide a spectral proof of the Riemann Hypothesis",
             "Berry-Keating conjecture suggests the operator is related to xp (position times momentum)",
             "Connections to random matrix theory (Montgomery-Odlyzko, GUE statistics)",
         ],
         implications=[
             "Would prove the Riemann Hypothesis via operator theory",
         ],
         common_errors=[
             "Confusing this with a proof of RH (it is a strategy, not a proof)",
             "Claiming the operator has been found (it has not)",
         ])

    _add("Erdos-Ginzburg-Ziv theorem extensions",
         status="OPEN", year=1961, domain="Analysis/PDE",
         key_facts=[
             "The original EGZ theorem is proved: among any 2n-1 integers, some n have sum divisible by n",
             "Higher-dimensional generalizations (Davenport constant for finite groups) remain partially open",
         ],
         implications=[
             "Fundamental to additive combinatorics",
         ],
         common_errors=[
             "Confusing the proved EGZ theorem with its open generalizations",
         ])

    _add("Littlewood conjecture",
         status="OPEN", year=1930, domain="Analysis/PDE",
         key_facts=[
             "For any two real numbers alpha, beta: lim inf n * ||n*alpha|| * ||n*beta|| = 0",
             "Where ||.|| is the distance to the nearest integer",
             "Einsiedler-Katok-Lindenstrauss (2006) proved counterexamples have Hausdorff dimension zero (if they exist)",
         ],
         implications=[
             "Fundamental to Diophantine approximation",
         ],
         common_errors=[
             "Claiming Littlewood is solved (it is not — only partial results exist)",
         ])

    _add("Mean curvature flow regularity",
         status="OPEN", year=1978, domain="Analysis/PDE",
         key_facts=[
             "Classification of all possible singularities in mean curvature flow",
             "Huisken (1984) proved convex hypersurfaces shrink to round points",
             "Generic singularities classified by Colding-Minicozzi",
             "Full singularity classification remains open",
         ],
         implications=[
             "Would complete the theory of geometric flows for hypersurfaces",
         ],
         common_errors=[
             "Claiming full singularity classification is done (it is not)",
         ])

    _add("Hot spots conjecture",
         status="OPEN", year=1974, domain="Analysis/PDE",
         key_facts=[
             "The hottest point of a convex domain (for the Neumann heat equation) eventually moves to the boundary",
             "Proved for certain classes of domains (acute triangles, lip domains)",
             "Open in full generality for convex domains",
         ],
         implications=[
             "Connects spectral geometry to physical heat diffusion",
         ],
         common_errors=[
             "Claiming it is fully proved (only for special domain classes)",
         ])

    # ── Computational ────────────────────────────────────────────────────

    _add("Graph isomorphism in P?",
         status="OPEN", year=1971, domain="Computational",
         key_facts=[
             "Is there a polynomial-time algorithm for graph isomorphism?",
             "Babai (2015) proved a quasipolynomial-time algorithm: exp(O((log n)^c))",
             "Not known to be NP-complete (and widely believed not to be)",
             "Polynomial for many special graph classes (planar, bounded degree, etc.)",
         ],
         implications=[
             "Would settle the complexity of a fundamental problem between P and NP-complete",
         ],
         common_errors=[
             "Claiming graph isomorphism is in P (Babai's result is quasipolynomial, not polynomial)",
             "Claiming graph isomorphism is NP-complete (very unlikely and no evidence for it)",
             "Confusing quasipolynomial time with polynomial time",
         ])

    _add("Separation of complexity classes (P vs PSPACE)",
         status="OPEN", year=1972, domain="Computational",
         key_facts=[
             "Is P strictly contained in PSPACE?",
             "Known: P is in NP is in PSPACE, but no separations are proved",
             "P != PSPACE would follow from P != NP but is also independently open",
             "PSPACE = IP (Shamir, 1992)",
         ],
         implications=[
             "Would establish that more space genuinely helps computation",
         ],
         common_errors=[
             "Claiming P != PSPACE is proved (it is not)",
             "Confusing with P vs NP (which is a weaker separation)",
         ])

    _add("Algebraic complexity: VP vs VNP",
         status="OPEN", year=1979, domain="Computational",
         key_facts=[
             "Valiant's algebraic analogue of P vs NP",
             "VP (algebraic P) vs VNP (algebraic NP): is the permanent harder than the determinant?",
             "Separation over finite fields would separate P from #P in Boolean complexity",
         ],
         implications=[
             "Would prove that the permanent is inherently harder than the determinant",
         ],
         common_errors=[
             "Claiming VP != VNP is proved (it is not)",
             "Confusing with the Boolean P vs NP problem",
         ])

    _add("Minimum circuit size problem (MCSP)",
         status="OPEN", year=1956, domain="Computational",
         key_facts=[
             "Given a truth table and a size parameter, is there a circuit of that size computing the function?",
             "Not known to be NP-complete, despite being in NP",
             "Connections to pseudorandomness, natural proofs barrier, and learning theory",
         ],
         implications=[
             "If NP-complete, would imply no natural proofs for strong circuit lower bounds",
         ],
         common_errors=[
             "Claiming MCSP is known to be NP-complete (it is not)",
         ])

    _add("Polynomial identity testing derandomization",
         status="OPEN", year=1980, domain="Computational",
         key_facts=[
             "Can we deterministically test if a polynomial given by an arithmetic circuit is identically zero?",
             "Schwartz-Zippel gives a randomized polynomial-time algorithm",
             "Deterministic PIT would imply circuit lower bounds (Kabanets-Impagliazzo 2004)",
         ],
         implications=[
             "Would yield explicit circuit lower bounds against NEXP",
         ],
         common_errors=[
             "Claiming deterministic PIT exists (only randomized algorithms are known)",
         ])

    _add("Planted clique conjecture",
         status="OPEN", year=1995, domain="Computational",
         key_facts=[
             "No polynomial-time algorithm can find a planted clique of size O(sqrt(n)) in a random graph",
             "Used as a hardness assumption for many average-case reductions",
             "No proof of hardness exists",
         ],
         implications=[
             "If true, implies hardness of many statistical estimation problems",
         ],
         common_errors=[
             "Claiming planted clique is proved hard (it is a conjecture, not a theorem)",
         ])

    _add("Khot's d-to-1 conjecture",
         status="OPEN", year=2002, domain="Computational",
         key_facts=[
             "Stronger version of the Unique Games Conjecture",
             "Would imply NP-hardness of coloring a 3-colorable graph with O(1) colors",
         ],
         implications=[
             "Would settle the approximate graph coloring problem",
         ],
         common_errors=[
             "Confusing with UGC (d-to-1 is stronger)",
         ])

    _add("BPP vs SUBEXP",
         status="OPEN", year=1997, domain="Computational",
         key_facts=[
             "Can randomized polynomial time be simulated in subexponential deterministic time?",
             "Impagliazzo-Wigderson: if E requires exponential-size circuits, then BPP = P",
             "Unconditional result unknown",
         ],
         implications=[
             "Would clarify the power of randomness in computation",
         ],
         common_errors=[
             "Claiming this is resolved (it is not unconditionally)",
         ])

    _add("Computational Goldbach",
         status="OPEN", year=2000, domain="Computational",
         key_facts=[
             "Is there a polynomial-time algorithm to find the Goldbach decomposition of a given even number?",
             "Assuming Goldbach conjecture is true, naive search is exponential",
             "Heuristic algorithms work in practice but have no worst-case guarantee",
         ],
         implications=[
             "Would connect additive number theory to computational complexity",
         ],
         common_errors=[
             "Confusing with the Goldbach conjecture itself (this is about algorithms, not existence)",
         ])

    _add("Matrix multiplication exponent",
         status="OPEN", year=1969, domain="Computational",
         key_facts=[
             "What is the exponent omega of matrix multiplication? Known: 2 <= omega < 2.372",
             "Strassen (1969) showed omega < 2.81",
             "Best current bound: omega < 2.3719 (Alman-Vassilevska Williams, Duan-Wu-Zhou)",
             "Conjectured that omega = 2 but far from proved",
         ],
         implications=[
             "Optimal matrix multiplication would accelerate many algorithms in linear algebra and graph theory",
         ],
         common_errors=[
             "Claiming omega = 2 is proved (it is conjectured, current best bound is ~2.372)",
         ])

    _add("Existential Theory of the Reals complete problems",
         status="OPEN", year=1988, domain="Computational",
         key_facts=[
             "ETR is between NP and PSPACE in complexity",
             "Complete problems include: realizability of abstract order types, stretchability of pseudolines",
             "ETR = NP would imply NP = co-NP (unlikely)",
         ],
         implications=[
             "Would clarify the complexity landscape between NP and PSPACE",
         ],
         common_errors=[
             "Claiming ETR problems are NP-complete (they are ETR-complete, a potentially harder class)",
         ])

    _add("NC vs P",
         status="OPEN", year=1979, domain="Computational",
         key_facts=[
             "Are all polynomial-time problems efficiently parallelizable?",
             "NC: problems solvable in polylogarithmic time with polynomially many processors",
             "P-complete problems (e.g., circuit value problem) are believed not in NC",
             "Separation would show inherent sequential bottlenecks exist",
         ],
         implications=[
             "Would formalize the limits of parallel computation",
         ],
         common_errors=[
             "Claiming NC = P is proved (it is not)",
             "Claiming NC != P is proved (it is not)",
         ])

    return db


# Module-level database (loaded once)
_DB = _build_database()


# ─── Alias map for fuzzy matching ─────────────────────────────────────────────

_ALIASES = {
    "rh": "riemann hypothesis",
    "riemann": "riemann hypothesis",
    "p vs np": "p vs np",
    "p != np": "p vs np",
    "p=np": "p vs np",
    "pvsnp": "p vs np",
    "navier stokes": "navier-stokes existence and smoothness",
    "navier-stokes": "navier-stokes existence and smoothness",
    "ns regularity": "navier-stokes regularity",
    "yang mills": "yang-mills existence and mass gap",
    "yang-mills": "yang-mills existence and mass gap",
    "hodge": "hodge conjecture",
    "bsd": "birch and swinnerton-dyer conjecture",
    "birch swinnerton-dyer": "birch and swinnerton-dyer conjecture",
    "poincare": "poincare conjecture",
    "goldbach": "goldbach conjecture",
    "twin primes": "twin prime conjecture",
    "twin prime": "twin prime conjecture",
    "collatz": "collatz conjecture",
    "abc": "abc conjecture",
    "legendre": "legendre conjecture",
    "catalan": "catalan conjecture",
    "flt": "fermat's last theorem",
    "fermat": "fermat's last theorem",
    "jacobian": "jacobian conjecture",
    "hadamard": "hadamard conjecture",
    "invariant subspace": "invariant subspace problem",
    "kervaire": "kervaire invariant one problem",
    "continuum": "continuum hypothesis",
    "ch": "continuum hypothesis",
    "kakeya": "kakeya conjecture",
    "carleson": "carleson's theorem",
    "graph isomorphism": "graph isomorphism in p?",
    "gi": "graph isomorphism in p?",
    "schanuel": "schanuel's conjecture",
    "sendov": "sendov's conjecture",
    "ugc": "unique games conjecture",
    "unique games": "unique games conjecture",
    "smooth poincare": "smooth poincare conjecture in dimension 4",
    "baum-connes": "baum-connes conjecture",
    "baum connes": "baum-connes conjecture",
    "borel": "borel conjecture",
    "novikov": "novikov conjecture",
    "andrews-curtis": "andrews-curtis conjecture",
    "hot spots": "hot spots conjecture",
    "littlewood": "littlewood conjecture",
    "matrix multiplication": "matrix multiplication exponent",
    "omega": "matrix multiplication exponent",
    "inverse galois": "inverse galois problem",
    "vaught": "vaught's conjecture",
    "planted clique": "planted clique conjecture",
    "nc vs p": "nc vs p",
    "vp vs vnp": "algebraic complexity: vp vs vnp",
    "hilbert-polya": "hilbert-polya conjecture",
    "hilbert polya": "hilbert-polya conjecture",
    "p = bpp": "p = bpp?",
    "p=bpp": "p = bpp?",
    "bpp": "p = bpp?",
    "mcsp": "minimum circuit size problem (mcsp)",
    "pit": "polynomial identity testing derandomization",
    "erdos discrepancy": "erdos discrepancy problem",
    "erdos-straus": "erdos-straus conjecture",
    "brocard": "brocard's problem",
    "bunyakovsky": "bunyakovsky conjecture",
    "brennan": "brennan conjecture",
    "zeeman": "zeeman conjecture",
}


def _resolve_name(name: str) -> Optional[ConjectureInfo]:
    """Resolve a conjecture name with fuzzy matching."""
    key = name.lower().strip()

    # Direct match
    if key in _DB:
        return _DB[key]

    # Alias match
    if key in _ALIASES:
        return _DB.get(_ALIASES[key])

    # Substring match — find all conjectures whose name contains the query
    matches = [info for db_key, info in _DB.items() if key in db_key]
    if len(matches) == 1:
        return matches[0]

    # Reverse substring — query contains the conjecture name
    matches = [info for db_key, info in _DB.items() if db_key in key]
    if len(matches) == 1:
        return matches[0]

    # Word overlap match
    query_words = set(key.split())
    best_match = None
    best_overlap = 0
    for db_key, info in _DB.items():
        db_words = set(db_key.split())
        overlap = len(query_words & db_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = info
    if best_overlap >= 2:
        return best_match

    return None


# ─── Issue and report dataclasses ─────────────────────────────────────────────

@dataclass
class ConjectureIssue:
    """A single issue found when checking a conjecture claim."""
    check_type: str          # STATUS_CHECK, IMPLICATION_CHECK, PARTIAL_RESULT_CHECK, CONTROVERSY_CHECK
    severity: str            # HIGH, MODERATE, LOW, INFO
    description: str
    details: Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        return f"  [{self.severity}] {self.check_type}: {self.description}"


@dataclass
class ConjectureReport:
    """Result of check_conjecture() or check_claim()."""
    verdict: str                          # PASS, WARN, or FAIL
    conjecture: Optional[ConjectureInfo]
    claimed_status: Optional[str]
    actual_status: Optional[str]
    issues: List[ConjectureIssue]
    notes: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        if self.conjecture:
            lines.append(f"  Conjecture Status Check: {self.verdict}")
            lines.append(f"{'=' * 60}")
            lines.append(f"  Conjecture: {self.conjecture.name}")
            lines.append(f"  Actual status: {self.actual_status}")
            if self.claimed_status:
                lines.append(f"  Claimed status: {self.claimed_status}")
        else:
            lines.append(f"  Conjecture Status Check: {self.verdict}")
            lines.append(f"{'=' * 60}")
            lines.append("  Conjecture not found in database")
        lines.append("")

        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in sorted(self.issues,
                                key=lambda i: {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}.get(i.severity, 4)):
                lines.append(str(issue))
            lines.append("")

        if self.notes:
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    - {note}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# Alias for natural-language claim checking
ClaimReport = ConjectureReport


# ─── Status normalization ─────────────────────────────────────────────────────

_STATUS_SYNONYMS = {
    "OPEN": "OPEN",
    "UNSOLVED": "OPEN",
    "UNPROVEN": "OPEN",
    "UNKNOWN": "OPEN",
    "SOLVED": "SOLVED",
    "PROVED": "SOLVED",
    "PROVEN": "SOLVED",
    "RESOLVED": "SOLVED",
    "TRUE": "SOLVED",
    "FALSE": "SOLVED",
    "DISPUTED": "DISPUTED",
    "CONTROVERSIAL": "DISPUTED",
    "CONTESTED": "DISPUTED",
    "INDEPENDENT": "INDEPENDENT",
    "UNDECIDABLE": "INDEPENDENT",
    "MOSTLY_SOLVED": "MOSTLY_SOLVED",
    "MOSTLY SOLVED": "MOSTLY_SOLVED",
    "PARTIALLY_SOLVED": "MOSTLY_SOLVED",
    "PARTIALLY SOLVED": "MOSTLY_SOLVED",
    "CONJECTURED_TRUE": "CONJECTURED_TRUE",
    "CONJECTURED TRUE": "CONJECTURED_TRUE",
    "BELIEVED_TRUE": "CONJECTURED_TRUE",
    "LIKELY_TRUE": "CONJECTURED_TRUE",
}


def _normalize_status(status: str) -> Optional[str]:
    """Normalize a claimed status string."""
    key = status.strip().upper().replace("-", "_")
    return _STATUS_SYNONYMS.get(key)


# ─── Check functions ──────────────────────────────────────────────────────────

def check_conjecture(name: str, claimed_status: str = None) -> ConjectureReport:
    """Look up a conjecture and optionally validate a claimed status.

    Args:
        name: conjecture name (fuzzy matched — "RH", "Riemann", etc.)
        claimed_status: optional claimed status to validate (e.g., "SOLVED", "OPEN")

    Returns:
        ConjectureReport with verdict, issues, and notes.
    """
    info = _resolve_name(name)
    if info is None:
        return ConjectureReport(
            verdict="FAIL",
            conjecture=None,
            claimed_status=claimed_status,
            actual_status=None,
            issues=[ConjectureIssue(
                check_type="STATUS_CHECK",
                severity="HIGH",
                description=f"Conjecture '{name}' not found in database",
            )],
            notes=[f"Known conjectures: use list_conjectures() to see all {len(_DB)} entries"],
        )

    issues = []
    notes = []
    actual_status = info.status

    # Add key facts as notes
    for fact in info.key_facts[:3]:
        notes.append(fact)

    if claimed_status is not None:
        normalized = _normalize_status(claimed_status)
        if normalized is None:
            issues.append(ConjectureIssue(
                check_type="STATUS_CHECK",
                severity="MODERATE",
                description=f"Unrecognized status '{claimed_status}'. "
                            f"Valid statuses: OPEN, SOLVED, DISPUTED, INDEPENDENT, MOSTLY_SOLVED, CONJECTURED_TRUE",
            ))
        else:
            # ── STATUS_CHECK ─────────────────────────────────────────
            if normalized != actual_status:
                # Determine severity based on how wrong the claim is
                if actual_status == "OPEN" and normalized == "SOLVED":
                    severity = "HIGH"
                    desc = (f"Claimed SOLVED but {info.name} is OPEN. "
                            f"No accepted proof exists as of 2025.")
                elif actual_status == "SOLVED" and normalized == "OPEN":
                    severity = "HIGH"
                    solver_info = f" by {info.solver} ({info.solver_year})" if info.solver else ""
                    desc = (f"Claimed OPEN but {info.name} is SOLVED"
                            f"{solver_info}.")
                elif actual_status == "DISPUTED" and normalized == "SOLVED":
                    severity = "MODERATE"
                    desc = (f"Claimed SOLVED but {info.name} is DISPUTED. "
                            f"A proof has been claimed but is not widely accepted.")
                elif actual_status == "INDEPENDENT" and normalized in ("SOLVED", "OPEN"):
                    severity = "HIGH" if normalized == "SOLVED" else "MODERATE"
                    desc = (f"Claimed {normalized} but {info.name} is INDEPENDENT of ZFC. "
                            f"It can be neither proved nor disproved in standard set theory.")
                elif actual_status == "MOSTLY_SOLVED" and normalized == "SOLVED":
                    severity = "MODERATE"
                    desc = (f"Claimed fully SOLVED but {info.name} is MOSTLY_SOLVED. "
                            f"Some cases remain open.")
                elif actual_status == "CONJECTURED_TRUE" and normalized == "SOLVED":
                    severity = "HIGH"
                    desc = (f"Claimed SOLVED but {info.name} is only CONJECTURED_TRUE. "
                            f"Widely believed but no unconditional proof exists.")
                else:
                    severity = "MODERATE"
                    desc = (f"Claimed {normalized} but actual status is {actual_status}.")

                issues.append(ConjectureIssue(
                    check_type="STATUS_CHECK",
                    severity=severity,
                    description=desc,
                    details={"claimed": normalized, "actual": actual_status},
                ))

            # ── CONTROVERSY_CHECK ────────────────────────────────────
            if actual_status == "DISPUTED":
                issues.append(ConjectureIssue(
                    check_type="CONTROVERSY_CHECK",
                    severity="MODERATE",
                    description=f"{info.name} has a disputed proof. "
                                f"Any reference should note the controversy.",
                    details={"status": "DISPUTED"},
                ))

            # ── PARTIAL_RESULT_CHECK ─────────────────────────────────
            if actual_status == "MOSTLY_SOLVED" and normalized == "SOLVED":
                issues.append(ConjectureIssue(
                    check_type="PARTIAL_RESULT_CHECK",
                    severity="MODERATE",
                    description=f"Partial result being presented as full proof. "
                                f"{info.name} has open cases remaining.",
                ))

    else:
        # No claimed status — just provide info and flag controversy
        if actual_status == "DISPUTED":
            issues.append(ConjectureIssue(
                check_type="CONTROVERSY_CHECK",
                severity="INFO",
                description=f"{info.name} has a disputed proof — note the controversy.",
            ))

    # ── Verdict logic ────────────────────────────────────────────────
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return ConjectureReport(
        verdict=verdict,
        conjecture=info,
        claimed_status=claimed_status,
        actual_status=actual_status,
        issues=issues,
        notes=notes,
    )


def check_claim(claim: str) -> ConjectureReport:
    """Parse a natural-language claim about a conjecture and validate it.

    Handles claims like:
        "the Riemann Hypothesis was proved in 2018"
        "Goldbach conjecture is solved"
        "P != NP was shown by Deolalikar"
        "the continuum hypothesis is true"
        "ABC conjecture was proved by Mochizuki"

    Args:
        claim: natural-language claim string

    Returns:
        ConjectureReport (aliased as ClaimReport) with verdict and issues.
    """
    claim_lower = claim.lower().strip()

    # ── Extract claimed status from the claim text ───────────────────
    claimed_status = None

    solved_patterns = [
        r"\bproved\b", r"\bproven\b", r"\bsolved\b", r"\bresolved\b",
        r"\bdemonstrated\b", r"\bestablished\b", r"\bshown to be true\b",
        r"\bconfirmed\b", r"\bhas been proved\b", r"\bwas proved\b",
    ]
    open_patterns = [
        r"\bunsolved\b", r"\bunproven\b", r"\bremains open\b",
        r"\bstill open\b", r"\bnot proved\b", r"\bnot proven\b",
    ]
    false_patterns = [
        r"\bdisproved\b", r"\brefuted\b", r"\bshown to be false\b",
        r"\bfalsified\b",
    ]
    true_patterns = [
        r"\bis true\b", r"\bwas true\b",
    ]
    false_literal_patterns = [
        r"\bis false\b", r"\bwas false\b",
    ]
    independent_patterns = [
        r"\bindependent\b", r"\bundecidable\b",
    ]

    for pat in solved_patterns:
        if re.search(pat, claim_lower):
            claimed_status = "SOLVED"
            break
    if claimed_status is None:
        for pat in open_patterns:
            if re.search(pat, claim_lower):
                claimed_status = "OPEN"
                break
    if claimed_status is None:
        for pat in false_patterns:
            if re.search(pat, claim_lower):
                claimed_status = "SOLVED"  # disproved = resolved
                break
    if claimed_status is None:
        for pat in independent_patterns:
            if re.search(pat, claim_lower):
                claimed_status = "INDEPENDENT"
                break
    if claimed_status is None:
        for pat in true_patterns:
            if re.search(pat, claim_lower):
                claimed_status = "SOLVED"
                break
    if claimed_status is None:
        for pat in false_literal_patterns:
            if re.search(pat, claim_lower):
                claimed_status = "SOLVED"
                break

    # ── Extract conjecture name ──────────────────────────────────────
    # Try each known conjecture name against the claim
    best_match = None
    best_len = 0
    for db_key, info in _DB.items():
        # Check if the full name appears in the claim
        if info.name.lower() in claim_lower:
            if len(info.name) > best_len:
                best_match = info
                best_len = len(info.name)

    # Try aliases
    if best_match is None:
        for alias, target_key in sorted(_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in claim_lower:
                best_match = _DB.get(target_key)
                if best_match:
                    break

    # Try each DB key as a substring
    if best_match is None:
        for db_key, info in sorted(_DB.items(), key=lambda x: -len(x[0])):
            if db_key in claim_lower:
                best_match = info
                break

    if best_match is None:
        return ConjectureReport(
            verdict="FAIL",
            conjecture=None,
            claimed_status=claimed_status,
            actual_status=None,
            issues=[ConjectureIssue(
                check_type="STATUS_CHECK",
                severity="HIGH",
                description=f"Could not identify a conjecture in the claim: '{claim}'",
            )],
            notes=["Try using a more specific conjecture name"],
        )

    # ── Run check_conjecture with extracted info ─────────────────────
    report = check_conjecture(best_match.name, claimed_status=claimed_status)

    # ── Additional checks from claim text ────────────────────────────

    # IMPLICATION_CHECK — detect false implication claims
    implication_patterns = [
        (r"riemann.{0,20}implies.{0,20}p\s*[!=]=?\s*np", "RH does not imply P != NP (or P = NP). They are unrelated problems."),
        (r"p\s*[!=]=?\s*np.{0,20}implies.{0,20}riemann", "P vs NP does not imply the Riemann Hypothesis. They are unrelated."),
        (r"goldbach.{0,20}implies.{0,20}twin\s*prime", "Goldbach conjecture does not imply the twin prime conjecture."),
        (r"collatz.{0,20}implies.{0,20}riemann", "The Collatz conjecture has no known implication for the Riemann Hypothesis."),
    ]
    for pat, desc in implication_patterns:
        if re.search(pat, claim_lower):
            report.issues.append(ConjectureIssue(
                check_type="IMPLICATION_CHECK",
                severity="HIGH",
                description=desc,
            ))

    # PARTIAL_RESULT_CHECK — detect confusion of partial with full
    partial_patterns = [
        (r"goldbach.{0,30}(solved|proved|proven)", "goldbach conjecture",
         "Helfgott (2013) proved WEAK Goldbach (3 primes for odd numbers). "
         "The strong Goldbach conjecture (2 primes for even numbers) is OPEN."),
        (r"twin\s*prime.{0,30}(solved|proved|proven)", "twin prime conjecture",
         "Zhang (2013) proved bounded gaps (<=246 by Maynard-Tao). "
         "Twin primes (gap = 2) specifically is still OPEN."),
        (r"collatz.{0,30}(solved|proved|proven)", "collatz conjecture",
         "Tao (2019) proved 'almost all' orbits reach values close to 1. "
         "The full Collatz conjecture for ALL starting values is still OPEN."),
        (r"bsd.{0,30}(solved|proved|proven)", "birch and swinnerton-dyer conjecture",
         "Only rank 0 and rank 1 cases are proved (Gross-Zagier, Kolyvagin). "
         "The full BSD conjecture for rank >= 2 is still OPEN."),
    ]
    for pat, conj_key, desc in partial_patterns:
        if re.search(pat, claim_lower) and best_match.name.lower() == _ALIASES.get(conj_key, conj_key):
            # Only add if not already caught by STATUS_CHECK
            if not any(i.check_type == "PARTIAL_RESULT_CHECK" for i in report.issues):
                report.issues.append(ConjectureIssue(
                    check_type="PARTIAL_RESULT_CHECK",
                    severity="HIGH",
                    description=desc,
                ))

    # Recalculate verdict after adding extra issues
    has_high = any(i.severity == "HIGH" for i in report.issues)
    has_moderate = any(i.severity == "MODERATE" for i in report.issues)
    if has_high:
        report.verdict = "FAIL"
    elif has_moderate:
        report.verdict = "WARN"

    return report


def list_conjectures(status: str = None) -> List[str]:
    """List all conjecture names, optionally filtered by status.

    Args:
        status: optional filter — one of OPEN, SOLVED, DISPUTED, INDEPENDENT,
                MOSTLY_SOLVED, CONJECTURED_TRUE. Case-insensitive.

    Returns:
        Sorted list of conjecture names.
    """
    results = []
    filter_status = status.upper().strip() if status else None
    for info in _DB.values():
        if filter_status is None or info.status == filter_status:
            results.append(info.name)
    return sorted(results)


def get_conjecture(name: str) -> Optional[ConjectureInfo]:
    """Get full information about a conjecture.

    Args:
        name: conjecture name (fuzzy matched).

    Returns:
        ConjectureInfo if found, None otherwise.
    """
    return _resolve_name(name)
