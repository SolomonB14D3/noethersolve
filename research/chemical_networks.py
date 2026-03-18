#!/usr/bin/env python3
"""
Chemical Reaction Network Conservation Laws

Stoichiometric compatibility classes define conserved quantities:
linear combinations of concentrations that stay constant under mass-action kinetics.

These come from the LEFT null space of the stoichiometry matrix S:
  If w^T S = 0, then d(w·c)/dt = w^T S v = 0  (conserved)

For complex networks (glycolysis, 20+ species), these are non-obvious.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Reaction:
    """A chemical reaction with reactants, products, and rate constant."""
    reactants: Dict[str, int]  # species -> stoichiometric coefficient
    products: Dict[str, int]
    k: float  # rate constant

@dataclass
class ChemicalNetwork:
    """A chemical reaction network."""
    name: str
    species: List[str]
    reactions: List[Reaction]

    def stoichiometry_matrix(self) -> np.ndarray:
        """Build stoichiometry matrix S where S[i,j] = net change in species i from reaction j."""
        n_species = len(self.species)
        n_reactions = len(self.reactions)
        S = np.zeros((n_species, n_reactions))

        for j, rxn in enumerate(self.reactions):
            for species, coef in rxn.reactants.items():
                i = self.species.index(species)
                S[i, j] -= coef
            for species, coef in rxn.products.items():
                i = self.species.index(species)
                S[i, j] += coef

        return S

    def conservation_laws(self) -> Tuple[np.ndarray, List[str]]:
        """Find conservation laws from left null space of S."""
        S = self.stoichiometry_matrix()
        # Left null space: vectors w where w^T S = 0
        # Equivalent to null space of S^T
        null = null_space(S.T)

        # Clean up near-zero entries
        null[np.abs(null) < 1e-10] = 0

        # Describe each conservation law
        descriptions = []
        for i in range(null.shape[1]):
            w = null[:, i]
            terms = []
            for j, coef in enumerate(w):
                if abs(coef) > 1e-10:
                    if abs(coef - 1.0) < 1e-10:
                        terms.append(f"[{self.species[j]}]")
                    elif abs(coef + 1.0) < 1e-10:
                        terms.append(f"-[{self.species[j]}]")
                    else:
                        terms.append(f"{coef:.3f}[{self.species[j]}]")
            descriptions.append(" + ".join(terms) if terms else "trivial")

        return null, descriptions

    def mass_action_rhs(self, t: float, c: np.ndarray) -> np.ndarray:
        """Right-hand side for mass-action ODE: dc/dt = S @ v(c)."""
        S = self.stoichiometry_matrix()
        v = np.zeros(len(self.reactions))

        for j, rxn in enumerate(self.reactions):
            rate = rxn.k
            for species, coef in rxn.reactants.items():
                i = self.species.index(species)
                rate *= max(0, c[i]) ** coef  # mass action
            v[j] = rate

        return S @ v

    def integrate(self, c0: np.ndarray, T: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate mass-action kinetics."""
        sol = solve_ivp(
            self.mass_action_rhs,
            (0, T),
            c0,
            t_eval=np.arange(0, T, dt),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        return sol.t, sol.y.T  # shape: (n_times, n_species)

    def check_conservation(self, c0: np.ndarray, T: float = 10.0) -> Dict:
        """Check which linear combinations are conserved."""
        null, descriptions = self.conservation_laws()
        t, c_history = self.integrate(c0, T)

        results = {
            "network": self.name,
            "n_species": len(self.species),
            "n_reactions": len(self.reactions),
            "n_conservation_laws": null.shape[1],
            "laws": []
        }

        for i in range(null.shape[1]):
            w = null[:, i]
            Q = c_history @ w  # conserved quantity over time
            mean_Q = np.mean(Q)
            frac_var = np.std(Q) / abs(mean_Q) if abs(mean_Q) > 1e-10 else 0

            results["laws"].append({
                "description": descriptions[i],
                "weights": w.tolist(),
                "mean": mean_Q,
                "frac_var": frac_var,
                "conserved": frac_var < 1e-6
            })

        return results


# ============================================================================
# Example Networks
# ============================================================================

def simple_AB_reaction() -> ChemicalNetwork:
    """Simple A <-> B reversible reaction."""
    return ChemicalNetwork(
        name="A_B_reversible",
        species=["A", "B"],
        reactions=[
            Reaction({"A": 1}, {"B": 1}, k=1.0),  # A -> B
            Reaction({"B": 1}, {"A": 1}, k=0.5),  # B -> A
        ]
    )


def enzyme_kinetics() -> ChemicalNetwork:
    """Michaelis-Menten enzyme kinetics: E + S <-> ES -> E + P"""
    return ChemicalNetwork(
        name="Michaelis_Menten",
        species=["E", "S", "ES", "P"],
        reactions=[
            Reaction({"E": 1, "S": 1}, {"ES": 1}, k=1.0),   # E + S -> ES
            Reaction({"ES": 1}, {"E": 1, "S": 1}, k=0.5),   # ES -> E + S
            Reaction({"ES": 1}, {"E": 1, "P": 1}, k=0.1),   # ES -> E + P
        ]
    )


def lotka_volterra() -> ChemicalNetwork:
    """Lotka-Volterra predator-prey as chemical network."""
    return ChemicalNetwork(
        name="Lotka_Volterra",
        species=["X", "Y"],  # prey, predator
        reactions=[
            Reaction({"X": 1}, {"X": 2}, k=1.0),      # X -> 2X (prey birth)
            Reaction({"X": 1, "Y": 1}, {"Y": 2}, k=0.5),  # X + Y -> 2Y (predation)
            Reaction({"Y": 1}, {}, k=0.3),            # Y -> 0 (predator death)
        ]
    )


def glycolysis_simplified() -> ChemicalNetwork:
    """Simplified glycolysis pathway (10 species)."""
    # Glucose -> G6P -> F6P -> FBP -> DHAP + G3P -> ... -> Pyruvate
    return ChemicalNetwork(
        name="Glycolysis_Simplified",
        species=["Glc", "G6P", "F6P", "FBP", "DHAP", "G3P", "BPG", "PG3", "PEP", "Pyr",
                 "ATP", "ADP", "NAD", "NADH"],
        reactions=[
            # Hexokinase: Glc + ATP -> G6P + ADP
            Reaction({"Glc": 1, "ATP": 1}, {"G6P": 1, "ADP": 1}, k=0.5),
            # Phosphoglucose isomerase: G6P <-> F6P
            Reaction({"G6P": 1}, {"F6P": 1}, k=1.0),
            Reaction({"F6P": 1}, {"G6P": 1}, k=0.8),
            # Phosphofructokinase: F6P + ATP -> FBP + ADP
            Reaction({"F6P": 1, "ATP": 1}, {"FBP": 1, "ADP": 1}, k=0.3),
            # Aldolase: FBP <-> DHAP + G3P
            Reaction({"FBP": 1}, {"DHAP": 1, "G3P": 1}, k=0.2),
            Reaction({"DHAP": 1, "G3P": 1}, {"FBP": 1}, k=0.1),
            # Triose phosphate isomerase: DHAP <-> G3P
            Reaction({"DHAP": 1}, {"G3P": 1}, k=2.0),
            Reaction({"G3P": 1}, {"DHAP": 1}, k=1.5),
            # GAPDH: G3P + NAD -> BPG + NADH
            Reaction({"G3P": 1, "NAD": 1}, {"BPG": 1, "NADH": 1}, k=0.4),
            # Phosphoglycerate kinase: BPG + ADP -> PG3 + ATP
            Reaction({"BPG": 1, "ADP": 1}, {"PG3": 1, "ATP": 1}, k=0.5),
            # Phosphoglycerate mutase + enolase: PG3 -> PEP
            Reaction({"PG3": 1}, {"PEP": 1}, k=0.6),
            # Pyruvate kinase: PEP + ADP -> Pyr + ATP
            Reaction({"PEP": 1, "ADP": 1}, {"Pyr": 1, "ATP": 1}, k=0.4),
        ]
    )


def brusselator() -> ChemicalNetwork:
    """Brusselator oscillating reaction."""
    return ChemicalNetwork(
        name="Brusselator",
        species=["A", "B", "X", "Y", "D", "E"],
        reactions=[
            Reaction({"A": 1}, {"X": 1}, k=1.0),           # A -> X
            Reaction({"X": 2, "Y": 1}, {"X": 3}, k=1.0),   # 2X + Y -> 3X
            Reaction({"B": 1, "X": 1}, {"Y": 1, "D": 1}, k=1.0),  # B + X -> Y + D
            Reaction({"X": 1}, {"E": 1}, k=1.0),           # X -> E
        ]
    )


def robertson_stiff() -> ChemicalNetwork:
    """Robertson's stiff chemical system (classic test problem)."""
    return ChemicalNetwork(
        name="Robertson_Stiff",
        species=["A", "B", "C"],
        reactions=[
            Reaction({"A": 1}, {"B": 1}, k=0.04),
            Reaction({"B": 2}, {"B": 1, "C": 1}, k=3e7),
            Reaction({"B": 1, "C": 1}, {"A": 1, "C": 1}, k=1e4),
        ]
    )


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("="*70)
    print("Chemical Reaction Network Conservation Laws")
    print("="*70)
    print()
    print("Testing stoichiometric conservation (null space of S^T)")
    print()

    networks = [
        (simple_AB_reaction(), np.array([1.0, 0.0])),
        (enzyme_kinetics(), np.array([1.0, 5.0, 0.0, 0.0])),
        (lotka_volterra(), np.array([2.0, 1.0])),
        (glycolysis_simplified(), np.array([
            1.0,  # Glc
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # intermediates
            5.0, 5.0,  # ATP, ADP
            3.0, 0.0   # NAD, NADH
        ])),
        (brusselator(), np.array([1.0, 2.0, 1.0, 1.0, 0.0, 0.0])),
    ]

    all_results = []

    for network, c0 in networks:
        print(f"\n{'='*60}")
        print(f"Network: {network.name}")
        print(f"{'='*60}")
        print(f"Species: {network.species}")
        print(f"Reactions: {len(network.reactions)}")

        # Compute conservation laws
        null, descriptions = network.conservation_laws()
        print("\nConservation laws (from null space of S^T):")
        for i, desc in enumerate(descriptions):
            print(f"  {i+1}. {desc}")

        if null.shape[1] == 0:
            print("  (no conservation laws)")

        # Verify numerically
        print("\nNumerical verification (T=10.0):")
        results = network.check_conservation(c0, T=10.0)
        all_results.append(results)

        for law in results["laws"]:
            status = "CONSERVED" if law["conserved"] else "BROKEN"
            print(f"  {law['description'][:40]:40s} frac_var={law['frac_var']:.2e} [{status}]")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"{'Network':<25} {'Species':>8} {'Reactions':>10} {'Cons. Laws':>12} {'All OK?'}")
    print("-"*70)

    for results in all_results:
        all_ok = all(law["conserved"] for law in results["laws"]) if results["laws"] else True
        status = "✓" if all_ok else "✗"
        print(f"{results['network']:<25} {results['n_species']:>8} {results['n_reactions']:>10} "
              f"{results['n_conservation_laws']:>12} {status:>7}")

    return all_results


if __name__ == "__main__":
    results = main()
