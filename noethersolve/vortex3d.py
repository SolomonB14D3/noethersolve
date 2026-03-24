"""3D vortex filament dynamics and Q_f conservation checker.

Implements the Biot-Savart law for 3D vortex filaments (thin tubes),
integrates the dynamics via RK45, and checks conservation of:
- Energy (Hamiltonian)
- Helicity
- Linear impulse
- Q_f family: Q_f = sum_ij Gamma_i Gamma_j f(r_ij)
  with emphasis on the scaling-critical kernel f(r) = r^{-2}

The key question: is Q_{r^{-2}} approximately conserved?
If yes, it bridges the "impossible triangle" for NS regularity.
"""

from dataclasses import dataclass
import math
import numpy as np
from typing import Callable


@dataclass
class VortexFilament:
    """A single vortex filament in 3D, discretized as a polygon of points."""
    points: np.ndarray  # (N, 3) array of filament centerline points
    circulation: float = 1.0
    core_radius: float = 0.01  # regularization parameter


@dataclass
class Vortex3DState:
    """State of a 3D vortex system."""
    filaments: list  # list of VortexFilament
    time: float = 0.0


@dataclass
class ConservationReport:
    """Report on conservation of a quantity over a trajectory."""
    name: str
    values: np.ndarray
    times: np.ndarray
    frac_var: float
    drift: float
    initial: float
    final: float

    def __str__(self):
        status = "PASS" if self.frac_var < 5e-3 else "FAIL"
        return (f"{self.name:<30} frac_var={self.frac_var:.2e}  "
                f"drift={self.drift:.2e}  [{status}]")


def biot_savart_velocity(point: np.ndarray, filaments: list,
                         exclude_self: bool = False,
                         self_filament_idx: int = -1,
                         self_point_idx: int = -1) -> np.ndarray:
    """Compute velocity at a point due to all vortex filaments via Biot-Savart.

    v(x) = sum_filaments (Gamma / 4pi) * integral dl' x (x - x') / |x - x'|^3

    Uses regularized kernel: |x - x'|^2 + delta^2 to avoid singularity.
    """
    v = np.zeros(3)

    for f_idx, fil in enumerate(filaments):
        gamma = fil.circulation
        pts = fil.points
        n = len(pts)
        delta2 = fil.core_radius ** 2

        for i in range(n):
            j = (i + 1) % n

            # Skip self-interaction for the segment containing the point
            if exclude_self and f_idx == self_filament_idx:
                if i == self_point_idx or j == self_point_idx:
                    continue

            dl = pts[j] - pts[i]  # tangent vector
            midpoint = 0.5 * (pts[i] + pts[j])
            r = point - midpoint
            r_mag2 = np.dot(r, r) + delta2
            r_mag3 = r_mag2 ** 1.5

            # Biot-Savart: (Gamma / 4pi) * dl x r / |r|^3
            cross = np.cross(dl, r)
            v += (gamma / (4 * math.pi)) * cross / r_mag3

    return v


def compute_velocities(state: Vortex3DState) -> list:
    """Compute velocity at every filament point."""
    velocities = []
    for f_idx, fil in enumerate(state.filaments):
        fil_vels = np.zeros_like(fil.points)
        for p_idx in range(len(fil.points)):
            fil_vels[p_idx] = biot_savart_velocity(
                fil.points[p_idx], state.filaments,
                exclude_self=True, self_filament_idx=f_idx,
                self_point_idx=p_idx
            )
        velocities.append(fil_vels)
    return velocities


def compute_energy(state: Vortex3DState) -> float:
    """Compute kinetic energy (regularized Biot-Savart integral).

    E = -(1/4pi) sum_{i!=j} Gamma_i Gamma_j integral integral
        dl_i . dl_j / |x_i - x_j|

    For discrete filaments, use segment-segment interaction.
    """
    energy = 0.0
    filaments = state.filaments

    for i, fil_i in enumerate(filaments):
        for j, fil_j in enumerate(filaments):
            if i >= j and len(filaments) > 1:
                continue  # avoid double counting inter-filament

            pts_i = fil_i.points
            pts_j = fil_j.points
            gi = fil_i.circulation
            gj = fil_j.circulation
            ni = len(pts_i)
            nj = len(pts_j)
            delta2 = max(fil_i.core_radius, fil_j.core_radius) ** 2

            for si in range(ni):
                si_next = (si + 1) % ni
                dl_i = pts_i[si_next] - pts_i[si]
                mid_i = 0.5 * (pts_i[si] + pts_i[si_next])

                for sj in range(nj):
                    if i == j and abs(si - sj) <= 1:
                        continue  # skip self and adjacent segments

                    sj_next = (sj + 1) % nj
                    dl_j = pts_j[sj_next] - pts_j[sj]
                    mid_j = 0.5 * (pts_j[sj] + pts_j[sj_next])

                    r = mid_i - mid_j
                    r_mag = math.sqrt(np.dot(r, r) + delta2)

                    energy += gi * gj * np.dot(dl_i, dl_j) / r_mag

    return -energy / (4 * math.pi)


def compute_helicity(state: Vortex3DState) -> float:
    """Compute helicity H = integral u . omega dx.

    For thin filaments: H = sum_i Gamma_i * integral u . dl_i
    where the integral is the circulation-weighted velocity flux.
    """
    helicity = 0.0

    for f_idx, fil in enumerate(state.filaments):
        pts = fil.points
        n = len(pts)
        gamma = fil.circulation

        for i in range(n):
            j = (i + 1) % n
            dl = pts[j] - pts[i]
            mid = 0.5 * (pts[i] + pts[j])

            # Velocity at midpoint (from all filaments)
            v = biot_savart_velocity(mid, state.filaments,
                                     exclude_self=True,
                                     self_filament_idx=f_idx,
                                     self_point_idx=i)
            helicity += gamma * np.dot(v, dl)

    return helicity


def compute_impulse(state: Vortex3DState) -> np.ndarray:
    """Compute linear impulse P = (1/2) sum_i Gamma_i integral x_i x dl_i."""
    impulse = np.zeros(3)

    for fil in state.filaments:
        pts = fil.points
        n = len(pts)
        gamma = fil.circulation

        for i in range(n):
            j = (i + 1) % n
            dl = pts[j] - pts[i]
            mid = 0.5 * (pts[i] + pts[j])
            impulse += 0.5 * gamma * np.cross(mid, dl)

    return impulse


def compute_qf(state: Vortex3DState, f: Callable[[float], float]) -> float:
    """Compute Q_f = sum_{i,j} Gamma_i Gamma_j f(r_ij).

    For filaments, this becomes a double integral over all segment pairs.
    r_ij is the distance between segment midpoints.
    """
    qf = 0.0
    filaments = state.filaments

    for i, fil_i in enumerate(filaments):
        for j, fil_j in enumerate(filaments):
            pts_i = fil_i.points
            pts_j = fil_j.points
            gi = fil_i.circulation
            gj = fil_j.circulation
            ni = len(pts_i)
            nj = len(pts_j)
            delta2 = max(fil_i.core_radius, fil_j.core_radius) ** 2

            for si in range(ni):
                si_next = (si + 1) % ni
                mid_i = 0.5 * (pts_i[si] + pts_i[si_next])
                len_i = np.linalg.norm(pts_i[si_next] - pts_i[si])

                for sj in range(nj):
                    if i == j and abs(si - sj) <= 1:
                        continue

                    sj_next = (sj + 1) % nj
                    mid_j = 0.5 * (pts_j[sj] + pts_j[sj_next])
                    len_j = np.linalg.norm(pts_j[sj_next] - pts_j[sj])

                    r = np.linalg.norm(mid_i - mid_j)
                    r_reg = math.sqrt(r ** 2 + delta2)

                    qf += gi * gj * f(r_reg) * len_i * len_j

    return qf


def make_vortex_ring(center: np.ndarray, radius: float, normal: np.ndarray,
                     circulation: float = 1.0, n_points: int = 32,
                     core_radius: float = 0.01) -> VortexFilament:
    """Create a circular vortex ring."""
    normal = normal / np.linalg.norm(normal)

    # Find two perpendicular vectors in the plane
    if abs(normal[2]) < 0.9:
        e1 = np.cross(normal, np.array([0, 0, 1]))
    else:
        e1 = np.cross(normal, np.array([1, 0, 0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    theta = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    points = np.array([
        center + radius * (math.cos(t) * e1 + math.sin(t) * e2)
        for t in theta
    ])

    return VortexFilament(points=points, circulation=circulation,
                          core_radius=core_radius)


def integrate_rk4(state: Vortex3DState, dt: float, n_steps: int,
                  record_every: int = 1) -> list:
    """Integrate vortex filament dynamics using RK4.

    Returns list of (time, state) snapshots.
    """
    trajectory = [(state.time, _copy_state(state))]

    for step in range(n_steps):
        # RK4 stages
        k1 = compute_velocities(state)
        s2 = _advance_state(state, k1, dt / 2)
        k2 = compute_velocities(s2)
        s3 = _advance_state(state, k2, dt / 2)
        k3 = compute_velocities(s3)
        s4 = _advance_state(state, k3, dt)
        k4 = compute_velocities(s4)

        # Update positions
        for f_idx, fil in enumerate(state.filaments):
            for p_idx in range(len(fil.points)):
                fil.points[p_idx] += (dt / 6) * (
                    k1[f_idx][p_idx] + 2 * k2[f_idx][p_idx] +
                    2 * k3[f_idx][p_idx] + k4[f_idx][p_idx]
                )
        state.time += dt

        if (step + 1) % record_every == 0:
            trajectory.append((state.time, _copy_state(state)))

    return trajectory


def _copy_state(state: Vortex3DState) -> Vortex3DState:
    new_fils = []
    for fil in state.filaments:
        new_fils.append(VortexFilament(
            points=fil.points.copy(),
            circulation=fil.circulation,
            core_radius=fil.core_radius
        ))
    return Vortex3DState(filaments=new_fils, time=state.time)


def _advance_state(state: Vortex3DState, velocities: list,
                   dt: float) -> Vortex3DState:
    new_state = _copy_state(state)
    for f_idx, fil in enumerate(new_state.filaments):
        for p_idx in range(len(fil.points)):
            fil.points[p_idx] += dt * velocities[f_idx][p_idx]
    new_state.time = state.time + dt
    return new_state


def check_conservation(trajectory: list,
                       quantity_fn: Callable,
                       name: str) -> ConservationReport:
    """Check conservation of a quantity along a trajectory."""
    times = np.array([t for t, _ in trajectory])
    values = np.array([quantity_fn(s) for _, s in trajectory])

    if isinstance(values[0], np.ndarray):
        # Vector quantity: check each component
        mag = np.linalg.norm(values, axis=1)
        mean_val = np.mean(mag)
        frac_var = np.std(mag) / abs(mean_val) if abs(mean_val) > 1e-15 else 0
        drift = (mag[-1] - mag[0]) / abs(mean_val) if abs(mean_val) > 1e-15 else 0
        return ConservationReport(
            name=name, values=mag, times=times,
            frac_var=frac_var, drift=drift,
            initial=float(mag[0]), final=float(mag[-1])
        )

    mean_val = np.mean(values)
    frac_var = np.std(values) / abs(mean_val) if abs(mean_val) > 1e-15 else 0
    drift = (values[-1] - values[0]) / abs(mean_val) if abs(mean_val) > 1e-15 else 0

    return ConservationReport(
        name=name, values=values, times=times,
        frac_var=frac_var, drift=drift,
        initial=float(values[0]), final=float(values[-1])
    )


def run_conservation_check(initial_conditions: str = "two_rings",
                           dt: float = 0.01, n_steps: int = 200,
                           n_points: int = 24) -> str:
    """Run a full conservation check on a standard initial condition.

    initial_conditions: "two_rings", "trefoil", "leapfrog"

    Returns a formatted report of all conserved quantities including Q_f family.
    """
    # Set up initial conditions
    if initial_conditions == "two_rings":
        ring1 = make_vortex_ring(
            center=np.array([0, 0, 0]), radius=1.0,
            normal=np.array([0, 0, 1]), circulation=1.0,
            n_points=n_points, core_radius=0.1
        )
        ring2 = make_vortex_ring(
            center=np.array([0, 0, 2]), radius=1.0,
            normal=np.array([0, 0, 1]), circulation=1.0,
            n_points=n_points, core_radius=0.1
        )
        state = Vortex3DState(filaments=[ring1, ring2])

    elif initial_conditions == "leapfrog":
        ring1 = make_vortex_ring(
            center=np.array([0, 0, 0]), radius=1.0,
            normal=np.array([0, 0, 1]), circulation=1.0,
            n_points=n_points, core_radius=0.1
        )
        ring2 = make_vortex_ring(
            center=np.array([0, 0, 0.5]), radius=0.8,
            normal=np.array([0, 0, 1]), circulation=1.0,
            n_points=n_points, core_radius=0.1
        )
        state = Vortex3DState(filaments=[ring1, ring2])

    elif initial_conditions == "trefoil":
        # Trefoil knot parametrization
        t = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
        points = np.column_stack([
            np.sin(t) + 2 * np.sin(2 * t),
            np.cos(t) - 2 * np.cos(2 * t),
            -np.sin(3 * t)
        ]) * 0.5
        fil = VortexFilament(points=points, circulation=1.0, core_radius=0.15)
        state = Vortex3DState(filaments=[fil])

    else:
        return f"Unknown IC: {initial_conditions}. Options: two_rings, leapfrog, trefoil"

    # Integrate
    trajectory = integrate_rk4(state, dt=dt, n_steps=n_steps, record_every=10)

    # Check conservation of standard quantities
    reports = []
    reports.append(check_conservation(trajectory, compute_energy, "Energy (H)"))
    reports.append(check_conservation(trajectory, compute_helicity, "Helicity"))
    reports.append(check_conservation(trajectory, compute_impulse, "|Impulse|"))

    # Q_f family
    kernels = {
        "Q_{r^{-2}} (CRITICAL)": lambda s: compute_qf(s, lambda r: r ** (-2)),
        "Q_{r^{-1}} (energy-like)": lambda s: compute_qf(s, lambda r: r ** (-1)),
        "Q_{-ln(r)} (2D optimal)": lambda s: compute_qf(s, lambda r: -math.log(r)),
        "Q_{r^{-3}} (supercritical)": lambda s: compute_qf(s, lambda r: r ** (-3)),
        "Q_{r^{-0.5}} (subcritical)": lambda s: compute_qf(s, lambda r: r ** (-0.5)),
        "Q_{r^2} (stretch-resistant)": lambda s: compute_qf(s, lambda r: r ** 2),
    }

    for name, fn in kernels.items():
        reports.append(check_conservation(trajectory, fn, name))

    # Format report
    lines = [
        f"{'=' * 60}",
        "  3D Vortex Filament Conservation Check",
        f"  IC: {initial_conditions}, dt={dt}, steps={n_steps}, points/ring={n_points}",
        f"{'=' * 60}",
        "",
    ]

    for r in reports:
        lines.append(str(r))

    lines.append("")
    lines.append("frac_var < 5e-3 → PASS (approximately conserved)")
    lines.append("frac_var < 1e-6 → near-exact conservation")

    # Highlight the critical kernel
    crit = [r for r in reports if "CRITICAL" in r.name][0]
    lines.append("")
    lines.append(f"*** KEY RESULT: Q_{{r^{{-2}}}} frac_var = {crit.frac_var:.2e} ***")
    if crit.frac_var < 5e-3:
        lines.append("Q_{r^{-2}} IS approximately conserved → potential regularity bridge")
    else:
        lines.append("Q_{r^{-2}} is NOT conserved → characterizes departure from regularity")
        lines.append(f"  Drift rate: {crit.drift:.4f} per unit time")

    return "\n".join(lines)
