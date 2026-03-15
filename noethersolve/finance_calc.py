"""Finance calculator — derives answers from first principles.

Covers Black-Scholes option pricing, put-call parity, present value,
and 2x2 game theory (Nash equilibrium).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BlackScholesReport:
    """Result of Black-Scholes option pricing."""
    option_type: str  # call or put
    price: float
    delta: float
    gamma: float
    theta: float  # per day
    vega: float  # per 1% vol change
    rho: float  # per 1% rate change
    d1: float
    d2: float
    S: float  # spot price
    K: float  # strike
    T: float  # time to expiry (years)
    r: float  # risk-free rate
    sigma: float  # volatility

    def __str__(self) -> str:
        lines = [
            f"Black-Scholes {self.option_type.upper()} Option:",
            f"  Price = {self.price:.4f}",
            f"  Greeks:",
            f"    Delta = {self.delta:.4f}",
            f"    Gamma = {self.gamma:.6f}",
            f"    Theta = {self.theta:.4f} /day",
            f"    Vega  = {self.vega:.4f} /1% vol",
            f"    Rho   = {self.rho:.4f} /1% rate",
            f"  Parameters: S={self.S}, K={self.K}, T={self.T}y, r={self.r}, σ={self.sigma}",
            f"  d1={self.d1:.4f}, d2={self.d2:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class PutCallParityReport:
    """Result of put-call parity check."""
    call_price: float
    put_price: float
    S: float
    K: float
    r: float
    T: float
    pv_strike: float  # present value of strike
    lhs: float  # C - P
    rhs: float  # S - PV(K)
    parity_holds: bool
    arbitrage_opportunity: Optional[str]

    def __str__(self) -> str:
        lines = [
            f"Put-Call Parity Check:",
            f"  C - P = {self.lhs:.4f}",
            f"  S - PV(K) = {self.rhs:.4f}",
            f"  Difference = {abs(self.lhs - self.rhs):.4f}",
            f"  Parity holds: {self.parity_holds}",
        ]
        if self.arbitrage_opportunity:
            lines.append(f"  Arbitrage: {self.arbitrage_opportunity}")
        return "\n".join(lines)


@dataclass
class NashEquilibriumReport:
    """Result of Nash equilibrium calculation for a 2x2 game."""
    pure_equilibria: List[Tuple[int, int]]
    mixed_equilibrium: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    dominant_strategies: List[str]
    game_type: str  # prisoners_dilemma, coordination, battle_of_sexes, etc.
    payoff_matrix_p1: List[List[float]]
    payoff_matrix_p2: List[List[float]]

    def __str__(self) -> str:
        lines = [
            f"Nash Equilibrium Analysis ({self.game_type}):",
            f"  Pure NE: {self.pure_equilibria if self.pure_equilibria else 'none'}",
        ]
        if self.mixed_equilibrium:
            p1, p2 = self.mixed_equilibrium
            lines.append(f"  Mixed NE: P1=({p1[0]:.3f}, {p1[1]:.3f}), P2=({p2[0]:.3f}, {p2[1]:.3f})")
        if self.dominant_strategies:
            lines.append(f"  Dominant strategies: {', '.join(self.dominant_strategies)}")
        return "\n".join(lines)


@dataclass
class PresentValueReport:
    """Result of present/future value calculation."""
    present_value: float
    future_value: float
    rate: float  # per period
    periods: int
    compounding: str  # discrete or continuous
    effective_annual_rate: Optional[float]

    def __str__(self) -> str:
        lines = [
            f"Time Value of Money:",
            f"  Present Value = {self.present_value:.2f}",
            f"  Future Value  = {self.future_value:.2f}",
            f"  Rate = {self.rate*100:.2f}% per period",
            f"  Periods = {self.periods}",
            f"  Compounding: {self.compounding}",
        ]
        if self.effective_annual_rate is not None:
            lines.append(f"  Effective annual rate = {self.effective_annual_rate*100:.4f}%")
        return "\n".join(lines)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> BlackScholesReport:
    """Price a European option using Black-Scholes formula.

    Assumes log-normal prices with constant volatility (the model's
    core assumption — a common LLM weak spot).

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
        sigma: Volatility (annualized, e.g., 0.20 for 20%)
        option_type: "call" or "put"

    Returns:
        BlackScholesReport with price and all Greeks.
    """
    if S <= 0 or K <= 0:
        raise ValueError("Stock and strike prices must be positive")
    if T <= 0:
        raise ValueError("Time to expiry must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")

    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1

    # Greeks (same for call and put except delta and rho)
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
    theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
             - r * K * math.exp(-r * T) * _norm_cdf(d2 if option_type == "call" else -d2)
             * (1 if option_type == "call" else -1))
    theta_daily = theta / 365.0
    vega = S * _norm_pdf(d1) * sqrt_T / 100  # per 1% vol
    if option_type == "call":
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100
    else:
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100

    return BlackScholesReport(
        option_type=option_type,
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta_daily,
        vega=vega,
        rho=rho,
        d1=d1,
        d2=d2,
        S=S, K=K, T=T, r=r, sigma=sigma,
    )


def put_call_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    tolerance: float = 0.01,
) -> PutCallParityReport:
    """Check put-call parity: C - P = S - PV(K).

    Put-call parity relates call, put, stock, and risk-free bond prices
    (a common LLM weak spot — models often say it relates only calls to stocks).

    Args:
        call_price: Market call option price
        put_price: Market put option price
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        T: Time to expiry in years
        tolerance: Acceptable deviation

    Returns:
        PutCallParityReport with arbitrage analysis.
    """
    pv_k = K * math.exp(-r * T)
    lhs = call_price - put_price
    rhs = S - pv_k
    diff = lhs - rhs
    holds = abs(diff) <= tolerance

    arb = None
    if not holds:
        if diff > tolerance:
            arb = (f"Call overpriced relative to put by {diff:.4f}. "
                   "Strategy: sell call, buy put + stock, borrow PV(K).")
        else:
            arb = (f"Put overpriced relative to call by {-diff:.4f}. "
                   "Strategy: sell put, short stock, buy call, lend PV(K).")

    return PutCallParityReport(
        call_price=call_price,
        put_price=put_price,
        S=S, K=K, r=r, T=T,
        pv_strike=pv_k,
        lhs=lhs, rhs=rhs,
        parity_holds=holds,
        arbitrage_opportunity=arb,
    )


def nash_equilibrium_2x2(
    payoff_matrix_p1: List[List[float]],
    payoff_matrix_p2: List[List[float]],
) -> NashEquilibriumReport:
    """Find Nash equilibria of a 2x2 game.

    Args:
        payoff_matrix_p1: 2x2 payoff matrix for player 1.
            payoff_matrix_p1[i][j] = P1's payoff when P1 plays i, P2 plays j.
        payoff_matrix_p2: 2x2 payoff matrix for player 2.

    Returns:
        NashEquilibriumReport with pure and mixed equilibria.
    """
    if len(payoff_matrix_p1) != 2 or len(payoff_matrix_p1[0]) != 2:
        raise ValueError("Must be 2x2 payoff matrices")
    if len(payoff_matrix_p2) != 2 or len(payoff_matrix_p2[0]) != 2:
        raise ValueError("Must be 2x2 payoff matrices")

    a = payoff_matrix_p1
    b = payoff_matrix_p2

    # Find pure Nash equilibria
    pure_ne = []
    for i in range(2):
        for j in range(2):
            # Check if i is best response for P1 given P2 plays j
            p1_br = a[i][j] >= a[1 - i][j]
            # Check if j is best response for P2 given P1 plays i
            p2_br = b[i][j] >= b[i][1 - j]
            if p1_br and p2_br:
                pure_ne.append((i, j))

    # Find dominant strategies
    dominant = []
    # P1: row 0 dominates if a[0][j] > a[1][j] for all j
    if a[0][0] > a[1][0] and a[0][1] > a[1][1]:
        dominant.append("P1: row 0 strictly dominates")
    elif a[1][0] > a[0][0] and a[1][1] > a[0][1]:
        dominant.append("P1: row 1 strictly dominates")
    if b[0][0] > b[0][1] and b[1][0] > b[1][1]:
        dominant.append("P2: col 0 strictly dominates")
    elif b[0][1] > b[0][0] and b[1][1] > b[1][0]:
        dominant.append("P2: col 1 strictly dominates")

    # Find mixed Nash equilibrium
    # P1 mixes so that P2 is indifferent:
    # p*b[0][0] + (1-p)*b[1][0] = p*b[0][1] + (1-p)*b[1][1]
    mixed_ne = None
    denom_p2 = (b[0][0] - b[1][0]) - (b[0][1] - b[1][1])
    denom_p1 = (a[0][0] - a[0][1]) - (a[1][0] - a[1][1])

    if abs(denom_p2) > 1e-10 and abs(denom_p1) > 1e-10:
        p = (b[1][1] - b[1][0]) / denom_p2  # P1's mix probability for row 0
        q = (a[1][1] - a[0][1]) / denom_p1  # P2's mix probability for col 0
        if 0 < p < 1 and 0 < q < 1:
            mixed_ne = ((p, 1 - p), (q, 1 - q))

    # Classify game type
    game_type = _classify_2x2_game(a, b)

    return NashEquilibriumReport(
        pure_equilibria=pure_ne,
        mixed_equilibrium=mixed_ne,
        dominant_strategies=dominant,
        game_type=game_type,
        payoff_matrix_p1=[list(row) for row in a],
        payoff_matrix_p2=[list(row) for row in b],
    )


def present_value(
    future_value: float,
    rate: float,
    periods: int,
    continuous: bool = False,
) -> PresentValueReport:
    """Calculate present value of a future cash flow.

    Args:
        future_value: Future amount
        rate: Interest rate per period
        periods: Number of compounding periods
        continuous: If True, use continuous compounding

    Returns:
        PresentValueReport.
    """
    if periods < 0:
        raise ValueError("periods must be non-negative")

    if continuous:
        pv = future_value * math.exp(-rate * periods)
        ear = math.exp(rate) - 1
        compounding = "continuous"
    else:
        pv = future_value / (1 + rate) ** periods
        ear = (1 + rate) ** 1 - 1 if periods >= 1 else None
        compounding = "discrete"

    return PresentValueReport(
        present_value=pv,
        future_value=future_value,
        rate=rate,
        periods=periods,
        compounding=compounding,
        effective_annual_rate=ear,
    )


def future_value(
    present_val: float,
    rate: float,
    periods: int,
    continuous: bool = False,
) -> PresentValueReport:
    """Calculate future value of a present cash flow.

    Args:
        present_val: Present amount
        rate: Interest rate per period
        periods: Number of compounding periods
        continuous: If True, use continuous compounding

    Returns:
        PresentValueReport.
    """
    if periods < 0:
        raise ValueError("periods must be non-negative")

    if continuous:
        fv = present_val * math.exp(rate * periods)
        ear = math.exp(rate) - 1
        compounding = "continuous"
    else:
        fv = present_val * (1 + rate) ** periods
        ear = (1 + rate) - 1 if periods >= 1 else None
        compounding = "discrete"

    return PresentValueReport(
        present_value=present_val,
        future_value=fv,
        rate=rate,
        periods=periods,
        compounding=compounding,
        effective_annual_rate=ear,
    )


def _classify_2x2_game(a, b) -> str:
    """Classify a 2x2 game by structure."""
    # Prisoner's dilemma: both have dominant strategy to defect, mutual cooperation better
    # Check symmetric structure
    if (a[0][0] == b[0][0] and a[1][1] == b[1][1] and
            a[0][1] == b[1][0] and a[1][0] == b[0][1]):
        # Symmetric game
        if a[1][0] > a[0][0] > a[1][1] > a[0][1]:
            return "prisoners_dilemma"
        if a[0][0] > a[1][0] and a[1][1] > a[0][1]:
            return "coordination"
    # Battle of the sexes: two pure NE, players prefer different ones
    # Chicken/hawk-dove
    return "generic_2x2"
