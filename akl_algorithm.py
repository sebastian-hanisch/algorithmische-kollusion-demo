"""Kern: Logit-Duopol, Preisraster, Nash und Monopol, zwei tabellarische Q-Lernende, Dauerzyklus und Abweichungstest.

Reine Standardbibliothek in der Lernschleife (rund 200 000 Perioden je Sekunde); scipy nur im Orakeltest.
"""

import math
import random
from dataclasses import dataclass, field

import akl_constants as C


# --- Markt -------------------------------------------------------------------------------------------------------------------------------------

def shares(p1: float, p2: float) -> tuple[float, float]:
    """Logit-Nachfrageanteile beider Anbieter bei den Preisen p1, p2."""
    e1 = math.exp((C.QUALITY - p1) / C.MU)
    e2 = math.exp((C.QUALITY - p2) / C.MU)
    e0 = math.exp(C.OUTSIDE / C.MU)
    d = e1 + e2 + e0
    return e1 / d, e2 / d


def profits(p1: float, p2: float) -> tuple[float, float]:
    q1, q2 = shares(p1, p2)
    return (p1 - C.COST) * q1, (p2 - C.COST) * q2


def nash_price() -> float:
    """Symmetrisches Bertrand-Nash-Gleichgewicht: p = c + µ / (1 − q(p, p)), Lösung per Bisektion."""
    f = lambda p: p - C.COST - C.MU / (1.0 - shares(p, p)[0])
    lo, hi = C.COST + 1e-9, C.COST + 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def monopoly_price() -> float:
    """Gemeinsamer Preis, der die Summe beider Gewinne maximiert (Goldener Schnitt; Zielfunktion ist eingipfelig)."""
    f = lambda p: sum(profits(p, p))
    lo, hi = C.COST + 1e-9, C.COST + 10.0
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(200):
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = f(b)
        else:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = f(a)
    return (lo + hi) / 2


def price_grid(m: int = C.GRID_M, xi: float = C.GRID_XI) -> list[float]:
    """m äquidistante Preise von p_N − xi·(p_M − p_N) bis p_M + xi·(p_M − p_N)."""
    pn, pm = nash_price(), monopoly_price()
    d = pm - pn
    lo, hi = pn - xi * d, pm + xi * d
    return [lo + (hi - lo) * i / (m - 1) for i in range(m)]


def profit_table(prices: list[float]) -> list[list[tuple[float, float]]]:
    return [[profits(a, b) for b in prices] for a in prices]


def best_response(rival_index: int, table, agent: int = 0) -> int:
    """Index des Preises, der den Sofortgewinn gegen einen festen Rivalenpreis maximiert (symmetrisches Spiel: agent 0)."""
    m = len(table)
    return max(range(m), key=lambda i: table[i][rival_index][0])


def grid_nash_pairs(table) -> list[tuple[int, int]]:
    """Alle reinen Nash-Gleichgewichte des Einmalspiels auf dem Preisraster (beide beste Antwort auf den anderen)."""
    m = len(table)
    out = []
    for i in range(m):
        for j in range(m):
            if table[i][j][0] >= max(table[x][j][0] for x in range(m)) - 1e-12 and table[i][j][1] >= max(table[i][y][1] for y in range(m)) - 1e-12:
                out.append((i, j))
    return out


def collusion_index(avg_profit: float, nash_profit: float, monopoly_profit: float) -> float:
    """Δ = (π̄ − π_N) / (π_M − π_N): 0 bei Nash-Gewinn, 1 beim gemeinsamen Monopolgewinn (Calvano et al. 2020)."""
    return (avg_profit - nash_profit) / (monopoly_profit - nash_profit)


# --- Lernen ------------------------------------------------------------------------------------------------------------------------------------

def td_update(q_old: float, reward: float, next_max: float, alpha: float, gamma: float) -> float:
    """Q(s,a) ← (1−α)·Q(s,a) + α·(r + γ·max_a' Q(s',a'))."""
    return (1.0 - alpha) * q_old + alpha * (reward + gamma * next_max)


def epsilon(t: int, beta: float) -> float:
    """Explorationswahrscheinlichkeit nach t Perioden: exp(−β·t)."""
    return math.exp(-beta * t)


def initial_q(table, gamma: float, agent: int) -> list[float]:
    """Anfangswert je eigene Aktion: diskontierter Durchschnittsgewinn gegen einen gleichverteilt wählenden Rivalen (Calvano et al.)."""
    m = len(table)
    if agent == 0:
        return [sum(table[a][b][0] for b in range(m)) / ((1.0 - gamma) * m) for a in range(m)]
    return [sum(table[b][a][1] for b in range(m)) / ((1.0 - gamma) * m) for a in range(m)]


@dataclass
class LearningResult:
    periods: int
    converged: bool
    memory: bool
    prices: list[float]
    greedy: list[list[int]]                     # greedy[k][zustand] = gewählte Aktion des Agenten k
    cycle_states: list[int]
    cycle_prices: list[tuple[float, float]]
    avg_profit: float
    delta: float
    avg_price: float
    trace: list[tuple[int, float, float]] = field(default_factory=list)   # (Periode, mittlerer Preis im Fenster, Erkundung ε)
    q: list | None = None                                                  # nur wenn angefordert


def learn(seed: int, alpha: float, beta: float, gamma: float, memory: bool = True, cap: int | None = None, stable: int | None = None,
          fixed_rival: int | None = None, record_every: int | None = None, keep_q: bool = False) -> LearningResult:
    """Zwei unabhängige Q-Lernende, ε-gierig mit ε_t = exp(−β·t); läuft bis die gierige Strategie `stable` Perioden unverändert ist
    oder `cap` erreicht ist. `fixed_rival`: Aktionsindex, den Agent 2 immer spielt und nicht lernt (Ein-Agenten-Grenzfall)."""
    cap = C.PERIOD_CAP if cap is None else cap
    stable = C.STABLE_PERIODS if stable is None else stable
    record_every = C.RECORD_EVERY if record_every is None else record_every
    rng = random.Random(seed)
    prices = price_grid()
    m = len(prices)
    pi = profit_table(prices)
    n_states = m * m if memory else 1
    q = [[initial_q(pi, gamma, k)[:] for _ in range(n_states)] for k in range(2)]
    state = rng.randrange(n_states) if memory else 0
    greedy = [[max(range(m), key=q[k][s].__getitem__) for s in range(n_states)] for k in range(2)]
    unchanged = 0
    t = 0
    trace: list[tuple[int, float, float]] = []
    win_sum, win_n = 0.0, 0
    rand, randrange, exp = rng.random, rng.randrange, math.exp
    while t < cap:
        eps = exp(-beta * t)
        row1, row2 = q[0][state], q[1][state]
        a1 = randrange(m) if rand() < eps else max(range(m), key=row1.__getitem__)
        if fixed_rival is None:
            a2 = randrange(m) if rand() < eps else max(range(m), key=row2.__getitem__)
        else:
            a2 = fixed_rival
        r1, r2 = pi[a1][a2]
        nxt = a1 * m + a2 if memory else 0
        changed = False
        row1[a1] = (1.0 - alpha) * row1[a1] + alpha * (r1 + gamma * max(q[0][nxt]))
        g = max(range(m), key=row1.__getitem__)
        if g != greedy[0][state]:
            greedy[0][state] = g
            changed = True
        if fixed_rival is None:
            row2[a2] = (1.0 - alpha) * row2[a2] + alpha * (r2 + gamma * max(q[1][nxt]))
            g = max(range(m), key=row2.__getitem__)
            if g != greedy[1][state]:
                greedy[1][state] = g
                changed = True
        unchanged = 0 if changed else unchanged + 1
        win_sum += prices[a1] + prices[a2]
        win_n += 2
        state = nxt
        t += 1
        if t % record_every == 0:
            trace.append((t, win_sum / win_n, eps))
            win_sum, win_n = 0.0, 0
        if unchanged >= stable:
            break
    converged = unchanged >= stable
    if fixed_rival is not None:
        for s in range(n_states):
            greedy[1][s] = fixed_rival
    cyc = cycle_from(greedy, state, m, memory)
    cyc_prices = [(prices[greedy[0][x]], prices[greedy[1][x]]) for x in cyc]
    avg_profit = sum((profits(*p)[0] + profits(*p)[1]) / 2 for p in cyc_prices) / len(cyc_prices)
    pn, pm = nash_price(), monopoly_price()
    delta = collusion_index(avg_profit, profits(pn, pn)[0], profits(pm, pm)[0])
    avg_price = sum((p[0] + p[1]) / 2 for p in cyc_prices) / len(cyc_prices)
    return LearningResult(t, converged, memory, prices, greedy, cyc, cyc_prices, avg_profit, delta, avg_price, trace, q if keep_q else None)


def cycle_from(greedy, start: int, m: int, memory: bool) -> list[int]:
    """Dauerzyklus der Zustände, wenn beide ab `start` ihre gierige Aktion spielen."""
    seen: dict[int, int] = {}
    order: list[int] = []
    s = start
    while s not in seen:
        seen[s] = len(order)
        order.append(s)
        s = greedy[0][s] * m + greedy[1][s] if memory else 0
    return order[seen[s]:]


# --- Abweichungstest ---------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Deviation:
    baseline: list[tuple[float, float]]        # Preise (Agent 1, Agent 2) je Periode ohne Abweichung
    deviated: list[tuple[float, float]]        # mit erzwungener Abweichung von Agent 1 in Periode 0
    gain: float                                # diskontierter Gewinn von Agent 1: Abweichung minus Basislinie
    rival_dip: float                           # tiefster Preis des Rivalen nach der Abweichung minus sein Basispreis (negativ = er senkt)
    return_period: int | None                  # erste Periode, ab der der Rivale dauerhaft wieder im Basisband liegt


def deviation_test(result: LearningResult, gamma: float, deviation_price: float | None = None, horizon: int | None = None) -> Deviation | None:
    """Agent 1 wird in Periode 0 auf den Rasterpreis nächst dem Nash-Preis gezwungen, danach spielen beide wieder gierig.
    Ohne Gedächtnis gibt es keinen Zustand, an dem der Rivale die Abweichung sehen könnte: dann None."""
    if not result.memory:
        return None
    horizon = C.DEVIATION_HORIZON if horizon is None else horizon
    prices = result.prices
    m = len(prices)
    target = nash_price() if deviation_price is None else deviation_price
    dev_action = min(range(m), key=lambda i: abs(prices[i] - target))
    start = result.cycle_states[0]

    def play(forced: int | None) -> list[tuple[int, int]]:
        s, out = start, []
        for t in range(horizon):
            a1, a2 = result.greedy[0][s], result.greedy[1][s]
            if t == 0 and forced is not None:
                a1 = forced
            out.append((a1, a2))
            s = a1 * m + a2
        return out

    base, dev = play(None), play(dev_action)
    disc = lambda path: sum(gamma ** t * profits(prices[a], prices[b])[0] for t, (a, b) in enumerate(path))
    step = prices[1] - prices[0]
    base_p = [(prices[a], prices[b]) for a, b in base]
    dev_p = [(prices[a], prices[b]) for a, b in dev]
    dip = min(p[1] for p in dev_p[1:]) - base_p[0][1]
    band = 0.51 * step
    ret = None
    for t in range(1, horizon):
        if all(abs(dev_p[u][1] - base_p[u][1]) <= band for u in range(t, horizon)):
            ret = t
            break
    return Deviation(base_p, dev_p, disc(dev) - disc(base), dip, ret)
