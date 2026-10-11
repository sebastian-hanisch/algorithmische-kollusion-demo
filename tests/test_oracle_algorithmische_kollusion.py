"""Orakel-Tests: jeder Kernbaustein gegen einen unabhängigen Rechenweg.

1. Nash- und Monopolpreis gegen scipy (Nullstelle der numerisch abgeleiteten Gewinnfunktion, beschränkte Minimierung).
2. Gewinntabelle gegen eine direkte numpy-Rechnung der Logit-Formel.
3. Ein-Agenten-Grenzfall: Gegen einen festen Rivalen lernt der Agent die beste Antwort, die ein Aufzählen über das Raster liefert.
4. Die Lernschleife gegen eine zweite, schlichte Implementierung (numpy-Tabellen, ein Agent nach dem anderen) mit demselben Zufallsstrom.
5. Der Dauerzyklus gegen eine Simulation der gierigen Strategien über viele Perioden.
"""

import random

import pytest

import akl_algorithm as A
import akl_constants as C

np = pytest.importorskip("numpy")
scipy_opt = pytest.importorskip("scipy.optimize")


def test_nash_and_monopoly_against_scipy():
    def dpi1(p):
        h = 1e-6
        return (A.profits(p + h, p)[0] - A.profits(p - h, p)[0]) / (2 * h)

    root = scipy_opt.brentq(dpi1, 1.05, 3.0)
    assert A.nash_price() == pytest.approx(root, abs=1e-6)
    res = scipy_opt.minimize_scalar(lambda p: -sum(A.profits(p, p)), bounds=(1.05, 4.0), method="bounded", options={"xatol": 1e-10})
    assert A.monopoly_price() == pytest.approx(res.x, abs=1e-5)


def test_profit_table_against_direct_numpy_formula():
    prices = np.array(A.price_grid())
    table = A.profit_table(list(prices))
    e = np.exp((C.QUALITY - prices) / C.MU)
    for i in range(len(prices)):
        for j in range(len(prices)):
            denom = e[i] + e[j] + np.exp(C.OUTSIDE / C.MU)
            assert table[i][j][0] == pytest.approx((prices[i] - C.COST) * e[i] / denom, rel=1e-12)
            assert table[i][j][1] == pytest.approx((prices[j] - C.COST) * e[j] / denom, rel=1e-12)


@pytest.mark.parametrize("rival", [0, 4, 9, 14])
def test_single_agent_learns_the_best_response_to_a_fixed_rival(rival):
    prices = A.price_grid()
    table = A.profit_table(prices)
    expected = max(range(len(prices)), key=lambda i: table[i][rival][0])
    r = A.learn(11, 0.15, 5e-4, 0.0, memory=False, cap=30_000, stable=5_000, fixed_rival=rival)
    assert r.greedy[0][0] == expected
    assert r.cycle_prices == [(prices[expected], prices[rival])]


def _reference_run(seed, alpha, beta, gamma, memory, periods):
    """Schlichte zweite Implementierung: dieselbe Zufallsfolge, aber numpy-Tabellen und je Agent ein eigener Block."""
    rng = random.Random(seed)
    prices = A.price_grid()
    m = len(prices)
    pi = np.array([[A.profits(a, b) for b in prices] for a in prices])      # [a1, a2, Agent]
    n_states = m * m if memory else 1
    q = np.zeros((2, n_states, m))
    for s in range(n_states):
        q[0, s] = pi[:, :, 0].mean(axis=1) / (1.0 - gamma)
        q[1, s] = pi[:, :, 1].mean(axis=0) / (1.0 - gamma)
    s = rng.randrange(n_states) if memory else 0
    for t in range(periods):
        eps = np.exp(-beta * t)
        acts = []
        for k in range(2):
            if rng.random() < eps:
                acts.append(rng.randrange(m))
            else:
                acts.append(int(np.argmax(q[k, s])))
        a1, a2 = acts
        s2 = a1 * m + a2 if memory else 0
        for k, a in enumerate(acts):
            target = pi[a1, a2, k] + gamma * q[k, s2].max()
            q[k, s, a] = (1.0 - alpha) * q[k, s, a] + alpha * target
        s = s2
    return q


@pytest.mark.parametrize("memory", [True, False])
def test_learning_loop_against_reference_implementation(memory):
    r = A.learn(21, 0.15, 3e-4, 0.9, memory=memory, cap=4_000, stable=10**9, keep_q=True)
    ref = _reference_run(21, 0.15, 3e-4, 0.9, memory, 4_000)
    assert np.allclose(np.array(r.q), ref, rtol=0, atol=1e-10)


def test_cycle_profit_against_a_long_simulation_of_the_greedy_strategies():
    r = A.learn(2, 0.15, 5e-4, 0.9, cap=200_000, stable=20_000)
    assert r.converged
    m = len(r.prices)
    length = len(r.cycle_states)
    s = r.cycle_states[0]
    total, n = 0.0, 60 * length
    for _ in range(n):
        a1, a2 = r.greedy[0][s], r.greedy[1][s]
        p = A.profits(r.prices[a1], r.prices[a2])
        total += (p[0] + p[1]) / 2
        s = a1 * m + a2
    assert total / n == pytest.approx(r.avg_profit, rel=1e-9)
    pn, pm = A.nash_price(), A.monopoly_price()
    assert r.delta == pytest.approx((r.avg_profit - A.profits(pn, pn)[0]) / (A.profits(pm, pm)[0] - A.profits(pn, pn)[0]), rel=1e-9)
