"""Einheiten des Kerns an Handbeispielen: Markt, Raster, Q-Update, Erkundung, Zyklus, Abweichungstest, Lernschleife."""

import math

import pytest

import akl_algorithm as A
import akl_constants as C


def test_demand_and_profit_by_hand_at_equal_prices():
    # p1 = p2 = 2 = Qualität: beide Exponenten null, Außenangebot ebenfalls: je 1/3 Anteil, Gewinn (2 − 1) / 3
    q1, q2 = A.shares(2.0, 2.0)
    assert q1 == pytest.approx(1 / 3) and q2 == pytest.approx(1 / 3)
    assert A.profits(2.0, 2.0)[0] == pytest.approx(1 / 3)


def test_cheaper_provider_gets_more_demand_and_shares_stay_below_one():
    q1, q2 = A.shares(1.5, 1.8)
    assert q1 > q2 and q1 + q2 < 1.0
    assert A.shares(1.5, 1.8)[0] == pytest.approx(A.shares(1.8, 1.5)[1])


def test_reference_prices_match_the_literature_setting():
    assert A.nash_price() == pytest.approx(1.4729, abs=1e-4)
    assert A.monopoly_price() == pytest.approx(1.9250, abs=1e-4)


def test_grid_is_equidistant_and_brackets_both_reference_prices():
    g = A.price_grid()
    pn, pm = A.nash_price(), A.monopoly_price()
    assert len(g) == C.GRID_M
    steps = [b - a for a, b in zip(g, g[1:])]
    assert max(steps) - min(steps) < 1e-12
    assert g[0] == pytest.approx(pn - 0.1 * (pm - pn)) and g[-1] == pytest.approx(pm + 0.1 * (pm - pn))


def test_td_update_by_hand():
    # (1 − 0,2)·1 + 0,2·(0,5 + 0,9·2) = 0,8 + 0,46
    assert A.td_update(1.0, 0.5, 2.0, 0.2, 0.9) == pytest.approx(1.26)


def test_exploration_schedule():
    assert A.epsilon(0, 1e-5) == 1.0
    assert A.epsilon(round(math.log(2) / 1e-5), 1e-5) == pytest.approx(0.5, abs=1e-5)


def test_initial_q_by_hand():
    table = [[(1.0, 5.0), (2.0, 0.0)], [(0.0, 2.0), (4.0, 3.0)]]
    assert A.initial_q(table, 0.0, 0) == [pytest.approx(1.5), pytest.approx(2.0)]      # Zeilenmittel des eigenen Gewinns
    assert A.initial_q(table, 0.0, 1) == [pytest.approx(3.5), pytest.approx(1.5)]      # Spaltenmittel des Gewinns von Agent 2
    assert A.initial_q(table, 0.5, 0)[0] == pytest.approx(3.0)                          # geteilt durch (1 − γ)


def test_collusion_index_endpoints():
    assert A.collusion_index(0.2, 0.2, 0.3) == 0.0
    assert A.collusion_index(0.3, 0.2, 0.3) == 1.0
    assert A.collusion_index(0.25, 0.2, 0.3) == pytest.approx(0.5)


def test_grid_game_has_two_pure_nash_equilibria():
    prices = A.price_grid()
    pairs = A.grid_nash_pairs(A.profit_table(prices))
    assert [(round(prices[i], 3), round(prices[j], 3)) for i, j in pairs] == [(1.466, 1.466), (1.505, 1.505)]


def test_cycle_from_follows_the_greedy_strategies():
    # zwei Preise, Zustand = 2·a1 + a2: 0 → (1,0)=2 → (0,1)=1 → (1,1)=3 → (0,0)=0
    g = [[1, 1, 0, 0], [0, 1, 1, 0]]
    assert A.cycle_from(g, 0, 2, True) == [0, 2, 1, 3]
    # Vorlaufzustand 1 führt in den Fixpunkt 0
    assert A.cycle_from([[0, 0, 0, 0], [0, 0, 0, 0]], 1, 2, True) == [0]
    assert A.cycle_from([[0], [0]], 0, 2, False) == [0]


def test_deviation_without_reaction_equals_the_one_period_difference(two_price_result):
    d = A.deviation_test(two_price_result, 0.9, deviation_price=1.4, horizon=6)
    expected = A.profits(1.4, 1.9)[0] - A.profits(1.9, 1.9)[0]
    assert d.gain == pytest.approx(expected)
    assert d.rival_dip == 0.0 and d.return_period == 1
    assert d.baseline[0] == (1.9, 1.9) and d.deviated[0] == (1.4, 1.9)


def test_deviation_with_one_period_punishment_by_hand(punishing_result):
    gamma = 0.9
    d = A.deviation_test(punishing_result, gamma, deviation_price=1.4, horizon=6)
    base = A.profits(1.9, 1.9)[0]
    expected = (A.profits(1.4, 1.9)[0] - base) + gamma * (A.profits(1.9, 1.4)[0] - base)
    assert d.gain == pytest.approx(expected)
    assert d.rival_dip == pytest.approx(1.4 - 1.9)
    assert d.deviated[1] == (1.9, 1.4) and d.return_period == 2


def test_deviation_test_needs_memory(two_price_result):
    two_price_result.memory = False
    assert A.deviation_test(two_price_result, 0.9) is None


def test_learning_is_deterministic_for_a_seed():
    a = A.learn(3, 0.15, 2e-4, 0.95, cap=40_000, stable=5_000)
    b = A.learn(3, 0.15, 2e-4, 0.95, cap=40_000, stable=5_000)
    assert a.greedy == b.greedy and a.periods == b.periods and a.delta == b.delta


def test_cap_stops_the_run_and_reports_not_converged():
    r = A.learn(0, 0.15, 1e-6, 0.95, cap=3_000)
    assert r.periods == 3_000 and not r.converged


def test_single_period_changes_exactly_one_cell_per_agent_by_the_td_rule():
    gamma, alpha = 0.9, 0.2
    r = A.learn(5, alpha, 1e-5, gamma, memory=False, cap=1, keep_q=True)
    prices = A.price_grid()
    pi = A.profit_table(prices)
    q0 = [A.initial_q(pi, gamma, 0), A.initial_q(pi, gamma, 1)]
    changed = [[a for a in range(C.GRID_M) if r.q[k][0][a] != q0[k][a]] for k in range(2)]
    assert [len(c) for c in changed] == [1, 1]
    a1, a2 = changed[0][0], changed[1][0]
    assert r.q[0][0][a1] == pytest.approx(A.td_update(q0[0][a1], pi[a1][a2][0], max(q0[0]), alpha, gamma))
    assert r.q[1][0][a2] == pytest.approx(A.td_update(q0[1][a2], pi[a1][a2][1], max(q0[1]), alpha, gamma))
