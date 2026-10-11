"""Jede Zahl aus README und App wird hier über die echten Auswertungsfunktionen aus der vorgerechneten Tabelle nachgerechnet (DEMO-PLAYBOOK §4)."""

import statistics

import pytest

import akl_algorithm as A
import akl_constants as C
import akl_evaluation as E

BASE = (C.DEFAULT_ALPHA, float(C.DEFAULT_BETA), C.DEFAULT_GAMMA, "both")


def summary(alpha=BASE[0], beta=BASE[1], gamma=BASE[2], memory="both"):
    cfg = E.find_config(alpha, beta, gamma, memory)
    assert cfg is not None
    return cfg, E.summarize(cfg)


def test_table_covers_every_configuration_with_twenty_seeds():
    data = E.load_sweeps()
    assert data["meta"]["seeds"] == list(C.SWEEP_SEEDS) and len(data["configs"]) == 13
    for c in data["configs"]:
        assert len(c["runs"]) == 20 and [r["seed"] for r in c["runs"]] == list(C.SWEEP_SEEDS)


def test_reference_prices_and_grid_nash_indices():
    mk = E.market()
    assert mk.nash_price == pytest.approx(1.4729, abs=5e-5) and mk.monopoly_price == pytest.approx(1.9250, abs=5e-5)
    assert [round(g[2], 2) for g in sorted(mk.grid_nash, key=lambda g: g[2])] == [-0.02, 0.12]


def test_default_configuration():
    cfg, s = summary()
    assert s["median"] == pytest.approx(0.79, abs=0.005) and s["min"] == pytest.approx(0.61, abs=0.005) and s["max"] == pytest.approx(0.95, abs=0.005)
    assert s["converged"] == 20 and s["periods_median"] == pytest.approx(613_679, abs=1)
    assert min(r["periods"] for r in cfg["runs"]) == pytest.approx(395_379, abs=1) and max(r["periods"] for r in cfg["runs"]) == pytest.approx(676_127, abs=1)
    assert all(r["delta"] > 0.5 for r in cfg["runs"])


def test_deviation_statistics_in_the_default_configuration():
    cfg, s = summary()
    assert (s["dev_unprofitable"], s["dev_rival_cuts"], s["dev_n"]) == (18, 18, 20)
    returns = [r["dev_return"] for r in cfg["runs"]]
    assert statistics.median([x for x in returns if x is not None]) == 5 and sum(x is None for x in returns) == 9
    assert min(r["dev_gain"] for r in cfg["runs"]) == pytest.approx(-0.408, abs=0.001) and max(r["dev_gain"] for r in cfg["runs"]) == pytest.approx(0.023, abs=0.001)


def test_without_memory_every_run_ends_at_a_single_price_and_no_deviation_test_exists():
    cfg, s = summary(memory="none")
    assert s["median"] == pytest.approx(0.946, abs=0.0005) and s["min"] == pytest.approx(0.868, abs=0.0005) and s["max"] == pytest.approx(0.994, abs=0.0005)
    assert all(r["cycle"] == 1 for r in cfg["runs"]) and s["dev_n"] == 0
    assert s["periods_median"] == pytest.approx(472_493, abs=1)
    assert s["median"] > summary()[1]["median"]


def test_median_index_rises_with_the_shadow_of_the_future():
    medians = {g: summary(gamma=g)[1]["median"] for g in C.SWEEP_GAMMAS}
    assert [round(medians[g], 3) for g in C.SWEEP_GAMMAS] == [0.213, 0.289, 0.376, 0.597, 0.790]
    assert list(medians.values()) == sorted(medians.values())
    assert summary(gamma=0.0)[1]["dev_unprofitable"] == 3


def test_exploration_decay_and_learning_rate():
    b = {x: summary(beta=x) for x in C.SWEEP_BETAS}
    assert b[0.4][1]["median"] == pytest.approx(0.931, abs=0.0005) and b[0.4][1]["periods_median"] == pytest.approx(1_804_668, abs=1)
    assert b[10.0][1]["median"] == pytest.approx(0.698, abs=0.0005) and b[20.0][1]["median"] == pytest.approx(0.707, abs=0.0005)
    assert b[20.0][1]["periods_median"] == pytest.approx(149_808, abs=1)
    a = [summary(alpha=x)[1]["median"] for x in C.SWEEP_ALPHAS]
    assert [round(v, 3) for v in a] == [0.707, 0.779, 0.790, 0.768, 0.723]
    assert all(s[1]["converged"] == 20 for s in b.values())


def test_a_live_rerun_of_the_fast_configuration_matches_the_table():
    """Stichprobe: vier Seeds der schnellsten Konfiguration (β = 20·10⁻⁵) neu gerechnet; Band, weil Plattform-Ulps den Lauf kippen könnten."""
    cfg = E.find_config(C.DEFAULT_ALPHA, 20.0, C.DEFAULT_GAMMA, "both")
    seeds = C.SWEEP_SEEDS[:4]
    live = [A.learn(s, C.DEFAULT_ALPHA, 20.0 * C.BETA_UNIT, C.DEFAULT_GAMMA, cap=C.SWEEP_CAP).delta for s in seeds]
    table = [r["delta"] for r in cfg["runs"] if r["seed"] in seeds]
    assert statistics.median(live) == pytest.approx(statistics.median(table), abs=0.1)
