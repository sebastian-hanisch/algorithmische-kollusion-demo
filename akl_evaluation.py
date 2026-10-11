"""Kennzahlen, Verdikt-Logik und Auswertungen über den Kern (Komposition der Einheiten aus akl_algorithm) und die vorgerechnete Tabelle."""

import json
import os
import statistics
from dataclasses import dataclass
from functools import lru_cache

import akl_algorithm as A
import akl_constants as C

SWEEPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "akl_sweeps.json")


def fmt(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int(x: int) -> str:
    """Ganzzahl mit Punkt als Tausendertrenner."""
    return f"{int(x):,}".replace(",", ".")


@dataclass(frozen=True)
class Settings:
    alpha: float = C.DEFAULT_ALPHA
    beta: int = C.DEFAULT_BETA          # in Einheiten von 1e-5
    gamma: float = C.DEFAULT_GAMMA
    memory: str = C.DEFAULT_MEMORY
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Market:
    nash_price: float
    monopoly_price: float
    nash_profit: float
    monopoly_profit: float
    prices: tuple[float, ...]
    grid_nash: tuple[tuple[float, float, float], ...]      # (Preis 1, Preis 2, Index Δ) der reinen Nash-Gleichgewichte auf dem Raster


@lru_cache(maxsize=1)
def market() -> Market:
    pn, pm = A.nash_price(), A.monopoly_price()
    prices = A.price_grid()
    table = A.profit_table(prices)
    pin, pim = A.profits(pn, pn)[0], A.profits(pm, pm)[0]
    gn = tuple((prices[i], prices[j], A.collusion_index(table[i][j][0], pin, pim)) for i, j in A.grid_nash_pairs(table))
    return Market(pn, pm, pin, pim, tuple(prices), gn)


@dataclass(frozen=True)
class Analysis:
    result: A.LearningResult
    deviation: A.Deviation | None
    market: Market


def analyse(settings: Settings) -> Analysis:
    r = A.learn(settings.seed, settings.alpha, settings.beta * C.BETA_UNIT, settings.gamma, memory=settings.memory == "both")
    return Analysis(r, A.deviation_test(r, settings.gamma), market())


def level_of(delta: float) -> str:
    """Sprachliche Stufe des Kollusionsindex."""
    if delta < 0.1:
        return "nahe am Wettbewerbspreis"
    if delta < 0.5:
        return "teils überhöht"
    return "deutlich überhöht"


def verdict(an: Analysis, settings: Settings) -> tuple[str, str]:
    r, mk = an.result, an.market
    state = f"Der Lauf ist nach {fmt_int(r.periods)} Perioden konvergiert" if r.converged else f"Der Lauf hat nach {fmt_int(r.periods)} Perioden die Obergrenze erreicht (**nicht konvergiert**)"
    head = (f"{state}: mittlerer Preis {fmt(r.avg_price)} (Nash {fmt(mk.nash_price)}, gemeinsames Monopol {fmt(mk.monopoly_price)}), Kollusionsindex Δ = {fmt(r.delta)}, "
            f"also {level_of(r.delta)}.")
    if not r.memory:
        tail = " Ohne Gedächtnis kann der Rivale eine Abweichung gar nicht sehen: Es gibt keine Strafphase, und die höheren Preise entstehen trotzdem."
        return ("info" if r.delta < 0.5 else "warning"), head + tail
    d = an.deviation
    if d.gain < 0:
        tail = (f" Der Abweichungstest: Wer einmal auf den Nash-Preis geht, verliert über {C.DEVIATION_HORIZON} Perioden {fmt(-d.gain, 3)} an diskontiertem Gewinn, "
                f"weil der Rivale seinen Preis um bis zu {fmt(-d.rival_dip)} senkt.")
    else:
        tail = (f" Der Abweichungstest: Die Abweichung auf den Nash-Preis **lohnt sich** hier (+{fmt(d.gain, 3)} diskontierter Gewinn über {C.DEVIATION_HORIZON} Perioden), "
                "obwohl der Rivale reagiert: Die gelernte Vergeltung trägt in diesem Lauf nicht.")
    return ("warning" if r.delta >= 0.5 else "info"), head + tail


# --- Vorgerechnete Tabelle ---------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_sweeps() -> dict | None:
    try:
        with open(SWEEPS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_config(alpha: float, beta: float, gamma: float, memory: str) -> dict | None:
    data = load_sweeps()
    if data is None:
        return None
    for c in data["configs"]:
        if abs(c["alpha"] - alpha) < 1e-9 and abs(c["beta"] - beta) < 1e-9 and abs(c["gamma"] - gamma) < 1e-9 and c["memory"] == memory:
            return c
    return None


def summarize(config: dict) -> dict:
    runs = config["runs"]
    d = [r["delta"] for r in runs]
    dev = [r for r in runs if r["dev_gain"] is not None]
    return {
        "n": len(runs), "mean": statistics.fmean(d), "median": statistics.median(d), "min": min(d), "max": max(d),
        "converged": sum(r["converged"] for r in runs), "periods_median": int(statistics.median([r["periods"] for r in runs])),
        "dev_n": len(dev),
        "dev_unprofitable": sum(r["dev_gain"] < 0 for r in dev),
        "dev_rival_cuts": sum(r["dev_dip"] < -1e-9 for r in dev),
    }


def sweep_series(kind: str) -> list[dict]:
    """Konfigurationen einer Reihe ('alpha', 'beta', 'gamma'): je Wert der variierten Größe (die übrigen auf der Voreinstellung)."""
    data = load_sweeps()
    if data is None:
        return []
    base = {"alpha": C.DEFAULT_ALPHA, "beta": float(C.DEFAULT_BETA), "gamma": C.DEFAULT_GAMMA}
    out = []
    for c in data["configs"]:
        if c["memory"] != "both":
            continue
        if all(abs(c[k] - base[k]) < 1e-9 for k in base if k != kind):
            out.append({"x": c[kind], "config": c, "summary": summarize(c)})
    return sorted(out, key=lambda e: e["x"])
