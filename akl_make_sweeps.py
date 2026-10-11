"""Rechnet die Tabelle `akl_sweeps.json` vor (einmalig, mehrere Minuten): je Konfiguration 20 feste Zufalls-Seeds, ein Lauf bis zur Konvergenz.

    python akl_make_sweeps.py            # schreibt akl_sweeps.json neben dieses Skript
    python akl_make_sweeps.py --check    # nur die Konfigurationsliste zeigen

Die App liest die Datei nur; die Tests prüfen jede README-Zahl gegen sie und reproduzieren einzelne Konfigurationen als Stichprobe.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import akl_algorithm as A
import akl_constants as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "akl_sweeps.json")


def configs() -> list[dict]:
    """Eine Größe variiert, der Rest bleibt auf der Voreinstellung. Doppelte Konfigurationen fallen weg."""
    base = {"alpha": C.DEFAULT_ALPHA, "beta": float(C.DEFAULT_BETA), "gamma": C.DEFAULT_GAMMA, "memory": "both"}
    out, seen = [], set()

    def add(label, **change):
        c = {**base, **change}
        key = (c["alpha"], c["beta"], c["gamma"], c["memory"])
        if key not in seen:
            seen.add(key)
            out.append({"label": label, **c})

    add("Voreinstellung")
    for b in C.SWEEP_BETAS:
        add(f"beta={b}", beta=float(b))
    for a in C.SWEEP_ALPHAS:
        add(f"alpha={a}", alpha=a)
    for g in C.SWEEP_GAMMAS:
        add(f"gamma={g}", gamma=g)
    add("kein Gedächtnis", memory="none")
    return out


def one(task: tuple) -> dict:
    cfg, seed = task
    r = A.learn(seed, cfg["alpha"], cfg["beta"] * C.BETA_UNIT, cfg["gamma"], memory=cfg["memory"] == "both", cap=C.SWEEP_CAP)
    d = A.deviation_test(r, cfg["gamma"])
    return {"seed": seed, "delta": r.delta, "periods": r.periods, "converged": r.converged, "cycle": len(r.cycle_states), "avg_price": r.avg_price,
            "dev_gain": None if d is None else d.gain, "dev_dip": None if d is None else d.rival_dip, "dev_return": None if d is None else d.return_period}


def main() -> None:
    cfgs = configs()
    if "--check" in sys.argv:
        for c in cfgs:
            print(c)
        print(len(cfgs), "Konfigurationen,", len(cfgs) * len(C.SWEEP_SEEDS), "Läufe")
        return
    tasks = [(c, s) for c in cfgs for s in C.SWEEP_SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(6, (os.cpu_count() or 2) - 1))) as pool:
        results = list(pool.map(one, tasks, chunksize=2))
    n = len(C.SWEEP_SEEDS)
    data = {"meta": {"seeds": list(C.SWEEP_SEEDS), "cap": C.SWEEP_CAP, "stable": C.STABLE_PERIODS, "horizon": C.DEVIATION_HORIZON,
                     "seconds": round(time.time() - t0)}, "configs": []}
    for i, c in enumerate(cfgs):
        data["configs"].append({**c, "runs": results[i * n:(i + 1) * n]})
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print("geschrieben:", OUT, f"({len(tasks)} Läufe, {round(time.time() - t0)} s)")


if __name__ == "__main__":
    main()
