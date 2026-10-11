"""Konstanten: Marktmodell, Lernparameter, Regler-Grenzen, Experimente, Farben und Presets."""

# --- Marktmodell (Calvano, Calzolari, Denicolò, Pastorello 2020): Logit-Nachfrage, zwei Anbieter, ein Außenangebot -------------------------------
QUALITY = 2.0          # a_i: Qualitätsindex beider Anbieter
OUTSIDE = 0.0          # a_0: Index des Außenangebots (Nicht-Kauf)
MU = 0.25              # Preissensitivität: je kleiner, desto schärfer der Preiswettbewerb
COST = 1.0             # Grenzkosten c
GRID_M = 15            # Anzahl der Preise im Raster
GRID_XI = 0.1          # das Raster reicht um XI·(p_M − p_N) über Nash und Monopol hinaus

# --- Lernen ------------------------------------------------------------------------------------------------------------------------------------
STABLE_PERIODS = 100_000        # Konvergenz: gierige Strategie so viele Perioden unverändert
PERIOD_CAP = 2_500_000          # Obergrenze je Lauf (live); längere Läufe stehen nur in der vorgerechneten Tabelle
RECORD_EVERY = 5_000            # Takt der Lernkurve
DEVIATION_HORIZON = 25          # Perioden, über die der Abweichungstest bewertet wird

ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.05, 0.50, 0.05, 0.15
BETA_MIN, BETA_MAX, BETA_STEP, DEFAULT_BETA = 1, 20, 1, 2     # in Einheiten von 1e-5 je Periode
BETA_UNIT = 1e-5
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.0, 0.95, 0.05, 0.95
SEED_MIN, SEED_MAX, DEFAULT_SEED = 0, 9999, 0

MEMORY_KEYS = ("both", "none")
MEMORY_LABELS = {"both": "Beide letzten Preise", "none": "Kein Gedächtnis"}
DEFAULT_MEMORY = "both"

# --- Vorgerechnete Tabelle (akl_sweeps.json) -------------------------------------------------------------------------------------------------
SWEEP_SEEDS = tuple(range(1000, 1020))      # feste Zufalls-Seeds: nie ein einzelner Lauf als Befund
SWEEP_CAP = 4_500_000
SWEEP_ALPHAS = (0.05, 0.10, 0.15, 0.30, 0.50)
SWEEP_BETAS = (0.4, 2.0, 10.0, 20.0)         # in 1e-5 (0,4 = Calvano et al.: 4·10⁻⁶)
SWEEP_GAMMAS = (0.0, 0.25, 0.50, 0.75, 0.95)

COLOR_A = "#2F6B65"
COLOR_B = "#C0662B"
COLOR_NASH = "#4c78a8"
COLOR_MONOPOLY = "#B5483A"
COLOR_NEUTRAL = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (Voreinstellung)": {"alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA, "gamma": DEFAULT_GAMMA, "memory": "both", "seed": DEFAULT_SEED},
    "Kein Gedächtnis": {"alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA, "gamma": DEFAULT_GAMMA, "memory": "none", "seed": DEFAULT_SEED},
    "Kurzsichtig (γ = 0)": {"alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA, "gamma": 0.0, "memory": "both", "seed": DEFAULT_SEED},
    "Mittlere Voraussicht (γ = 0,5)": {"alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA, "gamma": 0.5, "memory": "both", "seed": DEFAULT_SEED},
    "Kurze Erkundung (β = 20·10⁻⁵)": {"alpha": DEFAULT_ALPHA, "beta": 20, "gamma": DEFAULT_GAMMA, "memory": "both", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Zwei Lernende wie bei Calvano et al.: Gedächtnis der letzten beiden Preise, γ = 0,95. Ein Lauf braucht im Median rund 600.000 Perioden.",
    "Kein Gedächtnis": "Beide kennen nur ihre eigenen Erfahrungen, nicht den letzten Preis des anderen: Eine Vergeltung ist gar nicht möglich.",
    "Kurzsichtig (γ = 0)": "Beide zählen nur den Gewinn der laufenden Periode: Es gibt keinen Grund, auf spätere Perioden Rücksicht zu nehmen.",
    "Mittlere Voraussicht (γ = 0,5)": "Spätere Perioden zählen halb so viel wie die laufende.",
    "Kurze Erkundung (β = 20·10⁻⁵)": "Die Erkundung klingt zehnmal schneller ab als im Standardfall: Beide legen sich früh fest.",
}
