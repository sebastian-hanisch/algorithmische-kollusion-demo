"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzen und Einrasten."""

import akl_constants as C
import akl_evaluation as ev
import akl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX and C.BETA_MIN <= p["beta"] <= C.BETA_MAX and C.GAMMA_MIN <= p["gamma"] <= C.GAMMA_MAX, name
        assert p["memory"] in C.MEMORY_KEYS and C.SEED_MIN <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert ev.Settings(first["alpha"], first["beta"], first["gamma"], first["memory"], first["seed"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_clamp_limits_and_snaps():
    spec = P.SETTING_SPECS
    assert P.clamp(spec["alpha_slider"], 9.0) == C.ALPHA_MAX and P.clamp(spec["alpha_slider"], -1.0) == C.ALPHA_MIN
    assert P.clamp(spec["alpha_slider"], 0.17) == 0.15                          # rastet auf den Schritt 0,05
    assert P.clamp(spec["gamma_slider"], 0.97) == 0.95                          # und bleibt im Bereich
    assert P.clamp(spec["beta_slider"], 99) == C.BETA_MAX and isinstance(P.clamp(spec["beta_slider"], 7.4), int)
    assert P.clamp(spec["memory_select"], "none") == "none" and P.clamp(spec["memory_select"], "xx") == C.DEFAULT_MEMORY


def test_every_preset_is_a_sweep_configuration_or_has_a_table_neighbour():
    """Jedes Preset außer der Seed-Wahl liegt in der vorgerechneten Tabelle: Die Hilfetexte können sich auf Mehr-Seed-Befunde stützen."""
    for name, p in C.PRESETS.items():
        assert ev.find_config(p["alpha"], float(p["beta"]), p["gamma"], p["memory"]) is not None, name
