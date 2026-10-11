"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button."""

import random
from dataclasses import dataclass

import streamlit as st

import akl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX, C.ALPHA_STEP),
    "beta_slider": SettingSpec("beta", int, C.DEFAULT_BETA, C.BETA_MIN, C.BETA_MAX, C.BETA_STEP),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX, C.GAMMA_STEP),
    "memory_select": SettingSpec("memory", str, C.DEFAULT_MEMORY, choices=C.MEMORY_KEYS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, C.SEED_MIN, C.SEED_MAX),
}
PRESET_KEYS = {"alpha": "alpha_slider", "beta": "beta_slider", "gamma": "gamma_slider", "memory": "memory_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp(spec: SettingSpec, value):
    """Begrenzt auf die Reglergrenzen und rastet auf den Reglerschritt ein; Auswahlwerte außerhalb der Liste werden zur Voreinstellung."""
    if spec.choices is not None:
        return value if value in spec.choices else spec.default
    value = min(spec.hi, max(spec.lo, value))
    if spec.step:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, max(spec.lo, value))
    return round(value, 2) if spec.caster is float else int(round(value))


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        raw = qp[spec.url_param]
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        try:
            value = spec.caster(raw)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = clamp(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(C.SEED_MIN, C.SEED_MAX)
