"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an Minimum und Maximum
und prüft, dass kein Python-Fehler auftritt. Die Lernläufe sind für die Tests kurz gedeckelt (PERIOD_CAP), die Tabelle kommt vorgerechnet aus akl_sweeps.json."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import akl_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "\u25b6"


@pytest.fixture(autouse=True)
def short_runs(monkeypatch):
    monkeypatch.setattr(C, "PERIOD_CAP", 30_000)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_permalink_is_clamped_and_snapped():
    at = AppTest.from_file(APP_PATH)
    at.query_params["alpha"] = "9"
    at.query_params["beta"] = "-4"
    at.query_params["gamma"] = "0.97"
    at.query_params["memory"] = "xx"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["alpha_slider"] == 0.5
    assert at.session_state["beta_slider"] == 1
    assert at.session_state["gamma_slider"] == 0.95
    assert at.session_state["memory_select"] == "both"


def test_the_page_explains_the_learners_and_the_index():
    text = " ".join(m.value for m in _fresh_app().markdown)
    for word in ("Q-Lernen", "Erkundung", "Gedächtnis", "Voraussicht", "Kollusionsindex", "Nash-Preis", "Monopolpreis"):
        assert word in text, word


def test_without_memory_there_is_no_deviation_test_and_the_page_says_why():
    at = AppTest.from_file(APP_PATH)
    at.query_params["memory"] = "none"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Ohne Gedächtnis gibt es keinen Abweichungstest" in i.value for i in at.info)


def test_the_page_does_not_claim_a_legal_verdict():
    text = " ".join(m.value for m in _fresh_app().markdown)
    assert "keine rechtliche Bewertung" in text
