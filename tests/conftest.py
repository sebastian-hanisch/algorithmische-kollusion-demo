"""Gemeinsame Fixtures: kleine, von Hand rechenbare Lernergebnisse (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import akl_algorithm as A  # noqa: E402


@pytest.fixture
def two_price_result():
    """Zwei Preise (1,4; 1,9), beide spielen in jedem Zustand den hohen Preis. Zustand = 2·a1 + a2."""
    return A.LearningResult(periods=0, converged=True, memory=True, prices=[1.4, 1.9], greedy=[[1, 1, 1, 1], [1, 1, 1, 1]], cycle_states=[3],
                            cycle_prices=[(1.9, 1.9)], avg_profit=0.0, delta=0.0, avg_price=1.9)


@pytest.fixture
def punishing_result():
    """Wie two_price_result, aber der zweite Agent senkt im Zustand (niedrig, hoch) = 1 auf den niedrigen Preis (eine Periode Strafe)."""
    return A.LearningResult(periods=0, converged=True, memory=True, prices=[1.4, 1.9], greedy=[[1, 1, 1, 1], [1, 0, 1, 1]], cycle_states=[3],
                            cycle_prices=[(1.9, 1.9)], avg_profit=0.0, delta=0.0, avg_price=1.9)
