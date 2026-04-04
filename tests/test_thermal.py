"""Tests — core.thermal"""

import pytest
import numpy as np
from thermal_engine.core.thermal import (
    u_value_from_layers,
    u_value_ground_floor_iso13370,
    transmission_heat_loss_coefficient,
    transmission_losses_monthly,
    thermal_time_constant_h,
    effective_heat_capacity,
    HOURS_PER_MONTH,
)


# ─── u_value_from_layers ─────────────────────────────────────────────────────

def test_u_value_concrete_200mm():
    # Béton 20 cm (λ=2.3) seul
    layers = [(0.20, 2.30)]
    u = u_value_from_layers(layers)
    r = 0.13 + 0.20 / 2.30 + 0.04
    assert abs(u - 1 / r) < 1e-6


def test_u_value_insulated_wall():
    # Béton 20 cm + laine 10 cm (λ=0.035)
    layers = [(0.20, 2.30), (0.10, 0.035)]
    u = u_value_from_layers(layers)
    assert u < 0.35   # Bien isolé
    assert u > 0.0


def test_u_value_decreases_with_more_insulation():
    layers_base  = [(0.20, 2.30), (0.10, 0.035)]
    layers_more  = [(0.20, 2.30), (0.20, 0.035)]
    assert u_value_from_layers(layers_more) < u_value_from_layers(layers_base)


def test_u_value_empty_layers_raises():
    with pytest.raises(ValueError):
        u_value_from_layers([])


def test_u_value_custom_surface_resistances():
    layers = [(0.20, 2.30)]
    u_std  = u_value_from_layers(layers, rsi=0.13, rse=0.04)
    u_roof = u_value_from_layers(layers, rsi=0.10, rse=0.04)
    # Rsi plus faible → R total plus faible → U plus élevé
    assert u_roof > u_std


def test_u_value_passive_house_wall():
    # Mur BBC : béton + 20 cm laine + placo
    layers = [(0.20, 2.30), (0.20, 0.035), (0.013, 0.21)]
    u = u_value_from_layers(layers)
    assert u < 0.18   # Niveau BBC/RE2020


# ─── u_value_ground_floor_iso13370 ───────────────────────────────────────────

def test_ground_floor_u_value_positive():
    u = u_value_ground_floor_iso13370(
        floor_area_m2=50.0, perimeter_m=30.0,
        wall_u_value=1.5,
        floor_layers=[(0.20, 2.30), (0.08, 0.038)],
    )
    assert 0.0 < u < 2.0


def test_ground_floor_u_value_decreases_with_insulation():
    base = u_value_ground_floor_iso13370(
        50.0, 30.0, 1.5, [(0.20, 2.30)]
    )
    insulated = u_value_ground_floor_iso13370(
        50.0, 30.0, 1.5, [(0.20, 2.30), (0.10, 0.038)]
    )
    assert insulated < base


# ─── transmission_heat_loss_coefficient ──────────────────────────────────────

def test_ht_single_surface():
    elements = [{"type": "surface", "u_value": 1.0, "area_m2": 10.0, "b_factor": 1.0}]
    assert abs(transmission_heat_loss_coefficient(elements) - 10.0) < 1e-9


def test_ht_multiple_surfaces():
    elements = [
        {"type": "surface", "u_value": 2.0, "area_m2": 5.0,  "b_factor": 1.0},
        {"type": "surface", "u_value": 1.5, "area_m2": 10.0, "b_factor": 1.0},
    ]
    expected = 2.0 * 5.0 + 1.5 * 10.0
    assert abs(transmission_heat_loss_coefficient(elements) - expected) < 1e-9


def test_ht_with_linear_bridge():
    elements = [
        {"type": "surface",       "u_value": 1.0, "area_m2": 10.0, "b_factor": 1.0},
        {"type": "linear_bridge", "psi": 0.1,     "length_m": 20.0, "b_factor": 1.0},
    ]
    expected = 10.0 + 0.1 * 20.0
    assert abs(transmission_heat_loss_coefficient(elements) - expected) < 1e-9


def test_ht_b_factor_zero_ignored():
    elements = [
        {"type": "surface", "u_value": 2.0, "area_m2": 10.0, "b_factor": 0.0},
    ]
    assert transmission_heat_loss_coefficient(elements) == 0.0


def test_ht_default_b_factor_is_one():
    elements = [{"type": "surface", "u_value": 1.0, "area_m2": 5.0}]
    assert abs(transmission_heat_loss_coefficient(elements) - 5.0) < 1e-9


# ─── transmission_losses_monthly ─────────────────────────────────────────────

def test_monthly_losses_shape():
    losses = transmission_losses_monthly(100.0, 19.0, [3.0] * 12)
    assert len(losses) == 12


def test_monthly_losses_cold_month_higher():
    monthly_temps = [3.0, 4.0, 8.0, 12.0, 16.0, 20.0,
                     22.0, 22.0, 18.0, 13.0, 7.0, 4.0]
    losses = transmission_losses_monthly(100.0, 19.0, monthly_temps)
    # Janvier (3°C) doit avoir plus de pertes que Juillet (22°C)
    assert losses[0] > losses[6]


def test_monthly_losses_zero_above_setpoint():
    # Si temp extérieure > consigne, pas de pertes
    losses = transmission_losses_monthly(100.0, 19.0, [25.0] * 12)
    assert all(l == 0.0 for l in losses)


def test_monthly_losses_unit():
    # H_T=100 W/K, ΔT=10°C, 744h (janv) → 100 * 10 * 744 / 1000 = 744 kWh
    losses = transmission_losses_monthly(100.0, 19.0, [9.0] * 12)
    jan_expected = 100.0 * 10.0 * HOURS_PER_MONTH[0] / 1000.0
    assert abs(losses[0] - jan_expected) < 0.1


# ─── effective_heat_capacity ─────────────────────────────────────────────────

def test_heat_capacity_medium():
    c = effective_heat_capacity(100.0, "medium")
    assert c == 165.0 * 100.0


@pytest.mark.parametrize("cls", ["very_light", "light", "medium", "heavy", "very_heavy"])
def test_heat_capacity_all_classes_positive(cls):
    assert effective_heat_capacity(100.0, cls) > 0


def test_heat_capacity_increases_with_mass():
    c_light = effective_heat_capacity(100.0, "light")
    c_heavy = effective_heat_capacity(100.0, "heavy")
    assert c_heavy > c_light


# ─── thermal_time_constant_h ─────────────────────────────────────────────────

def test_time_constant_formula():
    c   = 16500.0  # kJ/K
    h_t = 100.0    # W/K
    h_v = 50.0     # W/K
    tau = thermal_time_constant_h(c, h_t, h_v)
    assert abs(tau - 16500.0 / 150.0) < 1e-6


def test_time_constant_zero_h():
    assert thermal_time_constant_h(1000.0, 0.0, 0.0) == 0
