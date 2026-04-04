"""Tests d'intégration — simulation.needs + simulation.renovation"""

import pytest
import numpy as np
import pandas as pd

from thermal_engine.climate.epw_models import WeatherSeries, EPWLocation
from thermal_engine.io.geojson_loader import load_building
from thermal_engine.simulation.needs import compute_building_needs
from thermal_engine.simulation.renovation import (
    InsulateWalls,
    InsulateRoof,
    ReplaceWindows,
    ReplaceHeatingSystem,
    InstallMVHR,
    RenovationScenario,
    simulate_renovation,
    build_standard_scenarios,
    simulate_multiple_scenarios,
)
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_weather():
    """Météo synthétique Lyon pour les tests (pas de fichier EPW requis)."""
    ts = pd.date_range("2023-01-01 01:00", periods=8760, freq="h")
    monthly_temps = [3.0, 4.5, 8.0, 11.5, 15.5, 19.5, 22.5, 22.0,
                     17.5, 12.5, 6.5, 3.5]
    months = ts.month.to_numpy() - 1
    t_base = np.array(monthly_temps)[months]
    dry_bulb = t_base + 2.0 * np.sin(np.pi * ts.hour.to_numpy() / 12)

    monthly_ghi = [30, 50, 90, 130, 155, 170, 185, 165, 120, 75, 35, 25]
    ghi = np.zeros(8760)
    for m in range(12):
        mask = (months == m) & (ts.hour.to_numpy() >= 6) & (ts.hour.to_numpy() <= 20)
        if mask.sum() > 0:
            peak = monthly_ghi[m] * 1000 / (mask.sum() * 0.6)
            ghi[mask] = np.maximum(0, peak * np.exp(-0.5 * ((ts.hour.to_numpy()[mask] - 13) / 4) ** 2))

    return WeatherSeries(
        location=EPWLocation("Lyon", "ARA", "France", "synthetic", "07481",
                             45.756, 4.854, 1.0, 200.0),
        dry_bulb_temp_c=dry_bulb,
        dew_point_temp_c=dry_bulb - 3,
        relative_humidity=np.full(8760, 70.0),
        atmospheric_pressure_pa=np.full(8760, 101325.0),
        ghi_wh_m2=ghi, dhi_wh_m2=ghi * 0.3,
        dni_wh_m2=np.maximum(0, ghi * 0.7),
        wind_speed_m_s=np.full(8760, 2.5),
        wind_direction_deg=np.full(8760, 180.0),
        timestamps=ts,
    )


@pytest.fixture(scope="module")
def sample_building():
    geojson_path = EXAMPLES_DIR / "batiment_mixte_lyon.geojson"
    if not geojson_path.exists():
        pytest.skip(f"Fichier GeoJSON manquant : {geojson_path}")
    return load_building(geojson_path)


# ─── Tests de chargement du bâtiment ─────────────────────────────────────────

def test_building_loaded(sample_building):
    assert sample_building.building_id == "bat_mixte_lyon_001"
    assert sample_building.n_zones == 2


def test_building_zones_have_area(sample_building):
    for zone in sample_building.zones:
        assert zone.floor_area_m2 > 0


def test_building_total_area(sample_building):
    assert sample_building.total_floor_area_m2 > 0


def test_building_adjacency_detected(sample_building):
    # Les deux zones ont la même emprise → elles devraient être adjacentes
    z0 = sample_building.zones[0]
    z1 = sample_building.zones[1]
    assert (z1.zone_id in z0.adjacent_zone_ids or
            z0.zone_id in z1.adjacent_zone_ids)


# ─── Tests de simulation des besoins ─────────────────────────────────────────

def test_compute_needs_returns_result(sample_building, synthetic_weather):
    result = compute_building_needs(sample_building, synthetic_weather)
    assert result is not None
    assert result.building_id == sample_building.building_id


def test_compute_needs_dpe_class(sample_building, synthetic_weather):
    result = compute_building_needs(sample_building, synthetic_weather)
    assert result.dpe_class in list("ABCDEFG")


def test_compute_needs_positive_values(sample_building, synthetic_weather):
    result = compute_building_needs(sample_building, synthetic_weather)
    assert result.heating_need_kwh >= 0
    assert result.final_energy_kwh > 0
    assert result.primary_energy_kwh > 0
    assert result.cost_eur > 0
    assert result.co2_kg > 0


def test_compute_needs_ep_per_m2(sample_building, synthetic_weather):
    result = compute_building_needs(sample_building, synthetic_weather)
    # Ordre de grandeur attendu pour un bâtiment des années 70 : 100–400 kWh/m²/an
    assert 50 < result.primary_energy_kwh_m2 < 600


def test_compute_needs_12_monthly_values(sample_building, synthetic_weather):
    result = compute_building_needs(sample_building, synthetic_weather)
    for zr in result.zone_results:
        assert len(zr.heating_need_monthly) == 12
        assert len(zr.solar_gains_monthly) == 12


def test_compute_needs_hourly_method(sample_building, synthetic_weather):
    result = compute_building_needs(sample_building, synthetic_weather, method="hourly")
    assert result.dpe_class in list("ABCDEFG")
    assert result.heating_need_kwh >= 0


def test_compute_needs_monthly_vs_hourly_order_of_magnitude(sample_building, synthetic_weather):
    monthly = compute_building_needs(sample_building, synthetic_weather, method="monthly")
    hourly  = compute_building_needs(sample_building, synthetic_weather, method="hourly")
    # Les deux méthodes doivent donner des résultats dans le même ordre de grandeur (±50%)
    ratio = monthly.heating_need_kwh / max(1, hourly.heating_need_kwh)
    assert 0.5 < ratio < 2.0


# ─── Tests de rénovation ─────────────────────────────────────────────────────

def test_insulate_walls_reduces_u(sample_building):
    zone_before = sample_building.zones[0]
    u_before = zone_before.envelope.walls.u_value_w_m2k

    action = InsulateWalls(
        action_id="test_ite", label="ITE test",
        description="Test isolation",
        insulation_material_id="mineral_wool",
        insulation_thickness_m=0.14,
        cost_min_eur=10000, cost_max_eur=20000,
    )
    new_building = action.apply(sample_building)
    u_after = new_building.zones[0].envelope.walls.u_value_w_m2k

    assert u_after < u_before


def test_renovation_original_unchanged(sample_building):
    u_original = sample_building.zones[0].envelope.walls.u_value_w_m2k

    action = InsulateWalls(
        action_id="test", label="Test", description="Test",
        insulation_material_id="mineral_wool", insulation_thickness_m=0.20,
        cost_min_eur=5000, cost_max_eur=10000,
    )
    _ = action.apply(sample_building)

    assert abs(sample_building.zones[0].envelope.walls.u_value_w_m2k - u_original) < 1e-9


def test_renovation_reduces_heating_need(sample_building, synthetic_weather):
    baseline = compute_building_needs(sample_building, synthetic_weather)

    scenario = RenovationScenario(
        scenario_id="test", label="Test", description="Test",
        actions=[
            InsulateRoof(
                action_id="roof", label="Toiture", description="Isolation toiture",
                insulation_material_id="mineral_wool", insulation_thickness_m=0.25,
                cost_min_eur=5000, cost_max_eur=8000,
            ),
        ],
    )
    result = simulate_renovation(sample_building, scenario, synthetic_weather, baseline=baseline)
    assert result.after.heating_need_kwh <= result.baseline.heating_need_kwh


def test_renovation_result_cost_savings_positive(sample_building, synthetic_weather):
    scenarios = build_standard_scenarios(sample_building)
    # Le scénario complet doit économiser de l'argent
    complete_scenario = scenarios[-1]  # BBC
    result = simulate_renovation(sample_building, complete_scenario, synthetic_weather)
    assert result.cost_savings_eur_per_year > 0


def test_renovation_dpe_improves_with_bbc(sample_building, synthetic_weather):
    scenarios = build_standard_scenarios(sample_building)
    baseline  = compute_building_needs(sample_building, synthetic_weather)
    bbc       = simulate_renovation(sample_building, scenarios[-1], synthetic_weather, baseline)
    order = list("ABCDEFG")
    assert order.index(bbc.after.dpe_class) <= order.index(baseline.dpe_class)


def test_simulate_multiple_scenarios(sample_building, synthetic_weather):
    scenarios = build_standard_scenarios(sample_building)
    results   = simulate_multiple_scenarios(sample_building, scenarios, synthetic_weather)
    assert len(results) == len(scenarios)
    # Chaque scénario doit avoir la même baseline
    ep_baseline = results[0].baseline.primary_energy_kwh
    for r in results[1:]:
        assert abs(r.baseline.primary_energy_kwh - ep_baseline) < 1e-3


def test_standard_scenarios_increasing_cost(sample_building):
    scenarios = build_standard_scenarios(sample_building)
    costs = [s.total_cost_center_eur for s in scenarios]
    assert costs[0] < costs[1] < costs[2]


# ─── Tests to_dict / JSON-serialisable ───────────────────────────────────────

def test_needs_result_to_dict(sample_building, synthetic_weather):
    import json
    result = compute_building_needs(sample_building, synthetic_weather)
    d = result.to_dict()
    # Doit être sérialisable sans erreur
    serialized = json.dumps(d)
    assert len(serialized) > 100


def test_renovation_result_to_dict(sample_building, synthetic_weather):
    import json
    scenarios = build_standard_scenarios(sample_building)
    result    = simulate_renovation(sample_building, scenarios[0], synthetic_weather)
    d = result.to_dict()
    serialized = json.dumps(d)
    assert "dpe_improvement" in d
    assert len(serialized) > 100
