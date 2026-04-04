"""Tests — models.materials + data.material_db"""

import pytest
from thermal_engine.data.material_db import get_material, MATERIAL_DATABASE
from thermal_engine.models.materials import (
    MaterialLayer,
    LayeredComposition,
    composition_from_dict,
)


# ─── material_db ─────────────────────────────────────────────────────────────

def test_get_known_material():
    mat = get_material("mineral_wool")
    assert mat.lambda_w_mk == 0.035
    assert mat.rho_kg_m3 == 30


def test_get_unknown_material_raises():
    with pytest.raises(KeyError, match="inconnu"):
        get_material("unknown_material_xyz")


def test_all_materials_positive_lambda():
    for mat_id, mat in MATERIAL_DATABASE.items():
        assert mat.lambda_w_mk > 0, f"{mat_id}: λ doit être > 0"


def test_all_materials_positive_rho():
    for mat_id, mat in MATERIAL_DATABASE.items():
        assert mat.rho_kg_m3 > 0, f"{mat_id}: ρ doit être > 0"


# ─── MaterialLayer ───────────────────────────────────────────────────────────

def test_layer_thermal_resistance():
    layer = MaterialLayer("mineral_wool", 0.10)
    expected_r = 0.10 / 0.035
    assert abs(layer.thermal_resistance_m2k_w - expected_r) < 1e-9


def test_layer_heat_capacity():
    layer = MaterialLayer("concrete_dense", 0.20)
    # C = ρ × Cp × e / 1000 [kJ/m²K]
    expected = 2300 * 1000 * 0.20 / 1000
    assert abs(layer.heat_capacity_kj_m2k - expected) < 1e-6


def test_layer_resolves_material():
    layer = MaterialLayer("eps_insulation", 0.08)
    assert layer.lambda_w_mk == 0.038
    assert "Polystyrène" in layer.description


# ─── LayeredComposition ──────────────────────────────────────────────────────

def test_composition_u_value_single_layer():
    comp = LayeredComposition([MaterialLayer("mineral_wool", 0.10)])
    r = 0.13 + 0.10 / 0.035 + 0.04
    assert abs(comp.u_value_w_m2k - 1 / r) < 1e-6


def test_composition_u_value_multilayer():
    comp = LayeredComposition([
        MaterialLayer("concrete_dense", 0.20),
        MaterialLayer("mineral_wool",   0.10),
        MaterialLayer("plasterboard",   0.013),
    ])
    r = 0.13 + 0.20 / 2.30 + 0.10 / 0.035 + 0.013 / 0.21 + 0.04
    assert abs(comp.u_value_w_m2k - 1 / r) < 1e-5


def test_composition_total_thickness():
    comp = LayeredComposition([
        MaterialLayer("concrete_dense", 0.20),
        MaterialLayer("mineral_wool",   0.10),
    ])
    assert abs(comp.total_thickness_m - 0.30) < 1e-9


def test_composition_add_insulation_exterior():
    original = LayeredComposition([MaterialLayer("concrete_dense", 0.20)])
    improved = original.add_insulation_layer("mineral_wool", 0.12, "exterior")
    assert improved.u_value_w_m2k < original.u_value_w_m2k
    assert len(improved.layers) == 2
    # Couche ajoutée en premier (extérieur)
    assert improved.layers[0].material_id == "mineral_wool"


def test_composition_add_insulation_interior():
    original = LayeredComposition([MaterialLayer("concrete_dense", 0.20)])
    improved = original.add_insulation_layer("mineral_wool", 0.12, "interior")
    assert improved.layers[-1].material_id == "mineral_wool"


def test_composition_original_unchanged_after_add():
    original = LayeredComposition([MaterialLayer("concrete_dense", 0.20)])
    u_before = original.u_value_w_m2k
    _ = original.add_insulation_layer("mineral_wool", 0.12)
    assert abs(original.u_value_w_m2k - u_before) < 1e-9


def test_composition_to_dict():
    comp = LayeredComposition([MaterialLayer("mineral_wool", 0.10)])
    d = comp.to_dict()
    assert "u_value_w_m2k" in d
    assert "layers" in d
    assert len(d["layers"]) == 1


# ─── composition_from_dict ───────────────────────────────────────────────────

def test_composition_from_dict_roundtrip():
    data = {
        "layers": [
            {"material_id": "concrete_dense", "thickness_m": 0.20},
            {"material_id": "mineral_wool",   "thickness_m": 0.10},
        ],
        "surface_resistance_interior": 0.13,
        "surface_resistance_exterior": 0.04,
    }
    comp = composition_from_dict(data)
    assert len(comp.layers) == 2
    assert comp.layers[0].material_id == "concrete_dense"
    assert comp.u_value_w_m2k < 1.0
