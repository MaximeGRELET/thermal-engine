# thermal-engine

Physics-based building energy analysis library — ISO 13790, multi-zone GeoJSON, EPW weather, renovation scenarios.

## Overview

`thermal-engine` is a standalone Python package that computes building heating and cooling needs from first principles. It takes a **GeoJSON FeatureCollection** describing building geometry and a **EPW weather file** (or synthetic weather data) as inputs, and returns structured JSON results including DPE classification, energy balances, and renovation scenario comparisons.

### Key capabilities

- **ISO 13790** monthly quasi-static method and hourly RC 1-node simulation
- **ISO 6946** U-value calculation for layered wall/roof/floor compositions
- **ISO 13370** equivalent U-value for ground floors in contact with soil
- **EN ISO 14683** linear thermal bridge Ψ coefficients (3 quality levels)
- **Hay-Davies (1980)** irradiance model on tilted surfaces
- **Spencer (1971)** solar position equations
- **DPE décret 2021** dual criterion (primary energy + CO₂), worst class wins
- **Multi-zone** buildings with automatic adjacency detection via Shapely
- **Immutable renovation actions** — original building is never mutated
- **EPW parser** — full 8760-h EnergyPlus Weather file support

---

## Installation

```bash
# From source (development)
pip install -e ".[dev]"

# As a dependency in another project
pip install git+https://github.com/MaximeGRELET/thermal-engine.git
```

**Requirements**: Python ≥ 3.11, numpy ≥ 1.26, pandas ≥ 2.1, scipy ≥ 1.11, shapely ≥ 2.0

---

## Quick start

```python
from thermal_engine.io.geojson_loader import load_building
from thermal_engine.climate.epw_parser import parse_epw
from thermal_engine.simulation.needs import compute_building_needs
from thermal_engine.simulation.renovation import build_standard_scenarios, simulate_multiple_scenarios

# Load building geometry from GeoJSON
building = load_building("examples/batiment_mixte_lyon.geojson")

# Load weather data from EPW file
weather = parse_epw("Lyon.epw")

# Compute baseline energy needs (ISO 13790 monthly method)
result = compute_building_needs(building, weather)
print(f"DPE class: {result.dpe_class}")
print(f"Primary energy: {result.primary_energy_kwh_m2:.0f} kWh EP/m²/an")
print(f"CO₂: {result.co2_kg_m2:.1f} kg CO₂/m²/an")

# Compare standard renovation scenarios
scenarios = build_standard_scenarios(building)
results = simulate_multiple_scenarios(building, scenarios, weather)
for r in results:
    print(f"{r.scenario.label}: {r.after.dpe_class} | {r.cost_savings_eur_per_year:.0f} €/an | ROI {r.simple_payback_years:.0f} ans")

# Export to JSON
import json
print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
```

---

## GeoJSON input format

Each Feature represents one thermal zone. The polygon must be a 2D footprint (WGS84 coordinates or local metric).

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [4.854000, 45.756000],
          [4.854154, 45.756000],
          [4.854154, 45.756090],
          [4.854000, 45.756090],
          [4.854000, 45.756000]
        ]]
      },
      "properties": {
        "zone_id": "rdc_commercial",
        "building_id": "bat_mixte_lyon_001",
        "zone_type": "commercial",
        "height_m": 4.0,
        "construction_year": 1972,
        "floors": 1,
        "is_ground_floor": true,
        "has_roof": false,
        "heating_setpoint_c": 19.0,
        "cooling_setpoint_c": 26.0,
        "thermal_mass_class": "heavy",
        "infiltration_ach": 0.6,
        "envelope": {
          "walls": {
            "layers": [
              {"material_id": "concrete_dense", "thickness_m": 0.20},
              {"material_id": "mineral_wool",   "thickness_m": 0.08},
              {"material_id": "plasterboard",   "thickness_m": 0.013}
            ]
          },
          "roof": {
            "layers": [
              {"material_id": "concrete_dense", "thickness_m": 0.20},
              {"material_id": "mineral_wool",   "thickness_m": 0.12}
            ]
          },
          "ground_floor": {
            "layers": [{"material_id": "concrete_dense", "thickness_m": 0.25}]
          },
          "windows": {
            "u_value_w_m2k": 2.8,
            "g_value": 0.6,
            "wwr_by_orientation": {"S": 0.30, "N": 0.10, "E": 0.15, "O": 0.15}
          }
        },
        "energy_system": {
          "system_id": "chaudiere_gaz",
          "type": "gas_boiler",
          "covers": "heating",
          "efficiency_nominal": 0.87,
          "fuel": "natural_gas"
        }
      }
    }
  ]
}
```

### Supported material IDs

| ID | Description | λ (W/m·K) |
|----|-------------|-----------|
| `concrete_dense` | Béton lourd | 2.30 |
| `concrete_cellular` | Béton cellulaire | 0.11 |
| `brick_hollow` | Brique creuse | 0.40 |
| `brick_solid` | Brique pleine | 0.84 |
| `mineral_wool` | Laine minérale | 0.035 |
| `glass_wool` | Laine de verre | 0.032 |
| `eps_insulation` | Polystyrène expansé | 0.038 |
| `xps_insulation` | Polystyrène extrudé | 0.033 |
| `wood_fiber` | Fibre de bois | 0.038 |
| `cellulose` | Ouate de cellulose | 0.040 |
| `plasterboard` | Plaque de plâtre | 0.21 |
| `timber_frame` | Ossature bois | 0.13 |

Full list: [`src/thermal_engine/data/material_db.py`](src/thermal_engine/data/material_db.py)

---

## Simulation methods

### Monthly (ISO 13790 §12)

Default method. Computes monthly heating/cooling needs using the quasi-static utilization factor approach:

```
Q_H = Q_loss - η_H × Q_gain    [kWh/month]
```

Where η_H = f(γ_H, a_H) is the dimensionless gain utilization factor depending on the gain/loss ratio γ_H and the building time constant τ.

### Hourly RC (ISO 13790 Annex C)

Optional 1-node resistor-capacitor model:

```python
result = compute_building_needs(building, weather, method="hourly")
```

The RC model resolves diurnal temperature swings and provides hourly indoor temperature profiles. Results are within ±50% of the monthly method for annual totals.

---

## Renovation scenarios

Renovation actions are immutable — `apply()` returns a deep copy of the building with modified parameters. The original is never mutated.

```python
from thermal_engine.simulation.renovation import (
    InsulateWalls, InsulateRoof, InsulateFloor,
    ReplaceWindows, ReplaceHeatingSystem, InstallMVHR,
    RenovationScenario, simulate_renovation,
)

action = InsulateWalls(
    action_id="ite_14cm",
    label="ITE 14 cm laine minérale",
    description="Isolation thermique par l'extérieur",
    insulation_material_id="mineral_wool",
    insulation_thickness_m=0.14,
    cost_min_eur=15000,
    cost_max_eur=22000,
)

scenario = RenovationScenario(
    scenario_id="scenario_1",
    label="Isolation enveloppe",
    description="ITE + isolation toiture",
    actions=[action, InsulateRoof(...)],
)

result = simulate_renovation(building, scenario, weather)
print(f"Gain DPE: {result.baseline.dpe_class} → {result.after.dpe_class}")
print(f"Économies: {result.cost_savings_eur_per_year:.0f} €/an")
print(f"Retour sur investissement: {result.simple_payback_years:.0f} ans")
```

### Built-in standard scenarios

```python
scenarios = build_standard_scenarios(building)
# Returns 3 scenarios ordered by increasing cost/ambition:
# [0] Confort+      — walls + windows
# [1] Rénovation RT — walls + roof + windows + heating system
# [2] BBC           — full envelope + MVHR + heat pump
```

---

## Package structure

```
thermal-engine/
├── src/thermal_engine/
│   ├── core/
│   │   ├── geometry.py       # Polygon area, wall segments, azimuth
│   │   ├── thermal.py        # U-values ISO 6946, H_T, monthly losses
│   │   ├── ventilation.py    # H_V, MVHR recovery efficiency
│   │   ├── solar.py          # Solar position, Hay-Davies, window gains
│   │   ├── schedules.py      # Occupancy profiles, internal gains (8760h)
│   │   └── systems.py        # System KPIs, DPE classification
│   ├── climate/
│   │   ├── epw_models.py     # WeatherSeries dataclass (8760-h numpy arrays)
│   │   └── epw_parser.py     # parse_epw() — reads .epw files
│   ├── models/
│   │   ├── materials.py      # MaterialLayer, LayeredComposition
│   │   ├── energy_systems.py # GasBoiler, HeatPump, SolarThermal, MVHR
│   │   └── building.py       # Building, Zone, EnvelopeConfig, WindowConfig
│   ├── simulation/
│   │   ├── needs.py          # compute_building_needs() — ISO 13790
│   │   └── renovation.py     # RenovationAction subclasses, simulate_renovation()
│   ├── io/
│   │   ├── geojson_loader.py # load_building() from GeoJSON FeatureCollection
│   │   └── report_builder.py # build_analysis_report(), save_report()
│   └── data/
│       ├── material_db.py        # ~30 materials per EN ISO 10456
│       └── thermal_bridges_db.py # Ψ coefficients per EN ISO 14683
├── tests/
│   ├── test_geometry.py    # 35 tests
│   ├── test_thermal.py     # 27 tests
│   ├── test_solar.py       # 17 tests
│   ├── test_materials.py   # 17 tests
│   └── test_simulation.py  # 20 integration tests (133 total)
├── examples/
│   └── batiment_mixte_lyon.geojson
└── pyproject.toml
```

---

## Running tests

```bash
# All tests
pytest

# With coverage
pytest --cov=thermal_engine --cov-report=html

# Specific module
pytest tests/test_simulation.py -v
```

---

## Output JSON structure

`result.to_dict()` returns:

```json
{
  "building_id": "bat_mixte_lyon_001",
  "method": "monthly",
  "dpe_class": "D",
  "dpe_co2_class": "C",
  "heating_need_kwh": 12450.3,
  "cooling_need_kwh": 890.1,
  "final_energy_kwh": 14300.5,
  "primary_energy_kwh": 15158.5,
  "primary_energy_kwh_m2": 186.5,
  "co2_kg": 2860.1,
  "co2_kg_m2": 35.2,
  "cost_eur": 1716.1,
  "total_floor_area_m2": 81.3,
  "zone_results": [
    {
      "zone_id": "rdc_commercial",
      "heating_need_kwh": 7230.0,
      "heating_need_monthly": [980, 850, 700, 420, 180, 20, 0, 0, 80, 350, 720, 930],
      "solar_gains_monthly": [120, 180, 310, 420, 510, 560, 580, 530, 400, 260, 140, 100],
      "h_t_w_k": 85.3,
      "h_v_w_k": 42.1,
      "floor_area_m2": 47.3
    }
  ]
}
```

---

## Standards and references

| Standard | Scope |
|----------|-------|
| ISO 13790:2008 | Energy performance of buildings — calculation of energy use for space heating and cooling |
| ISO 6946:2017 | Building components — thermal resistance and transmittance |
| ISO 13370:2017 | Thermal performance — heat transfer via the ground |
| EN ISO 14683:2017 | Thermal bridges — linear thermal transmittance |
| EN ISO 9806:2017 | Solar energy collectors — test methods |
| EnergyPlus EPW | Weather data format (NREL/DOE) |
| Hay & Davies (1980) | Solar radiation on inclined surfaces |
| Spencer (1971) | Solar position equations |
| Décret DPE 2021 | French energy performance certificate method |

---

## License

MIT — see [LICENSE](LICENSE)
