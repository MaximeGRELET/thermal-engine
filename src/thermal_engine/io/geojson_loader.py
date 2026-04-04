"""
Chargeur GeoJSON → Building.

Parse un fichier FeatureCollection GeoJSON décrivant un bâtiment multi-zones.
Valide les données, résout les matériaux et construit le modèle physique.

Schéma GeoJSON attendu :
  - FeatureCollection.properties : métadonnées du bâtiment
  - Chaque Feature : une zone thermique
    - geometry.type = "Polygon"
    - geometry.coordinates = emprise 2D [[x,y], ...]
    - properties.zone_id, .usage, .height_m, .n_floors
    - properties.envelope.walls / roof / ground_floor / windows
    - properties.energy_systems []
    - properties.ventilation {}
    - properties.occupancy {}
"""

from __future__ import annotations
import json
import copy
from pathlib import Path

from ..models.building import (
    Building, Zone, EnvelopeConfig, WindowConfig,
    OccupancyProfile, ThermalSetpoints,
)
from ..models.materials import LayeredComposition, MaterialLayer, composition_from_dict
from ..models.energy_systems import (
    EnergySystem, SolarThermalSystem, VentilationSystem,
    system_from_dict, ventilation_from_dict,
)
from ..core.geometry import (
    polygon_area_m2, compute_zone_geometry, find_shared_edges,
)


class GeoJSONValidationError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def load_building(source: str | Path | dict) -> Building:
    """
    Charge un bâtiment depuis un fichier GeoJSON ou un dict Python.

    Parameters
    ----------
    source : str | Path | dict
        Chemin vers un fichier .geojson / .json, ou dict déjà chargé.

    Returns
    -------
    Building
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Fichier GeoJSON introuvable : {path}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

    return _parse_feature_collection(data)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing de la FeatureCollection
# ─────────────────────────────────────────────────────────────────────────────

def _parse_feature_collection(data: dict) -> Building:
    """Parse le GeoJSON et construit le Building."""
    if data.get("type") != "FeatureCollection":
        raise GeoJSONValidationError(
            f"Le GeoJSON doit être de type 'FeatureCollection', reçu : {data.get('type')}"
        )

    features = data.get("features", [])
    if not features:
        raise GeoJSONValidationError("Le GeoJSON ne contient aucune feature (zone).")

    bprops = data.get("properties", {})
    building_id = bprops.get("building_id", data.get("id", "building_001"))
    name        = bprops.get("name", building_id)

    # Localisation
    loc_data = bprops.get("location", {})
    latitude  = float(loc_data.get("latitude", bprops.get("latitude", 48.86)))
    longitude = float(loc_data.get("longitude", bprops.get("longitude", 2.35)))
    city      = loc_data.get("city", bprops.get("city", ""))

    # Parse les zones
    zones = []
    for i, feature in enumerate(features):
        try:
            zone = _parse_zone_feature(feature, i)
        except (KeyError, ValueError) as exc:
            fid = feature.get("id", f"feature_{i}")
            raise GeoJSONValidationError(
                f"Erreur dans la feature '{fid}' : {exc}"
            ) from exc
        zones.append(zone)

    # Détection des adjacences entre zones (murs mitoyens)
    _compute_adjacencies(zones)

    building = Building(
        building_id = building_id,
        name        = name,
        zones       = zones,
        location    = (latitude, longitude),
        city        = city,
    )
    return building


def _parse_zone_feature(feature: dict, index: int) -> Zone:
    """Parse un Feature GeoJSON en Zone."""
    props   = feature.get("properties", {})
    geom    = feature.get("geometry", {})
    fid     = feature.get("id", props.get("zone_id", f"zone_{index:03d}"))

    # ─── Géométrie ───────────────────────────────────────────────
    if geom.get("type") != "Polygon":
        raise GeoJSONValidationError(
            f"Zone '{fid}' : geometry.type doit être 'Polygon', reçu '{geom.get('type')}'."
        )
    raw_coords = geom["coordinates"][0]   # Outer ring
    # Extrait uniquement x, y (ignore z si présent)
    footprint  = [[c[0], c[1]] for c in raw_coords]

    # Surface calculée géométriquement
    area = polygon_area_m2(footprint)

    # ─── Propriétés de base ──────────────────────────────────────
    zone_id   = props.get("zone_id", fid)
    label     = props.get("zone_label", props.get("label", zone_id))
    usage     = props.get("usage", "residential")
    height_m  = float(props.get("height_m", 3.0))
    n_floors  = int(props.get("n_floors", max(1, round(height_m / 3.0))))
    year_built= int(props.get("year_built", props.get("construction_year", 1975)))
    is_ground = bool(props.get("is_ground_floor", True))
    construction_class = props.get("construction_class", "medium")

    # ─── Enveloppe ───────────────────────────────────────────────
    env_data  = props.get("envelope", {})
    envelope  = _parse_envelope(env_data, year_built)

    # ─── Ventilation ─────────────────────────────────────────────
    vent_data = props.get("ventilation", {})
    if not vent_data:
        # Valeur par défaut selon l'année de construction
        from ..core.ventilation import ach_from_construction_year
        ach, _ = ach_from_construction_year(year_built, "maison")
        vent_data = {"type": "natural", "air_change_rate_h": ach}
    ventilation = ventilation_from_dict(vent_data)

    # ─── Systèmes énergétiques ───────────────────────────────────
    systems_data = props.get("energy_systems", [])
    if not systems_data:
        # Système par défaut : gaz si avant 2000, PAC sinon
        fuel = "natural_gas" if year_built < 2000 else "electricity"
        sys_type = "gas_boiler" if year_built < 2000 else "heat_pump_air_water"
        systems_data = [{"system_id": "sys_default", "type": sys_type,
                         "fuel": fuel, "covers": ["heating", "dhw"],
                         "efficiency_nominal": 0.87 if year_built < 2000 else 3.2}]
    energy_systems = [system_from_dict(s) for s in systems_data]

    # ─── Occupation ──────────────────────────────────────────────
    occ_data = props.get("occupancy", {})
    from ..core.schedules import USAGE_DEFAULTS
    defaults = USAGE_DEFAULTS.get(usage, USAGE_DEFAULTS["residential"])
    occupancy = OccupancyProfile(
        schedule_name    = occ_data.get("schedule", defaults["schedule"]),
        n_persons        = int(occ_data.get("n_occupants", occ_data.get("n_persons", 0))),
        n_persons_per_m2 = float(occ_data.get("n_persons_per_m2", defaults["n_persons_per_m2"])),
        heat_per_person_w= float(occ_data.get("heat_per_person_w", defaults["heat_per_person_w"])),
        appliances_w_m2  = float(occ_data.get("appliances_w_m2", defaults["appliances_w_m2"])),
        lighting_w_m2    = float(occ_data.get("lighting_w_m2", defaults["lighting_w_m2"])),
    )

    # ─── Consignes ───────────────────────────────────────────────
    sp_data = props.get("setpoints", {})
    setpoints = ThermalSetpoints(
        heating_day_c   = float(sp_data.get("heating_day_c", 19.0)),
        heating_night_c = float(sp_data.get("heating_night_c", 16.0)),
        cooling_c       = float(sp_data.get("cooling_c", 26.0)),
    )

    zone = Zone(
        zone_id            = zone_id,
        label              = label,
        usage              = usage,
        footprint_coords   = footprint,
        height_m           = height_m,
        n_floors           = n_floors,
        year_built         = year_built,
        envelope           = envelope,
        ventilation        = ventilation,
        energy_systems     = energy_systems,
        occupancy          = occupancy,
        setpoints          = setpoints,
        construction_class = construction_class,
        floor_area_m2      = round(area * n_floors, 2),
        heated_volume_m3   = round(area * height_m, 2),
        is_ground_floor    = is_ground,
    )
    return zone


def _parse_envelope(env_data: dict, year_built: int) -> EnvelopeConfig:
    """Parse le dict d'enveloppe en EnvelopeConfig."""
    # ─── Murs ─────────────────────────────────────────────────────
    walls_data = env_data.get("walls", {})
    walls_comp_data = walls_data.get("composition", walls_data)
    if "layers" in walls_comp_data:
        walls = composition_from_dict(walls_comp_data)
    else:
        walls = _default_wall_composition(year_built)

    # ─── Toiture ──────────────────────────────────────────────────
    roof_data = env_data.get("roof", {})
    roof_comp_data = roof_data.get("composition", roof_data)
    if "layers" in roof_comp_data:
        roof = composition_from_dict({**roof_comp_data, "rsi_m2k_w": 0.10, "rse_m2k_w": 0.04})
    else:
        roof = _default_roof_composition(year_built)

    # ─── Plancher bas ─────────────────────────────────────────────
    floor_data = env_data.get("ground_floor", env_data.get("floor", {}))
    floor_comp_data = floor_data.get("composition", floor_data)
    if "layers" in floor_comp_data:
        ground_floor = composition_from_dict({**floor_comp_data, "rsi_m2k_w": 0.17, "rse_m2k_w": 0.04})
    else:
        ground_floor = _default_floor_composition(year_built)

    # ─── Fenêtres ─────────────────────────────────────────────────
    win_data = env_data.get("windows", {})
    wwr_raw  = win_data.get("wwr_by_orientation", win_data.get("wwr", {}))
    if not wwr_raw:
        # WWR par défaut selon usage et orientation
        wwr_raw = _default_wwr(year_built)
    glazing = win_data.get("glazing", {})
    windows = WindowConfig(
        wwr_by_orientation = wwr_raw,
        uw_w_m2k    = float(glazing.get("uw_w_m2k", _default_uw_window(year_built))),
        g_value     = float(glazing.get("g_value", 0.62 if year_built > 1990 else 0.75)),
        frame_factor= float(glazing.get("frame_factor", 0.70)),
        shading_factor= float(win_data.get("shading_factor", 1.0)),
    )

    return EnvelopeConfig(
        walls      = walls,
        roof       = roof,
        ground_floor=ground_floor,
        windows    = windows,
        roof_type  = env_data.get("roof", {}).get("type", "flat"),
        roof_pitch_deg=float(env_data.get("roof", {}).get("pitch_deg", 30.0)),
        thermal_bridge_quality=env_data.get("thermal_bridge_quality", "default"),
    )


def _compute_adjacencies(zones: list[Zone]) -> None:
    """
    Calcule les adjacences entre zones et marque les murs mitoyens.
    Modifie zones in-place en mettant à jour adjacent_zone_ids.
    """
    n = len(zones)
    for i in range(n):
        for j in range(i + 1, n):
            shared = find_shared_edges(
                zones[i].footprint_coords,
                zones[j].footprint_coords,
                tolerance_m=0.20,
            )
            if shared:
                if zones[j].zone_id not in zones[i].adjacent_zone_ids:
                    zones[i].adjacent_zone_ids.append(zones[j].zone_id)
                if zones[i].zone_id not in zones[j].adjacent_zone_ids:
                    zones[j].adjacent_zone_ids.append(zones[i].zone_id)


# ─────────────────────────────────────────────────────────────────────────────
# Compositions par défaut selon l'année de construction
# ─────────────────────────────────────────────────────────────────────────────

def _default_wall_composition(year: int) -> LayeredComposition:
    """Composition de mur typique selon l'époque de construction."""
    if year < 1948:
        return LayeredComposition([
            MaterialLayer("stone_limestone", 0.45),
            MaterialLayer("plaster_coat", 0.02),
        ])
    elif year < 1975:
        return LayeredComposition([
            MaterialLayer("brick_hollow", 0.20),
            MaterialLayer("air_gap", 0.04),
            MaterialLayer("plasterboard", 0.013),
        ])
    elif year < 1990:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("mineral_wool", 0.05),
            MaterialLayer("plasterboard", 0.013),
        ])
    elif year < 2005:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("mineral_wool", 0.08),
            MaterialLayer("plasterboard", 0.013),
        ])
    elif year < 2013:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("mineral_wool", 0.12),
            MaterialLayer("plasterboard", 0.013),
        ])
    else:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("mineral_wool", 0.18),
            MaterialLayer("plasterboard", 0.013),
        ])


def _default_roof_composition(year: int) -> LayeredComposition:
    if year < 1975:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("bitumen_membrane", 0.005),
        ], rsi_m2k_w=0.10, rse_m2k_w=0.04)
    elif year < 1990:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("eps_insulation", 0.05),
            MaterialLayer("bitumen_membrane", 0.005),
        ], rsi_m2k_w=0.10, rse_m2k_w=0.04)
    elif year < 2013:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("eps_insulation", 0.12),
            MaterialLayer("bitumen_membrane", 0.005),
        ], rsi_m2k_w=0.10, rse_m2k_w=0.04)
    else:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("eps_insulation", 0.20),
            MaterialLayer("bitumen_membrane", 0.005),
        ], rsi_m2k_w=0.10, rse_m2k_w=0.04)


def _default_floor_composition(year: int) -> LayeredComposition:
    if year < 1975:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("screed", 0.05),
        ], rsi_m2k_w=0.17, rse_m2k_w=0.04)
    elif year < 2005:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("eps_insulation", 0.04),
            MaterialLayer("screed", 0.05),
        ], rsi_m2k_w=0.17, rse_m2k_w=0.04)
    else:
        return LayeredComposition([
            MaterialLayer("concrete_dense", 0.20),
            MaterialLayer("eps_insulation", 0.10),
            MaterialLayer("screed", 0.05),
        ], rsi_m2k_w=0.17, rse_m2k_w=0.04)


def _default_wwr(year: int) -> dict[str, float]:
    """WWR par défaut selon l'époque de construction."""
    if year < 1975:
        return {"north": 0.10, "south": 0.20, "east": 0.15, "west": 0.15}
    elif year < 2000:
        return {"north": 0.15, "south": 0.25, "east": 0.20, "west": 0.20}
    else:
        return {"north": 0.15, "south": 0.35, "east": 0.20, "west": 0.20}


def _default_uw_window(year: int) -> float:
    """Uw [W/m²K] typique selon l'époque de construction."""
    if year < 1975:
        return 5.5   # Simple vitrage
    elif year < 1990:
        return 3.0   # Double vitrage ancien
    elif year < 2005:
        return 2.8
    elif year < 2013:
        return 1.6
    else:
        return 1.0   # Triple vitrage ou DV performant
