"""
Modèles de données du bâtiment multi-zones.

Un Building contient une ou plusieurs zones thermiques (Zone).
Chaque zone est décrite par son emprise 2D + hauteur + propriétés thermiques.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from .materials import LayeredComposition
from .energy_systems import EnergySystem, SolarThermalSystem, VentilationSystem


Usage = Literal[
    "residential", "office", "retail", "restaurant",
    "school", "warehouse", "industrial", "other",
]

ConstructionClass = Literal["very_light", "light", "medium", "heavy", "very_heavy"]


@dataclass
class WindowConfig:
    """
    Configuration des vitages d'une zone.

    Les fenêtres sont définies par WWR (Window-to-Wall Ratio) par orientation
    et des propriétés de vitrage communes.
    """
    wwr_by_orientation: dict[str, float]   # {"north": 0.15, "south": 0.35, ...}
    uw_w_m2k: float = 2.8                  # Coefficient de transmission [W/m²K]
    g_value: float  = 0.62                 # Facteur solaire (SHGC)
    frame_factor: float = 0.70             # Fraction vitrée (1 - fraction cadre)
    shading_factor: float = 1.0            # Facteur d'ombrage global

    def wwr(self, orientation: str) -> float:
        """Retourne le WWR pour une orientation donnée."""
        return float(self.wwr_by_orientation.get(orientation.lower(), 0.20))


@dataclass
class EnvelopeConfig:
    """Configuration de l'enveloppe d'une zone."""
    walls: LayeredComposition
    roof: LayeredComposition
    ground_floor: LayeredComposition | None
    windows: WindowConfig
    roof_type: str = "flat"                 # "flat" | "pitched" | "shed"
    roof_pitch_deg: float = 30.0
    # Qualité de traitement des ponts thermiques
    thermal_bridge_quality: str = "default"  # "default" | "improved" | "optimised"


@dataclass
class OccupancyProfile:
    """Profil d'occupation d'une zone."""
    schedule_name: str = "residential_standard"
    n_persons: int = 0
    n_persons_per_m2: float = 0.035   # Utilisé si n_persons == 0
    heat_per_person_w: float = 80.0
    appliances_w_m2: float = 4.0
    lighting_w_m2: float = 2.0

    def effective_n_persons(self, floor_area_m2: float) -> int:
        """Nombre de personnes effectif (valeur fixe ou calculée depuis densité)."""
        if self.n_persons > 0:
            return self.n_persons
        return max(1, int(self.n_persons_per_m2 * floor_area_m2))


@dataclass
class ThermalSetpoints:
    """Températures de consigne."""
    heating_day_c: float   = 19.0
    heating_night_c: float = 16.0
    cooling_c: float       = 26.0


@dataclass
class Zone:
    """
    Zone thermique d'un bâtiment.

    Décrite par son emprise 2D + hauteur.
    Les surfaces (murs, toiture, plancher) sont calculées géométriquement.
    """
    zone_id: str
    label: str
    usage: Usage
    footprint_coords: list[list[float]]   # Coordonnées 2D du polygone d'emprise
    height_m: float                        # Hauteur totale de la zone [m]
    n_floors: int                          # Nombre de niveaux
    year_built: int
    envelope: EnvelopeConfig
    ventilation: VentilationSystem
    energy_systems: list[EnergySystem | SolarThermalSystem]
    occupancy: OccupancyProfile
    setpoints: ThermalSetpoints = field(default_factory=ThermalSetpoints)
    construction_class: ConstructionClass = "medium"
    # Champs calculés (remplis par le loader ou le moteur)
    floor_area_m2: float = 0.0
    heated_volume_m3: float = 0.0
    is_ground_floor: bool = True   # Cette zone est-elle au contact du sol ?
    adjacent_zone_ids: list[str] = field(default_factory=list)

    @property
    def floor_height_m(self) -> float:
        """Hauteur d'un niveau [m]."""
        return self.height_m / max(1, self.n_floors)

    def dhw_need_kwh_per_year(self) -> float:
        """
        Besoin ECS annuel estimé [kWh/an].
        Formule RE 2020 : 20 kWh/m²/an (résidentiel) ou selon l'usage.
        """
        defaults = {
            "residential": 20.0,
            "office":       5.0,
            "restaurant":  50.0,
            "school":       5.0,
            "retail":       2.0,
        }
        kwh_m2 = defaults.get(self.usage, 10.0)
        return kwh_m2 * max(self.floor_area_m2, 1.0)


@dataclass
class Building:
    """
    Bâtiment complet composé de plusieurs zones thermiques.

    Les zones peuvent avoir des emprises différentes (ex. RDC commercial + étages résidentiels).
    """
    building_id: str
    name: str
    zones: list[Zone]
    location: tuple[float, float]   # (latitude_deg, longitude_deg)
    city: str = ""

    @property
    def total_floor_area_m2(self) -> float:
        return sum(z.floor_area_m2 for z in self.zones)

    @property
    def total_volume_m3(self) -> float:
        return sum(z.heated_volume_m3 for z in self.zones)

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @property
    def year_built(self) -> int:
        """Année de construction la plus ancienne parmi les zones."""
        return min(z.year_built for z in self.zones) if self.zones else 0

    def get_zone(self, zone_id: str) -> Zone | None:
        """Retourne une zone par son identifiant."""
        for z in self.zones:
            if z.zone_id == zone_id:
                return z
        return None

    def to_summary_dict(self) -> dict:
        """Résumé JSON-serialisable du bâtiment."""
        return {
            "building_id":   self.building_id,
            "name":          self.name,
            "city":          self.city,
            "location":      {"latitude": self.location[0], "longitude": self.location[1]},
            "n_zones":       self.n_zones,
            "total_floor_area_m2": round(self.total_floor_area_m2, 1),
            "total_volume_m3":     round(self.total_volume_m3, 1),
            "year_built":          self.year_built,
            "zones": [
                {
                    "zone_id":      z.zone_id,
                    "label":        z.label,
                    "usage":        z.usage,
                    "floor_area_m2":round(z.floor_area_m2, 1),
                    "height_m":     z.height_m,
                    "n_floors":     z.n_floors,
                    "year_built":   z.year_built,
                    "u_wall":       round(z.envelope.walls.u_value_w_m2k, 3),
                    "u_roof":       round(z.envelope.roof.u_value_w_m2k, 3),
                    "u_window":     z.envelope.windows.uw_w_m2k,
                }
                for z in self.zones
            ],
        }
