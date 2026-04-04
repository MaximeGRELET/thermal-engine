"""
Moteur de scénarios de rénovation.

Les rénovations sont modélisées comme des changements de paramètres
dans le BuildingModel. Chaque action retourne une COPIE du bâtiment
(immutabilité — l'original n'est jamais modifié).

Cela permet d'évaluer plusieurs scénarios en parallèle sur la même
référence.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..models.building import Building, Zone, EnvelopeConfig, WindowConfig
from ..models.materials import LayeredComposition, MaterialLayer
from ..models.energy_systems import (
    EnergySystem, SolarThermalSystem, VentilationSystem,
    HeatPump, GasBoiler, system_from_dict, ventilation_from_dict,
)
from .needs import BuildingNeedsResult, compute_building_needs
from ..climate.epw_models import WeatherSeries


# ─────────────────────────────────────────────────────────────────────────────
# Actions de rénovation (primitives)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenovationAction:
    """
    Action de rénovation abstraite.

    Chaque sous-classe implémente apply() qui retourne un NOUVEAU Building
    avec les paramètres modifiés. L'original est inchangé.
    """
    action_id: str
    label: str
    description: str
    cost_min_eur: float        # Coût minimal [€] hors aides
    cost_max_eur: float        # Coût maximal [€]
    lifetime_years: int = 30
    zone_ids: list[str] = field(default_factory=list)  # [] = s'applique à toutes les zones

    @property
    def cost_center_eur(self) -> float:
        return (self.cost_min_eur + self.cost_max_eur) / 2

    def apply(self, building: Building) -> Building:
        raise NotImplementedError(f"{self.__class__.__name__}.apply() non implémenté.")

    def _target_zones(self, building: Building) -> list[Zone]:
        """Retourne les zones ciblées (toutes si zone_ids est vide)."""
        if not self.zone_ids:
            return building.zones
        return [z for z in building.zones if z.zone_id in self.zone_ids]

    def to_dict(self) -> dict:
        return {
            "action_id":     self.action_id,
            "label":         self.label,
            "description":   self.description,
            "cost_min_eur":  self.cost_min_eur,
            "cost_max_eur":  self.cost_max_eur,
            "cost_center_eur": self.cost_center_eur,
            "lifetime_years":self.lifetime_years,
            "zone_ids":      self.zone_ids,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Isolation des parois
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InsulateWalls(RenovationAction):
    """
    Ajout d'une couche d'isolation sur les murs extérieurs.
    Peut être ITE (Isolation Thermique par l'Extérieur) ou ITI.
    """
    insulation_material_id: str = "mineral_wool"
    insulation_thickness_m: float = 0.14
    position: Literal["exterior", "interior"] = "exterior"

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        for zone in self._target_zones(new_building):
            new_comp = zone.envelope.walls.add_insulation_layer(
                material_id = self.insulation_material_id,
                thickness_m = self.insulation_thickness_m,
                position    = self.position,
            )
            zone.envelope = EnvelopeConfig(
                walls      = new_comp,
                roof       = zone.envelope.roof,
                ground_floor=zone.envelope.ground_floor,
                windows    = zone.envelope.windows,
                roof_type  = zone.envelope.roof_type,
                roof_pitch_deg=zone.envelope.roof_pitch_deg,
                thermal_bridge_quality=self._upgrade_tb_quality(
                    zone.envelope.thermal_bridge_quality, self.position),
            )
        return new_building

    @staticmethod
    def _upgrade_tb_quality(current: str, position: str) -> str:
        """L'ITE améliore naturellement la qualité des ponts thermiques."""
        if position == "exterior" and current == "default":
            return "improved"
        return current


@dataclass
class InsulateRoof(RenovationAction):
    """Isolation de la toiture (combles perdus, sarking, ou toiture terrasse)."""
    insulation_material_id: str = "mineral_wool"
    insulation_thickness_m: float = 0.20

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        for zone in self._target_zones(new_building):
            new_roof = zone.envelope.roof.add_insulation_layer(
                material_id = self.insulation_material_id,
                thickness_m = self.insulation_thickness_m,
                position    = "exterior",   # Sarking ou toiture-terrasse
            )
            zone.envelope = EnvelopeConfig(
                walls       = zone.envelope.walls,
                roof        = new_roof,
                ground_floor= zone.envelope.ground_floor,
                windows     = zone.envelope.windows,
                roof_type   = zone.envelope.roof_type,
                roof_pitch_deg=zone.envelope.roof_pitch_deg,
                thermal_bridge_quality=zone.envelope.thermal_bridge_quality,
            )
        return new_building


@dataclass
class InsulateFloor(RenovationAction):
    """Isolation du plancher bas (sous-dalle ou plancher suspendu)."""
    insulation_material_id: str = "xps_insulation"
    insulation_thickness_m: float = 0.10

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        for zone in self._target_zones(new_building):
            if zone.envelope.ground_floor is None:
                continue
            new_floor = zone.envelope.ground_floor.add_insulation_layer(
                material_id = self.insulation_material_id,
                thickness_m = self.insulation_thickness_m,
                position    = "exterior",
            )
            zone.envelope = EnvelopeConfig(
                walls       = zone.envelope.walls,
                roof        = zone.envelope.roof,
                ground_floor= new_floor,
                windows     = zone.envelope.windows,
                roof_type   = zone.envelope.roof_type,
                roof_pitch_deg=zone.envelope.roof_pitch_deg,
                thermal_bridge_quality=zone.envelope.thermal_bridge_quality,
            )
        return new_building


# ─────────────────────────────────────────────────────────────────────────────
# Remplacement des menuiseries
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplaceWindows(RenovationAction):
    """
    Remplacement des menuiseries.
    Modifie Uw, g-value et le facteur de cadre.
    """
    new_uw_w_m2k: float    = 1.0     # Double vitrage performant ou triple
    new_g_value: float     = 0.62
    new_frame_factor: float= 0.72

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        for zone in self._target_zones(new_building):
            old_w = zone.envelope.windows
            new_windows = WindowConfig(
                wwr_by_orientation = old_w.wwr_by_orientation,
                uw_w_m2k           = self.new_uw_w_m2k,
                g_value            = self.new_g_value,
                frame_factor       = self.new_frame_factor,
                shading_factor     = old_w.shading_factor,
            )
            zone.envelope = EnvelopeConfig(
                walls       = zone.envelope.walls,
                roof        = zone.envelope.roof,
                ground_floor= zone.envelope.ground_floor,
                windows     = new_windows,
                roof_type   = zone.envelope.roof_type,
                roof_pitch_deg=zone.envelope.roof_pitch_deg,
                thermal_bridge_quality=zone.envelope.thermal_bridge_quality,
            )
        return new_building


# ─────────────────────────────────────────────────────────────────────────────
# Systèmes de chauffage et ventilation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplaceHeatingSystem(RenovationAction):
    """Remplacement du système de chauffage principal."""
    new_system_config: dict = field(default_factory=dict)   # dict JSON du nouveau système

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        new_sys = system_from_dict(self.new_system_config)
        for zone in self._target_zones(new_building):
            # Supprime les anciens systèmes de chauffage
            zone.energy_systems = [
                s for s in zone.energy_systems
                if "heating" not in getattr(s, "covers", [])
            ]
            zone.energy_systems.insert(0, new_sys)
        return new_building


@dataclass
class InstallMVHR(RenovationAction):
    """
    Installation d'une VMC double flux avec récupération de chaleur.
    Remplace le système de ventilation existant.
    """
    air_change_rate_h: float = 0.4          # Débit VMC [vol/h]
    heat_recovery_efficiency: float = 0.85  # Rendement récupérateur [0–1]
    specific_power_w_m3h: float = 0.25      # Puissance spécifique [W/(m³/h)]

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        new_vent = VentilationSystem(
            vent_type               = "mec_double_flux",
            air_change_rate_h       = self.air_change_rate_h,
            heat_recovery_efficiency= self.heat_recovery_efficiency,
            specific_power_w_m3h    = self.specific_power_w_m3h,
        )
        for zone in self._target_zones(new_building):
            zone.ventilation = new_vent
        return new_building


@dataclass
class AddSolarThermal(RenovationAction):
    """Installation de capteurs solaires thermiques."""
    collector_area_m2: float = 6.0
    covers: list[str] = field(default_factory=lambda: ["dhw"])
    tilt_deg: float = 35.0
    orientation_deg: float = 180.0

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        solar_sys = system_from_dict({
            "system_id":         f"solar_{self.action_id}",
            "type":              "solar_thermal",
            "collector_area_m2": self.collector_area_m2,
            "covers":            self.covers,
            "tilt_deg":          self.tilt_deg,
            "orientation_deg":   self.orientation_deg,
        })
        for zone in self._target_zones(new_building):
            zone.energy_systems.append(solar_sys)
        return new_building


@dataclass
class ImproveAirtightness(RenovationAction):
    """Amélioration de l'étanchéité à l'air (traitement des fuites)."""
    target_ach: float = 0.3   # Taux de renouvellement résiduel après travaux

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        for zone in self._target_zones(new_building):
            old_vent = zone.ventilation
            new_ach  = min(self.target_ach, old_vent.air_change_rate_h)
            zone.ventilation = VentilationSystem(
                vent_type               = old_vent.vent_type,
                air_change_rate_h       = new_ach,
                heat_recovery_efficiency= old_vent.heat_recovery_efficiency,
                specific_power_w_m3h    = old_vent.specific_power_w_m3h,
            )
        return new_building


@dataclass
class OptimiseThermalBridges(RenovationAction):
    """Traitement et optimisation des ponts thermiques."""
    quality_level: str = "improved"   # "improved" | "optimised"

    def apply(self, building: Building) -> Building:
        new_building = copy.deepcopy(building)
        for zone in self._target_zones(new_building):
            zone.envelope = EnvelopeConfig(
                walls       = zone.envelope.walls,
                roof        = zone.envelope.roof,
                ground_floor= zone.envelope.ground_floor,
                windows     = zone.envelope.windows,
                roof_type   = zone.envelope.roof_type,
                roof_pitch_deg=zone.envelope.roof_pitch_deg,
                thermal_bridge_quality=self.quality_level,
            )
        return new_building


# ─────────────────────────────────────────────────────────────────────────────
# Scénario de rénovation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenovationScenario:
    """
    Un scénario = une liste ordonnée d'actions de rénovation.
    Les actions sont appliquées séquentiellement.
    """
    scenario_id: str
    label: str
    description: str
    actions: list[RenovationAction]

    @property
    def total_cost_min_eur(self) -> float:
        return sum(a.cost_min_eur for a in self.actions)

    @property
    def total_cost_max_eur(self) -> float:
        return sum(a.cost_max_eur for a in self.actions)

    @property
    def total_cost_center_eur(self) -> float:
        return (self.total_cost_min_eur + self.total_cost_max_eur) / 2

    def apply_to_building(self, building: Building) -> Building:
        """
        Applique toutes les actions séquentiellement sur une copie du bâtiment.
        Le bâtiment original n'est JAMAIS modifié.
        """
        result = building
        for action in self.actions:
            result = action.apply(result)
        return result

    def to_dict(self) -> dict:
        return {
            "scenario_id":          self.scenario_id,
            "label":                self.label,
            "description":          self.description,
            "total_cost_min_eur":   round(self.total_cost_min_eur, 0),
            "total_cost_max_eur":   round(self.total_cost_max_eur, 0),
            "actions":              [a.to_dict() for a in self.actions],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Résultats de scénario
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenovationResult:
    """Résultat de la comparaison avant / après rénovation."""
    scenario: RenovationScenario
    baseline: BuildingNeedsResult
    after: BuildingNeedsResult

    @property
    def energy_savings_kwh(self) -> float:
        return self.baseline.final_energy_kwh - self.after.final_energy_kwh

    @property
    def primary_energy_savings_kwh(self) -> float:
        return self.baseline.primary_energy_kwh - self.after.primary_energy_kwh

    @property
    def cost_savings_eur_per_year(self) -> float:
        return self.baseline.cost_eur - self.after.cost_eur

    @property
    def co2_savings_kg_per_year(self) -> float:
        return self.baseline.co2_kg - self.after.co2_kg

    @property
    def simple_payback_years(self) -> float:
        cost_center = self.scenario.total_cost_center_eur
        savings = self.cost_savings_eur_per_year
        if savings <= 0:
            return float("inf")
        return cost_center / savings

    @property
    def heating_need_reduction_pct(self) -> float:
        if self.baseline.heating_need_kwh <= 0:
            return 0.0
        return (1 - self.after.heating_need_kwh / self.baseline.heating_need_kwh) * 100

    @property
    def dpe_improvement(self) -> str:
        return f"{self.baseline.dpe_class} → {self.after.dpe_class}"

    def to_dict(self) -> dict:
        return {
            "scenario":                      self.scenario.to_dict(),
            "baseline_dpe":                  self.baseline.dpe_class,
            "after_dpe":                     self.after.dpe_class,
            "dpe_improvement":               self.dpe_improvement,
            "heating_need_baseline_kwh":     round(self.baseline.heating_need_kwh, 0),
            "heating_need_after_kwh":        round(self.after.heating_need_kwh, 0),
            "heating_need_reduction_pct":    round(self.heating_need_reduction_pct, 1),
            "primary_energy_baseline_kwh_m2":round(self.baseline.primary_energy_kwh_m2, 1),
            "primary_energy_after_kwh_m2":   round(self.after.primary_energy_kwh_m2, 1),
            "energy_savings_kwh":            round(self.energy_savings_kwh, 0),
            "primary_energy_savings_kwh":    round(self.primary_energy_savings_kwh, 0),
            "cost_savings_eur_per_year":     round(self.cost_savings_eur_per_year, 0),
            "co2_savings_kg_per_year":       round(self.co2_savings_kg_per_year, 0),
            "investment_min_eur":            round(self.scenario.total_cost_min_eur, 0),
            "investment_max_eur":            round(self.scenario.total_cost_max_eur, 0),
            "simple_payback_years":          round(self.simple_payback_years, 1),
            "baseline_full":                 self.baseline.to_dict(),
            "after_full":                    self.after.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def simulate_renovation(
    building: Building,
    scenario: RenovationScenario,
    weather: WeatherSeries,
    method: str = "monthly",
    baseline: BuildingNeedsResult | None = None,
) -> RenovationResult:
    """
    Évalue un scénario de rénovation.

    Parameters
    ----------
    building : Building        Bâtiment de référence (non modifié)
    scenario : RenovationScenario
    weather : WeatherSeries
    method : "monthly" | "hourly"
    baseline : BuildingNeedsResult | None
        Résultat de référence pré-calculé (évite un recalcul si plusieurs scénarios)

    Returns
    -------
    RenovationResult
    """
    if baseline is None:
        baseline = compute_building_needs(building, weather, method)

    # Applique les actions → nouveau bâtiment (copie profonde)
    renovated_building = scenario.apply_to_building(building)

    # Calcule les besoins après rénovation
    after_result = compute_building_needs(renovated_building, weather, method)

    return RenovationResult(
        scenario = scenario,
        baseline = baseline,
        after    = after_result,
    )


def simulate_multiple_scenarios(
    building: Building,
    scenarios: list[RenovationScenario],
    weather: WeatherSeries,
    method: str = "monthly",
) -> list[RenovationResult]:
    """
    Évalue plusieurs scénarios en calculant la baseline une seule fois.

    Returns
    -------
    list[RenovationResult]  Dans le même ordre que scenarios
    """
    baseline = compute_building_needs(building, weather, method)
    return [
        simulate_renovation(building, scenario, weather, method, baseline)
        for scenario in scenarios
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Scénarios prédéfinis
# ─────────────────────────────────────────────────────────────────────────────

def build_standard_scenarios(
    building: Building,
    floor_area_m2: float | None = None,
) -> list[RenovationScenario]:
    """
    Génère les 3 scénarios standards (léger / intermédiaire / complet)
    adaptés au bâtiment.

    Parameters
    ----------
    building : Building
    floor_area_m2 : float | None
        Si None, utilise la surface totale du bâtiment.
    """
    area = floor_area_m2 or building.total_floor_area_m2

    return [
        RenovationScenario(
            scenario_id = "light",
            label       = "Rénovation légère",
            description = "Actions rapides et peu onéreuses sur l'enveloppe",
            actions     = [
                InsulateRoof(
                    action_id              = "roof_insulation",
                    label                  = "Isolation combles perdus",
                    description            = "Laine minérale soufflée, R≥7 m²K/W",
                    insulation_material_id = "mineral_wool",
                    insulation_thickness_m = 0.25,
                    cost_min_eur           = area * 20,
                    cost_max_eur           = area * 40,
                    lifetime_years         = 40,
                ),
                ReplaceWindows(
                    action_id       = "windows_dv",
                    label           = "Fenêtres double vitrage performant",
                    description     = "Uw ≤ 1.3 W/m²K",
                    new_uw_w_m2k    = 1.3,
                    new_g_value     = 0.62,
                    cost_min_eur    = area * 60,
                    cost_max_eur    = area * 100,
                    lifetime_years  = 25,
                ),
            ],
        ),

        RenovationScenario(
            scenario_id = "intermediate",
            label       = "Rénovation intermédiaire",
            description = "Enveloppe + VMC double flux",
            actions     = [
                InsulateRoof(
                    action_id="roof_insulation", label="Isolation toiture (sarking/soufflage)",
                    description="R≥7 m²K/W",
                    insulation_material_id="mineral_wool", insulation_thickness_m=0.25,
                    cost_min_eur=area * 25, cost_max_eur=area * 45, lifetime_years=40,
                ),
                InsulateWalls(
                    action_id="wall_ite", label="Isolation murs par l'extérieur (ITE)",
                    description="R≥3.7 m²K/W — polystyrène ou laine de roche",
                    insulation_material_id="eps_insulation", insulation_thickness_m=0.14,
                    position="exterior",
                    cost_min_eur=area * 100, cost_max_eur=area * 160, lifetime_years=30,
                ),
                ReplaceWindows(
                    action_id="windows_hpe", label="Menuiseries haute performance",
                    description="Uw ≤ 1.1 W/m²K",
                    new_uw_w_m2k=1.1, new_g_value=0.62,
                    cost_min_eur=area * 80, cost_max_eur=area * 130, lifetime_years=25,
                ),
                InstallMVHR(
                    action_id="mvhr", label="VMC double flux",
                    description="Récupération de chaleur η≥75%",
                    heat_recovery_efficiency=0.80,
                    cost_min_eur=area * 30, cost_max_eur=area * 50, lifetime_years=20,
                ),
            ],
        ),

        RenovationScenario(
            scenario_id = "bbc_retrofit",
            label       = "Rénovation globale BBC",
            description = "Rénovation complète visant la classe B ou mieux",
            actions     = [
                InsulateRoof(
                    action_id="roof_insulation", label="Isolation toiture R≥9",
                    description="R≥9 m²K/W — laine minérale haute épaisseur",
                    insulation_material_id="mineral_wool_hd", insulation_thickness_m=0.32,
                    cost_min_eur=area * 30, cost_max_eur=area * 55, lifetime_years=40,
                ),
                InsulateWalls(
                    action_id="wall_ite_hpe", label="ITE haute performance",
                    description="R≥4.5 m²K/W",
                    insulation_material_id="mineral_wool_hd", insulation_thickness_m=0.18,
                    position="exterior",
                    cost_min_eur=area * 130, cost_max_eur=area * 200, lifetime_years=30,
                ),
                InsulateFloor(
                    action_id="floor_insulation", label="Isolation plancher bas",
                    description="XPS R≥3 m²K/W",
                    insulation_material_id="xps_insulation", insulation_thickness_m=0.12,
                    cost_min_eur=area * 25, cost_max_eur=area * 45, lifetime_years=40,
                ),
                ReplaceWindows(
                    action_id="windows_tv", label="Triple vitrage",
                    description="Uw ≤ 0.9 W/m²K, g=0.50",
                    new_uw_w_m2k=0.9, new_g_value=0.50, new_frame_factor=0.72,
                    cost_min_eur=area * 120, cost_max_eur=area * 180, lifetime_years=30,
                ),
                InstallMVHR(
                    action_id="mvhr_hpe", label="VMC double flux haute performance",
                    description="η≥85%",
                    heat_recovery_efficiency=0.85, air_change_rate_h=0.35,
                    cost_min_eur=area * 40, cost_max_eur=area * 70, lifetime_years=20,
                ),
                ReplaceHeatingSystem(
                    action_id="pac_installation", label="Pompe à chaleur air/eau",
                    description="SCOP≥3.5, départ 45°C",
                    new_system_config={
                        "system_id": "pac_001", "type": "heat_pump_air_water",
                        "fuel": "electricity", "covers": ["heating", "dhw"],
                        "efficiency_nominal": 3.5, "carnot_fraction": 0.42,
                        "t_sink_design_c": 45.0,
                    },
                    cost_min_eur=area * 70, cost_max_eur=area * 120, lifetime_years=20,
                ),
                OptimiseThermalBridges(
                    action_id="thermal_bridges", label="Traitement des ponts thermiques",
                    description="Ruptures thermiques + continuité de l'isolation",
                    quality_level="optimised",
                    cost_min_eur=area * 10, cost_max_eur=area * 25, lifetime_years=40,
                ),
            ],
        ),
    ]
