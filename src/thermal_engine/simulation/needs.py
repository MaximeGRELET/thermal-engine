"""
Calcul des besoins énergétiques selon ISO 13790.

Deux méthodes disponibles :
  1. "monthly" — Méthode quasi-stationnaire mensuelle (ISO 13790 §12)
     Rapide, standard pour les DPE et bilans énergétiques réglementaires.
  2. "hourly"  — Simulation horaire simplifiée (modèle RC à 1 nœud)
     Plus précis, utile pour l'analyse des pics et le confort.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..models.building import Building, Zone
from ..models.energy_systems import HeatPump, SolarThermalSystem
from ..climate.epw_models import WeatherSeries
from ..core.thermal import (
    transmission_heat_loss_coefficient,
    transmission_losses_monthly,
    thermal_time_constant_h,
    effective_heat_capacity,
    HOURS_PER_MONTH,
)
from ..core.ventilation import (
    ventilation_heat_loss_coefficient,
    ventilation_losses_monthly,
)
from ..core.solar import (
    prepare_irradiance_series,
    solar_gains_monthly,
)
from ..core.schedules import (
    internal_gains_monthly,
    USAGE_DEFAULTS,
)
from ..core.systems import compute_system_kpis, dpe_final
from ..core.geometry import (
    compute_zone_geometry,
    find_shared_edges,
    wwr_for_azimuth,
    orientation_label,
)
from ..data.thermal_bridges_db import get_psi


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres de calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationParams:
    """
    Overrides appliqués sur une zone au moment de la simulation.
    Toute valeur à None est ignorée (le modèle utilise sa valeur calculée).

    Usage :
        params = {
            "zone_rdc": CalibrationParams(u_walls=1.5, infiltration_ach=0.8),
        }
        result = compute_building_needs(building, weather, calibration=params)
    """
    # Enveloppe — U-values (W/m²K)
    u_walls:   float | None = None
    u_roof:    float | None = None
    u_floor:   float | None = None
    u_windows: float | None = None

    # Vitrages — taux moyen tous azimuts (fraction 0-1)
    wwr_override: float | None = None

    # Ventilation / infiltration (vol/h)
    infiltration_ach:  float | None = None   # remplace zone.ventilation.infiltration_ach
    ventilation_ach:   float | None = None   # remplace zone.ventilation.mechanical_ach

    # Consignes de température (°C)
    t_heating: float | None = None
    t_cooling: float | None = None

    # Apports internes (W/m²) — override global
    internal_gains_w_m2: float | None = None

    # Altitude (m) — affecte la densité de l'air → H_V
    altitude_m: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Résultats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoneNeedsResult:
    """Résultats de calcul pour une zone."""
    zone_id: str
    zone_label: str
    floor_area_m2: float

    # Besoins nets (demande de chaleur / froid après utilisation des apports)
    heating_need_kwh: float
    cooling_need_kwh: float
    dhw_need_kwh: float

    # Déperditions brutes
    transmission_losses_kwh: float
    ventilation_losses_kwh: float
    total_losses_kwh: float

    # Apports
    solar_gains_kwh: float
    internal_gains_kwh: float
    total_gains_kwh: float

    # Détail mensuel
    heating_need_monthly: list[float]   # kWh/mois × 12
    cooling_need_monthly: list[float]
    transmission_monthly: list[float]
    ventilation_monthly: list[float]
    solar_gains_monthly: list[float]
    internal_gains_monthly: list[float]

    # Bilan système
    final_energy_kwh: float
    primary_energy_kwh: float
    cost_eur: float
    co2_kg: float
    dpe_class: str

    # Détail des éléments d'enveloppe
    envelope_breakdown: dict = field(default_factory=dict)

    # Données horaires (uniquement renseignées avec method="hourly")
    t_int_hourly: list[float] | None = None        # °C × 8760
    q_heat_hourly_kw: list[float] | None = None    # kW × 8760
    comfort_hours_above_26c: int | None = None      # heures/an > 26°C
    coldest_week_start_h: int | None = None         # indice horaire de début

    def to_dict(self) -> dict:
        d = {
            "zone_id":                self.zone_id,
            "zone_label":             self.zone_label,
            "floor_area_m2":          round(self.floor_area_m2, 1),
            "heating_need_kwh":       round(self.heating_need_kwh, 0),
            "cooling_need_kwh":       round(self.cooling_need_kwh, 0),
            "dhw_need_kwh":           round(self.dhw_need_kwh, 0),
            "transmission_losses_kwh":round(self.transmission_losses_kwh, 0),
            "ventilation_losses_kwh": round(self.ventilation_losses_kwh, 0),
            "solar_gains_kwh":        round(self.solar_gains_kwh, 0),
            "internal_gains_kwh":     round(self.internal_gains_kwh, 0),
            "final_energy_kwh":       round(self.final_energy_kwh, 0),
            "primary_energy_kwh":     round(self.primary_energy_kwh, 0),
            "primary_energy_kwh_m2":  round(self.primary_energy_kwh / max(1, self.floor_area_m2), 1),
            "cost_eur":               round(self.cost_eur, 0),
            "co2_kg":                 round(self.co2_kg, 0),
            "dpe_class":              self.dpe_class,
            "heating_need_monthly":   [round(v, 0) for v in self.heating_need_monthly],
            "solar_gains_monthly":    [round(v, 0) for v in self.solar_gains_monthly],
            "envelope_breakdown":     self.envelope_breakdown,
        }
        if self.t_int_hourly is not None:
            d["hourly"] = {
                "t_int":                [round(v, 1) for v in self.t_int_hourly],
                "q_heat_kw":            [round(v, 2) for v in self.q_heat_hourly_kw],
                "comfort_h_above_26c":  self.comfort_hours_above_26c,
                "coldest_week_start_h": self.coldest_week_start_h,
            }
        return d


@dataclass
class BuildingNeedsResult:
    """Résultats agrégés pour le bâtiment complet."""
    building_id: str
    building_name: str
    total_floor_area_m2: float
    zone_results: list[ZoneNeedsResult]

    # Totaux bâtiment
    heating_need_kwh: float
    cooling_need_kwh: float
    dhw_need_kwh: float
    final_energy_kwh: float
    primary_energy_kwh: float
    cost_eur: float
    co2_kg: float
    dpe_class: str
    primary_energy_kwh_m2: float
    co2_kg_m2: float

    method: str = "monthly"

    def to_dict(self) -> dict:
        return {
            "building_id":           self.building_id,
            "building_name":         self.building_name,
            "method":                self.method,
            "total_floor_area_m2":   round(self.total_floor_area_m2, 1),
            "heating_need_kwh":      round(self.heating_need_kwh, 0),
            "cooling_need_kwh":      round(self.cooling_need_kwh, 0),
            "dhw_need_kwh":          round(self.dhw_need_kwh, 0),
            "final_energy_kwh":      round(self.final_energy_kwh, 0),
            "primary_energy_kwh":    round(self.primary_energy_kwh, 0),
            "primary_energy_kwh_m2": round(self.primary_energy_kwh_m2, 1),
            "cost_eur":              round(self.cost_eur, 0),
            "co2_kg":                round(self.co2_kg, 0),
            "co2_kg_m2":             round(self.co2_kg_m2, 2),
            "dpe_class":             self.dpe_class,
            "zones":                 [z.to_dict() for z in self.zone_results],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Calcul principal
# ─────────────────────────────────────────────────────────────────────────────

def compute_building_needs(
    building: Building,
    weather: WeatherSeries,
    method: str = "monthly",
    t_setpoint_heating: float | None = None,
    t_setpoint_cooling: float | None = None,
    calibration: dict[str, "CalibrationParams"] | None = None,
) -> BuildingNeedsResult:
    """
    Calcule les besoins énergétiques du bâtiment complet.

    Parameters
    ----------
    building : Building
    weather : WeatherSeries
    method : "monthly" | "hourly"
    t_setpoint_heating : float | None
        Override global de la consigne de chauffage.
    t_setpoint_cooling : float | None
        Override global de la consigne de refroidissement.
    calibration : dict[zone_id → CalibrationParams] | None
        Overrides fins par zone (U-values, WWR, ACH, consignes, apports internes…).
        Les clés sont les zone_id. Les zones sans entrée utilisent leurs valeurs par défaut.

    Returns
    -------
    BuildingNeedsResult
    """
    calibration = calibration or {}
    zone_results = []
    for zone in building.zones:
        cal = calibration.get(zone.zone_id) or calibration.get("*")  # "*" = toutes zones
        t_heat = (cal.t_heating if cal and cal.t_heating is not None
                  else t_setpoint_heating or zone.setpoints.heating_day_c)
        t_cool = (cal.t_cooling if cal and cal.t_cooling is not None
                  else t_setpoint_cooling or zone.setpoints.cooling_c)
        if method == "hourly":
            result = _compute_zone_needs_hourly(zone, weather, t_heat, t_cool, cal)
        else:
            result = _compute_zone_needs_monthly(zone, weather, t_heat, t_cool, cal)
        zone_results.append(result)

    # Agrégation bâtiment
    total_area = sum(z.floor_area_m2 for z in building.zones)
    total_heat = sum(z.heating_need_kwh for z in zone_results)
    total_cool = sum(z.cooling_need_kwh for z in zone_results)
    total_dhw  = sum(z.dhw_need_kwh for z in zone_results)
    total_ef   = sum(z.final_energy_kwh for z in zone_results)
    total_ep   = sum(z.primary_energy_kwh for z in zone_results)
    total_cost = sum(z.cost_eur for z in zone_results)
    total_co2  = sum(z.co2_kg for z in zone_results)

    ep_m2  = total_ep / max(1, total_area)
    co2_m2 = total_co2 / max(1, total_area)
    dpe    = dpe_final(ep_m2, co2_m2)

    return BuildingNeedsResult(
        building_id          = building.building_id,
        building_name        = building.name,
        total_floor_area_m2  = total_area,
        zone_results         = zone_results,
        heating_need_kwh     = round(total_heat, 0),
        cooling_need_kwh     = round(total_cool, 0),
        dhw_need_kwh         = round(total_dhw, 0),
        final_energy_kwh     = round(total_ef, 0),
        primary_energy_kwh   = round(total_ep, 0),
        cost_eur             = round(total_cost, 0),
        co2_kg               = round(total_co2, 0),
        dpe_class            = dpe,
        primary_energy_kwh_m2= round(ep_m2, 1),
        co2_kg_m2            = round(co2_m2, 2),
        method               = method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Méthode mensuelle ISO 13790
# ─────────────────────────────────────────────────────────────────────────────

def _compute_zone_needs_monthly(
    zone: Zone,
    weather: WeatherSeries,
    t_heating: float,
    t_cooling: float,
    cal: "CalibrationParams | None" = None,
) -> ZoneNeedsResult:
    """Calcul mensuel ISO 13790 pour une zone."""
    monthly_temps = weather.monthly_mean_temperature()
    floor_height  = zone.floor_height_m
    area          = zone.floor_area_m2
    volume        = zone.heated_volume_m3

    # ─── 1. Coefficient H_T (transmission) ───────────────────────
    geo = compute_zone_geometry(
        zone_id           = zone.zone_id,
        footprint_coords  = zone.footprint_coords,
        height_m          = zone.height_m,
        n_floors          = zone.n_floors,
        roof_type         = zone.envelope.roof_type,
        roof_pitch_deg    = zone.envelope.roof_pitch_deg,
        is_ground_floor   = zone.is_ground_floor,
    )

    elements, windows_config = _build_envelope_elements(zone, geo, floor_height, cal)
    h_t = transmission_heat_loss_coefficient(elements)

    # ─── 2. Coefficient H_V (ventilation) ────────────────────────
    eff_ach = _resolve_ach(zone, cal)
    altitude_m = cal.altitude_m if cal and cal.altitude_m is not None else 0.0
    h_v = ventilation_heat_loss_coefficient(volume, eff_ach, altitude_m=altitude_m)

    # ─── 3. Pertes mensuelles ─────────────────────────────────────
    q_tr  = transmission_losses_monthly(h_t, t_heating, monthly_temps)
    q_ve  = ventilation_losses_monthly(volume, eff_ach, t_heating, monthly_temps,
                                       altitude_m=altitude_m)
    q_loss = q_tr + q_ve

    # ─── 4. Apports solaires ─────────────────────────────────────
    irr_series = prepare_irradiance_series(weather, windows_config)
    q_sol_monthly = solar_gains_monthly(irr_series, windows_config, weather.timestamps)

    # ─── 5. Apports internes ─────────────────────────────────────
    if cal and cal.internal_gains_w_m2 is not None:
        # Override global en W/m² → kWh/mois
        q_int_monthly = np.array([
            cal.internal_gains_w_m2 * area * h / 1000
            for h in HOURS_PER_MONTH
        ])
    else:
        n_persons = zone.occupancy.effective_n_persons(area)
        q_int_monthly = internal_gains_monthly(
            floor_area_m2    = area,
            n_persons        = n_persons,
            schedule_name    = zone.occupancy.schedule_name,
            heat_per_person_w= zone.occupancy.heat_per_person_w,
            appliances_w_m2  = zone.occupancy.appliances_w_m2,
            lighting_w_m2    = zone.occupancy.lighting_w_m2,
        )

    # ─── 6. Besoin de chauffage (méthode gain-utilisation ISO 13790) ─
    tau = thermal_time_constant_h(
        effective_heat_capacity(area, zone.construction_class),
        h_t, h_v,
    )
    q_heat_monthly, q_cool_monthly = _iso13790_monthly_needs(
        q_loss, q_sol_monthly + q_int_monthly, tau, t_heating, t_cooling,
    )

    # ─── 7. ECS ──────────────────────────────────────────────────
    dhw_need = zone.dhw_need_kwh_per_year()

    # ─── 8. Bilan système ─────────────────────────────────────────
    solar_irr_monthly = np.array(weather.monthly_total_ghi())   # kWh/m²
    sys_kpis = compute_system_kpis(
        heating_need_kwh        = float(np.sum(q_heat_monthly)),
        cooling_need_kwh        = float(np.sum(q_cool_monthly)),
        dhw_need_kwh            = dhw_need,
        systems                 = zone.energy_systems,
        monthly_temps_c         = monthly_temps,
        solar_irradiance_monthly= solar_irr_monthly,
    )

    # ─── 9. DPE zone ─────────────────────────────────────────────
    ep_m2  = sys_kpis["primary_energy_kwh"] / max(1, area)
    co2_m2 = sys_kpis["co2_kg"] / max(1, area)
    dpe    = dpe_final(ep_m2, co2_m2)

    # ─── Répartition des déperditions par élément ─────────────────
    breakdown = _envelope_breakdown(elements, q_tr, q_ve)

    return ZoneNeedsResult(
        zone_id                  = zone.zone_id,
        zone_label               = zone.label,
        floor_area_m2            = area,
        heating_need_kwh         = round(float(np.sum(q_heat_monthly)), 0),
        cooling_need_kwh         = round(float(np.sum(q_cool_monthly)), 0),
        dhw_need_kwh             = round(dhw_need, 0),
        transmission_losses_kwh  = round(float(np.sum(q_tr)), 0),
        ventilation_losses_kwh   = round(float(np.sum(q_ve)), 0),
        total_losses_kwh         = round(float(np.sum(q_loss)), 0),
        solar_gains_kwh          = round(float(np.sum(q_sol_monthly)), 0),
        internal_gains_kwh       = round(float(np.sum(q_int_monthly)), 0),
        total_gains_kwh          = round(float(np.sum(q_sol_monthly + q_int_monthly)), 0),
        heating_need_monthly     = [round(float(v), 0) for v in q_heat_monthly],
        cooling_need_monthly     = [round(float(v), 0) for v in q_cool_monthly],
        transmission_monthly     = [round(float(v), 0) for v in q_tr],
        ventilation_monthly      = [round(float(v), 0) for v in q_ve],
        solar_gains_monthly      = [round(float(v), 0) for v in q_sol_monthly],
        internal_gains_monthly   = [round(float(v), 0) for v in q_int_monthly],
        final_energy_kwh         = sys_kpis["final_energy_kwh"],
        primary_energy_kwh       = sys_kpis["primary_energy_kwh"],
        cost_eur                 = sys_kpis["cost_eur"],
        co2_kg                   = sys_kpis["co2_kg"],
        dpe_class                = dpe,
        envelope_breakdown       = breakdown,
    )


def _iso13790_monthly_needs(
    q_loss: np.ndarray,   # kWh/mois — pertes
    q_gains: np.ndarray,  # kWh/mois — apports totaux
    tau: float,           # constante de temps [h]
    t_heating: float,
    t_cooling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Facteur d'utilisation des apports (ISO 13790 §12.2.1.1).

    Pour le chauffage :
      γ_H = Q_gains / Q_loss
      a_H = a_H0 + τ/τ_H0   (a_H0 = 1, τ_H0 = 15 h)
      η_H = (1 - γ_H^a_H) / (1 - γ_H^(a_H+1))  si γ ≠ 1
      η_H = a_H / (a_H + 1)                       si γ = 1
      Q_H = max(0, Q_loss - η_H × Q_gains)

    Pour le refroidissement :
      Calcul symétrique (η_C utilise γ_C = 1/γ_H)
    """
    TAU_H0 = 15.0   # h (ISO 13790 Tableau 12)
    A_H0   = 1.0

    a_h = A_H0 + tau / TAU_H0

    # ── Chauffage ──
    q_loss_safe = np.where(q_loss > 0, q_loss, 1.0)
    gamma_h = np.where(q_loss > 0, q_gains / q_loss_safe, 0.0)
    eta_h   = np.where(
        np.abs(gamma_h - 1.0) < 1e-6,
        a_h / (a_h + 1),
        (1 - gamma_h ** a_h) / (1 - gamma_h ** (a_h + 1)),
    )
    eta_h = np.clip(eta_h, 0.0, 1.0)
    q_heat = np.maximum(0, q_loss - eta_h * q_gains)

    # ── Refroidissement ──
    a_c   = a_h   # Symétrique (simplification)
    gamma_c = np.where(q_gains > 0, q_loss / q_gains, 0.0)
    eta_c   = np.where(
        np.abs(gamma_c - 1.0) < 1e-6,
        a_c / (a_c + 1),
        (1 - gamma_c ** a_c) / (1 - gamma_c ** (a_c + 1)),
    )
    eta_c = np.clip(eta_c, 0.0, 1.0)
    # En été (q_loss ~ 0), les apports créent un besoin de refroidissement
    q_cool = np.maximum(0, q_gains - eta_c * q_loss)

    return q_heat, q_cool


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Helpers calibration
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ach(zone: Zone, cal: "CalibrationParams | None") -> float:
    """Retourne le taux de renouvellement d'air effectif (infiltration + mécanique)."""
    if cal is None:
        return zone.ventilation.effective_ach
    infil = cal.infiltration_ach if cal.infiltration_ach is not None else zone.ventilation.infiltration_ach
    mech  = cal.ventilation_ach  if cal.ventilation_ach  is not None else zone.ventilation.mechanical_ach
    return infil + mech


# ─────────────────────────────────────────────────────────────────────────────
# Construction des éléments d'enveloppe depuis la géométrie
# ─────────────────────────────────────────────────────────────────────────────

def _build_envelope_elements(
    zone: Zone,
    geo,   # ZoneGeometry
    floor_height: float,
    cal: "CalibrationParams | None" = None,
) -> tuple[list[dict], list[dict]]:
    """
    Construit la liste des éléments thermiques (pour H_T) et la config des fenêtres.

    Returns
    -------
    elements : list[dict]    Éléments pour transmission_heat_loss_coefficient
    windows_config : list[dict]  Configuration pour solar_gains_through_window
    """
    env      = zone.envelope
    # U-values : calibration override > valeur calculée depuis couches
    u_wall   = cal.u_walls   if cal and cal.u_walls   is not None else env.walls.u_value_w_m2k
    u_roof   = cal.u_roof    if cal and cal.u_roof    is not None else env.roof.u_value_w_m2k
    u_floor  = cal.u_floor   if cal and cal.u_floor   is not None else getattr(env.ground_floor, "u_value_w_m2k", 0.5)
    u_win    = cal.u_windows if cal and cal.u_windows is not None else env.windows.uw_w_m2k
    tb_qual  = env.thermal_bridge_quality
    # WWR global override (fraction 0-1), appliqué uniformément à tous les azimuts
    wwr_global = cal.wwr_override if cal and cal.wwr_override is not None else None

    elements       = []
    windows_config = []

    # ─── Murs et fenêtres (par segment) ──────────────────────────
    for seg in geo.wall_segments:
        if not seg.is_exterior:
            continue   # Mur mitoyen → b = 0

        for floor_idx in range(zone.n_floors):
            wall_area_gross = seg.length_m * floor_height

            # WWR pour cette orientation (override global si fourni)
            if wwr_global is not None:
                wwr = wwr_global
            else:
                orient_simple = _simplify_orientation(seg.azimuth_deg)
                wwr = env.windows.wwr(orient_simple)
            win_area  = wall_area_gross * wwr
            wall_area_net = wall_area_gross - win_area

            # Élément mur (surface nette)
            elements.append({
                "id":       f"{seg.segment_id}_f{floor_idx}",
                "type":     "surface",
                "category": "wall",
                "u_value":  u_wall,
                "area_m2":  wall_area_net,
                "b_factor": 1.0,
            })

            # Élément fenêtre
            if win_area > 0:
                win_id = f"win_{seg.segment_id}_f{floor_idx}"
                elements.append({
                    "id":       win_id,
                    "type":     "surface",
                    "category": "window",
                    "u_value":  u_win,
                    "area_m2":  win_area,
                    "b_factor": 1.0,
                })
                windows_config.append({
                    "window_id":    win_id,
                    "area_m2":      win_area,
                    "azimuth_deg":  seg.azimuth_deg,
                    "tilt_deg":     90.0,    # Fenêtres verticales
                    "g_value":      env.windows.g_value,
                    "frame_factor": env.windows.frame_factor,
                    "shading_factor": env.windows.shading_factor,
                })

        # Ponts thermiques linéiques autour des fenêtres (révéal)
        if wwr > 0:
            try:
                psi_win = get_psi("window_reveal", tb_qual)
                # Périmètre estimé d'une fenêtre typique (2×(l+h), approx)
                win_perimeter = 2 * (1.2 + 1.5) * wwr * zone.n_floors
                elements.append({
                    "id":       f"{seg.segment_id}_tb_win",
                    "type":     "linear_bridge",
                    "psi":      psi_win,
                    "length_m": seg.length_m * win_perimeter / seg.length_m * seg.length_m,
                    "b_factor": 1.0,
                })
            except KeyError:
                pass

    # ─── Toiture ─────────────────────────────────────────────────
    elements.append({
        "id":       "roof",
        "type":     "surface",
        "category": "roof",
        "u_value":  u_roof,
        "area_m2":  geo.roof_area_m2,
        "b_factor": 1.0,
    })
    # Pont thermique mur-toiture
    try:
        psi_roof = get_psi("wall_roof", tb_qual)
        elements.append({
            "id":       "tb_wall_roof",
            "type":     "linear_bridge",
            "psi":      psi_roof,
            "length_m": geo.perimeter_m,
            "b_factor": 1.0,
        })
    except KeyError:
        pass

    # ─── Plancher bas ────────────────────────────────────────────
    if zone.is_ground_floor and env.ground_floor is not None:
        if cal and cal.u_floor is not None:
            u_floor_eff = cal.u_floor
        else:
            from ..core.thermal import u_value_ground_floor_iso13370
            u_floor_eff = u_value_ground_floor_iso13370(
                floor_area_m2 = geo.ground_floor_area_m2,
                perimeter_m   = geo.perimeter_m,
                wall_u_value  = u_wall,
                floor_layers  = [(l.thickness_m, l.lambda_w_mk) for l in env.ground_floor.layers],
            )
        elements.append({
            "id":       "ground_floor",
            "type":     "surface",
            "category": "floor",
            "u_value":  u_floor_eff,
            "area_m2":  geo.ground_floor_area_m2,
            "b_factor": 1.0,
        })
        # Pont thermique mur-plancher RDC
        try:
            psi_floor = get_psi("wall_floor_ground", tb_qual)
            elements.append({
                "id":       "tb_wall_floor_ground",
                "type":     "linear_bridge",
                "psi":      psi_floor,
                "length_m": geo.perimeter_m,
                "b_factor": 1.0,
            })
        except KeyError:
            pass

    return elements, windows_config


def _simplify_orientation(azimuth_deg: float) -> str:
    """Retourne "north"/"south"/"east"/"west" depuis un azimut."""
    if azimuth_deg < 45 or azimuth_deg >= 315:
        return "north"
    elif 45 <= azimuth_deg < 135:
        return "east"
    elif 135 <= azimuth_deg < 225:
        return "south"
    else:
        return "west"


def _envelope_breakdown(
    elements: list[dict],
    q_tr_monthly: np.ndarray,
    q_ve_monthly: np.ndarray,
) -> dict:
    """Calcule la répartition des déperditions par poste."""
    def sum_by_cat(cat):
        return sum(
            el["area_m2"] * el["u_value"]
            for el in elements
            if el.get("category") == cat and el.get("type") == "surface"
        )

    total_ht_surfaces = sum(
        el["area_m2"] * el["u_value"]
        for el in elements
        if el.get("type") == "surface"
    )
    total_ht_bridges = sum(
        el["psi"] * el["length_m"]
        for el in elements
        if el.get("type") == "linear_bridge"
    )
    total_ht = total_ht_surfaces + total_ht_bridges or 1

    q_tr_total = float(np.sum(q_tr_monthly))
    q_ve_total = float(np.sum(q_ve_monthly))
    q_total    = q_tr_total + q_ve_total or 1

    return {
        "walls_kwh":           round(q_tr_total * sum_by_cat("wall") / total_ht, 0),
        "windows_kwh":         round(q_tr_total * sum_by_cat("window") / total_ht, 0),
        "roof_kwh":            round(q_tr_total * sum_by_cat("roof") / total_ht, 0),
        "floor_kwh":           round(q_tr_total * sum_by_cat("floor") / total_ht, 0),
        "thermal_bridges_kwh": round(q_tr_total * total_ht_bridges / total_ht, 0),
        "ventilation_kwh":     round(q_ve_total, 0),
        "walls_pct":           round(100 * q_tr_total * sum_by_cat("wall") / (total_ht * q_total), 1),
        "windows_pct":         round(100 * q_tr_total * sum_by_cat("window") / (total_ht * q_total), 1),
        "roof_pct":            round(100 * q_tr_total * sum_by_cat("roof") / (total_ht * q_total), 1),
        "floor_pct":           round(100 * q_tr_total * sum_by_cat("floor") / (total_ht * q_total), 1),
        "thermal_bridges_pct": round(100 * q_tr_total * total_ht_bridges / (total_ht * q_total), 1),
        "ventilation_pct":     round(100 * q_ve_total / q_total, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Méthode horaire (modèle RC simplifié)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_zone_needs_hourly(
    zone: Zone,
    weather: WeatherSeries,
    t_heating: float,
    t_cooling: float,
    cal: "CalibrationParams | None" = None,
) -> ZoneNeedsResult:
    """
    Simulation horaire RC à 1 nœud.

    dT_int/dt = (Q_heating - Q_losses + Q_gains) / C_th

    Q_losses = H_T × (T_int - T_ext) + H_V × (T_int - T_ext)
    Q_gains  = Q_solar + Q_internal

    Schéma d'Euler implicite pour la stabilité.
    """
    from ..core.solar import prepare_irradiance_series, solar_gains_through_window
    from ..core.schedules import get_occupancy_schedule, internal_gains_hourly

    area   = zone.floor_area_m2
    volume = zone.heated_volume_m3
    geo    = compute_zone_geometry(
        zone_id          = zone.zone_id,
        footprint_coords = zone.footprint_coords,
        height_m         = zone.height_m,
        n_floors         = zone.n_floors,
        roof_type        = zone.envelope.roof_type,
        is_ground_floor  = zone.is_ground_floor,
    )
    elements, windows_config = _build_envelope_elements(zone, geo, zone.floor_height_m, cal)
    h_t = transmission_heat_loss_coefficient(elements)
    altitude_m = cal.altitude_m if cal and cal.altitude_m is not None else 0.0
    eff_ach    = _resolve_ach(zone, cal)
    h_v = ventilation_heat_loss_coefficient(volume, eff_ach, altitude_m=altitude_m)
    h_total = h_t + h_v   # W/K

    # Capacité thermique [J/K]
    c_th = effective_heat_capacity(area, zone.construction_class) * 1000   # kJ → J

    # Apports solaires horaires (W)
    irr_series = prepare_irradiance_series(weather, windows_config)
    q_solar_w = np.zeros(8760)
    for win in windows_config:
        wid = win["window_id"]
        q_solar_w += solar_gains_through_window(
            irr_series.get(wid, np.zeros(8760)),
            area_m2       = win["area_m2"],
            g_value       = win["g_value"],
            frame_factor  = win["frame_factor"],
            shading_factor= win["shading_factor"],
        )

    # Apports internes horaires (W)
    if cal and cal.internal_gains_w_m2 is not None:
        q_int_w = np.full(8760, cal.internal_gains_w_m2 * area)
    else:
        occ = get_occupancy_schedule(zone.occupancy.schedule_name)
        n_pers = zone.occupancy.effective_n_persons(area)
        q_int_w = internal_gains_hourly(
            floor_area_m2       = area,
            n_persons           = n_pers,
            occupation_schedule = occ,
            heat_per_person_w   = zone.occupancy.heat_per_person_w,
            appliances_w_m2     = zone.occupancy.appliances_w_m2,
            lighting_w_m2       = zone.occupancy.lighting_w_m2,
        )
    q_gains_w = q_solar_w + q_int_w

    # Simulation Euler implicite
    t_ext  = weather.dry_bulb_temp_c
    t_int  = np.full(8760, t_heating)
    q_heat = np.zeros(8760)
    q_cool = np.zeros(8760)
    dt     = 3600.0   # 1 heure en secondes

    t_prev = t_heating
    for i in range(8760):
        # Bilan sans système : Euler implicite
        # C × (T_n - T_{n-1}) / dt = Q_gains - H × (T_n - T_ext)
        # T_n = (C × T_{n-1} / dt + Q_gains + H × T_ext) / (C/dt + H)
        t_no_system = ((c_th * t_prev / dt)
                       + q_gains_w[i]
                       + h_total * t_ext[i]) / (c_th / dt + h_total)

        if t_no_system < t_heating:
            t_int[i] = t_heating
            # Chauffage nécessaire = différence entre ce que le bâtiment atteint et la consigne
            q_heat[i] = h_total * (t_heating - t_ext[i]) - q_gains_w[i]
            q_heat[i] = max(0, q_heat[i])
        elif t_no_system > t_cooling:
            t_int[i] = t_cooling
            q_cool[i] = q_gains_w[i] - h_total * (t_cooling - t_ext[i])
            q_cool[i] = max(0, q_cool[i])
        else:
            t_int[i] = t_no_system

        t_prev = t_int[i]

    # Conversion en mensuels
    ts = weather.timestamps
    s_heat = pd.Series(q_heat, index=ts)
    s_cool = pd.Series(q_cool, index=ts)
    s_sol  = pd.Series(q_solar_w, index=ts)
    s_int  = pd.Series(q_int_w, index=ts)

    q_heat_m = s_heat.groupby(s_heat.index.month).sum().values / 1000  # kWh
    q_cool_m = s_cool.groupby(s_cool.index.month).sum().values / 1000
    q_sol_m  = s_sol.groupby(s_sol.index.month).sum().values / 1000
    q_int_m  = s_int.groupby(s_int.index.month).sum().values / 1000

    # Pertes mensuelles (rétrocalcul)
    q_tr_m = transmission_losses_monthly(h_t, t_heating, weather.monthly_mean_temperature())
    q_ve_m = ventilation_losses_monthly(volume, eff_ach,
                                         t_heating, weather.monthly_mean_temperature(),
                                         altitude_m=altitude_m)

    dhw_need = zone.dhw_need_kwh_per_year()
    monthly_temps = weather.monthly_mean_temperature()
    solar_irr_monthly = np.array(weather.monthly_total_ghi())
    sys_kpis = compute_system_kpis(
        heating_need_kwh        = float(np.sum(q_heat_m)),
        cooling_need_kwh        = float(np.sum(q_cool_m)),
        dhw_need_kwh            = dhw_need,
        systems                 = zone.energy_systems,
        monthly_temps_c         = monthly_temps,
        solar_irradiance_monthly= solar_irr_monthly,
    )

    ep_m2  = sys_kpis["primary_energy_kwh"] / max(1, area)
    co2_m2 = sys_kpis["co2_kg"] / max(1, area)
    breakdown = _envelope_breakdown(elements, q_tr_m, q_ve_m)

    # ── Champs horaires ────────────────────────────────────────────────────────
    comfort_h = int(np.sum(t_int > 26.0))

    # Semaine la plus froide : fenêtre glissante de 168h maximisant le besoin
    q_heat_rolling = np.convolve(q_heat, np.ones(168), mode='valid')
    coldest_start  = int(np.argmax(q_heat_rolling))

    return ZoneNeedsResult(
        zone_id                = zone.zone_id,
        zone_label             = zone.label,
        floor_area_m2          = area,
        heating_need_kwh       = round(float(np.sum(q_heat_m)), 0),
        cooling_need_kwh       = round(float(np.sum(q_cool_m)), 0),
        dhw_need_kwh           = round(dhw_need, 0),
        transmission_losses_kwh= round(float(np.sum(q_tr_m)), 0),
        ventilation_losses_kwh = round(float(np.sum(q_ve_m)), 0),
        total_losses_kwh       = round(float(np.sum(q_tr_m + q_ve_m)), 0),
        solar_gains_kwh        = round(float(np.sum(q_sol_m)), 0),
        internal_gains_kwh     = round(float(np.sum(q_int_m)), 0),
        total_gains_kwh        = round(float(np.sum(q_sol_m + q_int_m)), 0),
        heating_need_monthly   = [round(float(v), 0) for v in q_heat_m],
        cooling_need_monthly   = [round(float(v), 0) for v in q_cool_m],
        transmission_monthly   = [round(float(v), 0) for v in q_tr_m],
        ventilation_monthly    = [round(float(v), 0) for v in q_ve_m],
        solar_gains_monthly    = [round(float(v), 0) for v in q_sol_m],
        internal_gains_monthly = [round(float(v), 0) for v in q_int_m],
        final_energy_kwh       = sys_kpis["final_energy_kwh"],
        primary_energy_kwh     = sys_kpis["primary_energy_kwh"],
        cost_eur               = sys_kpis["cost_eur"],
        co2_kg                 = sys_kpis["co2_kg"],
        dpe_class              = dpe_final(ep_m2, co2_m2),
        envelope_breakdown     = breakdown,
        t_int_hourly           = [round(float(v), 1) for v in t_int],
        q_heat_hourly_kw       = [round(float(v) / 1000, 2) for v in q_heat],
        comfort_hours_above_26c= comfort_h,
        coldest_week_start_h   = coldest_start,
    )
