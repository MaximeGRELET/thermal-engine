"""
Modèles de calcul des systèmes énergétiques.

Calcule la production et les besoins en énergie finale / primaire
pour chaque type de système : chaudières, PAC, solaire thermique.
"""

from __future__ import annotations
import numpy as np
from ..models.energy_systems import (
    EnergySystem, HeatPump, SolarThermalSystem, VentilationSystem,
    FUEL_PROPERTIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chaudières et systèmes à combustion
# ─────────────────────────────────────────────────────────────────────────────

def boiler_seasonal_efficiency(
    nominal_efficiency: float,
    partial_load_correction: float = 0.92,
) -> float:
    """
    Rendement saisonnier effectif d'une chaudière.

    En fonctionnement réel, le rendement est inférieur au rendement nominal
    (pertes à l'arrêt, régulation, modulation partielle).

    Parameters
    ----------
    nominal_efficiency : float
        Rendement nominal (PCS) de la chaudière
    partial_load_correction : float
        Facteur de correction charge partielle (0.85–0.95)
    """
    return nominal_efficiency * partial_load_correction


def final_energy_demand(
    heating_need_kwh: float,
    system: EnergySystem,
    monthly_temps_c: list[float] | None = None,
) -> float:
    """
    Énergie finale [kWh] nécessaire pour couvrir un besoin de chauffage.

    Pour les PAC, utilise le SCOP calculé dynamiquement si les températures
    mensuelles sont fournies.

    Parameters
    ----------
    heating_need_kwh : float   Besoin de chauffage net [kWh]
    system : EnergySystem      Système énergétique
    monthly_temps_c : list     Températures mensuelles (pour SCOP PAC)

    Returns
    -------
    float : Énergie finale [kWh EF]
    """
    if isinstance(system, HeatPump) and monthly_temps_c is not None:
        scop = system.seasonal_cop(monthly_temps_c)
        eff = scop * (1 - system.distribution_losses)
    else:
        eff = system.effective_efficiency()

    if eff <= 0:
        return 0.0
    return heating_need_kwh / eff


def compute_system_kpis(
    heating_need_kwh: float,
    cooling_need_kwh: float,
    dhw_need_kwh: float,
    systems: list[EnergySystem | SolarThermalSystem],
    monthly_temps_c: list[float] | None = None,
    solar_irradiance_monthly: np.ndarray | None = None,
) -> dict:
    """
    Calcule les KPI énergétiques (énergie finale, primaire, coût, CO₂)
    pour un ensemble de systèmes couvrant les besoins de la zone.

    Logique de répartition :
      - Les besoins sont couverts dans l'ordre des systèmes.
      - Le solaire thermique couvre en priorité l'ECS et le chauffage.
      - Le système principal couvre le restant.

    Returns
    -------
    dict avec clés :
      final_energy_kwh, primary_energy_kwh, cost_eur, co2_kg
      + breakdown par système
    """
    remaining_heating = heating_need_kwh
    remaining_cooling = cooling_need_kwh
    remaining_dhw     = dhw_need_kwh

    total_ef   = 0.0
    total_ep   = 0.0
    total_cost = 0.0
    total_co2  = 0.0
    breakdown  = []

    for sys in systems:
        if isinstance(sys, SolarThermalSystem):
            # Estimation simplifiée de la production solaire thermique annuelle
            if solar_irradiance_monthly is not None:
                annual_irr = float(np.sum(solar_irradiance_monthly)) * 1000  # kWh → Wh… non
                # irradiance_monthly est en kWh/m², on la convertit en énergie brute
                eta_mean = sys.eta_0 * 0.65   # Rendement moyen annuel approximé
                production_kwh = float(np.sum(solar_irradiance_monthly)) * sys.collector_area_m2 * eta_mean
            else:
                # Estimation simplifiée : 350–500 kWh/m² de capteur/an (Europe centrale)
                production_kwh = sys.collector_area_m2 * 400.0

            sol_cover = min(production_kwh, remaining_dhw + (0.3 * remaining_heating))
            if "dhw" in sys.covers:
                dhw_covered = min(sol_cover, remaining_dhw)
                remaining_dhw -= dhw_covered
                sol_cover -= dhw_covered
            if "heating" in sys.covers and sol_cover > 0:
                heat_covered = min(sol_cover, remaining_heating)
                remaining_heating -= heat_covered

            breakdown.append({
                "system_id":    sys.system_id,
                "system_type":  "solar_thermal",
                "label":        sys.label,
                "production_kwh": round(production_kwh, 0),
                "final_energy_kwh": 0,    # L'énergie solaire est "gratuite"
                "primary_energy_kwh": 0,
                "cost_eur":     0,
                "co2_kg":       0,
            })
            continue

        # Systèmes conventionnels
        needs_covered = 0.0
        if "heating" in sys.covers:
            needs_covered += remaining_heating
            remaining_heating = 0
        if "cooling" in sys.covers:
            needs_covered += remaining_cooling
            remaining_cooling = 0
        if "dhw" in sys.covers:
            needs_covered += remaining_dhw
            remaining_dhw = 0

        if needs_covered <= 0:
            continue

        ef   = final_energy_demand(needs_covered, sys, monthly_temps_c)
        fuel = FUEL_PROPERTIES.get(sys.fuel, FUEL_PROPERTIES["natural_gas"])
        ep   = ef * fuel.primary_energy_factor
        cost = ef * fuel.cost_eur_kwh
        co2  = ef * fuel.co2_factor_kg_kwh

        total_ef   += ef
        total_ep   += ep
        total_cost += cost
        total_co2  += co2

        breakdown.append({
            "system_id":          sys.system_id,
            "system_type":        sys.system_type,
            "label":              sys.label,
            "needs_covered_kwh":  round(needs_covered, 0),
            "final_energy_kwh":   round(ef, 0),
            "primary_energy_kwh": round(ep, 0),
            "cost_eur":           round(cost, 0),
            "co2_kg":             round(co2, 0),
        })

    return {
        "final_energy_kwh":   round(total_ef, 0),
        "primary_energy_kwh": round(total_ep, 0),
        "cost_eur":           round(total_cost, 0),
        "co2_kg":             round(total_co2, 0),
        "breakdown_by_system":breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DPE
# ─────────────────────────────────────────────────────────────────────────────

# Seuils DPE énergie primaire (kWh EP/m²/an) — décret 2021
_DPE_SEUILS = [(70, "A"), (110, "B"), (180, "C"), (250, "D"), (330, "E"), (420, "F")]


def dpe_class(primary_energy_kwh_m2: float) -> str:
    """Retourne la lettre DPE (A–G) selon la consommation en énergie primaire."""
    for seuil, lettre in _DPE_SEUILS:
        if primary_energy_kwh_m2 <= seuil:
            return lettre
    return "G"


def dpe_co2_class(co2_kg_m2: float) -> str:
    """
    Classe DPE selon les émissions CO₂ (kg CO₂/m²/an) — décret 2021.
    Le DPE final est le max des deux classes (énergie et CO₂).
    """
    seuils_co2 = [(6, "A"), (11, "B"), (30, "C"), (50, "D"), (70, "E"), (100, "F")]
    for seuil, lettre in seuils_co2:
        if co2_kg_m2 <= seuil:
            return lettre
    return "G"


def dpe_final(primary_energy_kwh_m2: float, co2_kg_m2: float) -> str:
    """
    Classe DPE finale = max(classe énergie, classe CO₂).
    La lettre la moins bonne (la plus éloignée de A) est retenue.
    """
    order = ["A", "B", "C", "D", "E", "F", "G"]
    c_ep  = dpe_class(primary_energy_kwh_m2)
    c_co2 = dpe_co2_class(co2_kg_m2)
    return order[max(order.index(c_ep), order.index(c_co2))]
