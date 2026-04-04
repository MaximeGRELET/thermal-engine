"""
Construction et sérialisation des rapports énergétiques en JSON.
"""

from __future__ import annotations
import json
from datetime import datetime
from ..simulation.needs import BuildingNeedsResult
from ..simulation.renovation import RenovationResult
from ..climate.epw_parser import summarize_epw
from ..climate.epw_models import WeatherSeries


def build_analysis_report(
    needs: BuildingNeedsResult,
    weather: WeatherSeries | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Construit un rapport JSON complet d'analyse énergétique.

    Parameters
    ----------
    needs : BuildingNeedsResult
    weather : WeatherSeries | None
    metadata : dict | None         Métadonnées libres (date, analyste, etc.)

    Returns
    -------
    dict   Rapport JSON-serialisable
    """
    report = {
        "report_type":    "energy_analysis",
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "method":         needs.method,
        "metadata":       metadata or {},
        "building":       {
            "building_id":         needs.building_id,
            "name":                needs.building_name,
            "total_floor_area_m2": needs.total_floor_area_m2,
        },
        "climate":        summarize_epw(weather) if weather else {},
        "results":        needs.to_dict(),
        "summary": {
            "dpe_class":             needs.dpe_class,
            "primary_energy_kwh_m2": needs.primary_energy_kwh_m2,
            "co2_kg_m2":             needs.co2_kg_m2,
            "annual_cost_eur":       needs.cost_eur,
            "heating_need_kwh":      needs.heating_need_kwh,
        },
    }
    return report


def build_renovation_report(
    results: list[RenovationResult],
    weather: WeatherSeries | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Construit un rapport de comparaison de scénarios de rénovation.

    Parameters
    ----------
    results : list[RenovationResult]
    weather : WeatherSeries | None
    metadata : dict | None
    """
    if not results:
        return {"error": "Aucun scénario fourni."}

    baseline = results[0].baseline
    scenarios_list = []
    for res in results:
        scenarios_list.append({
            "scenario_id":                  res.scenario.scenario_id,
            "label":                        res.scenario.label,
            "dpe_before":                   res.baseline.dpe_class,
            "dpe_after":                    res.after.dpe_class,
            "heating_need_reduction_pct":   round(res.heating_need_reduction_pct, 1),
            "primary_energy_before_kwh_m2": round(res.baseline.primary_energy_kwh_m2, 1),
            "primary_energy_after_kwh_m2":  round(res.after.primary_energy_kwh_m2, 1),
            "cost_savings_eur_per_year":    round(res.cost_savings_eur_per_year, 0),
            "co2_savings_kg_per_year":      round(res.co2_savings_kg_per_year, 0),
            "investment_min_eur":           round(res.scenario.total_cost_min_eur, 0),
            "investment_max_eur":           round(res.scenario.total_cost_max_eur, 0),
            "simple_payback_years":         round(res.simple_payback_years, 1),
            "actions":                      [a.to_dict() for a in res.scenario.actions],
            "full_result":                  res.to_dict(),
        })

    return {
        "report_type":  "renovation_comparison",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata":     metadata or {},
        "climate":      summarize_epw(weather) if weather else {},
        "building": {
            "building_id":         baseline.building_id,
            "name":                baseline.building_name,
            "total_floor_area_m2": baseline.total_floor_area_m2,
            "dpe_baseline":        baseline.dpe_class,
            "primary_energy_baseline_kwh_m2": baseline.primary_energy_kwh_m2,
            "annual_cost_baseline_eur":        baseline.cost_eur,
        },
        "scenarios": scenarios_list,
    }


def save_report(report: dict, filepath: str, indent: int = 2) -> None:
    """Sauvegarde un rapport JSON dans un fichier."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=indent)


def load_report(filepath: str) -> dict:
    """Charge un rapport JSON depuis un fichier."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)
