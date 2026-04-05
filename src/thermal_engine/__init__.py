"""
OptiBuilding Physics — Moteur de calcul énergétique physique des bâtiments.

API publique :
  - load_building(geojson_path)          → Building
  - parse_epw(epw_path)                  → WeatherSeries
  - compute_building_needs(building, weather)  → BuildingNeedsResult
  - simulate_renovation(building, scenario, weather) → RenovationResult
  - simulate_multiple_scenarios(...)     → list[RenovationResult]
  - build_standard_scenarios(building)  → list[RenovationScenario]
  - build_analysis_report(...)          → dict (JSON)
  - build_renovation_report(...)        → dict (JSON)
  - save_report(report, path)           → None
"""

from .io.geojson_loader import load_building
from .climate.epw_parser import parse_epw, summarize_epw
from .simulation.needs import compute_building_needs, CalibrationParams
from .simulation.renovation import (
    simulate_renovation,
    simulate_multiple_scenarios,
    build_standard_scenarios,
    RenovationScenario,
    InsulateWalls,
    InsulateRoof,
    InsulateFloor,
    ReplaceWindows,
    ReplaceHeatingSystem,
    InstallMVHR,
    AddSolarThermal,
    ImproveAirtightness,
    OptimiseThermalBridges,
)
from .io.report_builder import build_analysis_report, build_renovation_report, save_report
from .models.building import Building, Zone
from .models.energy_systems import (
    GasBoiler, HeatPump, WoodBoiler, ElectricHeater,
    DistrictHeatSystem, SolarThermalSystem, VentilationSystem,
)

__version__ = "2.0.0"
__all__ = [
    # I/O
    "load_building", "parse_epw", "summarize_epw",
    # Calcul
    "compute_building_needs", "CalibrationParams",
    # Rénovation
    "simulate_renovation", "simulate_multiple_scenarios", "build_standard_scenarios",
    "RenovationScenario",
    "InsulateWalls", "InsulateRoof", "InsulateFloor", "ReplaceWindows",
    "ReplaceHeatingSystem", "InstallMVHR", "AddSolarThermal",
    "ImproveAirtightness", "OptimiseThermalBridges",
    # Rapports
    "build_analysis_report", "build_renovation_report", "save_report",
    # Modèles
    "Building", "Zone",
    "GasBoiler", "HeatPump", "WoodBoiler", "ElectricHeater",
    "DistrictHeatSystem", "SolarThermalSystem", "VentilationSystem",
]
