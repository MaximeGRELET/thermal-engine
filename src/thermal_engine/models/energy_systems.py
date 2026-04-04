"""
Modèles des systèmes énergétiques des bâtiments.

Couvre : chaudières, PAC, solaire thermique, réseau de chaleur,
         systèmes ECS, refroidissement.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

# Types littéraux
EnergyCarrier = Literal[
    "natural_gas", "lpg", "fuel_oil",
    "electricity", "wood_pellets", "wood_chips", "wood_logs",
    "district_heat", "district_cool", "solar_thermal", "geothermal",
]
SystemCoverage = Literal["heating", "cooling", "dhw"]
DistributionType = Literal[
    "radiators", "floor_heating", "fan_coil", "air_handling",
    "baseboard", "ceiling_panel", "none",
]


@dataclass
class FuelProperties:
    """Propriétés d'un vecteur énergétique."""
    carrier: EnergyCarrier
    primary_energy_factor: float   # Facteur énergie primaire (EP/EF)
    co2_factor_kg_kwh: float       # kg CO₂ par kWh d'énergie finale
    cost_eur_kwh: float            # €/kWh d'énergie finale (tarif 2024)
    label: str


FUEL_PROPERTIES: dict[str, FuelProperties] = {
    "natural_gas": FuelProperties(
        "natural_gas", primary_energy_factor=1.0,
        co2_factor_kg_kwh=0.227, cost_eur_kwh=0.115, label="Gaz naturel",
    ),
    "lpg": FuelProperties(
        "lpg", primary_energy_factor=1.0,
        co2_factor_kg_kwh=0.272, cost_eur_kwh=0.138, label="GPL",
    ),
    "fuel_oil": FuelProperties(
        "fuel_oil", primary_energy_factor=1.0,
        co2_factor_kg_kwh=0.324, cost_eur_kwh=0.145, label="Fioul domestique",
    ),
    "electricity": FuelProperties(
        "electricity", primary_energy_factor=2.3,
        co2_factor_kg_kwh=0.052, cost_eur_kwh=0.2516, label="Électricité",
    ),
    "wood_pellets": FuelProperties(
        "wood_pellets", primary_energy_factor=1.0,
        co2_factor_kg_kwh=0.030, cost_eur_kwh=0.065, label="Granulés de bois",
    ),
    "wood_chips": FuelProperties(
        "wood_chips", primary_energy_factor=1.0,
        co2_factor_kg_kwh=0.030, cost_eur_kwh=0.045, label="Plaquettes bois",
    ),
    "wood_logs": FuelProperties(
        "wood_logs", primary_energy_factor=1.0,
        co2_factor_kg_kwh=0.030, cost_eur_kwh=0.055, label="Bûches",
    ),
    "district_heat": FuelProperties(
        "district_heat", primary_energy_factor=0.77,
        co2_factor_kg_kwh=0.109, cost_eur_kwh=0.090, label="Réseau de chaleur urbain",
    ),
    "district_cool": FuelProperties(
        "district_cool", primary_energy_factor=0.77,
        co2_factor_kg_kwh=0.040, cost_eur_kwh=0.080, label="Réseau de froid urbain",
    ),
}


@dataclass
class EnergySystem:
    """
    Système énergétique générique.

    Pour les PAC, efficacite_nominale représente le COP nominal.
    Pour les chaudières, c'est le rendement (PCS).
    Pour le solaire thermique, c'est le rendement optique η₀.
    """
    system_id: str
    covers: list[SystemCoverage]
    efficiency_nominal: float      # COP ou rendement nominal
    system_type: str = ""          # Défini dans __post_init__ des sous-classes
    fuel: EnergyCarrier = "natural_gas"
    distribution_type: DistributionType = "radiators"
    distribution_losses: float = 0.05   # Pertes de distribution (fraction)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.system_id

    @property
    def fuel_props(self) -> FuelProperties:
        return FUEL_PROPERTIES.get(self.fuel, FUEL_PROPERTIES["electricity"])

    def effective_efficiency(self) -> float:
        """Efficacité nette (pertes de distribution incluses)."""
        return self.efficiency_nominal * (1 - self.distribution_losses)

    def final_energy_for_need(self, need_kwh: float) -> float:
        """Énergie finale [kWh] pour couvrir un besoin [kWh]."""
        eff = self.effective_efficiency()
        if eff <= 0:
            return 0.0
        return need_kwh / eff

    def primary_energy_for_need(self, need_kwh: float) -> float:
        """Énergie primaire [kWh EP] pour couvrir un besoin [kWh]."""
        ef = self.final_energy_for_need(need_kwh)
        return ef * self.fuel_props.primary_energy_factor

    def cost_for_need(self, need_kwh: float) -> float:
        """Coût [€] pour couvrir un besoin [kWh]."""
        ef = self.final_energy_for_need(need_kwh)
        return ef * self.fuel_props.cost_eur_kwh

    def co2_for_need(self, need_kwh: float) -> float:
        """Émissions CO₂ [kg] pour couvrir un besoin [kWh]."""
        ef = self.final_energy_for_need(need_kwh)
        return ef * self.fuel_props.co2_factor_kg_kwh

    def to_dict(self) -> dict:
        return {
            "system_id":         self.system_id,
            "system_type":       self.system_type,
            "label":             self.label,
            "fuel":              self.fuel,
            "covers":            self.covers,
            "efficiency_nominal":self.efficiency_nominal,
            "distribution_type": self.distribution_type,
        }


@dataclass
class GasBoiler(EnergySystem):
    """Chaudière gaz (standard ou à condensation)."""
    condensing: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.system_type = "condensing_boiler" if self.condensing else "gas_boiler"
        if not self.label:
            self.label = "Chaudière gaz condensation" if self.condensing else "Chaudière gaz standard"


@dataclass
class FuelOilBoiler(EnergySystem):
    """Chaudière fioul."""
    def __post_init__(self):
        super().__post_init__()
        self.system_type = "fuel_oil_boiler"
        if not self.label:
            self.label = "Chaudière fioul"


@dataclass
class ElectricHeater(EnergySystem):
    """Chauffage électrique à effet Joule."""
    def __post_init__(self):
        super().__post_init__()
        self.system_type = "electric_resistance"
        if not self.label:
            self.label = "Convecteurs électriques"


@dataclass
class HeatPump(EnergySystem):
    """
    Pompe à chaleur (air/air, air/eau, eau/eau, géothermie).

    Le COP dépend de la température source et de la température de départ
    réseau. On modélise le COP par la fraction du COP de Carnot.
    """
    source: Literal["air", "ground_water", "ground_horizontal"] = "air"
    carnot_fraction: float = 0.40   # Fraction du COP de Carnot (typique : 0.35–0.50)
    t_sink_design_c: float = 45.0   # Température de départ réseau en régime nominal [°C]

    def __post_init__(self):
        super().__post_init__()
        self.system_type = f"heat_pump_{self.source.replace('_', '-')}"
        if not self.label:
            labels = {
                "air":               "Pompe à chaleur air/eau",
                "ground_water":      "Pompe à chaleur eau/eau (nappe)",
                "ground_horizontal": "Pompe à chaleur géothermique (capteurs horizontaux)",
            }
            self.label = labels.get(self.source, "Pompe à chaleur")

    def cop_at_conditions(self, t_source_c: float, t_sink_c: float | None = None) -> float:
        """
        COP instantané selon la fraction de Carnot.

        COP_Carnot = T_sink / (T_sink - T_source)  [températures en K]
        COP_réel   = carnot_fraction × COP_Carnot
        """
        t_sink = t_sink_c if t_sink_c is not None else self.t_sink_design_c
        t_source_k = t_source_c + 273.15
        t_sink_k   = t_sink + 273.15
        if t_sink_k <= t_source_k:
            return self.efficiency_nominal   # COP dégradé si source > sink
        cop_carnot = t_sink_k / (t_sink_k - t_source_k)
        return max(1.0, self.carnot_fraction * cop_carnot)

    def seasonal_cop(self, monthly_temps_c: list[float]) -> float:
        """
        SCOP annuel approximé sur la base des températures mensuelles.
        Pondéré par les degrés-heures de chauffe de chaque mois.
        """
        import numpy as np
        cops = [self.cop_at_conditions(t) for t in monthly_temps_c]
        return float(np.mean(cops))


@dataclass
class WoodBoiler(EnergySystem):
    """Chaudière ou poêle biomasse (bûches, granulés, plaquettes)."""
    has_buffer_tank: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.system_type = "wood_boiler"
        if not self.label:
            fuel_labels = {"wood_pellets": "Chaudière granulés", "wood_logs": "Chaudière bûches"}
            self.label = fuel_labels.get(self.fuel, "Chaudière biomasse")


@dataclass
class DistrictHeatSystem(EnergySystem):
    """Raccordement réseau de chaleur urbain."""
    def __post_init__(self):
        super().__post_init__()
        self.system_type = "district_heating"
        if not self.label:
            self.label = "Réseau de chaleur urbain"


@dataclass
class SolarThermalSystem:
    """
    Système solaire thermique (ECS ou combiné chauffage + ECS).

    Modèle EN ISO 9806 simplifié : rendement en fonction de (T_m - T_amb) / G.
    """
    system_id: str
    collector_area_m2: float
    covers: list[SystemCoverage]    # ["dhw"] ou ["dhw", "heating"]
    eta_0: float = 0.80             # Rendement optique (intersection axe Y de la courbe)
    a1_w_m2k: float = 3.5          # Coefficient de perte linéaire [W/m²K]
    a2_w_m2k2: float = 0.015       # Coefficient de perte quadratique [W/m²K²]
    tilt_deg: float = 35.0         # Inclinaison des capteurs [°]
    orientation_deg: float = 180.0  # Azimut (180 = sud)
    storage_volume_l: float = 0.0  # Volume ballon [L] (0 = calculé auto)
    label: str = "Solaire thermique"
    fuel: EnergyCarrier = "solar_thermal"
    system_type: str = "solar_thermal"

    def efficiency_at_conditions(
        self,
        irradiance_w_m2: float,
        t_mean_collector_c: float,
        t_ambient_c: float,
    ) -> float:
        """
        Rendement instantané du capteur solaire thermique.

        η = η₀ - a₁(T_m - T_amb)/G - a₂(T_m - T_amb)²/G
        """
        if irradiance_w_m2 <= 0:
            return 0.0
        delta_t = t_mean_collector_c - t_ambient_c
        reduced_temp = delta_t / irradiance_w_m2
        eta = (self.eta_0
               - self.a1_w_m2k * reduced_temp
               - self.a2_w_m2k2 * (delta_t ** 2) / irradiance_w_m2)
        return max(0.0, eta)

    def power_output_w(
        self,
        irradiance_w_m2: float,
        t_mean_collector_c: float,
        t_ambient_c: float,
    ) -> float:
        """Puissance utile produite [W]."""
        eta = self.efficiency_at_conditions(irradiance_w_m2, t_mean_collector_c, t_ambient_c)
        return eta * irradiance_w_m2 * self.collector_area_m2

    def to_dict(self) -> dict:
        return {
            "system_id":          self.system_id,
            "system_type":        self.system_type,
            "label":              self.label,
            "collector_area_m2":  self.collector_area_m2,
            "covers":             self.covers,
            "tilt_deg":           self.tilt_deg,
            "orientation_deg":    self.orientation_deg,
            "eta_0":              self.eta_0,
        }


@dataclass
class VentilationSystem:
    """Système de ventilation du bâtiment ou de la zone."""
    vent_type: Literal["natural", "mec_extract", "mec_double_flux"]
    air_change_rate_h: float        # Taux de renouvellement d'air [vol/h]
    heat_recovery_efficiency: float = 0.0   # 0 pour nat/extraction, 0.75–0.90 pour double-flux
    specific_power_w_m3h: float = 0.0       # Puissance spécifique [W/(m³/h)] pour VMC

    @property
    def effective_ach(self) -> float:
        """
        Taux de renouvellement d'air effectif tenant compte de la récupération.
        Pour le calcul des pertes thermiques : ACH_eff = ACH × (1 - η_récup)
        """
        return self.air_change_rate_h * (1.0 - self.heat_recovery_efficiency)

    def to_dict(self) -> dict:
        return {
            "vent_type":                self.vent_type,
            "air_change_rate_h":        self.air_change_rate_h,
            "heat_recovery_efficiency": self.heat_recovery_efficiency,
            "effective_ach":            round(self.effective_ach, 3),
        }


def system_from_dict(data: dict) -> EnergySystem | SolarThermalSystem:
    """
    Construit un système énergétique depuis un dict JSON.

    Le champ "type" détermine la sous-classe instanciée.
    """
    sys_type = data.get("type", data.get("system_type", "gas_boiler"))
    common = {
        "system_id":          data.get("system_id", "sys_001"),
        "fuel":               data.get("fuel", "natural_gas"),
        "covers":             data.get("covers", ["heating"]),
        "efficiency_nominal": float(data.get("efficiency_nominal", data.get("efficiency", 0.87))),
        "distribution_type":  data.get("distribution", data.get("distribution_type", "radiators")),
        "distribution_losses":float(data.get("distribution_losses", 0.05)),
        "label":              data.get("label", ""),
    }

    if sys_type in ("gas_boiler", "condensing_boiler"):
        return GasBoiler(**common, condensing=(sys_type == "condensing_boiler"))
    elif sys_type == "fuel_oil_boiler":
        return FuelOilBoiler(**{**common, "fuel": "fuel_oil"})
    elif sys_type == "electric_resistance":
        return ElectricHeater(**{**common, "fuel": "electricity", "efficiency_nominal": 1.0})
    elif sys_type in ("heat_pump_air_water", "heat_pump_air_air", "heat_pump_air"):
        return HeatPump(**{**common, "fuel": "electricity",
                           "source": "air",
                           "carnot_fraction": float(data.get("carnot_fraction", 0.40)),
                           "t_sink_design_c": float(data.get("t_sink_design_c", 45.0))})
    elif sys_type in ("heat_pump_ground_water", "heat_pump_geothermal"):
        return HeatPump(**{**common, "fuel": "electricity",
                           "source": "ground_water",
                           "carnot_fraction": float(data.get("carnot_fraction", 0.45)),
                           "t_sink_design_c": float(data.get("t_sink_design_c", 35.0))})
    elif sys_type == "wood_boiler":
        fuel = data.get("fuel", "wood_pellets")
        return WoodBoiler(**{**common, "fuel": fuel})
    elif sys_type == "district_heating":
        return DistrictHeatSystem(**{**common, "fuel": "district_heat"})
    elif sys_type == "solar_thermal":
        return SolarThermalSystem(
            system_id       = data.get("system_id", "sol_001"),
            collector_area_m2=float(data.get("collector_area_m2", 4.0)),
            covers          = data.get("covers", ["dhw"]),
            eta_0           = float(data.get("eta_0", 0.80)),
            a1_w_m2k        = float(data.get("a1_w_m2k", 3.5)),
            tilt_deg        = float(data.get("tilt_deg", 35.0)),
            orientation_deg = float(data.get("orientation_deg", 180.0)),
        )
    else:
        # Fallback sur système générique gaz
        return GasBoiler(**common)


def ventilation_from_dict(data: dict) -> VentilationSystem:
    return VentilationSystem(
        vent_type              = data.get("type", "natural"),
        air_change_rate_h      = float(data.get("air_change_rate_h", 0.5)),
        heat_recovery_efficiency=float(data.get("heat_recovery_efficiency", 0.0)),
        specific_power_w_m3h   = float(data.get("specific_power_w_m3h", 0.0)),
    )
