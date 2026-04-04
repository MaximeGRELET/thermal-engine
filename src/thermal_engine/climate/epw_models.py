"""
Modèles de données pour les séries climatiques issues des fichiers EPW.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EPWLocation:
    """Métadonnées de localisation du fichier EPW."""
    city: str
    state_province: str
    country: str
    source: str
    wmo_station_id: str
    latitude_deg: float        # °N positif, °S négatif
    longitude_deg: float       # °E positif, °O négatif
    timezone_offset: float     # h par rapport à UTC
    elevation_m: float         # m NGF


@dataclass
class WeatherSeries:
    """
    Série météorologique annuelle complète (8760 heures).

    Toutes les séries temporelles sont des arrays numpy de longueur 8760.
    Convention EPW : heure 1 = 1er janvier 01h00, heure 8760 = 31 décembre 24h00.
    """
    location: EPWLocation

    # ─── Température et humidité ───────────────────────────────────
    dry_bulb_temp_c: np.ndarray    # Température sèche [°C]
    dew_point_temp_c: np.ndarray   # Point de rosée [°C]
    relative_humidity: np.ndarray  # Humidité relative [%]
    atmospheric_pressure_pa: np.ndarray  # Pression [Pa]

    # ─── Rayonnement solaire [Wh/m²] ──────────────────────────────
    ghi_wh_m2: np.ndarray          # Global Horizontal Irradiance
    dhi_wh_m2: np.ndarray          # Diffuse Horizontal Irradiance
    dni_wh_m2: np.ndarray          # Direct Normal Irradiance

    # ─── Vent ─────────────────────────────────────────────────────
    wind_speed_m_s: np.ndarray     # Vitesse [m/s]
    wind_direction_deg: np.ndarray # Direction [°]

    # ─── Index temporel ──────────────────────────────────────────
    timestamps: pd.DatetimeIndex = field(default=None)

    def __post_init__(self):
        if self.timestamps is None:
            # Génère les 8760 timestamps d'une année type (non bissextile)
            self.timestamps = pd.date_range(
                start="2023-01-01 01:00:00",
                periods=8760,
                freq="h",
            )

    # ─── Propriétés dérivées ──────────────────────────────────────

    @property
    def dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame pandas complet (8760 lignes)."""
        return pd.DataFrame({
            "timestamp":            self.timestamps,
            "dry_bulb_temp_c":      self.dry_bulb_temp_c,
            "dew_point_temp_c":     self.dew_point_temp_c,
            "relative_humidity":    self.relative_humidity,
            "atmospheric_pressure_pa": self.atmospheric_pressure_pa,
            "ghi_wh_m2":            self.ghi_wh_m2,
            "dhi_wh_m2":            self.dhi_wh_m2,
            "dni_wh_m2":            self.dni_wh_m2,
            "wind_speed_m_s":       self.wind_speed_m_s,
            "wind_direction_deg":   self.wind_direction_deg,
        })

    def heating_degree_days(self, base_temp_c: float = 18.0) -> float:
        """
        Degrés-jours de chauffage (DJC) sur base de température donnée.
        DJC = sum(max(0, T_base - T_ext_mean_day)) sur tous les jours.
        """
        df = self.dataframe
        df["date"] = self.timestamps.date
        daily_mean = df.groupby("date")["dry_bulb_temp_c"].mean()
        return float(np.maximum(0, base_temp_c - daily_mean).sum())

    def cooling_degree_days(self, base_temp_c: float = 26.0) -> float:
        """Degrés-jours de climatisation."""
        df = self.dataframe
        df["date"] = self.timestamps.date
        daily_mean = df.groupby("date")["dry_bulb_temp_c"].mean()
        return float(np.maximum(0, daily_mean - base_temp_c).sum())

    def monthly_mean_temperature(self) -> list[float]:
        """Températures moyennes mensuelles [°C] — liste de 12 valeurs."""
        df = self.dataframe.copy()
        df["month"] = self.timestamps.month
        return df.groupby("month")["dry_bulb_temp_c"].mean().tolist()

    def monthly_total_ghi(self) -> list[float]:
        """Rayonnement global horizontal mensuel total [kWh/m²] — 12 valeurs."""
        df = self.dataframe.copy()
        df["month"] = self.timestamps.month
        return (df.groupby("month")["ghi_wh_m2"].sum() / 1000).tolist()

    def design_winter_temperature(self, percentile: float = 2.5) -> float:
        """
        Température de dimensionnement hiver (°C).
        Par défaut : percentile 2,5 % des températures horaires annuelles.
        """
        return float(np.percentile(self.dry_bulb_temp_c, percentile))

    def design_summer_temperature(self, percentile: float = 97.5) -> float:
        """Température de dimensionnement été (°C)."""
        return float(np.percentile(self.dry_bulb_temp_c, percentile))
