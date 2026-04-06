"""
Parser de fichiers EPW (EnergyPlus Weather Format).

Format documenté : https://energyplus.net/weather/format
Structure :
  Ligne 1  : LOCATION — métadonnées de localisation
  Lignes 2–8 : en-têtes divers (DESIGN CONDITIONS, TYPICAL PERIODS, etc.)
  Lignes 9+  : données horaires (8760 lignes, 35 colonnes)

Colonnes pertinentes (index 0-based) :
  0  : Year
  1  : Month
  2  : Day
  3  : Hour (1–24)
  6  : Dry Bulb Temperature [°C]
  7  : Dew Point Temperature [°C]
  8  : Relative Humidity [%]
  9  : Atmospheric Pressure [Pa]
  13 : Global Horizontal Irradiance [Wh/m²]
  15 : Direct Normal Irradiance [Wh/m²]
  16 : Diffuse Horizontal Irradiance [Wh/m²]
  20 : Wind Direction [°]
  21 : Wind Speed [m/s]
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .epw_models import EPWLocation, WeatherSeries


def parse_epw(filepath: str | Path) -> WeatherSeries:
    """
    Parse un fichier EPW et retourne une WeatherSeries complète.

    Parameters
    ----------
    filepath : str | Path
        Chemin vers le fichier .epw

    Returns
    -------
    WeatherSeries
        Série climatique annuelle (8760 heures)

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le fichier n'a pas le format EPW attendu.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier EPW introuvable : {filepath}")

    with filepath.open(encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 9:
        raise ValueError(f"Fichier EPW trop court ({len(lines)} lignes). Format invalide.")

    location = _parse_location(lines[0])
    data = _parse_hourly_data(lines[8:])

    return WeatherSeries(
        location               = location,
        dry_bulb_temp_c        = data["dry_bulb_temp_c"],
        dew_point_temp_c       = data["dew_point_temp_c"],
        relative_humidity      = data["relative_humidity"],
        atmospheric_pressure_pa= data["atmospheric_pressure_pa"],
        ghi_wh_m2              = data["ghi_wh_m2"],
        dhi_wh_m2              = data["dhi_wh_m2"],
        dni_wh_m2              = data["dni_wh_m2"],
        wind_speed_m_s         = data["wind_speed_m_s"],
        wind_direction_deg     = data["wind_direction_deg"],
    )


def _parse_location(line: str) -> EPWLocation:
    """Parse la première ligne LOCATION du fichier EPW."""
    parts = [p.strip() for p in line.split(",")]
    # LOCATION,City,State,Country,Source,WMO,Lat,Lon,TZ,Elev
    try:
        return EPWLocation(
            city             = parts[1] if len(parts) > 1 else "Unknown",
            state_province   = parts[2] if len(parts) > 2 else "",
            country          = parts[3] if len(parts) > 3 else "",
            source           = parts[4] if len(parts) > 4 else "",
            wmo_station_id   = parts[5] if len(parts) > 5 else "",
            latitude_deg     = float(parts[6]) if len(parts) > 6 else 0.0,
            longitude_deg    = float(parts[7]) if len(parts) > 7 else 0.0,
            timezone_offset  = float(parts[8]) if len(parts) > 8 else 0.0,
            elevation_m      = float(parts[9]) if len(parts) > 9 else 0.0,
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Impossible de parser la ligne LOCATION : {line!r}") from exc


def _parse_hourly_data(data_lines: list[str]) -> dict[str, np.ndarray]:
    """
    Parse les lignes de données horaires du fichier EPW.

    Utilise pandas read_csv sur les données en mémoire pour la performance.
    Retourne un dict de arrays numpy (longueur 8760).
    """
    # Filtrage des lignes non-vides
    valid_lines = [ln for ln in data_lines if ln.strip() and not ln.strip().startswith("LOCATION")]

    if len(valid_lines) < 8760:
        raise ValueError(
            f"Données horaires incomplètes : {len(valid_lines)} lignes trouvées, 8760 attendues."
        )

    # On prend exactement 8760 lignes
    content = "".join(valid_lines[:8760])

    # Colonnes EPW (index 0-based, nommées pour lisibilité)
    col_names = [f"c{i}" for i in range(35)]

    try:
        df = pd.read_csv(
            pd.io.common.StringIO(content),
            header=None,
            names=col_names,
            dtype=object,
            on_bad_lines="skip",
        )
        # Convert numeric columns to float (non-numeric values → NaN)
        for col in col_names:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception as exc:
        raise ValueError(f"Erreur de parsing des données horaires EPW : {exc}") from exc

    def _safe_col(col_idx: int, default: float = 0.0) -> np.ndarray:
        """Retourne la colonne si elle existe, sinon un array de valeurs par défaut."""
        col = f"c{col_idx}"
        if col in df.columns:
            arr = df[col].to_numpy(dtype=float)
            # Remplace les valeurs aberrantes EPW (ex. 9999) par la médiane
            arr = np.where(np.abs(arr) > 9990, np.nan, arr)
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                median_val = np.nanmedian(arr)
                arr[nan_mask] = median_val
            return arr
        return np.full(8760, default)

    return {
        "dry_bulb_temp_c":         _safe_col(6),
        "dew_point_temp_c":        _safe_col(7),
        "relative_humidity":       _safe_col(8),
        "atmospheric_pressure_pa": _safe_col(9, default=101325.0),
        "ghi_wh_m2":               np.maximum(0, _safe_col(13)),
        "dni_wh_m2":               np.maximum(0, _safe_col(15)),
        "dhi_wh_m2":               np.maximum(0, _safe_col(16)),
        "wind_direction_deg":      _safe_col(20),
        "wind_speed_m_s":          np.maximum(0, _safe_col(21)),
    }


def summarize_epw(weather: WeatherSeries) -> dict:
    """
    Retourne un résumé JSON-serialisable du fichier EPW chargé.
    Utile pour validation et affichage.
    """
    loc = weather.location
    return {
        "location": {
            "city":             loc.city,
            "country":          loc.country,
            "latitude_deg":     loc.latitude_deg,
            "longitude_deg":    loc.longitude_deg,
            "timezone_offset":  loc.timezone_offset,
            "elevation_m":      loc.elevation_m,
        },
        "climate_summary": {
            "temp_mean_annual_c":       round(float(np.mean(weather.dry_bulb_temp_c)), 2),
            "temp_min_c":               round(float(np.min(weather.dry_bulb_temp_c)), 1),
            "temp_max_c":               round(float(np.max(weather.dry_bulb_temp_c)), 1),
            "design_winter_temp_c":     round(weather.design_winter_temperature(), 1),
            "design_summer_temp_c":     round(weather.design_summer_temperature(), 1),
            "heating_degree_days_18":   round(weather.heating_degree_days(18.0), 0),
            "cooling_degree_days_26":   round(weather.cooling_degree_days(26.0), 0),
            "ghi_annual_kwh_m2":        round(float(np.sum(weather.ghi_wh_m2)) / 1000, 0),
            "wind_speed_mean_m_s":      round(float(np.mean(weather.wind_speed_m_s)), 2),
        },
        "monthly_mean_temp_c":      [round(v, 1) for v in weather.monthly_mean_temperature()],
        "monthly_ghi_kwh_m2":       [round(v, 1) for v in weather.monthly_total_ghi()],
    }
