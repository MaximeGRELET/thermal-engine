"""
Calculs solaires : position du soleil, irradiance sur surfaces inclinées, apports passifs.

Méthodes :
  - Position solaire : formules de Spencer (1971) + équation du temps
  - Irradiance sur surface : modèle de Hay-Davies (1980)
    (meilleur compromis précision/complexité vs Perez)
  - Apports solaires passifs : EN ISO 13790 section 11.3

Toutes les fonctions sont pures (sans état, sans I/O).
Entrées/sorties en grandeurs scalaires ou arrays numpy pour vectorisation sur 8760 heures.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

SOLAR_CONSTANT = 1367.0   # W/m² (irradiance extraterrestre)


# ─────────────────────────────────────────────────────────────────────────────
# Position du soleil
# ─────────────────────────────────────────────────────────────────────────────

def day_of_year(month: int | np.ndarray, day: int | np.ndarray) -> int | np.ndarray:
    """Jour de l'année julien (1–365)."""
    days_in_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cum = np.cumsum(days_in_month)
    m = np.asarray(month)
    d = np.asarray(day)
    return cum[m - 1] + d


def solar_declination_rad(doy: int | np.ndarray) -> float | np.ndarray:
    """
    Déclinaison solaire δ [rad] — formule de Spencer (1971).

    B = (doy - 1) × 360/365 [rad]
    δ = 0.006918 - 0.399912·cos(B) + 0.070257·sin(B)
      - 0.006758·cos(2B) + 0.000907·sin(2B)
      - 0.002697·cos(3B) + 0.00148·sin(3B)
    """
    b = 2 * np.pi * (np.asarray(doy) - 1) / 365
    dec = (0.006918
           - 0.399912 * np.cos(b)
           + 0.070257 * np.sin(b)
           - 0.006758 * np.cos(2 * b)
           + 0.000907 * np.sin(2 * b)
           - 0.002697 * np.cos(3 * b)
           + 0.001480 * np.sin(3 * b))
    return dec


def equation_of_time_min(doy: int | np.ndarray) -> float | np.ndarray:
    """
    Équation du temps E [minutes] — formule de Spencer (1971).

    Correction entre le temps solaire vrai et le temps civil.
    """
    b = 2 * np.pi * (np.asarray(doy) - 1) / 365
    e_min = 229.18 * (0.000075
                      + 0.001868 * np.cos(b)
                      - 0.032077 * np.sin(b)
                      - 0.014615 * np.cos(2 * b)
                      - 0.04089  * np.sin(2 * b))
    return e_min


def solar_hour_angle_rad(
    hour_local: float | np.ndarray,
    longitude_deg: float,
    timezone_offset: float,
    doy: int | np.ndarray,
) -> float | np.ndarray:
    """
    Angle horaire solaire ω [rad].

    ω = 0 à midi solaire vrai ; négatif le matin, positif l'après-midi.

    Parameters
    ----------
    hour_local : float | array   Heure locale (1–24 convention EPW)
    longitude_deg : float        Longitude [°], E positif
    timezone_offset : float      Décalage UTC [h] (ex. +1 pour Paris hiver)
    doy : int | array            Jour de l'année

    Returns
    -------
    ω en radians
    """
    # Méridien de référence du fuseau horaire
    lon_std = timezone_offset * 15.0   # °
    eot = equation_of_time_min(doy)    # minutes
    # Heure solaire vraie [h]
    h_solar = (np.asarray(hour_local) - 0.5          # milieu de l'heure EPW
               + (longitude_deg - lon_std) / 15.0    # correction longitude
               + eot / 60.0)                          # équation du temps
    omega = np.radians(15.0 * (h_solar - 12.0))
    return omega


def solar_altitude_rad(
    latitude_rad: float,
    declination_rad: float | np.ndarray,
    hour_angle_rad: float | np.ndarray,
) -> float | np.ndarray:
    """
    Hauteur solaire β [rad] au-dessus de l'horizon.

    sin(β) = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(ω)
    """
    sin_beta = (np.sin(latitude_rad) * np.sin(declination_rad)
                + np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))
    return np.arcsin(np.clip(sin_beta, -1.0, 1.0))


def solar_azimuth_rad(
    latitude_rad: float,
    declination_rad: float | np.ndarray,
    hour_angle_rad: float | np.ndarray,
    altitude_rad: float | np.ndarray,
) -> float | np.ndarray:
    """
    Azimut solaire γ_s [rad], mesuré depuis le Sud (0=S, π/2=O, -π/2=E).
    Convention : positif vers l'Ouest.

    cos(γ_s) = (sin(β)·sin(φ) - sin(δ)) / (cos(β)·cos(φ))
    """
    cos_beta = np.cos(altitude_rad)
    denom = cos_beta * np.cos(latitude_rad)
    cos_gamma = np.where(
        np.abs(denom) > 1e-6,
        (np.sin(altitude_rad) * np.sin(latitude_rad) - np.sin(declination_rad)) / denom,
        0.0,
    )
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    # Signe : négatif le matin (ω < 0)
    return np.where(hour_angle_rad < 0, -gamma, gamma)


def compute_solar_position(
    latitude_deg: float,
    longitude_deg: float,
    timezone_offset: float,
    doy_array: np.ndarray,    # shape (8760,)
    hour_array: np.ndarray,   # shape (8760,) — heures EPW 1–24
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la position solaire pour une série de 8760 heures.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (altitude_rad, azimuth_rad_from_south) — shape (8760,)
        altitude ≥ 0 uniquement pour heures diurnes
    """
    lat_rad = np.radians(latitude_deg)
    dec     = solar_declination_rad(doy_array)
    omega   = solar_hour_angle_rad(hour_array, longitude_deg, timezone_offset, doy_array)
    alt     = solar_altitude_rad(lat_rad, dec, omega)
    azm     = solar_azimuth_rad(lat_rad, dec, omega, alt)
    return alt, azm


# ─────────────────────────────────────────────────────────────────────────────
# Irradiance sur surface inclinée — modèle Hay-Davies
# ─────────────────────────────────────────────────────────────────────────────

def extraterrestrial_irradiance(doy: int | np.ndarray) -> float | np.ndarray:
    """Irradiance extraterrestre I₀ [W/m²] (correction distance Soleil-Terre)."""
    b = 2 * np.pi * np.asarray(doy) / 365
    return SOLAR_CONSTANT * (1.00011
                             + 0.034221 * np.cos(b)
                             + 0.001280 * np.sin(b)
                             + 0.000719 * np.cos(2 * b)
                             + 0.000077 * np.sin(2 * b))


def incidence_angle_cos(
    tilt_rad: float,
    surface_azimuth_rad: float,   # Azimut surface depuis Sud (même convention que soleil)
    solar_altitude_rad_arr: np.ndarray,
    solar_azimuth_rad_arr: np.ndarray,
) -> np.ndarray:
    """
    Cosinus de l'angle d'incidence θ entre le rayon solaire et la normale à la surface.

    cos(θ) = sin(β)·cos(Σ) + cos(β)·sin(Σ)·cos(γ_s - γ)

    Où Σ = inclinaison surface, γ = azimut surface, β = altitude solaire, γ_s = azimut solaire.
    """
    beta = solar_altitude_rad_arr
    gamma_s = solar_azimuth_rad_arr
    sigma = tilt_rad
    gamma = surface_azimuth_rad

    cos_theta = (np.sin(beta) * np.cos(sigma)
                 + np.cos(beta) * np.sin(sigma) * np.cos(gamma_s - gamma))
    return np.clip(cos_theta, 0.0, 1.0)


def irradiance_on_tilted_surface_hay_davies(
    ghi: np.ndarray,          # W/m² Global Horizontal
    dhi: np.ndarray,          # W/m² Diffuse Horizontal
    dni: np.ndarray,          # W/m² Direct Normal
    tilt_deg: float,          # Inclinaison surface [°] (0=horizontal, 90=vertical)
    surface_azimuth_deg: float, # Azimut surface [°] depuis Nord (0=N, 90=E, 180=S, 270=O)
    solar_altitude_rad: np.ndarray,
    solar_azimuth_rad: np.ndarray,   # Azimut solaire depuis Sud
    doy_array: np.ndarray,
) -> np.ndarray:
    """
    Irradiance sur une surface inclinée [W/m²] — modèle Hay-Davies (1980).

    Décompose en 3 composantes :
      I_beam  = DNI × cos(θ)    [rayonnement direct]
      I_diff  = DHI × (Ai + (1-Ai) × (1+cos(Σ))/2)   [diffus isotrope + circumsolaire]
      I_refl  = GHI × ρ_sol × (1-cos(Σ))/2   [réfléchi par le sol, ρ=0.2]

    Où Ai = DNI / I₀ = anisotropy index (facteur Hay-Davies)

    Parameters
    ----------
    surface_azimuth_deg : float
        Convention : 0=Nord, 90=Est, 180=Sud, 270=Ouest
    solar_azimuth_rad : np.ndarray
        Convention : 0=Sud, π/2=Ouest, -π/2=Est (convention solaire européenne)
    """
    sigma_rad = np.radians(tilt_deg)
    # Conversion azimut surface : de convention Nord vers convention Sud
    gamma_surface_rad = np.radians(surface_azimuth_deg - 180.0)

    # Angle d'incidence
    cos_theta = incidence_angle_cos(
        sigma_rad, gamma_surface_rad,
        solar_altitude_rad, solar_azimuth_rad,
    )

    # Masque nuit (altitude solaire ≤ 0)
    daylight = solar_altitude_rad > 0

    # Indice d'anisotropie Hay-Davies
    i0 = extraterrestrial_irradiance(doy_array)
    cos_zenith = np.sin(solar_altitude_rad)
    cos_zenith_safe = np.where(cos_zenith > 0.001, cos_zenith, 0.001)
    a_i = np.where(daylight, np.clip(dni / (i0 * cos_zenith_safe), 0.0, 1.0), 0.0)

    # Facteur de vue ciel : (1 + cos Σ) / 2
    sky_view_factor = (1 + np.cos(sigma_rad)) / 2

    # Composante directe
    i_beam = dni * cos_theta

    # Composante diffuse (Hay-Davies)
    i_diff = dhi * (a_i * cos_theta / np.where(cos_zenith_safe > 0, cos_zenith_safe, 1)
                    + (1 - a_i) * sky_view_factor)

    # Composante réfléchie par le sol (albédo = 0.20)
    rho_ground = 0.20
    i_refl = ghi * rho_ground * (1 - np.cos(sigma_rad)) / 2

    irr = np.where(daylight, np.maximum(0, i_beam + i_diff + i_refl), 0.0)
    return irr


# ─────────────────────────────────────────────────────────────────────────────
# Apports solaires passifs (fenêtres)
# ─────────────────────────────────────────────────────────────────────────────

def solar_gains_through_window(
    irradiance_w_m2: np.ndarray,   # W/m²
    area_m2: float,                # Surface de la fenêtre (baie)
    g_value: float,                # Facteur solaire du vitrage (SHGC)
    frame_factor: float = 0.70,    # Fraction vitrée (1 - fraction cadre)
    shading_factor: float = 1.0,   # Facteur d'ombrage (masques, stores)
) -> np.ndarray:
    """
    Apports solaires passifs à travers une fenêtre [W].

    Q_sol = I_surface × A_w × g × F_F × F_sh

    Où A_eff = A_w × F_F × F_sh est la surface solaire effective.

    Parameters
    ----------
    g_value : float       Facteur solaire total du vitrage (0.3–0.8 selon double/triple)
    frame_factor : float  Fraction transparente (0.6–0.85)
    shading_factor : float Facteur d'ombrage global (0.5–1.0)

    Returns
    -------
    np.ndarray : Apports [W], shape (8760,)
    """
    a_eff = area_m2 * frame_factor * shading_factor
    return irradiance_w_m2 * a_eff * g_value


def solar_gains_monthly(
    irradiance_series: dict[str, np.ndarray],   # {window_id: array (8760,) W/m²}
    windows_config: list[dict],                  # [{area_m2, g_value, frame_factor, shading_factor}]
    timestamps: "pd.DatetimeIndex | None" = None,
) -> np.ndarray:
    """
    Apports solaires mensuels totaux [kWh/mois] — somme sur toutes les fenêtres.

    Parameters
    ----------
    irradiance_series : dict
        Irradiance horaire sur chaque orientation de fenêtre (8760 valeurs en W/m²).
        Les clés doivent correspondre aux window_id dans windows_config.
    windows_config : list[dict]
        Configuration de chaque fenêtre.
    timestamps : pd.DatetimeIndex | None
        Index temporel (optionnel, pour grouper par mois).

    Returns
    -------
    np.ndarray : shape (12,) en kWh
    """
    import pandas as pd

    total_gains_w = np.zeros(8760)
    for win in windows_config:
        win_id = win.get("window_id", win.get("id", ""))
        irr = irradiance_series.get(win_id, np.zeros(8760))
        gains = solar_gains_through_window(
            irr,
            area_m2       = float(win.get("area_m2", 0)),
            g_value       = float(win.get("g_value", 0.62)),
            frame_factor  = float(win.get("frame_factor", 0.70)),
            shading_factor= float(win.get("shading_factor", 1.0)),
        )
        total_gains_w += gains

    if timestamps is None:
        timestamps = pd.date_range("2023-01-01 01:00", periods=8760, freq="h")

    s = pd.Series(total_gains_w, index=timestamps)
    return (s.groupby(s.index.month).sum().values / 1000.0)   # kWh


def prepare_irradiance_series(
    weather: "WeatherSeries",
    windows_config: list[dict],
) -> dict[str, np.ndarray]:
    """
    Calcule l'irradiance horaire sur toutes les fenêtres d'une zone.

    Parameters
    ----------
    weather : WeatherSeries   Données climatiques EPW
    windows_config : list     Liste des fenêtres avec tilt_deg, azimuth_deg

    Returns
    -------
    dict[str, np.ndarray] : {window_id: irradiance W/m², shape (8760,)}
    """
    import pandas as pd
    loc = weather.location

    # Génère les arrays doy et hour depuis le DatetimeIndex
    ts = weather.timestamps
    doy_arr  = ts.day_of_year.to_numpy(dtype=float)
    hour_arr = ts.hour.to_numpy(dtype=float) + 1.0   # Convention EPW : 1–24

    alt_rad, azm_rad = compute_solar_position(
        latitude_deg    = loc.latitude_deg,
        longitude_deg   = loc.longitude_deg,
        timezone_offset = loc.timezone_offset,
        doy_array       = doy_arr,
        hour_array      = hour_arr,
    )

    result = {}
    for win in windows_config:
        win_id = win.get("window_id", win.get("id", "win_?"))
        irr = irradiance_on_tilted_surface_hay_davies(
            ghi              = weather.ghi_wh_m2,
            dhi              = weather.dhi_wh_m2,
            dni              = weather.dni_wh_m2,
            tilt_deg         = float(win.get("tilt_deg", 90.0)),
            surface_azimuth_deg=float(win.get("azimuth_deg", win.get("orientation_deg", 180.0))),
            solar_altitude_rad = alt_rad,
            solar_azimuth_rad  = azm_rad,
            doy_array          = doy_arr,
        )
        result[win_id] = irr

    return result
