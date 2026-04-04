"""
Profils d'occupation, d'usage et apports internes.

Les profils sont définis sur 8760 heures (une année non-bissextile).
Convention : heure 0 = 1er janvier à 00h–01h.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Profils d'occupation horaires (valeur entre 0.0 et 1.0)
# ─────────────────────────────────────────────────────────────────────────────

def _build_weekly_profile(
    hour_weekday: list[float],   # 24 valeurs
    hour_weekend: list[float],   # 24 valeurs
) -> np.ndarray:
    """
    Construit un vecteur de 8760 valeurs à partir de profils jour/weekend.
    Répète le schéma hebdomadaire sur 52 semaines + correction.
    """
    assert len(hour_weekday) == 24 and len(hour_weekend) == 24
    profile = np.zeros(8760)
    for day in range(365):
        hour_start = day * 24
        hour_end   = hour_start + 24
        if hour_end > 8760:
            break
        # Lundi=0, ..., Dimanche=6 (jour 0 = 1er janvier 2023 = dimanche)
        weekday = (day + 6) % 7   # 0=lundi, 6=dimanche
        if weekday < 5:
            profile[hour_start:hour_end] = hour_weekday
        else:
            profile[hour_start:hour_end] = hour_weekend
    return profile


def get_occupancy_schedule(schedule_name: str) -> np.ndarray:
    """
    Retourne un profil d'occupation horaire (8760 valeurs, 0–1).

    Profils disponibles :
      "residential_standard"  : Résidentiel standard (RT 2012 / RE 2020)
      "office_standard"       : Bureau 9h–18h semaine
      "retail_standard"       : Commerce 9h–20h
      "restaurant_standard"   : Restaurant 11h–15h + 18h–23h
      "school_standard"       : École 8h–17h semaine, hors vacances
      "always_on"             : Présence continue (1.0 partout)
      "always_off"            : Aucune occupation (0.0 partout)
    """
    if schedule_name == "always_on":
        return np.ones(8760)
    if schedule_name == "always_off":
        return np.zeros(8760)

    profiles = {
        "residential_standard": (
            # Semaine : pointes matin et soir
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8,    # 0–5 nuit
             0.9, 1.0, 0.7, 0.3, 0.3, 0.3,    # 6–11 matin
             0.3, 0.3, 0.3, 0.3, 0.5, 0.8,    # 12–17 après-midi
             1.0, 1.0, 1.0, 1.0, 0.9, 0.8],   # 18–23 soirée
            # Weekend : occupation plus continue
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
             0.8, 0.9, 1.0, 1.0, 1.0, 0.9,
             0.9, 0.9, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 0.9, 0.8],
        ),
        "office_standard": (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.1, 0.7, 1.0, 1.0, 0.8,
             0.5, 0.8, 1.0, 1.0, 0.8, 0.4,
             0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 24,  # Weekend vide
        ),
        "retail_standard": (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.5, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.5, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        "restaurant_standard": (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.3,
             1.0, 1.0, 0.5, 0.1, 0.0, 0.1,
             0.5, 1.0, 1.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.1, 0.5,
             1.0, 1.0, 0.7, 0.3, 0.1, 0.2,
             0.7, 1.0, 1.0, 1.0, 0.5, 0.1],
        ),
        "school_standard": (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
             0.7, 1.0, 1.0, 1.0, 0.5, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 24,
        ),
    }

    if schedule_name not in profiles:
        raise ValueError(
            f"Profil inconnu : '{schedule_name}'. "
            f"Disponibles : {list(profiles.keys()) + ['always_on', 'always_off']}"
        )

    weekday_profile, weekend_profile = profiles[schedule_name]
    return _build_weekly_profile(weekday_profile, weekend_profile)


# ─────────────────────────────────────────────────────────────────────────────
# Profil de température de consigne
# ─────────────────────────────────────────────────────────────────────────────

def heating_setpoint_schedule(
    t_setpoint_day: float = 19.0,
    t_setpoint_night: float = 16.0,
    occupation_schedule: np.ndarray | None = None,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Profil de température de consigne chauffage [°C].

    Utilise le profil d'occupation : si occupation > threshold → consigne jour,
    sinon consigne nuit (réduit/hors-gel).

    Returns
    -------
    np.ndarray : shape (8760,) en °C
    """
    if occupation_schedule is None:
        occupation_schedule = get_occupancy_schedule("residential_standard")
    occupied = occupation_schedule > threshold
    return np.where(occupied, t_setpoint_day, t_setpoint_night)


def cooling_setpoint_schedule(
    t_setpoint: float = 26.0,
    occupation_schedule: np.ndarray | None = None,
    threshold: float = 0.3,
) -> np.ndarray:
    """Profil de température de consigne refroidissement [°C]."""
    if occupation_schedule is None:
        occupation_schedule = np.ones(8760)
    occupied = occupation_schedule > threshold
    return np.where(occupied, t_setpoint, t_setpoint + 4.0)   # +4°C nuit


# ─────────────────────────────────────────────────────────────────────────────
# Apports internes
# ─────────────────────────────────────────────────────────────────────────────

def internal_gains_hourly(
    floor_area_m2: float,
    n_persons: int,
    occupation_schedule: np.ndarray,
    heat_per_person_w: float = 80.0,      # W/personne (activité sédentaire)
    appliances_w_m2: float = 4.0,          # W/m² équipements
    lighting_w_m2: float = 2.0,            # W/m² éclairage
    lighting_schedule: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apports internes horaires [W].

    Q_int(t) = Q_personnes(t) + Q_équipements(t) + Q_éclairage(t)

    Parameters
    ----------
    heat_per_person_w : float
        Chaleur sensible dégagée par personne [W]
        (80 W assis, 120 W activité légère, 175 W debout actif)
    appliances_w_m2 : float
        Puissance des équipements [W/m²]
        (4 W/m² résidentiel, 10–15 W/m² bureaux)
    lighting_w_m2 : float
        Puissance d'éclairage [W/m²]
        (2 W/m² résidentiel LED, 8–12 W/m² bureaux)

    Returns
    -------
    np.ndarray : Apports internes [W], shape (8760,)
    """
    q_persons    = n_persons * heat_per_person_w * occupation_schedule
    q_appliances = appliances_w_m2 * floor_area_m2 * occupation_schedule
    if lighting_schedule is None:
        lighting_schedule = occupation_schedule
    q_lighting   = lighting_w_m2 * floor_area_m2 * lighting_schedule

    return q_persons + q_appliances + q_lighting


def internal_gains_monthly(
    floor_area_m2: float,
    n_persons: int,
    schedule_name: str = "residential_standard",
    heat_per_person_w: float = 80.0,
    appliances_w_m2: float = 4.0,
    lighting_w_m2: float = 2.0,
) -> np.ndarray:
    """
    Apports internes mensuels [kWh/mois].

    Returns
    -------
    np.ndarray : shape (12,) en kWh
    """
    import pandas as pd
    occ = get_occupancy_schedule(schedule_name)
    gains_w = internal_gains_hourly(
        floor_area_m2 = floor_area_m2,
        n_persons     = n_persons,
        occupation_schedule = occ,
        heat_per_person_w   = heat_per_person_w,
        appliances_w_m2     = appliances_w_m2,
        lighting_w_m2       = lighting_w_m2,
    )
    ts = pd.date_range("2023-01-01 01:00", periods=8760, freq="h")
    s = pd.Series(gains_w, index=ts)
    return (s.groupby(s.index.month).sum().values / 1000.0)   # kWh


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres d'occupation par usage (valeurs par défaut RE 2020 / RT 2012)
# ─────────────────────────────────────────────────────────────────────────────

USAGE_DEFAULTS: dict[str, dict] = {
    "residential": {
        "schedule":          "residential_standard",
        "heat_per_person_w": 80.0,
        "appliances_w_m2":   4.0,
        "lighting_w_m2":     2.0,
        "n_persons_per_m2":  0.035,   # ~1 pers/28 m²
    },
    "office": {
        "schedule":          "office_standard",
        "heat_per_person_w": 80.0,
        "appliances_w_m2":   15.0,    # Écrans, serveurs...
        "lighting_w_m2":     10.0,
        "n_persons_per_m2":  0.10,    # ~1 pers/10 m²
    },
    "retail": {
        "schedule":          "retail_standard",
        "heat_per_person_w": 100.0,
        "appliances_w_m2":   8.0,
        "lighting_w_m2":     15.0,
        "n_persons_per_m2":  0.20,
    },
    "restaurant": {
        "schedule":          "restaurant_standard",
        "heat_per_person_w": 100.0,
        "appliances_w_m2":   25.0,    # Équipements cuisine
        "lighting_w_m2":     12.0,
        "n_persons_per_m2":  0.30,
    },
    "school": {
        "schedule":          "school_standard",
        "heat_per_person_w": 80.0,
        "appliances_w_m2":   5.0,
        "lighting_w_m2":     8.0,
        "n_persons_per_m2":  0.40,
    },
}
