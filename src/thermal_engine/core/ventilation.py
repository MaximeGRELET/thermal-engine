"""
Calculs des pertes thermiques par ventilation et infiltration.

Références : ISO 13790, EN 15242, EN 15251.
"""

from __future__ import annotations
import numpy as np

# Propriétés de l'air à conditions standard (20°C, 1013 hPa)
RHO_AIR = 1.204       # Masse volumique de l'air [kg/m³]
CP_AIR  = 1006.0      # Chaleur spécifique de l'air [J/kg·K]
# Facteur volumique : ρ × Cp = 1.204 × 1006 ≈ 1211 J/(m³·K) ≈ 0.336 Wh/(m³·K)
RHO_CP_AIR = RHO_AIR * CP_AIR / 3600   # Wh/(m³·K) → pour calculs en Wh directement


def ventilation_heat_loss_coefficient(
    volume_m3: float,
    effective_ach: float,
    wind_speed_m_s: float = 0.0,
    wind_correction: bool = False,
) -> float:
    """
    Coefficient de déperdition par ventilation H_V [W/K].

    H_V = ρ_air · Cp_air · V_dot
        = ρ_air · Cp_air · Volume · ACH_eff / 3600

    Correction éolienne (EN 15242) optionnelle :
      ACH_eff_wind = ACH_eff + 0.04 × v_vent  (approximation)

    Parameters
    ----------
    volume_m3 : float          Volume chauffé [m³]
    effective_ach : float      Taux de renouvellement effectif [vol/h]
                               (pour VMC double flux : ACH × (1 - η_récup))
    wind_speed_m_s : float     Vitesse du vent [m/s]
    wind_correction : bool     Appliquer la correction éolienne

    Returns
    -------
    float : H_V [W/K]
    """
    ach = effective_ach
    if wind_correction and wind_speed_m_s > 0:
        # Correction simplifiée : infiltrations supplémentaires dues au vent
        ach_extra = 0.04 * wind_speed_m_s
        ach = ach + ach_extra

    # H_V = ρ × Cp × V_dot  [W/K]
    # V_dot = Volume × ACH / 3600  [m³/s]
    h_v = RHO_AIR * CP_AIR * volume_m3 * ach / 3600.0
    return float(h_v)


def ventilation_losses_hourly(
    h_v_w_k: float,
    t_interior: np.ndarray,
    t_exterior: np.ndarray,
) -> np.ndarray:
    """
    Déperditions horaires par ventilation [W].

    Q_ve(t) = H_V · max(0, θᵢ - θₑ)

    Parameters
    ----------
    h_v_w_k : float           H_V [W/K]
    t_interior : np.ndarray   Températures intérieures [°C], shape (8760,)
    t_exterior : np.ndarray   Températures extérieures [°C], shape (8760,)

    Returns
    -------
    np.ndarray : shape (8760,) en W
    """
    return h_v_w_k * np.maximum(0, t_interior - t_exterior)


def ventilation_losses_monthly(
    volume_m3: float,
    effective_ach: float,
    t_interior_c: float,
    monthly_mean_temp_c: list[float],
    hours_per_month: np.ndarray | None = None,
) -> np.ndarray:
    """
    Déperditions mensuelles par ventilation [kWh/mois].

    Parameters
    ----------
    volume_m3 : float
    effective_ach : float
    t_interior_c : float
    monthly_mean_temp_c : list[float]    12 valeurs
    hours_per_month : np.ndarray         Heures par mois (défaut : année standard)

    Returns
    -------
    np.ndarray : shape (12,) en kWh
    """
    if hours_per_month is None:
        from .thermal import HOURS_PER_MONTH
        hours_per_month = HOURS_PER_MONTH

    h_v = ventilation_heat_loss_coefficient(volume_m3, effective_ach)
    temps = np.array(monthly_mean_temp_c)
    delta_t = np.maximum(0, t_interior_c - temps)
    return h_v * delta_t * hours_per_month / 1000.0


def mvhr_recovered_heat_monthly(
    volume_m3: float,
    ach_total: float,
    heat_recovery_efficiency: float,
    t_interior_c: float,
    monthly_mean_temp_c: list[float],
    hours_per_month: np.ndarray | None = None,
) -> np.ndarray:
    """
    Chaleur récupérée par la VMC double flux [kWh/mois].

    Q_rec = ρ × Cp × V_dot × η × (θᵢ - θₑ) × t_m / 1000

    Returns
    -------
    np.ndarray : shape (12,) en kWh (valeur positive = énergie économisée)
    """
    if hours_per_month is None:
        from .thermal import HOURS_PER_MONTH
        hours_per_month = HOURS_PER_MONTH

    h_v_total = ventilation_heat_loss_coefficient(volume_m3, ach_total)
    temps = np.array(monthly_mean_temp_c)
    delta_t = np.maximum(0, t_interior_c - temps)
    return h_v_total * heat_recovery_efficiency * delta_t * hours_per_month / 1000.0


def infiltration_ach_from_n50(
    n50_vol_h: float,
    e_factor: float = 0.07,
    f_factor: float = 15.0,
) -> float:
    """
    Taux d'infiltration moyen [vol/h] à partir du test d'étanchéité n50.

    Méthode simplifiée (EN 15242 / ASHRAE) :
      ACH_inf = n50 / f

    Valeurs typiques de n50 :
      - Maison ancienne, aucun traitement : 15–20 vol/h
      - Construction standard RT 2005     : 4–7 vol/h
      - BBC / RT 2012                     : ≤ 1.0 vol/h (maison) / ≤ 1.5 (appartement)
      - Passif                            : ≤ 0.6 vol/h

    Parameters
    ----------
    n50_vol_h : float   Débit sous 50 Pa [vol/h] (issu du test BlowerDoor)
    e_factor : float    Facteur d'exposition (0.02–0.10, défaut 0.07 = expo modérée)
    f_factor : float    Facteur de protection (15 = exposition modérée)

    Returns
    -------
    float : ACH d'infiltration moyen annuel [vol/h]
    """
    return n50_vol_h * e_factor


def ach_from_construction_year(year: int, building_type: str = "maison") -> tuple[float, float]:
    """
    Estime le taux de renouvellement d'air (ACH total) et le n50 typique
    selon l'année de construction (source : ADEME, CSTB 2020).

    Returns
    -------
    tuple[float, float] : (ACH_naturel [vol/h], n50_estimé [vol/h])
    """
    # Données empiriques
    if year < 1975:
        return (1.0, 15.0)
    elif year < 1990:
        return (0.8, 10.0)
    elif year < 2005:
        return (0.6, 7.0)
    elif year < 2013:
        return (0.5, 4.0)
    else:
        return (0.4, 1.0 if building_type == "maison" else 1.5)
