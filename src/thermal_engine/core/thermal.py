"""
Calculs de transmission thermique.

Toutes les fonctions sont pures (sans état).
Références : EN ISO 6946, EN ISO 13370, EN ISO 14683, ISO 13790.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient U des parois
# ─────────────────────────────────────────────────────────────────────────────

def u_value_from_layers(
    layers: list[tuple[float, float]],   # (thickness_m, lambda_W_mK)
    rsi: float = 0.13,
    rse: float = 0.04,
) -> float:
    """
    Calcule le coefficient U [W/m²K] d'une paroi multicouche.
    Méthode : EN ISO 6946 — résistances en série.

    U = 1 / (Rsi + Σ(eᵢ/λᵢ) + Rse)
    """
    if not layers:
        raise ValueError("La composition de paroi est vide.")
    r_total = rsi + rse + sum(e / lam for e, lam in layers if lam > 0)
    return 1.0 / r_total


def u_value_ground_floor_iso13370(
    floor_area_m2: float,
    perimeter_m: float,
    wall_u_value: float,
    floor_layers: list[tuple[float, float]],
    rsi: float = 0.17,
) -> float:
    """
    U-value équivalent du plancher bas sur terre-plein selon EN ISO 13370.

    B' = 2·A / P  (dimension caractéristique du plancher)
    dt = épaisseur totale des couches du plancher + de la paroi (fictive)

    Pour dt < B' :
    Uf = (2·λsol)/(π·B' + dt) · ln(π·B'/dt + 1)

    Parameters
    ----------
    floor_area_m2 : float   Surface du plancher
    perimeter_m : float     Périmètre exposé du plancher
    wall_u_value : float    U-value du mur (pour calcul dt)
    floor_layers : list     Couches du plancher (e_m, lambda_W_mK)

    Returns
    -------
    float : Uf [W/m²K]
    """
    LAMBDA_SOL = 2.0   # Conductivité thermique du sol [W/m·K] (EN ISO 13370 valeur par défaut)

    if perimeter_m <= 0:
        return 0.5   # Valeur de repli pour bâtiments sans périmètre

    B_prime = 2 * floor_area_m2 / perimeter_m

    # Épaisseur fictive du plancher (résistances ramenées en épaisseur équivalente)
    r_floor = rsi + sum(e / lam for e, lam in floor_layers if lam > 0)
    r_wall  = 1.0 / wall_u_value if wall_u_value > 0 else 0
    dt = LAMBDA_SOL * (r_floor + r_wall)

    if dt < B_prime:
        u_f = (2 * LAMBDA_SOL) / (np.pi * B_prime + dt) * np.log(np.pi * B_prime / dt + 1)
    else:
        u_f = LAMBDA_SOL / (0.457 * B_prime + dt)

    return float(np.clip(u_f, 0.05, 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient de déperdition thermique H_T [W/K]
# ─────────────────────────────────────────────────────────────────────────────

def transmission_heat_loss_coefficient(
    elements: list[dict],
) -> float:
    """
    Calcule le coefficient de déperdition par transmission H_T [W/K].

    H_T = Σ(Uᵢ · Aᵢ · bᵢ) + Σ(Ψⱼ · lⱼ · bⱼ)

    Où bᵢ est le facteur de réduction de température (0 ≤ b ≤ 1) :
      - b = 1 pour les parois en contact avec l'extérieur
      - b < 1 pour les parois vers espace non chauffé (garage, combles non isolés...)
      - b = 0 pour les parois vers espace chauffé contigu

    Parameters
    ----------
    elements : list[dict]
        Chaque dict doit avoir :
          - "u_value" : float [W/m²K]
          - "area_m2" : float
          - "b_factor" : float (défaut 1.0)
          - "type" : "surface" | "linear_bridge"
          Pour les ponts thermiques linéiques :
          - "psi" : float [W/m·K]
          - "length_m" : float

    Returns
    -------
    float : H_T [W/K]
    """
    h_t = 0.0
    for el in elements:
        b = float(el.get("b_factor", 1.0))
        if el.get("type") == "linear_bridge":
            h_t += el["psi"] * el["length_m"] * b
        else:
            h_t += el["u_value"] * el["area_m2"] * b
    return h_t


# ─────────────────────────────────────────────────────────────────────────────
# Pertes horaires par transmission
# ─────────────────────────────────────────────────────────────────────────────

def transmission_losses_hourly(
    h_t_w_k: float,
    t_interior: np.ndarray,
    t_exterior: np.ndarray,
) -> np.ndarray:
    """
    Déperditions horaires par transmission [W].

    Q_tr(t) = H_T · (θᵢ - θₑ)

    Parameters
    ----------
    h_t_w_k : float           Coefficient H_T [W/K]
    t_interior : np.ndarray   Températures intérieures [°C], shape (8760,)
    t_exterior : np.ndarray   Températures extérieures [°C], shape (8760,)

    Returns
    -------
    np.ndarray : Pertes en W, shape (8760,)
    """
    return h_t_w_k * np.maximum(0, t_interior - t_exterior)


# ─────────────────────────────────────────────────────────────────────────────
# Pertes mensuelles (méthode ISO 13790)
# ─────────────────────────────────────────────────────────────────────────────

# Nombre d'heures par mois (année non-bissextile)
HOURS_PER_MONTH = np.array([744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744], dtype=float)

# Jours par mois
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)


def transmission_losses_monthly(
    h_t_w_k: float,
    t_interior_c: float,
    monthly_mean_temp_c: list[float],
) -> np.ndarray:
    """
    Déperditions mensuelles par transmission [kWh/mois].

    Q_tr,m = H_T · (θᵢ - θₑ,m) · t_m / 1000

    Parameters
    ----------
    h_t_w_k : float                Coefficient H_T [W/K]
    t_interior_c : float           Température intérieure de consigne [°C]
    monthly_mean_temp_c : list     Températures extérieures moyennes mensuelles (12 valeurs) [°C]

    Returns
    -------
    np.ndarray : shape (12,) en kWh
    """
    temps = np.array(monthly_mean_temp_c)
    delta_t = np.maximum(0, t_interior_c - temps)
    return h_t_w_k * delta_t * HOURS_PER_MONTH / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Facteur de température pour espaces non chauffés (b-factor)
# ─────────────────────────────────────────────────────────────────────────────

def b_factor_unheated_space(
    h_iu_w_k: float,
    h_ue_w_k: float,
) -> float:
    """
    Facteur de réduction de température b pour un espace non chauffé
    adjacent (ex. garage, combles perdus non isolés).

    b = H_ue / (H_iu + H_ue)

    Où H_iu = coefficient entre espace chauffé et espace non chauffé
        H_ue = coefficient entre espace non chauffé et extérieur

    Returns
    -------
    float : b ∈ [0, 1]
    """
    denom = h_iu_w_k + h_ue_w_k
    if denom <= 0:
        return 1.0
    return float(np.clip(h_ue_w_k / denom, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Inertie thermique
# ─────────────────────────────────────────────────────────────────────────────

def thermal_time_constant_h(
    heat_capacity_kj_k: float,
    h_t_w_k: float,
    h_v_w_k: float,
) -> float:
    """
    Constante de temps thermique du bâtiment τ [h].

    τ = C / (H_T + H_V)   [h]

    C = capacité thermique effective [kJ/K]
    H_T + H_V = coefficient global de déperdition [W/K]

    Parameters
    ----------
    heat_capacity_kj_k : float   Capacité thermique effective [kJ/K]
    h_t_w_k : float              Coefficient de transmission [W/K]
    h_v_w_k : float              Coefficient de ventilation [W/K]
    """
    h_total = h_t_w_k + h_v_w_k
    if h_total <= 0:
        return 0
    return float(heat_capacity_kj_k / h_total)


def effective_heat_capacity(
    floor_area_m2: float,
    construction_class: str = "medium",
) -> float:
    """
    Capacité thermique effective du bâtiment [kJ/K].

    Estimée par classe de construction (ISO 13790 Annexe F).

    Classes :
      "very_light"  : 80 kJ/m²K  (ossature bois, intérieur léger)
      "light"       : 110 kJ/m²K
      "medium"      : 165 kJ/m²K (construction standard)
      "heavy"       : 260 kJ/m²K (béton, pierre)
      "very_heavy"  : 370 kJ/m²K (béton massif)
    """
    kappa_map = {
        "very_light": 80,
        "light":      110,
        "medium":     165,
        "heavy":      260,
        "very_heavy": 370,
    }
    kappa = kappa_map.get(construction_class, 165)   # kJ/m²K
    return float(kappa * floor_area_m2)
