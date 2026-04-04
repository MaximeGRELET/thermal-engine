"""
Base de données des ponts thermiques linéiques.

Coefficients Ψ (psi) en W/m·K selon EN ISO 14683:2017.
Ces valeurs correspondent aux tableaux normatifs — elles sont conservatives
(approche par défaut) et peuvent être remplacées par des valeurs calculées
numériquement (approche détaillée).

Catégories de qualité de la liaison :
  - "default"   : valeurs par défaut ISO 14683 (conservatif)
  - "improved"  : construction soignée, traitement partiel des ponts
  - "optimised" : traitement complet (ETICS, couche de rupture, etc.)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalBridgeRef:
    bridge_type: str
    description: str
    psi_default: float    # W/m·K — qualité standard
    psi_improved: float   # W/m·K — construction soignée
    psi_optimised: float  # W/m·K — traitement optimisé


THERMAL_BRIDGE_DATABASE: dict[str, ThermalBridgeRef] = {
    # ─── Liaisons mur ────────────────────────────────────────────
    "wall_roof": ThermalBridgeRef(
        "wall_roof", "Liaison mur / toiture",
        psi_default=0.10, psi_improved=0.06, psi_optimised=0.02,
    ),
    "wall_floor_ground": ThermalBridgeRef(
        "wall_floor_ground", "Liaison mur / plancher bas (RDC sur terre-plein)",
        psi_default=0.45, psi_improved=0.25, psi_optimised=0.10,
    ),
    "wall_floor_intermediate": ThermalBridgeRef(
        "wall_floor_intermediate", "Liaison mur / plancher intermédiaire",
        psi_default=0.20, psi_improved=0.10, psi_optimised=0.04,
    ),
    "wall_corner_external": ThermalBridgeRef(
        "wall_corner_external", "Angle de mur extérieur (angle sortant)",
        psi_default=0.05, psi_improved=0.03, psi_optimised=0.01,
    ),
    "wall_corner_internal": ThermalBridgeRef(
        "wall_corner_internal", "Angle de mur intérieur (angle rentrant)",
        psi_default=-0.05, psi_improved=-0.03, psi_optimised=0.00,
    ),
    "wall_partition": ThermalBridgeRef(
        "wall_partition", "Liaison mur extérieur / cloison intérieure",
        psi_default=0.10, psi_improved=0.05, psi_optimised=0.01,
    ),

    # ─── Fenêtres ────────────────────────────────────────────────
    "window_reveal": ThermalBridgeRef(
        "window_reveal", "Encadrement de fenêtre (tableau)",
        psi_default=0.10, psi_improved=0.06, psi_optimised=0.02,
    ),
    "window_sill": ThermalBridgeRef(
        "window_sill", "Appui de fenêtre (sous-bassement)",
        psi_default=0.10, psi_improved=0.06, psi_optimised=0.02,
    ),
    "window_lintel": ThermalBridgeRef(
        "window_lintel", "Linteau de fenêtre",
        psi_default=0.10, psi_improved=0.06, psi_optimised=0.02,
    ),

    # ─── Balcons et porte-à-faux ─────────────────────────────────
    "balcony_slab": ThermalBridgeRef(
        "balcony_slab", "Dalle de balcon traversante",
        psi_default=0.80, psi_improved=0.40, psi_optimised=0.15,
    ),

    # ─── Piliers / poteaux ────────────────────────────────────────
    "column_steel": ThermalBridgeRef(
        "column_steel", "Poteau acier traversant l'enveloppe",
        psi_default=0.50, psi_improved=0.25, psi_optimised=0.10,
    ),
    "column_concrete": ThermalBridgeRef(
        "column_concrete", "Poteau béton traversant l'enveloppe",
        psi_default=0.30, psi_improved=0.15, psi_optimised=0.05,
    ),
}


def get_psi(
    bridge_type: str,
    quality: str = "default",
) -> float:
    """
    Retourne le coefficient Ψ (W/m·K) pour un type de pont thermique.

    Parameters
    ----------
    bridge_type : str
        Clé dans THERMAL_BRIDGE_DATABASE
    quality : str
        "default" | "improved" | "optimised"

    Returns
    -------
    float
        Coefficient Ψ en W/m·K
    """
    if bridge_type not in THERMAL_BRIDGE_DATABASE:
        known = ", ".join(sorted(THERMAL_BRIDGE_DATABASE.keys()))
        raise KeyError(
            f"Type de pont thermique inconnu : '{bridge_type}'.\n"
            f"Types disponibles : {known}"
        )
    entry = THERMAL_BRIDGE_DATABASE[bridge_type]
    field_map = {
        "default":   "psi_default",
        "improved":  "psi_improved",
        "optimised": "psi_optimised",
    }
    if quality not in field_map:
        raise ValueError(f"Qualité invalide : '{quality}'. Choisir parmi : {list(field_map)}")
    return getattr(entry, field_map[quality])
