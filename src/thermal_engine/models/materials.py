"""
Modèles de données pour les matériaux et compositions de parois.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from ..data.material_db import get_material, MaterialProps


@dataclass
class MaterialLayer:
    """Une couche de matériau dans une composition de paroi."""
    material_id: str
    thickness_m: float          # Épaisseur [m]
    _props: MaterialProps = field(init=False, repr=False)

    def __post_init__(self):
        self._props = get_material(self.material_id)

    @property
    def lambda_w_mk(self) -> float:
        return self._props.lambda_w_mk

    @property
    def rho_kg_m3(self) -> float:
        return self._props.rho_kg_m3

    @property
    def cp_j_kgk(self) -> float:
        return self._props.cp_j_kgk

    @property
    def description(self) -> str:
        return self._props.description

    @property
    def thermal_resistance_m2k_w(self) -> float:
        """Résistance thermique de la couche [m²·K/W]."""
        return self.thickness_m / self.lambda_w_mk

    @property
    def heat_capacity_kj_m2k(self) -> float:
        """Capacité thermique surfacique [kJ/m²·K] — utilisée pour l'inertie thermique."""
        return self.rho_kg_m3 * self.cp_j_kgk * self.thickness_m / 1000


@dataclass
class LayeredComposition:
    """
    Composition multicouche d'une paroi (mur, toiture, plancher).

    Les résistances surfaciques Rsi et Rse suivent EN ISO 6946 :
      - Mur vertical    : Rsi = 0.13, Rse = 0.04 m²K/W
      - Toiture         : Rsi = 0.10, Rse = 0.04 m²K/W
      - Plancher bas    : Rsi = 0.17, Rse = 0.04 m²K/W (côté sol)
      - Plancher haut   : Rsi = 0.10, Rse = 0.04 m²K/W (côté air)
    """
    layers: list[MaterialLayer]
    rsi_m2k_w: float = 0.13     # Résistance superficielle intérieure [m²K/W]
    rse_m2k_w: float = 0.04     # Résistance superficielle extérieure [m²K/W]

    @property
    def total_resistance_m2k_w(self) -> float:
        """Résistance thermique totale RT = Rsi + ΣRi + Rse [m²K/W]."""
        r_layers = sum(lay.thermal_resistance_m2k_w for lay in self.layers)
        return self.rsi_m2k_w + r_layers + self.rse_m2k_w

    @property
    def u_value_w_m2k(self) -> float:
        """Coefficient de transmission thermique U = 1/RT [W/m²K]."""
        return 1.0 / self.total_resistance_m2k_w

    @property
    def total_thickness_m(self) -> float:
        return sum(lay.thickness_m for lay in self.layers)

    @property
    def heat_capacity_kj_m2k(self) -> float:
        """Capacité thermique effective de la paroi [kJ/m²K]."""
        return sum(lay.heat_capacity_kj_m2k for lay in self.layers)

    def add_insulation_layer(
        self,
        material_id: str,
        thickness_m: float,
        position: str = "exterior",
    ) -> "LayeredComposition":
        """
        Retourne une NOUVELLE composition avec une couche d'isolation ajoutée.

        Parameters
        ----------
        material_id : str
            Identifiant du matériau isolant
        thickness_m : float
            Épaisseur [m]
        position : "exterior" | "interior"
            Côté d'ajout de l'isolant
        """
        import copy
        new_layer = MaterialLayer(material_id=material_id, thickness_m=thickness_m)
        new_layers = copy.copy(self.layers)
        if position == "exterior":
            new_layers.insert(0, new_layer)
        else:
            new_layers.append(new_layer)
        return LayeredComposition(
            layers     = new_layers,
            rsi_m2k_w  = self.rsi_m2k_w,
            rse_m2k_w  = self.rse_m2k_w,
        )

    def to_dict(self) -> dict:
        return {
            "u_value_w_m2k":      round(self.u_value_w_m2k, 3),
            "total_thickness_m":  round(self.total_thickness_m, 3),
            "heat_capacity_kj_m2k": round(self.heat_capacity_kj_m2k, 1),
            "layers": [
                {
                    "material_id":  lay.material_id,
                    "description":  lay.description,
                    "thickness_m":  lay.thickness_m,
                    "lambda_w_mk":  lay.lambda_w_mk,
                    "R_m2k_w":      round(lay.thermal_resistance_m2k_w, 3),
                }
                for lay in self.layers
            ],
        }


def composition_from_dict(data: dict) -> LayeredComposition:
    """
    Construit une LayeredComposition depuis un dict JSON.

    Format attendu :
    {
        "layers": [{"material_id": "...", "thickness_m": 0.20}, ...],
        "rsi_m2k_w": 0.13,  # optionnel
        "rse_m2k_w": 0.04   # optionnel
    }
    """
    layers = [
        MaterialLayer(
            material_id = lay["material_id"],
            thickness_m = float(lay["thickness_m"]),
        )
        for lay in data["layers"]
    ]
    return LayeredComposition(
        layers    = layers,
        rsi_m2k_w = float(data.get("surface_resistance_interior", data.get("rsi_m2k_w", 0.13))),
        rse_m2k_w = float(data.get("surface_resistance_exterior", data.get("rse_m2k_w", 0.04))),
    )
