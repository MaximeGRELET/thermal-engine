"""
Base de données des matériaux de construction.

Propriétés thermiques selon EN ISO 10456:2007 et RT 2012 / RE 2020.
Chaque entrée contient :
  lambda_w_mk  : conductivité thermique [W/m·K]
  rho_kg_m3   : masse volumique [kg/m³]
  cp_j_kgk    : chaleur spécifique [J/kg·K]
  description  : label humain
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialProps:
    material_id: str
    description: str
    lambda_w_mk: float   # Conductivité thermique [W/m·K]
    rho_kg_m3: float     # Masse volumique [kg/m³]
    cp_j_kgk: float      # Chaleur spécifique [J/kg·K]


MATERIAL_DATABASE: dict[str, MaterialProps] = {
    # ─── Béton et maçonnerie ───────────────────────────────────────
    "concrete_dense": MaterialProps(
        "concrete_dense", "Béton lourd",
        lambda_w_mk=2.30, rho_kg_m3=2300, cp_j_kgk=1000,
    ),
    "concrete_cellular": MaterialProps(
        "concrete_cellular", "Béton cellulaire (AAC)",
        lambda_w_mk=0.11, rho_kg_m3=500, cp_j_kgk=1000,
    ),
    "concrete_lightweight": MaterialProps(
        "concrete_lightweight", "Béton léger",
        lambda_w_mk=0.60, rho_kg_m3=1400, cp_j_kgk=1000,
    ),
    "brick_full": MaterialProps(
        "brick_full", "Brique pleine terre cuite",
        lambda_w_mk=0.76, rho_kg_m3=1800, cp_j_kgk=840,
    ),
    "brick_hollow": MaterialProps(
        "brick_hollow", "Brique creuse terre cuite",
        lambda_w_mk=0.32, rho_kg_m3=1000, cp_j_kgk=840,
    ),
    "brick_monomur": MaterialProps(
        "brick_monomur", "Brique monomur (30 cm)",
        lambda_w_mk=0.087, rho_kg_m3=780, cp_j_kgk=1000,
    ),
    "stone_limestone": MaterialProps(
        "stone_limestone", "Pierre calcaire",
        lambda_w_mk=1.40, rho_kg_m3=2000, cp_j_kgk=1000,
    ),
    "stone_granite": MaterialProps(
        "stone_granite", "Granite / gneiss",
        lambda_w_mk=3.50, rho_kg_m3=2800, cp_j_kgk=1000,
    ),
    "screed": MaterialProps(
        "screed", "Chape ciment",
        lambda_w_mk=1.40, rho_kg_m3=2000, cp_j_kgk=1000,
    ),

    # ─── Isolants ─────────────────────────────────────────────────
    "mineral_wool": MaterialProps(
        "mineral_wool", "Laine minérale (verre ou roche)",
        lambda_w_mk=0.035, rho_kg_m3=30, cp_j_kgk=840,
    ),
    "mineral_wool_hd": MaterialProps(
        "mineral_wool_hd", "Laine minérale haute densité",
        lambda_w_mk=0.032, rho_kg_m3=70, cp_j_kgk=840,
    ),
    "eps_insulation": MaterialProps(
        "eps_insulation", "Polystyrène expansé (EPS)",
        lambda_w_mk=0.038, rho_kg_m3=20, cp_j_kgk=1450,
    ),
    "xps_insulation": MaterialProps(
        "xps_insulation", "Polystyrène extrudé (XPS)",
        lambda_w_mk=0.034, rho_kg_m3=35, cp_j_kgk=1450,
    ),
    "pur_pir_insulation": MaterialProps(
        "pur_pir_insulation", "Polyuréthane / PIR",
        lambda_w_mk=0.022, rho_kg_m3=32, cp_j_kgk=1400,
    ),
    "wood_fiber": MaterialProps(
        "wood_fiber", "Panneau fibre de bois",
        lambda_w_mk=0.038, rho_kg_m3=160, cp_j_kgk=2100,
    ),
    "hemp_wool": MaterialProps(
        "hemp_wool", "Laine de chanvre",
        lambda_w_mk=0.040, rho_kg_m3=35, cp_j_kgk=1700,
    ),
    "sheep_wool": MaterialProps(
        "sheep_wool", "Laine de mouton",
        lambda_w_mk=0.040, rho_kg_m3=25, cp_j_kgk=1700,
    ),
    "cellulose_blown": MaterialProps(
        "cellulose_blown", "Ouate de cellulose projetée",
        lambda_w_mk=0.040, rho_kg_m3=55, cp_j_kgk=1900,
    ),
    "aerogel": MaterialProps(
        "aerogel", "Panneau aérogel silice",
        lambda_w_mk=0.015, rho_kg_m3=150, cp_j_kgk=1000,
    ),
    "vacuum_insulation": MaterialProps(
        "vacuum_insulation", "Panneau isolant sous vide (VIP)",
        lambda_w_mk=0.007, rho_kg_m3=200, cp_j_kgk=800,
    ),

    # ─── Bois et dérivés ─────────────────────────────────────────
    "timber_softwood": MaterialProps(
        "timber_softwood", "Bois résineux (épicéa, pin)",
        lambda_w_mk=0.13, rho_kg_m3=500, cp_j_kgk=1600,
    ),
    "timber_hardwood": MaterialProps(
        "timber_hardwood", "Bois feuillu (chêne)",
        lambda_w_mk=0.18, rho_kg_m3=700, cp_j_kgk=1600,
    ),
    "osb_panel": MaterialProps(
        "osb_panel", "Panneau OSB",
        lambda_w_mk=0.13, rho_kg_m3=650, cp_j_kgk=1700,
    ),
    "plywood": MaterialProps(
        "plywood", "Contreplaqué",
        lambda_w_mk=0.13, rho_kg_m3=600, cp_j_kgk=1600,
    ),

    # ─── Plâtre et finitions ──────────────────────────────────────
    "plasterboard": MaterialProps(
        "plasterboard", "Plaque de plâtre (BA13)",
        lambda_w_mk=0.21, rho_kg_m3=900, cp_j_kgk=1000,
    ),
    "plaster_coat": MaterialProps(
        "plaster_coat", "Enduit plâtre",
        lambda_w_mk=0.57, rho_kg_m3=1300, cp_j_kgk=1000,
    ),
    "render_cement": MaterialProps(
        "render_cement", "Enduit ciment extérieur",
        lambda_w_mk=1.00, rho_kg_m3=1800, cp_j_kgk=1000,
    ),

    # ─── Toiture ─────────────────────────────────────────────────
    "roof_tile": MaterialProps(
        "roof_tile", "Tuile terre cuite",
        lambda_w_mk=1.00, rho_kg_m3=2000, cp_j_kgk=800,
    ),
    "slate": MaterialProps(
        "slate", "Ardoise",
        lambda_w_mk=2.20, rho_kg_m3=2800, cp_j_kgk=760,
    ),
    "bitumen_membrane": MaterialProps(
        "bitumen_membrane", "Membrane bitumineuse",
        lambda_w_mk=0.17, rho_kg_m3=1100, cp_j_kgk=1000,
    ),

    # ─── Divers ──────────────────────────────────────────────────
    "air_gap": MaterialProps(
        "air_gap", "Lame d'air non ventilée",
        lambda_w_mk=0.18, rho_kg_m3=1.2, cp_j_kgk=1006,
    ),
    "steel": MaterialProps(
        "steel", "Acier",
        lambda_w_mk=50.00, rho_kg_m3=7800, cp_j_kgk=500,
    ),
    "glass": MaterialProps(
        "glass", "Verre (cloison)",
        lambda_w_mk=1.00, rho_kg_m3=2500, cp_j_kgk=750,
    ),
}


def get_material(material_id: str) -> MaterialProps:
    """Retourne les propriétés d'un matériau depuis la base de données."""
    if material_id not in MATERIAL_DATABASE:
        known = ", ".join(sorted(MATERIAL_DATABASE.keys()))
        raise KeyError(
            f"Matériau inconnu : '{material_id}'.\n"
            f"Matériaux disponibles : {known}"
        )
    return MATERIAL_DATABASE[material_id]
