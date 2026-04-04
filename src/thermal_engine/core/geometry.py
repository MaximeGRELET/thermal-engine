"""
Calculs géométriques pour les bâtiments décrits par emprises 2D + hauteur.

Toutes les fonctions sont pures (sans état, sans I/O).
Les coordonnées peuvent être :
  - Métriques (Lambert 93, UTM, etc.) → utilisées telles quelles
  - Géographiques (WGS84 lon/lat) → converties en mètres locaux

Conventions :
  - Azimut : 0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest (sens horaire)
  - Inclinaison : 0° = horizontal, 90° = vertical
"""

from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import shared_paths, snap
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

Coords2D = list[list[float]]   # Liste de [x, y] (ou [lon, lat])


class WallSegment(NamedTuple):
    """Un segment de mur issu de l'emprise 2D."""
    segment_id: str
    length_m: float
    azimuth_deg: float      # Orientation de la face extérieure (0=N, 90=E, 180=S, 270=O)
    is_exterior: bool       # False si mur mitoyen/adjacent à une autre zone
    x0: float               # Coordonnée x du premier sommet (système métrique local)
    y0: float
    x1: float
    y1: float


class ZoneGeometry(NamedTuple):
    """Géométrie dérivée d'une zone bâtie."""
    zone_id: str
    floor_area_m2: float
    perimeter_m: float
    wall_segments: list[WallSegment]
    roof_area_m2: float           # = floor_area pour toiture plate, + pour toiture pentue
    ground_floor_area_m2: float   # Surface en contact avec le sol (RDC uniquement)
    volume_m3: float
    # Rapport largeur/longueur (forme compacité)
    compactness: float            # Rapport surface enveloppe / volume


# ─────────────────────────────────────────────────────────────────────────────
# Détection du système de coordonnées
# ─────────────────────────────────────────────────────────────────────────────

def is_geographic(coords: Coords2D) -> bool:
    """
    Détecte si les coordonnées sont en WGS84 (lon/lat) ou métriques.
    Si |x| ≤ 180 et |y| ≤ 90, on suppose des coordonnées géographiques.
    """
    if not coords:
        return False
    x_vals = [p[0] for p in coords]
    y_vals = [p[1] for p in coords]
    return (max(abs(v) for v in x_vals) <= 181 and
            max(abs(v) for v in y_vals) <= 91)


def geographic_to_local_metric(coords: Coords2D) -> tuple[np.ndarray, float, float]:
    """
    Convertit des coordonnées WGS84 (lon/lat) en coordonnées métriques locales.

    Utilise une projection tangente locale (approximation valide sur < 50 km).
    Le premier point devient l'origine (0, 0).

    Returns
    -------
    local_coords : np.ndarray, shape (N, 2)
        Coordonnées en mètres
    origin_lon : float
    origin_lat : float
    """
    pts = np.array(coords, dtype=float)
    origin_lon = pts[0, 0]
    origin_lat = pts[0, 1]

    R = 6_371_000.0  # Rayon moyen de la Terre [m]
    lat_rad = np.radians(origin_lat)

    dx = np.radians(pts[:, 0] - origin_lon) * R * np.cos(lat_rad)
    dy = np.radians(pts[:, 1] - origin_lat) * R

    local = np.column_stack([dx, dy])
    return local, origin_lon, origin_lat


def to_metric_coords(coords: Coords2D) -> np.ndarray:
    """
    Retourne les coordonnées en mètres (conversion auto si géographiques).

    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    if is_geographic(coords):
        local, _, _ = geographic_to_local_metric(coords)
        return local
    return np.array(coords, dtype=float)[:, :2]   # On ignore Z si présent


# ─────────────────────────────────────────────────────────────────────────────
# Calculs sur polygone 2D
# ─────────────────────────────────────────────────────────────────────────────

def polygon_area_m2(coords: Coords2D) -> float:
    """
    Aire d'un polygone 2D en m² (formule de Shoelace / Gauss).
    Gère les coordonnées métriques et géographiques.
    """
    pts = to_metric_coords(coords)
    poly = Polygon(pts)
    return float(poly.area)


def polygon_perimeter_m(coords: Coords2D) -> float:
    """Périmètre d'un polygone 2D en mètres."""
    pts = to_metric_coords(coords)
    poly = Polygon(pts)
    return float(poly.length)


def segment_length_m(x0: float, y0: float, x1: float, y1: float) -> float:
    """Longueur d'un segment [m]."""
    return float(np.hypot(x1 - x0, y1 - y0))


def segment_azimuth_deg(x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Azimut de la face extérieure d'un mur [°].

    Le vecteur mur est (x1-x0, y1-y0). La normale extérieure pointe à 90°
    dans le sens horaire. L'azimut suit la convention géographique (0=N, 90=E).

    Hypothèse : le polygone de l'emprise est orienté dans le sens trigonométrique
    (anti-horaire), donc la normale extérieure est à droite du vecteur segment.
    Si le polygone est horaire, il faut inverser (géré dans extract_wall_segments).
    """
    dx = x1 - x0
    dy = y1 - y0
    # Normale extérieure (rotation -90° du vecteur segment = sens horaire)
    nx =  dy
    ny = -dx
    # Azimut (0=Nord, 90=Est) depuis (nx, ny) en coordonnées locales (x=Est, y=Nord)
    azimuth = np.degrees(np.arctan2(nx, ny)) % 360
    return float(azimuth)


def orientation_label(azimuth_deg: float) -> str:
    """Retourne le label d'orientation (N, NE, E, SE, S, SO, O, NO)."""
    sectors = [
        (22.5,  "N"), (67.5,  "NE"), (112.5, "E"),  (157.5, "SE"),
        (202.5, "S"), (247.5, "SO"), (292.5, "O"),   (337.5, "NO"),
    ]
    for limit, label in sectors:
        if azimuth_deg < limit:
            return label
    return "N"


def wwr_for_azimuth(
    azimuth_deg: float,
    wwr_by_orientation: dict[str, float],
) -> float:
    """
    Retourne le Window-to-Wall Ratio pour un azimut donné.

    wwr_by_orientation est un dict avec clés "north"/"south"/"east"/"west"
    (ou abréviations "n"/"s"/"e"/"o").
    """
    key_map = {
        "n": "north", "s": "south", "e": "east", "o": "west", "w": "west",
        "ne": "north", "se": "south", "so": "west", "no": "north",
        "nw": "north", "sw": "south",
    }
    label = orientation_label(azimuth_deg).lower()
    # Normalisation anglais/français
    resolved = key_map.get(label, label)
    return float(wwr_by_orientation.get(resolved, wwr_by_orientation.get(label, 0.20)))


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des murs depuis l'emprise
# ─────────────────────────────────────────────────────────────────────────────

def extract_wall_segments(
    coords: Coords2D,
    zone_id: str,
    exterior_flags: list[bool] | None = None,
) -> list[WallSegment]:
    """
    Extrait les segments de mur depuis l'emprise 2D d'une zone.

    Parameters
    ----------
    coords : Coords2D
        Coordonnées de l'emprise (polygone fermé ou ouvert, ordre quelconque)
    zone_id : str
        Identifiant de la zone
    exterior_flags : list[bool] | None
        Si fourni, indique pour chaque segment s'il est extérieur.
        Si None, tous les segments sont considérés extérieurs.

    Returns
    -------
    list[WallSegment]
        Un segment par arête du polygone.
    """
    pts = to_metric_coords(coords)

    # Fermeture du polygone si nécessaire
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # Assure l'orientation anti-horaire (sens trigonométrique)
    # pour que la normale extérieure soit correcte.
    poly = Polygon(pts)
    if poly.area < 0 or not poly.exterior.is_ccw:
        pts = pts[::-1]

    segments = []
    n = len(pts) - 1   # nombre d'arêtes
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        length = segment_length_m(x0, y0, x1, y1)
        if length < 0.01:   # Ignore les segments dégénérés (< 1 cm)
            continue
        azimuth = segment_azimuth_deg(x0, y0, x1, y1)
        is_ext = exterior_flags[i] if exterior_flags is not None else True
        segments.append(WallSegment(
            segment_id  = f"{zone_id}_seg_{i:03d}",
            length_m    = length,
            azimuth_deg = azimuth,
            is_exterior = is_ext,
            x0=x0, y0=y0, x1=x1, y1=y1,
        ))
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Détection des murs adjacents entre zones
# ─────────────────────────────────────────────────────────────────────────────

def find_shared_edges(
    coords_a: Coords2D,
    coords_b: Coords2D,
    tolerance_m: float = 0.10,
) -> list[tuple[float, float, float, float]]:
    """
    Détecte les arêtes partagées entre deux emprises de zones.

    Retourne une liste de segments (x0, y0, x1, y1) représentant les portions
    de périmètre communes (murs mitoyens).

    Parameters
    ----------
    tolerance_m : float
        Tolérance géométrique pour considérer deux points comme identiques [m].
    """
    pts_a = to_metric_coords(coords_a)
    pts_b = to_metric_coords(coords_b)

    poly_a = Polygon(pts_a)
    poly_b = Polygon(pts_b)

    # Snap pour aligner les géométries proches dans la tolérance
    poly_b_snapped = snap(poly_b, poly_a, tolerance_m)

    # Chemins partagés — Shapely 2.x retourne une GeometryCollection
    result = shared_paths(poly_a.exterior, poly_b_snapped.exterior)
    # result.geoms[0] = forward paths, result.geoms[1] = backward paths
    geom_parts = list(result.geoms) if not result.is_empty else []

    shared = []
    for geom in geom_parts:
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            geoms = [geom]
        elif isinstance(geom, MultiLineString):
            geoms = list(geom.geoms)
        else:
            geoms = list(geom.geoms)
        for line in geoms:
            coords_line = list(line.coords)
            for j in range(len(coords_line) - 1):
                x0, y0 = coords_line[j]
                x1, y1 = coords_line[j + 1]
                if segment_length_m(x0, y0, x1, y1) > tolerance_m:
                    shared.append((x0, y0, x1, y1))

    return shared


def compute_exterior_flags(
    coords: Coords2D,
    shared_edges: list[tuple[float, float, float, float]],
    tolerance_m: float = 0.10,
) -> list[bool]:
    """
    Pour chaque segment de l'emprise, détermine s'il est extérieur
    ou mitoyen (présent dans shared_edges).

    Returns
    -------
    list[bool]
        True = extérieur, False = mitoyen/intérieur
    """
    pts = to_metric_coords(coords)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    flags = []
    n = len(pts) - 1
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        is_shared = False
        for sx0, sy0, sx1, sy1 in shared_edges:
            # Vérifie si le milieu du segment courant est sur une arête partagée
            smid_x = (sx0 + sx1) / 2
            smid_y = (sy0 + sy1) / 2
            if np.hypot(mid_x - smid_x, mid_y - smid_y) < tolerance_m * 2:
                is_shared = True
                break
        flags.append(not is_shared)

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Calcul de la géométrie complète d'une zone
# ─────────────────────────────────────────────────────────────────────────────

def compute_zone_geometry(
    zone_id: str,
    footprint_coords: Coords2D,
    height_m: float,
    n_floors: int = 1,
    roof_type: str = "flat",
    roof_pitch_deg: float = 30.0,
    is_ground_floor: bool = True,
    shared_edges: list[tuple[float, float, float, float]] | None = None,
) -> ZoneGeometry:
    """
    Calcule toute la géométrie d'une zone bâtie depuis son emprise 2D.

    Parameters
    ----------
    footprint_coords : Coords2D
        Emprise 2D (peut être en lon/lat ou en mètres)
    height_m : float
        Hauteur totale de la zone [m]
    n_floors : int
        Nombre de niveaux
    roof_type : str
        "flat" | "pitched" | "shed"
    roof_pitch_deg : float
        Inclinaison de la toiture pour les toitures pentues [°]
    is_ground_floor : bool
        True si la zone est au contact du sol (RDC)
    shared_edges : list | None
        Arêtes mitoyennes avec d'autres zones (issues de find_shared_edges)

    Returns
    -------
    ZoneGeometry
    """
    # Aires et périmètre
    floor_area = polygon_area_m2(footprint_coords)
    perimeter  = polygon_perimeter_m(footprint_coords)

    # Toiture
    if roof_type == "flat":
        roof_area = floor_area
    elif roof_type in ("pitched", "gable"):
        # Surface de toiture = emprise / cos(pente) × 2 versants
        roof_area = floor_area / np.cos(np.radians(roof_pitch_deg)) * 1.0
        # Note : ×1.0 car on suppose 2 versants mais chaque versant = emprise/2 / cos
        # → surface totale = emprise × 1/cos(pitch) (identité géométrique)
    elif roof_type == "shed":
        roof_area = floor_area / np.cos(np.radians(roof_pitch_deg))
    else:
        roof_area = floor_area

    # Volume
    volume = floor_area * height_m

    # Murs extérieurs (avec détection des mitoyens si fourni)
    ext_flags = None
    if shared_edges:
        ext_flags = compute_exterior_flags(footprint_coords, shared_edges)
    wall_segments = extract_wall_segments(footprint_coords, zone_id, ext_flags)

    # Compacité : surface enveloppe / volume
    floor_height = height_m / n_floors if n_floors > 0 else height_m
    wall_area_ext = sum(
        seg.length_m * floor_height for seg in wall_segments if seg.is_exterior
    ) * n_floors
    envelope_area = wall_area_ext + roof_area + (floor_area if is_ground_floor else 0)
    compactness = envelope_area / volume if volume > 0 else 0

    return ZoneGeometry(
        zone_id             = zone_id,
        floor_area_m2       = round(floor_area, 2),
        perimeter_m         = round(perimeter, 2),
        wall_segments       = wall_segments,
        roof_area_m2        = round(roof_area, 2),
        ground_floor_area_m2= round(floor_area if is_ground_floor else 0, 2),
        volume_m3           = round(volume, 2),
        compactness         = round(compactness, 3),
    )
