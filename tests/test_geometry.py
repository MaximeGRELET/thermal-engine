"""Tests — core.geometry"""

import math
import pytest
from thermal_engine.core.geometry import (
    is_geographic,
    polygon_area_m2,
    polygon_perimeter_m,
    segment_length_m,
    segment_azimuth_deg,
    orientation_label,
    extract_wall_segments,
    compute_zone_geometry,
    to_metric_coords,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

# Rectangle 10 m × 20 m en coordonnées métriques locales (non géographiques).
# On commence à 1000 pour dépasser le seuil de détection géographique (>181).
RECT_10x20 = [[1000, 1000], [1020, 1000], [1020, 1010], [1000, 1010], [1000, 1000]]

# Même rectangle en coordonnées WGS84 proches de Lyon (lat ~45.75°)
# 20 m Est  ≈ 20 / (111320 * cos(45.75°)) ≈ 0.0002530°
# 10 m Nord ≈ 10 / 111320 ≈ 0.0000899°
RECT_GEOGRAPHIC = [
    [4.854000, 45.756000],
    [4.854253, 45.756000],
    [4.854253, 45.756090],
    [4.854000, 45.756090],
    [4.854000, 45.756000],
]


# ─── is_geographic ───────────────────────────────────────────────────────────

def test_is_geographic_true():
    assert is_geographic(RECT_GEOGRAPHIC) is True


def test_is_geographic_false_large_coords():
    coords = [[1000.0, 500.0], [1020.0, 500.0], [1020.0, 510.0], [1000.0, 510.0]]
    assert is_geographic(coords) is False


def test_is_geographic_empty():
    assert is_geographic([]) is False


# ─── polygon_area_m2 ─────────────────────────────────────────────────────────

def test_polygon_area_metric_rectangle():
    area = polygon_area_m2(RECT_10x20)
    assert abs(area - 200.0) < 0.1  # 10 × 20 = 200 m²


def test_polygon_area_geographic():
    area = polygon_area_m2(RECT_GEOGRAPHIC)
    # On accepte ±5% d'erreur (projection locale)
    assert abs(area - 200.0) / 200.0 < 0.05


def test_polygon_area_square():
    square = [[1000, 1000], [1005, 1000], [1005, 1005], [1000, 1005], [1000, 1000]]
    area = polygon_area_m2(square)
    assert abs(area - 25.0) < 0.01


def test_polygon_area_triangle():
    triangle = [[1000, 1000], [1006, 1000], [1003, 1004], [1000, 1000]]
    area = polygon_area_m2(triangle)
    assert abs(area - 12.0) < 0.01  # base=6, h=4, aire=12


# ─── polygon_perimeter_m ─────────────────────────────────────────────────────

def test_perimeter_rectangle():
    perim = polygon_perimeter_m(RECT_10x20)
    assert abs(perim - 60.0) < 0.1  # 2*(10+20)=60


# ─── segment_length_m ────────────────────────────────────────────────────────

def test_segment_length_horizontal():
    assert abs(segment_length_m(0, 0, 10, 0) - 10.0) < 1e-9


def test_segment_length_diagonal():
    # 3-4-5 right triangle
    assert abs(segment_length_m(0, 0, 3, 4) - 5.0) < 1e-9


def test_segment_length_zero():
    assert segment_length_m(5, 5, 5, 5) == 0.0


# ─── segment_azimuth_deg ─────────────────────────────────────────────────────

def test_azimuth_south_facing_wall():
    # Mur horizontal allant vers l'Est (x croissant, y=0)
    # normale extérieure pointe vers le Sud → azimut 180°
    az = segment_azimuth_deg(0, 0, 10, 0)
    assert abs(az - 180.0) < 1.0


def test_azimuth_north_facing_wall():
    # Mur allant vers l'Ouest → normale vers le Nord
    az = segment_azimuth_deg(10, 0, 0, 0)
    assert abs(az - 0.0) < 1.0 or abs(az - 360.0) < 1.0


def test_azimuth_east_facing_wall():
    # Mur allant vers le Nord → normale vers l'Est
    az = segment_azimuth_deg(0, 0, 0, 10)
    assert abs(az - 90.0) < 1.0


def test_azimuth_west_facing_wall():
    # Mur allant vers le Sud → normale vers l'Ouest
    az = segment_azimuth_deg(0, 10, 0, 0)
    assert abs(az - 270.0) < 1.0


# ─── orientation_label ───────────────────────────────────────────────────────

@pytest.mark.parametrize("azimuth,expected", [
    (0,   "N"),
    (10,  "N"),
    (45,  "NE"),
    (90,  "E"),
    (135, "SE"),
    (180, "S"),
    (225, "SO"),
    (270, "O"),
    (315, "NO"),
    (350, "N"),
])
def test_orientation_label(azimuth, expected):
    assert orientation_label(azimuth) == expected


# ─── extract_wall_segments ───────────────────────────────────────────────────

def test_wall_segments_count():
    segs = extract_wall_segments(RECT_10x20, "zone_test")
    assert len(segs) == 4


def test_wall_segments_all_exterior_by_default():
    segs = extract_wall_segments(RECT_10x20, "zone_test")
    assert all(s.is_exterior for s in segs)


def test_wall_segments_lengths():
    segs = extract_wall_segments(RECT_10x20, "zone_test")
    lengths = sorted(s.length_m for s in segs)
    assert abs(lengths[0] - 10.0) < 0.5
    assert abs(lengths[1] - 10.0) < 0.5
    assert abs(lengths[2] - 20.0) < 0.5
    assert abs(lengths[3] - 20.0) < 0.5


def test_wall_segments_with_interior_flag():
    exterior = [True, False, True, True]  # 2e mur = mitoyen
    segs = extract_wall_segments(RECT_10x20, "zone_test", exterior)
    interior = [s for s in segs if not s.is_exterior]
    assert len(interior) == 1


# ─── compute_zone_geometry ───────────────────────────────────────────────────

def test_zone_geometry_area():
    geo = compute_zone_geometry("z1", RECT_10x20, height_m=3.0)
    assert abs(geo.floor_area_m2 - 200.0) < 0.5


def test_zone_geometry_volume():
    geo = compute_zone_geometry("z1", RECT_10x20, height_m=3.0)
    assert abs(geo.volume_m3 - 600.0) < 1.0


def test_zone_geometry_roof_flat():
    geo = compute_zone_geometry("z1", RECT_10x20, height_m=3.0, roof_type="flat")
    assert abs(geo.roof_area_m2 - 200.0) < 0.5


def test_zone_geometry_roof_pitched():
    pitch = 30.0
    geo = compute_zone_geometry("z1", RECT_10x20, height_m=3.0, roof_type="pitched",
                                roof_pitch_deg=pitch)
    expected = 200.0 / math.cos(math.radians(pitch))
    assert abs(geo.roof_area_m2 - expected) < 1.0


def test_zone_geometry_ground_floor_contact():
    geo = compute_zone_geometry("z1", RECT_10x20, height_m=3.0, is_ground_floor=True)
    assert abs(geo.ground_floor_area_m2 - 200.0) < 0.5


def test_zone_geometry_no_ground_contact():
    geo = compute_zone_geometry("z1", RECT_10x20, height_m=3.0, is_ground_floor=False)
    assert geo.ground_floor_area_m2 == 0.0
