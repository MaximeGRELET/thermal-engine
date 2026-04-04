"""Tests — core.solar"""

import math
import pytest
import numpy as np
from thermal_engine.core.solar import (
    solar_declination_rad,
    equation_of_time_min,
    solar_altitude_rad,
    solar_hour_angle_rad,
    incidence_angle_cos,
    irradiance_on_tilted_surface_hay_davies,
    solar_gains_through_window,
)


# ─── solar_declination_rad ───────────────────────────────────────────────────

def test_declination_summer_solstice():
    # Solstice d'été ≈ jour 172, δ ≈ +23.45°
    dec = solar_declination_rad(172)
    assert 22 < math.degrees(dec) < 24


def test_declination_winter_solstice():
    # Solstice d'hiver ≈ jour 355, δ ≈ -23.45°
    dec = solar_declination_rad(355)
    assert -24 < math.degrees(dec) < -22


def test_declination_equinox():
    # Équinoxe de printemps ≈ jour 80, δ ≈ 0°
    dec = solar_declination_rad(80)
    assert abs(math.degrees(dec)) < 5


def test_declination_range():
    for doy in range(1, 366):
        dec = solar_declination_rad(doy)
        assert -0.42 < dec < 0.42  # ±23.45° en radians ≈ ±0.41 rad


# ─── equation_of_time_min ────────────────────────────────────────────────────

def test_eot_range():
    # L'équation du temps est toujours entre -17 et +17 minutes
    for doy in range(1, 366):
        eot = equation_of_time_min(doy)
        assert -17 < eot < 17


# ─── solar_altitude_rad ──────────────────────────────────────────────────────

def test_altitude_negative_at_night():
    # Angle horaire ω=π (minuit solaire), latitude quelconque → altitude négative
    lat = math.radians(45.0)
    dec = solar_declination_rad(172)  # été
    alt = solar_altitude_rad(lat, dec, math.pi)
    assert alt < 0


def test_altitude_positive_at_noon():
    # Midi solaire (ω=0), été, latitude 45° → altitude positive
    lat = math.radians(45.0)
    dec = solar_declination_rad(172)
    alt = solar_altitude_rad(lat, dec, 0.0)
    assert alt > 0


def test_altitude_max_at_noon():
    lat = math.radians(45.0)
    dec = solar_declination_rad(172)
    alt_noon    = solar_altitude_rad(lat, dec, 0.0)
    alt_morning = solar_altitude_rad(lat, dec, math.radians(-60))
    assert alt_noon > alt_morning


# ─── incidence_angle_cos ─────────────────────────────────────────────────────

def test_incidence_horizontal_surface():
    # Surface horizontale (tilt=0) → cos(θ) = sin(altitude)
    lat_r = math.radians(45.0)
    dec   = solar_declination_rad(172)
    omega = 0.0  # midi

    alt = solar_altitude_rad(lat_r, dec, omega)
    cos_theta = incidence_angle_cos(
        tilt_rad           = 0.0,
        surface_azimuth_rad= 0.0,
        solar_altitude_rad_arr = np.array([alt]),
        solar_azimuth_rad_arr  = np.array([0.0]),
    )
    expected = math.sin(alt)
    assert abs(float(cos_theta[0]) - expected) < 0.05


def test_incidence_clipped_positive():
    # cos(θ) doit être >= 0 (surface ne reçoit pas de rayonnement négatif)
    cos_theta = incidence_angle_cos(
        tilt_rad=math.radians(90),
        surface_azimuth_rad=0.0,
        solar_altitude_rad_arr=np.array([-0.5]),   # nuit
        solar_azimuth_rad_arr=np.array([0.0]),
    )
    assert float(cos_theta[0]) >= 0.0


# ─── irradiance_on_tilted_surface_hay_davies ─────────────────────────────────

def test_irradiance_zero_at_night():
    irr = irradiance_on_tilted_surface_hay_davies(
        ghi=np.array([0.0]), dhi=np.array([0.0]), dni=np.array([0.0]),
        tilt_deg=30.0, surface_azimuth_deg=180.0,
        solar_altitude_rad=np.array([-0.1]),
        solar_azimuth_rad=np.array([0.0]),
        doy_array=np.array([172.0]),
    )
    assert float(irr[0]) == 0.0


def test_irradiance_positive_midday():
    irr = irradiance_on_tilted_surface_hay_davies(
        ghi=np.array([800.0]), dhi=np.array([150.0]), dni=np.array([700.0]),
        tilt_deg=30.0, surface_azimuth_deg=180.0,
        solar_altitude_rad=np.array([math.radians(50)]),
        solar_azimuth_rad=np.array([0.0]),
        doy_array=np.array([172.0]),
    )
    assert float(irr[0]) > 0.0


def test_irradiance_non_negative():
    # Pas d'irradiance négative dans tous les cas
    n = 100
    ghi = np.random.uniform(0, 1000, n)
    dhi = ghi * 0.3
    dni = ghi * 0.7
    irr = irradiance_on_tilted_surface_hay_davies(
        ghi=ghi, dhi=dhi, dni=dni,
        tilt_deg=45.0, surface_azimuth_deg=180.0,
        solar_altitude_rad=np.random.uniform(-0.5, 1.2, n),
        solar_azimuth_rad=np.random.uniform(-1.5, 1.5, n),
        doy_array=np.full(n, 172.0),
    )
    assert np.all(irr >= 0.0)


# ─── solar_gains_through_window ──────────────────────────────────────────────

def test_solar_gains_proportional_to_area():
    irr = np.array([500.0])
    g1 = solar_gains_through_window(irr, area_m2=1.0, g_value=0.6)
    g2 = solar_gains_through_window(irr, area_m2=2.0, g_value=0.6)
    assert abs(float(g2[0]) / float(g1[0]) - 2.0) < 1e-9


def test_solar_gains_zero_irradiance():
    gains = solar_gains_through_window(np.array([0.0]), area_m2=2.0, g_value=0.6)
    assert float(gains[0]) == 0.0


def test_solar_gains_shading_reduces():
    irr = np.array([500.0])
    g_full   = solar_gains_through_window(irr, 2.0, 0.6, shading_factor=1.0)
    g_shaded = solar_gains_through_window(irr, 2.0, 0.6, shading_factor=0.5)
    assert float(g_shaded[0]) < float(g_full[0])


def test_solar_gains_formula():
    irr = np.array([600.0])
    gains = solar_gains_through_window(irr, area_m2=3.0, g_value=0.5,
                                        frame_factor=0.8, shading_factor=0.9)
    expected = 600.0 * 3.0 * 0.5 * 0.8 * 0.9
    assert abs(float(gains[0]) - expected) < 1e-6
