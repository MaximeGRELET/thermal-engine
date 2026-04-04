# Changelog

All notable changes to `thermal-engine` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.1.0] — 2026-04-04

### Added
- Physics-based thermal calculation engine for buildings
- Multi-zone building model loaded from GeoJSON FeatureCollection (2D footprint + height)
- EPW weather file parser (EnergyPlus format) → `WeatherSeries` (8760 hourly records)
- **ISO 13790** monthly quasi-static method for heating/cooling needs
- **Hourly RC simulation** (1-node implicit Euler) as alternative method
- Solar gains via **Hay-Davies** irradiance model on tilted surfaces
- Solar position using **Spencer (1971)** equations
- **ISO 6946** U-value calculation for layered wall compositions
- **ISO 13370** equivalent U-value for ground floors
- **EN ISO 14683** linear thermal bridge coefficients (Ψ)
- Automatic detection of adjacent zones (shared walls) via Shapely
- Material database — ~30 materials with λ, ρ, Cp (EN ISO 10456)
- Energy systems: gas boiler, condensing boiler, fuel oil, electric resistance,
  heat pump air/water, heat pump ground/water, wood boiler, district heating,
  solar thermal (EN ISO 9806)
- Ventilation: natural, mechanical extract, MVHR double-flux with heat recovery
- Occupation schedules: residential, office, retail, restaurant, school
- Renovation actions (immutable — returns deep copy of building):
  `InsulateWalls`, `InsulateRoof`, `InsulateFloor`, `ReplaceWindows`,
  `ReplaceHeatingSystem`, `InstallMVHR`, `AddSolarThermal`,
  `ImproveAirtightness`, `OptimiseThermalBridges`
- Pre-built standard renovation scenarios (light / intermediate / BBC retrofit)
- DPE classification (dual criterion: primary energy + CO₂, decree 2021)
- JSON-serialisable output for all results
