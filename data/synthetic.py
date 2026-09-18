"""Synthetic household demand + solar generation series for the week-1 pipeline.

Shapes, not truth. Every series here is produced in exactly the form the real
data will be loaded into — half-hourly, chronological, kWh per tick — so that
replacing this module with the CER (demand) and PVGIS (solar) loaders is an
import swap in the orchestrator, not a pipeline rewrite. See `data/README.md`.

Reference units:
    - CER Smart Metering Trial publishes half-hourly *kW* readings; divide by 2
      to get kWh per 30-minute interval (`kw_readings_to_kwh` does this).
    - PVGIS publishes hourly generation; a 3 kW Irish array yields roughly
      7-8 kWh on an average day, which is what the defaults here reproduce.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from shared.schemas import AgentRole, HouseholdProfile

TICK_MINUTES = 30
MINUTES_PER_HOUR = 60

# --- Demand shape (kWh per tick, before the per-household scale factor) -------
OVERNIGHT_BASE_KWH = 0.10
MORNING_PEAK_HOUR = 7.75
MORNING_PEAK_KWH = 0.45
MORNING_PEAK_WIDTH_H = 1.0
EVENING_PEAK_HOUR = 18.75
EVENING_PEAK_KWH = 0.55
EVENING_PEAK_WIDTH_H = 1.4
WEEKEND_PEAK_SHIFT_H = 1.5  # weekend mornings start later and flatten out
WEEKEND_MORNING_DAMPING = 0.6
TICK_NOISE_FRACTION = 0.18  # relative jitter applied to each half-hour reading

# --- Solar shape (PVGIS-like, Irish latitude) --------------------------------
SOLAR_NOON_HOUR = 13.0
SOLAR_WIDTH_H = 2.6
PEAK_OUTPUT_FRACTION = 0.55  # AC output at solar noon as a fraction of nameplate kW
DAYLIGHT_START_HOUR = 6.0
DAYLIGHT_END_HOUR = 21.0
CLOUD_FACTOR_RANGE = (0.35, 1.0)  # whole-day clearness multiplier
OVERCAST_DAY_PROBABILITY = 0.25
OVERCAST_FACTOR_RANGE = (0.12, 0.35)

# --- Role scaling ------------------------------------------------------------
_ROLE_DEMAND_SCALE = {
    AgentRole.PURE_CONSUMER: 1.0,
    AgentRole.PROSUMER: 1.0,
    AgentRole.PURE_PRODUCER: 0.35,  # e.g. a barn/field array with minimal on-site load
}


def kw_readings_to_kwh(
    readings_kw: Sequence[float], tick_minutes: int = TICK_MINUTES
) -> list[float]:
    """Convert CER-style average-power readings (kW) to energy per tick (kWh)."""
    hours_per_tick = tick_minutes / MINUTES_PER_HOUR
    return [reading_kw * hours_per_tick for reading_kw in readings_kw]


def ticks_per_day(tick_minutes: int = TICK_MINUTES) -> int:
    """Number of simulation ticks in one day at the given tick resolution."""
    if tick_minutes <= 0 or (24 * MINUTES_PER_HOUR) % tick_minutes != 0:
        raise ValueError("tick_minutes must be a positive divisor of 1440")
    return 24 * MINUTES_PER_HOUR // tick_minutes


def _bell(hour: float, centre_h: float, width_h: float) -> float:
    """Unit-height Gaussian in hours-of-day — the building block of both shapes."""
    return math.exp(-((hour - centre_h) ** 2) / (2.0 * width_h**2))


def _household_seed(household_id: str, seed: int) -> int:
    """Stable per-household seed so each household differs but the set is reproducible."""
    return (seed * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(household_id))) % (2**31)


def _demand_tick_kwh(hour: float, is_weekend: bool) -> float:
    """Base demand for one tick (kWh), before household scaling and noise."""
    morning_centre = MORNING_PEAK_HOUR + (WEEKEND_PEAK_SHIFT_H if is_weekend else 0.0)
    morning_height = MORNING_PEAK_KWH * (WEEKEND_MORNING_DAMPING if is_weekend else 1.0)
    morning = morning_height * _bell(hour, morning_centre, MORNING_PEAK_WIDTH_H)
    evening = EVENING_PEAK_KWH * _bell(hour, EVENING_PEAK_HOUR, EVENING_PEAK_WIDTH_H)
    return OVERNIGHT_BASE_KWH + morning + evening


def _solar_tick_kwh(hour: float, solar_capacity_kw: float, tick_minutes: int) -> float:
    """Clear-sky generation for one tick (kWh), before the day's cloud multiplier."""
    if solar_capacity_kw <= 0 or not DAYLIGHT_START_HOUR <= hour <= DAYLIGHT_END_HOUR:
        return 0.0
    hours_per_tick = tick_minutes / MINUTES_PER_HOUR
    peak_kw = solar_capacity_kw * PEAK_OUTPUT_FRACTION
    return peak_kw * _bell(hour, SOLAR_NOON_HOUR, SOLAR_WIDTH_H) * hours_per_tick


def _cloud_factor(rng: random.Random) -> float:
    """Whole-day clearness multiplier in (0, 1]; occasionally a heavily overcast day."""
    if rng.random() < OVERCAST_DAY_PROBABILITY:
        return rng.uniform(*OVERCAST_FACTOR_RANGE)
    return rng.uniform(*CLOUD_FACTOR_RANGE)


def generate_household_series(
    profile: HouseholdProfile,
    days: int,
    tick_minutes: int = TICK_MINUTES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Generate one household's (demand_kwh, solar_kwh) series, chronological, per tick.

    Both lists are ``days * ticks_per_day(tick_minutes)`` long and in kWh per tick.
    Demand carries morning/evening peaks over an overnight base, damped and shifted
    at weekends; solar is a daylight bell curve scaled by the profile's panel
    capacity and knocked down by a per-day cloud factor. ``seed`` is combined with
    ``profile.household_id``, so a fixed seed reproduces the whole cohort exactly
    while each household still has its own load shape and weather.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    per_day = ticks_per_day(tick_minutes)

    rng = random.Random(_household_seed(profile.household_id, seed))
    household_scale = rng.uniform(0.75, 1.35) * _ROLE_DEMAND_SCALE[profile.role]

    demand_kwh: list[float] = []
    solar_kwh: list[float] = []
    for day in range(days):
        is_weekend = day % 7 >= 5
        cloud = _cloud_factor(rng)
        for tick_of_day in range(per_day):
            hour = tick_of_day * tick_minutes / MINUTES_PER_HOUR
            noise = 1.0 + rng.uniform(-TICK_NOISE_FRACTION, TICK_NOISE_FRACTION)
            base = _demand_tick_kwh(hour, is_weekend)
            demand_kwh.append(max(0.0, base * household_scale * noise))
            clear_sky = _solar_tick_kwh(hour, profile.solar_capacity_kw, tick_minutes)
            intraday = 1.0 + rng.uniform(-0.12, 0.12) if clear_sky > 0 else 1.0
            solar_kwh.append(max(0.0, clear_sky * cloud * intraday))
    return demand_kwh, solar_kwh


def default_profiles(num_households: int = 4) -> list[HouseholdProfile]:
    """Build a mixed cohort of household profiles: prosumers, a consumer, a producer.

    Tariffs are indicative Irish residential rates (import ~0.25 EUR/kWh, export
    ~0.07 EUR/kWh) — the gap between them is what makes P2P trading worth doing.
    """
    if num_households < 1:
        raise ValueError("num_households must be >= 1")
    roles = [
        AgentRole.PROSUMER,
        AgentRole.PURE_CONSUMER,
        AgentRole.PROSUMER,
        AgentRole.PURE_PRODUCER,
    ]
    solar_capacities_kw = {AgentRole.PROSUMER: 3.0, AgentRole.PURE_PRODUCER: 6.0}

    profiles: list[HouseholdProfile] = []
    for index in range(num_households):
        role = roles[index % len(roles)]
        solar_capacity_kw = solar_capacities_kw.get(role, 0.0)
        profiles.append(
            HouseholdProfile(
                household_id=f"hh_{index:03d}",
                role=role,
                has_solar=solar_capacity_kw > 0,
                solar_capacity_kw=solar_capacity_kw,
                battery_capacity_kwh=5.0 if solar_capacity_kw > 0 else 0.0,
                grid_export_tariff_eur_per_kwh=0.07,
                grid_import_tariff_eur_per_kwh=0.25,
            )
        )
    return profiles
