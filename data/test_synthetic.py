"""Tests for the synthetic household generator.

These assert the *shape* of the series — daily energy in a plausible residential
range, peaks where peaks belong, solar only in daylight — because everything
downstream (forecast error, savings, CO2) is meaningless if the inputs aren't
roughly household-shaped.
"""

from data.synthetic import (
    default_profiles,
    generate_household_series,
    kw_readings_to_kwh,
    ticks_per_day,
)
from shared.schemas import AgentRole


def test_default_profiles_are_a_mixed_cohort() -> None:
    """A four-household cohort covers prosumers, a consumer and a producer."""
    profiles = default_profiles(4)
    assert len(profiles) == 4
    assert {p.role for p in profiles} == {
        AgentRole.PROSUMER,
        AgentRole.PURE_CONSUMER,
        AgentRole.PURE_PRODUCER,
    }
    assert len({p.household_id for p in profiles}) == 4
    consumer = next(p for p in profiles if p.role == AgentRole.PURE_CONSUMER)
    assert not consumer.has_solar and consumer.solar_capacity_kw == 0.0


def test_import_tariff_exceeds_export_tariff() -> None:
    """The import/export spread is the reason P2P trading is worth anything."""
    for profile in default_profiles(4):
        assert profile.grid_import_tariff_eur_per_kwh > profile.grid_export_tariff_eur_per_kwh


def test_series_length_matches_requested_days() -> None:
    """Both series are exactly days * ticks_per_day long."""
    profile = default_profiles(1)[0]
    demand_kwh, solar_kwh = generate_household_series(profile, days=3)
    assert len(demand_kwh) == len(solar_kwh) == 3 * ticks_per_day()


def test_series_are_deterministic_for_a_given_seed() -> None:
    """Same profile + seed reproduces the run exactly; a different seed does not."""
    profile = default_profiles(1)[0]
    first = generate_household_series(profile, days=2, seed=11)
    assert first == generate_household_series(profile, days=2, seed=11)
    assert first != generate_household_series(profile, days=2, seed=12)


def test_households_differ_from_each_other_under_one_seed() -> None:
    """One seed still yields distinct load shapes per household, not four clones."""
    profiles = default_profiles(4)
    demands = [tuple(generate_household_series(p, days=2, seed=5)[0]) for p in profiles]
    assert len(set(demands)) == len(demands)


def test_daily_demand_is_in_a_plausible_residential_range() -> None:
    """A household draws single-digit-to-low-teens kWh a day, not 1 or 100."""
    profile = next(p for p in default_profiles(4) if p.role == AgentRole.PURE_CONSUMER)
    demand_kwh, _ = generate_household_series(profile, days=14, seed=3)
    daily_kwh = sum(demand_kwh) / 14
    assert 5.0 < daily_kwh < 20.0


def test_demand_peaks_in_the_evening_not_overnight() -> None:
    """Evening demand clearly exceeds the overnight base — the shape agents trade against."""
    profile = default_profiles(1)[0]
    demand_kwh, _ = generate_household_series(profile, days=7, seed=4)
    per_day = ticks_per_day()
    overnight = [demand_kwh[d * per_day + t] for d in range(7) for t in range(0, 8)]
    evening = [demand_kwh[d * per_day + t] for d in range(7) for t in range(36, 42)]
    assert sum(evening) / len(evening) > 2.0 * sum(overnight) / len(overnight)


def test_solar_is_zero_overnight_and_positive_at_midday() -> None:
    """Generation follows daylight: nothing at 02:00, real output around solar noon."""
    profile = next(p for p in default_profiles(4) if p.has_solar)
    _, solar_kwh = generate_household_series(profile, days=5, seed=6)
    per_day = ticks_per_day()
    assert all(solar_kwh[d * per_day + t] == 0.0 for d in range(5) for t in range(0, 10))
    assert all(solar_kwh[d * per_day + 26] > 0.0 for d in range(5))


def test_no_solar_household_generates_nothing() -> None:
    """A household without panels never produces a non-zero generation reading."""
    profile = next(p for p in default_profiles(4) if not p.has_solar)
    _, solar_kwh = generate_household_series(profile, days=3, seed=8)
    assert set(solar_kwh) == {0.0}


def test_larger_array_generates_more() -> None:
    """Daily yield scales with panel capacity."""
    profiles = default_profiles(4)
    small = next(p for p in profiles if p.solar_capacity_kw == 3.0)
    large = next(p for p in profiles if p.solar_capacity_kw == 6.0)
    _, small_solar = generate_household_series(small, days=14, seed=9)
    _, large_solar = generate_household_series(large, days=14, seed=9)
    assert sum(large_solar) > 1.5 * sum(small_solar)


def test_all_readings_are_non_negative() -> None:
    """Noise never pushes a reading below zero — negative kWh would corrupt clearing."""
    for profile in default_profiles(4):
        demand_kwh, solar_kwh = generate_household_series(profile, days=7, seed=10)
        assert min(demand_kwh) >= 0.0
        assert min(solar_kwh) >= 0.0


def test_kw_readings_convert_to_kwh_by_interval_length() -> None:
    """CER publishes average kW per half-hour; energy is that divided by two."""
    assert kw_readings_to_kwh([2.0, 1.0, 0.0]) == [1.0, 0.5, 0.0]
    assert kw_readings_to_kwh([2.0], tick_minutes=60) == [2.0]


def test_ticks_per_day_rejects_non_divisor_resolutions() -> None:
    """A tick length that doesn't divide the day would silently misalign every series."""
    assert ticks_per_day(30) == 48
    assert ticks_per_day(15) == 96
    for bad in (0, -30, 7):
        try:
            ticks_per_day(bad)
        except ValueError:
            continue
        raise AssertionError(f"ticks_per_day({bad}) should have raised")
