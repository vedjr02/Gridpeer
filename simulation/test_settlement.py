"""Tests for grid-baseline settlement — the P2P-versus-grid-only comparison."""

from datetime import datetime

import pytest

from shared.schemas import AgentDecision, AgentRole, HouseholdProfile, OrderSide
from simulation.market import clear_tick
from simulation.settlement import settle_tick

TICK = 1
TIMESTAMP = datetime(2026, 1, 1, 8, 0)
IMPORT_TARIFF = 0.25
EXPORT_TARIFF = 0.07


def profile(household_id: str) -> HouseholdProfile:
    """A household on the standard tariffs these tests use."""
    return HouseholdProfile(
        household_id=household_id,
        role=AgentRole.PROSUMER,
        has_solar=True,
        solar_capacity_kw=3.0,
        battery_capacity_kwh=5.0,
        grid_export_tariff_eur_per_kwh=EXPORT_TARIFF,
        grid_import_tariff_eur_per_kwh=IMPORT_TARIFF,
    )


def order(
    household_id: str,
    side: OrderSide,
    quantity_kwh: float,
    limit_price_eur_per_kwh: float,
) -> AgentDecision:
    return AgentDecision(
        household_id=household_id,
        tick=TICK,
        side=side,
        quantity_kwh=quantity_kwh,
        limit_price_eur_per_kwh=limit_price_eur_per_kwh,
        strategy_name="test_fixture",
    )


def test_a_matched_trade_saves_both_sides_money():
    """The whole thesis of the project, in one tick, calculated by hand.

    hh_sell offers 2.0 kWh from 0.10, hh_buy will pay up to 0.20, so they clear at
    the 0.15 midpoint. Against the grid the buyer would have paid 0.25/kWh and the
    seller been paid only 0.07/kWh, so:

        buyer: pays 2.0 * 0.15 = 0.30 instead of 2.0 * 0.25 = 0.50  -> saves 0.20
        seller: earns 2.0 * 0.15 = 0.30 instead of 2.0 * 0.07 = 0.14 -> gains 0.16

    Both sides land inside the import/export tariff gap. That gap is the entire
    margin the marketplace exists to capture.
    """
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=2.0, limit_price_eur_per_kwh=0.20),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    settlements = settle_tick(
        state, {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}, orders
    )

    buyer = settlements["hh_buy"]
    assert buyer.p2p_cost_eur == pytest.approx(0.30)
    assert buyer.grid_only_cost_eur == pytest.approx(0.50)
    assert buyer.savings_eur == pytest.approx(0.20)
    assert buyer.traded_kwh == 2.0
    assert buyer.grid_fallback_kwh == 0.0

    seller = settlements["hh_sell"]
    assert seller.p2p_cost_eur == pytest.approx(-0.30)  # negative cost = revenue
    assert seller.grid_only_cost_eur == pytest.approx(-0.14)
    assert seller.savings_eur == pytest.approx(0.16)


def test_an_unmatched_order_falls_back_to_the_grid_and_saves_nothing():
    """No match means no saving — P2P and grid-only must come out identical.

    An unmatched buyer still needs the energy and imports it at the import tariff.
    Reporting any saving here would inflate the headline number with trades that
    never happened.
    """
    buy = order("hh_buy", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.08)
    sell = order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.12)
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=[buy, sell])
    assert state.trades == []

    settlements = settle_tick(
        state, {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}, [buy, sell]
    )

    buyer = settlements["hh_buy"]
    assert buyer.p2p_cost_eur == pytest.approx(IMPORT_TARIFF)
    assert buyer.grid_only_cost_eur == pytest.approx(IMPORT_TARIFF)
    assert buyer.savings_eur == pytest.approx(0.0)
    assert buyer.grid_fallback_kwh == 1.0
    assert buyer.traded_kwh == 0.0

    seller = settlements["hh_sell"]
    assert seller.p2p_cost_eur == pytest.approx(-EXPORT_TARIFF)
    assert seller.savings_eur == pytest.approx(0.0)


def test_a_partial_fill_settles_the_traded_and_fallback_portions_separately():
    """3.0 kWh wanted, 1.0 kWh matched: 1.0 trades, 2.0 imports from the grid."""
    orders = [
        order("hh_buy", OrderSide.BUY, quantity_kwh=3.0, limit_price_eur_per_kwh=0.20),
        order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    settlements = settle_tick(
        state, {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}, orders
    )

    buyer = settlements["hh_buy"]
    assert buyer.traded_kwh == 1.0
    assert buyer.grid_fallback_kwh == 2.0
    # 1.0 kWh at the 0.15 midpoint + 2.0 kWh imported at 0.25.
    assert buyer.p2p_cost_eur == pytest.approx(0.15 + 0.50)
    # Grid-only: all 3.0 kWh imported at 0.25.
    assert buyer.grid_only_cost_eur == pytest.approx(0.75)
    assert buyer.savings_eur == pytest.approx(0.10)


def test_households_keep_their_own_tariffs():
    """Tariffs are per household, so settlement must not use a shared default."""
    cheap_importer = profile("hh_buy").model_copy(
        update={"grid_import_tariff_eur_per_kwh": 0.16}
    )
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    settlements = settle_tick(
        state, {"hh_buy": cheap_importer, "hh_sell": profile("hh_sell")}, orders
    )

    # This buyer's grid alternative is cheaper, so the same trade saves it less.
    assert settlements["hh_buy"].grid_only_cost_eur == pytest.approx(0.16)
    assert settlements["hh_buy"].savings_eur == pytest.approx(0.01)


def test_a_household_that_sat_out_the_tick_settles_at_zero():
    """Passing the original book keeps non-participants in the run at zero cost."""
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)
    profiles = {
        "hh_buy": profile("hh_buy"),
        "hh_sell": profile("hh_sell"),
        "hh_quiet": profile("hh_quiet"),
    }

    settlements = settle_tick(state, profiles, orders)

    assert "hh_quiet" not in settlements  # it submitted no order at all

    quiet_order = order("hh_quiet", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.01)
    with_quiet = settle_tick(state, profiles, [*orders, quiet_order])
    assert with_quiet["hh_quiet"].p2p_cost_eur == 0.0
    assert with_quiet["hh_quiet"].savings_eur == 0.0


def test_settling_without_a_profile_is_an_error():
    """A missing profile means missing tariffs — guessing them would corrupt the result."""
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    with pytest.raises(KeyError, match="hh_sell"):
        settle_tick(state, {"hh_buy": profile("hh_buy")}, orders)


def test_savings_never_exceed_the_tariff_gap():
    """Sanity bound: a trade can only ever capture the import/export spread.

    If this ever fails, the settlement is inventing money — the most damaging
    possible bug in this project, because it would make the headline savings
    figure indefensible.
    """
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=EXPORT_TARIFF),
        order("hh_buy", OrderSide.BUY, quantity_kwh=2.0, limit_price_eur_per_kwh=IMPORT_TARIFF),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    settlements = settle_tick(
        state, {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}, orders
    )

    gap_eur = 2.0 * (IMPORT_TARIFF - EXPORT_TARIFF)
    total_savings = sum(s.savings_eur for s in settlements.values())
    assert total_savings == pytest.approx(gap_eur)
    assert all(s.savings_eur >= 0 for s in settlements.values())


# ---------------------------------------------------------------------------
# Settling against meters (actual_net_position_kwh)
# ---------------------------------------------------------------------------


def test_selling_energy_you_do_not_have_cannot_make_money():
    """The exploit metered settlement exists to close, calculated by hand.

    hh_sell has no surplus at all but offers 50 kWh from 0.20; hh_buy needs only
    1 kWh but bids for 50 up to 0.24. They clear 50 kWh at the 0.22 midpoint (EUR 11).
    Settled on orders alone, hh_sell would book EUR 7.50 of "savings" for energy it
    never generated. Against the meters:

        hh_sell: earns 11.00, must import the 50 kWh it did not have: 50 * 0.25 = 12.50
                 p2p 1.50 vs grid-only 0.00                              -> saves -1.50
        hh_buy:  pays 11.00, exports the 49 kWh it did not use: 49 * 0.07 = 3.43
                 p2p 7.57 vs grid-only 1 * 0.25 = 0.25                   -> saves -7.32
    """
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=50.0, limit_price_eur_per_kwh=0.20),
        order("hh_buy", OrderSide.BUY, quantity_kwh=50.0, limit_price_eur_per_kwh=0.24),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    settlements = settle_tick(
        state,
        {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")},
        actual_net_position_kwh={"hh_sell": 0.0, "hh_buy": -1.0},
    )

    seller = settlements["hh_sell"]
    assert seller.p2p_cost_eur == pytest.approx(1.50)
    assert seller.grid_only_cost_eur == pytest.approx(0.0)
    assert seller.savings_eur == pytest.approx(-1.50)
    assert seller.grid_fallback_kwh == pytest.approx(50.0)

    buyer = settlements["hh_buy"]
    assert buyer.p2p_cost_eur == pytest.approx(7.57)
    assert buyer.grid_only_cost_eur == pytest.approx(0.25)
    assert buyer.savings_eur == pytest.approx(-7.32)


@pytest.mark.parametrize(
    ("orders", "actual_net_position_kwh"),
    [
        pytest.param(
            [
                order("hh_sell", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
                order("hh_buy", OrderSide.BUY, quantity_kwh=2.0, limit_price_eur_per_kwh=0.20),
            ],
            {"hh_sell": 2.0, "hh_buy": -2.0},
            id="full-match",
        ),
        pytest.param(
            [
                order("hh_buy", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.08),
                order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.12),
            ],
            {"hh_sell": 1.0, "hh_buy": -1.0},
            id="no-match",
        ),
        pytest.param(
            [
                order("hh_buy", OrderSide.BUY, quantity_kwh=3.0, limit_price_eur_per_kwh=0.20),
                order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
            ],
            {"hh_sell": 1.0, "hh_buy": -3.0},
            id="partial-fill",
        ),
    ],
)
def test_when_orders_match_the_meters_both_settlements_agree(orders, actual_net_position_kwh):
    """Metered settlement changes nothing for an honest, perfectly forecast book."""
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)
    profiles = {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}

    by_orders = settle_tick(state, profiles, orders)
    by_meters = settle_tick(state, profiles, actual_net_position_kwh=actual_net_position_kwh)

    assert by_meters.keys() == by_orders.keys()
    for household_id, expected in by_orders.items():
        actual = by_meters[household_id]
        assert actual.traded_kwh == pytest.approx(expected.traded_kwh)
        assert actual.grid_fallback_kwh == pytest.approx(expected.grid_fallback_kwh)
        assert actual.p2p_cost_eur == pytest.approx(expected.p2p_cost_eur)
        assert actual.grid_only_cost_eur == pytest.approx(expected.grid_only_cost_eur)


def test_forecast_error_is_settled_at_grid_tariffs():
    """Both sides traded 2.0 kWh at 0.15, but the forecasts were off.

    hh_sell generated only 1.5 kWh, so it imports the 0.5 kWh shortfall:
        p2p -0.30 + 0.5 * 0.25 = -0.175 vs grid-only -1.5 * 0.07 = -0.105  -> saves 0.07
    hh_buy actually needed 2.5 kWh, so it imports the extra 0.5 kWh:
        p2p  0.30 + 0.5 * 0.25 =  0.425 vs grid-only  2.5 * 0.25 =  0.625  -> saves 0.20
    """
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=2.0, limit_price_eur_per_kwh=0.20),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)

    settlements = settle_tick(
        state,
        {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")},
        actual_net_position_kwh={"hh_sell": 1.5, "hh_buy": -2.5},
    )

    seller = settlements["hh_sell"]
    assert seller.grid_fallback_kwh == pytest.approx(0.5)
    assert seller.p2p_cost_eur == pytest.approx(-0.175)
    assert seller.savings_eur == pytest.approx(0.07)

    buyer = settlements["hh_buy"]
    assert buyer.grid_fallback_kwh == pytest.approx(0.5)
    assert buyer.p2p_cost_eur == pytest.approx(0.425)
    assert buyer.savings_eur == pytest.approx(0.20)


def test_a_metered_household_that_did_not_trade_settles_at_the_grid():
    """No order does not mean no energy: it still imported, and saved nothing."""
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=[])

    settlements = settle_tick(
        state, {"hh_quiet": profile("hh_quiet")}, actual_net_position_kwh={"hh_quiet": -1.0}
    )

    quiet = settlements["hh_quiet"]
    assert quiet.traded_kwh == 0.0
    assert quiet.p2p_cost_eur == pytest.approx(IMPORT_TARIFF)
    assert quiet.grid_only_cost_eur == pytest.approx(IMPORT_TARIFF)
    assert quiet.savings_eur == pytest.approx(0.0)


def test_a_household_that_traded_without_a_meter_reading_is_an_error():
    """Settling a trader with no metered position would silently trust its order."""
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=1.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=1.0, limit_price_eur_per_kwh=0.20),
    ]
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders)
    profiles = {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}

    with pytest.raises(KeyError, match="hh_sell"):
        settle_tick(state, profiles, actual_net_position_kwh={"hh_buy": -1.0})


def test_a_metered_household_without_a_profile_is_an_error():
    """Metered settlement needs tariffs just as much as order-based settlement does."""
    state = clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=[])

    with pytest.raises(KeyError, match="hh_quiet"):
        settle_tick(state, {}, actual_net_position_kwh={"hh_quiet": -1.0})


# ---------------------------------------------------------------------------
# Network charges on peer-to-peer trades
# ---------------------------------------------------------------------------


def _two_kwh_trade():
    orders = [
        order("hh_sell", OrderSide.SELL, quantity_kwh=2.0, limit_price_eur_per_kwh=0.10),
        order("hh_buy", OrderSide.BUY, quantity_kwh=2.0, limit_price_eur_per_kwh=0.20),
    ]
    profiles = {"hh_buy": profile("hh_buy"), "hh_sell": profile("hh_sell")}
    return clear_tick(tick=TICK, timestamp=TIMESTAMP, orders=orders), profiles, orders


@pytest.mark.parametrize("metered", [False, True])
def test_network_charge_is_paid_by_the_buyer_by_default(metered):
    """Hand-worked: 2.0 kWh at 0.15, charge 0.04/kWh.

    The buyer pays 0.30 + 0.08 = 0.38 instead of 0.50: saves 0.12 (was 0.20).
    The seller is untouched.
    """
    state, profiles, orders = _two_kwh_trade()
    meters = {"hh_buy": -2.0, "hh_sell": 2.0} if metered else None
    s = settle_tick(state, profiles, orders, actual_net_position_kwh=meters,
                    network_charge_eur_per_kwh=0.04)
    assert s["hh_buy"].p2p_cost_eur == pytest.approx(0.38)
    assert s["hh_buy"].savings_eur == pytest.approx(0.12)
    assert s["hh_sell"].p2p_cost_eur == pytest.approx(-0.30)


@pytest.mark.parametrize("metered", [False, True])
def test_network_charge_can_be_split_with_the_seller(metered):
    state, profiles, orders = _two_kwh_trade()
    meters = {"hh_buy": -2.0, "hh_sell": 2.0} if metered else None
    s = settle_tick(state, profiles, orders, actual_net_position_kwh=meters,
                    network_charge_eur_per_kwh=0.04, network_charge_seller_share=0.5)
    assert s["hh_buy"].p2p_cost_eur == pytest.approx(0.30 + 0.04)
    assert s["hh_sell"].p2p_cost_eur == pytest.approx(-0.30 + 0.04)


def test_a_charge_equal_to_the_tariff_gap_wipes_out_the_saving():
    """Break-even: each traded kWh saves the community (import - export) - charge."""
    state, profiles, orders = _two_kwh_trade()
    gap = IMPORT_TARIFF - EXPORT_TARIFF
    s = settle_tick(state, profiles, orders, network_charge_eur_per_kwh=gap)
    assert sum(x.savings_eur for x in s.values()) == pytest.approx(0.0, abs=1e-12)


def test_invalid_network_charges_are_rejected():
    state, profiles, orders = _two_kwh_trade()
    with pytest.raises(ValueError, match="network_charge_eur_per_kwh"):
        settle_tick(state, profiles, orders, network_charge_eur_per_kwh=-0.01)
    with pytest.raises(ValueError, match="seller_share"):
        settle_tick(state, profiles, orders, network_charge_seller_share=1.5)
