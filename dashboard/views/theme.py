"""Shared visual language for the dashboard: colour, units, and chart defaults.

One household keeps one colour everywhere it appears, so a viewer can track
"hh_002" across the savings chart and the forecast charts without re-reading a
legend. Every axis label carries its unit (kWh, EUR, EUR/kWh) because energy
trading bugs are usually unit bugs, and a mislabelled axis hides one.

Palette: the validated eight-slot categorical set (adjacent-pair CVD delta-E 9.1,
normal-vision 19.6 on a light surface). Slots are assigned in fixed order and
never cycled — past eight households the extras fold into one muted "other"
colour rather than repeating a hue and implying a false identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt

# Axis/metric unit strings. Import these rather than retyping a unit — a typo in
# a literal is how a kW axis ends up labelled kWh.
EUR = "EUR"
EUR_PER_KWH = "EUR/kWh"
KWH = "kWh"
TICK_AXIS = "Tick (half-hourly)"

#: Fixed categorical hue order. Never re-order: colour follows the household,
#: not its rank, so a filter that drops households must not repaint the rest.
_HOUSEHOLD_HUES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

#: Ninth-and-beyond households share this; see module docstring.
_OVERFLOW_HUE = "#8a8a85"

#: Non-categorical roles. The grid baseline is deliberately muted — it is the
#: reference the P2P line is read against, not a competing series.
GRID_BASELINE_HUE = "#8a8a85"
P2P_HUE = _HOUSEHOLD_HUES[0]
REFERENCE_RULE_HUE = "#52514e"


def household_colours(household_ids: list[str]) -> dict[str, str]:
    """Map each household to its fixed colour, assigned in sorted-id order.

    Pass the run's whole roster. Prefer :class:`HouseholdPalette`, which holds
    that roster so a chart drawn from a subset cannot shift anyone's colour.
    """
    ordered = sorted(set(household_ids))
    return {
        household_id: _HOUSEHOLD_HUES[i] if i < len(_HOUSEHOLD_HUES) else _OVERFLOW_HUE
        for i, household_id in enumerate(ordered)
    }


@dataclass(frozen=True)
class HouseholdPalette:
    """Colour assignment for one run, fixed once from its full household roster.

    Built from every household in the run, so a chart that happens to plot only
    some of them — a household that never traded, a filtered view — still draws
    each one in the colour it has everywhere else. Colour follows the household,
    never its position in whatever list a chart was handed.
    """

    colours: dict[str, str]

    @classmethod
    def for_run(cls, household_ids: list[str]) -> HouseholdPalette:
        """Fix the palette from every household taking part in the run."""
        return cls(colours=household_colours(household_ids))

    def scale(self) -> alt.Scale:
        """Altair colour scale pinning each household to its colour."""
        return alt.Scale(domain=list(self.colours), range=list(self.colours.values()))

    def colour(self, title: str = "Household") -> alt.Color:
        """Colour encoding for a per-household series, legend always shown."""
        return alt.Color(
            "household_id:N",
            scale=self.scale(),
            title=title,
            legend=alt.Legend(orient="right", symbolType="stroke"),
        )


def basis_colour() -> alt.Color:
    """Colour encoding for the P2P-vs-grid-baseline comparison.

    The baseline is deliberately muted: it is the reference the peer-to-peer bar
    is read against, not a competing series of equal standing.
    """
    from dashboard.views.transforms import GRID_BASIS, P2P_BASIS

    return alt.Color(
        "basis:N",
        scale=alt.Scale(domain=[GRID_BASIS, P2P_BASIS], range=[GRID_BASELINE_HUE, P2P_HUE]),
        title="Cost basis",
        legend=alt.Legend(orient="top", direction="horizontal"),
    )


def tick_axis() -> alt.X:
    """Shared x encoding: simulation tick, labelled with its resolution."""
    return alt.X("tick:Q", title=TICK_AXIS, axis=alt.Axis(grid=False))


def style_faceted(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """Styling for a faceted (column-per-household) chart.

    Faceted charts cannot carry ``configure_view`` per facet the way a single
    chart does, so the shared axis/legend styling is applied at the top level
    and the per-facet frame is stripped separately.
    """
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#52514e",
            titleColor="#52514e",
            titleFontWeight="normal",
            gridColor="#e8e7e3",
            domainColor="#d5d4cf",
            tickColor="#d5d4cf",
        )
        .configure_header(labelColor="#0b0b0b", titleColor="#52514e", titleFontWeight="normal")
        .configure_legend(labelColor="#0b0b0b", titleColor="#52514e", titleFontWeight="normal")
    )


def style(chart: alt.Chart, height: int = 280) -> alt.Chart:
    """Apply recessive grid/axis styling and a consistent height to a chart."""
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#52514e",
            titleColor="#52514e",
            titleFontWeight="normal",
            gridColor="#e8e7e3",
            domainColor="#d5d4cf",
            tickColor="#d5d4cf",
        )
        .configure_legend(labelColor="#0b0b0b", titleColor="#52514e", titleFontWeight="normal")
    )
