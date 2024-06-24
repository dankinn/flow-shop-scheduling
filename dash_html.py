"""
This file is forked from apps/dash-clinical-analytics/app.py under the following license

MIT License

Copyright (c) 2019 Plotly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifications are licensed under

Apache License, Version 2.0
(see ./LICENSE for details)

"""

from __future__ import annotations

from src.job_shop_scheduler import HybridSamplerType, SamplerType
from dash import dcc, html

from app_configs import (
    CLASSICAL_TAB_LABEL,
    DESCRIPTION,
    DWAVE_TAB_LABEL,
    SHOW_CQM,
    MAIN_HEADER,
    SCENARIOS,
    SOLVER_TIME,
    THEME_COLOR_SECONDARY,
    THUMBNAIL
)

SAMPLER_TYPES = {SamplerType.HYBRID: "Quantum Hybrid" if SHOW_CQM else "Quantum Hybrid (NL)", SamplerType.HIGHS: "Classical (HiGHS)"}
HYBRID_SAMPLER_TYPES = {HybridSamplerType.NL: "NL", HybridSamplerType.CQM: "CQM"}


def description_card():
    """A Div containing dashboard title & descriptions."""
    return html.Div(
        id="description-card",
        children=[html.H1(MAIN_HEADER), html.P(DESCRIPTION)],
    )


def dropdown(label: str, id: str, options: list, wrapper_id: str = "", wrapper_class_name: str = "") -> html.Div:
    """Slider element for value selection."""

    return html.Div(
        id = wrapper_id,
        className = wrapper_class_name,
        children=[
            html.Label(label),
            dcc.Dropdown(
                id=id,
                options=options,
                value=options[0]["value"],
                clearable=False,
                searchable=False,
            ),
        ],
    )


def checklist(label: str, id: str, options: list) -> html.Div:
    """Slider element for value selection."""
    return html.Div(
        children=[
            html.Label(label),
            dcc.Checklist(
                id=id,
                options=options,
                value=[options[0]["value"]],
            ),
        ],
    )


def generate_solution_tab(label: str, title: str, tab: str, index: int) -> dcc.Tab:
    """Generates solution tab containing, solution graphs, sort functionality, and
    problem details dropdown.

    Returns:
        dcc.Tab: A Tab containing the solution graph and problem details.
    """
    return dcc.Tab(
        label=label,
        id=f"{tab}-tab",
        className="tab",
        value=f"{tab}-tab",
        disabled=True,
        children=[
            html.Div(
                className="solution-card",
                children=[
                    html.Div(
                        className="gantt-chart-card",
                        children=[
                            html.Div(
                                className="gantt-heading",
                                children=[
                                    html.H3(
                                        title,
                                        className="gantt-title",
                                    ),
                                    html.Button(
                                        id={"type": "sort-button", "index": index},
                                        className="sort-button",
                                        children="Sort by start time",
                                        n_clicks=0
                                    ),
                                ],
                            ),
                            html.Div(
                                className="graph-wrapper",
                                children=[
                                    html.Div(
                                        id={"type": "gantt-chart-visible-wrapper", "index": index},
                                        children=[
                                            dcc.Graph(
                                                id={"type": "gantt-chart-jobsort", "index": index},
                                                responsive=True,
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        id={"type": "gantt-chart-hidden-wrapper", "index": index},
                                        className="display-none",
                                        children=[
                                            dcc.Graph(
                                                id={"type": "gantt-chart-startsort", "index": index},
                                                responsive=True,
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        [
                            html.Hr(),
                            html.Div(
                                [
                                    problem_details(tab, index),
                                    html.H5(id=f"{tab}-stats-makespan"),
                                ],
                                className="problem-details"
                            ),
                        ],
                        className="problem-details-parent",
                    ),
                ],
            ),
        ],
    ),


def generate_control_card() -> html.Div:
    """Generates the control card for the dashboard.

    Contains the dropdowns for selecting the scenario, model, and solver.

    Returns:
        html.Div: A Div containing the dropdowns for selecting the scenario,
        model, and solver.
    """

    scenario_options = [{"label": scenario, "value": scenario} for scenario in SCENARIOS]
    sampler_options = [
        {"label": label, "value": sampler_type.value}
        for sampler_type, label in SAMPLER_TYPES.items()
    ]
    hybrid_sampler_options = [
        {"label": label, "value": hybrid_sampler_type.value}
        for hybrid_sampler_type, label in HYBRID_SAMPLER_TYPES.items()
    ]

    return html.Div(
        id="control-card",
        children=[
            dropdown(
                "Scenario (jobs x operations)",
                "scenario-select",
                scenario_options,
            ),
            checklist(
                "Solver (hybrid and/or classical)",
                "solver-select",
                sorted(sampler_options, key=lambda op: op["value"]),
            ),
            dropdown(
                "Quantum Hybrid Solver",
                "hybrid-select",
                sorted(hybrid_sampler_options, key=lambda op: op["value"]),
                "hybrid-select-wrapper",
                "" if SHOW_CQM else "display-none",
            ),
            html.Label("Solver Time Limit (seconds)"),
            dcc.Input(
                id="solver-time-limit",
                type="number",
                **SOLVER_TIME,
            ),
            html.Div(
                id="button-group",
                children=[
                    html.Button(id="run-button", children="Run Optimization", n_clicks=0),
                    html.Button(
                        id="cancel-button",
                        children="Cancel Optimization",
                        n_clicks=0,
                        className="display-none",
                    ),
                ],
            ),
        ],
    )


def problem_details(solver: str, index: int) -> html.Div:
    """generate the problem details section.

    Args:
        solver: Which solver tab to generate the section for. Either "dwave" or "highs"
        index: Unique element id to differentiate matching elements.

    Returns:
        html.Div: Div containing a collapsable table.
    """
    return html.Div(
        [
            html.Div(
                id={"type": "to-collapse-class", "index": index},
                className="details-collapse-wrapper collapsed",
                children=[
                    html.Button(
                        id={"type": "collapse-trigger", "index": index},
                        className="details-collapse",
                        children=[
                            html.H5("Problem Details"),
                            html.Div(className="collapse-arrow"),
                        ],
                    ),
                    html.Div(
                        className="details-to-collapse",
                        children=[
                            html.Table(
                                className="solution-stats-table",
                                children=[
                                    html.Tbody(
                                        children=[
                                            html.Tr(
                                                [
                                                    html.Td("Scenario"),
                                                    html.Td(id=f"{solver}-stats-scenario"),
                                                    html.Td("Solver"),
                                                    html.Td(id=f"{solver}-stats-solver"),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Number of Jobs"),
                                                    html.Td(id=f"{solver}-stats-jobs"),
                                                    html.Td("Solver Time Limit (s)"),
                                                    html.Td(id=f"{solver}-stats-time-limit"),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Number of Operations"),
                                                    html.Td(id=f"{solver}-stats-resources"),
                                                    html.Td("Wall Clock Time (s)"),
                                                    html.Td(id=f"{solver}-stats-wall-clock-time"),
                                                ]
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

# set the application HTML
def set_html(app):
    app.layout = html.Div(
        id="app-container",
        children=[
            dcc.Store("running-dwave"),
            dcc.Store("running-classical"),
            # Banner
            html.Div(id="banner", children=[html.Img(src=THUMBNAIL)]),
            html.Div(
                id="columns",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                [  # Fixed width Div to collapse
                                    html.Div(
                                        [  # Padding and content wrapper
                                            description_card(),
                                            generate_control_card(),
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                html.Button(
                                    id={"type": "collapse-trigger", "index": 0},
                                    className="left-column-collapse",
                                    children=[html.Div(className="collapse-arrow")],
                                ),
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        id="right-column",
                        children=[
                            dcc.Tabs(
                                id="tabs",
                                value="input-tab",
                                children=[
                                    dcc.Tab(
                                        label="Input",
                                        value="input-tab",
                                        className="tab",
                                        children=[
                                            html.Div(
                                                className="gantt-chart-card",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "Unscheduled Jobs and Operations",
                                                                className="gantt-title",
                                                            ),
                                                        ],
                                                        className="gantt-heading",
                                                    ),
                                                    dcc.Loading(
                                                        id="loading-icon-input",
                                                        parent_className="loading-graph",
                                                        type="circle",
                                                        color=THEME_COLOR_SECONDARY,
                                                        children=[
                                                            dcc.Graph(
                                                                id="unscheduled-gantt-chart",
                                                                responsive=True,
                                                                config={"displayModeBar": False},
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    # dcc.Tab seems to return a list so these need to be unpacked
                                    *generate_solution_tab(DWAVE_TAB_LABEL, "Leap Hybrid Solver", "dwave", 0),
                                    *generate_solution_tab(CLASSICAL_TAB_LABEL, "HiGHS Classical Solver", "highs", 1),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
