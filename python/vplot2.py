"""
Plotting.

refs:
- scatter https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html
- marker https://plotly.com/python/marker-style
"""

import copy
import enum
import typing
import pandas as pd
import plotly.graph_objects as go


class Color(enum.Enum):
    # ref: https://matplotlib.org/stable/gallery/color/named_colors.html
    RED = "orangered"
    PINK = "deeppink"
    ORANGE = "darkorange"
    YELLOW = "gold"
    GREEN = "forestgreen"
    AQUAMARINE = "mediumaquamarine"
    BLUE = "steelblue"
    VIOLET = "blueviolet"
    LIGHT_GREY = "lightgrey"
    GREY = "grey"
    BLACK = "black"


class Figure:
    def __init__(self):
        self.subplots = []
        self.data = []
        self.layout = {}

    def add_subplot(
        self,
        x_domain: typing.Tuple[float, float] = None,
        y_domain: typing.Tuple[float, float] = None,
        x_share_with: "Subplot" = None,
        x_skip_no_data=False,
        rangeslider_visible=False,
    ) -> "Subplot":
        new_subplot = Subplot(self, x_share_with=x_share_with)
        self.subplots.append(new_subplot)

        if x_domain:
            self.layout[new_subplot.xaxis_layout_id] = dict(domain=x_domain)
        if y_domain:
            self.layout[new_subplot.yaxis_layout_id] = dict(domain=y_domain)

        assert not (x_skip_no_data and x_share_with)
        if not x_share_with:
            self.layout[new_subplot.xaxis_layout_id] = dict(
                rangeslider=dict(visible=rangeslider_visible),
                anchor=new_subplot.yaxis_id,
            )
        if x_skip_no_data:
            self.layout[new_subplot.xaxis_layout_id]["type"] = "category"

        return new_subplot

    def to_go(self) -> go.Figure:
        fig = go.Figure(
            data=self.data,
            layout=self.layout,
        )
        return fig

    def show(self):
        self.to_go().show()

    def to_html(self, filepath):
        self.to_go().write_html(filepath)


class Subplot:
    # TODO: add domain or col/row
    # TODO: set reference to mention in added traces/data
    def __init__(self, fig: Figure, x_share_with: "Subplot" = None):
        def _new_axis_id(fig: Figure, axis=typing.Literal["x", "y"]) -> str:
            """
            Create new axis id for subplot.
            """
            assert axis in ["x", "y"]
            if len(fig.subplots) == 0:
                return axis
            return f"{axis}{len(fig.subplots)+1}"

        def _new_axis_layout_id(fig: Figure, axis=typing.Literal["x", "y"]) -> str:
            """
            Create new axis layout id for subplot.
            """
            assert axis in ["x", "y"]
            if len(fig.subplots) == 0:
                return f"{axis}axis"
            return f"{axis}axis{len(fig.subplots)+1}"

        self.fig = fig

        if x_share_with:
            self.xaxis_id = x_share_with.xaxis_id
        else:
            self.xaxis_id = _new_axis_id(fig, "x")

        self.xaxis_layout_id = _new_axis_layout_id(fig, "x")
        self.yaxis_id = _new_axis_id(fig, "y")
        self.yaxis_layout_id = _new_axis_layout_id(fig, "y")

    def add(self, go):
        # copy object to avoid modifying original
        saved_go = copy.copy(go)
        saved_go.xaxis = self.xaxis_id
        saved_go.yaxis = self.yaxis_id
        self.fig.data.append(saved_go)

    def add_ohlc(
        self,
        data_df,
        col_open="Open",
        col_high="High",
        col_low="Low",
        col_close="Close",
        line_width=0.7,
        name="ohlc",
    ):
        assert type(data_df.index) == pd.DatetimeIndex

        self.fig.data.append(
            go.Candlestick(
                x=data_df.index,
                open=data_df[col_open],
                high=data_df[col_high],
                low=data_df[col_low],
                close=data_df[col_close],
                name=name,
                line=dict(width=line_width),
                xaxis=self.xaxis_id,
                yaxis=self.yaxis_id,
            )
        )
