"""
Plotting.

refs:
- scatter https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html
- marker https://plotly.com/python/marker-style
"""

import enum
import itertools
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


class PlotlySubplot:
    # TODO: add domain or col/row
    # TODO: set reference to mention in added traces/data
    def __init__(self, domain=None):
        self.data = []
        self.layout = {}

    def add(self, go):
        self.data.append(go)

    def add_ohlc(
        self,
        data_df,
        col_open="Open",
        col_high="High",
        col_low="Low",
        col_close="Close",
        line_width=0.7,
        name="ohlc",
        skip_no_data=True,
    ):
        assert type(data_df.index) == pd.DatetimeIndex

        self.data.append(
            go.Candlestick(
                x=data_df.index,
                open=data_df[col_open],
                high=data_df[col_high],
                low=data_df[col_low],
                close=data_df[col_close],
                name=name,
                line=dict(width=line_width),
            ),
        )

        # TODO: make generic to target related x,y
        if skip_no_data:
            self.layout["xaxis"] = dict(
                type="category",
                rangeslider=dict(visible=False),
                anchor="y",
            )


class PlotlyPlot:
    def __init__(self, subplots: list[PlotlySubplot]):
        data = list(itertools.chain.from_iterable(s.data for s in subplots))
        layout = dict(itertools.chain.from_iterable(s.layout.items() for s in subplots))

        self.fig = go.Figure(
            data=data,
            layout=layout,
        )

    def show(self):
        self.fig.show()

    def to_html(self, filepath):
        self.fig.write_html(filepath)
