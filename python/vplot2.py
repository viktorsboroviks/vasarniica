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
import webcolors


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
    WHITE = "white"

    @staticmethod
    def to_rgba_str(css_color_name, alpha=1.0) -> str:
        if css_color_name is None:
            return "rgba(0,0,0,0)"
        r, g, b = webcolors.name_to_rgb(css_color_name)
        return f"rgba({r},{g},{b},{alpha})"


DEFAULT_LINE_WIDTH = 0.7


class PlotlyFigure:
    def __init__(
        self,
        font_size: int = 8,
        template: str = "simple_white",
        showlegend: bool = False,
        dry_run: bool = False,
    ):
        """
        template - set manually because of certain visual bugs
                   in plotly default templates
        """
        self.dry_run = dry_run

        self.subplots = []
        self.data = []
        self.layout = {}
        self.layout["shapes"] = []
        self.layout["annotations"] = []
        self.layout["font"] = {}
        self.layout["font"]["size"] = font_size

        self.template = template
        if template == "simple_white":
            self.layout["font"]["color"] = "black"
            self.layout["paper_bgcolor"] = "white"
            self.layout["plot_bgcolor"] = "white"

        self.layout["showlegend"] = showlegend

        # hover over all subplots at once
        self.layout["hovermode"] = "x"
        self.layout["hoversubplots"] = "axis"

    def add_subplot(
        self,
        x_domain: typing.Tuple[float, float] = None,
        y_domain: typing.Tuple[float, float] = None,
        x_share_with: "PlotlySubplot" = None,
        x_skip_no_data=False,
        rangeslider_visible=False,
    ) -> "PlotlySubplot":
        new_subplot = PlotlySubplot(
            self,
            x_domain=x_domain,
            y_domain=y_domain,
            x_share_with=x_share_with,
        )
        self.subplots.append(new_subplot)
        if self.dry_run:
            return new_subplot

        self.layout[new_subplot.xaxis_layout_id] = {}
        self.layout[new_subplot.yaxis_layout_id] = {}

        if x_domain:
            self.layout[new_subplot.xaxis_layout_id]["domain"] = x_domain
        if y_domain:
            self.layout[new_subplot.yaxis_layout_id]["domain"] = y_domain

        assert not (x_skip_no_data and x_share_with)
        if not x_share_with:
            self.layout[new_subplot.xaxis_layout_id]["rangeslider"] = dict(
                visible=rangeslider_visible
            )
            self.layout[new_subplot.xaxis_layout_id]["anchor"] = new_subplot.yaxis_id
        if x_skip_no_data:
            self.layout[new_subplot.xaxis_layout_id]["type"] = "category"

        if self.template == "simple_white":
            self.layout[new_subplot.xaxis_layout_id]["showgrid"] = False
            self.layout[new_subplot.xaxis_layout_id]["zeroline"] = False
            self.layout[new_subplot.xaxis_layout_id]["showline"] = True
            self.layout[new_subplot.xaxis_layout_id]["linecolor"] = "black"
            self.layout[new_subplot.xaxis_layout_id]["ticks"] = "outside"
            self.layout[new_subplot.xaxis_layout_id]["tickcolor"] = "black"

            self.layout[new_subplot.yaxis_layout_id]["showgrid"] = False
            self.layout[new_subplot.yaxis_layout_id]["zeroline"] = False
            self.layout[new_subplot.yaxis_layout_id]["showline"] = True
            self.layout[new_subplot.yaxis_layout_id]["linecolor"] = "black"
            self.layout[new_subplot.yaxis_layout_id]["ticks"] = "outside"
            self.layout[new_subplot.yaxis_layout_id]["tickcolor"] = "black"

        # crosshair
        self.layout[new_subplot.xaxis_layout_id]["showspikes"] = True
        self.layout[new_subplot.xaxis_layout_id]["spikemode"] = "across"
        self.layout[new_subplot.xaxis_layout_id]["spikesnap"] = "cursor"
        self.layout[new_subplot.xaxis_layout_id]["spikedash"] = "solid"
        self.layout[new_subplot.xaxis_layout_id]["spikecolor"] = Color.to_rgba_str(
            "black", 0.15
        )
        # workaround to disable border lines around spikes
        # ref: https://stackoverflow.com/questions/64287427/how-to-change-white-border-color-around-plotly-spike-lines
        self.layout[new_subplot.xaxis_layout_id]["spikethickness"] = -2

        self.layout[new_subplot.yaxis_layout_id]["showspikes"] = True
        self.layout[new_subplot.yaxis_layout_id]["spikemode"] = "across"
        self.layout[new_subplot.yaxis_layout_id]["spikesnap"] = "cursor"
        self.layout[new_subplot.yaxis_layout_id]["spikedash"] = "solid"
        self.layout[new_subplot.yaxis_layout_id]["spikecolor"] = Color.to_rgba_str(
            "black", 0.15
        )
        self.layout[new_subplot.yaxis_layout_id]["spikethickness"] = -2

        return new_subplot

    def to_go(self) -> go.Figure:
        fig = go.Figure(
            data=self.data,
            layout=self.layout,
        )
        return fig

    def show(self):
        if self.dry_run:
            return

        self.to_go().show()

    def to_html(self, filepath):
        if self.dry_run:
            return

        self.to_go().write_html(filepath)


class PlotlySubplot:
    # TODO: add domain or col/row
    # TODO: set reference to mention in added traces/data
    def __init__(
        self,
        fig: PlotlyFigure,
        x_domain,
        y_domain,
        x_share_with: "PlotlySubplot" = None,
    ):
        def _new_axis_id(fig: PlotlyFigure, axis=typing.Literal["x", "y"]) -> str:
            """
            Create new axis id for subplot.
            """
            assert axis in ["x", "y"]
            if len(fig.subplots) == 0:
                return axis
            return f"{axis}{len(fig.subplots)+1}"

        def _new_axis_layout_id(
            fig: PlotlyFigure, axis=typing.Literal["x", "y"]
        ) -> str:
            """
            Create new axis layout id for subplot.
            """
            assert axis in ["x", "y"]
            if len(fig.subplots) == 0:
                return f"{axis}axis"
            return f"{axis}axis{len(fig.subplots)+1}"

        self.fig = fig
        self.x_domain = x_domain
        self.y_domain = y_domain

        if x_share_with:
            self.xaxis_id = x_share_with.xaxis_id
        else:
            self.xaxis_id = _new_axis_id(fig, "x")

        self.xaxis_layout_id = _new_axis_layout_id(fig, "x")
        self.yaxis_id = _new_axis_id(fig, "y")
        self.yaxis_layout_id = _new_axis_layout_id(fig, "y")

    def add(self, go):
        if self.fig.dry_run:
            return

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
        color_up=Color.GREEN.value,
        color_down=Color.RED.value,
        line_width=DEFAULT_LINE_WIDTH,
        name="ohlc",
    ):
        if self.fig.dry_run:
            return

        assert type(data_df.index) == pd.DatetimeIndex

        self.fig.data.append(
            go.Candlestick(
                x=data_df.index,
                open=data_df[col_open],
                high=data_df[col_high],
                low=data_df[col_low],
                close=data_df[col_close],
                increasing_line_color=color_up,
                increasing_fillcolor=color_up,
                decreasing_line_color=color_down,
                decreasing_fillcolor=color_down,
                line=dict(width=line_width),
                name=name,
                xaxis=self.xaxis_id,
                yaxis=self.yaxis_id,
            )
        )

    def add_hline(
        self,
        y,
        line_color=Color.BLUE.value,
        line_width=DEFAULT_LINE_WIDTH,
        line_dash="solid",
        opacity=1.0,
    ):
        if self.fig.dry_run:
            return

        # fig.add_hline not used because go.Figure is not created yet
        self.fig.layout["shapes"].append(
            dict(
                type="line",
                xref="paper",
                yref=self.yaxis_id,
                x0=0,
                x1=1,
                y0=y,
                y1=y,
                line=dict(
                    color=Color.to_rgba_str(line_color, opacity),
                    width=line_width,
                    dash=line_dash,
                ),
            )
        )

    def add_vline(
        self,
        x,
        line_color=Color.BLUE.value,
        line_width=DEFAULT_LINE_WIDTH,
        line_dash="solid",
        opacity=1.0,
    ):
        if self.fig.dry_run:
            return

        self.fig.layout["shapes"].append(
            dict(
                type="line",
                xref=self.xaxis_id,
                yref="paper",
                x0=x,
                x1=x,
                y0=0,
                y1=1,
                line=dict(
                    color=Color.to_rgba_str(line_color, opacity),
                    width=line_width,
                    dash=line_dash,
                ),
            )
        )

    def add_vrect(
        self,
        x0,
        x1,
        line_color=None,
        line_opacity=1.0,
        line_width=0,
        line_dash=None,
        fill_color=Color.GREY.value,
        fill_opacity=1.0,
    ):
        if self.fig.dry_run:
            return

        self.fig.layout["shapes"].append(
            dict(
                type="rect",
                xref=self.xaxis_id,
                yref="paper",
                x0=x0,
                x1=x1,
                y0=self.y_domain[0] if self.y_domain else 0,
                y1=self.y_domain[1] if self.y_domain else 1,
                line=dict(
                    color=Color.to_rgba_str(line_color, line_opacity),
                    width=line_width,
                    dash=line_dash,
                ),
                fillcolor=Color.to_rgba_str(fill_color, fill_opacity),
            )
        )

    def add_annotation(
        self,
        x,
        y,
        text,
        font_color=Color.BLACK.value,
        font_size=10,
        ax=0,
        ay=0,
        show_arrow=False,
        arrow_size=1,
        arrow_head=1,
        arrow_color=Color.BLACK.value,
        arrow_opacity=1.0,
    ):
        if self.fig.dry_run:
            return

        if show_arrow:
            arrowcolor = Color.to_rgba_str(arrow_color, arrow_opacity)
        else:
            arrowcolor = Color.to_rgba_str(None, 0.0)

        self.fig.layout["annotations"].append(
            dict(
                x=x,
                y=y,
                xref=self.xaxis_id,
                yref=self.yaxis_id,
                text=text,
                showarrow=True,
                arrowhead=arrow_head,
                arrowsize=arrow_size,
                arrowcolor=arrowcolor,
                ax=ax,
                ay=ay,
                font=dict(
                    color=font_color,
                    size=font_size,
                ),
            )
        )
