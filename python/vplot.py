"""
Plotting.
"""

# pylint: disable=too-many-lines
import base64
import enum
import pathlib
import typing
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpfo
import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go


class Color(enum.Enum):
    """
    Some common CSS color names.
    ref: https://matplotlib.org/stable/gallery/color/named_colors.html
    """

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

    def next(self):
        """
        Return the next color after this.
        """
        cls = self.__class__
        members = list(cls)
        index = members.index(self) + 1
        if index >= len(members):
            return members[0]
        return members[index]

    @staticmethod
    def to_css(color) -> str:
        """
        Return string representation of the color.
        """
        if isinstance(color, Color):
            ret = color.value
        elif isinstance(color, str):
            ret = color
        elif color is None:
            ret = None
        else:
            raise TypeError
        return ret


class MarkerSymbol(enum.Enum):
    """
    List of supported Marker symbols.
    """

    TRIANGLE_UP = 1
    TRIANGLE_DOWN = 2
    CIRCLE = 3
    EMPTY_CIRCLE = 4


class Dash(enum.Enum):
    """
    List of supported Dashes.
    """

    SOLID = 1
    DOT = 2
    DASH = 3
    LONGDASH = 4
    DASHDOT = 5
    LONGDASHDOT = 6


# pylint: disable=too-few-public-methods
class Lines:
    """
    Lines.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x: typing.Iterable = None,
        y: typing.Iterable = None,
        color: str | Color = None,
        width: float = 1.0,
        dash: Dash = Dash.DOT,
    ):
        """
        Init.

        Args:
            color: string with the CSS color name or a color code.
        """
        assert isinstance(x, (typing.Iterable, type(None)))
        assert isinstance(y, (typing.Iterable, type(None)))
        assert isinstance(color, (str, Color, type(None)))
        assert isinstance(width, float)
        assert isinstance(dash, Dash)

        self.x = x
        self.y = y
        self.color = color
        self.width = width
        self.dash = dash


class Bar:
    """
    Bar chart.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x: typing.Iterable,
        y: typing.Iterable,
        color: str | Color = None,
        name: str = None,
        showlegend: bool = False,
        showannotation: bool = False,
    ):
        """
        Init.

        Args:
            color: string with the CSS color name or a color code.
        """
        assert isinstance(x, typing.Iterable)
        assert isinstance(y, typing.Iterable)
        assert isinstance(color, (str, Color, type(None)))
        assert isinstance(name, (str, type(None)))
        assert isinstance(showlegend, bool)
        assert isinstance(showannotation, bool)

        self.x = x
        self.y = y
        self.color = color
        self.name = name
        self.showlegend = showlegend
        self.showannotation = showannotation


# pylint: disable=too-many-instance-attributes
class Histogram:
    """
    Histogram.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=redefined-builtin
    def __init__(
        self,
        data: list[typing.Iterable],
        bins: int = 10,
        range: tuple[float, float] = None,
        is_horizontal: bool = False,
        is_probability_density: bool = False,
        width: float = None,
        color: str | Color = None,
        fill: typing.Literal[None, "solid", "transparent"] = None,
        name: str = None,
        showlegend: bool = False,
        showannotation: bool = False,
    ):
        """
        Init.

        Args:
            a, bins, range, density:
                https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
            color: string with the CSS color name or a color code.

        Warning!
        See examples how underlying np.histogram behaves for `bins`.
        """
        assert isinstance(data, typing.Iterable)
        assert isinstance(is_horizontal, bool)
        assert isinstance(is_probability_density, bool)
        assert isinstance(width, (float, type(None)))
        assert isinstance(color, (str, Color, type(None)))
        assert isinstance(fill, (str, type(None)))
        if isinstance(fill, str):
            assert fill in ("solid", "transparent")
        assert isinstance(name, (str, type(None)))
        assert isinstance(showlegend, bool)
        assert isinstance(showannotation, bool)

        self.data = data
        self.bins = bins
        self.range = range
        self.is_horizontal = is_horizontal
        self.is_probability_density = is_probability_density
        self.width = width
        self.color = color
        self.fill = fill
        self.name = name
        self.showlegend = showlegend
        self.showannotation = showannotation


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
class Trace:
    """
    Generic trace.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x: typing.Iterable,
        y: typing.Iterable,
        secondary_y: bool = False,
        color: str | Color = None,
        width: float = None,
        dash: Dash = Dash.SOLID,
        name: str = None,
        showlegend: bool = False,
        showannotation: bool = False,
    ):
        """
        Init.

        Args:
            color: string with the CSS color name or a color code.
        """
        assert isinstance(x, typing.Iterable)
        assert isinstance(y, typing.Iterable)
        assert isinstance(secondary_y, bool)
        assert isinstance(color, (str, Color, type(None)))
        assert isinstance(width, (int, float, type(None)))
        assert isinstance(dash, Dash)
        assert isinstance(name, (str, type(None)))
        assert isinstance(showlegend, bool)
        assert isinstance(showannotation, bool)

        self.x = x
        self.y = y
        self.secondary_y = secondary_y
        self.color = color
        self.width = width
        self.dash = dash
        self.name = name
        self.showlegend = showlegend
        self.showannotation = showannotation


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
class Candlestick(Trace):
    """
    Candlestick trace.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        x: pd.Index,
        yopen: pd.Series,
        yhigh: pd.Series,
        ylow: pd.Series,
        yclose: pd.Series,
        width: float = None,
        name: str = "OHLC",
        showlegend: bool = False,
        showannotation: bool = False,
    ):
        """
        Init.
        """
        assert isinstance(x, pd.Index)
        assert isinstance(yopen, pd.Series)
        assert isinstance(yhigh, pd.Series)
        assert isinstance(ylow, pd.Series)
        assert isinstance(yclose, pd.Series)
        assert isinstance(width, (int, float, type(None)))
        assert isinstance(name, str)
        assert isinstance(showlegend, bool)
        assert isinstance(showannotation, bool)

        self.x = x
        self.yopen = yopen
        self.yhigh = yhigh
        self.ylow = ylow
        self.yclose = yclose
        self.color = None
        self.width = width
        self.name = name
        self.showlegend = showlegend
        self.showannotation = showannotation


# pylint: disable=too-few-public-methods
class Scatter(Trace):
    """
    Scatter trace.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x: pd.Index,
        y: pd.Series,
        secondary_y: bool = False,
        color: str | Color = None,
        width: int | float = None,
        dash: Dash = Dash.SOLID,
        name: str = None,
        showlegend: bool = False,
        showannotation: bool = False,
        mode: typing.Literal["lines", "lines+markers", "markers"] = "lines",
        marker_symbol: MarkerSymbol = None,
        marker_size: int = None,
        marker_yshift: float = 0,
    ):
        """
        Init.

        Args:
            color: string with the CSS color name or a color code;
            mode: string with the following options
                'lines'
                'lines+markers'
                'markers'
        """
        Trace.__init__(
            self,
            x=x,
            y=y,
            secondary_y=secondary_y,
            color=color,
            width=width,
            dash=dash,
            name=name,
            showlegend=showlegend,
            showannotation=showannotation,
        )

        assert mode in ("lines", "lines+markers", "markers")
        assert isinstance(marker_symbol, (MarkerSymbol, type(None)))
        assert not (mode in ("markers", "lines+markers") and marker_symbol is None)
        assert isinstance(marker_size, (float, int, type(None)))
        assert isinstance(marker_yshift, (float, int))

        self.mode = mode
        self.marker_symbol = marker_symbol
        self.marker_size = marker_size
        self.marker_yshift = marker_yshift


class Step(Trace):
    """
    Step trace.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x: pd.Index,
        y: pd.Series,
        secondary_y: bool = False,
        color: str | Color = None,
        width: int | float = None,
        dash: Dash = Dash.SOLID,
        name: str = None,
        showlegend: bool = False,
        showannotation: bool = False,
    ):
        """
        Init.

        Args:
            color: string with the CSS color name or a color code;
        """
        Trace.__init__(
            self,
            x=x,
            y=y,
            secondary_y=secondary_y,
            color=color,
            width=width,
            dash=dash,
            name=name,
            showlegend=showlegend,
            showannotation=showannotation,
        )


# pylint: disable=too-few-public-methods
class Subplot:
    """
    Subplot containing one or several traces.
    """

    # special identifier used in several plotly calls
    _plotly_id: int

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def __init__(
        self,
        traces: list[Trace | Bar | Histogram | None],
        lines: list[Lines] = None,
        x_min=None,
        x_max=None,
        col: int | list[int] = 1,
        row: int | list[int] = 1,
        x_title: str = None,
        y_title: str = None,
        subtitle_text: str = None,
        subtitle_x: float = 0,
        subtitle_y: float = 0,
        legendgroup_name: str = None,
        log_y: bool = False,
    ):
        """
        Init.

        Args:
            traces: a list of Traces to be plotted;
            lines: a list of horizontal/vertical Lines to be plotted;
            x_min, x_max: can be used to prevent Plotly x axis changing width
            col: column number of this subplot or a range [int, int];
            row: row number of this subplot or a range [int, int];
            x_title: str
            y_title: str
            subtitle_text: str
            subtitle_x: float in relative units
            subtitle_y: float in relative units
            legendgroup_name: if not None - subtitle for all traces under
                              this subplot;
            log_y: use logarithmic scale

        Change `col`, `row` only if there is more than one
        subplot in a plot.
        """
        assert isinstance(traces, (list, type(None)))
        if traces is not None:
            assert len(traces) > 0
            for t in traces:
                assert isinstance(t, (Trace, Bar, Histogram))
        assert isinstance(lines, (list, type(None)))
        if lines is not None:
            assert len(lines) > 0
            for ln in lines:
                assert isinstance(ln, Lines)
        assert isinstance(col, (int, list))
        if isinstance(col, int):
            assert col >= 1
        elif isinstance(col, list):
            assert len(col) == 2
            for c in col:
                assert isinstance(c, int)
        assert isinstance(row, (int, list))
        if isinstance(row, int):
            assert row >= 1
        elif isinstance(row, list):
            assert len(row) == 2
            for c in row:
                assert isinstance(c, int)
        assert isinstance(x_title, (str, type(None)))
        assert isinstance(y_title, (str, type(None)))
        assert isinstance(subtitle_text, (str, type(None)))
        assert isinstance(subtitle_x, (int, float))
        assert isinstance(subtitle_y, (int, float))
        assert isinstance(legendgroup_name, (str, type(None)))
        assert isinstance(log_y, bool)
        assert (x_min is None and x_max is None) or (
            x_min is not None and x_max is not None
        )

        self.traces = traces
        self.lines = lines
        self.x_min = x_min
        self.x_max = x_max
        self.col = col
        self.row = row
        self.x_title = x_title
        self.y_title = y_title
        self.subtitle_text = subtitle_text
        self.subtitle_x = subtitle_x
        self.subtitle_y = subtitle_y
        self.legendgroup_name = legendgroup_name
        self.log_y = log_y

    def get_plotly_id_str(self):
        """
        Get plotly id.
        """

        assert self._plotly_id > 0
        if self._plotly_id == 1:
            return ""
        return str(self._plotly_id)


# pylint: disable=too-few-public-methods
class LogicSignalSubplot(Subplot):
    """
    Logic signal subplot containing one or several Step traces.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        steps: list[Step],
        lines: list[Lines] = None,
        col: int | list[int] = 1,
        row: int | list[int] = 1,
        x_title: str = None,
        y_title: str = None,
        subtitle_text: str = None,
        subtitle_x: float = 0,
        subtitle_y: float = 0,
        legendgroup_name: str = None,
        yshift: float = 1.1,
    ):
        """
        Init.

        Args:
            steps: a list of Steps to be plotted;
            lines: a list of horizontal/vertical Lines to be plotted;
            col: column number of this subplot or a range [int, int];
            row: row number of this subplot or a range [int, int];
            x_title: str
            y_title: str
            subtitle_text: str
            subtitle_x: float in relative units
            subtitle_y: float in relative units
            legendgroup_name: if not None - subtitle for all traces under
                              this subplot
            yshift: value for shifting one logical signal against another.

        Change `col`, `row` only if there is more than one
        subplot in a plot.
        """
        Subplot.__init__(
            self,
            traces=steps,
            lines=lines,
            col=col,
            row=row,
            x_title=x_title,
            y_title=y_title,
            subtitle_text=subtitle_text,
            subtitle_x=subtitle_x,
            subtitle_y=subtitle_y,
            legendgroup_name=legendgroup_name,
        )

        assert isinstance(steps, list)
        assert len(steps) > 0
        for s in steps:
            assert isinstance(s, Step)
        assert isinstance(yshift, (int, float))

        self.steps = self.traces
        self.yshift = yshift


# pylint: disable=too-few-public-methods
class PieSubplot(Subplot):
    """
    Pie subplot.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        labels: list[str],
        values: list[float],
        col: int | list[int] = 1,
        row: int | list[int] = 1,
        subtitle_text: str = None,
        subtitle_x: float = 0,
        subtitle_y: float = 0,
        hole: float = 0,
        colors: list[str | Color] = None,
        name: str = None,
        showlegend: bool = False,
    ):
        """
        Init.

        Args:
            labels: a list of labels for the pie;
            values: a list of values corresponding to labels;
            col: column number of this subplot or a range [int, int];
            row: row number of this subplot or a range [int, int];
            subtitle_text: str
            subtitle_x: float in relative units
            subtitle_y: float in relative units
            hole: ratio of a hole inside the pie;
            colors: a list of colors;
            name:
            showlegend:

        Change `col`, `row` only if there is more than one
        subplot in a plot.
        """
        Subplot.__init__(
            self,
            traces=None,
            lines=None,
            col=col,
            row=row,
            subtitle_text=subtitle_text,
            subtitle_x=subtitle_x,
            subtitle_y=subtitle_y,
            legendgroup_name=None,
        )

        assert isinstance(labels, list)
        for label in labels:
            assert isinstance(label, str)
        assert isinstance(values, list)
        for value in values:
            assert isinstance(value, (float, int, np.integer, np.floating))
        assert len(labels) > 0
        assert len(labels) == len(values)
        assert isinstance(hole, (float, int))
        assert 0.0 <= hole <= 1.0
        assert isinstance(colors, (list, type(None)))
        if colors is not None:
            assert len(colors) == len(labels)
            for color in colors:
                assert isinstance(color, (str, Color))
        assert isinstance(name, str)
        assert isinstance(showlegend, bool)

        self.labels = labels
        self.values = values
        self.hole = hole
        self.colors = colors
        self.name = name
        self.showlegend = showlegend


class ImageSubplot(Subplot):
    """
    Image subplot.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image_path: str,
        col: int | list[int] = 1,
        row: int | list[int] = 1,
        subtitle_text: str = None,
        subtitle_x: float = 0,
        subtitle_y: float = 0,
    ):
        """
        Init.

        Args:
            image_path: a list of labels for the pie;
            col: column number of this subplot or a range [int, int];
            row: row number of this subplot or a range [int, int];
            subtitle_text: str
        """
        Subplot.__init__(
            self,
            traces=None,
            lines=None,
            col=col,
            row=row,
            subtitle_text=subtitle_text,
            subtitle_x=subtitle_x,
            subtitle_y=subtitle_y,
        )

        assert isinstance(image_path, str)
        self.image_path = image_path


class Plot:
    """
    Plot containing one or several Subplots.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def __init__(
        self,
        subplots: list[Subplot],
        share_x: bool | str = True,
        share_y: bool | str = False,
        lines: list[Lines] = None,
        title_text: str = None,
        height: int = None,
        width: int = None,
        row_ratios: list[float] = None,
        col_ratios: list[float] = None,
        font_size: int | float = None,
        grid: bool = True,
    ):
        """
        Init.

        Args:
            subplots: a list of subplots to be plotted;
            share_x: share x axis between subplots
            share_y: share y axis between subplots
            lines: a list of horizontal/vertical Lines to be plotted;
            title_text: str
            height: plot height in pixels;
            width: plot width in pixels;
            row_ratios: list of row height ratios,
                         e.g. [0.5, 0.2, 0.3];
            col_ratios: list of column width ratios,
                         e.g. [0.5, 0.2, 0.3].
            font_size: int
            grid: add grid to the plot
        """
        assert isinstance(subplots, list)
        assert len(subplots) > 0
        for s in subplots:
            assert isinstance(s, Subplot)
            # pylint: disable=invalid-name
            for s2 in subplots:
                assert not (
                    s is not s2 and s.col == s2.col and s.row == s2.row
                ), "Both rows and columns of different suplots should not overlap"
        assert isinstance(share_x, (bool, str))
        assert isinstance(share_y, (bool, str))
        assert isinstance(lines, (list, type(None)))
        if lines:
            assert len(lines) > 0
            for line in lines:
                assert isinstance(line, Lines)
        assert isinstance(title_text, (str, type(None)))
        assert isinstance(height, (int, type(None)))
        if height:
            assert height >= 0
        assert isinstance(width, (int, type(None)))
        if width:
            assert width >= 0
        assert isinstance(row_ratios, (list, type(None)))
        if row_ratios:
            for r in row_ratios:
                assert isinstance(r, (int, float))
        assert isinstance(col_ratios, (list, type(None)))
        if col_ratios:
            for c in col_ratios:
                assert isinstance(c, (int, float))
        assert isinstance(font_size, (int, float, type(None)))
        assert isinstance(grid, bool)

        self.subplots = subplots
        self.share_x = share_x
        self.share_y = share_y
        self.lines = lines
        self.title_text = title_text
        self.height = height
        self.width = width
        self.row_ratios = row_ratios
        self.col_ratios = col_ratios
        self.font_size = font_size
        self.grid = grid
        self._init_plot_dimensions()

    def _init_plot_dimensions(self):
        rows = 1
        cols = 1
        for s in self.subplots:
            if isinstance(s.row, int):
                rows = max(s.row, rows)
            else:
                rows = max(s.row[1], rows)
            if isinstance(s.col, int):
                cols = max(s.col, cols)
            else:
                cols = max(s.col[1], cols)
        self.rows = rows
        self.cols = cols

    def image(self, filename, scale=None):
        """
        Generate image file depending on the filename extension.
        """
        raise NotImplementedError()

    def html(self, filename):
        """
        Generate .html file.
        """
        raise NotImplementedError()

    def to_file(self, filename, scale=3):
        """
        Generate file based on the extension of the filename.
        """
        extension = pathlib.Path(filename).suffix
        if extension == ".html":
            self.html(filename)
        else:
            self.image(filename, scale=scale)


class PlotlyPlot(Plot):
    """
    Plotly plot containing one or several Subplots.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        subplots: list[Subplot],
        share_x: bool = True,
        share_y: bool = False,
        lines: list[Lines] = None,
        title_text: str = None,
        height: int = None,
        width: int = None,
        row_ratios: list[float] = None,
        col_ratios: list[float] = None,
        font_size: int = None,
        grid: bool = True,
    ):
        """
        Init.

        Args:
            subplots: a list of subplots to be plotted;
            share_x: share x axis between subplots
            share_y: share y axis between subplots
            lines: a list of horizontal/vertical Lines to be plotted;
            title_text: str
            height: plot height in pixels;
            width: plot width in pixels;
            row_ratios: list of row height ratios,
                         e.g. [0.5, 0.2, 0.3];
            col_ratios: list of column width ratios,
                         e.g. [0.5, 0.2, 0.3].
            font_size: int
            grid: add grid to the plot
        """
        Plot.__init__(
            self,
            subplots,
            share_x,
            share_y,
            lines,
            title_text,
            height,
            width,
            row_ratios,
            col_ratios,
            font_size,
            grid,
        )

        self._init_specs()

    # pylint: disable=too-many-branches
    # generate plotly specs for make_subplots()
    # pylint: disable=line-too-long
    # ref: https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html  # noqa
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def _init_specs(self):

        # create an array with accurate col/row dimensions,
        # filled with index of occupying subplot
        # if the slot is empty, set to -1
        specs_arr = np.empty((self.rows, self.cols), dtype=int)
        specs_arr[:] = -1
        for i, s in enumerate(self.subplots):
            # rows and cols in plotly are indexed from 1
            if isinstance(s.row, int):
                row_min = s.row - 1
                row_max = row_min + 1
            else:
                row_min = s.row[0] - 1
                row_max = s.row[1]
            if isinstance(s.col, int):
                col_min = s.col - 1
                col_max = col_min + 1
            else:
                col_min = s.col[0] - 1
                col_max = s.col[1]
            specs_arr[row_min:row_max, col_min:col_max] = i

        # go over the array row-by-row
        # assign every subplot an id based on their global position
        # pylint: disable=too-many-nested-blocks
        processed_subplot_i = set()
        plotly_id = 1
        for row in specs_arr:
            for subplot_i in row:
                if subplot_i not in processed_subplot_i:
                    # pylint: disable=protected-access
                    self.subplots[subplot_i]._plotly_id = plotly_id
                    processed_subplot_i.add(subplot_i)
                    subplot_has_secondary_y = False
                    if self.subplots[subplot_i].traces is not None:
                        for t in self.subplots[subplot_i].traces:
                            if not isinstance(t, Histogram) and t.secondary_y:
                                subplot_has_secondary_y = True
                    if subplot_has_secondary_y:
                        plotly_id += 2
                    else:
                        plotly_id += 1

        # go over the array row-by-row
        # if entry is empty -> add {}
        # if entry is not empty -> fill, based on type and span
        # if entry is already used -> add None
        # pylint: disable=too-many-nested-blocks
        used_i = set()
        specs = []
        # rows and cols in np are indexed from 0
        for irow in range(0, self.rows):
            specs_row = []
            for icol in range(0, self.cols):
                i = specs_arr[irow, icol]
                # entry empty
                if i == -1:
                    specs_entry = {}
                # new entry
                elif i not in used_i:
                    specs_entry = {}
                    s = self.subplots[i]
                    if isinstance(s, PieSubplot):
                        type_str = "domain"
                        for t in s.traces:
                            assert not t.secondary_y
                    elif isinstance(s, ImageSubplot):
                        type_str = "image"
                    else:
                        type_str = "xy"
                        for t in s.traces:
                            if not isinstance(t, Histogram) and t.secondary_y:
                                specs_entry["secondary_y"] = True
                    specs_entry["type"] = type_str
                    if isinstance(s.row, list):
                        specs_entry["rowspan"] = s.row[1] - s.row[0] + 1
                    if isinstance(s.col, list):
                        specs_entry["colspan"] = s.col[1] - s.col[0] + 1
                    used_i.add(i)
                # entry already described before
                else:
                    specs_entry = None
                specs_row.append(specs_entry)
            specs.append(specs_row)

        self.specs = specs

    @staticmethod
    def _get_marker(
        marker_symbol: MarkerSymbol | None,
        marker_size: int | None,
        marker_yshift: float,
        color: str = None,
        line_width: float = 0.5,
    ):
        assert isinstance(marker_symbol, (MarkerSymbol, type(None)))
        assert isinstance(marker_size, (float, int, type(None)))
        assert isinstance(marker_yshift, (float, int))
        assert isinstance(color, (str, type(None)))
        assert isinstance(line_width, (float, int))

        if marker_symbol == MarkerSymbol.TRIANGLE_UP:
            marker = {
                "size": marker_size,
                "standoff": 10 + marker_yshift,
                "symbol": "triangle-up",
                "angle": 0,
                "color": color,
            }
        elif marker_symbol == MarkerSymbol.TRIANGLE_DOWN:
            # standoff can only be >0, so to move the triangle down
            # above with standoff=10, it is rotated 180deg and
            # 'triangle-up' is used
            marker = {
                "size": marker_size,
                "standoff": 10 - marker_yshift,
                "symbol": "triangle-up",
                "angle": 180,
                "color": color,
            }
        elif marker_symbol == MarkerSymbol.CIRCLE:
            marker = {
                "size": marker_size,
                "symbol": "circle",
                "angle": 0,
                "color": color,
            }
        elif marker_symbol == MarkerSymbol.EMPTY_CIRCLE:
            marker = {
                "size": marker_size,
                "symbol": "circle",
                "angle": 0,
                "color": "white",
                "line": {
                    "color": color,
                    "width": line_width,
                },
            }
        elif marker_symbol is None:
            marker = None
        else:
            raise TypeError

        return marker

    @staticmethod
    def _get_scatter_line_width(width: int | float | type(None)):
        # scatter lines cannot be set thick by default
        assert isinstance(width, (int, float, type(None)))

        if width is None:
            ret = 1.0
        else:
            ret = width
        return ret

    @staticmethod
    def _get_candlestick_line_width(width: int | float | type(None)):
        # candlestick lines are too thick by default
        assert isinstance(width, (int, float, type(None)))

        if width is None:
            ret = 0.7
        else:
            ret = width
        return ret

    @staticmethod
    def _get_hv_line_width(width: int | float | type(None)):
        # horizontal and vertical lines cannot be set thick by default
        assert isinstance(width, (int, float, type(None)))

        if width is None:
            ret = 1.0
        else:
            ret = width
        return ret

    @staticmethod
    def _get_line_dash(dash: Dash):
        assert isinstance(dash, Dash)

        if dash == Dash.SOLID:
            ret = "solid"
        elif dash == Dash.DOT:
            ret = "dot"
        elif dash == Dash.DASH:
            ret = "dash"
        elif dash == Dash.LONGDASH:
            ret = "longdash"
        elif dash == Dash.DASHDOT:
            ret = "dashdot"
        elif dash == Dash.LONGDASHDOT:
            ret = "longdashdot"
        else:
            raise ValueError
        return ret

    @staticmethod
    def _get_annotation_text(legendgroup_name: str | type(None), name: str) -> str:
        assert isinstance(legendgroup_name, (str, type(None)))
        assert isinstance(name, str)

        if legendgroup_name:
            return f"{legendgroup_name}: {name}"
        return name

    @staticmethod
    def _add_bar_to_fig(fig: go.Figure, subplot: Subplot, bar: Bar):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, Subplot)
        assert isinstance(bar, Bar)

        fig.add_trace(
            go.Bar(
                x=bar.x,
                y=bar.y,
                marker_color=Color.to_css(bar.color),
                showlegend=bar.showlegend,
                name=bar.name,
                # do not truncate long hover text
                hoverlabel={"namelength": -1},
                legendgroup=subplot.legendgroup_name,
                legendgrouptitle_text=subplot.legendgroup_name,
            ),
            col=subplot.col,
            row=subplot.row,
        )

        if bar.showannotation:
            # I also tried adding text trace over the particular data points,
            # but plotly manages annotation text much more gracefully,
            # w/o croppling it or the underlying trace
            fig.add_annotation(
                x=bar.x[-1],
                y=bar.y.iloc[-1],
                text=PlotlyPlot._get_annotation_text(
                    subplot.legendgroup_name, bar.name
                ),
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                col=subplot.col,
                row=subplot.row,
            )

    @staticmethod
    def _add_histogram_to_fig(fig: go.Figure, subplot: Subplot, histogram: Histogram):
        count, index = np.histogram(
            a=histogram.data,
            bins=histogram.bins,
            range=histogram.range,
            density=histogram.is_probability_density,
        )

        if histogram.is_horizontal:
            x = count
            y = index
        else:
            x = index
            y = count

        if histogram.fill:
            if histogram.is_horizontal:
                fill = "tozerox"
            else:
                fill = "tozeroy"
            if histogram.fill == "solid":
                fillcolor = Color.to_css(histogram.color)
            elif histogram.fill == "transparent":
                fillcolor = None
        else:
            fill = None
            fillcolor = None

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                line={"shape": "hv"},
                line_color=Color.to_css(histogram.color),
                line_width=PlotlyPlot._get_scatter_line_width(histogram.width),
                fill=fill,
                fillcolor=fillcolor,
                showlegend=histogram.showlegend,
                name=histogram.name,
                # do not truncate long hover text
                hoverlabel={"namelength": -1},
                legendgroup=subplot.legendgroup_name,
                legendgrouptitle_text=subplot.legendgroup_name,
            ),
            col=subplot.col,
            row=subplot.row,
        )

    @staticmethod
    def _add_scatter_to_fig(fig: go.Figure, subplot: Subplot, trace: Trace):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, Subplot)
        assert isinstance(trace, Trace)

        color = Color.to_css(trace.color)
        line_width = PlotlyPlot._get_scatter_line_width(trace.width)
        fig.add_trace(
            go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                line_color=color,
                line_width=line_width,
                line_dash=PlotlyPlot._get_line_dash(trace.dash),
                marker=PlotlyPlot._get_marker(
                    trace.marker_symbol,
                    trace.marker_size,
                    trace.marker_yshift,
                    color=color,
                    line_width=line_width,
                ),
                showlegend=trace.showlegend,
                name=trace.name,
                # do not truncate long hover text
                hoverlabel={"namelength": -1},
                legendgroup=subplot.legendgroup_name,
                legendgrouptitle_text=subplot.legendgroup_name,
            ),
            secondary_y=trace.secondary_y,
            col=subplot.col,
            row=subplot.row,
        )

        if trace.showannotation:
            # I also tried adding text trace over the particular data points,
            # but plotly manages annotation text much more gracefully,
            # w/o croppling it or the underlying trace
            fig.add_annotation(
                x=trace.x[-1],
                y=trace.y.iloc[-1],
                text=PlotlyPlot._get_annotation_text(
                    subplot.legendgroup_name, trace.name
                ),
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                col=subplot.col,
                row=subplot.row,
            )

    @staticmethod
    def _add_step_to_fig(
        fig: go.Figure,
        subplot: Subplot,
        trace: Trace,
        yshift: int | float = 0,
        annotation_yshift: int | float = 0,
    ):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, Subplot)
        assert isinstance(trace, Trace)
        assert isinstance(yshift, (int, float))
        assert isinstance(annotation_yshift, (int, float))

        fig.add_trace(
            go.Scatter(
                x=trace.x,
                y=trace.y + yshift,
                mode="lines",
                line={"shape": "hv"},
                line_color=Color.to_css(trace.color),
                line_width=PlotlyPlot._get_scatter_line_width(trace.width),
                line_dash=PlotlyPlot._get_line_dash(trace.dash),
                showlegend=trace.showlegend,
                name=trace.name,
                # do not truncate long hover text
                hoverlabel={"namelength": -1},
                legendgroup=subplot.legendgroup_name,
                legendgrouptitle_text=subplot.legendgroup_name,
            ),
            secondary_y=trace.secondary_y,
            col=subplot.col,
            row=subplot.row,
        )

        if trace.showannotation:
            fig.add_annotation(
                x=trace.x[-1],
                y=trace.y.iloc[-1] + yshift,
                text=PlotlyPlot._get_annotation_text(
                    subplot.legendgroup_name, trace.name
                ),
                yshift=annotation_yshift,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                col=subplot.col,
                row=subplot.row,
            )

    @staticmethod
    def _add_candlestick_to_fig(fig: go.Figure, subplot: Subplot, trace: Trace):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, Subplot)
        assert isinstance(trace, Trace)

        fig.add_trace(
            go.Candlestick(
                x=trace.x,
                open=trace.yopen,
                high=trace.yhigh,
                low=trace.ylow,
                close=trace.yclose,
                showlegend=trace.showlegend,
                name=trace.name,
                # do not truncate long hover text
                hoverlabel={"namelength": -1},
                legendgroup=subplot.legendgroup_name,
                legendgrouptitle_text=subplot.legendgroup_name,
            ),
            col=subplot.col,
            row=subplot.row,
        )
        fig.update_traces(
            line_width=PlotlyPlot._get_candlestick_line_width(trace.width),
            selector={"type": "candlestick"},
        )

    def _add_subplot_to_fig(self, fig: go.Figure, subplot: Subplot):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, Subplot)

        row = subplot.row
        col = subplot.col
        if isinstance(row, typing.Iterable):
            row = row[0]
        if isinstance(col, typing.Iterable):
            col = col[0]

        if subplot.traces is not None:
            for t in subplot.traces:
                if isinstance(t, Bar):
                    PlotlyPlot._add_bar_to_fig(fig, subplot, t)
                elif isinstance(t, Histogram):
                    PlotlyPlot._add_histogram_to_fig(fig, subplot, t)
                elif isinstance(t, Scatter):
                    PlotlyPlot._add_scatter_to_fig(fig, subplot, t)
                elif isinstance(t, Step):
                    PlotlyPlot._add_step_to_fig(fig, subplot, t)
                elif isinstance(t, Candlestick):
                    PlotlyPlot._add_candlestick_to_fig(fig, subplot, t)
                else:
                    raise ValueError(f"{type(t)} is an unsupported trace class.")
        if subplot.x_min is not None and subplot.x_max is not None:
            fig.update_xaxes(range=[subplot.x_min, subplot.x_max])
        if subplot.log_y:
            fig.update_yaxes(type="log", col=col, row=row)
        if subplot.x_title:
            fig.update_xaxes(title_text=subplot.x_title, col=col, row=row)
        if subplot.y_title:
            fig.update_yaxes(title_text=subplot.y_title, col=col, row=row)

    def _add_logic_signal_subplot_to_fig(
        self, fig: go.Figure, subplot: LogicSignalSubplot
    ):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, LogicSignalSubplot)

        row = subplot.row
        col = subplot.col
        if isinstance(row, typing.Iterable):
            row = row[0]
        if isinstance(col, typing.Iterable):
            col = col[0]

        steps = []
        for s in subplot.steps:
            if isinstance(s, Step):
                steps.append(s)
            else:
                raise ValueError(f"{type(s)} is an unsupported trace class.")
        yshift = (len(steps) - 1) * subplot.yshift
        for s in steps:
            PlotlyPlot._add_step_to_fig(fig, subplot, s, yshift, annotation_yshift=-14)
            yshift = yshift - subplot.yshift
        # do not show y axis
        fig.update_yaxes(visible=False, row=row, col=col)

    def _add_pie_subplot_to_fig(self, fig: go.Figure, subplot: PieSubplot):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, PieSubplot)

        row = subplot.row
        col = subplot.col
        if isinstance(row, typing.Iterable):
            row = row[0]
        if isinstance(col, typing.Iterable):
            col = col[0]

        if subplot.colors:
            colors = [Color.to_css(color) for color in subplot.colors]
            marker = {"colors": colors}
        else:
            marker = None

        fig.add_trace(
            go.Pie(
                labels=subplot.labels,
                values=subplot.values,
                marker=marker,
                showlegend=subplot.showlegend,
                name=subplot.name,
                textposition="outside",
                insidetextorientation="horizontal",
                # do not truncate long hover text
                hoverlabel={"namelength": -1},
                legendgroup=subplot.legendgroup_name,
                legendgrouptitle_text=subplot.legendgroup_name,
            ),
            col=col,
            row=row,
        )

    def _add_image_subplot_to_fig(self, fig: go.Figure, subplot: PieSubplot):
        assert isinstance(fig, go.Figure)
        assert isinstance(subplot, ImageSubplot)

        row = subplot.row
        col = subplot.col
        if isinstance(row, typing.Iterable):
            row = row[0]
        if isinstance(col, typing.Iterable):
            col = col[0]

        # we need a dummy invisible plot as a frame
        # to be able to place image on top of it
        fig.add_trace(
            go.Scatter(),
            col=subplot.col,
            row=subplot.row,
        )
        fig.update_xaxes(visible=False, row=subplot.row, col=subplot.col)
        fig.update_yaxes(visible=False, row=subplot.row, col=subplot.col)

        extension = pathlib.Path(subplot.image_path).suffix
        # pylint: disable=consider-using-with
        image_base64 = base64.b64encode(open(subplot.image_path, "rb").read())
        if extension == ".svg":
            source = f"data:image/svg+xml;base64,{image_base64.decode()}"
        elif extension == ".png":
            source = f"data:image/png;base64,{image_base64.decode()}"
        else:
            raise ValueError(f"image file extension {extension} is not supported")

        fig.add_layout_image(
            source=source,
            xref=f"x{subplot.get_plotly_id_str()} domain",
            yref=f"y{subplot.get_plotly_id_str()} domain",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.5,
            sizex=1,
            sizey=1,
        )

    @staticmethod
    def _add_line_to_fig(
        fig: go.Figure,
        line: Lines,
        col: int | typing.Iterable | typing.Literal["all"] = "all",
        row: int | typing.Iterable | typing.Literal["all"] = "all",
    ):
        assert isinstance(fig, go.Figure)
        assert isinstance(line, Lines)
        assert isinstance(col, (int, str))
        if isinstance(col, str):
            assert col == "all"
        assert isinstance(row, (int, str, typing.Iterable))
        if isinstance(row, typing.Iterable):
            row = row[0]
        if isinstance(row, str):
            assert row == "all"

        if line.x is not None:
            # there is a bug that requires you to force `opacity` and
            # `line_width` to some value if a `simple_white` theme is used:
            # pylint: disable=line-too-long
            # ref: https://stackoverflow.com/questions/67327670/plotly-add-hline-doesnt-work-with-simple-white-template  # noqa
            for x in line.x:
                fig.add_vline(
                    x=x,
                    opacity=1,
                    line_width=PlotlyPlot._get_hv_line_width(line.width),
                    line_dash=PlotlyPlot._get_line_dash(line.dash),
                    line_color=Color.to_css(line.color),
                    col=col,
                    row=row,
                )
        if line.y is not None:
            for y in line.y:
                fig.add_hline(
                    y=y,
                    opacity=1,
                    line_width=PlotlyPlot._get_hv_line_width(line.width),
                    line_dash=PlotlyPlot._get_line_dash(line.dash),
                    line_color=Color.to_css(line.color),
                    col=col,
                    row=row,
                )

    def _add_lines_to_fig(self, fig: go.Figure):
        assert isinstance(fig, go.Figure)

        # add plot lines
        if self.lines:
            for line in self.lines:
                PlotlyPlot._add_line_to_fig(fig, line)
        # add subplot lines
        for s in self.subplots:
            if s.lines:
                for line in s.lines:
                    PlotlyPlot._add_line_to_fig(fig, line, col=s.col, row=s.row)

    def _add_subtitles_to_fig(self, fig: go.Figure):
        assert isinstance(fig, go.Figure)

        annotations = []
        for s in self.subplots:
            if s.subtitle_text:
                annotations += [
                    {
                        "text": s.subtitle_text,
                        "xref": f"x{s.get_plotly_id_str()} domain",
                        "yref": f"y{s.get_plotly_id_str()} domain",
                        "x": s.subtitle_x,
                        "y": s.subtitle_y,
                        "xanchor": "left",
                        "showarrow": False,
                    },
                ]
        fig.update_layout(annotations=annotations)

    def _get_fig(self) -> go.Figure:
        fig = plotly.subplots.make_subplots(
            rows=self.rows,
            cols=self.cols,
            shared_xaxes=self.share_x,
            shared_yaxes=self.share_y,
            specs=self.specs,
            row_heights=self.row_ratios,
            column_widths=self.col_ratios,
        )
        # set title
        if self.title_text:
            fig.update_layout(title={"text": self.title_text})
        # set dimensions
        fig.update_layout(width=self.width, height=self.height)
        # best theme in plotly
        fig.update_layout(template="simple_white")
        # set font size
        if self.font_size:
            fig.update_layout(font={"size": self.font_size})
        # set grid
        fig.update_yaxes(showgrid=self.grid)
        fig.update_xaxes(showgrid=self.grid)

        for s in self.subplots:
            if isinstance(s, LogicSignalSubplot):
                self._add_logic_signal_subplot_to_fig(fig, s)
            elif isinstance(s, PieSubplot):
                self._add_pie_subplot_to_fig(fig, s)
            elif isinstance(s, ImageSubplot):
                self._add_image_subplot_to_fig(fig, s)
            elif isinstance(s, Subplot):
                self._add_subplot_to_fig(fig, s)
            else:
                raise NameError("Unknown subplot")

        self._add_lines_to_fig(fig)
        self._add_subtitles_to_fig(fig)

        return fig

    def image(self, filename, scale=3):
        """
        Generate image file based on filename extension.
        """
        fig = self._get_fig()
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.write_image(filename, scale=scale)

    def html(self, filename):
        """
        Generate .html file.
        """
        fig = self._get_fig()
        # setup crosshair cursor
        fig.update_layout(hovermode="x")
        fig.update_xaxes(
            showspikes=True,
            spikemode="across+toaxis",
            spikesnap="cursor",
            spikedash="dot",
            spikecolor=Color.to_css(Color.BLACK),
            spikethickness=1,
        )  # in px
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
            spikecolor=Color.to_css(Color.BLACK),
            spikethickness=1,
        )  # in px
        # slider is conflicting with subplots
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.write_html(filename)


class MplPlot(Plot):
    """
    matplotlib plot containing one or several Subplots.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        subplots: list[Subplot],
        share_x: bool = True,
        share_y: bool = False,
        lines: list[Lines] = None,
        height: int = None,
        width: int = None,
        dpi: int = 100,
        row_ratios: list[float] = None,
        col_ratios: list[float] = None,
        font_size: int = 10,
        grid: bool = True,
    ):
        """
        Init.

        Args:
            subplots: a list of subplots to be plotted;
            specs: orientation of subplots within the graph,
                   see https://plotly.com/python/subplots/
            share_x: share x axis between subplots
            share_y: share y axis between subplots
            lines: a list of horizontal/vertical Lines to be plotted;
            height: plot height in pixels;
            width: plot width in pixels;
            row_ratios: list of row height ratios,
                         e.g. [0.5, 0.2, 0.3];
            col_ratios: list of column width ratios,
                         e.g. [0.5, 0.2, 0.3].
            font_size: int
            grid: add grid to the plot
        """
        Plot.__init__(
            self,
            subplots,
            share_x,
            share_y,
            lines,
            height,
            width,
            row_ratios,
            col_ratios,
            font_size,
            grid,
        )

        assert isinstance(dpi, int)
        self.dpi = dpi

    def image(self, filename, scale=None):
        """
        Generate image file based on filename extension.
        """
        fig = self._get_fig()
        fig.savefig(filename)

    def html(self, filename):
        """
        Generate .html file.
        """
        # Does not work reliably with mpld3
        # many features missing, formatting changes and glitches
        raise NotImplementedError

    def _get_fig(self) -> mpl.figure.Figure:
        """
        Return mpl fig.
        """
        # set font size
        # it must be done before the fig is initialized to affect all fields
        if self.font_size:
            # inspired by
            # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
            small_size = self.font_size * 0.66
            medium_size = self.font_size
            bigger_size = self.font_size * 1.3
            plt.rc("font", size=small_size)
            plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
            plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
            plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
            plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
            plt.rc("legend", fontsize=small_size)  # legend fontsize
            plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title

        # set grid
        plt.grid(visible=self.grid, which="both", axis="both")

        # set dpi
        # setting via fig(dpi=) parameter does not work
        # default is 100
        plt.rc("figure", dpi=self.dpi)

        px = 1 / plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(self.width * px, self.height * px))

        # squeeze=False always returns a 2d array of Axes
        # otherwise it could be squeezed to 1d or 0d array if not enough items.
        axs = fig.subplots(
            nrows=self.rows,
            ncols=self.cols,
            squeeze=False,
            width_ratios=self.col_ratios,
            height_ratios=self.row_ratios,
            sharex=self.share_x,
            sharey=self.share_y,
        )
        for s in self.subplots:
            ax = MplPlot._get_ax(axs, s.row, s.col)
            if isinstance(s, LogicSignalSubplot):
                self._add_logic_signal_subplot_to_ax(ax, s)
            elif isinstance(s, Subplot):
                self._add_subplot_to_ax(ax, s)
        self._add_lines_to_axs(axs)
        return fig

    @staticmethod
    def _get_ax(axs: np.ndarray[np.ndarray], row: int, col: int) -> mpl.axes.Axes:
        assert isinstance(axs, np.ndarray)
        for ax in axs:
            assert isinstance(ax, np.ndarray)
        assert isinstance(row, int)
        assert row >= 1
        assert isinstance(col, int)
        assert col >= 1

        return axs[row - 1][col - 1]

    @staticmethod
    def _add_logic_signal_subplot_to_ax(ax: mpl.axes.Axes, subplot: Subplot):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, Subplot)

        steps = []
        for s in subplot.steps:
            if isinstance(s, Step):
                steps.append(s)
            else:
                raise ValueError(f"{type(s)} is an unsupported trace class.")
        yshift = (len(steps) - 1) * subplot.yshift
        for s in steps:
            MplPlot._add_step_to_ax(ax, subplot, s, yshift, annotation_yshift=-14)
            yshift = yshift - subplot.yshift
        # do not show y axis
        ax.get_yaxis().set_visible(False)

    @staticmethod
    def _add_step_to_ax(
        ax: mpl.axes.Axes,
        subplot: Subplot,
        trace: Trace,
        yshift: float = 0,
        annotation_yshift: float = 0,
    ):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, Subplot)
        assert isinstance(trace, Trace)
        assert isinstance(yshift, (int, float))
        assert isinstance(annotation_yshift, (int, float))

        ax.step(
            x=trace.x,
            y=trace.y + yshift,
            color=Color.to_css(trace.color),
            linestyle=MplPlot._get_line_dash(trace.dash),
            linewidth=trace.width,
            label=trace.name,
        )
        if trace.showlegend:
            ax.legend()

        if trace.showannotation:
            ax.annotate(
                text=MplPlot._get_annotation_text(subplot.legendgroup_name, trace.name),
                xy=(trace.x[-1], trace.y.iloc[-1] + yshift),
                textcoords={"offset pixels": (0, annotation_yshift)},
                horizontalalignment="right",
                verticalalignment="bottom",
            )

    @staticmethod
    def _get_line_dash(dash: Dash) -> str | typing.Tuple:
        """
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html#linestyles
        """
        assert isinstance(dash, Dash)

        if dash == Dash.SOLID:
            ret = "solid"
        elif dash == Dash.DOT:
            ret = "dotted"
        elif dash == Dash.DASH:
            ret = "dashed"
        elif dash == Dash.DASHDOT:
            ret = "dashdot"
        elif dash == Dash.LONGDASH:
            ret = (5, (10, 3))
        elif dash == Dash.LONGDASHDOT:
            ret = (0, (10, 1, 1, 1))
        else:
            raise ValueError
        return ret

    @staticmethod
    def _get_annotation_text(legendgroup_name: str, name: str) -> str:
        assert isinstance(legendgroup_name, str)
        assert isinstance(name, str)

        if legendgroup_name:
            return f"{legendgroup_name}: {name}"
        return name

    @staticmethod
    def _add_subplot_to_ax(ax: mpl.axes.Axes, subplot: Subplot):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, Subplot)

        for t in subplot.traces:
            if isinstance(t, Scatter):
                MplPlot._add_scatter_to_ax(ax, subplot, t)
            elif isinstance(t, Step):
                MplPlot._add_step_to_ax(ax, subplot, t)
            elif isinstance(t, Candlestick):
                MplPlot._add_candlestick_to_ax(ax, t)
            else:
                raise ValueError(f"{type(t)} is an unsupported trace class.")

    def _add_lines_to_axs(self, axs: np.ndarray[np.ndarray]):
        assert isinstance(axs, np.ndarray)
        for ax in axs:
            assert isinstance(ax, np.ndarray)

        # add plot lines
        if self.lines:
            for line in self.lines:
                self._add_line_to_axs(axs, line)
        # add subplot lines
        for s in self.subplots:
            if s.lines:
                for line in s.lines:
                    MplPlot._add_line_to_ax(MplPlot._get_ax(axs, s.row, s.col), line)

    @staticmethod
    def _add_line_to_ax(ax: mpl.axes.Axes, line: Lines):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(line, Lines)

        if line.x is not None:
            for x in line.x:
                ax.axvline(
                    x=x,
                    linestyle=MplPlot._get_line_dash(line.dash),
                    color=Color.to_css(line.color),
                    linewidth=line.width,
                )
        if line.y is not None:
            for y in line.y:
                ax.axhline(
                    y=y,
                    linestyle=MplPlot._get_line_dash(line.dash),
                    color=Color.to_css(line.color),
                    linewidth=line.width,
                )

    def _add_line_to_axs(self, axs: np.ndarray[mpl.axes.Axes], line: Lines):
        assert isinstance(axs, np.ndarray)
        for ax in axs:
            assert isinstance(ax, np.ndarray)
        assert isinstance(line, Lines)

        for col in range(self.cols):
            for row in range(self.rows):
                MplPlot._add_line_to_ax(MplPlot._get_ax(axs, row + 1, col + 1), line)

    @staticmethod
    def _add_scatter_to_ax(ax: mpl.axes.Axes, subplot: Subplot, trace: Trace):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, Subplot)
        assert isinstance(trace, Trace)

        color = Color.to_css(trace.color)
        if trace.mode == "lines":
            linestyle = MplPlot._get_line_dash(trace.dash)
            linewidth = trace.width
            marker = None
        elif trace.mode == "lines+markers":
            linestyle = MplPlot._get_line_dash(trace.dash)
            linewidth = trace.width
            marker = MplPlot._get_marker(trace.marker_symbol)
        elif trace.mode == "markers":
            linestyle = "None"
            linewidth = None
            marker = MplPlot._get_marker(trace.marker_symbol)
        else:
            raise ValueError

        ax.plot(
            trace.x,
            trace.y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            label=trace.name,
        )

        if trace.showlegend:
            ax.legend()

        if trace.showannotation:
            ax.annotate(
                text=MplPlot._get_annotation_text(subplot.legendgroup_name, trace.name),
                xy=(trace.x[-1], trace.y.iloc[-1]),
                horizontalalignment="right",
                verticalalignment="bottom",
            )

    @staticmethod
    def _get_marker(marker_symbol: MarkerSymbol):
        assert isinstance(marker_symbol, MarkerSymbol)

        if marker_symbol == MarkerSymbol.TRIANGLE_UP:
            marker = "^"
        elif marker_symbol == MarkerSymbol.TRIANGLE_DOWN:
            marker = "v"
        elif marker_symbol is None:
            marker = None
        else:
            raise TypeError

        return marker

    @staticmethod
    def _add_candlestick_to_ax(ax: mpl.axes.Axes, trace: Trace):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(trace, Trace)

        # original interface does not require pd and does not mess up
        # default ax formatting
        # ref: https://github.com/matplotlib/mplfinance/issues/592
        mpfo.candlestick_ohlc(
            ax,
            zip(
                mpl.dates.date2num(trace.x.to_pydatetime()),
                trace.yopen,
                trace.yhigh,
                trace.ylow,
                trace.yclose,
            ),
            colorup=Color.to_css(Color.GREEN),
            colordown=Color.to_css(Color.RED),
            width=0.6,
        )

        if trace.showlegend:
            ax.legend()
