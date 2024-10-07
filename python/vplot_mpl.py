# pylint: disable=R0801
"""
Plotting.
"""

import typing
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpfo
import vplot


class MplPlot(vplot.Plot):
    """
    matplotlib plot containing one or several Subplots.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        subplots: list[vplot.Subplot],
        share_x: bool = True,
        share_y: bool = False,
        lines: list[vplot.Lines] = None,
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
        vplot.Plot.__init__(
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
            if isinstance(s, vplot.LogicSignalSubplot):
                self._add_logic_signal_subplot_to_ax(ax, s)
            elif isinstance(s, vplot.Subplot):
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
    def _add_logic_signal_subplot_to_ax(ax: mpl.axes.Axes, subplot: vplot.Subplot):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, vplot.Subplot)

        steps = []
        for s in subplot.steps:
            if isinstance(s, vplot.Step):
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
        subplot: vplot.Subplot,
        trace: vplot.Trace,
        yshift: float = 0,
        annotation_yshift: float = 0,
    ):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, vplot.Subplot)
        assert isinstance(trace, vplot.Trace)
        assert isinstance(yshift, (int, float))
        assert isinstance(annotation_yshift, (int, float))

        ax.step(
            x=trace.x,
            y=trace.y + yshift,
            color=vplot.Color.to_css(trace.color),
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
    def _get_line_dash(dash: vplot.Dash) -> str | typing.Tuple:
        """
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html#linestyles
        """
        assert isinstance(dash, vplot.Dash)

        if dash == vplot.Dash.SOLID:
            ret = "solid"
        elif dash == vplot.Dash.DOT:
            ret = "dotted"
        elif dash == vplot.Dash.DASH:
            ret = "dashed"
        elif dash == vplot.Dash.DASHDOT:
            ret = "dashdot"
        elif dash == vplot.Dash.LONGDASH:
            ret = (5, (10, 3))
        elif dash == vplot.Dash.LONGDASHDOT:
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
    def _add_subplot_to_ax(ax: mpl.axes.Axes, subplot: vplot.Subplot):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, vplot.Subplot)

        for t in subplot.traces:
            if isinstance(t, vplot.Scatter):
                MplPlot._add_scatter_to_ax(ax, subplot, t)
            elif isinstance(t, vplot.Step):
                MplPlot._add_step_to_ax(ax, subplot, t)
            elif isinstance(t, vplot.Candlestick):
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
    def _add_line_to_ax(ax: mpl.axes.Axes, line: vplot.Lines):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(line, vplot.Lines)

        if line.x is not None:
            for x in line.x:
                ax.axvline(
                    x=x,
                    linestyle=MplPlot._get_line_dash(line.dash),
                    color=vplot.Color.to_css(line.color),
                    linewidth=line.width,
                )
        if line.y is not None:
            for y in line.y:
                ax.axhline(
                    y=y,
                    linestyle=MplPlot._get_line_dash(line.dash),
                    color=vplot.Color.to_css(line.color),
                    linewidth=line.width,
                )

    def _add_line_to_axs(self, axs: np.ndarray[mpl.axes.Axes], line: vplot.Lines):
        assert isinstance(axs, np.ndarray)
        for ax in axs:
            assert isinstance(ax, np.ndarray)
        assert isinstance(line, vplot.Lines)

        for col in range(self.cols):
            for row in range(self.rows):
                MplPlot._add_line_to_ax(MplPlot._get_ax(axs, row + 1, col + 1), line)

    @staticmethod
    def _add_scatter_to_ax(
        ax: mpl.axes.Axes, subplot: vplot.Subplot, trace: vplot.Trace
    ):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(subplot, vplot.Subplot)
        assert isinstance(trace, vplot.Trace)

        color = vplot.Color.to_css(trace.color)
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
    def _get_marker(marker_symbol: vplot.MarkerSymbol):
        assert isinstance(marker_symbol, vplot.MarkerSymbol)

        if marker_symbol == vplot.MarkerSymbol.TRIANGLE_UP:
            marker = "^"
        elif marker_symbol == vplot.MarkerSymbol.TRIANGLE_DOWN:
            marker = "v"
        elif marker_symbol is None:
            marker = None
        else:
            raise TypeError

        return marker

    @staticmethod
    def _add_candlestick_to_ax(ax: mpl.axes.Axes, trace: vplot.Trace):
        assert isinstance(ax, mpl.axes.Axes)
        assert isinstance(trace, vplot.Trace)

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
            colorup=vplot.Color.to_css(vplot.Color.GREEN),
            colordown=vplot.Color.to_css(vplot.Color.RED),
            width=0.6,
        )

        if trace.showlegend:
            ax.legend()
