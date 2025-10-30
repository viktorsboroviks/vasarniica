"""
Finance computations.
"""

import copy
import os
import datetime
import typing
import numpy as np
import pandas as pd
import yfinance as yf
import vcore
import vlog


CACHE_DIR = "__vfin_cache__"


def fetch_ticker(
    ticker,
    interval: str = "1d",
    start=None,
    end=None,
    cache_dir=os.path.join(os.getcwd(), CACHE_DIR),
) -> pd.DataFrame:
    """
    Fetch ticker data from yfinance.

    Args:
        ticker: String with a ticker name or ISIN.
        interval: (optional) String, default is '1d'.
        start: (optional) Start datetime.
        end: (optional) End datetime.
        cache_dir: (optional) String with a path to a cache dir
                   with downloaded csv files.
                   To disable cache set to None.

    Returns:
        pd.DataFrame

    yfinance docs:
        https://github.com/ranaroussi/yfinance/wiki/Tickers
    """

    # TODO: reconsider how this fetch_ticker and Data class are organized
    #       for a more graceful interface
    # TODO: interrupt if no network
    # TODO: reuse subset of already fetched data if possible
    # TODO: extend/merge fetched data cache into one file for every ticker+scale
    def get_filename(ticker, interval, start, end):
        now = datetime.datetime.now()
        if end is None:
            # for intervals >1d consider daily scale to reduce caching
            if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
                end = datetime.datetime(year=now.year, month=now.month, day=now.day)
            else:
                end = now
        if start is None:
            start = datetime.datetime(1990, 1, 1)
        return f"{ticker}_{interval}_{start}_{end}.csv"

    def read_cache(ticker, interval, start, end, cache_dir):
        if not os.path.isdir(cache_dir):
            return None

        filename = get_filename(ticker, interval, start, end)
        if not os.path.isfile(os.path.join(cache_dir, filename)):
            vlog.debug("cache file not found")
            return None
        # parameters are required to parse dates as indexes
        vlog.debug(f"cache file found, reading {ticker} data")
        return pd.read_csv(
            os.path.join(cache_dir, filename), index_col=0, parse_dates=True
        )

    def read_yfinance(ticker, interval, start, end):
        def flatten_pd_multiindex(data):
            data.columns = data.columns.to_flat_index()
            data.columns = [col[0] for col in data.columns]
            data.columns.name = None

        vlog.debug(f"downloading {ticker} data")
        data = yf.download(ticker, interval=interval, start=start, end=end)
        flatten_pd_multiindex(data)

        # pylint: disable=line-too-long
        # detecting yfinance download errors is not trivial
        # see: https://stackoverflow.com/questions/67690778/how-to-detect-failed-downloads-using-yfinance
        # pylint: disable=protected-access
        download_errors = yf.shared._ERRORS.keys()
        if download_errors:
            raise ConnectionError(f"download error: {download_errors}")
        return data

    # pylint: disable=too-many-arguments
    def save_cache(data, ticker, interval, start, end, cache_dir):
        if not cache_dir:
            return
        filename = get_filename(ticker, interval, start, end)
        if os.path.isfile(os.path.join(cache_dir, filename)):
            return
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        data.to_csv(os.path.join(cache_dir, filename))
        vlog.debug(f"cache file created {os.path.join(cache_dir, filename)}")

    if cache_dir:
        data = read_cache(ticker, interval, start, end, cache_dir)
    if data is None:
        data = read_yfinance(ticker, interval, start, end)
    assert isinstance(data, pd.DataFrame)
    if cache_dir:
        save_cache(data, ticker, interval, start, end, cache_dir)

    vlog.debug(f"data for {ticker} fetched")
    return data


def get_buy_price(src_price: float, slippage_pct: float) -> float:
    """
    Get buy price, using slippage.
    """
    assert isinstance(src_price, float)
    assert isinstance(slippage_pct, float)

    return src_price * (100 + slippage_pct) / 100.0


def get_sell_price(src_price: float, slippage_pct: float) -> float:
    """
    Get sell price, using slippage.
    """
    assert isinstance(src_price, float)
    assert isinstance(slippage_pct, float)

    return src_price * (100 - slippage_pct) / 100.0


def get_short_price(
    in_slice: np.ndarray,
    col_name: str,
    prev_short_price: float,
    index: int = -1,
    first_value: float = 1.0,
) -> float:
    """
    Get short price.
    """
    assert isinstance(in_slice, np.ndarray)
    assert len(in_slice) > 0
    assert isinstance(col_name, str)
    assert isinstance(index, int)
    assert isinstance(prev_short_price, float)

    if len(in_slice) == -index:
        return first_value

    return in_slice[index - 1][col_name] / in_slice[index][col_name] * prev_short_price


def get_xy_signal_rise(
    index: typing.Iterable, signal_table: typing.Iterable, value_table: typing.Iterable
) -> tuple[typing.Iterable, typing.Iterable]:
    """
    Return x, y coordinate pairs from value_table when signal_table is True.
    """
    assert len(index) == len(signal_table) == len(value_table)

    x = []
    y = []
    prev_signal = False
    # pylint: disable=consider-using-enumerate
    for i in range(0, len(index)):
        signal = signal_table[i]
        if signal and not prev_signal:
            x.append(index[i])
            y.append(value_table[i])
        prev_signal = signal
    return x, y


# pylint: disable=too-few-public-methods
class Instrument:
    """
    Description of the financial instrument used in a Strategy.
    """

    DEFAULT_KWARGS = {
        "type": vcore.Val(str, default_value=""),
        "ticker": vcore.Val(str, default_value=""),
        "ISIN": vcore.Val(str, default_value=""),
        "currency": vcore.Val(str, default_value=""),
        "slippage_pct": vcore.Val(float, default_value=0),
        "margin_pct": vcore.Val(float, default_value=0, min_value=0),
        "min_traded_quantity": vcore.Val(float, default_value=0, min_value=0),
        "max_hold_quantity": vcore.Val(float, default_value=0, min_value=0),
        "fee_hold_annual_pct": vcore.Val(
            float, default_value=0, min_value=0, max_value=100
        ),
        "fee_fx_pct": vcore.Val(float, default_value=0, min_value=0, max_value=100),
        "fee_fx_on_result_pct": vcore.Val(
            float, default_value=0, min_value=0, max_value=100
        ),
        "fee_sell_pct": vcore.Val(float, default_value=0, min_value=0, max_value=100),
        "fee_sell_per_share": vcore.Val(float, default_value=0, min_value=0),
        "interest_hold_overnight": vcore.Val(float, default_value=0),
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.DEFAULT_KWARGS:
                raise KeyError(
                    f"Invalid key: {k}. Allowed keys are: {self.DEFAULT_KWARGS.keys()}"
                )
            if self.DEFAULT_KWARGS[k].dtype is float:
                assert isinstance(v, (float, int))
            else:
                assert isinstance(v, self.DEFAULT_KWARGS[k].dtype)

            if self.DEFAULT_KWARGS[k].min_value is not None:
                assert v >= self.DEFAULT_KWARGS[k].min_value

            if self.DEFAULT_KWARGS[k].max_value is not None:
                assert v <= self.DEFAULT_KWARGS[k].max_value

        self.__dict__.update(kwargs)

        # set remaining parameters to default values
        default_kwargs = copy.copy(self.DEFAULT_KWARGS)
        for k in self.DEFAULT_KWARGS:
            if k in kwargs:
                del default_kwargs[k]
        for k, v in default_kwargs.items():
            setattr(self, k, v.default_value)
