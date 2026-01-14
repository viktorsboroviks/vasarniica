"""
Technical Analysis.
"""

import numpy as np
import pandas as pd


def trailing_atr(
    comp_f,
    ref: pd.Series,
    atr: pd.Series,
    atr_mult: float = 1.0,
) -> pd.Series:
    """
    examples:
        data["trailing ATR min"] = vta.trailing_atr(
            min, data["Low"], -data["ATR"], ATR_MULTIPLIER
        )

        data["trailing ATR max"] = vta.trailing_atr(
            max, data["High"], data["ATR"], ATR_MULTIPLIER
        )
    """
    assert len(ref) == len(atr)
    assert len(ref) > 0
    res = pd.Series(index=ref.index, dtype=float)
    res.iloc[0] = ref.iloc[0] + atr_mult * atr.iloc[0]
    for i in range(1, len(ref)):
        res.iloc[i] = comp_f(
            ref.iloc[i] + atr_mult * atr.iloc[i],
            res.iloc[i - 1],
        )
    return res


def trailing_atr(
    atr_ref: pd.Series,
    atr: pd.Series,
    atr_mult: float,
    atr_up: bool,
    reset_ref: pd.Series,
    reset_below: bool,
):
    """
    example 1 - trailing atr low:
    data_df["trailing_atr2_lo"] = vta.trailing_atr(
        atr_ref=data_df["Low"],
        atr=atr,
        atr_mult=2.0,
        atr_up=True,
        reset_ref=data_df["Low"],
        reset_below=True,
    )

    example 2 - chandelier low:
    data_df["chandelier_atr2_lo"] = vta.trailing_atr(
        atr_ref=data_df["High"],
        atr=atr,
        atr_mult=2.0,
        atr_up=True,
        reset_ref=data_df["Low"],
        reset_below=True,
    )
    """
    assert len(atr_ref) == len(reset_ref)
    assert len(atr_ref) == len(atr)
    assert len(atr_ref) > 0
    res = pd.Series(index=atr_ref.index, dtype=float)
    prev_ta = atr_ref.iloc[0] - atr_mult * atr.iloc[0]
    for i in range(len(atr_ref)):
        if atr_up:
            ta = max(atr_ref.iloc[i] - atr_mult * atr.iloc[i], prev_ta)
        else:
            ta = min(atr_ref.iloc[i] - atr_mult * atr.iloc[i], prev_ta)

        if reset_below:
            ta = min(reset_ref.iloc[i], ta)
        else:
            ta = max(reset_ref.iloc[i], ta)

        res.iloc[i] = ta
        prev_ta = ta
    return res


def sharpe_from_equity(equity_series: pd.Series, scaling_factor: float) -> float:
    """
    Calculate the Sharpe Ratio from an equity curve, scaled by a factor.

    The calculation follows the standard formula:
    Sharpe = (Mean Excess Return / Std Dev of Return) * sqrt(Scaling Factor)

    Note: This implementation assumes a risk-free rate of 0.0.

    Args:
        equity_series: Time series of portfolio equity (cash + open positions).
        scaling_factor: The factor used to scale the ratio.
                        Should contain number of active periods per period
                        (usually year), when market was open / strategy was running.

    Returns:
        The scaled Sharpe Ratio. Returns 0.0 if standard deviation is zero.

    Ref:
        For annualized value:
        < 1.0       - poor
        1.0 to 1.5  - acceptable
        1.5 to 2.0  - good
        > 2.0       - excellent
    """
    # make sure calculating on numeric type
    # convert invalid values to NaN and drop them
    equity = pd.to_numeric(equity_series, errors="coerce").dropna()
    assert len(equity) > 2

    returns = equity.pct_change().dropna()
    assert len(returns) > 2

    # set degrees of freedom to 1 for sample standard deviation
    # not total population
    sigma = float(returns.std(ddof=1))
    if sigma == 0.0 or not np.isfinite(sigma):
        return 0.0

    return returns.mean() / sigma * np.sqrt(scaling_factor)
