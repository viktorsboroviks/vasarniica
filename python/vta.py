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
