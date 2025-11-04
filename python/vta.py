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
