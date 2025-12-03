import typing
import pandas as pd


def partitions(df: pd.DataFrame, expr: str | typing.Callable, i_no_gaps=True):
    """
    Iterate over pandas df and yield (val, i_start, i_end)
    whenever the expression value stays the same for consecutive rows.

    Parameters
    ----------
    df : pandas.DataFrame
    expr : str | callable
        Column name or function(row) → value
    i_no_gaps : bool, default True
        If True, i_end is the index of the beginning of the next partition.
    """
    if isinstance(expr, str):
        getter = lambda row: row[expr]
    else:
        getter = expr

    val_prev = None
    i_start = None
    i_prev = None

    for i, row in df.iterrows():
        val = getter(row)

        if val_prev is None:
            # first row
            i_start = i
        elif val != val_prev:
            # partition ended
            if i_no_gaps:
                yield val_prev, i_start, i
            else:
                yield val_prev, i_start, i_prev
            i_start = i

        val_prev = val
        i_prev = i

    # yield last partition
    if val_prev is not None:
        if i_no_gaps:
            yield val_prev, i_start, i
        else:
            yield val_prev, i_start, i_prev
