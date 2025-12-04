import typing
import pandas as pd


def partitions(
    df: pd.DataFrame,
    expr: str | typing.Callable[[pd.Series], typing.Any],
    i_no_gaps: bool = True,
):
    """
    Iterate over pandas df and yield (val, i_start, i_end)
    whenever the expression value stays the same for consecutive rows.

    Parameters
    ----------
    df : pd.DataFrame
    expr : str | callable
        Column name or function(row) → value
    i_no_gaps : bool, default True
        If True, i_end is the index of the beginning of the next partition.
    """
    if isinstance(expr, str):
        # simple column
        vals = df[expr]
    else:
        # apply callable row-wise
        vals = df.apply(expr, axis=1)

    group_id = vals.ne(vals.shift()).cumsum()

    groups = [group for _, group in df.groupby(group_id)]
    for gi in range(len(groups)):
        i_start = groups[gi].index[0]

        if i_no_gaps and (gi + 1 < len(groups)):
            i_end = groups[gi + 1].index[0]
        else:
            i_end = groups[gi].index[-1]
        val = vals.loc[i_start]
        yield val, i_start, i_end
