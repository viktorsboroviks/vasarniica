"""Generic functions."""

import copy
import dataclasses
import typing
import networkx as nx
import vgraph


# @dataclasses.dataclass(frozen=True)
@dataclasses.dataclass
class Val:
    """
    Value.
    """

    dtype: type
    default_value: typing.Any = None
    min_value: typing.Any = None
    max_value: typing.Any = None


def remove_duplicates(lst: list) -> list:
    """
    Return a list without duplicates while preserving the order of elements.
    """
    return list(dict.fromkeys(lst))


def merge_lists(lists: list[typing.Iterable]) -> typing.Iterable:
    """
    Merge multiple lists while preserving the original order of their elements.
    Each list should have no duplicating elements.
    """
    # input check
    assert isinstance(lists, list)
    assert len(lists) > 0
    for lst in lists:
        assert isinstance(lst, typing.Iterable)
        assert len(lst) > 0
        assert len(lst) == len(
            set(lst)
        ), f"Duplicate values detected within a list: {lst}"

    # create dag from paths
    g = nx.DiGraph()
    for lst in lists:
        if len(lst) == 1:
            g.add_node(lst[0])
        else:
            nx.add_path(g, lst)

    # dag check: error if loop created
    try:
        nx.find_cycle(g)
        raise ValueError("Merge resulted in a cyclic path")
    except nx.exception.NetworkXNoCycle:
        pass

    # remove all duplicating paths and shortcuts
    g = nx.transitive_reduction(g)

    vgraph.dag_merge_parallel_paths(g)
    vgraph.dag_merge_leaves(g)
    # at this point only subgraphs could remain, join them
    vgraph.dag_join_all_paths(g)

    # orphan nodes are not joinded with other paths
    res = vgraph.dag_all_orphan_nodes(g)
    # it is possible that there is no single path except orphan nodes
    all_paths = vgraph.dag_all_paths(g)
    if len(all_paths) > 0:
        res = res + all_paths[0]
    return res


def copy_list(lst) -> list:
    """
    Copy a list of objects by creating shallow copies of each object.

    Useful when the list contains mutable objects that should not be shared
    and deepcopy is not necessary or excessive.
    """
    return [copy.copy(c) for c in lst]


def it_flatten_dict(dct: dict) -> typing.Iterator[dict]:
    """
    Flatten a dictionary with list values into multiple dictionaries.

    Useful for slicing one multidimensional set of parameters
    into multiple sets of flat one-dimensional parameters.

    Example:
        initial_dict = {
            "a": [1, 2],
            "b": "x",
            "c": [True, False],
        }
        for d in it_flatten_dict(initial_dict):
            print(d)

        # Output:
        {'a': 1, 'b': 'x', 'c': True}
        {'a': 1, 'b': 'x', 'c': False}
        {'a': 2, 'b': 'x', 'c': True}
        {'a': 2, 'b': 'x', 'c': False}
    """
    assert isinstance(dct, dict)
    ret_dct = dct.copy()
    result_yielded = False
    for k, v in dct.items():
        if isinstance(v, list):
            assert len(v) > 0
            # do not yield single-value lists as it would cause
            # unnecessary recursion
            if len(v) == 1:
                ret_dct[k] = v[0]
                continue

            result_yielded = True
            for list_v in v:
                rec_ret_dct = ret_dct.copy()
                rec_ret_dct[k] = list_v
                yield from it_flatten_dict(rec_ret_dct)
            return

    if not result_yielded:
        yield ret_dct
