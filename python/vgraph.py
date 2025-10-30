"""Functions for working with graphs."""

import itertools
import networkx as nx


def paths_cross(pa: list, pb: list) -> bool:
    """
    Return true if two paths of a graph cross.
    """
    assert isinstance(pa, list)
    assert isinstance(pb, list)
    return len(set(pa) & set(pb)) > 0


def dag_node_is_root(g: nx.DiGraph, node) -> bool:
    """
    Return true if node of a DAG is a root.
    """
    assert isinstance(g, nx.DiGraph)
    return g.in_degree(node) == 0


def dag_node_is_leaf(g: nx.DiGraph, node) -> bool:
    """
    Return true if node of a DAG is a leaf.
    """
    assert isinstance(g, nx.DiGraph)
    return g.out_degree(node) == 0


def dag_all_paths(digraph: nx.DiGraph) -> list:
    """
    Return a list of all paths in a directed graph.
    If there are orhpan nodes, return those as well.
    """
    assert isinstance(digraph, nx.DiGraph)

    # ref: https://stackoverflow.com/questions/55711945/networkx-getting-all-possible-paths-in-dag
    res = []
    roots = []
    leaves = []
    for node in digraph.nodes:
        if dag_node_is_root(digraph, node):
            roots.append(node)
        elif dag_node_is_leaf(digraph, node):
            leaves.append(node)

    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(digraph, root, leaf):
                res.append(path)

    return res


def dag_all_orphan_nodes(g: nx.DiGraph) -> list:
    """
    Return a list of all orphan nodes.
    """
    assert isinstance(g, nx.DiGraph)

    res = []
    paths = dag_all_paths(g)

    for node in g.nodes:
        node_in_path = False
        for path in paths:
            if node in path:
                node_in_path = True
        if not node_in_path:
            res.append(node)

    return res


def dag_merge_parallel_paths(g: nx.DiGraph):
    """
    Merge parallel paths of a graph.

    Paths
        A->B->C->D
        A->X->Y->Z->C
    would be merged into
        A->B->X->Y->Z->C->D
    """

    def _get_two_shortest_parallel_paths(g: nx.DiGraph, pa: list, pb: list):
        parallel_paths = []

        common_nodes = set(pa) & set(pb)
        for na, nb in itertools.permutations(common_nodes, 2):
            simple_paths = list(nx.all_simple_paths(g, na, nb))
            # check if simple paths are parallel
            if len(simple_paths) > 1:
                # when recording the parallel paths, sort by lenght
                # so when processing we could begin with the shortest
                parallel_paths.append(sorted(simple_paths, key=len))

        if len(parallel_paths) == 0:
            # no parallel paths found
            return []

        # find shortest simple path
        shortest_parallel_paths = sorted(parallel_paths, key=lambda a: len(a[0]))[0]
        return shortest_parallel_paths[0], shortest_parallel_paths[1]

    def _merge_two_shortest_parallel_paths(g: nx.DiGraph, pa: list, pb: list):
        # this funcion assumes:
        # - both paths have only unique nodes
        # - both paths begin and end in the same nodes
        # - there are no common nodes between two paths
        #   except the beginning and the end
        assert pa[0] == pb[0] and pa[-1] == pb[-1]
        assert len(set(pa)) == len(pa) and len(set(pb)) == len(pb)
        assert len(set(pa) & set(pb)) == 2

        if len(pa) == 2:
            # from A->C and A->B->C:
            # remove A->C, keep A->B->C
            g.remove_edge(pa[0], pa[1])
        elif len(pb) == 2:
            g.remove_edge(pb[0], pb[1])
        else:
            # from A->B->C and A->X->C:
            # A->B->C => A->B
            g.remove_edge(pa[-2], pa[-1])
            # A->X->C => X->C
            g.remove_edge(pb[0], pb[1])
            # A->B, X->C => A->B->X->C
            g.add_edge(pa[-2], pb[1])

    assert isinstance(g, nx.DiGraph)

    # compare all paths between each other
    paths = dag_all_paths(g)
    continue_merging = True

    # merge pairs of parallel paths, starting with shortest,
    # until there are no parallel paths
    while continue_merging:
        # shortest paths are taken to not deal with common nodes between
        # two long parallel paths
        for pa, pb in itertools.permutations(paths, 2):
            spp = _get_two_shortest_parallel_paths(g, pa, pb)
            if spp != []:
                _merge_two_shortest_parallel_paths(g, spp[0], spp[1])
                break
        continue_merging = False


def dag_merge_leaves(g: nx.DiGraph):
    """
    Merge leaves right next to its node, branching off.

    Paths
        A->B->C->D
        A->X->Y
    would be merged into
        A->X->Y->B->C->D

    Paths
        A->B->C->D
        C->X->Y
    would be merged into
        A->B->C->X->Y->D
    """

    def _merge_root(g: nx.DiGraph, root: list, path: list):
        # pa: A->B->C->D
        # path: X->Y->C->D
        # pa merge with path: A->B->X->Y->C->D
        for i, root_node in enumerate(root):
            if root_node in path:
                # A->B->C->D => A->B
                g.remove_edge(root[i - 1], root[i])
                # A->B, X->Y->C->D => A->B->X->Y->C->D
                g.add_edge(root[i - 1], path[0])
                return

    def _merge_leaf(g: nx.DiGraph, leaf: list, path: list):
        # leaf: A->B->C->D
        # path: A->B->X->Y
        # merge: A->B->X->Y->C->D
        for i, leaf_node in enumerate(leaf):
            if leaf_node not in path:
                # A->B->C->D => C->D
                g.remove_edge(leaf[i - 1], leaf[i])
                # C->D, A->B->X->Y => A->B->X->Y->C->D
                g.add_edge(path[-1], leaf[i])
                return

    assert isinstance(g, nx.DiGraph)

    while True:
        paths = dag_all_paths(g)
        if len(paths) < 2:
            break
        cross_paths_found = False
        for pa, pb in itertools.permutations(paths, 2):
            if paths_cross(pa, pb):
                # A->B, A->X
                # A->B->C, A->B->X
                # A->Z, Y->Z
                # A->Y->Z, X->Y->Z
                cross_paths_found = True
                if g.in_degree(pa[0]) == 0 and pa[0] not in pb:
                    _merge_root(g, pa, pb)
                else:
                    _merge_leaf(g, pa, pb)
                break
        if not cross_paths_found:
            return


def dag_join_all_paths(g: nx.DiGraph):
    """
    Join all paths into one.
    """
    assert isinstance(g, nx.DiGraph)

    paths = dag_all_paths(g)
    if len(paths) > 1:
        for pa, pb in itertools.pairwise(paths):
            # A->B, C->D => A->B->C->D
            g.add_edge(pa[-1], pb[0])
