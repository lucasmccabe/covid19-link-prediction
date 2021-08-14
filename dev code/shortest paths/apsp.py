import networkx as nx
import numpy as np
import time
import json
import itertools
import sys
import igraph as ig
import os

# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import floyd_warshall

if __name__ == "__main__":
    arguments = sys.argv
    node_range = range(
        int(arguments[1]),
        int(arguments[2])+1
    )
    print(
        "Finding shortest paths in range [%d, %d]" \
        %(node_range[0], node_range[-1])
    )

KG = nx.read_gpickle("training_KG_concepts2.gpickle")

iKG = ig.Graph.from_networkx(KG)

start = time.time()

for node1 in node_range:
    # shortest_paths = iKG.get_all_shortest_paths(node1)
    shortest_path_lengths = [
        len(path) \
        if path \
        else np.nan \
        for path in iKG.get_shortest_paths(node1)]

    # with open(
    #             "D:/smdc-2021-covid-kg/shortest_paths/%s_%s_paths.txt" \
    #             %(node_range[0], node_range[-1]),
    #             "a"
    #         ) as f:
    #     f.write(str(shortest_paths)+"\n")
    with open(
                "D:/smdc-2021-covid-kg/shortest_paths/%s_%s_lengths.txt" \
                %(node_range[0], node_range[-1]),
                "a"
            ) as f:
        f.write(str(shortest_path_lengths)+"\n")

print((time.time() - start)*(46669/len(node_range))/(60*60*24))
