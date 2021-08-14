import networkx as nx
import os
import linecache
import pickle
import numpy as np
import joblib

class KGBrowser():
    """
    Utilities for browsing/interacting with processed data from SMCDC 2021
    Challenge 2.
    """
    def __init__(self):
        self.KG = nx.read_gpickle("data/graph/training_KG_concepts.gpickle")
        self.G = nx.convert_node_labels_to_integers(self.KG)
        self.all_nodes = list(self.KG.nodes)
        self.betweenness_dict_10000 = None
        self.embeddings = None
        # classifiers
        self.deepwalk_mlp_69336 = None
        self.deepwalk_lr_69336 = None
        self.clf_geodesic_69336 = None

    def link_prob(self, node1, node2, method = "embeddings_lr"):
        """
        Estimates the probability of an edge between concept node1 and node2.

        Parameters
        ----------
        node1 : str
            source node
        node2 : str
            destination node
        method : str
            one of the following:
                - "embeddings_lr" : Uses the DeepWalk embedding-based
                    LR classifier to estimate the probability of an edge between
                    node1 and node2.
                - "embeddings_mlp" : Uses the DeepWalk embedding-based
                    MLP classifier to estimate the probability of an edge between
                    node1 and node2.
                - "geodesic" : TBD

        Raises
        ------
        ValueError : when method is unrecognized
        """
        if method == "embeddings_lr":
            if self.deepwalk_lr_69336 is None:
                self.deepwalk_lr_69336 = joblib.load(
                    "classifiers/deepwalk_lr_69336.sav"
                )
            if self.embeddings is None:
                self.embeddings = np.load(
                    "data/embeddings/training_graph_embeddings.npy"
                )
            return self._link_prob_embeddings(
                node1,
                node2,
                classifier = "lr"
            )
        if method == "embeddings_mlp":
            if self.deepwalk_mlp_69336 is None:
                self.deepwalk_mlp_69336 = joblib.load(
                    "classifiers/deepwalk_mlp_69336.sav"
                )
            if self.embeddings is None:
                self.embeddings = np.load(
                    "data/embeddings/training_graph_embeddings.npy"
                )
            return self._link_prob_embeddings(
                node1,
                node2,
                classifier = "mlp"
            )
        elif method == "geodesic":
            if self.clf_geodesic_69336 is None:
                self.clf_geodesic_69336 = joblib.load(
                    "classifiers/geodesic_gbc_69336.sav"
                )
            return self._link_prob_geodesic(node1, node2)
        else:
            raise ValueError("Unrecognized method.")
        return None

    def _link_prob_embeddings(self, node1, node2, classifier = "lr"):
        """
        Uses the DeepWalk embedding-based classifier to estimate the probability
        of an edge between node1 and node2.

        Parameters
        ----------
        node1 : str
            source node
        node2 : str
            destination node
        classifier : str
            one of the following:
            - "lr" : Uses the DeepWalk embedding-based
                LR classifier to estimate the probability of an edge between
                node1 and node2.
            - "mlp" : Uses the DeepWalk embedding-based
                MLP classifier to estimate the probability of an edge between
                node1 and node2.

        Raises
        ------
        ValueError : when classifier is unrecognized
        """
        node1 = self.all_nodes.index(node1)
        node2 = self.all_nodes.index(node2)
        if classifier == "mlp":
            X = [np.concatenate([self.embeddings[node1], self.embeddings[node2]])]
            return self.deepwalk_mlp_69336.predict_proba(X)[:, 1][0]
        elif classifier == "lr":
            X = [np.multiply(self.embeddings[node1], self.embeddings[node2])]
            return self.deepwalk_lr_69336.predict_proba(X)[:, 1][0]
        else:
            raise ValueError("Unrecognized classifier.")


    def _link_prob_geodesic(self, node1, node2):
        """
        Uses SPL and estimated betweenness centrality to estimate the probability
        of an edge between node1 and node2.

        Parameters
        ----------
        node1 : str
            source node
        node2 : str
            destination node
        """
        spl = self.retrieve_spl(node1, node2, type = "label")
        if spl != spl:
            spl = 5
        X = [
            [
                self.retrieve_betweenness(node1),
                self.retrieve_betweenness(node2),
                spl
            ]
        ]
        return self.clf_geodesic_69336.predict_proba(X)[:, 1][0]

    def retrieve_spl(self, node1, node2, type = "label"):
        """
        Low-overhead retrieval of pre-computed shortest path length from node1
        to node2 in the Training Concept Graph.

        Parameters
        ----------
        node1 : str or int
            source node
        node2 : str or int
            destination node
        type : str
            one of the following:
                - "index" : node1, node2 are provided as the graph index of the
                    node
                - "label" : node1, node2 are provided as concept labels
                    (e.g. 'C4071858')

        Returns
        -------
        ~output~ : int
            shortest path length
        """
        if type == "label":
            node1 = self.all_nodes.index(node1)
            node2 = self.all_nodes.index(node2)
        elif type != "index":
            raise ValueError("Unrecognized type.")

        file_ranges = [
            [int(i) for i in s.split("_")[:2]] \
            for s \
            in os.listdir("data/shortest_paths") \
            if not s.endswith(".md")
        ]
        for r in file_ranges:
            if r[0]<=node1<=r[1]:
                file = "data/shortest_paths/%d_%d_lengths.txt"%(r[0], r[1])
                break
        line = node1 - r[0] + 1
        lengths = linecache.getline(
            file,
            line)
        return [
            int(i) \
            if i != "nan" \
            else np.nan
            for i \
            in lengths[1:-2].split(", ")
        ][node2]

    def retrieve_betweenness(self, node, normalized = True):
        """
        Retrieve pre--computed estimated normalized betweenness centrality

        Parameters
        ----------
        node : str
            the label (e.g. 'C4071858') for the node you want to retrieve the
                statistic for
        normalized : bool
            whether you want the normalized value or not.
        """
        if self.betweenness_dict_10000 is None:
            self.betweenness_dict_10000 = np.load(
                "data/betweenness/betweenness_dict_10000.npy",
                allow_pickle=True
            ).item()
        if normalized:
            return self.betweenness_dict_10000[node]
        N = len(self.G.nodes)
        return self.betweenness_dict_10000[node]*(N-1)*(N-2)/2
