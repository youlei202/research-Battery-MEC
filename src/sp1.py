import math
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class NetworkFlowProblem(object):

    def __init__(self, arrival_rates: List, weights: np.array) -> None:

        self.n = len(arrival_rates)
        self.demand = np.sum(arrival_rates)

        self.G = nx.DiGraph()

        self.G.add_node('S', demand=-self.demand)
        self.G.add_node('D', demand=self.demand)

        for i in range(self.n):
            self.G.add_node(f'L{i}')
            self.G.add_node(f'R{i}')

        for i in range(self.n):
            self.G.add_edge('S', f'L{i}', capacity=arrival_rates[i])

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.G.add_edge(f'L{i}', f'R{j}', weight=weights[i][j])

        for i in range(self.n):
            self.G.add_edge(f'R{i}', 'D')

        self.G.add_edge('S', 'D')

        self.capacities = {(u, v): c for u, v,
                           c in self.G.edges(data="capacity")}
        self.flow = {}
        self.X = np.zeros((self.n, self.n))

        self.drawing_settings = {}
        self.drawing_settings['node_pos'] = self._get_node_positions()
        self.drawing_settings['node_colors'] = ["skyblue" if i in {
            "S", "D"} else "lightgray" for i in self.G.nodes]

    def solve(self) -> Tuple[int, Dict]:
        self.flow_cost, self.flow_dict = nx.network_simplex(self.G)
        self._get_flows()

        return self.flow_cost, self.flow_dict

    def visualize_network(self, figsize: Optional[Tuple[int, int]] = (16, 9)) -> None:
        _, ax = plt.subplots(figsize=figsize)

        node_pos = self.drawing_settings['node_pos']
        node_colors = self.drawing_settings['node_colors']

        nx.draw(self.G, node_pos, ax=ax, with_labels=True,
                node_color=node_colors, node_shape='s')
        nx.draw_networkx_edge_labels(
            self.G, node_pos, ax=ax, edge_labels=self.capacities)

    def visualize_flow(self, figsize: Optional[Tuple[int, int]] = (16, 9)) -> Any:
        """Visualize flow returned by the `check_valid_flow` funcion."""
        flow_graph = self._check_valid_flow()

        fig, ax = plt.subplots(figsize=figsize)
        node_pos = self.drawing_settings['node_pos']
        node_colors = self.drawing_settings['node_colors']

        # Draw the full graph for reference
        nx.draw(
            self.G, node_pos, ax=ax, node_color=node_colors, edge_color="lightgrey", with_labels=True
        )

        # Draw the example flow on top
        flow_nc = [
            "skyblue" if n in {"S", "D"} else flow_graph.nodes[n].get(
                "color", "lightgrey")
            for n in flow_graph
        ]
        flow_ec = [flow_graph[u][v].get("edgecolor", "black")
                   for u, v in flow_graph.edges]
        edge_labels = {(u, v): lbl for u, v,
                       lbl in flow_graph.edges(data="label")}
        nx.draw(flow_graph, node_pos, ax=ax,
                node_color=flow_nc, edge_color=flow_ec)
        nx.draw_networkx_edge_labels(
            self.G, node_pos, edge_labels=edge_labels, ax=ax)

        return fig, ax

    def _get_flows(self) -> None:
        self.flow = {}
        for L, value in self.flow_dict.items():
            for R, w in value.items():
                if w > 0:
                    self.flow[(L, R)] = w
        self.X = np.matrix([[self.flow.get((f'L{i}', f'R{j}'), 0) for j in range(
            self.n)] for i in range(self.n)])

    def _get_node_positions(self) -> Dict:
        node_pos = {}

        y_pos_step = 0.2
        x_step = 1
        for i in range(self.n):
            node_pos[f'L{i}'] = (x_step*1, y_pos_step * i)
            node_pos[f'R{i}'] = (x_step*2, y_pos_step * i)

        node_pos['S'] = (0, y_pos_step * self.n)
        node_pos['D'] = (x_step*3, y_pos_step * self.n)

        return node_pos

    def _check_valid_flow(self) -> nx.DiGraph:
        source_node = 'S'
        target_node = 'D'

        H = nx.DiGraph()
        H.add_edges_from(self.flow.keys())

        for (u, v), f in self.flow.items():
            capacity = self.G[u][v].get("capacity", np.inf)
            H[u][v]["label"] = f"{f}/{capacity}"
            # Capacity constraint
            if f > self.G[u][v].get("capacity", np.inf):
                H[u][v]["edgecolor"] = "red"
                print(
                    f"Invalid flow: capacity constraint violated for edge ({u!r}, {v!r})")
            # Conservation of flow
            if v not in {source_node, target_node}:
                incoming_flow = sum(
                    self.flow[(i, v)] if (i, v) in self.flow else 0 for i in self.G.predecessors(v)
                )
                outgoing_flow = sum(
                    self.flow[(v, o)] if (v, o) in self.flow else 0 for o in self.G.successors(v)
                )
                if not math.isclose(incoming_flow, outgoing_flow):
                    print(
                        f"Invalid flow: flow conservation violated at node {v}")
                    H.nodes[v]["color"] = "red"
        return H
