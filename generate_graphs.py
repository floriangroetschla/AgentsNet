import argparse
import os
import json
import random
from utils import relabel_and_name_vertices, generate_ba_graph, generate_ws_graph, generate_delaunay_triangulation
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

GRAPH_MODELS = {
    "ws": generate_ws_graph,
    "ba": generate_ba_graph,
    "dt": generate_delaunay_triangulation,
}


OUTPUT_DIR = "graphs"


def get_graph_path(graph_model, graph_size, num_sample):
    filename = f"graph_{graph_model}_{graph_size}_{num_sample}.json"
    return os.path.join(OUTPUT_DIR, filename)


def get_graph(graph_model, graph_size, num_sample):
    filepath = get_graph_path(graph_model, graph_size, num_sample)
    print(f"Loading graph from {filepath}")
    with open(filepath) as f:
        graph_dict = json.load(f)
    return json_graph.node_link_graph(graph_dict["graph"], edges="links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
    parser.add_argument("--samples_per_graph_model", type=int, default=3)
    parser.add_argument("--graph_sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(args.seed)
    graph_seeds = [random.randint(1, 10000) for _ in range(args.samples_per_graph_model)]
    for graph_model in args.graph_models:
        for graph_size in args.graph_sizes:
            for i in range(args.samples_per_graph_model):
                graph = GRAPH_MODELS[graph_model](graph_size, seed=graph_seeds[i])
                graph = relabel_and_name_vertices(graph)
                filepath = get_graph_path(graph_model, graph_size, i)

                with open(filepath, "w") as f:
                    json.dump({
                        "graph": json_graph.node_link_data(graph, edges="links"),
                        "num_nodes": len(graph.nodes()),
                        "diameter": nx.diameter(graph),
                        "max_degree": max(dict(graph.degree()).values()),
                        "graph_seed": graph_seeds[i],
                        "seed": args.seed,
                    }, f, indent=4)

                plt.figure()
                nx.draw(graph, with_labels=True)
                plt.savefig(filepath[:-5] + ".png")
                plt.close()
                print(f"Graph saved to {filepath}")

