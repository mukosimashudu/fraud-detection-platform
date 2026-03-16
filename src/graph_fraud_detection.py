import networkx as nx
import pandas as pd

from src.database import read_transactions


def prepare_graph_data(sample_size=3000):
    """
    Load transactions and create simulated banking entities.
    Since the credit card dataset is anonymized, we derive
    pseudo-entities for graph fraud detection.
    """

    df = read_transactions().copy()

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Simulated entities derived from anonymized features
    df["card_id"] = "card_" + df.index.astype(str)
    df["merchant_id"] = "merchant_" + (df["Amount"] // 10).astype(int).astype(str)
    df["device_id"] = "device_" + (df["V1"] * 100).round().astype(int).astype(str)
    df["ip_id"] = "ip_" + (df["V2"] * 100).round().astype(int).astype(str)

    return df


def build_fraud_graph(sample_size=3000):
    """
    Build graph:
    card -> merchant
    card -> device
    device -> ip
    """

    df = prepare_graph_data(sample_size=sample_size)

    G = nx.Graph()

    for _, row in df.iterrows():
        card = row["card_id"]
        merchant = row["merchant_id"]
        device = row["device_id"]
        ip_addr = row["ip_id"]

        G.add_node(card, node_type="card")
        G.add_node(merchant, node_type="merchant")
        G.add_node(device, node_type="device")
        G.add_node(ip_addr, node_type="ip")

        G.add_edge(card, merchant)
        G.add_edge(card, device)
        G.add_edge(device, ip_addr)

    return G, df


def detect_suspicious_communities(sample_size=3000, min_size=5):
    """
    Detect suspicious graph communities using modularity-based clustering.
    """

    G, df = build_fraud_graph(sample_size=sample_size)

    communities = list(nx.algorithms.community.greedy_modularity_communities(G))

    suspicious = [c for c in communities if len(c) >= min_size]

    results = []
    for idx, community in enumerate(suspicious, start=1):
        subgraph = G.subgraph(community)

        cards = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "card"]
        merchants = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "merchant"]
        devices = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "device"]
        ips = [n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "ip"]

        results.append({
            "community_id": idx,
            "total_nodes": subgraph.number_of_nodes(),
            "edges": subgraph.number_of_edges(),
            "cards": len(cards),
            "merchants": len(merchants),
            "devices": len(devices),
            "ips": len(ips),
            "sample_nodes": list(community)[:10]
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["total_nodes", "edges"],
        ascending=False
    ) if results else pd.DataFrame(
        columns=["community_id", "total_nodes", "edges", "cards", "merchants", "devices", "ips", "sample_nodes"]
    )

    return results_df


def graph_summary(sample_size=3000):
    G, _ = build_fraud_graph(sample_size=sample_size)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G),
        "density": nx.density(G)
    }


if __name__ == "__main__":
    summary = graph_summary()
    print("Graph Summary:", summary)

    suspicious_df = detect_suspicious_communities()
    print(suspicious_df.head())