import networkx as nx
import pandas as pd

from src.database import read_transactions


def build_transaction_graph():

    df = read_transactions()

    G = nx.Graph()

    for i, row in df.iterrows():

        card = f"card_{i}"
        merchant = f"merchant_{int(row['Amount'] // 10)}"
        device = f"device_{int(row['V1'] * 100)}"
        ip = f"ip_{int(row['V2'] * 100)}"

        G.add_edge(card, merchant)
        G.add_edge(card, device)
        G.add_edge(device, ip)

    return G


def detect_suspicious_clusters():

    G = build_transaction_graph()

    communities = nx.algorithms.community.greedy_modularity_communities(G)

    suspicious_groups = [c for c in communities if len(c) > 5]

    return suspicious_groups