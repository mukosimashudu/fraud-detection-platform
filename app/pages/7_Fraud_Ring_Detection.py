import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from src.graph_fraud_detection import (
    graph_summary,
    detect_suspicious_communities,
    build_fraud_graph
)

st.title("Fraud Ring Detection")

st.write(
    "Detect suspicious transaction communities using graph analytics. "
    "This simulates relationships between cards, merchants, devices, and IP addresses."
)

# ----------------------------------------------------
# GRAPH SUMMARY
# ----------------------------------------------------

summary = graph_summary(sample_size=3000)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Nodes", summary["nodes"])
col2.metric("Edges", summary["edges"])
col3.metric("Connected Components", summary["connected_components"])
col4.metric("Graph Density", f"{summary['density']:.6f}")

# ----------------------------------------------------
# COMMUNITY DETECTION
# ----------------------------------------------------

st.subheader("Suspicious Communities")

min_size = st.slider(
    "Minimum community size",
    min_value=3,
    max_value=20,
    value=5
)

communities_df = detect_suspicious_communities(
    sample_size=3000,
    min_size=min_size
)

if communities_df.empty:
    st.info("No suspicious communities found.")
else:
    st.dataframe(communities_df)

# ----------------------------------------------------
# FRAUD NETWORK VISUALIZATION
# ----------------------------------------------------

st.subheader("Fraud Network Visualization")

G, df = build_fraud_graph(sample_size=300)

# limit nodes for visualization
nodes_to_show = st.slider(
    "Number of nodes to display",
    min_value=50,
    max_value=300,
    value=150
)

sub_nodes = list(G.nodes())[:nodes_to_show]
subgraph = G.subgraph(sub_nodes)

# create pyvis network
net = Network(
    height="600px",
    width="100%",
    bgcolor="#111111",
    font_color="white"
)

for node, data in subgraph.nodes(data=True):

    node_type = data.get("node_type", "unknown")

    if node_type == "card":
        color = "#00FFFF"
    elif node_type == "merchant":
        color = "#FFD700"
    elif node_type == "device":
        color = "#FF5733"
    elif node_type == "ip":
        color = "#8E44AD"
    else:
        color = "#AAAAAA"

    net.add_node(
        node,
        label=node,
        color=color
    )

for source, target in subgraph.edges():
    net.add_edge(source, target)

net.repulsion(
    node_distance=120,
    spring_length=150
)

html_path = "fraud_network.html"
net.save_graph(html_path)

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=600)