import pickle

with open("./resources/graph_with_cs.pkl", "rb") as f:
    G = pickle.load(f)

for i in G.nodes:
    print(G.nodes[i])