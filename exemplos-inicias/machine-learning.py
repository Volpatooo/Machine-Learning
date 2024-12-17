# pip install python-igraph
# pip install cairocffi
# no terminal


import igraph as ig
import matplotlib.pyplot as plt

# Criando um Grafo
g = ig.Graph()

# Add 7 vértices (pessoas)
g.add_vertices(7)


# Adicionando as arestas entre os vértices
g.add_edges([(0, 3), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (4, 6)])


# Adicionando atributos aos vertices
g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]

# Adicionando atributos as arestas
g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]


# Definindo cores de acordo com o gênero
color_dict = {"m": "blue", "f": "pink"}
g.vs["color"] = [color_dict[gender] for gender in g.vs["gender"]]


# Layout do grafo (Frunchterman-Reingold)
layout = g.layout("fr")


# Plotar o grafo com destaque nos nomes (tamanho maior da fonte)
plt.figure(figsize=(8, 8))
ig.plot(
    g,
    layout=layout,
    vertex_label=g.vs["name"],
    vertex_label_size=15,
    vertex_label_color="black",
    vertex_label_fontweight="bold",
    vertex_color=g.vs["color"],
    vertex_size=40,
    edge_width=[2 if formal else 1 for formal in g.es["is_formal"]],
    margin=50,
    bbox=(600, 600)
)

