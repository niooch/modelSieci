import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

def generate_spanning_tree(num_nodes):
    """
    Generate a random spanning tree for a graph with num_nodes nodes.
    This ensures connectivity by:
    - Shuffling the list of nodes.
    - Connecting each subsequent node to a randomly chosen node among those already added.
    """
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    T = nx.Graph()
    T.add_nodes_from(nodes)
    for i in range(1, num_nodes):
        T.add_edge(nodes[i], random.choice(nodes[:i]))
    return T

def generate_connected_graph(num_nodes=20, min_edges=20, max_edges=30):
    """
    Generate a connected graph with num_nodes nodes and a random number of edges
    between min_edges and max_edges.
    The function begins with a spanning tree (ensuring connectivity) and adds extra edges.
    """
    num_edges = random.randint(min_edges, max_edges)
    # Generate a spanning tree first.
    G = generate_spanning_tree(num_nodes)
    current_edges = G.number_of_edges()  # Exactly num_nodes - 1 edges.

    # Build a set of all possible edges not already in the spanning tree.
    possible_edges = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if not G.has_edge(i, j):
                possible_edges.add((i, j))

    # Add extra edges to reach the target number.
    extra_edges = num_edges - current_edges
    if extra_edges > 0:
        extra_edges_selected = random.sample(list(possible_edges), extra_edges)
        G.add_edges_from(extra_edges_selected)

    return G

def generate_packet_intensity_matrix(G, intensity_min=1, intensity_max=10):
    """
    Generate a packet intensity matrix N from the graph G.

    For an undirected edge (i, j) in G, assume that both directions (i -> j and j -> i)
    can have traffic. Each directed connection gets its own random intensity value.

    The matrix is of size |V| x |V| with zeros on the diagonal and zero for non-adjacent nodes.
    """
    num_nodes = G.number_of_nodes()
    N = np.zeros((num_nodes, num_nodes), dtype=int)

    # For every edge in the graph, assign random intensities to both directions.
    for u, v in G.edges():
        N[u][v] = random.randint(intensity_min, intensity_max)
        N[v][u] = random.randint(intensity_min, intensity_max)

    return N


def generate_edge_capacity_matrix(G, base_capacity=21000, scaling_factor=2000):
    """
    Generuje macierz przepustowości krawędzi, gdzie przepustowość zależy od stopni węzłów.

    Przepustowość krawędzi (u,v) = base_capacity + scaling_factor * (degree(u) + degree(v))
    """
    num_nodes = G.number_of_nodes()
    capacity_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    node_degrees = dict(G.degree())

    # Dla każdej krawędzi w grafie przypisz przepustowość zależną od stopnia węzłów
    for u, v in G.edges():
        # Obliczamy przepustowość w oparciu o sumę stopni węzłów
        capacity = base_capacity + scaling_factor * (node_degrees[u] + node_degrees[v])
        # Macierz jest symetryczna dla krawędzi nieskierowanych
        capacity_matrix[u][v] = capacity
        capacity_matrix[v][u] = capacity

    return capacity_matrix


def generate_edge_reliability_matrix(G, base_prob=0.9, scaling_factor=0.01):
    """
    Generuje macierz prawdopodobieństw, że krawędź nie ulegnie awarii.

    Prawdopodobieństwo dla krawędzi (u,v) = min(base_prob + scaling_factor * (degree(u) + degree(v)), 0.99)
    """
    num_nodes = G.number_of_nodes()
    reliability_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    node_degrees = dict(G.degree())

    # Dla każdej krawędzi w grafie przypisz prawdopodobieństwo zależne od stopnia węzłów
    for u, v in G.edges():
        # Obliczamy prawdopodobieństwo w oparciu o sumę stopni węzłów
        # Ograniczamy maksymalną wartość do 0.99
        prob = min(base_prob + scaling_factor * (node_degrees[u] + node_degrees[v]), 0.99)
        # Macierz jest symetryczna dla krawędzi nieskierowanych
        reliability_matrix[u][v] = prob
        reliability_matrix[v][u] = prob

    return reliability_matrix


if __name__ == '__main__':
    num_nodes = 20
    # Generate a random, connected graph.
    G = generate_connected_graph(num_nodes=num_nodes)

    # Save the graph topology (edge list) to a text file.
    nx.write_edgelist(G, "graph_topology.txt", data=False)
    print("Graph topology saved to graph_topology.txt")

    # Generate the packet intensity matrix N based on the graph.
    N = generate_packet_intensity_matrix(G)

    # Save the matrix N to a text file. Each row corresponds to a node.
    np.savetxt("packet_intensity_matrix.txt", N, fmt='%d', delimiter=' ')
    print("Packet intensity matrix saved to packet_intensity_matrix.txt")

    # Generuj i zapisz macierz przepustowości krawędzi
    capacity_matrix = generate_edge_capacity_matrix(G)
    np.savetxt("edge_capacity_matrix.txt", capacity_matrix, fmt='%d', delimiter=' ')
    print("Edge capacity matrix saved to edge_capacity_matrix.txt")

    # Generuj i zapisz macierz prawdopodobieństw niezawodności krawędzi
    reliability_matrix = generate_edge_reliability_matrix(G)
    np.savetxt("edge_reliability_matrix.txt", reliability_matrix, fmt='%.4f', delimiter=' ')
    print("Edge reliability matrix saved to edge_reliability_matrix.txt")

    # (Optional) Visualize the generated graph.
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    plt.title("Random Connected Graph: 20 Nodes, {} Edges".format(G.number_of_edges()))
    plt.savefig("graph_topology.png")
