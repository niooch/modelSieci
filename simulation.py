import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# =================== PARAMETERS ===================
# (User–defined parameters; these are also inputs in the exercise.)
# Probability that any given edge is functional (nieuszkodzona) in a time interval.
p = 0.95
# Threshold on the average delay (seconds) for acceptable performance.
T_max = 0.5
# Average packet size in bits.
m = 1000
# Baseline capacity per edge in bits per second.
# (For example, if capacity=15000 bits/s and m=1000 bits, then each channel can carry 15 packets per second.)
CAPACITY_baseline = 21000
# Number of Monte Carlo trials to estimate reliability.
num_trials = 10000

# Filenames for baseline network parameters.
GRAPH_FILE = "graph_topology.txt"
MATRIX_FILE = "packet_intensity_matrix.txt"

# =================== UTILITY FUNCTIONS ===================

def load_graph(edge_file=GRAPH_FILE):
    """
    Load the baseline graph (topology) from an edge list file.
    Each line in the file should have: u v
    """
    G = nx.read_edgelist(edge_file, nodetype=int, data=False)
    # Ensure nodes are integers 0,...,n-1.
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G

def load_matrix(matrix_file=MATRIX_FILE):
    """
    Load the packet intensity matrix N from a text file.
    The file is assumed to be space-delimited.
    """
    N = np.loadtxt(matrix_file, dtype=float)
    return N

def compute_delay(G, N, capacity, m):
    """
    Compute the network average delay T as defined by:

       T = (1/G_total) * sum_{e in E} [ a(e) / ( (capacity/m) - a(e) ) ]

    where the sum is taken over *directed* channels defined by the baseline (undirected)
    topology. For each undirected edge (u, v) in G, both directions are included:
    – from u to v (using N[u,v]) and from v to u (using N[v,u]).

    G_total is the sum of all intensity values on the operating links (as defined in N).

    If for any channel we have (capacity/m) <= a(e), the delay is taken as infinite.
    """
    total_intensity = 0.0
    total_delay = 0.0
    for u, v in G.edges():
        # For each direction. (Assumes N is indexed by node numbers.)
        for (i, j) in [(u, v), (v, u)]:
            a = N[i, j]
            total_intensity += a
            cap_packets = capacity / m  # capacity in packets per second
            if a >= cap_packets:
                # This channel is overloaded; delay becomes infinite.
                return float('inf')
            delay = a / (cap_packets - a)
            total_delay += delay
    # Avoid division by zero.
    if total_intensity == 0:
        return float('inf')
    T = total_delay / total_intensity
    return T

def estimate_reliability(G, N, capacity, m, T_max, p, num_trials):
    """
    Estimate the reliability of the network, defined as the probability that in a given time interval:
        - The network (with edge failures) remains connected, and
        - The average delay T (computed on the baseline parameters) is less than T_max.

    In our simplified model, we assume that if the network is connected then the delay is computed
    using the baseline intensity matrix and capacity (i.e. re-routing does not change T).

    Since T is computed deterministically from N, capacity, and m, if T >= T_max the trial fails.

    Returns:
       reliability: estimated probability (between 0 and 1)
       T_value: the computed delay for the intact (baseline) network.
    """
    # Compute the baseline average delay T.
    T_value = compute_delay(G, N, capacity, m)
    if T_value >= T_max or T_value == float('inf'):
        # Even with all edges active, delay is too high.
        return 0.0, T_value

    success_count = 0
    for _ in range(num_trials):
        # For each trial, simulate edge failures.
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        # For each edge in the baseline topology, include it with probability p.
        for edge in G.edges():
            if random.random() < p:
                H.add_edge(*edge)
        # If the resulting graph is connected, count the trial as a success.
        if nx.is_connected(H):
            success_count += 1
    connectivity_prob = success_count / num_trials
    # Since T_value < T_max for the intact network,
    # the overall reliability is given by the probability that the network remains connected.
    return connectivity_prob, T_value

# =================== EXPERIMENT FUNCTIONS ===================

def experiment_intensity_scaling(G, N, capacity, m, T_max, p, num_trials, scales):
    """
    With fixed topology and capacity, vary the intensity matrix by scaling it.
    For each scaling factor, compute T and estimate reliability.
    Returns a list of (scale, T, reliability) tuples.
    """
    results = []
    for scale in scales:
        N_scaled = N * scale
        T_val = compute_delay(G, N_scaled, capacity, m)
        reliability, _ = estimate_reliability(G, N_scaled, capacity, m, T_max, p, num_trials)
        results.append((scale, T_val, reliability))
        print(f"Scale={scale:.2f} -> T = {T_val:.4f}, Reliability = {reliability:.4f}")
    return results

def experiment_capacity(G, N, capacities, m, T_max, p, num_trials):
    """
    With fixed topology and intensity matrix, vary the capacity.
    For each capacity value, compute T and estimate reliability.
    Returns a list of (capacity, T, reliability) tuples.
    """
    results = []
    for cap in capacities:
        T_val = compute_delay(G, N, cap, m)
        reliability, _ = estimate_reliability(G, N, cap, m, T_max, p, num_trials)
        results.append((cap, T_val, reliability))
        print(f"Capacity={cap} bits/s -> T = {T_val:.4f}, Reliability = {reliability:.4f}")
    return results

def experiment_topology(G, N, capacity, m, T_max, p, num_trials, num_extra_edges_list):
    """
    Starting with the baseline topology G, gradually add extra edges.
    For each augmented topology, we assume that the new edge(s) get a capacity equal to the baseline,
    and assign them an intensity equal to the average intensity of the baseline (for each direction).

    Returns a list of (extra_edges, T, reliability) tuples.
    """
    results = []
    # Compute average intensity per directed edge over the baseline edges.
    total_intensity = 0.0
    count = 0
    for u, v in G.edges():
        for (i, j) in [(u, v), (v, u)]:
            total_intensity += N[i, j]
            count += 1
    avg_intensity = total_intensity / count if count > 0 else 0

    # Build a complete graph on the same nodes.
    nodes = list(G.nodes())
    complete_edges = set()
    for i in nodes:
        for j in nodes:
            if i < j:
                complete_edges.add((i, j))

    # Find the set of edges not in the baseline.
    baseline_edges = set(G.edges())
    extra_possible = list(complete_edges - baseline_edges)

    # For each number of extra edges to add, augment a copy of the baseline graph.
    for extra in num_extra_edges_list:
        G_aug = G.copy()
        N_aug = N.copy()
        if extra > len(extra_possible):
            extra_edges = extra_possible
        else:
            extra_edges = random.sample(extra_possible, extra)
        # Add each extra edge to the augmented graph and update the intensity matrix.
        for (u, v) in extra_edges:
            G_aug.add_edge(u, v)
            # Assume symmetric traffic on new links equal to the average.
            N_aug[u, v] = avg_intensity
            N_aug[v, u] = avg_intensity
        T_val = compute_delay(G_aug, N_aug, capacity, m)
        reliability, _ = estimate_reliability(G_aug, N_aug, capacity, m, T_max, p, num_trials)
        results.append((extra, T_val, reliability))
        print(f"Extra edges={extra} -> T = {T_val:.4f}, Reliability = {reliability:.4f}")
    return results

# =================== MAIN SIMULATION ===================

if __name__ == '__main__':
    # Load baseline network topology and intensity matrix.
    G = load_graph(GRAPH_FILE)
    N = load_matrix(MATRIX_FILE)

    print("Baseline network:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")

    # Estimate reliability for the baseline parameters.
    baseline_reliability, T_baseline = estimate_reliability(G, N, CAPACITY_baseline, m, T_max, p, num_trials)
    print(f"\nBaseline delay T = {T_baseline:.4f}, Reliability = {baseline_reliability:.4f}\n")

    # ----- Experiment 1: Vary intensity scaling -----
    print("Experiment 1: Varying the intensity matrix (N) scaling:")
    scales = np.linspace(1.0, 2.0, 11)  # e.g. scale factors from 1.0 to 2.0
    exp1_results = experiment_intensity_scaling(G, N, CAPACITY_baseline, m, T_max, p, num_trials, scales)

    # ----- Experiment 2: Vary capacity -----
    print("\nExperiment 2: Varying the capacity (c) of each edge:")
    # For instance, vary capacity from 10000 bits/s to 20000 bits/s.
    capacities = np.linspace(CAPACITY_baseline, CAPACITY_baseline*2, 11)
    exp2_results = experiment_capacity(G, N, capacities, m, T_max, p, num_trials)

    # ----- Experiment 3: Vary topology by adding extra edges -----
    print("\nExperiment 3: Varying topology by adding extra edges:")
    # For example, try adding from 0 up to 10 extra edges.
    extra_edges_list = list(range(0, 11))
    exp3_results = experiment_topology(G, N, CAPACITY_baseline, m, T_max, p, num_trials, extra_edges_list)

    # (Optional) Plotting the results for visual inspection.
    plt.figure(figsize=(12, 4))

    # Experiment 1 Plot: Intensity scaling vs Reliability and Delay
    plt.subplot(1, 3, 1)
    scales_list, T_list, rel_list = zip(*exp1_results)
    plt.plot(scales_list, rel_list, marker='o')
    plt.xlabel("Intensity Scale Factor")
    plt.ylabel("Reliability Pr[T < T_max]")
    plt.title("Varying N intensities")

    # Experiment 2 Plot: Capacity vs Reliability and Delay
    plt.subplot(1, 3, 2)
    cap_list, T_cap_list, rel_cap_list = zip(*exp2_results)
    plt.plot(cap_list, rel_cap_list, marker='o')
    plt.xlabel("Capacity (bits/s)")
    plt.ylabel("Reliability Pr[T < T_max]")
    plt.title("Varying Capacity")

    # Experiment 3 Plot: Extra edges vs Reliability and Delay
    plt.subplot(1, 3, 3)
    extra_list, T_topo_list, rel_topo_list = zip(*exp3_results)
    plt.plot(extra_list, rel_topo_list, marker='o')
    plt.xlabel("Number of Extra Edges")
    plt.ylabel("Reliability Pr[T < T_max]")
    plt.title("Varying Topology")

    plt.tight_layout()
    plt.savefig("plot1.png")
