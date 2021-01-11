#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:26:19 2021

@author: javierrd
"""

# ------------------------------ IMPORT PACKAGES -----------------------------
# Pennylane packages (for the quantum circuit)
import pennylane as qml
from pennylane import numpy as np

# TensorFlow packages (for optimizing the quantum circuit)
import tensorflow as tf

# Torch packages (for the neural network)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Other packages
import networkx as nx
from tqdm.notebook import tqdm
import os
import random

# ---------------------------- GRAPH LOADER ----------------------------------
# Loading graph
def graph_loader(path, num_graphs):
    or_path = os.getcwd()
    os.chdir(path)
    graphs = []
    for i in range(num_graphs):
        nodes = np.loadtxt("G"+str(i)+"_nodes")
        edges = np.loadtxt("G"+str(i)+"_edges")
        weights = np.loadtxt("G"+str(i)+"_weights")
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        k = 0
        for e in edges:
            G[e[0]][e[1]]["weight"] = weights[k]
            k +=1
        graphs.append(G)
    return graphs
def new_G(G):
    nodes = []
    edges = []
    new_G = nx.Graph()
    for i in list(G.nodes):
        nodes.append(int(i))
    for e in list(G.edges):
        e0 = int(e[0])
        e1 = int(e[1])
        edges.append((e0,e1,G[e0][e1]["weight"]))
    new_G.add_nodes_from(nodes)
    new_G.add_weighted_edges_from(edges)
    return new_G

path = "/home/jrivera/Documents/QAOA_FNN/Graphs_QAOA_NN"
graphs = graph_loader(path, 3)
G = new_G(graphs[0])

# ------------------------- NEURAL NETWORK DEFINITION -------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(G.nodes), len(G.nodes), False)
        nn.init.eye_(self.fc1.weight)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return x

def adjacency_matrix(G):
    adj = torch.zeros((len(G.nodes),len(G.nodes)))
    for edge in G.edges:
        i = edge[0]
        j = edge[1]
        adj[i,j] = G[i][j]["weight"]
        adj[j,i] = G[j][i]["weight"]
    return adj

def MaxCut_NN(G,x):
    adj = adjacency_matrix(G).double()
    A_batch = adj.repeat(x.shape[0],1,1).double()
    first_prod = torch.bmm(A_batch, x.view(-1, x.shape[1], 1))
    second_prod = torch.bmm(x.view(x.shape[0], 1, x.shape[1]), first_prod)
    return 0.5*torch.sum(second_prod)

# ------------------------- QAOA DEFINITIONS ----------------------------------
# Define the depth my circuit depth
p = 1

# Cost gate
def U_C(gamma):
    for e in list(G.edges):
        wire1 = int(e[0])
        wire2 = int(e[1])
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(G[wire1][wire2]["weight"]*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])

# Mixer gate
def U_M(gamma):
    for n in list(G.nodes):
        qml.RX(gamma, wires = n)

# Definition of the circuit together with the device
dev = qml.device('default.qubit.tf', wires = len(G.nodes))
@qml.qnode(dev, interface = "tf", diff_method = "backprop")
def circuit(gamma, beta, **kwargs):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(gamma[j])
        U_M(beta[j])
    return qml.probs(wires = list(range(len(G.nodes))))

def string_to_tens(x):
    tens = torch.zeros(len(x))
    i = 0
    for el in x:
        tens[i] = float(el)
        i+=1
    return tens

def adjacency_matrix(G):
    adj = torch.zeros((len(G.nodes),len(G.nodes)))
    for edge in G.edges:
        i = edge[0]
        j = edge[1]
        adj[i,j] = G[i][j]["weight"]
        adj[j,i] = G[j][i]["weight"]
    return adj

def MaxCut_NN_QAOA(G,x, net):
    x = 1-2*x
    z = torch.sign(net(x)).double()
    adj = adjacency_matrix(G).double()
    return 0.5*torch.matmul(z,torch.mv(adj,z))

def cost_function(gamma, beta, net):
    counts = {}
    result = circuit(gamma, beta)
    # In the following line, change 2 --> your number of qubits
    for i in range(len(result[0])):
        counts[f"{i:05b}"] = result[0][i]
    E = np.array([])
    for bitstring in counts.keys():
        x = string_to_tens(bitstring).double()
        E = np.append(E,1*float(MaxCut_NN_QAOA(G, x, net)))
        #E += -energy*counts[bitstring]
    return sum(E*result[0])

# ----------------------- STATISTICAL NOISY CIRCUIT --------------------------
dev = qml.device('default.qubit', wires = len(G.nodes), analytic = False, shots = 500)
@qml.qnode(dev, interface = "torch")
def circuit_stat(gamma, beta, **kwargs):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(gamma[j])
        U_M(beta[j])
    return [qml.sample(qml.PauliZ(i)) for i in range(len(G.nodes))]

# ------------------------- RANDOM INITIALIZATIONS ----------------------------
n_random = 200 # number of random seeds

gammas = np.zeros((n_random, p))
betas = np.zeros((n_random, p))

for i in range(n_random):
    for j in range(p):
        gammas[i][j] = 7*random.random()
        betas[i][j] = 7*random.random()
        
# -------------------------------- OPTIMIZERS ---------------------------------
def QAOA_opt(gamma, beta, opt, net):
    with tf.GradientTape() as tape:
        cost = cost_function(gamma, beta, net)
    gradients = tape.gradient(cost, [gamma, beta])
    opt.apply_gradients(zip(gradients, [gamma, beta]))
    return gamma, beta, cost
def NN_opt1(net, gamma, beta, opt_NN):
    # Quantum circuit samples and reshaping
    result = circuit_stat(gamma.numpy(), beta.numpy())
    nodes, shots = np.shape(result)
    result = result.reshape(shots, nodes)
    input = result.double()
    
    # NN and optimization
    x = net(input)
    E = MaxCut_NN(G,x)
    cost = E/shots
    cost.backward()
    opt_NN.step()
    opt_NN.zero_grad()
    return cost.detach().numpy()

# ----------------------------- RUNNING THE CODE ------------------------------
# Arrays for saving the results

E_initial = np.zeros(np.shape(gammas)[0])
E_final = np.zeros(np.shape(gammas)[0])

for i in tqdm(range(np.shape(gammas)[0])):
    # Define gamma and beta
    gamma = tf.Variable([gammas[i][k] for k in range(p)], dtype=tf.float64)
    beta = tf.Variable([betas[i][k] for k in range(p)], dtype=tf.float64)
    
    # Initialize the NN
    net = Net().double()
    net_QAOA = Net().double()
    
    # Initialize the optimizers
    opt_QAOA = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt_QAOA2 = tf.keras.optimizers.Adam(learning_rate= 0.1)
    opt_NN = optim.SGD(net.parameters(), lr = 0.05)
    
    # Define number of NN steps
    NN_steps = 80
    QAOA2_steps = 200
    QAOA3_steps = 50
    
    # First optimization step
    for k in range(50):
        gamma, beta, cost = QAOA_opt(gamma, beta, opt_QAOA, net_QAOA)
    E_initial[i] = cost
    
    # NN optimization. Second optimization step
    for k in range(NN_steps):
        NN_opt1(net, gamma, beta, opt_NN)
    
    # QAOA optimization in NN energy landscape. Third optimization step
    for k in range(QAOA2_steps):
        gamma, beta, cost = QAOA_opt(gamma, beta, opt_QAOA2, net)
    
    # QAOA optimization step in original landscape. Fourth optimization step
    for k in range(QAOA3_steps):
        gamma, beta, cost = QAOA_opt(gamma, beta, opt_QAOA, net_QAOA)
    E_final[i] = cost
    
# ------------------------------ SAVING THE RESULTS --------------------------
path_saving = "/home/jrivera/Documents/QAOA_FNN/Results/Candle/5-qubit-new/ADAM_01/NN_"+str(NN_steps)
os.chdir(path_saving)
np.savetxt("E_final_p_" + str(p), E_final)
np.savetxt("E_initial_p_" + str(p), E_initial)