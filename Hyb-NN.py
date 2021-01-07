#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:25:30 2020

@author: javierrd
"""

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
from tqdm import tqdm
import os
import random
import copy

# ------------------------------ GRAPH LOADER --------------------------------

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

path = "Graphs"
graphs = graph_loader(path, 3)
G = new_G(graphs[0]) # 0 for 5-qubit, 1 for 11-qubit and 2 for 8-qubit

# ------------------------------ NEURAL NETWORK -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(G.nodes), len(G.nodes), False)
        nn.init.eye_(self.fc1.weight)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return x

class aNNealer(nn.Module):
    def __init__(self, tnsor):
        self.tnsor = tnsor
        super(aNNealer, self).__init__()
        self.fc1 = nn.Linear(len(G.nodes), len(G.nodes), False)
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(self.tnsor)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return x

def MaxCut_NN(G,x):
    adj = adjacency_matrix(G)
    return 0.5*torch.matmul(x,torch.mv(adj,x))

# Definition of the annealing functions for the aNNealer step
def annealing_f(x):
    return 0.5*(1-np.tanh(0.05*(x-50)))

def annealing_g(x):
    return 0.5*(np.tanh(0.05*(x-50))+1)

# Definition of the W matrix in the aNNealer step
def W_new(x, tnsor):
    identity = torch.ones(len(G.nodes))
    ident = torch.diag(identity)
    return annealing_f(x)*tnsor + annealing_g(x)*ident

# ---------------------- QUANTUM CIRCUIT DEFINITIONS --------------------------
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

# Define the circuit depth
p = 8

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
    z = torch.sign(net(x))
    adj = adjacency_matrix(G)
    return 0.5*torch.matmul(z,torch.mv(adj,z))

def cost_function(gamma, beta, net):
    counts = {}
    result = circuit(gamma, beta)
    # In the following line, change 2 --> your number of qubits
    for i in range(len(result[0])):
        counts[f"{i:05b}"] = result[0][i]
    E = np.array([])
    for bitstring in counts.keys():
        x = string_to_tens(bitstring)
        E = np.append(E,1*float(MaxCut_NN_QAOA(G, x, net)))
        #E += -energy*counts[bitstring]
    return sum(E*result[0])

# ------------------- RANDOM NUMBER LOADER / GENERATOR -----------------------
# Generator of random parameters
n_random = 150 # number of random seeds

gammas = np.zeros((n_random, p))
betas = np.zeros((n_random, p))

for i in range(n_random):
    for j in range(p):
        gammas[i][j] = 7*random.random()
        betas[i][j] = 7*random.random()


# ------------------------- STATISTICAL NOISY CIRCUIT ------------------------
dev = qml.device('default.qubit.tf', wires = len(G.nodes), analytic = False, shots = 400)
@qml.qnode(dev, interface = "tf")
def circuit_stat(gamma, beta, **kwargs):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(gamma[j])
        U_M(beta[j])
    return [qml.sample(qml.PauliZ(i)) for i in range(len(G.nodes))]

# -------------------------------- OPTIMIZERS --------------------------------
def QAOA_opt(gamma, beta, opt, net):
    with tf.GradientTape() as tape:
        cost = cost_function(gamma, beta, net)
    gradients = tape.gradient(cost, [gamma, beta])
    opt.apply_gradients(zip(gradients, [gamma, beta]))
    return gamma, beta, cost

def NN_opt(net, gamma, beta, opt_NN):
    # Identity definition
    identity = torch.ones(len(G.nodes))
    ident = torch.diag(identity)
    
    result = circuit_stat(gamma,beta).numpy()
    nodes, shots = np.shape(result)
    E = []
    opt_NN.zero_grad()
    for i in range(shots):
        el = [np.float(result[j][i]) for j in range(nodes)]
        input = torch.tensor(el)
        x = net(input)
        E.append(MaxCut_NN(G,x))
    cost = sum(E)/shots + 0.4*torch.sum(torch.abs(net.fc1.weight -ident))
    #print(torch.sum(torch.abs(net.fc1.weight -ident)))
    cost.backward()
    opt_NN.step()

    
    
def checker(gamma,beta,net):
    result = circuit_stat(gamma,beta).numpy()
    nodes, shots = np.shape(result)
    E = []
    for i in range(shots):
        el = [np.float(result[j][i]) for j in range(nodes)]
        el = torch.tensor(el)
        x = torch.sign(net(el))
        E.append(MaxCut_NN(G,x))
    cost = sum(E)/shots
    return cost

def checker_QAOA(gamma,beta):
    result = circuit_stat(gamma,beta).numpy()
    nodes, shots = np.shape(result)
    E = []
    net = Net()
    for i in range(shots):
        el = [np.float(result[j][i]) for j in range(nodes)]
        el = torch.tensor(el)
        E.append(MaxCut_NN(G,el))
    cost = sum(E)/shots
    return cost.detach().numpy()

# --------------------------- RUNNING THE CODE -------------------------------
# Lists for saving the energies
E_after = []
E_QAOA = []
E_hybrid = []

for i in tqdm(range(n_random)):
    # Define gamma and beta
    gamma = tf.Variable([gammas[i][k] for k in range(p)], dtype=tf.float64)
    beta = tf.Variable([betas[i][k] for k in range(p)], dtype=tf.float64)
    

    gamma_QAOA = copy.deepcopy(gamma)
    gamma_hybrid = copy.deepcopy(gamma)
    gamma_NN = copy.deepcopy(gamma)
    beta_QAOA = copy.deepcopy(beta)
    beta_hybrid = copy.deepcopy(beta)
    beta_NN = copy.deepcopy(beta)
    print(gamma_hybrid, beta_hybrid)
    # Initialize the NN
    net_QAOA = Net()
    net_hybrid = Net()
    net_NN = Net()
    
    # Initialize the optimizers
    opt_QAOA = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt_hybrid = optim.SGD(net_hybrid.parameters(), lr = 0.05)
    opt_NN = optim.SGD(net_NN.parameters(), lr = 0.05)
    
    # Optimization
    for i in tqdm(range(500)):
        # QAOA optimization
        gamma_QAOA, beta_QAOA, cost_QAOA = QAOA_opt(gamma_QAOA, beta_QAOA, opt_QAOA, net_QAOA)
        
        # Hybrid Optimization
        gamma_hybrid, beta_hybrid, cost_hybrid = QAOA_opt(gamma_hybrid, beta_hybrid, opt_QAOA, net_hybrid)
        NN_opt(net_hybrid, gamma_hybrid, beta_hybrid, opt_hybrid)
	
    E_hybrid.append(float(cost_hybrid))
    E_QAOA.append(float(cost_QAOA))
    
    # Save the final angles such that we can modify them with perturbing the others
    gamma_after = copy.deepcopy(gamma_hybrid)
    beta_after = copy.deepcopy(beta_hybrid)
    W = net_hybrid.fc1.weight
    for i in tqdm(range(150)):
        # QAOA optimization
        W_prim = W_new(i, W)
        aNN = aNNealer(W_prim)
        gamma_after, beta_after, cost_after = QAOA_opt(gamma_after, beta_after, opt_QAOA, aNN)
    E_after.append(float(cost_after))
    
# ------------------------- SAVING THE RESULTS -------------------------------
path_saving = "Path for saving the results"
os.chdir(path_saving)
np.savetxt("E_hybrid_p_"+str(p), E_hybrid)
np.savetxt("E_QAOA_p_"+str(p), E_QAOA)
np.savetxt("E_after_p_"+str(p), E_after)
