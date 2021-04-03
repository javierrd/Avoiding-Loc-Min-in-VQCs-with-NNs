#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:26:19 2021

@author: javierrd
"""

# ------------------------------ IMPORT PACKAGES -----------------------------
# Pennylane packages (for the quantum circuit)
import pennylane as qml
import numpy as np

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

# ---------------------------- GRAPH LOADER ----------------------------------
# Loading graph
def graph_loader(path, num_graphs):
    # or_path = os.getcwd()
    # os.chdir(path)
    graphs = []
    for i in range(num_graphs):
        nodes = np.loadtxt(path+"G"+str(i)+"_nodes")
        edges = np.loadtxt(path+"G"+str(i)+"_edges")
        weights = np.loadtxt(path+"G"+str(i)+"_weights")
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

def adjacency_matrix(G):
    adj = np.zeros((len(G.nodes),len(G.nodes)))
    for edge in G.edges:
        i = edge[0]
        j = edge[1]
        adj[i,j] = G[i][j]["weight"]
        adj[j,i] = G[j][i]["weight"]
    return np.array([adj])

def string_to_tens(x):
    tens = np.zeros(len(x))
    i = 0
    for el in x:
        tens[i] = float(el)
        i+=1
    return tens

def MaxCut_NN_single_bitstring(G,x):
    Adj = adjacency_matrix(G)
    first_prod = tf.linalg.matvec(Adj, x)
    inner_prod = tf.reduce_sum(tf.multiply(x, first_prod))
    return 0.5*inner_prod

def MaxCut_NN(G,x):
    Adj = adjacency_matrix(G)
    A_batch =  tf.tile(Adj, [x.shape[0], 1,1])
    A_batch = tf.cast(A_batch, tf.float32)
    first_prod = tf.linalg.matvec(A_batch, x)
    inner_prod = tf.reduce_sum(tf.multiply(x, first_prod),1)
    inner_prod = tf.cast(inner_prod, dtype = tf.float64)
    return 0.5*inner_prod

def get_energies_of_all_strings(gamma, beta, qcircuit, G=0, p=0):
    counts = {}
    result = qcircuit(gamma, beta, G=G, p=p)
    # In the following line, change 2 --> your number of qubits
    for i in range(len(result[0])):
        counts[f"{i:05b}"] = result[0][i]
    E = np.array([])
    configs = []
    for bitstring in counts.keys():
        x = string_to_tens(bitstring)
        x = 2*x-1
        E = np.append(E,1*MaxCut_NN_single_bitstring(G,x))
        configs.append(x)
    return E, np.array(configs)

"""
    The following funtion is exactly the same as the one introduced above
    but for the random quantum circuit, which needs a single set of parameters
    instead of two
"""
def get_energies_of_all_strings_rand(params, qcircuit, sequence=0, G=0, p=0):
    counts = {}
    result = qcircuit(params, G=G, p=p, sequence=sequence)
    # In the following line, change 2 --> your number of qubits
    for i in range(len(result[0])):
        counts[f"{i:08b}"] = result[0][i]
    E = np.array([])
    configs = []
    for bitstring in counts.keys():
        x = string_to_tens(bitstring)
        x = 2*x-1
        E = np.append(E,1*MaxCut_NN_single_bitstring(G,x))
        configs.append(x)
    return E, np.array(configs)
