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

# ------------------------- QAOA CIRCUIT -------------------------------------
def U_C(G,gamma):
    for e in list(G.edges):
        wire1 = int(e[0])
        wire2 = int(e[1])
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(G[wire1][wire2]["weight"]*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])
# Mixer gate
def U_M(G,gamma):
    for n in list(G.nodes):
        qml.RX(gamma, wires = n)
        
# Definition of the QAOA circuit
def circuit(gamma, beta, G=0, p=0):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(G, gamma[j])
        U_M(G, beta[j])
    return qml.probs(wires = list(range(len(G.nodes))))

# Statistical noisy implementation 
def circuit_stat(gamma, beta, G=0, p=0):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(G, gamma[j])
        U_M(G, beta[j])
    return [qml.sample(qml.PauliZ(i)) for i in range(len(G.nodes))]

# -------------------------- RANDOM VQE ---------------------------------------
# Random gates
gates = [qml.RX, qml.RY, qml.RZ]

# Random unitaries
def Random_unitary(params, G = 0, sequence = 0):
    for i in range(len(G.nodes)):
        gates[sequence[i]](params[i], wires = i)


# Definition of the random circuit
def random_circuit(params, G=0, sequence=0, p=0, **kwargs):
    for i in range(len(G.nodes)):
        qml.RY(np.pi/4, wires = i)
    for j in range(p):
        Random_unitary(params, G, sequence)
    for i in range(len(G.nodes)-1):
        qml.CZ(wires= [i, i+1])
    return qml.probs(wires = list(range(len(G.nodes))))

# Statistical noisy implementation
def random_circuit_stat(params, G=0, sequence=0, p=0, **kwargs):
    for i in range(len(G.nodes)):
        qml.RY(np.pi/4, wires = i)
    for j in range(p):
        Random_unitary(params, G,sequence)
    for i in range(len(G.nodes)-1):
        qml.CZ(wires= [i, i+1])
    return [qml.sample(qml.PauliZ(i)) for i in range(len(G.nodes))]
