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
from tqdm.notebook import tqdm
import os
import random

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
# Definition of the circuit together with the device
def circuit(gamma, beta, G=0, p=0):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(G, gamma[j])
        U_M(G, beta[j])
    return qml.probs(wires = list(range(len(G.nodes))))
# ----------------------- STATISTICAL NOISY CIRCUIT --------------------------
def circuit_stat(gamma, beta, G=0, p=0):
    for i in range(len(G.nodes)):
        qml.Hadamard(wires = i)
    for j in range(p):
        U_C(G, gamma[j])
        U_M(G, beta[j])
    return [qml.sample(qml.PauliZ(i)) for i in range(len(G.nodes))]
