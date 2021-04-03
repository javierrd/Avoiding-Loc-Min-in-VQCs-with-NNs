#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:09:42 2021

@author: javierrd
"""

# ------------------------------ IMPORT PACKAGES -----------------------------
# Pennylane packages (for the quantum circuit)
import pennylane as qml
#from pennylane import numpy as np
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
from helper import *
from Q_circuits import *

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

p = 1 # Number of QAOA REPETITIONS
path = path = "Graphs/"
graphs = graph_loader(path, 6)
G = new_G(graphs[-1])

n_random = 200 # number of random seeds


dev1 = qml.device('default.qubit.tf', wires = len(G.nodes))
dev2 = qml.device('default.qubit', wires = len(G.nodes), analytic = False, shots = 500)

qcircuit = qml.QNode(circuit, dev1, interface = "tf", diff_method = "backprop")
qcircuit_stat = qml.QNode(circuit_stat, dev1, interface = "tf", diff_method = "backprop")

# ------------------------- RANDOM INITIALIZATIONS ----------------------------
gammas = 7*np.random.rand(n_random, p)
betas = 7*np.random.rand(n_random, p)
print(gammas, betas)
# We only have to do this once!
Energies, configs = get_energies_of_all_strings(gammas[0], betas[0], qcircuit, G=G, p=p)
configs = tf.cast(configs,tf.float32)

def cost_function(gamma, beta, Energies, qcircuit, Net=None, G=0, p=0):
    result = qcircuit(gamma, beta, G=G, p=p)
    if Net == None:
        configs_new = configs
    else:
        configs_new = tf.sign(Net(configs))
    E = MaxCut_NN(G,configs_new)
    return tf.reduce_sum(E*result[0])

# -------------------------------- OPTIMIZATION ---------------------------------
def QAOA_opt(gamma, beta, opt, Energies, qcircuit, Net=None, G=0, p=0):
    with tf.GradientTape() as tape:
        cost = cost_function(gamma, beta, Energies, qcircuit, Net=Net, G=G, p=p)
    gradients = tape.gradient(cost, [gamma, beta])
    opt.apply_gradients(zip(gradients, [gamma, beta]))
    return gamma, beta, cost

def NN_opt1(net_tf, gamma, beta, opt, qcircuit_stat):
    # Quantum circuit samples and reshaping
    result = qcircuit_stat(gamma, beta, G=G, p=p)
    shape = np.shape(result)
    result = tf.reshape(result, (shape[1], shape[0]))
    result = tf.cast(result,tf.float32)
    with tf.GradientTape() as tape:
        x = net_tf(result)
        E = MaxCut_NN(G,x)
        cost = tf.reduce_sum(E)/shots
    gradients = tape.gradient(cost, net_tf.trainable_variables)
    opt.apply_gradients(zip(gradients, net_tf.trainable_variables))
    return cost

# ----------------------------- RUNNING THE CODE ------------------------------
# Arrays for saving the results

E_initial = np.zeros(np.shape(gammas)[0])
E_final = np.zeros(np.shape(gammas)[0])

for i in tqdm(range(np.shape(gammas)[0])):
    # Define gamma and beta
    gamma = tf.Variable([gammas[i][k] for k in range(p)], dtype=tf.float64)
    beta = tf.Variable([betas[i][k] for k in range(p)], dtype=tf.float64)
    qcircuit(gamma, beta, G=G, p=p)
    qcircuit_stat(gamma, beta, G=G, p=p)
    initializer = tf.keras.initializers.Identity()
    net_tf = tf.keras.models.Sequential([
      tf.keras.layers.Dense(len(G.nodes), kernel_initializer=initializer),
      tf.keras.layers.Activation("tanh")
    ])
    print(net_tf.layers[0].weights)
    # Initialize the optimizers
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt_NN = tf.keras.optimizers.SGD(learning_rate = 0.05)
    
    # Define number of NN steps
    NN_steps = 25
    QAOA2_steps = 150
    QAOA3_steps = 200
    shots = 1000

    #First optimization step only QAOA
    for k in range(QAOA3_steps): # 50 before
        #print(k)
        gamma, beta, cost = QAOA_opt(gamma, beta, opt, Energies, qcircuit, Net=None, G=G, p=p)
    E_initial[i] = cost
    
    # NN optimization. Second optimization step
    for k in range(NN_steps):
        #print(k)
        NN_opt1(net_tf, gamma, beta, opt_NN, qcircuit_stat)
    
    # QAOA optimization in NN energy landscape. Third optimization step
    for k in range(QAOA2_steps):
        #print(k)
        gamma, beta, cost = QAOA_opt(gamma, beta, opt, Energies, qcircuit, Net=net_tf, G=G, p=p)
    # QAOA optimization step in original landscape. Fourth optimization step
    for k in range(QAOA3_steps):
        gamma, beta, cost = QAOA_opt(gamma, beta, opt, Energies, qcircuit, Net=None, G=G, p=p)
    E_final[i] = cost

