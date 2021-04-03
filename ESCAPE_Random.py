#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:48:57 2021

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
from tqdm import tqdm
import os
import random
from helper import *
from Q_circuits import *

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

p = 1 # Number of circuit repetitions (not applicable here)
path = "Graphs/"
graphs = graph_loader(path, 4)
G = new_G(graphs[-1])
n_random = 200 # number of random seeds


dev1 = qml.device('default.qubit.tf', wires = len(G.nodes))
dev2 = qml.device('default.qubit', wires = len(G.nodes), analytic = False, shots = 500)

qcircuit = qml.QNode(random_circuit, dev1, interface = "tf", diff_method = "backprop")
qcircuit_stat = qml.QNode(random_circuit_stat, dev1, interface = "tf", diff_method = "backprop")

# ------------------------- RANDOM INITIALIZATIONS ----------------------------
params_set = 7*np.random.rand(n_random, len(G.nodes)*p)
sequence = np.array([0,0,1,1,2,2,0,1])

# We only have to do this once!
Energies, configs = get_energies_of_all_strings_rand(params_set[0], qcircuit, G=G, p=p, sequence=sequence)
configs = tf.cast(configs,tf.float32)

def cost_function(params, Energies, qcircuit, sequence=0, Net=None, G=0, p=0):
    result = qcircuit(params, G=G, p=p, sequence = sequence)
    if Net == None:
        configs_new = configs
    else:
        configs_new = tf.sign(Net(configs))
    E = MaxCut_NN(G,configs_new)
    return tf.reduce_sum(E*result[0])

# -------------------------------- OPTIMIZATION ---------------------------------
def QAOA_opt(params, opt, Energies, qcircuit, sequence=0, Net=None, G=0, p=0):
    with tf.GradientTape() as tape:
        cost = cost_function(params, Energies, qcircuit, sequence = sequence, Net=Net, G=G, p=p)
    gradients = tape.gradient(cost, [params])
    opt.apply_gradients(zip(gradients, [params]))
    return params, cost

def NN_opt1(net_tf, params, opt, qcircuit_stat, sequence=0):
    # Quantum circuit samples and reshaping
    result = qcircuit_stat(params, G=G, p=p, sequence=sequence)
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

E_initial = np.zeros(n_random)
E_final = np.zeros(n_random)

for i in tqdm(range(n_random)):
    # Define gamma and beta
    params = tf.Variable([params_set[i][k] for k in range(len(G.nodes)*p)], dtype=tf.float64)
    print(sequence)
    qcircuit(params, G=G, p=p, sequence=sequence)
    qcircuit_stat(params, G=G, p=p, sequence=sequence)
    initializer = tf.keras.initializers.Identity()
    net_tf = tf.keras.models.Sequential([
      tf.keras.layers.Dense(len(G.nodes), kernel_initializer=initializer),
      tf.keras.layers.Activation("tanh")
    ])
    # Initialize the optimizers
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt_NN = tf.keras.optimizers.SGD(learning_rate = 0.05)
    
    # Define number of NN steps
    NN_steps = 150
    QAOA2_steps = 100
    QAOA3_steps = 300
    QAOA4_steps = 100
    shots = 1000

    #First optimization step only QAOA
    for k in range(QAOA3_steps): # 50 before
        #print(k)
        params, cost = QAOA_opt(params, opt, Energies, qcircuit, Net=None, G=G, p=p, sequence=sequence)
    E_initial[i] = cost
    
    # NN optimization. Second optimization step
    for k in range(NN_steps):
        #print(k)
        NN_opt1(net_tf, params, opt_NN, qcircuit_stat, sequence=sequence)
    
    # QAOA optimization in NN energy landscape. Third optimization step
    for k in range(QAOA2_steps):
        #print(k)
        params, cost = QAOA_opt(params, opt, Energies, qcircuit, Net=net_tf, G=G, p=p, sequence=sequence)
    # QAOA optimization step in original landscape. Fourth optimization step
    for k in range(QAOA4_steps):
        params, cost = QAOA_opt(params, opt, Energies, qcircuit, Net=None, G=G, p=p, sequence=sequence)
    E_final[i] = cost

