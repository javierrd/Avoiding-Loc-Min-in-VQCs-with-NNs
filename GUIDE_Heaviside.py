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
import copy

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

p = 4
path = "Graphs/"
graphs = graph_loader(path, 4)
G = new_G(graphs[0])
n_random = 200 # number of random seeds


dev1 = qml.device('default.qubit.tf', wires = len(G.nodes))
dev2 = qml.device('default.qubit', wires = len(G.nodes), analytic = False, shots = 500)

qcircuit = qml.QNode(circuit, dev1, interface = "tf", diff_method = "backprop")
qcircuit_stat = qml.QNode(circuit_stat, dev1, interface = "tf", diff_method = "backprop")

# ------------------------- RANDOM INITIALIZATIONS ----------------------------
gammas = 7*np.random.rand(n_random, p)
betas = 7*np.random.rand(n_random, p)
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

def configs_generator(gamma, beta, conf, shots):
    probs = qcircuit(gamma, beta, G=G, p=p).numpy()[0]
    conf = configs.numpy()
    a = np.random.choice(np.arange(0, len(probs)), p=probs, size = shots)
    results = conf[a]
    results_tf = tf.convert_to_tensor(results)
    results_tf = tf.cast(results_tf, tf.float32)
    return results_tf

def NN_opt1(net_tf, gamma, beta, opt, qcircuit_stat, step):
    # Quantum circuit samples and reshaping
    result = qcircuit_stat(gamma, beta, G=G, p=p)
    shape = np.shape(result)
    result = tf.reshape(result, (shape[1], shape[0]))
    result = tf.cast(result,tf.float32)
    NNweights = net_tf.layers[0].weights
    if step == 0:
        with tf.GradientTape() as tape:
            x = net_tf(result)
            E = MaxCut_NN(G,x)
        #penalty = tf.reduce_sum(NNweights[0] - tf.eye(5))
        #print(penalty)
            cost = tf.reduce_sum(E)/shots
    else:
        with tf.GradientTape() as tape:
            x = net_tf(result)
            E = MaxCut_NN(G,x)
            penalty = tf.reduce_sum(tf.abs(NNweights[0] - tf.eye(8)))
            cost = tf.reduce_sum(E)/shots + 0.05*tf.abs(tf.cast(penalty, dtype = tf.float64))
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
    gamma_hyb = copy.deepcopy(gamma)
    beta_hyb = copy.deepcopy(beta)
    qcircuit(gamma, beta, G=G, p=p)
    qcircuit_stat(gamma, beta, G=G, p=p)
    initializer = tf.keras.initializers.Identity()
    net_tf = tf.keras.models.Sequential([
      tf.keras.layers.Dense(len(G.nodes), kernel_initializer=initializer, use_bias = False),
      tf.keras.layers.Activation("tanh")
    ])
    # Initialize the optimizers
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt_NN = tf.keras.optimizers.SGD(learning_rate = 0.05)
    
    # Define number of NN steps
    Hyb_steps = 300
    QAOA2_steps = 100
    VanQAOA = 200
    shots = 1000

    # Vanilla QAOA optimization
    for k in range(VanQAOA): # 50 before
        #print(k)
        gamma, beta, cost = QAOA_opt(gamma, beta, opt, Energies, qcircuit, Net=None, G=G, p=p)
    E_initial[i] = cost
    
    # Hybrid optimization
    for k in range(Hyb_steps):
        #print(k)
        gamma_hyb, beta_hyb, cost = QAOA_opt(gamma_hyb, beta_hyb, opt, Energies, qcircuit, Net=net_tf, G=G, p=p)
        NN_opt1(net_tf, gamma_hyb, beta_hyb, opt_NN, qcircuit_stat, k)

    # QAOA optimization step in original landscape. Fourth optimization step
    for k in range(QAOA2_steps):
        gamma_hyb, beta_hyb, cost = QAOA_opt(gamma_hyb, beta_hyb, opt, Energies, qcircuit, Net=None, G=G, p=p)
    E_final[i] = cost
