# Relaxation-of-QAOA-energy-landscapes
In this repository you can find the codes that have been used for obtaining the results shown in the article "Relaxation of QAOA energy landscape with a Neural Network" by J. Rivera-Dean, Patrick Huembeli and Joseph Bowles.

The codes are structured in different parts, which we now specify:

- **Graph Loader.** Here we load some graphs that have been generated previously and whose data is saved in the folder `Graphs`. We manage these graphs via the `networkx` package.
- **Neural network definitions**. Here we define the neural network using `pytorch`.
- **QAOA definitions**. Here we define the QAOA gates and the corresponding quantum circuit. Furthermore, we also introduce some other functions such as the cost function definition, for which we shall perform the calculations analitycally. For running the QAOA we use the Pennylane package together with tensorflow.
- **Random number generator**. Generates the initial QAOA parameters randomly between [0,2*pi].
- **Statistical noisy circuit**. This is a circuit which uses some angles for the QAOA circuit, and generates a set of bitstrings which fed the neural network.
- **Optimization**. Definition of the QAOA and the neural network optimizers.
- **Running the code**. Here is where the code runs.
- **Saving the resuls**. Save the results that we want to analyze prior to a *saving directory* definition.

More information about Pennylane can be found in: V. Bergholm et al. Pennylane: Automatic differentiation of hybrid quantum-classical computations arXiv:1811.04968 (2018) url: https://arxiv.org/abs/1811.04968.

More information about keras (optimization for tensorflow) can be found in: F. Chollet. Keras. https://github.com/keras-team/keras (2015)

More information about PyTorch can be found in: A. Paszke et al. PyTorch: An imperative style, high-performance deep learning library. In H. Wallach et al. (eds.), *Advances in Neural Information Processing Systems* vol. 32 (Curan Associates, Inc., 2019), pp. 8026-8037. url: https://proceedings.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf. More information can also be found in: https://pytorch.org/
