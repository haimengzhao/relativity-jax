# Relativity with Jax

In this repo, we use the automatic differentiation in Jax to compute Christoffel symbols, Riemann curvature tensors, Ricci tensors, Ricci scalars and Einstein tensors. GPUs & TPUs are supported. These results can be further used for the computation in Riemannian geometry, numerical relativity, the training of physics-inspired neural networks, etc.

The implementations are in `riemann_geo.py`. Two Jupyter notebooks give examples of usage:

- `sphere.ipynb` computes the Christoffel symbols, Riemann curvature tensor, Ricci tensor and Ricci scalar of the 2 dimensional sphere.
- `nn.ipynb` uses a fully-connected neural network to parameterize the metric. Then we use gradient descent to minimize a loss function, and obtain the solutions we want.
