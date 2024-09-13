# Smooth Max Pooling

## Overview

Traditional max pooling operations can result in sparse gradients, which might affect the training of neural networks. The smooth max pooling implementation here uses LogSumExp to approximate the maximum operation while ensuring that gradients are distributed across all neurons.

## Mathematical Background

The LogSumExp (LSE) function provides a smooth approximation to the maximum function. Given a vector $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, the LSE function is defined as:

$$
\text{LSE}(x_1, x_2, \ldots, x_n) = \log \left( \sum_{i=1}^n \exp(x_i) \right)
$$

### Bounds

The LSE function approximates the maximum with the following bounds:

$$
\max(x_1, x_2, \ldots, x_n) \leq \text{LSE}(x_1, x_2, \ldots, x_n) \leq \max(x_1, x_2, \ldots, x_n) + \log(n)
$$

where $n$ is the number of elements in the vector.

### Convexity and Derivatives

The LSE function is convex and strictly increasing. Its gradient corresponds to the softmax function:

$$
\frac{\partial}{\partial x_i} \text{LSE}(\mathbf{x}) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}
$$

This ensures smooth gradients and better propagation through the network.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
