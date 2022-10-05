# Description

Notes for CMU DL Systems Course (2022 online public run).

# Resources:
* [Course main page](https://dlsyscourse.org/)
* [YouTube channel](https://www.youtube.com/channel/UC3-KIvmiIaZimgXMNt7F99g)


# Notes

## [Lecture 3 (Part I)](https://www.youtube.com/watch?v=OyrqSYJs7NQ) - "Manual" Neural Networks

* The main point of using multi-layer Neural Networks is to automate feature construction process.
We no longer need to design features ourselves - instead we put some restrictions on the type of features
(by defining activation function) and features are constructed by the NN automatically.
* Random Fourier features: $cos(W),\ W -$ is a (fixed) matrix of random Gaussian samples.
These features work great for many problems.
* A **Neural Network** refers to a particular type of hapythesis class, 
consisting of multiple, parameterized differentiable functions (aka layers) 
composed together in any manner to form the output. <br>
These functions are composed via non-linear functions (aka **activation functions**) 
that make Neural Network a **non-linear hypothesis class**.
* **Deep Learning** refers to a Machine learning that uses multi-layer Neural Networks (non-linear functions) as hypothesis class. <br>
The only constrain on the network te be considered "deep" is to have $\ge 1$ hidden layer - i.e. belong to a non-linear hypothesis class.
* A **Layer** is usually referred to as weights that transform one features to other (i.e. weights matrix)
* In most cases NN perform well even without a bias term in linear layers
* Intermediate features of multi-layer NN are called:
  *   hidden layers
  *   activations
  *   neurons
* **Universal approximation theorem**: a one-hidden-layer NN can approximate any smooth function _f_ over closed domain $\mathcal{D}$ given any accuracy $\epsilon > 0$ <br>
The size of a hidden layer grows exponentially compared to the input size.

Why to use deep networks?
* 1 hidden layer is enough to approximate any smooth function (according to Universal approximation theorem)<br>
However size of hidden layer grows exponentially compared to the input size.
* Empirically it's better (more efficient in terms of number of parameters)
to have a number of hidden layers instead of single hidden layer.
* I.e. empirically it seems like they work better for a fixed parameter count!
* Especially for structured networks (convnets, recurrent nets, transformers).

There are other not very important reasons to use Deep Networks:
* Deep networks resemble the way that human brain works. 
The brain does multi-stage processing of its inputs before it reaches the final decision making center.
* From circuit theory. Certain functions can be represented much more efficiently 
using a multi-stage architecture. A classic example is the **parity function**.
However parity function is not very relevant to functions that we approximate in practise.
Besides, gradient descent is horrible at learning functions like parity.

### Questions:
* What kinds of non-linear activation functions exist besides element-wise functions?
* Activation functions define constraint on features that NN constructs.
Can we remove this constraint by not specifying activation function explicitly 
and let NN decide what activation function to use?

## [Lecture 3 (Part II)](https://www.youtube.com/watch?v=JLg1HkzDsKI) - "Manual" Neural Networks
