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
* Random Fourier features: $cos(W),\ W$ - is a (fixed) matrix of random Gaussian samples.
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
  * hidden layers
  * activations
  * neurons
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
However parity function is not very relevant to functions that we approximate in practice.
Besides, gradient descent is horrible at learning functions like parity.

### Questions:

* What kinds of non-linear activation functions exist besides element-wise functions?
* Activation functions define constraint on features that NN constructs.
Can we remove this constraint by not specifying activation function explicitly 
and let NN decide what activation function to use?


## [Lecture 3 (Part II)](https://www.youtube.com/watch?v=JLg1HkzDsKI) - "Manual" Neural Networks
* TODO: add general formula to compute gradients of multilayer feedworward network


## [Lecture 4](https://www.youtube.com/watch?v=56WUlMEeAuA) - Automatic Differentiation


## [Lecture 5](https://www.youtube.com/watch?v=cNADlHfHQHg) - Automatic Differentiation Implementation


## [Lecture 6](https://www.youtube.com/watch?v=CukpVt-1PA4) - Fully connected networks, optimization, initialization

### Fully connected networks
* Now that we have automatic differentiation framework we can add bias terms to linear layers. We didn't add bias previously because it's hard to manually derive gradients in this case
* To save memory we don't need to store the whole matrix of bias terms.<br>
  Istead we store single bias-vector and make use of **broadcasting**.

### Optimization

* When using Gradient Descent with large step size (learning rate) we can "bounce" between different points in parameter space or even diverge
* This can be fixed by using smaller step sizes, but the convergence time increases
* Simple Gradient Descent is rarely used in practice. There are more efficient variants of optimization algorithms

* Newtons method is an example of second order optimization method.<br>
  It uses both gradients and hessians to make parameter updates.
  In convex optimization this helps to directly point to global minimum.<br>
  When using $\alpha=1$ the method is said to use **full-steps**,
  otherwise (when $\alpha<1$) the algorithm is called **damped** Newtons method.<br>
  Main disadvantages:
  * It's very inefficient to compute second order gradients (hessians).<br>
    One of the reasons is that the size of hessian matrix for parameter-vector is quadratic 
    in the size of paramter-vector ( $n^2$ ).<br>
    However there are somewhat efficient methods to compute approximates of hessian.
  * For non-convex optimization (as is the case with real world deep learning problems), it's very unclear that we even **want** to use the Newton direction
  * It's also difficult to derive stochastic version of Newtons method

* Main optimization methods that are used in practice are:
  * **Momentum** (with/without bias correction, **Nesterov Momentum**).
    Uses exponential moving average of gradients to make parameter updates.
    This helps to avoid "bouncing" as in Gradient Descent with large step size
    * It usually converges faster than simple Gradient Descent
    * It makes use of the **momentum** term - exponential average of historical gradients. This introduces a king of **inertia** in parameter updates.
    * In biased version updates during first iterations are smaller than actual gradients (because of parameter $\beta<1$) - momentum term is **warming up**.
    * Unbiased version scales momentum term by a factor that corresponds to how long the optimization is run
    * Sometimes we want to use smaller steps in the beginning of the training - e.g. when we don't have good estimates of the right gradient direction. Thus biased Momentum migh work better than unbiased.
    * Nesterov variant uses parameters on the current and not previous iteration to compute momentum term. Sometimes it works better than the regular Momentum (converges faster).
  
  * **Adam** (with/without bias correction) - example of adaptive gradient method
    * The scale of gradients can vary widely for different parameters, especially across different layers of a deep network, different layer types, etc.
    * Adaptive gradient methods attempt to estimate this scale over iterations and then re-scale the gradient update accordingly.<br>
      This puts all gradient terms on similar scale.
    * In addition to **momentum** term, Adam introduces **adaptive scale estimation** - exponential average of squares of historical gradients.
    * Actually both **momentum** and **adaptive scale estimation** terms are biases in the same way. So when dividing one by other a kind of naturall unbiasing happens (although the latter term is under a square root). In general there are different variants of scaling, unbiasing of Adam.

* **Stochastic variant of Gradient Descent** is cheaper, faster but is also more noisy compared to Gradient Descent algorithm.<br>
  By the end of an epoch we will have multiple updates to parameters using SGD compared to single parameters update using plain Gradient Descent. And this single update is not necesserally better than the result of multiple SGD updates.
* Sometimes we even want noise to be present in parameters updates (as in the case with SGD). This can help us to get out of local minimum or other nasty regions in the loss function landscape.

* The amount of valid intuition about optimization methods we get from looking at simple (convex, quadratic) optimization problems **is limited**.<br>
  We need to **constantly experiment** to gain an understanding/intuition of how these methods actually affect deep networks of different types

### Initialization

#### Choice of initialization matters!

* We can't initialize parameters of feedforward network with all 0s, because all activations and gradiens will be 0 as well. A local minimum, but a bad one :)
* Let's initialize weights of feedforward network (100 hidden units, depth 50, ReLU nonlinearities) with random gaussians with mean = 0.
  * The choice of variance has huge implication for the resulting performance, weights, magnitude of gradients
  * $\large \sigma^2  = \frac{1}{n}, \sigma^2 = \frac{2}{n}, \sigma^2 = \frac{3}{n}$ variances differ only by the constant factor. However they result in norm of feedforward activations and norm of gradients that differ **on the order of magnitudes**!

#### Weights don't actually move "that much" during training
* It's really important where you start, what initialization use
* Differences in weights initialization with subsequent optimization can far outweight the differences in weights optimization alone.
* After optimization model weights often stay relatively close to their initialization values.
* **Kaiming initizalition**: $w_{i}\sim\mathcal{N}(0; \frac{2}{n_{i}})$, where $n_i$ is the size of individual actication on layer $i$<br>
  When **ReLU is a non-linearity function**, Kaiming initialization ensures that $z_i \sim \mathcal{N}(0, 1)$ for each hidden layer.<br>
  Derivation for element of activation matrix:<br>
  
  $z_i \sim \mathcal{N}(0,I_{n_i}),\ z_i \in \mathbb{R}^{n_i}$ - activation vector<br>
  
  $\large w_i \sim \mathcal{N}(0,\frac{2}{n} I_{n_i}),\ w_i \in \mathbb {R}^{n_i}$ - parameter vector <br>
  
  $Var(w_i z_i) = Var(\sum\limits^{n}\limits_{k=1}{w_{i, k} z_{i, k}}) = n Var(w_{i,k}) Var(z_{i,k})$ - because $w_i$ and $z_i$ are independent<br>
  
  ReLU operation keeps only positive values in the sum - approximately half of values for vector from gaussian. Thus:<br>
  
  $Var(\large z_{i+1}) = Var(ReLU(w_i z_i)) = Var(ReLU(\sum\limits^{n}\limits_{k=1}{w_{i, k} z_{i, k}})) = \frac{n}{2} Var(w_{i,k}) Var(z_{i,k}) = \frac{n}{2} \frac{2}{n} 1 = 1$<br>
  
  $\mathbb{E}z_{i+1} = \mathbb{E}(ReLU(w_i z_i)) = \sum\limits^{n}\limits_{k=1} ReLU(\mathbb{E}(w_{i, k} z_{i, k})) = \sum\limits^{n}\limits_{k=1} ReLU(\mathbb{E}w_{i, k} \mathbb{E}z_{i, k}) = 0$


### Questions:
* why parameter scaling is that important when optimizing parameters? why Adam sometimes performs better than Momentum?<br>
  need to get more intuition in differences of various optimization algorithms.<br>
  A helpful [Distill article](https://distill.pub/2017/momentum) about Momentum.
* how unbiased version of Momentum and Adam will look like if we use $\alpha^{*} = \alpha (1 - \beta)$ and won't use $(1-\beta)$ term?
* Prove that $Var(xy) = Var(x) \ Var(y)$ for independent $x$, $y$ variables


## [Lecture 7](https://www.youtube.com/watch?v=fzKNkS_5E6U) - Neural Network Abstractions

### Programming abstractions
* It's useful to learn from programming abstractions used in different automatic differentiation frameworks. It helps to:
  * understand why the abstractions were designed in this way
  * learn how to design new abstractions

* Caffe
  * Created based on `cuda-convnet` that was developed by Alex Krizhevsky in AlexNet framework
  * Introduces **Layer** abstractions (exp, sum, ...). It's a class that implements `forward` and `backward` methods.
  * Forward and backward computations are performed in-place.

* Tensorflow 1.0
  * Based on a **computational graph** architecture:
    * At first we describe forward computations
    * Gradients computations are performed by extending computational graph with new nodes.
      This nodes are computed using forward pass operator methods implemented early.
  * First DL framework that used computational graph architecture was `Theano`. 
    TensorFlow 1.0 shares some concepts with Theano.
  * TensorFlow was developed by Google. And Google usually develops its products in a way to support
    parallel and distributed execution
  * Designed in **declarative** style. This approach helps to separate developing environment from execution one.
    Any Tensorflow script consists of two parts: 
    * **Declaration** part. the computation graph is defined here
    * **Run part**. Here `tf.Session` object is created.
      It optimizes computational graph, caches nodes and runs computations on specified hardware (CPUs, GPUs, TPUs) - 
      local or remote, single machine or distributed cluster, sequentially or in parallel mode.

* PyTorch (needle as well):
  * Defined in **imperative** (also known as **define by run**) concept
  * The first framework that used imperative style to define computational graph was `Chainer`. PyTorch is based on it.
  * Executes computations as we construct computational graph
  * Allows for easy mixing of python control flow and computational graph construction.<br>
    e.g. we can construct computataional graph **dynamically** by examining its nodes values.<br>
    This allows to process input sequences of variable length (as in NLP) or build stochastic computational graph
    where number of layers is determined randomly.
    In contrast, TensorFlow is said to provide the way to build only **static** computational graphs.
  * Such imperative style is widely used in research and model prototyping
  * It's much more convenient to debug PyTorch program compared to TensorFlow because of imperative style computations.
  * However TensorFlow 1 still provides much more opportunities to optimize computational graph. 
    This optimization benefits both training and inference.
  * However there are number of optimization techniques for PyTorch: lazy evaluation, just in time compilation, etc.
    
### High level modular library components

