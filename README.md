# Description

Notes for CMU DL Systems Course (2022 online public run).

# Resources:
* 💻 [Course main page](https://dlsyscourse.org/)
* 🎥 [YouTube channel](https://www.youtube.com/channel/UC3-KIvmiIaZimgXMNt7F99g)
* 📒 [Public jupyter notebooks from lectures](https://github.com/dlsyscourse/public_notebooks)

# Table of Contents
* Lecture 3
  * [Part I - "Manual" Neural Networks](#lec3-1)
  * [Part II - "Manual" Neural Networks](#lec3-2)
* [Lecture 4 - Automatic Differentiation](#lec4)
* [Lecture 5 - Automatic Differentiation Implementation](#lec5)
* [Lecture 6 - Fully connected networks, optimization, initialization](#lec6)
* [Lecture 7 - Neural Network Abstractions](#lec7)
* [Lecture 8 - Neural Network Library Implementation](#lec8)
* [Lecture 9 - Normalization and Regularization](#lec9)
* [Lecture 10 - Convolutional Networks](#lec10)
* [Lecture 11 - Hardware Acceleration](#lec11)
* [Lecture 12 - GPU Acceleration](#lec12)
* [Lecture 13 - Hardware Acceleration Implemention](#lec13)
* [Lecture 14 - Implementing Convolutions](#lec14)
* [Lecture 15 - Training Large Models](#lec15)

# Notes

<a id="lec3-1"></a>

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


<a id="lec3-2"></a>

## [Lecture 3 (Part II)](https://www.youtube.com/watch?v=JLg1HkzDsKI) - "Manual" Neural Networks
* TODO
* TODO: add general formula to compute gradients of multilayer feedworward network


<a id="lec4"></a>

## [Lecture 4](https://www.youtube.com/watch?v=56WUlMEeAuA) - Automatic Differentiation

* TODO

* Define **adjoint** $\overline{v}_i$ as a partial derivative of output scalar $y$ (typically a loss function)
  with respect to an intermediate value node $v_i$:<br>
  $\overline{v}_i = \dfrac{dy}{d v_i}$

* **Partial derivatives** with respect to same parameter vary in dimensions depending on 
  what variable is been differentiated:

  For example, $x \in \mathbb{R}^m,\ W \in \mathbb{R}^{m, n},\ f = x W \in \mathbb{R}^n$

  $y = y(f) \in \mathbb{R}$ - a scalar valued function. Thus $\dfrac{dy}{df} \in \mathbb{R}^n$
  
  $\dfrac{df}{dx} = W^T \in \mathbb{R}^{n,m}$

  $\dfrac{dy}{dx} = \dfrac{dy}{df} \dfrac{df}{dx} = \dfrac{dy}{df} W^T \in \mathbb{R}^m$

  $shape(\dfrac{dy}{dx}) = shape(x)$
  
  $shape(\dfrac{df}{dx}) \ne shape(x)$


<a id="lec5"></a>

## [Lecture 5](https://www.youtube.com/watch?v=cNADlHfHQHg) - Automatic Differentiation Implementation
* TODO

<a id="lec6"></a>

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
  
  $\large Var(w_i z_i) = Var(\sum\limits^{n}\limits_{k=1}{w_{i, k} z_{i, k}}) = [w_i \text{ and } z_i \text{ are independent}] = n Var(w_{i,k}) Var(z_{i,k}) = n \frac{2}{n} 1 = 2$ <br>
  
  ReLU operation keeps only positive values in the sum - approximately half of values for vector from gaussian. Thus:<br>
  
  $Var(\large z_{i+1}) = Var(ReLU(w_i z_i)) = Var(ReLU(\sum\limits^{n}\limits_{k=1}{w_{i, k} z_{i, k}})) = \frac{n}{2} Var(w_{i,k}) Var(z_{i,k}) = \frac{n}{2} \frac{2}{n} 1 = 1$<br>
  
  $\mathbb{E}z_{i+1} = \mathbb{E}(ReLU(w_i z_i)) = \sum\limits^{n}\limits_{k=1} ReLU(\mathbb{E}(w_{i, k} z_{i, k})) = \sum\limits^{n}\limits_{k=1} ReLU(\mathbb{E}w_{i, k} \mathbb{E}z_{i, k}) = 0$


### Questions:
* Why parameter scaling is that important when optimizing parameters? why Adam sometimes performs better than Momentum?<br>
  need to get more intuition in differences of various optimization algorithms.<br>
  A helpful [Distill article](https://distill.pub/2017/momentum) about Momentum.
* How unbiased version of Momentum and Adam will look like if we use $\alpha^{*} = \alpha (1 - \beta)$ and won't use $(1-\beta)$ term?
* How to overcome the problem that trained weights are relatively close to initialization weights? Do we need to
  construct an ensemble of models initialized with different strategies?
* Prove that $Var(xy) = Var(x) \ Var(y)$ for independent $x$, $y$ variables


<a id="lec7"></a>

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

#### Module
* Helps to compose things together
* Any Module follows a common principle of "tensor in, tensor out".
  Sometimes Module can take several tensors as inputs, but usually - one.
* Module does the following:
  * Defines a forward pass: a way to compute outputs for given inputs
  * Provides storage and defines initialization of trainable parameters
  * In some Deep Learning frameworks modules also handle gradients computation (like in Caffe).<br>
    But in frameworks based on computational graph abstraction (TensorFlow, PyTorch) we only need to define
    a forward pass for a module - gradient computations are performed **automatically** 
    by Tensor and Operation (add, div, sum, etc.) objects!

* Loss function is a special kind of Module. It takes input tensors and outputs scalar value (tensor of rank 0)
* Multiple losses can be combined together. For example in the case of object detection and recognition one loss
  would find object bounding boxes and the other one will predict its class value. We can then take (possibly weighted)
  sum of two losses and execute `.backward()` method of scalar that is result of summation to propagate gradients.
* Some modules should behave differently while training and inference. For example we often don't need to compute
  loss values during inference. A special Module's parameter allows to control this behaviour.

#### Optimizer
* **Optimizer** is another high level component. It allows to:
  * acces model parameters and perform optimization steps on them
  * keep track of state variables such as momentum terms for SGD with momentum
* Regularization can be performed:
  * Either in loss function: $l = l + ||w||_2$
  * Or directly in the optimizer, as a part of weights update. This is called **weight decay**


#### Initialization
* We need to consider magnitudes of initialization values:
  * If we initialize parameters with too small values, after a few layers of propagation, values will become close to 0
  * If initializing with too large values, after a few layers of propagation, values will explode
* Initialization routines that try to control weights magnitude depending on model architecture to ensure that
  overall magnitude of values (activations? gradients?) do not change very much.

#### Data loader and preprocessing
* Besides data loading and preprocessing, allows to shuffle and augment data by applying various transformations
* Data transformations usually need to be implemented as separate modules.<br>
  This allows to easily combine them in preprocessing pipelines, change one transformation with another.<br>
  Combining all data transformations and augmentations in a single function is usually a bad approach :)
* Data augmentation can account for significant portion of prediction accuracy boost for Deep Learning models

#### Other components:
* Learning rate schedulers
* Hyperparameter tuning component
* Trainer - model training abstraction
* Callbacks to store movel metrics. e.g. TensorBoard
* Experiment tracking frameworks

#### Deep learning is modular in nature
* We can build new models on the backbone of existing ones by simply changing the loss function, for example
* If we want to explore the effect of various optimization strategies on model performance we simply 
  change the Optimizer - the rest of pipeline remains the same


#### Comparison of Caffe and PyTorch. Separation of gradient computation from module composition
* It's highly effective to **separate gradient computation from module composition** in 2 distinct levels of API.
* In Caffe each Module defines both forward and backward passes. 
  So, gradient computation is coupled with module composition.<br>
  This makes it hard to implement complex modules such as Residual module. Because gradient computation in this case
  is not trivial: we need to manually specify gradient computations for all intermediate parameters 
  in the Residual block to be able to obtain gradient for input Tensor.
* In PyTorch for each module we define only the forward pass (besides weights initialization). 
  When actual tensors are passed through the module object,
  computational graph gets extended by new nodes. Gradients computation and backpropagation is handled by
  Tensor and Operation (add, div, sum, etc.) objects. 
  This separation helps to build complex models easily, e.g. those that contain Residual modules.<br>
  Such Module implementation allows to focus on constructing new nodes for computational graph 
  and forget about automatic differentiation.

<a id="lec8"></a>

## [Lecture 8](https://www.youtube.com/watch?v=uB81vGRrH0c) - Neural Network Library Implementation

### Gradient update implementation

* Gradient update should be performed on detached tensors. Otherwise **memory leak** will happen:
    * because a new node is added to computational graph (memory is spent on this node)
    * and, moreover, this node holds references to its inputs and to their inputs recursively. 
      thus, previous graph nodes (there might be a lot of them) cannot be released by garbage collector.

  It's also important to detach **both** previous parameter tensor **and gradient tensor**.
  ```python
  w = w.detach() - lr * grad.detach()
  ```
* Gradient update is usually implemented inplace. To do so we need to add setter method `data` that accesses underlying tensor data. 
  Alternatively, we can use `realize_cached_data()` method:
  ```python
  w.data = w.detach() - lr * grad.detach()
  ```

### Numerical stability

* limited float numbers precision may result in numerical errors and overflows
* for example, `exp(100) == nan`
* in order to compute softmax for arbitrary inputs we can apply following transformation:<br>
  $\large z_i = \dfrac{exp(x_i)}{\sum\limits_{k} exp(x_k)} = \dfrac{exp(x_i - c)}{\sum\limits_{k} exp(x_k - c)},\ c \in  \mathbb{R}$<br>
  we can take $c = max(x_i)$. thus all inputs for `exp` operation will be $\le 0$ 
  and this will improve numerical stability.
* similar principles hold when we compute **logsoftmax** and **logsumexp** operations


### Other notes

* It's important to appreciate simple and elegant modular design of modern Deep Learning frameworks. 
  This design might seem obvious now, but back in the past it was not yet invented.


<a id="lec9"></a>

## [Lecture 9](https://www.youtube.com/watch?v=ky7qiKyZmnE) - Normalization and Regularization

### Initialization
* Initialization matters a lot
* MNIST 50 layer network example:
  With different variance values used to initialize weights:
  * before training:
    * activations norm **change linearly** over layers. the slope of change depends on variance value used
    * gradients norm stays about the same across layers. 
      magnitude of gradient norm depends on variance value used (is proportional)
    * weights variance stays about the same across layers. 
      magnitude of weights variance depends on variance value used (is proportional)
  * after training completed (< 5% error):
    * activations norm increases over layers for all variance value used
    * gradients norm changes (no clear dependency. some curve) over layers.
      no clear dependency on variance value used
    * weights variance stays about the same across layers. 
      magnitude of weights variance depends on variance value used (is proportional)
  * **NOTE**: weights variance across layers is almost the same **before and after training**.
    training somehow did not impact weights variance.<br>
    **Question**: Does the same weights variance indicate that weight stay about the same? the network should have
    learned -> weights should have been optimized. but it's weird that variance stayed the same

### Normalization
* Instead of (or together with) trying to initialize all weights "correctly" we can directly change layer activations
  to have any required mean and variance.<br>
  We can perform such normalization either element-wise (**Layer norm**) or feature-wise (**Batch norm**)
* Normalization allows to wory about initialization much less. 
  We are no longer afraid of exploding or vanishing activation. 
  But the final model weights after training still depend on initialization approach.
* Using either Layer norm or Batch norm fixes the issue with activation norm increasing or decreasing over layers
  and huge difference in gradients magnitudes. 
  Now all activations and gradients stay relatively constant across layers 
  and their magnitude almost doesn't depend on variance value used in initizalition

#### Layer normalization
* **Layer norm** normalizes activations on-the-fly. 
  It adds a new layer $Z_{i+1}$ that equals to a normalized previous layer acivations:<br>
  $z_{i+1} = \dfrac{z_i - \mathbb{E}[z_i]}{\sqrt{Var[z_i] + \epsilon}}, z_i$ - 
  is an activation for *i*-th data sample.<br>
  Layer norm is applied **example-wise** (row-wise): 
  activations for each data sample are normalized independently.
* **For standard Fully Connected Networks it's often harder to train model to have low loss when Layer Norm is used**<br>
  One of the reason is that relative norms and variances of different examples 
  may be a useful feature to discriminate (e.g. classify) examples.

#### Batch normalization
* Contrary to Layer norm, **Batch norm** normalizes activations feature-wise (column-wise) over the minibatch
* This allows for different data samples to preserve different mean and variance (across their features).
  And this might be used by network to discriminate examples.
* At the same time activations are get normalized. That brings control over activations magnitude and helps to prevent
  them from exploding in the deeper layers
* Such feature-wise normalization resembles the way features get preprocessed for classical machine learning algorithms.
* An important note about Batch Norm is that it introduces **minibatch dependence**: 
  outputs for single example in a minibatch depend on the rest examples in minibatch
* To deal with minibatch depency at test (evaluation) time Batch Norm layers keeps track of **running averages** 
  for mean and variance of each feature:<br>
  <!--  For some reason we need to escape `_` in subscripts here in order for GitHub to render the formula correctly -->
  $\hat{\mu}\_{i} = (1 - \beta) \hat{\mu}\_{i-1} + \beta \mu\_{i}$<br>
  $\hat{\sigma}^2_{i} = (1 - \beta) \hat{\sigma}^2_{i-1} + \beta \sigma^2_{i}$<br>
  where $\beta > 0$ - is a momentum parameter
* **Inference for trained networks that contain Batch Norm layers must be run only in `eval` mode**.<br>
  Doing so in `train` mode
  makes Batch Norm to update running averages for mean and variance. So normalization becomes incorrect 😅<br>
* **However**! Running Batch Norm at test time substantially improves classifiers performance 
  on distribution shift (out-of-distribution data)!<br>
  Paper: [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)
* Batch Norm was developed as a techique to help optimization or avoid Dropout.<br>
  Then lots of research happened that challange explanations why Batch Norm works.<br>
  And now it's getting a third life as a technique to improve networks robustness to distributional shift.
* TODO: Add links to Batch Norm exploration papers

### Regularization
* Deep Neural Networks are often **overparameterized models**: 
  they contain more parameters than the number of training examples
* Under certain assumptions it means that such network may fit training data completely (loss == 0)
* This may lead to overfitting: model performs well on training data but can't generalize to unseen data
* However, Deep Neural Networks are still able to generalize well (though many large models will often still overfit)
* To fight overfitting regularization is used
* **Regularization** - is a process of limiting complexity of a the hypothesis class
  to help models generalize better to unseen data
* There are 2 types of regularization
  * **Implicit regularization**
    * Complexity is limited by particular algorithms or architectures used
    * If we say that class of Deep Neural Networks is overparameterized, we consider every possible point in
      weights feature space and, I guess, every possible model architecture.<br>
      By making **specific choices on model architecture and optimization** we implicitly limit number of available
      implementations of this hypothesis class.<br>
      In practice, when we train a network **we are not optimizing over the space of all possible neural networks**.
    * **SGD with a given weight initialization** is an example of implicit regularization. 
      The reason is that particular weight initialization makes only specific part of weights feature space 
      attainable during training (recall that weights do not change much during training).
      It means model can't reach every possible point in weights feature space - 
      thus limiting complexity of a hypothesis class.
  * **Explicit regularization**
    * Refers to modifications made to the network and training procedure explicitly intended to regularize the network
    * Most common examples: 
      * $L_2$ regularization, a.k.a. weight decay
      * Dropout

#### $L_2$ regularization (weight decay)
**TODO. Detalize notes**
* apply to activations of network, not to weights
* plain implementation changes overall activation (and weights, I guess)

* in classical Machine Learning smaller weights meant smoother function. Lipschitz constant
* general optimization problem of all Machine Learning :)
* updated version of weight update :)
* $\lambda$ - tradeoff between actual loss and regularization term
* **weight decay** etymology
* $L_2$ norm term is better considered as a part of a loss function and not of optimization algorithm.
  However, now it's often implemented as part of optimization algorightms.
* Caveat. Unclear how weights norm affect training.<br>
  Parameters magnitude may be a bad proxy of deep network complexity
* we can ignore weight decay completely. especially when using normalization layers (Batch Norm, Layer Norm)<br>
  (?) normalization of activations leads to normalization in weights (?)

#### Dropout
**TODO. Detalize notes**

* plain dropout changes distribution (magnitude) of activations. variance of weights no longer remain steady.
* need to scale weights by probability to keep them
* typically (not always) applied only during training, not in test time
* may seem very odd: we must be massively changing the function that we are approximating
* makes network robust to missing activations. but it's not obvious way of thining. better to thinks of it as 
  a **Stochastic approximation**
* Dropout takes the idea of a stochastic approximation 
  (like in SGD we approximate true gradient due to noise introduced by batch selection)
  to activations of a network. 
  Doing so provides similar degree of regularization as SGD provides for traditional training objective

### Interaction of initialization, optimization, normalization and regularization
**TODO. Detalize notes**

* There are many ways to ease networks optimizations procedure and make them perform better
* And all of them interact with each other
* We don't really know how these interactions work or what is the reason and how does each individual techique works
* We don't even know how Batch Norm works exactly :)<br>
  And there have been a lot of discussion on the nature of Batch Norm (see above)<br>
  There are similar discussions for other techniques: dropout, weight decay, etc.
* TODO: refactor 3 points below as they are kind of similar
  * Sometimes it's shocking how similar very different architectures and training methods work
    So we don't need to try out all possible combinations of design choices, 
    as most of them will perform relatively similar
  * The "good news" is that in many cases, it seems possible to get similarly good results with wildly different 
    architectural and methodological choices
  * I.E. we can get good performance with variety of different methods.

### Questions:
* when explaining the effect of initialization norms and variances were used together. probably "variance" is a typo?
  if not, why we used weights variance instead of weights norm?<br>
  probably because in initialization we control variance and not norm.<br>
  if so, it would also be interesting to examing weights norm across layers.
* Initialization: Does the same weights variance indicate that weight stay about the same? the network should have
  learned -> weights should have been optimized. but it's weird that variance stayed the same
* Batch Norm: can we use simple averaging instead of exponential averaging to compute running averages?
* Batch Norm: why do we use running averages only at the test time? Can we use these estimates for mean and variance
  while training instead of current minibatch statistics?
* Change in activations magnitude changes weight magnitude, I guess. 
  This was implicitly meant in discussion of:
  * uselessness of weight decay for networks with normalization layers
  * plain dropout changing activations distribution. and as a result changing distribution of weights


<a id="lec10"></a>

## [Lecture 10](https://www.youtube.com/watch?v=-5RPPjn0hPg) - Convolutional Networks 

### Elements of practical convolutions

* Drawbacks of using Fully Connected Layers for images:
  * A LOT of parameters per single layers
  * FCN do not capture image invariants (objects, patterns). 
    A slight shift of image by 1 pixel might lead to completely different activations
* One of dominant ideas in modern Deep Learning is to 
  choose such types of architectures that preserve 
  initial structure of data. For example, convolutions help to preserve image structure and invariances.
* Convolutions in DL and Signal Processing mean different things.
  Signal Processnig analog for "convolution" in Deep Learning is "correlation".
* Multi-channel convolutions contain a convolutional filter for
  **each input-output channel pair**. Single output channel is sum
  of convolutions over all input channels
* A better view on convolutions is to view:
  * each input "pixel" as a vector with len = number input channels
  * each output "pixel" as a vector with len = number output channels
  * each filter "pixel" as a matrix 
    $W_{i,j} \in \large \mathbb{R}^{c_{in} \times c_{out}}$
  * output "pixel" vector is calculated as **matrix-vector product**:<br>
    $\large z_{i,j} = [k=3] = x_{i-1,j-1} W_{i-1, j-1} + x_{i-1,j} W_{i-1, j} + ... + x_{i+1,j+1} W_{i+1, j+1}$
* Such way of thinking helps to represent a single convolution as a set of
  $k \times k$ matrix multiplications, and their result is then summed.<br>
  This **helps to implement convolutions**
* Usually, only odd-sized filters are used. Because it's more convenient
  to calculate output shapes, padding size compared to even-sized filters.
* To build high level representations and reduce computations we often 
  want to reduce image resolution (H x W). This is usually achieved with
  the help of **Pooling** and **Strided Convolutions**
* There are cases when convolutions still have large number of parameters -
  e.g. when there is large number of input or output channels.
  This can lead to overfitting + slow computations.<br>
  In this cases **Grouped Convolutions** are used - groups of channels in
  output depend only on corresponding groups of input channels 
  (equivalently, enforce filter weight matrices to be **block-diagonal**).<br>
  An extreme case of Grouped Convolutions are **Depth-wise Convolustions** -
  single input channel maps to a single output channel.
* Convolutions have a relatively small **receptive field** size.
  It's the key idea of convolutions (we analyze only part of input image),
  but sometimes it poses a problem and we want to increase receptive field.<br>
  This could be done with **Dilations** - filter points get spread out.<br>
  However, dilations are less used these days - people tend to use 
  **concatenation to patches** instead

### Differentiating
* It's important to implement convolutions as **atomic operations**.<br>
  Because we don't want to store all intermediate matrix-vector multiplication results 
  before summing them up - as it creates huge memory consumption in computational graph.

* If we flatten outputs $f$, we can rewrite the convolution $f = conv(x, W)$
  as a **matrix-vector product** treating **inputs** $x$ as the vector:
  
  $f = conv(x, W) = \widehat{W} x$, where 

  $x = (x_1, x_2, ..., x_m)^T$

  $\widehat{W} = band(w_1, w_2, ..., w_{k \times k})$ is a **banded matrix** for a $k \times k$ filter $W$.

* Using this version of convolution, we can derive partial derivative $\dfrac{df}{dx}$ as:<br>
  $\dfrac{df}{dx} = \dfrac{d \widehat{W} x}{dx} = \widehat{W}$

  And the adjoint $\large \overline{x}$ is derived as follows:
  
  $\large \overline{x} = \dfrac{dy}{dx} = \dfrac{dy}{df} \dfrac{df}{dx} = \widehat{W}^T \dfrac{dy}{df}$

  $\widehat{W}^T = band(w_{k \times k}, ..., w_2, w_1)$ - 
  is also a banded matrix, but with reverse order of diagonals.
  
  * Based on that we can **represent adjoint $\large \overline{x}$ as convolution** 
    of incoming gradient $\dfrac{dy}{df}$ with a flipped filter (flipped order of filter-values):
  
    $\large \overline{x} = \widehat{W}^T \dfrac{dy}{df} = conv(\dfrac{dy}{df}, flip(W))$

    i.e. **multiplying by the transpose of a convolution is equivalent 
    to convolving with a flipped version of the filter.**

  * And we don't even need to construct and store either of $\widehat{W}$ and $\widehat{W}^T$ matrices.<br>
    It would also be impractical because these matrices contain a lot of 0s.


* To compute another adjoint, $\large \overline{W}$, we can similarly rewrite the convolution as a matrix-vector
  product, now treating the **filter** as a vector (instead of treating input as a vector, like previously)
  and expanding $x$ vector to a $\widehat{X}$ matrix using **im2col** operation:

  $f = conv(x, W) = \widehat{X} w$

  $\dfrac{df}{dW} = \widehat{X}$

  * Matrix $\widehat{X}$ is much more dense compared to a $\widehat{W}$
  * And it turns out that in many cases **the most efficient way to implement convolultions**<br>
    is to first explicity construct $\widehat{X}$ im2col matrix 
    and then perform the convolution as a matrix-matrix product (see [Lecture 14](#lec14))<br>
    $f = conv(x, W) = \widehat{X} w$
  * Matrix $\widehat{X}$ has duplicated values from $x$ (e.g. multiple copies of $x_1$).<br>
    And it often ends up been worthwhile to duplicate the memory 
    for the sake of been a bit more efficient 
    when it comes to the computations of matrix-matrix multiplication. 
  * We want to create $\widehat{X}$ matrix in convolutional operator (in `ops`) 
    instead of in Computational graph (in a `Layer`) 
    to avoid creating a lot of memory for redundand nodes.

* To implement convolutions properly we first need to understand 
  how matrices and vectors and tensors are stored in memory.<br>
  Convolutions may be implemented efficiently by manipulating stride operations 
  in the internal representation of matrices.

### Convolution implementations comparison
* If the kernel size is small, convolution via im2col is the fastest one.<br>
  When kernel is large, it is still faster than a bunch of matrix multiplications (?)
* When the kernel is small a number of other optimizations can be implemented (as in pytorch)
* Practical comparison is performed in [Lecture 14](#lec14) and in `Lecture_14_code_notes.ipynb` notebook


<a id="lec11"></a>

## [Lecture 11](https://www.youtube.com/watch?v=es6s6T1bTtI) - Hardware Acceleration

* Each type of runtime device (cpu, gpu, tpu, mobile, etc.) requires individual tensor linear algebry library
* Next we will describe main optimization techniques used for efficient computations
### Vectorization
* Leverages vector registers to load contiguous blocks of memory.<br>
  e.g. `load_float4` can load 4 contiguous blocks of float values.
* Requires arguments to be alligned to fixed length. because we read in blocks.
* That requires to allocate memory for intermediate computations in aligned way as well. The memory must be aligned both to:
  * platform word size (8 bytes for 64-bit platform)
  * and to the block size used in vectorized computations

### Data layout and strides

* CPU requires data to be stored in flat way (1D array)
* To store multidimensional arrays we need to flatten them
* Orders:
  * Row major: `A[i, j] = A[i * A.shape[1] + j]`
  * Column major: `A[i, j] = A[j * A.shape[0] + i]`
  * **Strides format**: `A[i, j] = A[i * A.strides[0] + j * A.strides[1]]`<br>
    More general order: row major and column major can be derived from strides format.<br>
    Also **generalizes well to a mutli-dimensional arrays**.
* Strides format:
  * Advantages. Can easily perform:
    * Slice: change the begin offset and shape
    * Transpose: swap the strides
    * Broadcast: insert a stride equals 0
  * Disadvantages: memory access no longer continuous
    * Makes vectorization harder
    * Many linear algebra operations may require compact (make continuous?) the array first

### Parallelization

* Executes the computation on multiple threads
* OpenMP is an example of parallelization framework

### Matrix multiplication 

* We will consider matrix multiplication
  in a following transposed variant:<br>
  $C = A B^T,\ C_{i,j} = \sum\limits_k A_{i,k} B_{j,k} $<br>
  All matrices have $n \times n$ size.
* Many libraries use same vanilla $O(n^3)$ algorithm 
  but apply optimization techniques to make computations efficient
* Depending on where the data resides, 
  the time cost of fetching the data can be very different.<br>
  [Latency Numbers Every Programmer Should Know ](https://gist.github.com/jboner/2841832)<br>
  Access to DRAM (200ns) is 400 times slower than acces to L1 Cache (0.5 ns)
* Optimizations discussed in this lecture:
  * Register tiling
  * Cache line aware tiling
  * Their combination
* The **main trick** in optimization is:
  * load the data
  * store it on the fast memory
  * **reuse it as much as possible**

#### Register tiling 
  * Decreases number of DRAM accesses 
    and increases number of registers usage
  * We split A into $v_1 \times v_3$ - sized blocks<br>
    and B into $v_2 \times v_3$ - sized blocks.<br>
    And for each `a, b` block pair we compute dot product 
    `c = dot(a, b.T)` of size $v_1 \times v_2$
    and add it to corresponding block in C matrix.
  * `a, b, c` are stored in registers
  * The dot product above may be computed using simple for loops
  * Data load cost: $dramspeed \times (n^3/v_2 + n^3/v_1)$
  * Number of registers used on each iteration: 
    $v_1 v_3 + v_2 v_3 + v_1 v_2$
  * Most of processors have limited number of registers.<br>
    We want to have $v_1$ and $v_2$ as large as possible to reduce 
    data loading costs,<br>
    but we also need to make sure that total number of registers
    does not exceed number of available registers.
  * $v_3$ does not affect loading cost (number of operations).<br>
    We can pick $v_3 = 1$ and then increase $v_1$ and $v_2$ 
    as large as needed.
  * Ususally we set $v_1 = v_2$, but sometimes
    we need a bit of assymetry, 
    if number of available registers is not a perfect square (?)
  * The reason that memory loading cost is reduced is that
    we **reuse** already loaded data.<br>
    We reuse `a` $v_2$ times and reuse `b` $v_1$ times.

#### Cache line aware tiling
  * Prefetch line blocks in L1 cache. No registers are used.
  * Compute dot product for each row pairs of prefetched line blocks.
    This could be done using Regitster tiling as above (leveraging registers that load data from L1 cache now)
* Combine them together: Cache line aware tiling + internal Register tiling

#### Data reuse patterns
* Besides parallelizaiton and vectorization we can speed up computations
  by **reusing data** during computations.
  * It helps to avoid loading the same data in different places
  * We can also copy data that is being reused to a faster memory to speed up data read/writes
* We can analyse individual equations to find **possibilities to perform tiling**
  * For example, in Cache line aware tiling we reuse same matrix rows
  * And in inner matrix multiplication operation: `C[i][j] = sum(A[i][k] * B[j][k], axis=k)`<br>
    access to `A` matrix is independent of `j` dimension of `B` matrix.<br>
    So we can tile the `j` dimension of B matrix by `v` and reuse A data `v` times<br>
    (the exact same thing that I come up with during Homework 3 😅. `(i -> k -> j)` order of loops.)
  * It helps to look for missing iterators to find tiling and memory reuse possibilities.<br>
    e.g. in `C[i][j] = sum(A[i][k] * B[j][k], axis=k)` example from above<br>
    for `A[i][k]` the missing iterator is `j` and for `B[j][k]` the missing iterator is `i`.


<a id="lec12"></a>

## [Lecture 12](https://www.youtube.com/watch?v=jYCxVirq4d0) - GPU Acceleration

* GPU was designed to perform a lot of identical operations using a lot of cores.<br>
  CPU, however, is a general purpose compute device with smaller number of cores and larger Control units.
* We are using CUDA's terminology. Usually there is a direct mapping between CUDA concepts and
  other GPU programming models: opencl, sycl, metal
* GPU operates under **Single instruction, multiple threads (SIMT)** paradigm.<br>
  All the threads execute same code with different context (thread id, block id)
* GPU computational model consists of 2 levels:
  * Threads are grouped into **blocks**. 
    Threads within same block have **shared memory**.
  * Blocks are grouped into a **launch grid**.<br>
    GPU kerner execution means launching the grid of thread blocks.
* To launch a GPU kernel we still need a **host side of a code** (CPU side). Its main purposes:
  * Call memory allocation on GPU
  * Copy data from CPU to GPU (using PCIe bus, currently)
  * Initializes number of thread blocks and number of threads per block
  * Launch GPU kernel
  * Copy result from GPU to CPU memory
  * Release memory on GPU
* Data transfer from host to device and vice versa takes a lot of time!<br>
  PCIe bus introduces bottleneck by limiting the speed of memory copy between GPU and CPU.<br>
  That's why we need to **keep data on GPU as long as possible**!

### GPU Memory Hierarchy

* Each thread block can be mapped to a **Stream Multiple Processor** (SMP)
  that contains multiple **computing cores**  
* Each thread gets mapped to a single computing core 
  within the Stream Multiple Processor
* GPU memory hierarchy:
  * There is a **global memory**
  * Each thread block has **shared memory**
  * Each thread has its own **registers**
* We use shared memory to **reduce number of data loads** from global slow memory (DRAM).
  Data from global memory is first loaded into a faster shared memory by cooperative fetching, 
  and then each thread within a block reads data from shared memory.<br>
  This also helps to increase memory reuse across different threads within the same thread block.
* **Cooperative fetching** means that multiple threads in a block
  load corresponding portions of shared data simultaneously.

### Matrix multiplication on GPU

* We will consider matrix multiplication 
  in a following transposed variant:<br>
  $C = A^T B,\ C_{i,j} = \sum\limits_k A_{k,i} B_{k,j}$<br>
  All matrices have $N \times N$ size.
* Thread level: **Register Tiling**<br>
  * Load portions of DRAM data into thread registers, 
  perform calculatation and save resultant tile back to DRAM.<br>
  Similar to CPU Register tiling from Lecture 11
  * Does not use thread block shared memory. Uses only thread registers  
* Thread Block level: **Shared Memory Tiling**
  * Each of the thread blocks computes $L \times L$ submatrix of C
  * Each of the threads computes $V \times V$ submatrix
  * Size of a thread block: $L / V \times L / V = L^2 / V^2$ threads
  * $S$ is a tiling factor on a reduction dimension
  * $L \times S$ regions of A and B matrices are loaded into a shared memory by each of a thread block
  * Number of global memory to shared memory copy operations: $2 N^3 / L$<br>
    Number of shared memory to registers copy operations: $2 N^3 / V$
  * The **reason to use a shared memory** is that some threads will use the same data.<br>
    Instead of loading it each time by each individual thread, it's better to pre-fetch it into a shared memory.
  * Shared memory fetching is slow (? probably due to the relatively slow nature of a global memory)
  * GPU is able to do **context switching** and launch computations 
    even if shared memory fetching is not fully completed (?).<br>
    In that case GPU launches computations on idle threads that have already loaded their portions of shared data.<br>
    If sufficient amount of threads is available, this allows for data loading and computations to run concurrently.
* How to choose S, L, V parameters? 
  * **tradeoff**: number of registers vs number of threads available - 
    total amount of registers on each SMT is a constant.<br>
    If we want to use more registers per thread, than the total number of threads will be lower.<br>
    The fewer threads are launched, the less (computational?) power we get.
  * **tradeoff**: amount of shared memory vs number of thread blocks available.<br>
    Larger amount of shared memory leads to a smaller number of thread blocks that can fit on the same SMP.<br>
    And if a thread block is stalled (e.g. with data loading) 
    there might be no other thread blocks to context switch to.
  * People tend to use **autotune** to find best params for each particular task
  * One can also perform problem analysis and come up with analytical solution 
    to choose hyperparams (but it's harder).

#### Other GPU optimization techniques
There are techniques other than simple parallel execution (SIMT) and the use of shared memory
that allow to get a maximum benefit of a GPU accelerator if used in combination:
* Global memory continuous read.<br>
  Ensure that all the threads within a thread block read data from a 
  continuous region
* Shared memory bank conflict<br>
  Make sure that each of a thread writes 
  to a different shared memory banks.
* Software pipelining<br>
  Allows to do data loading and computations (transformations?)
  in a concurrent fashion.
* Warp level optimizations<br>
  Perform certain computations at a warp level - a smaller granularity
  than a thread block.
* Tensor Core<br>
  Another type of acceleration unit.

#### CUDA learning resources:
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Slides. CUDA C/C++ Basics. Supercomputing 2011 Tutorial](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)


<a id="lec13"></a>

## [Lecture 13](https://www.youtube.com/watch?v=XdhUZRXA7fg) - Hardware Acceleration Implemention

* Manipulating with array strides, shape and offset change **the view** of the data
* To perform calculations (e.g. element-wise addition) 
  we need to use **compact()** function.<br>
  It creates a new tensor from a specified view in order to perform operations correctly
  and not change underlying data of the original view.
  * For example, to reshape a `x[0, ::3, 1:]` view of a (2,4,3)-tensor `x` we need to call:<br>
  `x[0, ::3, 1:].compact().reshape((4,))`
* However, some operations in numpy and pytorch can work with arguments that are not contiguous in memory.
  They do not call **compact()** function and work directly with strides.
* In some cases it's usefull to explicitly call **compact()**.
  For example for Matrix Multiplications or Convolutions it's helpful to use compacted (flattened) arrays


<a id="lec14"></a>

## [Lecture 14](https://www.youtube.com/watch?v=XdhUZRXA7fg) - Implementing Convolutions

### Implementation
* See `Lecture_14_code_notes.ipynb` notebook 
  and [Public jupyter notebooks from lectures repo](https://github.com/dlsyscourse/public_notebooks) 
  for implementation details

### Notes
* We are going to store inputs as $N \times H \times W \times C_{in}$ tensor as opposed to pytorch way of $N \times C_{in} \times H \times W$. The reason is that it makes more sense when dealing with matrix multiplications
* And we are going to store filters as $K \times K \times C_{in} \times C_{out}$ tensor as opposed to pytorch way of $C_{out} \times C_{in} \times K \times K$

### Naive implementation. 7 loops :)
* very slow, obviously. ~10 seconds vs 2ms in pytorch convolutions on CPU 
  for `(10, 32, 32, 8), (3, 3, 8, 16)` shaped inputs
* ```python
  def conv_naive(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    assert K % 2 == 1

    out = np.zeros(shape=(N, H - K + 1, W - K + 1, C_out))

    # batch
    for ib in range(N):
        # image
        for i in range(H - K + 1):
            for j in range(W - K + 1):
                # channels
                for icin in range(C_in):
                    for icout in range(C_out):
                        # kernel
                        for ik in range(0, K):
                            for jk in range(0, K):
                                out[ib, i, j, icout] += Z[ib, i + ik, j + jk, icin] * weight[ik, jk, icin, icout]
    
    return out
  ```

### Convolution as number of matrix multiplications
* much better compared to a naive implementation
* still not good: 
  * 2 `for` loops in python
  * too many calls in automatic differentiation tool
  * not efficient for large filter sizes
* we leverage batch type of matrix multiplication:
  ```python
  def conv_matrix_mult(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    assert K % 2 == 1

    out = np.zeros(shape=(N, H - K + 1, W - K + 1, C_out))

    for i in range(K):
        for j in range(K):
            out += Z[:, i:i + H - K + 1, j:j + W - K + 1, :] @ weight[i, j, :, :]

    return out
    ```
* 2-3 times slower than pytorch implementation on CPU

### Convolution as single matrix multiplication. im2col opeartion
* We can easily (if implemented correctly 🙃) index tensors and create complex subtensors
  by manipulating underlying memory using strides and shapes.<br>
  Here are [nice examples](https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20) to illustrate the point.
* im2col constructs a tensor by using $O(K^2)$ more memory than the original image, which can be quite costly for large kernel sizes.

#### Implementation
* The main idea: im2col -> reshape (makes tensor to be contiguous in memory, duplicates required tensor elements) ->
  matrix multiplication -> reshape:
  ```python
  def im2col(arr: np.ndarray, K):
    B, H, W, Cin = arr.shape
    Bs, Hs, Ws, Cs = arr.strides

    out = np.lib.stride_tricks.as_strided(
        arr, 
        shape=(B, H - K + 1, W - K + 1, K, K, Cin),
        strides=(Bs, Hs, Ws, Hs, Ws, Cs)
    )
    # numpy makes array contiguous in memory before reshape if needed - like in this case.
    # here we not only change the shape, but also duplicate needed values of input tensor.
    # thus, underlying data copy is required.
    out = out.reshape(-1, K * K * Cin)
    
    return out

  def conv_im2col(Z, weight):
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    assert K % 2 == 1

    Z_im2col = im2col(Z, K)
    out = Z_im2col @ weight.reshape(-1, C_out)
    out = out.reshape(N, H - K + 1, W - K + 1, C_out)

    return out
  ```

### Questions
* I guess it's easier to add padding when treating convolutions as single matrix multiplication via im2col
* TODO: What happens to gradients computation if we use multi-channel version of im2col? They should remain the same.
  Need to derive them and check.


<a id="lec15"></a>

## [Lecture 15](https://www.youtube.com/watch?v=HSzVogM5IPo) - Training Large Models

* Main sources of memory consumption:
  * Model weights
  * Optimizer states
  * Intermediate activation values (used during backward pass)

### Techniques for memory saving
* For inference we need only $O(1)$ memory to compute the result. We don't need to keep track of intermediate
  activation values. And in fact we can use only 2 memory buffers (for simple case, e.g. without skip connections): 
  buffer A to store `x` and buffer B to store `f(x)`. Then we write `f(x)` to A and `g(f(x))` to B and proceed. 
  For more complex cases, due to locality of intermediate activation nodes dependencies, we still can perform
  inference with a constant memory usage.
* Training N-layer network requires $O(N)$ memory without any optimizations.
  The reason is that intermediate activation values being used 
  during backward pass.
* **Activation checkpointing** (or simply **checkpointing**, or **re-materialization technique**) helps to 
  reduce amount of memory needed to run training.
  * The idea is that we save (checkpoint) only a portion of intermediate activations during a forward pass
  * While doing backward pass we recompute only the needed portion of intermediate activations starting from the 
    last checkpoint available.<br>
    For that we use additional memory buffer.
  * If we checkpoint only each K-th intermediate activation, the overall memory cost has following bound:<br>
    $O(N/K) + O(K)$. checkpoint cost + re-computation cost
  * If we choose $K = \sqrt{N}$, then we get sublinear memory cost: $O(\sqrt{N})$
  * However, recomputations introduce additional computations. We are in fact computing forward pass once again
    (2 times in total) during recomputation (between each pair of successive checkpoints).
  * we can choose only to recompute relatively cheap layers (ReLU, Linear) 
    as opposed to more computationaly heavy layers (matrix multiplication, convolution). 
    In this case memory saving will be less, but the compute will take less time.

### Parallel and distributed training
* TODO
