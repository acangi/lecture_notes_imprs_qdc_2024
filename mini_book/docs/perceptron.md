---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: 'inline'
---

<a href="https://colab.research.google.com/github/acangi/lecture_notes_imprs_qdc_2024/blob/main/mini_book/docs/notebooks/perceptron.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```{hint}
Run the notebook in the colab environment.
```

(h1:perceptron)=

# Perceptron

## Neural Networks

### Biological Neurons
#### Basic Facts about Neurons
A **neuron**, also known as a nerve cell, is the basic functional unit of the nervous system. Neurons are specialized cells that receive, process and transmit information through electrical and chemical signals.

A neuron is made up of three main parts:

* **Cell body (soma)**:
Contains the nucleus and other organelles necessary for basic cell functions.

* **Dendrites**:
Tree-like projections that receive signals from other nerve cells and transmit them to the cell body.

* **Axon**:
A long, thin extension that conducts electrical signals away from the cell body to other nerve cells, muscles or glands. The end of the axon branches into axon terminals that release neurotransmitters to transmit signals.

```{figure} images/Neurons_cerebral_cortex.png
---
width: 400px
name: Neurons in the cerebral cortex
align: center
---
Neurons in the cerebral cortex
```
#### Anatomy and Function of Neurons
The function of a neuron can be divided into several steps:

* **Reception of signals**: Dendrites receive chemical signals from neighboring neurons. These signals lead to changes in the membrane potential of the neuron.

**Generation of an action potential**: When the membrane potential reaches a certain threshold, an action potential is triggered. This is a rapid change in the membrane potential that spreads along the axon.

* **Transmission of the action potential**: The action potential travels along the axon to the axon terminals. This occurs through a series of depolarizations and repolarizations of the cell membrane.

**Signal transmission**: At the axon terminal, the action potential leads to the release of neurotransmitters into the synaptic gap, the space between the axon terminal of the sending neuron and the dendrites of the receiving neuron. The neurotransmitters bind to receptors on the membrane of the receiving neuron and can trigger a new electrical signal there.

This variability of information transmission is reflected in the weights of artificial neuronal networks.

```{figure} images/Neuron_anatomy.png
---
width: 600px
name: Anatomie eines Neurons
align: center
---
Anatomy of a neuron
```

### Artificial Neurons

#### History and Development of artificial neural networks
* **Early ideas and inspirations (1940s)**:

  * McCulloch and Pitts (1943): They developed the first mathematical model of a neuron, laying the foundation for neural network theory. Their model showed that a neuron can function as a simple binary switch.
  
```{figure} images/McCulloch-Pitts-cell.png
---
width: 600px
name: McCulloch-Pitts Neuron
align: center
---
McCulloch-Pitts neuron
```

* **Perzeptron Model (1950s and 1960s)**:
  
  * Frank Rosenblatt (1958): He invented the perceptron, a simple single-level neural network that could solve linear classification tasks. The perceptron was capable of learning by adapting its weights through a simple learning process.

  * Criticism of the perceptron (1969): Marvin Minsky and Seymour Papert published the book "Perceptrons" in which they pointed out the limitations of the perceptron, particularly its inability to solve nonlinear problems such as the XOR problem. This led to a decline in interest in neural networks.

```{figure} images/Perceptron-xor-task.png
---
width: 600px
name: Perzeptron-Modell
align: center
---
Perzeptron model
```
* **Revival and development of the multilayer perceptron (1980s):**

  * Backpropagation algorithm (1986): David Rumelhart, Geoffrey Hinton and Ronald Williams introduced the backpropagation algorithm to efficiently adjust the weights in multilayer neural networks. This solved the problem of non-linear classification and opened up new possibilities for artificial neural networks.

## Perzeptron Model

### Structure
The simple perceptron model consists of an input layer and an output layer.

In the input layer, the input is $\vec{x} = (x_1, x_2,\dots, x_n)$.

The output layer consists of a single neuron. It contains the network input $z$ and the output value $y$.

This network can be used for binary classification, i.e. the network can decide for an input whether it belongs to a certain category. The classification is expressed by the output value $y$.

```{figure} images/Perceptron.png
---
width: 600px
name: Perceptron
align: center
---
Perceptron
```

### Forward Propagation
In the following, the functionality of the simple perceptron model will be demonstrated with the help of vector notation and the path from input $\vec{x}$ to output $y$ will be traced. These steps are also referred to as forward propagation.

The input is represented as a feature vector $\vec{x} = (x_1, x_2,\dots, x_n)$. In other words
\begin{align}
\vec{x} &=
\begin{pmatrix}
x_1\\ \vdots\\ x_n
\end{pmatrix} ,
\end{align}
where the feature vector has length $n$ und a feature is denoted by $x_i \in \mathbb{R}$.


The parameters of the perceptron are the weights $w_i \in \mathbb{R}$. They are represented in terms of a so-called weight vector:
\begin{align}
\vec{w} &=
\begin{pmatrix}
w_1\\ \vdots\\ w_n
\end{pmatrix} .
\end{align}


#### Network Input
In the first step, the network input $z$ is calculated from the feature vector $\vec{x}$. This results from the sum of the values of the input neurons multiplied by their respective weights:
\begin{align}
z &= \sum_{i=1}^n w_i x_i = w_1 x_1 + \dots + w_n x_n\ . 
\end{align}

We can express this in a compact form in vector notation as matrix multiplication:
\begin{align}
z &= \vec{w}^T \vec{x}\\
  &= (w_1, \dots, w_n) \begin{pmatrix} x_1\\ \vdots\\ x_n \end{pmatrix}\\
  &= w_1 x_1 + \dots + w_n x_n\,,  
\end{align}
where $\vec{w}^T$ denotes a transposed weight vector (which is a row vector).

#### Activation and Output
In the second and final step, we calculate the activation of the output neuron, which also corresponds to the output of the perceptron model.

An activation function $g$ is applied to the network input:
\begin{align}
y &= g(z)\ .
\end{align}

### Learning Process
Now that we have understood the propagation of data through the perceptron, let us turn our attention to the learning process. In the context of the perceptron model, we understand learning as the gradual adaptation of the weights $\vec{w}$ to the desired target function with the help of **training data**.

#### Training Data
Training data is, for example, a series of data with a label (e.g. images with the categorical assignment "cat" and "non-cat").  The training data can therefore be written as pairs of feature vectors and labels:
\begin{align}
(\vec{x}^k, y^k) ~~~ k \in \{1, \dots, N\}\,,
\end{align}
where we assume that the training data has a dimension $N$.

With a neural network, we must distinguish between the calculated output of the current network $\bar{y}^k \in \{0,1\}$ and the correct output of a training example $y^k \in \{0,1\}$.


#### Learning Step
Learning means that we calculate the output $\bar{y}$ for each training example and then adjust the weights. This is called a **learning step**.

We can therefore adjust the weight vector $\vec{w}$ for the pairs of feature vectors and labels $(\vec{x}^k, y^k)$ in each learning step by adding a change of all weights to the current value:
\begin{align}
w_i := w_i + \Delta w_i\ .
\end{align}

We will define a concrete rule for adjusting the weights in the next section. Before that, however, let's look at the properties of the weight update:

* If the calculated output $\bar{y}$ is greater than the reference value $y$, the weight update should be negative, i.e. the weight of this neuron should be weakened.
* If the calculated output $\bar{y}$ is smaller than the reference value $y$, the weight update should be positive, i.e. the weight is increased.
* If the calculated output $\bar{y}$ and the reference value $y$ are the same, no weighting update should take place.

### Gradient Descent
In the following, we will derive a learning rule that can be used to update the weights of a perceptron.

#### Loss Function
In order to derive a suitable learning rule, we must first define the term loss function $L$. It is defined as
\begin{align}
L(w) &= \frac{1}{2N} \sum_{k=1}^N \left[ y^k - \bar{y}^k \right]^2\ . 
\end{align}

The loss function is therefore defined as the mean value of the squared errors, which is very common.

How can you visualize the loss function? It is a multidimensional function of the weights or the weight vector $\vec{w}$. An intuitive idea can be obtained by looking at the figure in which the loss function is represented as a function of two representative weights ($w_0$ and $w_1$).

```{figure} images/landscape_0.png
---
width: 600px
name: landscape_0
align: center
---
Parameter landscape of the loss function
```

We can see that the loss function here spans a two-dimensional (and generally a multi-dimensional) parameter landscape.

#### Learning rule: Minimum of the loss landscape
We can now use the loss function to derive a learning rule by setting ourselves the goal of minimizing the value of the loss function. This means that we look for valleys in the parameter landscape.

```{figure} images/landscape_2.png
---
width: 600px
name: landscape_2
align: center
---
Gradient descent in the parameter landscape of the loss function
```
### Weight Update
With this goal in mind, we can derive the weight update $w := w + \Delta w$. To do this, we use the gradient of the loss landscape, which informs us about the largest increase in the error landscape. We therefore take the negative gradient and thus obtain our learning rule:

\begin{align}
\Delta \vec{w} &= -\alpha\, \nabla L(w)\\
         &= -\alpha\, \begin{pmatrix} \frac{\partial L(w)}{\partial w_1}\\
                            \vdots \\
                            \frac{\partial L(w)}{\partial w_n}
            \end{pmatrix}\,,
\end{align}
where we have also introduced the learning rate $\alpha \in [0,1]$.
As can be seen in the figure, updating the weights in the direction of the negative gradient causes us to run in the direction of the minimum of the loss landscape.

For the learning rule, we now need the partial derivatives of the loss function according to the weights $w_1, \dots, w_n$, which we now calculate.

To do this, we start with the definition of the loss function and reshape it until we can derive it according to the individual weights:

\begin{align}
L(w) &= \frac{1}{2N} \sum_{k=1}^N \left[ y^k - \bar{y}^k \right]^2\ .
\end{align}

To simplify the derivation, we choose the identical mapping $g(z) = z$ as the activation function.

We insert it into the definition of the loss function and obtain
\begin{align}
L(w) &= \frac{1}{2N} \sum_{k=1}^N \left[ y^k - g(z^k) \right]^2\ .
\end{align}

We now evaluate the activation function, which simplifies the expression to:
\begin{align}
L(w) &= \frac{1}{2N} \sum_{k=1}^N \left( y^k - z^k \right)^2\ .
\end{align}

Now we insert the expression for the network input:
\begin{align}
L(w) &= \frac{1}{2N} \sum_{k=1}^N \left( y^k - \vec{w}^T \vec{x}^k \right)^2\ .
\end{align}

Now we can explicitly calculate the derivative with respect to the weight $w_i$:
\begin{align}
\frac{\partial L}{\partial w_i}
&= \frac{\partial}{\partial w_i} \frac{1}{2N} \sum_{k=1}^N \left( y^k - \vec{w}^T \vec{x}^k \right)^2 \\
&= \frac{1}{2N} \sum_{k=1}^N \frac{\partial}{\partial w_i} \left( y^k - \vec{w}^T \vec{x}^k \right)^2 \\
&= \frac{1}{2N} \sum_{k=1}^N  2 \left( y^k - \vec{w}^T \vec{x}^k \right) \frac{\partial}{\partial w_i} \left(y^k - \vec{w}^T \vec{x}^k \right) \\
&= \frac{1}{N} \sum_{k=1}^N \left( y^k - \vec{w}^T \vec{x}^k \right) \frac{\partial}{\partial w_i} \left[-(w_1, \dots, w_i, \dots, w_n) \begin{pmatrix} x^k_1\\ \vdots\\  x^k_i\\ \vdots\\ x^k_n \end{pmatrix} \right] \\
&= -\frac{1}{N} \sum_{k=1}^N \left( y^k - \vec{w}^T \vec{x}^k \right) \frac{\partial}{\partial w_i} \left[ (w_1 x^k_1 + \dots + w_i x^k_i + \dots + w_n x^k_n) \right] \\
&= -\frac{1}{N} \sum_{k=1}^N \left( y^k - \vec{w}^T \vec{x}^k \right) x^k_i\ .
\end{align}

At the end we use the definition of the activation function again and obtain:
\begin{align}
\frac{\partial L}{\partial w_i}
&=-\frac{1}{N} \sum_{k=1}^N \left[ y^k - g(z^k) \right] x^k_i\ .
\end{align}

This gives us a learning rule for a specific weight:
\begin{align}
\Delta w_i
&=-\alpha \frac{\partial L(w)}{\partial w_i}\\
&=\frac{\alpha}{N} \sum_{k=1}^N \left[ y^k - g(z^k) \right] x^k_i\ .
\end{align}

Since the output of the perceptron is $\bar{y}^k=g(z^k)$, we thus obtain
\begin{align}
\Delta w_i
&=\frac{\alpha}{N} \sum_{k=1}^N \left( y^k - \bar{y}^k \right) x^k_i\ .
\end{align}

Thus, we have derived our learning rule using gradient descent. As we can see, we first need to process the sum over all training data ($N$ times) to calculate our weight update for a learning step.

# Application example: Binary Classification with the Percepetron Model

### Learning Algorithm
With the help of the derived learning rule, we can now formulate a learning algorithm.

1. initialize all weights $\vec{w} = (w_0, ..., w_n)$.
2. for each epoch:
     * Set $\Delta w_i = 0$
     * For each set of training data ($x^k, y^k$), $k=1, \dots, N$:
       * Calculate output $y^k$.
       * Calculate weight update: $\Delta w_i^k = \Delta w_i^k + (y^k-\bar{y}^k)x_i^k$.
     * Calculate the mean of all weight updates over the training data: $\Delta w_i = \frac{\alpha}{N}\sum_{k=1}^N \Delta w_i^k$.
     * Update all weights $w_i = w_i + \Delta w_i$


### Implementation of the Perceptron
#### Perceptron Class
First we define a class `Perceptron`:
```{code-cell} ipython3
class Perceptron():
    
    # Constructor with default values
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    # Training with x (features) and y (labels)
    def train(self, x, y, epochs):
        rnd = np.random.RandomState(42)
        n_samples = x.shape[0]
        print(f"Train on {n_samples} samples")
        
        # Weights: 1 + dim(x) for the bias neuron
        self.w = rnd.normal(loc=0, scale=0.01, size=1 + x.shape[1]) 
    
        # List for storing loss function in each epoch
        self.loss = [] 
        
        # List for storing prediction in each epoch
        self.prediction = [] 

        # List for storing binary accuracy in each epoch
        self.accuracy = [] 

        # Loop over epochs
        for i in range(epochs):
            z = self.net_input(x) # Network input for all training data
            y_hat = self.activation(z) # Activation for all training data
            diff = y - y_hat # Loss vector for all training data
            
            # Weight update
            self.w[1:] += self.alpha * x.T.dot(diff)
            
            # Weight update for bias neuron
            self.w[0] += self.alpha * diff.sum()
            
            # Save SSE of loss for each epoch
            l = (diff**2).sum() / 2.0 
            self.loss.append(l)
            print(f"Epoch {i+1}/{epochs} - loss: {l:.4f}")

            # Save prediction for each epoch
            prediction = self.predict(x)
            self.prediction.append(prediction)

            # Save binary accuracy for each epoch
            accuracy = self.measure_accuracy(y,self.predict(x))
            self.accuracy.append(accuracy)
        
        return self
    
    # Activation function
    def activation(self, z):
        return z
    
    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0] 
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, 0)

    def measure_accuracy(self, y_true, y_pred):
        N = y_true.shape[0]
        accuracy = (y_true == y_pred).sum() / N
        return accuracy
```

#### Learning Process
The learning or training of the perceptron takes place in the function 'train', in particular in the loop over the epochs (lines 20 - 35).

```
for i in range(epochs):
            z = self.net_input(x) # Network input for all training data
            y_hat = self.activation(z) # Activation for all training data
            diff = y - y_hat # Loss vector for all training data
            
            # Weight update
            self.w[1:] += self.alpha * x.T.dot(diff)
            
            # Weight update for bias neuron
            self.w[0] += self.alpha * diff.sum()
```


#### Batch processing of the training data
We have implemented the perceptron in such a way that the matrix $x$ contains all feature vectors. This ensures that all training data is run through before the weights are updated. 
The feature vectors are stacked on top of each other to form an $N \times 3$ input matrix. It should be noted that the first column of the matrix represents the bias neuron, i.e. contains only ones. By matrix multiplying the $N \times 3$ input matrix with the $3 \times 1$ weighting vector, we obtain an $N \times 1$ output vector that contains all outputs for all training examples.

The net input is calculated in the `net_input` function:
```
def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0] 
```

To give us a better understanding, we illustrate the operations.

```{figure} images/Training_01.png
---
width: 600px
name: training_01
align: center
---
```

The input vectors are actually row vectors and stacked on top of each other.

However, this picture is not entirely correct. In fact, the bias neuron is treated differently in the implementation, as the following figure shows:

```{figure} images/Training_02.png
---
width: 600px
name: training_02
align: center
---
```

This corresponds to the following line in the implementation:
```
np.dot(x, self.w[1:]) + self.w[0]
```

#### Weight Update 
The weight update
\begin{align}
\vec{w} &= \vec{w} + \Delta\vec{w}\\
\Delta w_i &= -\alpha\frac{\partial L}{\partial w_i} = \frac{\alpha}{N}\sum_{k=1}^N \left[y^k - g(z^k) \right] x_i^k
\end{align}
is implemented in the following way.

```{figure} images/Training_03.png
---
width: 600px
name: training_03
align: center
---
```

### Data Set
The sample data sets of the `scikit-learn` library can be used to illustrate the perceptron model.

```{code-cell} ipython3 
:tags: [hide-cell]

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

from matplotlib import cm
from matplotlib import rcParams
from matplotlib.ticker import LinearLocator
from matplotlib import cbook
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import axes3d
```

```{code-cell} ipython3
from sklearn import datasets
data_wine = datasets.load_wine()
```

```{code-cell} ipython3
data_wine.keys()
```

```{code-cell} ipython3
np.shape(data_wine.data)
```

The output data is saved in the `target` category. The input data is located in the `data` category. The following features are available as input:
```{code-cell} ipython3
data_wine.feature_names
```

In our example, we want to use two features to solve a binary classification problem using the perceptron model. Therefore, we analyze the dataset by plotting the three categories as a function of different features. It turns out that the two features "alcohol" and "flavonoids" lead to a linearly separable data set between the categories "0" (dark blue) and "2" (yellow). 

```{code-cell} ipython3
plt.scatter(data_wine.data.T[0], data_wine.data.T[6], c=data_wine.target)
plt.xlabel(data_wine.feature_names[0])
plt.ylabel(data_wine.feature_names[6])
plt.show()
```

We now prepare our data set for training the perceptron model. We start with the output data. 

```{code-cell} ipython3
y = np.concatenate((data_wine.target[:59], data_wine.target[130:]))
# Change values to 0 and 1.
y = np.where(y == 2, 1, 0)
```

Similarly, we prepare the input data.

```{code-cell} ipython3
input_features = data_wine.data.T
input_features = input_features[[0,6]]
x = np.concatenate((input_features.T[:59], input_features.T[130:]))
```

At the end, we check once again whether the data has been prepared correctly.

```{code-cell} ipython3
plt.scatter(x.T[0][:59], x.T[1][:59], c='blue', label='Kategorie 0')
plt.scatter(x.T[0][59:], x.T[1][59:], c='red', label='Kategorie 1')
plt.xlabel(data_wine.feature_names[0])
plt.ylabel(data_wine.feature_names[6])
plt.legend(loc='upper left')
plt.show()
```

### Training Examples

We train a model with a learning rate of $\alpha=0.01$ over a period of 20 epochs.
```{code-cell} ipython3
model1 = Perceptron(alpha=0.01)
model1.train(x, y, epochs=20)
```


We then train another model with a learning rate of $\alpha=0.0001$.
```{code-cell} ipython3
model2 = Perceptron(alpha=0.0001)
model2.train(x, y, epochs=20)
```

We illustrate the loss function (top) and the accuracy (bottom) for training with both learning rates.
```{code-cell} ipython3
:tags: [hide-output]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,12))

ax[0][0].plot(range(1, len(model1.loss)+1), np.log10(model1.loss), marker='o')
ax[0][0].set_xlabel('Epochen')
ax[0][0].set_ylabel('log(SSE)')
ax[0][0].set_title('Perzeptron, alpha=0.01')

ax[0][1].plot(range(1, len(model2.loss)+1), np.log10(model2.loss), marker='o')
ax[0][1].set_xlabel('Epochen')
ax[0][1].set_ylabel('log(SSE)')
ax[0][1].set_title('Perzeptron, alpha=0.0001')

ax[1][0].plot(range(1, len(model1.loss)+1), model1.accuracy, marker='o')
ax[1][0].set_ylim(0, 1)
ax[1][0].set_xlabel('Epochen')
ax[1][0].set_ylabel('Accuracy')
ax[1][0].set_title('Perzeptron, alpha=0.01')

ax[1][1].plot(range(1, len(model2.loss)+1), model2.accuracy, marker='o')
ax[1][1].set_ylim(0, 1)
ax[1][1].set_xlabel('Epochen')
ax[1][1].set_ylabel('Accuracy')
ax[1][1].set_title('Perzeptron, alpha=0.0001')

plt.show()
```


## Training strategies

### Standardization
Standardize the two inputs:

```{code-cell} ipython3
x_st = np.copy(x)
x_st[:, 0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x_st[:, 1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()
```

```{code-cell} ipython3
model3 = Perceptron(alpha=0.01)
model3.train(x_st, y, epochs=20)
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,12))

ax[0][0].plot(range(1, len(model1.loss)+1), np.log10(model1.loss), marker='o')
ax[0][0].set_xlabel('Epochen')
ax[0][0].set_ylabel('log(SSE)')
ax[0][0].set_title('Perzeptron, alpha=0.01')

ax[0][1].plot(range(1, len(model2.loss)+1), np.log10(model2.loss), marker='o')
ax[0][1].set_xlabel('Epochen')
ax[0][1].set_ylabel('log(SSE)')
ax[0][1].set_title('Perzeptron, alpha=0.001')

ax[0][2].plot(range(1, len(model3.loss)+1), np.log10(model3.loss), marker='o')
ax[0][2].set_xlabel('Epochen')
ax[0][2].set_ylabel('log(SSE)')
ax[0][2].set_title('Perzeptron, std, alpha=0.01')

ax[1][0].plot(range(1, len(model1.loss)+1), model1.accuracy, marker='o')
ax[1][0].set_ylim(0, 1)
ax[1][0].set_xlabel('Epochen')
ax[1][0].set_ylabel('Accuracy')
ax[1][0].set_title('Perzeptron, alpha=0.01')

ax[1][1].plot(range(1, len(model2.loss)+1), model2.accuracy, marker='o')
ax[1][1].set_ylim(0, 1)
ax[1][1].set_xlabel('Epochen')
ax[1][1].set_ylabel('Accuracy')
ax[1][1].set_title('Perzeptron, alpha=0.001')

ax[1][2].plot(range(1, len(model3.loss)+1), model3.accuracy, marker='o')
ax[1][2].set_ylim(0, 1)
ax[1][2].set_xlabel('Epochen')
ax[1][2].set_ylabel('Accuracy')
ax[1][2].set_title('Perzeptron, std, alpha=0.01')

plt.show()
```


