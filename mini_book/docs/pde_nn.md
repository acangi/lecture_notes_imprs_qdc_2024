---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: lectures_imprs
mystnb:
  execution_mode: 'off'
---

<a href="https://colab.research.google.com/github/acangi/lecture_notes_imprs_qdc_2024/blob/main/mini_book/docs/notebooks/02_TD_Schrodinger.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```{hint}
Run the notebook in the colab environment.
```

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.image as img
import copy

import ipywidgets as widgets
from ipywidgets import interact

import urllib.request as urllib2
```


# Solving differential equations with neural networks
In this lecture we will learn how to solve differential equations using machine learning. More specifically, we will look into two ways how to leverage neural networks to find solutions of differential equations. 
- The first method is a data-driven approach using feedforward neural networks.
- The second method is a physics-driven approach using physics-informed neural networks. 

## Time Dependent Quantum Harmonic Oscillator
As a simple example consider the time-dependent Schrödinger equation for the quantum harmonic oscillator. 
```{hint}
We will use atomic units unless specified: $\hbar=m=e=1$. 
```

We begin with the time-dependent Schrödinger equation 
\begin{equation}
i \frac{\partial \phi(\mathbf{r}, t)}{\partial t}-\hat{H} \phi(\mathbf{r}, t)=0\ .
\end{equation} 
The Hamiltonian for a one-dimensional harmonic oscillator is
\begin{equation}
\hat{H_x}=-\frac{1}{2}\frac{\partial^{2}}{\partial x^{2}}+\frac{\omega^{2}}{2}x^{2}
\end{equation}
The analytical solutions $\phi(x,t) \in \mathbb{C}$ are
\begin{equation}
\phi_n(x) = \phi_0(x) \frac{1}{\sqrt{2^n n!}}Her_n(\sqrt{\omega}x)\exp\left(-i E_n t\right)\,,
\end{equation}
where
\begin{equation}
\phi_0(x) = \sqrt[4]{\frac{\omega}{\pi}}e^{\left(-\frac{\omega x^2}{2}\right)}
\end{equation}
is the ground state wavefunction and
\begin{equation}
\phi_1(x) = \phi_0(x) \sqrt{2\omega}x
\end{equation}
the first excited state.
Here, $Her_n(\sqrt{\omega}x)$ denotes the $n$th Hermite polynomial and $E_n = (n+\frac{1}{2})\omega$ the eigenvalues of the harmonic oscillator with frequency $\omega$.

In the following, we will consider the superposition of two eigenstates which is defined as
\begin{equation}
\psi_{m,n}(x,t) = \frac{1}{\sqrt{2}}\left( e^{\left(-i E_m t\right)}\phi_m(x) + e^{\left(-i E_n t\right)}\phi_n(x)\right)
\end{equation}


### Base case: $\psi_{0,1}(x,t)$
Let us have a closer look at the analytical solution for $\psi_{0,1}(x,t)$.

\begin{equation}
\psi_{0,1}(x,t) = \frac{1}{\sqrt{2}}\left( e^{\left(-i E_0 t\right)}\phi_m(x) + e^{\left(-i E_1 t\right)}\phi_n(x)\right)
\end{equation}

The $n$th Hermite polynomial is defined as:
\begin{equation}
H_{n}(y)=(-1)^{n} e^{y^{2}} \frac{d^{n}}{d y^{n}} e^{-y^{2}}
\end{equation}

```{dropdown} Exercise 
a) Derive the first Hermite Polynomial ($H_{1}(y)$) in terms of $x$ for $y=\sqrt{\omega}x$:

Derivation here
$H_{1}(\sqrt{\omega}x) = $

b) Write down the ground state and first excited state energies for QHO:

$E_0 =$,
$E_1=$

c) Use your results from $a$ and $b$ to write down $\psi_{0,1}(x,t)$, and simplify it:

$\psi_{0,1}(x,t) = $
```
```{dropdown} Solution 
a) Derive the first Hermite Polynomial ($H_{1}(y)$) in terms of $x$ for $y=\sqrt{\omega}x$:

Derivation here
$H_{1}(\sqrt{\omega}x) = $

b) Write down the ground state and first excited state energies for QHO:

$E_0 =$,
$E_1=$

c) Use your results from $a$ and $b$ to write down $\psi_{0,1}(x,t)$, and simplify it:

$\psi_{0,1}(x,t) = $
```

## Implementing the one-dimensional quantum harmonic oscillator
Now that we have our analytical solution, let's write some code to compute it for arbitrary values of $\{x,t,\omega\}$

```{code-cell} ipython3
def get_analytical_solution_base(X,T,omega):
    # Analytical solution for first two states. X,T are numpy arrays, omega is a float.
    # Hint: Avoid using for loops, use numpy functions to calculate the wavefunction.
    
    #Solution:
    phi_0 = (omega / np.pi) ** (1. / 4.) * np.exp(-X * omega * X / 2.0)
    phi_1 = phi_0 *  np.sqrt(omega / 2.) * 2.0 * X
    psi = np.sqrt(1. / 2.) * (np.exp(-1j * omega/2 * T) * phi_0 + np.exp(-1j* 3/2 * omega * T) * phi_1)

    return psi
```

Let's define the domain for our problem:
We will restrict our domain to $\mathbf{x} \in [-\pi,\pi], t \in [0,T]$, with fixed boundary conditions $x_0, x_b = 0$. 

Let's create some data using the function we just wrote.

```{code-cell} ipython3
:tags: [hide-output]

def get_data_set(x_dom, t_dom, delta_X, delta_T, omega, analytical_solution_fun):
    # Helper function to generate datasets from analytical solutions.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = np.arange(x_dom[0], x_dom[1], delta_X).astype(float)
    t = np.arange(t_dom[0], t_dom[1], delta_T).astype(float)
    X, T = np.meshgrid(x, t)

    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)

    psi_val = analytical_solution_fun(X,T,omega)
    u = np.real(psi_val)
    v = np.imag(psi_val)

    train_input = np.hstack((X,T))
    train_output = np.hstack((u,v))

    train_x = torch.tensor(train_input).to(device)
    train_y = torch.tensor(train_output).to(device)

    return train_x, train_y
```

```{code-cell} ipython3
:tags: [hide-output]

# Define domain in code

L = float(np.pi)
omega = 1
delta_T = 0.01
delta_X = 0.01
x_dom = [-L,L]
t_dom = [0, 2*L]
analytical_solution_function = get_analytical_solution_base

test_x, test_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, analytical_solution_function)
```

Here `delta_T` and `delta_X` are the grid spacing in our domain. Data will be generated on this grid. You cna move the slider to see how the probabilty density of the QHO evolves in time. Our system looks like this:

```{code-cell} ipython3
@interact(t=widgets.IntSlider(min=0, max=62, step=1, value=0))
def interactive_viz_colour(t):
    tstr = str(t).zfill(2)
    f = urllib2.urlopen("https://github.com/GDS-Education-Community-of-Practice/DSECOP/tree/main/Learning_the_Schrodinger_Equation/")
    fname = f"res/plots/waveform/psi_0_1/t_{tstr}.png"
    image = img.imread(fname)
    plt.figure(figsize=(8,6))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
```

## Fully Connected Neural Network
Now that we have the analytical solution, let's build a neural network to solve it. 

### Implementation
We will be using Mean Squared Error to quantify the difference between the true value and predicted values.
It is defined as 
\begin{equation}
\mathrm{MSE}=\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}
\end{equation}
where $Y_i$ and $\hat{Y_i}$ are the $i$th predicted and true values respectively. Implement this in code below:

```{dropdown} Exercise
Implement mse function.
```

```{code-cell} ipython3
:tags: [hide-cell]
def get_mse(y_true, y_pred):
  mse = np.mean((y_true - y_pred)**2)
  return mse
```

A neural network consists of chained linear regression nodes (perceptrons) and activation functions. The first layer is called the input layer, the layers in the middele are called output layers and the final layer is called the output layer. A neural network surves as a function approximator between the input and the ouput.

![Domains](https://github.com/GDS-Education-Community-of-Practice/DSECOP/blob/main/Learning_the_Schrodinger_Equation/res/fig/nn_architecture.png?raw=1)

Since neural networks are constrained to $\mathbb{R}$, the complex valued solution can be represented as 
\begin{equation}
\psi(x,t) = u + iv    
\end{equation} where $u = \operatorname{Re}(\psi)$ and $v=\operatorname{Im}(\psi)$

```{dropdown} Exercise
What are input and output variables for this neural network?
```

```{dropdown} Solution
Inputs: $x,t$
Outputs: $u,v$
```

```{dropdown} Exercise
Write the one-dimensional time-dependent Schrödinger equation in terms of $u$ and $v$.
```

```{dropdown} Solution
The time-dependent Schrödinger equation can be written as
\begin{equation}
    \left(-\frac{\partial v}{\partial t}+\frac{1}{2}\frac{\partial^{2} u}{\partial x^{2}}-\frac{\omega^{2}}{2}x^{2}\right)\psi + i \left(\frac{\partial u}{\partial t}+\frac{1}{2}\frac{\partial^{2} v}{\partial x^{2}}-\frac{\omega^{2}}{2}x^{2}\right)\psi = 0
\end{equation}
```

In the first case, the training data is generated on a high resolution grid from the analytical solution described above. 
The neural network $\psi_{net}: \mathbb{R}^{1+1}\mapsto \mathbb{R}^{2}$ is constructed, with inputs $(x,t)$ and outputs $(u,v)$.

Let's look at the pipeline for creating, training and testing Neural Networks. 

```{hint}
In this case, we are using the entire dataset in training for demonstrating the workflow. This is not done in practice and leads to overfitting. 
```

Generate training and test data:
```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1

# Grid spacing
delta_T = 0.1
delta_X = 0.1
# Domains
x_dom = [-L,L]
t_dom = [0, 2*L]

train_x, train_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_base)
test_x, test_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_base)
```

Create the architecture of Neural Network. This is a fully connected neural network (FCN):
```{code-cell} ipython3
class NN(nn.Module):
    
    def __init__(self, n_in, n_out, n_h, n_l, activation):
        super().__init__()

        
        self.f_in = nn.Linear(n_in, n_h)
        
        layers = []
        for l in range(n_l - 1):
            layers.append(nn.Linear(n_h, n_h))
            layers.append(activation)

        self.f_h = nn.Sequential(*layers)

        self.f_out = nn.Linear(n_h, n_out)
        
    def forward(self, x):
        x = self.f_in(x)
        x = activation(x)

        x = self.f_h(x)
        x = self.f_out(x)
        return x
```

Create a function to train the network:
```{code-cell} ipython3
:tags: [hide-output]

def train_nn(model, n_epochs, train_x, train_y):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    loss_list = np.zeros(n_epochs)
    print("Epoch \t Loss")
    for i in range(n_epochs):
        optimizer.zero_grad()
        pred_y = model(train_x)

        loss = torch.mean((train_y-pred_y)**2)
        
        loss_list[i] = loss.detach().cpu().numpy()

        loss.backward()
        optimizer.step()

        if i % 500 == 499:
            print(f"{i} {loss}")
            
    return model, loss_list
```

Create a function to quantify the loss and errors
```{code-cell} ipython3
:tags: [hide-output]

def get_model_error(model, test_x, test_y):
    model.eval()
    with torch.no_grad():

        pred_y = model(test_x)
        
        pred_u, pred_v, pred_dens = get_density(pred_y)
        test_u, test_v, test_dens = get_density(test_y)
        
        loss_u = torch.mean((pred_u - test_u)**2)
        loss_v = torch.mean((pred_v - test_v)**2)
        loss_dens = torch.mean((pred_dens - test_dens)**2)

        print(f"Model loss: \n loss_u = {loss_u} \n loss_v = {loss_v} \n loss_dens = {loss_dens}")
        
    return pred_y
```

Function for inference and plotting:
```{code-cell} ipython3
:tags: [hide-output]

def inference(model, test_x, test_y,n_x, n_t, x_dom, t_dom, omega,plot_name="plot"):
    
    model.eval()
    with torch.no_grad():

        pred_y = model(test_x)
        
        pred_u, pred_v, pred_dens = get_density(pred_y)
        test_u, test_v, test_dens = get_density(test_y)

        loss_u = torch.mean((pred_u - test_u)**2)
        loss_v = torch.mean((pred_v - test_v)**2)
        loss_dens = torch.mean((pred_dens - test_dens)**2)
         
        # print(loss_u)
        # print(loss_v)
            
        get_plots_norm_colorbar(test_y,pred_y, n_x, n_t, x_dom, t_dom, plot_name)
        # return pred_y
```

### Example: Testing interpolation
Now that we have all the pieces in place, we can use this code to train a neural network on our domain.
```{code-cell} ipython3
:tags: [hide-output]

torch.manual_seed(314)
activation =  nn.Tanh()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_nn = NN(2,2,20,3,activation).to(device)
n_epochs = 10000
```

```{code-cell} ipython3
:tags: [hide-output]

%%time
model_nn, loss_list = train_nn(model_nn, n_epochs, train_x, train_y)
```

The losses for different terms are:
```{code-cell} ipython3
:tags: [hide-output]

pred_y = get_model_error(model_nn, test_x, test_y)
```

And we can plot our reults with the inference function:
```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1
delta_T = 0.1
delta_X = 0.1
    
n_x = np.shape(np.arange(x_dom[0], x_dom[1], delta_X))[0]
n_t = np.shape(np.arange(t_dom[0], t_dom[1], delta_T))[0]

print(delta_T)

inference(model_nn, test_x, test_y, n_x, n_t, x_dom, t_dom, omega, 'nn')
```

Reading these snapshot plots: 
The first column consists of plots of neural network predictions The second column consists of true values and the last column is the Absolute Error between ground truth and predictions.

The rows are real($\psi$), imaginary($\psi$) and density $|\psi|^2$ .

In this case, since we used all the data over the domain for training, we get great results, where the NN approximates $\psi$ very well.

Now for a more realistic case. We might not have a lot of experimental data covering the entire domain of a system. To approximate that, let's train a neural network on a low resolution grid in a reduced domain $x \in [-\pi/4,\pi/4]$.

### Example: Testing transferability
```{code-cell} ipython3
:tags: [hide-output]

# Generate reduced training set
L = float(np.pi)
omega = 1

delta_T = 0.1
delta_X = 0.1
x_dom = [-L/4,L/4]
t_dom = [0, 2*L]

train_x, train_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_base)
```

We will test it's performance on a high resolution grid across the entire domain:
```{code-cell} ipython3
:tags: [hide-output]

delta_T = 0.01
delta_X = 0.01
x_dom = [-L,L]
t_dom = [0, 2*L]

test_x, test_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_base)
```

```{code-cell} ipython3
:tags: [hide-output]

torch.manual_seed(314)
activation =  nn.Tanh()

model_nn = NN(2,2,20,3,activation).to(device)
n_epochs = 10000
```

```{code-cell} ipython3
:tags: [hide-output]

%%time
model_nn, loss_list = train_nn(model_nn, n_epochs, train_x, train_y,)
```

So far so good, the training loss is small. But across the entire domain, the loss is:
```{code-cell} ipython3
:tags: [hide-output]

y_pred = get_model_error(model_nn, test_x, test_y)
```

The error is orders of magnitude higher than our data point values! Let's look at the plots:
```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1
delta_T = 0.01
delta_X = 0.01
    
n_x = np.shape(np.arange(x_dom[0], x_dom[1], delta_X))[0]
n_t = np.shape(np.arange(t_dom[0], t_dom[1], delta_T))[0]

inference(model_nn, test_x, test_y, n_x, n_t, x_dom, t_dom, omega, 'nn_reduced')
```
The values are similar in the domain that we trained on $x \in [-0.78,0.78]$ but diverge considerably near the boundaries.

To tackle this problem, we will add some information about the governing equation in the neural network.


## Physics-Informed Neural Networks
Physics Informed Neural Networks are constructed by encoding the constraints posed by a given differential equation and its boundary conditions into the loss function of a Fully Connected Network. This constraint guides the network to approximate the solution of the differential equation.

For a system $ f $, with solution $ u(\mathbf{x},t) $, governed by the following equation

\begin{align}\label{eq:pde}
f(u) &:=u_{t}+\mathcal{N}[u;\lambda], \mathbf{x} \in \Omega, t \in [0,T] \\
f(u) &= 0
\end{align} 
where $\mathcal{N}[u;\lambda]$ is a differential operator parameterised by $ \lambda $, $ \Omega \in \mathbb{R^D} $, $ \mathbf{x} = (x_1,x_2,...,x_d) $ with boundary conditions 
 \begin{equation}
\mathcal{B}(u, \mathbf{x},t)=0 \quad \text { on } \quad \partial \Omega
\end{equation}
 and initial conditions
 \begin{equation}
\mathcal{T}(u, \mathbf{x},t)=0 \quad \text { at } \quad t = 0
\end{equation}


![PINN architecture](https://github.com/GDS-Education-Community-of-Practice/DSECOP/blob/main/Learning_the_Schrodinger_Equation/res/fig/PINN_diagrams.png?raw=1)

A neural network $u_{net}: \mathbb{R}^{D+1}\mapsto \mathbb{R}^{1}$ is constructed as a surrogate model for the true solution $u$, 
\begin{equation}
f_{net}=f(u_{net})
\end{equation}
The constraints imposed by the system are encoded in the loss term $L$ for neural network optimisation
\begin{equation}
L={\color{green} L_{f}}+{\color{red} L_{BC}}+{\color{blue} L_{IC}}
\label{eq:pinn_loss}
\end{equation}
where $L_{f}$ denotes the error in the solution within the interior points of the system, enforcing the PDE. This error is calculated for $N_f$ collocation points.
\begin{equation}
\color{green} 
L_{f}=\frac{1}{N_{f}} \sum_{i=1}^{N_{f}}\left|f_{net}\left(\mathbf{x}_{f}^{i}, t_{f}^{i}\right)\right|^{2}
\end{equation}
\begin{equation}
\color{red} 
L_{BC}=\frac{1}{N_{BC}} \sum_{i=1}^{N_{BC}}\left|u\left(\mathbf{x}_{BC}^{i}, t_{BC}^{i}\right)-u^{i}\right|^{2}
\end{equation}
\begin{equation}
\color{blue} 
L_{IC}=\frac{1}{N_{IC}} \sum_{i=1}^{N_{IC}}\left|u\left(\mathbf{x}_{IC}^{i}, t_{IC}^{i}\right)-u^{i}\right|^{2}
\end{equation}
$L_{BC}$ and $L_{IC}$ represent the constraints imposed by the boundary and initial conditions, calculated on a set of $N_{BC}$ boundary points and $N_{IC}$ initial points respectively, with $u_i$ being the ground truth.

<img src="https://github.com/GDS-Education-Community-of-Practice/DSECOP/blob/main/Learning_the_Schrodinger_Equation/res/fig/colloc_points.png?raw=1" alt="Domain" width="400"/>

Once sufficiently trained, the network can be used as a solver for the PDE, potentially for a range of parameters $ \lambda $.


Since PINNs can be used to solve systems of arbitrary resolutions once they are trained and generalise well over different parameter spaces, they might be used to accelerate the solution of PDEs. 

### Implementation
The following function creates the collocation points for the equation, boundary and initial conditions.

```{code-cell} ipython3
:tags: [hide-output]

def get_physics_colloc_points(x_dom, t_dom, delta_X, delta_T,analytical_solution_function):

    x = np.arange(x_dom[0], x_dom[1], delta_X).astype(float)
    t = np.arange(t_dom[0], t_dom[1], delta_T).astype(float)
    X, T = np.meshgrid(x, t)
    
    x_physics = np.expand_dims(X.flatten(), axis=-1)
    t_physics = np.expand_dims(T.flatten(), axis=-1)

    x_physics = torch.tensor(x_physics).requires_grad_(True).to(device)
    t_physics = torch.tensor(t_physics).requires_grad_(True).to(device)

    f_colloc = torch.hstack((x_physics, t_physics)).to(device)
    
    t_ic = np.zeros_like(x)
    X_ic, T_ic = np.meshgrid(x, t_ic)
    
    x_ic = np.expand_dims(X_ic.flatten(), axis=-1)
    t_ic = np.expand_dims(T_ic.flatten(), axis=-1)
    
    ic_sol = analytical_solution_function(x_ic,t_ic, omega)
    ic = np.hstack((np.real(ic_sol), np.imag(ic_sol)))
    
    ic = torch.tensor(ic).requires_grad_(False).to(device)
    
    x_ic = torch.tensor(x_ic).requires_grad_(False).to(device)
    t_ic = torch.tensor(t_ic).requires_grad_(False).to(device)

    ic_colloc = torch.hstack((x_ic, t_ic))
    
    x_b = np.array(x_dom)
    X_b, T_b = np.meshgrid(x_b, t)
    x_b = np.expand_dims(X_b.flatten(), axis=-1)
    t_b = np.expand_dims(T_b.flatten(), axis=-1)

    x_b = torch.tensor(x_b).requires_grad_(False).to(device)
    t_b = torch.tensor(t_b).requires_grad_(False).to(device)

    b_colloc = torch.hstack((x_b, t_b))
    
    return x_physics, t_physics,f_colloc, b_colloc, ic_colloc, ic

omega = 1
x_dom = [-L,L]
t_dom = [0,2*L]
delta_x = 0.2
delta_t = 0.2
analytical_solution_function = get_analytical_solution_base
x_physics, t_physics, f_colloc, b_colloc, ic_colloc, ic = get_physics_colloc_points(x_dom, t_dom, delta_x, delta_t, analytical_solution_function)
```

```{dropdown} Exercise
What will the loss terms look like in this case? Hint: Split the Schrodinger equation in real and imaginary parts to calculate the equation loss.
```

```{dropdown} Solution

The time-dependent Schrödinger equation can be written as
\begin{equation}
    \left(-\frac{\partial v}{\partial t}+\frac{1}{2}\frac{\partial^{2} u}{\partial x^{2}}-\frac{\omega^{2}}{2}x^{2}\right) + i \left(\frac{\partial u}{\partial t}+\frac{1}{2}\frac{\partial^{2} v}{\partial x^{2}}-\frac{\omega^{2}}{2}x^{2}\right) = 0
\end{equation}

The loss function $L$ is given by 
\begin{equation}
L=L_{f}+L_{BC}+L_{IC}
\end{equation}

\begin{equation*}
L_{BC}=\frac{1}{N_{BC}} \sum_{i=1}^{N_{BC}}\left(\left|u\left(\mathbf{r}_{BC}^{i}, t_{BC}^{i}\right)-u^{i}\right|^{2}+\left|v\left(\mathbf{r}_{BC}^{i}, t_{BC}^{i}\right)-v^{i}\right|^{2}\right)
\end{equation*}
\begin{equation*}
L_{IC}=\frac{1}{N_{IC}} \sum_{i=1}^{N_{IC}}\left(\left|v\left(\mathbf{r}_{IC}^{i}, t_{IC}^{i}\right)-u^{i}\right|^{2}+\left|v\left(\mathbf{r}_{IC}^{i}, t_{IC}^{i}\right)-v^{i}\right|^{2}\right)
\end{equation*}
\begin{equation*}
L_{f}=\frac{1}{N_{f}} \sum_{i=1}^{N_{f}}\left|f_{net}\left(\mathbf{r}_{f}^{i}, t_{f}^{i}\right)\right|^{2}
\end{equation*}
At each training step, the loss function is calculated on $N_f$ collocation points, sampled randomly from the grid.
```

We need to modify the training loop to calculate the new physics informed loss function:
```{code-cell} ipython3
:tags: [hide-output]

def train_pinn(model, n_epochs, train_x, train_y, x_physics, t_physics, f_colloc, b_colloc, ic_colloc, ic):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_list = np.zeros(n_epochs)
    print("Epoch \t Loss \t PDE Loss")
    for i in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(train_x)

        loss1 = torch.mean((y_pred-train_y)**2)

        # calculate loss on colloc points

        y_pred = model(f_colloc)

        u = y_pred[:,0]
        v = y_pred[:,1]

        du_dx  = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
        du_dxx  = torch.autograd.grad(du_dx, x_physics, torch.ones_like(du_dx), create_graph=True)[0]
        du_dt = torch.autograd.grad(u,  t_physics, torch.ones_like(y_pred[:,0]),  create_graph=True)[0]
        if debug:
            print("flag_2")
        dv_dx  = torch.autograd.grad(v, x_physics, torch.ones_like(v), create_graph=True)[0]
        dv_dxx  = torch.autograd.grad(dv_dx, x_physics, torch.ones_like(dv_dx), create_graph=True)[0]
        dv_dt = torch.autograd.grad(v,  t_physics, torch.ones_like(v),  create_graph=True)[0]
        if debug:
            print("flag_3")
        loss_u = -du_dt - 1/2 * (dv_dxx + omega**2/2 * x_physics**2) * v.view(-1,1)
        loss_v = -dv_dt + 1/2 * (du_dxx + omega**2/2 * x_physics**2) * u.view(-1,1)

        loss_physics = torch.stack((loss_u, loss_v))
        if debug:
            print("flag_4")
        y_pred_b = model(b_colloc)
        y_pred_ic = model(ic_colloc)
        if debug:
            print("flag_5")
        loss_b = torch.mean(y_pred_b**2)
        loss_ic = torch.mean((y_pred_ic - ic)**2)

        loss2 =  (torch.mean(loss_physics**2) + loss_b + loss_ic)
        if debug:
            print("flag_6")
        

        loss = loss1 + (1e-4) * loss2# add two loss terms together
        loss_list[i] = loss.detach().cpu().numpy()

        loss.backward()
        optimizer.step()
        if debug:
            print("flag_7")
        if i % 500 == 499:
            print(f"{i} {loss} {loss2}")
    
    return model, loss_list
```
### Example: Testing Transferability
We will now train the PINN on the same reduced domain:
```{code-cell} ipython3
:tags: [hide-output]

debug = False
torch.manual_seed(123)
activation =  nn.Tanh()

n_epochs = 10000
model_pinn = NN(2,2,32,3,activation).to(device)
model_pinn, loss_pinn = train_pinn(model_pinn, n_epochs, train_x, train_y, x_physics, t_physics, f_colloc, b_colloc, ic_colloc, ic)
```

```{code-cell} ipython3
:tags: [hide-output]

y_pred = get_model_error(model_pinn, test_x, test_y)
```

```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1
delta_T = 0.01
delta_X = 0.01
    
n_x = np.shape(np.arange(x_dom[0], x_dom[1], delta_X))[0]
n_t = np.shape(np.arange(t_dom[0], t_dom[1], delta_T))[0]

inference(model_pinn, test_x, test_y, n_x, n_t, x_dom, t_dom, omega, 'pinn_reduced')
```

The PINN model matches the system well, even when trained on reduced data.


### Training Considerations
#### Architecture
The architecture and associated hyperparameters have significant impact on the performance of the PINN. The learning model architecture can be customized depending on the nature of the domain. For example, CNNs, RNNs and GNNs can be used for spatial, temporal and interacting problems respectively. For the one-dimensional quantum harmonic oscillator workflow, we use a FCN with 3 layers of 20 neurons each. 

Optimizer selection is important for convergence of the learner and to avoid solutions at local minima. It has been shown that a combination of Adam  at early training and L-BFGS for later stages has been effective for solving a variety of PDEs through PINNs. We use the Adam optimizer only given the relatively simple nature of this problem.

```{dropdown} Exercise
Try running the PINN with ReLU activation function. What are your observations? Why would this activation function not work?
```

```{dropdown} Solution
As we saw in calculating the residual, the choice of activation functions is constrained by the fact that they have to be $(n+1)$-differentiable for $n$-order PDEs. For the one-dimensional quantum harmonic oscillator problem we use $\tanh$ activation functions because they are 3-differentiable.
```

#### Collocation Points
The accuracy of the NN increases with increase in density of collocation points. However, the computational resources required also increases exponentially with the increase in density of points. This can lead to training bottlenecks, especially for high dimensional systems. A trade-off has to be made between the desired accuracy and number of collocation points because training the NN on a large number of points may lead to overfitting and adversely affect generalisability. The distribution of points can also be customised according to the problem domain. Density could be increased around areas of sharp discontinuities to capture more information about the domain. 

```{dropdown} Exercise 
What is the effect of changing collocation points on the training time and accuracy of the Neural Network? You can run it multiple times with different grids of collocation point to observe this.
```

```{dropdown} Solution
Training time and accuracy both increase exponentially with collocation grid size.

<img src="https://github.com/GDS-Education-Community-of-Practice/DSECOP/blob/main/Learning_the_Schrodinger_Equation/res/fig/pinn_colloc_time.png?raw=1" alt="Domain" width="300"/>

<img src="https://github.com/GDS-Education-Community-of-Practice/DSECOP/blob/main/Learning_the_Schrodinger_Equation/res/fig/pinn_colloc_performance.png?raw=1" alt="Domain" width="300"/>
```

## Predicting higher energy states with fully connected and physics-informed neural networks
We will now check the performance of NNs on systems with higher energy states. For demonstration, the system used is $\psi_{1,3}(x,t)$.  Let's derive the analytical equation first.

```{dropdown} Exercise 
a) Derive the first Hermite Polynomial ($H_{3}(y)$) in terms of $x$ for $y=\sqrt{\omega}x$:

Derivation here
$H_{3}(\sqrt{\omega}x) = $

b) Write down the ground state and first excited state energies for QHO:

$E_1 =$,
$E_3=$

c) Use your results from $a$ and $b$ to write down $\psi_{0,1}(x,t)$, and simplify it:

$\psi_{1,3}(x,t) = $
```

Now that we have our analytical solution, let's write some code to compute it for arbitrary values of $x,t,\omega$:

```{code-cell} ipython3
:tags: [hide-output]

def get_analytical_solution_1_3(X,T, omega):
    #Solution:
    phi_0 =(omega / np.pi) ** (1. / 4.) * np.exp(-X * omega * X / 2.0)
    phi_1 = phi_0 *  np.sqrt(omega / 2.) * 2.0 * X
    phi_3 = phi_0 * 1/np.sqrt(48) * (8. * omega**(3./2.) * X**3. - 12. * omega**(1./2.) * X)
    psi = np.sqrt(1. / 2.) * (np.exp(-1j * 3./2. * omega * T) * phi_1 + np.exp(-1j * 7./2. * omega * T) * phi_3)

    return psi
```

You can explore the system here:
```{code-cell} ipython3

@interact(t=widgets.IntSlider(min=0, max=62, step=1, value=0))
def interactive_viz_colour(t):
    tstr = str(t).zfill(2)
    f = urllib2.urlopen("https://github.com/GDS-Education-Community-of-Practice/DSECOP/tree/main/Learning_the_Schrodinger_Equation/")
    fname = f"res/plots/waveform/psi_1_3/t_{tstr}.png"
    image = img.imread(fname)
    plt.figure(figsize=(8,6))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
```

### Fully connected neural network
Using a reduced grid with Fully Connected Network, we have:


```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1

delta_T = 0.1
delta_X = 0.1
x_dom = [-L/4,L/4]
t_dom = [0, 2*L]

train_x, train_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_1_3)
```

```{code-cell} ipython3
:tags: [hide-output]

delta_T = 0.01
delta_X = 0.01
x_dom = [-L,L]
t_dom = [0, 2*L]

test_x, test_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_1_3)
```

```{code-cell} ipython3
:tags: [hide-output]

torch.manual_seed(314)
activation =  nn.Tanh()

model_nn_he = NN(2,2,32,3,activation).to(device)
optimizer = torch.optim.Adam(model_nn_he.parameters(),lr=1e-3)

n_epochs = 10000
```

```{code-cell} ipython3
:tags: [hide-output]

%%time
model_nn_he, loss_list = train_nn(model_nn_he, n_epochs, train_x, train_y)
```

```{code-cell} ipython3
:tags: [hide-output]

y_pred = get_model_error(model_nn_he, test_x, test_y)
```

```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1
delta_T = 0.01
delta_X = 0.01
    
n_x = np.shape(np.arange(x_dom[0], x_dom[1], delta_X))[0]
n_t = np.shape(np.arange(t_dom[0], t_dom[1], delta_T))[0]

inference(model_nn_he, test_x, test_y, n_x, n_t, x_dom, t_dom, omega, 'high_nn')
```
This does not capture any detail beyond the domain, as seen in Abs Err plots. 

### Physics-informed neural network
Let us now rerun this example with a PINN.

```{code-cell} ipython3
:tags: [hide-output]

omega = 1
x_dom = [-L,L]
t_dom = [0,2*L]
delta_x = 0.2
delta_t = 0.2
analytical_solution_function = get_analytical_solution_1_3
x_physics, t_physics, f_colloc, b_colloc, ic_colloc, ic = get_physics_colloc_points(x_dom, t_dom, delta_x, delta_t, analytical_solution_function)
```

```{code-cell} ipython3
:tags: [hide-output]

debug = False
torch.manual_seed(314)
activation =  nn.Tanh()

n_epochs = 10000
model_pinn_he = NN(2,2,32,3,activation).to(device)
```

```{code-cell} ipython3
:tags: [hide-output]

model_pinn_he, loss_pinn = train_pinn(model_pinn_he, n_epochs, train_x, train_y, x_physics, t_physics, f_colloc, b_colloc, ic_colloc, ic)
```

```{code-cell} ipython3
:tags: [hide-output]

y_pred = get_model_error(model_pinn_he, test_x, test_y)
```

```{code-cell} ipython3
:tags: [hide-output]

L = float(np.pi)
omega = 1
delta_T = 0.01
delta_X = 0.01
    
n_x = np.shape(np.arange(x_dom[0], x_dom[1], delta_X))[0]
n_t = np.shape(np.arange(t_dom[0], t_dom[1], delta_T))[0]
test_x, test_y = get_data_set(x_dom, t_dom, delta_X, delta_T, omega, get_analytical_solution_1_3)
```

```{code-cell} ipython3
:tags: [hide-output]

inference(model_pinn_he, test_x, test_y, n_x, n_t, x_dom, t_dom, omega, 'high_pinn')
```

For higher energy states, it is difficult to capture the details with fully connected layers. PINNs can be extended with Recurrent Neural Networks to tackle this.