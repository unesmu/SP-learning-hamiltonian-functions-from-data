# Semester project : Learning Hamiltonian functions from data

## Abstract

Recent advances in physics-informed neural networks have been successful at using neural networks to capture the dynamics of physical systems such as the simple and double
pendulum **[1][2][3][4]**. These neural networks are endowed with physics-based inductive biases
by incorporating Hamiltonian dynamics in their architecture. These architectures leverage
neural ordinary differential equations **[5]** for this purpose. The goal of this project is to
design physics-informed architectures based on the port-Hamiltonian framework to model
dynamical systems. The simple pendulum and the Furuta pendulum were selected for this
purpose.


<p align="center">
  <img width="800"  src="https://github.com/unesmu/SP-learning-hamiltonian-functions-from-data/blob/d22b3a6378b258452aaf5858ee02db44024f9e44/furuta_pendulum/data/TRAJECTORIES_test_set5test%20(1).png"> 
</p>
<p align="center">
    <strong>Figure 1: </strong> Expanding-wide-HNN model experiments with the Furuta pendulum when G is known (See the <a href="https://github.com/unesmu/SP-learning-hamiltonian-functions-from-data/blob/f549d406f0ca7460e64a615c0808ce2bf6217d33/report.pdf">report</a> for more details and figures)  
</p>
<p align="center">
  <img width="800"  src="https://github.com/unesmu/SP-learning-hamiltonian-functions-from-data/blob/d22b3a6378b258452aaf5858ee02db44024f9e44/simple_pendulum/101hz_simple_pend.png"> 
</p>
<p align="center">
    <strong>Figure 2: </strong> Input-HNN model experiments with the simple pendulum when G is known
</p>

<p align="center">
  <img width="500"  src="https://github.com/unesmu/SP-learning-hamiltonian-functions-from-data/blob/a415df09f39e9ad1b6671c8754b929b747bc3b57/furuta_pendulum/data/FIG_-IHNN%20(1).png"> 
</p>
<p align="center">
    <strong>Figure 3:</strong> Input-HNN model summary 
</p>




## Repository content

- `simple_pendulum` : this folder contains the code and experiments that were run on the simplependulum
- `furuta_pendulum` : this folder contains the code and experiments that were run on the Furuta pendulum

Both of these folders are organized in the following way :

- `data`: contains folders that contains experiment runs, where the plots, models, stats, and train/test sets are saved

- `notebook` : contains jupyter notebooks that run script files in the `src` folder
- `src`: contains the python script files used to run the experiments

## Requirements

```
matplotlib==3.5.3
numpy==1.23.3
torchdiffeq==0.2.2
dill==0.3.5.1
torch==1.12.1
jupyter==1.0.0
```

## How to run

### Case 1: If you will run this on your own computer

First create a virtual environment using either conda or venv, then activate that virtual environment.

You can do so by using the following command in windows terminal (assuming you have a terminal open at the project directory):  
`python -m venv pendulumenv` 

Then activate it by running the activate script (assuming you are on windows 10):  
`C:\path\to\pendulumenv\Scripts\activate`

Then open terminal (anaconda terminal or git bash), cd to this repository's directory, and run : `pip install -r requirements.txt`

You can now open one of the notebooks and run the code after selectiong the `pendulumenv` environment inside jupyter.


### Case 2: If you will run this code using google colab

First download this repository, place it in a folder named `1_SP_Ham_func` in the main directory of your google drive. Then run the notebooks. You need to give the notebook acess to your google drive. This does not work with epfl google accounts and only with a private google account.

If you have the directory setup as above you can directly click one of these buttons:

 `TODO ADD CLICKABLE BUTTONS` 

## References

**[1]** S. Greydanus, M. Dzamba, and J. Yosinski, “Hamiltonian neural networks,” Advances in Neural Information Processing Systems, vol. 32, 2019.
**[2]** Y. D. Zhong, B. Dey, and A. Chakraborty, “Symplectic ode-net: Learning hamiltonian dynamics with control,” arXiv preprint arXiv:1909.12077, 2019.  
**[3]** “Dissipative symoden: Encoding hamiltonian dynamics with dissipation and control into deep learning,” arXiv preprint arXiv:2002.08860, 2020.  
**[4]** S. A. Desai, M. Mattheakis, D. Sondak, P. Protopapas, and S. J. Roberts, “Porthamiltonian neural networks for learning explicit time-dependent dynamical systems,”
Physical Review E, vol. 104, no. 3, p. 034312, 2021.  
**[5]** R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” Advances in neural information processing systems, vol. 31, 2018.  

