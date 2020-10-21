# ABCdiscrete: Approximate Bayesian Computation for Discrete Data
The repository contains the python code required to reproduce the experiments carried out in the following paper:

- Ilze A. Auzina, and Jakub M. Tomczak, 'ABC-di:Approximate Bayesian Computation for Discrete Data', 2020, [arXiv](https://arxiv.org/abs/2010.09790)

## Requirements 
The code requires: 
- python 3.5 or above
- numpy 
- scipy
- scikit-image
- matplotlib
- nasbench software and its dependencies (https://github.com/google-research/nasbench)


## Run the Experiments 
1. Open the `experiments` directory 
2. Select one of the experiments of interest
3. Check the settings and update the **pythonpath** and the **datapath** (if needed), see example below:
    - PYTHONPATH = '/home/username/location/ABCdiscrete/experiments'
    - DATA_PATH = '/home/username/location/nasbench_only108.tfrecord'
4. Run the experiment 

*the python code is ran in multiple parallel processes (MAX_PROCESS), thus, check how many nodes you have available*

## Evaluate the Experiments
1. Open the `evaluate` directory
2. Select the experiment you want to evaluate
3. Check the settings and update the **pythonpath** and the **datapath** 
4. Run the evaluation

## Overall Design 
The repository is organized in 7 folders, which details are described below:
- **experiments**: the directory contains the main execution files for each experiment (every experiment has a separate execution file).
- **testbeds** : the directory contains the use-cases utilised for the experiments. The super class `main_usecase.py` specifies the functionalities that any use-case must posses (if you want to implement an additional use-case). 
- **algorithms**: contains the super class, `main_sampling.py` that specifies the minimum required functions, and the subclasses:
    - population-based MCMC `mcmc.py`
    - population-based ABC `abc.py`
- **kernels**: contains the possible proposal distributions. 
- **results**: the directory where the results will be stored.
- **evaluate**: contains the execution files for the evaluation.
- **utils**: contains additional functionalities such as plotting or creating text files to aid storing the results in a more user-friendly way. 

