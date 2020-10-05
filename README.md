# ABCdiscrete
A python repository for a likelihood-free inference method for discerete data, ABC-discrete

# Contents
The repostiroy is orgniazed in 7 folders, which details are described below:
1. **experiments** main execution file folder - this folder contains the main python files for each one of the experiments executed
2. **testbeds** contains the use-cases utalized for the experiments, with a super class `main_usecase.py` that specifies the functionalities that a given use case must posses for the code to execute
3. **algorithms** contains a super class, `main_sampling.py` that specifies the minimum required functions that are shared across the different algirhtms, as well as subclasses such as:
- population-based MCMC `mcmc.py`
- approximate bayesian computation `abc.py`
- binarized neural network `bnn.py`
4. **kernels** contains alternative proposal methods. 
5. **results** location where the results of each experiment are stored 
6. **evaluate** contains the main evaluation files for each experiment 
7. **utils** contains additional functionalities such as plotting or creating text files to aid storing the results in a user-friendly set-up. 

# How to use 

Before running the code specify the python path for the experiments folder. Furthermore, for the experiments on binarized neural network and neural architecture search the actual data files must be dowanloaded and stored on your local device (specify python path in the testbeds files). Morover, for the NAS experiment the software package nasbench must be installed. 
