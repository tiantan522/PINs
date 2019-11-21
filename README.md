# PINs: Parameterized Indexed Value Function for Efficient Exploration in Reinforcement Learning 

This is the reference implementation of the algorithm PINs in the paper. This repository is based on environments from [bsuite](https://github.com/deepmind/bsuite). To install [bsuite], please follow instructions at https://github.com/deepmind/bsuite

#### Instructions

To run PINs on Cartpole-Swingup with sparse rewards, do:
`python bsuite_experiment.py`

Output files will be written to `result_path` in `bsuite_experiment.py` by default. The file `plot.py` can be used to plot episodic reward after training.  

#### Communication

If you have a problem running the code or spot a bug, please open an issue. 
Please direct other correspondence to Tian Tan: tiantan@stanford.edu 
