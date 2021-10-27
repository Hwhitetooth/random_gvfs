# Learning State Representations from Random Deep Action-conditional Predictions

This repository is the official implementation of 
[Learning State Representations from Random Deep Action-conditional Predictions](https://arxiv.org/abs/2102.04897).


## Requirements
Run the following command to create a new conda environment with all dependencies:
```bash
conda env create -f conda_cuda11_cp38_linux64.yml
```
Then activate the conda environment by
```bash
conda activate rgvfs
```
Or if you prefer using your own Python environment, run the following command to install the dependencies:
```bash
pip install -r requirements.txt
```


### Atari
OpenAI Gym does not contain the ROMs for the Atari games. 
Please refer to [atari-py](https://github.com/openai/atari-py) for how to download and import the ROMs.


### DeepMind Lab
Please refer to the instructions at [https://github.com/deepmind/lab](https://github.com/deepmind/lab) for installation.


## Training
Run the following command for the Atari experiment:
```bash
python -m experiments.atari
```
Run the following command for the DeepMind Lab experiments:
```bash
python -m experiments.dmlab
```

To run the stop-gradient experiments, use the argument `--stop_gradient`. 
For example, the following command runs the stop-gradient experiment in Atari
```bash
python -m experiments.atari --stop_gradient
```