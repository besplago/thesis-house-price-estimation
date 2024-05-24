# House Price Estimation

## Introduction
This is the code for our thesis project on house price estimation.

## Usage
It should be relatively simple to run our code. To run the code containing our models
do the following:

1. Clone the repository
2. Make sure you have the required libraries installed. You can do this by running `pip install -r requirements.txt`. Some of the packages might fail which will cause the command to stop. You can use the following command to install each individually: windows: `Get-Content .\requirements.txt | ForEach-Object {pip install $_}`, linux: `cat requirements.txt | xargs -n 1 pip install`
3. Open the `train_eval_models_CPU.ipynb` notebook
4. Run the notebook

All our models, except our DenseNet and VGG16-based models, are trained and evaluated in train_eval_models_cpu.ipynb. If it is desired to train the DenseNet model, it must be done through our train_eval_models_GPU.ipynb file. The procedure is the same, however, this notebook runs on an NVIDIA GPU. The running of this notebook depends to a large extent on the computer it is run on, and it is therefore difficult to make a guide on how to run it (For a starting point, check this guide: https://saturncloud.io/blog/how-to-run-jupyter-notebook-on-gpus/).

Our data exploration can be run in the same manner and is found in the file called `Data_Exploration.ipynb`.