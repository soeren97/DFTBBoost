# DFTBBoost

A graph convolutional neural network project aiming to enhance DFTB calculations to DFT level of theory.

Processed datasets can be found at https://sid.erda.dk/cgi-sid/ls.py?share_id=CAhOAMkq9r

## Setup enviroment

Create a conda enviroment and activate it

* conda create -n DFTBBoost
* conda activate DFTBBoost

Install pip and required packages. If GPU is available write it in the brackets otherwise write CPU

* conda install pip
* conda install .[GPU]

Activate pre-commit hooks 

* pre-commit install

The enviroment is now ready to be used.