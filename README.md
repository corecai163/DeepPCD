DeepPCD: Enabling AutoCompletion of Indoor Point Clouds with Deep Learning

##### The code is tested on nvidia gpus with cuda support.
## to check your gpu 
lspci | grep -i nvidia

#### Install nvidia drivers, cuda 11.1 and cuDNN.
##Please refer to the official website: 
https://developer.nvidia.com/cuda-toolkit

It should work with other cuda versions.

## to verify your installation, check
nvidia-smi
nvcc -V

#### Install required Dependencies

sudo apt install ninja-build

pip3 install -r requirements.txt

#### Data set
https://www.dropbox.com/scl/fo/5qqciu52p7lhpv9ve7kf4/h?rlkey=6873a7zyxqepe0a23arcszn2w&dl=0

#### Train
python3 train_myNet.py


#### Evaluate
python3 eval_myNet.py

#### 
#### If this helps you, please cite our work




