# Master's thesis - Benjamin Kessler

## Overview

This repository contains the code for my Master's thesis. The code heavily relies on Googles [neural_tangents](https://github.com/google/neural-tangents) to train and study neural networks of both finite and infinite width. All experiments are run on [sciCORE](https://scicore.unibas.ch) and utilize GPU via [XLA](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/index.md).

## Contents
* [Installation](#installation)
* [Environment](#environment)

## Installation

The instructions below describe how to install neural_tangents using [Anaconda](https://docs.anaconda.com) so that it utilizes the GPU on sciCORE. For a more generic installation, see [neural_tangent's](https://github.com/google/neural-tangents#installation) instructions.

As neural_tangents is built using [JAX](https://github.com/google/jax), you must first install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). JAX unfortunately does not bundle any of these as part of the `pip` package.

Note that Conda handles the dependencies between your packages for you.  CUDA is install by default on sciCORE. To check which versions are available, you can run
```
module spider
````
The CUDA 10 JAX wheels require cuDNN 7, whereas the CUDA 11 JAX wheels require cuDNN 8. Depending on which version you prefer, you can install them to your conda environment using
````
conda install -c cuda-forge cudatoolkit=version
conda install -c cuda-forge cudnn=version
````
Keep in mind that your desired version might not be available or compatible with the given version of TensorFlow. In this case, an older version should be considered. Furthermore, please make sure that your version is available on sciCORE. You can always check the version in your conda environment by running
````
conda list cudatoolkit
conda list cudnn
````

This concludes the installation of TensorFlow, CUDA and cuDNN. So all that remains is to install JAX and neural_tangents. Unfortunately, JAX is not yet downloadable via conda, so please install JAX using ``pip``:
```
conda install pip
```
Once ``pip`` is installed, you can install JAX by running
```
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.64+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
````
where the jaxlib version must correspond to the version of the existing CUDA installation you want to use, with `cuda112` for CUDA 11.2, `cuda111` for CUDA 11.1, `cuda110` for CUDA 11.0, `cuda102` for CUDA 10.2, and `cuda101` for CUDA 10.1.

Note that some GPU functionality in JAX expects the CUDA installation to be at `/usr/local/cuda-X.X`. In an anaconda environment, this is unfortunately not given. The quickest solution I found is to copy the `libdevice` file (dependent of CUDA version) found in `/anaconda/envs/your_env/lib` to a new folder with path `~/folder_name/nvvm/libdevice`. This can be done by running
````
cp anaconda/envs/your_env/lib/libdevice.10.bc ~/folder_name/nvvm/libdevice
````
This completes the installation of JAX with automatic GPU support.

Once JAX is installed, you can install neural_tangents by running
````
pip install neural_tangents
````
As a drawback from our relocating of the `libdevice`, we have to set the following environment variable before importing JAX:
````
XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/folder_name
`````
Or alternatively, call above environment variable before executing your python script. Note that you also have to load the corresponding CUDA and cuDNN modules in sciCORE:
```
ml CUDA/version % or just ml CUDA if version is correct by default
ml cuDNN
XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/folder_name python your_script.py
```

Note that [datasets.py](datasets.py) relies on `tensorflow-datasets` to load datasets. To install this, run
```
conda install -c conda-forge tensorflow-datasets
```
or (less advised)
```
pip install tensorflow tensorflow-datasets more-itertools --upgrade
```
## Environment

The environment that worked best for me personally (as of Mai 2021) was the following combination of packages:

| Name            | Version        | Build      | Channel  |
|-----------------|----------------|------------|----------|
| tensorflow-datasets  | 4.2.0     | pypi_0     | pypi     |
| cudatoolkit     | 10.1.243       | h6bb024c_0 | anaconda |
| cudnn           | 7.6.5          | cuda10.1_0 | anaconda |
| jax             | 0.2.12         | pypi_0     | pypi     |
| jaxlib          | 0.1.65+cuda101 | pypi_0     | pypi     |
| neural_tangents | 0.3.6          | pypi_0     | pypi     |

The results were computed on two rtx8000 graphic cards with a memory of 48601MiB each. Alternatively, given that the GPU supports CUDA 11, I used

| Name            | Version        | Build      | Channel  |
|-----------------|----------------|------------|----------|
| tensorflow-datasets  | 4.3.0     | pypi_0     | pypi     |
| cudatoolkit     | 11.2.2         | he111cf0_8 | conda-forge |
| cudnn           | 8.1.0.77       | h90431f1_0 | conda-forge |
| jax             | 0.2.13         | pypi_0     | pypi     |
| jaxlib          | 0.1.66+cuda111 | pypi_0     | pypi     |
| neural_tangents | 0.3.6          | pypi_0     | pypi     |

This environment should run on all partitions available.