# AI Accelerators Evaluation:

This repository is to aid the AD/AE evaluation for the paper titled "A Comprehensive Evaluation of AI Accelerators for Deep Learning Workloads".

The repository divided into three sub-directories covering all the experiments in the paper -
+ [Microkernels](./microkernels/) : Code for running Deep Learning Primitives or Microkernels. 
+ [Benchmarks](./benchmarks/) : Code for running Deep Learning Benchmarks.
+ [Scientific ML Applications (sciML)](./sciML/) : Code for running Scientific Machine Learning Applications. 

The codes in this repository are tested on [ThetaGPU](https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview) system at ANL. Here we provide instructions to run the experiments listed in the paper in two ways: 
* natively, using the scripts as-is with installed software packages listed above or 
* running using the container images with the instructions provided.

## Running Natively 

The scripts are generic and can be run on any A100 GPU system with the required packages installed, namely Tensorflow (version 2.4 or later), PyTorch (version 1.8 or later) and setting appropriate paths for CUDA 11.0 and cuDNN 8.

```bash
export PATH=<path-to-cuda>/bin:$PATH
export LD_LIBRARY_PATH=<path-to-cuda-libraries>/lib64:$LD_LIBRARY_PATH
```


## Running using Container

Singularity and Docker are container systems that allow users full control over their software environment. Users can create their own container images or choose from a set of pre-defined images, and specify that with the submitted jobs. In this repo, we provide Singularity and Docker container images to run the benchmarks and applications used in this work. Here we provide instructions to run the scripts with Singularity images in each directory. The steps to replicate the experiments with Docker images should be obvious. A simple example to run GEMM scripts with Docker image is provided below for reference.

### Singularity

* Download singularity container images for 
  * PyTorch from (https://doi.org/10.5281/zenodo.6629456) and 
  * Tensorflow from (https://doi.org/10.5281/zenodo.6629290)

* Run the benchmarks evaluated with container images provided. For example, to evaluate GEMMs with Pytorch, use the following command:
    ```bash
    singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python gemm_torch.py
    ```
* Each separate subfolder provides the commands for executing the other benchmarks and applications. Similar steps should be used for Tensorflow as well.


### Docker

* Download docker container for PyTorch and Tensorflow from 
  https://hub.docker.com/
    ```bash
    sudo docker pull zhenxie92/sc22-aiaccelerators-ae-pytorch
    or 
    sudo docker pull zhenxie92/sc22-aiaccelerators-ae-tf2.
    ```

* Run the benchmarks evaluated with container images provided. For example, to evaluate GEMMs with Pytorch, use the following command:
  
    ```bash
    sudo docker run --gpus all -it --rm zhenxie92/sc22-aiaccelerators-ae-pytorch python gemm_torch.py
    ```


