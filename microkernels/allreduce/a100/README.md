# All-Reduce

Neural networks today are often trained across multiple GPUs or even multiple systems, each with multiple GPUs. There are two main categories of techniques for doing this: synchronous and asynchronous. Synchronous techniques rely on keeping the parameters on all instances of the model synchronized, usually by making sure all instances of the model have the same copy of the gradients before taking an optimization step. The primitive usually used to perform this operation is called All-Reduce.

All-Reduce based on the number of ranks and the size of the data.

All-Reduce cases for testing:

| Size (# of floats) | Number of Ranks      | Application        |
|--------------------|----------------------|--------------------|
| 16777216           | 2                    | Speech Recognition |
| 16777216           | 4                    | Speech Recognition |
| 16777216           | 8                    | Speech Recognition |
| 64500000           | 16                   | Speech Recognition |

Allreduce configurations

Each GPU has 64500000 single floating point numbers, ~250 MBytes per GPU


## Running Natively

+ Setup environment as described in [readme](../../../README.md)
+ Run Code
    ```bash
    $ ./nccl_single_all_reduce numberofGPU 
    ```
    where numberofGPU is number of GPUs such as 2, 4, and 8

## Running using Singularity

* Download appropriate container image as descibed in [readme](../../../README.md)
* Run Code
    ```bash
    singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif ./nccl_single_all_reduce numberofGPU
    ```
    where numberofGPU is number of GPUs such as 2, 4, and 8
