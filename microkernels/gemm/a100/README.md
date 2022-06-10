# GEMM on A100

Dense matrix multiplication exist in almost all deep neural networks today.  They are used to implement fully connected layers and vanilla RNNs and are building blocks for other types of recurrent layers.

Common terminology to describe a matrix problem is the triple (M, N, K), which describes the sizes of the matrices involved.

GEMM kernels configurations :

| Name     |   M   |   N   |   K   |
|----------|-------|-------|-------|
| Kernel 1 | 64    | 1760  | 1760  |
| Kernel 2 | 2560  | 64    | 2560  |
| Kernel 3 | 1760  | 128   | 1760  |
| Kernel 4 | 2560  | 2560  | 2560  |


* `gemm_torch.py` : GEMM micro-benchmark code
* `gemm_results.csv` : Results from execution on ThetaGPU
* `gemm_results_scaling.csv` : Results of strong scaling for GEMM on ThetaGPU.


## Running Natively

+ Setup environment as described in [readme](../../../README.md)
+ Run Code
    ```bash
    $ python gemm_torch.py
    ```

## Running using Singularity

* Download appropriate container image as descibed in [readme](../../../README.md)
* Run Code
    ```bash
    singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python gemm_torch.py
    ```
