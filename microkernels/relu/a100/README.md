# ReLU on A100

The Rectified Linear Unit (ReLU) is the most commonly used activation function in deep learning models.

ReLU cases for testing:

| Name     | width | height | channel | batch |
|----------|-------|--------|---------|-------|
| Kernel 1 | 7     | 7      | 32      | 262144|
| Kernel 2 | 14    | 14     | 128     | 16384 |
| Kernel 3 | 54    | 54     | 1024    | 128   |
| Kernel 4 | 128   | 128    | 128     | 128   |



* `relu_bf16.py` : ReLU micro-benchmark code for bf16 data-type
* `relu_ff16.py` : ReLU micro-benchmark code for fp16 data-type
* `relu_fp32.py` : ReLU micro-benchmark code for fp32 data-type
* `relu_results.csv` : Results from execution on ThetaGPU



## Running Natively

+ Setup environment as described in [readme](../../../README.md)
+ Run Code
    ```bash
    $ python relu_bf16.py
    $ python relu_fp16.py
    $ python relu_fp32.py
    ```

## Runnig using Singularity

* Download appropriate containder image as descibed in [readme](../../../README.md)
* Run Code
    ```bash
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python relu_bf16.py
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python relu_fp16.py
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python relu_fp32.py
    ```
