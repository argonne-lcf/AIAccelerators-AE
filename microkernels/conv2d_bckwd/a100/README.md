# Conv2d Backward on A100 

Convolutions make up the vast majority of flops in networks that operate on images and videos and form important parts of networks such as speech and natural language modeling, thus making them perhaps the single most important layer from a performance perspective.

Conv2d kernel configurations :

| Name     | width | height | channel | batch | kernel | filter_w | filter_h | pad_w | pad_h | wstride | hstride |
|----------|-------|--------|---------|-------|--------|----------|----------|-------|-------|---------|---------|
| Kernel 1 | 7     | 7      | 32      | 131072| 32     | 3        | 3        | 0     | 0     | 1       | 1       |
| Kernel 2 | 14    | 14     | 128     | 4096  | 256    | 3        | 3        | 1     | 1     | 1       | 1       |
| Kernel 3 | 54    | 54     | 1024    | 16    | 1024   | 3        | 3        | 1     | 1     | 1       | 1       |
| Kernel 4 | 128   | 128    | 128     | 32    | 128    | 5        | 5        | 0     | 0     | 1       | 1       |


+ `conv2d_bckwd_bf16.py` : conv2d forward micro-benchmark code for BF16 data type
+ `conv2d_bckwd_fp16.py` : conv2d forward micro-benchmark code for FO16 data type
+ `conv2d_bckwd_fp32.py` : conv2d forward micro-benchmark code for FP32 data type
+ `results_conv2d_bckwd.csv` : Results from execution on ThetaGPU


## Running Natively

+ Setup environment as described in [readme](../../../README.md)
+ Run Code
    ```bash
    $ python conv2d_bckwd_bf16.py
    $ python conv2d_bckwd_fp16.py
    $ python conv2d_bckwd_fp32.py
    ```

## Runnig using Singularity

* Download appropriate containder image as descibed in [readme](../../../README.md)
* Run Code
    ```bash
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python conv2d_bckwd_bf16.py
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python conv2d_bckwd_fp16.py
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif python conv2d_bckwd_fp32.py
    ```
