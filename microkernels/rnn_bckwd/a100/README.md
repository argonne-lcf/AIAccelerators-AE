## Recurrent Layers

Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language. There are three types of recurrent cells, such as, vanilla RNNs, LSTMs and GRUs. We use the LSTM as a representative.

RNN cases for testing:

| Name     | time_step | batch_size | input_size | hidden_size |
|----------|-----------|------------|------------|-------------|
| Kernel 1 | 50        | 64         | 256        | 256         |
| Kernel 2 | 25        | 32         | 512        | 512         |
| Kernel 3 | 25        | 16         | 512        | 512         |
| Kernel 4 | 50        | 32         | 512        | 512         |


+ `rnn_bckwd_fp16.py` : rnn forward micro-benchmark code for FP16 data type
+ `rnn_bckwd_fp32.py` : rnn forward micro-benchmark code for FP32 data type
+ `results_rnn_bckwd.csv` : Results from execution on ThetaGPU


## Running Natively

+ Setup environment as described in [readme](../../../README.md)
+ Run Code
    ```bash
    $ run_bckwd_fp16.sh
    $ run_bckwd_fp32.sh
    ```

## Runnig using Singularity

* Download appropriate containder image as descibed in [readme](../../../README.md)
* Run Code
    ```bash
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif run_bckwd_fp16.sh
    $ singularity exec --nv --bind /path:/bind_path /path/sc22-aiaccelerators-ae-pytorch.sif run_bckwd_fp32.sh
    ```
