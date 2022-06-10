# Resnet50 on NVIDIA A100 

The Resnet50 benchmark here is from NVIDIA DeepLearning Examples Github Repo. 
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md

## Running Resnet50

Important Files - 
* `ConvNets` is the direcotry which consists of relevant code from this repo. 
* `logs.txt` : File contains the outout of running benchmark
* `benchmark_fp32.json` & `benchmark_amp.json` file contain the detailed performance numbers from the runs. 

### Initial Setup
    
  * Create directories called `train` and `val` needed for running of this code. 

### Running ResNet50 Benchmark

* Running for automatic Mixed Precision for TRAINING 
    ```bash
        $ python ./launch.py --model resnet50 --precision AMP --mode benchmark_training --platform DGXA100 . --raport-file benchmark_amp.json --epochs 1 --prof 100
    ```

* Running for FP32 Precision for TRAINING
    ```bash
    $ python ./launch.py --model resnet50 --precision FP32 --mode benchmark_training --platform DGXA100 . --raport-file benchmark_fp32.json --epochs 1 --prof 100
    ```

* Running for automatic Mixed Precision for INFERENCE 
    ```bash
        $ python ./launch.py --model resnet50 --precision AMP --mode benchmark_inference --platform DGXA100 . --raport-file benchmark.json --epochs 1 --prof 100
    ```

* Running for FP32 Precision for INFERENCE
    ```bash
    $ python ./launch.py --model resnet50 --precision TF32 --mode benchmark_inference --platform DGXA100 . --raport-file benchmark.json --epochs 1 --prof 100
    ```

