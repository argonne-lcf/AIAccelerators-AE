# UNet on NVIDIA A100

## Prerequisites

```bash
python >=3.8.10
scikit-image >= 0.19.1
tqdm >= 4.62.3 
matplotlib = 3.5.0
pytorch == 1.10.0
torchvision == 0.11.1
numpy=1.20.3
```

## Running UNet

* Download the dataset from  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

* Update the path to the dataset correctly under  `data_loaders` function in the `train.py`

* Train UNet
    ```bash 
    python ./train.py
    ```

## Scalability Study

* Perform an evaluation to understand how the UNet's throughput varies with an increasing number of GPUs. We scale the number of GPUs of NVIDIA A100 from 1 to 8 running UNet with the Kaggle-3m dataset

* Run scalability
    ```bash 
    sh ./scalability_testing.sh
    ```
