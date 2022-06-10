# BraggNN on NVIDIA A100

## Prepare Dataset

* `setup_data.sh` : Download and prepare the dataset by running this script. If successful, the current directory should have directory named 'dataset'


## Install Prerequisites

Install the required prequisites using
```bash
pip install -r requirements.txt
```
## Train model

* This script run the model training & validation by running the following command:
```bash
python main.py -device gpu -dataset <path to dataset directory: Default='./dataset'>
```

* Use ```python main.py -h``` to list all the available settings

