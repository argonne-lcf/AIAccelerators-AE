# Training UNO model with IPU
This repo contains the code is a stripped down version of UNO project located at: https://github.com/ECP-CANDLE/Benchmarks
The code is adapted to use Graphcore's IPU hardware for UNO model's training and inference.

## Install Poplar SDK for IPU
Follow the instructions described [here](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#sdk-installation) to install the POPSDK for Tensorflow. This code uses tensorflow & keras framework which is also supported for the IPU. To learn more about using tensorflow for IPU, please refer to the [documentation](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/index.html) and available [tutorials](https://github.com/graphcore/tutorials/tree/master/tutorials/tensorflow2)

### Create venv

```bash
virtualenv -p python3.6 ~/workspace/tensorflow_uno_env
source ~/workspace/tensorflow_uno_env/bin/activate
```

## Install Tensorflow

```bash
pip install /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/tensorflow-2.4.4+gc2.4.0+139613+8debb698097+amd_znver1-cp36-cp36m-linux_x86_64.whl
```

## Clone Repo

```bash
git clone https://github.com/daman-khaira/uno.git
cd uno/
git checkout master
```

## Install Prerequisites
Once the correct python environment is activated as described in the previous section, install the required prequisites using
```
cd Uno/
pip install -r requirements.txt
```
## Run training on AUC dataset

First, download the AUC dataset using the following command:
```
#mkdir data_dir
#wget -o data_dir/top_21_auc_1fold.uno.h5 http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
```
To train the UNO model on the dataset downloaded previously, use:
```
python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data /localdata/damank/dataset/UNO/H5/top_21_auc_1fold.uno.h5 -e 3 --save_weights save/saved.model.weights.h5

#  python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data data_dir/top_21_auc_1fold.uno.h5 -e 3 --save_weights save/saved.model.weights.h5
```
