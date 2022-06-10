### Training UNO model with A100
This repo contains the code is a stripped down version of UNO project located at: https://github.com/ECP-CANDLE/Benchmarks.


## Install Prerequisites
Once the correct python environment is activated as described in the previous section, install the required prequisites using
```
pip install -r requirements.txt
```
## Generate tfRecords
Generate the tfRecords using the following script.
```
./generate_tf_records.sh 
```
This generates ALL_TFR or CCLE_TFR tfrecords file on which the training is done using 
## Run CCLE / ALL dataset 

```
./run_all.sh
```


