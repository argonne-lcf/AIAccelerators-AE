# Uno on NVIDIA A100

* Uno source is available at https://github.com/ECP-CANDLE/Benchmarks/tree/develop/Pilot1/Uno 
* All instructions to install the prerequisites are in the [README](./A100/Uno/README.md) file in this repo.


# Running Natively

* For AUC model configuration with CCLE dataset, use the command 
     ```bash
     python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data top_21_auc_1fold.uno.h5 -e 5
     ```

* For larger dataset with all train sources, follow the instructions listed in the README file in that repo. Generate pre-staged dataset file. Use `--export_data` to specify the file name and use a large batch size to speed up.

     ```bash
     python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 4096 --export_data All.h5 --shuffle True
     ```

* Training with pre-staged dataset. Use `--use_exported_data` to point dataset file.

     ```bash
     python uno_baseline_keras2.py --train_sources all --cache cache/all  --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True -z 512 --use_exported_data All.h5 --cp True --shuffle True --tb True
     ```

* Inferencing with pre-staged dataset.
     ```bash
     python uno_infer.py --data All.h5 --model_file model.h5 --n_pred 30
     ```


