from __future__ import division, print_function

import logging
import os, sys, shutil

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


import uno as benchmark
# Import Candle libraries
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
import candle

from uno_data import DataFeeder, CombinedDataGenerator, TFRecordsHandler, CombinedDataLoader
from uno_tfr_utils import *
from uno_baseline_keras2 import extension_from_parameters, initialize_parameters

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_tfr_from_h5( args, logger=None ):
    loader = CombinedDataLoader(seed=args.rng_seed)
    loader.load(cache=args.cache,
              ncols=args.feature_subsample,
              agg_dose=args.agg_dose,
              cell_features=args.cell_features,
              drug_features=args.drug_features,
              drug_median_response_min=args.drug_median_response_min,
              drug_median_response_max=args.drug_median_response_max,
              use_landmark_genes=args.use_landmark_genes,
              use_filtered_genes=args.use_filtered_genes,
              cell_feature_subset_path=args.cell_feature_subset_path or args.feature_subset_path,
              drug_feature_subset_path=args.drug_feature_subset_path or args.feature_subset_path,
              preprocess_rnaseq=args.preprocess_rnaseq,
              single=args.single,
              train_sources=args.train_sources,
              test_sources=args.test_sources,
              embed_feature_source=not args.no_feature_source,
              encode_response_source=not args.no_response_source,
              use_exported_data=args.use_exported_data,
              )

    loader.partition_data(cv_folds=args.cv, train_split=(1-args.val_split), val_split=args.val_split,
                            cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug,
                            cell_subset_path=args.cell_subset_path, drug_subset_path=args.drug_subset_path)

    # Record the feature length and array for TF Records
    feature_len = 0
    feature_arr = []
    for _,val in loader.input_features.items():
        curr_len = loader.feature_shapes[val][0]
        feature_len += curr_len
        feature_arr.append(curr_len)

    partitions = ['train', 'val', 'test']
    tfr_writer = TFRecordsHandler( partition=partitions, write=True, directory=args.export_data, feature_len=feature_len, feature_sz_arr=feature_arr)

    for partition in ['train', 'val']:

      data_feeder = DataFeeder( partition= partition, filename=args.use_exported_data, batch_size=args.batch_size, shuffle=False, single=args.single, agg_dose=args.agg_dose, on_memory=args.on_memory_loader)
      
      for di in range(data_feeder.steps):
        x_list, y = data_feeder[di]
        feature_mat = np.concatenate( [fi.values.astype(np.float32) for fi in x_list], axis=1)
        y_arr = y.values.astype(np.float32)
        tfr_writer.write_to_file(feature_mat, y_arr, partition)

        if logger is not None:
          logger.info('Generating {} dataset. {} / {}'.format(partition, di, data_feeder.steps))

def create_tfr_from_generator(args, logger=None):
  """ Not implemented yet: CombinedDataGenerator will be used to create tf records """
  return



def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))

    # Use exported data
    shutil.rmtree(args.export_data, ignore_errors=True)

    if args.use_exported_data is not None:
      create_tfr_from_h5(args, logger=logger)
    else:
      create_tfr_from_generator(args, logger=logger)
    
    return


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
