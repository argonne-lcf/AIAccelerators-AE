#! /usr/bin/env python

from __future__ import division, print_function

import logging
import os, sys, shutil

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
#from tensorflow.python import ipu
from tensorflow.keras.models import Model

import uno as benchmark

# Import Candle libraries
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
import candle

import uno_data
from uno_data import CombinedDataLoader, CombinedDataGenerator, DataFeeder, TFDataFeeder, TFRecordsHandler
from model import build_model
from uno_tfr_utils import *

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.dropout > 0:
        ext += '.DR={}'.format(args.dropout)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.no_gen:
        ext += '.ng'
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += '.D{}={}'.format(i + 1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i + 1, n)

    return ext


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'corr': corr}


def log_evaluation(metric_outputs, logger, description='Comparing y_true and y_pred:'):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info('  {}: {:.4f}'.format(metric, value))


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class MultiGPUCheckpoint(ModelCheckpoint):

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


class SimpleWeightSaver(Callback):

    def __init__(self, fname):
        self.fname = fname

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

    def on_train_end(self, logs={}):
        self.model.save_weights(self.fname)

def initialize_parameters(default_model='uno_default_model.txt'):

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, default_model, 'keras',
                                    prog='uno_baseline', desc='Build neural network based models to predict tumor response to single and paired drugs.')

    # Initialize parameters
    gParameters = candle.finalize_parameters(unoBmk)
    # benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

def getDataLoader(args) -> CombinedDataLoader:
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
    return loader

def exportData(args: dict, loader:CombinedDataLoader):
    fname = args.export_data
    target = args.agg_dose or 'Growth'

    loader.partition_data(cv_folds=args.cv, train_split=(1-args.val_split), val_split=args.val_split,
                            cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug,
                            cell_subset_path=args.cell_subset_path, drug_subset_path=args.drug_subset_path)

    config_min_itemsize = {'Sample': 30, 'Drug1': 10}
    if not args.single:
        config_min_itemsize['Drug2'] = 10

    partitions = ['train', 'val', 'test']

    # Open a pandas file handler
    store = pd.HDFStore(fname, complevel=9, complib='blosc:lz4') if (args.export_data is not None) else None

    # Record the feature length and array for TF Records
    feature_len = 0
    feature_arr = []
    for _,val in loader.input_features.items():
        curr_len = loader.feature_shapes[val][0]
        feature_len += curr_len
        feature_arr.append(curr_len)

    tfr_writer = None
    if args.export_tfrecords is not None:
        tfr_writer = TFRecordsHandler( partition=partitions, write=True, directory=args.export_tfrecords, feature_len=feature_len, feature_sz_arr=feature_arr)

    for partition in partitions:
        gen = CombinedDataGenerator(loader, partition=partition, batch_size=args.batch_size, shuffle=args.shuffle)
        for i in range(gen.steps):
            logger.info('Generating {} dataset. {} / {}'.format(partition, i, gen.steps))
            x_list, y = gen.get_slice(size=args.batch_size, dataframe=True, single=args.single)

            if args.export_tfrecords is not None:
                feature_mat = np.concatenate( [fi.values.astype(np.float32) for fi in x_list], axis=1)
                y_arr = y[target].values.astype(np.float32)
                tfr_writer.write_to_file(feature_mat, y_arr, partition)

            if store is not None:
                for j, input_feature in enumerate(x_list):
                    input_feature.columns = [''] * len(input_feature.columns)
                    store.append('x_{}_{}'.format(partition, j), input_feature.astype('float32'), format='table')
                store.append('y_{}'.format(partition), y.astype({target: 'float32'}), format='table', min_itemsize=config_min_itemsize)
            

    # save input_features and feature_shapes from loader
    if store is not None:
        store.put('model', pd.DataFrame())
        store.get_storer('model').attrs.input_features = loader.input_features
        store.get_storer('model').attrs.feature_shapes = loader.feature_shapes

        store.close()
    
    logger.info('Completed generating {}'.format(fname))
    return

def run( params, ipu_strategy = None ):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))

    loader = getDataLoader(args)

    target = args.agg_dose or 'Growth'
    val_split = args.val_split

    if args.export_data or args.export_tfrecords:
        return exportData(args, loader)

    ## TODO: Make sure args.use_exported_data is not none

    model = build_model(loader, args)
    logger.info('Combined model:')
    model.summary(print_fn=logger.info)
    # plot_model(model, to_file=prefix+'.model.png', show_shapes=True)

    if args.cp:
        model_json = model.to_json()
        with open(prefix + '.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={:.5g}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold + 1, cv))
            cv_ext = '.cv{}'.format(fold + 1)

        #with ipu_strategy.scope():
        if True:
            model = build_model(loader, args, logger=logger)
            if args.initial_weights:
                logger.info("Loading initial weights from {}".format(args.initial_weights))
                model.load_weights(args.initial_weights)

            optimizer = optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
            base_lr = args.base_lr or K.get_value(optimizer.lr)
            if args.learning_rate:
                K.set_value(optimizer.lr, args.learning_rate)

            if args.use_exported_data is not None:
                train_gen = TFDataFeeder(partition='train', tfr_directory= args.use_tfrecords,  filename=args.use_exported_data, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single, agg_dose=args.agg_dose, on_memory=args.on_memory_loader)
                val_gen  = TFDataFeeder(partition='val', tfr_directory= args.use_tfrecords, filename=args.use_exported_data, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single, agg_dose=args.agg_dose, on_memory=args.on_memory_loader)
                test_gen = TFDataFeeder(partition='test', tfr_directory= args.use_tfrecords, filename=args.use_exported_data, batch_size=args.batch_size, shuffle=args.shuffle, single=args.single, agg_dose=args.agg_dose, on_memory=args.on_memory_loader)
            else:
                pass


            # calculate trainable and non-trainable params
            params.update(candle.compute_trainable_params(model))

            candle_monitor = candle.CandleRemoteMonitor(params=params)
            timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
            es_monitor = keras.callbacks.EarlyStopping(patience=10, verbose=1)

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            warmup_lr = LearningRateScheduler(warmup_scheduler)
            checkpointer = MultiGPUCheckpoint(prefix + cv_ext + '.model.h5', save_best_only=True)
            tensorboard = TensorBoard(log_dir="tb/{}{}{}".format(args.tb_prefix, ext, cv_ext))
            history_logger = LoggingCallback(logger.debug)

            callbacks = [candle_monitor, timeout_monitor, history_logger]
            if args.es:
                callbacks.append(es_monitor)
            if args.reduce_lr:
                callbacks.append(reduce_lr)
            if args.warmup_lr:
                callbacks.append(warmup_lr)
            if args.cp:
                callbacks.append(checkpointer)
            if args.tb:
                callbacks.append(tensorboard)
            if args.save_weights:
                logger.info("Will save weights to: " + args.save_weights)
                callbacks.append(MultiGPUCheckpoint(args.save_weights))

            df_val = val_gen.get_response(copy=True)
            y_val = df_val[target].values
            y_shuf = np.random.permutation(y_val)
            log_evaluation(evaluate_prediction(y_val, y_shuf), logger, description='Between random pairs in y_val:')

            logger.info('Data points per epoch: train = %d, val = %d, test = %d', train_gen.size, val_gen.size, test_gen.size)
            logger.info('Steps per epoch: train = %d, val = %d, test = %d', train_gen.steps, val_gen.steps, test_gen.steps)

            #model.set_asynchronous_callbacks(asynchronous=True)
            
            # Adjust steps per execution
            # If user specified a negative number for steps per execution, it means entire epoch is run on IPU
            if ( args.steps_per_execution < 0 ):
                args.steps_per_execution = train_gen.steps

            model.compile(loss=args.loss, optimizer=optimizer, metrics=[candle.mae, candle.r2], steps_per_execution=args.steps_per_execution)

            history = model.fit( train_gen.tf_dataset,
                                 epochs=args.epochs,
                                 callbacks=callbacks,
                                 steps_per_epoch=(train_gen.steps//args.steps_per_execution)*args.steps_per_execution,
                                 validation_data=val_gen.tf_dataset,
                                 validation_steps=(val_gen.steps)
                               )

        candle.plot_metrics(history, title=None, skip_ep=0, outdir=os.path.dirname(args.save_path), add_lr=True)


    if K.backend() == 'tensorflow':
        K.clear_session()

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    """ 
    # Number of replicas
    num_ipus = params['num_replicas']
    num_io_tiles = params['num_io_tiles']

    # Standard IPU TensorFlow setup.
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus
    # Set number of tiles only for I/O
    if num_io_tiles > 0:
      ipu_config.io_tiles.num_io_tiles = num_io_tiles
      ipu_config.io_tiles.place_ops_on_io_tiles = True

    ipu_config.configure_ipu_system()


    # Create an execution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy(enable_dataset_iterators=False)
    """
    run(params, None)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
