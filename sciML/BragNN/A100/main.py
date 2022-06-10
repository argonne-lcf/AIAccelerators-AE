import pickle
pickle.DEFAULT_PROTOCOL=4

import argparse, os, time, sys, shutil, logging
from util import str2bool, str2tuple, s2ituple
from execute import execute

def parse_args():
    parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
    parser.add_argument('-device', type=str, default="ipu", help='Run on "gpu" or "ipu"', choices=['gpu', 'ipu'])
    parser.add_argument('-expName',type=str, default="run_log", help='Experiment name')
    parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
    parser.add_argument('-mbsz',   type=int, default=512, help='mini batch size')
    parser.add_argument('-maxep',  type=int, default=500, help='max training epoches')
    parser.add_argument('-fcsz',   type=s2ituple, default='16_8_4_2', help='size of dense layers')
    parser.add_argument('-psz',    type=int, default=11, help='working patch size')
    parser.add_argument('-aug',    type=int, default=1, help='augmentation size')
    parser.add_argument('-dataset',           help='path to the dataset directory (default: ./dataset)', type =str, default='./dataset' )
    # IPU specific operations
    parser.add_argument('-num-threads',       help='Number of worker threads to be used in data loader', type=int, default=8 )
    parser.add_argument('-device-iter',       help='number of mini batches processed on IPU per execution step' ,type=int, default=10)
    parser.add_argument('-num-train-replica', help='Number of IPUs used for training in data-parallel mode', type=int, default=1 )
    parser.add_argument('-num-infer-replica', help='Number of IPUs used for validation/inference in data-parallel mode', type=int, default=1 )
    parser.add_argument('-no-cache',          help='disable model caching (default: False). Compilation time will increase by disabling caching', action="store_true" )
    parser.add_argument('-benchmark',         help='Benchmark the dataloader and IPU model for epochs specified by -maxep argument', action="store_true" )
    parser.add_argument('-profile',           help='profile the IPU run (default: False). Loading a cached model with profiling on might result in a compilation error', action="store_true" )

    args, unparsed = parser.parse_known_args()

    if (args.profile):
        print("Model caching is disabled when profile mode is on")
        args.no_cache = True

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    return args


if __name__ == "__main__":
    args = parse_args()
    #set debugging flags
    if args.profile:
        print("Run with profiling on")
        os.environ['POPLAR_ENGINE_OPTIONS'] = '{"autoReport.all":"true", "autoReport.directory":"./profile"}'

    execute(args)
