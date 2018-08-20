#!/usr/bin/env python
"""
Script for training model.
Use `train.py -h` to see an auto-generated description of advanced options.
"""

import argparse
import sys, os
import numpy as np
import random as rn
import tensorflow as tf
import keras
from keras import backend as K


from genomeloader.wrapper import TwoBitWrapper, FastaWrapper, BedWrapper
from genomeloader.generator import BedGenerator

from pillownet.model import unet, sliding, double_stranded


def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Input BED file containing ground truth intervals', type=str)
    parser.add_argument('-m', '--model', default='sliding', choices=['sliding', 'detection'],
                        help='Model type (default: sliding)', type=str)
    parser.add_argument('-r', '--revcomp', action='store_true', default=False,
                        help='Consider both the given strand and the reverse complement strand when searching for '
                             'motifs in a complementable alphabet (default: consider given strand only).')
    parser.add_argument('--window', '-w', type=int, required=False,
                        default=200,
                        help='Size of window in base pairs (default: 200).')
    parser.add_argument('--validchroms', '-v', type=str, required=False, nargs='+',
                        default=['chr11'],
                        help='Chromosome(s) to set aside for validation (default: chr11).')
    parser.add_argument('--testchroms', '-t', type=str, required=False, nargs='+',
                        default=['chr1', 'chr8', 'chr21'],
                        help='Chromosome(s) to set aside for testing (default: chr1, chr8, chr21).')
    parser.add_argument('--seqlen', '-L', type=int, required=False,
                        default=1000,
                        help='Length of sequence input (default: 1000).')
    parser.add_argument('-e', '--epochs',
                        help='Max number of epochs to train (default: 5).',
                        type=int, default=5)
    parser.add_argument('-s', '--seed',
                        help='Random seed for reproducibility (default: 1337).',
                        type=int, default=1337)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gf', '--genomefasta', type=str,
                       help='Genome FASTA file.')
    group.add_argument('-gt', '--genometwobit', type=str,
                       help='Genome twobit file.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--outputdir', type=str,
                       help='The output directory. Causes error if the directory already exists.')
    group.add_argument('-oc', '--outputdirc', type=str,
                       help='The output directory. Will overwrite if directory already exists.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    epochs = args.epochs
    seed = args.seed
    revcomp = args.revcomp
    model_type = args.model
    valid_chroms = args.validchroms
    test_chroms = args.testchroms
    config = tf.ConfigProto()
    gpu_list = K.tensorflow_backend._get_available_gpus()
    if len(gpu_list) == 0: # CPU mode
        use_multiprocessing = False
        workers = 1
    else:
        use_multiprocessing = True
        workers = 3

    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.gpu_options.visible_device_list = "0"
    # set_session(tf.Session(config=config))
    # Random seeds for reproducibility
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

    # Load genome
    if args.genomefasta is None:
        genome = TwoBitWrapper(args.genometwobit)
    else:
        genome = FastaWrapper(args.genomefasta)

    # Load bed file
    bed_file = args.input
    bed = BedWrapper(bed_file)
    bed_train, bed_valid, bed_test = bed.train_valid_test_split(valid_chroms=valid_chroms, test_chroms=test_chroms)

    # Make data generator
    seq_len = args.seqlen
    return_sequences = args.model == 'detection'
    generator_train = BedGenerator(bed_train, genome, seq_len=seq_len, negatives_ratio=1,
                                   return_sequences=return_sequences)
    generator_valid = BedGenerator(bed_valid, genome, seq_len=seq_len, negatives_ratio=1,
                                   return_sequences=return_sequences, shuffle=False, jitter_mode=None)
    generator_test = BedGenerator(bed_train, genome, seq_len=seq_len, negatives_ratio=1,
                                  return_sequences=return_sequences, shuffle=False, jitter_mode=None)

    if args.outputdir is None:
        overwrite = True
        output_dir = args.outputdirc
    else:
        overwrite = False
        output_dir = args.outputdir
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=output_dir,
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(os.path.join(output_dir+'/', "weights.h5"),
                                        verbose=0, save_weights_only=False, monitor='val_loss')
    ]

    if model_type == 'detection':
        model = unet(size=seq_len)
    else:
        model = sliding(size=seq_len)
    model.summary()
    if revcomp:
        model_single = model
        model = double_stranded(model_single)

    model.fit_generator(generator=generator_train, validation_data=None, epochs=epochs, callbacks=callbacks,
                        use_multiprocessing=True, workers=3)


if __name__ == '__main__':
    main()
