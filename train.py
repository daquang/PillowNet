#!/usr/bin/env python
"""
Script for training model.
Use `train.py -h` to see an auto-generated description of advanced options.
"""

import argparse
import sys
import os
import errno
import shutil
import numpy as np
import random as rn
import tensorflow as tf
import keras
from tqdm import trange

from genomeloader.wrapper import TwoBitWrapper, FastaWrapper, BedWrapper, BigWigWrapper, BedGraphWrapper
from genomeloader.generator import MultiBedGenerator

from pillownet.model import unet, sliding, double_stranded, simple_window


def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input BED file(s) containing ground truth intervals', type=str)
    parser.add_argument('-ic', '--inputcross', required=False, nargs='*', default=None,
                        help='Input BED file(s) containing ground truth intervals for cross cell type validation. By '
                             'default, ground truth intervals in the reference cell type are used as a validation set.',
                        type=str)
    parser.add_argument('-bl', '--blacklist', required=False,
                        default='resources/blacklist.bed.gz',
                        help='Blacklist BED file.', type=str)
    parser.add_argument('-x', '--extra', required=False,
                        default=None,
                        help='BED file of extra regions to include in training.', type=str)
    parser.add_argument('-bw', '--bigwigs', type=str, required=False, nargs='*',
                        default=None,
                        help='Input bigwig files.')
    parser.add_argument('-bwl', '--bigwigslabel', type=str, required=False, nargs='*',
                        default=None,
                        help='Label bigwig files.')
    parser.add_argument('-bwc', '--bigwigscross', type=str, required=False, nargs='*',
                        default=None,
                        help='Input bigwig files for cross cell type validation.')
    parser.add_argument('-bwcl', '--bigwigscrosslabel', type=str, required=False, nargs='*',
                        default=None,
                        help='Label bigwig files for cross cell type validation.')
    parser.add_argument('-m', '--model', default='detection', choices=['sliding', 'detection', 'simple'],
                        help='Model type (default: detection)', type=str)
    parser.add_argument('-l', '--loss', default='bce_dice', choices=['bce', 'dice', 'focal', 'bce_dice', 'focal_dice',
                                                                     'tversky', 'jaccardlog', 'bce_jaccardlog',
                                                                     'bce_tversky',
                                                                     'mse_regression', 'msle_regression',
                                                                     'poisson_regression'],
                        help='Loss function (default: bce_dice)', type=str)
    parser.add_argument('-r', '--revcomp', action='store_true', default=False,
                        help='Consider both the given strand and the reverse complement strand (default: consider given'
                             ' strand only).')
    parser.add_argument('-w', '--window', type=int, required=False,
                        default=200,
                        help='Size of window in base pairs (default: 200). Only matters for sliding model.')
    parser.add_argument('-nr', '--negativesratio', type=int, required=False,
                        default=1,
                        help='Number of negative windows to sample per positive window (default: 1).')
    parser.add_argument('-v', '--validchroms', type=str, required=False, nargs='+',
                        default=['chr11'],
                        help='Chromosome(s) to set aside for validation (default: chr11).')
    parser.add_argument('-t', '--testchroms', type=str, required=False, nargs='+',
                        default=['chr1', 'chr8', 'chr21'],
                        help='Chromosome(s) to set aside for testing (default: chr1, chr8, chr21).')
    parser.add_argument('-L', '--seqlen', type=int, required=False,
                        default=None,
                        help='Length of sequence input (default: 1018 for sliding, 3084 for cropped detection, 1024 for'
                             ' non-cropped detection).')
    parser.add_argument('-f', '--filters', type=int, required=False,
                        default=32,
                        help='Number of filters in the first block (default: 32).')
    parser.add_argument('-k', '--kernelsize', type=int, required=False,
                        default=None,
                        help='Kernel size (default: 20 for sliding, 11 for detection.')
    parser.add_argument('-d', '--depth', type=int, required=False,
                        default=None,
                        help='Number of blocks (default: 1 for sliding, 5 for detection.')
    parser.add_argument('-sp', '--samepadding', action='store_true', default=False,
                        help='Use same padding (no cropping) in the detection model (default: valid padding).')
    parser.add_argument('-ns', '--noskip', action='store_true', default=False,
                        help='Do not use skip connections in the detection model (default: use skip connections).')
    parser.add_argument('-re', '--recurrent', action='store_true', default=False,
                        help='Add a recurrent layer (default: no recurrent layer).')
    parser.add_argument('-bs', '--batchsize',
                        help='Batch size (default: 128).',
                        type=int, default=128)
    parser.add_argument('-e', '--epochs',
                        help='Max number of epochs to train (default: 5).',
                        type=int, default=5)
    parser.add_argument('-er', '--epochsreset',
                        help='Number of epochs to reset negative sampling (default: 10).',
                        type=int, default=5)
    parser.add_argument('-s', '--seed',
                        help='Random seed for reproducibility (default: 1337).',
                        type=int, default=1337)
    parser.add_argument('-p', '--processes',
                        help='Number of parallel process workers (default: 3. If set to 0, then multiprocessing will '
                             'not be used).',
                        type=int, default=3)
    group = parser.add_mutually_exclusive_group(required=False)
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
    seq_len = args.seqlen
    window_len = args.window
    crop = not args.samepadding
    skip = not args.noskip
    recurrent = args.recurrent
    negatives_ratio = args.negativesratio
    filters = args.filters
    kernel_size = args.kernelsize
    depth = args.depth
    batch_size = args.batchsize
    epochs = args.epochs
    epochs_reset = args.epochsreset
    workers = args.processes
    seed = args.seed
    revcomp = args.revcomp
    model_type = args.model
    loss = args.loss
    valid_chroms = args.validchroms
    test_chroms = args.testchroms
    # Random seeds for reproducibility
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

    if workers > 0:
        use_multiprocessing = True
        thread_safe = True
    else:
        workers = 0
        use_multiprocessing = False
        thread_safe = False

    signals = []

    # Load genome
    if args.genometwobit is not None:
        genome = TwoBitWrapper(args.genometwobit, thread_safe=thread_safe)
        signals.append(genome)
    elif args.genomefasta is not None:
        genome = FastaWrapper(args.genomefasta, thread_safe=thread_safe)
        signals.append(genome)
    else:
        genome = None

    # Load input bigWigs
    bigwig_files = args.bigwigs
    bigwigs = [] if bigwig_files is None else [BigWigWrapper(bigwig_file, thread_safe=thread_safe) for bigwig_file in
                                               bigwig_files]
    signals.extend(bigwigs)

    # Check if at least one signal file is provided
    if len(signals) == 0:
        raise ValueError('You must provide at least one input signal.')

    # Load bed file
    bed_files = args.input
    beds = [BedWrapper(bed_file) for bed_file in bed_files]
    beds_train, beds_valid, beds_test = list(zip(*[bed.train_valid_test_split(valid_chroms=valid_chroms,
                                                                              test_chroms=test_chroms)
                                                   for bed in beds]))

    # Load blacklist file
    blacklist_file = args.blacklist
    blacklist = None if blacklist_file is None else BedWrapper(blacklist_file)
    if blacklist is None:
        blacklist_train, blacklist_valid, blacklist_test = None, None, None
    else:
        blacklist_train, blacklist_valid, blacklist_test = blacklist.train_valid_test_split(valid_chroms=valid_chroms,
                                                                                            test_chroms=test_chroms)

    # Load extra BED file
    extra_file = args.extra
    extra = None if extra_file is None else BedWrapper(extra_file)
    if extra is None:
        extra_train, extra_valid, extra_test = None, None, None
    else:
        extra_train, extra_valid, extra_test = extra.train_valid_test_split(valid_chroms=valid_chroms,
                                                                            test_chroms=test_chroms)

    # Load label bigWigs (if any)
    bigwiglabel_files = args.bigwigslabel
    output_signals = [] if bigwiglabel_files is None else [BigWigWrapper(bigwiglabel_file, thread_safe=thread_safe)
                                                           for bigwiglabel_file in bigwiglabel_files]

    # Make model
    if len(signals) == 1:
        if genome is not None:
            input_channel = 4
        else:
            input_channel = 1
    else:
        input_channel = (len(signals)) * [1]
        if genome is not None:
            input_channel[0] = 4
    output_channel = len(beds)
    if model_type == 'detection':
        if seq_len is None:
            seq_len = 3084 if crop else 1024
        if kernel_size is None:
            kernel_size = 11
        if depth is None:
            depth = 5
        model = unet(size=seq_len, input_channel=input_channel, output_channel=output_channel, skip=skip, crop=crop,
                     recurrent=recurrent, filters=filters, kernel_size=kernel_size, depth=depth, loss=loss)
        output_size = model.output_shape[1]
    elif model_type == 'sliding':
        if seq_len is None:
            seq_len = 1018
        if kernel_size is None:
            kernel_size = 20
        if depth is None:
            depth = 1
        model = sliding(size=seq_len, input_channel=input_channel, output_channel=output_channel, recurrent=recurrent,
                        filters=filters, kernel_size=kernel_size, depth=depth)
        output_size = None
    else:
        if seq_len is None:
            seq_len = 2048
        if kernel_size is None:
            kernel_size = 1001
        model = simple_window(size=seq_len, kernel_size=kernel_size, loss=loss)
        output_size = model.output_shape[1]


    # Make data generator
    return_sequences = args.model != 'sliding'
    jitter_mode = model_type
    generator_train = MultiBedGenerator(beds=beds_train, signals=signals, output_signals=output_signals,
                                        batch_size=batch_size, extra=extra_train, blacklist=blacklist, seq_len=seq_len,
                                        window_len=window_len, output_seq_len=output_size, epochs_reset=epochs_reset,
                                        negatives_ratio=negatives_ratio, jitter_mode=jitter_mode, shuffle=True,
                                        return_sequences=return_sequences)

    # If cross cell type BED files are included, make validation/test generators with cross cell type
    if args.inputcross is not None:
        cross_bed_files = args.input
        if len(bed_files) != len(cross_bed_files):
            raise ValueError('Different number of BED files for reference and cross cell types.')
        cross_beds = [BedWrapper(cross_bed_file) for cross_bed_file in cross_bed_files]
        cross_beds_train, cross_beds_valid, cross_beds_test = list(zip(*[cross_bed.train_valid_test_split(
                                                                         valid_chroms=valid_chroms,
                                                                         test_chroms=test_chroms)
                                                                         for cross_bed in cross_beds]))
        cross_bigwig_files = args.bigwigscross
        cross_bigwigs = [] if cross_bigwig_files is None else [BigWigWrapper(cross_bigwig_file, thread_safe=thread_safe)
                                                               for cross_bigwig_file in cross_bigwig_files]
        if len(bigwigs) != len(cross_bigwigs):
            raise ValueError('Different number of bigWig files for reference and cross cell types.')

        cross_signals = []
        if genome is not None:
            cross_signals.append(genome)
        cross_signals.extend(cross_bigwigs)

        # Load label bigWigs for cross cell type (if any)
        cross_bigwiglabel_files = args.bigwigscrosslabel
        cross_output_signals = [] if cross_bigwiglabel_files is None else [BigWigWrapper(cross_bigwiglabel_file,
                                                                                         thread_safe=thread_safe)
                                                                           for cross_bigwiglabel_file in
                                                                           cross_bigwiglabel_files]

        generator_valid = MultiBedGenerator(beds=cross_beds_valid, signals=cross_signals, batch_size=batch_size,
                                            output_signals=cross_output_signals, extra=extra_valid,
                                            blacklist=blacklist, seq_len=seq_len, window_len=window_len,
                                            output_seq_len=output_size, negatives_ratio=negatives_ratio,
                                            jitter_mode=None, return_sequences=return_sequences, shuffle=False)
        generator_test = MultiBedGenerator(beds=cross_beds_test, signals=cross_signals, batch_size=batch_size,
                                           output_signals=cross_output_signals, extra=extra_test,
                                           blacklist=blacklist, seq_len=seq_len, window_len=window_len,
                                           output_seq_len=output_size, negatives_ratio=negatives_ratio,
                                           jitter_mode=None, return_sequences=return_sequences, shuffle=False)
    else:  # Else make validation/test generators with the reference cell type
        generator_valid = MultiBedGenerator(beds=beds_valid, signals=signals, output_signals=output_signals,
                                            extra=extra_valid, blacklist=blacklist, seq_len=seq_len,
                                            window_len=window_len, output_seq_len=output_size, batch_size=batch_size,
                                            negatives_ratio=negatives_ratio, jitter_mode=None,
                                            return_sequences=return_sequences, shuffle=False)
        generator_test = MultiBedGenerator(beds=beds_test, signals=signals, output_signals=output_signals,
                                           extra=extra_test, blacklist=blacklist, seq_len=seq_len,
                                           window_len=window_len, output_seq_len=output_size, batch_size=batch_size,
                                           negatives_ratio=negatives_ratio, jitter_mode=None,
                                           return_sequences=return_sequences, shuffle=False)

    if 'regression' not in loss:  # housekeeping stuff for binary output
        # Compute percentage of zero labels in validation set
        total_valid_labels = 0
        total_valid_nonzero_labels = np.zeros(len(beds))
        axis = (0, 1) if return_sequences else 0
        for i in trange(len(generator_valid)):
            _, y = generator_valid[i]
            total_valid_labels += y.size / len(beds)
            total_valid_nonzero_labels += np.count_nonzero(y, axis=axis)
        fraction_valid_zeros = 1 - total_valid_nonzero_labels / total_valid_labels
        print('Overall fraction of valid zero labels: %.4f' % fraction_valid_zeros.mean())
        for i in range(len(beds)):
            print('Fraction of valid zero labels for BED %i: %.4f' % (i, fraction_valid_zeros[i]))
        chroms_train = beds_train[0].chroms()
        chroms_size = signals[0].chroms_size()
        # Total number of nucleotides in all training chromosomes
        total_chroms_train_size = sum([chroms_size[c] for c in chroms_train])
        if blacklist is not None:
            total_chroms_train_size -= blacklist_train.sum_intervals()
        # Total number of positive labels in all training chromosomes
        total_positives_train = np.array([bed_train.sum_intervals() for bed_train in beds_train])
        new_biases = np.log(total_positives_train) - np.log(total_chroms_train_size - total_positives_train)
        # new_biases = np.log((1 - fraction_valid_zeros) / fraction_valid_zeros)
        # new_biases = np.zeros(len(total_positives_train))
        # Set bias in final layer based on fraction of zero labels in validation set
        final_layer = model.layers[-1]
        final_layer_weights = final_layer.get_weights()
        final_layer_weights[1] = new_biases  # np.log((1 - fraction_valid_zeros) / fraction_valid_zeros)
        final_layer.set_weights(final_layer_weights)
    """elif len(signals) == 1 and len(output_signals) == 1:
        # Compute initial MSE between input and output bigWigs
        total_valid_labels = 0
        se = 0
        for i in trange(len(generator_valid)):
            x, y = generator_valid[i]
            total_valid_labels += y.size
            if x.size > y.size:
                y_seq_size = y.shape[1]
                x_seq_size = x.shape[1]
                x_seq_start = int(x_seq_size / 2 - y_seq_size / 2)
                x_seq_stop = x_seq_start + y_seq_size
                x = x[:, x_seq_start:x_seq_stop, :]
            se += np.square(x - y).sum()
        mse = se / total_valid_labels
        print('Benchmark MSE on validation data: %.4f' % mse)"""

    model_single = model
    if revcomp:
        model = double_stranded(model_single)
    model.summary()

    if args.outputdir is None:
        overwrite = True
        output_dir = args.outputdirc
    else:
        overwrite = False
        output_dir = args.outputdir

    try:
        os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            if not overwrite:
                print(('Output directory (%s) already exists '
                       'but you specified not to overwrite it') % output_dir)
                sys.exit(1)
            else:
                print(('Output directory (%s) already exists '
                       'so it will be overwritten') % output_dir)
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=output_dir,
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(os.path.join(output_dir+'/', "best_weights.h5"),
                                        verbose=1, save_best_only=True, monitor='val_loss', save_weights_only=False)
    ]

    model.fit_generator(generator=generator_train,
                        validation_data=generator_valid, epochs=epochs,
                        callbacks=callbacks, use_multiprocessing=use_multiprocessing, workers=workers)

    model.save(output_dir + '/final_weights.h5', include_optimizer=False)


if __name__ == '__main__':
    main()
