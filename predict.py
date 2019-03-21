#!/usr/bin/env python
"""
Script for generating predictions from a trained model.
Use `predict.py -h` to see an auto-generated description of advanced options.
"""

import argparse
import numpy as np

from keras.models import load_model
from tqdm import tqdm, trange
import pybedtools as pbt
import pyBigWig as pbw

from genomeloader.wrapper import TwoBitWrapper, FastaWrapper, BedWrapper, BigWigWrapper, BedGraphWrapper
from genomeloader.generator import MultiBedGenerator

from pillownet.layer import ReverseComplement, Reverse, Motifs


def get_args():
    parser = argparse.ArgumentParser(description='Generating predictions.',
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-w', '--weights', required=True,
                        help='Input model weights.', type=str)
    parser.add_argument('-o', '--output', required=True,
                        help='Output bigWig file of predictions.', type=str)
    parser.add_argument('-bl', '--blacklist', required=False,
                        default='resources/blacklist.bed.gz',
                        help='Blacklist BED file.', type=str)
    parser.add_argument('-bw', '--bigwigs', type=str, required=False, nargs='*',
                        default=None,
                        help='Input bigwig files.')
    parser.add_argument('-s', '--step',
                        help='Step size and window size to make predictions for.',
                        type=int, default=50)
    parser.add_argument('-t', '--threshold',
                        help='Remove all signal values below threshold (default: 1e-2).',
                        type=float, default=1e-2)
    parser.add_argument('-at', '--autothreshold', action='store_true', default=False,
                        help='Automatically set threshold.')
    parser.add_argument('-p', '--processes',
                        help='Number of parallel process workers (default: 3. If set to 0, then multiprocessing will '
                             'not be used).',
                        type=int, default=3)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-c', '--chroms', type=str, nargs='+',
                       default=['chr1', 'chr8', 'chr21'],
                       help='Chromosome(s) to make predictions for.')
    group.add_argument('-wg', '--wholegenome', action='store_true', default=False,
                       help='Make predictions for the whole genome.')
    group.add_argument('-ax', '--autox', action='store_true', default=False,
                       help='Predict on autosomes and X chromosome.')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-gf', '--genomefasta', type=str,
                       help='Genome FASTA file.')
    group.add_argument('-gt', '--genometwobit', type=str,
                       help='Genome twobit file.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    weights_file = args.weights
    output_file = args.output
    step = args.step
    workers = args.processes
    threshold = args.threshold
    autothreshold = args.autothreshold

    bw = pbw.open(output_file, 'w')

    if workers > 0:
        use_multiprocessing = True
        thread_safe = True
    else:
        workers = 0
        use_multiprocessing = False
        thread_safe = False

    signals = []

    if args.genometwobit is not None:
        genome = TwoBitWrapper(args.genometwobit, thread_safe=thread_safe)
        signals.append(genome)
    elif args.genomefasta is not None:
        genome = FastaWrapper(args.genomefasta, thread_safe=thread_safe)
        signals.append(genome)
    else:
        genome = None

    # Load bigwigs
    bigwig_files = args.bigwigs
    bigwigs = [] if bigwig_files is None else [BigWigWrapper(bigwig_file, thread_safe=thread_safe) for bigwig_file in
                                               bigwig_files]
    signals.extend(bigwigs)

    chroms_size = signals[0].chroms_size()

    # Load blacklist file
    blacklist_file = args.blacklist
    blacklist = None if blacklist_file is None else BedWrapper(blacklist_file)

    if args.wholegenome:
        chroms = signals[0].chroms()
    elif args.autox:
        chroms = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                  'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX']
    else:
        chroms = args.chroms

    header = []
    for chrom in chroms:
        chrom_size = chroms_size[chrom]
        header.append((chrom, chrom_size))
    bw.addHeader(header)

    model = load_model(weights_file, custom_objects={'ReverseComplement': ReverseComplement,
                                                     'Reverse': Reverse,
                                                     'Motifs': Motifs}, compile=False)

    if autothreshold:
        input_shape = np.array(model.input_shape)
        input_shape[0] = 1
        input_zeros = np.zeros(input_shape)
        output_zeros = model.predict(input_zeros)
        threshold = output_zeros.max() * 1.01
        print('The new threshold is: %f' % threshold)

    return_sequences = len(model.output_shape) == 3
    multi_task = model.output_shape[-1] > 1

    # multi_task still not supported
    if multi_task:
        print('Multi-task not supported yet')
        return

    dna_input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    seq_len = dna_input_shape[1]
    output_seq_len = None

    if return_sequences:
        step = model.output_shape[1]
        output_seq_len = model.output_shape[1]

    pbar = tqdm(chroms)
    for chrom in pbar:
        pbar.set_description('Processing %s' % chrom)
        chrom_size = chroms_size[chrom]
        chrom_windows_bt = pbt.BedTool().window_maker(genome={chrom: (0, chrom_size)}, w=step, s=step)
        chrom_windows = BedWrapper(chrom_windows_bt.fn, sort_bed=False)
        generator = MultiBedGenerator(beds=[chrom_windows], signals=signals, seq_len=seq_len,
                                      output_seq_len=output_seq_len, negatives_ratio=0, jitter_mode=None,
                                      shuffle=False, return_sequences=return_sequences, return_output=False,
                                      left_justify=True)
        chrom_start = 0
        for i in trange(len(generator)):
            batch = generator[i]
            #chrom_end = chrom_start + step * len(batch)
            predictions_batch = model.predict(batch)
            values = predictions_batch.ravel()
            chrom_end = chrom_start + len(values)
            starts = np.arange(chrom_start, chrom_end)
            if blacklist is not None:
                values_blacklist = blacklist[chrom, chrom_start:chrom_end].ravel()
                values[values_blacklist] = 0
            if chrom_end > chrom_size:
                crop_size = chrom_size - chrom_end
                values = values[:crop_size]
                starts = starts[:crop_size]
            chrom_start = chrom_end
            above_threshold = values >= threshold
            values = values[above_threshold]
            if len(values) == 0:
                continue
            starts = starts[above_threshold]
            bw.addEntries(chroms=chrom, starts=starts, span=1, values=values)

    bw.close()


if __name__ == '__main__':
    main()
