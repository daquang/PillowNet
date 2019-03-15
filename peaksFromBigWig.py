#!/usr/bin/env python
"""
Script for converting bigWigs of predictions to peaks.
Use `peaksFromBigWig.py -h` to see an auto-generated description of advanced options.
"""

import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from genomeloader.wrapper import BedWrapper, BigWigWrapper


def get_args():
    parser = argparse.ArgumentParser(description="Evaluating predictions.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', '--predictions', required=True,
                        help='BigWig of predictions.', type=str)
    parser.add_argument('-o', '--output', required=True,
                        help='Output BED of intervals.', type=str)
    parser.add_argument('-l', '--minlength', required=False, default=147,
                        help='Minimum size of peaks (147 bps).', type=int)
    parser.add_argument('-t', '--threshold', required=False, default=0.5,
                        help='Minimum probability to call peaks (default: 0.5).', type=float)
    parser.add_argument('-bl', '--blacklist', required=False,
                        default=None,
                        help='Blacklist BED file.', type=str)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-c', '--chroms', type=str, nargs='+',
                        default=['chr1', 'chr8', 'chr21'],
                        help='Chromosome(s) to generate peaks on.')
    group.add_argument('-wg', '--wholegenome', action='store_true', default=False,
                        help='Generate peaks the whole genome.')
    group.add_argument('-ax', '--autox', action='store_true', default=False,
                       help='Generate peaks on autosomes and X chromosome.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    bigwig_file = args.predictions
    threshold = args.threshold
    min_length = args.minlength
    output_file = args.output

    # Load blacklist file
    blacklist_file = args.blacklist
    blacklist = None if blacklist_file is None else BedWrapper(blacklist_file)

    # Load bigwig of predictions
    bw = BigWigWrapper(bigwig_file)

    if args.wholegenome:
        chroms = bw.chroms()
    elif args.autox:
        chroms = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                  'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX']
    else:
        chroms = args.chroms

    peaks_chrom = []
    peaks_start = []
    peaks_end = []
    peaks_value = []
    pbar = tqdm(chroms)
    for chrom in pbar:
        pbar.set_description('Processing %s' % chrom)
        chrom_values = np.ravel(bw[chrom])
        chrom_bools = chrom_values > threshold
        chrom_bools[0] = False
        chrom_bools[-1] = False
        if blacklist is not None:
            values_blacklist = blacklist[chrom, 0:len(chrom_bools)].ravel()
            chrom_values[values_blacklist] = 0
            chrom_bools[values_blacklist] = False
        starts_ends = np.where(chrom_bools[:-1] != chrom_bools[1:])[0] + 1
        starts_ends = starts_ends.reshape((int(starts_ends.size / 2), 2))
        lens = starts_ends[:, 1] - starts_ends[:, 0]
        starts_ends = starts_ends[lens >= min_length]
        peaks_chrom.extend(len(starts_ends) * [chrom])
        peaks_start.extend(starts_ends[:, 0])
        peaks_end.extend(starts_ends[:, 1])
        peaks_value.extend([np.mean(chrom_values[se[0]:se[1]]) for se in starts_ends])

    bw.close()
    peaks_df = pd.DataFrame()
    peaks_df['chrom'] = peaks_chrom
    peaks_df['start'] = peaks_start
    peaks_df['end'] = peaks_end
    peaks_df['value'] = peaks_value
    peaks_df.to_csv(output_file, sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()
