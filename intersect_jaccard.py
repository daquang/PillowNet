#!/usr/bin/env python
"""
intersectbed-like tool that uses Jaccard index (IOU) instead of fraction overlap.
Use `intersect_jaccard.py -h` to see an auto-generated description of advanced options.
"""

import argparse
from genomeloader.wrapper import BedWrapper


def get_args():
    parser = argparse.ArgumentParser(description='Report A intervals that meet desired IOU threshold overlaps with B.',
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-a', '--a', required=True,
                        help='A BED.', type=str)
    parser.add_argument('-b', '--b', required=True,
                        help='B BED.', type=str)
    parser.add_argument('-t', '--threshold', required=False, default=0.5,
                        help='Jaccard/IOU threshold (default: 0.5).', type=float)
    parser.add_argument('-v', '--inverse', action='store_true', default=False,
                        help='Only report those entries in A that have no overlap in B.')
    parser.add_argument('-bl', '--blacklist', required=False,
                        default=None,
                        help='Blacklist BED file.', type=str)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-c', '--chroms', type=str, nargs='+',
                        default=['chr1', 'chr8', 'chr21'],
                        help='Chromosome(s) to evaluate on.')
    group.add_argument('-wg', '--wholegenome', action='store_true', default=False,
                        help='Evaluate on the whole genome.')
    group.add_argument('-ax', '--autox', action='store_true', default=False,
                       help='Evaluate on autosomes and X chromosome.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    inverse = args.inverse
    a = BedWrapper(args.a)
    b = BedWrapper(args.b)
    iou_threshold = args.threshold

    # Load blacklist file
    blacklist_file = args.blacklist
    blacklist = None if blacklist_file is None else BedWrapper(blacklist_file)
    if blacklist is not None: # clip away parts of BED files that overlap blacklist intervals
        new_b_bt = b.bt.subtract(blacklist.bt)
        b = BedWrapper(new_b_bt.fn)
        new_a_bt = a.bt.subtract(blacklist.bt)
        a = BedWrapper(new_a_bt.fn)

    if not args.wholegenome:
        if args.autox:
            chroms = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                      'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX']
        else:
            chroms = args.chroms
        _, _, a = a.train_valid_test_split(valid_chroms=[], test_chroms=chroms)
        _, _, b = b.train_valid_test_split(valid_chroms=[], test_chroms=chroms)

    for row, bt_row in zip(a.df.itertuples(), a.bt):
        chrom = getattr(row, 'chrom')
        start = getattr(row, 'chromStart')
        end = getattr(row, 'chromEnd')
        potential_gt_intervals = b.search(chrom, start, end)
        overlaps_positive = False
        for potential_gt_interval in potential_gt_intervals:
            row_iou = iou(start, end, potential_gt_interval.begin, potential_gt_interval.end)
            if row_iou >= iou_threshold:
                overlaps_positive = True
                break
        if overlaps_positive ^ inverse:
            print(bt_row, end='')


def iou(a_start, a_end, b_start, b_end):
    if a_start > b_start:
        a_start, a_end, b_start, b_end = b_start, b_end, a_start, a_end
    if a_end < b_start:
        return 0
    intersection = a_end - b_start
    union = b_end - a_start
    return abs(1.0 * intersection / union)


if __name__ == '__main__':
    main()
