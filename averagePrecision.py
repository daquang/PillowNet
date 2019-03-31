#!/usr/bin/env python
"""
Script for computing average precision.
Use `averagePrecision.py -h` to see an auto-generated description of advanced options.
"""

import argparse
import os
import shutil
import errno
from tqdm import tqdm
import numpy as np
from genomeloader.wrapper import BedWrapper, BedGraphWrapper, NarrowPeakWrapper, BroadPeakWrapper
from sklearn.metrics import auc
import matplotlib
matplotlib.use('Agg')
from matplotlib import style
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Evaluating predictions.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-abg', '--abedgraph',
                        help='BEDGraph of predicted intervals.', type=str)
    group.add_argument('-anp', '--anarrowpeak',
                       help='NarrowPeak of predicted intervals.', type=str)
    group.add_argument('-abp', '--abroadpeak',
                       help='NarrowPeak of predicted intervals.', type=str)
    parser.add_argument('-b', '--b', required=True,
                        help='BED of ground truth intervals.', type=str)
    parser.add_argument('-i', '--signalcolumn', required=False,
                        help='Column index in prediction BED(s) containing signal value.', type=int)
    parser.add_argument('-t', '--threshold', required=False, default=0.5,
                        help='IOU threshold (default: 0.5).', type=float)
    parser.add_argument('-o', '--output', required=False, default=None,
                        help='Output directory (optional).', type=str)
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
    output = args.output
    b = BedWrapper(args.b)
    if args.abedgraph is not None:
        a = BedGraphWrapper(args.abedgraph)
    elif args.anarrowpeak is not None:
        a = NarrowPeakWrapper(args.anarrowpeak)
    else:
        a = BroadPeakWrapper(args.abroadpeak)
    data_col = a.col_names[a.data_col - 1]
    iou_threshold = args.threshold

    if not args.wholegenome:
        if args.autox:
            chroms = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                      'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX']
        else:
            chroms = args.chroms
        _, _, a = a.train_valid_test_split(valid_chroms=[], test_chroms=chroms)
        _, _, b = b.train_valid_test_split(valid_chroms=[], test_chroms=chroms)

    num_gt_peaks = len(b)
    true_positives_detected = 0
    false_positives_detected = 0
    #gt_genomic_interval_tree = b.genomic_interval_tree
    num_pr_peaks = len(a)
    predictions_df = a.df
    predictions_df.sort_values(by=data_col, ascending=False, inplace=True)

    thresholds = []
    recalls = []
    precisions = []
    labels = []

    pbar = tqdm(iterable=predictions_df.itertuples(), total=num_pr_peaks)
    for row in pbar:
        chrom = getattr(row, 'chrom')
        start = getattr(row, 'chromStart')
        end = getattr(row, 'chromEnd')
        value = getattr(row, data_col)
        thresholds.append(value)
        #chrom_gt_tree = gt_genomic_interval_tree[chrom]
        #potential_gt_intervals = chrom_gt_tree.overlap(start, end)
        potential_gt_intervals = b.search(chrom, start, end)
        overlaps_positive = False
        for potential_gt_interval in potential_gt_intervals:
            row_iou = iou(start, end, potential_gt_interval.begin, potential_gt_interval.end)
            if row_iou >= iou_threshold:
                #chrom_gt_tree.remove(potential_gt_interval)
                true_positives_detected += 1
                overlaps_positive = True
                break
        if not overlaps_positive:
            false_positives_detected += 1
        labels.append(overlaps_positive)
        recalls.append(1.0 * true_positives_detected / num_gt_peaks)
        precisions.append(1.0 * true_positives_detected / (true_positives_detected + false_positives_detected))

    thresholds = np.array(thresholds)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    thresholds, unique_indices, unique_counts = np.unique(thresholds, return_index=True, return_counts=True)
    unique_indices = unique_indices + unique_counts - 1
    recalls = recalls[unique_indices]
    precisions = precisions[unique_indices]
    ap = auc(recalls, precisions)
    print('The average precision is %f' % ap)
    jaccard = a.bt.jaccard(b.bt)['jaccard']
    print('The Jaccard index is %f' % jaccard)
    if output is not None:
        plt.ioff()
        style.use('ggplot')
        try:
            os.makedirs(output)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                    shutil.rmtree(output)
                    os.makedirs(output)
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(output + '/pr.pdf')
        np.save(output + '/recalls.npy', recalls)
        np.save(output + '/precisions.npy', precisions)


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
