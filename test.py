#!/usr/bin/env python
"""
Script for evaluating predictions.
Use `test.py -h` to see an auto-generated description of advanced options.
"""

import argparse
from sklearn.metrics import roc_curve, precision_recall_curve, auc, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
from tqdm import tqdm, trange

from genomeloader.wrapper import BedWrapper, BigWigWrapper


def get_args():
    parser = argparse.ArgumentParser(description="Evaluating predictions.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', '--predictions', required=True,
                        help='BigWig of predictions.', type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--labels', required=False,
                        help='BigWig of ground truth labels.', type=str)
    group.add_argument('-b', '--bed', required=False,
                        help='BED of ground truth intervals.', type=str)
    parser.add_argument('-t', '--testbed', required=False,
                        help='BED of intervals to perform evaluation on.', type=str)
    parser.add_argument('-bl', '--blacklist', required=False,
                        default=None,
                        help='Blacklist BED file.', type=str)
    parser.add_argument('-ac', '--aggregatechromosomes', action='store_true', default=False,
                        help='If no test BED provided, evaluate as an aggregate across all test chromosomes. Will '
                             'consume more memory (default: evaluate at a per-chromosome level).')
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
    bigwig_file = args.predictions
    labels_bigwig_file = args.labels
    bed_file = args.bed
    aggregate = args.aggregatechromosomes

    if args.labels is None and args.bed is None:
        raise ValueError('You must supply ground truth BED or bigWig file')

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

    test_over_intervals = False
    if args.testbed is not None:
        bed_test = BedWrapper(args.testbed)
        _, _, bed_test = bed_test.train_valid_test_split(valid_chroms=[], test_chroms=chroms)
        test_over_intervals = True

    if labels_bigwig_file is not None:
        # Load bigWig of ground truth labels
        labels_bw = BigWigWrapper(labels_bigwig_file)
        if test_over_intervals:
            test_regression_over_intervals(bed_test, labels_bw, bw, blacklist)
        else:
            test_regression(chroms, labels_bw, bw, blacklist, aggregate)
    else:
        # Load BED file of ground truth intervals
        bed = BedWrapper(bed_file)
        if test_over_intervals:
            test_classification_over_intervals(bed_test, bed, bw, blacklist)
        else:
            test_classification(chroms, bed, bw, blacklist, aggregate)


def test_regression_over_intervals(bed_test, labels_bw, bw, blacklist):
    y_true = []
    y_pred = []
    pbar = trange(len(bed_test))
    for i in pbar:
        interval = bed_test.df.iloc[i]
        chrom = interval.chrom
        chromStart = interval.chromStart
        chromEnd = interval.chromEnd
        predictions = bw[chrom, chromStart:chromEnd]
        labels = labels_bw[chrom, chromStart:chromEnd]
        if blacklist is not None:
            values_blacklist = ~ blacklist[chrom, chromStart:chromEnd]
            predictions = predictions[values_blacklist]
            labels = labels[values_blacklist]
        y_true.append(labels)
        y_pred.append(predictions)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    pearson, pearson_p = pearsonr(y_pred, y_true)
    spearman, spearman_p = spearmanr(y_pred, y_true)
    print('MSE:', mse)
    print('Pearson R:', pearson)
    print('Spearman R:', spearman)


def test_regression(chroms, labels_bw, bw, blacklist, aggregate):
    chroms_size = bw.chroms_size()

    mses = []
    pearsons = []
    spearmans = []

    y_true = []
    y_pred = []

    pbar = tqdm(chroms)
    for chrom in pbar:
        pbar.set_description('Processing %s' % chrom)
        chrom_size = chroms_size[chrom]
        chrom_predictions = bw[chrom]
        chrom_labels = labels_bw[chrom, 0:chrom_size]
        if blacklist is not None:
            chrom_blacklist = ~ blacklist[chrom, 0:chrom_size]
            chrom_predictions = chrom_predictions[chrom_blacklist]
            chrom_labels = chrom_labels[chrom_blacklist]
        mse = mean_squared_error(chrom_labels, chrom_predictions)
        pearson, pearson_p = pearsonr(chrom_predictions, chrom_labels)
        spearman, spearman_p = spearmanr(chrom_predictions, chrom_labels)
        mses.append(mse)
        pearsons.append(pearson)
        spearmans.append(spearman)
        if aggregate:
            y_true.append(chrom_labels)
            y_pred.append(chrom_predictions)
    if aggregate:
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        mse_mean = mean_squared_error(y_true, y_pred)
        pearson_mean, pearson_p_mean = pearsonr(y_pred, y_true)
        spearman_mean, spearman_p_mean = spearmanr(y_pred, y_true)
    else:
        mse_mean = np.mean(mses)
        pearson_mean = np.mean(pearsons)
        spearman_mean = np.mean(spearmans)
    print('Chromosomes:', chroms)
    print('MSEs:', mses)
    print('MSE (chromosome average):', mse_mean)
    print('Pearson Rs:', pearsons)
    print('Pearson R (chromosome average):', pearson_mean)
    print('Spearman Rs:', spearmans)
    print('Spearman R (chromosome average):', spearman_mean)


def dice_coef(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)
    denom = np.sum(y_true + y_pred)
    return np.mean(2. * intersect / denom)


def test_classification_over_intervals(bed_test, bed, bw, blacklist):
    y_true = []
    y_pred = []

    pbar = trange(len(bed_test))
    for i in pbar:
        interval = bed_test.df.iloc[i]
        chrom = interval.chrom
        chromStart = interval.chromStart
        chromEnd = interval.chromEnd
        predictions = bw[chrom, chromStart:chromEnd]
        labels = bed[chrom, chromStart:chromEnd]
        if blacklist is not None:
            values_blacklist = ~ blacklist[chrom, chromStart:chromEnd]
            predictions = predictions[values_blacklist]
            labels = labels[values_blacklist]
        y_true.append(labels)
        y_pred.append(predictions)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    frac = 1.0 * y_true.sum() / len(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    dice = dice_coef(y_true, y_pred)

    bw.close()
    jaccard = dice(2 - dice)
    print('Positive fraction:', frac)
    print('Dice coefficient:', dice)
    print('Jaccard index:', jaccard)
    print('auROC:', auroc)
    print('auPR:', aupr)
    """
    pylab.subplot(121)
    pylab.plot(fpr, tpr, label=chrom + ' (auROC=%0.2f)' % auroc)
    pylab.plot([0, 1], [0, 1], 'k--', label='Random')
    pylab.legend(loc='lower right')
    pylab.xlabel('FPR')
    pylab.ylabel('TPR')

    pylab.subplot(122)
    pylab.plot(recall, precision, label=chrom + ' (auPR=%0.2f)' % aupr)
    pylab.legend(loc='upper right')
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.show()
    """


def test_classification(chroms, bed, bw, blacklist, aggregate):
    chroms_size = bw.chroms_size()

    fracs = []
    aurocs = []
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    auprs = []
    dices = []

    y_true = []
    y_pred = []

    pbar = tqdm(chroms)
    for chrom in pbar:
        pbar.set_description('Processing %s' % chrom)
        chrom_size = chroms_size[chrom]
        chrom_predictions = bw[chrom]
        chrom_labels = bed[chrom, 0:chrom_size]
        if blacklist is not None:
            chrom_blacklist = ~ blacklist[chrom, 0:chrom_size]
            chrom_predictions = chrom_predictions[chrom_blacklist]
            chrom_labels = chrom_labels[chrom_blacklist]
        frac = 1.0 * chrom_labels.sum() / len(chrom_labels)
        fracs.append(frac)
        fpr, tpr, _ = roc_curve(chrom_labels, chrom_predictions)
        auroc = auc(fpr, tpr)
        aurocs.append(auroc)
        fprs.append(fpr)
        tprs.append(tpr)
        precision, recall, _ = precision_recall_curve(chrom_labels, chrom_predictions)
        precisions.append(precision)
        recalls.append(recall)
        aupr = auc(recall, precision)
        auprs.append(aupr)
        dice = dice_coef(chrom_labels, chrom_predictions)
        dices.append(dice)
        if aggregate:
            y_true.append(chrom_labels)
            y_pred.append(chrom_predictions)
    if aggregate:
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        dice_mean = dice_coef(y_true, y_pred)
        jaccard_mean = dice_mean / (2 - dice_mean)
        fpr_mean, tpr_mean, _ = roc_curve(y_true, y_pred)
        precision_mean, recall_mean, _ = precision_recall_curve(y_true, y_pred)
        auroc_mean = auc(fpr_mean, tpr_mean)
        aupr_mean = auc(recall_mean, precision_mean)
    else:
        dice_mean = np.mean(dices)
        jaccards = [s / (2 - s) for s in dices]
        jaccard_mean = np.mean(jaccards)
        auroc_mean = np.mean(aurocs)
        aupr_mean = np.mean(auprs)

    bw.close()
    print('Chromosomes:', chroms)
    print('Positive fractions:', fracs)
    print('Dice coefficients:', dices)
    print('Dice coefficient (chromosome average):', dice_mean)
    print('Jaccard indexes:', jaccards)
    print('Jaccard index (chromosome average):', jaccard_mean)
    print('auROCs:', aurocs)
    print('auROC (chromosome average):', auroc_mean)
    print('auPRs:', auprs)
    print('auPR (chromosome average):', aupr_mean)
    """
    pylab.subplot(121)
    for i, chrom in enumerate(chroms):
        pylab.plot(fprs[i], tprs[i], label=chrom + ' (auROC=%0.2f)' % aurocs[i])
    pylab.plot([0, 1], [0, 1], 'k--', label='Random')
    pylab.legend(loc='lower right')
    pylab.xlabel('FPR')
    pylab.ylabel('TPR')

    pylab.subplot(122)
    for i, chrom in enumerate(chroms):
        pylab.plot(recalls[i], precisions[i], label=chrom + ' (auPR=%0.2f)' % auprs[i])
    pylab.legend(loc='upper right')
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.show()
    """


if __name__ == '__main__':
    main()
