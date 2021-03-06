''' Implements the WEAT tests from sent-bias repo
    https://github.com/W4ngatang/sent-bias/blob/master/sentbias/weat.py

    This script runs WEAT by evaluating bias over the full sets A (AX \cup AY) and B (BX \cup BY).
'''

import logging as log
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats
import torch
from torch.nn.functional import cosine_similarity as f_cossim

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.
# A = AX \cup AY and B = BX \cup BY

def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """

    AB = torch.stack([AB[i] for i in range(len(AB))])
    cossims = torch.zeros((len(XY), len(AB)))
    dims = torch.Size( (len(AB), len(XY[0])) )

    for xy in XY:
        cossims[xy, :] = f_cossim(XY[xy].expand(dims), AB)
    return cossims


def s_wAB(A, B, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[:, A].mean(dim=1) - cossims[:, B].mean(dim=1)


def s_XAB(X, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    return slightly more computationally efficient version of WEAT
    statistic for p-value computation.
    Caliskan defines the WEAT statistic s(X, Y, A, B) as
        sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    where s(w, A, B) is defined as
        mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    The p-value is computed using a permutation test on (X, Y) over all
    partitions (X', Y') of X union Y with |X'| = |Y'|.
    However, for all partitions (X', Y') of X union Y,
        s(X', Y', A, B)
      = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = C,
    a constant.  Thus
        sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
      = C + 2 sum_{x in X'} s(x, A, B).
    By monotonicity,
        s(X', Y', A, B) > s(X, Y, A, B)
    if and only if
        [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
    that is,
        sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
    Thus we only need use the first component of s(X, Y, A, B) as our
    test statistic.
    """
    return s_wAB_memo[X].sum()


def s_XYAB(X, Y, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)


def p_val_permutation_test(X, Y, A, B, n_samples, cossims, parametric=False):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    #X = torch.tensor(list(X), dtype=torch.int)
    #Y = torch.tensor(list(Y), dtype=torch.int)
    #A = torch.tensor(list(A), dtype=torch.int)
    #B = torch.tensor(list(B), dtype=torch.int)
    X,Y = list(X), list(Y)
    A,B = list(A), list(B)

    # TODO fixme
    if len(X) < len(Y):
        Y = Y[:len(X)]
    elif len(X) > len(Y):
        X = X[:len(Y)]
    assert len(X) == len(Y), f'len X {len(X)}, len Y {len(Y)}'
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = X + Y

    if parametric:
        log.info('Using parametric test')
        s = s_XYAB(X, Y, s_wAB_memo)

        log.info('Drawing {} samples'.format(n_samples))
        samples = []
        for _ in range(n_samples):
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si)

        # Compute sample standard deviation and compute p-value by
        # assuming normality of null distribution
        log.info('Inferring p-value based on normal distribution')
        (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        log.info('Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}'.format(
            shapiro_test_stat, shapiro_p_val))
        sample_mean = torch.mean(samples)
        sample_std = torch.std(samples)
        log.info('Sample mean: {:.2g}, sample standard deviation: {:.2g}'.format(
            sample_mean, sample_std))
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else:
        log.info('Using non-parametric test')
        s = s_XAB(X, s_wAB_memo)
        total_true = 0
        total_equal = 0
        total = 0

        run_sampling = len(X) > 20 # large to compute num partitions, so sample
        if run_sampling:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            log.info('Drawing {} samples (and biasing by 1)'.format(n_samples - total))
            for _ in range(n_samples - 1):
                np.random.shuffle(XY)
                Xi = XY[:size]
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
            log.info(f'Using exact test ({num_partitions} partitions)')
            for Xi in it.combinations(XY, len(X)):
                #Xi = torch.tensor(Xi, dtype=torch.int)
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        if total_equal:
            log.warning('Equalities contributed {}/{} to p-value'.format(total_equal, total))
        if not isinstance(total_true / total, float):
            raise Exception(f'nan {total_true}, {total}')
        return total_true / total


def mean_s_wAB(X, A, B, cossims):
    return torch.mean(s_wAB(A, B, cossims[X]))


def stdev_s_wAB(X, A, B, cossims):
    return torch.std(s_wAB(A, B, cossims[X]))

def effect_size(X, Y, A, B, cossims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X, Y = list(X), list(Y)
    A, B = list(A), list(B)
    assert X != Y
    numerator = mean_s_wAB(X, A, B, cossims) - mean_s_wAB(Y, A, B, cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )

def convert_keys_to_ints_combine(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(list(X.items()) +\
                                                 list(Y.items())))
    )

''' "classifier case": WEAT where A=AX \cup AY and B=BX \cup BY
'''
def run_test(X, Y, AX, AY, BX, BY, n_samples, cat_X, cat_Y, cat_A, cat_B, parametric=False):
    ''' Run a WEAT.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''
    
    # take union over attribute images; images differ by target XY
    A = convert_keys_to_ints_combine(AX, AY)
    B = convert_keys_to_ints_combine(BX, BY)

    # First convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    log.info("Computing cosine similarities...")
    cossims = construct_cossim_lookup(XY, AB).cuda()

    log.info(f"Null hypothesis: no difference between {cat_X} and {cat_Y} in association to attributes {cat_A} and {cat_B}")
    log.info("Computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, n_samples, cossims=cossims, parametric=parametric)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    log.info("esize: %g", esize)

    return esize, pval
