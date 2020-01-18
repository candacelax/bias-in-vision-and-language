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
from torch.nn.functional import cosine_similarity as torch_cossim

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
    cossims = np.zeros((len(XY), len(AB)))
    dims = torch.Size( (len(AB), len(XY[0])) )

    for xy in XY:
        cossims[xy, :] = torch_cossim(XY[xy].expand(dims),
                                      AB)
    return cossims


def s_wAB(A, B, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)


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
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A = np.array(list(A), dtype=np.int)
    B = np.array(list(B), dtype=np.int)

    assert len(X) == len(Y)
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = np.concatenate((X, Y))

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
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
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
            log.info('Using exact test ({} partitions)'.format(num_partitions))
            for Xi in it.combinations(XY, len(X)):
                Xi = np.array(Xi, dtype=np.int)
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
        return total_true / total


def mean_s_wAB(X, A, B, cossims):
    return np.mean(s_wAB(A, B, cossims[X]))


def stdev_s_wAB(X, A, B, cossims):
    return np.std(s_wAB(A, B, cossims[X]), ddof=1)


def effect_size(X, Y, A, B, cossims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator


def convert_keys_to_ints(X, Y=None):
    if Y:
        return (
            dict((i, v) for (i, (k, v)) in enumerate(X.items())),
            dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
        )
    else:
        return (
            dict((i, v) for (i, (k, v)) in enumerate(X.items()))
        )

    
def convert_keys_to_ints_combine(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(list(X.items()) +\
                                                 list(Y.items())))
    )

''' "classifier case": WEAT where A=AX \cup AY and B=BX \cup BY
'''
def get_general_vals(encs, n_samples, parametric=False):
    ''' Run a WEAT.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''
    X, Y = encs["targ_X"]["encs"], encs["targ_Y"]["encs"]
    AX, AY = encs["attr_A_X"]["encs"], encs["attr_A_Y"]["encs"]
    BX, BY = encs["attr_B_X"]["encs"], encs["attr_B_Y"]["encs"]

    #----- 1.
    log.info("Computing cosine similarities [ cos(X, AX) - cos(X, AY) ]...")
    X = convert_keys_to_ints(X)
    Y = convert_keys_to_ints(Y)
    
    (AX, AY) = convert_keys_to_ints(AX, AY)
    AXY = AX.copy()
    AXY.update(AY)

    (BX, BY) = convert_keys_to_ints(BX, BY)
    BXY = BX.copy()
    BXY.update(BY)

    cossims_X = construct_cossim_lookup(X, AXY)
    AX = np.array(list(AX), dtype=np.int)
    AY = np.array(list(AY), dtype=np.int)
    X_AXonAY = s_wAB(AX, AY, cossims_X).sum()

    cossims_X = construct_cossim_lookup(X, BXY)
    BX = np.array(list(BX), dtype=np.int)
    BY = np.array(list(BY), dtype=np.int)
    X_BXonBY = s_wAB(BX, BY, cossims_X).sum()

    cossims_Y = construct_cossim_lookup(X, AXY)
    Y_AXonAY = s_wAB(AX, AY, cossims_Y).sum()
    
    cossims_Y = construct_cossim_lookup(X, BXY)
    Y_BXonBY = s_wAB(BX, BY, cossims_Y).sum()


    #----- 2.
    log.info("Computing cosine similarities [ cos(X, AX+AY) - cos(X, BX+BY) ]...")
    X, Y = encs["targ_X"]["encs"], encs["targ_Y"]["encs"]
    AX, AY = encs["attr_A_X"]["encs"], encs["attr_A_Y"]["encs"]
    BX, BY = encs["attr_B_X"]["encs"], encs["attr_B_Y"]["encs"]
    X = convert_keys_to_ints(X)
    Y = convert_keys_to_ints(Y)
    
    # take union over attribute images; images differ by target XY
    A = convert_keys_to_ints_combine(AX, AY)
    B = convert_keys_to_ints_combine(BX, BY)
    (A, B) = convert_keys_to_ints(A, B)

    AB = A.copy()
    AB.update(B)

    cossims_X = construct_cossim_lookup(X, AB)
    cossims_Y = construct_cossim_lookup(Y, AB)

    A = np.array(list(A), dtype=np.int)
    B = np.array(list(B), dtype=np.int)
    X_AonB = s_wAB(A, B, cossims_X).sum()
    Y_AonB = s_wAB(A, B, cossims_Y).sum()
    

    #----- 3.
    log.info("Computing cosine similarities [ cos(X, AX+BX) - cos(X, AY+BY) ]...")
    X, Y = encs["targ_X"]["encs"], encs["targ_Y"]["encs"]
    AX, AY = encs["attr_A_X"]["encs"], encs["attr_A_Y"]["encs"]
    BX, BY = encs["attr_B_X"]["encs"], encs["attr_B_Y"]["encs"]
    X = convert_keys_to_ints(X)
    Y = convert_keys_to_ints(Y)
    
    # take union over attribute images; images differ by target XY
    ABX = convert_keys_to_ints_combine(AX, BX)
    ABY = convert_keys_to_ints_combine(AY, BY)
    (ABX, ABY) = convert_keys_to_ints(ABX, ABY)

    AB = ABX.copy()
    AB.update(ABY)

    cossims_X = construct_cossim_lookup(X, AB)
    cossims_Y = construct_cossim_lookup(Y, AB)

    ABX = np.array(list(ABX), dtype=np.int)
    ABY = np.array(list(ABY), dtype=np.int)
    X_ABXonABY = s_wAB(ABX, ABY, cossims_X).sum()
    Y_ABXonABY = s_wAB(ABX, ABY, cossims_Y).sum()
    
    return {'X_AXonAY' : X_AXonAY,
            'X_BXonBY' : X_BXonBY,
            'Y_AXonAY' : Y_AXonAY,
            'Y_BXonBY' : Y_BXonBY,
            'X_AonB' : X_AonB,
            'Y_AonB' : Y_AonB,
            'X_ABXonABY' : X_ABXonABY,
            'Y_ABXonABY' : Y_ABXonABY}

                          
