''' Implements the WEAT tests from sent-bias repo
    https://github.com/W4ngatang/sent-bias/blob/master/sentbias/weat.py
'''

import logging as log
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.


def cossim(x, y):
    b = np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """

    #cossims = np.zeros((len(XY), len(AB)))
    cossims = np.zeros((max(XY)+1, len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
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


def p_val_permutation_test(X, Y, A_X, B_X, A_Y, B_Y, n_samples,
                           cossims_X, cossims_Y, parametric=False):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A_X = np.array(list(A_X), dtype=np.int)
    B_X = np.array(list(B_X), dtype=np.int)
    A_Y = np.array(list(A_Y), dtype=np.int)
    B_Y = np.array(list(B_Y), dtype=np.int)

    assert len(X) == len(Y)
    size = len(X)

    s_wAB_X_memo = s_wAB(A_X, B_X, cossims=cossims_X) # avg cos_sim for each x across A_X and B_X
    s_wAB_Y_memo = s_wAB(A_Y, B_Y, cossims=cossims_Y) # avg cos_sim for each y across A_Y and B_Y

    XY = np.concatenate((X, Y))

    if parametric:
        raise Exception('not implemented')
        # log.info('Using parametric test')
        # s = s_XYAB(X, Y, s_wAB_memo)

        # log.info('Drawing {} samples'.format(n_samples))
        # samples = []
        # for _ in range(n_samples):
        #     np.random.shuffle(XY)
        #     Xi = XY[:size]
        #     Yi = XY[size:]
        #     assert len(Xi) == len(Yi)
        #     si = s_XYAB(Xi, Yi, s_wAB_memo)
        #     samples.append(si)

        # # Compute sample standard deviation and compute p-value by
        # # assuming normality of null distribution
        # log.info('Inferring p-value based on normal distribution')
        # (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        # log.info('Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}'.format(
        #     shapiro_test_stat, shapiro_p_val))
        # sample_mean = np.mean(samples)
        # sample_std = np.std(samples, ddof=1)
        # log.info('Sample mean: {:.2g}, sample standard deviation: {:.2g}'.format(
        #     sample_mean, sample_std))
        # p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        # return p_val

    else:
        log.info('Using non-parametric test')
        s = s_XAB(X, s_wAB_X_memo) # really just s_wAB_X_memo.sum() bc X is only target
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
                Xi = set(XY[:size])
                # for random samples, sort by w \in X and w \in Y
                Xi_X = np.array(list(Xi.intersection(X)), dtype=np.int)
                Xi_Y = np.array(list(Xi.intersection(Y)), dtype=np.int)

                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi_X, s_wAB_X_memo) +\
                     s_XAB(Xi_Y, s_wAB_Y_memo)
                     
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
                Xi_X = np.array(list(Xi.intersection(X)), dtype=np.int)
                Xi_Y = np.array(list(Xi.intersection(Y)), dtype=np.int)
                
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi_X, s_wAB_X_memo) +\
                     s_XAB(Xi_Y, s_wAB_Y_memo)
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

# each attribute varies by target
def stdev_s_wAB(X, Y, A_X, B_X, A_Y, B_Y, cossims_X, cossims_Y):
    valX = s_wAB(A_X, B_X, cossims_X[X])
    valY = s_wAB(A_Y, B_Y, cossims_Y[Y])
    vals = np.concatenate((valX, valY))
    return np.std(vals, ddof=1)

def effect_size(X, Y, A_X, B_X, A_Y, B_Y, cossims_X, cossims_Y):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A_X = list(A_X)
    B_X = list(B_X)
    A_Y = list(A_Y)
    B_Y = list(B_Y)
    
    numerator = mean_s_wAB(X, A_X, B_X, cossims=cossims_X) -\
                                mean_s_wAB(Y, A_Y, B_Y, cossims=cossims_Y)
    denominator = stdev_s_wAB(X, Y, A_X, B_X, A_Y, B_Y, cossims_X, cossims_Y)
    return numerator / denominator

def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )

def run_test(encs, n_samples, parametric=False):
    ''' Run a WEAT with gender-specific images.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''
    X, Y = encs["targ_X"]["encs"], encs["targ_Y"]["encs"]
    A_X, B_X = encs["attr_A_X"]["encs"], encs["attr_B_X"]["encs"]
    A_Y, B_Y = encs["attr_A_Y"]["encs"], encs["attr_B_Y"]["encs"]

    # First convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X,Y)
    (A_X, B_X) = convert_keys_to_ints(A_X, B_X)
    (A_Y, B_Y) = convert_keys_to_ints(A_Y, B_Y)

    AB_X = A_X.copy()
    AB_X.update(B_X)

    AB_Y = A_Y.copy()
    AB_Y.update(B_Y)
    
    log.info("Computing cosine similarities...")
    cossims_X = construct_cossim_lookup(X, AB_X)
    cossims_Y = construct_cossim_lookup(Y, AB_Y)

    log.info("Null hypothesis: no difference between %s and %s in association to attributes %s and %s",
             encs["targ_X"]["category"], encs["targ_Y"]["category"],
             encs["attr_A_X"]["category"], encs["attr_B_X"]["category"])
    log.info("Computing pval...")
    pval = p_val_permutation_test(X, Y,
                                  A_X, B_X,
                                  A_Y, B_Y,
                                  n_samples,
                                  cossims_X=cossims_X,
                                  cossims_Y=cossims_Y,
                                  parametric=parametric)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A_X, B_X, A_Y, B_Y, cossims_X, cossims_Y)
    log.info("esize: %g", esize)
    return esize, pval
