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


def p_val_permutation_test(X, Y, A_ID1, B_ID1, A_ID2, B_ID2, n_samples,
                           cossims_X, cossims_Y, parametric=False):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A_ID1 = np.array(list(A_ID1), dtype=np.int)
    B_ID1 = np.array(list(B_ID1), dtype=np.int)
    A_ID2 = np.array(list(A_ID2), dtype=np.int)
    B_ID2 = np.array(list(B_ID2), dtype=np.int)

    assert len(X) == len(Y)
    size = len(X)

    #s_wAB_memo = s_wAB(A, B, cossims=cossims)
    s_wAB_ID1_memo = s_wAB(A_ID1, B_ID1, cossims=cossims_X)
    s_wAB_ID2_memo = s_wAB(A_ID2, B_ID2, cossims=cossims_Y)

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
        #s = s_XAB(X, s_wAB_memo)
        s = s_XAB(X, s_wAB_ID1_memo)
        total_true = 0
        total_equal = 0
        total = 0

        num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
        if num_partitions > n_samples:
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
                #si = s_XAB(Xi, s_wAB_memo)
                si = s_XAB(Xi, s_wAB_ID1_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            # FIXME check this math
            s_XAB_ID1 = lambda s: s_XAB(s, s_wAB_ID1_memo)
            s_XAB_ID2 = lambda s: s_XAB(s, s_wAB_ID2_memo)
            
            log.info('Using exact test ({} partitions)'.format(num_partitions))
            for Xi in it.combinations(XY, len(X)):
                #Xi = np.array(Xi, dtype=np.int)
                Xi_ID1 = np.array([x for x in Xi if x in X], dtype=np.int)
                Xi_ID2 = np.array([x for x in Xi if x in Y], dtype=np.int)
                
                assert 2 * len(Xi) == len(XY)
                #si = s_XAB(Xi, s_wAB_memo)
                si = s_XAB(Xi_ID1, s_wAB_ID1_memo) +\
                     s_XAB(Xi_ID2, s_wAB_ID2_memo)
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


#def stdev_s_wAB(X, A, B, cossims):
#    return np.std(s_wAB(A, B, cossims[X]), ddof=1)

# two IDs
def stdev_s_wAB(X, Y, A_ID1, B_ID1, A_ID2, B_ID2, cossims_X, cossims_Y):
    valX = s_wAB(A_ID1, B_ID1, cossims_X[X])
    valY = s_wAB(A_ID2, B_ID2, cossims_Y[Y])
    vals = np.concatenate((valX, valY))
    return np.std(vals, ddof=1)


def effect_size(X, Y, A_ID1, B_ID1, A_ID2, B_ID2, cossims_X, cossims_Y):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A_ID1 = list(A_ID1)
    B_ID1 = list(B_ID1)
    A_ID2 = list(A_ID2)
    B_ID2 = list(B_ID2)

    #numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    #denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    numerator = mean_s_wAB(X, A_ID1, B_ID1, cossims=cossims_X) -\
                                mean_s_wAB(Y, A_ID2, B_ID2, cossims=cossims_Y)

    #denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(X, Y, A_ID1, B_ID1, A_ID2, B_ID2, cossims_X, cossims_Y)
    return numerator / denominator


def convert_keys_to_ints(X, Y=None, starting_idx=0):
    if Y is None:
        return (
            dict((i+starting_idx, v) for (i, (k, v)) in enumerate(X.items()))
        )
    else:
        return (
            dict((i, v) for (i, (k, v)) in enumerate(X.items())),
            dict((i + starting_idx + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
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
    X, Y = encs["targ1"]["encs"], encs["targ2"]["encs"]
    A_ID1, B_ID1 = encs["attr1_ID1"]["encs"], encs["attr2_ID1"]["encs"]
    A_ID2, B_ID2 = encs["attr1_ID1"]["encs"], encs["attr2_ID2"]["encs"]

    # First convert all keys to ints to facilitate array lookups
    #(X, Y) = convert_keys_to_ints(X, Y)
    X = convert_keys_to_ints(X)
    Y = convert_keys_to_ints(Y, starting_idx=len(X))

    (A_ID1, B_ID1) = convert_keys_to_ints(A_ID1, B_ID1)
    (A_ID2, B_ID2) = convert_keys_to_ints(A_ID2, B_ID2)

    AB_ID1 = A_ID1.copy()
    AB_ID1.update(B_ID1)

    AB_ID2 = A_ID2.copy()
    AB_ID2.update(B_ID2)
    
    log.info("Computing cosine similarities...")
    #cossims = construct_cossim_lookup(XY, AB)
    cossims_X = construct_cossim_lookup(X, AB_ID1)
    cossims_Y = construct_cossim_lookup(Y, AB_ID2)

    log.info("Null hypothesis: no difference between %s and %s in association to attributes %s and %s",
             encs["targ1"]["category"], encs["targ2"]["category"],
             encs["attr1_ID1"]["category"], encs["attr2_ID2"]["category"])
    log.info("Computing pval...")
    #pval = p_val_permutation_test(X, Y, A, B, n_samples, cossims=cossims, parametric=parametric)
    pval = p_val_permutation_test(X, Y,
                                  A_ID1, B_ID1,
                                  A_ID2, B_ID2,
                                  n_samples,
                                  cossims_X=cossims_X,
                                  cossims_Y=cossims_Y,
                                  parametric=parametric)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A_ID1, B_ID1, A_ID2, B_ID2, cossims_X, cossims_Y)
    log.info("esize: %g", esize)
    return esize, pval



if __name__ == "__main__":
    X = {"x" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    Y = {"y" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = {"a" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    B = {"b" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = X
    B = Y

    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    log.info("computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, cossims=cossims, n_samples=10000)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    log.info("esize: %g", esize)
