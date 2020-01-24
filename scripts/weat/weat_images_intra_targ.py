''' Implements the WEAT tests from sent-bias repo
    https://github.com/W4ngatang/sent-bias/blob/master/sentbias/weat.py

    This script runs WEAT by evaluating bias over the full sets A (AX \cup AY) and B (BX \cup BY).
'''

import logging as log
import math
import itertools as it
import numpy as np
from random import shuffle
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
    AB = torch.stack([AB[i] for i in AB.keys()])
    cossims = torch.zeros( (len(XY), len(AB)) )
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
    return cossims[:, A].mean(dim=1) - cossims[:, B].mean(dim=1)


def p_val_permutation_test(X, A_X, B_X, A_Y, B_Y, n_samples,
                           cossims_attrX, cossims_attrY,
                           parametric=False):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''

    X = np.array(list(X), dtype=np.int)
    A_X = np.array(list(A_X), dtype=np.int)
    B_X = np.array(list(B_X), dtype=np.int)
    A_Y = np.array(list(A_Y), dtype=np.int)
    B_Y = np.array(list(B_Y), dtype=np.int)

    if parametric:
        raise Exception('not implemented')

    else:
        log.info('Using non-parametric test')
        s =  s_wAB(A_X, B_X, cossims=cossims_attrX).sum()
        total_true = 0
        total_equal = 0
        total = 0
        
        run_sampling = len(X) > 20 # large to compute num partitions, so sample
        A = [('X', a) for a in A_X] + [('Y', a) for a in A_Y]
        B = [('X', b) for b in B_X] + [('Y', b) for b in B_Y]

        if run_sampling:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            log.info('Drawing {} samples (and biasing by 1)'.format(n_samples - total))

            for _ in range(n_samples - 1):
                if _ % 10000 == 0:
                    print('on', _)
                    
                shuffle(A)
                shuffle(B)

                Ai = A[:len(A_X)]
                Bi = B[:len(B_X)]

                Aix = [a for set_,a in Ai if set_ == 'X']
                Aiy = [a for set_,a in Ai if set_ == 'Y']

                Bix = [b for set_,b in Bi if set_ == 'X']
                Biy = [b for set_,b in Bi if set_ == 'Y']

                si = s_wAB(Aix, Bix, cossims=cossims_attrX).sum() + \
                     s_wAB(Aiy, Biy, cossims=cossims_attrY).sum()
                    
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1
        else:
            raise Exception('Not implemented')

        if total_equal:
            log.warning('Equalities contributed {}/{} to p-value'.format(total_equal, total))

        return total_true / total

def s_wAB(A, B, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)
    
def mean_s_wAB(X, A, B, cossims):
    #return np.mean(s_wAB(A, B, cossims[X]))
    return torch.mean(s_wAB(A,B,cossims[X]))

# each attribute varies by target
def stdev_s_wAB(X, A_X, B_X, A_Y, B_Y, cossims_X, cossims_Y):
    valX = s_wAB(A_X, B_X, cossims_X[X])
    valY = s_wAB(A_Y, B_Y, cossims_Y[X])
    vals = np.concatenate((valX, valY))
    return np.std(vals, ddof=1)

def effect_size(X, A_X, B_X, A_Y, B_Y, cossims_attrX, cossims_attrY):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    A_X = list(A_X)
    B_X = list(B_X)
    A_Y = list(A_Y)
    B_Y = list(B_Y)
    
    numerator = mean_s_wAB(X, A_X, B_X, cossims=cossims_attrX) -\
                                mean_s_wAB(X, A_Y, B_Y, cossims=cossims_attrY)
    denominator = stdev_s_wAB(X, A_X, B_X, A_Y, B_Y, cossims_attrX, cossims_attrY)
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

def run_test(encs, n_samples, parametric=False):
    ''' Run a WEAT.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''
    X, Y = encs["targ_X"]["encs"], encs["targ_Y"]["encs"]
    A_X, A_Y = encs["attr_A_X"]["encs"], encs["attr_A_Y"]["encs"]
    B_X, B_Y = encs["attr_B_X"]["encs"], encs["attr_B_Y"]["encs"]

    # First convert all keys to ints to facilitate array lookups
    X = convert_keys_to_ints(X)
    Y = convert_keys_to_ints(Y)
    
    (A_X, B_X) = convert_keys_to_ints(A_X, B_X)
    (A_Y, B_Y) = convert_keys_to_ints(A_Y, B_Y)
    
    AB_X = A_X.copy()
    AB_X.update(B_X)
    
    AB_Y = A_Y.copy()
    AB_Y.update(B_Y)


    log.info("Computing cosine similarities...")
    cossims_XonX = construct_cossim_lookup(X, AB_X)
    cossims_XonY = construct_cossim_lookup(X, AB_Y)

    # first X on attrX attrY
    log.info("Null hypothesis: no difference between %s in association to attributes %s and %s across images",
             encs["targ_X"]["category"], encs["attr_A_X"]["category"], encs["attr_B_X"]["category"])

    log.info("Computing pval...")
    pval_x = p_val_permutation_test(X, A_X, B_X, A_Y, B_Y, n_samples,
                                    cossims_attrX=cossims_XonX,
                                    cossims_attrY=cossims_XonY,
                                    parametric=parametric)
    log.info("pval: %g", pval_x)

    log.info("computing effect size...")
    esize_x = effect_size(X, A_X, B_X, A_Y, B_Y,
                          cossims_XonX, cossims_XonY)
    log.info("esize: %g", esize_x)

    
    # now Y on attrX attrY
    log.info("Null hypothesis: no difference between %s in association to attributes %s and %s across images",
             encs["targ_Y"]["category"], encs["attr_A_X"]["category"], encs["attr_B_X"]["category"])

    log.info("Computing pval...")
    cossims_YonX = construct_cossim_lookup(Y, AB_X)
    cossims_YonY = construct_cossim_lookup(Y, AB_Y)
    pval_y = p_val_permutation_test(Y, A_X, B_X, A_Y, B_Y, n_samples,
                                    cossims_attrX=cossims_YonX,
                                    cossims_attrY=cossims_YonY,
                                    parametric=parametric)
    log.info("pval: %g", pval_y)

    log.info("computing effect size...")
    esize_y = effect_size(Y, A_X, B_X, A_Y, B_Y,
                          cossims_YonX, cossims_YonY)
    log.info("esize: %g", esize_y)
    return esize_x.item(), pval_x, esize_y.item(), pval_y
