''' Implements the WEAT tests from sent-bias repo
    https://github.com/W4ngatang/sent-bias/blob/master/sentbias/weat.py

    This script runs WEAT by evaluating bias over the full sets A (AX \cup AY) and B (BX \cup BY).
'''

from copy import copy
import ctypes
import itertools as it
import logging as log
import math
from multiprocessing import Pool, Process, Value, Manager, Array
import numpy as np
from random import shuffle
import scipy.special
import scipy.stats
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity as torch_cossim
from warnings import warn

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.
# A = AX \cup AY and B = BX \cup BY

NUM_PARALLEL = 100
NUM_SHUFFLES = 10

def inner_p_test(args):# A, B, len_A_X, len_B_X, cossims_attrX, cossims_attrY, return_vals_list):
    A, B, len_AX, len_BX, cossims_attrX, cossims_attrY, return_vals_list = args
    A, B = copy(A), copy(B)

    # shuffle
    count = 0
    while count < NUM_SHUFFLES:
        count+=1
        A = A[np.random.permutation(len(A))][:len_AX]
        indices_Ax = np.array(np.where(A == 'X')).transpose()[:,0]
        indices_Ay = np.array(np.where(A == 'Y')).transpose()[:,0]
        Aix, Aiy = A[indices_Ax][:,1].astype('float'), A[indices_Ay][:,1].astype('float')
        if len(Aix) > 0 and len(Aiy) > 0:
            break 
        if count == NUM_SHUFFLES:
            warn('no suitable split found for A')
            return

    count = 0
    while count < NUM_SHUFFLES:
        count+=1
        B = B[np.random.permutation(len(B))][:len_BX]
        indices_Bx = np.array(np.where(B == 'X')).transpose()[:,0]
        indices_By = np.array(np.where(B == 'Y')).transpose()[:,0]
        Bix, Biy = B[indices_Bx][:,1].astype('float'), B[indices_By][:,1].astype('float')
        if len(Bix) > 0 and len(Biy) > 0:
            break
        if count == NUM_SHUFFLES:
            warn('no suitable split found for B')
            return
    
    si = s_wAB(Aix, Bix, cossims=cossims_attrX).sum() + s_wAB(Aiy, Biy, cossims=cossims_attrY).sum()
    return_vals_list.append(si)
    return si

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
        A = np.array([('X', a) for a in A_X] + [('Y', a) for a in A_Y])
        B = np.array([('X', b) for b in B_X] + [('Y', b) for b in B_Y])
        len_AX = max(2, len(A_X)) # need at least 2 samples so we can get an X and Y
        len_BX = max(2, len(B_X)) # need at least 2 samples so we can get an X and Y

        if run_sampling:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            log.info('Drawing {} samples (and biasing by 1)'.format(n_samples - total))

            results = []
            manager = Manager()
            return_vals_list = manager.list()
            for idx in tqdm(range(1, math.ceil(n_samples))):
                # shuffle
                A = A[np.random.permutation(len(A))]
                B = B[np.random.permutation(len(B))]

                # top chunk of list
                A_slice, B_slice = A[:len_AX], B[:len_BX]
                indices_Ax = np.array(np.where(A_slice == 'X')).transpose()[:,0]
                indices_Ay = np.array(np.where(A_slice == 'Y')).transpose()[:,0]
                indices_Bx = np.array(np.where(B_slice == 'X')).transpose()[:,0]
                indices_By = np.array(np.where(B_slice == 'Y')).transpose()[:,0]

                Aix, Aiy = A[indices_Ax][:,1].astype('float'), A[indices_Ay][:,1].astype('float')
                Bix, Biy = B[indices_Bx][:,1].astype('float'), B[indices_By][:,1].astype('float')
                if len(Aix) == 0 or len(Aiy) == 0 or len(Bix) == 0 or len(Biy) == 0:
                    continue
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
    try:
        return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)
    except:
        return cossims[:, A].mean(dim=1) - cossims[:, B].mean(dim=1)
    
def mean_s_wAB(X, A, B, cossims):
    #return np.mean(s_wAB(A, B, cossims[X]))
    return torch.mean(s_wAB(A,B,cossims[X]))

# each attribute varies by target
def stdev_s_wAB(X, A_X, B_X, A_Y, B_Y, cossims_X, cossims_Y):
    valX = s_wAB(A_X, B_X, cossims_X[X]).cpu().numpy()
    valY = s_wAB(A_Y, B_Y, cossims_Y[X]).cpu().numpy()
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

def run_test(X, Y, AX, AY, BX, BY, n_samples, cat_X, cat_Y, cat_A, cat_B, parametric=False):
    ''' Run a WEAT.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''

    # First convert all keys to ints to facilitate array lookups
    X = convert_keys_to_ints(X)
    Y = convert_keys_to_ints(Y)
    
    (AX, BX) = convert_keys_to_ints(AX, BX)
    (AY, BY) = convert_keys_to_ints(AY, BY)
    
    AB_X = AX.copy()
    AB_X.update(BX)
    
    AB_Y = AY.copy()
    AB_Y.update(BY)


    log.info("Computing cosine similarities...")
    cossims_XonX = construct_cossim_lookup(X, AB_X).cuda()
    cossims_XonY = construct_cossim_lookup(X, AB_Y).cuda()
    # first X on attrX attrY
    log.info(f"Null hypothesis: no difference between {cat_X} in association to attributes {cat_A} and {cat_B} across images")

    log.info("Computing pval...")
    pval_x = p_val_permutation_test(X, AX, BX, AY, BY, n_samples,
                                   cossims_attrX=cossims_XonX,
                                   cossims_attrY=cossims_XonY,
                                   parametric=parametric)
    log.info("pval: %g", pval_x)

    log.info("computing effect size...")
    esize_x = effect_size(X, AX, BX, AY, BY,
                          cossims_XonX, cossims_XonY)
    log.info("esize: %g", esize_x)

    
    # now Y on attrX attrY
    log.info(f"Null hypothesis: no difference between {cat_Y} in association to attributes {cat_A} and {cat_B} across images")
    
    log.info("Computing pval...")
    cossims_YonX = construct_cossim_lookup(Y, AB_X).cuda()
    cossims_YonY = construct_cossim_lookup(Y, AB_Y).cuda()
    pval_y = p_val_permutation_test(Y, AX, BX, AY, BY, n_samples,
                                  cossims_attrX=cossims_YonX,
                                   cossims_attrY=cossims_YonY,
                                   parametric=parametric)
    log.info("pval: %g", pval_y)

    log.info("computing effect size...")
    esize_y = effect_size(Y, AX, BX, AY, BY,
                          cossims_YonX, cossims_YonY)
    log.info("esize: %g", esize_y)
    return esize_x.item(), pval_x, esize_y.item(), pval_y
