"""
Performs iterative proportional fitting (also called "raking") on N variables.
"""

import os
import sys
import numpy as np

from ipfn import ipfn
from collections import Counter


# raking models for two variables
MODELS_2 = [
    [[0], [1]],
    [[1], [0]],
    [[0,1]],
]

# raking models for three variables
MODELS_3 = [
    # 1D marginals only
    [[0], [1], [2]],
    [[0], [2], [1]],
    [[1], [0], [2]],
    [[1], [2], [0]],
    [[2], [0], [1]],
    [[2], [1], [0]],
    
    # mixed marginals
    [[0], [1,2]],
    [[1], [0,2]],
    [[2], [0,1]],
    [[0,1], [2]],
    [[0,2], [1]],
    [[1,2], [0]],
]

# raking models for four variables
MODELS_4 = [
    # 1D marginals only
    [[0], [1], [2], [3]],
    
    # single 2D interaction
    [[0], [1], [2,3]],
    [[0], [2], [1,3]],
    [[0], [3], [1,2]],
    [[1], [2], [0,3]],
    [[1], [3], [0,2]],
    [[2], [3], [0,1]],
    [[2,3], [0], [1]],
    [[1,3], [0], [2]],
    [[1,2], [0], [3]],
    [[0,3], [1], [2]],
    [[0,2], [1], [3]],
    [[0,1], [2], [3]],
    
    # 2D interactions only
    [[0,1], [2,3]],
    [[0,2], [1,3]],
    [[0,3], [1,2]],
    [[1,2], [0,3]],
    [[1,3], [0,2]],
    [[2,3], [0,1]],    
]

# raking models for five variables
MODELS_5 = [
    # 1D marginals only
    [[0], [1], [2], [3], [4]],
    
    # single 2D interaction
    [[0], [1], [2], [3,4]],
    [[0], [1], [3], [2,4]],
    [[0], [1], [4], [2,3]],
    [[0], [2], [3], [1,4]],
    [[0], [2], [4], [1,3]],
    [[0], [3], [4], [1,2]],
    [[1], [2], [3], [0,4]],
    [[1], [2], [4], [0,3]],
    [[1], [3], [4], [0,2]],
    [[2], [3], [4], [0,1]],

    # 2D marginals (assuming order doesn't matter)

    [[0,1], [2,3], [4]],
    [[0,1], [2,4], [3]],
    [[0,1], [3,4], [2]],
    [[0,2], [1,3], [4]],
    [[0,2], [1,4], [3]],
    [[0,2], [3,4], [1]],
    [[0,3], [1,2], [4]],
    [[0,3], [1,4], [2]],
    [[0,3], [2,4], [1]],
    [[0,4], [1,2], [3]],
    [[0,4], [1,3], [2]],
    [[0,4], [2,3], [1]],
    [[1,2], [3,4], [0]],
    [[1,3], [2,4], [0]],
    [[1,4], [2,3], [0]],
]

# raking models for six variables
MODELS_6 = [

    # 1D marginals only
    [[0], [1], [2], [3], [4], [5]],    
    
    # single interaction
    [[0], [1], [2], [3], [4,5]],
    [[0], [1], [2], [4], [3,5]],
    [[0], [1], [3], [4], [2,5]],
    [[0], [2], [3], [4], [1,5]],
    [[1], [2], [3], [4], [0,5]],
    [[0], [1], [2], [5], [3,4]],
    [[0], [1], [3], [5], [2,4]],
    [[0], [2], [3], [5], [1,4]],
    [[1], [2], [3], [5], [0,4]],
    [[0], [1], [4], [5], [2,3]],
    [[0], [2], [4], [5], [1,3]],
    [[1], [2], [4], [5], [0,3]],
    [[0], [3], [4], [5], [1,2]],
    [[1], [3], [4], [5], [0,2]],
    [[2], [3], [4], [5], [0,1]],

    # two interaction terms
    # 0123
    [[4], [5], [0,1], [2,3]],
    # 0124
    [[3], [5], [0,1], [2,4]],
    # 0125
    [[3], [4], [0,1], [2,5]],
    # 0134
    [[2], [5], [0,1], [3,4]],
    # 0135
    [[2], [4], [0,1], [3,5]],
    # 0145
    [[2], [3], [0,1], [4,5]],
    # 0213
    [[4], [5], [0,2], [1,3]],
    # 0214
    [[3], [5], [0,2], [1,4]],
    # 0215
    [[3], [4], [0,2], [1,5]],
    # 0234
    [[1], [5], [0,2], [3,4]],
    # 0235
    [[1], [4], [0,2], [3,5]],
    # 0245
    [[1], [3], [0,2], [4,5]],
    # 0312
    [[4], [5], [0,3], [1,2]],
    # 0314
    [[2], [5], [0,3], [1,4]],
    # 0315
    [[2], [4], [0,3], [1,5]],
    # 0324
    [[1], [5], [0,3], [2,4]],
    # 0325
    [[1], [4], [0,3], [2,5]],
    # 0345
    [[1], [2], [0,3], [4,5]],
    # 0412
    [[3], [5], [0,4], [1,2]],
    # 0413
    [[2], [5], [0,4], [1,3]],
    # 0415
    [[2], [3], [0,4], [1,5]],
    # 0423
    [[1], [5], [0,4], [2,3]],
    # 0425
    [[1], [3], [0,4], [2,5]],
    # 0435
    [[1], [2], [0,4], [3,5]],
    # 0512
    [[3], [4], [0,5], [1,2]],
    # 0513
    [[2], [4], [0,5], [1,3]],
    # 0514
    [[2], [3], [0,5], [1,4]],
    # 0523
    [[1], [4], [0,5], [2,3]],
    # 0524
    [[1], [3], [0,5], [2,4]],
    # 0534
    [[1], [2], [0,5], [3,4]],
    
    # triple interaction, assuming order doesn't matter
    [[0,1], [2,3], [4,5]],
    [[0,1], [2,4], [3,5]],
    [[0,1], [2,5], [3,4]],
    
    [[0,2], [1,3], [4,5]],
    [[0,2], [1,4], [3,5]],
    [[0,2], [1,5], [3,4]],
    
    [[0,3], [1,2], [4,5]],
    [[0,3], [1,4], [2,5]],
    [[0,3], [1,5], [2,4]],
    
    [[0,4], [1,2], [3,5]],
    [[0,4], [1,3], [2,5]],
    [[0,4], [1,5], [2,3]],
    
    [[0,5], [1,2], [3,4]],
    [[0,5], [1,3], [2,4]],
    [[0,5], [2,4], [1,3]],
]


###############################################################################
def _is_valid_model(dim, model):
    """
    A model is a list of lists, such as [[0], [1,2], [2,3,4], [5]].
    
    This function performs a set of consistency and correctness checks
    on a given model.
    """
    
    # a valid model must include all indices on the interval [0, dim)
    required_index_set = {i for i in range(dim)}

    # this is the set of all axes in the current model
    member_set = set()

    for member_list in model:
        for elt in member_list:
            assert elt in required_index_set

        # member lists cannot contain duplicate indices
        ctr = Counter(member_list)
        for v in ctr.values():
            assert 1 == v

        # update the set of all axes in the current model
        for elt in member_list:
            member_set.add(elt)

    # these sets must be equal for a valid model
    diff_set = required_index_set - member_set
    return 0 == len(diff_set)


###############################################################################
def _contingency_table(sample_arrays, bin_counts, weights=None):
    """
    Construct a contingency table from the sample arrays.
    
    The parameter 'sample_arrays' is a list of numpy arrays, all of identical
    length. Each array contains the samples for one of the variables.
    
    The parameter 'bin_counts' is a list of integers, one for each array in
    the sample_arrays list.
    
    ASSUMPTIONS:
        1. Samples are categorical with values represented by consecutive
           integers starting at zero.
        2. Each array of samples contains an identical number of elements.
        3. The "sample_arrays" and "bin_counts" arrays have the same number
           of elements.
    """
    
    # need an array for each bin
    assert len(sample_arrays) == len(bin_counts)

    # all arrays in samples have identical length (derived from dataframe)
    num_samples = len(sample_arrays[0])

    # ensure that the weights array has the same length as the sample arrays
    if weights is not None:
        assert len(weights) == num_samples
        
    # table of counts
    table = np.zeros(tuple(bin_counts))
    
    # fill table
    for q in range(num_samples):
        
        # build index tuple from the qth element of each sample array
        index_tup = tuple([sample_arrays[i][q] for i in range(len(sample_arrays))])
    
        w = 1
        if weights is not None:
            w = weights[q]
            
        table[index_tup] += w
        
    # check counts
    total = np.sum(table)
    if weights is None:
        assert np.isclose(total, num_samples)
    else:
        assert np.isclose(total, np.sum(weights))
    
    return table


###############################################################################
def _compute_marginals(model, table):
    """
    Compute marginals for raking using the index specifications in 'model',
    which is a list of lists.
            
    A valid model should ensure that each axis appears at least once, with no
    repeats in any sub-list.

    Examples of valid models:

        [[0], [1], [2], [3], [4], [5]]
        [[1], [2], [4], [5], [0,3]]
        [[0,1], [2,3], [4,5]]
    """
    
    # convert model to tuples for hashing
    tup_model = [tuple(item) for item in model]
        
    marginals = []
    for marginal_axes in tup_model:
        
        # determine axes to be summed over
        dim = len(table.shape)
        sum_axes = {q for q in range(dim)}
        for a in marginal_axes:
            # remove the axes for the current marginal
            sum_axes.remove(a)

        # if no axes left to sum over, the marginal is the contingency table itself
        if 0 == len(sum_axes):
            marginal = table.copy()
        else:
            # convert to tuple if more than one axis, integer if not
            if len(sum_axes) > 1:
                sum_axes = tuple(sum_axes)
            else:
                sum_axes = sum_axes.pop()

            # compute the marginal for this set of marginal axes
            marginal = np.sum(table, axis=sum_axes)
            
        marginals.append(marginal)
        
    assert len(marginals) == len(model)
    return marginals


###############################################################################
def _individual_weights(sample_arrays, raked, unraked, target_pop):
    """
    Compute weights for each individual in the source dataframe.
    """

    # each variable has an identical number of samples
    num_samples = len(sample_arrays[0])
        
    # weights for individual cells
    cell_weights = np.divide(raked, unraked)
    
    # set NaNs to zero (occur where unraked == 0, so get a divide by zero)
    cell_weights[np.isnan(cell_weights)] = 0
    assert not np.isnan(cell_weights).any()
        
    # compute weighted sample - should sum to target population
    weighted_sample = np.multiply(cell_weights, unraked)
    weighted_sample_pop = np.sum(weighted_sample)
    if not np.isclose(weighted_sample_pop, target_pop):
        # these weights are no good
        population_diff_pct = 100.0 * np.abs(target_pop - weighted_sample_pop) / target_pop
        print('\tWeighted population differs from target by {0:.2f}%, discarding...'.format(population_diff_pct))
        return None

    # extract weights from table using recoded data as coords    
    weights = []
    for q in range(num_samples):
        
        # build index tuple from the qth element of each sample array
        index_tup = tuple([sample_arrays[i][q] for i in range(len(sample_arrays))])
        
        # get the weight from this cell
        w = cell_weights[index_tup]
        weights.append(w)
        
    # need one weight for each row in the source dataframe
    assert len(weights) == num_samples
        
    # sum of the weights should sum to target population
    assert np.isclose(target_pop, np.sum(weights))
    return np.array(weights)


###############################################################################
def _min_cell_count(marginals):
    """
    Find the minimum cell count across all marginal arrays for a given model.
    """
   
    min_count = sys.maxsize     
    for q, marginal in enumerate(marginals):
        for cell_count in np.nditer(marginal):
            if 0 == cell_count:
                # skip any collapsed cells
                continue
            if cell_count < min_count:
                min_count = cell_count
    
    return min_count


###############################################################################
def rake(model, source_samples, target_samples, target_weights, bin_counts, target_population):
    """
    Parameters:
    
        source_samples: list of numpy arrays, one for each variable
        target_samples: list of numpy arrays, one for each variable
        target_weights: list of weights, one for each row in the target dataframe
    """

    # the number of variables to be simultaneously raked
    dim = len(bin_counts)
    
    if not _is_valid_model(dim, model):
        raise ValueError('Invalid model: {0}'.format(model))
    
    # must have one weight for each target sample
    #assert len(target_weights) == len(target_samples[0])
    if len(target_weights) != len(target_samples[0]):
        raise ValueError('Different numbers of target weights and samples')
    
    # compute the target population, either weighted or unweighted
    if target_weights is None:
        target_n = len(target_samples[0])
    else:
        target_n = np.sum(target_weights)
        assert target_n == target_population
    
    # generate the source table
    table_s = _contingency_table(source_samples, bin_counts)
        
    # generate the target table
    table_t = _contingency_table(target_samples, bin_counts, target_weights)
    
    # compute the target marginals for this raking model
    marginals_t = _compute_marginals(model, table_t)
    
    # find the minimum cell count in each *source* marginal
    marginals_s = _compute_marginals(model, table_s)
    smallest_cell = _min_cell_count(marginals_s)
    #print('\tmin source cell count: {0}'.format(smallest_cell))
    
    # make a copy of the unraked table, otherwise will be overwritten by ipfn
    unraked = table_s.copy()
    
    # do the iterations
    ipf_obj = ipfn.ipfn(original         = table_s,
                        aggregates       = marginals_t,
                        dimensions       = model,
                        convergence_rate = 1e-6,
                        max_iteration    = 5000,
                        verbose          = 2)
    raked, flag, info = ipf_obj.iteration()
    
    # a flag value of "1" means convergence
    if 1 == flag:

        # compute weights for each row of the source dataframe
        weights = _individual_weights(source_samples, raked, unraked, target_n)
        if weights is not None:
            
            # compute error metric: compute the marginals for the raked data and compare with target
            marginals_r = _compute_marginals(model, raked)

            # accumulate sum of squared residuals here
            sum_sq = 0.0
            for index, marginal in enumerate(marginals_r):

                # Compute the difference array between the result marginal and the target marginal.
                # All elements should be essentially zero.
                diff = marginal - marginals_t[index]

                # Compute the sum of the squares of each element of the diff matrix.
                # This is *elementwise* multiplication, not matrix multiplication.
                residual = np.sum(np.multiply(diff, diff))
                sum_sq += residual

            # Frobenius norm
            fnorm = np.sqrt(sum_sq)

            return weights, unraked, table_t, raked, fnorm, smallest_cell

    return None

