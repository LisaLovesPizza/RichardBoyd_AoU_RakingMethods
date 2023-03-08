"""
Performs iterative proportional fitting (also called "raking") on N variables.
"""

import os
import sys
import numpy as np
import pandas as pd

from ipfn import ipfn
from collections import Counter

from . import models


# map of the number of variables to list of raking models
_MODEL_MAP = {
     2 : models.MODELS_2,
     3 : models.MODELS_3,
     4 : models.MODELS_4,
     5 : models.MODELS_5,
     6 : models.MODELS_6,
     7 : models.MODELS_7,
     8 : models.MODELS_8,
     9 : models.MODELS_9,
    10 : models.MODELS_10,
}


# all state variables needed for raking
class State(object):    

    def __init__(self):
        self._rake_map = None
        self._rakevars = None
        self._source_col_map = None
        self._target_col_map = None
        self._bin_counts = None
        
        self._source_samples = None
        self._target_samples = None
        self._target_weights = None
        self._target_population = None
        

    @property    
    def target_population(self):
        return self._target_population

    @property
    def target_weights(self):
        return self._target_weights

    @property
    def target_samples(self):
        return self._target_samples

    @property
    def source_samples(self):
        return self._source_samples

    @property
    def bin_counts(self):
        return self._bin_counts
        
        
    def initialize(self, rake_data, bin_counts):
        """
        Initialize all state variables.
        """

        # map of variable enum to (source_col, target_col) tuple
        self._rake_map = {enumvar:(source_col, target_col) for enumvar, source_col, target_col in rake_data}

        # list of enumerated variables used for raking
        self._rakevars = [tup[0] for tup in rake_data]

        # map of variable enum to source file col name
        self._source_col_map = {self._rakevars[i]:rake_data[i][1] for i in range(len(self._rakevars))}

        # map of variable enum to PUMS file col name
        self._target_col_map = {self._rakevars[i]:rake_data[i][2] for i in range(len(self._rakevars))}

        # number of bins for each variable
        self._bin_counts = bin_counts
        
        self._source_samples = None
        self._target_samples = None
        self._target_weights = None
        self._target_population = None


    def set_source_samples(self, rake_data, source_df):
        """
        """

        # get the names of the source dataframe columns in the order matching the variables
        ordered_source_cols = [self._source_col_map[enumvar] for enumvar in self._rakevars]
        print('\nSource columns, in order: ')
        for c in ordered_source_cols:
            print('\t{0}'.format(c))
        
        # list of np.arrays for the source samples
        self._source_samples = [np.array(source_df[col].values) for col in ordered_source_cols]

        # source and target need the same number of variables
        if self._target_samples is not None:
            assert len(self._target_samples) == len(self._source_samples)
        

    def set_target_samples(self, rake_data, target_df, target_weight_col):
        """
        """

        # get the names of the target columns in the same order as the RAKEVARS array
        ordered_target_cols = [self._target_col_map[enumvar] for enumvar in self._rakevars]
        print('\nTarget columns, in order: ')
        for c in ordered_target_cols:
            print('\t{0}'.format(c))

        # list of np.arrays for the target samples
        self._target_samples = [np.array(target_df[col].values) for col in ordered_target_cols]
        
        # set TARGET_WEIGHTS to None to rake to an unweighted target population
        if target_weight_col is not None:
            self._target_weights = np.array(target_df[target_weight_col].values)
            self._target_population = np.sum(self._target_weights)
            print('\nTarget population (weighted): {0}'.format(self._target_population))
        else:
            self._target_weights = None
            self._target_population = len(self._target_samples[0])
            print('\nTarget population (unweighted): {0}'.format(self._target_population))

        # source and target need the same number of variables
        if self._source_samples is not None:
            assert len(self._source_samples) == len(self._target_samples)


# state instance variable        
_state = State()


# these functions provide access to readonly properties
def target_population():
    return _state.target_population

def bin_counts():
    return _state.bin_counts

def source_samples():
    return _state.source_samples

def target_samples():
    return _state.target_samples


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
def _scoring_metric(weights_m, avg_wt):
    """
    David Marker's modification of the variability score.
    """
    
    # will sum all weights >= this fraction * max_wt
    MAX_WT_FRAC = 0.9
    
    # maximum weight for this model
    wmax = np.max(weights_m)
    
    # cutoff value; compute the sum of all weights >= this cutoff
    w_cutoff = MAX_WT_FRAC * wmax
    
    samples = []
    for w in weights_m:
        if w >= w_cutoff:
            samples.append(w)
            
    w_sum = np.sum(samples)
    
    # w_sum represents this percent of the weighted total population
    pct = w_sum / _state.target_population
     
    # the score is the product of the max wt and the fraction above the cutoff,
    # divided by the avg weight
    score = wmax * pct / avg_wt
    return score
    

###############################################################################
def _rake_single_model(model,
                       source_samples,
                       target_samples,
                       target_weights,
                       bin_counts,
                       target_population):
    """
    Parameters:

        model             : the single raking model to run
        source_samples    : list of numpy arrays, one for each variable
        target_samples    : list of numpy arrays, one for each variable
        target_weights    : list of weights, one for each row in the target dataframe
        bin_counts        : number of bins for each variable
        target_population : sum of the target weights
    """

    # the number of variables to be simultaneously raked
    dim = len(bin_counts)
    
    if not _is_valid_model(dim, model):
        raise ValueError('Invalid model: {0}'.format(model))
    
    # compute the target population, either weighted or unweighted
    if target_weights is None:
        target_n = len(target_samples[0])
    else:
        # must have one weight for each target sample
        if len(target_weights) != len(target_samples[0]):
            raise ValueError('Different numbers of target weights and samples')
        
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


###############################################################################
def rake(rake_data,
         source_df,
         target_df,
         bin_counts,
         min_cell_size = 50,
         target_weight_col=None,
         model_list=None):
    """
    list of models, None means to run all for the given dimension
    """

    if len(bin_counts) != len(rake_data):
        raise ValueError('raking.rake: lengths of bin_counts and rake_data lists must match')
    
    # check source and target dataframes
    #for enumvar, src_col, tgt_col in rake_data:
    for q, tup in enumerate(rake_data):
        bin_count = bin_counts[q]
        enumvar, src_col, tgt_col = tup
        
        # the source and target columns must exist in the respective dataframes
        if not src_col in source_df:
            raise ValueError('raking.rake: column "{0}" not in source dataframe'.format(src_col))
        if not tgt_col in target_df:
            raise ValueError('raking.rake: column "{0}" not in target dataframe'.format(tgt_col))

        # cannot have any missing values
        src_missing = source_df[src_col].isna().sum()
        if src_missing != 0:
            raise ValueError('raking.rake: source dataframe column "{0}" has mising values'.format(src_col))
        tgt_missing = target_df[tgt_col].isna().sum()
        if tgt_missing != 0:
            raise ValueError('raking.rake: target dataframe column "{0}" has missing values'.format(tgt_col))
        
        src_values = source_df[src_col].values
        min_src = np.min(src_values)
        max_src = np.max(src_values)
        if min_src < 0 or min_src >= bin_count or max_src < 0 or max_src >= bin_count:
            raise ValueError('raking.rake: source dataframe contains values exceeding the bin values [{0}, {1}]'.
                             format(0, bin_count-1))
        
        tgt_values = target_df[tgt_col].values
        min_tgt = np.min(tgt_values)
        max_tgt = np.max(tgt_values)
        if min_tgt < 0 or min_tgt >= bin_count or max_tgt < 0 or max_tgt >= bin_count:
            raise ValueError('raking.rake: target dataframe contains values outside the bin range [{0}, {1}]'.
                             format(0, bin_count-1))

    # initialize all state variables
    _state.initialize(rake_data, bin_counts)
    _state.set_source_samples(rake_data, source_df)
    _state.set_target_samples(rake_data, target_df, target_weight_col)

    # get the models to run
    if model_list is None:
        # run all raking models for the number of variables
        num_vars = len(rake_data)
        if not num_vars in _MODEL_MAP:
            raise ValueError('raking.rake: unsupported number of variables ({0})'.format(num_vars))
        
        model_list = _MODEL_MAP[num_vars]
        
    
    # list of (max_wt, smallest_cell_size, model_index) for each convergent model
    data = []

    # map of model_index => weights for that model
    weight_map = {}

    # main loop over models
    for model_index, model in enumerate(model_list):

        print('\n[{0}/{1}]\tModel {2}:\n'.format(model_index+1, len(model_list), model))        

        raking_result_tuple = _rake_single_model(model,
                                                 _state.source_samples,
                                                 _state.target_samples,
                                                 _state.target_weights,
                                                 _state.bin_counts,
                                                 _state.target_population)

        if raking_result_tuple is None:
            print('\n*** No acceptable raking model was found. ***')
        else:
            weights_m, unraked_m, target_m, raked_m, fnorm_m, smallest_cell_m = raking_result_tuple
            if smallest_cell_m < min_cell_size:
                print('\tSource marginal cell size {0} less than MIN_CELL_SIZE {1}, discarding...'.
                      format(smallest_cell_m, min_cell_size))
            else: 
                wmax = np.max(weights_m)
                print('\tMax weight: {0:.3f}'.format(wmax))
                data.append( (wmax, smallest_cell_m, model_index))
                weight_map[model_index] = weights_m    


    #
    # score the results
    #
    
    # The average weight is used to standardize the scoring metric.
    sample_count = len(_state.source_samples[0])
    avg_wt = _state.target_population / sample_count

    # compute a score for each model
    scores = []
    for wmax, smallest_cell, model_index in data:
        model = model_list[model_index]
        weights_m = weight_map[model_index]
        score = _scoring_metric(weights_m, avg_wt)
        scores.append(score)


    # collect results in a new dataframe
    result_df = pd.DataFrame(data=data, columns=['Max Weight', 'Min Cell', 'Model Index'])
    result_df = result_df.assign(**{'Score':scores})
    model_details = result_df['Model Index'].map(lambda q: model_list[q])
    result_df = result_df.assign(**{'Model':model_details})
    
    # sort models by increasing max_weight
    result_df = result_df.sort_values(by=['Max Weight'])
    result_df = result_df.reset_index(drop=True)

    return result_df, weight_map
