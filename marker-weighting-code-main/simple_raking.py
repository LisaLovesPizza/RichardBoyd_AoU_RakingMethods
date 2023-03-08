"""
This code computes weights for tuples from an input file so that the weighted
marginal distributions match those of the weighted PUMS census totals.

The input file must use the same coding scheme as the recoded PUMS USA data.
The recoding scheme can be found in the file "src/coding.py".

Usage example:

To rake a sample of 50k synthetic AllOfUs tuples to match the PUMS totals for states AL, GA, and MS:

    python ./simple_raking.py --sourcefile data/synthetic_aou_50k.csv --outfile data/synthetic_weighted.csv --codes "1,13,28"


"""

import os
import re
import ast
import sys
import math
import argparse
import numpy as np
import pandas as pd
from ipfn import ipfn
from collections import Counter, defaultdict

from src import plots, pdf, coding

# reference names for the variables to be raked, which also specifies
# the ordering of the variables
VARIABLES = coding.VARIABLES

# PDFs of raked data must differ from targets by less than this for success
# (this is the max diff for each bin of each PDF)
THRESHOLD = 0.01

# path to PUMS *recoded* USA file
PUMS_RECODED_FILE_PATH = 'data/pums_usa_2020_recoded.csv'

# Whether to use weighted PUMS counts, which are required for true population counts (PUMS 'PWGTP' variable).
# Set to None to use an unweighted target population.
PUMS_WEIGHT_COL = 'PWGTP'

# map the NAME of each variable to the NAME of the corresponding recoded PUMS variable
PUMS_COL_MAP = { 
    VARIABLES[0] : 'RACE_GROUPING',
    VARIABLES[1] : 'EDUCATION_GROUPING',
    VARIABLES[2] : 'INSURANCE_GROUPING',
    VARIABLES[3] : 'INCOME_GROUPING',
    VARIABLES[4] : 'SEX_AT_BIRTH',
}


# map the NAME of each variable to the NAME of the corresponding recoded AllOfUs variable
AOU_COL_MAP = {    
    VARIABLES[0] : 'race',
    VARIABLES[1] : 'education',
    VARIABLES[2] : 'insurance',
    VARIABLES[3] : 'income',
    VARIABLES[4] : 'sex',    
}

COL_NAME_MAP = AOU_COL_MAP


###############################################################################
def table_5d(i, j, k, l, m, samples, bin_counts, weights=None):
    """
    Construct a 5d contingency table from samples of five source or target variables.
    
    ASSUMPTIONS:
        1. Samples are categorical with values represented by a contiguous group of integers starting at zero.
        2. Each array of samples contains the same number of elements.
    """
    
    samples_i = samples[i]
    samples_j = samples[j]
    samples_k = samples[k]
    samples_l = samples[l]
    samples_m = samples[m]
    
    # each array must contain the same number of elements
    n = len(samples_i)
    assert len(samples_j) == n
    assert len(samples_k) == n
    assert len(samples_l) == n  
    assert len(samples_m) == n
    if weights is not None:
        assert len(weights) == n
        
    # table of counts
    table = np.zeros((bin_counts[i], bin_counts[j], bin_counts[k], bin_counts[l], bin_counts[m]))
    
    # fill table
    for q in range(n):
        i = samples_i[q]
        j = samples_j[q]
        k = samples_k[q]
        l = samples_l[q]
        m = samples_m[q]
        
        w = 1
        if weights is not None:
            w = weights[q]
            
        table[i,j,k,l,m] += w
        
    # check counts
    total = np.sum(table)
    
    if weights is None:
        assert np.isclose(total, n)
    else:
        assert np.isclose(total, np.sum(weights))
    
    return table


###############################################################################
def marginals_5d(model, table):
    """
    Compute marginals for 5D raking using the index specifications in 'model', which is a list of lists.
    
    The valid elements of 'model' are:
    
        [0] : 1D marginal along axis 0 (sum over axes (1,2,3,4))
        [1] : 1D marginal along axis 1 (sum over axes (0,2,3,4))
        [2] : 1D marginal along axis 2 (sum over axes (0,1,3,4))
        [3] : 1D marginal along axis 3 (sum over axes (0,1,2,4))
        [4] : 1D marginal along axis 4 (sum over axes (0,1,2,3))
        
        [0,1] : 2D marginal for axes 01 (sum over axes (2,3,4))
        [0,2] : 2D marginal for axes 02 (sum over axes (1,3,4))
        [0,3] : 2D marginal for axes 03 (sum over axes (1,2,4))
        [0,4] : 2D marginal for axes 04 (sum over axes (1,2,3))
        [1,2] : 2D marginal for axes 12 (sum over axes (0,3,4))
        [1,3] : 2D marginal for axes 13 (sum over axes (0,2,4))
        [1,4] : 2D marginal for axes 14 (sum over axes (0,2,3))
        [2,3] : 2D marginal for axes 23 (sum over axes (0,1,4))
        [2,4] : 2D marginal for axes 24 (sum over axes (0,1,3))
        [3,4] : 2D marginal for axes 34 (sum over axes (0,1,2))
        
        [0,1,2] : 3D marginal for axes 012 (sum over axes (3,4))
        [0,1,3] : 3D marginal for axes 013 (sum over axes (2,4))
        [0,1,4] : 3D marginal for axes 014 (sum over axes (2,3))
        [0,2,3] : 3D marginal for axes 023 (sum over axes (1,4))
        [0,2,4] : 3D marginal for axes 024 (sum over axes (1,3))
        [0,3,4] : 3D marginal for axes 034 (sum over axes (1,2))
        [1,2,3] : 3D marginal for axes 123 (sum over axes (0,4))
        [1,2,4] : 3D marginal for axes 124 (sum over axes (0,3))
        [1,3,4] : 3D marginal for axes 134 (sum over axes (0,2))
        [2,3,4] : 3D marginal for axes 234 (sum over axes (0,1))
        
        [0,1,2,3] : 4D marginal for axes 0123 (sum over axis 4)
        [0,1,2,4] : 4D marginal for axes 0124 (sum over axis 3)
        [0,1,3,4] : 4D marginal for axes 0134 (sum over axis 2)
        [0,2,3,4] : 4D marginal for axes 0234 (sum over axis 1)
        [1,2,3,4] : 4D marginal for axes 1234 (sum over axis 0)
        
        
    A valid model should ensure that each axis appears at least once.
    """
    
    # sum axes for each possibility
    SUM_AXES = {
        (0,) : (1,2,3,4),
        (1,) : (0,2,3,4),
        (2,) : (0,1,3,4),
        (3,) : (0,1,2,4),
        (4,) : (0,1,2,3),
        (0,1) : (2,3,4),
        (0,2) : (1,3,4),
        (0,3) : (1,2,4),
        (0,4) : (1,2,3),
        (1,2) : (0,3,4),
        (1,3) : (0,2,4),
        (1,4) : (0,2,3),
        (2,3) : (0,1,4),
        (2,4) : (0,1,3),
        (3,4) : (0,1,2),
        (0,1,2) : (3,4),
        (0,1,3) : (2,4),
        (0,1,4) : (2,3),
        (0,2,3) : (1,4),
        (0,2,4) : (1,3),
        (0,3,4) : (1,2),
        (1,2,3) : (0,4),
        (1,2,4) : (0,3),
        (1,3,4) : (0,2),
        (2,3,4) : (0,1),
        (0,1,2,3) : 4,
        (0,1,2,4) : 3,
        (0,1,3,4) : 2,
        (0,2,3,4) : 1,
        (1,2,3,4) : 0,
    }
    
    # convert model to tuples for hashing
    tup_model = [tuple(item) for item in model]
    
    # check the entries in model
    expected = {
        (0,), (1,), (2,), (3,),(4,),
        (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4),
        (0,1,2), (0,1,3), (0,1,4), (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4), (2,3,4),
        (0,1,2,3), (0,1,2,4), (0,1,3,4), (0,2,3,4), (1,2,3,4),
    }
    for item in tup_model:
        assert item in expected
        
    marginals = []
    for marginal_axes in tup_model:
        # look up the axes to be summed over
        sum_axes = SUM_AXES[marginal_axes]
        marginal = np.sum(table, axis=sum_axes)
        marginals.append(marginal)
        
    assert len(marginals) == len(model)
    return marginals


###############################################################################
def sample_weights(i, j, k, l, m, samples, raked, unraked, target_pop):
    """
    Compute weights for each individual in the source dataframe.
    """
    
    samples_i = samples[i]
    samples_j = samples[j]
    samples_k = samples[k]
    samples_l = samples[l]
    samples_m = samples[m]    
    
    # each array must contain the same number of elements
    n = len(samples_i)
    assert len(samples_j) == n
    assert len(samples_k) == n
    assert len(samples_l) == n
    assert len(samples_m) == n
        
    # weights for individual cells
    cell_weights = np.divide(raked, unraked)
    
    # set NaNs to zero (occur where unraked == 0)
    cell_weights[np.isnan(cell_weights)] = 0
    assert not np.isnan(cell_weights).any()
        
    # compute weighted sample - should sum to target population
    weighted_sample = np.multiply(cell_weights, unraked)
    weighted_sample_pop = np.sum(weighted_sample)
    if not np.isclose(weighted_sample_pop, target_pop):
        # these weights are no good
        print('\tPopulations differ...discarding')
        return None
    
    # extract weights from table using recoded data as coords
    weights = []
    for q in range(n):
        i = samples_i[q]
        j = samples_j[q]
        k = samples_k[q]
        l = samples_l[q]
        m = samples_m[q]
        w = cell_weights[i,j,k,l,m]
        weights.append(w)
        
    # need one weight for each row in the source dataframe
    assert len(weights) == n
        
    # sum of the weights should sum to target population
    assert np.isclose(target_pop, np.sum(weights))
    return np.array(weights)


###############################################################################
def rake_5d(model, i, j, k, l, m, source_samples, target_samples, target_weights, bin_counts):
    """
    Perform raking on the five variables at indices i, j, k, l, and m.
    
    SOURCE_SAMPLES is a list of identical-length sample arrays for each source variable.
    TARGET_SAMPLES is the same for the target variables.
    """   
    
    # number of source and target samples
    source_n = len(source_samples[i])
    
    if target_weights is None:
        target_n = len(target_samples[j])
    else:
        target_n = np.sum(target_weights)
    
    # generate the source table for the variables at indices i, j, k, and l
    table_s = table_5d(i,j,k,l,m, source_samples, bin_counts)
        
    # generate the target table for the variables at indices i, j, k, and l
    table_t = table_5d(i,j,k,l,m, target_samples, bin_counts, target_weights)
    
    # compute the target marginals
    marginals_t = marginals_5d(model, table_t)
    
    # make a copy of the unraked table, otherwise will be overwritten by ipfn
    unraked = table_s.copy()
    
    # perform 5D raking
    ipf_obj = ipfn.ipfn(original         = table_s,
                        aggregates       = marginals_t,
                        dimensions       = model,
                        convergence_rate = 1e-6,
                        max_iteration    = 5000,
                        verbose          = 2)
    raked, flag, info = ipf_obj.iteration()
    
    if 1 == flag:
        
        # convergence, but weights may still be bad

        # compute weights for each row of the source dataframe
        weights = sample_weights(i,j,k,l,m, source_samples, raked, unraked, target_n)
        if weights is not None:
            
            # check by computing the marginals for the result array and comparing with the target
            marginals_r = marginals_5d(model, raked)

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
        
            return weights, unraked, table_t, raked, fnorm
        
    return None, None, None, None, None


###############################################################################
def run(model, source_df, pums_df, use_target_weights=True):

    #
    # setup target data
    #
        
    # get the names of the target columns in the same order as the VARIABLES array
    ordered_target_cols = [PUMS_COL_MAP[var_name] for var_name in VARIABLES]

    # samples taken in order of the variables in VARIABLES
    TARGET_SAMPLES = [np.array(pums_df[col].values) for col in ordered_target_cols]    

    # set TARGET_WEIGHTS to None to rake to an unweighted target population
    if use_target_weights:
        TARGET_WEIGHTS = np.array(pums_df[PUMS_WEIGHT_COL].values)
        TARGET_POPULATION = np.sum(TARGET_WEIGHTS)
        print('Target population (weighted): {0}'.format(TARGET_POPULATION))
    else:
        TARGET_WEIGHTS = None
        TARGET_POPULATION = len(TARGET_SAMPLES[0])    
        print('Target population (unweighted): {0}'.format(TARGET_POPULATION))


    #
    # setup source data
    #

    # get the names of the source dataframe columns in the order matching the variables
    ordered_source_cols = [COL_NAME_MAP[var_name] for var_name in VARIABLES]

    # build a list of np.arrays containing the data for each col
    SOURCE_SAMPLES = [np.array(source_df[col].values) for col in ordered_source_cols]

    print('Sample population: {0}'.format(len(SOURCE_SAMPLES[0])))


    #
    # compute the number of bins required for each variable
    #

    # bin counts in order of the variables
    BIN_COUNTS = []

    for i, var in enumerate(VARIABLES):
        BIN_COUNTS.append(coding.BIN_COUNTS[var])

    # maximum-length variable name, used for prettyprinting
    maxlen = max([len(var_name) for var_name in VARIABLES])

    print('Bin counts: ')
    for i in range(len(VARIABLES)):
        print('{0:>{2}} : {1}'.format(VARIABLES[i], BIN_COUNTS[i], maxlen))
    
    
    #
    # rake the data
    #

    # do 5D raking on all variables
    i,j,k,l,m = (0,1,2,3,4)
    weights, unraked, target, raked, fnorm = rake_5d(model, i,j,k,l,m,
                                                     SOURCE_SAMPLES,
                                                     TARGET_SAMPLES, TARGET_WEIGHTS,
                                                     BIN_COUNTS)
    if fnorm is None:
        print('\n*** No acceptable raking model was found. ***')
    else:
        # compute PDFs
        target_pdfs = {}
        source_raked_pdfs = {}
        source_unraked_pdfs = {}

        # target pdfs
        for q in range(len(VARIABLES)):
            target_pdfs[q] = pdf.to_pdf(BIN_COUNTS[q], TARGET_SAMPLES[q], weights=TARGET_WEIGHTS)

        # unraked source pdfs
        for q in range(len(VARIABLES)):
            source_unraked_pdfs[q] = pdf.to_pdf(BIN_COUNTS[q], SOURCE_SAMPLES[q], weights=None)

        # raked source pdfs
        for q in range(len(VARIABLES)):
            source_raked_pdfs[q] = pdf.to_pdf(BIN_COUNTS[q], SOURCE_SAMPLES[q], weights=weights)

        print('Raking model: {0}'.format(model))
        print('Raking on variables {0}, {1}, {2}, {3}, and {4} '.format(VARIABLES[i], VARIABLES[j],
                                                                        VARIABLES[k], VARIABLES[l],
                                                                        VARIABLES[m]))    

        # diff between raked and target pdfs
        print('\nPDF diffs after raking on {0}, {1}, {2}, {3}, and {4}:\n'.format(VARIABLES[i],
                                                                                  VARIABLES[j],
                                                                                  VARIABLES[k],
                                                                                  VARIABLES[l],
                                                                                  VARIABLES[m]))
        diffs = []
        for q in range(len(VARIABLES)):
            diff = source_raked_pdfs[q] - target_pdfs[q]
            diffs.append(diff)
            print('{0:>{2}} : {1}'.format(VARIABLES[q], 
                                          np.array_str(diff, precision=5, suppress_small=True),
                                          maxlen))

        # check the diff vectors for the presence of any diff > 0.01 (i.e. 1%)
        all_ok = True
        for diff_vector in diffs:
            if np.any(diff_vector > THRESHOLD):
                all_ok = False

        # sum of the weights
        sum_of_weights = np.sum(weights)
        print('Sum of the weights : {0:.3f}'.format(sum_of_weights))
        print('  Population total : {0:.3f}'.format(TARGET_POPULATION))
        print('        Difference : {0:.3f}'.format(TARGET_POPULATION - sum_of_weights))
        print('\nRaked PDFs differ from target PDFs by less than {0}%: {1}'.format(int(THRESHOLD * 100),
                                                                                   all_ok))

        return all_ok, source_df, weights


###############################################################################
def to_state_code_list(code_arg_string):
    """
    """

    assert code_arg_string is not None
    
    # remove quotes if present
    code_str = re.sub(r'["]', '', code_arg_string)

    items = code_str.split(',')
    
    state_codes = []
    for item in items:
        if item.isdigit():
            # single state code
            code = int(item.strip())
            state_codes.append(code)
        else:
            # must be a range
            assert '-' in item
            first, last = item.split('-')
            first = int(first.strip())
            last  = int(last.strip())
            for code in range(first, last+1):
                state_codes.append(code)

    return state_codes

    
###############################################################################        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = 'Rake AllOfUs data in 5D to match PUMS targets.'
    )

    # parser.add_argument('--target',
    #                     type=str,
    #                     dest='target_file',
    #                     required=True,
    #                     help='path to recoded PUMS file')

    parser.add_argument('--outfile',
                        dest='outfile',
                        required=True,
                        help='output file name')

    parser.add_argument('--sourcefile',
                        type=str,
                        dest='source_file',
                        required=True,
                        help='path to recoded AllOfUs file')

    # parser.add_argument('--no_weights',
    #                     action='store_true',
    #                     help='do NOT use target weights')

    parser.add_argument('--model',
                        type=str,
                        dest='model',
                        help='raking model (list of lists, default is [[0],[1],[2],[3],[4]]')

    parser.add_argument('--codes',
                        type=str,
                        dest='codes',
                        help='Quoted, comma-separated list of PUMS state codes, i.e. "13, 17, 22-25". Omit to use data from all states.')
                        
    args = parser.parse_args()

    # target_file = args.target_file
    # if not os.path.isfile(target_file):
    #     print('*** Target file not found: "{0}"'.format(target_file))
    #     sys.exit(-1)

    source_file = args.source_file
    if not os.path.isfile(source_file):
        print('*** Source file not found: "{0}"'.format(source_file))
        sys.exit(-1)

    output_file = args.outfile

    # default model is to rake each variable to its 1D marginal
    model = [[0],[1],[2],[3],[4]]
    if args.model is not None and args.model:
        # convert string to list
        model = ast.literal_eval(args.model)

    state_codes = None
    if args.codes is not None and args.codes:
        state_codes = to_state_code_list(args.codes)

    # target file is always the PUMS recoded file for the entire USA + DC
    target_file = PUMS_RECODED_FILE_PATH
    
    use_target_weights = True
    # if args.no_weights:
    #     use_target_weights = False
    #     print('Target weights will NOT be used.')

    print('\n   ARGUMENTS')
    print(' Source file : {0}'.format(source_file))
    print(' Output file : {0}'.format(target_file))
    print('       Model : {0}'.format(model))
    print(' State codes : {0}'.format(state_codes))
    print()
        
    #
    # load target data
    #
    
    print('Loading target file "{0}"...'.format(target_file))
    pums_usa_df = pd.read_csv(target_file)

    # check that all state codes represent actual states (or DC)
    if state_codes is not None:
        for code in state_codes:
            assert code >= pums_usa_df['ST'].min()
            assert code <= pums_usa_df['ST'].max()

        # extract all rows for these states
        pums_df = pums_usa_df.loc[pums_usa_df['ST'].isin(state_codes)]

    # keep specified data cols and weight col, drop all others
    keep_cols = [col for col in PUMS_COL_MAP.values()]
    keep_cols.append(PUMS_WEIGHT_COL)
    pums_df = pums_df[keep_cols]
    pums_df = pums_df.reset_index(drop=True)

    
    #
    # load source data
    #
    
    print('Loading source file: "{0}"'.format(source_file))
    source_df = pd.read_csv(source_file)
    

    #
    # rake the data
    #
    
    success, source_df, weights = run(model,
                                      source_df,
                                      pums_df,
                                      use_target_weights)
    if not success:
        print('PDFs differed by more than the expected tolerance, exiting...')
        sys.exit(0)
    
    else:

        #
        # write weighted source data to disk
        #

        final_df = source_df.assign(weight = weights)
        
        # write output file
        final_df.to_csv(output_file, index=False)
        print('Wrote file "{0}".'.format(output_file))        
    
