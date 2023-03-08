"""
This code generates a file containing all raking models with pairwise interactions
for up to ten variables.

The file is written to the same directory.

Usage:

    python ./modelgen.py

"""

import os
import sys
import itertools

# indentation of four spaces
_I4 = '    '

_OUTFILE_NAME = 'models.py'


###############################################################################
def _to_list(tup_model):
    """
    Convert the model from a list of tuples into a list of lists.
    """
    
    list_rep = []
    for item in tup_model:
        if tuple == type(item):
            # interaction term
            lr = list(item)
        else:
            # integer, single axis
            lr = [item]
        list_rep.append(lr)

    return list_rep


###############################################################################
def _dedup(model_list):
    """
    Return the unique models as a list of tuples. The tuples are returned
    in sorted order.
    """

    uniques = set()
    for model in model_list:
        uniques.add(model)

    unique_model_list = sorted(uniques)
    return unique_model_list


###############################################################################
def _interaction1(n):
    """
    Generate all raking models for n variables containing a single
    pairwise interaction term.
    """

    # indices of the n variables
    index_set = set([q for q in range(n)])

    model_list = []
    
    # all pairwise interactions (n indices taken two at a time)
    # s1 is a set of two-element tuples
    s1 = itertools.combinations(index_set, 2)
    for p1 in s1:
        # remove index pair p1 from the index set to get univariate indices
        univariate_indices = index_set - set(p1)

        # build the model: univariate portion first, then interaction
        model_u = []
        if len(univariate_indices) > 0:
            model_u = sorted([(u) for u in univariate_indices])
            
        model_i = [p1]
        model = model_u + model_i
        model_list.append(tuple(model))

    unique_model_list = _dedup(model_list)
    return unique_model_list
        

###############################################################################
def _interaction2(n):
    """
    Generate all raking models for n variables containing two pairwise
    interaction terms.
    """

    # indices of the n variables
    index_set = set([q for q in range(n)])

    model_list = []

    # first pairwise interaction (n indices taken two at a time)
    # s1 is a set of two-element index tuples
    s1 = itertools.combinations(index_set, 2)
    for p1 in s1:
        # remove index pair p1 from the index set
        # choose next pair from these indices
        remaining = index_set - set(p1)
        # s2 is also a set of two-element index tuples
        s2 = itertools.combinations(remaining, 2)
        for p2 in s2:
            univariate_indices = remaining - set(p2)

            # build the model: univariate portion first, then interactions
            model_u = []
            if len(univariate_indices) > 0:
                model_u = sorted([(u) for u in univariate_indices])
                
            model_i = sorted([p1, p2])
            model = model_u + model_i
            model_list.append(tuple(model))

    unique_model_list = _dedup(model_list)
    return unique_model_list


###############################################################################
def _interaction3(n):
    """
    Generate all raking models for n variables containing three pairwise
    interaction terms.
    """

    # indices of the n variables
    index_set = set([q for q in range(n)])

    model_list = []

    # first pairwise interaction (n indices taken two at a time)
    # s1 is a set of two-element index tuples
    s1 = itertools.combinations(index_set, 2)
    for p1 in s1:
        # remove index pair p1 from the index set
        # choose next pair from these indices
        remaining = index_set - set(p1)
        # s2 is also a set of two-element index tuples
        s2 = itertools.combinations(remaining, 2)
        for p2 in s2:
            remaining2 = remaining - set(p2)
            s3 = itertools.combinations(remaining2, 2)
            for p3 in s3:
                univariate_indices = remaining2 - set(p3)

                # build the model: univariate portion first, then interactions
                model_u = []
                if len(univariate_indices) > 0:
                    model_u = sorted([(u) for u in univariate_indices])
                    
                model_i = sorted([p1, p2, p3])
                model = model_u + model_i
                model_list.append(tuple(model))

    unique_model_list = _dedup(model_list)
    return unique_model_list


###############################################################################
def _interaction4(n):
    """
    Generate all raking models for n variables containing four pairwise
    interaction terms.
    """

    # indices of the n variables
    index_set = set([q for q in range(n)])

    model_list = []

    # first pairwise interaction (n indices taken two at a time)
    # s1 is a set of two-element index tuples
    s1 = itertools.combinations(index_set, 2)
    for p1 in s1:
        # remove index pair p1 from the index set
        # choose next pair from these indices
        remaining = index_set - set(p1)
        # the next set of interactin pairs comes from the remaining indices,
        # taken two at a time
        s2 = itertools.combinations(remaining, 2)
        for p2 in s2:
            # p2 is the second interaction pair
            remaining2 = remaining - set(p2)
            s3 = itertools.combinations(remaining2, 2)
            for p3 in s3:
                # p3 is the third interaction pair
                remaining3 = remaining2 - set(p3)
                s4 = itertools.combinations(remaining3, 2)
                for p4 in s4:
                    # p4 is the fourth interaction pair                
                    univariate_indices = remaining3 - set(p4)

                    # build the model: univariate portion first, then interactions
                    model_u = []
                    if len(univariate_indices) > 0:
                        model_u = sorted([(u) for u in univariate_indices])

                    model_i = sorted([p1, p2, p3, p4])
                    model = model_u + model_i
                    model_list.append(tuple(model))

    unique_model_list = _dedup(model_list)
    return unique_model_list


###############################################################################
def _interaction5(n):
    """
    Generate all raking models for n variables containing five pairwise
    interaction terms.
    """

    # indices of the n variables
    index_set = set([q for q in range(n)])

    model_list = []

    # first pairwise interaction (n indices taken two at a time)
    # s1 is a set of two-element index tuples
    s1 = itertools.combinations(index_set, 2)
    for p1 in s1:
        # remove index pair p1 from the index set
        # choose next pair from these indices
        remaining = index_set - set(p1)
        # the next set of interactin pairs comes from the remaining indices,
        # taken two at a time
        s2 = itertools.combinations(remaining, 2)
        for p2 in s2:
            # p2 is the second interaction pair
            remaining2 = remaining - set(p2)
            s3 = itertools.combinations(remaining2, 2)
            for p3 in s3:
                # p3 is the third interaction pair
                remaining3 = remaining2 - set(p3)
                s4 = itertools.combinations(remaining3, 2)
                for p4 in s4:
                    # p4 is the fourth interaction pair
                    remaining4 = remaining3 - set(p4)
                    s5 = itertools.combinations(remaining4, 2)
                    for p5 in s5:
                        # p5 is the fifth interaction pair                    
                        univariate_indices = remaining4 - set(p5)

                        # build the model: univariate portion first, then interactions
                        model_u = []
                        if len(univariate_indices) > 0:
                            model_u = sorted([(u) for u in univariate_indices])

                        model_i = sorted([p1, p2, p3, p4, p5])
                        model = model_u + model_i
                        model_list.append(tuple(model))

    unique_model_list = _dedup(model_list)
    return unique_model_list


###############################################################################
def _print_models(model_list, append_comma=True):
    """
    Print models to stdout. Optionally append a comma to each so that the
    output can be cut and pastsed into python code.
    """

    for model in model_list:
        list_rep = _to_list(model)
        if append_comma:
            print('{0},'.format(list_rep))
        else:
            print(list_rep)
            

###############################################################################
def _write_univariate(n, outfile):
    
    outfile.write('{0}# univariate\n'.format(_I4))
    univariate_model = [[q] for q in range(n)]
    outfile.write('{0}{1},\n'.format(_I4, univariate_model))

def _write_single_interaction(n, outfile):
    models = _interaction1(n)
    outfile.write('{0}# single interaction ({1})\n'.format(_I4, len(models)))
    for model in models:
        list_rep = _to_list(model)
        outfile.write('{0}{1},\n'.format(_I4, list_rep))

def _write_dual_interaction(n, outfile):
    models = _interaction2(n)
    outfile.write('{0}# dual interaction ({1})\n'.format(_I4, len(models)))
    for model in models:
        list_rep = _to_list(model)
        outfile.write('{0}{1},\n'.format(_I4, list_rep))
        
def _write_three_interaction(n, outfile):
    models = _interaction3(n)
    outfile.write('{0}# triple interaction ({1})\n'.format(_I4, len(models)))
    for model in models:
        list_rep = _to_list(model)
        outfile.write('{0}{1},\n'.format(_I4, list_rep))

def _write_four_interaction(n, outfile):
    models = _interaction4(n)
    outfile.write('{0}# four interactions ({1})\n'.format(_I4, len(models)))
    for model in models:
        list_rep = _to_list(model)
        outfile.write('{0}{1},\n'.format(_I4, list_rep))

def _write_five_interaction(n, outfile):
    models = _interaction5(n)
    outfile.write('{0}# five interactions ({1})\n'.format(_I4, len(models)))
    for model in models:
        list_rep = _to_list(model)
        outfile.write('{0}{1},\n'.format(_I4, list_rep))
        
        
###############################################################################
if __name__ == '__main__':

    with open(_OUTFILE_NAME, 'wt') as outfile:

        outfile.write('# All pairwise interaction models for up to ten variables.\n\n')
        
        outfile.write('MODELS_2 = [\n')
        _write_univariate(2, outfile)
        outfile.write(']\n')
        
        outfile.write('\nMODELS_3 = [\n')
        _write_univariate(3, outfile)
        outfile.write('\n')
        _write_single_interaction(3, outfile)
        outfile.write('\n')
        outfile.write(']\n')

        outfile.write('\nMODELS_4 = [\n')
        _write_univariate(4, outfile)
        outfile.write('\n')
        _write_single_interaction(4, outfile)
        outfile.write('\n')
        _write_dual_interaction(4, outfile)
        outfile.write(']\n')

        outfile.write('\nMODELS_5 = [\n')
        _write_univariate(5, outfile)
        outfile.write('\n')
        _write_single_interaction(5, outfile)
        outfile.write('\n')
        _write_dual_interaction(5, outfile)
        outfile.write(']\n')

        outfile.write('\nMODELS_6 = [\n')
        _write_univariate(6, outfile)
        outfile.write('\n')
        _write_single_interaction(6, outfile)
        outfile.write('\n')
        _write_dual_interaction(6, outfile)
        outfile.write('\n')
        _write_three_interaction(6, outfile)
        outfile.write(']')

        outfile.write('\nMODELS_7 = [\n')
        _write_univariate(7, outfile)
        outfile.write('\n')
        _write_single_interaction(7, outfile)
        outfile.write('\n')
        _write_dual_interaction(7, outfile)
        outfile.write('\n')
        _write_three_interaction(7, outfile)
        outfile.write(']')

        outfile.write('\nMODELS_8 = [\n')
        _write_univariate(8, outfile)
        outfile.write('\n')
        _write_single_interaction(8, outfile)
        outfile.write('\n')
        _write_dual_interaction(8, outfile)
        outfile.write('\n')
        _write_three_interaction(8, outfile)
        outfile.write('\n')
        _write_four_interaction(8, outfile)
        outfile.write(']')
        
        outfile.write('\nMODELS_9 = [\n')
        _write_univariate(9, outfile)
        outfile.write('\n')
        _write_single_interaction(9, outfile)
        outfile.write('\n')
        _write_dual_interaction(9, outfile)
        outfile.write('\n')
        _write_three_interaction(9, outfile)
        outfile.write('\n')
        _write_four_interaction(9, outfile)
        outfile.write(']')

        outfile.write('\nMODELS_10 = [\n')
        _write_univariate(10, outfile)
        outfile.write('\n')
        _write_single_interaction(10, outfile)
        outfile.write('\n')
        _write_dual_interaction(10, outfile)
        outfile.write('\n')
        _write_three_interaction(10, outfile)
        outfile.write('\n')
        _write_four_interaction(10, outfile)
        outfile.write('\n')
        _write_five_interaction(10, outfile)
        outfile.write(']')

    
