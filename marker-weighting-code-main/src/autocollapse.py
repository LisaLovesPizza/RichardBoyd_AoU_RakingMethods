"""
Performs auto-collapsing based on a bin count threshold. Any bins with counts
below the threshold have their data moved to an adjacent bin. A zero count is
left in the source bin.
"""

import os
import sys

from collections import Counter


###############################################################################
def collapse_unordered(samples, rebin_threshold, other_key):
    """
    """
    
    changes = None
    
    # bin the data and get the counts for each value
    ctr = Counter(samples)
    
    
    # find the index of the bin with the smallest count
    move_key = None
    values = [v for k,v in ctr.items() if k != other_key]
    if 0 == len(values):
        # all counts are already in the "other" bin
        return samples, changes
    else:
        min_count = min(values)
        if min_count < rebin_threshold:
            for k,v in ctr.items():
                if v == min_count:
                    move_key = k
                    break
                
    if move_key is None:
        # nothing to do
        return samples, changes
    elif move_key == other_key:
        # the "other" bin has the minimum count, nothing to do
        return samples, changes
    else:
        # move the below-threshold key's counts to the other_key unless
        # that key's count is zero, which would result in a meaningless
        # swap of the counts
        if other_key not in ctr.keys():
            # zero value
            return samples, changes
        else:
            print('\tcollapse_unordered: change key {0} to key {1}'.format(move_key, other_key))
            new_samples = []
            for s in samples:
                if s == move_key:
                    new_samples.append(other_key)
                else:
                    new_samples.append(s)
                
    changes = (move_key, other_key)
    return new_samples, changes


###############################################################################
def collapse_ordered(samples, rebin_threshold):
    """
    """
    
    changes = None
    
    # bin the data and get the counts for each value
    ctr = Counter(samples)
    
    # sort the keys in order and get counts in the same order
    sorted_keys = sorted([k for k in ctr.keys()])
    counts = [ctr[k] for k in sorted_keys]
        
    # find the index of the bin with the smallest count
    min_count = min(counts)
    min_bin_index = counts.index(min_count)
    if min_count >= rebin_threshold:
        # no need to rebin
        return samples, changes
    
    # collapse the bin with the minimum count into an adjacent bin
    
    # get count in the adjacent bin to the left
    left_count = None
    if min_bin_index > 0:
        check_index = min_bin_index - 1
        left_count = counts[check_index]
        
    # get count in the adjacent bin to the right
    right_count = None
    if min_bin_index < len(ctr) - 1:
        check_index = min_bin_index + 1
        right_count = counts[check_index]
        
    old_index = min_bin_index
    new_index = None
    if left_count is not None and right_count is not None:
        if left_count > right_count:
            # move right
            new_index = old_index+1
        else:
            # move left
            new_index = old_index-1
    elif left_count is None and right_count is not None:
        # move right
        new_index = old_index+1
    elif left_count is not None and right_count is None:
        # move left
        new_index = old_index-1
    else:
        # nothing to do
        return samples, changes
    
    old_key = sorted_keys[old_index]
    new_key = sorted_keys[new_index]
    print('\tcollapse_ordered: change key {0} to key {1}'.format(old_key, new_key))
    
    new_samples = []
    for s in samples:
        if s == old_key:
            new_samples.append(new_key)
        else:
            new_samples.append(s)
    
    changes = (old_key, new_key)
    return new_samples, changes


###############################################################################
def full_collapse_ordered(df, col_name, threshold):
    """
    Collapse the bin with the smallest count into an adjacent bin if the count is below
    the given threshold. Iterate until no further collapses are possible.
    
    Return the updated dataframe and a list of (before, after) index tuples containing
    the collapse sequence.
    """
    
    change_list = []
    
    samples = df[col_name].values
    sample_count = len(samples)
    
    collapsed, changes = collapse_ordered(samples, rebin_threshold=threshold)
    if changes is not None:
        change_list.append(changes)
    assert len(collapsed) == sample_count
    while changes is not None:
        collapsed, changes = collapse_ordered(collapsed, rebin_threshold=threshold)
        if changes is not None:
            change_list.append(changes)
        assert len(collapsed) == sample_count
        
    df = df.assign(**{col_name:collapsed})
    return df, change_list


###############################################################################
def full_collapse_unordered(df, col_name, threshold, other_col):
    """
    """
    
    change_list = []
    
    samples = df[col_name].values
    sample_count = len(samples)
    
    collapsed, changes = collapse_unordered(samples, threshold, other_col)
    if changes is not None:
        change_list.append(changes)
    assert len(collapsed) == sample_count
    while changes is not None:
        collapsed, changes = collapse_unordered(collapsed, threshold, other_col)
        if changes is not None:
            change_list.append(changes)
        assert len(collapsed) == sample_count
        
    df = df.assign(**{col_name:collapsed})
    return df, change_list


###############################################################################
def collapse_from_changelist(df, col_name, change_list):
    
    for orig_key, change_key in change_list:
        samples = df[col_name].values
        new_values = []
        for s in samples:
            t = s
            if s == orig_key:
                t = change_key
            new_values.append(t)
        df = df.assign(**{col_name:new_values})
        
    return df

