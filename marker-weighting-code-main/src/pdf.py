import os
import sys
import numpy as np


def to_pdf(num_bins, samples, weights = None):
    """
    Compute the PDF for a given variable. Each sample at index q is for a given person, and the
    weight at index q is also for that same person. If no weights are provided each sample gets
    a weight of 1.
    """
    
    # must have as many weights as samples
    if weights is not None:
        assert len(weights) == len(samples)
    
    # accumulate counts, normalize
    pdf = np.zeros(num_bins)
    
    # q is an index for samples and weights
    for q in range(len(samples)):
        # the next sample value, 0 <= value < num_bins
        val = samples[q]
        # the weight to give this value
        w = 1
        if weights is not None:
            w = weights[q]
        # accumulate
        pdf[val] += w
        
    total = np.sum(pdf)
    pdf /= total
    assert np.isclose(1.0, np.sum(pdf))
    return pdf
