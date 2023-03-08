import os
import sys
import numpy as np
import matplotlib.pyplot as plt



def dual_histogram(bin_count, var_name, source_samples, target_samples,
                   msg='', weights=None):
    """
    Plot a dual histogram for a given variable.
    """
    
    # bins are numbered from 0..(bin_count-1)
    binvals = [x for x in range(0, bin_count)]
    labels = ['{0}'.format(int(x)) for x in binvals]
        
    # start with zero in all bins, then fill in available values
    target_pdf = np.zeros(len(binvals))
    source_pdf = np.zeros(len(binvals))

    # compute source PDF
    for x in source_samples:
        index = binvals.index(x)
        source_pdf[index] += 1
        
    # if weights were provided, multiply the source counts by the weights
    # before normalization
    if weights is not None:
        assert len(weights) == len(source_pdf)
        # elementwise multiplication
        source_pdf = np.multiply(source_pdf, weights)
        
    source_count = np.sum(source_pdf)
    source_pdf /= source_count
    
    checksum = np.sum(source_pdf)
    assert np.isclose(checksum, 1.0)
        
    # compute target pdf
    for x in target_samples:
        index = binvals.index(x)
        target_pdf[index] += 1
    target_count = np.sum(target_pdf)
    target_pdf /= target_count
    
    checksum = np.sum(target_pdf)
    assert np.isclose(checksum, 1.0)
    
    assert len(source_pdf) == len(target_pdf)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(16,8))
    rects1 = ax.bar(x + width/2, target_pdf, width, label='Target', zorder=3) 
    rects2 = ax.bar(x - width/2, source_pdf, width, label='Source', zorder=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('{0}'.format(var_name), fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    
    if msg is not None:
        ax.set_title('{0} PDF, {1}'.format(var_name, msg), fontsize=16)
    else:
        ax.set_title('{0} PDF'.format(var_name), fontsize=16)
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    plt.grid(zorder=0)
    plt.show()


def dual_histogram_from_pdfs(var_name, source_pdf, target_pdf, labels=None, msg=None):
    """
    Plot a dual histogram from given PDFs.
    """

    bin_count = len(source_pdf)
    assert bin_count == len(target_pdf)
    
    # bins are numbered from 0..(bin_count-1)
    binvals = [x for x in range(0, bin_count)]
    xlabels = ['{0}'.format(int(x)) for x in binvals]
        
    x = np.arange(len(xlabels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(16,8))
    if labels is None:
        rects1 = ax.bar(x + width/2, target_pdf, width, label='Target', zorder=3) 
        rects2 = ax.bar(x - width/2, source_pdf, width, label='Source', zorder=3)
    else:
        rects1 = ax.bar(x + width/2, target_pdf, width, label=labels[1], zorder=3) 
        rects2 = ax.bar(x - width/2, source_pdf, width, label=labels[0], zorder=3)        

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('{0}'.format(var_name), fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    
    if msg is not None:
        ax.set_title('{0} PDF, {1}'.format(var_name, msg), fontsize=16)
    else:
        ax.set_title('{0} PDF'.format(var_name), fontsize=16)
    ax.set_xticks(x, xlabels)
    ax.legend()
    fig.tight_layout()
    plt.grid(zorder=0)
    plt.show()


def triple_histogram_from_pdfs(var_name, pdf1, pdf2, pdf3,
                               labels=['pdf1','pdf2','pdf3'], msg=None):
    """
    Plot a triple histogram from given PDFs, all of which have the same bins.
    """

    bin_count = len(pdf1)
    assert bin_count == len(pdf2)
    assert bin_count == len(pdf3)
    
    # bins are numbered from 0..(bin_count-1)
    binvals = [x for x in range(0, bin_count)]
    xlabels = ['{0}'.format(int(x)) for x in binvals]
        
    x = np.arange(len(xlabels))  # the label locations
    width = 0.5  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(16,8))
    rects1 = ax.bar(x + width/3, pdf3, width/3, label=labels[2], zorder=3) 
    rects2 = ax.bar(x,           pdf2, width/3, label=labels[1], zorder=3)
    rects3 = ax.bar(x - width/3, pdf1, width/3, label=labels[0], zorder=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('{0}'.format(var_name), fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    
    if msg is not None:
        ax.set_title('{0} PDF, {1}'.format(var_name, msg), fontsize=16)
    else:
        ax.set_title('{0} PDF'.format(var_name), fontsize=16)
    ax.set_xticks(x, xlabels)
    ax.legend()
    fig.tight_layout()
    plt.grid(zorder=0)
    plt.show()
    
