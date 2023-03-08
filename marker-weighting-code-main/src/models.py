import os
import sys
from collections import Counter


MODELS_2 = [
    [[0], [1]],
    [[1], [0]],
]

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

    # two interactions
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
    [[2], [5], [0,4], [1,3]]
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
        
    # triple interaction
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

