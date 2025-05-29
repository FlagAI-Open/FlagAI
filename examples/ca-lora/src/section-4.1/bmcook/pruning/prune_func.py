'''
    This implementation is modified from https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity.
'''

import multiprocessing
import sys
import torch
import numpy as np
import collections
from itertools import permutations

""" compute density (helper fn to compute % NNZs in a tensor) """
def fill(x):
    return float(x.nonzero().size(0))/torch.numel(x)

""" reshape matrix into m-dimensional vectors: (h,w) -> (hw/m, m) """
def reshape_1d(matrix, m):
    # If not a nice multiple of m, fill with zeroes.
    if matrix.shape[1] % m > 0:
        mat = torch.cuda.FloatTensor(matrix.shape[0], matrix.shape[1] + (m-matrix.shape[1]%m)).fill_(0)
        mat[:, :matrix.shape[1]] = matrix
        shape = mat.shape
        return mat.view(-1,m),shape
    else:
        return matrix.view(-1,m), matrix.shape

""" return all possible m:n patterns in a 1d vector """
valid_m4n2_1d_patterns = None
def compute_valid_1d_patterns(m,n):
    # Early exit if patterns was already created.
    global valid_m4n2_1d_patterns

    if m==4  and n==2 and valid_m4n2_1d_patterns  is not None: return valid_m4n2_1d_patterns
    patterns = torch.zeros(m)
    patterns[:n] = 1
    valid_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
    if m == 4  and n == 2: valid_m4n2_1d_patterns  = valid_patterns       
    return valid_patterns

""" m:n 1d structured best """
def mn_1d_best(matrix, m, n):
    # Find all possible patterns.
    patterns = compute_valid_1d_patterns(m,n).cuda()

    # Find the best m:n pattern (sum of non-masked weights).
    mask = torch.cuda.IntTensor(matrix.shape).fill_(1).view(-1,m)
    mat,shape = reshape_1d(matrix,m)

    pmax = torch.argmax(torch.matmul(mat.abs().float().cuda(), patterns.t()), dim=1)
    mask[:] = patterns[pmax[:]]
    mask = mask.view(matrix.shape)
    return torch.tensor(mask, dtype=matrix.dtype)

def m4n2_1d(mat):
    return mn_1d_best(mat, 4, 2)

def mn_2d_greedy(matrix, m, n):
    # Convert to numpy
    mat = matrix.cpu().detach().numpy()
    mask = np.ones(mat.shape, dtype=int)

    assert mat.shape[0] % m == 0, mat.shape
    assert mat.shape[1] % m == 0, mat.shape
    assert len(mat.shape) == 2
    rowCount = mat.shape[0]
    colCount = mat.shape[1]

    for rowStartIdx in range(0, rowCount, m):
        rowEndIdx = rowStartIdx + m
        for colStartIdx in range(0, colCount, m):
            colEndIdx = colStartIdx + m
            matrixSub = np.absolute(mat[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx])
            maskSub = mask[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx]
            maskSub.fill(0.0)
            matrixVecView = matrixSub.reshape(-1)
            linearIdx = np.argsort(matrixVecView)
            matrixIdx = [(int(x/m), x % m) for x in linearIdx]
            rowCounter = collections.Counter()
            colCounter = collections.Counter()
            for currIdx in range(len(linearIdx) - 1, -1, -1):
                currMatrixEntry = matrixIdx[currIdx]
                if (rowCounter[currMatrixEntry[0]] == n) or (colCounter[currMatrixEntry[1]] == n):
                    continue
                #end if
                maskSub[currMatrixEntry[0], currMatrixEntry[1]] = 1.0
                rowCounter[currMatrixEntry[0]] += 1
                colCounter[currMatrixEntry[1]] += 1

    return torch.tensor(mask, dtype=matrix.dtype).cuda()

def process_func(mat):
    m = 4
    n = 2
    mask = np.ones(mat.shape, dtype=int)
    mask.fill(0.0)
    matrixVecView = mat.reshape(-1)
    linearIdx = np.argsort(matrixVecView)
    matrixIdx = [(int(x/m), x % m) for x in linearIdx]
    rowCounter = collections.Counter()
    colCounter = collections.Counter()
    for currIdx in range(len(linearIdx) - 1, -1, -1):
        currMatrixEntry = matrixIdx[currIdx]
        if (rowCounter[currMatrixEntry[0]] == n) or (colCounter[currMatrixEntry[1]] == n):
            continue
        #end if
        mask[currMatrixEntry[0], currMatrixEntry[1]] = 1.0
        rowCounter[currMatrixEntry[0]] += 1
        colCounter[currMatrixEntry[1]] += 1
    return mask

def mn_2d_greedy_faster(matrix, m, n):
    assert m == 4
    assert n == 2
    # Convert to numpy
    mat = matrix.cpu().detach().numpy()
    mask = np.ones(mat.shape, dtype=int)

    assert mat.shape[0] % m == 0
    assert mat.shape[1] % m == 0
    assert len(mat.shape) == 2
    rowCount = mat.shape[0]
    colCount = mat.shape[1]

    inputs = []
    for rowStartIdx in range(0, rowCount, m):
        rowEndIdx = rowStartIdx + m
        for colStartIdx in range(0, colCount, m):
            colEndIdx = colStartIdx + m
            inputs.append(np.absolute(mat[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx]))
    
    pool = multiprocessing.Pool(16)
    masks = pool.imap(process_func, inputs, 4096)

    idx = 0
    for m_ in masks:
        rowStartIdx = idx // (colCount // m) * m
        colStartIdx = idx % (colCount // m) * m
        rowEndIdx = rowStartIdx + m
        colEndIdx = colStartIdx + m
        mask[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx] = m_
        idx += 1
    
    return torch.tensor(mask, dtype=matrix.dtype).cuda()

def m4n2_2d_greedy(mat):
    return mn_2d_greedy(mat, 4, 2)
