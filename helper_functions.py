import numpy as np
import sys
from math import factorial
from itertools import permutations

def match_row(M1_row, M2):
    """
    Given a vector of size 1xn and a matrix of size mxn, find the row of the matrix closest to the vector (in the Euclidean norm sense).

    Args:
        M1_row (numpy.ndarray): vector of size 1xn.
        M2 (numpy.ndarray): matrix of size mxn.
    
    Returns:
        match (int): the index of the row of M2 that is closest to M1_row.
        diff (float): the Euclidean norm of the difference between M1_row and its closest row in M2.
    """
    q, _ = np.shape(M2)
    diff = sys.float_info.max
    match = -1
    for i in range(q):
        compare_row = M2[i,:].copy()
        current_diff = np.linalg.norm(compare_row-M1_row)
        if current_diff < diff:
            diff = current_diff
            match = i
    return match, diff

def symmetric_indices(d, n):
    """
    Generate symmetric indices and permutation indices for symmetric tensors. 
        Obtained from the code provided by Kileel et al. in their paper "Subspace power method for symmetric tensor decomposition and generalized PCA".

    Args:
        d (int): the dimensionality of the tensor.
        n (int): the order of the tensor.

    Returns:
        symind (numpy.ndarray): an array containing the indices of symmetric permutations.
        findsym (numpy.ndarray): a tensor representing the permutation indices for symmetric tensors.
        nperm (numpy.ndarray): an array containing the number of unique permutations for each symmetry order.

    Citation:
        Joe Kileel and Joao M. Pereira. Subspace power method for symmetric tensor decomposition and generalized PCA. arXiv preprint arXiv:1912.04007, 2019.
    """
    symind = np.ones((n, d), dtype=int)
    symind[0] = np.arange(d)
    nperm = np.full(d, factorial(n))
    last_equal = np.ones(d)
    for j in range(1, n):
        reps = d - symind[j-1]
        symind = np.repeat(symind, reps, axis=1)
        ind = np.cumsum(reps)
        symind[j, ind[:-1]] = 1-reps[1:]
        symind[j, 0] = 0
        symind[j] = np.cumsum(symind[j], out=symind[j])
        new_last_equal = np.ones(symind.shape[1])
        new_last_equal[ind[:-1]] = last_equal[1:] + 1
        new_last_equal[0] = j + 1
        last_equal = new_last_equal
        nperm = np.repeat(nperm, reps) / last_equal

    dnsym = symind.shape[1]

    findsym = np.empty(d ** n, dtype=int)
    for perm in permutations(range(n)):
        findsym[symind[perm, :].T  @ (d ** np.arange(n))] = np.arange(dnsym)
    findsym = findsym.reshape([d] * n)

    return symind, findsym, nperm

def cube(x):
    """
    Calculate cubic roots while avoiding complex numbers.

    Args:
        x (float): number to take the cubic root of.
    
    Returns:
        float: cubic root of x.
    """
    if x >= 0:
        return x**(1/3)
    elif x < 0:
        return -(abs(x)**(1/3))

def find_best_permutation(matrixA, matrixB):
    """
    Given two matrices A and B, calculate the permutation of the rows of A yielding the closest matrix to B (in the Frobenius norm sense).

    Args:
        matrixA (numpy.ndarray): the matrix to permute the columns of.
        matrixB (numpy.ndarray): the matrix to compare against.
    
    Returns:
        P (numpy.ndarray): a permutation matrix minimizing ||matrixA*P - matrixB||
    """
    _,q = np.shape(matrixA)
    global_diff = sys.float_info.max
    global_match = None
    column_matrix = matrixA.T.tolist()
    columns = list(range(len(column_matrix)))
    for perm in permutations(columns):
        shuffled_column_matrix = []
        for idx in perm:
            shuffled_column_matrix.append(column_matrix[idx])
        new_matrix = np.array(shuffled_column_matrix).T
        new_diff = np.abs(np.linalg.norm(matrixB-new_matrix))
        if new_diff < global_diff:
            global_diff = new_diff
            global_match = perm
    P = np.zeros((q,q))
    for i in range(q):
        orig_column = global_match[i]
        P[orig_column][i] = 1
    return P

def lambda_diff(Lambda, Lambdar, perm):
    """
    Calculate the difference between the original Lambda and the estimated one.
   
    Args:
        Lambda (numpy.ndarray): the true matrix encoding the latent graph.
        Lambdar (numpy.ndarray): the recovered matrix encoding the latent graph, up to the relabeling of the latent nodes given by 
            the permutation arising from tensor decomposition in the observational context.
        perm (numpy.ndarray): permutation of the latent nodes in the observational context.
    
    Returns:
        float: the norm of the difference between the true matrix encoding the latent graph and the recovered one, 
            up to the permutation of the latent nodes in the observational context.
    """
    Lambdanperm = np.matmul(perm,np.matmul(Lambdar,np.transpose(perm)))
    return np.linalg.norm(Lambda-Lambdanperm)

def H_diff(H, Hr, perm):
    """
    Calculate the difference between the pseudoinverse of the true mixing matrix and that of the estimated one.

    Args:
        H (numpy.ndarray): the pseudoinverse of the true mixing matrix.
        Hr (numpy.ndarray): the pseudoinverse of the recovered mixing matrix, up to the relabeling of the latent nodes given by 
            the permutation arising from tensor decomposition in the observational context.
        perm (numpy.ndarray): permutation of the latent nodes in the observational context.
    
    Returns:
        diff (float): the norm of the difference between the pseudoinverse of the true mixing matrix and that of the recovered mixing matrix,
            up to the permutation of the latent nodes in the observational context.
        diff_abs (float): the norm of the difference between the absolute value (entrywise) of the true mixing matrix and that of the recovered mixing matrix,
            up to the permutation of the latent nodes in the observational context.
    """
    Hrnperm = np.matmul(perm,Hr)
    Hrnpermabs = np.abs(Hrnperm)
    Habs = np.abs(H)
    diff = np.linalg.norm(Hrnperm-H)
    diff_abs = np.linalg.norm(Hrnpermabs-Habs)
    return diff, diff_abs