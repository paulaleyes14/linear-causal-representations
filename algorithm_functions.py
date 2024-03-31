# --- Imports ---
# Standard libraries
from math import factorial
from itertools import product

# Local
import helper_functions as hp

# Third-party
from sklearn.preprocessing import normalize
import causaldag as cd
import numpy as np
import tensorly as tl
from PyMoments import kstat

def create_params(p, q):
    """
    Sample B and Lambdas.

    Args:
        p (int): number of observed variables.
        q (int): number of latent variables.

    Returns:
        tuple: a tuple containing the sampled B and Lambdas.
            - B (numpy.ndarray): the mixing matrix, with shape (p, q).
            - Lambdas (dict): a dictionary containing the observational and interventional Lambdas (perfect interventions).
                Keys indicate the context (str), and values are the corresponding Lambdas (numpy.ndarray)
    """
    # Sample H
    rgen = np.random.default_rng()
    cond_number = 100
    while cond_number > 9:
        H = rgen.uniform(low=-2, high=2, size=(q,p))
        Hnorm = normalize(H, axis=1)
        for i in range(q):
            index = max(enumerate(Hnorm[i,:]), key=lambda x: (abs(x[1])))[0]
            if Hnorm[i,index] < 0:
                Hnorm[i,:] = (-1)*Hnorm[i,:]
        cond_number = np.linalg.cond(Hnorm)
    B = np.linalg.pinv(Hnorm)
    # Sample the graph and Lambda
    Lambdas = {}
    dag = cd.rand.directed_erdos(q, density=0.75, random_order=False)
    g = cd.rand.rand_weights(dag)
    Lambda_obs = g.to_amat()
    Lambdas["obs"]=Lambda_obs
    # Calculate interventional Lambdas
    for i in range(q-1):
        new_Lambda = np.copy(Lambda_obs)
        new_Lambda[i] = 0
        Lambdas["{0}".format(i)] = new_Lambda
    Lambdas["{0}".format(q-1)] = np.copy(Lambda_obs)
    return B, Lambdas

def create_models(nmodels, nobserved, nlatent):
    """
    Generate random models. Each model is encoded as a tuple of its parameters (B, Lambdas).

    Args:
        nmodels (int): number of models to generate.
        nobserved (int): number of observed variables per model.
        nlatent (int): number of latent variables per model.
    
    Returns:
        list: a list containing the generated models. 
    """
    model_params = []
    for _ in range(nmodels):
        params = create_params(nobserved, nlatent)
        model_params.append(params)
    return model_params

def create_X_samples(nsamples, Lambda, B, i, nonlinear_X, alpha_X, nonlinear_Z, alpha_Z):
    """
    Generate sample data.

    Args:
        nsamples (int): number of samples of X to generate.
        Lambda (numpy.ndarray): matrix defining the latent graph.
        B (numpy.ndarray): mixing matrix.
        i (int): context indicator. Equals -1 in observational context and k in inteventional context corresponding to an intervention at Z_k.
        nonlinear_X (bool): if True, add nonlinearity in the transformation from latent to observed variables.
        alpha_X (float): coefficient quantifying amount of nonlinearity to add in the transformation from latent to observed variables. 
            Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): if True, add nonlinearity in the latent space.
        alpha_Z (float): coefficient quantifying amount of nonlinearity to add in the latent space. Equals 0 if nonlinear_Z is False.
    
    Returns:
        numpy.ndarray: the sample data.
        numpy.ndarray: the product B(I-Lambda)^{-1}D in the context specified by i.
    """
    p, q = np.shape(B)
    data = []
    rgen = np.random.default_rng()
    omega = np.diag(factorial(2)*np.ones(q))
    I = np.eye(q)
    matrix = np.linalg.inv(I-Lambda)
    if i != -1:
        int_scale = rgen.uniform(low=1.25, high=4)
        omega[i,i] = factorial(2)*(int_scale**3)
    for _ in range(nsamples):
        epsilon_v = rgen.exponential(size=(q,1))
        if i != -1:
            epsilon_int = rgen.exponential(scale=int_scale)
            epsilon_v[i] = epsilon_int
        Z = np.matmul(matrix,epsilon_v)
        if nonlinear_Z:
            Z = Z + alpha_Z*np.power(epsilon_v,2)
        X = np.matmul(B,Z)
        if nonlinear_X:
            Z_add = np.power(Z,2)
            if p > q:
                nzeros = np.zeros((p-q,1))
                Z_add = np.vstack((Z_add,nzeros))
            X = X + alpha_X*Z_add
        X = np.reshape(X,(p,))
        data.append(X)
    data = np.asarray(data)
    omega_scaled = np.cbrt(omega)
    product = np.matmul(np.matmul(B,matrix),omega_scaled)

    return data, product

def sample_cumulant(nsamples, Lambda, B, i, nonlinear_X, alpha_X, nonlinear_Z, alpha_Z):
    """
    Calculate the sample third-order cumulant of X.
        Adapted from the code provided by Wang et al. in their paper "Identifiability of overcomplete independent component analysis".

    Args:
        nsamples (int): number of samples to use to calculate cumulant.
        Lambda (numpy.ndarray): matrix defining the latent graph.
        B (numpy.ndarray): mixing matrix.
        i (int): context indicator. Equals -1 in observational context and k in inteventional context corresponding to an intervention at Z_k.
        nonlinear_X (bool): if True, add nonlinearity in the transformation from latent to observed variables.
        alpha_X (float): coefficient quantifying amount of nonlinearity to add in the transformation from latent to observed variables. 
            Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): if True, add nonlinearity in the latent space.
        alpha_Z (float): coefficient quantifying amount of nonlinearity to add in the latent space. Equals 0 if nonlinear_Z is False.
    
    Returns:
        numpy.ndarray: the sample third-order cumulant.
        numpy.ndarray: the population third-order cumulant.
        float: the Frobenius norm of the difference between the sample and the population cumulants.
        numpy.ndarray: the product B(I-Lambda)^{-1}D in the context specified by i.
    
    Citation:
        Kexin Wang and Anna Seigal. Identifiability of overcomplete independent component analysis. arXiv preprint arXiv:2401.14709, 2024.
    """
    # Calculate sample cumulant
    p, q = np.shape(B)
    data, real_product = create_X_samples(nsamples, Lambda, B, i,nonlinear_X, alpha_X, nonlinear_Z, alpha_Z)
    sym_indices, _, _ = hp.symmetric_indices(p,3)
    third_cumulants = np.apply_along_axis(lambda x: kstat(data,tuple(x)), 0, sym_indices)
    third_cumulants_dict = {tuple(sym_indices[:,n]): third_cumulants[n] for n in range(len(third_cumulants))}
    all_indices = np.array([list(i) for i in product(range(p), range(p),range(p))])
    values = np.apply_along_axis(lambda x: third_cumulants_dict[tuple(np.sort(x))], 1, all_indices)
    third_order_kstat = values.reshape(p, p, p)

    # Calculate population cumulant
    weights = np.ones(q)
    real_tensor = tl.cp_to_tensor((weights, [real_product, real_product, real_product]))
    diff = np.linalg.norm(third_order_kstat-real_tensor)

    return third_order_kstat, real_tensor, diff, real_product


def construct_cumulants_kstat(nsamples, Lambdas, B, nonlinear_X, alpha_X, nonlinear_Z, alpha_Z):
    """
    Construct cumulant tensors and products for all contexts.

    Args:
        nsamples (int): number of samples to use to calculate sample cumulants.
        Lambdas (dict): a dictionary containing the observational and interventional Lambdas (perfect interventions).
            Keys indicate the context (str), and values are the corresponding Lambdas (numpy.ndarray).
        B (numpy.ndarray): mixing matrix.
        nonlinear_X (bool): if True, add nonlinearity in the transformation from latent to observed variables.
        alpha_X (float): coefficient quantifying amount of nonlinearity to add in the transformation from latent to observed variables. 
            Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): if True, add nonlinearity in the latent space.
        alpha_Z (float): coefficient quantifying amount of nonlinearity to add in the latent space. Equals 0 if nonlinear_Z is False.
    
    Returns:
        dict: a dictionary containing the sample third-order cumulants.
            Keys indicate the context (str), and values are the corresponding sample cumulants (numpy.ndarray).
        dict: a dictionary containing the population third-order cumulants.
            Keys indicate the context (str), and values are the corresponding population cumulants (numpy.ndarray).
        dict: a dictionary containing the products B(I-Lambda)^{-1}D across contexts.
            Keys indicate the context (str), and values are the corresponding products (numpy.ndarray).
        float: mean difference between population and sample cumulant across contexts.
    """
    Ts = {}
    realTs = {}
    Products = {}
    tensor_diffs = []
    keys = Lambdas.keys()
    for key in keys:
        i = -1 if key == "obs" else int(key)
        sample_tensor, real_tensor, diff, prod = sample_cumulant(nsamples, Lambdas[key], B, i, nonlinear_X, alpha_X, nonlinear_Z, alpha_Z)
        Ts[key] = sample_tensor
        realTs[key] = real_tensor
        Products[key] = prod
        tensor_diffs.append(diff)
    mean_diff = np.mean(tensor_diffs)
    return Ts, realTs, Products, mean_diff

def simult_diag(T,q):
    """
    Calculate the rank-q symmetric CP decomposition of a tensor.

    Args:
        T (numpy.ndarray): the tensor to perform tensor decomposition on.
        q (int): the desired rank of the decomposition.
    
    Returns:
        numpy.ndarray: the normalized (by column) factor matrix of the decomposition.
    """
    p, _, _ = np.shape(T)
    a = np.random.rand(p)
    anorm = a/np.linalg.norm(a)
    b = np.random.rand(p)
    borth = b - np.inner(b,a)*a
    bnorm = borth/np.linalg.norm(borth)

    scaled_tensor_a = np.einsum('ijk,i->ijk', T, anorm)
    Ma = np.sum(scaled_tensor_a,axis=0)
    scaled_tensor_b = np.einsum('ijk,i->ijk', T, bnorm)
    Mb = np.sum(scaled_tensor_b,axis=0)

    prod = np.matmul(Ma, np.linalg.pinv(Mb))
    eigvalues, eigvectors = np.linalg.eig(prod)
    idx = np.argsort(np.abs(eigvalues))
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:,idx]

    return np.real_if_close(eigvectors[:,-q:], tol=1)

def decompose_tensors(Ts,q):
    """
    Decompose cumulants in all contexts using symmetric CP decomposition.

    Args:
        Ts (dict): a dictionary containing the cumulants to decompose.
            Keys indicate the context (str), and values are the corresponding cumulants (numpy.ndarray).
        q (int): number of latent variables.
    
    Returns:
        dict: a dictionary containing the recovered factor matrices.
            Keys indicate the context (str), and values are the corresponding factor matrices (numpy.ndarray).
    """
    Product_recovery = {}
    for key in Ts.keys():
        factor = simult_diag(Ts[key], q)
        Product_recovery[key] = factor
    return Product_recovery

def find_alphas(T, recovered_product):
    """
    Calculate tensor decomposition coefficients.

    Args:
        T (numpy.ndarray): the tensor whose decomposition's coefficients we want to calculate.
        recovered_product (numpy.ndarray): the normalized (by column) factor matrix recovered by performing tensor decomposition on T.
    
    Returns:
        numpy.ndarray: the coefficients minimizing the Frobenius norm of the difference between T and the reconstructed tensor.
    """
    p, q = np.shape(recovered_product)
    b = np.reshape(T, p**3)
    A = []
    for j in range(q):
        current_v = np.reshape(recovered_product[:,j],(p,1))
        current_tensor = tl.cp_to_tensor((np.ones(1),[current_v,current_v,current_v]))
        new_column = np.reshape(current_tensor,p**3)
        A.append(new_column)
    Amat = np.transpose(np.array(A))
    alphas, _, _, _  = np.linalg.lstsq(Amat, b, rcond=None)
    return alphas

def nonnorm_products(Ts, RProducts):
    """
    Given a dictionary of tensors and the normalized factor matrices of their rank-r symmetric CP decompositions, 
        calculate the non-normalized factor matrices and their pseudoinverses.

    Args:
        Ts (dict): a dictionary containing the cumulants in each context.
            Keys indicate the context (str), and values are the corresponding cumulants (numpy.ndarray).
        RProducts (dict): a dictionary containing the normalized (by column) factor matrices recovered by decomposing the tensors in each context.
            Keys indicate the context (str), and values are the corresponding normalized factor matrices (numpy.ndarray).
        
    Returns:
        dict: a dictionary containing the non-normalized factor matrices and their pseudoinverses.
            Keys indicate the context (str), and values are tuples containing the non-normalized factor matrices (numpy.ndarray) and their pseudoinverses (numpy.ndarray).
    """
    _, q = np.shape(RProducts["obs"])
    RProducts_nn = {}
    for key in Ts.keys():
        current_product = RProducts[key].copy()
        current_alphas = find_alphas(Ts[key],current_product)
        for i in range(q):
            current_product[:,i] = hp.cube(current_alphas[i])*current_product[:,i]
        current_product_inv = np.linalg.pinv(current_product)
        RProducts_nn[key] = (current_product, current_product_inv)
    return RProducts_nn

def recover_int_target(C, Ctilde, current_ints):
    """
    Determine intervention target and permutation matrix of interventional context where pseudoinverse of Ctilde was recovered.

    Args:
        C (numpy.ndarray): pseudoinverse of the product recovered in the observational context.
        Ctilde (numpy.ndarray): pseudoinvere of the product recovered in an interventional context with unknown intervention target.
        current_ints (set[int]): a set containing the intervention targets of the interventional contexts that have already been matched with an intervention target.
    
    Returns:
        tuple: a tuple containing the intervened variable in the context where the pseudoinverse of Ctilde was recovered 
            and the index indicating the relabeling of this variable in such context.
        numpy.ndarray: the permutation matrix encoding the relabeling of the latent nodes in the interventional context where the pseudoinverse of Ctilde was recovered.
    """
    q, _ = np.shape(C)
    diff_list = []
    matches_int = {}
    P = np.zeros((q,q))
    for k in range(q):
        matches_int[k] = 0

    # Initial matching of rows of C with rows of \tilde{C}
    for i in range(q):
        row_int, diff = hp.match_row(C[i,:].copy(), Ctilde)
        matches_int[row_int] += 1
        diff_list.append((i, row_int, diff))
    
    # Rematch incorrectly matched rows
    unmatched_rows = [k for k, v in matches_int.items() if v == 0]
    for (key, value) in matches_int.copy().items():
        if value > 1:
            tuple_list = list(filter(lambda x: x[1]==key,diff_list))
            tuple_list.sort(key = lambda x: x[2])
            tuple_list.remove(tuple_list[0])

            diff_list = [elt for elt in diff_list if elt not in tuple_list]
            
            for obs_row_ind, _, _ in tuple_list:
                row_int_new, diff_new = hp.match_row(C[obs_row_ind,:].copy(), Ctilde[unmatched_rows,:])
                new_match = (obs_row_ind, unmatched_rows[row_int_new], diff_new)
                unmatched_rows.remove(unmatched_rows[row_int_new])
                diff_list.append(new_match)
    
    diff_list.sort(reverse=True, key=lambda x: x[2])
    for (i, j, _) in diff_list:
        P[i,j]=1
    for tuple in diff_list:
        if not(tuple[0] in current_ints):
            return tuple, P

def match_int(Productsnn):
    """
    Determine the intervention target and undo relabeling of latent nodes arising from tensor decomposition in each interventional context.

    Args:
        RProductsnn (dict): a dictionary containing the non-normalized factor matrices recovered via tensor decomposition and least squares in each context.
            Keys indicate the context (str), and values are tuples containing the corresponding non-normalized factor matrix (numpy.ndarray) and its pseudoinverse (numpy.ndarray).

    Returns:
        dict: a dictionary mapping each context to its non-normalized factor matrix, its pseudoinverse (both after undoing relabeling of latent nodes) and its intervention target. 
            Keys indicate the context (str), and values are tuples containing the corresponding non-normalized factor matrix (numpy.ndarray), its pseudoinverse (numpy.ndarray),
                and a tuple containing the intervention target of the context (int) and the relabeling of the intervened variable in the context (int). The latter two are both -1
                in the observational context.
    """
    tuples = {}
    Prodnn, C = Productsnn["obs"]
    tuples["obs"] = (Prodnn, C, (-1,-1))
    keys = Productsnn.keys()
    current_ints = set()
    for key in keys:
        if key == "obs":
            continue
        else:
            Prodtildenn, Ctilde = Productsnn[key]
            _, q = np.shape(Prodtildenn)
            tuple, P = recover_int_target(C, Ctilde, current_ints)

            assert(np.linalg.matrix_rank(P)==q)
            current_ints.add(tuple[0])
            Prodtildenn_new = np.matmul(Prodtildenn, np.transpose(P))
            Cprimetilde = np.matmul(P, Ctilde)
            tuples[key] = (Prodtildenn_new, Cprimetilde, (tuple[0], tuple[1]))
    return tuples

def recover_H(tuples):
    """
    Recover the mixing matrix.

    Args:
        tuples (dict): a dictionary containing the factor matrices recovered via tensor decomposition in each context (with consistent labeling of latent nodes across contexts),
            its pseudoinverses, and the intervention targets. Keys indicate the context (str), and values are tuples containing the corresponding factor matrix (numpy.ndarray),
            its pseudoinvsrse (numpy.ndarray), and a tuple containing the intervention target of the context (int) and the relabeling of the intervened variable in the context (int).
            The latter two are both -1 in the observational context.
    
    Returns:
        numpy.ndarray: the pseudoinverse of the mixing matrix B.
    """
    keys = tuples.keys()
    Hpair = []
    for key in keys:
        if key == "obs":
            continue
        else:
            _, Cprimea, int_target = tuples[key]
            a, _ = int_target
            new_row = Cprimea[a,:]/(np.linalg.norm(Cprimea[a,:]))
            index = max(enumerate(new_row), key=lambda x: (abs(x[1])))[0]
            if new_row[index] < 0:
                new_row = new_row*(-1)
            row_pair = (new_row, a)
            Hpair.append(row_pair)
    Hpair.sort(key=lambda x: x[1])
    H = list(map(lambda x: x[0], Hpair))
    return np.array(H)

def recover_lambda(tuples, H, tol_param=1e-08):
    """
    Recover Lambda, the matrix encoding the latent graph.

    Args:
        tuples (dict): a dictionary containing the factor matrices recovered via tensor decomposition in each context (with consistent labeling of latent nodes across contexts),
            its pseudoinverses, and the intervention targets. Keys indicate the context (str), and values are tuples containing the corresponding factor matrix (numpy.ndarray),
            its pseudoinvsrse (numpy.ndarray), and a tuple containing the intervention target of the context (int) and the relabeling of the intervened variable in the context (int).
            The latter two are both -1 in the observational context.
        H (numpy.ndarray): the pseudoinverse of the mixing matrix B.
        tol_param (float): threshold used to determine if a vector v_r is in the span of a set of vectors {v_1,...,v_n}
    
    Returns:
        numpy.ndarray: the matrix encoding the latent graph.
    """
    p, q = np.shape(tuples["obs"][0])
    _, Invnn_obs, _ = tuples["obs"]
    nodes = set(list(range(q)))
    Lambda = np.zeros((q,q))
    Vs = {}
    Vs[0] = []
    keys = tuples.keys()
    for key in keys:
        if key == "obs":
            continue
        else:
            _, Invnn_tilde, int_target = tuples[key]
            a, _ = int_target
            row_obs = np.reshape(Invnn_obs[a,:],(p,1))
            row_int = Invnn_tilde[a,:]
            _, res_0, _, _ = np.linalg.lstsq(row_obs, row_int, rcond=None)
            if np.allclose(res_0,0,atol=tol_param):
                Vs[0].append(a)
                nodes.remove(int(key))
    for i in range(1,q):
        Vs[i] = Vs[i-1]
        for j in nodes.copy():
            int_var, _ = tuples["{0}".format(j)][2]
            indices = [int_var] + Vs[i-1]
            Hrows = H[indices, :]
            A = np.transpose(Hrows)
            b = Invnn_obs[int_var,:]
            sol, residual, _, _ = np.linalg.lstsq(A, b, rcond=None)
            gamma_j = sol[0]

            if np.allclose(residual,0,atol=tol_param):
                Vs[i].append(int_var)
                nodes.remove(j)
                for m in range(1,len(sol)):
                    current_k = indices[m]
                    Lambda[int_var,current_k] = (-1)*sol[m]/gamma_j
    return Lambda