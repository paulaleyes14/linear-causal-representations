# --- Imports ---
# Standard libraries
import os
from argparse import ArgumentParser

# Local
import algorithm_functions as af
import helper_functions as hp

# Third-party
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def run_sample(nsamples_list, nobserved, nlatent, nmodels, nonlinear_X, alpha_X, nonlinear_Z, alpha_Z):
    """
    Test performance of method for linear causal disentanglement using sample cumulants as input.

    Args:
        nsamples_list (list[int]): list containing the number of samples to use in constructing the cumulants.
        nobserved (int): number of observed variables.
        nlatent (int): number of latent variables.
        nmodels (int): number of models to generate. Errors will be averaged across them.
        nonlinear_X (bool): flag to add nonlinearity in the transformation from latent to observed variables. If True, nonlinearity will be added; if False, it won't.
        alpha_X (float): coefficient quantifying how much nonlinearity to add in the transformation from latent to observed variables. Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): flag to add nonlinearity in the latent space. If True, nonlinearity will be added; if False, it won't.
        alpha_Z (float): coefficient quantifying how much nonlinearity to add in the latent space. Equals 0 if nonlinear_Z is False.
    
    Returns:
        tuple: a tuple containing a list of the mean errors in estimating H for each number of samples (list[float]), a list of the mean errors in estimating abs(H) for
            each number of samples (list[float]), and a list of the mean errors in estimating Lambda for each number of samples (list[float]).
    """
    models = af.create_models(nmodels, nobserved, nlatent)
    diff_H_global = []
    diff_H_abs_global = []
    diff_lambda_global = []
    tensor_diff_global = []

    for nsamples in nsamples_list:
        print(nsamples)
        diff_H_local = []
        diff_H_abs_local = []
        diff_lambda_local = []
        tensor_diff_local = []
        for pair in models:
            B, Lambdas = pair
            Ts, _, Products, mean_diff = af.construct_cumulants_kstat(nsamples,Lambdas,B,nonlinear_X,alpha_X,nonlinear_Z,alpha_Z)
            tensor_diff_local.append(mean_diff)

            RProducts = af.decompose_tensors(Ts,nlatent)
            RProdnninv = af.nonnorm_products(Ts, RProducts)
            RProdnn = {key: values[0] for key, values in RProdnninv.items()}
            RProdinv = {key: values[1] for key, values in RProdnninv.items()}
            int_tuples = af.match_int(RProdnninv)

            Hr = af.recover_H(int_tuples)
            perm = hp.find_best_permutation(Products["obs"],RProdnn["obs"])

            diff_H, diff_H_abs = hp.H_diff(np.linalg.pinv(B),Hr,perm)
            diff_H_local.append(diff_H)
            diff_H_abs_local.append(diff_H_abs)

            Lambdar = af.recover_lambda(int_tuples,Hr,tol_param=0.01)
            diff_lambda = hp.lambda_diff(Lambdas["obs"],Lambdar,perm)
            diff_lambda_local.append(diff_lambda)
        # Once I have gone through my models, I want to save the mean. Then I should have one value per number of samples
        diff_H_global.append(np.mean(diff_H_local))
        print("Mean error in H for {0} samples is {1}".format(nsamples,np.mean(diff_H_local)))
        diff_H_abs_global.append(np.mean(diff_H_abs_local))
        print("Mean error in abs(H) for {0} samples is {1}".format(nsamples,np.mean(diff_H_abs_local)))
        diff_lambda_global.append(np.mean(diff_lambda_local))
        print("Mean error in Lambda for {0} samples is {1}".format(nsamples,np.mean(diff_lambda_local)))
        tensor_diff_global.append(np.mean(tensor_diff_local))
        print("Mean error in cumulants for {0} samples is {1}".format(nsamples, np.mean(tensor_diff_local)))
        
    return (diff_H_global, diff_H_abs_global, diff_lambda_global,  tensor_diff_global)

def plot(nlatent, nobserved, diff_H, diff_H_abs, diff_lambda, tensor_diff, nsamples_list, nonlinear_X, alpha_X, nonlinear_Z, alpha_Z):
    """
    Plot the results of testing the method for linear causal disentanglement on sample cumulants and save the figures.

    Args:
        nlatent (int): number of latent variables.
        nobserved (int): number of observed variables.
        diff_H (list[float]): list containing the mean errors in estimating H for each number of samples.
        diff_H_abs (list[float]): list containing the mean errors in estimating abs(H) for each number of samples.
        diff_lambda (list[float]): list containing the mean errors in estimating Lambda for each number of samples.
        nsamples_list (list[int]): list containing the number of samples to use in constructing the cumulants.
        nonlinear_X (bool): boolean indicating whether the results in diff_H, diff_H_abs and diff_lambda were obtained with nonlinearity 
            in the transformation from latent to observed variables.
        alpha_X (float): coefficient quantifying how much nonlinearity was added in the transformation from latent to observed variables 
            in the experiments yielding the errors in diff_H, diff_H_abs, and diff_lambda. Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): boolean indicating whether the results in diff_H, diff_H_abs and diff_lambda were obtained with nonlinearity 
            in the latent space.
        alpha_Z (float): coefficient quantifying how much nonlinearity was added in the latent space in the experiments yielding the errors 
            in diff_H, diff_H_abs, and diff_lambda. Equals 0 if nonlinear_Z is False.
    """
    match (nonlinear_X, nonlinear_Z):
        case (False,False):
            path = "figures/sample_cumulants/{0}latent{1}observed".format(nlatent, nobserved)
        case (True, False):
            path = "figures/sample_cumulants/{0}latent{1}observed/nonlinearX/{2}coeff".format(nlatent, nobserved, alpha_X)
        case (False, True):
            path = "figures/sample_cumulants/{0}latent{1}observed/nonlinearZ/{2}coeff".format(nlatent, nobserved, alpha_Z)
        case (True, True):
            path = "figures/sample_cumulants/{0}latent{1}observed/nonlinearXZ/{2}Xcoeff{3}Zcoeff".format(nlatent, nobserved, alpha_X, alpha_Z)
    os.makedirs(path,exist_ok=True)

    # Plot error in estimating H and abs(H)
    plt.figure(figsize=(8, 6))
    plt.plot(nsamples_list, diff_H, marker='o', linestyle='-',label='H')
    plt.xlabel('Number of samples')
    plt.ylabel('Mean Frobenius error in H')
    plt.grid(True)
    plt.legend(loc='best')
    path_H = path + "/H.png"
    plt.savefig(path_H)
    plt.plot(nsamples_list, diff_H_abs, marker='o', linestyle='-', label='abs(H)',color='green')
    plt.legend(loc='best')
    path_H_abs = path + "/HvsHabs.png"
    plt.savefig(path_H_abs)

    # Plot error in estimating Lambda
    plt.figure(figsize=(8, 6))
    plt.plot(nsamples_list, diff_lambda, marker='o', linestyle='-',label=r'$\Lambda$')
    plt.xlabel('Number of samples')
    plt.ylabel('Mean Frobenius error in ' + r'$\Lambda$')
    plt.grid(True)
    plt.legend()
    path_lambda = path + "/Lambda"
    plt.savefig(path_lambda)

    # Plot error in estimating H vs error in estimating cumulants
    _, ax1 = plt.subplots(figsize=(8,6))
    Hline, = ax1.plot(nsamples_list, diff_H, 'g-',label="H")
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Mean Frobenius error in H', color='k')

    ax2 = ax1.twinx()
    tensorsline, = ax2.plot(nsamples_list, tensor_diff, 'b-',label="Cumulants")
    ax2.set_ylabel('Mean Frobenius error in cumulant tensors', color='k')

    lines1 = [Hline, tensorsline]
    labels1 = [line.get_label() for line in lines1]
    ax1.legend(lines1, labels1, loc='upper right')

    path_Hvscum = path + "/Hvscumulants.png"
    plt.savefig(path_Hvscum)

    # Plot error in estimating Lambda vs error in estimating cumulants
    _, ax3 = plt.subplots(figsize=(8,6))

    Lambdaline, = ax3.plot(nsamples_list, diff_lambda, 'g-',label=r'$\Lambda$')
    ax3.set_xlabel('Number of samples')
    ax3.set_ylabel('Mean Frobenius error in ' + r'$\Lambda$', color='k')

    ax4 = ax3.twinx()
    tensorsline, = ax4.plot(nsamples_list, tensor_diff, 'b-',label="Cumulants")
    ax4.set_ylabel('Mean Frobenius error in cumulant tensors', color='k')

    lines = [Lambdaline, tensorsline]
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc='upper right')

    path_Lambdavscum = path + "/Lambdavscumulants"
    plt.savefig(path_Lambdavscum)

def main(nsamples_list,nlatent,nobserved,nonlinear_X,alpha_X,nonlinear_Z,alpha_Z):
    diff_H, diff_H_abs, diff_lambda, tensor_diff = run_sample(nsamples_list,nobserved,nlatent,500,nonlinear_X,alpha_X,nonlinear_Z,alpha_Z)
    plot(nlatent,nobserved,diff_H,diff_H_abs,diff_lambda,tensor_diff,nsamples_list,nonlinear_X,alpha_X,nonlinear_Z,alpha_Z)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nlatent", type=int, default=4)
    parser.add_argument("--nobserved", type=int, default=5)
    parser.add_argument("--nsamples_list", type=int, nargs="+", default=[2500, 5000, 10000, 30000, 50000, 100000, 250000])
    parser.add_argument("--nonlinear_X", action='store_true')
    parser.add_argument("--alpha_X", type=float)
    parser.add_argument("--nonlinear_Z", action='store_true')
    parser.add_argument("--alpha_Z", type=float)
    args = parser.parse_args()
    if args.nlatent > args.nobserved:
        parser.error("the number of latent variables must be at least that of observed")
    if args.nonlinear_X and args.alpha_X is None:
        args.alpha_X = 0.1
    elif not args.nonlinear_X and args.alpha_X is not None:
        parser.error("argument --alpha_X should not be provided when --nonlinear_X is False")
    if args.nonlinear_Z and args.alpha_Z is None:
        args.alpha_Z = 0.1
    elif not args.nonlinear_Z and args.alpha_Z is not None:
        parser.error("argument --alpha_Z should not be provided when --nonlinear_Z is False")
    main(args.nsamples_list,args.nlatent,args.nobserved,args.nonlinear_X,args.alpha_X,args.nonlinear_Z,args.alpha_Z)