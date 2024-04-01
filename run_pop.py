# --- Imports ---
# Standard libraries
import os

# Local
import algorithm_functions as af
import helper_functions as hp

# Third-party
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def run(nmodels):
    """
    Test performance of method for linear causal disentanglement using population cumulants as input.

    Args:
        nmodels (int): number of models to generate per value of (q, p). Errors will be averaged across them.
    
    Returns:
        latent_dict (dict): a dictionary mapping the number of latent variables to the mean errors in parameter estimation for different values of p.
            Keys are the number of latent variables (int), and the values are tuples containing the mean errors in estimating H,
                those in estimating abs(H), and those in estimating Lambda, all for values of p ranging from q to 15.
    """
    latent_dict = {}
    for nlatent in range(4,8):
        diff_H_global = []
        diff_H_abs_global = []
        diff_lambda_global = []
        for nobserved in range(nlatent,16):
            models = af.create_models(nmodels, nobserved, nlatent)
            diff_H_local = []
            diff_H_abs_local = []
            diff_lambda_local = []
            for pair in models:
                B, Lambdas = pair
                _, realTs, Products, _ = af.construct_cumulants_kstat(5,Lambdas,B,False,0,False,0)
                RProducts = af.decompose_tensors(realTs, nlatent)
                RProdnninv = af.nonnorm_products(realTs, RProducts)
                RProdnn = {key: values[0] for key, values in RProdnninv.items()}
                int_tuples = af.match_int(RProdnninv)

                perm = hp.find_best_permutation(Products["obs"],RProdnn["obs"])

                Hr = af.recover_H(int_tuples)
                
                diff_H, diff_H_abs = hp.H_diff(np.linalg.pinv(B),Hr,perm)
                diff_H_local.append(diff_H)
                diff_H_abs_local.append(diff_H_abs)

                Lambdar = af.recover_lambda(int_tuples,Hr,1e-08)
                diff_lambda = hp.lambda_diff(Lambdas["obs"],Lambdar,perm)
                diff_lambda_local.append(diff_lambda)
            # Once I have gone through my models, we save the mean. Then I should have one value per number of variables
            print("Number of latent variables is {0}".format(nlatent))
            print("Number of observed variables is {0}".format(nobserved))
            print("Mean difference in H is {0}".format(np.mean(diff_H_local)))
            print("Mean difference in H abs is {0}".format(np.mean(diff_H_abs_local)))
            print("Mean difference in Lambda is {0}".format(np.mean(diff_lambda_local)))
            diff_H_global.append(np.mean(diff_H_local))
            diff_H_abs_global.append(np.mean(diff_H_abs_local))
            diff_lambda_global.append(np.mean(diff_lambda_local))
        latent_dict[nlatent] = (diff_H_global, diff_H_abs_global, diff_lambda_global)
    return latent_dict

def plot(latent_dict, i):
    """
    Plot the results of testing the method for linear causal disentanglement on population cumulants and save the figures.

    Args:
        latent_dict (dict): a dictionary mapping the number of latent variables to the mean errors in parameter estimation for different values of p.
            Keys are the number of latent variables (int), and the values are tuples containing the mean errors in estimating H,
                those in estimating abs(H), and those in estimating Lambda, all for values of p ranging from q to 15.
        i (int): indicates what errorss to plot. If equal to 0, plot the errors in estimating H, if equal to 1, plot those in estimating abs(H), 
            and if equal to 2, plot those in estimating Lambda.
    """
    nobserved = np.linspace(4,15,12)
    plt.figure(figsize=(8, 6))
    for (key, value) in latent_dict.items():
        match i:
            case 0:
                label = name = "H"
            case 1:
                label = name = "abs(H)"
            case 2:
                label = r'$\Lambda$'
                name = "Lambda"
        yvalues = value[i]
        nones = (key-4)*[None]
        yvalues = nones + yvalues
        plt.plot(nobserved, yvalues, marker='o', linestyle='-',label='q={0}'.format(key))
    plt.xlabel('p')
    plt.ylabel('Mean Frobenius error in ' + label)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figures/pop_cumulants/{name}.png")

def main():
    latent_dict = run(500)
    os.makedirs("figures/pop_cumulants",exist_ok=True)
    for i in range(3):
        plot(latent_dict,i)

if __name__ == "__main__":
    main()