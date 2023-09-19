import numpy as np
import math
from scipy.optimize import minimize
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import itertools
import numdifftools as nd
import time
import esr.fitting.test_all as test_all
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier
import matplotlib.pyplot as plt
from scipy.stats import mode
import traceback

warnings.filterwarnings("ignore")

use_relative_dx = True            # CHANGE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
    
def load_loglike(comp, likelihood, data_start, data_end, split=True):
    """Load results of optimisation completed by test_all.py
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :data_start (int): minimum index of results we want to load (only if split=True)
        :data_end (int): maximum index of results we want to load (only if split=True)
        :split (bool, deault=True): whether to return subset of results given by data_start and data_end (True) or all data (False)
        
    Returns:
        :negloglike (list): list of minimum log-likelihoods
        :params (np.ndarray): list of parameters at maximum likelihood points. Shape = (nfun, nparam).

    """
    if rank == 0:
        print(likelihood.out_dir + "/negloglike_comp"+str(comp)+".dat")
    data = np.genfromtxt(likelihood.out_dir + "/negloglike_comp"+str(comp)+".dat")
    negloglike = np.atleast_1d(data[:,0])
    params = np.atleast_2d(data[:,1:])
    if split:
        negloglike = negloglike[data_start:data_end]               # Assuming same order of fcn and chi2 files
        params = params[data_start:data_end,:]
    return negloglike, params


def convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike, max_param=4):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for single function
    
    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :eq (sympy object): sympy object for the function we wish to fit to data
        :integrated (bool): whether eq_numpy has already been infisher matrixtegrated
        :theta_ML (list): the maximum likelihood values of the parameters
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :negloglike (float): the minimum log-likelihood for this function
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
    
    Returns:
        :params (list): the corrected maximum likelihood values of the parameters
        :negloglike (float): the corrected minimum log-likelihood for this function
        :deriv (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :codelen (float): the parameteric contribution to the description length of this function
        
    """
    nparam = simplifier.count_params([fcn_i], max_param)[0]
    infln_singularity = 0
    
    if nparam > 0:
        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_s = sympy.sympify(fcn_i,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
            derivative_expr = sympy.diff(eq_s, x)
            V_1 = sympy.lambdify([x] + all_a, derivative_expr, 'numpy')
            derivative_expr2 = sympy.diff(derivative_expr, x)
            V_2 = sympy.lambdify([x] + all_a, derivative_expr2, 'numpy')
            def fop(xx): 
                return likelihood.negloglike(xx,eq_numpy,fcn_i, dd1=V_1, dd2=V_2, integrated=integrated)
        
        else:       # 1 parameter
            eq_s = sympy.sympify(fcn_i,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0})
            derivative_expr = sympy.diff(eq_s, x)
            V_1 = sympy.lambdify([x, a0], derivative_expr, 'numpy')
            derivative_expr2 = sympy.diff(derivative_expr, x)
            V_2 = sympy.lambdify([x, a0], derivative_expr2, 'numpy')            
            def fop(xx):
                return likelihood.negloglike(xx,eq_numpy,fcn_i, dd1=V_1, dd2=V_2, integrated=integrated)

    else:  # 0 parameters
        eq_s = sympy.sympify(fcn_i, locals={"inv": inv, "square": square, "cube": cube, "x": x})
        derivative_expr = sympy.diff(eq_s, x)
        V_1 = sympy.lambdify(x, derivative_expr, 'numpy')
        derivative_expr2 = sympy.diff(derivative_expr, x)
        V_2 = sympy.lambdify(x, derivative_expr2, 'numpy')
        def fop(xx):
            return likelihood.negloglike(xx,eq_numpy,fcn_i,dd1=V_1, dd2=V_2, integrated=integrated)

    params = np.zeros(max_param)
    deriv = np.full(int(max_param * (max_param + 1) / 2), np.nan)

    # Step-sizes to try in case the function misbehvaes
    d_list = [.1,1.e-5, 10.**(-5.5), 10.**(-4.5), 1.e-6, 1.e-4, 10.**(-6.5), 10.**(-3.5), 1.e-7, 1.e-3, 10.**(-7.5), 10.**(-2.5), 1.e-8, 1.e-2, 1.e-9, 1.e-10, 1.e-11]
    
    method_list = ["central", "forward", "backward"]
    # method_list = ["central", "complex", "multicomplex"]

    if nparam == 0:
        codelen = np.nan
        return params, negloglike, deriv, codelen, infln_singularity
    
    

    try:
        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
    except Exception:
        print("BAD:", fcn_i, negloglike, np.isfinite(negloglike))
        Fisher_diag = np.nan
        deriv[:] = np.nan
        return params, negloglike, deriv, codelen, infln_singularity
    
    

    # Get Hessian
    theta_ML = theta_ML[:nparam]
    Hfun = nd.Hessian(fop)#,step=1e-13)
    Hmat = Hfun(theta_ML)
    
    # 2nd derivatives of -log(L) wrt params
    Fisher_diag = np.array([Hmat[i,i] for i in range(nparam)])
    # Precision to know constants
    Delta = np.sqrt(12./Fisher_diag)
    
    Nsteps = abs(np.array(theta_ML))/Delta
    
    for i in range(nparam):
        start = int(i * max_param - (i - 1) * i / 2)
        deriv[start:start+nparam-i] = Hmat[i,i:]
    
    n_iter = len(d_list)*len(method_list)
    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0):#  or (np.sum(Nsteps<1) > 0):
       Fisher_array = np.empty((n_iter, nparam))
       Hmat_array = np.empty((n_iter, nparam, nparam))
       e = 0
       for d2, meth in itertools.product(d_list, method_list):
            if use_relative_dx:
                Hfun = nd.Hessian(fop, step = np.abs(d2*theta_ML)+1.e-15, method=meth)
            else:
                Hfun = nd.Hessian(fop, step = d2, method=meth)
            Hmat = Hfun(theta_ML)
            Hmat_array[e] = Hmat
            e += 1

       Hmat_array_f = [] # filter array
       for matrix in Hmat_array:
           if not np.any(np.isnan(matrix)) and not np.any(np.isinf(matrix)) and np.all(np.diagonal(matrix) > 0):
               Hmat_array_f.append(matrix)
       Hmat_array_f = np.array(Hmat_array_f)
       Fisher_array = np.array([np.array([mat[i,i] for i in range(nparam)]) for mat in Hmat_array_f])
       Delta_array = np.sqrt(12./Fisher_array)
       Delta_array_round = [[format(num, ".3e") for num in row] for row in Delta_array]
       Delta_array_round = np.array(Delta_array_round, dtype=float)
       if len(Delta_array_round.shape) < 2:
           repeated_elements_exist = False
       else:
           repeated_elements_exist = len(Delta_array_round[:,0]) != len(set(Delta_array_round[:,0]))

       if repeated_elements_exist:
           Delta_mode = mode(Delta_array_round)[0][0]
           mode_ind = np.where(Delta_array_round == Delta_mode)[0][0]
           Fisher_diag = np.atleast_1d(Fisher_array[mode_ind])
           Delta = np.sqrt(12./Fisher_diag)
           Nsteps = abs(np.array(theta_ML))/Delta
           for i in range(nparam):
               start = int(i * max_param - (i - 1) * i / 2)
               deriv[start:start+nparam-i] = Hmat_array_f[mode_ind][i,i:]
       
       else: # try again with less precision
           Delta_array_round = [[format(num, ".1e") for num in row] for row in Delta_array]
           Delta_array_round = np.array(Delta_array_round, dtype=float)
           if len(Delta_array_round.shape) < 2:
               repeated_elements_exist = False
           else:
               repeated_elements_exist = len(Delta_array_round[:,0]) != len(set(Delta_array_round[:,0]))
           if not repeated_elements_exist:
               codelen = np.nan
               return params, negloglike, deriv, codelen, infln_singularity
           else:
               Delta_mode = mode(Delta_array_round)[0][0]
               mode_ind = np.where(Delta_array_round == Delta_mode)[0][0]
               Fisher_diag = np.atleast_1d(Fisher_array[mode_ind])
               Delta = np.sqrt(12./Fisher_diag)
               Nsteps = abs(np.array(theta_ML))/Delta  
               for i in range(nparam):
                   start = int(i * max_param - (i - 1) * i / 2)
                   deriv[start:start+nparam-i] = Hmat_array_f[mode_ind][i,i:]

    # Must indicate a bad fcn, so just need to make sure it doesn't have a good -log(L)
    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
        codelen = np.nan
        return params, negloglike, deriv, codelen, infln_singularity
    
    k = nparam
    theta_ML_orig = np.copy(theta_ML)
    negloglike_orig = np.copy(negloglike)
    
    if np.sum(Nsteps<1)>0:         # Possibly should reevaluate -log(L) with the param(s) set to 0, but doesn't matter unless the fcn is a very good one

        # First try setting any parameter to 0 that doesn't have at least
        # one precision step, and recompute -log(L).
        theta_ML[Nsteps<1] = 0.
        negloglike = fop(theta_ML)
        
        if np.isfinite(negloglike) and not negloglike==1e10:
            k -= np.sum(Nsteps<1)
            kept_mask = Nsteps>=1
        else:
            # Let's see if setting any of the parameters to zero is ok
            try_idx = np.arange(nparam)[Nsteps < 1]
            for r in reversed(range(1, len(try_idx))):
                for idx in itertools.combinations(try_idx, r):
                    theta_ML = np.copy(theta_ML_orig)
                    for idx_ in idx:
                        theta_ML[idx_] = 0.
                    negloglike = fop(theta_ML)
                    if np.isfinite(negloglike) and not negloglike==1e10:
                        break
            kept_mask = np.ones(len(theta_ML), dtype=bool)
            if np.isfinite(negloglike) and not negloglike==1e10: # can set to zero normally
                infln_singularity = -1
                k -= len(idx)
                codelen = np.nan
                negloglike = np.nan
                return params, negloglike, deriv, codelen, infln_singularity

            elif negloglike==1e10: # this parameter list doesn't give inflation
                infln_singularity = 1
                theta_ML = theta_ML_orig
                Fisher_diag[Nsteps<1] = 12./(theta_ML_orig[Nsteps<1]**2) #########
                negloglike = negloglike_orig
                k = nparam
            elif not np.isfinite(negloglike): # situation such as 1/a0, a0=0
                codelen = np.nan
                return params, negloglike, deriv, codelen, infln_singularity
                
                
            
        if k<0:
            print("This shouldn't have happened", flush=True)
            quit()
        elif k==0:
            codelen = np.nan
            # not interested in functions with no parameters so we discard it
            negloglike = np.nan
            return params, negloglike, deriv, codelen, infln_singularity
        
        Fisher_diag = Fisher_diag[kept_mask]     # Only consider these parameters in the codelen
        theta_ML = theta_ML[kept_mask]
    else:
        kept_mask = np.ones(len(theta_ML), dtype=bool)
    
    
    codelen = -k/2.*math.log(3.) + np.sum( 0.5*np.log(Fisher_diag) + np.log(abs(np.array(theta_ML))) )
    
    if codelen==0:
        codelen=np.nan # not interested in functions without parameters

    # New params after the setting to 0, padded to length max_param as always
    theta_ML = theta_ML_orig
    theta_ML[~kept_mask] = 0.
    params[:] = np.pad(theta_ML, (0, max_param-len(theta_ML)))
    return params, negloglike, deriv, codelen, infln_singularity


    
def main(comp, likelihood, tmax=5, try_integration=False):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for all functions and save to file
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        
    Returns:
        None
    
    """


    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')

    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood)
    negloglike, params_proc = load_loglike(comp, likelihood, data_start, data_end)
    max_param = params_proc.shape[1]

    codelen = np.zeros(len(fcn_list_proc))          # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), max_param])
    deriv = np.zeros([len(fcn_list_proc), int(max_param * (max_param+1) / 2)])
    infl_sing = np.zeros(len(fcn_list_proc))
    
    finish_fn = len(fcn_list_proc) -1 

    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank==0:    
            print(i, len(fcn_list_proc),'F', flush=True)
            
        if np.isnan(negloglike[i]) or np.isinf(negloglike[i]):
            codelen[i]=np.nan
            continue

        theta_ML = params_proc[i,:]
            
        if negloglike[i]<-8:
            tlim = 60 # after looking at exceptions I noticed many very good functions being timed out 
        else:
            tlim = 10 
        try:
            with simplifier.time_limit(tlim):
                fcn_i = fcn_list_proc[i].replace('\n', '')
                fcn_i = fcn_list_proc[i].replace('\'', '')
                fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
                params[i,:], negloglike[i], deriv[i,:], codelen[i], infl_sing[i] = convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param)
        except NameError as namer:
            print('NameError', namer)
            traceback.print_exc()
                # Occurs if function produced not implemented in numpy
            if try_integration:
                with simplifier.time_limit(tlim):
                    fcn_i = fcn_list_proc[i].replace('\n', '')
                    fcn_i = fcn_list_proc[i].replace('\'', '')
                    fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                    params[i,:], negloglike[i], deriv[i,:], codelen[i], infl_sing[i] = convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param)
            else:
                params[i,:] = 0.
                deriv[i,:] = 0.
                codelen[i] = 0
                infl_sing[i] = 0

        except Exception as exc:
            if negloglike[i]<-8:
                print(exc,'for',fcn_list_proc[i], negloglike[i])
            params[i,:] = 0.
            deriv[i,:] = 0.
            codelen[i] = 0
            infl_sing[i] = 0

        
    out_arr = np.transpose(np.vstack([codelen, negloglike] + [params[:,i] for i in range(max_param)]))
    out_arr_inflsing = np.transpose([infl_sing])
    
    out_arr_deriv = np.transpose(np.vstack([deriv[:,0], deriv[:,1], deriv[:,2], deriv[:,3], deriv[:,4], deriv[:,5], deriv[:,6], deriv[:,7], deriv[:,8], deriv[:,9]]))
    out_arr_deriv = np.transpose(np.vstack([deriv[:,i] for i in range(deriv.shape[1])]))
    
    np.savetxt(likelihood.temp_dir + '/infl_sing_'+str(comp)+'_'+str(rank)+'.dat', out_arr_inflsing, fmt='%.1e')
    np.savetxt(likelihood.temp_dir + '/codelen_deriv_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    np.savetxt(likelihood.temp_dir + '/derivs_'+str(comp)+'_'+str(rank)+'.dat', out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_deriv_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_comp'+str(comp)+'_deriv.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_deriv_'+str(comp)+'_*.dat'
        os.system(string)
        
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "infl_sing_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/infl_sing_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/infl_sing_'+str(comp)+'_*.dat'
        os.system(string)
        
        
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "derivs_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/derivs_'+str(comp)+'_*.dat'
        os.system(string)
        
    comm.Barrier()

    return
