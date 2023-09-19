import numpy as np
import math
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import numdifftools as nd
import time
import itertools
import esr.fitting.test_all as test_all
import esr.fitting.test_all_Fisher as test_all_Fisher
from esr.fitting.sympy_symbols import *

import esr.generation.simplifier as simplifier

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, likelihood, tmax=5, try_integration=False):
    """Apply results of fitting the unique functions to all functions and save to file
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        
    Returns:
        None
        
    """
    if rank==0:
        print('Entered match.py')
    
    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')

    def fop(xx):
        eq_s = sympy.sympify(fcn_i,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
        derivative_expr = sympy.diff(eq_s, x)
        V_1 = sympy.lambdify([x] + all_a, derivative_expr, 'numpy')
        derivative_expr2 = sympy.diff(derivative_expr, x)
        V_2 = sympy.lambdify([x] + all_a, derivative_expr2, 'numpy')
        try:
            return likelihood.negloglike(xx,eq_numpy,fcn_i, dd1=V_1, dd2=V_2, integrated=integrated)
        except:
            return np.nan
        
    def f1(xx):
        eq_s = sympy.sympify(fcn_i,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0})
        derivative_expr = sympy.diff(eq_s, x)
        V_1 = sympy.lambdify([x, a0], derivative_expr, 'numpy')
        derivative_expr2 = sympy.diff(derivative_expr, x)
        V_2 = sympy.lambdify([x, a0], derivative_expr2, 'numpy')
        try:
            return likelihood.negloglike(xx,eq_numpy,fcn_i, dd1=V_1, dd2=V_2, integrated=integrated)
        except:
            return np.nan
        
    invsubs_file = likelihood.fn_dir + "/compl_%i/inv_subs_%i.txt"%(comp,comp)
    match_file = likelihood.fn_dir + "/compl_%i/matches_%i.txt"%(comp,comp)
    infl_sing_file = likelihood.out_dir + "/infl_sing_comp%i.dat"%(comp)
    
    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood, unique=False)
    negloglike, params_meas = test_all_Fisher.load_loglike(comp, likelihood, data_start, data_end, split=False)
    max_param = params_meas.shape[1]
    
    all_inv_subs_proc = simplifier.load_subs(invsubs_file, max_param)[data_start:data_end]
    matches_proc = np.atleast_1d(np.loadtxt(match_file).astype(int))[data_start:data_end]
    infl_sing_file = np.atleast_1d(np.loadtxt(infl_sing_file).astype(int))

    all_fish = np.loadtxt(likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat')   # 2D array of shape (# unique fcns, 10)
    all_fish = np.atleast_2d(all_fish)

    codelen = np.zeros(len(fcn_list_proc))              # Both of these are also just for this proc
    negloglike_all = np.zeros(len(fcn_list_proc))
    index_arr = np.zeros(len(fcn_list_proc))
    params = np.zeros([len(fcn_list_proc), max_param])

    for i in range(len(fcn_list_proc)):                 # The part of all eqs analysed by this proc
        if i%1000==0 and rank==0:
            print(i, len(fcn_list_proc))

        fcn_i = fcn_list_proc[i].replace('\'', '')
        
        nparams = simplifier.count_params([fcn_i], max_param)[0]

        index = matches_proc[i]          # Index in total unique eqs file, common to all procs
        
        infl_sing_i = infl_sing_file[index] 
        

        if infl_sing_i==-1:
            continue
        
        index_arr[i] = index

        negloglike_all[i] = negloglike[index]           # Assign the likelihood of this variant to the that of the unique eq
        
        if np.isnan(negloglike[index]) or np.isinf(negloglike[index]):          # Element of the unique eqs file, common to all procs
            codelen[i] = np.nan
            continue

        if nparams==0:
            codelen[i] = np.nan
            continue
        else:
            k = nparams
            measured = params_meas[index,:nparams]
        
        fish_measured = all_fish[index,:]               # Access from the unique eqs all_fish array, common to all procs
                
        try:
            p, fish = simplifier.convert_params(measured, fish_measured, all_inv_subs_proc[i], n=max_param)
            if isinstance(p, float):
                p=[p]
            p = np.atleast_1d(p)
        except Exception as ex:
            codelen[i] = np.inf
            continue
        
        if np.sum(fish<=0)>0:
            codelen[i] = np.inf
            continue
        
        Delta = np.zeros(fish.shape)
        m = (fish != 0)
        Delta[m] = np.atleast_1d(np.sqrt(12./fish[m]))
        Delta[~m] = np.inf
        Nsteps = np.atleast_1d(np.abs(np.array(p)))
        m = (Delta != 0)
        Nsteps[m] /= Delta[m]
        Nsteps[~m] = np.nan
        
        negloglike_orig = np.copy(negloglike_all[i])
        ptrue=np.copy(p)
        
        if np.sum(Nsteps<1)>0:         # should reevaluate -log(L) with the param(s) set to 0, but doesn't matter unless the fcn is a very good one
            try:
                p[Nsteps<1] = 0.         # Set any parameter to 0 that doesn't have at least one precision step, and recompute -log(L).
            except (IndexError, TypeError):
                p=0.
            
            try:            # It's possible that after putting params to 0 the likelihood is botched, in which case give it nan
                fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
                if k==1:
                    eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                    negloglike_all[i] = f1(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
                else:
                    all_a = ' '.join([f'a{i}' for i in range(nparams)])
                    all_a = list(sympy.symbols(all_a, real=True))
                    eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
                    negloglike_all[i] = fop(p)

            except NameError:
                if try_integration:
                    fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                    if k==1:
                        eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                        negloglike_all[i] = f1(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
                    else:
                        all_a = ' '.join([f'a{i}' for i in range(nparams)])
                        all_a = list(sympy.symbols(all_a, real=True))
                        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
                        negloglike_all[i] = fop(p)
                else:
                    negloglike_all[i] = np.nan

            except Exception as exc:
                print('Exception', exc)
                negloglike_all[i] = np.nan

             
            if np.isfinite(negloglike_all[i]) and not negloglike_all[i]==1e10:
                k -= np.sum(Nsteps<1)
                kept_mask = Nsteps>=1
            else:
                # Let's see if setting any of the parameters to zero is ok
                try_idx = np.arange(nparams)[Nsteps < 1]
                for r in reversed(range(1, len(try_idx))):
                    for idx in itertools.combinations(try_idx, r):
                        p = np.copy(ptrue)
                        p[idx] = 0.
                        if k==1:
                            negloglike_all[i] = f1(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
                        else:
                            negloglike_all[i] = fop(p)
                        
                        if np.isfinite(negloglike_all[i]) and not negloglike_all[i]==1e10:
                            break
                kept_mask = np.ones(len(p), dtype=bool)
                if np.isfinite(negloglike_all[i]) and not negloglike_all[i]==1e10:
                    k -= len(idx)
                    kept_mask[idx] = 0
                elif negloglike_all[i]==1e10 and infl_sing_i!=-1:
                    p = ptrue
                    fish[Nsteps<1] = 12./(p[Nsteps<1]**2)
                    negloglike_all[i] = negloglike_orig
                    
                elif not np.isfinite(negloglike_all[i]): # situation such as 1/a0, a0=0
                    continue
            
            if k<0:
                print("This shouldn't have happened", flush=True)
                quit()
            elif k==0:                  # If we have no parameters left then the parameter codelength is 0 so we can move on
                codelen[i] = np.nan # not interested in functions with no parameters
                continue
            
            
            fish = fish[kept_mask]
            p = p[kept_mask]
            
        else:
            kept_mask = np.ones(len(p), dtype=bool)
        
        try:
            codelen[i] = -k/2.*math.log(3.) + np.sum( 0.5*np.log(fish) + np.log(abs(np.array(p))) )
        except:
            codelen[i] = np.nan
        
        p = ptrue
        p[~kept_mask]=0.
        
        if list(p).count(0)>(len(list(p))-nparams) and len(list(p))>1: # if one of the param is zero, this fcn already appeared, optimised, at lower complexity
            codelen[i] = np.nan
            continue

        if list(p).count(0)==len(list(p)) or codelen[i]==0: # not interested in fcn with no param
            codelen[i] = np.nan
            continue

        try:        # If p was an array, we can make a list out of it
            list_p = list(p)
            params[i,:] = np.pad(p, (0, max_param-len(p)))
        except:     # p is either a number or nothing
            if p:   # p is a number
                params[i,:] = 0
                params[i,0] = p
            else:
                params[i,:] = np.zeros(max_param)
        
        assert len(params[i,:])==max_param
        if negloglike_all[i]==1e10:
            negloglike_all[i]=np.nan
        

    
    out_arr = np.transpose(np.vstack([negloglike_all, codelen, index_arr] + [params[:,i] for i in range(max_param)]))
    
    np.savetxt(likelihood.temp_dir + '/codelen_matches_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')        # Save the data for this proc in Partial
    
    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_matches_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_matches_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_matches_'+str(comp)+'_*.dat'
        os.system(string)
        
    comm.Barrier()
        
    return

