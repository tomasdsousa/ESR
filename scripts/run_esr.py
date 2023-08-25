#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:07:34 2023

@author: tomass
"""

import sympy as sp
import esr.generation.generator as generatorx
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from scipy import optimize 
from scipy import integrate 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import warnings
import timeit
warnings.filterwarnings("error", category=integrate.IntegrationWarning)
import numba as nb

import inspect

#%% define functions
from esr.fitting.likelihood import Likelihood
import os
runname='inflation_extended'

@nb.jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64, nb.float64, nb.float64))
def trapz(xvals, yvals, start, finish, n):
    stp = (finish-start)/(n-1)
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*stp
    return area



def solve_inflation(eq_numpy, *a, fstring='', V11, V22, efolds=60, phi_max=40, print_n=True, plots=False, plot_title=''):
     """ Slow roll inflation function. Returns [100,100,100] for potentials that do not give rise to inflation, and [5,5,5]
     for potentials where the parameters create a singularity. This distinction is made just so we can make use of it in the negloglike 
     function. This version of the function is made to be inside ESR, so no constant potentials will be inputted at this point since
     they are ignored in esr/fitting/test_all.py.

        Args:
            :eq_numpy (numpy function): function to use which gives the potential function, V (esr generated function)
            :a (list): parameters
            :fstring (string): function string for plot title
            :V11 (numpy function): first derivative of eq_numpy
            :V22 (numpy function): second derivative of eq_numpy
            :efolds (float): efolds of expansion to consider
            :phi_max (float): x domain [-phi_max, phi_max] to consider
            :print_n (bool): whether to print the predictions of this potential
            :plots (bool): whether to plot the inflation region of the potential
            :plot_title (string): self-explanatory
            

        Returns:
            :ypred (list): the predictions for the spectral index, tensor-to-scalar ratio and energy scale [n, r, m]
            
        """
    

    scaling= 1e-9 # rescale units
    
    phi = np.linspace(-phi_max, phi_max, 30000)
    
    def V(x): return scaling*eq_numpy(x,*np.atleast_1d(a))  
    
    if np.isinf(V(1)) and np.isinf(V(2)):
        print('parameter singularity=',a,' in',fstring)
        return [5,5,5]
    
    def V1(x): return scaling*V11(x,*np.atleast_1d(a))
    
    def V2(x): return scaling*V22(x,*np.atleast_1d(a))
        
    def epsilon(phi0): return 0.5*( V1(phi0)/V(phi0) )**2
    
    def eta(phi0): return V2(phi0)/V(phi0)
    
    def integrand(phi0): return - V(phi0)/V1(phi0)
    
    def e_scale(phi0): return (V(phi0)/epsilon(phi0) )**(0.25)
     
    def ns(phi0): return 1 - 6*epsilon(phi0) + 2*eta(phi0) 
    
    def r(phi0): return 16 * epsilon(phi0) 
    
    def epsilon1(phi0): #function to calculate the root of. we want phi for epsilon=1 
        return epsilon(phi0) - 1
                   
    def integral(phi_star): 
        xvals1 = np.linspace(phi_star,phi_end,100)
        yvals1 = integrand(xvals1)
        return trapz(xvals1,yvals1,phi_star,phi_end, 100) - efolds
    
    "FIND PHI_END:"
    
    phi1 = np.linspace(-phi_max, phi_max, 30000) # redefinition of phi as this should work just as well with lower array size
    f = epsilon1(phi1)
    phi_ends=[]
    for i in range(len(phi1)-1):
        if f[i] * f[i+1] < 0: #there is a root in this interval
            result = optimize.root_scalar(epsilon1, bracket=[phi1[i],phi1[i+1]]) #find root
            root = result.root
            phi_ends.append(root)
    
    if len(phi_ends)==0: # unfit for inflation since \epsilon never crosses 1
        # print('100')
        return [100,100,100]
    

    "FIND PHI_START:"
    
    phi_stars=[]
    n_arr=[]
    r_arr=[]
    m_arr=[]
    for i in range(len(phi_ends)):
        phi_end=phi_ends[i]
        phi_star0=phi_end # guess for phi_start
        efold=0
        step=.2*np.abs(phi_end) # this will affect the running time since a smaller step means the integration function will be called more often
        if V(phi_end)>=0:  
            if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                while efold<efolds and phi_star0<phi_max:
                    phi_star0 +=  step 
                    xvals=np.linspace(phi_star0,phi_end,100) # creating this array for every integration iteration is what slows down this function the most 
                    yvals=integrand(xvals)
                    efold = trapz(xvals,yvals,phi_star0, phi_end,100)
                if efold>efolds:
                    phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
                    phi_star=phi_star.root
                else:
                    phi_star=1e5 #fake value just to fill array so that array sizes match
                    phi_stars.append(phi_star)
                    n_pre = 100
                    r_pre = 100
                    m_pre = 100
                    n_arr.append(n_pre)
                    r_arr.append(r_pre)
                    m_arr.append(m_pre)
                    continue # next phi_end
            else: 
                while efold<efolds and phi_star0>-phi_max:
                    phi_star0 -=  step
                    xvals=np.linspace(phi_star0,phi_end,100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals,yvals,phi_star0,phi_end,100)
                if efold>efolds:
                    phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
                    phi_star=phi_star.root
                else:
                    phi_star=1e5 #fake value just to fill array so that array sizes match
                    phi_stars.append(phi_star)
                    n_pre = 100
                    r_pre = 100
                    m_pre = 100
                    n_arr.append(n_pre)
                    r_arr.append(r_pre)
                    m_arr.append(m_pre)
                    continue # next phi_end
                

            if np.abs(phi_star)<=phi_max and epsilon(phi_star)<1 and efold!=70:
                phi_stars.append(phi_star)
                n_pre = ns(phi_star) # get n
                r_pre = r(phi_star) # get r
                m_pre = e_scale(phi_star) # get m
                n_arr.append(n_pre)
                r_arr.append(r_pre)
                m_arr.append(m_pre)
            else:
                if efold==70 or len(phi_ends)==1: # integration problems, discard function
                    # print('100')
                    return [100,100,100]
                else:
                    phi_star=1e5 #fake value just to fill array so that array sizes match
                    phi_stars.append(phi_star)
                    n_pre = 100
                    r_pre = 100
                    m_pre = 100
                    n_arr.append(n_pre)
                    r_arr.append(r_pre)
                    m_arr.append(m_pre)
        else: 
            # print(100)
            return [100,100,100]
    
    if all(t == 1e5 for t in phi_stars) or len(n_arr) == 0: # meaning we never found an actual phi_start
        # print('100')
        return [100,100,100]
    
    if len(n_arr) == 1: # this potential only has 1 inflationary trajectory
        n_out=n_arr[0]
        r_out=r_arr[0]
        m_out=m_arr[0]
        

        if plots:
            plt.figure()
            plt.plot(phi, V(phi))
            ylimmax=10*V(phi_star)
            plt.ylim(0,ylimmax)
            plt.xlim(-phi_max,phi_max)
            plt.xlabel('$\phi$')
            plt.ylabel('V($\phi$)')
            plt.title(str(plot_title))
            plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='green')
        
        if print_n:  
            print('n:', n_out)
            print('r:', r_out)
            print('m:', m_out)
        return [n_out,r_out,m_out]
        
        
    "select best pair of (n,r)" 
    # when a potential has more than one inflationary trajectory, output the one matching data better
    n_arr=np.array(n_arr)
    r_arr=np.array(r_arr)
    m_arr=np.array(m_arr)
    n_obs=0.9649
    sigma_n=0.0042 
    r_obs=0.000
    sigma_r=0.028
    m_obs=0.027
    sigma_m=0.0027
    fit_n = - np.log(sigma_n) - ((n_arr-n_obs)**2)/(2*(sigma_n**2))
    fit_r = - np.log(sigma_r) - ((r_arr-r_obs)**2)/(2*(sigma_r**2))
    fit_m = - np.log(sigma_m) - ((m_arr-m_obs)**2)/(2*(sigma_m**2))
    fit_i = fit_n + fit_r + fit_m
    arg=np.argmax(fit_i) # get best fitness
    n_out=n_arr[arg]
    r_out=r_arr[arg]
    m_out=m_arr[arg]

    phi_star = phi_stars[arg]
    if plots:
        plt.figure()
        plt.plot(phi, V(phi))
        ylimmax=10*V(phi_star)
        plt.ylim(0,ylimmax)
        plt.xlim(-phi_max,phi_max)
        plt.xlabel('$\phi$')
        plt.ylabel('V($\phi$)')
        plt.title(str(plot_title))
        plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='green')

    if print_n:
        print('n:', n_out)
        print('r:', r_out)
        print('m:', m_out)
    return n_out, r_out, m_out  
  


class GaussLikelihood(Likelihood):

    def __init__(self, data_file, run_name, data_dir=None):
        """Likelihood class used to fit a function directly using a Gaussian likelihood

        """
        super().__init__(data_file, data_file, run_name, data_dir=data_dir)
        self.ylabel = r'$r$'    # for plotting
        self.xvar, self.yvar, self.yerr = np.loadtxt(self.data_file, unpack=True)

    def negloglike(self, a, eq_numpy, fcn_i, dd1, dd2, **kwargs):
        """Negative log-likelihood for a given function.

        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y

        Returns:
            :nll (float): - log(likelihood) for this function and parameters
            
        """

        ypred = solve_inflation(eq_numpy,*a, V11=dd1, V22=dd2, fstring=fcn_i, print_n=False)

        if np.isnan(any(ypred)):
            print('solve output is nan')
            ypred = [100,100,100]
            
        if ypred==[100,100,100]:
            return 1e10
        if ypred==[5,5,5]: # some sort of 1/a0 situation
            return np.inf

        if not np.all(np.isreal(ypred)):
            return np.inf

        nll = np.sum(0.5 * (ypred - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))
        if np.isnan(nll):
            return np.inf
        return nll



#%% Fit parameters of potentials using Planck's gaussian likelihood
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.sympy_symbols
import esr.fitting.likelihood
import esr.plotting.plot

likelihood = GaussLikelihood('planck_ns.txt', 'inflation_extended', data_dir=os.getcwd())

comp=4            # no results below comp=4 for the 'inflation_extended' basis functions

esr.fitting.test_all.main(comp, likelihood, log_opt=True)
esr.fitting.test_all_Fisher.main(comp, likelihood)
esr.fitting.match.main(comp, likelihood)
esr.fitting.combine_DL.main(comp, likelihood)
# esr.plotting.plot.pareto_plot(likelihood.out_dir, 'pareto.png')
