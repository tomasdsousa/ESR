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

#final7=np.loadtxt('final_7.dat',delimiter=';',unpack=False, dtype=str)

import inspect

#check 18JUl

#%% define functions
from esr.fitting.likelihood import Likelihood
import os
runname='inflation'

@nb.jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64, nb.float64, nb.float64))
def trapz(xvals, yvals, start, finish, n):
    stp = (finish-start)/(n-1)
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*stp
    return area



def solve_inflation(eq_numpy, *a, fstring='', V11, V22, efolds=60, phi_max=40, print_n=True, plots=False, plot_title=''):
    # print(a)
    scaling= 1e-9
    
    phi = np.linspace(-phi_max, phi_max, 30000)
    
    def V(x): return scaling*eq_numpy(x,*np.atleast_1d(a))  
    
    if np.isinf(V(1)) and np.isinf(V(2)):
        print('inf parameter=',a,' in',fstring)
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
    
    phi1 = np.linspace(-phi_max, phi_max, 30000)
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
        #print('end',phi_end)
        #print(phi_end)
        phi_star0=phi_end # guess for phi_start
        efold=0
        step=.2*np.abs(phi_end)
        # d2_end=V2(phi_end)
        # d2_start=V2(phi_star0)
        if V(phi_end)>=0:  
            if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                while efold<efolds and phi_star0<phi_max:# and d2_end*d2_start>0:
                    phi_star0 +=  step 
                    # d2_start=V2(phi_star0)
                    xvals=np.linspace(phi_star0,phi_end,100)
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
                while efold<efolds and phi_star0>-phi_max:# and d2_end*d2_start>0:
                    phi_star0 -=  step
                    # d2_start=V2(phi_star0)
                    xvals=np.linspace(phi_star0,phi_end,100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals,yvals,phi_star0,phi_end,100)
                if efold>efolds:
                    # print(phi_star0,phi_end)
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
    
    #print('all',phi_stars)
    if all(t == 1e5 for t in phi_stars) or len(n_arr) == 0: # meaning we never found an actual phi_start
        # print('100')
        return [100,100,100]
    
    if len(n_arr) == 1: # this potential only has 1 inflationary trajectory
        # n_out=np.round(n_arr[0],decimals=5)
        # r_out=np.round(r_arr[0], decimals=7)
        # m_out=np.round(m_arr[0], decimals=4)
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
    # n_arr=np.round(n_arr, decimals=5)
    # r_arr=np.round(r_arr, decimals=7)
    # m_arr=np.round(m_arr, decimals=4)
    n_arr=np.array(n_arr)
    r_arr=np.array(r_arr)
    m_arr=np.array(m_arr)
    # observed values (planck 2018 paper):
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
        #run_name='inflation'
        super().__init__(data_file, data_file, run_name, data_dir=data_dir)
        self.ylabel = r'$r$'    # for plotting
        self.xvar, self.yvar, self.yerr = np.loadtxt(self.data_file, unpack=True)
        #self.xvar, self.yvar, self.yerr = np.array([0,0]), np.array([0.9649,0]), np.array([0.0042,0.028])

    def negloglike(self, a, eq_numpy, fcn_i, dd1, dd2, **kwargs):
        """Negative log-likelihood for a given function.

        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y

        Returns:
            :nll (float): - log(likelihood) for this function and parameters
            
        """
        #print('a0:',a)
        #print('st:',fcn_i)

        # try:
        ypred = solve_inflation(eq_numpy,*a, V11=dd1, V22=dd2, fstring=fcn_i, print_n=False)
        # except:
        #     #ypred=np.inf
        #     ypred=np.sum(0.5 * ([100,100,100] - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))


        if np.isnan(any(ypred)):
            print('solve output is nan')
            ypred = [100,100,100]
            
        if ypred==[100,100,100]:
            return 1e10
            # return np.nan
        if ypred==[5,5,5]: # some sort of 1/a0 situation
            return np.inf

        if not np.all(np.isreal(ypred)):
            # return np.sum(0.5 * ([100,100,100] - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))
            return np.inf
            # return np.nan

        nll = np.sum(0.5 * (ypred - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))
        if np.isnan(nll):
            # return np.sum(0.5 * ([100,100,100] - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))
            return np.inf
            # return np.nan
        return nll



#%% Fit parameters of potentials using Planck's gaussian likelihood
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.sympy_symbols
import esr.fitting.likelihood
import esr.plotting.plot

likelihood = GaussLikelihood('Desktop/planck_ns.txt', 'inflation_extended', data_dir=os.getcwd())
# likelihood = GaussLikelihood('planck_ns.txt', 'inflation', data_dir=os.getcwd())

comp=4

# esr.fitting.test_all.main(comp, likelihood, log_opt=True)
# esr.fitting.test_all_Fisher.main(comp, likelihood)
# esr.fitting.match.main(comp, likelihood)
esr.fitting.combine_DL.main(comp, likelihood)
#esr.fitting.plot.main(comp, likelihood)
# esr.plotting.plot.pareto_plot(likelihood.out_dir, 'pareto_aif.png')
# esr.plotting.plot.pareto_plot('Glamdring_results/results_AIFe_19Aug/', 'pareto_aif.png')


#%%  fitting single
# from esr.fitting.fit_single import single_function, fit_from_string
# likelihood = GaussLikelihood('planck_ns.txt', 'inflation', data_dir=os.getcwd())

# labels = ["square", "+", "a0", "x"]
# # labels = ["sqrt_abs","*", "x", "a0" ]
# # labels = ["/", "sqrt_abs","x", "a0" ]
# # labels = ["*", "a0", "square", "x"]
# # labels = ["+", "sqrt_abs", "x", "inv", "a0"]
# # labels = ["+", "sqrt_abs", "x",  "a0"]
# # labels = ["+", "a0", "inv", "x"]
# # labels= ["-", "exp", "a0", "square", "exp", "x"]
# # labels = ["*", "a0", "-", "1", "square", "/", "x", "a1"]
# # labels = ["-", "a0", "cube", "exp", "x"]
# # labels = ["square", "-", "a0", "x"]
# # labels = ['/' ,'a0' ,'exp' ,'exp' ,'x']
# # labels = ["square", "+", "a0", "exp", "x"]
# # labels = ["square", "-", "a0", "exp", "*", "a1", "x"]
# # labels = ["*","a0","square", "-", "a1", "exp", "*", "a2", "x"]
# # labels = ["exp", "*", "a0", "-", "a1", "exp", "x"]
# # labels = ["*", "a0", "+", "1", "cos", "*", "a1", "x"]
# basis_functions = [["x", "a"],  # type0
#                    ["inv", "square", "cube", "exp", "log", "sin", "sqrt_abs"],  # type1
#                    ["+", "*", "-", "/"]]  # type2

# logl_lcdm_si, dl_lcdm_si = single_function(labels, basis_functions,
#             likelihood,verbose=True, log_opt=False, pmin=-1, pmax=1)

# logl_lcdm_si, dl_lcdm_si, lbls = fit_from_string('a0 - exp(2*x)', basis_functions,
            # likelihood,verbose=True, log_opt=True, pmin=-11, pmax=-6)

#%%   make contour plots

# def negloglike_jax(equation, *a):
#     yv = np.array([0.9649, 0.0, 0.027])
#     ye = np.array([0.0042 , 0.028, 0.0027])
#     # yv = np.array([ 0.0])#, 0.027])
#     # ye = np.array([ 0.028])#, 0.0027])
#     # yv = np.array([ 0.027])
#     # ye = np.array([0.0027])
    
#     # trypred = solve_inflation(equation, *a)
#     # if trypred==[1]
#     yvector = np.array(solve_inflation(equation, *a))
#     if yvector[0]==100:
#         return np.inf
    
#     nll = np.sum(0.5 * (yvector - yv) ** 2 / ye ** 2 + 0.5 * np.log(2 * np.pi ) + np.log(ye) )
#     # print('nll',nll)
#     return nll

# def V(xx,*a):
#     # return (a[0] - np.exp(a[1]*xx))**2
#     # return np.exp(a[0]*(a[1] - np.exp(xx)))
#     return a[0]*(1  + np.cos(xx*a[1]))
#     # return a[0] - np.exp(xx)**3
#     # return np.exp(a[0]) - np.exp(xx)**2
#     # return a[0] + xx**2

# # print(negloglike_jax(V,[8e-12]))
# # st=' (a0 - exp(a1*x))**2'
# # eq=sp.sympify(st, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
# # derivative_expr = sp.diff(eq, x)
# # V1 = sp.lambdify(x, derivative_expr, 'numpy')
# # derivative_expr2 = sp.diff(derivative_expr, x)
# # V2 = sp.lambdify(x, derivative_expr2, 'numpy')

# # print(V1(3))

# def solve_inflation(eq_numpy, *a, fstring='', efolds=60, phi_max=400, print_n=True, plots=False, plot_title=''):

#     phi = np.linspace(-phi_max, phi_max, 30000)
#     # step=80/2999
#     #print('a2:',a)
#     # source_code = inspect.getsource(eq_numpy)
#     # print(source_code)
#     print('a:',a)
#     scaling = 1
#     def V(x): 
#         return scaling*eq_numpy(x,*np.atleast_1d(a))
    

#     # if len(a)==0:
#     #     #print('--------------')
#     #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x})
#     # if len(a)==1:
#     #     a0=a[0]
#     #     #print('--------------')
#     #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x, "a0": a0})
#     # elif len(a)==2:
#     #     a0=a[0]
#     #     a1=a[1]
#     #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1})
#     # elif len(a)==3:
#     #     a0=a[0]
#     #     a1=a[1]
#     #     a2=a[2]
#     #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2})
#     # elif len(a)==4:
#     #     a0=a[0]
#     #     a1=a[1]
#     #     a2=a[2]
#     #     a3=a[3]
#     #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})

#     # derivative_expr = sp.diff(eq, x)
#     # V1 = sp.lambdify(x, derivative_expr, 'numpy')
    
#     # derivative_expr2 = sp.diff(derivative_expr, x)
#     # V2 = sp.lambdify(x, derivative_expr2, 'numpy')
        
#     def epsilon(phi0):
#         return 0.5*( (scaling*V1(phi0))/V(phi0) )**2
    
#     def eta(phi0):
#         return (scaling*V2(phi0))/V(phi0)
    
#     def integrand(phi0):
#         return - V(phi0)/(V1(phi0)*scaling)
    
#     def e_scale(phi0):
#         return (V(phi0)/epsilon(phi0) )**(0.25)
     
#     def ns(phi0): #spectral index 
#       return 1 - 6*epsilon(phi0) + 2*eta(phi0) 
    
#     def r(phi0): # tensor-to-scalar ratio 
#         return 16 * epsilon(phi0) 
    
#     def epsilon1(phi0): #function to calculate the root of. we want phi for epsilon=1 
#         return epsilon(phi0) - 1
                   
#     def integral(phi_star): 
#         # try:
#         xvals1 = np.linspace(phi_star,phi_end,100)
#         yvals1 = integrand(xvals1)
#         return trapz(xvals1,yvals1,phi_star,phi_end, 100) - efolds
#             # return integrate.quad(integrand, phi_star, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0] - efolds
#         #except integrate.IntegrationWarning:
#         #     print(111)
#             # return 0
#         # except:
#             # print(222)
#             # return 0
    
#     "FIND PHI_END:"
    
#     phi1 = np.linspace(-phi_max, phi_max, 30000)
#     f = epsilon1(phi1)
#     phi_ends=[]
#     for i in range(len(phi1)-1):
#         if f[i] * f[i+1] < 0: #there is a root in this interval
#             result = optimize.root_scalar(epsilon1, bracket=[phi1[i],phi1[i+1]]) #find root
#             root = result.root
#             phi_ends.append(root)
#     # print(phi_ends)
#     if len(phi_ends)==0: # unfit for inflation since \epsilon never crosses 1
#         print('100')
#         return [100,100,100]
#         # return [100]

#     "FIND PHI_START:"
    
#     phi_stars=[]
#     n_arr=[]
#     r_arr=[]
#     m_arr=[]
#     for i in range(len(phi_ends)):
#         phi_end=phi_ends[i]
#         #print('end',phi_end)
#         #print(phi_end)
#         phi_star0=phi_end # guess for phi_start
#         efold=0
#         step=.2*np.abs(phi_end)
#         if V(phi_end)>=0:  
#             if V(phi_end+1e-4)>V(phi_end): # check direction of integration
#                 while efold<efolds and phi_star0<phi_max:# and eta(phi_star0)<1:
#                     # print(phi_end)
#                     # print('eta',eta(phi_star0))
#                     phi_star0 +=  step
#                     # try: 
#                     xvals=np.linspace(phi_star0, phi_end, 100)
#                     yvals=integrand(xvals)
#                     efold = trapz(xvals, yvals, phi_star0, phi_end ,100)
#                     # print('own',efold)
#                     # efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
#                     # print('sci',efold)
#                     # except integrate.IntegrationWarning:
#                         # efold=70 # exit while loop
#                     # except:
#                         #print(4)
#                         # efold=70 # exit while loop
#                 if efold>efolds:
#                     phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                     phi_star=phi_star.root
#                 else:
#                     phi_star=1e5 #fake value just to fill array so that array sizes match
#                     phi_stars.append(phi_star)
#                     n_pre = 100
#                     r_pre = 100
#                     m_pre = 100
#                     n_arr.append(n_pre)
#                     r_arr.append(r_pre)
#                     m_arr.append(m_pre)
#                     continue # next phi_end
#             else: 
#                 while efold<efolds and phi_star0>-phi_max:# and eta(phi_star0)<1:
#                     phi_star0 -=  step
#                     # try: 
#                     xvals=np.linspace(phi_star0,phi_end,100)
#                     yvals=integrand(xvals)
#                     efold = trapz(xvals,yvals,phi_star0,phi_end,100)
#                     # print('own2',efold)
#                     # efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
#                     # print('sci2',efold)
#                     # except integrate.IntegrationWarning:
#                     #     #print('s3')
#                         # efold=70
#                     # except:
#                         #print('s4')
#                         # efold=70
#                 if efold>efolds:
#                     # print(phi_star0,phi_end)
#                     phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                     phi_star=phi_star.root
#                 else:
#                     phi_star=1e5 #fake value just to fill array so that array sizes match
#                     phi_stars.append(phi_star)
#                     n_pre = 100
#                     r_pre = 100
#                     m_pre = 100
#                     n_arr.append(n_pre)
#                     r_arr.append(r_pre)
#                     m_arr.append(m_pre)
#                     continue # next phi_end
                

#             if np.abs(phi_star)<=phi_max and epsilon(phi_star)<1 and efold!=70:#eta(phi_star)<1 and 
#                 phi_stars.append(phi_star)
#                 n_pre = ns(phi_star) # get n
#                 r_pre = r(phi_star) # get r
#                 m_pre = e_scale(phi_star) # get m
#                 n_arr.append(n_pre)
#                 r_arr.append(r_pre)
#                 m_arr.append(m_pre)
#             else:
#                 if efold==70 or len(phi_ends)==1: # integration problems, discard function
#                     print('100')
#                     return [100,100,100]
#                     # return [100]

#                 else:
#                     phi_star=1e5 #fake value just to fill array so that array sizes match
#                     phi_stars.append(phi_star)
#                     n_pre = 100 
#                     r_pre = 100
#                     m_pre = 100
#                     n_arr.append(n_pre)
#                     r_arr.append(r_pre)
#                     m_arr.append(m_pre)
#         else: 
#             print(100)
#             return [100,100,100]
#             # return [100]#,100]
    
#     #print('all',phi_stars)
#     if all(t == 1e5 for t in phi_stars) or len(n_arr) == 0: # meaning we never found an actual phi_start
#         print('100')
#         return [100,100,100]
#         # return [100]#,100]
    
#     if len(n_arr) == 1: # this potential only has 1 inflationary trajectory
#         # n_out=np.round(n_arr[0],decimals=4)
#         # r_out=np.round(r_arr[0], decimals=3)
#         # m_out=np.round(m_arr[0], decimals=3)
#         n_out=n_arr[0]
#         r_out=r_arr[0]
#         m_out=m_arr[0]
        
#         # if n_out!=1.0 and r_out!=0.0:    
#         if plots:
#             plt.figure()
#             plt.plot(phi, V(phi))
#             ylimmax=10*V(phi_star)
#             plt.ylim(0,ylimmax)
#             plt.xlim(-phi_max,phi_max)
#             plt.xlabel('$\phi$')
#             plt.ylabel('V($\phi$)')
#             plt.title(str(plot_title))
#             plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='green')
        
#         if print_n:  
#             print('n:', n_out)
#             print('r:', r_out)
#             print('m:', m_out)
#         return [n_out,r_out,m_out]
#         # return n_out#,r_out]#,m_out]
#         # return m_out
        
#         # else: 
#         #     print(100)
#         #     return [100,100,100] 
        
#     "select best pair of (n,r)" 
#     # when a potential has more than one inflationary trajectory, output the one matching data better
#     # n_arr=np.round(n_arr, decimals=4)
#     # r_arr=np.round(r_arr, decimals=3)
#     # m_arr=np.round(m_arr, decimals=3)
#     n_arr=np.array(n_arr)
#     r_arr=np.array(r_arr)
#     m_arr=np.array(m_arr)
#     # observed values (planck 2018 paper):
#     n_obs=0.9649
#     sigma_n=0.0042 
#     r_obs=0.000
#     sigma_r=0.028
#     m_obs=0.027
#     sigma_m=0.0027
#     fit_n = - np.log(sigma_n) - ((n_arr-n_obs)**2)/(2*(sigma_n**2))
#     fit_r = - np.log(sigma_r) - ((r_arr-r_obs)**2)/(2*(sigma_r**2))
#     fit_m = - np.log(sigma_m) - ((m_arr-m_obs)**2)/(2*(sigma_m**2))
#     fit_i = fit_n + fit_r + fit_m
#     arg=np.argmax(fit_i) # get best fitness
#     n_out=n_arr[arg]
#     r_out=r_arr[arg]
#     m_out=m_arr[arg]

#     # if n_out!=1.0 and r_out!=0.0:
#     if plots:
#         plt.figure()
#         plt.plot(phi, V(phi))
#         ylimmax=10*V(phi_stars[arg])
#         plt.ylim(0,ylimmax)
#         plt.xlim(-phi_max,phi_max)
#         plt.xlabel('$\phi$')
#         plt.ylabel('V($\phi$)')
#         plt.title(str(plot_title))
#         plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='green')

#     if print_n:
#         print('n:', n_out)
#         print('r:', r_out)
#         print('m:', m_out)
#     return n_out, r_out, m_out  
#     # return n_out#, r_out
#     # return m_out


# plt.figure()
# plt.xlabel('a0')
# plt.ylabel('a1')
# # plt.title('nll of (exp(a0*(a1 - exp(x)))')
# # plt.xscale("log")
# # plt.yscale("log")

# num_points = 40
# x_values = np.linspace(0,0.8, num_points)
# y_values = np.linspace(-.4, .4, num_points)


# Z = np.zeros((num_points, num_points))
# for i in range(len(x_values)):
#     for j in range(len(y_values)):
#         a = [x_values[i],y_values[j]]
#         # print(a)
#         a0 = a[0]
#         a1 = a[1]
        
#         st='a0*(1+cos(a1*x))'
#         # st='exp(a0*(a1 - exp(x)))'
#         eq=sp.sympify(st, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
#         derivative_expr = sp.diff(eq, x)
#         V1 = sp.lambdify(x, derivative_expr, 'numpy')
#         derivative_expr2 = sp.diff(derivative_expr, x)
#         V2 = sp.lambdify(x, derivative_expr2, 'numpy')
#         Z[j, i] = negloglike_jax(V, *a)

# cplot = plt.pcolormesh(x_values, y_values, Z, cmap='jet') 
# plt.colorbar(cplot)
# plt.show()

# # " 1d fix a0 and plot a1"
# # plt.figure()
# # # plt.xlabel('a1')
# # # plt.ylabel('nll for n_s only')
# # # plt.title('fixed a0=1 for (a0 - exp(a1*x))**2')
# # low=-5
# # high=0
# # a1range = np.logspace(low,high,800)
# # plt.xlim(low,high)
# # yplot=[]
# # for _ in a1range:
# #     # a=[1e-8, _]
# #     # a0=1e-8
# #     # a1=_
# #     a = [_]
# #     a0 = _
# #     st='a0 + x**2'
# #     eq=sp.sympify(st, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "cos":cos, "sin":sin, "exp":exp, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
# #     derivative_expr = sp.diff(eq, x)
# #     V1 = sp.lambdify(x, derivative_expr, 'numpy')
# #     derivative_expr2 = sp.diff(derivative_expr, x)
# #     V2 = sp.lambdify(x, derivative_expr2, 'numpy')
    
# #     yplot_ = negloglike_jax(V, *a)
# #     yplot.append(yplot_)
# #     plt.plot(a0,yplot_,'r.')
# #     plt.show()


# print(Z[42,199])
# print(y_values[42])   
# find=np.min(Z)
# print(find)
# findi=np.where(Z==find)
# print(findi)

# yv = np.array([0.027])
# ye = np.array([0.0027])
# trypred = solve_inflation(equation, *a)
# if trypred==[1]
# yvector = yv
# nll = np.sum(0.5 * (yvector - yv) ** 2 / ye ** 2 + 0.5 * np.log(2 * np.array([np.pi]) ) + np.log(ye) )
# print('nll',nll)
#%%
# import cProfile
#x = sp.Symbol('x')
#import numba as nb
#exp = sp.Lambda(x, sp.exp(x, evaluate=True) )

#def G(x,*a): # use np.exp
    #return x**2
    #return x**2
    #return (1 - np.exp( - (.81)*x ) )**2
    #return x**(-2.0)
    #return 1/x
   # return np.exp(3/x)
    #return (1 - np.exp( a[0]*x) )**2
    #return 1 + np.cos(x/a[0])
    #return 0.42 + 1/x
    #return 8*x**3
    #return .356 - 1/x
    #return np.exp(-4*x)
    #return a[0]*x**2
    #return np.exp(np.exp(x**2))
    #return 1e-1 + x**(-3)
    #return 0.2 + np.exp(2*x)
    #return a[0]*np.exp(-x**2)
    #return (a[0]/x + x)**2

#=[]
#a0=a[0]
#st = 'exp(3/x)'
#st = '(1 - exp(-a0*x) )**2'
#st = 'x**2'

#eq=sp.sympify(st, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp, "pow": pow, "x": x, "a0": a0})#, "a1": a1, "a2": a2, "a3": a3})
#derivative_expr = sp.diff(eq, x)
#V1 = sp.lambdify(x, derivative_expr, 'numpy')
#tes=np.linspace(1,10,20)
#print(V1(tes))

#derivative_expr2 = sp.diff(derivative_expr, x)
#V2 = sp.lambdify(x, derivative_expr2, 'numpy')
#print(V2(2))
#code_block = """
#def solve_inflation(eq_numpy, *a, fstring, efolds=60, phi_max=40, print_n=True, plots=False, plot_title=''):

 #   phi = np.linspace(-phi_max, phi_max, 3000)
    
    #source_code = inspect.getsource(eq_numpy)
    #print(source_code)
    #print('a:',a)
    
   # def V(x): 
  #      return eq_numpy(x,*np.atleast_1d(a))
    
    # if len(a)==1:
    #     a0=a[0]
    # elif len(a)==2:
    #     a0=a[0]
    #     a1=a[1]
    # elif len(a)==3:
    #     a0=a[0]
    #     a1=a[1]
    #     a2=a[2]

    # eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
    # derivative_expr = sp.diff(eq, x)
    # V1 = sp.lambdify(x, derivative_expr, 'numpy')
    
    # derivative_expr2 = sp.diff(derivative_expr, x)
    # V2 = sp.lambdify(x, derivative_expr2, 'numpy')
    
    # try:
    #     cs = CubicSpline(phi, V(phi))
    #     def epsilon(phi): 
    #         return (1 / 2) * (cs(phi, 1) / (cs(phi) ) )**2
                     
    #     def eta(phi): 
    #         return cs(phi, 2) / cs(phi)
        
    #     def integrand(phi): #the integrand for the e-folds expression 
    #         return - cs(phi)/cs(phi,1)
        
    #     def e_scale(phi):
    #         return (cs(phi)/epsilon(phi))**(0.25)
        
#     def epsilon(phi):
#         return 0.5*( V1(phi)/V(phi) )**2
    
#     def eta(phi):
#         return V2(phi)/V(phi)
    
#     def integrand(phi):
#         return - V(phi)/V1(phi)
    
#     def e_scale(phi):
#         return (V(phi)/epsilon(phi) )**(0.25)
            
#     # except:
#     #     print('--CUBIC SPLINE ERROR--')
#     #     return [100,100,100]

     
#     def ns(phi): #spectral index 
#       return 1 - 6*epsilon(phi) + 2*eta(phi) 
    
#     def r(phi): # tensor-to-scalar ratio 
#         return 16 * epsilon(phi) 
    
#     def epsilon1(phi): #function to calculate the root of. we want phi for epsilon=1 
#         return epsilon(phi) - 1
                   
#     def integral(phi_star): 
#         try:
#             #return mpmath.quad(integrand, [phi_star, phi_end]) - efolds
#             return integrate.quad(integrand, phi_star, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0] - efolds
#         except integrate.IntegrationWarning:
#             return 0
#         except:
#             return 0
    
#     "FIND PHI_END:"
    
#     phi1 = np.linspace(-phi_max, phi_max, 3000)
#     f = epsilon1(phi1)
#     phi_ends=[]
#     for i in range(len(phi1)-1):
#         if f[i] * f[i+1] < 0: #there is a root in this interval
#             result = optimize.root_scalar(epsilon1, bracket=[phi1[i],phi1[i+1]]) #find root
#             root = result.root
#             phi_ends.append(root)
#     print('e',phi_ends)
#     if len(phi_ends)==0: # unfit for inflation since \epsilon never crosses 1
#         print('100')
#         return [100,100,100]
    

#     "FIND PHI_START:"
    
#     phi_stars=[]
#     n_arr=[]
#     r_arr=[]
#     m_arr=[]
#     #for phi_end in phi_ends:
#     for i in range(len(phi_ends)):
#         phi_end=phi_ends[i]
#         print('end',phi_end)
#         #print(phi_end)
#         phi_star0=phi_end # guess for phi_start
#         efold=0
#         step=1
#         if V(phi_end)>=0:  
#             if V(phi_end+1e-4)>V(phi_end): # check direction of integration
#                 while efold<efolds and phi_star0<phi_max:
#                     #print('efold',efold)
#                     #while np.abs(phi_star0)<phi_max:
#                     phi_star0 +=  step
#                     print(phi_star0)
#                     #if epsilon(phi_star0)<1:
#                     try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
#                     except integrate.IntegrationWarning:
#                         efold=70 # exit while loop
#                         print('s1')
#                     except:
#                         efold=70 # exit while loop
#                         print('s2')
#                 if efold>efolds:
#                     phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                     phi_star=phi_star.root
#                 else:
#                     phi_star=1e5 #fake value just to fill array so that array sizes match
#                     phi_stars.append(phi_star)
#                     n_pre = ns(phi_star) 
#                     r_pre = r(phi_star)
#                     m_pre = e_scale(phi_star)
#                     n_arr.append(n_pre)
#                     r_arr.append(r_pre)
#                     m_arr.append(m_pre)
#                     print('fail')
#                     continue # next phi_end
#             else: 
#                 while efold<efolds and phi_star0>-phi_max:
#                     phi_star0 -=  step
#                     print('i',phi_star0)
#                     #if epsilon(phi_star0)<1:
#                     try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
#                     except integrate.IntegrationWarning:
#                         print('s3')
#                         efold=70
#                     except:
#                         print('s4')
#                         efold=70
#                 if efold>efolds:
#                     phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                     phi_star=phi_star.root
#                 else:
#                     phi_star=1e5 #fake value just to fill array so that array sizes match
#                     phi_stars.append(phi_star)
#                     n_pre = 100
#                     r_pre = 100
#                     m_pre = 100
#                     n_arr.append(n_pre)
#                     r_arr.append(r_pre)
#                     m_arr.append(m_pre)
#                     continue # next phi_end
                

#             if np.abs(phi_star)<=phi_max and epsilon(phi_star)<1 and efold!=70:
#                 phi_stars.append(phi_star)
#                 n_pre = ns(phi_star) # get n
#                 r_pre = r(phi_star) # get r
#                 m_pre = e_scale(phi_star) # get m
#                 n_arr.append(n_pre)
#                 r_arr.append(r_pre)
#                 m_arr.append(m_pre)
#             else:
#                 if efold==70 or len(phi_ends)==1: # integration problems, discard function
#                     print('100')
#                     return [100,100,100]
#                 else:
#                     phi_star=1e5 #fake value just to fill array so that array sizes match
#                     phi_stars.append(phi_star)
#                     n_pre = ns(phi_star) 
#                     r_pre = r(phi_star)
#                     m_pre = e_scale(phi_star)
#                     n_arr.append(n_pre)
#                     r_arr.append(r_pre)
#                     m_arr.append(m_pre)
#         else: 
#             print(100)
#             return [100,100,100]
    
#     print('all',phi_stars)
#     if all(t == 1e5 for t in phi_stars) or len(n_arr) == 0: # meaning we never found an actual phi_start
#         print('1020')
#         return [100,100,100]
    
#     if len(n_arr) == 1: # this potential only has 1 inflationary trajectory
#         n_out=np.round(n_arr[0],decimals=4)
#         r_out=np.round(r_arr[0], decimals=3)
#         m_out=np.round(m_arr[0], decimals=3)
        
#         if n_out!=1.0 and r_out!=0.0:    
#             if plots:
#                 plt.figure()
#                 plt.plot(phi, V(phi))
#                 plt.ylim(0)
#                 plt.xlim(-phi_max,phi_max)
#                 plt.xlabel('$\phi$')
#                 plt.ylabel('V($\phi$)')
#                 plt.title(str(plot_title))
#                 plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='green')
            
#             if print_n:  
#                 print('n:', n_out)
#                 print('r:', r_out)
#                 print('m:', m_out)
#             return [n_out,r_out,m_out]
        
#         else: 
#             print(100)
#             return [100,100,100] 
        
#     "select best pair of (n,r)" 
#     # when a potential has more than one inflationary trajectory, output the one matching data better
#     n_arr=np.round(n_arr, decimals=4)
#     r_arr=np.round(r_arr, decimals=3)
#     m_arr=np.round(m_arr, decimals=3)
#     # observed values (planck 2018 paper):
#     n_obs=0.9649
#     sigma_n=0.0042 
#     r_obs=0.000
#     sigma_r=0.028
#     m_obs=0.027
#     sigma_m=0.0027
#     fit_n = - np.log(sigma_n) - ((n_arr-n_obs)**2)/(2*(sigma_n**2))
#     fit_r = - np.log(sigma_r) - ((r_arr-r_obs)**2)/(2*(sigma_r**2))
#     fit_m = - np.log(sigma_m) - ((m_arr-m_obs)**2)/(2*(sigma_m**2))
#     fit_i = fit_n + fit_r + fit_m
#     arg=np.argmax(fit_i) # get best fitness
#     n_out=n_arr[arg]
#     r_out=r_arr[arg]
#     m_out=m_arr[arg]

#     if n_out!=1.0 and r_out!=0.0:
#         if plots:
#             plt.figure()
#             plt.plot(phi, V(phi))
#             plt.ylim(0)
#             plt.xlim(-phi_max,phi_max)
#             plt.xlabel('$\phi$')
#             plt.ylabel('V($\phi$)')
#             plt.title(str(plot_title))
#             plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='green')
    
#         if print_n:
#             print('n:', n_out)
#             print('r:', r_out)
#             print('m:', m_out)
#         return n_out, r_out, m_out
    
#     else: 
#         print(100)
#         return [100,100,100] 

# #solvze_inflation(G,*a,fstring=st,plots=True)
# #"""



# def solve_inflation(eq_numpy, *a, efolds=60, phi_max=40, print_n=True, plots=False, plot_title=''):
        
#             phi = np.linspace(-phi_max, phi_max, 3000)

#             # source_code = inspect.getsource(eq_numpy)
#             # print(source_code)
#             # print(a)
            
#             def V(x): 
#                 return eq_numpy(x,*np.atleast_1d(a))
            
            
#             # Va = V(phi)
#             # Va1 = simps(phi,Va)
            
#             # g='(1 - exp( - (0.81)*x ) )**2'
#             # eq=sp.sympify(g,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})

#             # #derivative_expr = sp.diff(V(x), x)
#             # derivative_expr = sp.diff(eq, x)
#             # V1 = sp.lambdify(x, derivative_expr, 'numpy')
            
#             # derivative_expr2 = sp.diff(derivative_expr, x)
#             # V2 = sp.lambdify(x, derivative_expr2, 'numpy')
            
#             try:
                
#                 cs = CubicSpline(phi, V(phi))
                
#                 def epsilon(phi): 
#                     return 0.5 * (cs(phi, 1) / (cs(phi) ) )**2
               
#                 def eta(phi): 
#                     return cs(phi, 2) / cs(phi)
                
#                 def integrand(phi): #the integrand for the e-folds expression 
#                     return - cs(phi)/cs(phi,1)
                
#                 def e_scale(phi):
#                     return (cs(phi)/epsilon(phi))**(0.25)
#                       ###############
                     
#             # def epsilon(phi0):
#             #     return 0.5 * ( simps(phi0,Va) / V(phi0))**2
            
#             # #print(epsilon(phi))
            
#             # def eta(phi0):
#             #     return simps(phi0,Va1) / V(phi0)
            
#             # def integrand(phi0): #the integrand for the e-folds expression 
#             #     return - V(phi0)/simps(phi0,Va)
            
#             # def e_scale(phi0):
#             #     return (V(phi0)/epsilon(phi0))**(0.25)
            
#             # def epsilon(phi0):
#             #     return 0.5 * ( V1(phi0)/ V(phi0) )**2
            
#             # #print(epsilon(phi))
            
#             # def eta(phi0):
#             #     return V2(phi0) / V(phi0)
            
#             # def integrand(phi0): #the integrand for the e-folds expression 
#             #     return - V(phi0)/ V1(phi0)
            
#             # def e_scale(phi0):
#             #     return (V(phi0)/epsilon(phi0))**(0.25)
                
#                 # def epsilon(phi):
#                 #     return 0.5*( V1(phi)/V(phi) )*( V1(phi)/V(phi) )
#                 # print(epsilon(phi))
#                 # #print(epsilon(phi))
                
#                 # def eta(phi):
#                 #     return V2(phi)/V(phi)
                
#                 # def integrand(phi):
#                 #     return - V(phi)/V1(phi)
                
#                 # def e_scale(phi):
#                 #     return (V(phi)/epsilon(phi) )**(0.25)
                
#             except:
#                 print('--CUBIC SPLINE ERROR--')
#                 return [100,100,100]
        
             
#             def ns(phi0): #spectral index 
#               return 1 - 6*epsilon(phi0) + 2*eta(phi0) 
            
#             def r(phi0): # tensor-to-scalar ratio 
#                 return 16 * epsilon(phi0) 
            
#             def epsilon1(phi0): #function to calculate the root of. we want phi for epsilon=1 
#                 return epsilon(phi0) - 1
                           
#             def integral(phi_star): 
#                 try:
#                     #return mpmath.quad(integrand, [phi_star, phi_end]) - efolds
#                     return integrate.quad(integrand, phi_star, phi_end,limit=50,epsabs=1.49e-04,epsrel=1.49e-04)[0] - efolds
#                 except integrate.IntegrationWarning:
#                     print('IntegrationWarning')
#                     return 0
#                 except:
#                     return 0
            
#             "FIND PHI_END:"
            
#             #phi1 = np.linspace(-phi_max, phi_max, 3000)
#             f = epsilon1(phi)
#             phi_ends=[]
#             for i in range(len(phi)-1):
#                 if f[i] * f[i+1] < 0: #there is a root in this interval
#                     #print('i',i)
#                     result = optimize.root_scalar(epsilon1, bracket=[phi[i],phi[i+1]])#, rtol=1e-9, xtol=1e-9)#find root
#                     root = result.root
#                     phi_ends.append(root)
#             #print(phi_ends)
#             if len(phi_ends)==0: # unfit for inflation since \epsilon never crosses 1
#                 print('100')
#                 return [100,100,100]
            

#             "FIND PHI_START:"
            
#             phi_stars=[]
#             n_arr=[]
#             r_arr=[]
#             m_arr=[]
#             for phi_end in phi_ends:
#                 phi_star0=phi_end # guess for phi_start
#                 efold=0
#                 step=1
#                 if V(phi_end)>=0:  
#                     if V(phi_end+1e-4)>V(phi_end): # check direction of integration
#                         while efold<efolds:
#                             phi_star0 +=  step
#                             #if epsilon(phi_star0)<1:
#                             try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
#                             except integrate.IntegrationWarning:
#                                 print('IntegrationWarning')
#                                 efold=70 # exit while loop
#                             except:
#                                 efold=70 # exit while loop
#                         phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                         phi_star=phi_star.root
#                     else: 
#                         while efold<efolds:
#                             phi_star0 -=  step
#                             #if epsilon(phi_star0)<1:
#                             try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=50,epsabs=1.49e-04,epsrel=1.49e-04)[0]
#                             except integrate.IntegrationWarning:
#                                 print('IntegrationWarning')
#                                 efold=70
#                             except:
#                                 efold=70
#                         phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])#, rtol=1e-9, xtol=1e-9)
#                         phi_star=phi_star.root
                        
#                     if np.abs(phi_star)<=np.abs(phi_max) and epsilon(phi_star)<1 and efold!=70:
#                         phi_stars.append(phi_star)
#                         n_pre = ns(phi_star) # get n
#                         r_pre = r(phi_star) # get r
#                         m_pre = e_scale(phi_star)
#                         n_arr.append(n_pre)
#                         r_arr.append(r_pre)
#                         m_arr.append(m_pre)
#                     else: ## mistake in the code!!
#                         if efold==70 or len(phi_ends)==1: # integration problems, discard function
#                             print('100')
#                             return [100,100,100]
#                         else:
#                             phi_star=1e5 #fake value just to fill array so that array sizes match
#                             phi_stars.append(phi_star)
#                             n_pre = 100
#                             r_pre = 100
#                             m_pre = 100
#                             n_arr.append(n_pre)
#                             r_arr.append(r_pre)
#                             m_arr.append(m_pre)
#                 else: 
#                     print(100)
#                     return [100,100,100]
            
#             if all(phi_stars)==1e5 or len(n_arr) == 0: # meaning we never found an actual phi_start
#                 print('100')
#                 return [100,100,100]
            
#             if len(n_arr) == 1: # this potential only has 1 inflationary trajectory
#                 n_out=np.round(n_arr[0],decimals=4)
#                 r_out=np.round(r_arr[0], decimals=3)
#                 m_out=np.round(m_arr[0], decimals=3)
                
#                 # if n_out!=1.0 and r_out!=0.0:    
#                 if plots:
#                     plt.figure()
#                     plt.plot(phi, V(phi))
#                     plt.ylim(0)
#                     plt.xlim(-phi_max,phi_max)
#                     plt.xlabel('$\phi$')
#                     plt.ylabel('V($\phi$)')
#                     plt.title(str(plot_title))
#                     plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='green')
                
#                 if print_n:  
#                     print('n:', n_out)
#                     print('r:', r_out)
#                     print('m:', m_out)
#                 return [n_out,r_out,m_out]
                
#                 # else: 
#                 #     print(100)
#                 #     return [100,100,100] 
                
#             "select best pair of (n,r)" 
#             # when a potential has more than one inflationary trajectory, output the one matching data better
#             n_arr=np.round(n_arr, decimals=4)
#             r_arr=np.round(r_arr, decimals=3)
#             m_arr=np.round(m_arr, decimals=3)
#             # observed values (planck 2018 paper):
#             n_obs=0.9649
#             sigma_n=0.0042 
#             r_obs=0.000
#             sigma_r=0.028
#             m_obs=0.027
#             sigma_m=0.0027
#             fit_n = - np.log(sigma_n) - ((n_arr-n_obs)**2)/(2*(sigma_n**2))
#             fit_r = - np.log(sigma_r) - ((r_arr-r_obs)**2)/(2*(sigma_r**2))
#             fit_m = - np.log(sigma_m) - ((m_arr-m_obs)**2)/(2*(sigma_m**2))
#             fit_i = fit_n + fit_r + fit_m
#             arg=np.argmax(fit_i) # get best fitness
#             n_out=n_arr[arg]
#             r_out=r_arr[arg]
#             m_out=m_arr[arg]

#             # if n_out!=1.0 and r_out!=0.0:
#             if plots:
#                 plt.figure()
#                 plt.plot(phi, V(phi))
#                 plt.ylim(0)
#                 plt.xlim(-phi_max,phi_max)
#                 plt.xlabel('$\phi$')
#                 plt.ylabel('V($\phi$)')
#                 plt.title(str(plot_title))
#                 plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='green')
        
#             if print_n:
#                 print('n:', n_out)
#                 print('r:', r_out)
#                 print('m:', m_out)
#             return n_out, r_out, m_out
            
#             # else: 
#             #     print(100)
#             #     return [100,100,100] 

         
# solve_inflation(G, *a,print_n=False, plots=False)

# # #cProfile.run('solve_inflation(G,*a)', sort='cumulative')

# # """
# num_runs = 100  # You can adjust this number based on the accuracy you need

# # Measure the time taken for `num_runs` repetitions of the code block
# execution_times = timeit.repeat(stmt=code_block, globals=globals(), number=1, repeat=num_runs)

# # Calculate the average time
# average_time = sum(execution_times) / num_runs

# # Print the result
# print(f"Average time taken: {average_time:.6f} seconds")    
