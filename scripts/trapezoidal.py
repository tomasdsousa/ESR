#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:48:03 2023

@author: tomass
"""

import sympy as sp
import esr.generation.generator as generatorx
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from scipy import optimize 
from scipy import integrate 
import matplotlib.pyplot as plt
import warnings
import timeit
warnings.filterwarnings("error", category=integrate.IntegrationWarning)
import numpy as np 
import numba as nb
import cProfile
import time


#%%  integration function

@nb.jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64, nb.float64, nb.float64))
def trapz(xvals, yvals, start, finish, n):
    stp= (finish-start)/(n-1)
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*stp
    return area

#%% 

def G(x,*a): 
    return a[0]*x**2
    # return x**2
    # return a[2]*(a[0] - np.exp( (a[1])*x ) )**2
    # return a[0]*(1 - (x/a[1])**2)
    # return a[0]*(a[1] + x**2)**2
    # return a[0]*np.log(x) + 1
    # return (a[0] - np.exp(2*x))**9 
    # return a[0] + 1/x
    # return np.sqrt(x)
    # return 1 + np.cos(a[0]*x)
    # return x

st = 'a0*x**2'

a = [1.81e-2]
a0=a[0]
# a1=a[1]

eq=sp.sympify(st, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp,  "pow": pow, "cos": cos, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
derivative_expr = sp.diff(eq, x)
V1 = sp.lambdify(x, derivative_expr, 'numpy')
derivative_expr2 = sp.diff(derivative_expr, x)
V2 = sp.lambdify(x, derivative_expr2, 'numpy')

def solve_inflation(eq_numpy, *a, fstring='', efolds=60, phi_max=400, print_n=True, plots=False, plot_title=''):

    scaling = 1e-9
    
    phi = np.linspace(-phi_max, phi_max, 30000)
    
    def V(x): 
        return scaling*eq_numpy(x,*np.atleast_1d(a))
        
    def epsilon(phi0):
        return 0.5*( (scaling*V1(phi0)) / V(phi0) )**2
    
    def eta(phi0):
        return (scaling*V2(phi0))/V(phi0)
    
    def integrand(phi0):
        return - V(phi0)/(scaling*V1(phi0))
    
    def e_scale(phi0):
        return (V(phi0)/epsilon(phi0) )**(0.25)
     
    def ns(phi0): #spectral index 
      return 1 - 6*epsilon(phi0) + 2*eta(phi0) 
    
    def r(phi0): # tensor-to-scalar ratio 
        return 16 * epsilon(phi0) 
    
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
        print('100')
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
        step=0.01*np.abs(phi_end)
        # step=1
        if V(phi_end)>=0:  
            if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                while efold<efolds and phi_star0<phi_max:
                    phi_star0 +=  step
                    d2_start = V2(phi_star0)
                    xvals=np.linspace(phi_star0, phi_end, 100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals, yvals, phi_star0, phi_end ,100)
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
                    d2_start = V2(phi_star0)
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
                

            if np.abs(phi_star)<=phi_max and epsilon(phi_star)<1 and efold!=70:#eta(phi_star)<1 and 
                phi_stars.append(phi_star)
                n_pre = ns(phi_star) # get n
                r_pre = r(phi_star) # get r
                m_pre = e_scale(phi_star) # get m
                n_arr.append(n_pre)
                r_arr.append(r_pre)
                m_arr.append(m_pre)
            else:
                if efold==70 or len(phi_ends)==1: # integration problems, discard function
                    print('100')
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
            print(100)
            return [100,100,100]
    
    #print('all',phi_stars)
    if all(t == 1e5 for t in phi_stars) or len(n_arr) == 0: # meaning we never found an actual phi_start
        print('100')
        return [100,100,100]
    
    if len(n_arr) == 1: # this potential only has 1 inflationary trajectory
        n_out=n_arr[0]
        r_out=r_arr[0]
        m_out=m_arr[0]
        
        if plots:
            plt.figure()
            plt.plot(phi, V(phi))
            try:
                ylimmax=3*V(phi_star)
                plt.ylim(0,ylimmax)
            except:
                pass
            try:
                xlimmax=6*np.abs(phi_star)
                plt.xlim(-xlimmax,xlimmax)
            except:
                pass
            plt.xlabel('$\phi$')
            plt.ylabel('V($\phi$)')
            plt.title(str(plot_title))
            plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='brown')
        
        if n_out>0:
            if print_n:  
                print('n:', n_out)
                print('r:', r_out)
                print('m:', m_out)
            return [n_out,r_out,m_out]

        else:
            return [100,100,100]
        
        
    "select best pair of (n,r)" 
    # when a potential has more than one inflationary trajectory, output the one matching data better
    n_arr=np.array(n_arr)
    r_arr=np.array(r_arr)
    m_arr=np.array(m_arr)
    print('There are', len(n_arr), 'inflationary trajectories.')
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

    if n_out>0: # can happen with sinusoidal functions when integration skips over maxima
        
        phi_star = phi_stars[arg]
        if plots:
            plt.figure()
            plt.plot(phi, V(phi))
            try:
                ylimmax=3*V(phi_star)
                plt.ylim(0,ylimmax)
            except:
                pass
            try:
                xlimmax=6*np.abs(phi_star)
                plt.xlim(-xlimmax,xlimmax)
            except:
                pass
            plt.xlabel('$\phi$')
            plt.ylabel('V($\phi$)')
            plt.title(str(plot_title))
            plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='brown')
            
        if print_n:
            print('n:', n_out)
            print('r:', r_out)
            print('m:', m_out)
        return n_out, r_out, m_out  
    else:
        return [100,100,100]


q,qq,qqq=solve_inflation(G, *a, print_n=True, plots=True, plot_title=st)
yyy=np.array([q,qq,qqq])
yv=np.array([0.9649, 0.0, 0.027])
ye=np.array([0.0042 , 0.028, 0.0027])
nll = np.sum(0.5 * (yyy - yv) ** 2 / ye ** 2 + 0.5 * np.log(2 * np.pi) + np.log(ye))
print('------')
print('nll',nll)


