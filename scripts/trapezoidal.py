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



#%%
def V(x): 
    #return (1 - np.exp(-0.81*x))**2
    #return np.cos(x)
    #return x**8
    #return np.exp(x**3)
    return 1/x

xa=np.linspace(-4,4,100)
#step=28/99
ya=V(xa)

# code_block="""
res = np.trapz(ya, x=xa)
# """
print(res)

num_runs = 100  
execution_times = timeit.repeat(stmt=code_block, globals=globals(), number=1, repeat=num_runs)
average_time = sum(execution_times) / num_runs
print(f"Average time taken np.trapz: {average_time:.6f} seconds")    


# code_block="""
res1 = integrate.quad(V,-4,4)[0]
# """
print(res1)


num_runs = 100
execution_times = timeit.repeat(stmt=code_block, globals=globals(), number=1, repeat=num_runs)
average_time = sum(execution_times) / num_runs
print(f"Average time taken scipy: {average_time:.6f} seconds")    


#%%  write my own code 

@nb.jit(nb.float64(nb.float64[:], nb.float64[:], nb.float64))
def trap1(yvals, xvals, step):
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*step
    return area

@nb.jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64, nb.float64, nb.float64))
def trapz(xvals, yvals, start, finish, n):
    stp= (finish-start)/(n-1)
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*stp
    return area



# def integ(phi0):
#     return - V(phi0)/V1(phi0)



# xa=np.linspace(-0.6865890479690392, 0.31341095 ,1000)
# yval=V(xa)
# are = trapz(xa,yval,-4, 4,100)
# print(are)
# """
# print(area)

# num_runs = 10000
# execution_times = timeit.repeat(stmt=code_block, globals=globals(), number=1, repeat=num_runs)
# average_time = sum(execution_times) / num_runs
# print(f"Average time taken loop: {average_time:.6f} seconds")    

#%% put it into solve

# def G(x,*a): # use np.exp
    # return x**a[0]
    # return x**2
    # return a[2]*(a[0] - np.exp( (a[1])*x ) )**2
    # return a[0]*(1 - (x/a[1])**2)
    # return a[0]*(a[1] + x**2)**2
    # return a[0]*np.log(x) + 1
    # return (a[0] - np.exp(2*x))**9 # rank1 comp7
    # return a[0] + 1/x
    # return np.sqrt(x)
    # return 1 + np.cos(a[0]*x)
    # return x

a = [4.36e+00]
# a = [] #1e-9, 8500 give -12.12
# a=[ 1.084e-10 , 10.976]
# a=[-8.32804974e-305 , 2.82461325e-155]

a0=a[0]
# a1=a[1]


st = 'exp(a0/x)'
# st = 'a0 + 1/x'
eq=sp.sympify(st, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp,  "pow": pow, "cos": cos, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
# poten = sp.lambdify(x, eq, 'numpy')
derivative_expr = sp.diff(eq, x)
V1 = sp.lambdify(x, derivative_expr, 'numpy')
derivative_expr2 = sp.diff(derivative_expr, x)
V2 = sp.lambdify(x, derivative_expr2, 'numpy')

# print(V1(3))

def solve_inflation(eq_numpy, *a, fstring='', efolds=60, phi_max=400, print_n=True, plots=False, plot_title=''):

    scaling = 1e-9
    
    phi = np.linspace(-phi_max, phi_max, 30000)
    # step=80/2999
    #print('a2:',a)
    # source_code = inspect.getsource(eq_numpy)
    # print(source_code)
    # print('a:',a)
    
    def V(x): 
        return scaling*eq_numpy(x,*np.atleast_1d(a))
    

    # if len(a)==0:
    #     #print('--------------')
    #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x})
    # if len(a)==1:
    #     a0=a[0]
    #     #print('--------------')
    #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x, "a0": a0})
    # elif len(a)==2:
    #     a0=a[0]
    #     a1=a[1]
    #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1})
    # elif len(a)==3:
    #     a0=a[0]
    #     a1=a[1]
    #     a2=a[2]
    #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2})
    # elif len(a)==4:
    #     a0=a[0]
    #     a1=a[1]
    #     a2=a[2]
    #     a3=a[3]
    #     eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})

    # derivative_expr = sp.diff(eq, x)
    # V1 = sp.lambdify(x, derivative_expr, 'numpy')
    
    # derivative_expr2 = sp.diff(derivative_expr, x)
    # V2 = sp.lambdify(x, derivative_expr2, 'numpy')
        
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
        # try:
        xvals1 = np.linspace(phi_star,phi_end,100)
        yvals1 = integrand(xvals1)
        return trapz(xvals1,yvals1,phi_star,phi_end, 100) - efolds
            # return integrate.quad(integrand, phi_star, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0] - efolds
        #except integrate.IntegrationWarning:
        #     print(111)
            # return 0
        # except:
            # print(222)
            # return 0
    
    "FIND PHI_END:"
    
    phi1 = np.linspace(-phi_max, phi_max, 30000)
    f = epsilon1(phi1)
    phi_ends=[]
    for i in range(len(phi1)-1):
        if f[i] * f[i+1] < 0: #there is a root in this interval
            result = optimize.root_scalar(epsilon1, bracket=[phi1[i],phi1[i+1]]) #find root
            root = result.root
            phi_ends.append(root)
    # print(phi_ends)
    # time.sleep(2)
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
        # print('end',phi_end)
        # time.sleep(3)
        #print(phi_end)
        phi_star0=phi_end # guess for phi_start
        efold=0
        step=0.01*np.abs(phi_end)
        # step=1
        d2_end=V2(phi_end)
        d2_start=V2(phi_star0)
        if V(phi_end)>=0:  
            if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                while efold<efolds and phi_star0<phi_max:# and d2_end*d2_start>0:# and eta(phi_star0)<1:
                    # print(phi_end)
                    # print('eta',eta(phi_star0))
                    phi_star0 +=  step
                    d2_start = V2(phi_star0)
                    # print('d2',d2_start)
                    # try: 
                    xvals=np.linspace(phi_star0, phi_end, 100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals, yvals, phi_star0, phi_end ,100)
                    # print('own',efold)
                    # efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
                    # print('sci',efold)
                    # except integrate.IntegrationWarning:
                        # efold=70 # exit while loop
                    # except:
                        #print(4)
                        # efold=70 # exit while loop
                if efold>efolds:
                    phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
                    phi_star=phi_star.root
                    # print(phi_end, phi_star)
                    # time.sleep(3)
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
                while efold<efolds and phi_star0>-phi_max:# and d2_end*d2_start>0:# and eta(phi_star0)<1:
                    phi_star0 -=  step
                    d2_start = V2(phi_star0)
                    # print('d2',d2_start)
                    # try: 
                    xvals=np.linspace(phi_star0,phi_end,100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals,yvals,phi_star0,phi_end,100)
                    # print('own2',efold)
                    # efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
                    # print('sci2',efold)
                    # except integrate.IntegrationWarning:
                    #     #print('s3')
                        # efold=70
                    # except:
                        #print('s4')
                        # efold=70
                if efold>efolds:
                    # print(phi_star0,phi_end)
                    phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
                    phi_star=phi_star.root
                    # print(phi_end, phi_star)
                    # time.sleep(2)
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
        # n_out=np.round(n_arr[0],decimals=4)
        # r_out=np.round(r_arr[0], decimals=3)
        # m_out=np.round(m_arr[0], decimals=3)
        n_out=n_arr[0]
        r_out=r_arr[0]
        m_out=m_arr[0]
        
        # if n_out!=1.0 and r_out!=0.0:    
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
        
        # else: 
        #     print(100)
        #     return [100,100,100] 
        
    "select best pair of (n,r)" 
    # when a potential has more than one inflationary trajectory, output the one matching data better
    # n_arr=np.round(n_arr, decimals=4)
    # r_arr=np.round(r_arr, decimals=3)
    # m_arr=np.round(m_arr, decimals=3)
    n_arr=np.array(n_arr)
    r_arr=np.array(r_arr)
    m_arr=np.array(m_arr)
    # print('There are', len(n_arr), 'inflationary trajectories.')
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

    # if n_out!=1.0 and r_out!=0.0:

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


# code_block="""
q,qq,qqq=solve_inflation(poten, *a, print_n=True, plots=True, plot_title=st)
# """
yyy=np.array([q,qq,qqq])
yv=np.array([0.9649, 0.0, 0.027])
ye=np.array([0.0042 , 0.028, 0.0027])
nll = np.sum(0.5 * (yyy - yv) ** 2 / ye ** 2 + 0.5 * np.log(2 * np.pi) + np.log(ye))
print('------')
print('nll',nll)


# cProfile.run('solve_inflation(G,*a,print_n=False, plots=False, plot_title=st)', sort='cumulative')


# num_runs = 1000
# execution_times = timeit.repeat(stmt=code_block, globals=globals(), number=1, repeat=num_runs)
# average_time = sum(execution_times) / num_runs
# print(f"Average time taken solve: {average_time:.6f} seconds")  



#%%   

def eq_numpy(phi,*a):
    return np.sqrt(phi) + 1/a[0]

fstr = 'sqrt(x) + 1/a0'
a = [0]

eq = sympy.sympify(fstr,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0})
derivative_expr = sympy.diff(eq, x)
V_1 = sympy.lambdify([x, a0], derivative_expr, 'numpy')
derivative_expr2 = sympy.diff(derivative_expr, x)
V_2 = sympy.lambdify([x, a0], derivative_expr2, 'numpy')


def solve_inflation(eq_numpy, *a, fstring='', V11, V22, efolds=60, phi_max=400, print_n=True, plots=False, plot_title=''):

    scaling = 1e-9
    
    phi = np.linspace(-phi_max, phi_max, 30000)
    # step=80/2999
    #print('a2:',a)
    # source_code = inspect.getsource(eq_numpy)
    # print(source_code)
    # print('a:',a)
    
    def V(x): return scaling*eq_numpy(x,*np.atleast_1d(a))  
    
    if np.isinf(V(1)) and np.isinf(V(2)):
        print('coco')
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
    # print(phi_ends)
    # time.sleep(2)
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
        # print('end',phi_end)
        # time.sleep(3)
        #print(phi_end)
        phi_star0=phi_end # guess for phi_start
        efold=0
        step=0.01*np.abs(phi_end)
        # step=1
        d2_end=V2(phi_end)
        d2_start=V2(phi_star0)
        if V(phi_end)>=0:  
            if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                while efold<efolds and phi_star0<phi_max:# and d2_end*d2_start>0:# and eta(phi_star0)<1:
                    # print(phi_end)
                    # print('eta',eta(phi_star0))
                    phi_star0 +=  step
                    d2_start = V2(phi_star0)
                    # print('d2',d2_start)
                    # try: 
                    xvals=np.linspace(phi_star0, phi_end, 100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals, yvals, phi_star0, phi_end ,100)
                    # print('own',efold)
                    # efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
                    # print('sci',efold)
                    # except integrate.IntegrationWarning:
                        # efold=70 # exit while loop
                    # except:
                        #print(4)
                        # efold=70 # exit while loop
                if efold>efolds:
                    phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
                    phi_star=phi_star.root
                    # print(phi_end, phi_star)
                    # time.sleep(3)
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
                while efold<efolds and phi_star0>-phi_max:# and d2_end*d2_start>0:# and eta(phi_star0)<1:
                    phi_star0 -=  step
                    d2_start = V2(phi_star0)
                    # print('d2',d2_start)
                    # try: 
                    xvals=np.linspace(phi_star0,phi_end,100)
                    yvals=integrand(xvals)
                    efold = trapz(xvals,yvals,phi_star0,phi_end,100)
                    # print('own2',efold)
                    # efold = integrate.quad(integrand, phi_star0, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0]
                    # print('sci2',efold)
                    # except integrate.IntegrationWarning:
                    #     #print('s3')
                        # efold=70
                    # except:
                        #print('s4')
                        # efold=70
                if efold>efolds:
                    # print(phi_star0,phi_end)
                    phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
                    phi_star=phi_star.root
                    # print(phi_end, phi_star)
                    # time.sleep(2)
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
        # n_out=np.round(n_arr[0],decimals=4)
        # r_out=np.round(r_arr[0], decimals=3)
        # m_out=np.round(m_arr[0], decimals=3)
        n_out=n_arr[0]
        r_out=r_arr[0]
        m_out=m_arr[0]
        
        # if n_out!=1.0 and r_out!=0.0:    
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
        
        # else: 
        #     print(100)
        #     return [100,100,100] 
        
    "select best pair of (n,r)" 
    # when a potential has more than one inflationary trajectory, output the one matching data better
    # n_arr=np.round(n_arr, decimals=4)
    # r_arr=np.round(r_arr, decimals=3)
    # m_arr=np.round(m_arr, decimals=3)
    n_arr=np.array(n_arr)
    r_arr=np.array(r_arr)
    m_arr=np.array(m_arr)
    # print('There are', len(n_arr), 'inflationary trajectories.')
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

    # if n_out!=1.0 and r_out!=0.0:

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

ypred = solve_inflation(eq_numpy, *a, V11=V_1,V22=V_2,fstring=fstr, plots=True)
