#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  13 11:38:34 2023

@author: tomass
"""

import sympy as sp
import esr.generation.generator as generatorx
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import getdist
from scipy import integrate 
import warnings
warnings.filterwarnings("error", category=integrate.IntegrationWarning)
from getdist import plots, MCSamples
import numpy as np
import matplotlib.pyplot as plt


#%%
n1, r1 = np.loadtxt('plikHM_TTTEEE_lowl_lowE_BK15/base_r_plikHM_TTTEEE_lowl_lowE_BK15_1.txt', 
                    usecols=(7,8),unpack=True)
n2, r2 = np.loadtxt('plikHM_TTTEEE_lowl_lowE_BK15/base_r_plikHM_TTTEEE_lowl_lowE_BK15_2.txt', 
                    usecols=(7,8),unpack=True)
n3, r3 = np.loadtxt('plikHM_TTTEEE_lowl_lowE_BK15/base_r_plikHM_TTTEEE_lowl_lowE_BK15_3.txt', 
                    usecols=(7,8),unpack=True)
n4, r4 = np.loadtxt('plikHM_TTTEEE_lowl_lowE_BK15/base_r_plikHM_TTTEEE_lowl_lowE_BK15_4.txt', 
                    usecols=(7,8),unpack=True)

n0=np.concatenate((n1,n2,n3,n4))
r0=np.concatenate((r1,r2,r3,r4))

data=np.array([n0,r0])
data=np.transpose(data)

#%% PLOT

name2 = ["n","r"]

samples = MCSamples(samples=data, names=name2, 
        settings={'smooth_scale_2D':0.5, 'smooth_scale_1D':0.5},
        ranges={'r':[0,None]})#, 'n':[None,1]})


g = plots.get_single_plotter(width_inch=4, ratio=1)
g.plot_2d(samples, 'n', 'r', filled=True,lims=[0.95, 0.98, 1e-7, 5e-1])

plt.yscale('log')
#g.add_legend(['TT,TE,EE+lowE+lensing+BK15'], colored_text=True)

#%%

comp=6

print('Plotting n and r...')


outfile = np.loadtxt('.../final_'+str(comp)+'.dat' ,
                     dtype=str, delimiter=';')


@nb.jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64, nb.float64, nb.float64))
def trapz(xvals, yvals, start, finish, n):
    stp = (finish-start)/(n-1)
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*stp
    return area

eqns = outfile[:,1]
lik = outfile[:,4]
dcl = outfile[:,2]
# plt.figure()
# plt.title('Complexity '+str(comp))
# plt.yscale("log")
# plt.xlim(0.93, 1.00) # changed!
# plt.ylim(0, 0.5)
# plt.xlabel('Predicted spectral index $n_s$')
# plt.ylabel('Predicted tensor-to-scalar ratio $r$')
for i in range(10):
    if not np.isinf(float(dcl[i])) :    
        print(eqns[i])
        param = outfile[i,7:11]
        param = [float(elem) for elem in param]
        fstr=eqns[i]
        max_param = simplifier.get_max_param([fstr], verbose=False)
        fstr, fsym = simplifier.initial_sympify([fstr], max_param, parallel=False, verbose=False)
        fstr = fstr[0]
        fsym = fsym[fstr]

        nparam = simplifier.count_params([fstr], max_param)[0]

        if ("a0" in fstr) ==False:
            eq_numpy = sympy.lambdify(x, fsym, modules=["numpy"])
        elif nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, fsym, modules=["numpy"])
        else: 
            eq_numpy = sympy.lambdify([x, a0], fsym, modules=["numpy"])
            
        for z in range(len(fstr)):
            if fstr[z]=='/':
                fstr1 = fstr.replace('/','d')
                print('modified', fstr1)
        
        def solve_inflation(eq_numpy, *a, fstring='', rank=00, efolds=60, phi_max=400, print_n=True, plots=False, plot_title=''):
        
            scaling=1e-9
            phi = np.linspace(-phi_max, phi_max, 30000)
            
            def V(x): 
                return scaling*eq_numpy(x,*np.atleast_1d(a))
            
        
            if len(a)==0:
                eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x})
            if len(a)==1:
                a0=a[0]
                eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x, "a0": a0})
            elif len(a)==2:
                a0=a[0]
                a1=a[1]
                eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1})
            elif len(a)==3:
                a0=a[0]
                a1=a[1]
                a2=a[2]
                eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2})
            elif len(a)==4:
                a0=a[0]
                a1=a[1]
                a2=a[2]
                a3=a[3]
                eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
        
            derivative_expr = sp.diff(eq, x)
            V1 = sp.lambdify(x, derivative_expr, 'numpy')
            
            derivative_expr2 = sp.diff(derivative_expr, x)
            V2 = sp.lambdify(x, derivative_expr2, 'numpy')
                
            def epsilon(phi0):
                return 0.5*( (scaling*V1(phi0))/V(phi0) )**2
            
            def eta(phi0):
                return (V2(phi0)*scaling)/V(phi0)
            
            def integrand(phi0):
                return - V(phi0)/(V1(phi0)*scaling)
            
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
                step=.2*np.abs(phi_end)
                if V(phi_end)>=0:  
                    if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                        while efold<efolds and phi_star0<phi_max :
                            phi_star0 +=  step 
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
                        ylimmax=2*V(phi_star)
                        plt.ylim(0,ylimmax)
                    except:
                        pass
                    try:
                        xlimmax=4*np.abs(phi_star)
                        plt.xlim(-xlimmax,xlimmax)
                    except:
                        pass
                    plt.xlabel('$\phi$')
                    plt.ylabel('V($\phi$)')
                    plt.title(str(plot_title))
                    plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='brown')
                    try:
                        plt.savefig('compl_'+str(comp)+'_plots/'+fstr+'.png',dpi=250)
                    except:
                        plt.savefig('compl_'+str(comp)+'_plots/'+fstr1+'.png',dpi=250)

                    plt.close()
                
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
                try:
                    ylimmax=2*V(phi_star)
                    plt.ylim(0,ylimmax)
                except:
                    pass
                try:
                    xlimmax=4*np.abs(phi_star)
                    plt.xlim(-xlimmax,xlimmax)
                except:
                    pass
                plt.xlabel('$\phi$')
                plt.ylabel('V($\phi$)')
                plt.title(str(plot_title))
                plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='brown')
                try:
                    plt.savefig('compl_'+str(comp)+'_plots/'+fstr+'.png',dpi=250)
                except:
                    plt.savefig('compl_'+str(comp)+'_plots/'+fstr1+'.png',dpi=250)
                plt.close()
        
            if print_n:
                print('n:', n_out)
                print('r:', r_out)
                print('m:', m_out)
            return n_out, r_out, m_out  

        a=[elem for elem in param if elem != 0.0]
        print(a)
        lab=str(i+1)
        lab1=i+1
        n, r, m = solve_inflation(eq_numpy, *a, fstring=fstr, rank=lab1, plots=False)#, plot_title=fstr)  

        plt.plot(n,r,'r.')#,label=lab)
        # plt.legend()
        print('--')

plt.savefig('compl_'+str(comp)+'_plots/nr_plott_%i.png'%comp,dpi=300)
