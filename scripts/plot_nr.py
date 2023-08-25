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

#like = np.loadtxt('plikHM_TTTEEE_lowl_lowE_BK15/base_r_plikHM_TTTEEE_lowl_lowE_BK15_1.txt', 
 #                   usecols=1,unpack=True)

n0=np.concatenate((n1,n2,n3,n4))
r0=np.concatenate((r1,r2,r3,r4))

data=np.array([n0,r0])
data=np.transpose(data)

#%% PLOT PLANCK CHAINS

name2 = ["n","r"]

samples = MCSamples(samples=data, names=name2, 
        settings={'smooth_scale_2D':0.5, 'smooth_scale_1D':0.5},
        ranges={'r':[0,None]})#, 'n':[None,1]})


g = plots.get_single_plotter(width_inch=4, ratio=1)
g.plot_2d(samples, 'n', 'r', filled=True,lims=[0.95, 0.98, 1e-7, 5e-1])
#
plt.yscale('log')
#g.add_legend(['TT,TE,EE+lowE+lensing+BK15'], colored_text=True)
# plt.rcParams['scale'] = 'log'

#%%
comp=6

print('Plotting n and r...')


outfile = np.loadtxt('Glamdring_results/results_AIFe_19Aug/final_'+str(comp)+'_katz2.dat' ,
                     dtype=str, delimiter=';')
#outfile = np.loadtxt('final_'+str(comp)+'.dat' ,
                    #dtype=str, delimiter=';')


@nb.jit(nb.float64(nb.float64[:], nb.float64[:],nb.float64, nb.float64, nb.float64))
def trapz(xvals, yvals, start, finish, n):
    stp = (finish-start)/(n-1)
    area=0
    for i in range(len(xvals)-1):
        area+=0.5*( yvals[i]+ yvals[i+1] )*stp
    return area

#param = outfile
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
    #if float(lik[i])<284380680 and not np.isinf(float(dcl[i])) :
    if not np.isinf(float(dcl[i])) :    
        print(eqns[i])
        # turn string into eq_numpy
        # solve inflation
        # plot n,r
        # PLOT (N,R) FROM TABLE FUNCTION
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
            
        # eq=sympy.sympify(fstr, locals={"inv": inv, "square": square, "cube": cube, "sqrt": sqrt, "log": log, "exp":exp, "pow": pow, "x": x, "a0": a0, "a1": a1, "a2": a2, "a3": a3})
        # derivative_expr = sympy.diff(eq, x)
        # V1 = sympy.lambdify(x, derivative_expr, 'numpy')
        # derivative_expr2 = sympy.diff(derivative_expr, x)
        # V2 = sympy.lambdify(x, derivative_expr2, 'numpy')
        
        for z in range(len(fstr)):
            if fstr[z]=='/':
                fstr1 = fstr.replace('/','d')
                print('modified', fstr1)
        
        def solve_inflation(eq_numpy, *a, fstring='', rank=00, efolds=60, phi_max=400, print_n=True, plots=False, plot_title=''):
        
            scaling=1e-9
            phi = np.linspace(-phi_max, phi_max, 30000)
            # step=80/2999
            #print('a2:',a)
            # source_code = inspect.getsource(eq_numpy)
            # print(source_code)
            # print('a:',a)
            
            def V(x): 
                return scaling*eq_numpy(x,*np.atleast_1d(a))
            
        
            if len(a)==0:
                #print('--------------')
                eq=sp.sympify(fstring,locals={"inv": inv, "square": square, "cube": cube, "x": x})
            if len(a)==1:
                a0=a[0]
                #print('--------------')
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
                # try:
                xvals1 = np.linspace(phi_star,phi_end,100)
                yvals1 = integrand(xvals1)
                return trapz(xvals1,yvals1,phi_star,phi_end, 100) - efolds
                    # return integrate.quad(integrand, phi_star, phi_end,limit=50, epsabs=1.49e-04,epsrel=1.49e-04)[0] - efolds
                # except integrate.IntegrationWarning:
                #     print(111)
                    # return 0
                # except:
                #     print(222)
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
                #print('end',phi_end)
                #print(phi_end)
                phi_star0=phi_end # guess for phi_start
                efold=0
                step=.2*np.abs(phi_end)
                # d2_end=V2(phi_end)
                # d2_start=V2(phi_star0)
                if V(phi_end)>=0:  
                    if V(phi_end+1e-4)>V(phi_end): # check direction of integration
                        while efold<efolds and phi_star0<phi_max :
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
                        while efold<efolds and phi_star0>-phi_max:
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
                        ylimmax=2*V(phi_star)
                        plt.ylim(0,ylimmax)
                    except:
                        pass
                    try:
                        xlimmax=4*np.abs(phi_star)
                        plt.xlim(-xlimmax,xlimmax)
                    except:
                        pass
                    # plt.xlim(-phi_max,phi_max)
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
                # plt.xlim(-phi_max,phi_max)
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
        # n, r, m = solve_inflation(eq_numpy, *a, fstring=fstr, rank=lab1, plots=True, plot_title=fstr+', DL rank '+lab+', a: '+str(a))  
        n, r, m = solve_inflation(eq_numpy, *a, fstring=fstr, rank=lab1, plots=True)#, plot_title=fstr)  

        # plt.plot(n,r,'r.')#,label=lab)
        # plt.legend()
        print('--')

#nn=np.linspace(0.93,1,40000)
#rr = (16/6)*0.968 - (16/6)*nn +.22
#plt.plot(nn,rr)

# fig.savefig('Desktop/compl_'+str(comp)+'_plots/nr_plott_%i.png'%comp)
# plt.savefig('compl_'+str(comp)+'_plots/nr_plott_%i.png'%comp,dpi=300)

#%%



# # Define the replacement function
# def replace_strings(s):
#     return np.char.replace(np.char.replace(s, "x", "x0"), "a0", "x1")

# # Apply the replacement function element-wise
# modified_text_array = np.vectorize(replace_strings)(outfile)

# def replace_strings_2(s):
#     return np.char.replace(s, "ex0p", "exp")

# modified_text_array = np.vectorize(replace_strings_2)(modified_text_array)

replacements = {
    "x": "x0",
    "a0": "x1",
    "a1": "x2",
    "a2": "x3",
    "a3": "x4"
}

# Apply replacements using np.char.replace
modified_text_array = outfile
for old_str, new_str in replacements.items():
    modified_text_array = np.char.replace(modified_text_array, old_str, new_str)

replacements = {"ex0p": "exp"}
for old_str, new_str in replacements.items():
    modified_text_array = np.char.replace(modified_text_array, old_str, new_str)
#%%  old loop 

        # modified version of "solve_inflation": added the rank argument for plot
        # def solve_inflation(eq_numpy, *a, rank=00, efolds=60, phi_max=40, print_n=True, plots=True, plot_title=''):
                
        #             phi = np.linspace(-phi_max, phi_max, 30000)
                    
        #             def V(x): 
        #                 return eq_numpy(x,*np.atleast_1d(a))
                    
                    
        #             try:
        #                 cs = CubicSpline(phi, V(phi))
        #             except:
        #                 print('--CS CUBIC SPLINE ERROR--')
        #                 return [100,100]
                
                   
        #             def epsilon(phi): 
        #               try: return (1 / 2) * (cs(phi, 1) / (cs(phi) ) )**2
        #               except:
        #                   print('--EPS CUBIC SPLINE ERROR--')
        #                   return [100,100]
        #             #if all(epsilon(phi))==0: 
        #                 # return [100,100]
                     
        #             def eta(phi): 
        #               try: return cs(phi, 2) / cs(phi)
        #               except:
        #                   print('--ETA CUBIC SPLINE ERROR--')
        #                   return [100,100]
        #             #if all(eta(phi))==0: 
        #               #   return [100,100]
                        
        #             def integrand(phi): #the integrand for the e-folds expression 
        #               try: return - cs(phi)/cs(phi,1)
        #               except:
        #                   print('--INT CUBIC SPLINE ERROR--')    
        #                   return [100,100]
        #             # if all(integrand(phi))==0: 
        #               #   return [100,100]
                     
        #             def ns(phi): #spectral index 
        #               return 1 - 6*epsilon(phi) + 2*eta(phi) 
                    
        #             def r(phi): # tensor-to-scalar ratio 
        #                 return 16 * epsilon(phi) 
                    
        #             def epsilon1(phi): #function to calculate the root of. we want phi for epsilon=1 
        #                 return epsilon(phi) - 1 
                                   
        #             def integral(phi_star): 
        #                 try:
        #                     #return mpmath.quad(integrand, [phi_star, phi_end]) - efolds
        #                     return integrate.quad(integrand, phi_star, phi_end,limit=10000)[0] - efolds
        #                 except integrate.IntegrationWarning:
        #                     return 0
        #                 except:
        #                     return 0
                    
        #             "FIND PHI_END:"
                    
        #             phi1 = np.linspace(-phi_max, phi_max, 3000)
        #             f = epsilon1(phi1)
        #             phi_ends=[]
        #             #count=1
        #             for i in range(len(phi1)-1):
        #                 if f[i] * f[i+1] < 0: #there is a root in this interval
        #                     result = optimize.root_scalar(epsilon1, bracket=[phi1[i],phi1[i+1]])
        #                     root = result.root
        #                     #print('phi_end #',count, ' :',root)
        #                     #count+=1
        #                     phi_ends.append(root)
                    
        #             #print('phi_ends',phi_ends,'.')
        #             if len(phi_ends)==0: 
        #                 return [100,100]
                    

        #             "FIND PHI_START:"
                    
        #             phi_stars=[]
        #             n_arr=[]
        #             r_arr=[]
        #             #count=1
        #             for i in range(len(phi_ends)):
        #                 phi_end=phi_ends[i]
        #                 phi_star0=phi_end # guess for phi_start
        #                 efold=0
        #                 step=epsilon(phi_star0) # so that the step is smaller closer to maxima/minima
        #                 if V(phi_end)>=0:  # already have this check before actually
                            
        #                     if V(phi_end+1e-4)>V(phi_end): # check direction of integration
        #                         while efold<efolds: # efolds is defined in arguments
        #                         #print('epsilon',epsilon(phi_star0))
        #                             phi_star0 +=  step
        #                             if epsilon(phi_star0)<1:
        #                                 #try: efold = mpmath.quad(integrand, [phi_star0, phi_end])
        #                                 try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=10000)[0]
        #                                 except integrate.IntegrationWarning:
        #                                     # print('--------------------------------------------')
        #                                     efold=70 # exit while loop
        #                                 except:
        #                                     efold=70 # exit while loop
        #                         phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
        #                         phi_star=phi_star.root
        #                         #phi_star = optimize.brentq(integral,phi_star0,phi_end)
        #                     else: 
        #                         while efold<efolds:
        #                         #print('epsilon',epsilon(phi_star0))
        #                             phi_star0 -=  step
        #                             if epsilon(phi_star0)<1:
        #                                 #try: efold = mpmath.quad(integrand, [phi_star0, phi_end])
        #                                 try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=10000)[0]
        #                                 except integrate.IntegrationWarning:
        #                                     # print('--------------------------------------------')
        #                                     efold=70
        #                                 except:
        #                                     efold=70
        #                         #phi_star = optimize.brentq(integral,phi_star0,phi_end)
        #                         phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
        #                         phi_star=phi_star.root
        #                     if np.abs(phi_star)<=np.abs(phi_max) and efold!=70:
        #                         phi_stars.append(phi_star)
        #                         #print('-------')
        #                         #print('phi_star #',count, ' :',phi_star)
        #                         #count+=1
        #                         #print(phi_star)
        #                         n_pre = ns(phi_star) 
        #                         r_pre = r(phi_star)
        #                         n_arr.append(n_pre)
        #                         r_arr.append(r_pre)
        #                     else:
        #                         if len(phi_ends)==1:                           
        #                             return [100,100]
        #                         else:
        #                             #count+=1
        #                             phi_star=1000000 #fake value just to fill array so that sizes match
        #                             #print(phi_star)
        #                             phi_stars.append(phi_star)
        #                             n_pre = ns(phi_star) 
        #                             r_pre = r(phi_star)
        #                             n_arr.append(n_pre)
        #                             r_arr.append(r_pre)
        #                 else: return [100,100]
                    
        #             if all(phi_stars)==1000000:
        #                 return [100,100]
        #             if len(n_arr) == 0:
        #                 return [100,100]
                    
        #             if len(n_arr) == 1:
        #                 n_out=np.round(n_arr[0],decimals=4)
        #                 r_out=np.round(r_arr[0], decimals=3)
                            
        #                 if plots==True:
        #                     plt.figure()
        #                     plt.plot(phi, V(phi))
        #                     plt.ylim(0)
        #                     plt.xlim(-phi_max,phi_max)
        #                     plt.xlabel('$\phi$')
        #                     plt.ylabel('V($\phi$)')
        #                     plt.title(str(plot_title))
        #                     plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='green')
        #                     # plt.savefig('nrplots/compl_'+str(comp)+'/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)                        
        #                     #plt.savefig('Desktop/compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
        #                     plt.savefig('compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
        #                     plt.close()
                            
        #                 if n_out!=1.0 and r_out!=0.0: 
        #                     if print_n==True:  
        #                         print('n:', n_out)
        #                         print('r:', r_out)
        #                     return [n_out,r_out]
        #                 else:                 
        #                     return [100,100]
        #             #print('phi_star',phi_stars,'.')
        #             "select best pair of (n,r)"
        #             n_arr=np.round(n_arr, decimals=4)
        #             r_arr=np.round(r_arr, decimals=3)
        #             lis=[]
        #             n_obs=0.9649
        #             sigma_n=0.0042 
        #             r_obs=0.000
        #             sigma_r=0.028 
        #             for i in range(len(n_arr)): 
        #                 fit_n = norm.logpdf(n_pre, loc=n_obs, scale=sigma_n)
        #                 fit_r = norm.logpdf(r_pre, loc=r_obs, scale=sigma_r)*np.heaviside(n_pre,1)
        #                 fit_i=fit_n+fit_r
        #                 lis.append(fit_i)
                    
        #             arg=np.argmax(lis) # get best fitness
        #             n_out=n_arr[arg]
        #             r_out=r_arr[arg]

        #             if plots==True:
        #                 plt.figure()
        #                 plt.plot(phi, V(phi))
        #                 plt.ylim(0)
        #                 plt.xlim(-phi_max,phi_max)
        #                 plt.xlabel('$\phi$')
        #                 plt.ylabel('V($\phi$)')
        #                 plt.title(str(plot_title))
        #                 plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='green')
        #                 #plt.savefig('nrplots/compl_'+str(comp)+'/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)                           
        #                 #plt.savefig('Desktop/compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
        #                 plt.savefig('compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
        #                 plt.close()
                     
        #             if type(n_out)==int or float and type(r_out)==int or float:
        #                 if n_out!=1.0 and r_out!=0.0:
        #                     if print_n==True: 
        #                         print('n:', n_out)
        #                         print('r:', r_out)
        #                     return n_out, r_out
        #                 else: return [100,100]
        #             else: return [100,100]

# for i in range(len(eqns)):
#     #if float(lik[i])<284380680 and not np.isinf(float(dcl[i])) :
#     if not np.isinf(float(dcl[i])) :    
#         print(eqns[i])
#         # turn string into eq_numpy
#         # solve inflation
#         # plot n,r
#         # PLOT (N,R) FROM TABLE FUNCTION
#         param = outfile[i,7:11]
#         param = [float(elem) for elem in param]
#         fstr=eqns[i]
#         max_param = simplifier.get_max_param([fstr], verbose=False)
#         fstr, fsym = simplifier.initial_sympify([fstr], max_param, parallel=False, verbose=False)
#         fstr = fstr[0]
#         fsym = fsym[fstr]

#         nparam = simplifier.count_params([fstr], max_param)[0]

#         if ("a0" in fstr) ==False:
#             eq_numpy = sympy.lambdify(x, fsym, modules=["numpy"])
#         elif nparam > 1:
#             all_a = ' '.join([f'a{i}' for i in range(nparam)])
#             all_a = list(sympy.symbols(all_a, real=True))
#             eq_numpy = sympy.lambdify([x] + all_a, fsym, modules=["numpy"])
#         else:
#             eq_numpy = sympy.lambdify([x, a0], fsym, modules=["numpy"])
            
#         # modified version of "solve_inflation": added the rank argument for plot
#         def solve_inflation(eq_numpy, *a, rank=00, efolds=60, phi_max=40, print_n=True, plots=True, plot_title=''):
                
#                     phi = np.linspace(-phi_max, phi_max, 30000)
                    
#                     def V(x): 
#                         return eq_numpy(x,*np.atleast_1d(a))
                    
                    
#                     try:
#                         cs = CubicSpline(phi, V(phi))
#                     except:
#                         print('--CS CUBIC SPLINE ERROR--')
#                         return [100,100]
                
                   
#                     def epsilon(phi): 
#                       try: return (1 / 2) * (cs(phi, 1) / (cs(phi) ) )**2
#                       except:
#                           print('--EPS CUBIC SPLINE ERROR--')
#                           return [100,100]
#                     #if all(epsilon(phi))==0: 
#                         # return [100,100]
                     
#                     def eta(phi): 
#                       try: return cs(phi, 2) / cs(phi)
#                       except:
#                           print('--ETA CUBIC SPLINE ERROR--')
#                           return [100,100]
#                     #if all(eta(phi))==0: 
#                       #   return [100,100]
                        
#                     def integrand(phi): #the integrand for the e-folds expression 
#                       try: return - cs(phi)/cs(phi,1)
#                       except:
#                           print('--INT CUBIC SPLINE ERROR--')    
#                           return [100,100]
#                     # if all(integrand(phi))==0: 
#                       #   return [100,100]
                     
#                     def ns(phi): #spectral index 
#                       return 1 - 6*epsilon(phi) + 2*eta(phi) 
                    
#                     def r(phi): # tensor-to-scalar ratio 
#                         return 16 * epsilon(phi) 
                    
#                     def epsilon1(phi): #function to calculate the root of. we want phi for epsilon=1 
#                         return epsilon(phi) - 1 
                                   
#                     def integral(phi_star): 
#                         try:
#                             #return mpmath.quad(integrand, [phi_star, phi_end]) - efolds
#                             return integrate.quad(integrand, phi_star, phi_end,limit=10000)[0] - efolds
#                         except integrate.IntegrationWarning:
#                             return 0
#                         except:
#                             return 0
                    
#                     "FIND PHI_END:"
                    
#                     phi1 = np.linspace(-phi_max, phi_max, 3000)
#                     f = epsilon1(phi1)
#                     phi_ends=[]
#                     #count=1
#                     for i in range(len(phi1)-1):
#                         if f[i] * f[i+1] < 0: #there is a root in this interval
#                             result = optimize.root_scalar(epsilon1, bracket=[phi1[i],phi1[i+1]])
#                             root = result.root
#                             #print('phi_end #',count, ' :',root)
#                             #count+=1
#                             phi_ends.append(root)
                    
#                     #print('phi_ends',phi_ends,'.')
#                     if len(phi_ends)==0: 
#                         return [100,100]
                    

#                     "FIND PHI_START:"
                    
#                     phi_stars=[]
#                     n_arr=[]
#                     r_arr=[]
#                     #count=1
#                     for i in range(len(phi_ends)):
#                         phi_end=phi_ends[i]
#                         phi_star0=phi_end # guess for phi_start
#                         efold=0
#                         step=epsilon(phi_star0) # so that the step is smaller closer to maxima/minima
#                         if V(phi_end)>=0:  # already have this check before actually
                            
#                             if V(phi_end+1e-4)>V(phi_end): # check direction of integration
#                                 while efold<efolds: # efolds is defined in arguments
#                                 #print('epsilon',epsilon(phi_star0))
#                                     phi_star0 +=  step
#                                     if epsilon(phi_star0)<1:
#                                         #try: efold = mpmath.quad(integrand, [phi_star0, phi_end])
#                                         try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=10000)[0]
#                                         except integrate.IntegrationWarning:
#                                             # print('--------------------------------------------')
#                                             efold=70 # exit while loop
#                                         except:
#                                             efold=70 # exit while loop
#                                 phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                                 phi_star=phi_star.root
#                                 #phi_star = optimize.brentq(integral,phi_star0,phi_end)
#                             else: 
#                                 while efold<efolds:
#                                 #print('epsilon',epsilon(phi_star0))
#                                     phi_star0 -=  step
#                                     if epsilon(phi_star0)<1:
#                                         #try: efold = mpmath.quad(integrand, [phi_star0, phi_end])
#                                         try: efold = integrate.quad(integrand, phi_star0, phi_end,limit=10000)[0]
#                                         except integrate.IntegrationWarning:
#                                             # print('--------------------------------------------')
#                                             efold=70
#                                         except:
#                                             efold=70
#                                 #phi_star = optimize.brentq(integral,phi_star0,phi_end)
#                                 phi_star=optimize.root_scalar(integral, bracket=[phi_star0,phi_end])
#                                 phi_star=phi_star.root
#                             if np.abs(phi_star)<=np.abs(phi_max) and efold!=70:
#                                 phi_stars.append(phi_star)
#                                 #print('-------')
#                                 #print('phi_star #',count, ' :',phi_star)
#                                 #count+=1
#                                 #print(phi_star)
#                                 n_pre = ns(phi_star) 
#                                 r_pre = r(phi_star)
#                                 n_arr.append(n_pre)
#                                 r_arr.append(r_pre)
#                             else:
#                                 if len(phi_ends)==1:                           
#                                     return [100,100]
#                                 else:
#                                     #count+=1
#                                     phi_star=1000000 #fake value just to fill array so that sizes match
#                                     #print(phi_star)
#                                     phi_stars.append(phi_star)
#                                     n_pre = ns(phi_star) 
#                                     r_pre = r(phi_star)
#                                     n_arr.append(n_pre)
#                                     r_arr.append(r_pre)
#                         else: return [100,100]
                    
#                     if all(phi_stars)==1000000:
#                         return [100,100]
#                     if len(n_arr) == 0:
#                         return [100,100]
                    
#                     if len(n_arr) == 1:
#                         n_out=np.round(n_arr[0],decimals=4)
#                         r_out=np.round(r_arr[0], decimals=3)
                            
#                         if plots==True:
#                             plt.figure()
#                             plt.plot(phi, V(phi))
#                             plt.ylim(0)
#                             plt.xlim(-phi_max,phi_max)
#                             plt.xlabel('$\phi$')
#                             plt.ylabel('V($\phi$)')
#                             plt.title(str(plot_title))
#                             plt.axvspan(phi_ends[0], phi_stars[0], alpha=0.5, color='green')
#                             # plt.savefig('nrplots/compl_'+str(comp)+'/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)                        
#                             #plt.savefig('Desktop/compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
#                             plt.savefig('compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
#                             plt.close()
                            
#                         if n_out!=1.0 and r_out!=0.0: 
#                             if print_n==True:  
#                                 print('n:', n_out)
#                                 print('r:', r_out)
#                             return [n_out,r_out]
#                         else:                 
#                             return [100,100]
#                     #print('phi_star',phi_stars,'.')
#                     "select best pair of (n,r)"
#                     n_arr=np.round(n_arr, decimals=4)
#                     r_arr=np.round(r_arr, decimals=3)
#                     lis=[]
#                     n_obs=0.9649
#                     sigma_n=0.0042 
#                     r_obs=0.000
#                     sigma_r=0.028 
#                     for i in range(len(n_arr)): 
#                         fit_n = norm.logpdf(n_pre, loc=n_obs, scale=sigma_n)
#                         fit_r = norm.logpdf(r_pre, loc=r_obs, scale=sigma_r)*np.heaviside(n_pre,1)
#                         fit_i=fit_n+fit_r
#                         lis.append(fit_i)
                    
#                     arg=np.argmax(lis) # get best fitness
#                     n_out=n_arr[arg]
#                     r_out=r_arr[arg]

#                     if plots==True:
#                         plt.figure()
#                         plt.plot(phi, V(phi))
#                         plt.ylim(0)
#                         plt.xlim(-phi_max,phi_max)
#                         plt.xlabel('$\phi$')
#                         plt.ylabel('V($\phi$)')
#                         plt.title(str(plot_title))
#                         plt.axvspan(phi_ends[arg], phi_stars[arg], alpha=0.5, color='green')
#                         #plt.savefig('nrplots/compl_'+str(comp)+'/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)                           
#                         #plt.savefig('Desktop/compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
#                         plt.savefig('compl_'+str(comp)+'_plots/solve_'+str(rank)+'compl_'+str(comp)+'.png',dpi=250)
#                         plt.close()
                     
#                     if type(n_out)==int or float and type(r_out)==int or float:
#                         if n_out!=1.0 and r_out!=0.0:
#                             if print_n==True: 
#                                 print('n:', n_out)
#                                 print('r:', r_out)
#                             return n_out, r_out
#                         else: return [100,100]
#                     else: return [100,100]

#         a=[elem for elem in param if elem != 0.0]
#         #print(a)
        
#         n,r=solve_inflation(eq_numpy, *a, rank=i, plots=True, plot_title=fstr)  
#         plt.plot(n,r,'r.')
#         print('--')