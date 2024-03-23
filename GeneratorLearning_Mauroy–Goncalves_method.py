#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:24:17 2024

@author: ym
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:30:03 2024

@author: ym
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#import DataCollection as dc
from numpy.polynomial import hermite
import time
import multiprocessing
from scipy.linalg import logm

M_for_1d = 100
M = M_for_1d**2

m_monomial = 3
n_monomial = 4

eta = lambda x: (x[0]**n) * (x[1]**m)

xx = np.linspace(-1, 1, M_for_1d)
yy = np.linspace(-1, 1, M_for_1d)
x_mesh, y_mesh = np.meshgrid(xx, yy)

ROI = [[-1, 1], [-1, 1]]

span = 0.5
t_span = [0, span]
NN = 1000
t_eval=np.linspace(0, span, NN)
lamda = 1e6

def ode_function(t, var):
    x, y = var
    return [y, -x + (1 - x**2)*y] 

#Definie the trajectory solving and modification function for each initial condition.
def solve_ode(initial_setup, t_span, t_eval, NN, ROI):
    y0 = initial_setup
    solution = solve_ivp(ode_function, t_span, y0, t_eval=np.linspace(0, span, NN), method='Radau', dense_output=True, atol=1e-10, rtol=1e-9) 
    data0 = solution.y[0]
    data1 = solution.y[1]
    return [[data0[-1], data1[-1]]]

def ode_data_generator(initial_setups, t_span, t_eval, NN, ROI):
    print('Start solving ODE')
    tic1 = time.time()
    #results = pool.map(solve_ode, initial_setups)
    results = pool.starmap(solve_ode, [(setup, t_span, t_eval, NN, ROI) for setup in initial_setups])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    I = np.stack([results[i][0] for i in range(M)], axis=0)
    print('ODE solving time = {} sec'.format(time.time()-tic1))

    total_time = time.time() - tic1
    print(f"Total time for data generation: {total_time:.2f} seconds")
    return I


if __name__ == "__main__": 
    
    sample = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    initial_setups = [[*sample[i]] for i in range(M)]
    
    
    #Use all available CPU cores
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    ##############################################################################
    
    I = ode_data_generator(initial_setups, t_span, t_eval, NN, ROI)
    
 
    ##############################################################################
    
    
    X = np.zeros((M, m_monomial*n_monomial))
    Y = np.zeros((M, m_monomial*n_monomial))
    
    i = 0
    for m in range (m_monomial):
        for n in range (n_monomial):
            eta2 = lambda x: np.power(x[:,1], m)*np.power(x[:,0], n)
            X[:,i] = eta2(sample)
            Y[:,i] = eta2(I)
            i+=1
            
    X_TX = (X.T)@ X
    X_TY = (X.T) @ Y
        
    pinv = np.linalg.pinv(X_TX)

    K = pinv @ X_TY
    eigenvalues, eigenvectors = np.linalg.eig(K)
    L = logm(K)
    
    weight_for_f1 = L[:,1].real.reshape(m_monomial, n_monomial).T
    weight_for_f2 = L[:,4].real.reshape(m_monomial, n_monomial).T
    
    print('Matrix of Weights for f1: ', weight_for_f1)
    print('Matrix of Weights for f2: ', weight_for_f2)
    
    weight_for_f1_imag = L[:,1].imag.reshape(m_monomial, n_monomial).T
    weight_for_f2_imag = L[:,4].imag.reshape(m_monomial, n_monomial).T
    
    print('Matrix of Weights (Imag Parts) for f1: ', weight_for_f1_imag)
    print('Matrix of Weights (Imag Parts) for f2: ', weight_for_f2_imag)
    
    
    