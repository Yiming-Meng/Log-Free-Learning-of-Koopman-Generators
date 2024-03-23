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
import os

M_for_1d = 100
M = M_for_1d**2

m_monomial = 3
n_monomial = 4

xx = np.linspace(-1, 1, M_for_1d)
yy = np.linspace(-1, 1, M_for_1d)
x_mesh, y_mesh = np.meshgrid(xx, yy)

ROI = [[-1, 1], [-1, 1]]

span = 1
t_span = [0, span]
NN = 3
t_eval=np.linspace(0, span, NN)
lamda = 1e6




#Definie the trajectory solving and modification function for each initial condition.
def solve_ode(initial_setup, t_span, t_eval, NN, ROI, m, n):
    eta = lambda x: (x[0]**n) * (x[1]**m)
    def ode_function(t, var):
        x, y, I = var
        return [y, -x + (1 - x**2)*y, lamda**2 * np.exp(-lamda * t) * eta([x,y])]
    y0 = initial_setup
    solution = solve_ivp(ode_function, t_span, y0, t_eval=np.linspace(0, span, NN), method='Radau', dense_output=True, atol=1e-10, rtol=1e-9) 
    data0 = solution.y[0]
    data1 = solution.y[1]
    integral = solution.y[2]
    return [[data0[-1], data1[-1]], integral[-1]  - lamda * eta([y0[0], y0[1]])]

def ode_data_generator(initial_setups, t_span, t_eval, NN, ROI, m, n):
    print('Start solving ODE')
    tic1 = time.time()
    results = pool.starmap(solve_ode, [(setup, t_span, t_eval, NN, ROI, m, n) for setup in initial_setups])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    I = np.stack([results[i][1] for i in range(M)], axis=0)
    print('ODE solving & modification time = {} sec'.format(time.time()-tic1))
    total_time = time.time() - tic1
    print(f"Total time for data generation: {total_time:.2f} seconds")
    return I


if __name__ == "__main__": 
       
    sample = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    initial_setups = [[*sample[i], 0] for i in range(M)]
    
    print('Data Generating...')
    
    for m in range (m_monomial):
        for n in range (n_monomial):
            print('Generating Data for $x_1^{0}x_2{1}$ and $Lx_1^{0}x_2^{1}$'.format(n,m))
            filenameX = f'XX_{m}_{n}_samples_{M}_span_{span}.npy'
            filenameY = f'YY_{m}_{n}_samples_{M}_span_{span}.npy'
            
            eta2 = lambda x: np.power(x[:,1], m)*np.power(x[:,0], n)  
    
            #Use all available CPU cores
            num_processes = multiprocessing.cpu_count()  
            #print('cpu count =', num_processes)
            pool = multiprocessing.Pool(processes=num_processes)
    
    
            if os.path.exists(filenameX):
                # Path exists, ask the user if they want to overwrite
                overwrite = input("The files " + "'" + filenameX + "'" + "and" +  "'" + filenameY + "'" + " already exists. Do you want to overwrite it? (yes/no): ")
                if overwrite.lower() == "no":
                    Y = np.load(filenameY)
                    print("Pass.")
                else:
                    YY = ode_data_generator(initial_setups, t_span, t_eval, NN, ROI, m, n)
            else:
                YY = ode_data_generator(initial_setups, t_span, t_eval, NN, ROI, m, n)
    ##############################################################################
            #plotting
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            mesh = ax.plot_surface(x_mesh, y_mesh, Y.reshape(x_mesh.shape), cmap='GnBu', rstride=5, cstride=5, linewidth=0.1)
            # Add labels and a color bar
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            cbar = fig.colorbar(mesh, shrink=0.7)
            cbar.set_label('$Lx_1^{0}x_2^{1}$'.format(n,m))
            ax.grid(True, linestyle='--', linewidth=0.2, color='gray')
            plt.show()
            
            
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            mesh = ax.plot_surface(x_mesh, y_mesh, eta2(sample).reshape(x_mesh.shape), cmap='GnBu', rstride=5, cstride=5, linewidth=0.1)
            # Add labels and a color bar
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            cbar = fig.colorbar(mesh, shrink=0.7)
            cbar.set_label('$x_1^{0}x_2^{1}$'.format(n,m))
            ax.grid(True, linestyle='--', linewidth=0.2, color='gray')
            plt.show()
    
            
            XX = eta2(sample)
    
            np.save(filenameX, XX)
            np.save(filenameY, YY)
    ##############################################################################
    
    print('Learning Infinitesimal Generator...')
    X = np.zeros((M, m_monomial*n_monomial))
    Y = np.zeros((M, m_monomial*n_monomial))

    i = 0
    for m in range (m_monomial):
        for n in range (n_monomial):
            filenameX = f'XX_{m}_{n}_samples_{M}_span_{span}.npy'
            filenameY = f'YY_{m}_{n}_samples_{M}_span_{span}.npy'
            X[:,i] = np.load(filenameX)
            Y[:,i] = np.load(filenameY)
            i+=1
            
    X_TX = (X.T)@ X
    X_TY = (X.T) @ Y
        
    pinv = np.linalg.pinv(X_TX)
    L = pinv @ X_TY
    
    weight_for_f1 = L[:,1].reshape(m_monomial, n_monomial).T
    weight_for_f2 = L[:,4].reshape(m_monomial, n_monomial).T
    
    print('Matrix of Weights for f1: ', weight_for_f1)
    print('Matrix of Weights for f2: ', weight_for_f2)
    
    
    
    
    
    
    