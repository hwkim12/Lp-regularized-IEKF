import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kv
from IEKF_SL import IEKF_SL
from IEKF import IEKF

np.random.seed(123456789)

# Matrix Dimension and noise level
n = 30
p = 300
sigma = 0.1

# Construction of Matrix and Data
A = np.random.normal(0, 1, size = [n, p])

xt = np.zeros(p)
xt[20] = 0.8
xt[113] = 1.02
xt[161] = 0.53
xt[248] = 0.78

param_index = [20, 113, 161, 248]
y = A@xt + np.random.normal(0, sigma, n)

# Initialization and Covariance Matrices
x0 = np.zeros(p)
Q = np.diag(np.concatenate((sigma**2*np.ones(n), sigma*np.ones(p))))
R = np.diag(sigma**2*np.ones(n))
P = np.diag(sigma*np.ones(p))

# Forward Map 
def h_x(x):
    return np.array(A@x)

# Parameters for Ensemble based optimization
inner_niter = 30
num_ens = 300
alpha = 0.5
P_xx = P_x = P

# Parameters and initializations for VIAS
m_x = np.zeros(p)
C_x = np.diag(np.ones(p))
a_x = m_x**2 + np.diag(C_x)
a_x = np.array(a_x)

beta = 0.05

b = 2*beta
s = 0.005 - 0.5


# Regularization parameter for IEKF_SL and IEKF
r = 1/3
#r = 1

for k in range(50):
  # VIAS requires more outer iterations at least like 30 iterations
  # Hierarchical IEKF takes less than 10 iterations
  
  # VIAS Procedure 
  L = np.zeros(p)
  for i in range(p):
        L[i] = kv(s-1, np.sqrt(a_x[i]*b))/kv(s, np.sqrt(a_x[i]*b))*np.sqrt(b/a_x[i])
  D = np.diag(np.array(L))
  C_x = np.linalg.inv((np.transpose(A)@A)/(sigma**2) + D)
  m_x = C_x@np.transpose(A)@y/(sigma**2)
  a_x = m_x**2 + np.diag(C_x)
  a_x = np.array(a_x)
  
  # IEKF-SL/IEKF Procedure 
  if k < 7:
      # IEKF-SL iteration
      xx_mat = IEKF(x0 = np.zeros(p), alpha = 0.5, inner_niter = 30, h_x = h_x, y = y, P = P_xx, R = R, num_ens = num_ens)
      xx_mat = pd.DataFrame(xx_mat)
    
      xx_l = xx_mat.apply(np.quantile, axis = 0, q = 0.025, result_type ='expand')
      xx_l = np.array(xx_l)
      xx_u = xx_mat.apply(np.quantile, axis = 0, q = 0.975, result_type ='expand') 
      xx_u = np.array(xx_u)
      
      xx = xx_mat.apply(np.mean, axis = 0, result_type ='expand') 
      xx = np.array(xx)
      
      theta_xx = np.array((xx**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
      P_xx = np.diag(theta_xx)
      
      # IEKF iteration
      x_mat = IEKF_SL(x0 = np.zeros(p), alpha = 0.5, inner_niter = 30, h_x = h_x, y = y, P = P_x, R = R, num_ens = num_ens) 
      x_mat = pd.DataFrame(x_mat)
      
      x_l = x_mat.apply(np.quantile, axis = 0, q = 0.025, result_type ='expand')
      x_l = np.array(x_l)
      x_u = x_mat.apply(np.quantile, axis = 0, q = 0.975, result_type ='expand') 
      x_u = np.array(x_u)
      
      x = x_mat.apply(np.mean, axis = 0, result_type ='expand') 
      x = np.array(x)
  
      theta_x = np.array((x**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
      P_x = np.diag(theta_x)

  
  #print(k)
    

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

ax[0].plot(range(p), xt, 's', label = 'Truth', color = 'red', alpha = 1)
ax[0].errorbar(np.linspace(0,p-1,p), m_x, yerr = 1.96*np.sqrt(np.diag(C_x)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
ax[0].set_title('VIAS estimation', fontsize = 16)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].legend(loc = "upper right", fontsize = 12)
ax[0].set_ylim(-0.2,1.4)

ax[1].plot(range(p), xt, 's', label = 'Truth', color = 'red', alpha = 1)
ax[1].errorbar(np.linspace(0,p-1,p), x, yerr = (x-x_l, x_u-x), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='IEKF-SL')
ax[1].set_title('IEKF-SL estimation', fontsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].legend(fontsize=12, loc = "upper right")
ax[1].set_ylim(-0.2,1.4)

ax[2].plot(range(p), xt, 's', label = 'Truth', color = 'red', alpha = 1)
ax[2].errorbar(np.linspace(0,p-1,p), xx, yerr = (xx-xx_l, xx_u-xx), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='IEKF')
ax[2].set_title('IEKF estimation', fontsize = 16)
ax[2].tick_params(axis='both', which='major', labelsize=12)
ax[2].legend(fontsize=12, loc = "upper right")
ax[2].set_ylim(-0.2,1.4)

