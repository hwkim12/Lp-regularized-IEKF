import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IEKF_SL import IEKF_SL
from IEKF import IEKF
from sklearn.linear_model import LassoLarsCV

np.random.seed(1234567)

# Matrix Dimension and noise level
n = 100
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

# Regularization parameter for IEKF_SL and IEKF
#r = 1/3
r = 1

# Parameters for IAS
beta = 0.05
b = 2*beta
s = 0.005 - 0.5

D_theta = np.identity(p)

IEKF_SL_NORM = []
IEKF_NORM = []

# Plotting norm difference to see the convergence
for k in range(20):
    # IAS procedure
    x_ias = np.linalg.inv(D_theta + (np.transpose(A)@A/sigma**2))@np.transpose(A)@y/sigma**2
    theta_x_ias = np.sqrt(x_ias**2/2)
    D_theta = np.diag(1/theta_x_ias)
    
    
    # IEKF procedure
    xx_mat = IEKF(x0 = np.zeros(p), alpha = 0.5, inner_niter = 30, h_x = h_x, y = y, P = P_xx, R = R, num_ens = num_ens)
    xx_mat = pd.DataFrame(xx_mat)
    xx = xx_mat.apply(np.mean, axis = 0, result_type ='expand') 
    xx = np.array(xx)
    IEKF_NORM.append(np.linalg.norm(xx-xt))
    
    theta_xx = np.array((xx**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
    P_xx = np.diag(theta_xx)
    
    # IEKF-SL procedure
    x_mat = IEKF_SL(x0 = np.zeros(p), alpha = 0.5, inner_niter = 30, h_x = h_x, y = y, P = P_x, R = R, num_ens = num_ens)
    x_mat = pd.DataFrame(x_mat)
    x = x_mat.apply(np.mean, axis = 0, result_type ='expand') 
    x = np.array(x)
    IEKF_SL_NORM.append(np.linalg.norm(x-xt))
    
    theta_x = np.array((x**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
    P_x = np.diag(theta_x)

    #print(k)

model_CV = LassoLarsCV(cv=10, normalize=False).fit(A, y)


plt.figure()
plt.plot(range(20), IEKF_NORM, label = "L05-IEKF", color = 'red')
plt.plot(range(20), IEKF_SL_NORM, label = "L05-IEKF-SL", color = 'blue')
plt.title('L2-Norm of difference', fontsize = 16)
plt.xlabel('Number of outer iterations')
plt.legend(loc="upper right", fontsize = 10)

plt.figure()
plt.plot(range(p), xt, 'o', label = 'Truth', color = 'red', alpha = 1)
plt.plot(range(p), x, 'p', label = 'Lp-IEKF-SL', color = 'blue', alpha = 1)
plt.plot(range(p), xx, 'D', label = 'Lp-IEKF', color = 'green', alpha = 1)
plt.plot(range(p), x_ias, 'X', label = 'IAS', color = 'khaki', alpha = 1)
plt.plot(range(p), model_CV.coef_, '^', label = "LASSO(CV)", color = 'purple', alpha = 1)
plt.title('Parameter estimation', fontsize = 16)
plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize = 10)
    

# Simulations below are for investigating how the parameter r controls the level of sparsity
rseq = np.linspace(0.1, 2, 20)

IEKF_SL_PATH = []
IEKF_PATH = []

NITER = 30 # inner iteration

for r in rseq:
    PP1 = np.diag(sigma*np.ones(p))
    PP2 = np.diag(sigma*np.ones(p))
    for k in range(NITER):
        XX1_mat = IEKF_SL(x0 = np.zeros(p), alpha = 0.5, inner_niter = 30, h_x = h_x, y = y, P = PP1, R = R, num_ens = num_ens)
        XX1_mat = pd.DataFrame(XX1_mat)
        XX1 = XX1_mat.apply(np.mean, axis = 0, result_type ='expand') 
        XX1 = np.array(XX1)
        
        THETA1 = np.array((XX1**2)**(1/(r+1))/(2*r)**(1/(r+1)))
        PP1 = np.diag(THETA1)
        
        XX2_mat = IEKF(x0 = np.zeros(p), alpha = 0.5, inner_niter = 30, h_x = h_x, y = y, P = PP2, R = R, num_ens = num_ens)
        XX2_mat = pd.DataFrame(XX2_mat)
        XX2 = XX2_mat.apply(np.mean, axis = 0, result_type ='expand') 
        XX2 = np.array(XX2)
        
        THETA2 = np.array((XX2**2)**(1/(r+1))/(2*r)**(1/(r+1)))
        PP2 = np.diag(THETA2)
        
        if k == (NITER-1):
            IEKF_SL_PATH.append(np.linalg.norm(np.delete(XX1, param_index)))
            IEKF_PATH.append(np.linalg.norm(np.delete(XX2, param_index))) 
    print(r)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))

ax[0].plot(range(p), xt, 's', label = 'Truth', color = 'red', alpha = 1)
ax[0].plot(range(p), x, 's', label = 'IEKF-SL', color = 'blue', alpha = 1)
ax[0].plot(range(p), xx, 's', label = 'IEKF', color = 'green', alpha = 1)
ax[0].plot(range(p), x_ias, 's', label = 'IAS', color = 'purple', alpha = 1)
ax[0].plot(range(p), model_CV.coef_, 's', label = "LASSO(CV)", color = "khaki", alpha = 1)
ax[0].set_title('Parameter estimation', fontsize = 16)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].legend(bbox_to_anchor=(1.02, 1.0), fontsize = 12)
ax[0].set_ylim(-0.2,1.4)

ax[1].plot(rseq, IEKF_PATH, label = 'IEKF', color = 'red', alpha = 1)
ax[1].plot(rseq, IEKF_SL_PATH, label = 'IEKF-SL', color = 'blue', alpha = 1)
ax[1].set_title('Zero-component estimation', fontsize = 16)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
#ax[1].set_ylim(-0.2,1.4)

plt.figure()
plt.plot(rseq, IEKF_PATH, label = 'IEKF', color = 'red', alpha = 1)
plt.plot(rseq, IEKF_SL_PATH, label = 'IEKF-SL', color = 'blue', alpha = 1)
plt.title('Zero-component estimation', fontsize = 16)
plt.xlabel('r')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc="upper left", fontsize=10)
