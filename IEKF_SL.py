import numpy as np
import pandas as pd
import scipy.stats as st

def IEKF_SL(x0, alpha, inner_niter, h_x, y, P, R, num_ens):
    ## x0: Prior Mean
    ## alpha: Step Size
    ## inner_niter: total number of ensemble based optimization iteration
    ## h_u: Forward Map
    ## y: Data
    ## P: Prior Covariance
    ## R: Measurement Covariance
    ## num_ens: Number of Ensembles
    
    p = len(x0)
    n = len(y)
    
    U = st.multivariate_normal.rvs(mean = x0, cov = P, size = num_ens)
    U = pd.DataFrame(U)
    
    
    for i in range(inner_niter):
        m = pd.DataFrame(U.apply(np.sum, axis = 0, result_type ='expand')/num_ens) ## Ensemble Mean of p by 1 vector
        hu = pd.DataFrame(U.apply(h_x, axis = 1, result_type ='expand')) ## Forward map evaluation of all ensembles n, this is of size number of ensembles by n
        h = pd.DataFrame(hu.apply(np.sum, axis = 0)/num_ens)
        
        PYY_i = hu.transpose() - h.values ## n by num_ens matrix
        PUU_i = U.transpose() - m.values ## p by num_ens matrix
        
        Puu_i = np.zeros(shape = [p,p])
        Puy_i = np.zeros(shape = [p,n])
        
        for j in range(num_ens):
            Puu_i = Puu_i + np.outer(np.array(PUU_i.iloc[:,j]), np.array(PUU_i.iloc[:,j]))
            Puy_i = Puy_i + np.outer(np.array(PUU_i.iloc[:,j]), np.array(PYY_i.iloc[:,j]))
        
        Puu_i = Puu_i/num_ens
        Puy_i = Puy_i/num_ens
        
        H_i = np.transpose(Puy_i)@np.linalg.inv(Puu_i) ## statistical linearization
        #print(np.cov(U))
        #print(11111111)
        #print(Puu_i)

        #H_i = np.transpose(Puy_i)@np.diag(1./np.diag(Puu_i)) ## statistical linearization with Covariance Tapering
        #print(11111111)
        #print(1./np.diag(Puu_i))
        K_i = P@np.transpose(H_i)@np.linalg.inv(H_i@P@np.transpose(H_i)+R) ## kalman gain
        y_i = np.random.multivariate_normal(mean = y, cov = 2*R/alpha, size = num_ens)
        m_i = np.random.multivariate_normal(mean = x0, cov = 2*P/alpha, size = num_ens)
        
        for k in range(num_ens):
            U.iloc[k,:] = U.iloc[k,:] + alpha*((K_i@(y_i[k,:]-hu.iloc[k,:])) + (np.diag(np.ones(p)) - K_i@H_i)@(m_i[k,:] - U.iloc[k,:]))
        
        #print(i)
    return U

np.random.seed(12345)

### Matrix Dimension and noise level
n = 30
p = 300
sigma = 0.1

### Construction of Matrix and Data
A = np.random.rand(n, p)

xt = np.zeros(p)
xt[20] = 0.8
xt[113] = 1.02
xt[161] = 0.53
xt[248] = 0.78

param_index = [20, 113, 161, 248]

y = A@xt + np.random.normal(0, sigma, n)

### Initialization and Covariance Matrices
x0 = np.zeros(p)
Q = np.diag(np.concatenate((sigma**2*np.ones(n), sigma*np.ones(p))))
R = np.diag(sigma**2*np.ones(n))
P = np.diag(sigma*np.ones(p))

### Forward Map 
def h_x(x):
    return np.array(A@x)

### Parameters for Ensemble based optimization
inner_niter = 20
num_ens = 200
alpha = 0.5

#print(IEKF_SL(x0, 0.5, 20, h_x, y, P, R, num_ens))