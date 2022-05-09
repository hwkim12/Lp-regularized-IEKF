import numpy as np
import pandas as pd
import scipy.stats as st

def TEKI(x0, alpha, inner_niter, h_x, y, P, R, num_ens):
    ## x0: Prior Mean
    ## alpha: Step Size
    ## inner_niter: total number of ensemble based optimization iteration
    ## h_u: Forward Map
    ## y: Data
    ## P: Prior Covariance
    ## R: Measurement Covariance
    ## num_ens: Number of Ensembles
    
    U = st.multivariate_normal.rvs(mean = x0, cov = P, size = num_ens) ### num_ens by size of paramter 
    U = pd.DataFrame(U)
    
    q_diag = np.concatenate((np.diag(R), np.diag(P)))
    Q = np.diag(q_diag)
    
    
    for i in range(inner_niter):
        m = pd.DataFrame(U.apply(np.sum, axis = 0, result_type ='expand')/num_ens) ## Ensemble Mean of p by 1 vector
        hu = pd.DataFrame(U.apply(h_x, axis = 1, result_type ='expand')) ## Forward map evaluation of all ensembles, this is of num_ens by n
        h = pd.DataFrame(hu.apply(np.sum, axis = 0)/num_ens)
        
        GU = pd.concat([hu, U], axis = 1)
        g = pd.DataFrame(GU.apply(np.sum, axis = 0, result_type = 'expand'))/num_ens
        
        PZZ_i = GU.transpose() - g.values ## (n+p) by num_ens matrix
        PUZ_i = U.transpose() - m.values ## p by num_ens matrix
        
        Pzz_i = np.zeros(shape = [len(g),len(g)])
        Puz_i = np.zeros(shape = [len(m),len(g)])
        
        for j in range(num_ens):
            Pzz_i = Pzz_i + np.outer(np.array(PZZ_i.iloc[:,j]), np.array(PZZ_i.iloc[:,j]))
            Puz_i = Puz_i + np.outer(np.array(PUZ_i.iloc[:,j]), np.array(PZZ_i.iloc[:,j]))
        
        Pzz_i = Pzz_i/num_ens
        Puz_i = Puz_i/num_ens
        
        z = np.concatenate((y, x0))
        
        K_i = Puz_i@np.linalg.inv(Pzz_i+Q/alpha) ## kalman gain
        z_i = np.random.multivariate_normal(mean = z, cov = Q/alpha, size = num_ens)
        #y_i = np.random.multivariate_normal(mean = y, cov = 2*R/alpha, size = num_ens)
        #m_i = np.random.multivariate_normal(mean = x0, cov = 2*P/alpha, size = num_ens)
        
        for k in range(num_ens):
            U.iloc[k,:] = U.iloc[k,:] + (K_i@(z_i[k,:]-GU.iloc[k,:]))
        
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