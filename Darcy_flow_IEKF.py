# IEKF+IAS for the Darcy-Flow Example 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IEKF import IEKF

np.random.seed(1)

# Finite Difference Solver of the Elliptic PDE on [0,1] by [0,1]
def darcy_fin_diff(f_var, K, h_diff):
    n = K.shape[0]
    mat_temp = np.transpose(K.iloc[1:(n-1), 1:(n+1)])
    mat_temp = mat_temp.melt()
    U = np.diag(mat_temp['value'])
    U = pd.DataFrame(U)
    zero_vec = pd.DataFrame(np.zeros((U.shape[0], n)))
    U = pd.concat([zero_vec, U], axis = 1, ignore_index= True)
    zero_vec = pd.DataFrame(np.zeros((n, U.shape[1])))
    U = pd.concat([U, zero_vec], axis = 0, ignore_index= True)
    
    mat_temp2 = np.transpose(K.iloc[1:(n-2), 1:(n+1)])
    mat_temp2 = mat_temp2.melt()
    mat_temp22 = K.iloc[(n-2), 1:(n+1)] + K.iloc[n-1, 1:(n+1)]
    L = np.diag([*mat_temp2['value'], *mat_temp22])
    L = pd.DataFrame(L)
    zero_vec = pd.DataFrame(np.zeros((n, L.shape[1])))
    L = pd.concat([zero_vec, L], axis = 0, ignore_index= True)
    zero_vec = pd.DataFrame(np.zeros((L.shape[0], n)))
    L = pd.concat([L, zero_vec], axis = 1, ignore_index= True)
    
    K_U = K.iloc[1, 1:n]
    K_U = np.array(K_U)
    K_U[0] = K_U[0] + K.iloc[1, 0]
    K_U = np.append(K_U, 0)
    
    K_L = K.iloc[1, 1:n]
    K_L = np.array(K_L)
    K_L[n-2] = K_L[n-2] + K.iloc[1, n]
    K_L = np.append(K_L, 0)
    
    K_M = 2*K.iloc[1, 1:(n+1)]
    K_M = np.array(K_M) + np.array(K.iloc[1, 0:n])
    K_M = np.array(K_M) + np.array(K.iloc[0, 1:(n+1)])
    K_M = -K_M
    
    b = 100*K.iloc[0, 1:(n+1)]/h_diff**2
    b[1] = b[1] + 1000*K.iloc[1, 0]/(K.iloc[1, 1]*h_diff)
    
    for kk in np.linspace(2, n-1, num = n-2):
        kk = int(kk)
        temp1 = np.array(K.iloc[kk, 1:n])
        temp1[0] = temp1[0] + K.iloc[kk,0]
        if kk != (n-1):
            temp1 = np.append(temp1, 0)
        K_U = [*K_U, *temp1]
        
        temp2 = np.array(K.iloc[kk, 1:n])
        temp2[n-2] = temp2[n-2] + K.iloc[kk,n]
        if kk != (n-1):
            temp2 = np.append(temp2, 0)
        K_L = [*K_L, *temp2]
        
        temp3 = np.array(2*K.iloc[kk, 1:(n+1)])
        temp3 = temp3 + np.array(K.iloc[kk, 0:n])
        temp3 = temp3 + np.array(K.iloc[(kk-1), 1:(n+1)])
        temp3 = -temp3
        K_M = [*K_M, *temp3] 
        
        temp4 = np.zeros(n)
        temp4[0] = 1000*K.iloc[kk, 0]/(K.iloc[kk, 1]*h_diff)
        b = [*b, *temp4]
    
    K_U = np.diag(K_U)
    K_U = pd.DataFrame(K_U)
    zero_vec = pd.DataFrame(np.zeros((1, K_U.shape[1])))
    K_U = pd.concat([K_U, zero_vec], axis = 0, ignore_index = True)
    zero_vec = pd.DataFrame(np.zeros((K_U.shape[0], 1)))
    K_U = pd.concat([zero_vec, K_U], axis = 1, ignore_index = True)
    
    K_L = np.diag(K_L)
    K_L = pd.DataFrame(K_L)
    zero_vec = pd.DataFrame(np.zeros((K_L.shape[0], 1)))
    K_L = pd.concat([K_L, zero_vec], axis = 1, ignore_index = True)
    zero_vec = pd.DataFrame(np.zeros((1, K_L.shape[1])))
    K_L = pd.concat([zero_vec, K_L], axis = 0, ignore_index = True)
    
    K_M = np.diag(K_M)
    K_M = pd.DataFrame(K_M)
    
    AA = U + L + K_U + K_L + K_M
    AA = AA/(h_diff**2)
    AA = pd.DataFrame(AA)
    
    f_var = f_var.melt()
    f_var = f_var['value']
    sol = np.linalg.solve(AA, -f_var-b)
    
    return sol

# Construction of grid on [0,1] by [0,1]
ngrid = 15
xgrid = ygrid = np.linspace(0, 1, num = ngrid)
h_diff = xgrid[1]
xgridEX = np.linspace(0, 1, num = 15)
xgridEX = np.insert(xgridEX, 0, -h_diff)


X2grid = np.tile(xgridEX, (ngrid, 1))
X2grid = pd.DataFrame(X2grid)
X2grid = X2grid.melt()
X2grid = X2grid["value"]

Y2grid = np.tile(ygrid, (ngrid+1, 1))
Y2grid = pd.DataFrame(Y2grid)
Y2grid = Y2grid.T
Y2grid = Y2grid.melt()
Y2grid = Y2grid["value"]
    
# Construction of the basis matrix
A = pd.DataFrame(np.ones(len(X2grid)))
for i in range(20):
    for j in range(20):
        if i == 0 and j == 0:
            A = A
        else:
            A = pd.concat([A, np.cos(i*np.pi*X2grid)*np.cos(j*np.pi*Y2grid)], axis = 1, ignore_index = True)


# True parameter
u = np.zeros(A.shape[1])

u[1] = 0.2
u[7] = 0.3

u[22*6] = 0.1
u[36*6] = -0.1

u[42*7] = -0.2
u[55*7] = -0.1

# Construction of the log_diffusion coefficient
logk = pd.DataFrame(A@u)
logk_mat = logk.values.reshape(ngrid+1, ngrid)
logk_mat = pd.DataFrame(logk_mat)
logk_mat = logk_mat.transpose()

K = np.exp(logk_mat)
K = pd.DataFrame(K)

# Defining source term 
f = np.zeros((len(ygrid)-1, len(xgrid)))
f = pd.DataFrame(f)

f_ind1 = np.logical_and([xgrid > 4/6], [xgrid <= 5/6])[0]
f_ind1 = pd.DataFrame(f_ind1)
f_ind1 = f_ind1.squeeze()
f_ind1 = f_ind1[f_ind1].index.values - 1

f.iloc[f_ind1, ] = 137

f_ind2 = np.logical_and([xgrid > 5/6], [xgrid <= 1])[0]
f_ind2 = pd.DataFrame(f_ind2)
f_ind2 = f_ind2.squeeze()
f_ind2 = f_ind2[f_ind2].index.values - 1

f.iloc[f_ind2, ] = 274

# Noiseless Data
res = darcy_fin_diff(f, K, h_diff)
res_mat = np.matrix(res)
res_mat = res_mat.reshape(len(ygrid), len(xgrid)-1)
res_mat = pd.DataFrame(res_mat)

# Log diffusion coefficient function
def logK_func(U):
    LOGK = pd.DataFrame(A@U)
    LOGK = LOGK.values.reshape(ngrid+1, ngrid)
    LOGK = pd.DataFrame(LOGK)
    LOGK = LOGK.transpose()
    
    return LOGK

# Forward Map
def h_u(U):
    LOGK = logK_func(U)
    KAPPA = np.exp(LOGK)
    
    return darcy_fin_diff(f, KAPPA, h_diff)


# Generating Data
sigma = 0.1
y = res + np.random.normal(0, sigma, len(res))

# Noise covariance and Prior Covariance
R = sigma*np.identity(len(y))
P_x = 0.1*np.identity(len(u))
    
# Step size and Initialization
alpha = 0.5
u0 = np.zeros(len(u))

# Inner iteration number and ensemble size
inner_niter = 30
num_ens = 400


# Regularization parameter for IEKF_SL and IEKF
r = 1/3
#r = 1

# Average width of the Credible Intervals and l2-error
avg_width = []
l2_error = []

# Outer iteration
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
for k in range(7):
    x_mat = IEKF(x0 = np.zeros(len(u)), alpha = alpha, inner_niter = 30, h_x = h_u, y = y, P = P_x, R = R, num_ens = num_ens)
    x_mat = pd.DataFrame(x_mat)
    
    # Generalizes to nonlinear
    x_l = x_mat.apply(np.quantile, axis = 0, q = 0.025, result_type ='expand')
    x_l = np.array(x_l)
    x_u = x_mat.apply(np.quantile, axis = 0, q = 0.975, result_type ='expand') 
    x_u = np.array(x_u)
    
    x = x_mat.apply(np.mean, axis = 0, result_type ='expand') 
    x = np.array(x)

    theta_x = np.array((x**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
    P_x = np.diag(theta_x)
    
    avg_width.append(np.mean(x_u-x_l))
    l2_error.append(np.linalg.norm(x-u))


    if k == 0:
        ax[0].plot(range(len(u)), u, 'D', label = 'Truth', color = 'red', alpha = 1)
        ax[0].plot(range(len(u)), x, 'o', label = 'IEKF', color = 'blue', alpha = 1)
        ax[0].errorbar(range(len(u)), x, yerr = (x-x_l, x_u-x), fmt='.', color='blue', ecolor='skyblue', capsize=5, capthick = 1.25, label='IEKF-CR')
        ax[0].set_title('Recovery based on 1st L05-IEKF', fontsize = 16)
        #ax[0].set_title('Recovery based on 1st L1-IEKF', fontsize = 16)
        ax[0].tick_params(axis='both', which='major', labelsize=12)
        ax[0].legend(loc = "upper right", fontsize = 12)
        ax[0].set_ylim(-1,1)
    if k == 3:
        ax[1].plot(range(len(u)), u, 'D', label = 'Truth', color = 'red', alpha = 1)
        ax[1].plot(range(len(u)), x, 'o', label = 'IEKF', color = 'blue', alpha = 1)
        ax[1].errorbar(range(len(u)), x, yerr = (x-x_l, x_u-x), fmt='.', color='blue', ecolor='skyblue', capsize=5, capthick = 1.25, label='IEKF-CR')
        ax[1].set_title('Recovery based on 4th L05-IEKF', fontsize = 16)
        #ax[1].set_title('Recovery based on 4th L1-IEKF', fontsize = 16)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        ax[1].legend(loc = "upper right", fontsize = 12)
        ax[1].set_ylim(-1,1)
    if k == 6:
        ax[2].plot(range(len(u)), u, 'D', label = 'Truth', color = 'red', alpha = 1)
        ax[2].plot(range(len(u)), x, 'o', label = 'IEKF', color = 'blue', alpha = 1)
        ax[2].errorbar(range(len(u)), x, yerr = (x-x_l, x_u-x), fmt='.', color='blue', ecolor='skyblue', capsize=5, capthick = 1.25, label='IEKF-CR')
        ax[2].set_title('Recovery based on 7th L05-IEKF', fontsize = 16)
        #ax[2].set_title('Recovery based on 7th L1-IEKF', fontsize = 16)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        ax[2].legend(loc = "upper right", fontsize = 12)
        ax[2].set_ylim(-1,1)
    print(k)

  
  