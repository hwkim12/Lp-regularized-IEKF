# IEKF+IAS for the Nonlinear Toy Example 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IEKF import IEKF

np.random.seed(1234567)

# Grid construction in [0,1] by [0,1] 
xgrid = np.linspace(0, 1, 21)
ygrid = np.linspace(0, 1, 21)

KK = 30 # Number of each sin/cos basis components


# Solution of the first order PDE at (x,y) 
def PDE_SOL(x,y,uu):
    sin_term_x = []
    cos_term_x = []
    
    sin_term_xy = []
    cos_term_xy = []
    
    for k in range(KK):
        sin_term_x.append(np.sin((k+1)*np.pi*x)*(1/((k+1)*np.pi)))
        cos_term_x.append(-np.cos((k+1)*np.pi*x)*(1/((k+1)*np.pi)))
        
        sin_term_xy.append(np.sin((k+1)*np.pi*(x+y))*(1/((k+1)*np.pi)))
        cos_term_xy.append(-np.cos((k+1)*np.pi*(x+y))*(1/((k+1)*np.pi)))
    
    init_cond = np.cos(x+y)
    
    trig_term_x = np.concatenate((np.array(sin_term_x), np.array(cos_term_x)))
    trig_term_xy = np.concatenate((np.array(sin_term_xy), np.array(cos_term_xy)))
    
    return init_cond*np.exp(np.dot(trig_term_x, uu)-np.dot(trig_term_xy, uu))

# Forward map
def h_u(uu):
    outcome_mat = np.zeros((len(xgrid), len(ygrid)))
    for i in range(len(xgrid)):
        x = xgrid[i]
        for j in range(len(ygrid)):
            y = ygrid[j]
            outcome_mat[i, j] = PDE_SOL(x, y, uu)
    
    return outcome_mat.flatten()

# True parameter
true_u = np.zeros(KK*2)
true_u[0] = 0.4
true_u[2] = 0.4
true_u[5] = -0.4
true_u[KK] = -0.2
true_u[KK+2] = -0.4
true_u[KK+5] = 0.2

true_u = true_u*3

# Data construction
noiseless_dat= h_u(true_u)
sigma = 0.1
y = noiseless_dat + np.random.normal(0, sigma, len(xgrid)**2)

# Matrix for recovering functions
sin_term = np.sin(np.pi*xgrid)
cos_term = np.cos(np.pi*xgrid)
sin_term = pd.DataFrame(sin_term)
cos_term = pd.DataFrame(cos_term)

sin_cos_mat = pd.concat([sin_term, cos_term], axis = 1)

for k in range(KK+1):
    if k > 1:
        sin_term = np.sin(k*np.pi*xgrid)
        cos_term = np.cos(k*np.pi*xgrid)
        sin_term = pd.DataFrame(sin_term)
        cos_term = pd.DataFrame(cos_term)
        sin_cos_mat = pd.concat([sin_cos_mat, sin_term, cos_term], axis = 1)


tsin_cos_mat = sin_cos_mat.T # This is for building recovery function

# Regularization parameter value
r = 1/3
#r = 1

# Covariance matrices, Step size, Initial Mean
R = sigma*np.identity(len(y))
alpha= 0.1
u0 = np.zeros(len(true_u))

# Inner iteration and Number of Ensembles
inner_niter = 20
num_ens = 100

# Prior Covariance Matrix
P2 = 0.04*np.identity(len(true_u))

# Initial IEKF
IEKF_ens = IEKF(u0, alpha, inner_niter, h_u, y, P2, R, num_ens)
IEKF_ens = pd.DataFrame(IEKF_ens)
U2 = IEKF_ens.apply(np.mean, axis = 0, result_type ='expand') 
U2 = np.array(U2)
THETA_2 = np.array((U2**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
P2 = np.diag(THETA_2)

# Constructing Approximate Credible Intervals
IEKF_ens_mat = np.matrix(IEKF_ens)
tsin_cos_mat_temp = np.matrix(tsin_cos_mat)
IEKF_recov = IEKF_ens_mat@tsin_cos_mat_temp
IEKF_recov = pd.DataFrame(IEKF_recov)

lower_curve_IEKF = IEKF_recov.apply(np.quantile, axis = 0, q = 0.025, result_type ='expand')
lower_curve_IEKF = np.array(lower_curve_IEKF)
upper_curve_IEKF = IEKF_recov.apply(np.quantile, axis = 0, q = 0.975, result_type ='expand') 
upper_curve_IEKF = np.array(upper_curve_IEKF)

# Average width of the Credible Intervals and l2-error
avg_width = []
l2_error = []

# Plotting
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
ax[0].plot(xgrid, sin_cos_mat@true_u, label = 'Truth', color = 'red', alpha = 1)
ax[0].plot(xgrid, sin_cos_mat@U2, label = 'IEKF', color = 'blue', alpha = 1)
ax[0].fill_between(xgrid, lower_curve_IEKF, upper_curve_IEKF, alpha = 0.2, color = 'blue')
ax[0].set_title('Recovery based on 1st L0.5-IEKF', fontsize = 16)
#ax[0].set_title('Recovery based on 1st L1-IEKF', fontsize = 16)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].legend(loc = "upper right", fontsize = 12)
ax[0].set_ylim(-4,5)
avg_width.append(np.mean(upper_curve_IEKF-lower_curve_IEKF))
l2_error.append(np.linalg.norm(true_u-U2)/np.sqrt(len(true_u)))

# Outer iteration for Lp-IEKF
for k in range(3):
    IEKF_ens = IEKF(u0, alpha, inner_niter, h_u, y, P2, R, num_ens)
    IEKF_ens = pd.DataFrame(IEKF_ens)
    U2 = IEKF_ens.apply(np.mean, axis = 0, result_type ='expand') 
    U2 = np.array(U2)
    THETA_2 = np.array((U2**2)**(1/(r+1))/(2*r)**(1/(r+1))) ### Lp regularization ###
    P2 = np.diag(THETA_2)
    
    # Credible Interval
    IEKF_ens_mat = np.matrix(IEKF_ens)
    tsin_cos_mat_temp = np.matrix(tsin_cos_mat)
    IEKF_recov = IEKF_ens_mat@tsin_cos_mat_temp
    IEKF_recov = pd.DataFrame(IEKF_recov)
    
    
    lower_curve_IEKF = IEKF_recov.apply(np.quantile, axis = 0, q = 0.025, result_type ='expand')
    lower_curve_IEKF = np.array(lower_curve_IEKF)
    upper_curve_IEKF = IEKF_recov.apply(np.quantile, axis = 0, q = 0.975, result_type ='expand') 
    upper_curve_IEKF = np.array(upper_curve_IEKF)

    if k == 0:
        ax[1].plot(xgrid, sin_cos_mat@true_u, label = 'Truth', color = 'red', alpha = 1)
        ax[1].plot(xgrid, sin_cos_mat@U2, label = 'IEKF', color = 'blue', alpha = 1)
        ax[1].fill_between(xgrid, lower_curve_IEKF, upper_curve_IEKF, alpha = 0.2, color = 'blue')
        ax[1].set_title('Recovery based on 2nd L0.5-IEKF', fontsize = 16)
        #ax[1].set_title('Recovery based on 2nd L1-IEKF', fontsize = 16)
        ax[1].tick_params(axis='both', which='major', labelsize=12)
        ax[1].legend(loc = "upper right", fontsize = 12)
        ax[1].set_ylim(-4,5)
        avg_width.append(np.mean(upper_curve_IEKF-lower_curve_IEKF))
        l2_error.append(np.linalg.norm(true_u-U2)/np.sqrt(len(true_u)))
                
    if k == 2:
        ax[2].plot(xgrid, sin_cos_mat@true_u, label = 'Truth', color = 'red', alpha = 1)
        ax[2].plot(xgrid, sin_cos_mat@U2, label = 'IEKF', color = 'blue', alpha = 1)
        ax[2].fill_between(xgrid, lower_curve_IEKF, upper_curve_IEKF, alpha = 0.2, color = 'blue')
        ax[2].set_title('Recovery based on 4th L0.5-IEKF', fontsize = 16)
        #ax[2].set_title('Recovery based on 4th L1-IEKF', fontsize = 16)
        ax[2].tick_params(axis='both', which='major', labelsize=12)
        ax[2].legend(loc = "upper right", fontsize = 12)
        ax[2].set_ylim(-4,5)
        avg_width.append(np.mean(upper_curve_IEKF-lower_curve_IEKF))
        l2_error.append(np.linalg.norm(true_u-U2)/np.sqrt(len(true_u)))
  
    #print(k)
