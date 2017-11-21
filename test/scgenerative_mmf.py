from sklearn import datasets, neighbors, preprocessing
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.csgraph import laplacian
from scipy.sparse import kron

# local imports
from sclllvm_test import scLL_LVM_test
from os import chdir
import os
chdir("../")

from lllvm import scLL_LVM
from lllvm.utils import chol_inv, matrix_normal_log_star, matrix_normal_log_star2, matrix_normal_log_star_std


# Plotting configuration
#%matplotlib inline
#rcParams['figure.figsize'] = (10,8)
#sns.set()


#generate fake data to get a graph that represents data lying on a manifold
x, t_true = datasets.make_swiss_roll(50, 0.01)
x = (x - np.mean(x,0)) / np.array([1, 1, 1])
x = x.T
Dy,N = x.shape
Dt = 1

G = neighbors.kneighbors_graph(x.T, 10, mode='connectivity')
G = G + G.T
G.data = np.ones_like(G.data)
neighbors = G.tolil().rows


#simulate data from the model
alpha = 1.0; gamma =0.1; epsilon = .0001; V_inv = np.identity(Dy) / gamma

#this isn't actually used; just to get the shapes right
t_init = np.random.multivariate_normal([0] * Dt * N, np.identity(Dt * N)*.05).reshape((1,N))
C_init = np.random.multivariate_normal([0] * Dt * N * Dy, np.identity(Dt * N * Dy)*0.05).reshape(Dy,Dt*N)
y = x * np.random.binomial(1,.1,Dy * N).reshape(x.shape)

model = scLL_LVM_test(G,epsilon,alpha,V_inv,C_init,t_init,x,y,0.01, 0.001, 0.001, 1.5)
y,x,t,C = model.simulate()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(10, -0)
ax.scatter(x[0,:], x[1,:], x[2,:], c=t.flatten(), marker='o')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(10, -0)
ax.scatter(y[0,:], y[1,:], y[2,:], c=t.flatten(), marker='o')
plt.show()


model = scLL_LVM_test(G,epsilon,alpha,V_inv,C,t,x,y,0.01, 0.001, 0.001, 1.5)
print("Likelihood under true latent variables: ", model.likelihood())

t_init = t + np.random.multivariate_normal([0] * Dt * N, np.identity(Dt * N )*5.5).reshape(t.shape)
C_init = C + np.random.multivariate_normal([0] * Dt * N * Dy, np.identity(Dt * N * Dy)*5.5).reshape(Dy,Dt*N)
x_init = x + np.random.multivariate_normal([0] * Dt * N * Dy, np.identity( N * Dy)*5.5).reshape(Dy,N)
rcParams['figure.figsize'] = (16,5)
plt.subplot(1, 3, 1)
plt.scatter(t_init.flatten(),t.flatten(),c=t.flatten())
plt.subplot(1, 3, 2)
plt.scatter(C_init.flatten(),C.flatten())
plt.subplot(1, 3, 3)
plt.scatter(x_init.flatten()[model.dropouts],x.flatten()[model.dropouts])
plt.show()


model = scLL_LVM_test(G, epsilon, alpha, V_inv,
                      C_init, t_init, y, y,
                      0.01, 0.001, 0.001, 1.5)
print("Likelihood under initialization: ", model.likelihood())

n_burn = 20000

for i in range(n_burn):
    print("\rStep %d of %d, Current Likelihood: %d, Acceptance Rate: %d, Current: %d " % (i+1,n_burn, model.likelihoods[-1],model.accept_rate * 100.0, model.a_current * 100.0), end="")
    model.MH_step(burn_in=True)

