from sklearn import datasets, neighbors, preprocessing
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from lllvm import LL_LVM

data = datasets.make_swiss_roll(300, 0.01)
#x = preprocessing.normalize(data[0])
x = data[0] / np.sum(data[0],0)
x = x.T
t_true = data[1].T
Dy,N = x.shape
Dt = 1

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[0,:], x[1,:], x[2,:], c=t_true, marker='o')
# plt.show()

#tinit = np.random.multivariate_normal([0] * Dt * N, np.identity(Dt * N)*.25).reshape((1,N))
tinit = np.random.uniform(-1.5, 1.5, size=Dt * N).reshape((Dt,N))
Cinit = np.random.multivariate_normal([0] * Dt * N * Dy, np.identity(Dt * N * Dy)*0.25).reshape(Dy,Dt*N)

#build (undirected) nearest neighbor graph
G = neighbors.kneighbors_graph(x.T, 9, mode='connectivity')
G = G + G.T
G.data = np.ones_like(G.data)

#set user-defined parameters
alpha = 1.0
gamma = 5.0
epsilon = .00001
V = np.identity(Dy) / gamma
#V = np.cov(x) * gamma

#to initialize from the priors
#degree = np.sum(G,1)
#L = np.diag(degree) - G
#omega_inv = np.kron(2*L, np.identity(Dt))
#Pi_inv = chol_inv(alpha * np.identity(N*Dt) + omega_inv)
#J = np.kron(np.ones(shape=(N,1)),np.identity(Dt))
#C_priorcov = ln.inv(epsilon * J * J.T + omega_inv)
#tinit = np.random.multivariate_normal(np.zeros(Dt*N),Pi_inv).reshape((Dt,N))
#Cinit = np.random.multivariate_normal(np.zeros(shape=(Dy*N*Dt)),np.kron(np.identity(Dy),C_priorcov)).reshape((Dy,Dt*N))


model = LL_LVM(G,epsilon,alpha,V,Cinit,tinit,x,.0005)
#model.likelihood()
#model.propose()
#model.likelihood(proposed=True)

for i in range(10):
    print(i)
    model.MH_step(burn_in=True)
for i in range(30):
    print(i)
    model.MH_step(burn_in=False)

print(model.acceptance / 4000.0)
print(model.likelihoods)
t = model.tfinal / 3000.0

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[0,:], x[1,:], x[2,:], c=t, marker='o')
# plt.show()
