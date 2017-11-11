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
Dy, N = x.shape
Dt = 1

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[0,:], x[1,:], x[2,:], c=t_true, marker='o')
# plt.show()

tinit = np.random.uniform(-1.5, 1.5, size=(Dt, N))
Cinit = 0.25*np.random.randn(Dy, N*Dt)

# build (undirected) nearest neighbor graph
G = neighbors.kneighbors_graph(x.T, 9, mode='connectivity')
G = G + G.T
G.data = np.ones_like(G.data)

# set user-defined parameters
alpha = 1.0
gamma = 5.0
epsilon = .00001
V = np.identity(Dy) / gamma

model = LL_LVM(G, epsilon, alpha, V, Cinit, tinit, x, .0005)

for i in range(100):
    print(i)
    model.MH_step(burn_in=True)
for i in range(300):
    print(i)
    model.MH_step(burn_in=False)

print(model.accept_rate)
print(model.likelihoods)
t = model.t_mean

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[0,:], x[1,:], x[2,:], c=t, marker='o')
# plt.show()
