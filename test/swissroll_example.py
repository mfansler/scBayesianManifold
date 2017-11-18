from sklearn import datasets, neighbors, preprocessing
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from lllvm import LL_LVM
from lllvm.utils import infer_graph, initialize_C, initialize_t
#from lllvm.plot import plot_G, plot_C_arrows

np.random.seed(6471)

x, t_true = datasets.make_swiss_roll(500, 0.01)
x = (x - x.mean(0)) / x.std(0)
x = x.T
Dy, N = x.shape
Dt = 1

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[0,:], x[1,:], x[2,:], c=t_true, marker='o')
# plt.show()


# build (undirected) nearest neighbor graph
G = infer_graph(x, k=10, keep_asymmetries=False, delta=2.5)
t_init = initialize_t(G, x)
C_init = initialize_C(x, t_init, G)

#ax = plot_G(t_true, x, G)
#ax.view_init(10, -80)
#plt.show()

#ax = plot_C_arrows(x, t_true.reshape((1,N)), C_init, scale_C=1/20)
#ax.view_init(10, -80)
#plt.show()

# set user-defined parameters
alpha = 1.0
gamma = 0.01
epsilon = .00001
V_inv = gamma * np.identity(Dy)

model = LL_LVM(G, epsilon, alpha, V_inv, C_init, t_init, x, .0005, 0.01)

n_burn = 1000
n_samp = 1000
for i in range(n_burn):
    print("\rStep %d of %d; log-likelihood = %f" % (i + 1, n_burn, model.likelihoods[-1]), end="")
    model.MH_step(burn_in=True)
for i in range(n_samp):
    print("\rStep %d of %d; log-likelihood = %f" % (i + 1, n_samp, model.likelihoods[-1]), end="")
    model.MH_step(burn_in=False)

print(model.accept_rate)
#print(model.likelihoods)
#t = model.t_mean

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[0,:], x[1,:], x[2,:], c=t, marker='o')
# plt.show()
