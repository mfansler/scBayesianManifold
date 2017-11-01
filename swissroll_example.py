from sklearn import datasets
from sklearn import neighbors
from numpy import linalg as ln
import scipy

data = datasets.make_swiss_roll(100,.01)
x = data[0].T
t_true = data[1].T
Dy,N = x.shape
Dt = 1

tinit = np.random.multivariate_normal([0] * Dt * N, np.identity(Dt * N)*5).reshape((1,N))
Cinit = np.random.multivariate_normal([0] * Dt * N * Dy, np.identity(Dt * N * Dy)*5).reshape(Dy,Dt*N)

#set user-defined parameters
alpha = 15.0
gamma = 1.0
epsilon = .00000001
V = np.identity(Dy) * gamma
#V = np.cov(x)

#build nearest neighbor graph
G = neighbors.kneighbors_graph(x.T,3).toarray()

model = LL_LVM(G,epsilon,alpha,V,Cinit,tinit,x)
