data = datasets.make_swiss_roll(100,.01)
x = preprocessing.normalize(data[0])
x = x.T
t_true = data[1].T
Dy,N = x.shape
Dt = 1

tinit = np.random.multivariate_normal([0] * Dt * N, np.identity(Dt * N)*5).reshape((1,N))
Cinit = np.random.multivariate_normal([0] * Dt * N * Dy, np.identity(Dt * N * Dy)*5).reshape(Dy,Dt*N)

#build nearest neighbor graph
G = neighbors.kneighbors_graph(x.T,3).toarray()
G = ((G + G.T) > 0) * 1.0

#to initialize from the prior on t
#degree = np.sum(G,1)
#L = np.diag(degree) - G
#omega_inv = np.kron(2*L, np.identity(Dt))
#Pi_inv = chol_inv(alpha * np.identity(N*Dt) + omega_inv)
#tinit = np.random.multivariate_normal(np.zeros(Dt*N),Pi_inv).reshape((Dt,N))

#set user-defined parameters
alpha = 1.0
gamma =20.0
epsilon = .01
V = np.identity(Dy) * gamma
#V = np.cov(x)

model = LL_LVM(G,epsilon,alpha,V,Cinit,tinit,x,5.0)
model.likelihood()
model.propose()
model.likelihood(proposed=True)



