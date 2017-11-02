data = datasets.make_swiss_roll(100,.01)
x = preprocessing.normalize(data[0])
x = x.T
t_true = data[1].T
Dy,N = x.shape
Dt = 1

tinit = np.random.multivariate_normal([0] * Dt * N, np.identity(Dt * N)*.25).reshape((1,N))
Cinit = np.random.multivariate_normal([0] * Dt * N * Dy, np.identity(Dt * N * Dy)*.25).reshape(Dy,Dt*N)

#build nearest neighbor graph
G = neighbors.kneighbors_graph(x.T,7).toarray()
G = ((G + G.T) > 0) * 1.0

#set user-defined parameters
alpha = 1.0
gamma =100.0
epsilon = .00001
#V = np.identity(Dy) * gamma
V = np.cov(x) * gamma

#to initialize from the priors
#degree = np.sum(G,1)
#L = np.diag(degree) - G
#omega_inv = np.kron(2*L, np.identity(Dt))
#Pi_inv = chol_inv(alpha * np.identity(N*Dt) + omega_inv)
#J = np.kron(np.ones(shape=(N,1)),np.identity(Dt))
#C_priorcov = ln.inv(epsilon * J * J.T + omega_inv)
#tinit = np.random.multivariate_normal(np.zeros(Dt*N),Pi_inv).reshape((Dt,N))
#Cinit = np.random.multivariate_normal(np.zeros(shape=(Dy*N*Dt)),np.kron(np.identity(Dy),C_priorcov)).reshape((Dy,Dt*N))


model = LL_LVM(G,epsilon,alpha,V,Cinit,tinit,x,.5)
model.likelihood()
model.propose()
model.likelihood(proposed=True)

for i in range(10000):
    model.MH_step()




