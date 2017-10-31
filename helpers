from numpy import linalg as ln

def L(G):
    n = G.shape[0]
    return np.diag(G * np.ones(shape=(n,1))) - G

def omega_inv(L,Dt):
    np.kron(2*L, np.identity(Dt))

def matrix_normal(Astar,M,U,V):
    n,p = A.shape
    return (np.exp(-.5 * ln.inv(V) * (A-M).T * \
    ln.inv(U) * (A-M)) / ((2 * np.pi)**(n*p/2.0) \ 
    * ln.det(V)**(n/2.0) * ln.det(U)**(p/2.0))

