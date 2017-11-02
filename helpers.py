def matrix_normal(A,M,U,V):
    n,p = A.shape
    A = np.matrix(A); M = np.matrix(M); U = np.matrix(U); V = np.matrix(V)
    return np.exp(-.5 * np.trace(ln.inv(V) * (A-M).T * ln.inv(U) * (A-M))) / ((2 * np.pi)**(n*p/2.0) * ln.det(V)**(n/2.0) * ln.det(U)**(p/2.0))

def matrix_normal_log(A,M,U,V):
    n,p = A.shape
    A = np.matrix(A); M = np.matrix(M); U = np.matrix(U); V = np.matrix(V)
    return -.5 * np.trace(ln.inv(V) * (A-M).T * ln.inv(U) * (A-M)) - np.log(((2 * np.pi)**(n*p/2.0) * ln.det(V)**(n/2.0) * ln.det(U)**(p/2.0))

def chol_inv(X):
    L = np.linalg.cholesky(X)
    Linv = scipy.linalg.solve_triangular(L,np.identity(L.shape[0]),lower=True)
    return Linv.T * np.matrix(Linv)











