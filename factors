from numpy import linalg as ln
import scipy

#T will be a vector of user-defined parameters
#Cstar is a Dy by N*Dt dimensional matrix 
def C_GT(Cstar,L):
    J = np.kron(np.ones(N),np.identity(Dt))
    V = epsilon* J * J.T + omega_inv(L,Dt)
    return matrix_normal(Cstar,np.zeros(Dy),np.identity(Dy),ln.inv(V))
 
#tstar is a Dt by N dimensional matrix     
def t_GT(tstar,L):
    return matrix_norma(tstar,np.zeros(Dt),np.identity(Dt),ln.inv(alpha*np.identity(N) + 2*L))

def x_CtGT(xstar,L):
    sigma_x = np.kron((epsilon * np.ones(shape=(N,1)) * np.ones(shape=(N,1)).T).T, np.identity(Dy))
    sigma_x = sigma_x + 2 * np.kron(L * Vinv)
    mu_x = sigma_x * e 
    scipy.stats.multivariate_normal.pdf(xstar,mu_x,sigma_x)
    