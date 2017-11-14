from .utils import chol_inv, matrix_normal_log_star_std, matrix_normal_log_star2
import numpy as np
import numpy.linalg as ln
from scipy.stats import bernoulli
from scipy.sparse.csgraph import laplacian
from scipy.sparse import kron, eye
from scipy.sparse.linalg import inv as sp_inv
from scipy.linalg import inv
import scipy.stats

class LL_LVM2:
    def __init__(self, G, epsilon, alpha, V, Cinit, tinit, xinit, stepsize, yobserved=0):
        """
        G is the N by N nearest-neighbor graph adjacency matrix
        Cinit is the Dy by N*Dt matrix of initial linear maps
        tinit is the Dt by N matrix of initial low-dimensional embeddings
        xinit is the Dy by N matrix of initial true expression
        yobserved is the Dy by N observed expression
        """
        # pre-compute fixed parameters #
        # dims
        self.N = G.shape[0]
        self.Dy = xinit.shape[0]
        self.Dt = tinit.shape[0]

        # constants
        self.alpha, self.epsilon, self.stepsize = alpha, epsilon, stepsize

        self.V = V
        self.Vinv = chol_inv(V)
        self.J = kron(np.ones(shape=(self.N, 1)), eye(self.Dt))

        # graph laplacian
        self.L = laplacian(G)
        
        # precision matrices
        self.omega_inv = kron(2*self.L, eye(self.Dt))
        self.sigma_x_inv = kron(np.full((self.N, self.N), self.epsilon), eye(self.Dy)) + kron(2*self.L, self.Vinv)

        # ToDo: Can we avoid computing this?
        self.sigma_x = chol_inv(self.sigma_x_inv.todense())

        # precision matrices for priors
        self.t_priorprc = self.alpha*eye(self.N) + 2*self.L
        self.C_priorprc = epsilon*self.J.dot(self.J.T) + self.omega_inv

        # create a dictionary of neighbors
        self.neighbors = G.tolil().rows

        # initialize latent variables and observations
        # just put observed if standard LL_LVM
        self.C, self.t, self.x = Cinit, tinit, xinit
        self.xvec = self.x.reshape((self.N*self.Dy, 1), order="F")
        
        self.e = self._e(self.C.reshape(self.Dy, self.N, self.Dt), self.t)

        # final means
        self.C_mean = Cinit
        self.t_mean = tinit

        # ToDo: Eliminate unnecessary reshaping
        # cache (x * Sig_x * x)
        self.x_SigX_x = self.xvec.T.dot(self.sigma_x_inv.dot(self.xvec))

        # counts
        self.num_samples, self.accept_rate = 0, 0

        self.Cprop, self.tprop, self.eprop = self.C, self.t, self.e

        # initialize variables to store trace and likelihood
        self.trace, self.likelihoods = [], [self.likelihood()]
        
        self.Pi = kron(chol_inv(self.alpha * np.eye(self.N) + 2*self.L) , np.eye(self.Dt))
        self.C_priorcov = chol_inv(self.C_priorprc.todense())
    def _e(self, C, t):

        # precompute differences in t
        # NB: t[:,i] - t[:,j] = t_diff[i, :, j]
        t_diff = np.kron(np.ones((self.N, 1)), t).reshape(self.N, self.Dt, self.N)
        t_diff = t_diff - t_diff.transpose()

        # ToDo: Parallelize this loop
        e = np.empty((self.Dy, self.N))
        for i in range(self.N):
            e[:, i] = np.sum((C[:,i,:] + C[:,j,:]).dot(t_diff[i, :, j]) for j in self.neighbors[i])
        e = - self.Vinv.dot(e)

        return e.flatten(order='F')#.reshape((self.N*self.Dy, 1),order="F")

    def _loglik_x_star(self, e):
        # ToDo: eliminate reshaping
        return -.5 * (self.x_SigX_x - 2*self.xvec.T.dot(e) + e.T.dot(self.sigma_x.dot(e)))[0]

    def _log_MV_normal(self,x,mu,prc):
        x, mu, prc = np.matrix(x), np.matrix(mu).T, np.matrix(prc)
        return float(-.5 *  (x-mu).T * chol_inv(prc) * (x-mu))
    
    # calculate likelihood for proposed latent variables
    def likelihood(self):
        
        C, t, e = self.Cprop, self.tprop, self.eprop

        Cfactor = matrix_normal_log_star_std(C, self.C_priorprc)
        tfactor = matrix_normal_log_star_std(t, self.t_priorprc)
        xfactor = self._loglik_x_star(e)

        # print(Cfactor, tfactor, xfactor)
        return (Cfactor + tfactor + xfactor)[0]
    
    # update with Metropolis-Hastings step
    def update(self):
        # calculate likelihood
        Lprime = self.likelihood()
        L = self.likelihoods[-1]
        # calculate acceptance probability
        a = 1.0 if Lprime > L else np.exp(Lprime - L)
        accept = bernoulli.rvs(a)
        
        # update the variables
        if accept:
            self.C = np.copy(self.Cprop)
            self.t = np.copy(self.tprop)
            self.e = np.copy(self.eprop)
            self.likelihoods.append(Lprime)

            # ToDo: add acceptance for x here for noisy version

        else:
            self.likelihoods.append(L)

        return accept

    # propose a new value based on current values
    def propose(self,i,var):
        draw = float(np.random.randn(1))
        if var == "C":
            vec = np.zeros(self.Dy*self.N*self.Dt)
            vec[i] = draw
            vec = vec.reshape((self.Dy, self.N*self.Dt),order='F')
            self.Cprop = self.C +vec*self.stepsize
        if var == "t":
            vec = np.zeros(self.Dt * self.N)
            vec[i] = draw
            vec = vec.reshape((self.Dt, self.N),order="F")
            self.tprop = self.t + vec*self.stepsize
            
        self.eprop = self._e(self.Cprop.reshape(self.Dy, self.N, self.Dt), self.tprop)

        # ToDo: add proposal for x here for noisy version

    def MH_step(self, burn_in=False):
        for i in range(self.Dt * self.N):
            self.propose(i,'t')
            accept = self.update()
        for i in range(self.Dy*self.N*self.Dt):
            self.propose(i,'C')
            accept = self.update()
        
        self.trace.append(self.t[0,0])
        if not burn_in:
            self.num_samples += 1
            self.accept_rate = ((self.num_samples - 1)*self.accept_rate + accept)/self.num_samples
            self.C_mean = ((self.num_samples - 1)*self.C_mean + self.C)/float(self.num_samples)
            self.t_mean = ((self.num_samples - 1)*self.t_mean + self.t)/float(self.num_samples)
    
    def autocorrelation(self,maxlag):
        trace = np.array(self.trace)[np.arange(self.num_samples,len(self.trace))]
        return [np.corrcoef(trace[0:len(trace)-i], trace[i:len(trace)])[0,1] for i in np.arange(1,maxlag)]
        
    def simulate(self):
        Pi = kron(chol_inv(self.alpha * np.eye(self.N) + 2*self.L) , np.eye(self.Dt))
        C_priorcov = chol_inv(self.C_priorprc.todense())
        t = np.random.multivariate_normal(np.zeros(self.Dt*self.N),Pi.todense()).reshape((self.Dt,self.N),order='F')
        C = np.random.multivariate_normal(np.zeros(shape=(self.Dy*self.N*self.Dt)),kron(C_priorcov,np.eye(self.Dy)).todense()).reshape((self.Dy,self.Dt*self.N),order='F')
        e = self._e(C.reshape(self.Dy, self.N, self.Dt), t)
        mu_x = np.matrix(self.sigma_x) * np.matrix(e).T
        x = np.random.multivariate_normal(mean=np.array(mu_x)[:,0], cov=self.sigma_x).reshape((self.Dy,self.N),order="F")
        return x, t, C
