from .utils import chol_inv, matrix_normal_log_star_std, matrix_normal_log_star2
import numpy as np
import numpy.linalg as ln
from scipy.stats import bernoulli
from scipy.sparse.csgraph import laplacian
from scipy.sparse import kron, eye
from scipy.sparse.linalg import inv as sp_inv
from scipy.linalg import inv
import scipy.stats

class scLL_LVM2:
    def __init__(self, G, epsilon, alpha, V, Cinit, tinit, xinit, yobserved, stepsize, ld):
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
        self.alpha, self.epsilon, self.ld, self.stepsize = alpha, epsilon, ld, stepsize

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

        #latent and observed variables
        self.C, self.t = Cinit, tinit
        self.e = self._e(self.C.reshape(self.Dy, self.N, self.Dt), self.t)
        self.x = xinit.reshape((self.N*self.Dy,1), order = 'F')
        self.y = yobserved.reshape((self.N*self.Dy,1), order='F')

        # final means
        self.C_mean = Cinit
        self.t_mean = tinit
        self.x_mean = self.x
        
        #this needs to be recomputed every time
        self.x_SigX_x = self.x.T.dot(self.sigma_x_inv.dot(self.x))
        
        # counts
        self.num_samples, self.accept_rate = 0, 0

        self.Cprop, self.tprop, self.eprop, self.xprop = self.C, self.t, self.e, self.x

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

    def _loglik_x_star(self, e, x):
        return -.5 * (self.x_SigX_x - 2*x.T.dot(e) + 
            e.T.dot(self.sigma_x.dot(e)))[0]

    def _loglik_dropouts(self, x):
        discrete = np.floor(np.exp(x))
        diff = x - self.y
        if np.alltrue(diff > 0):
            return self.ld * np.sum(diff)
        else:
            return -1 * float('inf')
        
    # calculate likelihood for proposed latent variables
    def likelihood(self):
        
        C, t, e, x = self.Cprop, self.tprop, self.eprop, self.xprop

        Cfactor = matrix_normal_log_star_std(C, self.C_priorprc)
        tfactor = matrix_normal_log_star_std(t, self.t_priorprc)
        xfactor = self._loglik_x_star(e,x)
        dropouts = self._loglik_dropouts(x)

        # print(Cfactor, tfactor, xfactor)
        return (Cfactor + tfactor + xfactor + dropouts)[0]
    
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
            self.x = np.copy(self.xprop)
            self.likelihoods.append(Lprime)
            # ToDo: add acceptance for x here for noisy version

        else:
            self.likelihoods.append(L)

        return accept

    # propose a new value based on current values
    def propose(self):
        self.Cprop = self.C + np.random.randn(self.Dy, self.N*self.Dt)*self.stepsize
        self.tprop = self.t + np.random.randn(self.Dt, self.N)*self.stepsize
        self.eprop = self._e(self.Cprop.reshape(self.Dy, self.N, self.Dt), self.tprop)
        self.xprop = self.x + np.random.randn(self.Dy, self.N)*self.stepsize
        self.x_SigX_x = self.xprop.T.dot(self.sigma_x.dot(self.xprop))
        
    def MH_step(self, burn_in=False):
        self.propose()
        accept = self.update()
        self.trace.append(self.t[0,0])
        if not burn_in:
            self.num_samples += 1
            self.accept_rate = ((self.num_samples - 1)*self.accept_rate + accept)/self.num_samples
            self.C_mean = ((self.num_samples - 1)*self.C_mean + self.C)/float(self.num_samples)
            self.t_mean = ((self.num_samples - 1)*self.t_mean + self.t)/float(self.num_samples)
            self.x_mean = ((self.num_samples - 1)*self.x_mean + self.x)/float(self.num_samples)
    
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
        x = np.random.multivariate_normal(mean=np.array(mu_x)[:,0], cov=self.sigma_x)
        x1 = np.floor(np.exp(x * 10))
        h = [np.random.binomial(int(xi),self.ld) for xi in x1]
        y = x1 - np.array(h)
        y = y.reshape((self.Dy,self.N),order="F")
        return x.reshape((self.Dy,self.N),order="F"), y, t, C
