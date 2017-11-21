from .utils import chol_inv, matrix_normal_log_star_std, matrix_normal_log_star2
import numpy as np
import numpy.linalg as ln
from scipy.stats import bernoulli
from scipy.sparse.csgraph import laplacian
from scipy.sparse import kron, eye
from scipy.sparse.linalg import inv as sp_inv
from scipy.linalg import inv
import scipy.stats


class scLL_LVM_test:
    def __init__(self, G, epsilon, alpha, V_inv,
                 C_init, t_init, x_init, y_obs,
                 stepsize_C, stepsize_t, stepsize_x, ld):
        """
        G is the N by N nearest-neighbor graph adjacency matrix
        C_init is the Dy by N*Dt matrix of initial linear maps
        t_init is the Dt by N matrix of initial low-dimensional embeddings
        x_init is the Dy by N matrix of initial true expression
        y_obs is the Dy by N observed expression
        """
        # pre-compute fixed parameters #
        # dims
        self.N = G.shape[0]
        self.Dy = x_init.shape[0]
        self.Dt = t_init.shape[0]

        # constants
        self.alpha, self.epsilon, self.ld = alpha, epsilon, ld
        self.stepsize_C, self.stepsize_t, self.stepsize_x = stepsize_C, stepsize_t, stepsize_x
        self.V_inv = V_inv
        self.J = kron(np.ones(shape=(self.N, 1)), eye(self.Dt))

        # graph laplacian
        self.L = laplacian(G)

        # precision matrices
        self.omega_inv = kron(2*self.L, eye(self.Dt))
        self.sigma_x_inv = kron(np.full((self.N, self.N), self.epsilon), eye(self.Dy)) + kron(2 * self.L, self.V_inv)

        # ToDo: Can we avoid computing this?
        self.sigma_x = chol_inv(self.sigma_x_inv.todense())

        # precision matrices for priors
        self.t_priorprc = self.alpha*eye(self.N) + 2*self.L
        self.C_priorprc = epsilon*self.J.dot(self.J.T) + self.omega_inv

        # create a dictionary of neighbors
        self.neighbors = G.tolil().rows

        #latent and observed variables
        self.C, self.t = C_init, t_init
        self.e = self._e(self.C.reshape(self.Dy, self.N, self.Dt), self.t)
        yflat = y_obs.reshape((self.N * self.Dy, 1), order='F')
        self.x = yflat
        # ToDo: Czech this
        self.dropouts = np.where(yflat==0)[0] #indexing as in the mu_x vector
        self.x[self.dropouts] = x_init.reshape(self.x.shape, order="F")[self.dropouts]

        # final means
        self.C_mean = np.empty_like(C_init)
        self.t_mean = np.empty_like(t_init)
        self.x_mean = np.empty_like(self.x)

        # this needs to be recomputed every time
        self.x_SigX_x = self.x.T.dot(self.sigma_x_inv.dot(self.x))
        self.x_SigX_xprop = self.x_SigX_x

        # counts
        self.num_samples = self.num_samples_tot = self.accept_rate = self.a_current = 0

        self.Cprop, self.tprop, self.eprop, self.xprop = self.C, self.t, self.e, self.x

        # initialize variables to store trace and likelihood
        self.ll_comp_C, self.ll_comp_t, self.ll_comp_x = [], [], []
        self.trace, self.trace_x, self.likelihoods = [], [], [self.likelihood()]

        # constants for simulation
        self.Pi = kron(chol_inv(self.alpha * np.eye(self.N) + 2*self.L) , np.eye(self.Dt))
        self.C_priorcov = chol_inv(self.C_priorprc.todense())
        
    def _e(self, C, t):

        # precompute differences in t
        # NB: t[:,j] - t[:,i] = t_diff[i, :, j]
        t_diff = np.kron(np.ones((self.N, 1)), t).reshape(self.N, self.Dt, self.N)
        t_diff = t_diff - t_diff.transpose()

        # ToDo: Parallelize this loop
        e = np.empty((self.Dy, self.N))
        for i in range(self.N):
            js = self.neighbors[i]
            e[:, i] = np.tensordot(C[:,[i],:] + C[:,js,:], t_diff[[i],:,js])
            # simple_e = np.sum((C[:,i,:] + C[:,j,:]).dot(t_diff[i, :, j]) for j in self.neighbors[i])
            # assert np.allclose(e[:, i], simple_e), "tensordot not matching sum"
        e = - self.V_inv.dot(e)

        return e.flatten(order='F')
    
    def _loglik_x_star(self, e, x, x_SigX_x):
        return float(-.5*(x_SigX_x - 2*x.T.dot(e) + e.T.dot(self.sigma_x.dot(e))))
    
    # calculate likelihood for proposed latent variables
    def likelihood(self):
        
        C, t, e, x, x_SigX_x = self.Cprop, self.tprop, self.eprop, self.xprop, self.x_SigX_xprop

        Cfactor = matrix_normal_log_star_std(C, self.C_priorprc)
        tfactor = matrix_normal_log_star_std(t, self.t_priorprc)
        xfactor = self._loglik_x_star(e, x, x_SigX_x)

        self.ll_comp_C.append(Cfactor)
        self.ll_comp_t.append(tfactor)
        self.ll_comp_x.append(xfactor)

        return Cfactor + tfactor + xfactor
    
    def set_stepsize(self, new_step):
        self.stepsize_C = new_step
    
    # update with Metropolis-Hastings step
    def update(self):
        # calculate likelihood
        Lprime = self.likelihood()
        L = self.likelihoods[-1]
        # calculate acceptance probability
        a = 1.0 if Lprime > L else np.exp(Lprime - L)
        #print(Lprime, L)
        self.a_current = a
        accept = bernoulli.rvs(a)
        
        # update the variables
        if accept:
            self.C = np.copy(self.Cprop)
            self.t = np.copy(self.tprop)
            self.e = np.copy(self.eprop)
            self.x = np.copy(self.xprop)
            self.x_SigX_x = np.copy(self.x_SigX_xprop)
            self.likelihoods.append(Lprime)

        else:
            self.likelihoods.append(L)

        return accept

    # propose a new value based on current values
    def propose(self):
        self.Cprop = self.C + np.random.randn(self.Dy, self.N*self.Dt)*self.stepsize_C
        self.tprop = self.t + np.random.randn(self.Dt, self.N)*self.stepsize_t
        self.eprop = self._e(self.Cprop.reshape(self.Dy, self.N, self.Dt), self.tprop)
        self.xprop = np.copy(self.x)
        self.xprop[self.dropouts] = self.x[self.dropouts] + np.random.randn(len(self.dropouts), 1)*self.stepsize_x
        self.x_SigX_xprop = self.xprop.T.dot(self.sigma_x_inv.dot(self.xprop))
        
    def MH_step(self, burn_in=False):
        self.propose()
        accept = self.update()
        self.trace.append(self.t[0,0])
        self.trace_x.append(self.x[0])
        self.num_samples_tot += 1
        self.accept_rate = ((self.num_samples_tot - 1)*self.accept_rate + accept)/float(self.num_samples_tot)
        if not burn_in:
            self.num_samples += 1
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
        x = np.random.multivariate_normal(mean=np.array(mu_x)[:,0], cov=self.sigma_x).reshape((self.Dy,self.N),order="F")
        #drop_prob = np.exp(self.ld * -1 * np.exp(x))
        drop_prob = np.ones(x.shape) * np.exp(-1 *self.ld)
        h = [np.random.binomial(1,1-prob) for prob in drop_prob]
        y = x * h
        return y, x, t, C