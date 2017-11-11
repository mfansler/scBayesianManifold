from .utils import chol_inv, matrix_normal_log_star, matrix_normal_log_star2
import numpy as np
import numpy.linalg as ln
from scipy.stats import bernoulli
from scipy.sparse.csgraph import laplacian
from scipy.sparse import kron, eye
from scipy.sparse.linalg import inv as sp_inv
from scipy.linalg import inv


class LL_LVM:
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

        self.Cpropcov = eye(self.N * self.Dy * self.Dt) * stepsize
        self.tpropcov = eye(self.N * self.Dt) * stepsize
        
        # create a dictionary of neighbors
        self.neighbors = G.tolil().rows

        # initialize latent variables and observations
        # just put observed if standard LL_LVM
        self.C, self.t, self.x = Cinit, tinit, xinit

        # list of Dy by Dt numpy arrays for each observation's linear map C_i
        if yobserved != 0:  # for the noisy LL_LVM model
            self.y = yobserved
        
        self.e = self._e(self.C.reshape(self.Dy, self.N, self.Dt), self.t)

        # final means
        self.C_mean = Cinit
        self.t_mean = tinit

        # cache (x * Sig_x * x)
        self.x_SigX_x = self.x.reshape((1, self.N*self.Dy)).dot(self.sigma_x_inv.dot(self.x.reshape((self.N*self.Dy, 1))))

        # counts
        self.num_samples, self.accept_rate = 0, 0

        self.Cprop, self.tprop, self.eprop = self.C, self.t, self.e

        # initialize variables to store trace and likelihood
        self.trace, self.likelihoods = [], [self.likelihood()]

    def _e(self, C, t):
        return np.array([-1 * np.array([self.Vinv.dot(C[:,i,:] + C[:,j,:]).dot(t[:,i] - t[:,j]) for j in self.neighbors[i]]).sum(0) for i in range(self.N)]).flatten()

    def _loglik_x_star(self, e):
        return -0.5*(self.x_SigX_x - 2*self.x.reshape((1, self.N*self.Dy)).dot(e)
                     + e.T.dot(self.sigma_x.dot(e)))[0]

    # calculate likelihood for proposed latent variables
    def likelihood(self):
        
        C, t, e = self.Cprop, self.tprop, self.eprop

        Cfactor = matrix_normal_log_star2(C, np.zeros(shape=(self.Dy, self.N*self.Dt)),
                                          eye(self.Dy), self.C_priorprc)
        tfactor = matrix_normal_log_star2(t, np.zeros(self.Dt), eye(self.Dt), self.t_priorprc)
        xfactor = self._loglik_x_star(e.reshape((self.N * self.Dy, 1)))

        # print(Cfactor, tfactor, xfactor)
        return Cfactor + tfactor + xfactor
    
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
    def propose(self):
        self.Cprop = self.C + np.random.randn(self.Dy, self.N*self.Dt)*self.stepsize
        self.tprop = self.t + np.random.randn(self.Dt, self.N)*self.stepsize
        self.eprop = self._e(self.Cprop.reshape(self.Dy, self.N, self.Dt), self.tprop)

        # ToDo: add proposal for x here for noisy version

    def MH_step(self, burn_in=False):
        self.propose()
        accept = self.update()
        if not burn_in:
            self.num_samples += 1
            self.accept_rate = ((self.num_samples - 1)*self.accept_rate + accept)/self.num_samples
            self.C_mean = ((self.num_samples - 1)*self.C_mean + self.C)/self.num_samples
            self.t_mean = ((self.num_samples - 1)*self.t_mean + self.t)/self.num_samples
