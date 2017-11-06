class LL_LVM:
    def __init__(self,G,epsilon,alpha,V,Cinit,tinit,xinit,stepsize,yobserved=0):
        """
        G is the N by N nearest-neighbor graph adjacency matrix
        Cinit is the Dy by N*Dt matrix of initial linear maps
        tinit is the Dt by N matrix of initial low-dimensional embeddings
        xinit is the Dy by N matrix of initial true expression
        yobserved is the Dy by N observed expression
        """
        #pre-compute fixed parameters
        self.N = G.shape[0]; self.Dy = xinit.shape[0]; self.Dt = tinit.shape[0]
        self.epsilon = epsilon; self.alpha = alpha; self.V = V; self.Vinv = chol_inv(V)
        degree = np.sum(G,1)
        self.L = np.diag(degree) - G
        self.omega_inv = np.kron(2*self.L, np.identity(self.Dt))
        self.J = np.kron(np.ones(shape=(self.N,1)),np.identity(self.Dt))
        self.sigma_x_inv = np.kron(self.epsilon * np.ones(shape=(self.N,1)) * np.ones(shape=(1,self.N)), np.identity(self.Dy)) + np.kron(2*self.L , self.Vinv)
        #self.sigma_x = ln.inv(self.sigma_x_inv)
        
        #add some jitter to condition properly
       # jitter = np.random.chisquare(.5,self.Dy*self.N)
        #self.sigma_x_inv = self.sigma_x_inv + np.diag(jitter)
        self.sigma_x = chol_inv(self.sigma_x_inv)
        
        self.t_priorcov = ln.inv(alpha*np.identity(self.N) + 2*self.L)
        self.C_priorcov = ln.inv(epsilon * self.J * self.J.T + self.omega_inv)
        
        self.Cpropcov = np.identity(self.N * self.Dy * self.Dt) * stepsize
        self.tpropcov = np.identity(self.N * self.Dt) * stepsize
        
        #create a dictionary of neighbors
        self.neighbors = {i:np.where(G[i,:]==1)[0] for i in range(N)}
        
        #initialize latent variables and observations
        self.C = Cinit; self.t = tinit; self.x = xinit #just put observed if standard LL_LVM
        #list of Dy by Dt numpy arrays for each observation's linear map C_i
        self.Ci = [self.C[:,np.arange(i*self.Dt,(i+1)*self.Dt)] for i in range(self.N)]
        if yobserved!=0: #for the noisy LL_LVM model
            self.y = yobserved
        
        #self.e = [-1 * np.sum([self.Vinv * (self.Ci[i] + self.Ci[j])*(self.t[:,i] - self.t[:,j]) for j in self.neighbors[i]]) for i in range(self.N)]
        self.e = np.array([-1 * np.array([self.Vinv * np.matrix((self.Ci[i] + self.Ci[j]))*np.matrix((self.t[:,i] - self.t[:,j])) for j in self.neighbors[i]]).sum(0) for i in range(self.N)]).flatten()
        #initialize variables to store proposed values of each
        self.Cfinal = np.zeros(shape=(self.Dy,self.N*self.Dt))
        #self.Ciprop = [self.Cprop[:,np.arange(i*self.Dt,(i+1)*self.Dt)] for i in range(self.N)]
        self.tfinal = np.zeros(shape = (self.Dt, self.N))
        #self.xprop = xinit #np.zeros(shape = (self.Dy, self.N))
        
        self.acceptance = 0
        
        #initialize variables to store trace and likelihood
        self.trace = []; self.likelihoods = []
        
    #calculate likelihood for proposed latent variables
    def likelihood(self,proposed=False):
        
        #x factor changes and y factor added for noisy version **
        
        #proposed contains all latent variables in one array
        if proposed:
            #calculate likelihood under proposed value
            #Cfactor = np.log(matrix_normal(self.Cprop,np.zeros(shape=(self.Dy, self.N*self.Dt)),np.identity(self.Dy),self.C_priorcov))
            #tfactor = np.log(matrix_normal(self.tprop,np.zeros(self.Dt),np.identity(self.Dt),self.t_priorcov))
            Cfactor = matrix_normal_log_star(self.Cprop,np.zeros(shape=(self.Dy, self.N*self.Dt)),np.identity(self.Dy),self.C_priorcov)
            tfactor = matrix_normal_log_star(self.tprop,np.zeros(self.Dt),np.identity(self.Dt),self.t_priorcov)
            mu_x = np.matrix(self.sigma_x) * self.eprop.reshape((self.N*self.Dy,1))
            xfactor = scipy.stats.multivariate_normal.logpdf(self.x.reshape((self.N*self.Dy,1)).T,mean= list(mu_x.flat),cov=self.sigma_x)
            #print Cfactor, tfactor, xfactor
            return Cfactor + tfactor + xfactor
            
        else:
            #if proposed is false just calculate it under the current variables
            #Cfactor = np.log(matrix_normal(self.C,np.zeros(shape=(self.Dy, self.N*self.Dt)),np.identity(self.Dy),self.C_priorcov))
            #tfactor = np.log(matrix_normal(self.t,np.zeros(self.Dt),np.identity(self.Dt),self.t_priorcov))
            Cfactor = matrix_normal_log_star(self.C,np.zeros(shape=(self.Dy, self.N*self.Dt)),np.identity(self.Dy),self.C_priorcov)
            tfactor = matrix_normal_log_star(self.t,np.zeros(self.Dt),np.identity(self.Dt),self.t_priorcov)
            mu_x = np.matrix(self.sigma_x) * self.e.reshape((self.N*self.Dy,1))
            xfactor = scipy.stats.multivariate_normal.logpdf(self.x.reshape((self.N*self.Dy,1)).T,mean= list(mu_x.flat),cov=self.sigma_x)
            #print Cfactor, tfactor, xfactor
            return Cfactor + tfactor + xfactor
    
    #update with Metropolis-Hastings step
    def update(self):
        #calculate likelihoods
        Lprime = self.likelihood(proposed=True)
        L = self.likelihood()
        #calculate acceptance probability
        a = min(1.0,np.exp(Lprime - L))
        acceptance = np.random.choice([0,1],p=(1-a,a))
        
        #update the variables
        if acceptance:
            self.C = np.copy(self.Cprop)
            self.t = np.copy(self.tprop)
            self.Ci = [self.C[:,np.arange(i*self.Dt,(i+1)*self.Dt)] for i in range(self.N)]
            self.e = np.array([-1 * np.array([self.Vinv * np.matrix((self.Ci[i] + self.Ci[j]))*np.matrix((self.t[:,i] - self.t[:,j])) for j in self.neighbors[i]]).sum(0) for i in range(self.N)]).flatten()
            self.acceptance+=1
            #add acceptance for x here for noisy version**
    
    #propose a new value based on current values
    def propose(self):
        #drawn from MVN centered at C
        self.Cprop = np.random.multivariate_normal(self.C.reshape((self.Dy*self.Dt*self.N)),self.Cpropcov).reshape((self.Dy,self.Dt*self.N))
        self.Ciprop = [self.Cprop[:,np.arange(i*self.Dt,(i+1)*self.Dt)] for i in range(self.N)]
        #drawn from MVN centered at t
        self.tprop = np.random.multivariate_normal(self.t.reshape((self.Dt*self.N)),self.tpropcov).reshape((self.Dt,self.N))
        self.eprop = np.array([-1 * np.array([self.Vinv * np.matrix((self.Ciprop[i] + self.Ciprop[j]))*np.matrix((self.tprop[:,i] - self.tprop[:,j])) for j in self.neighbors[i]]).sum(0) for i in range(self.N)]).flatten()

        #add proposal for x here for noisy version**

    def MH_step(self,burn_in=False):
        #propose
        self.propose()
        #update
        self.update()
        #store new likelihood
        self.likelihoods.append(self.likelihood())
        if not burn_in:
            self.Cfinal = self.Cfinal + self.C
            self.tfinal = self.tfinal + self.t

