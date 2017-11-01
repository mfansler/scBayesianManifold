class LL_LVM:
    def __init__(self,G,epsilon,alpha,V,Cinit,tinit,xinit,yobserved=0):
        """
        G is the N by N nearest-neighbor graph adjacency matrix
        Cinit is the Dy by N*Dt matrix of initial linear maps
        tinit is the Dt by N matrix of initial low-dimensional embeddings
        xinit is the Dy by N matrix of initial true expression
        yobserved is the Dy by N observed expression
        """
        #pre-compute fixed parameters
        self.N = G.shape[0]; self.Dy = xinit.shape[0]; self.Dt = tinit.shape[0]
        self.epsilon = epsilon; self.alpha = alpha; self.V = V; self.Vinv = ln.inv(V)
        self.L = np.diag(G * np.ones(shape=(N,1))) - G
        self.omega_inv = np.kron(2*L, np.identity(Dt))
        self.J = np.kron(np.ones(shape=(N,1)),np.identity(Dt))
        
        #create a dictionary of neighbors
        self.neighbors = {i:np.where(G[i,:]==1)[0] for i in range(N)}
        
        #initialize latent variables and observations
        self.C = Cinit; self.t = tinit; self.x = xinit
        #list of Dy by Dt numpy arrays for each observation's linear map C_i
        self.Ci = [C[:,np.arange(i*Dt,(i+1)*Dt)] for i in range(N)]
        if yinit!=0:
            self.y = yobserved
        
        self.e = [-1 * np.sum([Vinv * (self.Ci[i] + self.Ci[j])*(self.t[:,i] - self.t[:,j]) for j in neighbors[i]]) for i in range(N)]
    
    #calculate likelihood for proposed latent variables
    def likelihood(self,proposed=False):
        #proposed contains all latent variables in one array
        if proposed:
            #calculate likelihood under proposed value
            
        else:
            #if proposed is empty just calculate it under the current variables
            
    
    #update with Metropolis-Hastings step
    def update(self,proposed):
        #calculate likelihoods
        
        #calculate acceptance probability
        
        #update the variables
        
        #update the precomputed values 
        self.e = [-1 * np.sum([Vinv * (self.Ci[i] + self.Ci[j])*(self.t[:,i] - self.t[:,j]) \ 
        for j in neighbors[i]]) for i in range(N)]
    
    #propose a new value    
    def propose(self):
        
    
    def MH_step(self):
        #propose
        #update
    

