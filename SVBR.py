import numpy as np
from numpy.linalg import norm as norm

class SVBR:
    """
    Main class
    """
    def __init__(self, random_state=None):
        self.mu = None
        self.m = None
        self.C = None
        
        self._rnd_state = random_state
        
    def __algorithm_21(self, X, Y, gamma, lam, delta=1e-7, T=500, eps=1e-4, verbose=True):
        
        np.random.seed(self._rnd_state)
        
        n, p = X.shape
        mu = np.zeros(p)
        m = np.zeros(p)
        C = 1e+4*np.eye(p)

        mu_old = np.random.rand(p)
        m_old = np.random.rand(p)

        t = 0  

        while t <= T and np.max([norm(mu-mu_old),norm(m-m_old)]) > eps:
            m_old = m
            mu_old = mu

            D_vbem = np.diag(np.sqrt(np.diag(C) + m**2 + delta**2))/lam
            D_em = np.diag(np.sqrt(mu**2 + delta**2))/lam

            K_vbem = D_vbem@X.T@np.linalg.inv(X@D_vbem@X.T + gamma**2*np.eye(n))
            K_em = D_em@X.T@np.linalg.inv(X@D_em@X.T + gamma**2*np.eye(n))

            m = K_vbem@Y
            C = (np.eye(p) - K_vbem@X)@D_vbem
            mu = K_em@Y                        

            t += 1
        if verbose: print("Num of iterations is {}".format(t-1))
        self.mu, self.m, self.C = mu, m, C
     
    def __algorithm_31(self, X, Y, M, gamma, lam, delta=1e-7, T=500, eps=1e-4, verbose=True):   
        np.random.seed(self._rnd_state)
        
        n, p = X.shape

        self.__algorithm_21(X[:M,:], Y[:M], gamma, lam, delta, T, eps, verbose=verbose)

        mu, m, C = self.mu.copy(), self.m.copy(), self.C.copy()
        mu_star = mu
        m_star = m

        X_prev = X[:M,:]
        for i in range(1,int(n/M)):
            Y_vbem = np.hstack((X_prev@m_star,Y[i*M:(i+1)*M]))
            Y_em = np.hstack((X_prev@mu_star,Y[i*M:(i+1)*M]))
            X_st = np.vstack((X_prev,X[i*M:(i+1)*M,:]))

            t = 0  

            mu_old = np.random.rand(p)
            m_old = np.random.rand(p)

            while t <= T and np.max([norm(mu-mu_old),norm(m-m_old)]) > eps:
                m_old = m
                mu_old = mu

                D_vbem = np.diag(np.sqrt(np.diag(C) + m**2 + delta**2))/lam
                D_em = np.diag(np.sqrt(mu**2 + delta**2))/lam

                K_vbem = D_vbem@X_st.T@np.linalg.inv(X_st@D_vbem@X_st.T + gamma**2*np.eye(2*M))
                K_em = D_em@X_st.T@np.linalg.inv(X_st@D_em@X_st.T + gamma**2*np.eye(2*M))

                m = m_star + K_vbem@(Y_vbem - X_st@m_star)
                C = (np.eye(p) - K_vbem@X_st)@D_vbem
                mu = mu_star + K_em@(Y_em - X_st@mu_star)                        

                t += 1  
            mu_star = mu.copy()
            m_star = m.copy()
            SIGMA,U = np.linalg.eig(X_st@X_st.T)
            ind = np.argsort(-SIGMA)[:M]
            X_prev = U[:,ind].T@X_st
            if verbose: print("Num of iterations per batch {} is {}".format(i+1, t-1))
        self.mu, self.m, self.C = np.real(mu_star), np.real(m_star), np.real(C)
            
    def fit(self, X, Y, gamma, lam, mode='monolithic', M=10, delta=1e-7, T=500, eps=1e-4, verbose=True):
        """Variational Bayesian LASSO

        Parameters
        ----------
        X : array
            Design matrix
        Y : array
            Labels
        M : int
            Batch size
        gamma : float
            Noise in the data 
        lam : float
            Hyperparameter that governs the sparsity of the solution
        mode: str
            Mode of operation
        delta : 
            Hyperparameter
        T : int
            Maximum number of iterations
        eps : float
            Convergence criterion
        verbose : bool, optional
            A flag used to print the number of iterations to the console (default is
            True)

        Returns
        -------
        mu, m, C : array
            MAP estimate, mean, covariance

        References
        ----------
        1. https://arxiv.org/abs/2102.12261    
        """ 
        assert mode in ['monolithic', 'online', 'approximate_online'], "mode should take 'monolithic', 'online', or 'approximate_online'"
        
        if mode=='monolithic':
            self.__algorithm_21(X=X, Y=Y, gamma=gamma, lam=lam, delta=delta, T=T, eps=eps, verbose=verbose)
        if mode=='approximate_online':
            self.__algorithm_31(X=X, Y=Y, M=M, gamma=gamma, lam=lam, delta=delta, T=T, eps=eps, verbose=verbose)        
    
    def predict(self, X):
        return X@self.mu, X@self.m