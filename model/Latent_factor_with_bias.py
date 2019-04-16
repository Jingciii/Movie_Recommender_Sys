
"""
Created on Mon April 15 2019
@author: Jingci Wang
"""

"""
This is a class for Latent factor model considering bias terms
"""

from .setting import loadMovieNames

nameDict = loadMovieNames()

class LFM_bias(object):
    ''' This class built a recommender system based on latent factor model with 
    stochastic gradient descent'''
    
    def __init__(self, train, test):
        self.ids = train.index
        self.columns = train.columns
        self.R = train.fillna(0).values
        self.T = test.fillna(0).values
        self.Q = None
        self.P = None
    
    def initialize(self, Q=None, P=None, k=1):
        '''initialize Q, P and latent factor k. The default values for Q and P
        are the results for SVD for R'''
        if (Q is not None) and (P is not None):
            self.Q = Q
            self.P = P
        else:
            u, s, vt = svds(self.R, k=k)
            s_diag=np.diag(s)
            self.Q = u
            self.P = s_diag.dot(vt).T
            
    def rmse_(self, R, Q, P, Bx, Bi, mu):
       # prediction = Q.dot(P.T)
       # prediction = prediction[R.nonzero()].flatten() 
       # true_value = R[R.nonzero()].flatten()
       # return math.sqrt(mean_squared_error(prediction, true_value))
        I = R != 0  # Indicator function which is zero for missing data
        ME = I * (R - (Bx + Bi - mu + np.dot(Q, P.T))) # Errors between real and predicted ratings
        MSE = ME**2  
        return np.sqrt(np.sum(MSE)/np.sum(I))
    
    def train(self, lr1, lr2, reg1, reg2, maxiter):
        '''SGD for training
        @lr1: learning rate for Q   @lr2: learning rate for P
        @reg1: regularizer for Q    @reg2: regularizer for P
        @maxiter: stop when epochs exceed this value
        '''
        
        train_errors = []
        test_errors = []
        users,items = self.R.nonzero() 
        for epoch in range(maxiter):
            # Include Bias term
            self.mu = self.R.mean().mean()
            bx = self.R.mean(axis=1)
            bi = self.R.mean(axis=0)
            self.Bx = np.tile(bx, self.R.shape[1]).reshape(self.R.shape[0], self.R.shape[1])
            self.Bi = np.tile(bi, self.R.shape[0]).reshape(self.R.shape[0], self.R.shape[1])
            dR = 2 * (self.R - (self.Bx + self.Bi - self.mu + self.Q.dot(self.P.T)))
            
            start_time = default_timer()
            for u, i in zip(users, items):
                dq = dR[u, i] * self.P[i, :] - 2 * reg2 * self.Q[u, :]
                self.Q[u, :] += lr1 * dq
                dp = dR[u, i] * self.Q[u, :] - 2 * reg1 * self.P[i, :]
                self.P[i, :] += lr2 * dp
            
            trg_err = self.rmse_(self.R, self.Q, self.P, self.Bx, self.Bi, self.mu)
            print("current training error: " + str(trg_err))
            train_errors = train_errors + [trg_err]
            test_err = self.rmse_(self.T, self.Q, self.P, self.Bx, self.Bi, self.mu)
            print("current test error: " + str(test_err))
            test_errors = test_errors + [test_err]
            print("Run took %.2f seconds for epoch " % (default_timer() - start_time) + str(epoch))
        self.train_errors = train_errors
        self.test_errors = test_errors
        return self.train_errors[-1], self.test_errors[-1]
            
    def trainplot(self):
        
        # Check performance by plotting train and test errors
        fig, ax = plt.subplots()
        ax.plot(self.train_errors, color="b", label='Training RMSE')
        ax.plot(self.test_errors, color="r", label='Test RMSE')
        ax.set_title("Error During Stochastic GD")
        ax.set_xlabel("Number of Epochs")
        ax.set_ylabel("RMSE")
        ax.legend()
    def pred_forall(self):
        self.pred = pd.DataFrame(self.Bx + self.Bi - mu + self.Q.dot(self.P.T), index=self.ids, columns=self.columns)

        
    def recommend(self, userid, k):
        '''Input userid to recommend top k movies that this user might like most'''
        
        rec = list(self.pred.loc[userid, :].argsort()[-k:][::-1].index)
        
        return list(map(lambda x: nameDict[x], rec))