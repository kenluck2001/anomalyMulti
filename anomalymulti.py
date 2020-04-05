import numpy as np
import math
from numpy.linalg import multi_dot
from scipy.stats import multivariate_normal
from utils import nearestPD



class probabilisticMultiEWMA:
    A_t = None
    C_t = None
    invC = None
    mean = None
    n = 0
    alpha, beta = 0, 0

    def __getCov(self, X):
        """
        @param: X is dimension (n x m) where n: number of samples, m: dimension of the sample
        @return: C is the covariance matrix
        """
        C = np.matmul(X.T, X)
        return C
	
    def __inverseCov(self, X):
        """
        @param: X is dimension (n x m) where n: number of samples, m: dimension of the sample
        @return: inverse covariance matrix
        """
        C = np.matmul(X.T, X)
        return np.linalg.inv(C)

    def __getCholeskyFactor(self, C):
        """
        @param: C is covariance matrix
        @return: A is the cholesky factor
        """
        A = np.linalg.cholesky(C)
        return A

    def __estimateInitParameters(self, n):
        """
        @param: n, number of samples in static setting
        @return: (alpha, beta) is a tuple of alpha, beta
        """
        C_cov = 2.0 / (math.pow(n, 2) + 6)
        alpha = 1 - C_cov
        beta = 1 - alpha
        return alpha, beta


    def __updateCovariance(self, alpha, beta, C_t, A_t, z_t):
        """
        @param: alpha, beta, A_t are parameters of the model
        @param: C_t is old covariance matrix, z_t as new data vector
        @return: C_tplus1 is updated covariance matrix
        """
        v_t = np.dot(A_t, z_t.T)
        C_tplus1 = (alpha * C_t)  + (beta * np.matmul(v_t, v_t.T)) 
        return C_tplus1
	
    def __updateCholeskyFactor(self, alpha, beta, A_t, z_t):
        """
        @param: alpha, beta, A_t are parameters of the model
        @param: z_t as new data vector
        @return: A_tplus1 is updated covariance matrix
        """
        v_t = np.dot(A_t, z_t.T)
        norm_z = np.linalg.norm(z_t)
        x = math.sqrt(alpha) * A_t
        w = beta * norm_z / alpha
        y = math.sqrt(alpha) * (math.sqrt(1 + w) - 1) * np.dot(v_t, z_t) / norm_z
        A_tplus1 = x + y
        return A_tplus1
	
    def __updateInverseCovariance(self, alpha, beta, invC_t, A_t, z_t):
        """
        @param: alpha, beta, A_t are parameters of the model
        @param: invC_t is old inverse covariance matrix, z_t as new data vector
        @return: invC_tplus1 is updated inverse covariance matrix
        """
        v_t = np.dot(A_t, z_t.T)
        hat_vt = (beta * v_t) / alpha
        y = multi_dot([invC_t, hat_vt, v_t.T, invC_t]) / (1 + multi_dot([hat_vt.T, invC_t, v_t]))
        invC_tplus1 = (invC_t - y) / alpha
        return invC_tplus1

    def __updateMean(self, mean, x):
        mean_tplus1 = ((self.n * mean) + x) / (self.n + 1)
        self.n = self.n + 1
        return mean_tplus1

    def init (self, X):
        self.n = len(X)
        self.C_t = self.__getCov(X)  
        self.C_t = nearestPD(self.C_t)
        self.invC = self.__inverseCov(X) 
        self.A_t = self.__getCholeskyFactor(self.C_t) 
        self.alpha, self.beta = self.__estimateInitParameters(self.n)  
        self.mean = np.mean(X, axis=0).ravel()

    def update(self, z_t):  
        self.C_t = self.__updateCovariance(self.alpha, self.beta, self.C_t, self.A_t, z_t)
        self.A_t = self.__updateCholeskyFactor(self.alpha, self.beta, self.A_t, z_t)
        self.invC = self.__updateInverseCovariance(self.alpha, self.beta, self.invC, self.A_t, z_t)

    def predict(self, x, threshold=0.001):
        """
        @param: x is the current data vector
        @param: mean is mean vector
        @param: cov is covariance matrix
        @return: score of anomaly
	    """
        score = multivariate_normal.pdf(x, mean=self.mean.ravel(), cov=self.C_t)
        self.mean = self.__updateMean(self.mean, x) # update mean vector
        return score

    def bulkPredict(self, Z):
        """
        @param: Z is the list of data vector
        @param: mean is mean vector
        @param: cov is covariance matrix
        @return: score of anomaly
	    """
        scores = []
        n, m = Z.shape

        for ind in range(1, n):
            cur = Z[ind-1].reshape((1,m))
            next = Z[ind].reshape((1,m))
            self.update(cur)
            score = self.predict(next)
            scores.append(score)
        return scores

    def getCurrentCovariance(self):
        """
        @return: covariance matrix
	    """
        return self.C_t

    def getCurrentInvCovariance(self):
        """
        @return: inverse covariance matrix
	    """
        return self.invC

    def getOriginalCovariance(self, X):
        """
        @return: original covariance matrix
	    """
        return self.__getCov( X)

    def getOriginalInvCovariance(self, X):
        """
        @return: original inverse covariance matrix
	    """
        return self.__inverseCov(X)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    X = np.random.rand(1000,15)
    z_t0 = np.random.rand(1,15) # new data
    anom = probabilisticMultiEWMA()
    anom.init(X)
    z_t0 = np.random.rand(1,15) # new data
    anom.update(z_t0)
    z_t1 = np.random.rand(1,15) # next new data
    print ("score: {}".format(anom.predict(z_t1)))
    Z = np.random.rand(1000,15)

    anom = probabilisticMultiEWMA()
    anom.init(X)
    pred = anom.bulkPredict(Z)
    print ("++++++++++++++++++++++")
    print (pred)
    print ("++++++++++++++++++++++")
