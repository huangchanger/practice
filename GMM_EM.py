import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def gaussian_prob(x, mu, sigma):
	return 1 / (math.sqrt(2*3.14159)*sigma) * math.exp(-(x-mu)**2 / (2*sigma**2))

if __name__ == '__main__':
    N_FEATURES = 1
    X,y = make_blobs(n_samples = 1000, n_features=N_FEATURES, random_state=1,centers=2)
    N,D = X.shape
    K = 2
    sigma = [1]*K
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    mu= kmeans.cluster_centers_[:,0]
    gamma = np.ones((len(X), len(mu)))
    X = X[:,0]
    pi = np.ones(K)/K
    K = len(mu)
    L = []
    plt.figure()
    plt.hist(X,bins=np.linspace(np.min(X), np.max(X), num=10))
    plt.show()
    while(True):
        #E step
        for i in range(len(X)):
            gamma[i][:] = np.array(\
                [pi[k]*gaussian_prob(X[i],mu[k],sigma[k]) for k in range(K)])
            sum_gamma_i = sum(gamma[i][:])
            gamma[i][:] = gamma[i]/sum_gamma_i
        #M step
        for k in range(K):
            Nk = sum(gamma[:][k])
            pi[k] = Nk / N
            mu[k] = 1/Nk * sum([gamma[i][k]*X[i] for i in range(N)])
            sigma[k] = 1/Nk * sum([gamma[i][k]*(X[i]-mu[k])**2 for i in range(N)])
        L.append(np.sum([np.log(np.sum([\
            pi[k]*gaussian_prob(X[i],mu[k],sigma[k]) for k in range(K)])) for i in range(N)]))
        if len(L) <= 1:
            continue
        if math.fabs(L[-1]-L[-2] < 1e-5):
            break
        print(L[-1])
    print(pi)
    print(mu)
    print(sigma)
