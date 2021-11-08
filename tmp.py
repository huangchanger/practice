import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def gaussian_prob(data,mean,cov):
    dim = np.shape(cov)[0]   
    covdet = np.linalg.det(cov) 
    covinv = np.linalg.inv(cov) 
    if covdet==0:           
        covdet = np.linalg.det(cov+np.eye(dim)*0.01)
        covinv = np.linalg.inv(cov+np.eye(dim)*0.01)
    m = data - mean
    z = -0.5 * np.dot(np.dot(m, covinv),m)  
    return 1.0/(np.power(np.power(2*np.pi,dim)*abs(covdet),0.5))*np.exp(z) 


if __name__ == '__main__':
    N_FEATURES = 2
    X,y = make_blobs(n_samples = 1000, n_features=N_FEATURES, random_state=1,centers=2)
    N,D = X.shape
    K = 2
    means_init = np.sum(X,axis=0) / N
    means = np.zeros((K,D))
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    means= kmeans.cluster_centers_
    convs = [0]*K
    for i in range(K):
        convs[i]=np.cov(X.T)
    plt.figure()
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    gamma = np.ones((N, K))
    pi = [0.2, 0.8]
    L = []
    iteration = 100
    while(iteration > 0):
        #E step
        for i in range(len(X)):
            gamma[i,:] = np.array([pi[k]*gaussian_prob(X[i],means[k],convs[k]) for k in range(K)])
            sum_gamma_i = sum(gamma[i,:])
            gamma[i,:] = gamma[i,:]/sum_gamma_i
        #M step
        for k in range(K):
            Nk = sum(gamma[:, k])
            pi[k] = Nk / N
            means[k] = 1/Nk * np.sum([gamma[i][k]*X[i] for i in range(N)],axis=0)
            xdiffs = X - means[k]
            convs[k] = (1.0/ Nk)*np.sum([gamma[n][k]* xdiffs[n].reshape(D,1) * xdiffs[n] for  n in range(N)],axis=0)
        L.append(np.sum([np.log(np.sum([\
            pi[k]*gaussian_prob(X[i],means[k],convs[k]) for k in range(K)])) for i in range(N)]))
        if len(L) == 1:
            continue
        if abs(L[-1] - L[-2]) < 1e-5:
            break
        iteration -= 1

    print('result: ')
    print(means)
    print(convs)
    print(pi)

