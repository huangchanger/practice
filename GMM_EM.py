import numpy as np


def gaussian_prob(x, mu, sigma):
	return 1 / (sqrt(2*3.14159)*sigma) * exp(-(x-mu)**2 / (2*sigma**2))

if __name__ == '__main__':
    mu = [1,2,3]
    sigma = [1,1,1]
    X = [1,2,3]
    N = len(X)
    gamma = np.ones((len(X), len(mu)))
    pi = [0.2,0.3,0.5]
    num_k = len(mu)
	#E step
    for i in range(len(X)):
        gamma[i][:] = np.array(\
            [pi[k]*gaussian_prob(X[i],mu[k],sigma[k]) for k in range(num_k)])
        sum_gamma_i = sum(gamma[i][:])
        gamma[i][:] = gamma[i]/sum_gamma_i
    #M step
    for k in range(num_k):
        Nk = sum(gamma[:][k])
        pi[k] = Nk / N
        mu[k] = 1/Nk * sum([gamma[i][k]*X[i] for i in range(N)])
        sigma[k] = 1/Nk * sum([gamma[i][k]*(X[i]-mu[k])**2 for i in range(N)])
    L = np.sum([np.log(np.sum([\
        pi[k]*gaussian_prob(X[i],mu[k],sigma[k]) for k in range(num_k)])) for i in range(N)])
    print(L)

