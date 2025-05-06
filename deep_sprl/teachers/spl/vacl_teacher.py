
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as norm_dist

class SVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A

    def dlnprob(self, theta):
        return -1 * np.matmul(theta - np.tile(self.mu, (theta.shape[0], 1)), self.A)
    

class VACLTeacher():
    def __init__(self, target_mean, target_variance, initial_mean, initial_variance, bounds, num_particles) -> None:
        self.lower=bounds[0]
        self.upper=bounds[1]
        self.n_particles=num_particles
        self.teacher_name='vacl'
        self.target=target_mean
        self.sigma=np.sqrt(target_variance)*40
        self.theta = np.random.normal(initial_mean, np.sqrt(initial_variance), [self.n_particles, 1])
        self.count=0
        self.value_buffer=[]
        self.context_buffer=[]
        self.A=1/(target_variance*40)
        self.model=SVN(mu=target_mean, A=self.A) # A is the inverse of variance
        self.fudge_factor = 1e-6
        self.historical_grad = 0
        self.iter=0
        
    def dlnprob(self, theta):
        # return -1 * np.matmul(theta - np.tile(self.model.mu, (theta.shape[0], 1)), self.model.A)*norm_dist.pdf(theta, self.model.mu, self.sigma)
        return -1 * np.matmul(theta - np.tile(self.model.mu, (theta.shape[0], 1)), self.model.A)
    
    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            # h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)
    
    def update(self, context_buffer, value_buffer, stepsize=0.1, alpha=0.9):
        if len(value_buffer)!=self.n_particles:
            context_buffer=context_buffer[:self.n_particles]
            value_buffer=value_buffer[:self.n_particles]
            # raise Exception("value buffer is not equal to the number of particles")
        # self.theta=context_buffer+np.random.rand(*context_buffer.shape)
        self.theta=context_buffer+np.random.normal(size=context_buffer.shape)
        values=np.array(value_buffer).squeeze()
        norm = np.linalg.norm(values)
        values = values / (norm+1e-6)

        # adagrad with momentum
        lnpgrad = self.dlnprob(self.theta)
        # calculating the kernel matrix
        kxy, dxkxy = self.svgd_kernel(self.theta, h=-1)
        grad_theta = ((np.matmul(kxy, lnpgrad) + dxkxy) / self.theta.shape[0])*values.reshape(*self.theta.shape)

        # adagrad
        if self.iter == 0:
            self.historical_grad = self.historical_grad + grad_theta ** 2
        else:
            self.historical_grad = alpha * self.historical_grad + (1 - alpha) * (grad_theta ** 2)
        if np.any(np.isnan(self.historical_grad)):
            raise ValueError('There is NAN')
        adj_grad = np.divide(grad_theta, self.fudge_factor + np.sqrt(self.historical_grad))
        # self.theta = self.theta + stepsize * adj_grad*np.expand_dims(values, 1)
        self.theta = self.theta + stepsize * adj_grad
        self.iter+=1
        
    def sample(self, size=1):
        # return np.random.uniform(low=self.lower, high=self.upper, size=size)
        samples=np.random.choice(self.theta.squeeze(), size=size, replace=False)
        samples=samples.reshape((size, 1))
        # print('sample is:',samples)
        return np.clip(samples, self.lower, self.upper)
    
    def update_distribution(self, avg_performance, contexts, values):
        self.update(contexts, values)