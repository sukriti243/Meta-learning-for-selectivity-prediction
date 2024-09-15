import torch
import gpytorch
import torch.nn as nn

class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, kernel, ard_num_dims=None, use_lengthscale_prior=True):
        #Set the likelihood noise and enable/disable learning
        # likelihood.noise_covar.raw_noise.requires_grad = True
        # likelihood.noise_covar.noise = 0.1
       
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        ## Linear kernel
        if kernel=='linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif kernel=='rbf' or kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        ## Matern kernel (52)
        elif kernel=='matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims))
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

