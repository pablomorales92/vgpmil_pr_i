from __future__ import print_function
import numpy as np
import cv2
from time import time
from scipy.stats import norm
from scipy.stats import multivariate_normal
import math
import random
import os

def probit_func(x):
    return norm.cdf(x)

def get_prob_all_less_zero(mean, var, num_samples=5000):
    """
    For a multivariate Gaussian with 'mean' and 'var', it computes the probability that all the components are less than zero, using MC estimation.
    Used to make predictions, see eq. (26) in arXiv paper. 
    """
    samples = np.random.multivariate_normal(mean=mean, cov=var, size=num_samples) # (num_samples, D)
    return np.mean(np.all(samples<0, axis=1))

def E_minf0(mu, var):
    """
    Computes the expectation of a 1D Gaussian truncated to (-\infty, 0). 
    This is a known closed-form expression (see e.g. Wikipedia and references therein).
    When mu and var are numpy arrays, the result is obtained element-wise.
    """
    std = np.sqrt(var)
    return mu - std*(norm.pdf(mu/std)/(1-probit_func(mu/std)))

def compute_norm_constant(Bags, mu_M, var):
    norm_constant = np.zeros_like(Bags, dtype=float)
    for bag_name in np.unique(Bags):
        idxs = (Bags==bag_name)
        norm_constant[idxs] = 1 - np.prod(1-probit_func(mu_M[idxs]/np.sqrt(var[idxs])))
    return norm_constant

class vgpmil_probit_smooth(object):
    def __init__(self, kernel, num_inducing=50, max_iter=10, normalize=True, verbose=False, mu_m_0=None, sigma_m_0_inv=None, batch_size=None, prior_weight=1, save_folder=None):
        """
        :param kernel: Specify the kernel to be used
        :param num_inducing: nr of inducing points
        :param max_iter: maximum number of iterations
        :param normalize: normalizes the data before training
        :param verbose: regulate verbosity
        :param mu_m_0: the mu of the smoothness distribution p(m). By default it is 0. Shape: (N,)
        :param sigma_m_0_inv: the (inv) sigma of the smoothness distribution p(m). By default it is close to zero (i.e. uniform prior). Shape (N,N).
                        It will be zero if the instances do not come from the same bag or if, coming from the same bag, there is no smoothness imposed.
                        It can be obtained by iterating over the samples (patches) and looking at the adjacent patches (not block-diagonal per se).
        :param prior_weight: the weight to be applied to the smoothness distribution p(m). A small/large value (i.e. 1e-4/1e4) means that the prior has a small/big influence.
        """
        self.kernel = kernel
        self.num_ind = num_inducing
        self.max_iter = max_iter
        self.normalize = normalize
        self.verbose = verbose
        self.mu_m_0 = mu_m_0
        self.sigma_m_0_inv = sigma_m_0_inv
        self.batch_size = batch_size
        self.prior_weight = prior_weight
        self.save_folder = save_folder

    def initialize(self, Xtrain, InstBagLabel, Bags, Z=None):
        """
        Initialize the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:  n-dim vector with the bag label of each instance
        :param Bags: n-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        """

        # Initialize mu_m_0 and sigma_m_0_inv
        if self.mu_m_0 is None:
            self.mu_m_0 = np.zeros(Xtrain.shape[0])
        if self.sigma_m_0_inv is None:
            sigma_m_0_inv = np.zeros((Xtrain.shape[0], Xtrain.shape[0]))


        # Normalize
        if self.normalize:
            self.data_mean, self.data_std = np.mean(Xtrain, 0), np.std(Xtrain, 0)
            self.data_std[self.data_std == 0] = 1.0
            Xtrain = (Xtrain - self.data_mean) / self.data_std

        # Compute inducing points if not provided
        if Z is not None:
            assert self.num_ind == Z.shape[0]
            self.Z = Z
        else:
            Xzeros = Xtrain[InstBagLabel == 0].astype("float32")
            Xones = Xtrain[InstBagLabel == 1].astype("float32")
            num_ind_pos = np.uint32(np.floor(self.num_ind * 0.5))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            nr_attempts = 10
            _, _, Z0 = cv2.kmeans(Xzeros, self.num_ind - num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
            _, _, Z1 = cv2.kmeans(Xones, num_ind_pos, None, criteria, attempts=nr_attempts, flags=cv2.KMEANS_RANDOM_CENTERS)
            self.Z = np.concatenate((Z0, Z1))
            if self.verbose:
                print("Inducing points are computed")

        # Computing useful kernel matrices
        self.Kzz = self.kernel.compute(self.Z)
        self.Kzzinv = np.linalg.inv(self.Kzz + np.identity(self.num_ind) * 1e-6)
        Kzx = self.kernel.compute(self.Z, Xtrain)
        self.KzziKzx = np.dot(self.Kzzinv, Kzx)

        # Computing useful matrices for each bag
        self.bag_idxs = {}
        self.mu_tilde_concat = np.zeros(Xtrain.shape[0])
        self.sigma_tilde = {}
        self.mu_tilde = {}
        self.bag_names_unique = np.unique(Bags).tolist()
        
        for name in self.bag_names_unique:
            self.bag_idxs[name] = np.where(Bags==name)[0]
            bag_size = len(self.bag_idxs[name])
            if not type(self.sigma_m_0_inv) is dict:
                sigma_m_0_inv_local = self.prior_weight*self.sigma_m_0_inv[np.ix_(self.bag_idxs[name], self.bag_idxs[name])].astype("float")
            else:
                sigma_m_0_inv_local = self.prior_weight*self.sigma_m_0_inv[name].astype("float")
            lowest_ev = np.linalg.eigvals(sigma_m_0_inv_local).min()
            if lowest_ev < -1e-10:
                print(name, lowest_ev)
            mu_m_0_local = self.mu_m_0[self.bag_idxs[name]]
            self.sigma_tilde[name] = np.linalg.inv( sigma_m_0_inv_local + np.eye(bag_size) )
            self.mu_tilde[name] = self.sigma_tilde[name] @ (sigma_m_0_inv_local @ mu_m_0_local)
            self.mu_tilde_concat[self.bag_idxs[name]] = self.mu_tilde[name]

        # The parameters for q(u)
        self.m = np.random.randn(self.num_ind)
        self.S = np.linalg.inv(self.Kzzinv + 
                                np.sum([self.KzziKzx[:,self.bag_idxs[name]] @
                                        self.sigma_tilde[name] @ 
                                        self.KzziKzx[:,self.bag_idxs[name]].T for name in self.bag_names_unique],axis=0)) # Notice that it is fixed during training (S=sigma_u)

        # The parameters for q(m)
        self.E_m = np.random.randn(Xtrain.shape[0])

    def train(self, Xtrain, InstBagLabel, Bags, Z=None, init=True, num_samples=5000):
        """
        Train the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:  n-dim vector with the bag label of each instance
        :param Bags: n-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        :param init: (opt) whether to initialize before training
        :param num_samples: (opt) number of samples to calculate the mean of the truncated Gaussian 
        """
        if init:
            start = time()
            self.initialize(Xtrain, InstBagLabel, Bags, Z=Z)
            stop = time()
            if self.verbose:
                print("Initialized. \tMinutes needed:\t", (stop - start) / 60.)

        mu_M, var = np.zeros_like(self.E_m), np.zeros_like(self.E_m)
        for it in range(self.max_iter):

            # self.print_m()

            start = time()
            if self.verbose:
                print("Iter %i/%i" % (it + 1, self.max_iter))
            
            ### Updating q(u)
            self.m = self.S @ (self.KzziKzx @ (self.E_m - self.mu_tilde_concat))

            ### Updating q(m) (We may not need to split the block diagonal matrix in submatrices)
            for name in self.bag_names_unique:
                mu_M[self.bag_idxs[name]] = self.sigma_tilde[name] @ \
                                            (self.KzziKzx[:,self.bag_idxs[name]].T @ self.m)
                var[self.bag_idxs[name]] = np.diag(self.sigma_tilde[name])

            mask_bag_label_0 = (InstBagLabel == 0)
            self.E_m[mask_bag_label_0] = E_minf0(mu=mu_M[mask_bag_label_0], var=var[mask_bag_label_0])
            mask_bag_label_1 = (InstBagLabel == 1)
            Z = compute_norm_constant(Bags[mask_bag_label_1], mu_M[mask_bag_label_1], var[mask_bag_label_1]) # Shape ( np.sum(mask_bag_label_1) )  (i.e. the same as any array sliced using mask_bag_label_1, e.g. the input arrays)
            self.E_m[mask_bag_label_1] = (mu_M[mask_bag_label_1] - (1-Z)*E_minf0(mu=mu_M[mask_bag_label_1], var=var[mask_bag_label_1]))/Z
                    
            stop = time()
            if self.verbose:
                print("Minutes needed: ", (stop - start) / 60.)
        # self.print_m()

    def predict(self, Xtest, bag_names_per_instance, sigma_m_0_inv=None, mu_m_0=None):
        """
        Predict instances
        :param Xtest: mxd matrix of n instances with d features
        :return: Instance Predictions
        """
        if mu_m_0 is None:
            mu_m_0 = np.zeros(Xtest.shape[0])
        if sigma_m_0_inv is None:
            sigma_m_0_inv = np.zeros((Xtest.shape[0], Xtest.shape[0]))

        if self.normalize:
            Xtest = (Xtest - self.data_mean) / self.data_std

        bag_names = np.unique(bag_names_per_instance)
        instance_probabilities = np.zeros(bag_names_per_instance.shape)
        mu_f_ast_all = np.zeros(bag_names_per_instance.shape)
        mu_m_ast_all = np.zeros(bag_names_per_instance.shape)
        bag_probabilities = []

        # for bag_name in bag_names:
        #     bag_mask = (bag_names_per_instance == bag_name)
        #     sigma_m_0_inv_local = self.prior_weight*sigma_m_0_inv[np.ix_(bag_mask,bag_mask)].astype("float")
        #     lowest_ev = np.linalg.eigvals(sigma_m_0_inv_local).min()
        #     if lowest_ev < -1e-10:
        #         print(bag_name, lowest_ev)

        for i, bag_name in enumerate(bag_names):
            
            if i%20==0:
                print("Test iter:", i,len(bag_names))

            bag_mask = (bag_names_per_instance == bag_name)

            # Computations for f
            Kzx = self.kernel.compute(self.Z, Xtest[bag_mask,:])
            KzziKzx = np.dot(self.Kzzinv, Kzx)
            sigma_f_ast = self.kernel.compute(Xtest[bag_mask,:], Xtest[bag_mask,:]) - \
                             KzziKzx.T @ (self.Kzz-self.S) @ KzziKzx  # Sigma_f^* = K_{XX} - K_{XZ}*K_{ZZ}^{-1}*(K_{ZZ}-S)*K_{ZZ}^{-1}*K_{ZX}
            # sigma_f_ast = self.kernel.compute_diag(Xtest[bag_mask,:]) - np.diag(np.einsum("ij,ij->j", Kzx, KzziKzx)) + \
            #                  KzziKzx.T @ (self.S @ KzziKzx)  # Sigma_f^* = diag(K_{XX} - K_{XZ}*K_{ZZ}^{-1}*K_{ZX}) + K_{XZ}*K_{ZZ}^{-1}*S*K_{ZZ}^{-1}*K_{ZX}
            mu_f_ast = np.dot(KzziKzx.T, self.m)                      # mu_f^* = K_{XZ} * K_{ZZ}^{-1} * self.m

            # Computations for m
            if not type(self.sigma_m_0_inv) is dict:
                sigma_m_0_inv_local = self.prior_weight*sigma_m_0_inv[np.ix_(bag_mask,bag_mask)].astype("float")
            else:
                sigma_m_0_inv_local = self.prior_weight*sigma_m_0_inv[bag_name].astype("float")
            mu_m_0_local = mu_m_0[bag_mask]
            try:
                sigma_tilde = np.linalg.inv(sigma_m_0_inv_local + np.eye(sigma_m_0_inv_local.shape[0]) )
            except:
                import pdb; pdb.set_trace()
            mu_tilde = sigma_tilde @ (sigma_m_0_inv_local @ mu_m_0_local)
            mu_m_ast = sigma_tilde @ mu_f_ast + mu_tilde
            sigma_m_ast = sigma_tilde + sigma_tilde @ sigma_f_ast @ sigma_tilde.T

            # Instance level and bag level predictions
            instance_probabilities[bag_mask] = probit_func(mu_m_ast/np.sqrt(np.diag(sigma_m_ast)))
            try:
                # bag_probabilities.append(1-multivariate_normal.cdf(x=np.zeros_like(mu_m_ast), mean=mu_m_ast, cov=sigma_m_ast+1e-8*np.eye(sigma_m_ast.shape[0])))
                bag_probabilities.append(1-get_prob_all_less_zero(mu_m_ast, sigma_m_ast+1e-8*np.eye(sigma_m_ast.shape[0])))
            except:
                import pdb; pdb.set_trace()
            mu_f_ast_all[bag_mask] = mu_f_ast
            mu_m_ast_all[bag_mask] = mu_m_ast
        
        bag_probabilities = np.array(bag_probabilities)
        return instance_probabilities, bag_probabilities, mu_f_ast_all, mu_m_ast_all

    def print_m(self):
        with open(os.path.join(self.save_folder, "m.txt"), "a") as f:
            np.savetxt(f, ((self.m).flatten())[None,:], delimiter=",", fmt="%.4f")