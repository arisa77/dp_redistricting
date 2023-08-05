import init
from ektelo import workload
from hdmm import inference
import geopandas
import numpy as np
import pickle
from utility import random_sample_workload, get_subworkloads

class Mechanism:
    
    def __init__(self, state, district_size, shapefile_path):

        self.state = state
        self.district_size = district_size
        self.gdf = geopandas.read_file(shapefile_path)
        
    def set_vap_strategy(self, A_vtd, vap_dom = 8):
        '''
        param A_vtd, strategy for a vtd comain
        '''
        
        self.A = workload.Kronecker([workload.Identity(vap_dom),A_vtd])
        self.Apinv = self.A.pinv()
        
    def set_strategy(self, A):
        
        self.A = A
        self.Apinv = A.pinv()
        
    def laplace(self, x, eps=np.sqrt(2), ntrials=1, seed=100):
        '''noisyly measure a data vector on a strategy using the Laplace Mechanism
        
        param x: a data vector
        param eps: epsilon
        param ntrials: number of trials
        param seed: seed for sampling the Laplace noise
        
        return ys: noisy measurements of ntrials
        '''
        
        prng = np.random.RandomState(seed)
        y = self.A.dot(x)
        ys = [y+prng.laplace(loc=0,scale=1/eps,size=y.shape) for i in range(ntrials)]
        
        return ys
    
    def post_process(self, y, engine, integer=True, W=None):
        
            
        if engine == 'ls':

            xest = self.Apinv.dot(y)

        elif engine == 'ls-round':

            xest = self.Apinv.dot(y)
            xest = np.clip(xest, a_min =0, a_max = np.inf)

        elif engine == 'ls-normalize':

            xest = self.Apinv.dot(y)
            sum_xest = np.sum(xest)
            xest = np.clip(xest, a_min =0, a_max = np.inf)
            xest = xest*sum_xest/np.sum(xest)

        elif engine == 'nnls':

            if np.sum(y < 0) == 0:#already non-negative values
                xest = self.Apinv.dot(y)
            else:
                xest = inference.nnls(self.A,y)

        elif engine == 'wnnls':
            if np.sum(y < 0) == 0:#already non-negative values
                xest = self.Apinv.dot(y)
            else:
                xest = inference.wnnls(W,self.A,y)
              

        else:
            print('does not exist such engine %s',self.engine)
            return

        if integer:
            xest = xest.round().astype(np.int)
            
            
        return xest
    
    
    def _get_counts(self, columns):
    
        values = self.gdf.loc[:,columns]
        values = np.asarray(values)
        return values  
    
    def run_noisy_measurement(self, columns, epsilons, ntrials=1, seed=100):
        
        data = {col: self._get_counts(col) for col in columns}
        dic = {}
        for eps in epsilons:
            print(eps)
            dic[eps] = {}
            for col, x in data.items():
                ys = self.laplace(x, eps=eps,ntrials = ntrials, seed = seed)
                dic[eps][col] = ys
        return dic
                
    
    
    def run_post_process(self, ys, columns, epsilons, engine = 'ls', integer=True, W=None):
        # scenario: run select-measaure paradigm, release noisy measrement and apply reconstruction for a given workload
        
        
        dic = {}
        for eps in epsilons:
            print(eps)
            dic[eps] = {}
            for col in columns:
                xests = []
                for W_test in get_subworkloads(W_tests, sub_size = self.district_size):
                    xests_per_test = [self.post_process(y, engine, integer=integer, W=W) for y in ys[eps][col]]
                    xests.append(xests_per_test)
                dic[eps][col] = np.stack(xests) #(test_size,num_trial,domain_size)
        return dic
    
#     def run(self, columns, epsilons, engine = 'ls', ntrials=1, seed=100, integer=True, W=None):
#         # scenario: run select-measure-reconstruct paradigm and then release noisy answers
        
        
#         data = {col: self._get_counts(col) for col in columns}
#         dic = {}
#         for eps in epsilons:
#             print(eps)
#             dic[eps] = {}
#             for col, x in data.items():
#                 ys = self.laplace(x, eps=eps, ntrials=ntrials,seed=seed)
#                 #xests = ys
#                 xests = []
#                 for i,y in enumerate(ys):
#                     #print(i)
#                     xests.append(self.post_process(y, engine, integer=integer, W=W))
#                 #xests = [self.post_process(y, engine, integer=integer, W=W) for y in ys]
#                 dic[eps][col] = xests
                
#         return dic
    def run(self, columns, epsilons, engine = 'ls', ntrials=1, seed=100, integer=True, W=None):
        # scenario: run select-measure-reconstruct paradigm and then release noisy answers
        
        
        data = self._get_counts(columns) # (num_vtds,num_att)
        if len(data.shape) == 2: # if 2d array
            x = data.T.reshape(-1) # (A1V1,A1V2,...,ANVN)
        dic = {}
        for eps in epsilons:
            print(eps)
            dic[eps] = {}
            ys = self.laplace(x, eps=eps, ntrials=ntrials,seed=seed)
            xests = [self.post_process(y, engine, integer=integer, W=W).reshape(data.T.shape) for y in ys]
            dic[eps] = xests 
        return dic
    