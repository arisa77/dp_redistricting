import init
from ektelo import workload
from hdmm import inference
import geopandas
import numpy as np
import pickle
from utility import random_sample_workload, get_subworkloads

class mechanism:
    
    def __init__(self,state,district_size,shapefile_path):

        self.state = state
        self.district_size = district_size
        self.shapefile_path = shapefile_path
    
    def set_engine(self,engine):
        self.engine = engine
        
    def set_workload(self,workload_path,proposal,index,sample_size=100):
        
        
        with open(workload_path,'rb') as f:
            data = pickle.load(f)
        
        W = data[proposal][index]['train']
        if sample_size is not None:
            W = random_sample_workload(W,sample_size,self.district_size).astype(float)
        self.W = W
        print('workload size is ', W.shape)
   
    
    def set_W(self,W):
        
        self.W = W
        
        
    def laplace(self,x,eps=np.sqrt(2),trials=1,seed=100):
        
        prng = np.random.RandomState(seed)
        
        self.I = workload.Identity(x.shape[0])
        ys = [self.I.dot(x)+prng.laplace(loc=0,scale=1/eps,size=self.I.shape[0]) for i in range(trials)]
        return ys
    
    def post_process(self,ys,integer=True,W_test = None):
        
    
        xests=[]
        
        A = self.I.sparse_matrix()
        
        for y in ys:
            
            if self.engine == 'ls':
                
                xest = A.dot(y)
                
            elif self.engine == 'ls-round':
                
                xest = A.dot(y)
                xest = np.clip(xest, a_min =0, a_max = np.inf)
                
            elif self.engine == 'ls-normalize':
                
                xest = A.dot(y)
                sum_xest = np.sum(xest)
                xest = np.clip(xest, a_min =0, a_max = np.inf)
                xest = xest*sum_xest/np.sum(xest)
                
            elif self.engine == 'nnls':
                
                if np.sum(y < 0) == 0:#already non-negative values
                    xest = A.dot(y)
                else:
                    xest = inference.nnls(A,y)
                
            elif self.engine == 'wnnls':
                if np.sum(y < 0) == 0:#already non-negative values
                    xest = A.dot(y)
                else:
                    if W_test is None:
                        xest = inference.wnnls(self.W,A,y)
                    else:
                        xest = inference.wnnls(W_test,A,y)
                
            else:
                print('does not exist such engine %s',self.engine)
                break

                
            if integer:
                xest = xest.round().astype(np.int)
            
            xests.append(xest)
            
        return xests
    
    
    def get_counts(self,column):
    
        gdf = geopandas.read_file(self.shapefile_path)
        values = gdf.loc[:,column]
        values = np.asarray(values)
        return values  
    
    def run_noisy_measurement(self,columns,epsilons,trials=1,seed=100):
        
        data = {col: self.get_counts(col) for col in columns}
        dic = {}
        for eps in epsilons:
            print(eps)
            dic[eps] = {}
            for col, x in data.items():
                ys = self.laplace(x,eps=eps,trials=trials,seed=seed)
                dic[eps][col] = ys
        return dic
                
    
    
    def run_post_process(self,ys,W_tests,columns,epsilons,engine = None,integer=True):
        # scenario: run select-measaure paradigm, release noisy measrement and apply reconstruction for a given workload
        
        self.engine = engine
        
        dic = {}
        for eps in epsilons:
            print(eps)
            dic[eps] = {}
            for col in columns:
                xests = []
                for W_test in get_subworkloads(W_tests,sub_size = self.district_size ):
                    xests_per_test = self.post_process(ys[eps][col],integer=integer,W_test=W_test) #a list of length num_trials
                    xests.append(xests_per_test)
                
                dic[eps][col] = np.stack(xests) #(test_size,num_trial,domain_size)
                #print(dic[eps][col].shape)
        return dic
    
    def run(self,columns,epsilons,engine = None,trials=1,seed=100,integer=True):
        # scenario: run select-measure-reconstruct paradigm and then release noisy answers
        
        self.engine = engine
        data = {col: self.get_counts(col) for col in columns}
        dic = {}
        for eps in epsilons:
            print(eps)
            dic[eps] = {}
            for col, x in data.items():
                ys = self.laplace(x,eps=eps,trials=trials,seed=seed)
                #xests = ys
                xests = self.post_process(ys,integer=integer)
                dic[eps][col] = xests
                
        return dic
    