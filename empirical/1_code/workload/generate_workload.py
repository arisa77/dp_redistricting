import os, re
import pickle
import sys
import numpy as np

sys.path.append("..")
from src.utility import encode_ensembles

NAME = 'workload'
PROJECT = 'ramm'

# set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)
# set up pipeline folder if missing
if os.path.exists(os.path.join('empirical', '2_pipeline')):
    pipeline = os.path.join('empirical', '2_pipeline', NAME)
else:
    pipeline = os.path.join('2_pipeline', NAME)
    
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))
########################################################

def split(indices,nsplit):
    split_indices=[]
    
    n = len(indices)
    if n%nsplit!=0:
        print('invalidate number of splits')
        return
    
    
    np.random.shuffle(indices)
    sp_indices=np.split(indices, nsplit)
    
    for i in range(nsplit):
        test_indices=list(sp_indices[i])
        train_indices = list(set(indices) - set(test_indices))
        split_indices.append((train_indices,test_indices))
    
    return split_indices

def train_test_CVsplit(W,sub_size,nsplit=5):
    data=[]
    n = int(W.shape[0]/sub_size)
    for train_index, test_index in split(np.arange(n),nsplit):
        d = {}
        W_train= np.vstack([W[i*sub_size:(i+1)*sub_size] for i in train_index]).astype(bool)
        W_test= np.vstack([W[i*sub_size:(i+1)*sub_size] for i in test_index]).astype(bool)
    
        data.append({'train':W_train,'test':W_test})
    return data

def generate_workloads(input_path, output_path):
    with open(input_path,'rb') as f:
        ass_maps = pickle.load(f)
    
    W = encode_ensembles(ass_maps).astype(float)   
    np.save(output_path,W)


if __name__ == "__main__":
    
    
    name = 'redistricting/MA-gerrychain'
    file = 'assignment_recom_0.02.pkl'
    input_path=os.path.join('empirical', '2_pipeline', name, 'store', file)
    output_path = os.path.join('empirical', '2_pipeline', NAME, 'store', "workload_ma_recom.npy")
    generate_workloads(input_path, output_path)
