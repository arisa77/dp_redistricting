import numpy as np
import os,re,sys
import geopandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
import argparse
import time
from datetime import datetime
import pickle
import gurobipy as gp
from gurobipy import GRB

from ektelo import workload
from ektelo.matrix import EkteloMatrix

PROJECT = 'ramm'
NAME = 'hierarchy/strategy_topdown'
# set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)
# set up pipeline folder if missing
if os.path.exists(os.path.join('empirical', '2_pipeline')):
    pipeline = os.path.join('empirical', '2_pipeline', NAME)
else:
    pipeline = os.path.join('/nfs/dreamlab/scratch1/atajima/',PROJECT, 'empirical', '2_pipeline', NAME)
    
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))
# set up data folder if missing
if os.path.exists(os.path.join('empirical', '0_data')):
    DATA_FOLDER = os.path.join('empirical','0_data')
else:
    DATA_FOLDER = os.path.join('/nfs/dreamlab/scratch1/atajima/',PROJECT,'empirical','0_data')
# set up pipeline folder if missing
if os.path.exists(os.path.join('empirical', '2_pipeline')):
    PIPELINE_FOLDER = os.path.join('empirical','2_pipeline')
else:
    PIPELINE_FOLDER = os.path.join('/nfs/dreamlab/scratch1/atajima/',PROJECT,'empirical','2_pipeline')


BUDGET = {
        'equal': [1/3, 1/3, 1/3], 
        'state-heavy': [0.5, 0.25, 0.25], 
        'county-heavy': [0.25, 0.5, 0.25],#IA optimal
        'county-heavy2': [0.2, 0.5, 0.3],
        'vtd-heavy1': [0.05, 0.25, 0.7],#MA optimal
        'vtd-heavy2': [0.02, 0.28, 0.7],#NC optimal
        'vtd-heavy3': [0.01, 0.24, 0.75],
        'vtd-heavy4': [0.1, 0.2, 0.7],#CT optimal
        'vtd-heavy5': [0.1, 0.4, 0.5],
        'vtd-veryheavy':[0.1,0.1,0.8],
        }
BUDGET2 = {
        'equal': [0.5,0.5], 
        'state-heavy': [0.7, 0.3], 
        'vtd-heavy1': [0.4, 0.6],
        'vtd-heavy2': [0.3, 0.7],
        'vtd-heavy3': [0.2, 0.8],
        'vtd-heavy4': [0.1, 0.9],
        'vtd-veryheavy':[0.05,0.95],
        }
def get_params(state):

    #config
    params = {}
    params['state'] = state

    if state == 'NC':

       
        params['district size'] = 13
        params['pop columns'] = ['TOTPOP']
        params['vap columns'] = ['HVAP','WVAP','BVAP','AMINVAP','ASIANVAP','NHPIVAP','OTHERVAP','2MOREVAP']
        params['vtd column'] = 'VTD_Key'
        params['county column'] = 'County' 
        params['shapefile'] = DATA_FOLDER+'/external/NC-shapefiles/NC_VTD/NC_VTD.shp'
        params['proposal'] = 'NC-recom'
        params['index'] = 0
        params['strategy path'] = PIPELINE_FOLDER+'/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['hierarchical strategy path'] = PIPELINE_FOLDER+'/29_hierarchical_workload_optimization/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        

    elif state == 'IA':
    
        params['district size'] = 4 
        params['pop columns'] = ['TOTPOP']
        params['vtd column'] = 'NAME10'
        params['county column'] = 'COUNTYFP10' 
        params['shapefile'] = DATA_FOLDER+'/external/IA-shapefiles/IA_counties/IA_counties.shp'
        params['proposal'] = 'IA-recom'
        params['index'] = 0
        params['strategy path'] = PIPELINE_FOLDER+'/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['hierarchical strategy path'] = PIPELINE_FOLDER+'/29_hierarchical_workload_optimization/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        
    elif state == 'MA':
        
        params['pop columns'] = ['POP10']
        params['district size'] = 9 
        params['vtd column'] = 'Name'
        params['county column'] = 'COUNTYFP10' 
        params['shapefile'] = DATA_FOLDER+'/manual/MA-shapefiles/MA_no_islands_12_16_county/MA_precincts_12_16.shp'
        params['proposal'] = 'MA-recom'
        params['index'] = 0
        params['strategy path'] = PIPELINE_FOLDER+'/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['hierarchical strategy path'] = PIPELINE_FOLDER+'/29_hierarchical_workload_optimization/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        
    elif state == 'CT':
        
        params['pop columns'] = ['TOTPOP']
        params['district size'] = 5
        params['vtd column'] = 'NAME10'
        params['county column'] = 'COUNTYFP10' 
        params['shapefile'] = DATA_FOLDER+'/external/CT-shapefiles/CT_precincts/CT_precincts.shp'  
        params['proposal'] = 'CT-recom'
        params['index'] = 0
        params['strategy path'] = PIPELINE_FOLDER+'/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['hierarchical strategy path'] = PIPELINE_FOLDER+'/29_hierarchical_workload_optimization/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
    else:
        print('unknown state')
        sys.exit()
    return params

def encode_workload(vtd_list):
    
    if not isinstance(vtd_list,np.ndarray):
        vtd_list = np.asarray(vtd_list)
        
    domain_size = len(vtd_list)
    

    queries=np.unique(vtd_list)
    
    W = []
    def encode_query(col):
        row = [0] * len(col)
        data = np.ones(len(col))
        return sparse.csr_matrix((data,(row,col)),shape = (1,domain_size),dtype=np.int8)

    
   
    for i in queries:
        vtds = np.where(vtd_list == i)[0]
        w = encode_query(vtds)
        W.append(w)

    W = sparse.vstack(tuple(W)).toarray()
    
    return W

def county_workload(gdf, params):

    W_county = encode_workload(gdf.groupby(params['county column']).ngroup().values)
    return EkteloMatrix(W_county.astype(float))


def hierarchical_inference_gb(noisy_children, noisy_parent, l1_reg=0, l2_reg=0, maxiter = 15000, non_negativity = False, verbose = False):
    """
    Solves the hierarchical constraint problem min || x - noisy_children || s.t. noisy_parent = sum(x) (and x >= 0 option) using gradient-based optimization
    
    :param noisy_children: numpy vector, 
    :param noisy_parent: a single scalar
    """
    
    #print(A.shape, y.shape, W.shape, ans.shape)
     
    #print(noisy_children.shape,noisy_parent)
    n = len(noisy_children)
        
    m = gp.Model("children")
    #m.setParam('TimeLimit', 10)
    if not verbose: m.params.OutputFlag = 0
    if non_negativity:
        xs = m.addVars(n, vtype=GRB.CONTINUOUS, name="Counts") #non-negative 
    else:
        #print("negative okay")
        xs = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Counts")
    
    
    obj = gp.quicksum([(xs[i] - noisy_children[i])**2 for i in range(n)])
    m.setObjective(obj, GRB.MINIMIZE)
    
    #m.addConstrs((ans[i] == gp.quicksum([ W.dense_matrix()[i][j]*xs[j] for j in range(W.shape[1])]) for i in range(W.shape[0]) ),name="parental_equality")
    m.addConstr(noisy_parent == gp.quicksum([xs[i] for i in range(n)]) ,name="parental_equality")
    m.optimize()
    xest = np.array([v.x for v in m.getVars()])
    
    #if non_negativity:
    #    xest = xest.round().astype(int)
    return xest



def laplace_measurement(A,x,eps=np.sqrt(2),trials=1,seed=None):   
    prng = np.random.RandomState(seed)
    delta = A.sensitivity()
    A = A.sparse_matrix()
    y = [A.dot(x)+prng.laplace(loc=0,scale=delta/eps,size=A.shape[0]) for i in range(trials)]
    return y

def laplace_workload(W,x,eps=np.sqrt(2),trials=1,seed=None,rounding=False):  
    
    prng = np.random.RandomState(seed)
    delta = max(np.sum(W,axis=0))#sensitivity
    
    if rounding:
        y = [(W.dot(x)+prng.laplace(loc=0,scale=delta/eps,size=W.shape[0])).round().astype(int) for i in range(trials)]
    else:
        y = [W.dot(x)+prng.laplace(loc=0,scale=delta/eps,size=W.shape[0]) for i in range(trials)]
    return y

def run_SCV(params):
    print(params['budgetsplit'])
    epsilons = [ params['epsilon']*p for p in BUDGET[params['budgetsplit']]]
    
    gdf = geopandas.read_file(params['shapefile'])
    # measurements
    W_state =  workload.Total(gdf.shape[0])
    W_county = county_workload(gdf, params)
    #with open(params['hierarchical strategy path'],'rb') as f:
    #    A_vtd = pickle.load(f)
    A_vtd = {}
    A_vtd['Identity'] = workload.Identity(gdf.shape[0])   
    #with open(params['hierarchical strategy path'],'rb') as f:
    #    A_vtd['p-Identity (H)'] = pickle.load(f)['p-Identity']
        
    # noisy measurements
    # if non-negativity constraint force the state counts be integer
    if params['non-negativity']:
        noisy_state = laplace_workload(W_state.matrix, gdf[params['pop columns'][0]].values,eps = epsilons[0],trials=params['trials'],seed=params['seed'])
    else:
        noisy_state = laplace_workload(W_state.matrix, gdf[params['pop columns'][0]].values,eps = epsilons[0],trials=params['trials'],seed=params['seed'])
        
    noisy_county = laplace_workload(W_county.matrix, gdf[params['pop columns'][0]].values,eps = epsilons[1],trials=params['trials'],seed=params['seed']) 
    y_vtd = {}
    for stg in  A_vtd.keys():
        y_vtd[stg] = laplace_measurement(A_vtd[stg],gdf[params['pop columns'][0]].values,eps = epsilons[2],trials=params['trials'])
    
    

    # state - county hierarchical constraints
    print("county")
    adjusted_county = [hierarchical_inference_gb(
        noisy_county[i], 
        noisy_state[i][0],
        non_negativity = params['non-negativity'])
        for i in range(params['trials'])
    ]

    # county - vtd hierarchical constraints
    
    adjusted_vtd = {}
    for stg in A_vtd.keys():
        print(stg)
        #if stg == 'p-Identity':
        #    break
        Apinv = A_vtd[stg].pinv()
        adjusted_vtd[stg] = []
        for i in range(params['trials']):
            xest  = Apinv.dot(y_vtd[stg][i])
            adjusted = xest.copy()
            for j in range(W_county.shape[0]):
                vtd_group = np.argwhere(W_county.matrix[j] == 1.0).reshape(-1)
                adjusted[vtd_group]= hierarchical_inference_gb(xest[vtd_group],adjusted_county[i][j],non_negativity = params['non-negativity'])
            adjusted_vtd[stg].append(adjusted)
            

#     print(noisy_state)
#     print(np.sum(adjusted_county[0]))
#     print(adjusted_county[0][:5])
#     print(W_county.dot(adjusted_vtd['Identity'][0])[:5])
#     print(W_county.dot(adjusted_vtd['p-Identity'][0])[:5])
    dic = {"state":noisy_state,"county":adjusted_county,"vtd":adjusted_vtd}

    if params['save']:
        if params['non-negativity']:
            path = os.path.join(pipeline, 'tmp','nonneg','adjusted_pop_'+params['state']+'_eps'+str(params['epsilon'])+'_'+params['budgetsplit']+'_seed'+str(params['seed'])+'.pickle')
        else:
            path = os.path.join(pipeline, 'tmp','adjusted_pop_'+params['state']+'_eps'+str(params['epsilon'])+'_'+params['budgetsplit']+'_seed'+str(params['seed'])+'.pickle')
        with open(path, 'wb') as f:
            pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def run_SCV_vap(params):
    print(params['budgetsplit'])
    epsilons = [ params['epsilon']*p for p in BUDGET[params['budgetsplit']]]
    
    gdf = geopandas.read_file(params['shapefile'])
    # measurements
    W_state =  workload.Total(gdf.shape[0])
    W_county = county_workload(gdf, params)
    #with open(params['hierarchical strategy path'],'rb') as f:
    #    A_vtd = pickle.load(f)
    A_vtd = {}
    A_vtd['Identity'] = workload.Identity(gdf.shape[0])      
    dic = {}
    # noisy measurements
    for vap_col in params['vap columns']:
        
        # noisy measurements
        noisy_state = laplace_workload(W_state.matrix, gdf[vap_col].values,eps = epsilons[0],trials=params['trials'],seed=params['seed']) 
        noisy_county = laplace_workload(W_county.matrix, gdf[vap_col].values,eps = epsilons[1],trials=params['trials'],seed=params['seed']) 
        y_vtd = {}
        for stg in  A_vtd.keys():
            y_vtd[stg] = laplace_measurement(A_vtd[stg],gdf[vap_col].values,eps = epsilons[2],trials=params['trials'])

    

        # state - county hierarchical constraints
        print("county")
        adjusted_county = [hierarchical_inference_gb(
            noisy_county[i], 
            noisy_state[i][0],
            non_negativity = params['non-negativity'])
            for i in range(params['trials'])
        ]

        # county - vtd hierarchical constraints

        adjusted_vtd = {}
        for stg in A_vtd.keys():
            print(stg)
            #if stg == 'p-Identity':
            #    break
            Apinv = A_vtd[stg].pinv()
            adjusted_vtd[stg] = []
            for i in range(params['trials']):
                xest  = Apinv.dot(y_vtd[stg][i])
                adjusted = xest.copy()
                for j in range(W_county.shape[0]):
                    vtd_group = np.argwhere(W_county.matrix[j] == 1.0).reshape(-1)
                    adjusted[vtd_group]= hierarchical_inference_gb(xest[vtd_group],adjusted_county[i][j],non_negativity = params['non-negativity'])
                adjusted_vtd[stg].append(adjusted)



        dic_vap = {"state":noisy_state,"county":adjusted_county,"vtd":adjusted_vtd}
        dic[vap_col] = dic_vap

    if params['save']:
        path = os.path.join(pipeline, 'tmp','adjusted_vap_'+params['state']+'_eps'+str(params['epsilon'])+'_'+params['budgetsplit']+'_seed'+str(params['seed'])+'.pickle')
        with open(path, 'wb') as f:
            pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
        

def run_SV(params):
    print(params['budgetsplit'])
    epsilons = [ params['epsilon']*p for p in BUDGET2[params['budgetsplit']]]
    
    gdf = geopandas.read_file(params['shapefile'])
    # measurements
    W_state =  workload.Total(gdf.shape[0])
#     with open(params['hierarchical strategy path'],'rb') as f:
#         A_vtd = pickle.load(f)
    A_vtd = {}
    A_vtd['Identity'] = workload.Identity(gdf.shape[0])   
    # noisy measurements
    # if non-negativity constraint force the state counts be integer
    if params['non-negativity']:
        noisy_state = laplace_workload(W_state.matrix, gdf[params['pop columns'][0]].values,eps = epsilons[0],trials=params['trials'],seed=params['seed'])
    else:
        noisy_state = laplace_workload(W_state.matrix, gdf[params['pop columns'][0]].values,eps = epsilons[0],trials=params['trials'],seed=params['seed'])
        

    y_vtd = {}
    for stg in  A_vtd.keys():
        y_vtd[stg] = laplace_measurement(A_vtd[stg],gdf[params['pop columns'][0]].values,eps = epsilons[1],trials=params['trials'])
    
    

    # state - vtd hierarchical constraints
    
    adjusted_vtd = {}
    for stg in A_vtd.keys():
        print(stg)
        #if stg == 'p-Identity':
        #    break
        Apinv = A_vtd[stg].pinv()
        adjusted_vtd[stg] = []
        for i in range(params['trials']):
            xest  = Apinv.dot(y_vtd[stg][i])
            adjusted = hierarchical_inference_gb(xest,noisy_state[i][0],non_negativity = params['non-negativity'])
            adjusted_vtd[stg].append(adjusted)
            

    
    dic = {"state":noisy_state,"vtd":adjusted_vtd}

    if params['save']:
        if params['non-negativity']:
            path = os.path.join(pipeline, 'tmp','nonneg','adjusted_level2_pop_'+params['state']+'_eps'+str(params['epsilon'])+'_'+params['budgetsplit']+'_seed'+str(params['seed'])+'.pickle')
        else:
            path = os.path.join(pipeline, 'tmp','adjusted_level2_pop_'+params['state']+'_eps'+str(params['epsilon'])+'_'+params['budgetsplit']+'_seed'+str(params['seed'])+'.pickle')
        with open(path, 'wb') as f:
            pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)


def default_params():
    """
    Return default parameters to run this program
    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['epsilon'] = np.sqrt(2)
    params['state'] = 'IA'
    params['seed'] = None
    params['save'] = True
    params['nonneg'] = False
    params['trials'] = 100
    params['budgetsplit'] = 'equal' # 'bottom heavy' 'top heavy'
    params['vap'] = False

    return params


if __name__ == '__main__':

   
    #setup args
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--state", metavar="state", default = 'NC',type=str, help="Which state to use")             
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--trials', type=int, help='number of trials')
    parser.add_argument('--seed', type=int, help='seed for RNG')
    parser.add_argument("--budgetsplit", metavar="budgetsplit", default = 'equal',type=str, help="how to split budget")    
    parser.add_argument('--save', action='store_true', help='save results')
    parser.add_argument('--nonneg', action = 'store_true', help='specify if non-negativity constraint is used')
    parser.add_argument('--vap', action='store_true', help='specify if vap counts are used')
    parser.set_defaults(**default_params())
    args = parser.parse_args()
    prng = np.random.RandomState(args.seed)
    
   
    # setupdata
    
    params = get_params(args.state)
    params['non-negativity'] = args.nonneg
    params['epsilon'] = args.epsilon
    params['trials'] = args.trials
    params['seed'] = args.seed
    params['save'] = args.save
    params['budgetsplit'] = args.budgetsplit
    params['vap'] = args.vap
    
    start_time = time.time()
    if params['vap']:
        run_SCV_vap(params)
    else:
        if params['state'] == 'IA':
            run_SV(params)
        else:
            run_SCV(params) 
    end_time = time.time()
    print('timestamp:',datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    print('time:',end_time - start_time)

    
#     for key in BUDGET2.keys():
#         params['budgetsplit'] = key
#         start_time = time.time()

        
#         run_SV(params) 
        
#         end_time = time.time()
#         print('timestamp:',datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
#         print('time:',end_time - start_time)

    

