import os, re, sys
import geopandas

sys.path.append("..")
from src.utility import *


NAME = 'optimize_noisy_workload'
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

####################################################

def get_default_params(state):
    params = {}
    if state == 'IA':
        
        params['state'] = 'IA'
        params['district size'] = 4 
        params['pop columns'] = ['TOTPOP']
        params['vtd column'] = 'NAME10'
        params['county column'] = 'COUNTYFP10' 
        params['shapefile'] = 'empirical/0_data/external/IA-shapefiles/IA_counties/IA_counties.shp'
        params['train workload'] = os.path.join('empirical', '2_pipeline', 'workload_julia', 'store', "workload_ia_compact_pop0.1_eps0.001_Identity_ls-round.npy")
        params['test workload'] = os.path.join('empirical', '2_pipeline', 'workload', 'store', "workload_ia_recom.npy")
    
    elif state == "NC":
        
        params['state'] = 'NC'
        params['district size'] = 13
        params['pop columns'] = ['TOTPOP']
        params['vtd column'] = 'VTD_Key'
        params['county column'] = 'County' 
        params['shapefile'] = 'empirical/0_data/external/NC-shapefiles/NC_VTD/NC_VTD.shp'
        params['train workload'] = os.path.join('empirical', '2_pipeline', 'workload_julia', 'store', "workload_nc_compact_pop0.1_eps0.001_Identity_ls-round.npy")
        params['test workload'] = os.path.join('empirical', '2_pipeline', 'workload', 'store', "workload_nc_recom.npy")

    elif state == 'MA':
        
        
        params['state'] = 'MA'
        params['district size'] = 9
        params['pop columns'] = ['POP10']
        params['vtd column'] = 'Name'
        params['county column'] = 'COUNTYFP10' 
        params['shapefile'] = 'empirical/0_data/manual/MA-shapefiles/MA_no_islands_12_16_county/MA_precincts_12_16.shp'
        params['train workload'] = os.path.join('empirical', '2_pipeline', 'workload_julia', 'store', "workload_ma_compact_pop0.1_eps0.001_Identity_ls-round.npy")
        params['test workload'] = os.path.join('empirical', '2_pipeline', 'workload', 'store', "workload_ma_recom.npy")
        
    elif state == 'CT':
        
        params['state'] = 'CT'
        params['pop columns'] = ['TOTPOP']
        params['district size'] = 5
        params['vtd column'] = 'NAME10'
        params['county column'] = 'COUNTYFP10' 
        params['shapefile'] = 'empirical/0_data/external/CT-shapefiles/CT_precincts/CT_precincts.shp' 
        params['train workload'] = os.path.join('empirical', '2_pipeline', 'workload_julia', 'store', "workload_ct_compact_pop0.1_eps0.001_Identity_ls-round.npy")
        params['test workload'] = os.path.join('empirical', '2_pipeline', 'workload', 'store', "workload_ct_recom.npy")

    return params

def select(filename,W_trains,strategies,p=7,opt=True,rows=None,save=False):
    '''
    :params: W_trains, a list of hierarchical workload being trained e.x: [W_state, W_county, W_districts]
    '''
    
    path = os.path.join('empirical', '2_pipeline', NAME, 'store', 'strategy-'+filename+'.pickle')
    if not opt:
        with open(path,'rb') as f:
            A = pickle.load(f)
    else:
        W_train = np.vstack(W_trains)
        A = {}
        print('optimizing...')
        for stg in strategies:
            print(stg)
            if stg == 'p-Identity' or stg == 'p-I' or stg == 'p-Identity (H)':
                A[stg] = opt_p_identity(W_train,p=p,restarts=1)
            elif stg == 'Identity' or stg == 'I':
                A[stg] = workload.Identity(W_train.shape[1])
            elif stg == 'row-weighted':
                A[stg] = opt_row_weighted(W_train, rows)
            else:
                print('%s is not defined'%(stg))
        if save:
            if os.path.exists(path):
                with open(path,'rb') as f:
                    A_update = pickle.load(f)
                for key in A.keys():
                    A_update[key] = A[key]
                with open(path,'wb') as f:
                    pickle.dump(A_update,f,pickle.HIGHEST_PROTOCOL)
            else:
                with open(path,'wb') as f:
                    pickle.dump(A,f,pickle.HIGHEST_PROTOCOL)
    
    for key in A.keys():
        print("train rootmse %s: %f"%(key,error.rootmse(W_train,A[key])))

    return A

def select_kron(filename,W_vaps,W_vtds,strategies,ps,opt=True,rows=None,save=False):
    '''
    :params: W_vtds, a list of workloads being trained e.x: [W_state, W_county, W_districts]
    '''
    
    path = os.path.join('empirical', '2_pipeline', NAME, 'store', 'strategy-'+filename+'.pickle')
    if not opt:
        with open(path,'rb') as f:
            A = pickle.load(f)
    else:
        W_vtds = np.vstack(W_vtds)
        workloads = workload.Kronecker([EkteloMatrix(W_vaps),EkteloMatrix(W_vtds)])
        A = {}
        print('optimizing...')
        for stg in strategies:
            print(stg)
            if stg == 'p-Identity' or stg == 'p-I':
                pid = templates.KronPIdentity([ps[0],ps[1]], [W_vaps.shape[1],W_vtds.shape[1]])
                pid.optimize(workloads)
                A[stg] = pid.strategy()
            elif stg == 'Identity' or stg == 'I':
                A[stg] = workload.Kronecker([workload.Identity(W_vaps.shape[1]),workload.Identity(W_vtds.shape[1])])
            else:
                print('%s is not defined'%(stg))
        if save:
            with open(path,'wb') as f:
                pickle.dump(A,f,pickle.HIGHEST_PROTOCOL)
    
    for key in A.keys():
        print("train rootmse %s: %f"%(key,error.rootmse(workloads,A[key])))

    return A

def county_workload(params):
    gdf = geopandas.read_file(params['shapefile'])
    W_county = encode_workload(gdf.groupby(params['county column']).ngroup().values)
    return W_county.astype(float)

def test_error(A, W_tests, eps_in):

    for key in A.keys():
        if key == 'p-Identity' or key == 'p-I' or key == 'p-Identity (H)':
            print("unseen test rootmse %s: %f"%(key,error.rootmse(W_tests, A[key],eps=np.sqrt(2)-eps_in)))
        else:
            print("unseen test rootmse %s: %f"%(key,error.rootmse(W_tests, A[key],eps=np.sqrt(2))))

def workload_opt(params, filename, eps_in, p=9):

    W_trains = np.load(params['train workload'])
    W_tests = np.load(params['test workload'])

    A = select(filename,[W_trains],['Identity','p-Identity'],p=p,opt=True,save=True)

    test_error(A, W_tests, eps_in)

def hierarchical_workload_opt(params, filename, weights, eps_in, p=9):
    assert len(weights) == 2 or len(weights) == 3
    W_districts = np.load(params['train workload'])
    W_tests = np.load(params['test workload'])  
    W_state = np.ones((1,W_trains.shape[1]))

    if len(weights)==3:
        W_county = county_workload(params)
        W_trains = [weights[0]*W_state, weights[1]*W_county, weights[2]*W_districts]
    else:
        W_trains = [weights[0]*W_state, weights[2]*W_districts]

    A = select(filename,W_trains,['Identity','p-Identity (H)'],p=p,opt=True,save=True)

def vap_kron_workload_opt(params, filename, eps_in, p=9):

    W_trains = np.load(params['train workload'])
    W_tests = np.load(params['test workload'])

    W_vap = np.array([[1,1,1,1,1,1,1,1],[1,0,1,1,1,1,1,1]]).astype(float)
    A = select_kron(filename,W_vap,[W_trains],['Identity','p-Identity'],p = [1,50],opt=True,save=True)


def main():
    eps_in = 0.001 # privacy budget spent to generate input/train workloads

    workload_opt(get_default_params('IA'), 'IA-recom-pop0.1-ep0.001', eps_in,p=9)
    workload_opt(get_default_params('NC'), 'NC-recom-pop0.1-ep0.001', eps_in, p=50)
    workload_opt(get_default_params('MA'), 'MA-recom-pop0.1-ep0.001', eps_in, p=45)
    workload_opt(get_default_params('CT'), 'CT-recom-pop0.1-ep0.001', eps_in, p=20)

    hierarchical_workload_opt(get_default_params('IA'), 'IA-recom-pop0.1-ep0.001', [130,1], eps_in,p=9)
    hierarchical_workload_opt(get_default_params('NC'), 'NC-recom-pop0.1-ep0.001', [500,50,1],eps_in, p=50)
    hierarchical_workload_opt(get_default_params('MA'), 'MA-recom-pop0.1-ep0.001', [260,130,1],eps_in, p=45)
    hierarchical_workload_opt(get_default_params('CT'), 'CT-recom-pop0.1-ep0.001', [170,65,1],eps_in, p=20)


    vap_kron_workload_opt(get_default_params("NC"),'NC-recom-pop0.1-ep1e-05',1e-05,p=[1,50])
