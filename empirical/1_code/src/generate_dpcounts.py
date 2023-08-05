from mechanism import Mechanism
import json
import numpy as np
import pickle
import os, re, math, time
from utility import random_sample_workload


#####################################################
#setup

NAME = 'generate_dpcounts'
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
        
######################################################

def get_parameters(state):

    params = {}

    if state == 'NC':
        params['state'] = 'NC'
        params['district size'] = 13
        params['epsilons'] = np.logspace(-5, 0, num=6) - 1e-06#np.logspace(-6, 0, num=7)
        #params['epsilons'] = np.logspace(-3, 0, num=10)
        params['ntrials'] = 500
        params['pop columns'] = ['TOTPOP']
        params['engines'] = ['ls','ls-round','ls-normalize']#'nnls']#,'wnnls']
        params['shapefile'] = 'empirical/0_data/external/NC-shapefiles/NC_VTD/NC_VTD.shp'
        params['workload path'] = 'empirical/2_pipeline/train_test_split/store/dataset_NC.pickle'
        params['train size'] = 100
        params['proposal'] = 'NC-recom'
        params['index'] = 0
        #params['strategy path'] = 'empirical/2_pipeline/29_hierarchical_workload_optimization/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['strategy path'] = 'empirical/2_pipeline/32_optimize_noisy_workload/store/strategy-NC-recom-pop0.1-ep1e-06.pickle'
        params['output dir'] = 'empirical/2_pipeline/generate_dpcounts/store/recom-pop-0.1-ep1e-06/'
        params['output'] = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+state+'.json'
        

    elif state == 'IA':
        params['state'] = 'IA'
        params['district size'] = 4
        params['epsilons'] = np.logspace(-5, 0, num=6) - 1e-06
        params['ntrials'] = 500
        params['pop columns'] = ['TOTPOP']
        params['engines'] = ['ls','ls-round','ls-normalize','nnls']#,'wnnls']
        params['shapefile'] = 'empirical/0_data/external/IA-shapefiles/IA_counties/IA_counties.shp'
        params['workload path'] = 'empirical/2_pipeline/train_test_split/store/dataset_IA.pickle'
        params['train size'] = "USE ALL"
        params['proposal'] = 'IA-recom'
        params['index'] = 0
        #params['strategy path'] = 'empirical/2_pipeline/29_hierarchical_workload_optimization/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['strategy path'] = 'empirical/2_pipeline/32_optimize_noisy_workload/store/strategy-IA-recom-pop0.1-ep1e-06.pickle'
        params['output dir'] = 'empirical/2_pipeline/generate_dpcounts/store/recom-pop-0.1-ep1e-06/'
        params['output'] = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+state+'.json'

    

    elif state == 'MA':
        params['state'] = 'MA'
        params['district size'] = 9
        params['epsilons'] = np.logspace(-6, 0, num=7)
        params['ntrials'] = 500
        params['pop columns'] = ['POP10']
        params['engines'] = ['ls-round']#,'wnnls']
        params['shapefile'] = 'empirical/0_data/manual/MA-shapefiles/MA_no_islands_12_16_county/MA_precincts_12_16.shp'
        params['proposal'] = 'MA-recom'
        params['index'] = 0
        #params['strategy path'] = 'empirical/2_pipeline/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['workload path'] = os.path.join('empirical', '2_pipeline', 'train_test_split', 'store', "dataset_MA.pickle")
        params['train size'] = "USE ALL"

        params['strategy path'] = 'empirical/2_pipeline/32_optimize_noisy_workload/store/strategy-MA-recom-pop0.1-ep0.001.pickle'
        params['output dir'] = 'empirical/2_pipeline/generate_dpcounts/store/recom-pop-0.1-ep0.001/'
        params['output'] = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+state+'.json'

    elif state == 'CT':
        params['state'] = 'CT'
        params['pop columns'] = ['TOTPOP']
        params['district size'] = 5
        params['epsilons'] = np.logspace(-6, 0, num=7)
        params['ntrials'] = 500
        params['shapefile'] = 'empirical/0_data/external/CT-shapefiles/CT_precincts/CT_precincts.shp'  
        params['proposal'] = 'CT-recom'
        params['index'] = 0
        params['strategy path'] = 'empirical/2_pipeline/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
        params['workload path'] = os.path.join('empirical', '2_pipeline', 'train_test_split', 'store', "dataset_CT.pickle")
        params['train size'] = "USE ALL"
        params['strategy path'] = 'empirical/2_pipeline/32_optimize_noisy_workload/store/strategy-CT-recom-pop0.1-ep0.001.pickle'
        params['output dir'] = 'empirical/2_pipeline/generate_dpcounts/store/recom-pop-0.1-ep0.001/'
        params['output'] = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+state+'.json'

    return params

def numpyconverter(obj):
    
    # Convert numpy to a Python int/float/list before serializing the object:
   
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
        
def execute(params):
    
    with open(params['strategy path'],'rb') as f:
        strategy = pickle.load(f)
    mechanism = Mechanism(params['state'],params['district size'],params['shapefile'])
    
    
    
    #baseline mechanism -- Identity
    print("baseline")
    mechanism.set_strategy(strategy['Identity'])
    baseline_results = {}
    for engine in params['engines']:
        print(engine)
        baseline_results[engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'])

    #HDMM -- p-Identity
    print("hdmm")
    mechanism.set_strategy(strategy['p-Identity'])
    hdmm_results = {}
    for engine in params['engines']:
        print(engine)
        hdmm_results[engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'])

    dic = {'Identity':baseline_results,'p-Identity':hdmm_results}
    with open(params['output dir'], 'w') as f:
        json.dump(dic, f,  indent=4, default=numpyconverter)
       
# def execute_postprocess(params,engine):
    
#     with open(params['strategy path'],'rb') as f:
#         strategy = pickle.load(f)
#     mechanism = Mechanism(params['state'],params['district size'],params['shapefile'])
    
#     if engine == 'wnnls':
#         with open(params['workload path'],'rb') as f:
#             Ws = pickle.load(f)
#         Wtrains = Ws[params['proposal']][params['index']]['train']
#         Wtrains = random_sample_workload(Wtrains,params['train size'],sub_size = params['district size'])
    
#     #baseline mechanism -- Identity
#     print("baseline")
#     mechanism.set_strategy(strategy['Identity'])
#     baseline_results = {}
#     if engine == 'wnnls':
#         baseline_results[engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'],W = Wtrains)
#     else:
#         baseline_results[engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'])

#     #HDMM -- p-Identity
#     print("hdmm")
#     mechanism.set_strategy(strategy['p-Identity'])
#     hdmm_results = {}
#     if engine == 'wnnls':
#         hdmm_results[engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'],W = Wtrains)
#     else:
#         hdmm_results[engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'])

#     dic = {'Identity':baseline_results,'p-Identity':hdmm_results}
#     outputfile = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+params['state']+'_'+engine+'2.json'
#     with open(outputfile, 'w') as f:
#         json.dump(dic, f,  indent=4, default=numpyconverter)

def execute_postprocess(params,engine):
    
    with open(params['strategy path'],'rb') as f:
        strategy = pickle.load(f)
    mechanism = Mechanism(params['state'],params['district size'],params['shapefile'])
    
    Wtrains = None
    if engine == 'wnnls':
        with open(params['workload path'],'rb') as f:
            Ws = pickle.load(f)
        Wtrains = Ws[params['proposal']][params['index']]['train']
        Wtrains = random_sample_workload(Wtrains,params['train size'],sub_size = params['district size'])
    
    dic = {}
    for key in strategy.keys():
        dic[key] = {}
        print("running for ",key)

        #generate noisy data
        mechanism = Mechanism(params['state'],params['district size'],params['shapefile'])
        mechanism.set_strategy(strategy[key])
        dic[key][engine] = mechanism.run(params['pop columns'],params['epsilons'],engine=engine,ntrials=params['ntrials'],W = Wtrains) 

    outputfile = params['output dir']+'dptotpop_'+params['state']+'_'+engine+'.json'
    with open(outputfile, 'w') as f:
        json.dump(dic, f,  indent=4, default=numpyconverter)

def single_execute(params,engine,eps):
    
    with open(params['strategy path'],'rb') as f:
        strategy = pickle.load(f)
    mechanism = Mechanism(params['state'],params['district size'],params['shapefile'])
    
    Wtrains = None
    if engine == 'wnnls':
        with open(params['workload path'],'rb') as f:
            Ws = pickle.load(f)
        Wtrains = Ws[params['proposal']][params['index']]['train']
        Wtrains = random_sample_workload(Wtrains,params['train size'],sub_size = params['district size'])
    
    dic = {}
    for key in strategy.keys():
        dic[key] = {}
        print("running for ",key)

        #generate noisy data
        mechanism = Mechanism(params['state'],params['district size'],params['shapefile'])
        mechanism.set_strategy(strategy[key])
        dic[key][engine] = mechanism.run(params['pop columns'],[eps],engine=engine,ntrials=params['ntrials'],W = Wtrains) 

    outputfile = params['output dir']+'dptotpop_'+params['state']+'_'+engine+'.json'
    with open(outputfile, 'w') as f:
        json.dump(dic, f,  indent=4, default=numpyconverter)


        
if __name__=='__main__':
    
    execute(get_parameters('IA'))
    #execute(get_parameters('NC'))
    #execute_postprocess(get_parameters('IA'),'ls-round')
    #execute_postprocess(get_parameters('NC'),'ls-round')
    #execute_postprocess(get_parameters('NC'),'nnls')
    #execute_postprocess(get_parameters('NC'),'wnnls')
    #execute_postprocess(get_parameters('MA'),'ls-round')
    #execute_postprocess(get_parameters('CT'),'ls-round')


    #single_execute(get_parameters('IA'),'ls-round',np.sqrt(2))
    #single_execute(get_parameters('NC'),'ls-round',np.sqrt(2))
    #single_execute(get_parameters('MA'),'ls-round',np.sqrt(2))
    #single_execute(get_parameters('CT'),'ls-round',np.sqrt(2))
