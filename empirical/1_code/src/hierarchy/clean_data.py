import numpy as np
import pickle
import json
import os, re
POPCOLUMN = 'TOTPOP'
POSTPROCESS = 'hierarchy'

PROJECT = 'ramm'

# set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

def numpyconverter(obj):
    
    # Convert numpy to a Python int/float/list before serializing the object:
   
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

def convert_to_json(inputfiles, outputfile):
    
    dic = {}
    for bgt_split in inputfiles.keys():
        dic[bgt_split] = {}
        dic[bgt_split][POSTPROCESS] = {}
        for eps in inputfiles[bgt_split].keys():
            with open(inputfiles[bgt_split][eps],'rb') as f:
                data = pickle.load(f)
            dic[bgt_split][POSTPROCESS][eps] = [ x.round().astype(int) for x in data['vtd']['Identity'] ]
    
    with open(outputfile, 'w') as f:
        json.dump(dic, f,  indent=4, default=numpyconverter)

def run_IA():

    epsilons = np.logspace(-5, 0, num=6)
    state = 'IA'
    bgt_splits = ['equal', 'vtd-heavy2']
    seed = None
    inputfiles = {}
    for bgt in bgt_splits:
        inputfiles[bgt] = {}
        for eps in epsilons:
            inputfiles[bgt][eps] = 'empirical/2_pipeline/hierarchy/strategy_topdown/tmp/nonneg/adjusted_level2_pop_'+state+'_eps'+str(eps)+'_'+bgt+'_seed'+str(seed)+'.pickle'
    
    outputfile = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+state+'_topdown.json'

    convert_to_json(inputfiles, outputfile)

def run_NC():

    epsilons = np.logspace(-5, 0, num=6)
    state = 'NC'
    bgt_splits = ['equal', 'vtd-heavy4']
    seed = None
    inputfiles = {}
    for bgt in bgt_splits:
        inputfiles[bgt] = {}
        for eps in epsilons:
            inputfiles[bgt][eps] = 'empirical/2_pipeline/hierarchy/strategy_topdown/tmp/nonneg/adjusted_pop_'+state+'_eps'+str(eps)+'_'+bgt+'_seed'+str(seed)+'.pickle'
    
    outputfile = 'empirical/2_pipeline/generate_dpcounts/store/dptotpop_'+state+'_topdown.json'

    convert_to_json(inputfiles, outputfile)


if __name__ == '__main__':

    #run_IA()
    run_NC()
