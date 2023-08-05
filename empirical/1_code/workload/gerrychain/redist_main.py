import geopandas
import numpy as np
import os, re, math, time

from redist_utility import *

### Set up ####
##############

NAME = 'redistricting/redist_main'
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

#####################
####################3


def get_params(state):
    
    params = {}
    
    if state == 'IA':
        
        params['shapefile'] = 'empirical/0_data/external/IA-shapefiles/IA_counties/IA_counties.shp'
        params['pop column'] =  'TOTPOP'
        params['assignment column'] = 'CD'
        params['parts'] = ['1','2','3','4']
        params['election'] = {'name':'PRES16','dic':{"Democratic":"PRES16D","Republican":"PRES16R"}}
        params['pop bound'] = 0.02
        params['total steps'] = 10000
        params['stats'] = {'column':'PRES16', 'path': os.path.join(pipeline,'store','IA_recom_pop0.02_stat.pkl')}
        params['assignment output'] = os.path.join(pipeline, 'store', 'IA_assignment_recom_pop0.02.pkl')
    
    elif state == 'TX':
        
        params['shapefile'] = 'empirical/0_data/external/TX-shapefiles/TX_vtds/TX_vtds.shp'
        params['pop column'] =  'TOTPOP'
        params['assignment column'] = 'USCD'
        params['parts'] =['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22','23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36']
        params['election'] = {'name':'PRES16','dic':{"Democratic":"PRES16D","Republican":"PRES16R"}}
        params['pop bound'] = 0.02
        params['total steps'] = 10000
        params['stats'] = {'column':'PRES16', 'path': os.path.join(pipeline,'store','TX_recom_pop0.02_stat.pkl')}
        params['assignment output'] = os.path.join(pipeline, 'store', 'TX_assignment_recom_pop0.02.pkl')

    return params
        
def execute(params):
    
    
    graph = create_graph(params['shapefile'])
    
    elections = [
        Election(params['election']['name'],params['election']['dic'])
    ]
    chain = redistChain(graph,params['pop column'],params['assignment column'],params['parts'],
                       pop_bound = params['pop bound'], total_steps = params['total steps'], elections = elections)
    
    chain.set_chain()
    chain.run_chain(params['assignment output'],params['stats'])
    
    
if __name__ == "__main__":
    
    #execute(get_params('IA')) # 6min
    execute(get_params('TX')) #59min
    
    
    