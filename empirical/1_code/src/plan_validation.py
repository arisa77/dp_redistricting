from baseline import mechanism as baseline_mechanism 
from hdmm_mec import mechanism as  hdmm_mechanism
from utility import random_sample_workload, get_subworkloads,encode_ensembles
from constraints import compute_overall_range
import numpy as np
import matplotlib.pyplot as plt
import pickle
import geopandas

def compute_mse_workload(W,x,noisy_data,district_size,column='TOTPOP'):
    
    #compute mean squared error (averaged across trials) of workload for each engine/post-processing
    
    mse = {}

    domain_size = W.shape[1]
    Ws = W.reshape(-1,district_size,domain_size)
    
    for engine in noisy_data.keys():
        errs = []
        for eps in noisy_data[engine].keys():
            xests = np.vstack(noisy_data[engine][eps][column]).T #shape = (domain_size,trials)
            errs.append(np.mean((Ws.dot(xests).transpose(2,0,1) - Ws.dot(x))**2)) #Ws.dot(xests) shape = (numplans,subsize,trials)
        mse[engine] = errs
    
    return mse
        
def validate_plan_ep(W,x,noisydata,threshold,epsilons,district_size,pop_column):
    
    acc = {'valid':[],'invalid':[]} #returns a list of list of accuracy; 
                                    #inner list size is len(epsilon) and a outer list size is num_plans
    dis = {'valid':[],'invalid':[]} #returns a list of absolute distance from threshold for each plan; list size is num_plans
    
    #W = shape =()
    
    domain_size = W.shape[1]
    Ws = W.reshape(-1,district_size,domain_size)
    
    gt_pop = Ws.dot(x) #shape = (num_plans,district_size)
    gt_ovg = (np.amax(gt_pop,axis=1) - np.amin(gt_pop,axis=1))/np.mean(gt_pop,axis=1)#shape = (num_plans,)
    #print(gt_ovg)
    valid_args = np.argwhere(gt_ovg <= threshold).reshape(-1)
    invalid_args = np.argwhere(gt_ovg > threshold).reshape(-1)
    for eps in epsilons:
        xests = np.stack(noisydata[eps])
        if xests.shape[1] == 1:
            xests = xests.transpose(1,0,2)[0] #(trials,domain_size) 
        
        noisy_pop = Ws.dot(xests.T) #shape = (num_plans,district_size,trials)
        noisy_ovg = (np.amax(noisy_pop,axis=1) - np.amin(noisy_pop,axis=1))/(np.mean(noisy_pop,axis=1))#shape = (num_plans,trials)
        
        outcome = np.where(np.tile(gt_ovg,(xests.shape[0],1)).T <= threshold, noisy_ovg <= threshold, noisy_ovg > threshold)#(num_plans,trials)
        
        acc['valid'].append(np.mean(outcome,axis=1)[valid_args])
        acc['invalid'].append(np.mean(outcome,axis=1)[invalid_args])
        
    distance = np.abs(threshold - gt_ovg)
    dis['valid'] = distance[valid_args]
    dis['invalid']= distance[invalid_args]
    
    acc['valid'] = np.vstack(acc['valid']) #(#epsilons,#plans) 
    acc['invalid'] = np.vstack(acc['invalid'])  
        
    return acc,dis
        
    '''
    for w in get_subworkloads(W,sub_size = district_size):
        gt_ovg = compute_overall_range(w.dot(x))
        accuracy = []
        for eps in epsilons:
            accurate = []
            for x_hat in noisydata[eps][pop_column]:

                if gt_ovg <= threshold: #valid on true counts
                    if compute_overall_range(w.dot(x_hat)) <= threshold:
                        accurate.append(1)
                    else:
                        accurate.append(0)
                else: #invalid on true counts
                    if compute_overall_range(w.dot(x_hat)) > threshold:
                        accurate.append(1)
                    else:
                        accurate.append(0)
            accuracy.append(np.mean(accurate))
        if gt_ovg <= threshold:
            acc['valid'].append(accuracy)
            dis['valid'].append(np.abs(threshold-gt_ovg))
        else:
            acc['invalid'].append(accuracy)
            dis['invalid'].append(np.abs(threshold-gt_ovg))
    
    return acc,dis
    '''
def validate_plan_mm(W,gtdf,noisyvaps,threshold,epsilons,district_size,minority_columns):
    
    acc = {'valid':[],'invalid':[]} #returns a list of list of accuracy; 
                                    #inner list size is len(epsilon) and a outer list size is num_plans
    
    domain_size = W.shape[1]
    Ws = W.reshape(-1,district_size,domain_size)
    
    minority_indices = [list(gtdf.columns).index(item) for item in minority_columns]
    
    
    minority = gtdf[minority_columns].sum(axis=1).values #(domain_size,)
    total = gtdf.sum(axis=1).values #(domain_size,)
    gt_num_MM = np.count_nonzero(Ws.dot(minority)/Ws.dot(total) > 0.5,axis=1) #(num_plans)
    
    valid_args = np.argwhere(gt_num_MM >= threshold).reshape(-1)
    invalid_args = np.argwhere(gt_num_MM < threshold).reshape(-1)
    
    for eps in epsilons:
        #xests = []
        #minority_indices = []
        #for i,race in enumerate(noisyvaps[eps].keys()):#convert a list of list into 2d array
        #    if race in minority_columns:
        #        minority_indices.append(i)
        
        xests = np.stack(noisyvaps[eps])
        if xests.shape[1] == len(gtdf.columns):
            xests = xests.transpose(1,0,2) #(num_races,trials,domain_size) 
        
        total = np.sum(xests,axis=0) #(num_races,trials,domain_size) ->(trials,domain_size)
        minority = np.sum(np.stack(xests)[minority_indices],axis=0) #(num_races,trials,domain_size) ->(trials,domain_size)
        
        num_MM = np.count_nonzero(Ws.dot(minority.T)/Ws.dot(total.T) > 0.5,axis=1) #(num_plans,district_size,trials) -> (num_plans,trials)
        
        outcome = np.where(np.tile(gt_num_MM,(num_MM.shape[1],1)).T >= threshold, num_MM >=threshold, num_MM < threshold)#(num_plans,trials)
        acc['valid'].append(np.mean(outcome,axis=1)[valid_args])
        acc['invalid'].append(np.mean(outcome,axis=1)[invalid_args])
        
    acc['valid'] = np.vstack(acc['valid']) #(#epsilons,#plans) 
    acc['invalid'] = np.vstack(acc['invalid'])    
    return acc
    '''
    for w in get_subworkloads(W,sub_size = district_size):
        accuracy = []
        gt_num_MM = np.count_nonzero(w.dot(truedata['nonwhite'])/w.dot(truedata['total']) > 0.5)
        
        for eps in epsilons:
            
            xests = {}
            minority_indices = []
            for i,race in enumerate(noisydata[eps].keys()):#convert a list of list into 2d array
                xests[race] = np.vstack(noisydata[eps][race]) #(trials,domain_size)
                
            total = np.sum(np.stack(list(xests.values())),axis=0) #(num_races,trials,domain_size) ->(trials,domain_size)
            nonwhite = total - xests[white_column] # (trials,domain_size)
            
            dis_total = w.dot(total.T)
            frac_nonwhite = w.dot(nonwhite.T)/np.where(dis_total==0,1,dis_total) #(district_size,trials)
            num_MMs = np.count_nonzero(frac_nonwhite > 0.5,axis=0) #trials,
            
            if gt_num_MM >= threshold:#valid on true counts
                accuracy.append(np.mean(num_MMs >= threshold))
            else: #invalid on true counts
                accuracy.append(np.mean(num_MMs <  threshold))
                
        if gt_num_MM >= threshold:
            acc['valid'].append(accuracy)
        else:
            acc['invalid'].append(accuracy)
            
    return acc
    '''
class experiment():
    
    _W_train,_W_test = None,None
    
    def __init__(self,state):
        
        self.set_parameters(state)
        
    def set_parameters(self,state):

        params = {}

        if state == 'NC':
            params['state'] = 'NC'
            params['district size'] = 13
            params['epsilons'] = np.logspace(-4, 0, num=8)
            params['trials'] = 20
            params['pop columns'] = ['TOTPOP']
            params['vap columns'] = ['HVAP','WVAP','BVAP','AMINVAP','ASIANVAP','NHPIVAP','OTHERVAP','2MOREVAP']
            params['white column'] = 'WVAP'
            params['nonwhite columns'] = ['HVAP','BVAP','AMINVAP','ASIANVAP','NHPIVAP','OTHERVAP','2MOREVAP']
            params['preplan column'] = 'newplan'
            params['engines'] = ['ls','ls-round','ls-normalize','nnls']#,'wnnls']
            params['shapefile'] = 'empirical/0_data/external/NC-shapefiles/NC_VTD/NC_VTD.shp'
            params['workload path'] = 'empirical/2_pipeline/train_test_split/store/dataset_NC.pickle'
            params['train size'] = 100
            params['proposal'] = 'NC-recom'
            params['index'] = 0
            params['strategy path'] = 'empirical/2_pipeline/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
            params['output'] = 'empirical/2_pipeline/plan validation/store/noisy_results_'+state+'.pickle'
            params['pipeline'] = 'empirical/2_pipeline/plan validation/store/'

        elif state == 'IA':
            params['state'] = 'IA'
            params['district size'] = 4
            params['epsilons'] = np.logspace(-4, 0, num=8)
            params['trials'] = 20
            params['pop columns'] = ['TOTPOP']
            params['vap columns'] = ['HVAP','WVAP','BVAP','AMINVAP','ASIANVAP','NHPIVAP','OTHERVAP','2MOREVAP']
            params['engines'] = ['ls','ls-round','ls-normalize','nnls']#,'wnnls']
            params['shapefile'] = 'empirical/0_data/external/IA-shapefiles/IA_counties/IA_counties.shp'
            params['workload path'] = 'empirical/2_pipeline/train_test_split/store/dataset_IA.pickle'
            params['train size'] = "USE ALL"
            params['proposal'] = 'IA-recom'
            params['index'] = 0
            params['strategy path'] = 'empirical/2_pipeline/13_error_on_various_proposals/store/strategy-'+params['proposal']+"-"+str(params['index'])+'.pickle'
            params['output'] = 'empirical/2_pipeline/plan_validation/store/noisy_results_'+state+'.pickle'
            params['pipeline'] = 'empirical/2_pipeline/plan_validation/store/'

        self._params = params
        
        self._W_train,self._W_test = None,None
        
    def get_params(self):
        
        return self._params

    def get_workload(self):
        
        if self._W_train is not None and self._W_test is not None :
            return self._W_train,self._W_test
        else:
            with open(self._params['workload path'],'rb') as f:
                W = pickle.load(f)[self._params['proposal']][self._params['index']]
            return W['train'].astype(float),W['test'].astype(float)

    def execute(self,pop_types='pop'):

        params = self._params
        with open(params['workload path'],'rb') as f:
            W = pickle.load(f)[params['proposal']][params['index']]

        W_train = W['train'].astype(float)
        W_test = W['test'].astype(float)
        
        if pop_types == 'pop':
            columns = params['pop columns']
        else:
            columns = params['vap columns']

        if isinstance(params['train size'],int):
            W_train = random_sample_workload(W_train,params['train size'],params['district size'])
        
        self._W_train = W_train
        self._W_test = W_test

        #baseline mechanism -- Identity
        print("baseline")
        baselineMec = baseline_mechanism(params['state'],params['district size'],params['shapefile'])
        baselineMec.set_W(W_train)
        baseline_results = {}
        for engine in params['engines']:
            print(engine)
            baseline_results[engine] = baselineMec.run(columns,params['epsilons'],engine=engine,trials=params['trials'])


        #HDMM -- p-Identity
        print("hdmm")
        hdmmMec = hdmm_mechanism(params['state'],params['district size'],params['shapefile'])
        hdmmMec.set_W(W_train)
        hdmmMec.set_strategy(params['strategy path'])
        hdmm_results = {}
        for engine in params['engines']:
            print(engine)
            hdmm_results[engine] = hdmmMec.run(columns,params['epsilons'],engine=engine,trials=params['trials'])

        self._results = {}
        self._results['baseline'] = baseline_results
        self._results['hdmm'] = hdmm_results
        
        with open(params['output'],'wb') as f:
            pickle.dump(self._results,f, pickle.HIGHEST_PROTOCOL)
        
        return baseline_results,hdmm_results
    
    def measure_error(self,column,plot_train=False):

        if plot_train:
            W = self._W_train
        else:
            W = self._W_test
            
        x = geopandas.read_file(self._params['shapefile']).values
        
        noisy_results = self._results
        
        fig, axes = plt.subplots(nrows=1, ncols=len(noisy_results.keys()),figsize=(10,5),sharey=True)

        mses = {}
        for strategy in noisy_results.keys():
            
            mse = compute_mse_workload(W,x,noisy_results,self._params['district size'],column=column)
            mses[strategy] = mse
            for engine in self._params["engines"]:

                ratio = np.array(mse['ls'])/np.array(mse[engine])
                axes[0].plot(epsilons,ratio,label=engine)

            axes[i].legend()
            axes[i].set_xscale('log')
            axes[i].set_title('Identity')
            axes[i].set_xlabel('privacy loss')
            axes[i].set_ylabel('Ratio of MSE on workload')


        plt.show()   

        return mses
    
    def validate_equal_pop():
        
        
        x = geopandas.read_file(self._params['shapefile']).values
        
        #compute threshold by simply dividning ensembles into two sets
        ovrgs = []
        for w in get_subworkloads(self._W_test,self._params['district size']):
            ovrgs.append(compute_overall_range(w.dot(x)))
        print('threshold is ', np.median(ovrgs))
        threshold = np.median(ovrgs)
        
        
        noisy_results = self._results
        accuracy = {}
        for strategy in noisy_results.keys():
            acc = {}
            for engine in self._params['engines']:
                acc[engine],_ = validity_check_ep(self._W_test,x,noisy_results[strategy][engine],threshold,self._params['epsilons'],self._params['district size'],self._params['pop columns'][0])
                
            accuracy[strategy] = acc
            
        #Plot strategy comparison in epsilon vs accuracy for each engine (valid plans)
        fig, axes = plt.subplots(nrows=1, ncols=len(self._params['engines']),figsize=(30,5),sharey=True)
        for i,engine in enumerate(self._params['engines']):
            
            for strategy in noisy_results.keys():
                result = np.mean(np.vstack(accuracy[strategy][engine]['valid']),axis=0)
                axes[i].plot(epsilons,result,label=strategy,marker='x')

        
            axes[i].legend()
            axes[i].set_xscale('log')
            axes[i].set_xlabel("Privacy loss")
            axes[i].set_ylabel("Expected accuracy")
            axes[i].set_title(engine)
        
        plt.show()
        
        #Plot strategy comparison in epsilon vs accuracy for each engine (invalid plans)
        fig, axes = plt.subplots(nrows=1, ncols=len(self._params['engines']),figsize=(30,5),sharey=True)
        for i,engine in enumerate(self._params['engines']):
            
            for strategy in noisy_results.keys():
                result = np.mean(np.vstack(accuracy[strategy][engine]['invalid']),axis=0)
                axes[i].plot(epsilons,result,label=strategy,marker='x')

        
            axes[i].legend()
            axes[i].set_xscale('log')
            axes[i].set_xlabel("Privacy loss")
            axes[i].set_ylabel("Expected accuracy")
            axes[i].set_title(engine)
        
        plt.show()
        
        #plot all the strategy and engine results into a single plot (valid plans)
        markers = ['o','x','+','^','s','v']
        colors = ['black','red','blue']
        for i,engine in enumerate(self._params['engines']):
            for j,trategy in enumerate(noisy_results.keys()):
                result = np.mean(np.vstack(accuracy[strategy][engine]['valid']),axis=0)
                plt.plot(epsilons,result,label=engine + '('+strategy+')',color = colors[j],marker = markers[i])

        plt.legend()
        plt.xscale('log')
        plt.xlabel("Privacy loss")
        plt.ylabel("Expected accuracy")
        
        
        #plot all the strategy and engine results into a single plot (invalid plans)
        markers = ['o','x','+','^','s','v']
        colors = ['black','red','blue']
        for i,engine in enumerate(self._params['engines']):
            for j,trategy in enumerate(noisy_results.keys()):
                result = np.mean(np.vstack(accuracy[strategy][engine]['invalid']),axis=0)
                plt.plot(epsilons,result,label=engine + '('+strategy+')',color = colors[j],marker = markers[i])

        plt.legend()
        plt.xscale('log')
        plt.xlabel("Privacy loss")
        plt.ylabel("Expected accuracy")
        
        
    def validate_mm(self):
        
        #compute threshold; num_mm on previous plans 
        gdf = geopandas.read_file(self._params['shapefile'])
        vap_races = gdf[self._params['vap columns']]
        total = vap_races.sum(axis=1).values
        white = gdf[self._params['white column']].values
        nonwhite = total-white
        W_pre = encode_ensembles([gdf[self._params['preplan column']].to_dict()]) #(district_size,domain_size)
        frac_nonwhite = W_pre.dot(nonwhite)/W_pre.dot(total)
        threshold = np.count_nonzero(frac_nonwhite > 0.5)
        print('threshold is ',threshold)

        
        noisy_results = self._results
        accuracy = {}
        for strategy in noisy_results.keys():
            acc = {}
            for engine in self._params['engines']:
                
                acc[engine] = validate_plan_mm(
                    self._W_test,
                    {'nonwhite':nonwhite,'total':total},
                    noisy_results[strategy][engine],
                    threshold,
                    self._params["epsilons"],
                    self._params['district size'],
                    self._params['white column'])
            accuracy[strategy] = acc
            
        return accuracy