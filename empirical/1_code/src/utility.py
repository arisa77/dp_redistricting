import init
import numpy as np
from hdmm import templates, error, more_templates,inference
from ektelo import workload
from ektelo.matrix import EkteloMatrix,VStack
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
import pickle
import seaborn as sns
from sklearn.metrics import pairwise_distances
import networkx as nx


def noisy_measurement(A,x,eps=np.sqrt(2),trials=1,seed=200):   
    prng = np.random.RandomState(seed)
    delta = A.sensitivity()
    A = A.sparse_matrix()
    y = [A.dot(x)+prng.laplace(loc=0,scale=delta/eps,size=A.shape[0]) for i in range(trials)]
    return y

def noisy_answer(W,x,eps=np.sqrt(2),trials=1,seed=200):  
    
    prng = np.random.RandomState(seed)
    delta = max(np.sum(W,axis=0))#sensitivity
    
    y = [W.dot(x)+prng.laplace(loc=0,scale=delta/eps,size=W.shape[0]) for i in range(trials)]
    return y

def inference_nnls_eq(W,A, y, th,sub_size,pr = 8e9,l1_reg=0, l2_reg=0, maxiter = 15000):
    
  
    def loss_and_grad(x):
        
        diff = A.dot(x)-y # shape = (# of queries,)
        res = 0.5 * np.sum(diff ** 2)
        overall_range = np.asarray([compute_overall_range(V.dot(x)) for V in get_subworkloads(W,sub_size)]) # shape = (# of plans,)
        penalty = np.sum(np.maximum(overall_range-th,0))
        f = res + pr*penalty + l1_reg*np.sum(x) + l2_reg*np.sum(x**2)
        #print(f,penalty)
        return f
    
    xinit = lsmr(A,y)[0]
    
    bnds = [(0,None)]*W.shape[1]
    xest,_,info = optimize.lbfgsb.fmin_l_bfgs_b(loss_and_grad,
                                                approx_grad = True,
                                                x0=xinit,
                                                pgtol=1e-4,
                                                bounds=bnds,
                                                maxiter=maxiter,
                                                m=1,
                                                )
    xest[xest < 0] = 0.0
    return xest

def inference_wnnls_eq(W,A, y, th,sub_size,pr = 5e10,l1_reg=0, l2_reg=0, maxiter = 15000):
    
   
    
    xhat = lsmr(A,y)[0]
    yhat = W.dot(xhat)
            
    def loss_and_grad(x):
        
        diff = W.dot(x)-yhat # shape = (# of queries,)
        res = 0.5 * np.sum(diff ** 2) 
        overall_range = np.asarray([compute_overall_range(V.dot(x)) for V in get_subworkloads(W,sub_size)]) # shape = (# of plans,)
        penalty = np.sum(np.maximum(overall_range-th,0))
        f = res + pr*penalty + l1_reg*np.sum(x) + l2_reg*np.sum(x**2)
        
        #print(f,penalty)
        return f

    xinit = xhat
    bnds = [(0,None)]*W.shape[1]
    xest,_,info = optimize.lbfgsb.fmin_l_bfgs_b(loss_and_grad,
                                                approx_grad = True,
                                                x0=xinit,
                                                pgtol=1e-4,
                                                bounds=bnds,
                                                maxiter=maxiter,
                                                m=1,
                                                
                                                )
    xest[xest < 0] = 0.0
    return xest

def post_process(W,A,ys,engine='ls',integer = True, args = None):
    
    Apinv = A.pinv()
    A = A.sparse_matrix()
        
    xests=[]
    for i in range(len(ys)):
        if engine == 'ls':
            xest = Apinv.dot(ys[i])
        elif engine == 'ls-round':
            xest = Apinv.dot(ys[i])
            xest = np.clip(xest, a_min =0, a_max = np.inf)
        elif engine == 'ls-normalize':
            xest = Apinv.dot(ys[i])
            sum_xest = np.sum(xest)
            xest = np.clip(xest, a_min =0, a_max = np.inf)
            xest = xest*sum_xest/np.sum(xest)
        elif engine == 'ls-normalize-budget':#sum = noisyly released total counts 
            xest = Apinv.dot(ys[i])
            sum_xest = args['total counts'][i]
            xest = np.clip(xest, a_min =0, a_max = np.inf)
            xest = xest*sum_xest/np.sum(xest)
        elif engine == 'nnls':
            xest = inference.nnls(A,ys[i])
        elif engine == 'wnnls':
            xest = inference.wnnls(W,A,ys[i])
        elif engine == 'nnls_eq':
            xest = inference_nnls_eq(W,A,ys[i],args['sub size'],args['penalty rate'])
        elif engine == 'wnnls_eq':
            xest = inference_wnnls_eq(W,A,ys[i],args['sub size'],args['penalty rate'])
        else:
            break
        
        if integer:
            xest = xest.round().astype(np.int)
        xests.append(xest)
        
        
    return xests

def opt_row_weighted(W, rows):
    #A = more_templates.RowWeighted(EkteloMatrix(rows))
    pid = more_templates.RowWeighted(rows)
    WTW = workload.ExplicitGram(W.T.dot(W))
    pid.optimize(WTW)
    #pid.optimize(EkteloMatrix(W))
    return pid.A

def opt_p_identity(W=None, p=4,restarts=None):
    
    WTW = workload.ExplicitGram(W.T.dot(W))
    
    pid = templates.PIdentity(p, W.shape[1])
    
    if restarts is None:
        pid.optimize(WTW)
    else:
        pid.restart_optimize(WTW, restarts)
    return pid.strategy()

def opt_aug_identity(W,imatrix,restarts=None):
    
    
    W = workload.ExplicitGram(W.T.dot(W))
    
    aid = templates.AugmentedIdentity(imatrix=imatrix)
    
    init = np.zeros(aid._params.size)
    k = len(init)-1
    init[:k] = np.linspace(0.5, 0, k)
    
    if restarts is None:    
        aid.optimize(W,init=init)
    else:
        aid.restart_optimize(W,restarts)
    print(aid._params)
    return aid.strategy(),aid._params



def check_row_match(matrix,row):
    index=[]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if row[j] != matrix[i][j]:
                break;
        if j==matrix.shape[1]-1:
            index.append(i)
    return index

def check_match(matrix,target_matrix):
    indices=[] #indices[i] ia a list of indices of matrix with which target_matrix[i] matches
    for i in range(target_matrix.shape[0]):
        index = check_row_match(matrix,target_matrix[i])
        indices.append(index)
        #print(str(i)+":")
        print(index)
'''
def check_duplicates(W1s,W2s,sub_size=13):
    
    def check(W1,W2):
        # return true if the two plans are the same
        for i in range(sub_size):
            if np.count_nonzero(np.sum((W1-W2[i]),axis=1))!=0:
                return False
        return True
    plan_ids=[]
    for j in range(int(W2s.shape[0]/sub_size)):
        #print(j)
        for i in range(int(W1s.shape[0]/sub_size)):
            re=check(W1s[i*sub_size:(i+1)*sub_size,:],W2s[j*sub_size:(j+1)*sub_size,:])
            if re==True:
                plan_ids.append(re)
                break;
    return plan_ids
'''
def check_self_duplicates(Ws,sub_size=13):
    
    Ws = get_subworkloads(Ws,sub_size)
    duplicates = []
    for i in range(len(Ws)-1):
        for j in range(i+1,len(Ws)):
            if np.array_equal(Ws[i],Ws[j]):
                duplicates.append((i,j))
    return duplicates

def check_duplicates(W1s,W2s,sub_size=13):
    
    W1s = get_subworkloads(W1s,sub_size)
    W2s = get_subworkloads(W2s,sub_size)
    duplicates = [] # a duplicate pair (id in W1s, id in W2s)
    for i in range(len(W1s)):
        for j in range(len(W2s)):
            if np.array_equal(W1s[i],W2s[j]):
                duplicates.append((i,j))
    return duplicates

def check_duplicates_per_query(W1s,W2s):
    

    duplicates = [] # a duplicate pair (id in W1s, id in W2s)
    for i in range(W1s.shape[0]):
        for j in range(W2s.shape[0]):
            if np.array_equal(W1s[i],W2s[j]):
                duplicates.append((i,j))
    return duplicates

def all_pairs_likelihood(plans):
    #compute the likelihood that two vtds are in the same district
    
    n,d = plans.shape
    
    PL = np.zeros((d,d))
    
    for i in range(n):
        for j in range(d):
            if plans[i][j]==1:
                for k in range(j+1,d):
                    if plans[i][k]==1:
                        PL[j][k]+=1
                        PL[k][j]+=1
    
    PL=PL+np.eye(d)*np.max(PL)
    
    PL = PL.astype(int)
    return PL

def get_partitions_similarity(P1,P2):
    
    # two paris of set partitions of partitining m elements into k groups
    # each partition is a k by m binary matrix P and P[i][j] indicates that j-th element is assinged to i-th group
    
    #return similarity of two partitions P1 and P2 between [0,1]. 
    #0 reperesents P1 and P2 are completely different
    #1 means P1 and P2 are completely the same partitioning
    
    similarity = 0
    
    #compute a jaccard similarity of every combination of the groups,one from P1 and another from P2.
    P1=P1.astype(bool)
    P2=P2.astype(bool)
    distance=1-pairwise_distances(P1,P2,metric='jaccard')
    
    #build a wegihted bipartite graph
    
    G = nx.Graph()
    a=['a'+str(i) for i in range(len(distance))]
    b=['b'+str(j) for j in range(len(distance[0]))]
    G.add_nodes_from(a,bipartite=0)
    G.add_nodes_from(b,bipartite=1)

    for i in range(len(distance)):
        for j in range(len(distance[i])):
                G.add_edge(a[i], b[j],weight=distance[i][j])
                
    matchings=nx.max_weight_matching(G)
    
    for edge in matchings:
        similarity+=G[edge[0]][edge[1]]['weight']
    similarity = similarity/len(matchings)
        
    return similarity

def get_subworkloads(Ws,sub_size=13):
    W = []
    num_plans = int(Ws.shape[0]/sub_size)
    for i in range(num_plans):
        offset=i*sub_size
        W.append(Ws[offset:offset+sub_size,:])
        
    return W
def uniform_sample_workload(workload,sample_size,sub_size=13):
    
    if workload.shape[0]<sample_size:
        print('sample size too big')
        return -1
    
    nplans=int(workload.shape[0]/sub_size)
    offsets=np.linspace(0,nplans-1,num=sample_size,dtype=int)
    samples = np.vstack([workload[i*sub_size:(i+1)*sub_size] for i in offsets])
 
    
    return samples
def random_sample_workload(workload,sample_size,sub_size=13,seed=100):
    
    np.random.RandomState(seed)
    
    nplans=int(workload.shape[0]/sub_size)
    if nplans<sample_size:
        print('sample size too big')
        return -1
    if nplans == sample_size:
        return workload
    
    offsets=np.random.choice(nplans-1, sample_size, replace=False)    
    samples = np.vstack([workload[i*sub_size:(i+1)*sub_size] for i in offsets])
    
    return samples

def random_sample(matrix,sample_size):
    
    return matrix[np.random.choice(matrix.shape[0],sample_size,replace=False)]

def train_test_split(workload,test_size=0.2,sub_size=13,seed=100):
    
    np.random.RandomState(seed)
    
    nplans=int(workload.shape[0]/sub_size)
    train_offsets = np.random.choice(nplans, int(nplans*(1-test_size)), replace=False)
    test_offsets = list(set(np.arange(nplans)) - set(train_offsets))
    train_workload= np.vstack([workload[i*sub_size:(i+1)*sub_size] for i in train_offsets])
    test_workload= np.vstack([workload[i*sub_size:(i+1)*sub_size] for i in test_offsets])
    
    return train_workload,test_workload
'''
def subworkload_error(W,A,eps=np.sqrt(2),sub_size=13):
    
    ew=[]
    num_plans = int(W.shape[0]/sub_size)
    
    W_list = get_subworkloads(W,sub_size=sub_size)
    for w in W_list:
        ew.append(error.per_query_error(EkteloMatrix(w),A,eps=eps,normalize=True))
         
    return ew
'''
def per_subworkload_error(W,A,eps=np.sqrt(2),sub_size=13):
    
    ew = []
    num_plans = int(W.shape[0]/sub_size)
    
    err = error.per_query_error(EkteloMatrix(W),A,eps=eps,normalize=True)
    for i in range(num_plans):
        offset=i*sub_size
        ew.append(err[offset:offset+sub_size])
     
    return ew


def per_data_error(A, eps=np.sqrt(2), delta=0, normalize=True):
    # calculate error on data (per domain) instead of workload
    if not (isinstance(A, EkteloMatrix) or isinstance(A, workload.ExplicitGram)):
        A = EkteloMatrix(A)
        
    delta = A.sensitivity()
    var = 2.0/eps**2
    AtA1 = A.gram().pinv()
    err = AtA1.diag()
    answer = var * delta**2 * err
    return np.sqrt(answer) if normalize else answer

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

def encode_ensembles(vtd_lists):
    # vtd_lists are the list of a dictionary whose keys are nodes and values are ID for congressional districts where the nodes are assigned
    W_r = []
    for vtd_list in vtd_lists:
        if type(vtd_list) is not list:
            vtd_list = [vtd_list[node] for node in sorted(vtd_list.keys())]
        w = encode_workload(vtd_list)
        W_r.append(w)

    W_r = np.vstack(tuple(W_r))

    return W_r

def analyze_error(W,strategies,labels,eps=np.sqrt(2)):
    
    
    n = W.dense_matrix().shape[0]
    df = pd.DataFrame({'district id':list(range(1,n+1)),
                      'num vtds':list(np.sum(W.dense_matrix(),axis=1).T)})
    
    for label in labels:
        
        A = strategies[label]
        W_error = error.rootmse(W,A,eps=eps)
        q_errors = error.per_query_error(W,A,eps=eps,normalize=True)
        
        df[label]=q_errors
        
    return df

def get_error_variance(data,labels):
    
    d = []
    keys=['label','overall error','error variance']
    for label in labels:
        
        overall_error = np.mean(list(data[label].values))
        error_variance = np.var(list(data[label].values))
        
        d.append(dict(zip(keys,[label,overall_error,error_variance])))
        
        
    return pd.DataFrame(d,columns=['label', 'overall error','error variance'])
    
def plot_errors2(W,strategies,eps=np.sqrt(2)):
    d = []
    n = W.dense_matrix().shape[0]
    df = pd.DataFrame({'district id':list(range(1,n+1)),
                      'num vtds':list(np.sum(W.dense_matrix(),axis=1).T)})
   
    
    for key, A in strategies.items():
        
        W_error = error.rootmse(W,A,eps=eps)
        q_errors = error.per_query_error(W,A,eps=eps,normalize=True)
        var = np.var(q_errors)
        df[key]=q_errors
        
        #ax = sns.lineplot(x = list(range(len(err))),y = sorted(err),palette = pal)
             
        d.append(dict(label = key,error = W_error, variance = var))
    
    
    #print(df)
    #return df
    SMALL_SIZE = 8
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 12

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #params = {'legend.fontsize': 15,
    #      'figure.figsize': (10, 5),
    #     'axes.labelsize': 15,
    #     'axes.titlesize':15,
    #     'xtick.labelsize':15,
    #     'ytick.labelsize':15}
    #plt.rcParams.update(params)
    
    #sort in accending order
    df=df.sort_values('num vtds')
    
    
    sns.set_style('ticks')
    pal = sns.color_palette('Greens_d',1)
    #ax = df[['district id','Identity','p-Identity']].plot(kind='bar',x='district id',figsize=(10, 4), legend=True, fontsize=12)
    ax = df[['district id','Identity','HDMM']].plot(kind='bar',x='district id',figsize=(10, 4), legend=True, fontsize=12)
    ax.axhline(y=np.mean(list(df['Identity'].values)), label='Identity',linestyle='--',color='blue')
    ax.axhline(y=np.mean(list(df['HDMM'].values)),label='HDMM',linestyle='--',color='orange')
    ax.set(ylabel='Error')
    #ax.get_xaxis().set_ticklabels([])
    #plt.legend()
    plt.savefig("plot/error_vs_district_"+str(eps)+".png", dpi=300, bbox_inches='tight')
    plt.show()
    
    sns.set_style('whitegrid')
    fig,ax=plt.subplots(1, 1,figsize=(6,3))
    x=list(range(0,n))
    #ax.plot(x,list(df['Identity'].values), color="blue", label="Identity", linestyle="-")
    #ax.plot(x,list(df['p-Identity'].values), color="red", label="p-Identity", linestyle="-")
    ax.plot(x,list(df['Identity'].values), color="blue", label="Identity (Baseline)", linestyle="-")
    ax.plot(x,list(df['HDMM'].values), color="red", label="HDMM", linestyle="-")
    
    ax.set(xlabel='district id',ylabel='Error')
    plt.xticks(x, list(df['district id'].values))
    ax.legend()
    plt.show()
    
    
    return pd.DataFrame(d,columns=['label', 'error','variance'])


def plot_errors(W,strategies,eps=np.sqrt(2)):
    d = []
    n = W.dense_matrix().shape[0]
    df = pd.DataFrame({'district id':list(range(1,n+1)),
                      'num vtds':list(np.sum(W.dense_matrix(),axis=1).T)})
   
    
    for key, A in strategies.items():
        
        W_error = error.rootmse(W,A,eps=eps)
        q_errors = error.per_query_error(W,A,eps=eps,normalize=True)
        #var = (np.max(q_errors)-np.min(q_errors))/np.mean(q_errors)
        var = np.var(q_errors)
        df[key]=q_errors
        
        #ax = sns.lineplot(x = list(range(len(err))),y = sorted(err),palette = pal)
        
                
        d.append(dict(label = key,error = W_error, variance = var))
    
    
    #print(df)
    #return df
    SMALL_SIZE = 8
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 12

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #params = {'legend.fontsize': 15,
    #      'figure.figsize': (10, 5),
    #     'axes.labelsize': 15,
    #     'axes.titlesize':15,
    #     'xtick.labelsize':15,
    #     'ytick.labelsize':15}
    #plt.rcParams.update(params)
    
    #sort in accending order
    df=df.sort_values('num vtds')
    
    
    sns.set_style('ticks')
    pal = sns.color_palette('Greens_d',1)
    #ax = df[['district id','Identity','p-Identity']].plot(kind='bar',x='district id',figsize=(10, 4), legend=True, fontsize=12)
    ax = df[['district id','Identity','HDMM']].plot(kind='bar',x='district id',figsize=(10, 4), legend=True, fontsize=12)
    ax.axhline(y=np.mean(list(df['Identity'].values)), label='Identity',linestyle='--',color='blue')
    ax.axhline(y=np.mean(list(df['HDMM'].values)),label='HDMM',linestyle='--',color='orange')
    ax.set(ylabel='Error')
    #ax.get_xaxis().set_ticklabels([])
    #plt.legend()
    plt.savefig("plot/error_vs_district_"+str(eps)+".png", dpi=300, bbox_inches='tight')
    plt.show()
    
    sns.set_style('whitegrid')
    fig,ax=plt.subplots(1, 1,figsize=(6,3))
    x=list(range(0,n))
    #ax.plot(x,list(df['Identity'].values), color="blue", label="Identity", linestyle="-")
    #ax.plot(x,list(df['p-Identity'].values), color="red", label="p-Identity", linestyle="-")
    ax.plot(x,list(df['Identity'].values), color="blue", label="Identity (Baseline)", linestyle="-")
    ax.plot(x,list(df['HDMM'].values), color="red", label="HDMM", linestyle="-")
    
    ax.set(xlabel='district id',ylabel='Error')
    plt.xticks(x, list(df['district id'].values))
    ax.legend()
    plt.show()
    
    
    return pd.DataFrame(d,columns=['label', 'error','variance'])

def plot2(err,name):
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,2))
    pal = sns.color_palette('Greens_d',1)
    ax = sns.lineplot(x = list(range(len(err))),y = sorted(err),palette = pal)
    #ax = sns.barplot(x = list(range(len(err))),y = sorted(err),palette = pal)
    ax.get_xaxis().set_ticklabels([])
    ax.set(ylabel='Error', title=f'Strategy = {name}')
    #plt.savefig(f'error_ensemble_old_{name}.png')
    #plt.savefig(f'error_ensemble_{name}.png')

    
    print(min(err), np.mean(err), max(err))


def evaluate_error(W,strategies):
    d = []
    for key, A in strategies.items():
        error_on_workload = error.rootmse(W,A)
        query_error_on_workload = error.per_query_error(W,A)
        #district_error = pd.DataFrame({'district':list(range(1,len(query_error_on_workload)+1)),'error':list(query_error_on_workload)})
        
        var_error_on_target = (np.max(query_error_on_workload)-np.min(query_error_on_workload))/np.mean(query_error_on_workload)
        plot2(query_error_on_workload,key)
        #plot(district_error,key)
        #district_error['num_vtds']=list(np.sum(W,axis=1).T)
        #print(district_error.sort_values(by=['error']))
        d.append(dict(label = key,error = error_on_workload, var = var_error_on_target))
    return pd.DataFrame(d)

def plot(err,total_err,name):
    
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

    sns.set_style('whitegrid')
    plt.figure(figsize=(10,2))
    pal = sns.color_palette('Greens_d',1)
    err=err.sort_values(by=['error'])
    ax = sns.barplot(x ='district',y = 'error',data=err,order=err['district'],palette = pal)
    #ax = sns.barplot(x = list(range(len(err))),y = sorted(err),palette = pal)
    #ax.axhline(total_err,color='red',ls="--")
    #ax.set(ylabel='Error', title=f'Strategy = {name}')
    ax.set_xlabel('District', fontsize=15)
    ax.set_ylabel('Error', fontsize=15)
 
    #plt.savefig(f'error_old_{name}.png')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',fontsize=15)
     
    plt.show()
    #plt.savefig('identity.png')
    
    #print(min(err['error']), np.mean(err['error']), max(err['error']))

def evaluate_error_with_districtID(W,strategies,eps=np.sqrt(2)):
    d = []
    for key, As in strategies.items():
        
        for A in As:
            print(A.shape,W.shape)
            error_on_workload = error.rootmse(W,A,eps=eps)
            query_error_on_workload = error.per_query_error(W,A,eps=eps,normalize=True)
            district_error = pd.DataFrame({'district':list(range(1,len(query_error_on_workload)+1)),'error':list(query_error_on_workload)})

            var_error_on_target = (np.max(query_error_on_workload)-np.min(query_error_on_workload))/np.mean(query_error_on_workload)

            plot(district_error,error_on_workload,key)
            district_error['num_vtds']=list(np.sum(W.dense_matrix(),axis=1).T)
            #print(district_error.sort_values(by=['error']))
            d.append(dict(label = key,test_rmse = error_on_workload, variance = var_error_on_target))
    return pd.DataFrame(d)
 

