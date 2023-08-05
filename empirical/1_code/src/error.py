import init
import pickle
import sys
import os, re, math, time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
import numpy as np
from hdmm import error
from ektelo.matrix import EkteloMatrix



palette = np.array(["green","orange","red","blue","gray",'purple','black','pink','brown','cyan','olive','darkcyan','lime'])

def get_subworkloads(Ws,sub_size=13):
    W = []
    num_plans = int(Ws.shape[0]/sub_size)
    for i in range(num_plans):
        offset=i*sub_size
        W.append(Ws[offset:offset+sub_size,:])
        
    return W
    
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

def empirical_error_per_precinct(x, xests,errors, label):
    # compute square root mean suqred error
    
    xests = np.vstack(xests) # shape = (ntrials, num_precincts)
    error = np.sqrt(np.mean((x - xests)**2, axis = 0))
    assert error.shape[0] == x.shape[0]

    errors[label] = error

def empirical_error_per_district(Ws, district_size, x, xests, errors, label):
    # compute mean suqred error
    
    xests = np.vstack(xests) # shape = (ntrials, num_precincts)

    errors[label] = [np.sqrt(np.mean((W.dot(x) - W.dot(xests.transpose()).transpose())**2, axis = 0)) for W in get_subworkloads(Ws,sub_size=district_size)]

def empirical_error_per_workload(Ws, x, xests, errors, label):
    # compute mean suqred error
    
    xests = np.vstack(xests) # shape = (ntrials, num_precincts)

    errors[label] = np.sqrt(np.mean((Ws.dot(x) - Ws.dot(xests.transpose()).transpose())**2, axis = 0))

def error_per_precinct(strategy, errors, label, eps = np.sqrt(2)):

    errors[label] = np.array(per_data_error(strategy,eps=eps))
    
def error_per_district(Ws, strategy, district_size, errors, label, eps = np.sqrt(2)):
    
    # Ws: a collection of redistricting workloads

    errors[label] = per_subworkload_error(Ws,strategy, eps=eps, sub_size=district_size)

def plot_error_per_district(errors,state,save=None):


    dfs = []
    for stg in errors.keys():
        error = np.mean(np.sort(np.vstack(errors[stg]), axis=1),axis=0)
        df = pd.DataFrame([[val for val in error]],columns=np.arange(len(error))).assign(method = stg).melt(id_vars = 'method')
        dfs.append(df)
    df = pd.concat(dfs,ignore_index=True)
    #print(df)
    sns.lineplot(x = 'variable', y= 'value', hue = 'method', style = 'method',markers=True,data = df, palette=list(palette[:len(errors.keys())]))
    plt.xlabel('District')
    plt.ylabel('Error')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(pipeline, 'store','error_per_district_'+state+'.eps'),dpi=300)
        plt.savefig(save,dpi=300)
    plt.show()

def plot_error_per_precinct(errors,state,save=None):
    dfs = []
    for stg in errors.keys():
        df = pd.DataFrame([[val for val in sorted(errors[stg])]],columns=np.arange(len(errors[stg]))).assign(method = stg).melt(id_vars = 'method')
        dfs.append(df)
    df = pd.concat(dfs,ignore_index=True)
    
    sns.lineplot(x = 'variable', y= 'value', hue = 'method', style = 'method', data = df,palette=list(palette[:len(errors.keys())]))
    plt.xlabel('VTD')
    plt.ylabel('Error')
    plt.tight_layout()
    if save:
        #plt.savefig(os.path.join(pipeline, 'store','error_per_vtd_'+state+'.eps'),dpi=300)
        plt.savefig(save,dpi=300)
    plt.show()

def plot_error_per_geounit(errors,state,geounit_name, save=None):
    dfs = []
    for stg in errors.keys():
        df = pd.DataFrame([[val for val in sorted(errors[stg])]],columns=np.arange(len(errors[stg]))).assign(method = stg).melt(id_vars = 'method')
        dfs.append(df)
    df = pd.concat(dfs,ignore_index=True)
    
    sns.lineplot(x = 'variable', y= 'value', hue = 'method', style = 'method', data = df,palette=list(palette[:len(errors.keys())]))
    plt.xlabel(geounit_name)
    plt.ylabel('Error')
    plt.tight_layout()
    if save:
        #plt.savefig(os.path.join(pipeline, 'store','error_per_vtd_'+state+'.eps'),dpi=300)
        plt.savefig(save,dpi=300)
    plt.show()

def plot_error_per_precinct_bgtsplit(errors,state,save=None):
    dfs = []
    for bgtsplit in errors.keys():
        for stg in errors[bgtsplit].keys():
            df = pd.DataFrame([[val for val in sorted(errors[bgtsplit][stg])]],columns=np.arange(len(errors[bgtsplit][stg]))).assign(method = stg).assign(budget_split = bgtsplit).melt(id_vars = ['method','budget_split'])
            dfs.append(df)
    df = pd.concat(dfs,ignore_index=True)
    
    sns.lineplot(x = 'variable', y= 'value', hue = 'method', style = 'budget_split',data = df,palette=list(palette[:len(df['method'].unique())]))
    plt.xlabel('VTD')
    plt.ylabel('Error')
    plt.tight_layout()
    if save:
        #plt.savefig(os.path.join(pipeline, 'store','error_per_vtd_'+state+'.eps'),dpi=300)
        plt.savefig(save,dpi=300)
    plt.show()
    
def plot_error_per_district_bgtsplit(errors,state,save=None):


    dfs = []
    for bgtsplit in errors.keys():
        for stg in errors[bgtsplit].keys():
            error = np.mean(np.sort(np.vstack(errors[bgtsplit][stg]), axis=1),axis=0)
            df = pd.DataFrame([[val for val in error]],columns=np.arange(len(error))).assign(method = stg).assign(budget_split = bgtsplit).melt(id_vars = ['method','budget_split'])
            dfs.append(df)
    df = pd.concat(dfs,ignore_index=True)
    #print(df)
    #sns.lineplot(x = 'variable', y= 'value', hue = 'method', style = 'budget_split',markers=True,data = df, palette=list(palette[:len(df['method'].unique())]))
    sns.lineplot(x = 'variable', y= 'value', hue = 'method', style = 'budget_split',markers=True,data = df, palette='rocket')
    plt.xlabel('District')
    plt.ylabel('Error')
    plt.tight_layout()
    if save:
        #plt.savefig(os.path.join(pipeline, 'store','error_per_district_'+state+'.eps'),dpi=300)
        plt.savefig(save,dpi=300)
    plt.show()
