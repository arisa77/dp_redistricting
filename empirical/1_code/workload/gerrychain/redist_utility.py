import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shapefile as shp
from gerrychain.metrics import mean_median,efficiency_gap
from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition.assignment import level_sets
from gerrychain import MarkovChain,constraints
from gerrychain.constraints import single_flip_contiguous,contiguous,UpperBound
from gerrychain.proposals import propose_random_flip,recom
from gerrychain.accept import always_accept
from gerrychain.tree import recursive_tree_part
from functools import partial
'''
def create_noisy_shapefile(shp_path,noisy_pop,save_path=None):
    'noisy_pop: a list of noisy_pop'
    gdf=geopandas.read_file(shp_path)
    
    new_gdf = gdf.copy()
    for i,xest in enumerate(xests):
        new_gdf["NOISYPOP-"+str(i+1)] = xest
        
    if save_path is not None:
        new_gdf.to_file(save_path)
    
    return new_gdf
'''

def create_graph(shp_path):
    return Graph.from_file(shp_path,ignore_errors=True)

def create_noisy_graph(shp_path,noisy_pop,columns):
    
    graph = Graph.from_file(shp_path,ignore_errors=True)
    df = pd.DataFrame(index=list(graph.nodes()))
    for i,xest in enumerate(noisy_pop):
        df[columns[i]] = xest
    graph.add_data(df)
    return graph
    
def read_shapefile(path):
    sf = shp.Reader(path)
    #fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]
    #fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]
    #converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)
    #assigning the coordinates
    df = df.assign(coords=shps)
    return df
def plot_statistics(mm_list,title):
    
    #sns.set()
    for mm, label in mm_list:
        #sns.distplot(mm,label=label)
        plt.hist(mm,label=label,alpha=0.3)
    plt.legend()
    plt.xlabel(title)
    plt.show()

class redistChain:
    
    def __init__(self,graph,pop_col,assignment_col,parts,pop_bound = 0.02,total_steps = 2000,elections = None,initial_seed = None,compact=True):
        
        self.graph = graph
        self.pop_col = pop_col
        self.assignment_col = assignment_col
        self.parts = parts
        self.pop_bound = pop_bound
        self.total_steps = total_steps
        self.elections = elections
        self.initial_seed = initial_seed
        self.compact = compact
    
    
    def random_seed(self):

        my_updaters = {
            'cut_edges':cut_edges,
            'population':Tally(self.pop_col,alias='population'), 
        }
        initial_partition = Partition(
            self.graph,
            assignment = self.assignment_col,
            updaters = my_updaters
        )
        ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
        #print(ideal_population)
        self.seed = recursive_tree_part(
            self.graph,
            parts = self.parts, #(0,13)
            pop_target=ideal_population, 
            pop_col=self.pop_col,
            epsilon = self.pop_bound,
            node_repeats=10, 
        )
        return self.seed
        

    def set_chain(self):

        my_updaters = {
                'cut_edges':cut_edges,
                'population':Tally(self.pop_col,alias='population'), 

            }
        
        if self.elections != None:
            #elections = [
            #Election(self.election[0],self.election[1])
            #]
            election_updaters = {election.name:election for election in self.elections}
            my_updaters.update(election_updaters)

        if self.initial_seed == None:
            
            initial_partition = Partition(
                self.graph,
                assignment = self.random_seed(),
                updaters = my_updaters
            )
        else:
            initial_partition = Partition(
                self.graph,
                assignment = self.initial_seed,
                updaters = my_updaters
            )

        self.ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
        proposal = partial(recom,
                           pop_col=self.pop_col,
                           pop_target=self.ideal_population,
                           epsilon=self.pop_bound,
                           node_repeats=10
                          )

        pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, self.pop_bound)

        if self.compact is False:
            self.chain = MarkovChain(
                proposal=proposal,
                constraints=[
                    pop_constraint,
                ],
                accept=always_accept,
                initial_state=initial_partition,
                total_steps=total_steps
            )
        else:
            compactness_bound = constraints.UpperBound(
                lambda p: len(p["cut_edges"]),
                2*len(initial_partition["cut_edges"])
            )
            self.chain = MarkovChain(
                proposal=proposal,
                constraints=[
                    pop_constraint,
                    compactness_bound
                ],
                accept=always_accept,
                initial_state=initial_partition,
                total_steps=self.total_steps
            )
        return self.chain
    
    def run_chain(self,save_path,statistics=None,debug=False,units=None):
        #run a chain and save a mapping of assignment for each generated partition
        #statistics = dict{}: column: string, path: string
        ass_list = []
        mm_list =[]
        eg_list = []
        seats_list = []
        percents = []
        cut_edges = []
        for (i,partition) in enumerate(self.chain.with_progress_bar()):
            if i%100==0:
                if debug:
                    print(i)
            if units is not None:
                partition.plot(units,figsize=(10,10),cmap=plt.cm.get_cmap('tab20', 13),legend=False)
                plt.axis('off')
                plt.show()
            ass_list.append(partition.assignment.to_dict())
            #print(partition.assignment.to_dict())
            if statistics != None:
                mm_list.append(mean_median(partition[statistics['column']]))
                eg_list.append(efficiency_gap(partition[statistics['column']]))
                seats_list.append(partition[statistics['column']].wins("Democratic"))
                percents.append(partition[statistics['column']].percents('Democratic'))
                cut_edges.append(len(partition['cut_edges']))
        with open(save_path,'wb') as f:
            pickle.dump(ass_list, f, pickle.HIGHEST_PROTOCOL)

        if statistics != None:
            stat = {}
            stat['mean-median'] = mm_list
            stat['efficiency-gap'] = eg_list
            stat['seats-d'] = seats_list
            stat['percents-d'] = percents
            stat['cut-edges'] = cut_edges
            with open(statistics['path'],'wb') as f:
                pickle.dump(stat, f, pickle.HIGHEST_PROTOCOL)

            #np.save(statistics['path'],st_list)
        
def save_maps(load_path,save_path,partition,units,every_print,debug=False):

    with open(load_path, 'rb') as f:
        ass_maps = pickle.load(f)

    ncols=10
    if int(len(ass_maps)/every_print)<ncols:
        ncols=int(len(ass_maps)/every_print)
    nrows=int(len(ass_maps)/every_print/ncols)
    if len(ass_maps) % ncols !=0:
        nrows +=1
    print(nrows,ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(50,10*nrows))
    count=0


    for (i,ass_map) in enumerate(ass_maps):
        if i%every_print!=0:
            continue
        if ncols*nrows <=count:
            break
        if debug:
            print(i,count)
        partition.assignment.update(ass_map)
        if nrows==1:
            ax = partition.plot(units,ax=axes[count])
        else:
            ax = partition.plot(units,ax=axes[int(count/ncols),int(count%ncols)])
        ax.set_axis_off()
        count+=1

    plt.savefig(save_path,bbox_inches='tight')
    plt.show()
    
def display_maps(graph,partitions,every_print,save_path = None,debug=False):
    
    #parms partitions: list of lists
    
    #df_partitions.columns = np.arange(df_partitions.shape[1]) #to match keys
    
    #num_partitions = df_partitions.shape[0]
    num_partitions = len(partitions)
    
    ncols=10
    if int(num_partitions/every_print)<ncols:
        ncols=int(num_partitions/every_print)
    nrows=int(num_partitions/every_print/ncols)
    if num_partitions % ncols !=0:
        nrows +=1
    print(nrows,ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(50,10*nrows))
    count=0


    for i in range(num_partitions):
        if i%every_print!=0:
            continue
        if ncols*nrows <=count:
            break
        if debug:
            print(i,count)
        partition = Partition(
            graph,
            #assignment = df_partitions.iloc[i].to_dict(),
            assignment = {j:district for j,district in enumerate(partitions[i])},
        )
        if nrows==1:
            ax = partition.plot(ax=axes[count])
        else:
            ax = partition.plot(ax=axes[int(count/ncols),int(count%ncols)])
        ax.set_axis_off()
        count+=1
    if save_path:
        plt.savefig(save_path,bbox_inches='tight')
    plt.show()

