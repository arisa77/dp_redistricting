import numpy as np
import csv
#import matplotlib.pyplot as plt
#import sys

TOTALPOP_ID = 'D000'
TOTALVAP_ID = 'D001'
HISPANIC_ID = 'D002'
WHITE_ID = 'D005'
AFRICANAMERICAN_ID = 'D006'
AMERICANINDIAN_ID = 'D007'
ASIAN_ID = 'D008'
NATIVEHAWAIIAN_ID = 'D009'
OTHER_ID = 'D010'
TWOORMORE_ID = 'D011'
HISPANIC = 'Hispanic'
WHITE = 'White'
AFRICANAMERICAN = 'African American'
AMERICANINDIAN = 'American Indian'
ASIAN = 'Asian'
NATIVEHAWAIIAN = 'Native Hawaiian'
OTHER = 'other'
TWOORMORE = "two or more races"
NONWHITE = 'non white-non-hispanic' # nonwhite_vap = total_vap - white_vap
MINORITY = (HISPANIC,AFRICANAMERICAN,AMERICANINDIAN,ASIAN,NATIVEHAWAIIAN,OTHER,TWOORMORE)



############
def load_data(csvfilename):
    pop_total =None
    vap_total =None
    vap_races ={}
    with open(csvfilename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
       
        for row in csvreader:
            id = row[0]
            data = np.array(list(map(int,row[2:])))
            
            if id == TOTALPOP_ID:  
                pop_total = data
            elif id == TOTALVAP_ID:
                vap_total = data
            elif id == HISPANIC_ID:
                vap_races[HISPANIC] = data
            elif id == WHITE_ID:
                vap_races[WHITE] = data
                vap_NW = vap_total - data
            elif id == AFRICANAMERICAN_ID:
                vap_races[AFRICANAMERICAN] = data
            elif id == AMERICANINDIAN_ID:
                vap_races[AMERICANINDIAN] = data
            elif id == ASIAN_ID:
                vap_races[ASIAN] = data
            elif id == NATIVEHAWAIIAN_ID:
                vap_races[NATIVEHAWAIIAN] = data
            elif id == OTHER_ID:
                vap_races[OTHER] = data
            elif id == TWOORMORE_ID:
                vap_races[TWOORMORE] = data
                
            
        return pop_total, vap_total, vap_races, vap_NW
        
#########################################################################
#######   Population Equality
def compute_ideal_pop(pop):
    
    # pop: a list of population counts for each district
    
    #ideal_pop: ideal population count where total state population count is divided by the number of districts
    
      
    num_districts=len(pop)
    total_state_pop=sum(pop)
    #pop_ideal = round(total_state_pop/num_districts)
    pop_ideal = total_state_pop/num_districts
    return pop_ideal
    
def compute_deviation(pop):
    
    
    #pop: a list of population counts for each district
    
    #deviations: a list of deviations for each district where the deviation is the difference between the population count in a district and the ideal population count (+ or -)
    
    pop_ideal = compute_ideal_pop(pop)
    deviations = (pop - pop_ideal)/pop_ideal
    
    return deviations

def compute_overall_range2(pop):
    
    #pop: a list of population counts for each district
    
    #overall_range: the difference between the largest positive deviation and the largest negative deviation

    deviations=compute_deviation(pop)
    overall_range = np.amax(deviations) - np.amin(deviations)
   
    return overall_range
def compute_overall_range(pop):
    
    # compute overall range without using deviations
    # simpler than the naive way 
    
    # overall range  = ( Max(pop) - Min (pop) )/(ideal pop)
    if compute_ideal_pop(pop)==0:
        return 0
    else:
        overall_range = (np.amax(pop) - np.amin(pop))/(compute_ideal_pop(pop))
    
    return overall_range
    
    
def test_equality_pop(pop,threshold):
    

    overall_range = compute_overall_range(pop)
    #overall_range = compute_overall_range2(pop)
    #print("overall range: ",overall_range)
    
    if(overall_range<=threshold):
        return True, overall_range
    else :
        return False, overall_range

    
#########################################################################
#######   Voting Rights Act

def compute_vap_total(vap_races):

    
    vap_total = [0.0]*len(vap_races[WHITE])
    vap_total =np.array(vap_total)
    for race in vap_races:
        vap_total +=vap_races[race]

    return vap_total

def compute_vap_fraction(vap_total,vap_race):
    
    vap_f=vap_race/vap_total
    
    return vap_f
    
def compute_num_majority_minority_districts(vap_f):
    
   
    num_MM_districts=np.count_nonzero(vap_f > 0.5)
    
    return num_MM_districts

def compute_num_mm_districts(vap_races,white_column):
    
    #vap_races: a dictionary containing a list of counts for each race
    
    #compute counts of non-white
    nonwhite = []
    for race in vap_races.keys():
        if race != white_column:
            nonwhite.append(vap_races[race])
    nonwhite = np.sum(np.vstack(nonwhite),axis=0)
    white = np.array(vap_races[white_column])
    
    nonwhite_frac = nonwhite/(nonwhite+white)
    
    num_MM_districts=np.count_nonzero(nonwhite_frac > 0.5)
    return num_MM_districts
    
def test_voting_rights_act2(vap_white,vap_race,M):
    #majority minority
    
    vap_f = np.where(vap_race>vap_white,1,0)
    mm=compute_num_majority_minority_districts(vap_f)
    
    if mm>=M:
        return True,mm
    else:
        return False,mm    

def test_voting_rights_act(vap_total,vap_race,M):
    #majority minority
    
    vap_f =compute_vap_fraction(vap_total,vap_race)
    mm=compute_num_majority_minority_districts(vap_f)
    
    if mm>=M:
        return True,mm
    else:
        return False,mm    

#########################################################################
#######   Noise
def checkValidateNum(array):
    #return false if there exist negative numbers in array, otherwise true
    array = array.flatten()   
    for ele in array:
        if ele < 0:
            return False
    return True
def toValidateCount(x):

    # x is an np.array

    #each value is to be rounded
    # if each value is negative, return 0.

    y = np.where(x>0,np.round(x),0)
    
    return y

            
def laplace(data,epsilon,trial=1, sensitivity=1):
    
    num = data.size 
    lamda = sensitivity/epsilon
    shape =(trial,num)
    
   
    noisy_data = data + np.random.laplace(0.0, lamda, shape)
   
    noisy_data = toValidateCount(noisy_data) 
    return noisy_data

def precompute(data):
    
    pop_total, vap_total, vap_races, vap_NW= data
    

    #compute ground truth overall range
    overall_range=compute_overall_range(pop_total)

    
    #compute ground truth # of minority districts    
    vap_f = compute_vap_fraction(vap_total,vap_NW)
    M = compute_num_majority_minority_districts(vap_f)
    
        
    return overall_range, M

    
def noisytest_EP(pop_total,threshold,eps=0.01,trial=1):
    
    
    results = []#a list of boolean; true if the condition meats, false otherwise
    overall_ranges = []
    init_state,_ = test_equality_pop(pop_total,threshold)# check if the ground truth plan meets the condition
    #add noise to population count in each district
    total_pop_noisy=laplace(pop_total,eps,trial=trial)
  
    checkPopNoisy = checkValidateNum(total_pop_noisy)

    for i in range(trial):
        #test population equality   
        result,overall_range=test_equality_pop(total_pop_noisy[i],threshold)
        if init_state==result:
            results.append(True)
        else:
            results.append(False)
        overall_ranges.append(overall_range)
    accuracy= np.mean(results)
    return checkPopNoisy,accuracy, overall_ranges

def noisytest_VRA(vap_races,threshold,eps=0.01,trial=1):
    
    
    
    #test VRA
    #print("VRA test")
    noisy_vap_races = []
    noisy_vap_totals=[]
    
    checkPopNoisy=True;
    for i in range(trial):
        noisy_vap = {}
        for race in vap_races:
            noisy_vap[race]=laplace(vap_races[race],eps,trial=1)[0]
            
            flag = checkValidateNum(noisy_vap[race])
            if flag==False:
                checkPopNoisy = False
                
        noisy_vap_races.append(noisy_vap)
        noisy_vap_total = compute_vap_total(noisy_vap)
        noisy_vap_totals.append(noisy_vap_total)

    
    numMMs = []
    results = []

    for i in range(trial):
        noisy_vap_NW = noisy_vap_totals[i] - noisy_vap_races[i][WHITE]
        #r,l = test_voting_rights_act(noisy_vap_totals[i],noisy_vap_NW, threshold)
        r,l = test_voting_rights_act2(noisy_vap_races[i][WHITE],noisy_vap_NW, threshold)
        numMMs.append(l)
        results.append(r)
    accuracy = np.mean(results)
    return checkPopNoisy,accuracy,numMMs



def noisytest(data,ep_type,red_type,eps=0.01,trial=1):
    
    # return True if a privatize redistricting data satisfies both equal population and VRA conditions.
    # for EP, global threshold is used and for VRA, majority minority is used
    
    pop_total, vap_total, vap_races, vap_NW= data
    overall_range,threshold_VRA = precompute(data)
    #define threshold
    
    
    if red_type=="congressional" and ep_type =="global":
        threshold_EP = 0.01 #1% overall range       
    elif red_type=="congressional" and ep_type =="local":
        threshold_EP = overall_range
    elif red_type == "state legislative" and ep_type == "global":
        threshold_EP = 0.1 #10% standard rule
    elif red_type == "state legislative" and ep_type == "local":
        threshold_EP = overall_range
    else:
        assert False
    
    init_state,_=test_equality_pop(pop_total,threshold_EP)
  
    results = []
   
    
    noisy_vap_races = []
    noisy_vap_totals=[]
    
    #add noise to population count in each district
    total_pop_noisy=laplace(pop_total,eps,trial=trial)
    
    #add noise to voting age population count for each race in each district 
    for i in range(trial):
        noisy_vap = {}
        for race in vap_races:
            noisy_vap[race]=laplace(vap_races[race],eps,trial=1)[0]
        noisy_vap_races.append(noisy_vap)
        noisy_vap_total = compute_vap_total(noisy_vap)
        noisy_vap_totals.append(noisy_vap_total)

    
    
  
    for i in range(trial):
        #test population equality   
        result_EP,dev=test_equality_pop(total_pop_noisy[i],threshold_EP)
        
        #test voting rights act
        noisy_vap_NW = noisy_vap_totals[i] - noisy_vap_races[i][WHITE]
        result_VRA,mm = test_voting_rights_act(noisy_vap_totals[i],noisy_vap_NW, threshold_VRA)
       
        result = result_EP and result_VRA #True if both of the conditions are true
        
        if init_state == result:#T->T or F->F
            results.append(True)
        else:
            results.append(False)
        
    accuracy = np.mean(results)
    return accuracy


