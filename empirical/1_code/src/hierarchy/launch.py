import sys
import argparse
import itertools
import subprocess
import os
import numpy as np
import time

"""
Script for launching a batch of experiments with slurm.  To use, simply call 
$ python launch.py topdown --slurm

If you would like to run sequentially on a local machine, simply omit the --slurm flag.
"""

def run_sequentially(commands):
    for cmd in commands:
        print('Running command:', cmd)
        subprocess.run(cmd, shell=True)

def run_with_slurm(commands, sbatch):
    opts = ' '.join(['--%s %s' % (k,v) for k,v in sbatch.items()])
    for cmd in commands:
        print('Running command:', cmd)
        cmd = 'sbatch %s --wrap "%s"' % (opts, cmd)
        subprocess.run(cmd, shell=True) 
        time.sleep(0.1) 

def generate_command(base, config):
    opts = [' --%s %s' % (k,v) for k,v in config.items()]
    return base + ''.join(opts)

def generate_all_commands(base, args):
    """
    :param base: the command to run the python file
    :param args: the configurations to use
        keys are arguments
        values are list possible settings
    
    Note: runs cartesian product of commands implied by args
    """
    keys = args.keys()
    vals = args.values()
    commands = []
    for config in itertools.product(*vals):
        opts = [' --%s %s' % (k,v) for k,v in zip(keys, config)]
        commands.append( base + ''.join(opts) )

    return commands

def main_experiments():
    """
    Get commands to run main.py using the cartesian product of all argument options below.
    """
    args = {}
    args['state'] = ['IA'] #'NC','MA','CT']
    args['epsilon'] = np.logspace(-5, 0, num=6)#np.logspace(-5, -4, num=2) #np.logspace(-4.5, -0.5, num=10) #np.logspace(-5, -1, num=10) #np.logspace(-4, 0, num=10)
    args['trials'] = [1000]
    #args['seed'] = [None] #456, 789,120,320,310,13,543,222,10]
    args['save'] = ['']
    #args['vap'] = ['']
    args['nonneg'] = ['']
    args['budgetsplit'] = ['equal','vtd-heavy2']

    commands = generate_all_commands('python strategy_topdown.py', args)

    sbatch = {}
    sbatch['job-name'] = 'topdown'
    sbatch['ntasks'] = 10
    sbatch['cpus-per-task'] = 1
    sbatch['mem'] = 2048
    sbatch['time'] = '3:00:00'
    sbatch['nodelist'] = 'compute-0-2'
    #sbatch['exclude'] = 'compute-0-[3-13]'

    return commands, sbatch

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('experiment', type=str, help='experiment to run')
    parser.add_argument('--slurm', action='store_true', help='run commands on slurm')

    args = parser.parse_args()

    if args.experiment == 'topdown':
        commands, sbatch = main_experiments()

    if args.slurm:
        run_with_slurm(commands, sbatch)
    else:
        run_sequentially(commands)   
 


