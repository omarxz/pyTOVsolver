#!/bin/bash

# job name
#SBATCH --job-name=GR_TOV

# Mail events (NONE, BEGIN, END, REQUEUE, FAIL, ALL)
#SBATCH --mail-type=END,FAIL,REQUEUE   
#SBATCH --mail-user=oramadan@hawaii.edu 
###############################################################################
# standard output and error log
#SBATCH --output=../output/output_%x.log
#SBATCH --error=../output/error_%x.err
###############################################################################
#SBATCH --partition=sandbox
#SBATCH --time=04:00:00

# ask for --nstasks & --cpus-per-task and let the slurm asks for the required nodes. Can use up to 640 cores (80tX4cpus)
# SBATCH --cpus-per-task=1
#SBATCH --ntasks=19 # nodes have 40 to 48 physical cores, so ask for 2 nodes with this number of tasks.
#SBATCH --nodes=1 --exclusive
###############################################################################
#SBATCH --mem=32GB
###############################################################################

# activate the environment

module purge
module load lang/Anaconda3
source activate py3.11

# are we okay?
which python
module list


cd /home/oramadan/Axions/pyTOVsolver/
python3 -u GR_TOV_solver.py

