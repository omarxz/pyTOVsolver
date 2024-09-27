#!/bin/bash

# job name
#SBATCH --job-name=fa2e18_ma12_gsN3e-23
# Mail events (NONE, BEGIN, END, REQUEUE, FAIL, ALL)
#SBATCH --mail-type=END,FAIL,REQUEUE   
#SBATCH --mail-user=oramadan@hawaii.edu 
###############################################################################
# standard output and error log
#SBATCH --output=../output/%x/output_%x.log
#SBATCH --error=../output/%x/error_%x.err
###############################################################################
#SBATCH --partition=shared,exclusive,gpu
#SBATCH --time=3-00:00:00
# SBATCH --cpus-per-task=1
# since I'm using exclusive, no need to specify cores as it will either be 20 or 40
# SBATCH --ntasks=39 
#SBATCH --nodes=1 --exclusive
###############################################################################
#SBATCH --mem=64GB
###############################################################################
# activate the environment
module purge
module load lang/Anaconda3
source activate
conda activate py3.11
# are we okay?
which python
module list

cd /home/oramadan/Axions/pyTOVsolver/

job_name="fa2e18_ma12_gsN3e-23"
# Print the current working directory
echo "Bash > Current directory: $(pwd)"
# Define the values you want to change
pre_name="$job_name"
g_s_N="3e-23"
ode_ma="1e-12 * 1e-9 * cmToGeVInv"
ode_fa="2e18 * PhiFaGeVToCGs"
num_of_stars="80"
# modify path to save in py script
sed -i "s/^pre_name = .*/pre_name = \"$pre_name\"/g" parallel_full_sol.py
# check that the edits are correct
echo "Bash > After replacing pre_name:"
grep "pre_name =" parallel_full_sol.py
# modify coupling to save in py script
sed -i "s/^g_s_N = .*/g_s_N = $g_s_N/g" parallel_full_sol.py
sed -i "s/^num_of_stars = .*/num_of_stars = $num_of_stars/g" parallel_full_sol.py
echo "Bash > After replacing num_of_stars:"
grep "num_of_stars =" parallel_full_sol.py
sed -i "s|^ode_system.ma = .*|ode_system.ma = $ode_ma|g" parallel_full_sol.py
sed -i "s|^ode_system.fa = .*|ode_system.fa = $ode_fa|g" parallel_full_sol.py
# check that the coupling is correct
echo "Bash > After replacing g_s_N:"
grep "g_s_N =" parallel_full_sol.py

python3 -u parallel_full_sol.py