#!/bin/bash
#PBS -l nodes=1:ppn=16

module load anaconda3/2019.10
source activate daba

cd /home/jruiz/DABA/Lorenz_96/experiments/

#GM CONFIGURATION CON RESAMPLING NETPF
MY_EXP=sensitivit_experiment_beta_multinf_gm_netpf_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log

#GM CONFIGURATION CON RESAMPLING LETPF
MY_EXP=sensitivit_experiment_beta_multinf_gm_letpf_multinature
python -u ./${MY_EXP}.py  > ./logs/${MY_EXP}.log







